#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — Vòng huấn luyện CVAE dual-decoder (thiết kế B4).

Cài đặt (Windows, trong venv đã kích hoạt):
    pip install torch sentencepiece
    (Có GPU NVIDIA thì lấy lệnh cài bản CUDA từ pytorch.org — KHÔNG bắt buộc,
     model ~6–9M tham số train được bằng CPU. Console lỗi tiếng Nhật →
     dùng Windows Terminal hoặc `set PYTHONUTF8=1`.)

LUÔN bắt đầu bằng sanity overfit (bắt bug trước khi tốn giờ train):
    python train.py --overfit 50 --epochs 300 --run sanity
    → kỳ vọng ce_trans/token < 0.5 và F1 ≈ 1.0 trên chính 50 câu đó. Không đạt = bug.

Tam giác ablation (mỗi lệnh một run, so sánh bằng metrics.csv):
    python train.py --variant seq2seq --run a_seq2seq
    python train.py --variant cvae1   --run b_cvae1
    python train.py --variant cvae2   --run c_cvae2

Đầu vào : data/splits/{train,dev}.jsonl
          (fallback: data/processed/mined.jsonl + cảnh báo — chỉ để chạy sớm)
Đầu ra  : runs/<run>/{config.json, metrics.csv, kl_dims.csv, ckpt_best.pt, ckpt_last.pt}
          artifacts/{spm_ja.model, spm_vi.model, grammar_vocab.json} (build một lần)
"""

from __future__ import annotations
import argparse, csv, json, math, random, sys, time
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":                      # console Windows → UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import sentencepiece as spm

from model import CVAE, PAD, UNK, BOS, EOS

CFG = dict(
    data_dir="data/splits", art_dir="artifacts", runs_dir="runs",
    vocab_ja=4000, vocab_vi=4000,                # hard_vocab_limit=False → tự co khi data nhỏ
    d_emb=256, enc_hid=256, dec_hid=512, d_z=64, attn=256, gr_emb=64, gr_hid=256,
    max_ja=80, max_vi=60,
    batch=64, epochs=60, lr=1e-3, clip=1.0, seed=13,
    lam=1.0,                                     # trọng số CE ngữ pháp (đã chuẩn hóa/token)
    beta_max=1.0, warmup_frac=0.15,              # KL annealing tuyến tính (A6)
    free_bits=0.2,                               # λ_fb mỗi chiều, nats (A6)
    p_wd=0.3,                                    # word dropout decoder DỊCH (A6)
    patience=10, log_every=20, dev_eval_n=200,
)

# ---------------------------------------------------------------- dữ liệu
def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def load_rows(data_dir: str):
    d = Path(data_dir)
    if (d / "train.jsonl").exists() and (d / "dev.jsonl").exists():
        return read_jsonl(d / "train.jsonl"), read_jsonl(d / "dev.jsonl"), None
    mined = Path("data/processed/mined.jsonl")
    if mined.exists():
        rows = read_jsonl(mined)
        cut = max(1, int(0.9 * len(rows)))
        warn = ("CHƯA có data/splits — dùng tạm mined.jsonl chia 90/10. "
                "Chỉ để sanity/chạy sớm; số liệu KHÔNG dùng cho báo cáo.")
        return rows[:cut], rows[cut:], warn
    sys.exit("Không thấy data/splits/ lẫn data/processed/mined.jsonl — chạy pipeline dữ liệu trước.")

def build_vocabs(cfg, rows_train, rows_dev):
    art = Path(cfg["art_dir"]); art.mkdir(parents=True, exist_ok=True)

    def spm_train(texts, prefix, vocab, coverage):
        if Path(f"{prefix}.model").exists():
            return
        tmp = Path(f"{prefix}.txt")
        tmp.write_text("\n".join(texts), encoding="utf-8")
        spm.SentencePieceTrainer.train(
            input=str(tmp), model_prefix=prefix, vocab_size=vocab,
            model_type="unigram", character_coverage=coverage,
            pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
            hard_vocab_limit=False, minloglevel=2)
        tmp.unlink()

    spm_train([r["ja"] for r in rows_train], str(art / "spm_ja"),
              cfg["vocab_ja"], 0.9995)
    spm_train([r["vi"] for r in rows_train], str(art / "spm_vi"),
              cfg["vocab_vi"], 1.0)

    gpath = art / "grammar_vocab.json"
    if not gpath.exists():
        tags = sorted({h["id"] for r in rows_train + rows_dev
                       for h in r.get("patterns", [])})
        gvoc = {"<pad>": PAD, "<unk>": UNK, "<bos>": BOS, "<eos>": EOS, "<none>": 4}
        for t in tags:
            gvoc[t] = len(gvoc)
        gpath.write_text(json.dumps(gvoc, ensure_ascii=False, indent=1), encoding="utf-8")

    sp_ja = spm.SentencePieceProcessor(model_file=str(art / "spm_ja.model"))
    sp_vi = spm.SentencePieceProcessor(model_file=str(art / "spm_vi.model"))
    gvoc = json.loads(gpath.read_text(encoding="utf-8"))
    return sp_ja, sp_vi, gvoc

def encode_rows(rows, sp_ja, sp_vi, gvoc, cfg):
    data = []
    for r in rows:
        x = sp_ja.encode(r["ja"], out_type=int)[: cfg["max_ja"]] + [EOS]
        y = [BOS] + sp_vi.encode(r["vi"], out_type=int)[: cfg["max_vi"]] + [EOS]
        tags = sorted({h["id"] for h in r.get("patterns", [])})
        gid = [gvoc[t] for t in tags if t in gvoc] or [gvoc["<none>"]]
        g = [BOS] + gid + [EOS]
        data.append(dict(x=x, y=y, g=g, gold=set(tags), vi=r["vi"],
                         id=r.get("id", ""), ja=r.get("ja", "")))
    return data

def pad_2d(seqs, device):
    T = max(len(s) for s in seqs)
    out = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out.to(device)

def make_batches(data, bs, shuffle, rng, device):
    idx = sorted(range(len(data)), key=lambda i: len(data[i]["x"]))
    chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        rng.shuffle(chunks)
    for ch in chunks:
        items = [data[i] for i in ch]
        y = pad_2d([it["y"] for it in items], device)
        g = pad_2d([it["g"] for it in items], device)
        yield dict(
            x=pad_2d([it["x"] for it in items], device),
            lens=torch.tensor([len(it["x"]) for it in items]),
            y_in=y[:, :-1], y_out=y[:, 1:],
            g_in=g[:, :-1], g_out=g[:, 1:],
            gold=[it["gold"] for it in items],
            refs=[it["vi"] for it in items],
            ids=[it["id"] for it in items],
            jas=[it["ja"] for it in items])

# ---------------------------------------------------------------- metric nội bộ
def chrf(hyps: list[str], refs: list[str], max_n: int = 6, beta: float = 2.0) -> float:
    """chrF xấp xỉ sacrebleu (bỏ khoảng trắng) — dùng cho early stopping;
    số báo cáo cuối chạy sacrebleu trong evaluate.py."""
    from collections import Counter
    tot, orders = 0.0, 0
    for n in range(1, max_n + 1):
        m = hp = rf = 0
        for h, r in zip(hyps, refs):
            h2, r2 = h.replace(" ", ""), r.replace(" ", "")
            hc = Counter(h2[i:i + n] for i in range(max(len(h2) - n + 1, 0)))
            rc = Counter(r2[i:i + n] for i in range(max(len(r2) - n + 1, 0)))
            m += sum((hc & rc).values()); hp += sum(hc.values()); rf += sum(rc.values())
        if hp == 0 or rf == 0:
            continue
        P, R = m / hp, m / rf
        tot += (1 + beta**2) * P * R / (beta**2 * P + R) if (P + R) else 0.0
        orders += 1
    return 100 * tot / max(orders, 1)

def prf(preds: list[set], golds: list[set]):
    tp = fp = fn = 0
    per = defaultdict(lambda: [0, 0, 0])
    for p, g in zip(preds, golds):
        for t in p & g: tp += 1; per[t][0] += 1
        for t in p - g: fp += 1; per[t][1] += 1
        for t in g - p: fn += 1; per[t][2] += 1
    micro = 2 * tp / max(2 * tp + fp + fn, 1)
    f1s = [2 * a / max(2 * a + b + c, 1) for a, b, c in per.values()]
    macro = sum(f1s) / max(len(f1s), 1)
    return micro, macro

# ---------------------------------------------------------------- eval dev
@torch.no_grad()
def evaluate(model, dev, sp_vi, inv_g, cfg, device):
    model.eval()
    ce_t = ce_g = kl = 0.0; nb = 0; mus = []; klvs = []
    for b in make_batches(dev, cfg["batch"], False, None, device):
        out = model(b, p_wd=0.0, sample_z=False)
        ce_t += out["ce_trans"].item()
        if model.use_gram: ce_g += out["ce_gram"].item()
        if model.use_z:
            kl += out["klv"].sum().item(); klvs.append(out["klv"].cpu())
            mus.append(out["mu"].cpu())
        nb += 1
    hyps, refs, preds, golds = [], [], [], []
    for b in make_batches(dev[: cfg["dev_eval_n"]], cfg["batch"], False, None, device):
        y, g, _ = model.predict(b["x"], b["lens"], cfg["max_vi"])
        for row in y.tolist():
            ids = []
            for i in row:
                if i == EOS: break
                if i != PAD: ids.append(i)
            hyps.append(sp_vi.decode(ids))
        refs += b["refs"]; golds += b["gold"]
        if g is not None:
            for row in g.tolist():
                st = set()
                for i in row:
                    if i == EOS: break
                    tok = inv_g.get(i)
                    if tok and not tok.startswith("<"): st.add(tok)
                preds.append(st)
    f1mi, f1ma = prf(preds, golds) if preds else (float("nan"),) * 2
    au = 0; klv_mean = None
    if mus:
        M = torch.cat(mus, 0)
        au = int((M.var(dim=0) > 0.01).sum().item())      # active units, δ=0.01 (A6)
        klv_mean = torch.stack(klvs).mean(0)
    return dict(ce_trans=ce_t / nb, ce_gram=(ce_g / nb if model.use_gram else float("nan")),
                kl=(kl / nb), chrf=chrf(hyps, refs),
                f1_micro=f1mi, f1_macro=f1ma, au=au, klv=klv_mean)

def monitor_score(ev, use_gram):
    if use_gram and ev["f1_micro"] == ev["f1_micro"]:     # not NaN
        return (ev["chrf"] + 100 * ev["f1_micro"]) / 2
    return ev["chrf"]

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--variant", default="cvae2", choices=["seq2seq", "cvae1", "cvae2"])
    ap.add_argument("--overfit", type=int, default=0, metavar="N")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="chỉ dùng N câu train đầu (chạy nhanh máy yếu; KHÁC --overfit: "
                         "vẫn giữ dev thật, p_wd, early stopping)")
    ap.add_argument("--dev-eval-n", type=int,
                    help="số câu dev greedy-decode mỗi epoch (mặc định 200 — bóp nhỏ cho nhanh)")
    ap.add_argument("--patience", type=int); ap.add_argument("--log-every", type=int)
    ap.add_argument("--epochs", type=int); ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float); ap.add_argument("--seed", type=int)
    ap.add_argument("--lam", type=float); ap.add_argument("--beta-max", type=float)
    ap.add_argument("--free-bits", type=float); ap.add_argument("--p-wd", type=float)
    ap.add_argument("--warmup-frac", type=float)
    ap.add_argument("--data-dir"); ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = dict(CFG); cfg["variant"] = args.variant
    for k in ("epochs", "batch", "lr", "seed", "lam", "data_dir",
              "beta_max", "free_bits", "p_wd", "warmup_frac",
              "dev_eval_n", "patience", "log_every"):
        v = getattr(args, k, None)
        if v is not None: cfg[k] = v

    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rows_tr, rows_de, warn = load_rows(cfg["data_dir"])
    if warn: print("⚠", warn)
    if args.overfit:
        rows_tr = rows_tr[: args.overfit]; rows_de = rows_tr
        cfg["p_wd"] = 0.0; cfg["patience"] = 10**9; cfg["dev_eval_n"] = len(rows_tr)
        cfg["art_dir"] = "artifacts_overfit"   # vocab tí hon KHÔNG được làm bẩn artifacts/
        print(f"CHẾ ĐỘ OVERFIT: {len(rows_tr)} câu, p_wd=0 — kỳ vọng CE→~0, F1→~1 "
              f"(vocab ghi vào artifacts_overfit/, tách khỏi ablation thật)")
    elif args.limit:
        rows_tr = rows_tr[: args.limit]
        print(f"GIỚI HẠN train {len(rows_tr)} câu (smoke-run máy yếu — số liệu KHÔNG "
              f"dùng cho báo cáo cuối; nhớ xóa artifacts/ trước để vocab build lại)")

    sp_ja, sp_vi, gvoc = build_vocabs(cfg, rows_tr, rows_de)
    inv_g = {v: k for k, v in gvoc.items()}
    tr = encode_rows(rows_tr, sp_ja, sp_vi, gvoc, cfg)
    de = encode_rows(rows_de, sp_ja, sp_vi, gvoc, cfg)

    model = CVAE(cfg, sp_ja.get_piece_size(), sp_vi.get_piece_size(), len(gvoc)).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"variant={cfg['variant']} | tham số: {n_par/1e6:.2f}M | device={device} | "
          f"train {len(tr):,} / dev {len(de):,} | vocab ja {sp_ja.get_piece_size()} "
          f"vi {sp_vi.get_piece_size()} gram {len(gvoc)}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n_batches = math.ceil(len(tr) / cfg["batch"])
    total_steps = cfg["epochs"] * n_batches
    warm = max(1, int(cfg["warmup_frac"] * total_steps))

    run = args.run or f"{cfg['variant']}_{int(time.time())}"
    rdir = Path(cfg["runs_dir"]) / run; rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    mcsv = open(rdir / "metrics.csv", "w", newline="", encoding="utf-8")
    mw = csv.writer(mcsv)
    mw.writerow(["step", "epoch", "phase", "ce_trans", "ce_gram", "kl",
                 "beta", "chrf", "f1_micro", "f1_macro", "au", "sec"])
    kcsv = open(rdir / "kl_dims.csv", "w", newline="", encoding="utf-8")
    kw = csv.writer(kcsv)
    kw.writerow(["epoch"] + [f"k{j}" for j in range(cfg["d_z"])])

    step, best, bad = 0, -1.0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); t0 = time.time()
        agg = defaultdict(float); cnt = 0
        for b in make_batches(tr, cfg["batch"], True, rng, device):
            step += 1
            beta = cfg["beta_max"] * min(1.0, step / warm)     # KL annealing (A6)
            out = model(b, p_wd=cfg["p_wd"], sample_z=True)
            loss = out["ce_trans"]
            kl_sum = 0.0
            if model.use_z:
                klv = out["klv"]
                kl_sum = klv.sum()
                assert kl_sum.item() > -1e-4, "KL ÂM — sai dấu công thức (A4)!"
                loss = loss + beta * torch.clamp(klv, min=cfg["free_bits"]).sum()  # free bits
            if model.use_gram:
                loss = loss + cfg["lam"] * out["ce_gram"]
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])   # chống cliff RNN
            opt.step()

            agg["ce_t"] += out["ce_trans"].item()
            if model.use_gram: agg["ce_g"] += out["ce_gram"].item()
            if model.use_z: agg["kl"] += kl_sum.item()   # .item() tránh UserWarning grad
            cnt += 1
            if step % cfg["log_every"] == 0:
                mw.writerow([step, epoch, "train",
                             f"{agg['ce_t']/cnt:.4f}",
                             f"{agg['ce_g']/cnt:.4f}" if model.use_gram else "",
                             f"{agg['kl']/cnt:.4f}" if model.use_z else "",
                             f"{beta:.3f}", "", "", "", "", ""])
                agg = defaultdict(float); cnt = 0

        ev = evaluate(model, de, sp_vi, inv_g, cfg, device)
        sec = time.time() - t0
        mw.writerow([step, epoch, "dev", f"{ev['ce_trans']:.4f}",
                     f"{ev['ce_gram']:.4f}", f"{ev['kl']:.4f}", f"{beta:.3f}",
                     f"{ev['chrf']:.2f}", f"{ev['f1_micro']:.4f}",
                     f"{ev['f1_macro']:.4f}", ev["au"], f"{sec:.1f}"])
        mcsv.flush()
        if ev["klv"] is not None:
            kw.writerow([epoch] + [f"{v:.4f}" for v in ev["klv"].tolist()]); kcsv.flush()

        score = monitor_score(ev, model.use_gram)
        ck = dict(model=model.state_dict(), cfg=cfg, gvoc=gvoc)
        torch.save(ck, rdir / "ckpt_last.pt")
        star = ""
        if score > best:
            best, bad = score, 0
            torch.save(ck, rdir / "ckpt_best.pt"); star = "  *best*"
        else:
            bad += 1
        print(f"epoch {epoch:3d} | dev ce_t {ev['ce_trans']:.3f} "
              f"ce_g {ev['ce_gram']:.3f} kl {ev['kl']:.2f} | "
              f"chrF {ev['chrf']:.1f} F1µ {ev['f1_micro']:.3f} AU {ev['au']} | "
              f"{sec:.0f}s{star}")
        if bad >= cfg["patience"]:
            print(f"Early stopping (patience {cfg['patience']}) — best={best:.2f}")
            break

    mcsv.close(); kcsv.close()
    print(f"\nXong. Kết quả trong {rdir}/ — metrics.csv + kl_dims.csv là đầu vào của plots.py.")

if __name__ == "__main__":
    main()
