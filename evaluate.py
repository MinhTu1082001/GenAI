#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py — chấm model trên dev (mặc định, dùng khi tune) hoặc test (MỘT LẦN).

    python evaluate.py runs/a_seq2seq runs/b_cvae1 runs/c_cvae2 --split dev
    python evaluate.py runs/c_cvae2 --split test --iwae-k 64

Xuất cho mỗi run vào runs/<run>/eval_<split>/:
    report.txt · per_pattern.csv · predictions.jsonl · mu_dump.csv · iwae_by_k.csv
Cộng bảng tổng hợp mọi run (+ hàng baseline rule-tagger nếu có fugashi)
→ eval_summary_<split>.csv

Số liệu:
    dịch      : chrF + BLEU (sacrebleu; thiếu thì fallback chrF nội bộ)
    ngữ pháp  : presence-F1 micro/macro trên TOÀN BỘ pattern và trên CORE SET
                (core = train≥30 & gold-test≥3, chỉ tính khi --split test)
    likelihood: NLL/token (teacher-forced, z=μ) · ELBO/token · IWAE_K/token (D7)
"""

from __future__ import annotations
import argparse, csv, json, math, sys
from collections import defaultdict
from pathlib import Path

import torch
import sentencepiece as spm

from model import CVAE, PAD, EOS
from train import encode_rows, make_batches, read_jsonl, chrf as chrf_internal

LOG2PI = math.log(2 * math.pi)


def load_run(run_dir: Path, device: str):
    ck = torch.load(run_dir / "ckpt_best.pt", map_location=device)
    cfg, gvoc = ck["cfg"], ck["gvoc"]
    art = Path(cfg["art_dir"])
    sp_ja = spm.SentencePieceProcessor(model_file=str(art / "spm_ja.model"))
    sp_vi = spm.SentencePieceProcessor(model_file=str(art / "spm_vi.model"))
    model = CVAE(cfg, sp_ja.get_piece_size(), sp_vi.get_piece_size(), len(gvoc)).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, cfg, gvoc, sp_ja, sp_vi


def pick_rows(split: str, data_dir: Path):
    if split == "test":
        print("※ NHẮC KỶ LUẬT: test chấm MỘT LẦN cho model cuối. Tune dùng --split dev.")
        gold = data_dir / "test_gold.jsonl"
        if gold.exists():
            return read_jsonl(gold), "gold"
        print("⚠ chưa có test_gold.jsonl — dùng test.jsonl (nhãn SILVER, chỉ tham khảo)")
        return read_jsonl(data_dir / "test.jsonl"), "silver"
    return read_jsonl(data_dir / "dev.jsonl"), "silver"


def row_level(r: dict) -> str:
    lv = {h.get("level") for h in r.get("patterns", [])}
    return "N4" if "N4" in lv else ("N5" if "N5" in lv else "none")


def grammar_table(preds: list[set], golds: list[set], core: set[str]):
    per = defaultdict(lambda: [0, 0, 0])                  # tag -> [tp, fp, fn]
    for p, g in zip(preds, golds):
        for t in p & g: per[t][0] += 1
        for t in p - g: per[t][1] += 1
        for t in g - p: per[t][2] += 1

    def agg(labels):
        tp = sum(per[t][0] for t in labels); fp = sum(per[t][1] for t in labels)
        fn = sum(per[t][2] for t in labels)
        micro = 2 * tp / max(2 * tp + fp + fn, 1)
        f1s = [2 * per[t][0] / max(2 * per[t][0] + per[t][1] + per[t][2], 1) for t in labels]
        return micro, (sum(f1s) / len(f1s) if f1s else float("nan"))

    all_labels = sorted(per)
    mi_all, ma_all = agg(all_labels)
    core_labels = sorted(t for t in all_labels if t in core)
    mi_c, ma_c = (agg(core_labels) if core_labels else (float("nan"),) * 2)
    return per, mi_all, ma_all, mi_c, ma_c


@torch.no_grad()
def logp_given_z(model, b, H, mask, summ, z):
    logits = model.dec_t.forward_teacher(b["y_in"], H, mask, summ, z, 0.0)
    lp = torch.log_softmax(logits, -1)
    tok = lp.gather(-1, b["y_out"].unsqueeze(-1)).squeeze(-1)
    m = b["y_out"].ne(PAD)
    return (tok * m).sum(1), m.sum(1)


@torch.no_grad()
def eval_run(run_dir: Path, rows, core: set[str], args, device):
    model, cfg, gvoc, sp_ja, sp_vi = load_run(run_dir, device)
    inv_g = {v: k for k, v in gvoc.items()}
    data = encode_rows(rows, sp_ja, sp_vi, gvoc, cfg)
    out_dir = run_dir / f"eval_{args.split}"; out_dir.mkdir(exist_ok=True)

    # ---------- 1. Greedy: dịch + tag + mu ----------
    hyps, refs, preds, golds, recs, mu_rows = [], [], [], [], [], []
    for b in make_batches(data, cfg["batch"], False, None, device):
        y, g, mu = model.predict(b["x"], b["lens"], cfg["max_vi"])
        for i, row_ids in enumerate(y.tolist()):
            ids = []
            for t in row_ids:
                if t == EOS: break
                if t != PAD: ids.append(t)
            hyps.append(sp_vi.decode(ids))
        refs += b["refs"]; golds += b["gold"]
        pr = [set() for _ in b["refs"]]
        if g is not None:
            for i, row_ids in enumerate(g.tolist()):
                for t in row_ids:
                    if t == EOS: break
                    tok = inv_g.get(t)
                    if tok and not tok.startswith("<"): pr[i].add(tok)
        preds += pr
        for i in range(len(b["refs"])):
            recs.append({"id": b["ids"][i], "ja": b["jas"][i], "ref": b["refs"][i],
                         "hyp": hyps[len(hyps) - len(b["refs"]) + i],
                         "gold": sorted(b["gold"][i]), "pred": sorted(pr[i])})
        if mu is not None:
            for i in range(len(b["refs"])):
                mu_rows.append([b["ids"][i]] + [f"{v:.4f}" for v in mu[i].tolist()])

    # ---------- 2. Dịch ----------
    try:
        import sacrebleu
        chrf_v = sacrebleu.corpus_chrf(hyps, [refs]).score
        bleu_v = sacrebleu.corpus_bleu(hyps, [refs]).score
        met_note = "sacrebleu"
    except ImportError:
        chrf_v, bleu_v, met_note = chrf_internal(hyps, refs), float("nan"), "chrF nội bộ (cài sacrebleu để có BLEU)"

    # ---------- 3. Ngữ pháp ----------
    per, mi_all, ma_all, mi_c, ma_c = grammar_table(preds, golds, core)

    # ---------- 4. Likelihood: NLL / ELBO / IWAE (D7) ----------
    Ks = sorted({k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= args.iwae_k} | {args.iwae_k})
    nll_sum = 0.0; tok_sum = 0; elbo_sum = 0.0
    iwae_sum = {k: 0.0 for k in Ks}
    for b in make_batches(data, cfg["batch"], False, None, device):
        H, summ, mu, logvar = model.enc(b["x"], b["lens"])
        mask = b["x"].ne(PAD)
        lp_mu, ntok = logp_given_z(model, b, H, mask, summ,
                                   mu if model.use_z else None)
        nll_sum += float(-lp_mu.sum()); tok_sum += int(ntok.sum())
        if model.use_z:
            kl_s = (0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)).sum(1)
            logws, logps = [], []
            for _ in range(args.iwae_k):
                eps = torch.randn_like(mu)
                z = mu + torch.exp(0.5 * logvar) * eps
                lp, _ = logp_given_z(model, b, H, mask, summ, z)
                logq = (-0.5 * (LOG2PI + logvar + eps.pow(2))).sum(1)
                logpz = (-0.5 * (LOG2PI + z.pow(2))).sum(1)
                logws.append(lp + logpz - logq); logps.append(lp)
            W = torch.stack(logws, 1)                       # (B, K)
            for k in Ks:
                iwae_sum[k] += float((torch.logsumexp(W[:, :k], 1) - math.log(k)).sum())
            elbo_sum += float((torch.stack(logps, 1).mean(1) - kl_s).sum())
    nll_tok = nll_sum / tok_sum
    elbo_tok = elbo_sum / tok_sum if model.use_z else float("nan")
    iwae_tok = {k: v / tok_sum for k, v in iwae_sum.items()} if model.use_z else {}

    # ---------- 5. Ghi file ----------
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "per_pattern.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["pattern", "tp", "fp", "fn", "P", "R", "F1", "core"])
        for t in sorted(per):
            tp, fp, fn = per[t]
            P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
            F = 2 * P * R / max(P + R, 1e-9)
            w.writerow([t, tp, fp, fn, f"{P:.3f}", f"{R:.3f}", f"{F:.3f}",
                        "yes" if t in core else "no"])
    if mu_rows:
        # mu_rows theo thứ tự BATCH (đã sort độ dài) ≠ thứ tự `rows` gốc → KHÔNG zip
        # thẳng, phải tra level qua id (mr[0]) kẻo fig7 t-SNE tô màu sai câu.
        level_of = {r["id"]: row_level(r) for r in rows}
        with open(out_dir / "mu_dump.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "level"] + [f"m{j}" for j in range(len(mu_rows[0]) - 1)])
            for mr in mu_rows:
                w.writerow([mr[0], level_of.get(mr[0], "none")] + mr[1:])
    if iwae_tok:
        with open(out_dir / "iwae_by_k.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["K", "bound_per_token"])
            for k in Ks:
                w.writerow([k, f"{iwae_tok[k]:.4f}"])

    res = dict(run=run_dir.name, chrf=chrf_v, bleu=bleu_v,
               f1_micro=mi_all, f1_macro=ma_all, f1_micro_core=mi_c, f1_macro_core=ma_c,
               nll_tok=nll_tok, elbo_tok=elbo_tok,
               iwae_tok=iwae_tok.get(args.iwae_k, float("nan")))
    rep = [f"run: {run_dir.name} | split: {args.split} | metric dịch: {met_note}",
           f"chrF {chrf_v:.2f} | BLEU {bleu_v:.2f}",
           f"F1 micro/macro (ALL)  : {mi_all:.4f} / {ma_all:.4f}",
           f"F1 micro/macro (CORE) : {mi_c:.4f} / {ma_c:.4f}  (core={len(core)} pattern)",
           f"NLL/token (z=μ) {nll_tok:.4f} | ELBO/token {elbo_tok:.4f} | "
           f"IWAE_{args.iwae_k}/token {res['iwae_tok']:.4f}",
           "(kỳ vọng lý thuyết: ELBO ≤ IWAE, bound tăng theo K — xem iwae_by_k.csv)"]
    (out_dir / "report.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep) + "\n")
    return res


def tagger_baseline(rows, core):
    try:
        from tagger import load_patterns, to_toks, find_patterns
        import fugashi
    except (ImportError, SystemExit):
        print("(bỏ qua baseline tagger — cần fugashi + patterns.yaml)")
        return None
    tg = fugashi.Tagger(); rules = load_patterns()
    preds = [{h["id"] for h in find_patterns(to_toks(tg(r["ja"])), rules)} for r in rows]
    golds = [{h["id"] for h in r.get("patterns", [])} for r in rows]
    _, mi, ma, mic, mac = grammar_table(preds, golds, core)
    return dict(run="tagger(rule)", chrf=float("nan"), bleu=float("nan"),
                f1_micro=mi, f1_macro=ma, f1_micro_core=mic, f1_macro_core=mac,
                nll_tok=float("nan"), elbo_tok=float("nan"), iwae_tok=float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--iwae-k", type=int, default=64)
    ap.add_argument("--data-dir", default="data/splits")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    rows, label_kind = pick_rows(args.split, data_dir)
    core: set[str] = set()
    if args.split == "test":
        tr_cnt = defaultdict(int)
        for r in read_jsonl(data_dir / "train.jsonl"):
            for pid in {h["id"] for h in r.get("patterns", [])}:
                tr_cnt[pid] += 1
        te_cnt = defaultdict(int)
        for r in rows:
            for pid in {h["id"] for h in r.get("patterns", [])}:
                te_cnt[pid] += 1
        core = {p for p in tr_cnt if tr_cnt[p] >= 30 and te_cnt[p] >= 3}
        print(f"Nhãn: {label_kind} | CORE SET = {len(core)} pattern (train≥30 & test≥3)\n")

    results = []
    for rd in args.runs:
        results.append(eval_run(Path(rd), rows, core, args, device))
    tb = tagger_baseline(rows, core)
    if tb:
        results.append(tb)

    cols = ["run", "chrf", "bleu", "f1_micro", "f1_macro",
            "f1_micro_core", "f1_macro_core", "nll_tok", "elbo_tok", "iwae_tok"]
    out = Path(f"eval_summary_{args.split}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            w.writerow([r["run"]] + [f"{r[c]:.4f}" if isinstance(r[c], float) else r[c]
                                     for c in cols[1:]])
    print(f"{'run':<16}{'chrF':>7}{'BLEU':>7}{'F1µ':>7}{'F1M':>7}{'F1µc':>7}{'NLL':>8}")
    for r in results:
        print(f"{r['run']:<16}{r['chrf']:>7.2f}{r['bleu']:>7.2f}{r['f1_micro']:>7.3f}"
              f"{r['f1_macro']:>7.3f}{r['f1_micro_core']:>7.3f}{r['nll_tok']:>8.3f}")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
