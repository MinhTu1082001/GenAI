#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo.py — sản phẩm cuối: câu Nhật vào → JSON dịch + chú giải ngữ pháp.

Phân công ba tầng (thiết kế B2, chống hallucination by design):
    MODEL  quyết CÓ pattern gì   (decoder ngữ pháp, chỉ nhìn z)
    RULES  quyết Ở ĐÂU           (matcher của tagger định vị span — cần fugashi)
    TEMPLATE quyết GIẢNG THẾ NÀO (templates.csv; thiếu thì fallback note trong patterns.yaml)

Dùng:
    python demo.py --run runs/c_cvae2 "昨日買ったばかりのパソコンが壊れてしまいました。"
    python demo.py --run runs/c_cvae2            (chế độ gõ từng câu, 'q' để thoát)

templates.csv (tùy chọn, UTF-8): pattern_id, meaning, explanation, common_mistake
"""

from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import sentencepiece as spm
import yaml

from model import CVAE, PAD, EOS


def load_run(run_dir: Path, device: str):
    ck = torch.load(run_dir / "ckpt_best.pt", map_location=device)
    cfg, gvoc = ck["cfg"], ck["gvoc"]
    art = Path(cfg["art_dir"])
    sp_ja = spm.SentencePieceProcessor(model_file=str(art / "spm_ja.model"))
    sp_vi = spm.SentencePieceProcessor(model_file=str(art / "spm_vi.model"))
    model = CVAE(cfg, sp_ja.get_piece_size(), sp_vi.get_piece_size(), len(gvoc)).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, cfg, gvoc, sp_ja, sp_vi


def load_knowledge():
    """patterns.yaml (tên hiển thị, level, note) + templates.csv (lời giảng)."""
    pats = {p["id"]: p for p in yaml.safe_load(Path("patterns.yaml").read_text(encoding="utf-8"))}
    tpl = {}
    tfile = Path("templates.csv")
    if tfile.exists():
        for r in csv.DictReader(open(tfile, encoding="utf-8-sig")):
            tpl[r["pattern_id"].strip()] = r
    return pats, tpl


def make_span_locator():
    try:
        from tagger import load_patterns, to_toks, match_at
        import fugashi
        tg = fugashi.Tagger(); rules = {p["id"]: p for p in load_patterns()}

        def locate(ja: str, pid: str) -> str | None:
            rule = rules.get(pid)
            if not rule:
                return None
            toks = to_toks(tg(ja))
            for i in range(len(toks)):
                j = match_at(toks, i, rule["seq"])
                if j is not None:
                    return "".join(t.surface for t in toks[i:j])
            return None
        return locate
    except (ImportError, SystemExit):
        return None


@torch.no_grad()
def analyze(ja: str, model, cfg, inv_g, sp_ja, sp_vi, pats, tpl, locate, device):
    ids = sp_ja.encode(ja, out_type=int)[: cfg["max_ja"]] + [EOS]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    lens = torch.tensor([len(ids)])
    y, g, _ = model.predict(x, lens, cfg["max_vi"])

    vi_ids = []
    for t in y[0].tolist():
        if t == EOS: break
        if t != PAD: vi_ids.append(t)
    translation = sp_vi.decode(vi_ids)

    pred = []
    if g is not None:
        for t in g[0].tolist():
            if t == EOS: break
            tok = inv_g.get(t)
            if tok and not tok.startswith("<") and tok not in pred:
                pred.append(tok)

    points = []
    for pid in sorted(pred):
        p = pats.get(pid, {})
        t = tpl.get(pid, {})
        points.append({
            "id": pid,
            "pattern": p.get("pattern", pid),
            "level": p.get("level", "?"),
            "in_sentence": locate(ja, pid) if locate else None,
            "meaning": (t.get("meaning") or p.get("note") or "").strip(),
            "explanation": (t.get("explanation") or "(chưa có template — bổ sung templates.csv)").strip(),
        })
        if (t.get("common_mistake") or "").strip():
            points[-1]["common_mistake"] = t["common_mistake"].strip()

    return {"input": ja, "translation": translation, "grammar_points": points}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/c_cvae2")
    ap.add_argument("--device", default=None)
    ap.add_argument("sentence", nargs="?")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, gvoc, sp_ja, sp_vi = load_run(Path(args.run), device)
    if not model.use_gram:
        print("⚠ run này không có decoder ngữ pháp (variant != cvae2) — chỉ dịch.")
    inv_g = {v: k for k, v in gvoc.items()}
    pats, tpl = load_knowledge()
    locate = make_span_locator()
    if locate is None:
        print("(không có fugashi → JSON sẽ thiếu in_sentence; pip install fugashi[unidic-lite])")

    def run_one(s: str):
        out = analyze(s.strip(), model, cfg, inv_g, sp_ja, sp_vi, pats, tpl, locate, device)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.sentence:
        run_one(args.sentence)
    else:
        print("Gõ câu tiếng Nhật (q để thoát):")
        for line in sys.stdin:
            s = line.strip()
            if s.lower() in ("q", "quit", "exit"):
                break
            if s:
                run_one(s)


if __name__ == "__main__":
    main()
