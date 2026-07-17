#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gold_ingest.py — nhập gold_annotation.csv (đã sửa tay trong Sheets/Excel)
→ data/splits/test_gold.jsonl.

Xác thực ba tầng trước khi nhận:
  1. Checksum: tập id trong CSV phải khớp test_checksum.txt (bằng chứng test chưa bị đụng)
  2. Mọi pattern id trong gold_patterns phải có trong patterns.yaml (bắt lỗi gõ)
  3. Đủ số dòng so với test.jsonl

Bonus: in preview P/R/F1 của rule tagger so với gold (so cột tagger_patterns vs
gold_patterns) — bản nháp của hàng baseline; số chính thức chạy evaluate.py.

Chạy:  python gold_ingest.py
"""

from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path

import yaml

SPLITS = Path("data/splits")
CSV_IN = SPLITS / "gold_annotation.csv"
TEST = SPLITS / "test.jsonl"
CHECK = SPLITS / "test_checksum.txt"
OUT = SPLITS / "test_gold.jsonl"
PATTERNS = Path("patterns.yaml")


def parse_tags(cell: str, split_span: bool = False) -> set[str]:
    out = set()
    for seg in (cell or "").split("|"):
        seg = seg.strip()
        if not seg:
            continue
        out.add(seg.split(":")[0].strip() if split_span else seg)
    return out


def main() -> None:
    for p in (CSV_IN, TEST, CHECK, PATTERNS):
        if not p.exists():
            sys.exit(f"Thiếu {p} — kiểm tra lại pipeline.")
    levels = {p["id"]: p["level"] for p in yaml.safe_load(PATTERNS.read_text(encoding="utf-8"))}
    test_rows = {json.loads(l)["id"]: json.loads(l) for l in open(TEST, encoding="utf-8")}

    try:
        rd = list(csv.DictReader(open(CSV_IN, encoding="utf-8-sig")))
    except UnicodeDecodeError:
        sys.exit("CSV không phải UTF-8 — trong Excel chọn 'Save As → CSV UTF-8' rồi chạy lại.")

    ids = [r["id"] for r in rd]
    if set(ids) != set(test_rows):
        extra = set(ids) - set(test_rows); missing = set(test_rows) - set(ids)
        sys.exit(f"Tập id CSV lệch test.jsonl (thừa {len(extra)}, thiếu {len(missing)}) — "
                 "KHÔNG được thêm/xóa dòng của gold_annotation.csv.")
    digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    frozen = CHECK.read_text(encoding="utf-8").strip()
    if digest != frozen:
        sys.exit("Checksum id KHÔNG khớp test_checksum.txt — test set đã bị thay đổi. Dừng.")

    unknown = sorted({t for r in rd for t in parse_tags(r.get("gold_patterns", ""))
                      if t not in levels})
    if unknown:
        sys.exit("gold_patterns chứa id không có trong patterns.yaml (lỗi gõ?): "
                 + ", ".join(unknown))

    out_rows, n_vi_fixed, n_notes = [], 0, 0
    tp = fp = fn = 0
    for r in rd:
        gold = parse_tags(r.get("gold_patterns", ""))
        tagg = parse_tags(r.get("tagger_patterns", ""), split_span=True)
        tp += len(tagg & gold); fp += len(tagg - gold); fn += len(gold - tagg)
        vi = (r.get("vi_sua") or "").strip() or r["vi"]
        if (r.get("vi_sua") or "").strip():
            n_vi_fixed += 1
        if (r.get("ghi_chu") or "").strip():
            n_notes += 1
        out_rows.append({"id": r["id"], "ja": r["ja"], "vi": vi,
                         "origin": test_rows[r["id"]].get("origin", ""),
                         "patterns": [{"id": t, "level": levels[t], "span": ""}
                                      for t in sorted(gold)]})

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    print(f"✓ {OUT} — {len(out_rows)} câu | vi sửa tay: {n_vi_fixed} | ghi chú: {n_notes}")
    print(f"Preview baseline TAGGER vs GOLD (micro): P={P:.3f} R={R:.3f} F1={F1:.3f}")
    print("(số chính thức + per-pattern: python evaluate.py ... --split test)")


if __name__ == "__main__":
    main()
