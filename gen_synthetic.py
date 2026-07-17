#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_synthetic.py — Bước (8): sinh cặp câu song ngữ từ khung để LẤP LỖ coverage.

Neo lý thuyết: dataset augmentation (Goodfellow §7.4); nhãn đúng 100% by construction.
Kỷ luật đã chốt:
  - synthetic CHỈ vào data/splits/train.jsonl (origin="synth")
  - tổng synthetic ≤ SYNTH_MAX_FRAC của train cuối
  - idempotent: mỗi lần chạy XÓA toàn bộ synth cũ rồi sinh lại (train_natural.jsonl
    được backup một lần, không bao giờ bị sửa)
Chạy SAU split_data.py:   python gen_synthetic.py
Sau khi chạy: XÓA artifacts/ trước chiến dịch train kế tiếp (vocab phải build lại).
"""

from __future__ import annotations
import itertools, json, random, re, sys
from collections import Counter
from pathlib import Path

import yaml

TRAIN = Path("data/splits/train.jsonl")
NATURAL_BACKUP = Path("data/splits/train_natural.jsonl")
FRAMES = Path("frames.yaml")
PATTERNS = Path("patterns.yaml")

TARGET_PER_PATTERN = 30    # khớp CORE_MIN_TRAIN của split_data.py
SYNTH_MAX_FRAC = 0.40      # synth ≤ 40% train CUỐI  →  synth ≤ 0.4/0.6 × natural
MAX_PER_FRAME = 80         # trần tổ hợp mỗi khung (tránh một khung nuốt hết quota)
SEED = 13

PLACEHOLDER = re.compile(r"\{([A-Za-z_]\w*)\.([A-Za-z_]\w*)\}")


def render(tpl: str, choice: dict[str, dict]) -> str:
    def sub(m):
        slot, field = m.group(1), m.group(2)
        entry = choice[slot]
        if field not in entry:
            sys.exit(f"Frame dùng {{{slot}.{field}}} nhưng entry {entry} thiếu trường '{field}' "
                     f"— bổ sung vào slots trong {FRAMES}")
        return str(entry[field])
    return PLACEHOLDER.sub(sub, tpl)


def expand_frame(frame: dict, slots: dict, rng: random.Random) -> list[tuple[str, str]]:
    names = sorted({m.group(1) for m in PLACEHOLDER.finditer(frame["ja"] + frame["vi"])})
    for n in names:
        if n not in slots:
            sys.exit(f"Frame {frame['id']} dùng slot '{n}' không có trong slots")
    combos = list(itertools.product(*[slots[n] for n in names]))
    rng.shuffle(combos)
    out = []
    for combo in combos[:MAX_PER_FRAME]:
        choice = dict(zip(names, combo))
        out.append((render(frame["ja"], choice), render(frame["vi"], choice)))
    return out


def main() -> None:
    if not TRAIN.exists():
        sys.exit("Chưa thấy data/splits/train.jsonl — chạy split_data.py trước.")
    rng = random.Random(SEED)
    frames_doc = yaml.safe_load(FRAMES.read_text(encoding="utf-8"))
    slots, frames = frames_doc["slots"], frames_doc["frames"]
    pats = yaml.safe_load(PATTERNS.read_text(encoding="utf-8"))
    levels = {p["id"]: p["level"] for p in pats}
    for f in frames:
        for pid in f["patterns"]:
            if pid not in levels:
                sys.exit(f"Frame {f['id']} khai pattern '{pid}' không có trong patterns.yaml")

    rows = [json.loads(l) for l in open(TRAIN, encoding="utf-8")]
    natural = [r for r in rows if r.get("origin") != "synth"]
    if not NATURAL_BACKUP.exists():          # backup MỘT lần, bất biến
        with open(NATURAL_BACKUP, "w", encoding="utf-8") as f:
            for r in natural:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = Counter(pid for r in natural for pid in {h["id"] for h in r.get("patterns", [])})
    deficit = {p: max(0, TARGET_PER_PATTERN - counts[p]) for p in levels}
    max_synth = int(SYNTH_MAX_FRAC / (1 - SYNTH_MAX_FRAC) * len(natural))

    # kho tổ hợp theo frame + chống trùng với câu tự nhiên
    seen_ja = {r["ja"] for r in natural}
    pools = {f["id"]: expand_frame(f, slots, rng) for f in frames}
    by_pattern: dict[str, list[dict]] = {}
    for f in frames:
        for pid in f["patterns"]:
            by_pattern.setdefault(pid, []).append(f)

    synth, added = [], Counter()
    for pid in sorted(deficit, key=lambda p: -deficit[p]):     # thiếu nhiều lấp trước
        if pid not in by_pattern:
            continue
        for f in by_pattern[pid]:
            while deficit[pid] > 0 and pools[f["id"]] and len(synth) < max_synth:
                ja, vi = pools[f["id"]].pop()
                if ja in seen_ja:
                    continue
                seen_ja.add(ja)
                synth.append({
                    "id": f"synth:{f['id']}:{len(synth)}", "ja": ja, "vi": vi,
                    "origin": "synth",
                    "patterns": [{"id": q, "level": levels[q], "span": ""}
                                 for q in f["patterns"]]})
                for q in f["patterns"]:
                    deficit[q] -= 1
                    added[q] += 1
            if deficit[pid] <= 0 or len(synth) >= max_synth:
                break

    # ---- kiểm chéo bằng tagger (tùy chọn — cần fugashi) ----
    try:
        from tagger import load_patterns, to_toks, find_patterns
        import fugashi
        tg = fugashi.Tagger()
        rules = load_patterns()
        miss = []
        for r in synth:
            found = {h["id"] for h in find_patterns(to_toks(tg(r["ja"])), rules)}
            want = {h["id"] for h in r["patterns"]}
            if not want <= found:
                miss.append((r["ja"], sorted(want - found)))
        if miss:
            print(f"⚠ {len(miss)} câu synth mà tagger KHÔNG thấy pattern đã khai "
                  f"(khung hoặc rule sai) — 3 ví dụ đầu:")
            for ja, m in miss[:3]:
                print(f"   {ja}  thiếu {m}")
        else:
            print("Kiểm chéo tagger: mọi câu synth đều khớp nhãn khai ✓")
    except (ImportError, SystemExit):
        print("(bỏ qua kiểm chéo tagger — cần `pip install fugashi[unidic-lite]`)")

    with open(TRAIN, "w", encoding="utf-8") as f:
        for r in natural + synth:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\ntrain.jsonl = {len(natural):,} tự nhiên + {len(synth):,} synth "
          f"({100*len(synth)/max(len(natural)+len(synth),1):.1f}% ≤ {int(SYNTH_MAX_FRAC*100)}%)")
    print(f"{'pattern':<22}{'trước':>7}{'thêm':>6}{'sau':>6}")
    for p in sorted(levels):
        if added[p] or counts[p]:
            print(f"{p:<22}{counts[p]:>7}{added[p]:>6}{counts[p]+added[p]:>6}")
    still = [p for p in levels if counts[p] + added[p] < TARGET_PER_PATTERN and p in by_pattern]
    if still:
        print("Vẫn dưới target (thêm entry vào slots hoặc thêm frame):", ", ".join(still))
    no_frame = [p for p, d in deficit.items() if d > 0 and p not in by_pattern]
    if no_frame:
        print("Pattern thiếu mà CHƯA có frame nào:", ", ".join(sorted(no_frame)))
    print("\nNHẮC: xóa artifacts/ trước chiến dịch train kế (vocab phải build lại trên train mới).")


if __name__ == "__main__":
    main()
