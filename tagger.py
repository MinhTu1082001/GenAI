#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tagger.py — Rule tagger ngữ pháp N5–N4 trên hình vị fugashi/UniDic.

Ba vai trò trong pipeline:
  (a) MINER   : lọc pool.jsonl lấy câu chứa pattern + giữ negatives
  (b) LABELER : silver label cho train, pre-annotate cho gold set
  (c) BASELINE: một hàng trong bảng ablation, chấm F1 trên gold như mọi model

Cài đặt (Colab):  !pip install "fugashi[unidic-lite]" pyyaml --quiet

Dùng theo thứ tự:
  python tagger.py --selftest                      # BẮT BUỘC chạy trước tiên
  python tagger.py --debug "買ったばかりです。"      # xem bảng hình vị khi viết rule
  python tagger.py --mine                          # pool.jsonl → mined.jsonl + coverage.csv
"""

from __future__ import annotations
import argparse, csv, json, random, sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit('Thiếu pyyaml — chạy:  pip install pyyaml')
try:
    import fugashi
except ImportError:
    sys.exit('Thiếu fugashi — chạy:  pip install "fugashi[unidic-lite]"')

PATTERNS_PATH = Path("patterns.yaml")
POOL_PATH     = Path("data/processed/pool.jsonl")
MINED_PATH    = Path("data/processed/mined.jsonl")
COVER_PATH    = Path("data/processed/coverage.csv")
NEG_FRACTION  = 0.15   # số negative giữ lại = 15% số câu có pattern
SEED = 13

# ---------------------------------------------------------------- Token
@dataclass
class Tok:
    surface: str
    lemma: str
    pos: str    # pos1-pos2-... nối bằng '-'
    cform: str  # 活用形 ('' nếu không có)

def to_toks(words) -> list[Tok]:
    out = []
    for w in words:
        f = w.feature
        pos = "-".join(p for p in (getattr(f, "pos1", None), getattr(f, "pos2", None),
                                   getattr(f, "pos3", None), getattr(f, "pos4", None))
                       if p and p != "*")
        lemma = getattr(f, "lemma", None) or w.surface
        cform = getattr(f, "cForm", None) or ""
        out.append(Tok(w.surface, lemma, pos, "" if cform == "*" else cform))
    return out

# ---------------------------------------------------------------- Matcher
def _aslist(x):
    return x if isinstance(x, list) else [x]

def tok_match(t: Tok, cond: dict) -> bool:
    if "surface" in cond and t.surface not in _aslist(cond["surface"]):
        return False
    if "lemma" in cond and t.lemma not in _aslist(cond["lemma"]):
        return False
    if "pos" in cond and not any(t.pos.startswith(p) for p in _aslist(cond["pos"])):
        return False
    if "cform" in cond and not any(t.cform.startswith(c) for c in _aslist(cond["cform"])):
        return False
    return True

def match_at(toks: list[Tok], i: int, seq: list[dict]) -> int | None:
    """Thử khớp dãy điều kiện bắt đầu tại token i. Trả về chỉ số kết thúc (exclusive)."""
    j = i
    for cond in seq:
        if j < len(toks) and tok_match(toks[j], cond):
            j += 1
        elif cond.get("opt"):
            continue
        else:
            return None
    return j

def find_patterns(toks: list[Tok], patterns: list[dict]) -> list[dict]:
    hits = []
    for p in patterns:
        for i in range(len(toks)):
            j = match_at(toks, i, p["seq"])
            if j is not None:
                hits.append({"id": p["id"], "level": p["level"],
                             "span": "".join(t.surface for t in toks[i:j])})
    return hits

# ---------------------------------------------------------------- Nạp patterns
def load_patterns() -> list[dict]:
    if not PATTERNS_PATH.exists():
        sys.exit(f"Không thấy {PATTERNS_PATH} — đặt patterns.yaml cạnh tagger.py")
    pats = yaml.safe_load(PATTERNS_PATH.read_text(encoding="utf-8"))
    ids = set()
    for p in pats:
        for k in ("id", "pattern", "level", "seq"):
            if k not in p:
                sys.exit(f"Pattern thiếu trường '{k}': {p}")
        if p["id"] in ids:
            sys.exit(f"Trùng id pattern: {p['id']}")
        ids.add(p["id"])
    print(f"Đã nạp {len(pats)} pattern từ {PATTERNS_PATH}")
    return pats

# ---------------------------------------------------------------- Debug
def dump_toks(toks: list[Tok]) -> None:
    print(f"  {'surface':<12}{'lemma':<12}{'cform':<18}pos")
    for t in toks:
        print(f"  {t.surface:<12}{t.lemma:<12}{t.cform:<18}{t.pos}")

def debug(sent: str, patterns: list[dict], tg) -> None:
    toks = to_toks(tg(sent))
    print(f"\nCÂU: {sent}")
    dump_toks(toks)
    hits = find_patterns(toks, patterns)
    if hits:
        print("KHỚP:", ", ".join(f"{h['id']}({h['span']})" for h in hits))
    else:
        print("KHỚP: (không có)")

# ---------------------------------------------------------------- Selftest
# Mỗi lần thêm pattern mới vào YAML → thêm ít nhất một câu test vào đây.
SELFTEST: list[tuple[str, set[str]]] = [
    ("昨日買ったばかりのパソコンが壊れてしまいました。", {"ta_bakari", "te_shimau"}),
    ("日本語を毎日勉強しています。",                     {"te_iru"}),
    ("明日までに宿題をしなければなりません。",           {"nakereba_naranai"}),
    ("音楽を聞きながら走るのが好きです。",               {"nagara"}),
    ("富士山に登ったことがありますか。",                 {"ta_koto_ga_aru"}),
    ("疲れているなら、早く寝たほうがいいですよ。",       {"te_iru", "hou_ga_ii"}),
    ("これはペンです。",                                 set()),
]

def selftest(patterns: list[dict], tg) -> None:
    fails = 0
    for sent, expected in SELFTEST:
        toks = to_toks(tg(sent))
        found = {h["id"] for h in find_patterns(toks, patterns)}
        missing, extra = expected - found, found - expected
        if missing:
            fails += 1
            print(f"FAIL  {sent}\n      thiếu: {sorted(missing)} | thấy: {sorted(found)}")
            dump_toks(toks)   # nhìn bảng này để sửa seq trong YAML
        else:
            mark = f"  (⚠ thêm: {sorted(extra)})" if extra else ""
            print(f"PASS  {sent}  → {sorted(found)}{mark}")
    print("\nKẾT QUẢ:", "TẤT CẢ PASS ✓" if fails == 0 else f"{fails} câu FAIL — sửa YAML rồi chạy lại")
    sys.exit(0 if fails == 0 else 1)

# ---------------------------------------------------------------- Mine
def mine(patterns: list[dict], tg) -> None:
    if not POOL_PATH.exists():
        sys.exit(f"Không thấy {POOL_PATH} — chạy prepare_data.py trước.")
    random.seed(SEED)
    pos_rows, neg_rows = [], []
    cover: dict[str, Counter] = defaultdict(Counter)  # pattern -> origin -> số CÂU
    n = 0
    with open(POOL_PATH, encoding="utf-8") as f:
        for line in f:
            n += 1
            if n % 20000 == 0:
                print(f"  … đã quét {n:,} câu")
            row = json.loads(line)
            toks = to_toks(tg(row["ja"]))
            hits = find_patterns(toks, patterns)
            if hits:
                row["patterns"] = hits
                pos_rows.append(row)
                for pid in {h["id"] for h in hits}:
                    cover[pid][row["origin"]] += 1
            else:
                neg_rows.append(row)

    k = min(len(neg_rows), int(NEG_FRACTION * len(pos_rows)))
    negs = random.sample(neg_rows, k) if k else []
    for r in negs:
        r["patterns"] = []
    MINED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MINED_PATH, "w", encoding="utf-8") as f:
        for r in pos_rows + negs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    origins = sorted({o for c in cover.values() for o in c})
    with open(COVER_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pattern"] + origins + ["TOTAL"])
        for pid in sorted(cover, key=lambda p: -sum(cover[p].values())):
            vals = [cover[pid].get(o, 0) for o in origins]
            w.writerow([pid] + vals + [sum(vals)])

    print(f"\npool: {n:,} câu | có pattern: {len(pos_rows):,} | negative giữ lại: {k:,}")
    print(f"→ {MINED_PATH}\n→ {COVER_PATH}  (ma trận pattern × nguồn)")
    print("\nPattern ÍT câu nhất (lỗ cần khung câu synthetic ở bước 8):")
    for pid in sorted(cover, key=lambda p: sum(cover[p].values()))[:10]:
        print(f"  {pid:<22}{sum(cover[pid].values()):>7,} câu")
    zero = [p["id"] for p in patterns if p["id"] not in cover]
    if zero:
        print("\nPattern 0 hit (nghi rule sai — kiểm bằng --debug với câu tự đặt):",
              ", ".join(zero))

# ---------------------------------------------------------------- Main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--debug", metavar="CÂU")
    ap.add_argument("--mine", action="store_true")
    args = ap.parse_args(["--mine"])

    pats = load_patterns()
    tg = fugashi.Tagger()
    if args.selftest:
        selftest(pats, tg)
    elif args.debug:
        debug(args.debug, pats, tg)
    elif args.mine:
        mine(pats, tg)
    else:
        ap.print_help()