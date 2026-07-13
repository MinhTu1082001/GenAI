#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_data.py — Bước (9): chia TRAIN/DEV/TEST + chuẩn bị gắn nhãn gold + ĐÓNG BĂNG.

CHẠY THỬ: lúc nào cũng được (idempotent).
CHẠY THẬT (rồi đóng băng): CHỈ sau khi patterns.yaml hoàn chỉnh và
`python tagger.py --mine` đã chạy lại lần cuối. Bắt đầu sửa gold_annotation.csv
= đóng băng: không re-split, không thêm pattern id, không đụng test.jsonl nữa.

Đầu vào : data/processed/mined.jsonl
Đầu ra  : data/splits/train.jsonl, dev.jsonl, test.jsonl
          data/splits/gold_annotation.csv  — mở bằng Google Sheets, sửa tay
          data/splits/README_gold.txt      — giao thức gắn nhãn (đọc TRƯỚC khi sửa)
          data/splits/split_report.txt     — thống kê + CORE PATTERN SET
          data/splits/test_checksum.txt    — sha256 danh sách id test (bằng chứng đóng băng)

Ghi chú: dữ liệu synthetic (bước 8) sau này CHỈ append vào train.jsonl.
"""

from __future__ import annotations
import csv, hashlib, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

MINED = Path("data/processed/mined.jsonl")
OUT   = Path("data/splits")

# ---------------- Tham số (chốt trước khi chạy thật) ----------------
TEST_SIZE  = 180
DEV_SIZE   = 400
MIN_TEST_PER_PATTERN = 3      # mục tiêu tối thiểu mỗi pattern trong test
TEST_NEG_FRACTION    = 0.10   # ~10% test là câu KHÔNG có pattern (đo precision)
TEST_EXCLUDE_ORIGINS = {"alt"}  # register báo chí + tránh leakage theo bài viết
ORIGIN_RANK = {"tatoeba": 0, "tatoeba_pivot": 1, "ted2020": 2,
               "opensubtitles": 3, "alt": 9}
JA_LEN_PREF   = 28            # test ưu tiên câu quanh độ dài này (register N5–N4)
JACCARD_DROP  = 0.60          # train bị xóa nếu giống test/dev từ mức này
CORE_MIN_TRAIN, CORE_MIN_TEST = 30, 3   # quy tắc CORE PATTERN SET (chốt trước!)
SEED = 13

# ---------------- Tiện ích ----------------
def ids_of(r: dict) -> set[str]:
    return {h["id"] for h in r.get("patterns", [])}

def trigrams(s: str) -> set[str]:
    return {s[i:i + 3] for i in range(max(len(s) - 2, 1))}

def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------- Main ----------------
def main() -> None:
    if not MINED.exists():
        sys.exit("Không thấy mined.jsonl — chạy `python tagger.py --mine` trước.")
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in open(MINED, encoding="utf-8")]
    OUT.mkdir(parents=True, exist_ok=True)

    pos = [r for r in rows if r.get("patterns")]
    neg = [r for r in rows if not r.get("patterns")]
    by_pat: dict[str, list[dict]] = defaultdict(list)
    for r in pos:
        for pid in ids_of(r):
            by_pat[pid].append(r)
    freq = {pid: len(v) for pid, v in by_pat.items()}

    # ---------- 1. Chọn TEST: tham lam, pattern hiếm trước ----------
    n_neg_test = int(TEST_SIZE * TEST_NEG_FRACTION)
    n_pos_test = TEST_SIZE - n_neg_test

    def test_score(r: dict) -> tuple:
        return (ORIGIN_RANK.get(r["origin"], 5), abs(len(r["ja"]) - JA_LEN_PREF))

    chosen: list[dict] = []
    chosen_ids: set[str] = set()
    covered: Counter = Counter()

    def take(r: dict) -> None:
        chosen.append(r)
        chosen_ids.add(r["id"])
        for pid in ids_of(r):
            covered[pid] += 1

    for pid in sorted(freq, key=lambda p: freq[p]):          # hiếm trước
        need = MIN_TEST_PER_PATTERN - covered[pid]
        if need <= 0 or len(chosen) >= n_pos_test:
            continue
        cands = sorted((r for r in by_pat[pid]
                        if r["id"] not in chosen_ids
                        and r["origin"] not in TEST_EXCLUDE_ORIGINS),
                       key=test_score)
        for r in cands[:need]:
            if len(chosen) >= n_pos_test:
                break
            take(r)

    # lấp chỗ trống còn lại: ưu tiên câu mang pattern đang ít được phủ
    rest = [r for r in pos if r["id"] not in chosen_ids
            and r["origin"] not in TEST_EXCLUDE_ORIGINS]
    rng.shuffle(rest)
    rest.sort(key=lambda r: (min((covered[p] for p in ids_of(r)), default=99),
                             test_score(r)))
    for r in rest:
        if len(chosen) >= n_pos_test:
            break
        take(r)

    negs_ok = [r for r in neg if r["origin"] not in TEST_EXCLUDE_ORIGINS]
    test_rows = chosen + rng.sample(negs_ok, min(n_neg_test, len(negs_ok)))
    test_ids = {r["id"] for r in test_rows}

    # ---------- 2. DEV: lấy ngẫu nhiên từ phần còn lại ----------
    remain = [r for r in rows if r["id"] not in test_ids]
    dev_rows = rng.sample(remain, min(DEV_SIZE, len(remain)))
    dev_ids = {r["id"] for r in dev_rows}

    # ---------- 3. TRAIN + chốt chặn leakage trigram ----------
    train_rows = [r for r in remain if r["id"] not in dev_ids]
    df: Counter = Counter()
    for r in train_rows:
        for g in trigrams(r["ja"]):
            df[g] += 1
    df_cap = max(20, int(0.005 * len(train_rows)))
    index: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(train_rows):
        for g in trigrams(r["ja"]):
            if df[g] <= df_cap:                # chỉ index trigram hiếm (prefilter)
                index[g].append(i)
    drop: set[int] = set()
    for r in test_rows + dev_rows:
        T = trigrams(r["ja"])
        shared: Counter = Counter()
        for g in T:
            for i in index.get(g, ()):
                shared[i] += 1
        for i, c in shared.items():
            if c >= 3 and i not in drop:
                U = trigrams(train_rows[i]["ja"])
                if len(T & U) / len(T | U) >= JACCARD_DROP:
                    drop.add(i)
    train_rows = [r for i, r in enumerate(train_rows) if i not in drop]

    # ---------- 4. Ghi splits ----------
    write_jsonl(train_rows, OUT / "train.jsonl")
    write_jsonl(dev_rows,   OUT / "dev.jsonl")
    write_jsonl(test_rows,  OUT / "test.jsonl")

    # ---------- 5. File annotation gold (Google Sheets) ----------
    with open(OUT / "gold_annotation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "origin", "ja", "vi", "vi_sua",
                    "tagger_patterns", "gold_patterns", "ghi_chu"])
        for r in test_rows:
            spans = "|".join(f"{h['id']}:{h['span']}" for h in r.get("patterns", []))
            golds = "|".join(sorted(ids_of(r)))
            w.writerow([r["id"], r["origin"], r["ja"], r["vi"], "", spans, golds, ""])

    (OUT / "README_gold.txt").write_text(GOLD_PROTOCOL, encoding="utf-8")

    # ---------- 6. Báo cáo + core set + checksum ----------
    def pat_count(rows_: list[dict]) -> Counter:
        c: Counter = Counter()
        for r in rows_:
            for pid in ids_of(r):
                c[pid] += 1
        return c

    tr_c, te_c, de_c = pat_count(train_rows), pat_count(test_rows), pat_count(dev_rows)
    core = sorted(p for p in freq
                  if tr_c[p] >= CORE_MIN_TRAIN and te_c[p] >= CORE_MIN_TEST)
    tail = sorted(p for p in freq if p not in core)

    L = [f"train {len(train_rows):,} | dev {len(dev_rows):,} | test {len(test_rows):,} "
         f"(negative trong test: {len(test_rows) - len(chosen):,}) | "
         f"train bị xóa vì near-dup với test/dev: {len(drop):,}",
         "", f"CORE PATTERN SET (train≥{CORE_MIN_TRAIN} & test≥{CORE_MIN_TEST}) — "
             f"{len(core)} pattern, headline macro-F1 tính trên nhóm này:",
         "  " + (", ".join(core) if core else "(trống — inventory còn mỏng, ĐỪNG đóng băng)"),
         "", f"TAIL ({len(tail)} pattern, chỉ báo mô tả): " + ", ".join(tail),
         "", f"{'pattern':<24}{'train':>7}{'dev':>6}{'test':>6}"]
    for p in sorted(freq, key=lambda x: -freq[x]):
        L.append(f"{p:<24}{tr_c[p]:>7}{de_c[p]:>6}{te_c[p]:>6}")
    origin_mix = Counter(r["origin"] for r in test_rows)
    L += ["", "origin trong test: " +
          ", ".join(f"{o}={c}" for o, c in origin_mix.most_common())]
    report = "\n".join(L)
    (OUT / "split_report.txt").write_text(report, encoding="utf-8")
    print(report)

    checksum = hashlib.sha256("\n".join(sorted(test_ids)).encode()).hexdigest()
    (OUT / "test_checksum.txt").write_text(checksum + "\n", encoding="utf-8")
    print(f"\nsha256(test ids) = {checksum[:16]}…  → data/splits/test_checksum.txt")
    print("ĐÓNG BĂNG: từ lúc sửa gold_annotation.csv, không re-split, không thêm "
          "pattern id,\nmọi tuning chỉ nhìn dev — test chỉ chấm MỘT LẦN ở cuối.")

GOLD_PROTOCOL = """GIAO THỨC GẮN NHÃN GOLD — đọc trước khi mở gold_annotation.csv
================================================================
Đơn vị nhãn: PATTERN-PRESENCE theo câu — cột gold_patterns là danh sách id
pattern THỰC SỰ xuất hiện trong câu, nối bằng dấu '|'. Span trong cột
tagger_patterns chỉ để tham khảo, không phải đối tượng chấm.

Quy trình cho từng dòng:
1) Đọc câu ja. Sửa cột gold_patterns (tagger đã điền sẵn):
   - XÓA id sai (false positive của tagger)
   - THÊM id thiếu (false negative) — chỉ dùng id có trong patterns.yaml
   - Pattern ngoài danh mục → ghi vào ghi_chu, KHÔNG bịa id mới
2) Kiểm tra bản dịch vi. Sai/lệch nghĩa → viết bản đúng vào vi_sua
   (để trống nếu vi ổn). Chỉ sửa SAI NGHĨA, không đánh bóng văn phong.
3) Không chắc → ghi_chu, đánh dấu xem lại sau; không đổi định nghĩa giữa chừng.

Kỷ luật:
- Không tra lại train/dev khi gắn nhãn.
- Định nghĩa core set (train≥30 & test≥3) đã chốt TRƯỚC — xem split_report.txt.
- Xong thì lưu CSV (giữ UTF-8); script gold_ingest sẽ sinh test_gold.jsonl.
- test.jsonl + test_checksum.txt không được sửa dưới mọi hình thức.
"""

if __name__ == "__main__":
    main()