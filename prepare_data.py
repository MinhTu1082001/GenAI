#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py — Bước (2)–(5): CHUẨN HÓA → LỌC → DEDUPE.

Đầu vào : data/interim/*.jsonl      (từ download_data.py, schema {id, ja, vi, origin})
Đầu ra  : data/processed/pool.jsonl — hồ câu sạch, CHƯA gắn nhãn, CHƯA chia split
          data/processed/stats.txt  — phễu lọc + percentiles + mẫu spot-check

Chạy:  !python prepare_data.py
Quy trình: chạy → đọc stats.txt → chỉnh NGƯỠNG nếu percentile cho thấy cắt oan →
chạy lại (idempotent, ghi đè). Bước kế tiếp của pipeline (tagger) đọc pool.jsonl.
CHÚ Ý: pool.jsonl chưa chia split — chưa train gì trên nó.
"""

from __future__ import annotations
import json, random, re, sys, unicodedata
from collections import Counter
from pathlib import Path

IN_DIR  = Path("data/interim")
OUT_DIR = Path("data/processed")

# ---------------- NGƯỠNG (calibrate bằng stats.txt, đừng đoán) ----------------
JA_CHARS    = (5, 60)      # độ dài vế Nhật: số KÝ TỰ sau chuẩn hóa
VI_WORDS    = (2, 32)      # độ dài vế Việt: số ÂM TIẾT (tách khoảng trắng)
RATIO_VI_JA = (0.20, 1.40) # số âm tiết Việt / số ký tự Nhật
MAX_LATIN_RATIO_JA = 0.30  # vế Nhật lai quá nhiều chữ Latin → loại
MAX_DIGIT_RATIO    = 0.30  # quá nhiều chữ số (bảng giá, tỉ số...) → loại
# Khi trùng lặp, giữ bản của nguồn đứng trước (bản dịch trực tiếp/người dịch ưu tiên):
PRIORITY = ["tatoeba", "alt", "tatoeba_pivot", "ted2020", "opensubtitles"]
SEED = 13

# ---------------- Regex dùng chung ----------------
RE_HIRA   = re.compile(r"[ぁ-ゖ]")
RE_CJK    = re.compile(r"[ぁ-ゖァ-ヺー一-龯々〆]")
RE_HANGUL = re.compile(r"[가-힣]")
RE_LATIN  = re.compile(r"[A-Za-z]")
RE_DIGIT  = re.compile(r"[0-9０-９]")
RE_TAG    = re.compile(r"<[^>]{0,40}>")            # <i>...</i> trong phụ đề
RE_URL    = re.compile(r"https?://|www\.", re.I)
RE_WS     = re.compile(r"\s+")
RE_MUSIC  = re.compile(r"[♪♬♩♫]+")
RE_NOTE   = re.compile(r"\([^)]*\)|（[^）]*）|\[[^\]]*\]")   # chú thích (tiếng động)…
RE_DASH   = re.compile(r"^\s*[-–—]+\s*")                     # gạch đầu thoại phụ đề
# Chỉ dùng cho khóa dedupe, không đụng text lưu trữ:
RE_SKEL   = re.compile(r"[。、．，！？!?・…‥「」『』（）()【】\[\]\"'“”‘’\s.,;:~〜ー-]")

VI_DIAC = set("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọ"
              "ồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ")
VI_COMMON = {"và","của","là","không","có","được","một","tôi","anh","em","bạn",
             "này","đã","đang","sẽ","cho","với","người","những","rồi","rất",
             "cũng","vào","ra","đi","làm","gì","thì","ở","như","về","khi","nào"}

# ---------------- Chuẩn hóa ----------------
def clean_subtitle(s: str) -> str:
    """Chỉ áp cho origin=opensubtitles: rác đặc thù phụ đề."""
    s = RE_DASH.sub("", s)
    s = RE_NOTE.sub("", s)
    return s

def norm_ja(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)   # thống nhất full/half-width
    s = RE_TAG.sub("", s)
    s = RE_MUSIC.sub("", s)
    s = RE_WS.sub("", s)                   # tiếng Nhật không dùng khoảng trắng
    return s.strip()

def norm_vi(s: str) -> str:
    s = unicodedata.normalize("NFC", s)    # chuẩn Unicode dấu tiếng Việt
    s = RE_TAG.sub("", s)
    s = RE_MUSIC.sub("", s)
    s = RE_WS.sub(" ", s)
    return s.strip()

# ---------------- Các cửa lọc (trả về lý do rớt, None = qua) ----------------
def reject_ja(ja: str) -> str | None:
    n = len(ja)
    if not (JA_CHARS[0] <= n <= JA_CHARS[1]):                return "ja_len"
    if not RE_HIRA.search(ja):                               return "ja_no_hiragana"
    if RE_HANGUL.search(ja):                                 return "ja_script_khac"
    if RE_URL.search(ja):                                    return "url"
    if len(RE_LATIN.findall(ja)) / n > MAX_LATIN_RATIO_JA:   return "ja_nhieu_latin"
    if len(RE_DIGIT.findall(ja)) / n > MAX_DIGIT_RATIO:      return "nhieu_chu_so"
    return None

def reject_vi(vi: str) -> str | None:
    words = vi.split()
    if not (VI_WORDS[0] <= len(words) <= VI_WORDS[1]):       return "vi_len"
    if RE_CJK.search(vi) or RE_HANGUL.search(vi):            return "vi_lan_cjk"
    if RE_URL.search(vi):                                    return "url"
    low = vi.lower()
    if len(RE_DIGIT.findall(low)) / max(len(low), 1) > MAX_DIGIT_RATIO:
        return "nhieu_chu_so"
    has_diac = any(c in VI_DIAC for c in low)
    has_word = any(w.strip(".,!?…;:\"'") in VI_COMMON for w in low.split())
    if not (has_diac or has_word):                           return "vi_khong_phai_tv"
    return None

def reject_pair(ja: str, vi: str) -> str | None:
    r = len(vi.split()) / max(len(ja), 1)
    if not (RATIO_VI_JA[0] <= r <= RATIO_VI_JA[1]):          return "lech_ty_le_dai"
    return None

# ---------------- Dedupe ----------------
def skeleton(ja: str) -> str:
    """Khóa trùng lặp: bỏ dấu câu, chữ số → '#'.
    Bắt được near-dup kiểu 「3時に起きた」/「5時に起きた」→ tính là một họ câu."""
    s = RE_SKEL.sub("", ja)
    s = RE_DIGIT.sub("#", s)
    return s

# ---------------- Percentile tiện dụng ----------------
def pct(xs: list[float], q: int) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]

# ---------------- Main ----------------
def main() -> None:
    random.seed(SEED)
    files = sorted(IN_DIR.glob("*.jsonl"))
    if not files:
        sys.exit("Không thấy data/interim/*.jsonl — chạy download_data.py trước.")
    # Xử lý theo thứ tự ưu tiên nguồn: bản tốt 'đến trước' nên thắng khi trùng khóa
    rank = {name: i for i, name in enumerate(PRIORITY)}
    files.sort(key=lambda p: rank.get(p.stem, 99))

    funnel: Counter = Counter()
    n_in: Counter = Counter()
    n_out: Counter = Counter()
    kept: dict[str, dict] = {}          # skeleton -> record

    for path in files:
        origin = path.stem
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                n_in[origin] += 1
                ja, vi = row["ja"], row["vi"]
                if origin == "opensubtitles":
                    ja, vi = clean_subtitle(ja), clean_subtitle(vi)
                ja, vi = norm_ja(ja), norm_vi(vi)

                reason = reject_ja(ja) or reject_vi(vi) or reject_pair(ja, vi)
                if reason:
                    funnel[reason] += 1
                    continue
                key = skeleton(ja)
                if key in kept:
                    funnel["trung_lap_ja"] += 1
                    continue
                kept[key] = {"id": row["id"], "ja": ja, "vi": vi, "origin": origin}
                n_out[origin] += 1
                funnel["GIU_LAI"] += 1

    rows = list(kept.values())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "pool.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- Báo cáo ----------
    ja_lens = [len(r["ja"]) for r in rows]
    vi_lens = [len(r["vi"].split()) for r in rows]
    ratios  = [len(r["vi"].split()) / max(len(r["ja"]), 1) for r in rows]

    L: list[str] = []
    L.append("== PHỄU LỌC (toàn bộ nguồn gộp) ==")
    for k, v in funnel.most_common():
        L.append(f"  {k:<20} {v:>10,}")
    L.append("\n== SỐ CÂU VÀO / RA THEO NGUỒN ==")
    for o in sorted(n_in, key=lambda x: rank.get(x, 99)):
        surv = 100 * n_out[o] / max(n_in[o], 1)
        L.append(f"  {o:<15} {n_in[o]:>10,} → {n_out[o]:>9,}  ({surv:4.1f}% sống)")
    L.append(f"\n  TỔNG pool.jsonl: {len(rows):,} cặp")
    L.append("\n== PERCENTILES trên tập sống sót (để calibrate NGƯỠNG) ==")
    L.append(f"  {'':<14}" + "".join(f"p{q:<7}" for q in (5, 25, 50, 75, 95)))
    L.append("  ja (ký tự)   " + "".join(f"{pct(ja_lens, q):<8.0f}" for q in (5, 25, 50, 75, 95)))
    L.append("  vi (âm tiết) " + "".join(f"{pct(vi_lens, q):<8.0f}" for q in (5, 25, 50, 75, 95)))
    L.append("  ratio vi/ja  " + "".join(f"{pct(ratios, q):<8.2f}" for q in (5, 25, 50, 75, 95)))
    L.append("\n== MẪU SPOT-CHECK (3 cặp ngẫu nhiên / nguồn — đọc bằng mắt!) ==")
    for o in sorted(n_out, key=lambda x: rank.get(x, 99)):
        sub = [r for r in rows if r["origin"] == o]
        for r in random.sample(sub, min(3, len(sub))):
            L.append(f"  [{o}] {r['ja']}")
            L.append(f"        ⇄ {r['vi']}")

    report = "\n".join(L)
    (OUT_DIR / "stats.txt").write_text(report, encoding="utf-8")
    print(report)
    print("\nĐã ghi data/processed/pool.jsonl và data/processed/stats.txt")
    print("Nếu percentile cho thấy ngưỡng cắt oan → chỉnh hằng số đầu file, chạy lại.")

if __name__ == "__main__":
    main()