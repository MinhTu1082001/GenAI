#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_data.py — Bước (1) của pipeline dữ liệu: TẢI & GHÉP cặp câu ja–vi.
Nguồn: Tatoeba (per-language exports), ALT (NICT), OPUS (TED2020, OpenSubtitles).

Chạy trên Colab (cần internet):
    !python download_data.py
Đầu ra:
    data/raw/            — file gốc, coi là BẤT BIẾN (copy lên Drive một lần)
    data/interim/*.jsonl — mỗi nguồn một JSONL: {id, ja, vi, origin}
    data/manifest.json   — URL, sha256, ngày tải  → nguyên liệu mục data provenance

Chú ý: đây CHỈ là bước tải + ghép. Chuẩn hóa / lọc / dedupe / tagger-mining
là các bước (2)–(6), nằm ở module sau — đừng trộn vào đây.
"""

from __future__ import annotations
import bz2, hashlib, io, json, sys, urllib.request, zipfile
from datetime import date
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/interim")

# Bật True để thêm cặp ja–vi GIÁN TIẾP qua tiếng Anh (2 bước trên đồ thị Tatoeba).
# Caveat: ja và vi là hai bản dịch độc lập của CÙNG một câu Anh → có thể lệch nghĩa.
# Bắt buộc spot-check và giữ origin="tatoeba_pivot" để lọc/ablation riêng.
PIVOT_QUA_TIENG_ANH = False

URLS = {
    "tatoeba": {
        "jpn_sentences": "https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2",
        "vie_sentences": "https://downloads.tatoeba.org/exports/per_language/vie/vie_sentences.tsv.bz2",
        "jpn_vie_links": "https://downloads.tatoeba.org/exports/per_language/jpn/jpn-vie_links.tsv.bz2",
        # chỉ dùng khi PIVOT_QUA_TIENG_ANH = True:
        "jpn_eng_links": "https://downloads.tatoeba.org/exports/per_language/jpn/jpn-eng_links.tsv.bz2",
        "eng_vie_links": "https://downloads.tatoeba.org/exports/per_language/eng/eng-vie_links.tsv.bz2",
    },
    "alt": {
        "zip": "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/ALT-Parallel-Corpus-20191206.zip",
    },
    "opus": {
        "ted2020": ["https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/ja-vi.txt.zip"],
        # thử v2024 trước, rớt thì lùi về v2018:
        "opensubtitles": [
            "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/ja-vi.txt.zip",
            "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/ja-vi.txt.zip",
        ],
    },
}

LICENSES = {
    "tatoeba": "CC BY 2.0 FR — ghi công tatoeba.org, kèm ngày export",
    "alt": "CC BY 4.0 — cite Riza et al. 2016 (O-COCOSDA) theo yêu cầu của NICT",
    "ted2020": "Nội dung TED: CC BY-NC-ND — dùng trong khuôn khổ học thuật",
    "opensubtitles": "Ghi công opensubtitles.org + cite Lison & Tiedemann 2016",
}

manifest = {"downloaded": str(date.today()), "licenses": LICENSES, "files": []}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> Path:
    """Tải nếu chưa có; luôn ghi (path, url, sha256) vào manifest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  ↓ {url}")
        urllib.request.urlretrieve(url, dest)
    manifest["files"].append({"path": str(dest), "url": url, "sha256": sha256(dest)})
    return dest


def fetch_first(urls: list[str], dest: Path) -> Path:
    """Thử lần lượt nhiều URL (ví dụ OpenSubtitles v2024 → v2018)."""
    last: Exception | None = None
    for u in urls:
        try:
            return fetch(u, dest)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  … {u} lỗi ({e}) — thử URL kế tiếp")
    raise last  # type: ignore[misc]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  → {path}  ({len(rows):,} cặp)")
    for r in rows[:2]:
        print(f"     ví dụ: {r['ja'][:40]} ⇄ {r['vi'][:40]}")


# ---------------------------------------------------------------- 1. TATOEBA
def load_tatoeba_sentences(path: Path) -> dict[int, str]:
    """File TSV: id <tab> lang(ISO 639-3) <tab> text. Parse phòng thủ 2–3 cột."""
    d: dict[int, str] = {}
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                sid, _lang, text = parts
            elif len(parts) == 2:
                sid, text = parts
            else:
                continue
            d[int(sid)] = text
    return d


def load_links(path: Path) -> list[tuple[int, int]]:
    """File links: mỗi dòng 'a<tab>b' nghĩa là câu b là bản dịch của câu a."""
    out: list[tuple[int, int]] = []
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            a, b = line.split()
            out.append((int(a), int(b)))
    return out


def step_tatoeba() -> None:
    print("[1/3] Tatoeba — cặp trực tiếp ja–vi")
    d = RAW / "tatoeba"
    ja = load_tatoeba_sentences(fetch(URLS["tatoeba"]["jpn_sentences"], d / "jpn_sentences.tsv.bz2"))
    vi = load_tatoeba_sentences(fetch(URLS["tatoeba"]["vie_sentences"], d / "vie_sentences.tsv.bz2"))
    print(f"  câu jpn: {len(ja):,} | câu vie: {len(vi):,}")

    rows, seen = [], set()
    for a, b in load_links(fetch(URLS["tatoeba"]["jpn_vie_links"], d / "jpn-vie_links.tsv.bz2")):
        # chấp nhận cả hai hướng ghi trong file, chuẩn hóa về (ja_id, vi_id)
        if a in ja and b in vi:
            pair = (a, b)
        elif b in ja and a in vi:
            pair = (b, a)
        else:
            continue
        if pair in seen:
            continue
        seen.add(pair)
        rows.append({"id": f"tatoeba:{pair[0]}-{pair[1]}",
                     "ja": ja[pair[0]], "vi": vi[pair[1]], "origin": "tatoeba"})
    write_jsonl(rows, OUT / "tatoeba.jsonl")

    if PIVOT_QUA_TIENG_ANH:
        print("  … pivot 2 bước qua tiếng Anh (origin=tatoeba_pivot)")
        eng_cua_ja: dict[int, list[int]] = {}
        for a, b in load_links(fetch(URLS["tatoeba"]["jpn_eng_links"], d / "jpn-eng_links.tsv.bz2")):
            ja_id, en_id = (a, b) if a in ja else (b, a)
            if ja_id in ja:
                eng_cua_ja.setdefault(en_id, []).append(ja_id)
        piv, seen2 = [], set()
        for a, b in load_links(fetch(URLS["tatoeba"]["eng_vie_links"], d / "eng-vie_links.tsv.bz2")):
            en_id, vi_id = (a, b) if b in vi else (b, a)
            if vi_id not in vi or en_id not in eng_cua_ja:
                continue
            for ja_id in eng_cua_ja[en_id]:
                key = (ja_id, vi_id)
                if key in seen or key in seen2:
                    continue
                seen2.add(key)
                piv.append({"id": f"tatoeba_pivot:{ja_id}-{en_id}-{vi_id}",
                            "ja": ja[ja_id], "vi": vi[vi_id], "origin": "tatoeba_pivot"})
        write_jsonl(piv, OUT / "tatoeba_pivot.jsonl")


# -------------------------------------------------------------------- 2. ALT
def step_alt() -> None:
    print("[2/3] ALT (NICT) — WikiNews dịch chuyên nghiệp, 13 ngôn ngữ")
    z = zipfile.ZipFile(fetch(URLS["alt"]["zip"], RAW / "alt" / "ALT-Parallel-Corpus-20191206.zip"))

    def load(suffix: str) -> dict[str, str]:
        member = next(n for n in z.namelist() if n.endswith(suffix))
        d: dict[str, str] = {}
        for line in io.TextIOWrapper(z.open(member), encoding="utf-8"):
            if "\t" not in line:
                continue
            sid, text = line.rstrip("\n").split("\t", 1)
            d[sid.strip()] = text.strip()
        return d

    ja, vi = load("data_ja.txt"), load("data_vi.txt")
    # ID câu dạng SNT.<id bài>.<số câu> → inner join; câu thiếu bản dịch tự rơi ra
    common = sorted(ja.keys() & vi.keys())
    print(f"  câu ja: {len(ja):,} | câu vi: {len(vi):,} | giao: {len(common):,}")
    rows = [{"id": f"alt:{s}", "ja": ja[s], "vi": vi[s], "origin": "alt"} for s in common]
    write_jsonl(rows, OUT / "alt.jsonl")


# ---------------------------------------------------- 3. OPUS (định dạng Moses)
def opus_moses(name: str, urls: list[str]) -> None:
    """Moses = zip chứa 2 file .ja/.vi thẳng hàng theo dòng (CÓ trùng lặp — dedupe ở bước 5)."""
    z = zipfile.ZipFile(fetch_first(urls, RAW / "opus" / f"{name}.ja-vi.txt.zip"))
    ja_f = next(n for n in z.namelist() if n.endswith(".ja"))
    vi_f = next(n for n in z.namelist() if n.endswith(".vi"))
    fj = io.TextIOWrapper(z.open(ja_f), encoding="utf-8")
    fv = io.TextIOWrapper(z.open(vi_f), encoding="utf-8")
    rows = [{"id": f"{name}:{i}", "ja": a.rstrip("\n"), "vi": b.rstrip("\n"), "origin": name}
            for i, (a, b) in enumerate(zip(fj, fv))]
    assert next(fj, None) is None and next(fv, None) is None, f"{name}: hai file lệch số dòng!"
    write_jsonl(rows, OUT / f"{name}.jsonl")


def step_opus() -> None:
    print("[3/3] OPUS")
    opus_moses("ted2020", URLS["opus"]["ted2020"])
    opus_moses("opensubtitles", URLS["opus"]["opensubtitles"])


if __name__ == "__main__":
    for step in (step_tatoeba, step_alt, step_opus):
        try:
            step()
        except Exception as e:  # noqa: BLE001
            print(f"  !! {step.__name__} thất bại: {e}\n"
                  f"     → mở trang nguồn kiểm tra URL/phiên bản mới rồi chạy lại; "
                  f"các nguồn khác không bị ảnh hưởng.", file=sys.stderr)
    (Path("data") / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nXong. data/manifest.json đã ghi URL + sha256 + license từng file."
          "\nSao chép nguyên thư mục data/ lên Drive MỘT LẦN rồi coi là bất biến;"
          "\ncác bước (2)–(6) chỉ đọc data/interim/*.jsonl, không đụng data/raw/.")
