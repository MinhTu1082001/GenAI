"""
check_unk.py — đo tỉ lệ <unk> của SentencePiece trên train/dev/test.

Vì sao tồn tại: phát hiện 2026-07-22 — spm_ja train với character_coverage=0.9995
làm 12.6% token JA thành <unk> (83.8% câu train dính UNK) → encoder mất kanji
nội dung, dịch sai nghĩa. Script này là GATE: sau khi sửa coverage=1.0 +
byte_fallback và rebuild artifacts/, chạy lại phải thấy UNK ~0.00%.

Chạy:   python check_unk.py            # sau khi artifacts/spm_*.model đã build
Đạt :   mọi split UNK < 0.1% token     # nếu vẫn >1% → artifacts/ cũ chưa bị xóa
"""
import json
import sys
from pathlib import Path

if sys.platform == "win32":                      # console Windows → UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sentencepiece as spm

ART = Path("artifacts")
SPLITS = Path("data/splits")


def do_side(name: str, model: Path, field: str) -> bool:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model))
    unk = sp.unk_id()
    print(f"\n[{name}] vocab={sp.get_piece_size()}  ({model})")
    ok = True
    for split in ("train", "dev", "test"):
        p = SPLITS / f"{split}.jsonl"
        if not p.exists():
            print(f"  {split:5s}: (chưa có {p})")
            continue
        ntok = nunk = nsent = nsent_unk = 0
        for line in open(p, encoding="utf-8"):
            ids = sp.encode(json.loads(line)[field], out_type=int)
            u = sum(1 for i in ids if i == unk)
            ntok += len(ids); nunk += u
            nsent += 1; nsent_unk += (u > 0)
        pct = 100 * nunk / max(ntok, 1)
        flag = "✓" if pct < 0.1 else "✗ CAO — xóa artifacts/spm_* rồi train lại"
        ok &= pct < 0.1
        print(f"  {split:5s}: UNK {pct:5.2f}% token | "
              f"{100 * nsent_unk / max(nsent, 1):5.1f}% câu dính UNK  {flag}")
    return ok


def main() -> None:
    missing = [m for m in ("spm_ja.model", "spm_vi.model") if not (ART / m).exists()]
    if missing:
        sys.exit(f"Chưa có {missing} — chạy train.py (bước build vocab) trước.")
    ok = do_side("JA", ART / "spm_ja.model", "ja")
    ok &= do_side("VI", ART / "spm_vi.model", "vi")
    print("\nKẾT LUẬN:", "PASS — tokenizer sạch UNK ✓" if ok
          else "FAIL — còn UNK; gần như chắc chắn artifacts/ là bản cũ (coverage 0.9995)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
