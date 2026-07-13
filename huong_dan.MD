# Trợ lý dịch Nhật→Việt + giải thích ngữ pháp N5–N4

### CVAE dual-decoder, train from scratch — bảng điều khiển dự án

## 1. Ràng buộc & phạm vi (đã chốt)

- **From scratch tuyệt đối**: không trọng số pretrained, không token nào do model
  khác sinh ra — áp cả lên _phả hệ dữ liệu_ (loại corpus mined kiểu CCMatrix/NLLB).
- Công cụ được phép ở tầng dữ liệu: SentencePiece (tự train trên corpus của mình),
  fugashi/MeCab (tra từ điển hình thái — không phải neural model).
- Phạm vi ngữ pháp: **N5–N4**, ~100–150 pattern. Model ~6M tham số, chạy CPU được.
- Ba tầng đầu ra: model quyết **có pattern gì** → rule matcher định vị **ở đâu**
  → từ điển template quyết **giảng thế nào** (hallucination tầng giải thích = 0 by design).

## 2. Kiến trúc & loss (tóm tắt một màn hình)

```
ja → BiGRU encoder → (μ, logσ²) → z = μ + σ⊙ε        (d_z = 64)
       │
       ├─ Decoder DỊCH:      GRU + attention Bahdanau, input mỗi bước [emb; c_t; z]
       └─ Decoder NGỮ PHÁP:  GRU, CHỈ nhận z (không attention) → chuỗi pattern-ID + <none>

L = CE_dịch/token + λ·CE_ngữpháp/token + β(t)·Σⱼ max(λ_fb, KL_j)
    KL annealing tuyến tính (warmup 15% bước) · free bits 0.2 · word dropout 0.3 (chỉ decoder dịch)
    assert KL ≥ 0 · eval dùng z = μ · tied embeddings
```

Tam giác ablation dùng chung code: `--variant seq2seq | cvae1 | cvae2`
(+ hàng baseline thứ tư: chính rule tagger, chấm F1 trên gold như mọi model).

## 3. File & trạng thái

| #   | File               | Vai trò                                                                         | Trạng thái                             |
| --- | ------------------ | ------------------------------------------------------------------------------- | -------------------------------------- |
| 1   | `download_data.py` | Tatoeba + ALT + OPUS → `data/interim/*.jsonl` + manifest (URL, sha256, license) | ✅ đã chạy                             |
| 2   | `prepare_data.py`  | Chuẩn hóa NFKC/NFC → 5 cửa lọc → dedupe skeleton → `pool.jsonl` + `stats.txt`   | ✅ đã chạy                             |
| 3   | `patterns.yaml`    | Danh mục pattern (schema + 7 mẫu)                                               | 🔶 **7/100–150 — critical path**       |
| 4   | `tagger.py`        | Miner + silver labeler + baseline; `--selftest / --debug / --mine`              | ✅ chạy được (theo 7 pattern)          |
| 5   | `split_data.py`    | Stratify test 180 + dev 400, chặn leakage trigram, core set, gold CSV, checksum | ⏳ chạy THẬT khi patterns xong         |
| 6   | `model.py`         | Kiến trúc CVAE 3 biến thể                                                       | ✅                                     |
| 7   | `train.py`         | Vòng train + metrics.csv/kl_dims.csv + overfit sanity (đã vá artifacts_overfit) | ✅ sanity xong · ▶ ablation tổng duyệt |

## 4. Lệnh chạy theo thứ tự

```bat
:: ---- DỮ LIỆU (lặp lại các bước 3–5 mỗi khi patterns.yaml đổi lớn) ----
python download_data.py
python prepare_data.py
python tagger.py --selftest
python tagger.py --mine
python split_data.py            :: CHỈ chạy thật khi patterns hoàn chỉnh → ĐÓNG BĂNG

:: ---- MODEL ----
python train.py --overfit 50 --epochs 300 --run sanity
rmdir /s /q artifacts           :: BẮT BUỘC mỗi khi dữ liệu train thay đổi
python train.py --variant seq2seq --run a_seq2seq
python train.py --variant cvae1   --run b_cvae1
python train.py --variant cvae2   --run c_cvae2
```

## 5. Kỷ luật bất di bất dịch

1. `artifacts/` xóa & build lại khi dữ liệu đổi; **giữ nguyên trong một chiến dịch ablation** (vocab chung mới so sánh được).
2. `test.jsonl` + `test_checksum.txt` đóng băng từ lúc bắt đầu sửa gold CSV; tuning chỉ nhìn dev; **test chấm một lần** ở cuối.
3. Dữ liệu synthetic (khi có) chỉ append vào **train**.
4. Dedupe trước split; near-dup đã có chốt chặn trigram trong split_data.
5. Mọi file I/O `encoding="utf-8"` (Windows mặc định cp1252 — bẫy số một).
6. Mỗi run một thư mục `runs/<tên>/` chứa config + metrics + checkpoint — số nào trong báo cáo cũng truy vết được về một run.

## 6. Việc còn lại

### Luồng TAY (critical path — mọi thứ downstream chờ mục 1)

1. **Mở rộng `patterns.yaml` → ~100–150 mục** (vòng lặp: `--debug` câu ví dụ → viết seq → thêm SELFTEST → `--selftest`; ~10–15 phút/pattern).
2. **Từ điển template** — làm song song với (1), đề xuất format `templates.csv`:
   `pattern_id, meaning, explanation, common_mistake` (demo.py sẽ đọc file này).
3. `--mine` lần cuối → soi `coverage.csv` tìm lỗ.
4. `split_data.py` thật → kiểm core set (~60–80 mục) → **đóng băng**.
5. Gắn nhãn `gold_annotation.csv` (~180 câu, 6–12h, đọc `README_gold.txt` trước).

### Luồng CODE (Claude viết tiếp, theo thứ tự cần)

6. `gen_synthetic.py` — khung câu song ngữ + slot từ vựng → lấp lỗ coverage (chỉ vào train).
7. `gold_ingest.py` — đọc CSV đã sửa → `test_gold.jsonl`.
8. `evaluate.py` — chấm test MỘT lần: F1 micro/macro trên core set + bảng theo pattern,
   chrF/BLEU (sacrebleu), ELBO/token + IWAE K=64; chấm luôn hàng baseline rule-tagger.
9. `plots.py` — sáu họ hình từ metrics.csv + kl_dims.csv: loss tổng, train/dev gap,
   phân rã ELBO + β(t), heatmap KL từng chiều + AU, mặt phẳng rate–distortion,
   ELBO vs IWAE; cộng t-SNE của μ tô màu theo level.
10. `demo.py` — ghép model + span-locator + template → JSON hoàn chỉnh (đúng vision ban đầu).

### Chiến dịch cuối + báo cáo

11. Ablation thật: 3 biến thể × 2–3 seed trên splits đóng băng; tune λ/free_bits/p_wd trên dev (~10–15 run).
12. `evaluate.py` trên test một lần / model cuối; bảng + hình.
13. Viết báo cáo: Step 1–2 đã có sẵn nguyên liệu (toàn bộ phần derive A0–A9 + bảng map
    nguồn Goodfellow/Mohri/CS236/CS294-158 trong hội thoại); mục data provenance
    (license, tuyên bố no-MT, ghi chú pivot ALT); discussion collapse + label noise/ERM;
    limitations (presence-level vs span, data nhỏ, trần silver = trần tagger).

## 7. Git (khuyến nghị làm ngay trong VS Code)

```bat
git init
git add *.py *.yaml README.md
git commit -m "pipeline + model v1"
```

`.gitignore`:

```
data/
runs/
artifacts*/
__pycache__/
*.pt
```
