# Trợ lý dịch Nhật→Việt + giải thích ngữ pháp N5–N4

# THEO CÁI NÀY NHÉ SẾP BÌNH DƯƠNG

Pipeline: **dữ liệu → gán nhãn → chia tập → train CVAE (3 biến thể) → chấm điểm → vẽ hình → demo JSON**.

chạy lần lượt mục **1 → 2 → 3 → 4** là đi từ số 0 đến demo (với 7 pattern có sẵn).
Mục **5–6** là việc chính của đề tài (mở rộng pattern, chấm điểm cuối). Mục **7–8** để tra cứu.

---

## 1 · Cài đặt — một lần sau khi clone

Cần: Python 3.10+ (đã thử 3.11), ~2 GB đĩa trống, internet cho lần tải đầu.

```bat
python -m venv venv                :: tạo môi trường ảo (thư mục venv/)
venv\Scripts\activate              :: bật nó (PowerShell gõ: .\venv\Scripts\Activate.ps1)
pip install -r requirements.txt    :: cài thư viện — torch ~200MB nên hơi lâu
```

Nếu console in lỗi ký tự Nhật/Việt: chạy thêm `set PYTHONUTF8=1`
(PowerShell: `$env:PYTHONUTF8="1"`). Chỉ cần một lần mỗi phiên.

---

## 2 · Tải + làm sạch dữ liệu — một lần (cần internet)

```bat
python download_data.py    :: tải Tatoeba + ALT, ghép cặp câu Nhật-Việt → data/interim/
python prepare_data.py     :: chuẩn hóa chữ, lọc câu rác, bỏ trùng → data/processed/pool.jsonl (~16.000 cặp)
```

Xong bước này mở `data/processed/stats.txt` xem số liệu lọc (không bắt buộc).

---

## 3 · Gán nhãn ngữ pháp + chia train/dev/test

```bat
python tagger.py --selftest    :: kiểm bộ luật pattern — phải PASS hết mới đi tiếp
python tagger.py --mine        :: quét pool, đánh dấu câu nào chứa pattern nào → mined.jsonl + coverage.csv
python split_data.py           :: chia train/dev/test — test 180 câu bị KHÓA bằng checksum, không đụng nữa
python gen_synthetic.py        :: sinh thêm câu từ khung frames.yaml, CHỈ thêm vào train
```

---

## 4 · Train + chấm + vẽ hình + demo — MỘT lệnh

Mở `config.ini` xem qua (mặc định đang là bản CHẠY NHANH cho máy yếu: 300 câu, 3 epoch), rồi:

```bat
python run.py
:: tự làm hết: xóa artifacts/ → train 3 biến thể (seq2seq / cvae1 / cvae2)
::             → chấm trên dev → vẽ hình → in JSON demo ra màn hình
```

- Kết quả nằm ở: `runs/<biến thể>/` (checkpoint + metrics + hình), `eval_summary_dev.csv`, `plots_compare/`.
- Muốn **số thật cho báo cáo**: trong `config.ini` sửa `limit = 0` (dùng toàn bộ train — CPU chạy vài giờ).
- Muốn tự tay chạy từng file thay vì run.py → xem mục 7.

Đến đây bạn đã có bản demo chạy được từ đầu đến cuối. 🎉

---

## 5 · Việc chính của đề tài: mở rộng pattern (làm TAY, lặp nhiều lần)

> Mục này là việc của NGƯỜI SOẠN nội dung. Cả 3 file kết quả (patterns.yaml / frames.yaml /
> templates.csv) đều nằm trên git — **người clone về bỏ qua mục 5**, chạy thẳng 1→4 là đủ.
> Với người soạn: chỉ **5a là vòng lặp bắt buộc**; 5b khi coverage thiếu; 5c làm một lần ở cuối.

Hiện có **7/~100–150 pattern** — chất lượng cả hệ phụ thuộc mục này. Ba file, mỗi file một vòng:

### 5a · patterns.yaml — thêm luật nhận diện pattern (~10–15 phút/pattern)

```bat
:: bước 1: xem máy tách câu ví dụ thành hình vị thế nào
python tagger.py --debug "昨日パソコンを買ったばかりです。"
:: bước 2: dựa vào bảng đó, viết mục mới (id + seq) vào patterns.yaml
:: bước 3: thêm 1 câu test vào SELFTEST[] trong tagger.py, rồi kiểm:
python tagger.py --selftest    :: PASS hết = pattern mới được nhận
```

### 5b · frames.yaml — khung câu bù cho pattern hiếm

```bat
python tagger.py --mine        :: mở coverage.csv, xem pattern nào ÍT câu
:: thêm khung + từ vựng vào frames.yaml cho pattern đó
python gen_synthetic.py        :: tự kiểm khung sinh đúng nhãn — có dòng ⚠ là khung/luật sai
```

### 5c · templates.csv — lời giảng cho từng pattern (demo.py đọc file này)

```bat
:: tạo sẵn khung CSV từ patterns.yaml (đỡ gõ tay từng dòng):
python -c "import yaml,csv; ps=yaml.safe_load(open('patterns.yaml',encoding='utf-8')); w=csv.writer(open('templates_seed.csv','w',newline='',encoding='utf-8-sig')); w.writerow(['pattern_id','meaning','explanation','common_mistake']); [w.writerow([p['id'],p.get('note',''),'','']) for p in ps]"
:: → đổi tên thành templates.csv, mở bằng Excel/Sheets, điền tay 2 cột cuối
```

**Sau mỗi đợt mở rộng lớn: chạy lại mục 3 rồi mục 4** (nhãn + split + vocab phải làm lại theo pattern mới).

---

## 6 · Chấm điểm cuối — làm MỘT lần, sau khi mọi thứ chốt

```bat
:: bước tay: mở data/splits/gold_annotation.csv, sửa nhãn máy gán sai (~6-12h)
::           đọc data/splits/README_gold.txt TRƯỚC khi sửa
python gold_ingest.py                                :: kiểm checksum + nhập nhãn đã sửa → test_gold.jsonl
python evaluate.py runs/cvae2 --split test --iwae-k 64   :: chấm test ĐÚNG MỘT LẦN cho model cuối
```

---

## 7 · (Tùy chọn) chạy từng file thay cho run.py

Đây chính là các lệnh run.py gọi bên trong — dùng khi muốn kiểm soát từng bước:

```bat
:: (tùy chọn) train thử 50 câu để bắt bug trước — ghi vocab riêng artifacts_overfit/
python train.py --overfit 50 --epochs 300 --run sanity

rmdir /s /q artifacts    :: xóa MỘT lần trước run đầu — 3 biến thể phải dùng CHUNG vocab
                         :: (PowerShell: Remove-Item -Recurse -Force artifacts)

:: train 3 biến thể — cờ giống hệt nhau mới so sánh được (cờ dưới = bản chạy nhanh)
python train.py --variant seq2seq --run seq2seq --limit 300 --epochs 3 --dev-eval-n 40
python train.py --variant cvae1   --run cvae1   --limit 300 --epochs 3 --dev-eval-n 40
python train.py --variant cvae2   --run cvae2   --limit 300 --epochs 3 --dev-eval-n 40
:: bản THẬT cho báo cáo: bỏ 3 cờ --limit --epochs --dev-eval-n

python evaluate.py runs/seq2seq runs/cvae1 runs/cvae2 --split dev --iwae-k 64
python plots.py runs/seq2seq runs/cvae1 runs/cvae2
python demo.py --run runs/cvae2 "昨日買ったばかりのパソコンが壊れてしまいました。"
```

---

## 8 · Quy tắc KHÔNG được phá

1. **artifacts/** (vocab): xóa + build lại khi dữ liệu train đổi; nhưng GIỮ NGUYÊN suốt một đợt so sánh 3 biến thể.
2. **test.jsonl**: đã khóa bằng checksum — không sửa, không chia lại. Mọi tinh chỉnh chỉ nhìn **dev**; test chấm đúng **một lần** ở mục 6.
3. Câu synthetic chỉ được thêm vào **train**, không bao giờ vào dev/test.
4. Mọi file mở bằng `encoding="utf-8"` — Windows mặc định cp1252, là bẫy số một.
5. Mỗi lần train một thư mục `runs/<tên>/` riêng — số nào trong báo cáo cũng truy được về một run.
