# Chạy từ đầu đến cuối sau khi clone

Dành cho người **mới clone repo về**, muốn chạy ra kết quả mà không phải soạn nội dung.

Bạn sẽ đi qua 5 bước, kết thúc bằng: 3 model đã train, bảng điểm, hình cho báo cáo,
và một bản demo dịch + chú giải ngữ pháp dạng JSON.

> Đây là bản rút gọn của `huong_dan.MD` (mục 1–4 và 6). Mục 5 của file đó là việc
> **soạn nội dung bằng tay** (viết luật ngữ pháp) — kết quả đã có sẵn trên git
> (`patterns.yaml`, `frames.yaml`), nên bạn bỏ qua được. Xem mục "Điều cần biết
> trước" bên dưới để rõ bạn nhận được gì và thiếu gì.

Cần: Python 3.10+ · ~2 GB đĩa trống · internet (chỉ cho bước 2).

---

## 0 · Điều cần biết trước — đọc 1 phút, đỡ mất công

**Repo KHÔNG chứa dữ liệu.** Thư mục `data/`, `artifacts/`, `runs/` đều bị gitignore.
Bạn phải tự chạy bước 2–3 để sinh ra chúng. Đây là chuyện bình thường (dữ liệu nặng),
không phải lỗi.

**Ba thứ cần lưu ý:**

1. **`config.ini` không có trên git** — chỉ có `config.ini.example`. Bước 1 sẽ hướng dẫn
   tạo. Nếu bỏ qua, `python run.py` báo *"Không thấy config.ini"*.
2. **`templates.csv` chưa có** (đây là phần 5c của huong_dan, chưa ai làm). Hệ vẫn chạy
   bình thường: demo lấy tạm phần `note` trong `patterns.yaml` làm nghĩa, còn ô
   `explanation` sẽ in `"(chưa có template — bổ sung templates.csv)"`. Muốn có lời giảng
   đầy đủ thì phải tự soạn — xem mục 5c của `huong_dan.MD`.
3. **Bước 5 (chấm test) cần gán nhãn tay mới có ý nghĩa.** Chạy được ngay về mặt máy móc,
   nhưng con số chỉ đúng nếu bạn đã sửa nhãn. Giải thích kỹ ở ngay bước đó.

**Bẫy tiếng Nhật trên Windows:** console mặc định cp1252 sẽ crash khi in chữ Nhật.
Chạy lệnh này **một lần mỗi khi mở terminal mới**:

```bat
set PYTHONUTF8=1
```
```powershell
$env:PYTHONUTF8 = "1"     # nếu dùng PowerShell
```

---

## 1 · Cài đặt — làm một lần

```bat
python -m venv venv                :: tạo môi trường ảo (thư mục venv/)
venv\Scripts\activate              :: bật nó — PowerShell gõ: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt    :: cài thư viện — torch ~200MB nên hơi lâu

copy config.ini.example config.ini :: tạo file cấu hình (BẮT BUỘC — repo không có sẵn)
                                   :: PowerShell: Copy-Item config.ini.example config.ini
```

**Chọn chế độ chạy.** Mở `config.ini` vừa tạo, xem 3 dòng trong mục `[train]`:

| Mục đích | limit | epochs | dev_eval_n | Thời gian train (CPU) |
|---|---|---|---|---|
| **Chạy thử cho biết** | `300` | `3` | `40` | ~10–20 phút |
| **Số thật cho báo cáo** | `0` | `60` | `200` | vài giờ |

File `config.ini.example` mặc định là **bản số thật**. Máy yếu hoặc muốn xem pipeline
chạy trót lọt trước, hãy sửa 3 dòng đó về `300 / 3 / 40`, chạy xong một vòng rồi mới đổi lại.

*Kiểm tra:* gõ `python -c "import torch, fugashi, yaml; print('OK')"` — in `OK` là xong bước 1.

---

## 2 · Tải và làm sạch dữ liệu — làm một lần, cần internet

```bat
python download_data.py    :: tải Tatoeba + ALT, ghép cặp câu Nhật–Việt → data/interim/
python prepare_data.py     :: chuẩn hóa, lọc câu rác, bỏ trùng → data/processed/pool.jsonl
```

Mất khoảng 10–30 phút tùy mạng. Xong sẽ có ~16.000 cặp câu.

*Kiểm tra:* mở `data/processed/stats.txt` — có bảng "PHỄU LỌC" và dòng `TỔNG pool.jsonl`
là đạt.

---

## 3 · Gán nhãn ngữ pháp và chia tập

```bat
python tagger.py --selftest    :: kiểm bộ luật — phải PASS HẾT mới được đi tiếp
python tagger.py --mine        :: quét pool, đánh dấu câu nào chứa pattern nào
python split_data.py           :: chia train/dev/test; test 180 câu bị KHÓA bằng checksum
python gen_synthetic.py        :: sinh thêm câu từ khung frames.yaml, CHỈ thêm vào train
```

Lệnh đầu tiên phải in `TẤT CẢ PASS ✓`. Nếu FAIL thì dừng lại — luật ngữ pháp đang hỏng,
mọi nhãn phía sau sẽ sai theo.

*Kiểm tra:* có 3 file `data/splits/train.jsonl`, `dev.jsonl`, `test.jsonl`, và
`data/splits/split_report.txt` cho biết mỗi tập bao nhiêu câu.

> ⚠️ **Từ giây phút này, đừng chạy lại bước 3 nữa** trừ khi bạn sửa `patterns.yaml`.
> Chạy lại sẽ chia tập khác đi, và mọi điểm số đã có sẽ không so sánh được với nhau.

---

## 4 · Train, chấm điểm, vẽ hình, demo — một lệnh

```bat
python run.py
```

Nó tự làm hết theo `config.ini`: xóa `artifacts/` → train 3 biến thể
(`seq2seq` / `cvae1` / `cvae2`) → chấm trên tập dev → vẽ hình → in JSON demo ra màn hình.

Kết quả:

| Ở đâu | Là gì |
|---|---|
| `runs/seq2seq/`, `runs/cvae1/`, `runs/cvae2/` | checkpoint, log `metrics.csv`, hình riêng từng run |
| `eval_summary_dev.csv` | bảng so sánh 3 biến thể — số chính cho báo cáo |
| `plots_compare/` | hình so sánh: chrF, F1, rate–distortion, IWAE |

Đến đây bạn đã có một hệ chạy được từ đầu đến cuối. 🎉

*Muốn chạy từng file thay vì `run.py`* (để xem rõ từng bước): mục 7 của `huong_dan.MD`.

---

## 5 · Chấm điểm cuối trên tập test — làm MỘT lần

Đây là mục 6 của `huong_dan.MD`. **Đọc kỹ phần cảnh báo trước khi chạy.**

```bat
python gold_ingest.py                                  :: kiểm checksum + nhập nhãn → test_gold.jsonl
python evaluate.py runs/cvae2 --split test --iwae-k 64  :: chấm test cho model cuối
```

> ### ⚠️ Con số chỉ có ý nghĩa nếu đã gán nhãn tay
>
> Bước 3 đã tạo sẵn `data/splits/gold_annotation.csv`, trong đó cột `gold_patterns`
> **do máy điền tạm** (chính là nhãn của rule tagger).
>
> - **Nếu bạn KHÔNG sửa file đó:** hai lệnh trên vẫn chạy trót lọt, nhưng "đáp án" chính
>   là nhãn máy. Hệ quả: dòng baseline tagger sẽ ra `P = R = F1 = 1.000` (nó đang tự
>   so với chính nó), và điểm F1 của model thực chất là *so với nhãn máy*, không phải
>   nhãn người. **Không dùng được cho báo cáo.**
> - **Muốn số thật:** mở `data/splits/gold_annotation.csv` bằng Excel/Google Sheets,
>   đọc `data/splits/README_gold.txt` trước, rồi sửa tay 180 câu (xóa nhãn máy gán sai,
>   thêm nhãn bị sót). Việc này tốn khoảng 6–12 giờ và cần trình độ N4 trở lên.
>
> Nói cách khác: bước 5 chạy được ngay, nhưng nó là *chỗ để đặt kết quả gán nhãn tay*,
> chứ bản thân nó không tạo ra nhãn đúng.

Kết quả nằm ở `eval_summary_test.csv` và `runs/cvae2/eval_test/report.txt`.

**Chỉ chấm test đúng một lần**, sau khi mọi thứ đã chốt. Trong lúc còn tinh chỉnh, luôn
nhìn tập **dev** (bước 4). Chấm test nhiều lần rồi chọn kết quả đẹp nhất là tự lừa mình:
con số sẽ tốt hơn thực tế.

---

## Gặp lỗi thì xem đây

| Thông báo | Nguyên nhân & cách xử lý |
|---|---|
| `Không thấy config.ini` | Chưa làm bước copy ở mục 1. Chạy `copy config.ini.example config.ini`. |
| `UnicodeEncodeError` / chữ Nhật thành `?????` | Chưa đặt `set PYTHONUTF8=1` cho terminal hiện tại. |
| `Thiếu fugashi` / `Thiếu pyyaml` | Chưa bật venv (`venv\Scripts\activate`) hoặc chưa `pip install -r requirements.txt`. |
| `Không thấy data/processed/pool.jsonl` | Chưa chạy xong bước 2. |
| `Không thấy mined.jsonl` | Chưa chạy `python tagger.py --mine` ở bước 3. |
| `Checksum id KHÔNG khớp` (ở bước 5) | Đã chạy lại `split_data.py` sau khi bắt đầu gán nhãn → tập test đã đổi. Nhãn gold cũ không còn khớp; phải gán lại từ đầu. |
| `X câu FAIL — sửa YAML rồi chạy lại` | Luật trong `patterns.yaml` hỏng. Đừng đi tiếp; xem mục 5a của `huong_dan.MD`. |
| Train chạy quá lâu | Sửa `config.ini`: `limit = 300`, `epochs = 3`, `dev_eval_n = 40`. |

---

## Bốn quy tắc đừng phá

1. **`artifacts/`** (bộ từ vựng): 3 biến thể phải dùng **chung một bộ**. `run.py` tự lo
   việc này — đừng xóa `artifacts/` giữa chừng một đợt chạy.
2. **`test.jsonl`** đã khóa bằng checksum: không sửa, không chia lại. Tinh chỉnh chỉ nhìn
   dev; test chấm đúng một lần.
3. **Câu synthetic chỉ vào `train`**, không bao giờ vào dev/test.
4. **Mỗi lần train ghi vào một thư mục `runs/<tên>/` riêng** — để mọi con số trong báo cáo
   đều truy ngược được về một lần chạy cụ thể.
