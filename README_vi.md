# robot_friction_detection

Phát hiện bất thường ma sát khớp trên robot phẳng 2 bậc tự do (2-DOF) bằng **mô phỏng động lực học** và **Random Forest**.

> Toàn bộ dữ liệu được tạo bằng mô phỏng. Kết quả **chưa được xác nhận trên robot thật**. Phát hiện bất thường ở đây
> **không phải** dự báo thời điểm hỏng hay tuổi thọ còn lại. Dự án được xây dựng với sự hỗ trợ của trợ lý AI (Claude).

- Câu hỏi nghiên cứu, kết quả và giới hạn: [`reports/standard/report_vi.md`](reports/standard/report_vi.md) (bản Word: `report_vi.docx`);
  bản tiếng Anh: [`reports/standard/report_en.md`](reports/standard/report_en.md) (`report_en.docx`, tạo bằng `make_report --lang en`);
  bản LaTeX ngắn gọn: [`reports/latex/jointguard_report.tex`](reports/latex/jointguard_report.tex)
- Giải thích từng giai đoạn để tự học và tự trình bày: [`docs/HUONG_DAN_HOC.md`](docs/HUONG_DAN_HOC.md)
- Kịch bản quay video demo: [`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md)
- Nhật ký tiến độ và các lệnh đã chạy: [`PROGRESS.md`](PROGRESS.md)

## Cấu trúc

```
configs/default.json      mọi tham số (robot, ma sát, bộ điều khiển, kịch bản, huấn luyện), đơn vị SI
src/
  config.py               đọc cấu hình, chống ghi đè kết quả
  dynamics.py             động học thuận, M(q), C(q,q̇)q̇, G(q), ma sát, giải gia tốc
  controller.py           PD + bù trọng lực + giới hạn mô-men
  trajectory.py           quỹ đạo sin trơn, quỹ đạo chuyển điểm (quintic)
  faults.py               lịch bơm lỗi ma sát + quy tắc gán nhãn
  simulator.py            vòng kín: RK4 1 kHz, điều khiển 100 Hz, cảm biến 100 Hz
  scenarios.py            lấy mẫu kịch bản (khỏe/mild/severe/gradual, khớp 1/2, stress)
  dataset.py              sinh dữ liệu hàng loạt headless, manifest, log lỗi số
  splits.py               chia train/val/test theo nhóm thí nghiệm
  features.py             đặc trưng cửa sổ (dùng chung offline & online)
  alarm.py                quy tắc duy trì cảnh báo
  training.py             Random Forest + baseline, chọn tham số/ngưỡng trên validation
  evaluation.py           chỉ số theo cửa sổ và theo sự kiện (báo nhầm, độ trễ, bỏ sót)
  online.py               bộ phát hiện streaming cho giao diện
  gui/app.py              giao diện Pygame
  plots.py, report.py     biểu đồ và báo cáo (MD + Word) từ kết quả thật
scripts/                  các bước chạy được riêng lẻ: python -m scripts.<tên>
tests/                    pytest: vật lý, rò rỉ dữ liệu, online=offline, quy tắc sự kiện
data/<name>/              signals/ (quan sát), truth/ (ground truth), manifest.csv, split_manifest.csv, features/
models/<tag>/             model_bundle.joblib, baseline.json, model_card.json, training_report.json
results/<tag>/            metrics.json, bảng theo cửa sổ/lần chạy, figures/
reports/<tag>/            report_vi.md, report_vi.docx, figures/
```

## Quy ước (conventions)

- Hệ tọa độ gốc tại khớp 1; x sang phải, y hướng lên; trọng lực theo −y. Robot chuyển động trong mặt phẳng thẳng đứng.
- q1 là góc khâu 1 so với trục +x; q2 là góc khâu 2 **so với khâu 1**. Chiều dương ngược chiều kim đồng hồ, đơn vị rad.
- Mô-men khớp dương theo chiều dương của góc, đơn vị N·m. Mọi đơn vị là SI.
- Sai số bám `e = q_des − q_meas`.

## Cài đặt

### Windows (PowerShell)

```powershell
cd robot_friction_detection
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1          # nếu bị chặn: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd robot_friction_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Các phiên bản trong `requirements.txt` là phiên bản đã dùng để tạo kết quả trong báo cáo (Python 3.13). Python 3.10 trở lên
nhiều khả năng cũng chạy được, nhưng chưa được kiểm thử. Dùng phiên bản thư viện khác có thể làm số liệu lệch nhẹ.

## Lệnh chạy

Mọi lệnh chạy từ thư mục gốc dự án, khi venv đang bật. Lệnh giống nhau trên PowerShell và bash.

| Việc | Lệnh |
|---|---|
| Kiểm tra vật lý | `python -m scripts.check_physics` |
| Toàn bộ kiểm thử | `python -m pytest -q` |
| Smoke pipeline (≈1–2 phút) | `python -m scripts.run_pipeline --mode smoke` |
| Toàn bộ pipeline standard (≈15–40 phút tùy máy, phần lâu nhất là tìm tham số RF) | `python -m scripts.run_pipeline --mode standard` |
| Sinh dữ liệu standard | `python -m scripts.generate_dataset --mode standard` |
| Tạo split (trước khi train) | `python -m scripts.make_splits --data data/standard` |
| Tạo đặc trưng | `python -m scripts.build_features --data data/standard` |
| Huấn luyện | `python -m scripts.train --data data/standard --tag standard` |
| Đánh giá (test + stress) | `python -m scripts.evaluate --data data/standard --model models/standard --tag standard` |
| Huấn luyện chẩn đoán (khớp nào, ma sát bao nhiêu) | `python -m scripts.train_diagnosis --data data/standard --model models/standard` |
| Đánh giá chẩn đoán (test + stress, cần chạy `evaluate` trước) | `python -m scripts.evaluate_diagnosis --data data/standard --model models/standard --results results/standard` |
| Biểu đồ chẩn đoán (cho báo cáo LaTeX `reports/latex/`) | `python -m scripts.make_diagnosis_figures --results results/standard --demo web/demo_runs.json` |
| Xuất dữ liệu cho trang web | `python -m scripts.export_web_demo --model models/standard --out web/demo_runs.json` |
| Biểu đồ | `python -m scripts.make_figures --results results/standard` |
| Báo cáo MD + Word | `python -m scripts.make_report --results results/standard --model models/standard --out reports/standard` |
| Mở demo | `python -m scripts.run_demo --model models/standard` |
| Demo tự động không cửa sổ (chụp ảnh) | `python -m scripts.demo_headless_check --model models/standard` |

**Không ghi đè âm thầm.** Nếu thư mục đích đã có kết quả, script sẽ dừng và báo lỗi. Hãy dùng tên mới
(`--name`, `--tag`) hoặc thêm `--overwrite` nếu thật sự muốn thay thế. Dự án đã có sẵn `data/standard`, `models/standard`
và `results/standard`, nên để chạy lại standard từ đầu, hãy dùng tên khác, ví dụ:

```bash
python -m scripts.generate_dataset --mode standard --name standard_v2
python -m scripts.make_splits --data data/standard_v2
python -m scripts.build_features --data data/standard_v2
python -m scripts.train --data data/standard_v2 --tag standard_v2
python -m scripts.evaluate --data data/standard_v2 --model models/standard_v2 --tag standard_v2
python -m scripts.make_figures --results results/standard_v2
python -m scripts.make_report --results results/standard_v2 --model models/standard_v2 --out reports/standard_v2
```

Với cùng seed, cấu hình và phiên bản thư viện, dữ liệu sinh lại sẽ giống hệt (đã kiểm thử ở mức từng lần chạy).

## Dữ liệu đi kèm

File `robot_friction_detection.zip` chứa mã nguồn, model đã huấn luyện, kết quả, biểu đồ và báo cáo, cùng các
`manifest.csv`/`split_manifest.csv`. Nó **không chứa** tín hiệu thô (`data/*/signals`, `truth`, `features`) để giữ file nhỏ.
Có hai cách để có lại tín hiệu thô:
- giải nén `data_standard_full.zip` vào thư mục gốc dự án (file này tạo ra `robot_friction_detection/data/standard/...`); hoặc
- tự sinh lại: `python -m scripts.generate_dataset --mode standard --name standard_regen`, rồi so sánh với manifest gốc
  (cùng seed và cùng phiên bản thư viện sẽ cho cùng tín hiệu).

Demo (`run_demo`) chỉ cần `models/standard`, không cần dữ liệu thô.

## Giao diện demo

Bố cục: bên trái là robot, đường đi mục tiêu của đầu robot (nét chấm) và vệt đường đi thực tế. Bên phải là mô-men, sai số bám
và điểm dự đoán của mô hình (kèm ngưỡng). Bên dưới là ô **AI PREDICTION** và ô riêng **SIMULATOR GROUND TRUTH**
(chỉ phục vụ người làm thí nghiệm, không phải đầu vào của AI).

| Phím | Tác dụng |
|---|---|
| Space | tạm dừng / tiếp tục |
| R | reset lần chạy, bộ nhớ cửa sổ và cảnh báo |
| 1 | chế độ khỏe (ma sát về giá trị khỏe) |
| 2 / 3 | ma sát khớp 1 / khớp 2 tăng đột ngột x3 |
| 4 | ma sát khớp 1 tăng dần lên x3,5 trong 10 s |
| L | đổi tải 0,5 → 1,5 → 2,0 kg (reset) |
| V | đổi tốc độ normal ↔ fast (reset) |
| S | lưu ảnh vào `results/screenshots/` |
| Esc | thoát và lưu log vào `results/demo_logs/` |

- Các phím lỗi chỉ thay đổi tham số ma sát trong mô hình vật lý. Cảnh báo chỉ đến từ mô hình đã huấn luyện.
- Nếu có `models/standard/diagnosis_bundle.joblib`, khi cảnh báo bật ô AI hiện thêm chẩn đoán, ví dụ
  `Joint 2 - friction increase: 213% (x3.1)`. Chẩn đoán dùng hai Random Forest (khớp nào, hệ số ma sát), huấn luyện
  chỉ trên cửa sổ lỗi của tập train (`src/diagnosis.py`). Trên test: đúng khớp 30/32 lần chạy tại lúc báo động, sai số
  hệ số ma sát trung vị 0,36×; stress: 23/26 và 0,53× (`results/standard/diagnosis_metrics.json`). Khi báo nhầm,
  chẩn đoán cũng sai.
- Thời gian mô phỏng được tiến theo đồng hồ thật, nên FPS thấp không làm robot chạy chậm. Nếu máy không đủ nhanh, giao diện
  hiện **NOT REAL-TIME** và real-time factor < 1.
- Nếu chưa có model, giao diện báo **MODEL NOT FOUND** kèm lệnh cần chạy, và không hiển thị dự đoán giả.
- Trong `WARMING UP`, chưa đủ dữ liệu cho cửa sổ đầu tiên hoặc robot đang trong 2 s khởi động trơn.
- macOS: nếu cửa sổ không hiện, hãy chạy từ Terminal (không chạy từ IDE) và cấp quyền nếu được hỏi.
  Linux không có màn hình: dùng `demo_headless_check`.
