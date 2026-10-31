# PROGRESS

Nhật ký tiến độ, lệnh đã chạy và việc còn lại. Cập nhật lần cuối: 2026-10-24.

## Môi trường đã dùng
Linux container, 4 CPU, Python 3.13.17, venv `.venv`, thư viện theo `requirements.txt`. Không có GPU và không có màn hình
(Pygame được chạy với `SDL_VIDEODRIVER=dummy`).

## Trạng thái theo giai đoạn

| Giai đoạn | Đã làm | Kiểm tra | Vấn đề còn lại |
|---|---|---|---|
| A. Mô phỏng | Động lực học 2 khâu (tải gộp vào khâu 2, quán tính rotor), ma sát nhớt + Coulomb trơn, RK4 1 ms, PD + bù trọng lực 100 Hz có bão hòa, quỹ đạo sin trơn và quỹ đạo chuyển điểm | 17 kiểm thử vật lý đạt. Giảm nửa bước (1 ms → 0,5 ms, lỗi x4, tải 2 kg): sai khác 8,2e-9 rad | Ban đầu, lỗi đột ngột rơi giữa các giai đoạn RK4 gây sai khác 1,7e-5 rad. Đã sửa (trước khi sinh dữ liệu) bằng cách giữ hệ số lỗi không đổi trong mỗi bước 1 ms |
| B. Dữ liệu | Smoke: 48 + 8 lần chạy, 16 s. Standard: 240 + 60 stress, 25 s. Manifest, seed, signals/truth tách riêng | 0 lần chạy mất ổn định. Tái lập từng lần chạy từ manifest đã được kiểm thử | Bão hòa mô-men: 0 trong miền, tối đa 1,7% chu kỳ trong stress |
| C. ML | 24 đặc trưng/cửa sổ; RF chọn trong 32 cấu hình (2 độ dài cửa sổ × 16 tham số) theo val AP; ngưỡng và số cửa sổ duy trì chọn theo val F1 với ràng buộc báo nhầm ≤ 0,5/phút; baseline sai số bám | Online = offline (kiểm thử). Đặc trưng nhân quả, không dùng thời gian hay metadata | `n_on = 8` nằm ở biên lưới `[1,3,5,8]` cho cả RF và baseline. Không chỉnh lại sau khi đã xem test |
| D. Đánh giá | Test và stress được đánh giá riêng, đúng một lần | Xem `results/standard/metrics.json` và báo cáo | Stress: báo nhầm nhiều (≈3,9/phút với RF) do tải và tốc độ ngoài miền |
| E. Giao diện | Pygame, phím 1–4, R, Space, S, Esc, L, V; WARMING UP; MODEL NOT FOUND; log CSV + events | Headless: ảnh chụp tự động; chế độ thời gian thực trên container: RTF ≈ 1,0 | Chưa thử trên máy có màn hình thật. Ở tốc độ fast, robot khỏe có 2–3 cảnh báo nhầm/30 s (xem docs/HUONG_DAN_HOC.md) |
| Báo cáo | `reports/standard/report_vi.md` + `.docx`, hình tự động | Đã chuyển docx sang PDF bằng LibreOffice để kiểm tra bố cục | Nhận xét định tính ở mục 8 được viết cho lần chạy này |
| Video | Kịch bản + hướng dẫn OBS/Game Bar | — | **Chưa có file video** |

## Lệnh đã chạy (theo thứ tự)
```
python -m venv .venv && pip install -r requirements.txt
python -m pytest -q                                   # 30 passed
python -m scripts.run_pipeline --mode smoke --overwrite
python -m scripts.generate_dataset --mode standard      # 300 runs, 0 failed, ~50 s on 4 cores
python -m scripts.make_splits --data data/standard
python -m scripts.build_features --data data/standard   # 69 000 windows per window length
python -m scripts.train --data data/standard --tag standard   # ~20 min (grid search)
python -m scripts.evaluate --data data/standard --model models/standard --tag standard   # ran once
python -m scripts.make_figures --results results/standard
python -m scripts.make_report --results results/standard --model models/standard --out reports/standard
python -m scripts.demo_headless_check --model models/standard
```
Sau khi có kết quả test, chỉ có những thay đổi *không ảnh hưởng tới mô hình hay số liệu*: bố cục biểu đồ, cỡ chữ bảng Word,
câu chữ trong báo cáo, và việc đặt `n_jobs=1` khi suy luận online (tăng tốc, cho kết quả giống hệt).

## Việc còn lại / gợi ý
- Chạy demo trên máy có màn hình (Windows/macOS) và quay video theo `docs/VIDEO_SCRIPT.md`.
- (Thí nghiệm mới, cần dùng một bộ stress mới vì bộ hiện tại đã được xem) Thêm đặc trưng tải khai báo hoặc phần dư mô hình,
  mở rộng lưới `n_on_grid`, rồi so sánh.
- Bootstrap theo nhóm để có khoảng tin cậy; GroupKFold trong tập phát triển.
