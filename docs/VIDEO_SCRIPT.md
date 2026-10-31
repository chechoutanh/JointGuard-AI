# Kịch bản video demo (≈ 90–120 giây)

> Video **chưa được quay**. Môi trường xây dựng dự án không có màn hình nên chỉ chụp được ảnh tĩnh
> (`results/screenshots/`). Hãy quay trên máy của bạn theo hướng dẫn bên dưới.

Chuẩn bị: `python -m scripts.run_demo --model models/standard`, mở sẵn `reports/standard/figures/comparison_rf_vs_baseline.png`
và `false_alarms_by_scenario.png`.

| Thời gian | Hình ảnh | Lời thoại gợi ý |
|---|---|---|
| 0:00–0:15 | Tiêu đề + cửa sổ demo đang WARMING UP | "Em mô phỏng một robot hai khớp trong mặt phẳng thẳng đứng, để kiểm tra xem từ mô-men và sai số chuyển động có phát hiện được ma sát khớp tăng bất thường mà không báo nhầm khi robot khỏe chạy nhanh hay mang tải nặng hay không. Toàn bộ là mô phỏng." |
| 0:15–0:30 | Robot khỏe, tải 0,5 kg, NORMAL | "Bên trái là robot: nét chấm là đường mục tiêu, nét liền là đường đi thật. Bên phải là mô-men, sai số bám và điểm dự đoán của mô hình. Ô màu vàng là ground truth của bộ mô phỏng, chỉ để người xem đối chiếu, AI không đọc ô này." |
| 0:30–0:50 | Nhấn `L` hai lần (tải 2,0 kg), chờ hết WARMING UP | "Robot vẫn khỏe nhưng mang tải nặng hơn: mô-men khớp 1 tăng rõ, còn mô hình vẫn báo NORMAL." *(Khi thử tự động trong 30 s ở tốc độ normal, cả ba mức tải đều không có cảnh báo nhầm. Hãy kiểm tra lại trên máy bạn trước khi quay.)* |
| 0:50–1:15 | Nhấn `3` (ma sát khớp 2 x3) | "Bây giờ em tăng ma sát khớp 2 lên ba lần. Phím này chỉ đổi tham số trong phương trình vật lý. Sai số khớp 2 lớn hơn, điểm dự đoán tăng, và sau một khoảng trễ ngắn (khi thử tự động: 0,8 s với tải 0,5 kg) hệ thống bật ANOMALY ALARM. Ô ground truth cho thấy độ trễ." |
| 1:15–1:30 | Nhấn `R`, rồi `V` (tốc độ fast) | "Một giới hạn em phát hiện được: khi robot khỏe chạy ở góc tốc độ cao của miền dữ liệu, mô hình đôi khi báo nhầm." *(Khi thử tự động: 2–3 cảnh báo nhầm trong 30 s.)* |
| 1:30–1:50 | Ảnh `comparison_rf_vs_baseline.png`, `false_alarms_by_scenario.png` | Đọc số từ báo cáo: số lần phát hiện/tổng, báo nhầm/phút, trên test và stress test, so với baseline. "Trong miền đã huấn luyện, Random Forest tốt hơn baseline. Ngoài miền, như tải 2–3 kg và tốc độ cao, cả hai báo nhầm nhiều, nên mô hình chưa tổng quát hóa." |
| 1:50–2:00 | Slide giới hạn | "Kết quả chưa kiểm chứng trên robot thật. Ma sát tăng chỉ là mô hình hóa một bất thường. Đây là phát hiện bất thường, chưa phải dự báo tuổi thọ. Dự án được làm với sự hỗ trợ của trợ lý AI." |

Lưu ý khi quay: không cắt ghép để che các cảnh báo nhầm, và không nói "chính xác 100%" khi kết quả chỉ dựa trên vài lần chạy.

## Cách quay màn hình

**Windows – Xbox Game Bar (có sẵn):** mở cửa sổ demo, nhấn `Win + G` → Capture → Record (hoặc `Win + Alt + R` để bắt
đầu/dừng). Video được lưu trong `Videos\Captures`. Game Bar chỉ quay một cửa sổ, nên các ảnh kết quả cần mở bằng chương
trình khác và quay riêng, hoặc dùng OBS.

**OBS Studio (Windows/macOS/Linux, miễn phí, obsproject.com):**
1. Sources → `+` → *Window Capture* (macOS: *macOS Screen Capture*), rồi chọn cửa sổ demo.
2. Thêm một *Display Capture* nếu muốn chuyển sang ảnh kết quả.
3. Settings → Output: định dạng `mp4` (hoặc `mkv` rồi Remux sang mp4), 1080p, 30 hoặc 60 fps.
4. Bấm *Start Recording*, làm theo kịch bản, rồi bấm *Stop Recording*.

**macOS:** `Cmd + Shift + 5` → Record Selected Portion.
