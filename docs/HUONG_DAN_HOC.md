# Hướng dẫn hiểu và tự trình bày dự án

Tài liệu này dành cho tác giả dự án. Nó giải thích từng giai đoạn theo bốn câu hỏi: *giải quyết vấn đề gì, vì sao chọn cách
này, chạy/quan sát gì để kiểm tra, cần hiểu điểm nào để tự giải thích*. Cuối tài liệu có bảng phân biệt **đã triển khai /
đã kiểm chứng / mới là giả định**, và một số câu hỏi phản biện thường gặp.

Số liệu kết quả luôn lấy từ `reports/standard/report_vi.md`. Tài liệu này chỉ trích lại một vài con số để minh họa.

---

## 0. Bức tranh tổng thể

```
quỹ đạo mục tiêu ─► bộ điều khiển PD + bù trọng lực ─► mô-men ─► robot mô phỏng (có ma sát, có thể bị bơm lỗi)
                         ▲                                              │
                         └──────────── cảm biến có nhiễu ◄──────────────┘
tín hiệu quan sát ─► cửa sổ 0,5–1 s ─► đặc trưng ─► Random Forest ─► điểm dự đoán ─► quy tắc duy trì ─► CẢNH BÁO
```

- **Đầu vào của AI:** góc và vận tốc đo, góc và vận tốc mục tiêu, mô-men áp dụng.
- **Đầu ra:** điểm dự đoán (0–1) mỗi 0,1 s và trạng thái cảnh báo bật/tắt.
- **Cách chứng minh:** kiểm thử vật lý độc lập với AI, chia dữ liệu theo nhóm thí nghiệm, đánh giá test và stress test
  riêng, so sánh với một baseline không cố ý làm yếu, và đo số báo nhầm/phút cùng độ trễ.

---

## Giai đoạn A — Mô phỏng vật lý (`src/dynamics.py`, `controller.py`, `trajectory.py`, `simulator.py`)

**Giải quyết vấn đề gì?** Không có robot thật, nên ta cần một "robot ảo" mà ở đó ma sát tăng gây ra hậu quả thông qua
**phương trình vật lý**: robot bị cản nhiều hơn, bộ điều khiển phải đẩy mạnh hơn, và sai số bám tăng. Ta không vẽ giả tín hiệu.

**Vì sao chọn cách này?**
- `M(q)q̈ + C(q,q̇)q̇ + G(q) + τf(q̇) = τ` là mô hình chuẩn của tay máy hai khâu (phương pháp Euler–Lagrange). Nó đủ
  đơn giản để tự kiểm tra bằng tay, nhưng vẫn có quán tính thay đổi theo tư thế, lực ly tâm/Coriolis và trọng lực.
- Ma sát được mô hình là `b·q̇ + τc·tanh(q̇/ε)`. Hàm `tanh` làm trơn bước nhảy của ma sát Coulomb khi q̇ đổi dấu.
  Nếu dùng `sign(q̇)`, tích phân số sẽ rung mạnh.
- Gia tốc được tính bằng cách **giải** hệ `M q̈ = r`, không nghịch đảo M. Cách này ổn định và rẻ hơn. Với ma trận 2×2,
  phép khử Gauss được viết tay và có kiểm thử so với `numpy.linalg.solve`.
- RK4 với bước 1 ms là phương pháp tích phân chính xác bậc 4 và dễ hiểu. Mô-men điều khiển được giữ nguyên giữa hai lần
  cập nhật 100 Hz (zero-order hold), giống bộ điều khiển số thật. Trọng lực, ma sát và quán tính vẫn được tính lại ở mỗi
  bước con của RK4.
- Bộ điều khiển PD + bù trọng lực là bộ điều khiển kinh điển. Nó **không bù ma sát**, nên ma sát tăng sẽ lộ ra trong tín
  hiệu. Bộ điều khiển chỉ biết tải *khai báo* (có sai lệch nhỏ) và không biết ma sát thật.
- Quán tính rotor quy đổi (`armature`) được cộng vào đường chéo của M. Lý do: với điều khiển 100 Hz, khâu quá nhẹ làm phần
  D của bộ điều khiển rời rạc mất ổn định. Robot công nghiệp thật cũng có quán tính rotor qua hộp số.

**Chạy/quan sát gì?**
- `python -m scripts.check_physics`: 17 kiểm thử vật lý, cộng với sai khác khi giảm bước tích phân và sai số bám khi robot khỏe.
- Mở demo, nhấn `3`, rồi quan sát sai số `e2` lớn hơn và mô-men `tau2` đổi dạng, trong khi quỹ đạo mục tiêu không đổi.

**Cần hiểu để tự giải thích:**
1. Vì sao `G(q)` phải bằng đạo hàm của thế năng (có kiểm thử cho điều này).
2. Vì sao ma sát luôn làm năng lượng giảm: `q̇·τf ≥ 0`.
3. Vì sao tải phải xuất hiện ở cả M và G. Nếu chỉ cộng vào G, robot "nặng" nhưng lại không có quán tính tương ứng.
4. Vì sao bộ điều khiển không được đọc hệ số ma sát lỗi. Nếu đọc, nó sẽ bù hoàn hảo và lỗi trở nên vô hình, hoặc ta vô tình
   cho AI biết đáp án.
5. Ý nghĩa của kiểm tra "giảm nửa bước": nếu kết quả gần như không đổi, sai số tích phân số đủ nhỏ so với hiệu ứng ta đo.

---

## Giai đoạn B — Sinh dữ liệu (`src/scenarios.py`, `dataset.py`)

**Giải quyết vấn đề gì?** AI cần nhiều ví dụ đa dạng: robot khỏe ở nhiều tốc độ và tải, và lỗi ở nhiều mức, thời điểm, khớp.
Mỗi ví dụ phải có nhãn đúng theo định nghĩa vật lý.

**Vì sao chọn cách này?**
- Nhãn đến từ **kịch bản vật lý** ("ma sát ≥ 1,5 lần mức khỏe"), không đến từ "sai số vượt ngưỡng". Nếu lấy nhãn theo sai số,
  AI chỉ học lại ngưỡng của chính ta, và robot khỏe chạy nhanh sẽ bị gán nhãn lỗi.
- Ma sát khỏe dao động ±15% giữa các lần chạy, nên AI không thể học thuộc một con số cố định.
- Nhiễu cảm biến được lấy từ **cùng một phân bố** cho mọi kịch bản. Nếu lần chạy lỗi có nhiễu khác, AI sẽ học nhận ra nhiễu
  thay vì nhận ra ma sát (một kiểu rò rỉ nhãn).
- Thời điểm lỗi ngẫu nhiên, nên AI không thể đoán theo "giây thứ 10".
- Lần chạy mất ổn định số được ghi vào `failed_runs.csv` và bị loại. Chúng không được biến thành ví dụ "lỗi".
- Một "nhóm" gồm 2 lần chạy giống nhau về quỹ đạo, tải và ma sát khỏe, chỉ khác nhiễu và thời điểm lỗi. Hai lần chạy trong
  cùng nhóm luôn nằm chung một tập, nếu không thì test sẽ chứa "bản sao" của dữ liệu train.

**Chạy/quan sát gì?**
- `data/standard/manifest.csv`: mỗi dòng là một lần chạy, có seed, tải, ma sát thật, thời điểm lỗi và `t_label`.
- `data/standard/signals/R0001.csv.gz` (thứ AI được thấy) so với `truth/R0001.csv.gz` (đáp án, AI không được thấy).
- Với lỗi tăng dần: xem cột `fric_mult` trong file truth, rồi kiểm tra nhãn bật đúng khi hệ số vượt 1,5.

**Cần hiểu:** sự khác nhau giữa *tín hiệu quan sát* và *ground truth*, lý do chia theo nhóm, và vì sao vùng ma sát
1,0–1,5 lần của lỗi tăng dần là vùng "mập mờ". Cảnh báo trong vùng đó không bị tính là báo nhầm, nhưng cũng chưa phải phát hiện.

---

## Giai đoạn C — Đặc trưng và Machine Learning (`src/features.py`, `training.py`, `alarm.py`)

**Giải quyết vấn đề gì?** Biến chuỗi tín hiệu thành con số tóm tắt mỗi 0,1 s để mô hình phân loại "bình thường" hay
"bất thường".

**Vì sao chọn cách này?**
- **Cửa sổ quá khứ** (0,5 hoặc 1 s): chỉ dùng dữ liệu đã có tại thời điểm dự đoán, giống hệt khi chạy thật. Không có bộ
  lọc hai chiều (vì nó "nhìn trước" tương lai).
- **Đặc trưng có ý nghĩa vật lý.** Ma sát luôn *cản theo hướng chuyển động*, nên robot luôn "trễ" sau mục tiêu theo hướng
  đang đi. Đặc trưng `e_dir_mean = mean(e·tanh(q̇_des/0,05))` đo đúng điều đó. Ngược lại, tải nặng chủ yếu kéo robot
  *xuống* theo trọng lực, còn chạy nhanh làm sai số tăng theo *gia tốc*. Các đặc trưng chuyển động mục tiêu (`qd_des_rms`,
  `qdd_des_rms`, `q_des_mean`) cho mô hình biết điều kiện vận hành, để nó không nhầm "chạy nhanh" với "ma sát cao". Trong
  kết quả, đặc trưng quan trọng nhất của RF là `j1_e_dir_mean`.
- **Random Forest** hợp với dữ liệu bảng cỡ vừa, ít phải chỉnh, khó overfit hơn một cây đơn, không cần chuẩn hóa, và có độ
  quan trọng đặc trưng để giải thích.
- **Baseline** ("sai số bám vượt ngưỡng đủ lâu") là cách một kỹ sư sẽ làm đầu tiên. Nó được chọn ngưỡng bằng đúng quy trình
  như RF, để so sánh công bằng.
- **Quy tắc duy trì cảnh báo:** cần nhiều cửa sổ dương liên tiếp mới bật, và nhiều cửa sổ âm liên tiếp mới tắt. Cách này
  giảm cảnh báo chập chờn, đổi lại độ trễ lớn hơn.
- Mọi lựa chọn (độ dài cửa sổ, tham số RF, ngưỡng, số cửa sổ duy trì) **chỉ dùng train/validation**.

**Chạy/quan sát gì?**
- `models/standard/training_report.json`: chứa điểm validation của mọi tổ hợp tham số đã thử.
- `results/standard/figures/feature_importance.png`.
- `python -m pytest tests/test_pipeline.py -v`: kiểm thử online = offline, tính nhân quả, và việc không dùng thời gian hay
  metadata.

**Cần hiểu:**
1. `predict_proba` của RF là **điểm** (tỷ lệ cây bỏ phiếu), chưa phải xác suất đã hiệu chuẩn.
2. Cửa sổ "trộn nhãn" (nửa khỏe, nửa lỗi) bị bỏ khi huấn luyện vì nhãn của nó mơ hồ. Khi đánh giá, chúng được giữ lại, vì
   khi chạy thật ta không được bỏ qua giai đoạn chuyển tiếp.
3. Đánh đổi giữa ngưỡng và số cửa sổ duy trì: ít báo nhầm hơn đi kèm phát hiện chậm hơn.

---

## Giai đoạn D — Đánh giá độc lập (`src/splits.py`, `evaluation.py`, `scripts/evaluate.py`)

**Giải quyết vấn đề gì?** Chứng minh mô hình hoạt động trên dữ liệu **chưa từng dùng để ra quyết định**, và đo những gì
người vận hành quan tâm: có phát hiện không, sau bao lâu, báo nhầm bao nhiêu lần mỗi phút.

**Vì sao chọn cách này?**
- Split được tạo **trước** khi huấn luyện, theo nhóm thí nghiệm, phân tầng theo kịch bản (60/20/20).
- Test thông thường (trong miền) và **stress test** (tải 2–3 kg, tốc độ cao hơn, quỹ đạo chuyển điểm chưa từng thấy) được báo
  cáo **riêng**. Stress test trả lời câu hỏi "nếu điều kiện vận hành khác lúc huấn luyện thì sao?".
- Chỉ số theo **sự kiện**: một cảnh báo là một lần chuyển sang trạng thái bật. Báo nhầm được tính trên mỗi phút robot khỏe.
  Độ trễ tính từ thời điểm lỗi được định nghĩa. Một cảnh báo nhầm đã bật từ trước **không** được tính là phát hiện.
- Test và stress được chạy đúng một lần với mô hình cuối. Sau khi xem kết quả, không có gì được chỉnh thêm.

**Chạy/quan sát gì?** `results/standard/metrics.json`, `runs_test_rf.csv` (từng lần chạy: báo nhầm, phát hiện, độ trễ) và
các hình trong `results/standard/figures/`.

**Cần hiểu để trả lời câu hỏi nghiên cứu (xem số chính xác trong báo cáo):**
- *Trong miền đã huấn luyện*, RF phát hiện được các lỗi trong test với số báo nhầm thấp, và vượt rõ baseline. Baseline buộc
  phải đặt ngưỡng cao để không báo nhầm khi robot khỏe chạy nhanh, nên bỏ sót nhiều lỗi nhẹ.
- *Ngoài miền* (stress test), cả hai đều báo nhầm nhiều, chủ yếu khi tải nặng và tốc độ cao hơn mức đã huấn luyện. Đây là
  kết quả **tiêu cực và quan trọng**: mô hình chưa tổng quát hóa ra ngoài miền dữ liệu. Nên trình bày điều này thẳng thắn,
  nó cho thấy bạn hiểu giới hạn của ML.
- Mỗi kịch bản trong test chỉ có 4–8 lần chạy, nên tỷ lệ phần trăm dao động mạnh. Hãy nói "6/6 lần" thay vì "100%".

---

## Giai đoạn E — Giao diện Pygame (`src/gui/app.py`, `src/online.py`)

**Giải quyết vấn đề gì?** Trình diễn trực quan: bơm lỗi bằng phím, rồi xem AI phản ứng theo thời gian thực.

**Vì sao chọn cách này?**
- Giao diện dùng **cùng hàm tạo đặc trưng** và **cùng quy tắc cảnh báo** như khi đánh giá offline. Có kiểm thử chứng minh
  hai đường cho ra đặc trưng và điểm giống hệt nhau.
- Cảnh báo chỉ đến từ mô hình. Ô "SIMULATOR GROUND TRUTH" để riêng và ghi rõ không phải đầu vào AI.
- Lịch vật lý, điều khiển, lấy mẫu, suy luận và hiển thị được tách nhau. Thời gian mô phỏng chạy theo đồng hồ thật, nên FPS
  thấp không làm robot chậm đi. Nếu máy quá chậm, giao diện báo NOT REAL-TIME.

**Chạy/quan sát gì?** `python -m scripts.run_demo`. Thứ tự gợi ý: chờ hết WARMING UP, nhấn `V` (nhanh) hoặc `L` (nặng) để
xem robot khỏe vẫn NORMAL, sau đó nhấn `3` và quan sát độ trễ đến ANOMALY ALARM.

**Cần hiểu:** vì sao có giai đoạn WARMING UP, và vì sao cảnh báo đến chậm hơn lúc bơm lỗi (cửa sổ 1 s và 8 cửa sổ duy trì
≈ 0,8 s, cộng thời gian để tín hiệu thay đổi).

**Kết quả đã quan sát khi thử demo** (robot khỏe, 30 s mô phỏng mỗi tổ hợp, chạy tự động):

| Tải | Tốc độ normal (0,30/0,38 Hz) | Tốc độ fast (0,50/0,55 Hz) |
|---|---|---|
| 0,5 kg | 0 cảnh báo | 3 cảnh báo nhầm |
| 1,5 kg | 0 cảnh báo | 2 cảnh báo nhầm |
| 2,0 kg | 0 cảnh báo | 3 cảnh báo nhầm |

Ở chế độ fast, *cả hai* khớp cùng chạy gần tần số lớn nhất của miền huấn luyện (0,55 Hz), với biên độ lớn. Tổ hợp này
nằm ở **góc của miền dữ liệu** và hiếm gặp khi huấn luyện, vì tần số của từng khớp được lấy ngẫu nhiên độc lập. Đây là ví dụ
cụ thể của vấn đề tổng quát hóa mà stress test cũng chỉ ra. Không nên giấu điều này trong video: hãy trình bày nó như một
giới hạn đã phát hiện.

---

## Đã triển khai / đã kiểm chứng / giả định

| Nội dung | Trạng thái |
|---|---|
| Động lực học 2 khâu, ma sát, tải, RK4, PD + bù trọng lực | **Đã triển khai**, **đã kiểm chứng bằng kiểm thử** (17 kiểm thử vật lý) |
| Sinh dữ liệu headless, seed, manifest, tái lập từng lần chạy | Đã triển khai, đã kiểm thử tái lập |
| Không rò rỉ: split theo nhóm, đặc trưng không dùng thời gian/metadata, nhân quả | Đã triển khai, đã kiểm thử tự động |
| Online = offline | Đã kiểm thử (đặc trưng và điểm giống hệt nhau) |
| Kết quả test/stress | Đã chạy thật. Số trong báo cáo đọc từ `metrics.json` |
| Giao diện | Đã chạy trên máy chủ không màn hình (SDL dummy), chụp ảnh tự động và đo real-time factor ≈ 1. **Chưa thử trên Windows/macOS có màn hình thật** |
| Thông số robot (khối lượng, chiều dài, ma sát danh định) | **Giả định hợp lý**, không lấy từ robot cụ thể nào |
| Ma sát tăng đại diện cho "khô dầu / mòn" | **Giả định mô hình hóa**; không chứng minh nguyên nhân |
| Hành vi trên robot thật | **Chưa kiểm chứng** |
| Video demo | **Chưa có file video**. Chỉ có kịch bản và hướng dẫn quay (`docs/VIDEO_SCRIPT.md`) |

---

## Câu hỏi phản biện thường gặp

1. **"Sao em biết mô phỏng đúng?"** Nhờ các kiểm thử độc lập với AI: bảo toàn năng lượng khi không ma sát, gradient thế
   năng, cân bằng tĩnh, hội tụ khi giảm bước tích phân. Tuy vậy, mô phỏng đúng *phương trình* không có nghĩa là đúng với
   *robot thật*.
2. **"AI có nhìn trộm đáp án không?"** Không. Đặc trưng chỉ lấy từ tín hiệu quan sát, có kiểm thử. Split theo nhóm được tạo
   trước khi huấn luyện, ngưỡng chọn trên validation, test chỉ chạy một lần.
3. **"Sao không dùng deep learning?"** Dữ liệu dạng bảng cỡ vừa, và cần giải thích được. RF đủ tốt trong miền. Vấn đề còn lại
   là tổng quát hóa ngoài miền, và một mô hình lớn hơn không tự giải quyết được vấn đề đó.
4. **"Sao kết quả stress tệ?"** Mô hình học quan hệ giữa sai số và điều kiện vận hành trong miền đã thấy. Khi tải và tốc độ
   vượt miền đó, sai số do quán tính và trọng lực bị hiểu nhầm thành ma sát. Hướng sửa: thêm đặc trưng phần dư mô hình hoặc
   tải khai báo, mở rộng dữ liệu, phát hiện dữ liệu ngoài miền.
5. **"Đây có phải dự báo bảo trì (RUL) không?"** Không. Đây là phát hiện *đã có* bất thường, không dự báo *khi nào* hỏng.

## Phần nào là của bạn?

Mã nguồn ban đầu được viết với sự hỗ trợ của trợ lý AI. Để dự án thực sự là của bạn và bảo vệ được trước hội đồng, gợi ý:
tự đọc và giải thích từng file trong `src/`, tự thay đổi ít nhất một quyết định rồi chạy lại và so sánh (ví dụ thêm đặc
trưng tải khai báo, đổi lưới `n_on_grid`, đổi ngưỡng nhãn 1,5), ghi lại bạn đã làm gì trong `PROGRESS.md`, và khi trình bày
hãy nói rõ công cụ AI đã hỗ trợ những phần nào.
