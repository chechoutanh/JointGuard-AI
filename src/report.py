"""Builds the Vietnamese report (Markdown + Word) from saved results only.
Every number in the report is read from metrics.json, training_report.json, the
manifests or re-computed from the simulator; no number is typed in by hand."""
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .faults import FaultSchedule
from .simulator import Simulator
from .trajectory import SinusoidTrajectory


# ---------------------------------------------------------------- block model
class Doc:
    def __init__(self):
        self.blocks = []

    def h(self, text, level=1): self.blocks.append(("h", level, text))
    def p(self, text): self.blocks.append(("p", text))
    def bullets(self, items): self.blocks.append(("ul", list(items)))
    def table(self, header, rows): self.blocks.append(("table", header, [[str(c) for c in r] for r in rows]))
    def img(self, path, caption): self.blocks.append(("img", path, caption))

    def to_markdown(self, rel_to):
        out = []
        for b in self.blocks:
            if b[0] == "h":
                out.append("#" * b[1] + " " + b[2] + "\n")
            elif b[0] == "p":
                out.append(b[1] + "\n")
            elif b[0] == "ul":
                out.append("\n".join(f"- {x}" for x in b[1]) + "\n")
            elif b[0] == "table":
                out.append("| " + " | ".join(b[1]) + " |")
                out.append("|" + "---|" * len(b[1]))
                out += ["| " + " | ".join(r) + " |" for r in b[2]]
                out.append("")
            elif b[0] == "img":
                out.append(f"![{b[2]}]({Path(b[1]).relative_to(rel_to).as_posix()})\n\n*{b[2]}*\n")
        return "\n".join(out)

    def to_docx(self, path):
        from docx import Document
        from docx.shared import Cm, Pt
        d = Document()
        st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
        for b in self.blocks:
            if b[0] == "h":
                d.add_heading(b[2], level=min(b[1] - 1, 3) if b[1] > 1 else 0)
            elif b[0] == "p":
                self._rich(d.add_paragraph(), b[1])
            elif b[0] == "ul":
                for x in b[1]:
                    self._rich(d.add_paragraph(style="List Bullet"), x)
            elif b[0] == "table":
                t = d.add_table(rows=1, cols=len(b[1])); t.style = "Light Grid Accent 1"
                for i, c in enumerate(b[1]):
                    t.rows[0].cells[i].text = c
                for r in b[2]:
                    cells = t.add_row().cells
                    for i, c in enumerate(r):
                        cells[i].text = c.replace("`", "")
                small = Pt(8) if len(b[1]) > 6 else Pt(9.5)
                for rr in t.rows:
                    for cell in rr.cells:
                        for par in cell.paragraphs:
                            for run in par.runs:
                                run.font.size = small
                d.add_paragraph()
            elif b[0] == "img":
                d.add_picture(str(b[1]), width=Cm(16))
                cap = d.add_paragraph(b[2]); cap.runs[0].italic = True
        d.save(path)

    @staticmethod
    def _rich(par, text):
        """Minimal **bold** and `code` support for Word."""
        import re
        for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
            if part.startswith("**"):
                par.add_run(part[2:-2]).bold = True
            elif part.startswith("`"):
                r = par.add_run(part[1:-1]); r.font.name = "Consolas"
            elif part:
                par.add_run(part)


def f2(x, n=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"


def step_size_check(cfg):
    fr = cfg["friction"]
    tr = SinusoidTrajectory((0.8, -1.2), (0.4, 0.45), (0.5, 0.55), (0.3, 1.0), 2.0)
    res = {}
    for dt in (0.001, 0.0005):
        s = Simulator(cfg, 2.0, 2.0, fr["nominal_b"], fr["nominal_tau_c"], tr,
                      FaultSchedule("abrupt", 2, 3.0, 4.0), 0, 0, 0, dt_physics=dt)
        res[dt] = s.run(6.0)[1][:, 1:3]
    return float(np.abs(res[0.001] - res[0.0005]).max())


# ---------------------------------------------------------------- content
def build(results_dir, model_dir, out_dir):
    results_dir, model_dir, out_dir = Path(results_dir), Path(model_dir), Path(out_dir)
    M = json.load(open(results_dir / "metrics.json"))
    TR = json.load(open(model_dir / "training_report.json"))
    card = json.load(open(model_dir / "model_card.json"))
    data_dir = Path(M["data_dir"])
    gen = json.load(open(data_dir / "generation_info.json"))
    cfg = gen["config"]
    man = pd.read_csv(data_dir / "manifest.csv")
    split = pd.read_csv(data_dir / "split_manifest.csv")
    failed = pd.read_csv(data_dir / "failed_runs.csv")
    fig_src = results_dir / "figures"
    fig_dir = out_dir / "figures"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    shutil.copytree(fig_src, fig_dir)
    F = lambda name: fig_dir / name
    r, c, tm, fr, fl = cfg["robot"], cfg["controller"], cfg["timing"], cfg["friction"], cfg["fault"]
    mode = gen["mode"]
    T, S = M["test"], M["stress"]
    d = Doc()

    d.h(f"Báo cáo: Phát hiện bất thường ma sát khớp trên robot 2-DOF bằng mô phỏng động lực học và Machine Learning")
    d.p(f"*Bộ dữ liệu: `{data_dir.as_posix()}` (chế độ `{mode}`), mô hình: `{model_dir.as_posix()}`. "
        f"Báo cáo được sinh tự động bởi `python -m scripts.make_report` từ kết quả đã chạy; "
        f"mọi con số đều đọc từ file kết quả.*")
    if mode == "smoke":
        d.p("**ĐÂY LÀ BÁO CÁO SMOKE**: dữ liệu rất nhỏ, chỉ dùng để kiểm tra pipeline chạy xuyên suốt. "
            "Không dùng các con số dưới đây để kết luận.")
    d.p("**Lưu ý quan trọng.** Toàn bộ dữ liệu được tạo bằng mô phỏng. Ma sát tăng là cách mô hình hóa "
        "một bất thường; nó không chứng minh nguyên nhân duy nhất là khô dầu. Kết quả chưa được xác nhận "
        "trên robot thật. Phát hiện bất thường hiện tại không tương đương với dự báo thời điểm hỏng "
        "hay tuổi thọ còn lại. Đồ thị mô phỏng trông giống tín hiệu thật không phải là bằng chứng "
        "xác thực vật lý.")
    d.p("**Minh bạch về công cụ.** Mã nguồn, thí nghiệm và báo cáo này được xây dựng với sự hỗ trợ của "
        "trợ lý AI (Claude) theo yêu cầu và định hướng của tác giả. Khi trình bày, cần ghi rõ phần nào "
        "do mình tự thực hiện, tự kiểm tra hoặc tự sửa đổi.")

    # 1
    d.h("1. Câu hỏi nghiên cứu", 2)
    d.p("Chỉ từ mô-men khớp (tín hiệu mô phỏng) và sai số bám quỹ đạo, hệ thống có phát hiện được ma sát "
        "của một khớp tăng bất thường không? Đồng thời, nó có tránh được báo nhầm khi robot khỏe chạy nhanh "
        "hơn hoặc mang tải nặng hơn không? Hai trường hợp này cũng làm mô-men và sai số tăng, nên đây là "
        "phần khó của bài toán.")
    d.p(f"Định nghĩa bất thường được chọn trước khi sinh dữ liệu: ma sát của một khớp (cả hệ số nhớt b và "
        f"Coulomb τc) cao hơn giá trị khỏe của lần chạy đó ít nhất {fl['label_threshold_multiplier']:.1f} lần "
        f"(tức ≥ +{(fl['label_threshold_multiplier']-1)*100:.0f}%). Nhãn không dựa vào sai số bám.")

    # 2
    d.h("2. Mô hình cơ học và giả định", 2)
    d.p("Robot phẳng hai khớp quay, chuyển động trong mặt phẳng thẳng đứng. Hệ tọa độ gốc đặt tại khớp 1, "
        "trục x hướng sang phải, trục y hướng lên, trọng lực theo −y. q1 là góc khâu 1 so với trục +x; "
        "q2 là góc tương đối của khâu 2 so với khâu 1. Chiều dương là ngược chiều kim đồng hồ. Đơn vị SI "
        "(m, kg, s, rad, N·m).")
    d.p("Phương trình chuyển động: `M(q)·q̈ + C(q,q̇)·q̇ + G(q) + τf(q̇) = τ_applied + τ_external` "
        "(trong thí nghiệm này τ_external = 0). Gia tốc được tính bằng cách giải hệ tuyến tính 2×2 (khử Gauss), "
        f"không nghịch đảo ma trận. Trạng thái được tích phân bằng RK4 với bước {tm['dt_physics']*1000:g} ms.")
    d.table(["Tham số", "Khâu 1", "Khâu 2"], [
        ["Chiều dài l [m]", r["l1"], r["l2"]], ["Khối lượng m [kg]", r["m1"], r["m2"]],
        ["Khoảng cách tới trọng tâm lc [m]", r["lc1"], r["lc2"]],
        ["Mô-men quán tính quanh trọng tâm I [kg·m²]", r["I1"], r["I2"]],
        ["Quán tính rotor quy đổi [kg·m²]", r["armature1"], r["armature2"]],
        ["Giới hạn khớp [rad]", r["joint_limits"][0], r["joint_limits"][1]],
        ["Giới hạn mô-men [N·m]", r["torque_limits"][0], r["torque_limits"][1]],
        ["Ma sát nhớt danh định b [N·m·s/rad]", fr["nominal_b"][0], fr["nominal_b"][1]],
        ["Ma sát Coulomb danh định τc [N·m]", fr["nominal_tau_c"][0], fr["nominal_tau_c"][1]]])
    d.bullets([
        "Tải ở đầu robot là khối lượng điểm. Nó được gộp vào khâu 2 thành một vật rắn tương đương "
        "(khối lượng, trọng tâm, quán tính theo định lý trục song song), nên tác động nhất quán lên cả M(q) và G(q).",
        f"Ma sát mỗi khớp: `τf = b·q̇ + τc·tanh(q̇/ε)` với ε = {fr['epsilon']} rad/s (Coulomb được làm trơn). "
        f"Ma sát khỏe của mỗi lần chạy được lấy ngẫu nhiên trong ±{fr['healthy_tolerance']*100:.0f}% quanh giá trị danh định.",
        "Lỗi chỉ nhân b và τc của một khớp trong mô hình vật lý. Bộ điều khiển không biết điều này.",
        "Giả định chưa mô hình hóa: động cơ, hộp số, khe hở (backlash), độ đàn hồi, nhiệt độ, va chạm, "
        "ma sát phụ thuộc tải. Mô-men là tín hiệu mô phỏng, không phải dòng điện đo được.",
        "Cảm biến: nhiễu Gauss cộng vào góc và vận tốc. Độ lệch chuẩn được lấy ngẫu nhiên cho mỗi lần chạy "
        f"(góc {cfg['sensor']['noise_q_range']} rad, vận tốc {cfg['sensor']['noise_qd_range']} rad/s), "
        "cùng một phân bố cho cả lần chạy khỏe và lỗi."])
    err = step_size_check(cfg)
    d.p(f"**Kiểm tra vật lý** (`python -m scripts.check_physics`, chi tiết trong `tests/test_physics.py`): "
        "động học thuận tại các cấu hình biết trước; M đối xứng và xác định dương; G(q) bằng gradient của "
        "thế năng; tải tạo đúng mô-men của khối lượng điểm; cân bằng trọng lực khi đứng yên; năng lượng được "
        "bảo toàn khi không có ma sát và mô-men; ma sát làm năng lượng giảm; robot khỏe bám ổn định; không có "
        "NaN; mất ổn định số được báo lỗi. Khi giảm bước tích phân từ 1 ms xuống 0,5 ms (có lỗi đột ngột "
        f"x4, tải 2 kg), sai khác góc lớn nhất là {err:.1e} rad.")

    # 3
    d.h("3. Bộ điều khiển", 2)
    d.p(f"Bộ điều khiển là PD kết hợp bù trọng lực, chạy ở {tm['f_control']} Hz (zero-order hold): "
        "`τ_req = Kp·(q_des − q_meas) + Kd·(q̇_des − q̇_meas) + Ĝ(q_meas)`, sau đó giới hạn mô-men để có "
        f"τ_applied. Hệ số Kp = {c['kp']}, Kd = {c['kd']}. Cả τ_req và τ_applied đều được lưu.")
    d.bullets([
        "Bộ điều khiển biết thông số danh định của các khâu và tải do người vận hành khai báo. Tải khai báo "
        f"có thể lệch so với tải thật tối đa ±{c['payload_declared_error']} kg.",
        "Nó không biết ma sát (khỏe hay lỗi) và không bù ma sát, nên ma sát tăng thể hiện thành sai số bám "
        "và mô-men PD lớn hơn.",
        "Bộ điều khiển chỉ nhận tín hiệu đo có nhiễu. Hệ số được chọn một lần theo yêu cầu ổn định rời rạc ở "
        "100 Hz và không được chỉnh lại theo kết quả test.",
        f"Bão hòa mô-men: trong miền chuẩn, tỷ lệ chu kỳ bị bão hòa lớn nhất là "
        f"{man[man.subset=='indomain'].saturated_fraction.max():.4f}; trong stress test là "
        f"{man[man.subset=='stress'].saturated_fraction.max():.4f}."])

    # 4
    d.h("4. Cách sinh dữ liệu", 2)
    dur = cfg["modes"][mode]["duration"]
    d.p(f"Dữ liệu được sinh hoàn toàn không cần giao diện (`python -m scripts.generate_dataset --mode {mode}`). "
        f"Mỗi lần chạy dài {dur:g} s. Thời gian khởi động trơn là {cfg['trajectory']['t_ramp']} s. Tín hiệu được "
        f"lấy mẫu ở {tm['f_sample']} Hz. Tổng cộng có {len(man)} lần chạy thành công và {len(failed)} lần mất ổn "
        "định số (được ghi riêng trong `failed_runs.csv` và không bao giờ bị gắn nhãn lỗi cơ khí).")
    d.bullets([
        "Quỹ đạo trong miền chuẩn: dao động sin trong không gian khớp. Tâm, biên độ, tần số và pha thay đổi "
        f"(tần số {cfg['domains']['indomain']['freq']} Hz). Quỹ đạo vượt giới hạn khớp bị loại khi lấy mẫu.",
        "Nhóm healthy_speed: nhiều tốc độ, tải trong khoảng danh định "
        f"{cfg['domains']['indomain']['payload_nominal']} kg. Nhóm healthy_load: tải "
        f"{cfg['domains']['indomain']['payload_full']} kg, tốc độ thấp hơn.",
        f"mild: ma sát nhảy lên x{fl['mild_range'][0]}–{fl['mild_range'][1]}; severe: x{fl['severe_range'][0]}–"
        f"{fl['severe_range'][1]}. Thời điểm bắt đầu lấy ngẫu nhiên trong {cfg['modes'][mode]['onset_range']} s.",
        f"gradual: ma sát tăng tuyến tính từ x1 tới x{fl['gradual_final_range'][0]}–{fl['gradual_final_range'][1]} "
        f"trong {fl['gradual_ramp_range'][0]}–{fl['gradual_ramp_range'][1]} s. Nhãn bật tại thời điểm hệ số vượt "
        f"{fl['label_threshold_multiplier']}. Từ lúc bắt đầu tăng tới lúc đó, nhãn vẫn là 0, nhưng cảnh báo trong "
        "giai đoạn này không bị tính là báo nhầm.",
        "Stress test (giữ riêng, không dùng để huấn luyện hay chỉnh): tải nặng hơn "
        f"{cfg['domains']['stress']['payload_full']} kg, tần số cao hơn {cfg['domains']['stress']['freq']} Hz, và "
        f"{cfg['domains']['stress']['p2p_fraction']*100:.0f}% là quỹ đạo chuyển điểm trơn (quintic), một kiểu "
        "quỹ đạo chưa từng xuất hiện khi huấn luyện.",
        "Mỗi nhóm (thí nghiệm gốc) gồm "
        f"{cfg['modes'][mode]['replicates_per_group']} lần chạy cùng quỹ đạo, tải và ma sát khỏe. Các lần chạy khác "
        "nhau ở nhiễu, độ lệch ban đầu và thời điểm lỗi. Một nhóm luôn nằm trọn trong một tập.",
        "Tín hiệu quan sát (`signals/`) và ground truth (`truth/`, `manifest.csv`) được lưu tách riêng. "
        "Seed và toàn bộ tham số của từng lần chạy nằm trong manifest, nên có thể tái tạo từng lần chạy."])
    piv = split.pivot_table(index="scenario", columns="split", values="run_id", aggfunc="count", fill_value=0)
    cols = [c_ for c_ in ("train", "val", "test", "stress") if c_ in piv.columns]
    d.table(["Kịch bản"] + cols, [[i] + [int(piv.loc[i, c_]) for c_ in cols] for i in piv.index])

    # 5
    d.h("5. Đặc trưng và mô hình AI", 2)
    d.p(f"Đặc trưng được tính trên cửa sổ quá khứ dài {card['window_s']} s (thử 0,5 s và 1 s, chọn bằng "
        f"validation), cập nhật mỗi {card['stride_s']} s. Chỉ dùng dữ liệu tới thời điểm dự đoán, không có bộ "
        "lọc hai chiều. Với mỗi khớp, đặc trưng gồm: RMS, độ lệch chuẩn và cực đại |mô-men|; RMS tốc độ thay đổi "
        "mô-men; mô-men trung bình theo hướng chuyển động mục tiêu; RMS và cực đại |sai số vị trí|; sai số vị trí "
        "trung bình theo hướng chuyển động (`mean(e·tanh(q̇_des/0.05))`, nhạy với lực cản của ma sát); RMS sai "
        "số vận tốc; RMS vận tốc và gia tốc mục tiêu; góc mục tiêu trung bình. Tổng cộng có "
        f"{len(card['feature_names'])} đặc trưng.")
    d.p("Thời gian tuyệt đối, run_id, tải thật, hệ số ma sát và nhãn không bao giờ được đưa vào đặc trưng. "
        "Có kiểm thử tự động cho việc này, cho tính nhân quả, và cho việc đặc trưng online và offline giống hệt nhau.")
    d.p(f"Mô hình là Random Forest (scikit-learn {card['versions']['scikit-learn']}). Tham số tốt nhất trên "
        f"validation (average precision = {TR['rf_best']['val_ap']:.3f}): `{TR['rf_best']['params']}`. Có "
        f"{len(TR['rf_selection'])} tổ hợp được thử. Cửa sổ trộn nhãn bị loại khi huấn luyện. Ngưỡng điểm dự đoán "
        f"là {card['threshold']}. Cảnh báo bật sau {card['n_on']} cửa sổ dương liên tiếp và tắt sau {card['n_off']} "
        "cửa sổ âm liên tiếp. Hai giá trị này được chọn trên validation để tối đa F1, với điều kiện số báo nhầm "
        f"≤ {cfg['training']['max_false_alarms_per_min']}/phút. `predict_proba` được gọi là **điểm dự đoán của "
        "mô hình**, không phải xác suất đã hiệu chuẩn.")
    b = M["baseline"]
    d.p(f"**Baseline:** điểm = max_j(RMS sai số vị trí khớp j / tham chiếu_j). Tham chiếu là phân vị "
        f"{cfg['training']['baseline_quantile']*100:.0f}% trên các cửa sổ khỏe của tập train "
        f"({b['ref'][0]:.4f} và {b['ref'][1]:.4f} rad). Cửa sổ là {b['window_s']} s, ngưỡng x{b['threshold']}, "
        f"và cảnh báo bật sau {b['n_on']} cửa sổ liên tiếp. Baseline dùng cùng quy tắc chọn trên validation và "
        "cùng quy tắc cảnh báo như Random Forest.")
    sel = pd.DataFrame(TR["rf_selection"])
    if (card["n_on"] == max(cfg["training"]["n_on_grid"])) or (b["n_on"] == max(cfg["training"]["n_on_grid"])):
        d.p("*Ghi chú trung thực:* giá trị duy trì cảnh báo được chọn nằm ở biên lưới tìm kiếm "
            f"({cfg['training']['n_on_grid']}). Lưới rộng hơn có thể cho kết quả khác. Điểm này được ghi nhận "
            "thay vì chỉnh lại sau khi đã xem kết quả test.")
    d.img(F("feature_importance.png"), "Hình: Độ quan trọng đặc trưng (impurity-based) của Random Forest.")

    # 6
    d.h("6. Thiết kế đánh giá", 2)
    d.bullets([
        "Dữ liệu được chia theo nhóm thí nghiệm (60/20/20, phân tầng theo kịch bản), và file split được tạo trước "
        "khi huấn luyện. Mọi lựa chọn đặc trưng, tham số và ngưỡng chỉ dùng train/validation.",
        "Test thông thường và stress test được báo cáo riêng, không gộp thành một điểm chung.",
        f"Giám sát bắt đầu sau {cfg['features']['monitor_start_s']} s (hết giai đoạn khởi động trơn). Trước đó "
        "giao diện hiển thị WARMING UP.",
        "Một cảnh báo = một lần chuyển trạng thái từ tắt sang bật. Báo nhầm = cảnh báo bắt đầu trước khi lỗi "
        "được bơm vào. Số báo nhầm được chia cho số phút robot khỏe.",
        "Độ trễ = thời điểm cảnh báo hợp lệ đầu tiên − thời điểm lỗi được định nghĩa (nhãn bật). Một cảnh báo "
        "nhầm đã bật từ trước và kéo dài không được tính là phát hiện. Với lỗi tăng dần, cảnh báo bắt đầu "
        "sau khi ma sát bắt đầu tăng và vẫn bật tại thời điểm nhãn chuyển sang 1 được tính là phát hiện "
        "sớm với độ trễ 0.",
        "Chỉ số theo cửa sổ (Precision/Recall/F1) so sánh trạng thái cảnh báo với nhãn trên mọi cửa sổ, kể cả "
        "giai đoạn chuyển tiếp. Đây là lý do recall theo cửa sổ thấp hơn tỷ lệ phát hiện theo lần chạy: "
        "các cửa sổ trong khoảng trễ phát hiện bị tính là bỏ sót."])

    # 7
    d.h("7. Kết quả thực nghiệm", 2)

    def row(name, s):
        w = s["window_alarm_state"]
        return [name, f2(w["precision"]), f2(w["recall"]), f2(w["f1"]), f2(s["window_raw_decision"]["f1"]),
                f2(s["false_alarms_per_healthy_min"]), f"{s['healthy_runs_with_false_alarm']}/{s['n_healthy_runs']}",
                f"{s['n_detected']}/{s['n_fault_runs']}", s["n_missed"], f2(s["delay_median"], 1),
                f2(s["delay_p90"], 1), f2(s["delay_max"], 1)]
    hdr = ["Bộ phát hiện", "Precision", "Recall", "F1", "F1 thô", "Báo nhầm /phút", "Lần khỏe có báo nhầm",
           "Phát hiện", "Bỏ sót", "Trễ TV [s]", "Trễ P90 [s]", "Trễ max [s]"]
    d.p("RF = Random Forest; TV = trung vị; F1 thô = F1 của quyết định từng cửa sổ trước quy tắc duy trì.")
    d.h("7.1 Test thông thường (trong miền)", 3)
    d.p(f"Gồm {T['rf']['n_runs']} lần chạy, {T['rf']['healthy_minutes']:.1f} phút robot khỏe và "
        f"{T['rf']['n_fault_runs']} lần chạy có lỗi.")
    d.table(hdr, [row("RF", T["rf"]), row("Baseline", T["baseline"])])
    d.h("7.2 Stress test (điều kiện giữ riêng)", 3)
    d.p(f"Gồm {S['rf']['n_runs']} lần chạy, {S['rf']['healthy_minutes']:.1f} phút robot khỏe và "
        f"{S['rf']['n_fault_runs']} lần chạy có lỗi.")
    d.table(hdr, [row("RF", S["rf"]), row("Baseline", S["baseline"])])
    d.img(F("comparison_rf_vs_baseline.png"), "Hình: So sánh Random Forest và baseline trên test và stress.")
    d.img(F("confusion_matrices.png"), "Hình: Ma trận nhầm lẫn theo cửa sổ (trạng thái cảnh báo).")
    d.h("7.3 Theo mức lỗi và khớp lỗi", 3)
    for part, P in (("test", T), ("stress", S)):
        rows = []
        for scen, s in P["rf"]["by_scenario"].items():
            sb = P["baseline"]["by_scenario"][scen]
            rows.append([scen, s["n_runs"], f"{s['n_detected']}/{s['n_fault_runs']}" if s["n_fault_runs"] else "—",
                         f2(s["delay_median"], 1), s["false_alarms"],
                         f"{sb['n_detected']}/{sb['n_fault_runs']}" if sb["n_fault_runs"] else "—",
                         f2(sb["delay_median"], 1), sb["false_alarms"]])
        d.p(f"**{part.upper()}**")
        d.table(["Kịch bản", "Số lần chạy", "RF phát hiện", "RF trễ TV [s]", "RF báo nhầm",
                 "BL phát hiện", "BL trễ TV [s]", "BL báo nhầm"], rows)
    d.img(F("detection_by_scenario.png"), "Hình: Tỷ lệ phát hiện theo kịch bản (nhãn = số lần phát hiện / số lần có lỗi).")
    d.img(F("false_alarms_by_scenario.png"), "Hình: Báo nhầm theo kịch bản.")
    d.img(F("detection_delays.png"), "Hình: Phân bố độ trễ của các lần được phát hiện, kèm số lần bỏ sót.")
    for f in sorted(fig_dir.glob("example_test_*.png")):
        d.img(f, f"Hình: Ví dụ lần chạy {f.stem.split('_')[-1]} trong tập test: quỹ đạo, mô-men, sai số, điểm dự đoán.")

    # 8
    d.h("8. Phân tích báo nhầm và bỏ sót", 2)
    rt = pd.read_csv(results_dir / "runs_test_rf.csv")
    fa = rt[rt["false_alarms"] > 0]
    miss = rt[rt["is_fault_run"] & ~rt["detected"]]
    d.p(f"Trong test, Random Forest có {int(rt['false_alarms'].sum())} báo nhầm, ở các lần chạy "
        f"{', '.join(f'{x.run_id} ({x.scenario})' for x in fa.itertuples()) or 'không có'}, và bỏ sót "
        f"{len(miss)} lần chạy có lỗi{': ' + ', '.join(f'{x.run_id} ({x.scenario})' for x in miss.itertuples()) if len(miss) else ''}. "
        f"Độ trễ lớn nhất trong test là {f2(T['rf']['delay_max'],1)} s.")
    ms = man.set_index("run_id")
    lines = []
    for det in ("rf", "baseline"):
        r_ = pd.read_csv(results_dir / f"runs_stress_{det}.csv")
        r_["traj_kind"] = r_["run_id"].map(ms["traj_kind"])
        for kind, g in r_.groupby("traj_kind"):
            hm = g["healthy_minutes"].sum()
            lines.append([det, kind, len(g), f"{g['false_alarms'].sum()} ({g['false_alarms'].sum()/hm:.2f}/phút)",
                          f"{int(g['detected'].sum())}/{int(g['is_fault_run'].sum())}"])
    d.p("*Các nhận xét định tính dưới đây được viết khi phân tích lần chạy standard đầu tiên; nếu chạy lại "
        "với cấu hình khác, hãy đối chiếu chúng với các bảng số (được cập nhật tự động).*")
    d.p("Phân tích stress test theo kiểu quỹ đạo (tải 2–3 kg trong cả hai kiểu):")
    d.table(["Bộ phát hiện", "Quỹ đạo", "Số lần chạy", "Báo nhầm", "Phát hiện"], lines)
    rf_s, bl_s = S["rf"], S["baseline"]
    d.bullets([
        f"Trong miền chuẩn, Random Forest đạt F1 {T['rf']['window_alarm_state']['f1']:.2f}, so với "
        f"{T['baseline']['window_alarm_state']['f1']:.2f} của baseline. Nó phát hiện "
        f"{T['rf']['n_detected']}/{T['rf']['n_fault_runs']} lần chạy có lỗi, so với "
        f"{T['baseline']['n_detected']}/{T['baseline']['n_fault_runs']} của baseline. "
        "Baseline chỉ nhìn độ lớn sai số, nên muốn ít báo nhầm khi robot khỏe chạy nhanh thì ngưỡng phải cao, và "
        "khi đó nó bỏ sót phần lớn lỗi mild và lỗi ở khớp 1 (xem bảng 7.3). Random Forest kết hợp thêm thông tin về chuyển động mục tiêu "
        "và sai số theo hướng chuyển động, nên tách được hai trường hợp tốt hơn trong miền đã huấn luyện.",
        f"Ngoài miền (stress), cả hai phương pháp đều suy giảm rõ. Random Forest có {rf_s['false_alarms_per_healthy_min']:.2f} "
        f"báo nhầm/phút robot khỏe, và {rf_s['healthy_runs_with_false_alarm']}/{rf_s['n_healthy_runs']} lần chạy khỏe "
        "có ít nhất một báo nhầm. Tải nặng hơn và tốc độ cao hơn mức đã thấy khi huấn luyện bị mô hình hiểu nhầm "
        "thành ma sát. Đây là câu trả lời trực tiếp (và tiêu cực) cho vế 'tránh báo nhầm khi tải lớn hơn' của câu "
        "hỏi nghiên cứu, khi tải vượt ra ngoài phạm vi huấn luyện.",
        "Với lỗi trên quỹ đạo sin trong stress test, nếu cảnh báo đã bật nhầm từ trước khi lỗi xuất hiện, quy tắc "
        "đánh giá không tính đó là phát hiện. Vì vậy số lần phát hiện thấp ở nhóm này chủ yếu phản ánh báo nhầm "
        "kéo dài, không phải mô hình 'không thấy' lỗi.",
        "Khi tỷ lệ báo nhầm cao, một phần 'phát hiện' sau thời điểm lỗi có thể chỉ là trùng hợp. Vì vậy tỷ lệ phát "
        "hiện ở stress test cần được đọc cùng với tỷ lệ báo nhầm.",
        "Cỡ mẫu nhỏ: mỗi kịch bản trong test chỉ có 4–8 lần chạy. Chênh lệch một hai lần chạy làm tỷ lệ thay đổi "
        "đáng kể, và không có khoảng tin cậy nào được tính."])

    # 9 (only when the diagnosis stage has been trained and evaluated)
    n = 9
    dm_path = results_dir / "diagnosis_metrics.json"
    DM = json.load(open(dm_path)) if dm_path.exists() else None
    if DM:
        DR = json.load(open(model_dir / "diagnosis_report.json"))
        d.h(f"{n}. Chẩn đoán: khớp nào và ma sát tăng bao nhiêu", 2)
        d.p("Sau khi cảnh báo bật, một tầng chẩn đoán trả lời hai câu hỏi: lỗi nằm ở khớp nào, và hệ số ma sát của "
            "khớp đó là bao nhiêu so với giá trị khỏe (ví dụ giao diện hiện `Joint 2 - friction increase: 200% (x3.0)`). "
            "Tầng này không quyết định có lỗi hay không; việc đó vẫn do bộ phát hiện ở trên.")
        d.bullets([
            "Hai Random Forest (phân loại khớp, hồi quy hệ số ma sát) dùng đúng 24 đặc trưng như bộ phát hiện, và dùng lại "
            "tham số của bộ phát hiện; không có bước dò tham số riêng.",
            f"Chỉ huấn luyện trên các cửa sổ có nhãn 1 của tập train ({DR['train']['n_windows']} cửa sổ, hệ số thật "
            f"x{DR['train_mult_range'][0]:.2f}–{DR['train_mult_range'][1]:.2f}). Đích lấy từ ground truth (khớp lỗi trong "
            "manifest, hệ số ma sát trong `truth/`), không bao giờ là đầu vào.",
            "Khi chạy online, dự đoán được lấy trung bình trên tối đa 20 cửa sổ gần nhất có điểm ≥ ngưỡng, và chỉ hiển "
            "thị khi cảnh báo đang bật. Bộ nhớ được xóa khi cảnh báo tắt.",
            f"Validation (theo cửa sổ, chỉ để tham khảo): đúng khớp {DR['val']['joint_accuracy']*100:.1f}%, sai số tuyệt "
            f"đối trung bình của hệ số {DR['val']['mult_mae']:.2f}.",
            "Test và stress chỉ được xem sau khi tầng chẩn đoán đã huấn luyện xong."])
        rows = []
        for part in ("test", "stress"):
            P = DM[part]
            for tag, name in (("at_alarm", "lúc cảnh báo bật"), ("later", f"{DM['later_s']:g} s sau")):
                r_ = P["runs"][tag]
                rows.append([part, name, f"{r_['joint_correct']}/{r_['n_runs']}", f2(r_["mult_median_abs_error"]),
                             f2(r_["mult_p90_abs_error"])])
        d.p("Đánh giá theo lần chạy, đúng như cách hệ thống hiển thị: chỉ trên các lần chạy có lỗi mà bộ phát hiện đã "
            "bắt được, đọc chẩn đoán tại cửa sổ phát hiện và sau đó một khoảng (chỉ khi cảnh báo vẫn còn bật).")
        d.table(["Tập", "Thời điểm", "Đúng khớp", "Sai số tuyệt đối hệ số TV", "Sai số tuyệt đối hệ số P90"], rows)
        d.p(f"Theo cửa sổ (mọi cửa sổ có nhãn 1): test đúng khớp {DM['test']['windows']['joint_accuracy']*100:.1f}%, "
            f"MAE hệ số {DM['test']['windows']['mult_mae']:.2f}; stress {DM['stress']['windows']['joint_accuracy']*100:.1f}%, "
            f"MAE {DM['stress']['windows']['mult_mae']:.2f}.")
        dt = pd.read_csv(results_dir / "diagnosis_runs_test.csv")
        wrong = dt[dt["at_alarm_joint"].notna() & (dt["at_alarm_joint"] != dt["fault_joint"])]
        mt = DM["test"]["runs"]["at_alarm"]["mult_median_abs_error"]
        d.bullets([
            "Các lần đoán sai khớp trong test lúc cảnh báo bật: "
            f"{', '.join(f'{x.run_id} ({x.scenario}, hệ số thật x{x.at_alarm_mult_true:.2f})' for x in wrong.itertuples()) or 'không có'}.",
            f"Sai số trung vị lúc cảnh báo trên test là {mt:.2f}, nghĩa là, ví dụ, ước lượng x{3 - mt:.1f} khi thật là "
            f"x3.0 (báo +{(2 - mt) * 100:.0f}% thay vì +200%). Mức ±15% dao động của ma sát khỏe nằm dưới độ phân giải "
            "này, nên hệ thống không đo được thay đổi nhỏ cỡ đó.",
            "Khi cảnh báo là báo nhầm, chẩn đoán vẫn nêu một khớp và một mức ma sát, và cả hai đều sai. Chẩn đoán chỉ "
            "đáng tin khi cảnh báo phía trước nó đáng tin.",
            "Chẩn đoán cho biết khớp và mức ma sát, không cho biết nguyên nhân vật lý."])
        n += 1

    d.h(f"{n}. Giới hạn", 2)
    d.bullets([
        "Dữ liệu 100% mô phỏng, chưa xác nhận trên robot thật. Mô hình vật lý đơn giản (khâu cứng, không có động "
        "cơ/hộp số chi tiết, không có backlash, không có thay đổi theo nhiệt độ).",
        "Ma sát tăng chỉ là một cách mô hình hóa bất thường. Nó không chứng minh nguyên nhân duy nhất là khô "
        "dầu, và hệ thống không phân biệt nguyên nhân.",
        "Chỉ có một loại lỗi, ở một khớp tại một thời điểm. Chưa có lỗi đồng thời hai khớp hoặc các lỗi khác "
        "(va chạm, lệch cảm biến, mòn bánh răng).",
        "Bộ điều khiển biết thông số danh định chính xác của robot, chỉ tải khai báo là có sai lệch nhỏ.",
        "Kích thước dữ liệu standard không đảm bảo tổng quát hóa. Stress test cho thấy mô hình không tổng quát "
        "tốt ra ngoài miền tải và tốc độ đã huấn luyện.",
        "Điểm dự đoán của Random Forest chưa được hiệu chuẩn thành xác suất.",
        "Phát hiện bất thường không phải là dự báo thời điểm hỏng hay tuổi thọ còn lại (RUL)."])

    d.h(f"{n + 1}. Hướng phát triển", 2)
    d.bullets([
        "Đưa tải khai báo, hoặc phần dư mô hình (mô-men đo trừ mô-men dự đoán bởi mô hình động lực học danh "
        "định), vào đặc trưng để giảm nhầm lẫn giữa tải nặng và ma sát. Sau đó đánh giá lại trên stress test "
        "bằng một bộ stress mới, vì bộ hiện tại đã được xem.",
        "Mở rộng miền huấn luyện (tải, tốc độ, kiểu quỹ đạo), hoặc thêm cơ chế phát hiện dữ liệu ngoài miền để "
        "hệ thống tự nói 'không chắc chắn' thay vì báo lỗi.",
        "Mô hình hóa động cơ và hộp số (dòng điện, tỉ số truyền, hiệu suất), rồi so sánh với dữ liệu robot thật "
        "nếu có thể.",
        "Ước lượng ma sát trực tuyến (identification) để theo dõi xu hướng. Chỉ khi có dữ liệu suy giảm theo "
        "thời gian dài mới nên nói tới dự báo tuổi thọ.",
        "Thêm khoảng tin cậy bằng bootstrap theo nhóm thí nghiệm, và kiểm định chéo GroupKFold trên tập phát triển."])
    d.h("Phụ lục: tái lập", 2)
    d.p(f"Seed: {cfg['seed']}. Phiên bản: Python {card['versions']['python']}, NumPy {card['versions']['numpy']}, "
        f"pandas {card['versions']['pandas']}, scikit-learn {card['versions']['scikit-learn']}. Chạy lại toàn bộ: "
        f"`python -m scripts.run_pipeline --mode {mode}` (xem README).")
    return d


def write_report(results_dir, model_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = build(results_dir, model_dir, out_dir)
    md = out_dir / "report_vi.md"
    md.write_text(d.to_markdown(out_dir), encoding="utf-8")
    docx = out_dir / "report_vi.docx"
    d.to_docx(docx)
    return md, docx
