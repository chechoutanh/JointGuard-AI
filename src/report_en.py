"""Builds the English report (Markdown + Word) from saved results only. Same content and
sources as the Vietnamese report (src/report.py); no number is typed in by hand."""
import json
import shutil
from pathlib import Path

import pandas as pd

from .report import Doc, f2, step_size_check


def build_en(results_dir, model_dir, out_dir):
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

    d.h("JointGuard AI: Detecting and Diagnosing Abnormal Joint Friction on a 2-DOF Robot Arm "
        "with Dynamics Simulation and Machine Learning")
    d.p(f"*Dataset: `{data_dir.as_posix()}` (`{mode}` mode), model: `{model_dir.as_posix()}`. This report is generated "
        "automatically by `python -m scripts.make_report --lang en` from the saved results; every number is read from "
        "a result file.*")
    if mode == "smoke":
        d.p("**THIS IS A SMOKE REPORT**: the dataset is tiny and only checks that the pipeline runs end to end. "
            "Do not draw conclusions from the numbers below.")
    d.p("**Important.** All data is simulated. Increased friction is one way of modelling an abnormality; it does not "
        "show that the only cause is lost lubrication. The results have not been validated on a physical robot. "
        "Anomaly detection is not the same as predicting time to failure or remaining useful life. Simulated signals "
        "that look realistic are not evidence of physical validity.")
    d.p("**Use of AI tools.** The code, experiments and this report were built with help from an AI coding assistant "
        "(Claude), following the author's questions and decisions.")

    # 1
    d.h("1. Research question", 2)
    d.p("Using only joint torque (a simulated signal) and trajectory tracking error, can a system detect that one "
        "joint's friction has risen abnormally? And can it avoid false alarms when a healthy robot moves faster or "
        "carries a heavier load? Both of those also raise torque and error, which is what makes the problem hard. "
        "Once an alarm is raised, can the system say which joint is affected and by how much?")
    d.p(f"The anomaly definition was fixed before any data was generated: the friction of one joint (both the viscous "
        f"coefficient b and the Coulomb term τc) is at least {fl['label_threshold_multiplier']:.1f} times that run's "
        f"healthy value (≥ +{(fl['label_threshold_multiplier'] - 1) * 100:.0f}%). The label never uses tracking error.")

    # 2
    d.h("2. Mechanical model and assumptions", 2)
    d.p("A planar arm with two revolute joints moving in a vertical plane. The origin is at joint 1, x points right, "
        "y points up and gravity acts along −y. q1 is the angle of link 1 from the +x axis; q2 is the angle of link 2 "
        "relative to link 1. Positive is counter-clockwise. SI units throughout (m, kg, s, rad, N·m).")
    d.p("Equation of motion: `M(q)·q̈ + C(q,q̇)·q̇ + G(q) + τf(q̇) = τ_applied + τ_external` (τ_external = 0 in these "
        "experiments). Accelerations come from solving the 2×2 linear system by Gaussian elimination, without "
        f"inverting the matrix. The state is integrated with RK4 at a {tm['dt_physics'] * 1000:g} ms step.")
    d.table(["Parameter", "Link 1", "Link 2"], [
        ["Length l [m]", r["l1"], r["l2"]], ["Mass m [kg]", r["m1"], r["m2"]],
        ["Distance to centre of mass lc [m]", r["lc1"], r["lc2"]],
        ["Inertia about centre of mass I [kg·m²]", r["I1"], r["I2"]],
        ["Reflected rotor inertia [kg·m²]", r["armature1"], r["armature2"]],
        ["Joint limits [rad]", r["joint_limits"][0], r["joint_limits"][1]],
        ["Torque limit [N·m]", r["torque_limits"][0], r["torque_limits"][1]],
        ["Nominal viscous friction b [N·m·s/rad]", fr["nominal_b"][0], fr["nominal_b"][1]],
        ["Nominal Coulomb friction τc [N·m]", fr["nominal_tau_c"][0], fr["nominal_tau_c"][1]]])
    d.bullets([
        "The tip payload is a point mass merged into link 2 as one equivalent rigid body (mass, centre of mass, inertia "
        "via the parallel-axis theorem), so it affects both M(q) and G(q) consistently.",
        f"Friction per joint: `τf = b·q̇ + τc·tanh(q̇/ε)` with ε = {fr['epsilon']} rad/s (smoothed Coulomb). Each run's "
        f"healthy friction is drawn within ±{fr['healthy_tolerance'] * 100:.0f}% of the nominal value.",
        "A fault only multiplies b and τc of one joint inside the physics model. The controller is not told.",
        "Not modelled: motors, gearboxes, backlash, elasticity, temperature, collisions, load-dependent friction. "
        "Torque is a simulated signal, not measured motor current.",
        "Sensors: additive Gaussian noise on angles and velocities. The standard deviation is drawn per run "
        f"(angle {cfg['sensor']['noise_q_range']} rad, velocity {cfg['sensor']['noise_qd_range']} rad/s) from the same "
        "distribution for healthy and faulty runs."])
    err = step_size_check(cfg)
    d.p("**Physics checks** (`python -m scripts.check_physics`, details in `tests/test_physics.py`): forward kinematics "
        "at known poses; M symmetric and positive definite; G(q) equal to the gradient of potential energy; the payload "
        "produces the torque of a point mass; static gravity balance; energy conserved without friction or torque; "
        "friction removes energy; a healthy arm tracks stably; no NaN; numerical blow-up is reported. Halving the step "
        f"from 1 ms to 0.5 ms (abrupt x4 fault, 2 kg payload) changes joint angles by at most {err:.1e} rad.")

    # 3
    d.h("3. Controller", 2)
    d.p(f"A PD controller with gravity compensation runs at {tm['f_control']} Hz (zero-order hold): "
        "`τ_req = Kp·(q_des − q_meas) + Kd·(q̇_des − q̇_meas) + Ĝ(q_meas)`, followed by torque saturation to give "
        f"τ_applied. Gains: Kp = {c['kp']}, Kd = {c['kd']}. Both τ_req and τ_applied are logged.")
    d.bullets([
        "The controller knows the nominal link parameters and the payload declared by the operator. The declared "
        f"payload may differ from the real one by up to ±{c['payload_declared_error']} kg.",
        "It does not know friction (healthy or faulty) and does not compensate it, so higher friction shows up as larger "
        "tracking error and larger PD torque.",
        "The controller only sees noisy measurements. Gains were chosen once for discrete-time stability at 100 Hz and "
        "never retuned on test results.",
        f"Torque saturation: the largest fraction of saturated control periods is "
        f"{man[man.subset == 'indomain'].saturated_fraction.max():.4f} in-range and "
        f"{man[man.subset == 'stress'].saturated_fraction.max():.4f} in the stress test."])

    # 4
    d.h("4. Data generation", 2)
    dur = cfg["modes"][mode]["duration"]
    d.p(f"Data is generated headless (`python -m scripts.generate_dataset --mode {mode}`). Each run lasts {dur:g} s, "
        f"with a {cfg['trajectory']['t_ramp']} s smooth start. Signals are sampled at {tm['f_sample']} Hz. In total there "
        f"are {len(man)} successful runs and {len(failed)} numerically unstable runs (logged separately in "
        "`failed_runs.csv` and never labelled as mechanical faults).")
    dm = cfg["domains"]
    d.bullets([
        "In-range trajectories: sinusoids in joint space with varied centre, amplitude, frequency and phase "
        f"(frequency {dm['indomain']['freq']} Hz). Trajectories that would leave the joint limits are rejected.",
        f"healthy_speed: varied speed, payload in the nominal range {dm['indomain']['payload_nominal']} kg. "
        f"healthy_load: payload {dm['indomain']['payload_full']} kg at lower speed.",
        f"mild: friction jumps to x{fl['mild_range'][0]}–{fl['mild_range'][1]}; severe: x{fl['severe_range'][0]}–"
        f"{fl['severe_range'][1]}. Onset time drawn from {cfg['modes'][mode]['onset_range']} s.",
        f"gradual: friction ramps linearly from x1 to x{fl['gradual_final_range'][0]}–{fl['gradual_final_range'][1]} "
        f"over {fl['gradual_ramp_range'][0]}–{fl['gradual_ramp_range'][1]} s. The label switches when the multiplier "
        f"passes {fl['label_threshold_multiplier']}. Before that the label is 0, but an alarm in that interval is not "
        "counted as a false alarm.",
        f"Stress test (held out, never used for training or tuning): heavier payloads {dm['stress']['payload_full']} kg, "
        f"higher frequencies {dm['stress']['freq']} Hz, and {dm['stress']['p2p_fraction'] * 100:.0f}% smooth "
        "point-to-point (quintic) trajectories, a trajectory type never seen in training.",
        f"Each group (base experiment) has {cfg['modes'][mode]['replicates_per_group']} runs sharing trajectory, payload "
        "and healthy friction; they differ in noise, initial offset and fault time. A group always stays inside one split.",
        "Observed signals (`signals/`) and ground truth (`truth/`, `manifest.csv`) are stored separately. Seeds and all "
        "parameters of every run are in the manifest, so each run can be reproduced."])
    piv = split.pivot_table(index="scenario", columns="split", values="run_id", aggfunc="count", fill_value=0)
    cols = [c_ for c_ in ("train", "val", "test", "stress") if c_ in piv.columns]
    d.table(["Scenario"] + cols, [[i] + [int(piv.loc[i, c_]) for c_ in cols] for i in piv.index])

    # 5
    d.h("5. Features and model", 2)
    d.p(f"Features are computed on a past window of {card['window_s']} s (0.5 s and 1 s tried, chosen on validation), "
        f"updated every {card['stride_s']} s. Only data up to the prediction time is used, with no two-sided filtering. "
        "Per joint: RMS, standard deviation and peak |torque|; RMS torque rate; mean torque along the target motion "
        "direction; RMS and peak |position error|; mean position error along the motion direction "
        "(`mean(e·tanh(q̇_des/0.05))`, sensitive to the drag of friction); RMS velocity error; RMS target velocity and "
        f"acceleration; mean target angle. {len(card['feature_names'])} features in total.")
    d.p("Absolute time, run_id, true payload, friction multipliers and labels are never features. Automated tests check "
        "this, check causality, and check that online and offline features are identical.")
    sig = F("signature_load_vs_friction.png")
    if sig.exists():
        d.img(sig, "Figure: torque RMS against direction-weighted tracking error for 1 s windows from recorded demo runs. "
                   "A heavier payload shifts windows right; tripled friction shifts them up.")
    d.p(f"The model is a Random Forest (scikit-learn {card['versions']['scikit-learn']}). Best parameters on validation "
        f"(average precision = {TR['rf_best']['val_ap']:.3f}): `{TR['rf_best']['params']}`, out of "
        f"{len(TR['rf_selection'])} configurations. Windows whose label changes inside them are dropped from training. "
        f"Score threshold: {card['threshold']}. The alarm turns on after {card['n_on']} consecutive positive windows and "
        f"off after {card['n_off']} consecutive negative ones. Both were chosen on validation to maximise F1 subject to "
        f"≤ {cfg['training']['max_false_alarms_per_min']} false alarms per minute. `predict_proba` is treated as a "
        "**model score**, not a calibrated probability.")
    b = M["baseline"]
    d.p(f"**Baseline:** score = max_j(RMS position error of joint j / reference_j). The reference is the "
        f"{cfg['training']['baseline_quantile'] * 100:.0f}th percentile on healthy training windows "
        f"({b['ref'][0]:.4f} and {b['ref'][1]:.4f} rad). Window {b['window_s']} s, threshold x{b['threshold']}, alarm "
        f"after {b['n_on']} consecutive windows. The baseline uses the same validation procedure and alarm rule as the "
        "Random Forest.")
    if (card["n_on"] == max(cfg["training"]["n_on_grid"])) or (b["n_on"] == max(cfg["training"]["n_on_grid"])):
        d.p("*Note:* the selected alarm persistence lies at the edge of the search grid "
            f"({cfg['training']['n_on_grid']}). A wider grid might give a different result. This is reported rather "
            "than changed after seeing test results.")
    d.img(F("feature_importance.png"), "Figure: impurity-based feature importance of the Random Forest.")

    # 6
    d.h("6. Evaluation design", 2)
    d.bullets([
        "Runs are split by experiment group (60/20/20, stratified by scenario), and the split file is written before "
        "training. All choices of features, parameters and thresholds use train/validation only.",
        "The regular test and the stress test are reported separately, never averaged.",
        f"Monitoring starts after {cfg['features']['monitor_start_s']} s (end of the smooth start). Before that the "
        "interface shows WARMING UP.",
        "One alarm = one off-to-on transition of the alarm state. A false alarm is an alarm that starts before the fault "
        "is injected. False alarms are divided by minutes of healthy operation.",
        "Delay = time of the first valid alarm − defined fault time (label switch). A false alarm that started earlier "
        "and is still on does not count as a detection. For gradual faults, an alarm that starts after friction begins "
        "rising and is still on when the label switches counts as an early detection with zero delay.",
        "Window-level metrics (precision/recall/F1) compare the alarm state with the label on every window, including "
        "transitions. That is why window recall is lower than the run-level detection rate: windows inside the "
        "detection delay count as misses."])

    # 7
    d.h("7. Results", 2)

    def row(name, s):
        w = s["window_alarm_state"]
        return [name, f2(w["precision"]), f2(w["recall"]), f2(w["f1"]), f2(s["window_raw_decision"]["f1"]),
                f2(s["false_alarms_per_healthy_min"]), f"{s['healthy_runs_with_false_alarm']}/{s['n_healthy_runs']}",
                f"{s['n_detected']}/{s['n_fault_runs']}", s["n_missed"], f2(s["delay_median"], 1),
                f2(s["delay_p90"], 1), f2(s["delay_max"], 1)]
    hdr = ["Detector", "Precision", "Recall", "F1", "Raw F1", "False alarms /min", "Healthy runs with FA",
           "Detected", "Missed", "Median delay [s]", "P90 delay [s]", "Max delay [s]"]
    d.p("RF = Random Forest; raw F1 = F1 of per-window decisions before the persistence rule.")
    d.h("7.1 Regular test (in range)", 3)
    d.p(f"{T['rf']['n_runs']} runs, {T['rf']['healthy_minutes']:.1f} minutes of healthy operation and "
        f"{T['rf']['n_fault_runs']} faulty runs.")
    d.table(hdr, [row("RF", T["rf"]), row("Baseline", T["baseline"])])
    d.h("7.2 Stress test (held-out conditions)", 3)
    d.p(f"{S['rf']['n_runs']} runs, {S['rf']['healthy_minutes']:.1f} minutes of healthy operation and "
        f"{S['rf']['n_fault_runs']} faulty runs.")
    d.table(hdr, [row("RF", S["rf"]), row("Baseline", S["baseline"])])
    d.img(F("comparison_rf_vs_baseline.png"), "Figure: Random Forest against the baseline on test and stress.")
    d.img(F("confusion_matrices.png"), "Figure: window-level confusion matrices (alarm state).")
    d.h("7.3 By fault level and joint", 3)
    for part, P in (("test", T), ("stress", S)):
        rows = []
        for scen, s in P["rf"]["by_scenario"].items():
            sb = P["baseline"]["by_scenario"][scen]
            rows.append([scen, s["n_runs"], f"{s['n_detected']}/{s['n_fault_runs']}" if s["n_fault_runs"] else "—",
                         f2(s["delay_median"], 1), s["false_alarms"],
                         f"{sb['n_detected']}/{sb['n_fault_runs']}" if sb["n_fault_runs"] else "—",
                         f2(sb["delay_median"], 1), sb["false_alarms"]])
        d.p(f"**{part.upper()}**")
        d.table(["Scenario", "Runs", "RF detected", "RF median delay [s]", "RF false alarms",
                 "BL detected", "BL median delay [s]", "BL false alarms"], rows)
    d.img(F("detection_by_scenario.png"), "Figure: detection rate per scenario (label = detected / faulty runs).")
    d.img(F("false_alarms_by_scenario.png"), "Figure: false alarms per scenario.")
    d.img(F("detection_delays.png"), "Figure: distribution of detection delays, with the number of misses.")
    for f in sorted(fig_dir.glob("example_test_*.png")):
        d.img(f, f"Figure: test run {f.stem.split('_')[-1]}: trajectory, torque, error and model score.")

    # 8
    d.h("8. Analysis of false alarms and misses", 2)
    rt = pd.read_csv(results_dir / "runs_test_rf.csv")
    fa = rt[rt["false_alarms"] > 0]
    miss = rt[rt["is_fault_run"] & ~rt["detected"]]
    d.p(f"On the test set the Random Forest raised {int(rt['false_alarms'].sum())} false alarms, in runs "
        f"{', '.join(f'{x.run_id} ({x.scenario})' for x in fa.itertuples()) or 'none'}, and missed {len(miss)} faulty "
        f"runs{': ' + ', '.join(f'{x.run_id} ({x.scenario})' for x in miss.itertuples()) if len(miss) else ''}. "
        f"The largest test delay was {f2(T['rf']['delay_max'], 1)} s.")
    ms = man.set_index("run_id")
    lines = []
    for det in ("rf", "baseline"):
        r_ = pd.read_csv(results_dir / f"runs_stress_{det}.csv")
        r_["traj_kind"] = r_["run_id"].map(ms["traj_kind"])
        for kind, g in r_.groupby("traj_kind"):
            hm = g["healthy_minutes"].sum()
            lines.append([det, kind, len(g), f"{g['false_alarms'].sum()} ({g['false_alarms'].sum() / hm:.2f}/min)",
                          f"{int(g['detected'].sum())}/{int(g['is_fault_run'].sum())}"])
    d.p("*The qualitative remarks below were written when analysing the first standard run; if you re-run with another "
        "configuration, check them against the tables (which update automatically).*")
    d.p("Stress test by trajectory type (2–3 kg payload in both):")
    d.table(["Detector", "Trajectory", "Runs", "False alarms", "Detected"], lines)
    rf_s = S["rf"]
    d.bullets([
        f"In range, the Random Forest reaches F1 {T['rf']['window_alarm_state']['f1']:.2f} against "
        f"{T['baseline']['window_alarm_state']['f1']:.2f} for the baseline, and detects "
        f"{T['rf']['n_detected']}/{T['rf']['n_fault_runs']} faulty runs against "
        f"{T['baseline']['n_detected']}/{T['baseline']['n_fault_runs']}. The baseline only sees error size, so to stay "
        "quiet on fast healthy motion its threshold must be high, and then it misses most mild faults and faults on "
        "joint 1 (table 7.3). The Random Forest also uses target motion and error along the motion direction, so it "
        "separates the two cases better within the training range.",
        f"Out of range (stress) both methods degrade clearly. The Random Forest has "
        f"{rf_s['false_alarms_per_healthy_min']:.2f} false alarms per healthy minute, and "
        f"{rf_s['healthy_runs_with_false_alarm']}/{rf_s['n_healthy_runs']} healthy runs have at least one. Heavier "
        "loads and faster motion than seen in training are mistaken for friction. This is a direct, negative answer to "
        "the 'avoid false alarms under heavier load' part of the question when the load leaves the training range.",
        "For faults on sinusoidal stress trajectories, an alarm that was already on before the fault is not counted as "
        "a detection, so the low detection count there mostly reflects long false alarms, not faults the model "
        "'did not see'.",
        "With a high false-alarm rate, some 'detections' after the fault may be coincidence, so stress detection rates "
        "should be read together with false-alarm rates.",
        "Small samples: each test scenario has only 4–8 runs. One or two runs change the rates noticeably, and no "
        "confidence intervals were computed."])

    # 9 (only when the diagnosis stage has been trained and evaluated)
    n = 9
    dm_path = results_dir / "diagnosis_metrics.json"
    DM = json.load(open(dm_path)) if dm_path.exists() else None
    if DM:
        DR = json.load(open(model_dir / "diagnosis_report.json"))
        d.h(f"{n}. Diagnosis: which joint, and how much", 2)
        d.p("Once the alarm is on, a diagnosis stage answers two questions: which joint is faulty, and what its friction "
            "multiplier is relative to the healthy value (for example the interface shows "
            "`Joint 2 - friction increase: 200% (x3.0)`). It never decides whether there is a fault; that remains the "
            "detector's job.")
        d.bullets([
            "Two Random Forests (joint classifier, friction-multiplier regressor) use the same 24 features as the "
            "detector and reuse its hyper-parameters; there is no separate tuning step.",
            f"Trained only on label-1 training windows ({DR['train']['n_windows']} windows, true multiplier "
            f"x{DR['train_mult_range'][0]:.2f}–{DR['train_mult_range'][1]:.2f}). Targets come from ground truth (faulty "
            "joint from the manifest, multiplier from `truth/`) and are never inputs.",
            "Online, predictions are averaged over at most the 20 most recent windows scoring ≥ the threshold, and shown "
            "only while the alarm is on. The buffer is cleared when the alarm turns off.",
            f"Validation (per window, for reference only): right joint {DR['val']['joint_accuracy'] * 100:.1f}%, mean "
            f"absolute multiplier error {DR['val']['mult_mae']:.2f}.",
            "Test and stress data were first examined after the diagnosis stage was trained."])
        rows = []
        for part in ("test", "stress"):
            P = DM[part]
            for tag, name in (("at_alarm", "alarm turns on"), ("later", f"{DM['later_s']:g} s later")):
                r_ = P["runs"][tag]
                rows.append([part, name, f"{r_['joint_correct']}/{r_['n_runs']}", f2(r_["mult_median_abs_error"]),
                             f2(r_["mult_p90_abs_error"])])
        d.p("Run-level evaluation, matching what the system shows: only faulty runs the detector caught, reading the "
            "diagnosis at the detection window and some time later (only if the alarm is still on).")
        d.table(["Set", "When", "Right joint", "Median abs. multiplier error", "P90 abs. multiplier error"], rows)
        d.p(f"Per window (all label-1 windows): test right joint {DM['test']['windows']['joint_accuracy'] * 100:.1f}%, "
            f"multiplier MAE {DM['test']['windows']['mult_mae']:.2f}; stress "
            f"{DM['stress']['windows']['joint_accuracy'] * 100:.1f}%, MAE {DM['stress']['windows']['mult_mae']:.2f}.")
        de = F("diagnosis_estimates.png")
        if de.exists():
            d.img(de, "Figure: estimated against true friction multiplier at the moment the alarm turns on.")
        dt = pd.read_csv(results_dir / "diagnosis_runs_test.csv")
        wrong = dt[dt["at_alarm_joint"].notna() & (dt["at_alarm_joint"] != dt["fault_joint"])]
        mt = DM["test"]["runs"]["at_alarm"]["mult_median_abs_error"]
        d.bullets([
            "Wrong joints on the test set at the moment of the alarm: "
            f"{', '.join(f'{x.run_id} ({x.scenario}, true multiplier x{x.at_alarm_mult_true:.2f})' for x in wrong.itertuples()) or 'none'}.",
            f"The median test error at the alarm is {mt:.2f}: for example an estimate of x{3 - mt:.1f} when the truth is "
            f"x3.0 (reported +{(2 - mt) * 100:.0f}% instead of +200%). The ±15% healthy variation is below this "
            "resolution, so the system cannot measure changes that small.",
            "Estimates are pulled toward the middle of the training range: small increases near x1.5 are overestimated "
            "and the largest ones are underestimated. The estimate separates moderate from severe faults; it is not a "
            "precise measurement.",
            "On a false alarm the diagnosis still names a joint and a friction level, and both are wrong. The diagnosis "
            "is only as trustworthy as the alarm before it.",
            "The diagnosis gives a joint and a friction level, not a physical cause."])
        n += 1

    d.h(f"{n}. Limitations", 2)
    d.bullets([
        "100% simulated data, not validated on a real robot. Simple physics (rigid links, no detailed motor/gearbox, "
        "no backlash, no temperature effects).",
        "Increased friction is only one way to model an abnormality. It does not show that the only cause is lost "
        "lubrication, and the system does not identify causes.",
        "One fault type, on one joint at a time. No simultaneous faults on both joints and no other faults (collisions, "
        "sensor offsets, gear wear).",
        "The controller knows the exact nominal robot parameters; only the declared payload is slightly off.",
        "The standard dataset size does not guarantee generalisation. The stress test shows the model does not "
        "generalise beyond the trained payload and speed range.",
        "Random Forest scores are not calibrated probabilities.",
        "Anomaly detection is not prediction of time to failure or remaining useful life (RUL)."])

    d.h(f"{n + 1}. Next steps", 2)
    d.bullets([
        "Add the declared payload, or the model residual (measured torque minus torque predicted by the nominal "
        "dynamics), as features to reduce confusion between heavy load and friction. Then re-evaluate on a new stress "
        "set, because the current one has already been examined.",
        "Widen the training range (payload, speed, trajectory types), or add out-of-distribution detection so the "
        "system says 'uncertain' instead of raising an alarm.",
        "Model motors and gearboxes (current, gear ratio, efficiency) and compare with real robot data if possible.",
        "Estimate friction online (system identification) to follow trends. Only with long-term degradation data "
        "should remaining-life prediction be discussed.",
        "Add bootstrap confidence intervals by experiment group and GroupKFold cross-validation on the development set."])
    d.h("Appendix: reproducibility", 2)
    d.p(f"Seed: {cfg['seed']}. Versions: Python {card['versions']['python']}, NumPy {card['versions']['numpy']}, "
        f"pandas {card['versions']['pandas']}, scikit-learn {card['versions']['scikit-learn']}. Full re-run: "
        f"`python -m scripts.run_pipeline --mode {mode}` (see README).")
    return d


def write_report_en(results_dir, model_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = build_en(results_dir, model_dir, out_dir)
    md = out_dir / "report_en.md"
    md.write_text(d.to_markdown(out_dir), encoding="utf-8")
    docx = out_dir / "report_en.docx"
    d.to_docx(docx)
    return md, docx
