"""Export recorded demo runs for the project web page (web/index.html).

Each scenario is run exactly like the Pygame demo (same trajectory, simulator, noise and
OnlineDetector), with fault keys pressed at fixed simulated times. Output is a compact
JSON with downsampled signals, the model score per window and the ground-truth panel.

  python -m scripts.export_web_demo --model models/standard --out web/demo_runs.json
"""
import argparse
import json
from pathlib import Path

from src.config import load_config
from src.dynamics import forward_kinematics
from src.faults import FaultSchedule
from src.online import OnlineDetector, load_bundle, load_diagnosis
from src.simulator import Simulator
from src.trajectory import SinusoidTrajectory
from src.gui.app import PAYLOADS, SPEEDS

DURATION = 30.0
PLOT_EVERY = 5   # 100 Hz -> 20 Hz for the page

# (id, title, payload kg, speed, fault: None | (kind, joint, t_start, mult, ramp_s))
SCENARIOS = [
    ("healthy", "Healthy robot", 0.5, "normal", None),
    ("heavy", "Healthy, heavy payload (2.0 kg)", 2.0, "normal", None),
    ("abrupt_j2", "Sudden friction x3 on joint 2", 0.5, "normal", ("abrupt", 2, 12.0, 3.0, None)),
    ("abrupt_j1", "Sudden friction x3 on joint 1", 0.5, "normal", ("abrupt", 1, 12.0, 3.0, None)),
    ("gradual_j1", "Friction on joint 1 creeps up to x3.5 over 10 s", 0.5, "normal",
     ("gradual", 1, 8.0, 3.5, 10.0)),
    ("fast", "Healthy, fast motion (edge of training range)", 0.5, "fast", None),
]


def run_scenario(cfg, bundle, diag_bundle, sid, title, payload, speed, fault, seed=1):
    assert payload in PAYLOADS
    f1, f2 = SPEEDS[speed]
    traj = SinusoidTrajectory((0.8, -1.25), (0.40, 0.45), (f1, f2), (0.0, 1.2), cfg["trajectory"]["t_ramp"])
    fr = cfg["friction"]
    sim = Simulator(cfg, payload, payload, fr["nominal_b"], fr["nominal_tau_c"], traj,
                    FaultSchedule(), 0.0005, 0.005, noise_seed=seed)
    det = OnlineDetector(bundle, diag_bundle)
    lt = cfg["fault"]["label_threshold_multiplier"]
    fault_set, t_label = False, None
    rows, windows = [], []
    feats = bundle["feature_names"]
    keep = [feats.index(n) for n in ("j1_e_dir_mean", "j2_e_dir_mean", "j1_tau_rms", "j2_tau_rms")]
    n_ticks = int(round(DURATION / sim.T_c))
    i = 0
    for _ in range(n_ticks):
        if fault and not fault_set and sim.t >= fault[2] - 1e-9:
            kind, j, _, mult, ramp = fault
            sim.fault = (FaultSchedule(kind, j, sim.t, mult, label_threshold=lt) if kind == "abrupt"
                         else FaultSchedule(kind, j, sim.t, mult, ramp, label_threshold=lt))
            t_label = sim.fault.t_label
            fault_set = True
        sig, truth = sim.step()
        if sig is None:
            continue
        res = det.push(list(sig[1:9]) + [sig[11], sig[12]])
        t = sig[0]
        if res["updated"]:
            x = det.last_features
            dg = res["diagnosis"]
            windows.append([round(t, 2), round(res["score"], 3), int(res["alarm"]),
                            *[round(float(x[k]), 5) for k in keep],
                            dg["joint"] if dg else 0, round(dg["mult"], 3) if dg else 0])
        if i % PLOT_EVERY == 0:
            x1, y1, x2, y2 = forward_kinematics(truth[1], truth[2], sim.params)
            _, _, xd, yd = forward_kinematics(sig[5], sig[6], sim.params)
            rows.append([round(t, 2), round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4),
                         round(xd, 4), round(yd, 4),
                         round(sig[11], 2), round(sig[12], 2),
                         round(sig[5] - sig[1], 5), round(sig[6] - sig[2], 5),
                         round(truth[5], 3), round(truth[6], 3), int(truth[7])])
        i += 1
    alarm_on = [w[0] for k, w in enumerate(windows) if w[2] and (k == 0 or not windows[k - 1][2])]
    first_after = next((t for t in alarm_on if t_label is not None and t >= t_label), None)
    false_alarms = [t for t in alarm_on if t_label is None or t < t_label]
    return {
        "id": sid, "title": title, "payload": payload, "speed": speed,
        "fault": None if not fault else {"kind": fault[0], "joint": fault[1], "t_start": fault[2],
                                         "mult": fault[3], "ramp_s": fault[4]},
        "t_label": t_label, "alarm_on": alarm_on, "false_alarms": false_alarms,
        "first_alarm_after_label": first_after,
        "delay_s": None if first_after is None else round(first_after - t_label, 2),
        "rows": rows, "windows": windows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/standard")
    ap.add_argument("--out", default="web/demo_runs.json")
    a = ap.parse_args(argv)
    cfg = load_config()
    bundle = load_bundle(a.model)
    diag_bundle = load_diagnosis(a.model)
    out = {
        "note": "Recorded with the project simulator and the trained Random Forest (OnlineDetector).",
        "model": {k: bundle[k] for k in ("window_s", "stride_s", "monitor_start_s", "threshold", "n_on", "n_off")},
        "importances": sorted(zip(bundle["feature_names"], map(float, bundle["model"].feature_importances_)),
                              key=lambda kv: -kv[1]),
        "row_cols": ["t", "x1", "y1", "x2", "y2", "xd", "yd", "tau1", "tau2", "e1", "e2", "m1", "m2", "label"],
        "window_cols": ["t", "score", "alarm", "j1_e_dir_mean", "j2_e_dir_mean", "j1_tau_rms", "j2_tau_rms",
                        "diag_joint", "diag_mult"],
        "scenarios": [],
    }
    for sc in SCENARIOS:
        r = run_scenario(cfg, bundle, diag_bundle, *sc)
        last = next((w for w in reversed(r["windows"]) if w[7]), None)
        print(f"{r['id']:11s} alarms_on={r['alarm_on']} false={len(r['false_alarms'])} delay={r['delay_s']} "
              f"last diagnosis={'joint %d x%.2f' % (last[7], last[8]) if last else '-'}")
        out["scenarios"].append(r)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    print("saved:", path, f"{path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
