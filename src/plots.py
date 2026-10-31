"""Figures generated only from saved results (metrics.json, per-window/per-run tables,
simulated signals). Nothing is typed in by hand."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .dataset import load_signals, load_truth
from .dynamics import RobotParams, forward_kinematics_np

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
DET_COLOR = {"rf": BLUE, "baseline": ORANGE}
DET_NAME = {"rf": "Random Forest", "baseline": "Baseline (tracking error)"}
SCEN_ORDER = ["healthy_speed", "healthy_load", "mild_j1", "mild_j2", "severe_j1", "severe_j2", "gradual_j1", "gradual_j2"]

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#b5b4ad", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9.5,
                     "axes.titlesize": 10.5, "axes.titleweight": "bold", "lines.linewidth": 1.4,
                     "legend.frameon": False})


def _save(fig, out, name):
    path = Path(out) / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def example_run(data_dir, results_dir, run_id, split, out, bundle_threshold):
    sig, tru = load_signals(data_dir, run_id), load_truth(data_dir, run_id)
    man = pd.read_csv(Path(data_dir) / "manifest.csv").set_index("run_id").loc[run_id]
    win = pd.read_csv(Path(results_dir) / f"windows_{split}_rf.csv.gz")
    win = win[win["run_id"] == run_id]
    runs = pd.read_csv(Path(results_dir) / f"runs_{split}_rf.csv").set_index("run_id").loc[run_id]
    t = sig["t"].to_numpy()
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 2.3])
    axp = fig.add_subplot(gs[:, 0])
    p = RobotParams.from_config(load_config())
    xd, yd = forward_kinematics_np(sig["q1_des"], sig["q2_des"], p)
    xa, ya = forward_kinematics_np(tru["q1_true"], tru["q2_true"], p)
    axp.plot(xd, yd, ":", color=GRAY, label="target tip path")
    axp.plot(xa, ya, color=BLUE, lw=1.0, label="actual tip path")
    axp.set_aspect("equal"); axp.set_xlabel("x [m]"); axp.set_ylabel("y [m]")
    axp.set_title("Tip path (forward kinematics)"); axp.legend(loc="lower left")
    axs = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    axs[0].plot(t, sig["q1_des"], ":", color=BLUE); axs[0].plot(t, sig["q1_meas"], color=BLUE, lw=1, label="joint 1")
    axs[0].plot(t, sig["q2_des"], ":", color=ORANGE); axs[0].plot(t, sig["q2_meas"], color=ORANGE, lw=1, label="joint 2")
    axs[0].set_ylabel("angle [rad]"); axs[0].set_title("Target (dotted) and measured joint angles")
    axs[1].plot(t, sig["tau1_applied"], color=BLUE, lw=0.9, label="tau1")
    axs[1].plot(t, sig["tau2_applied"], color=ORANGE, lw=0.9, label="tau2")
    axs[1].set_ylabel("N·m"); axs[1].set_title("Applied torque (simulated)")
    axs[2].plot(t, sig["q1_des"] - sig["q1_meas"], color=BLUE, lw=0.8, label="e1")
    axs[2].plot(t, sig["q2_des"] - sig["q2_meas"], color=ORANGE, lw=0.8, label="e2")
    axs[2].set_ylabel("rad"); axs[2].set_title("Tracking error")
    axs[3].plot(win["t_end"], win["score"], color=BLUE, lw=1.2, label="RF model score")
    axs[3].axhline(bundle_threshold, color=MUTED, ls="--", lw=1, label=f"threshold {bundle_threshold:.2f}")
    st = win["state"].to_numpy(bool); te = win["t_end"].to_numpy()
    axs[3].fill_between(te, 0, 1, where=st, step="post", color=BLUE, alpha=0.12, lw=0, label="alarm ON")
    axs[3].set_ylim(0, 1.02); axs[3].set_ylabel("score"); axs[3].set_xlabel("time [s]")
    axs[3].set_title("Model score and alarm state (score is not a calibrated probability)")
    for a in axs:
        a.set_xlim(t[0], t[-1])
        if np.isfinite(man["t_onset"]):
            a.axvline(man["t_onset"], color=INK, lw=1, ls="-.")
        if np.isfinite(man["t_label"]) and man["t_label"] != man["t_onset"]:
            a.axvline(man["t_label"], color=INK, lw=1, ls=":")
        if runs["detected"]:
            a.axvline(runs["t_detect"], color=AQUA, lw=1.5)
        a.legend(loc="upper right", ncol=3, fontsize=8, frameon=True, framealpha=0.85, edgecolor="none")
    desc = (f"{run_id} | {man['scenario']} | payload {man['payload']:.2f} kg | "
            f"fault x{man['mult_final']:.2f} " + (f"injected t={man['t_onset']:.1f} s" if np.isfinite(man['t_onset']) else "none"))
    if runs["is_fault_run"]:
        desc += (f" | detected at {runs['t_detect']:.1f} s (delay {runs['delay']:.1f} s)" if runs["detected"] else " | MISSED")
    desc += f" | false alarms: {runs['false_alarms']}"
    fig.suptitle(desc + "\n(dash-dot: fault injected, dotted: label threshold crossed, green: detection)", fontsize=9.5)
    return _save(fig, out, f"example_{split}_{run_id}.png")


def confusion(metrics, out):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7))
    for i, part in enumerate(("test", "stress")):
        for j, det in enumerate(("rf", "baseline")):
            ax = axes[i, j]
            c = metrics[part][det]["window_alarm_state"]["confusion"]
            m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
            ax.imshow(m / m.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for (r, k), v in np.ndenumerate(m):
                frac = v / m[r].sum()
                ax.text(k, r, f"{v}\n({frac:.1%})", ha="center", va="center",
                        color="white" if frac > 0.6 else INK, fontsize=10)
            ax.set_xticks([0, 1], ["pred normal", "pred anomaly"]); ax.set_yticks([0, 1], ["true normal", "true anomaly"])
            ax.grid(False)
            ax.set_title(f"{part.upper()} - {DET_NAME[det]}")
    fig.suptitle("Window-level confusion matrices of the ALARM STATE (row-normalised colour)", fontsize=10)
    return _save(fig, out, "confusion_matrices.png")


def comparison(metrics, out):
    items = [("F1 (window alarm state)", lambda s: s["window_alarm_state"]["f1"]),
             ("Detection rate (runs)", lambda s: s["detection_rate"]),
             ("False alarms / healthy min", lambda s: s["false_alarms_per_healthy_min"]),
             ("Median delay [s]", lambda s: s["delay_median"])]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    x = np.arange(2)
    for ax, (title, f) in zip(axes, items):
        for k, det in enumerate(("rf", "baseline")):
            vals = [f(metrics[p][det]) for p in ("test", "stress")]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=DET_COLOR[det], label=DET_NAME[det])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x, ["test", "stress"]); ax.set_title(title, fontsize=9.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Random Forest vs baseline (same windows, same alarm rule, thresholds from validation)", fontsize=10)
    return _save(fig, out, "comparison_rf_vs_baseline.png")


def by_scenario(metrics, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, part in zip(axes, ("test", "stress")):
        scen = [s for s in SCEN_ORDER if s in metrics[part]["rf"]["by_scenario"] and not s.startswith("healthy")]
        x = np.arange(len(scen))
        for k, det in enumerate(("rf", "baseline")):
            bs = metrics[part][det]["by_scenario"]
            vals = [bs[s]["detection_rate"] for s in scen]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=DET_COLOR[det], label=DET_NAME[det])
            for b, s in zip(bars, scen):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                        f"{bs[s]['n_detected']}/{bs[s]['n_fault_runs']}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(x, scen, rotation=20); ax.set_ylim(0, 1.15)
        ax.set_title(f"{part.upper()}: detected runs / fault runs per scenario")
    axes[0].set_ylabel("detection rate")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=8.5)
    fig.suptitle("Small counts per bar: read as counts, not as precise rates", fontsize=9.5, x=0.3)
    return _save(fig, out, "detection_by_scenario.png")


def false_alarms_by_scenario(metrics, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, part in zip(axes, ("test", "stress")):
        scen = [s for s in SCEN_ORDER if s in metrics[part]["rf"]["by_scenario"]]
        x = np.arange(len(scen))
        for k, det in enumerate(("rf", "baseline")):
            bs = metrics[part][det]["by_scenario"]
            vals = [bs[s]["false_alarms"] / bs[s]["healthy_minutes"] if bs[s]["healthy_minutes"] else 0 for s in scen]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=DET_COLOR[det], label=DET_NAME[det])
            for b, s in zip(bars, scen):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{bs[s]['false_alarms']}",
                        ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(x, scen, rotation=20)
        ax.set_title(f"{part.upper()}: false alarms per healthy minute (label = count)")
    axes[0].set_ylabel("false alarms / min")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=8.5)
    fig.suptitle("False alarms = alarm events before the fault is injected", fontsize=9.5, x=0.3)
    return _save(fig, out, "false_alarms_by_scenario.png")


def delays(results_dir, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, part in zip(axes, ("test", "stress")):
        data, labels = [], []
        for det in ("rf", "baseline"):
            r = pd.read_csv(Path(results_dir) / f"runs_{part}_{det}.csv")
            f = r[r["is_fault_run"]]
            d = f.loc[f["detected"], "delay"].to_numpy()
            data.append(d)
            labels.append(f"{DET_NAME[det]}\n{len(d)} detected, {int((~f['detected']).sum())} missed")
        bp = ax.boxplot(data, tick_labels=labels, widths=0.45, patch_artist=True, showfliers=False)
        for patch, det in zip(bp["boxes"], ("rf", "baseline")):
            patch.set_facecolor(DET_COLOR[det]); patch.set_alpha(0.25); patch.set_edgecolor(DET_COLOR[det])
        rng = np.random.default_rng(0)
        for i, (d, det) in enumerate(zip(data, ("rf", "baseline"))):
            ax.scatter(i + 1 + rng.uniform(-0.12, 0.12, len(d)), d, s=12, color=DET_COLOR[det], zorder=3)
        ax.set_title(f"{part.upper()}: detection delay of detected runs")
    axes[0].set_ylabel("delay after defined fault time [s]")
    return _save(fig, out, "detection_delays.png")


def importance(results_dir, out):
    imp = pd.read_csv(Path(results_dir) / "feature_importance.csv").head(14)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.barh(imp["feature"], imp["importance"], color=BLUE, height=0.6)
    ax.set_xlabel("impurity-based importance"); ax.set_title("Random Forest: top-14 feature importances")
    return _save(fig, out, "feature_importance.png")
