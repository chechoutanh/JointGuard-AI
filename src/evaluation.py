"""Evaluation metrics (Stage D.3).

Definitions (identical for Random Forest and baseline):
  * Window-level metrics compare the ALARM STATE of each window with its label
    (all windows kept, including healthy->fault transitions). Raw per-window
    decisions (score >= threshold) are reported too.
  * One alarm event = one OFF->ON transition of the alarm state.
  * Healthy time of a run = monitored time before the physical fault injection
    t_onset (the whole run for healthy runs). Alarm events starting in healthy time
    are FALSE ALARMS. Rate = false alarms / healthy minutes.
  * Defined fault time t_label (label switches to 1, see src/faults.py).
    Detection = first alarm event starting at t >= t_label; delay = t - t_label.
    Gradual faults only: an alarm that started inside [t_onset, t_label) (friction already
    rising, but below the anomaly threshold) and is still ON at t_label counts as an
    early detection with delay 0. An alarm that started before t_onset is a false alarm
    and is NEVER counted as a detection.
  * A faulty run with no detection before the end of the run is a MISS.
"""
import math

import numpy as np
import pandas as pd


def prf(y_true, y_pred):
    y_true = np.asarray(y_true, bool)
    y_pred = np.asarray(y_pred, bool)
    tp = int(np.sum(y_true & y_pred)); fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred)); tn = int(np.sum(~y_true & ~y_pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def per_run_events(win: pd.DataFrame, manifest: pd.DataFrame, stride_s: float):
    """win: columns run_id, t_end, label, state(bool), onset(bool), sorted by t_end per run."""
    meta = manifest.set_index("run_id")
    rows = []
    for run_id, df in win.groupby("run_id", sort=False):
        m = meta.loc[run_id]
        t = df["t_end"].to_numpy()
        onset_t = t[df["onset"].to_numpy(bool)]
        state = df["state"].to_numpy(bool)
        t_onset = float(m["t_onset"]); t_label = float(m["t_label"])
        t_first = t[0] - stride_s  # monitoring starts one stride before first decision
        t_end = t[-1]
        healthy_end = min(t_onset, t_end)
        healthy_min = max(healthy_end - t_first, 0.0) / 60.0
        fa = int(np.sum(onset_t < t_onset))
        early = int(np.sum((onset_t >= t_onset) & (onset_t < t_label)))
        row = {"run_id": run_id, "scenario": m["scenario"], "severity": m["severity"],
               "fault_joint": int(m["fault_joint"]), "fault_kind": m["fault_kind"],
               "t_onset": t_onset, "t_label": t_label, "healthy_minutes": healthy_min,
               "false_alarms": fa, "early_alarms": early, "is_fault_run": bool(math.isfinite(t_label) and t_label <= t_end),
               "detected": False, "t_detect": np.nan, "delay": np.nan, "detect_type": ""}
        if row["is_fault_run"]:
            after = onset_t[onset_t >= t_label]
            idx_label = int(np.searchsorted(t, t_label))
            # early-active case (gradual only): alarm on at first window >= t_label and started after t_onset
            if m["fault_kind"] == "gradual" and idx_label < len(t) and state[idx_label]:
                prev_onsets = onset_t[onset_t <= t[idx_label]]
                if len(prev_onsets) and prev_onsets[-1] >= t_onset:
                    row.update(detected=True, t_detect=t_label, delay=0.0, detect_type="early_active")
            if not row["detected"] and len(after):
                row.update(detected=True, t_detect=float(after[0]), delay=float(after[0] - t_label),
                           detect_type="new_alarm")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(win: pd.DataFrame, runs: pd.DataFrame):
    out = {"n_runs": int(len(runs)), "n_windows": int(len(win))}
    out["window_alarm_state"] = prf(win["label"], win["state"])
    out["window_raw_decision"] = prf(win["label"], win["decision"])
    hm = runs["healthy_minutes"].sum()
    out["healthy_minutes"] = float(hm)
    out["false_alarms"] = int(runs["false_alarms"].sum())
    out["false_alarms_per_healthy_min"] = float(runs["false_alarms"].sum() / hm) if hm else float("nan")
    out["healthy_runs_with_false_alarm"] = int((runs.loc[runs["fault_kind"] == "none", "false_alarms"] > 0).sum())
    out["n_healthy_runs"] = int((runs["fault_kind"] == "none").sum())
    out.update(_detection_block(runs))
    by = {}
    for scen, r in runs.groupby("scenario"):
        b = {"n_runs": int(len(r)), "false_alarms": int(r["false_alarms"].sum()),
             "healthy_minutes": float(r["healthy_minutes"].sum())}
        b.update(_detection_block(r))
        by[scen] = b
    out["by_scenario"] = by
    return out


def _detection_block(runs):
    f = runs[runs["is_fault_run"]]
    det = f[f["detected"]]
    d = det["delay"].to_numpy()
    return {"n_fault_runs": int(len(f)), "n_detected": int(len(det)), "n_missed": int(len(f) - len(det)),
            "detection_rate": float(len(det) / len(f)) if len(f) else float("nan"),
            "delay_median": float(np.median(d)) if len(d) else float("nan"),
            "delay_mean": float(np.mean(d)) if len(d) else float("nan"),
            "delay_p90": float(np.percentile(d, 90)) if len(d) else float("nan"),
            "delay_max": float(np.max(d)) if len(d) else float("nan"),
            "n_early_active": int((det["detect_type"] == "early_active").sum())}
