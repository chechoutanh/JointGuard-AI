"""Fault diagnosis (stage after detection): WHICH joint has the friction fault and HOW MUCH
its friction rose. It never decides whether there is a fault; that stays with the detector.

Training data: train-split windows with label 1 (friction >= label threshold), mixed
windows dropped. Targets come from ground truth: the faulty joint (manifest) and the
friction multiplier of that joint at the window's last sample (truth files).
Inputs are the same 24 causal window features as the detector.

Online use (DiagnosisTracker): per-window predictions are averaged over the most recent
positive windows (score >= threshold), at most `max_windows`, and shown only while the
alarm is on. The buffer is cleared when the alarm turns off.
"""
from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .dataset import load_truth
from .features import FEATURE_NAMES


def attach_true_multipliers(feat, data_dir):
    """Add mult1, mult2 (ground-truth friction multipliers at each window's t_end)."""
    parts = []
    for run_id, df in feat.groupby("run_id", sort=False):
        tr = load_truth(data_dir, run_id)
        idx = np.searchsorted(tr["t"].to_numpy(), df["t_end"].to_numpy() - 1e-9)
        df = df.copy()
        df["mult1"] = tr["fric_mult1"].to_numpy()[idx]
        df["mult2"] = tr["fric_mult2"].to_numpy()[idx]
        parts.append(df)
    return pd.concat(parts)


def diagnosis_targets(feat, manifest):
    """Rows usable for diagnosis (label 1, not mixed) with targets fault_joint and mult_true."""
    joint = manifest.set_index("run_id")["fault_joint"]
    d = feat[(feat["label"] == 1) & (~feat["mixed"])].copy()
    d["fault_joint"] = d["run_id"].map(joint).astype(int)
    d["mult_true"] = np.where(d["fault_joint"] == 1, d["mult1"], d["mult2"])
    return d


def train_diagnoser(train, params, seed):
    X = train[FEATURE_NAMES].to_numpy()
    clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **params).fit(X, train["fault_joint"].to_numpy())
    reg_params = {k: v for k, v in params.items() if k != "class_weight"}
    reg = RandomForestRegressor(random_state=seed, n_jobs=-1, **reg_params).fit(X, train["mult_true"].to_numpy())
    return clf, reg


def predict_windows(bundle, X):
    """Per-window predictions: P(fault is on joint 2) and estimated friction multiplier."""
    X = np.atleast_2d(X)
    classes = list(bundle["joint_model"].classes_)
    p2 = bundle["joint_model"].predict_proba(X)[:, classes.index(2)]
    return p2, bundle["mult_model"].predict(X)


def window_metrics(bundle, d):
    p2, m = predict_windows(bundle, d[FEATURE_NAMES].to_numpy())
    joint_hat = np.where(p2 >= 0.5, 2, 1)
    err = m - d["mult_true"].to_numpy()
    return {"n_windows": int(len(d)), "joint_accuracy": float(np.mean(joint_hat == d["fault_joint"].to_numpy())),
            "mult_mae": float(np.mean(np.abs(err))), "mult_median_abs_error": float(np.median(np.abs(err))),
            "mult_bias": float(np.mean(err))}


class DiagnosisTracker:
    def __init__(self, max_windows=20):
        self.buf = deque(maxlen=max_windows)

    def reset(self):
        self.buf.clear()

    def update(self, decision, alarm, p2, mult):
        """Feed one window. Returns the current diagnosis dict while the alarm is on, else None."""
        if decision:
            self.buf.append((float(p2), float(mult)))
        if not alarm:
            if not decision:
                self.buf.clear()
            return None
        if not self.buf:
            return None
        p = float(np.mean([b[0] for b in self.buf]))
        mult = float(np.mean([b[1] for b in self.buf]))
        joint = 2 if p >= 0.5 else 1
        return {"joint": joint, "joint_confidence": p if joint == 2 else 1.0 - p,
                "mult": mult, "increase_pct": (mult - 1.0) * 100.0, "n_windows": len(self.buf)}
