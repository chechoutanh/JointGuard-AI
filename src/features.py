"""Window features (Stage C.1). The SAME function `window_features` is used offline
(dataset building / evaluation) and online (Pygame demo, via src/online.py).

Inputs per sample (INPUT_COLUMNS) are only observable signals. Not used, on purpose:
absolute time, run metadata, true state, friction multipliers, labels.

Window grid: a window ends at sample index e (0-based) when (e + 1) % stride_n == 0,
(e + 1) >= window_n and t_end >= monitor_start_s. The window covers samples
[e - window_n + 1, e] (causal: past samples only, no filtering in either direction).
"""
import math

import numpy as np
import pandas as pd

INPUT_COLUMNS = ["q1_meas", "q2_meas", "qd1_meas", "qd2_meas",
                 "q1_des", "q2_des", "qd1_des", "qd2_des",
                 "tau1_applied", "tau2_applied"]
DIR_EPS = 0.05  # rad/s, smoothing for the motion-direction weight tanh(qd_des/DIR_EPS)

_PER_JOINT = ["tau_rms", "tau_std", "tau_maxabs", "dtau_rms", "tau_dir_mean",
              "e_rms", "e_maxabs", "e_dir_mean", "ev_rms",
              "qd_des_rms", "qdd_des_rms", "q_des_mean"]
FEATURE_NAMES = [f"j{j}_{n}" for j in (1, 2) for n in _PER_JOINT]


def window_features(block: np.ndarray, dt: float) -> np.ndarray:
    """block: (window_n, len(INPUT_COLUMNS)) array in INPUT_COLUMNS order."""
    out = []
    for j in range(2):
        q, qd = block[:, j], block[:, 2 + j]
        q_des, qd_des = block[:, 4 + j], block[:, 6 + j]
        tau = block[:, 8 + j]
        e = q_des - q
        ev = qd_des - qd
        w = np.tanh(qd_des / DIR_EPS)
        dtau = np.diff(tau) / dt
        qdd_des = np.diff(qd_des) / dt
        out += [math.sqrt(np.mean(tau * tau)), float(np.std(tau)), float(np.max(np.abs(tau))),
                math.sqrt(np.mean(dtau * dtau)), float(np.mean(tau * w)),
                math.sqrt(np.mean(e * e)), float(np.max(np.abs(e))), float(np.mean(e * w)),
                math.sqrt(np.mean(ev * ev)),
                math.sqrt(np.mean(qd_des * qd_des)), math.sqrt(np.mean(qdd_des * qdd_des)),
                float(np.mean(q_des))]
    return np.asarray(out)


def window_end_indices(n_samples, window_n, stride_n, f_sample, monitor_start_s):
    ends = []
    for e in range(window_n - 1, n_samples):
        if (e + 1) % stride_n == 0 and e / f_sample >= monitor_start_s - 1e-9:
            ends.append(e)
    return ends


def window_sizes(window_s, stride_s, f_sample):
    return int(round(window_s * f_sample)), int(round(stride_s * f_sample))


def run_features(signals: pd.DataFrame, window_s, stride_s, f_sample, monitor_start_s):
    """Features for every window of one run. Returns (X array, end indices)."""
    arr = signals[INPUT_COLUMNS].to_numpy(dtype=float)
    wn, sn = window_sizes(window_s, stride_s, f_sample)
    ends = window_end_indices(len(arr), wn, sn, f_sample, monitor_start_s)
    dt = 1.0 / f_sample
    X = np.array([window_features(arr[e - wn + 1:e + 1], dt) for e in ends]) if ends else \
        np.zeros((0, len(FEATURE_NAMES)))
    return X, ends


def window_labels(truth: pd.DataFrame, ends, window_n):
    """Label of a window = ground-truth label at its last sample; `mixed` if the label
    changes inside the window (dropped for training, kept for online evaluation)."""
    lab = truth["label"].to_numpy(dtype=int)
    y = np.array([lab[e] for e in ends], dtype=int)
    mixed = np.array([lab[e - window_n + 1:e + 1].min() != lab[e - window_n + 1:e + 1].max() for e in ends])
    return y, mixed
