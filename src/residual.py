"""Physics-residual features (experiment after the standard run; see docs/RESIDUAL_EXPERIMENT.md).

Idea: the controller knows the nominal arm and the payload the operator declared. Subtracting
the torque that this nominal model needs from the applied torque leaves what the model cannot
explain:

    r = tau_applied - tau_hat(q, qd, qdd; payload_declared)
      = tau_friction(qd) + dm * y(q, qd, qdd) + noise

because the dynamics are linear in the payload mass (y = tau_hat(payload=1) - tau_hat(payload=0)).
Friction depends on velocity, a payload error on pose and acceleration, so inside each window a
small least-squares fit separates them:

    unknowns theta = [b1, tc1, b2, tc2, dm]
    joint 1 rows:  r1 = b1*qd1 + tc1*tanh(qd1/eps) + dm*y1
    joint 2 rows:  r2 = b2*qd2 + tc2*tanh(qd2/eps) + dm*y2

A weak Tikhonov prior pulls theta towards the nominal values so that a joint that barely moves
gives "nominal" instead of a wild estimate. Its strength is fixed (PRIOR_SAMPLES) and was not
tuned on any test or stress data.

Inputs are observable only: measured angles/velocities, targets, applied torque and the
declared payload. No true state, friction value or label is used.
"""
import numpy as np
import pandas as pd

from .dynamics import ArmModel, RobotParams
from .features import INPUT_COLUMNS, window_end_indices, window_sizes

# Only dimensionless features, on purpose: a first version also exposed residual magnitudes
# (rms, fit error), which grow with speed, and the forest used them to mistake fast motion for
# friction again on the (already seen) stress split. See docs/RESIDUAL_EXPERIMENT.md.
RESIDUAL_FEATURE_NAMES = ["j1_fric_ratio", "j2_fric_ratio", "dm_hat", "j1_moving_frac", "j2_moving_frac"]
PRIOR_SAMPLES = 5.0  # prior weight, in "equivalent samples" per unknown
MOVE_THRESHOLD = 0.1  # rad/s; friction is only measured on samples where the joint really moves
ACC_HALF_WIDTH = 4  # samples; see run_residual_signals (chosen on train runs only)
MIN_MOVING = 10  # fewer moving samples than this -> friction ratio reported as nominal (1.0)


def _torque(model: ArmModel, q1, q2, qd1, qd2, qdd1, qdd2):
    """Vectorised M(q) qdd + C(q, qd) qd + G(q) (no friction)."""
    c2, s2 = np.cos(q2), np.sin(q2)
    m11 = model._a + 2.0 * model._beta * c2
    m12 = model._d + model._beta * c2
    h = -model._beta * s2
    c12 = np.cos(q1 + q2)
    t1 = m11 * qdd1 + m12 * qdd2 + h * (2.0 * qd1 * qd2 + qd2 * qd2) + model._g1 * np.cos(q1) + model._g2 * c12
    t2 = m12 * qdd1 + model._m22 * qdd2 - h * qd1 * qd1 + model._g2 * c12
    return t1, t2


def run_residual_signals(signals: pd.DataFrame, payload_declared, params: RobotParams, f_sample):
    """Per-sample residual torque r (2, n), payload sensitivity y (2, n) and measured qd (2, n).

    Acceleration: smoothed difference of the MEASURED velocity, centred on the control period
    during which tau_applied is held: qdd[i] = (qd[i+K] - qd[i+1-K]) / ((2K-1) dt), K = ACC_HALF_WIDTH.
    It needs K-1 earlier and K later samples, so samples without them are NaN and a window ending at
    sample e only uses samples up to e-K (still causal). A first version used the TARGET
    acceleration instead; the arm lags its target, and for a sinusoid that error is proportional to
    velocity and grows with frequency^2, i.e. it looks exactly like viscous friction at high speed.
    """
    dt = 1.0 / f_sample
    K = ACC_HALF_WIDTH
    q1, q2 = signals["q1_meas"].to_numpy(), signals["q2_meas"].to_numpy()
    qd1, qd2 = signals["qd1_meas"].to_numpy(), signals["qd2_meas"].to_numpy()
    qdd = []
    for v in (qd1, qd2):
        a = np.full(len(v), np.nan)
        if len(v) >= 2 * K:
            a[K - 1:len(v) - K] = (v[2 * K - 1:] - v[:len(v) - 2 * K + 1]) / ((2 * K - 1) * dt)
        qdd.append(a)
    m_decl, m0, m1 = ArmModel(params, payload_declared), ArmModel(params, 0.0), ArmModel(params, 1.0)
    th = _torque(m_decl, q1, q2, qd1, qd2, *qdd)
    t0 = _torque(m0, q1, q2, qd1, qd2, *qdd)
    t1 = _torque(m1, q1, q2, qd1, qd2, *qdd)
    r = np.vstack([signals["tau1_applied"].to_numpy() - th[0], signals["tau2_applied"].to_numpy() - th[1]])
    y = np.vstack([t1[0] - t0[0], t1[1] - t0[1]])
    return r, y, np.vstack([qd1, qd2])


def window_residual_features(r, y, qd, eps, b_nom, tc_nom):
    """r, y, qd: (2, window_n) arrays for one window. Returns RESIDUAL_FEATURE_NAMES values."""
    n = r.shape[1]
    z = np.zeros(n)
    s1, s2 = np.tanh(qd[0] / eps), np.tanh(qd[1] / eps)
    A = np.vstack([np.column_stack([qd[0], s1, z, z, y[0]]),
                   np.column_stack([z, z, qd[1], s2, y[1]])])
    rhs = np.concatenate([r[0], r[1]])
    theta0 = np.array([b_nom[0], tc_nom[0], b_nom[1], tc_nom[1], 0.0])
    # prior rows: weight so each counts like PRIOR_SAMPLES rows of typical size
    scale = np.sqrt(np.mean(A * A, axis=0)) + 1e-9
    w = np.sqrt(PRIOR_SAMPLES) * scale
    A_aug = np.vstack([A, np.diag(w)])
    rhs_aug = np.concatenate([rhs, w * theta0])
    theta = np.linalg.lstsq(A_aug, rhs_aug, rcond=None)[0]
    # total friction ratio per joint: payload-corrected residual vs nominal friction, projected on the
    # motion direction, over moving samples only (near standstill friction is not observable)
    ratio, moving = [], []
    for j, sj in ((0, s1), (1, s2)):
        mask = np.abs(qd[j]) > MOVE_THRESHOLD
        moving.append(float(mask.mean()))
        if mask.sum() < MIN_MOVING:
            ratio.append(1.0)
            continue
        f_nom = b_nom[j] * qd[j][mask] + tc_nom[j] * sj[mask]
        ratio.append(float(np.sum((r[j][mask] - theta[4] * y[j][mask]) * sj[mask]) / np.sum(f_nom * sj[mask])))
    return np.array([ratio[0], ratio[1], theta[4], moving[0], moving[1]])


def run_residual_features(signals, payload_declared, cfg, window_s):
    """Residual features for every window of one run, on the same window grid as run_features."""
    fs = cfg["timing"]["f_sample"]
    fc, fr = cfg["features"], cfg["friction"]
    params = RobotParams.from_config(cfg)
    r, y, qd = run_residual_signals(signals, payload_declared, params, fs)
    wn, sn = window_sizes(window_s, fc["stride_s"], fs)
    ends = window_end_indices(r.shape[1], wn, sn, fs, fc["monitor_start_s"])
    K = ACC_HALF_WIDTH  # last K samples of a window have no acceleration estimate yet
    X = np.array([window_residual_features(r[:, e - wn + 1:e - K + 1], y[:, e - wn + 1:e - K + 1],
                                           qd[:, e - wn + 1:e - K + 1], fr["epsilon"],
                                           fr["nominal_b"], fr["nominal_tau_c"]) for e in ends]) \
        if ends else np.zeros((0, len(RESIDUAL_FEATURE_NAMES)))
    return X, ends


def history_samples():
    """Extra past samples, beyond the window, the residual features need (acceleration stencil)."""
    return ACC_HALF_WIDTH - 1


def window_residual_from_block(block, payload_declared, cfg):
    """Online version: block = (window_n + history_samples(), len(INPUT_COLUMNS)) array ending at the
    current sample. Gives exactly the values run_residual_features computes for that window."""
    fr = cfg["friction"]
    K = ACC_HALF_WIDTH
    sig = pd.DataFrame(block, columns=INPUT_COLUMNS)
    r, y, qd = run_residual_signals(sig, payload_declared, RobotParams.from_config(cfg), cfg["timing"]["f_sample"])
    n = block.shape[0]
    sl = slice(K - 1, n - K)  # same samples as r[:, e-wn+1 : e-K+1] in run_residual_features
    return window_residual_features(r[:, sl], y[:, sl], qd[:, sl], fr["epsilon"], fr["nominal_b"], fr["nominal_tau_c"])
