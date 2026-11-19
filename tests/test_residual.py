"""Checks for the physics-residual features (src/residual.py)."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.dynamics import ArmModel, RobotParams
from src.faults import FaultSchedule
from src.residual import (RESIDUAL_FEATURE_NAMES, _torque, run_residual_features,
                          window_residual_features)
from src.simulator import Simulator, SIGNAL_COLUMNS
from src.trajectory import SinusoidTrajectory

CFG = load_config()
P = RobotParams.from_config(CFG)
B, TC, EPS = CFG["friction"]["nominal_b"], CFG["friction"]["nominal_tau_c"], CFG["friction"]["epsilon"]
NAMES = RESIDUAL_FEATURE_NAMES


def test_dynamics_linear_in_payload():
    """The least-squares model assumes tau(payload) = tau(0) + payload * y."""
    rng = np.random.default_rng(0)
    args = [rng.uniform(-2, 2, 50) for _ in range(6)]
    t0, t1, t25 = (np.array(_torque(ArmModel(P, m), *args)) for m in (0.0, 1.0, 2.5))
    assert np.allclose(t25 - t0, 2.5 * (t1 - t0), atol=1e-9)


def test_least_squares_recovers_known_parameters():
    rng = np.random.default_rng(1)
    n = 100
    t = np.arange(n) * 0.01
    qd = np.vstack([0.8 * np.sin(2.0 * t), 0.9 * np.cos(2.5 * t)])
    y = np.vstack([5.0 + rng.normal(0, 1, n), 2.0 + rng.normal(0, 1, n)])
    true = dict(b1=4.0, tc1=6.0, b2=1.2, tc2=2.0, dm=0.3)
    r = np.vstack([true["b1"] * qd[0] + true["tc1"] * np.tanh(qd[0] / EPS) + true["dm"] * y[0],
                   true["b2"] * qd[1] + true["tc2"] * np.tanh(qd[1] / EPS) + true["dm"] * y[1]])
    f = dict(zip(NAMES, window_residual_features(r, y, qd, EPS, B, TC)))
    assert f["dm_hat"] == pytest.approx(0.3, abs=0.05)
    assert f["j1_fric_ratio"] == pytest.approx(2.0, rel=0.1)
    assert f["j2_fric_ratio"] == pytest.approx(1.0, rel=0.1)


def _sim_features(payload, payload_declared, fault=None):
    traj = SinusoidTrajectory((0.8, -1.2), (0.35, 0.4), (0.4, 0.5), (0.0, 1.0), 2.0)
    sim = Simulator(CFG, payload, payload_declared, B, TC, traj, fault, 0.0, 0.0, 0)
    sig, _ = sim.run(10.0)
    X, _ = run_residual_features(pd.DataFrame(sig, columns=SIGNAL_COLUMNS), payload_declared, CFG, 1.0)
    return pd.DataFrame(X, columns=NAMES)


def test_payload_error_is_not_read_as_friction():
    """Heavier payload than declared: dm_hat finds it, friction estimates stay nominal."""
    f = _sim_features(payload=2.5, payload_declared=2.0).iloc[-30:].median()
    assert f["dm_hat"] == pytest.approx(0.5, abs=0.1)
    assert f["j1_fric_ratio"] == pytest.approx(1.0, abs=0.25)
    assert f["j2_fric_ratio"] == pytest.approx(1.0, abs=0.25)


def test_friction_fault_shows_on_the_right_joint():
    fault = FaultSchedule(kind="abrupt", joint=1, t_onset=4.0, mult_final=3.0, ramp=0.0, label_threshold=1.5)
    f = _sim_features(payload=1.0, payload_declared=1.0, fault=fault).iloc[-30:].median()
    assert f["j1_fric_ratio"] == pytest.approx(3.0, rel=0.25)
    assert f["j2_fric_ratio"] == pytest.approx(1.0, abs=0.25)


@pytest.mark.parametrize("window_s", [0.5, 1.0])
def test_online_residual_features_identical_to_offline(window_s):
    from sklearn.ensemble import RandomForestClassifier
    from src.dataset import simulate_spec
    from src.features import FEATURE_NAMES, INPUT_COLUMNS
    from src.online import OnlineDetector
    from src.scenarios import sample_run_specs
    spec = [s for s in sample_run_specs(CFG, "smoke", "indomain") if s["scenario"] == "severe_j1"][0]
    _, sig, _, _ = simulate_spec((spec, CFG))
    sig = pd.DataFrame(sig, columns=SIGNAL_COLUMNS)
    names = FEATURE_NAMES + NAMES
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, len(names)))
    rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, (X[:, -5] > 0).astype(int))
    bundle = {"model": rf, "feature_names": names, "window_s": window_s, "stride_s": 0.1,
              "monitor_start_s": 2.0, "f_sample": 100, "threshold": 0.5, "n_on": 3, "n_off": 5}
    X_off, ends = run_residual_features(sig, spec["payload_declared"], CFG, window_s)
    det = OnlineDetector(bundle, payload_declared=spec["payload_declared"], cfg=CFG)
    X_on, idx = [], []
    for i, row in enumerate(sig[INPUT_COLUMNS].to_numpy()):
        if det.push(row)["updated"]:
            X_on.append(det.last_features[-len(NAMES):]); idx.append(i)
    assert idx == ends
    np.testing.assert_array_equal(np.array(X_on), X_off)
