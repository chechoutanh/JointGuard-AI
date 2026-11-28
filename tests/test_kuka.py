"""Checks for the KUKA iiwa generalisation experiment (src/kuka.py)."""
import numpy as np
import pytest

pytest.importorskip("pybullet")

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src import kuka  # noqa: E402

CFG = load_config(PROJECT_ROOT / "configs" / "kuka.json")
RNG = np.random.default_rng(0)
Q, QD, QDD = (RNG.uniform(-1, 1, kuka.N) for _ in range(3))


def test_payload_is_affine_and_matters():
    """Interpolating between the 0 kg and 1 kg bodies must equal a body loaded at 2.5 kg."""
    b25 = kuka.load_body(CFG, 2.5)
    p, c = kuka._PB["p"], kuka._PB["c"]
    direct = np.asarray(p.calculateInverseDynamics(b25, list(Q), list(QD), list(QDD), physicsClientId=c))
    model = kuka.KukaModel(CFG, 2.5)
    assert np.allclose(model.inverse_dynamics(Q, QD, QDD) - model.arm * QDD, direct, atol=1e-9)
    # the payload really changes the torque (guards against a no-op payload)
    assert np.max(np.abs(direct - kuka.KukaModel(CFG, 0.0).inverse_dynamics(Q, QD, QDD) + model.arm * QDD)) > 1.0


def test_forward_inverse_consistent():
    m = kuka.KukaModel(CFG, 1.3)
    tau = m.inverse_dynamics(Q, QD, QDD)
    assert np.allclose(m.forward_dynamics(Q, QD, tau), QDD, atol=1e-8)


def _short(scenario, **kw):
    spec = next(s for s in kuka.sample_specs(CFG, "indomain") if s["scenario"] == scenario)
    return dict(spec, **kw)


def test_healthy_ratio_near_one_and_fault_ratio_tracks_multiplier():
    spec = _short("severe_j6", duration=8.0, t_onset=3.0, mult_final=3.0)
    sig = kuka.simulate(spec, CFG)
    f = kuka.run_features(sig, spec["payload_declared"], CFG, 1.0)
    pre, post = f[f.t_end < 3.0], f[f.t_end > 4.5]
    b6 = spec["b"][5] / CFG["friction"]["nominal_b"][5]  # this run's healthy friction vs nominal
    assert pre["j6_fric_ratio"].median() == pytest.approx(b6, abs=0.15)
    assert post["j6_fric_ratio"].median() == pytest.approx(3.0 * b6, rel=0.15)
    assert post["j1_fric_ratio"].median() < 1.3  # other joints unaffected


def test_payload_error_is_estimated():
    spec = _short("healthy_speed", duration=8.0)
    spec["payload_declared"] = spec["payload"] - 0.8
    sig = kuka.simulate(spec, CFG)
    f = kuka.run_features(sig, spec["payload_declared"], CFG, 1.0)
    assert f["dm_hat"].median() == pytest.approx(0.8, abs=0.25)
