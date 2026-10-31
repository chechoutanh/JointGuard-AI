"""Physics sanity checks (Stage A). Run: python -m pytest tests/test_physics.py -v"""
import math

import numpy as np
import pytest

from src.config import load_config
from src.dynamics import ArmModel, RobotParams, forward_kinematics, friction_torque, solve_sym2
from src.faults import FaultSchedule
from src.simulator import Simulator, SimulationDivergedError
from src.trajectory import SinusoidTrajectory, PointToPointTrajectory

CFG = load_config()
P = RobotParams.from_config(CFG)
B = CFG["friction"]["nominal_b"]
TC = CFG["friction"]["nominal_tau_c"]


def make_sim(payload=1.0, fault=None, traj=None, noise=(0.0, 0.0), dt=None, b=B, tc=TC, cfg=CFG):
    traj = traj or SinusoidTrajectory((0.8, -1.2), (0.35, 0.4), (0.4, 0.5), (0.0, 1.0), 2.0)
    return Simulator(cfg, payload, payload, b, tc, traj, fault, noise[0], noise[1], 0, dt_physics=dt)


def test_forward_kinematics_known_configs():
    l1, l2 = P.l1, P.l2
    cases = {(0.0, 0.0): (l1 + l2, 0.0), (math.pi / 2, 0.0): (0.0, l1 + l2),
             (0.0, math.pi / 2): (l1, l2), (math.pi / 2, -math.pi / 2): (l2, l1)}
    for (q1, q2), (x, y) in cases.items():
        _, _, xt, yt = forward_kinematics(q1, q2, P)
        assert xt == pytest.approx(x, abs=1e-12) and yt == pytest.approx(y, abs=1e-12)


@pytest.mark.parametrize("payload", [0.0, 1.0, 3.0])
def test_mass_matrix_symmetric_positive_definite(payload):
    m = ArmModel(P, payload)
    for q2 in np.linspace(-math.pi, math.pi, 37):
        M = m.mass_matrix_np([0.0, q2])
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_linear_solve_matches_numpy():
    m = ArmModel(P, 1.5)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q2 = rng.uniform(-3, 3)
        r = rng.normal(size=2) * 20
        M = m.mass_matrix_np([0, q2])
        x = solve_sym2(M[0, 0], M[0, 1], M[1, 1], r[0], r[1])
        assert np.allclose(x, np.linalg.solve(M, r))


def test_gravity_matches_potential_energy_gradient():
    m = ArmModel(P, 2.0)
    h = 1e-6
    for q1, q2 in [(0.3, -1.0), (1.2, -0.4), (-0.5, -2.0)]:
        g = m.gravity(q1, q2)
        d1 = (m.potential_energy(q1 + h, q2) - m.potential_energy(q1 - h, q2)) / (2 * h)
        d2 = (m.potential_energy(q1, q2 + h) - m.potential_energy(q1, q2 - h)) / (2 * h)
        assert g[0] == pytest.approx(d1, rel=1e-6) and g[1] == pytest.approx(d2, rel=1e-6)


def test_payload_enters_gravity_as_point_mass_at_tip():
    q1, q2, mp = 0.7, -1.1, 2.0
    g0 = ArmModel(P, 0.0).gravity(q1, q2)
    g1 = ArmModel(P, mp).gravity(q1, q2)
    # extra torque from point mass at tip: J^T (0, -mp g) -> moment arms are tip x positions
    x_tip = P.l1 * math.cos(q1) + P.l2 * math.cos(q1 + q2)
    x_rel = P.l2 * math.cos(q1 + q2)
    assert g1[0] - g0[0] == pytest.approx(mp * P.g * x_tip)
    assert g1[1] - g0[1] == pytest.approx(mp * P.g * x_rel)


def test_static_gravity_balance():
    """At rest with tau = G(q) the acceleration must be zero (friction is 0 at qd=0)."""
    m = ArmModel(P, 1.2)
    for q1, q2 in [(0.8, -1.2), (0.2, -0.5), (1.5, -2.2)]:
        g = m.gravity(q1, q2)
        a = m.forward_dynamics(q1, q2, 0, 0, g[0], g[1], 0.0, 0.0)
        assert abs(a[0]) < 1e-12 and abs(a[1]) < 1e-12


def test_controller_holds_static_pose_without_noise():
    traj = PointToPointTrajectory([(0.8, -1.2), (0.8, -1.2)], [1.0], [5.0])
    sim = make_sim(traj=traj)
    sig, tru = sim.run(3.0)
    assert np.max(np.abs(tru[:, 1:3] - [0.8, -1.2])) < 1e-9
    assert np.max(np.abs(tru[:, 3:5])) < 1e-9


def test_energy_conserved_without_friction_and_torque():
    m = ArmModel(P, 1.0)
    from src.simulator import Simulator as S
    sim = make_sim(b=(0, 0), tc=(0, 0))
    sim.state = [0.5, -0.8, 1.0, -1.5]
    sim.check_joint_limits = False   # free swinging arm, no controller
    sim.controller.compute = lambda *a: ((0.0, 0.0), (0.0, 0.0))
    E0 = m.total_energy(*sim.state)
    for _ in range(100):  # 1 s
        sim.step()
    assert abs(m.total_energy(*sim.state) - E0) / abs(E0) < 1e-6


def test_friction_dissipates_energy():
    m = ArmModel(P, 1.0)
    for qd in np.linspace(-5, 5, 41):
        assert qd * friction_torque(qd, 2.0, 3.0, 0.02) >= 0.0
    sim = make_sim()
    sim.state = [0.5, -0.8, 1.0, -1.5]
    sim.check_joint_limits = False   # free swinging arm, no controller
    sim.controller.compute = lambda *a: ((0.0, 0.0), (0.0, 0.0))
    E = [m.total_energy(*sim.state)]
    for _ in range(100):
        sim.step()
        E.append(m.total_energy(*sim.state))
    assert np.all(np.diff(E) <= 1e-9)
    assert E[-1] < E[0]


def test_healthy_robot_tracks_stably():
    sim = make_sim(payload=2.0, noise=(0.0005, 0.005))
    sig, tru = sim.run(20.0)
    e = sig[:, 5:7] - tru[:, 1:3]
    late = sig[:, 0] > 2.5
    assert np.max(np.abs(e[late])) < 0.08
    # no growth: error in last 5 s not larger than error in 5-10 s
    a = np.sqrt(np.mean(e[(sig[:, 0] > 5) & (sig[:, 0] < 10)] ** 2))
    b = np.sqrt(np.mean(e[sig[:, 0] > 15] ** 2))
    assert b < 1.5 * a
    assert sim.saturated_ticks == 0


def test_halving_time_step_changes_little():
    f = FaultSchedule("abrupt", 2, 3.0, 4.0)
    s1 = make_sim(fault=f)
    s2 = make_sim(fault=FaultSchedule("abrupt", 2, 3.0, 4.0), dt=0.0005)
    _, t1 = s1.run(6.0)
    _, t2 = s2.run(6.0)
    assert np.max(np.abs(t1[:, 1:3] - t2[:, 1:3])) < 1e-5


def test_fault_changes_physics_not_controller():
    f = FaultSchedule("abrupt", 1, 4.0, 3.5)
    healthy, faulty = make_sim(), make_sim(fault=f)
    _, th = healthy.run(8.0)
    _, tf = faulty.run(8.0)
    before = th[:, 0] < 4.0
    assert np.allclose(th[before, 1:5], tf[before, 1:5])          # identical until onset
    assert np.max(np.abs(th[~before, 1] - tf[~before, 1])) > 1e-3  # diverges after
    assert faulty.controller.kp == healthy.controller.kp


def test_no_nan_under_severe_fault_and_heavy_payload():
    for j in (1, 2):
        sim = make_sim(payload=3.0, fault=FaultSchedule("abrupt", j, 2.0, 4.0), noise=(0.001, 0.01))
        sig, tru = sim.run(10.0)
        assert np.all(np.isfinite(sig)) and np.all(np.isfinite(tru))


def test_divergence_is_reported():
    # explicit integrator instability: 10 ms RK4 step with a very stiff viscous term
    # (b/M * dt >> 2.8, outside the RK4 stability region)
    cfg = load_config()
    cfg["timing"]["dt_physics"] = 0.01
    sim = make_sim(cfg=cfg, b=(2000.0, 2000.0))
    with pytest.raises(SimulationDivergedError):
        sim.run(5.0)
    sim = make_sim()
    sim.state = [float("nan"), 0.0, 0.0, 0.0]
    with pytest.raises(SimulationDivergedError):
        sim.step()


def test_gradual_label_rule():
    f = FaultSchedule("gradual", 1, 5.0, 3.0, 10.0, 1.5)
    assert f.t_label == pytest.approx(5.0 + 10.0 * 0.5 / 2.0)
    assert f.label(f.t_label - 0.01) == 0 and f.label(f.t_label) == 1
    assert f.multiplier_scalar(4.9) == 1.0 and f.multiplier_scalar(20) == 3.0
