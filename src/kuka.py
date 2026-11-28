"""KUKA LBR iiwa (7-DOF) generalisation experiment (docs/KUKA_EXPERIMENT.md).

Same design as the 2-DOF pipeline, n joints instead of 2:
  * plant: rigid-body dynamics from pybullet (URDF in pybullet_data) + armature + friction
    b*qd + tau_c*tanh(qd/eps); RK4 at dt_physics, the fault multiplier held over one physics step
  * controller: PD + gravity compensation with the DECLARED payload, ZOH at f_control, saturation
  * faults: src.faults.FaultSchedule on one joint (1..7), same label rule
  * detector inputs: measured q, qd, targets and applied torque only

pybullet is used only as a dynamics library (mass matrix, inverse dynamics); its own
stepSimulation is never called. The payload is a point mass added to link 7 at its centre of
mass, so the torque is affine in the payload (checked in tests/test_kuka.py).
"""
import math

import numpy as np
import pandas as pd

from .faults import FaultSchedule
from .features import window_end_indices, window_sizes
from .residual import ACC_HALF_WIDTH, smoothed_acceleration, window_residual_features
from .scenarios import parse_scenario
from .simulator import SimulationDivergedError
from .trajectory import PointToPointTrajectory, SinusoidTrajectory, trajectory_from_dict, within_limits

N = 7
_PB = {}  # per-process pybullet state: client and bodies keyed by payload


def load_body(cfg, payload):
    """Load the URDF with `payload` kg added to link 7's mass (centre of mass and inertia kept, i.e.
    a point mass at link 7's centre of mass). Note: pybullet's changeDynamics(mass=...) does NOT
    change calculateInverseDynamics, hence a separate body per payload."""
    import re
    import tempfile
    from pathlib import Path
    import pybullet as p
    import pybullet_data
    if "c" not in _PB:
        _PB.update(p=p, c=p.connect(p.DIRECT), bodies={})
        p.setGravity(0, 0, -cfg["robot"]["g"], physicsClientId=_PB["c"])
    if payload not in _PB["bodies"]:
        src = Path(pybullet_data.getDataPath()) / cfg["robot"]["urdf"]
        text = src.read_text()
        start = text.index('<link name="lbr_iiwa_link_7">')
        mt = re.compile(r'<mass value="([0-9.eE+-]+)"/>').search(text, start)
        text = text[:mt.start(1)] + repr(float(mt.group(1)) + payload) + text[mt.end(1):]
        text = text.replace('filename="meshes/', f'filename="{(src.parent / "meshes").as_posix()}/')
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
            f.write(text)
        _PB["bodies"][payload] = p.loadURDF(f.name, useFixedBase=True, physicsClientId=_PB["c"])
        Path(f.name).unlink()
    return _PB["bodies"][payload]


class KukaModel:
    """Rigid-body model of the arm carrying `payload` kg, plus motor armature on the diagonal.
    The dynamics are affine in a point-mass payload, so every quantity is interpolated exactly
    between two bodies (payload 0 and 1 kg) instead of loading one body per payload value."""

    def __init__(self, cfg, payload):
        self.b0, self.b1 = load_body(cfg, 0.0), load_body(cfg, 1.0)
        self.p, self.c = _PB["p"], _PB["c"]
        self.payload = float(payload)
        self.arm = np.asarray(cfg["robot"]["armature"], float)

    def _mix(self, f):
        a = np.asarray(f(self.b0))
        return a + self.payload * (np.asarray(f(self.b1)) - a)

    def inverse_dynamics(self, q, qd, qdd):
        """M(q) qdd + C(q, qd) qd + G(q), armature included, no friction."""
        args = [list(map(float, v)) for v in (q, qd, qdd)]
        tau = self._mix(lambda b: self.p.calculateInverseDynamics(b, *args, physicsClientId=self.c))
        return tau + self.arm * np.asarray(qdd)

    def gravity(self, q):
        return self.inverse_dynamics(q, np.zeros(N), np.zeros(N))

    def mass_matrix(self, q):
        qs = list(map(float, q))
        M = self._mix(lambda b: self.p.calculateMassMatrix(b, qs, physicsClientId=self.c))
        M[np.diag_indices(N)] += self.arm
        return M

    def forward_dynamics(self, q, qd, tau_net):
        return np.linalg.solve(self.mass_matrix(q), tau_net - self.inverse_dynamics(q, qd, np.zeros(N)))


# --- closed-loop simulation ---------------------------------------------------------
def _fault(spec, cfg):
    return FaultSchedule(kind=spec["fault_kind"], joint=int(spec["fault_joint"]), t_onset=float(spec["t_onset"]),
                         mult_final=float(spec["mult_final"]), ramp=float(spec["ramp"]),
                         label_threshold=cfg["fault"]["label_threshold_multiplier"])


def simulate(spec, cfg):
    """Returns dict of sampled arrays: t, q, qd, q_des, qd_des, tau (n, 7) and label (n,)."""
    rb, tm = cfg["robot"], cfg["timing"]
    plant, ctrl = KukaModel(cfg, spec["payload"]), KukaModel(cfg, spec["payload_declared"])
    kp, kd = np.asarray(cfg["controller"]["kp"]), np.asarray(cfg["controller"]["kd"])
    lim_tau = np.asarray(rb["torque_limits"])
    b, tc, eps = np.asarray(spec["b"]), np.asarray(spec["tau_c"]), cfg["friction"]["epsilon"]
    traj, fault = trajectory_from_dict(spec["traj"]), _fault(spec, cfg)
    rng = np.random.default_rng(spec["noise_seed"])
    dt, T_c = tm["dt_physics"], 1.0 / tm["f_control"]
    n_sub, every = int(round(T_c / dt)), int(round(tm["f_control"] / tm["f_sample"]))
    jl = np.asarray(rb["joint_limits"])
    q = np.asarray(traj(0.0)[0]) + np.asarray(spec["q0_offset"])
    qd = np.zeros(N)
    out = {k: [] for k in ("t", "q", "qd", "q_des", "qd_des", "tau", "label")}
    saturated = 0

    def deriv(q, qd, tau, mult):
        f = mult * (b * qd + tc * np.tanh(qd / eps))
        return qd, plant.forward_dynamics(q, qd, tau - f)

    for k in range(int(round(spec["duration"] / T_c))):
        t = k * T_c
        q_m = q + rng.normal(0.0, spec["noise_q"], N)
        qd_m = qd + rng.normal(0.0, spec["noise_qd"], N)
        q_des, qd_des = map(np.asarray, traj(t))
        req = kp * (q_des - q_m) + kd * (qd_des - qd_m) + ctrl.gravity(q_m)
        tau = np.clip(req, -lim_tau, lim_tau)
        saturated += bool(np.any(tau != req))
        if k % every == 0:
            for key, v in (("t", t), ("q", q_m), ("qd", qd_m), ("q_des", q_des), ("qd_des", qd_des),
                           ("tau", tau), ("label", fault.label(t))):
                out[key].append(v)
        for j in range(n_sub):
            ts = t + j * dt + 1e-9  # same round-off guard as src/simulator.py
            mult = np.ones(N)
            if fault.joint:
                mult[fault.joint - 1] = fault.multiplier_scalar(ts)
            k1 = deriv(q, qd, tau, mult)
            k2 = deriv(q + 0.5 * dt * k1[0], qd + 0.5 * dt * k1[1], tau, mult)
            k3 = deriv(q + 0.5 * dt * k2[0], qd + 0.5 * dt * k2[1], tau, mult)
            k4 = deriv(q + dt * k3[0], qd + dt * k3[1], tau, mult)
            q = q + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            qd = qd + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd))):
            raise SimulationDivergedError(f"non-finite state at t={t:.3f}")
        if np.any(np.abs(qd) > rb["max_abs_velocity"]):
            raise SimulationDivergedError(f"velocity limit exceeded at t={t:.3f}")
        if np.any(q < jl[:, 0]) or np.any(q > jl[:, 1]):
            raise SimulationDivergedError(f"joint limit left at t={t:.3f}")
    res = {k: np.asarray(v) for k, v in out.items()}
    res["saturated_fraction"] = saturated / max(k + 1, 1)
    res["t_label"] = fault.t_label
    return res


# --- scenarios ----------------------------------------------------------------------
def _u(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _trajectory(rng, dom, cfg, freq_range, p2p, duration):
    lim, margin = cfg["robot"]["joint_limits"], cfg["trajectory"]["joint_margin"]
    for _ in range(200):
        c = tuple(_u(rng, r) for r in dom["center"])
        if p2p:
            wps, moves, dwells, t = [c], [], [cfg["trajectory"]["t_ramp"]], cfg["trajectory"]["t_ramp"]
            while t < duration:
                wps.append(tuple(ci + _u(rng, (-dom["amp"][1], dom["amp"][1])) for ci in c))
                moves.append(_u(rng, dom["p2p_move_time"]))
                dwells.append(_u(rng, dom["p2p_dwell"]))
                t += moves[-1] + dwells[-1]
            tr = PointToPointTrajectory(wps, moves, dwells[:len(moves)])
        else:
            tr = SinusoidTrajectory(c, tuple(_u(rng, dom["amp"]) for _ in range(N)),
                                    tuple(_u(rng, freq_range) for _ in range(N)),
                                    tuple(_u(rng, (0, 2 * math.pi)) for _ in range(N)), cfg["trajectory"]["t_ramp"])
        if within_limits(tr, duration, lim, margin):
            return tr
    raise RuntimeError("could not sample a trajectory inside joint limits")


def sample_specs(cfg, subset):
    """One dict per run. Fault groups of each kind cycle through joints 1..7."""
    m, dom, fcfg, fr = cfg["run"], cfg["domains"][subset], cfg["fault"], cfg["friction"]
    counts = m["indomain_counts"] if subset == "indomain" else m["stress_counts"]
    reps, duration, tol = m["replicates_per_group"], m["duration"], fr["healthy_tolerance"]
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 0 if subset == "indomain" else 1]))
    prefix = "K" if subset == "indomain" else "KS"
    specs, run_idx, group_idx = [], 0, 0
    for base, n in counts.items():
        if n % reps:
            raise ValueError(f"count for {base} must be a multiple of replicates_per_group")
        for g in range(n // reps):
            group_idx += 1
            scen = base if base.startswith("healthy") else f"{base}_j{g % N + 1}"
            kind, severity, joint = parse_scenario(scen)
            if subset == "stress":
                freq, payload, p2p = dom["freq"], _u(rng, dom["payload_full"]), rng.uniform() < dom["p2p_fraction"]
            elif scen == "healthy_speed":
                freq, payload, p2p = dom["freq"], _u(rng, dom["payload_nominal"]), False
            elif scen == "healthy_load":
                freq, payload, p2p = dom["freq_low_speed_group"], _u(rng, dom["payload_full"]), False
            else:
                freq, payload, p2p = dom["freq"], _u(rng, dom["payload_full"]), False
            traj = _trajectory(rng, dom, cfg, freq, p2p, duration)
            b = [x * _u(rng, (1 - tol, 1 + tol)) for x in fr["nominal_b"]]
            tc = [x * _u(rng, (1 - tol, 1 + tol)) for x in fr["nominal_tau_c"]]
            for _ in range(reps):
                run_idx += 1
                spec = {"run_id": f"{prefix}{run_idx:04d}", "group_id": f"{prefix}G{group_idx:03d}",
                        "subset": subset, "scenario": scen, "fault_kind": kind, "severity": severity,
                        "fault_joint": joint, "duration": duration, "payload": payload,
                        "payload_declared": max(0.0, payload + _u(rng, (-1, 1)) * cfg["controller"]["payload_declared_error"]),
                        "b": b, "tau_c": tc,
                        "noise_q": _u(rng, cfg["sensor"]["noise_q_range"]),
                        "noise_qd": _u(rng, cfg["sensor"]["noise_qd_range"]),
                        "q0_offset": [_u(rng, (-1, 1)) * cfg["trajectory"]["initial_offset_max"] for _ in range(N)],
                        "noise_seed": int(rng.integers(0, 2 ** 31 - 1)),
                        "traj": traj.to_dict(), "traj_kind": traj.kind,
                        "t_onset": math.inf, "mult_final": 1.0, "ramp": 0.0}
                if kind == "abrupt":
                    spec["t_onset"] = _u(rng, m["onset_range"])
                    spec["mult_final"] = _u(rng, fcfg["mild_range" if severity == "mild" else "severe_range"])
                elif kind == "gradual":
                    spec["t_onset"] = _u(rng, m["gradual_onset_range"])
                    spec["mult_final"] = _u(rng, fcfg["gradual_final_range"])
                    spec["ramp"] = _u(rng, fcfg["gradual_ramp_range"])
                specs.append(spec)
    return specs


# --- features -----------------------------------------------------------------------
RATIO_NAMES = [f"j{j}_fric_ratio" for j in range(1, N + 1)]
FEATURE_NAMES = RATIO_NAMES + ["dm_hat"] + [f"j{j}_moving_frac" for j in range(1, N + 1)] + \
    [f"j{j}_e_rms" for j in range(1, N + 1)]


def residual_signals(sig, payload_declared, cfg):
    """r = tau - tau_hat(declared payload), y = d tau / d payload, both (7, n); NaN where qdd is unknown."""
    qd = sig["qd"].T
    qdd = smoothed_acceleration(qd, 1.0 / cfg["timing"]["f_sample"])
    ok = np.all(np.isfinite(qdd), axis=0)
    r, y = np.full(qd.shape, np.nan), np.full(qd.shape, np.nan)
    models = [KukaModel(cfg, m) for m in (payload_declared, 0.0, 1.0)]
    idx = np.flatnonzero(ok)
    tau_hat = [np.array([mdl.inverse_dynamics(sig["q"][i], sig["qd"][i], qdd[:, i]) for i in idx]).T for mdl in models]
    r[:, idx] = sig["tau"][idx].T - tau_hat[0]
    y[:, idx] = tau_hat[2] - tau_hat[1]
    return r, y, qd


def run_features(sig, payload_declared, cfg, window_s):
    """One row per window (same grid and causality as the 2-DOF residual features)."""
    fc, fr, fs = cfg["features"], cfg["friction"], cfg["timing"]["f_sample"]
    r, y, qd = residual_signals(sig, payload_declared, cfg)
    e = sig["q_des"] - sig["q"]
    wn, sn = window_sizes(window_s, fc["stride_s"], fs)
    ends = window_end_indices(len(sig["t"]), wn, sn, fs, fc["monitor_start_s"])
    K, lab = ACC_HALF_WIDTH, sig["label"]
    rows = []
    for e_i in ends:
        sl = slice(e_i - wn + 1, e_i - K + 1)
        f = window_residual_features(r[:, sl], y[:, sl], qd[:, sl], fr["epsilon"], fr["nominal_b"], fr["nominal_tau_c"])
        w = lab[e_i - wn + 1:e_i + 1]
        rows.append([sig["t"][e_i], int(lab[e_i]), bool(w.min() != w.max())] + list(f) +
                    list(np.sqrt(np.mean(e[e_i - wn + 1:e_i + 1] ** 2, axis=0))))
    return pd.DataFrame(rows, columns=["t_end", "label", "mixed"] + FEATURE_NAMES)
