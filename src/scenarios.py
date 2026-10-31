"""Scenario sampling for batch data generation (Stage B).

A *group* (base experiment) fixes the trajectory, payload and the run's healthy
friction values. Each group has `replicates_per_group` runs that differ in sensor
noise (level and seed), small initial offset, declared-payload error and fault
timing/magnitude. Runs of the same group always end up in the same data split.

Scenario types: healthy_speed, healthy_load, mild_j{1,2}, severe_j{1,2}, gradual_j{1,2}.
Sensor-noise levels are sampled from the same range for every scenario type, so noise
carries no information about the label.
"""
import math

import numpy as np

from .trajectory import SinusoidTrajectory, PointToPointTrajectory, within_limits

SCENARIO_TYPES = ["healthy_speed", "healthy_load", "mild_j1", "mild_j2",
                  "severe_j1", "severe_j2", "gradual_j1", "gradual_j2"]


def parse_scenario(name):
    """-> (fault_kind, severity, joint)"""
    if name.startswith("healthy"):
        return "none", "healthy", 0
    sev, j = name.split("_j")
    return ("gradual" if sev == "gradual" else "abrupt"), sev, int(j)


def _u(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _sample_sinusoid(rng, dom, freq_range, cfg, duration):
    lim = cfg["robot"]["joint_limits"]
    margin = cfg["trajectory"]["joint_margin"]
    for _ in range(200):
        tr = SinusoidTrajectory(
            center=(_u(rng, dom["center_q1"]), _u(rng, dom["center_q2"])),
            amp=(_u(rng, dom["amp1"]), _u(rng, dom["amp2"])),
            freq=(_u(rng, freq_range), _u(rng, freq_range)),
            phase=(_u(rng, (0, 2 * math.pi)), _u(rng, (0, 2 * math.pi))),
            t_ramp=cfg["trajectory"]["t_ramp"])
        if within_limits(tr, duration, lim, margin):
            return tr
    raise RuntimeError("could not sample a trajectory inside joint limits")


def _sample_p2p(rng, dom, cfg, duration):
    lim = cfg["robot"]["joint_limits"]
    margin = cfg["trajectory"]["joint_margin"]
    for _ in range(200):
        c = (_u(rng, dom["center_q1"]), _u(rng, dom["center_q2"]))
        a = (dom["amp1"][1], dom["amp2"][1])
        wps = [c]
        moves, dwells = [], [cfg["trajectory"]["t_ramp"]]
        t = dwells[0]
        while t < duration:
            wps.append((c[0] + _u(rng, (-a[0], a[0])), c[1] + _u(rng, (-a[1], a[1]))))
            moves.append(_u(rng, dom["p2p_move_time"]))
            dwells.append(_u(rng, dom["p2p_dwell"]))
            t += moves[-1] + dwells[-1]
        tr = PointToPointTrajectory(wps, moves, dwells[:len(moves)])
        if within_limits(tr, duration, lim, margin):
            return tr
    raise RuntimeError("could not sample a p2p trajectory inside joint limits")


def sample_run_specs(cfg, mode, subset):
    """Returns a list of dicts, one per run, fully describing how to simulate it."""
    m = cfg["modes"][mode]
    dom = cfg["domains"][subset]
    duration = m["duration"]
    counts = m["indomain_counts"] if subset == "indomain" else m["stress_counts"]
    reps = m["replicates_per_group"]
    subset_code = 0 if subset == "indomain" else 1
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], subset_code]))
    fcfg, scfg, fr = cfg["fault"], cfg["sensor"], cfg["friction"]
    tol = fr["healthy_tolerance"]
    prefix = "R" if subset == "indomain" else "S"
    specs, run_idx, group_idx = [], 0, 0
    for scen in SCENARIO_TYPES:
        n = counts.get(scen, 0)
        if n % reps:
            raise ValueError(f"count for {scen} must be a multiple of replicates_per_group")
        kind, severity, joint = parse_scenario(scen)
        for _ in range(n // reps):
            group_idx += 1
            # --- base experiment (shared by replicates) ---
            if subset == "stress":
                freq_range = dom["freq"]
                payload = _u(rng, dom["payload_full"])
                use_p2p = rng.uniform() < dom["p2p_fraction"]
            else:
                use_p2p = False
                if scen == "healthy_speed":
                    freq_range, payload = dom["freq"], _u(rng, dom["payload_nominal"])
                elif scen == "healthy_load":
                    freq_range, payload = dom["freq_low_speed_group"], _u(rng, dom["payload_full"])
                else:
                    freq_range, payload = dom["freq"], _u(rng, dom["payload_full"])
            traj = _sample_p2p(rng, dom, cfg, duration) if use_p2p else \
                _sample_sinusoid(rng, dom, freq_range, cfg, duration)
            b = [fr["nominal_b"][i] * _u(rng, (1 - tol, 1 + tol)) for i in range(2)]
            tc = [fr["nominal_tau_c"][i] * _u(rng, (1 - tol, 1 + tol)) for i in range(2)]
            for r in range(reps):
                run_idx += 1
                spec = {
                    "run_id": f"{prefix}{run_idx:04d}", "group_id": f"{prefix}G{group_idx:03d}",
                    "subset": subset, "scenario": scen, "fault_kind": kind,
                    "severity": severity, "fault_joint": joint, "duration": duration,
                    "payload": payload,
                    "payload_declared": max(0.0, payload + _u(rng, (-1, 1)) * cfg["controller"]["payload_declared_error"]),
                    "b1": b[0], "b2": b[1], "tau_c1": tc[0], "tau_c2": tc[1],
                    "noise_q": _u(rng, scfg["noise_q_range"]),
                    "noise_qd": _u(rng, scfg["noise_qd_range"]),
                    "q0_offset1": _u(rng, (-1, 1)) * cfg["trajectory"]["initial_offset_max"],
                    "q0_offset2": _u(rng, (-1, 1)) * cfg["trajectory"]["initial_offset_max"],
                    "noise_seed": int(rng.integers(0, 2 ** 31 - 1)),
                    "traj": traj.to_dict(), "traj_kind": traj.kind,
                    "t_onset": math.inf, "mult_final": 1.0, "ramp": 0.0,
                }
                if kind == "abrupt":
                    spec["t_onset"] = _u(rng, m["onset_range"])
                    spec["mult_final"] = _u(rng, fcfg["mild_range" if severity == "mild" else "severe_range"])
                elif kind == "gradual":
                    spec["t_onset"] = _u(rng, m["gradual_onset_range"])
                    spec["mult_final"] = _u(rng, fcfg["gradual_final_range"])
                    spec["ramp"] = _u(rng, fcfg["gradual_ramp_range"])
                specs.append(spec)
    return specs
