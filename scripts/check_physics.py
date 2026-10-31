"""Run the physics test-suite and print a short numerical report (Stage A).

  python -m scripts.check_physics
"""
import sys

import numpy as np
import pytest

from src.config import load_config
from src.faults import FaultSchedule
from src.simulator import Simulator
from src.trajectory import SinusoidTrajectory


def report():
    cfg = load_config()
    fr = cfg["friction"]
    tr = SinusoidTrajectory((0.8, -1.2), (0.4, 0.45), (0.5, 0.55), (0.3, 1.0), 2.0)
    print("\nStep-size check (abrupt x4 fault on joint 2 at t=3 s, payload 2 kg, 6 s):")
    res = {}
    for dt in (0.001, 0.0005, 0.00025):
        s = Simulator(cfg, 2.0, 2.0, fr["nominal_b"], fr["nominal_tau_c"], tr,
                      FaultSchedule("abrupt", 2, 3.0, 4.0), 0, 0, 0, dt_physics=dt)
        res[dt] = s.run(6.0)[1][:, 1:3]
    print(f"  max |q(dt=1ms) - q(dt=0.5ms)|   = {np.abs(res[0.001] - res[0.0005]).max():.2e} rad")
    print(f"  max |q(dt=0.5ms) - q(dt=0.25ms)| = {np.abs(res[0.0005] - res[0.00025]).max():.2e} rad")
    print("Healthy tracking (payload 0/2 kg, 20 s, nominal friction, noise on):")
    for payload in (0.0, 2.0):
        s = Simulator(cfg, payload, payload, fr["nominal_b"], fr["nominal_tau_c"], tr, None, 0.0005, 0.005, 0)
        sig, tru = s.run(20.0)
        e = sig[sig[:, 0] > 2.5][:, 5:7] - tru[tru[:, 0] > 2.5][:, 1:3]
        print(f"  payload {payload} kg: RMS error j1={np.sqrt((e[:,0]**2).mean()):.4f} rad, "
              f"j2={np.sqrt((e[:,1]**2).mean()):.4f} rad, max |tau| = {np.abs(sig[:, 11:13]).max(0).round(1)} N*m, "
              f"saturated ticks = {s.saturated_ticks}")


if __name__ == "__main__":
    code = pytest.main(["-q", "tests/test_physics.py"])
    report()
    sys.exit(code)
