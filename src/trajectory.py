"""Smooth joint-space reference trajectories.

* SinusoidTrajectory: q_des_i(t) = c_i + s(t) * A_i * sin(2*pi*f_i*t + phi_i)
  with a quintic smooth-start factor s(t) (0 -> 1 over t_ramp) so the reference
  starts at rest at the centre configuration (no step in position/velocity).
* PointToPointTrajectory: quintic (minimum-jerk) moves between waypoints with
  dwell periods. Used only for the held-out stress test.
All trajectories return (q_des, qd_des) analytically.
"""
import math

import numpy as np


def _smoothstep(t, T):
    """Quintic 0->1 blend with zero 1st and 2nd derivative at both ends. Returns (s, ds/dt)."""
    if t <= 0.0:
        return 0.0, 0.0
    if t >= T:
        return 1.0, 0.0
    x = t / T
    s = x ** 3 * (10 - 15 * x + 6 * x * x)
    ds = (30 * x * x - 60 * x ** 3 + 30 * x ** 4) / T
    return s, ds


class SinusoidTrajectory:
    kind = "sinusoid"

    def __init__(self, center, amp, freq, phase, t_ramp):
        self.center = tuple(map(float, center))
        self.amp = tuple(map(float, amp))
        self.freq = tuple(map(float, freq))
        self.phase = tuple(map(float, phase))
        self.t_ramp = float(t_ramp)

    def __call__(self, t):
        s, ds = _smoothstep(t, self.t_ramp)
        q, qd = [], []
        for i in range(2):
            w = 2.0 * math.pi * self.freq[i]
            sn = math.sin(w * t + self.phase[i])
            cs = math.cos(w * t + self.phase[i])
            q.append(self.center[i] + s * self.amp[i] * sn)
            qd.append(ds * self.amp[i] * sn + s * self.amp[i] * w * cs)
        return (q[0], q[1]), (qd[0], qd[1])

    def to_dict(self):
        return {"kind": self.kind, "center": self.center, "amp": self.amp,
                "freq": self.freq, "phase": self.phase, "t_ramp": self.t_ramp}


class PointToPointTrajectory:
    kind = "p2p"

    def __init__(self, waypoints, move_times, dwell_times):
        """waypoints: list of (q1, q2); segment k moves waypoints[k] -> waypoints[k+1]
        in move_times[k] seconds after dwelling dwell_times[k] seconds at waypoints[k]."""
        self.waypoints = [tuple(map(float, w)) for w in waypoints]
        self.move_times = list(map(float, move_times))
        self.dwell_times = list(map(float, dwell_times))
        self._starts = []
        t = 0.0
        for k in range(len(self.move_times)):
            t += self.dwell_times[k]
            self._starts.append(t)
            t += self.move_times[k]
        self.t_end = t

    def __call__(self, t):
        for k, ts in enumerate(self._starts):
            T = self.move_times[k]
            if t < ts:
                w = self.waypoints[k]
                return w, (0.0, 0.0)
            if t < ts + T:
                s, ds = _smoothstep(t - ts, T)
                a, b = self.waypoints[k], self.waypoints[k + 1]
                return ((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])),
                        (ds * (b[0] - a[0]), ds * (b[1] - a[1])))
        return self.waypoints[-1], (0.0, 0.0)

    def to_dict(self):
        return {"kind": self.kind, "waypoints": self.waypoints,
                "move_times": self.move_times, "dwell_times": self.dwell_times}


def trajectory_from_dict(d):
    d = dict(d)
    kind = d.pop("kind")
    if kind == "sinusoid":
        return SinusoidTrajectory(**d)
    if kind == "p2p":
        return PointToPointTrajectory(**d)
    raise ValueError(kind)


def within_limits(traj, duration, joint_limits, margin, dt=0.01):
    """True if the reference stays inside joint limits (minus margin) for the whole run."""
    for t in np.arange(0.0, duration + dt, dt):
        q, _ = traj(float(t))
        for i in range(2):
            lo, hi = joint_limits[i]
            if not (lo + margin <= q[i] <= hi - margin):
                return False
    return True
