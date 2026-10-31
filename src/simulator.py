"""Closed-loop simulation: plant (RK4 at dt_physics) + controller (zero-order hold at
f_control) + sensors (f_sample). The robot angle is never set from the target; it only
evolves by integrating the equation of motion.

Timeline of one control period [t_k, t_k + T_c):
  1. sensors read true state at t_k and add Gaussian noise (q_meas, qd_meas)
  2. controller computes tau_requested from measurements; saturation -> tau_applied
  3. tau_applied is held constant while the plant is integrated with RK4 for
     T_c / dt_physics sub-steps. Friction, gravity, Coriolis and the (possibly
     time-varying) fault multiplier are re-evaluated at every RK4 stage.
"""
import math

import numpy as np

from .dynamics import ArmModel, RobotParams, friction_torque
from .controller import PDGravityController
from .faults import FaultSchedule


class SimulationDivergedError(RuntimeError):
    """Numerical instability or physically invalid state. Such runs are logged
    separately and are NEVER labelled as mechanical faults."""


SIGNAL_COLUMNS = ["t", "q1_meas", "q2_meas", "qd1_meas", "qd2_meas",
                  "q1_des", "q2_des", "qd1_des", "qd2_des",
                  "tau1_req", "tau2_req", "tau1_applied", "tau2_applied"]
TRUTH_COLUMNS = ["t", "q1_true", "q2_true", "qd1_true", "qd2_true",
                 "fric_mult1", "fric_mult2", "label"]


class Simulator:
    def __init__(self, cfg, payload, payload_declared, friction_b, friction_tau_c,
                 trajectory, fault: FaultSchedule = None, noise_q=0.0, noise_qd=0.0,
                 noise_seed=0, q0=None, dt_physics=None):
        self.cfg = cfg
        self.params = RobotParams.from_config(cfg)
        self.model = ArmModel(self.params, payload)
        self.controller = PDGravityController.from_config(cfg, self.params, payload_declared)
        self.b = tuple(map(float, friction_b))
        self.tau_c = tuple(map(float, friction_tau_c))
        self.eps = float(cfg["friction"]["epsilon"])
        self.trajectory = trajectory
        self.fault = fault or FaultSchedule()
        self.noise_q = float(noise_q)
        self.noise_qd = float(noise_qd)
        self.rng = np.random.default_rng(noise_seed)
        tm = cfg["timing"]
        self.dt = float(dt_physics or tm["dt_physics"])
        self.T_c = 1.0 / tm["f_control"]
        self.n_sub = int(round(self.T_c / self.dt))
        if abs(self.n_sub * self.dt - self.T_c) > 1e-12:
            raise ValueError("control period must be an integer multiple of dt_physics")
        ratio = tm["f_control"] / tm["f_sample"]
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("f_control must be an integer multiple of f_sample")
        self.sample_every = int(round(ratio))
        self.max_vel = float(cfg["robot"]["max_abs_velocity"])
        self.joint_limits = cfg["robot"]["joint_limits"]
        qd0 = trajectory(0.0)[0]
        q0 = q0 if q0 is not None else qd0
        self.state = [float(q0[0]), float(q0[1]), 0.0, 0.0]
        self.k = 0  # control tick counter
        self.check_joint_limits = True
        self.saturated_ticks = 0

    @property
    def t(self):
        return self.k * self.T_c

    # --- plant -------------------------------------------------------------
    def _deriv(self, s, m1, m2, tau1, tau2):
        q1, q2, qd1, qd2 = s
        tf1 = friction_torque(qd1, self.b[0] * m1, self.tau_c[0] * m1, self.eps)
        tf2 = friction_torque(qd2, self.b[1] * m2, self.tau_c[1] * m2, self.eps)
        a1, a2 = self.model.forward_dynamics(q1, q2, qd1, qd2, tau1, tau2, tf1, tf2)
        return (qd1, qd2, a1, a2)

    def _rk4(self, s, t, h, tau1, tau2):
        # The fault multiplier is a slowly varying *parameter*: it is held constant over
        # one physics step (evaluated at the step start) so that an abrupt onset never
        # falls between RK4 stages. All state-dependent terms are evaluated per stage.
        m1, m2 = self.fault.multipliers(t)
        k1 = self._deriv(s, m1, m2, tau1, tau2)
        s2 = [s[i] + 0.5 * h * k1[i] for i in range(4)]
        k2 = self._deriv(s2, m1, m2, tau1, tau2)
        s3 = [s[i] + 0.5 * h * k2[i] for i in range(4)]
        k3 = self._deriv(s3, m1, m2, tau1, tau2)
        s4 = [s[i] + h * k3[i] for i in range(4)]
        k4 = self._deriv(s4, m1, m2, tau1, tau2)
        return [s[i] + h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4)]

    def _check_state(self):
        q1, q2, qd1, qd2 = self.state
        if not all(math.isfinite(v) for v in self.state):
            raise SimulationDivergedError(f"non-finite state at t={self.t:.3f}: {self.state}")
        if abs(qd1) > self.max_vel or abs(qd2) > self.max_vel:
            raise SimulationDivergedError(f"velocity exceeds {self.max_vel} rad/s at t={self.t:.3f}")
        for i, q in enumerate((q1, q2) if self.check_joint_limits else ()):
            lo, hi = self.joint_limits[i]
            if not (lo <= q <= hi):
                raise SimulationDivergedError(f"joint {i+1} left its limits at t={self.t:.3f}: {q:.3f}")

    # --- one control period ------------------------------------------------------
    def step(self):
        """Advance one control period. Returns (signal_row, truth_row) sampled at the start
        of the period (None, None if this tick is not a sampling tick)."""
        t = self.t
        q1, q2, qd1, qd2 = self.state
        q_meas = (q1 + self.rng.normal(0.0, self.noise_q), q2 + self.rng.normal(0.0, self.noise_q))
        qd_meas = (qd1 + self.rng.normal(0.0, self.noise_qd), qd2 + self.rng.normal(0.0, self.noise_qd))
        q_des, qd_des = self.trajectory(t)
        tau_req, tau_app = self.controller.compute(q_meas, qd_meas, q_des, qd_des)
        if tau_req != tau_app:
            self.saturated_ticks += 1
        sig = truth = None
        if self.k % self.sample_every == 0:
            m1, m2 = self.fault.multipliers(t)
            sig = (t, q_meas[0], q_meas[1], qd_meas[0], qd_meas[1], q_des[0], q_des[1],
                   qd_des[0], qd_des[1], tau_req[0], tau_req[1], tau_app[0], tau_app[1])
            truth = (t, q1, q2, qd1, qd2, m1, m2, self.fault.label(t))
        s = self.state
        for j in range(self.n_sub):
            # +1e-9 guards against k*dt round-off putting a grid-aligned onset one step late
            s = self._rk4(s, t + j * self.dt + 1e-9, self.dt, tau_app[0], tau_app[1])
        self.state = s
        self.k += 1
        self._check_state()
        return sig, truth

    def run(self, duration):
        n_ticks = int(round(duration / self.T_c))
        sig_rows, truth_rows = [], []
        for _ in range(n_ticks):
            sig, truth = self.step()
            if sig is not None:
                sig_rows.append(sig)
                truth_rows.append(truth)
        return np.array(sig_rows), np.array(truth_rows)
