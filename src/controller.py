"""PD + gravity-compensation joint controller with per-joint torque saturation.

What the controller knows:
  * Nominal link parameters (identical to the simulated arm).
  * The payload *declared* by the operator (payload_declared), which may differ
    slightly from the true payload. It never sees the true payload directly.
  * It knows NOTHING about friction (healthy or faulty) and does not compensate it.
It only receives measured (noisy) joint angles and velocities.
"""
from .dynamics import ArmModel, RobotParams


class PDGravityController:
    def __init__(self, params: RobotParams, kp, kd, torque_limits, payload_declared):
        self.kp = tuple(float(k) for k in kp)
        self.kd = tuple(float(k) for k in kd)
        self.torque_limits = tuple(float(t) for t in torque_limits)
        self.set_payload_declared(params, payload_declared)

    def set_payload_declared(self, params, payload_declared):
        self.payload_declared = float(payload_declared)
        self._model = ArmModel(params, self.payload_declared)

    def compute(self, q_meas, qd_meas, q_des, qd_des):
        """Returns (tau_requested, tau_applied) as 2-tuples. Applied = saturated requested."""
        g1, g2 = self._model.gravity(q_meas[0], q_meas[1])
        req1 = self.kp[0] * (q_des[0] - q_meas[0]) + self.kd[0] * (qd_des[0] - qd_meas[0]) + g1
        req2 = self.kp[1] * (q_des[1] - q_meas[1]) + self.kd[1] * (qd_des[1] - qd_meas[1]) + g2
        lim1, lim2 = self.torque_limits
        app1 = min(max(req1, -lim1), lim1)
        app2 = min(max(req2, -lim2), lim2)
        return (req1, req2), (app1, app2)

    @classmethod
    def from_config(cls, cfg, params, payload_declared):
        c = cfg["controller"]
        return cls(params, c["kp"], c["kd"], cfg["robot"]["torque_limits"], payload_declared)
