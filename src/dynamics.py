"""Rigid-body dynamics and kinematics of a planar 2-link (RR) arm in a vertical plane.

Conventions (see README):
  * Base frame: origin at joint 1, x to the right, y up. Gravity acts along -y.
  * q1: angle of link 1 measured from +x axis, counter-clockwise positive [rad].
  * q2: angle of link 2 relative to link 1, counter-clockwise positive [rad].
  * Joint torques tau_i are positive counter-clockwise [N*m].

Equation of motion:
  M(q) qdd + C(q, qd) qd + G(q) + tau_f(qd) = tau_applied + tau_external

The payload is a point mass at the tip of link 2. It is merged into link 2 as an
equivalent rigid body (mass, centre of mass, inertia via the parallel axis theorem),
so the same parameters enter both the inertia matrix and the gravity vector.
Reflected actuator (rotor) inertia ``armature_i`` is added to the diagonal of M.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotParams:
    l1: float
    l2: float
    m1: float
    m2: float
    lc1: float
    lc2: float
    I1: float  # inertia of link 1 about its own centre of mass [kg m^2]
    I2: float
    armature1: float
    armature2: float
    g: float

    @classmethod
    def from_config(cls, cfg):
        r = cfg["robot"]
        return cls(r["l1"], r["l2"], r["m1"], r["m2"], r["lc1"], r["lc2"],
                   r["I1"], r["I2"], r["armature1"], r["armature2"], r["g"])


class ArmModel:
    """Dynamics of the arm for a given payload mass. Pure-float math for speed."""

    def __init__(self, params: RobotParams, payload: float = 0.0):
        p = params
        self.params = p
        self.payload = float(payload)
        mp = self.payload
        # Equivalent link 2 (link 2 + point payload at its tip).
        m2e = p.m2 + mp
        lc2e = (p.m2 * p.lc2 + mp * p.l2) / m2e
        I2e = p.I2 + p.m2 * (p.lc2 - lc2e) ** 2 + mp * (p.l2 - lc2e) ** 2
        self.m2e, self.lc2e, self.I2e = m2e, lc2e, I2e
        d = m2e * lc2e ** 2 + I2e
        self._a = p.m1 * p.lc1 ** 2 + p.I1 + m2e * p.l1 ** 2 + d + p.armature1
        self._beta = m2e * p.l1 * lc2e
        self._d = d
        self._m22 = d + p.armature2
        self._g1 = (p.m1 * p.lc1 + m2e * p.l1) * p.g
        self._g2 = m2e * lc2e * p.g

    # --- individual terms -------------------------------------------------
    def mass_matrix(self, q2):
        c2 = math.cos(q2)
        m11 = self._a + 2.0 * self._beta * c2
        m12 = self._d + self._beta * c2
        return m11, m12, self._m22

    def coriolis_vector(self, q2, qd1, qd2):
        """Returns C(q, qd) qd (Coriolis + centrifugal torques)."""
        h = -self._beta * math.sin(q2)
        return h * (2.0 * qd1 * qd2 + qd2 * qd2), -h * qd1 * qd1

    def gravity(self, q1, q2):
        c12 = math.cos(q1 + q2)
        return self._g1 * math.cos(q1) + self._g2 * c12, self._g2 * c12

    def mass_matrix_np(self, q):
        m11, m12, m22 = self.mass_matrix(q[1])
        return np.array([[m11, m12], [m12, m22]])

    # --- energy -----------------------------------------------------------
    def kinetic_energy(self, q2, qd1, qd2):
        m11, m12, m22 = self.mass_matrix(q2)
        return 0.5 * (m11 * qd1 * qd1 + 2.0 * m12 * qd1 * qd2 + m22 * qd2 * qd2)

    def potential_energy(self, q1, q2):
        p = self.params
        return (p.m1 * p.lc1 * math.sin(q1) + self.m2e * (p.l1 * math.sin(q1) + self.lc2e * math.sin(q1 + q2))) * p.g

    def total_energy(self, q1, q2, qd1, qd2):
        return self.kinetic_energy(q2, qd1, qd2) + self.potential_energy(q1, q2)

    # --- forward dynamics ---------------------------------------------------
    def forward_dynamics(self, q1, q2, qd1, qd2, tau1, tau2, tf1=0.0, tf2=0.0):
        """Solve M qdd = tau - C qd - G - tau_f for qdd (2x2 Gaussian elimination, no inverse)."""
        m11, m12, m22 = self.mass_matrix(q2)
        c1, c2 = self.coriolis_vector(q2, qd1, qd2)
        g1, g2 = self.gravity(q1, q2)
        r1 = tau1 - c1 - g1 - tf1
        r2 = tau2 - c2 - g2 - tf2
        return solve_sym2(m11, m12, m22, r1, r2)


def solve_sym2(m11, m12, m22, r1, r2):
    """Solve the symmetric 2x2 system [[m11,m12],[m12,m22]] x = r by elimination."""
    f = m12 / m11
    x2 = (r2 - f * r1) / (m22 - f * m12)
    x1 = (r1 - m12 * x2) / m11
    return x1, x2


def friction_torque(qd, b, tau_c, eps):
    """Viscous + smoothed Coulomb friction: b*qd + tau_c*tanh(qd/eps). Opposes motion."""
    return b * qd + tau_c * math.tanh(qd / eps)


def forward_kinematics(q1, q2, params: RobotParams):
    """Returns elbow (x1, y1) and tip (x2, y2) positions in the base frame [m]."""
    x1 = params.l1 * math.cos(q1)
    y1 = params.l1 * math.sin(q1)
    x2 = x1 + params.l2 * math.cos(q1 + q2)
    y2 = y1 + params.l2 * math.sin(q1 + q2)
    return x1, y1, x2, y2


def forward_kinematics_np(q1, q2, params: RobotParams):
    """Vectorised FK for arrays of joint angles; returns tip x, y."""
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    x = params.l1 * np.cos(q1) + params.l2 * np.cos(q1 + q2)
    y = params.l1 * np.sin(q1) + params.l2 * np.sin(q1 + q2)
    return x, y
