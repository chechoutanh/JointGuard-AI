"""Friction fault injection. Faults ONLY scale the physical friction parameters
(b and tau_c) of one joint inside the plant model; the controller is unaware.

Kinds:
  * none    : multiplier = 1 at all times.
  * abrupt  : multiplier jumps from 1 to `mult_final` at t_onset.
  * gradual : multiplier rises linearly from 1 at t_onset to `mult_final` at t_onset + ramp.

Label rule (defined before any data is generated, independent of tracking error):
  anomaly (1) when the injected multiplier >= label_threshold (default 1.5, i.e.
  friction at least +50 % above the run's healthy value). For abrupt faults this is
  exactly t_onset; for gradual faults it is the crossing time
  t_label = t_onset + ramp * (label_threshold - 1) / (mult_final - 1).
"""
import math
from dataclasses import dataclass, asdict


@dataclass
class FaultSchedule:
    kind: str = "none"          # none | abrupt | gradual
    joint: int = 0              # 1 or 2 (0 for none)
    t_onset: float = math.inf
    mult_final: float = 1.0
    ramp: float = 0.0
    label_threshold: float = 1.5

    def multiplier_scalar(self, t):
        if self.kind == "none" or t < self.t_onset:
            return 1.0
        if self.kind == "abrupt":
            return self.mult_final
        if self.kind == "gradual":
            frac = min((t - self.t_onset) / self.ramp, 1.0) if self.ramp > 0 else 1.0
            return 1.0 + frac * (self.mult_final - 1.0)
        raise ValueError(self.kind)

    def multipliers(self, t):
        m = self.multiplier_scalar(t)
        if self.joint == 1:
            return m, 1.0
        if self.joint == 2:
            return 1.0, m
        return 1.0, 1.0

    @property
    def t_label(self):
        """Time at which the ground-truth label switches to anomaly (inf if never)."""
        if self.kind == "none" or self.mult_final < self.label_threshold:
            return math.inf
        if self.kind == "abrupt":
            return self.t_onset
        return self.t_onset + self.ramp * (self.label_threshold - 1.0) / (self.mult_final - 1.0)

    def label(self, t):
        return 1 if t >= self.t_label else 0

    def to_dict(self):
        return asdict(self)
