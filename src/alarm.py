"""Alarm persistence rule shared by the Random Forest detector and the baseline.

Raw decision per window: score >= threshold.
Alarm turns ON after `n_on` consecutive positive windows and OFF after `n_off`
consecutive negative windows. One *alarm event* = one OFF -> ON transition.
"""
import numpy as np


class AlarmLogic:
    def __init__(self, n_on, n_off):
        self.n_on, self.n_off = int(n_on), int(n_off)
        self.reset()

    def reset(self):
        self.state = False
        self._pos = 0
        self._neg = 0

    def update(self, decision: bool):
        """Returns (state, turned_on_now)."""
        if decision:
            self._pos += 1
            self._neg = 0
        else:
            self._neg += 1
            self._pos = 0
        turned_on = False
        if not self.state and self._pos >= self.n_on:
            self.state, turned_on = True, True
        elif self.state and self._neg >= self.n_off:
            self.state = False
        return self.state, turned_on


def alarm_sequence(decisions, n_on, n_off):
    logic = AlarmLogic(n_on, n_off)
    states = np.zeros(len(decisions), dtype=bool)
    onsets = np.zeros(len(decisions), dtype=bool)
    for i, d in enumerate(decisions):
        states[i], onsets[i] = logic.update(bool(d))
    return states, onsets
