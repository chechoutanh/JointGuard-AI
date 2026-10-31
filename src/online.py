"""Online (streaming) detector used by the Pygame demo. It calls exactly the same
`window_features` function as the offline pipeline, on the same window grid, and the
same AlarmLogic. If a diagnosis bundle exists, it also reports which joint and how much
friction while the alarm is on (src/diagnosis.py). If no trained model exists it raises ModelNotFoundError instead of
producing any fake prediction.
"""
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from .alarm import AlarmLogic
from .diagnosis import DiagnosisTracker, predict_windows
from .features import INPUT_COLUMNS, window_features, window_sizes


class ModelNotFoundError(FileNotFoundError):
    pass


def load_bundle(model_dir):
    path = Path(model_dir) / "model_bundle.joblib"
    if not path.exists():
        raise ModelNotFoundError(
            f"Model file not found: {path}\nTrain it first, e.g.:\n"
            f"  python -m scripts.run_pipeline --mode standard\n"
            f"or step by step: generate_dataset -> make_splits -> build_features -> train")
    return joblib.load(path)


def load_diagnosis(model_dir):
    """Optional diagnosis stage; None if it has not been trained."""
    path = Path(model_dir) / "diagnosis_bundle.joblib"
    return joblib.load(path) if path.exists() else None


class OnlineDetector:
    def __init__(self, bundle, diagnosis=None):
        self.bundle = bundle
        self.diagnosis = diagnosis
        if diagnosis:
            for m in (diagnosis["joint_model"], diagnosis["mult_model"]):
                m.n_jobs = 1
            self.tracker = DiagnosisTracker(diagnosis["max_windows"])
        self.model = bundle["model"]
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1   # single-sample inference: thread start-up costs more than it saves
        self.f_sample = bundle["f_sample"]
        self.window_n, self.stride_n = window_sizes(bundle["window_s"], bundle["stride_s"], self.f_sample)
        self.monitor_start_s = bundle["monitor_start_s"]
        self.threshold = bundle["threshold"]
        self.alarm = AlarmLogic(bundle["n_on"], bundle["n_off"])
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=self.window_n)
        self.n = 0
        self.alarm.reset()
        if self.diagnosis:
            self.tracker.reset()
        self.diag = None
        self.score = None
        self.last_features = None
        self.status = "WARMING UP"

    def push(self, sample):
        """sample: mapping or sequence with INPUT_COLUMNS values for one sample.
        Returns dict(updated, score, decision, alarm, turned_on, status)."""
        if hasattr(sample, "keys"):
            sample = [sample[c] for c in INPUT_COLUMNS]
        self.buf.append(np.asarray(sample, dtype=float))
        e = self.n            # 0-based index of this sample within the run
        self.n += 1
        res = {"updated": False, "score": self.score, "alarm": self.alarm.state,
               "turned_on": False, "status": self.status, "diagnosis": self.diag}
        if self.n < self.window_n or e / self.f_sample < self.monitor_start_s - 1e-9:
            self.status = res["status"] = "WARMING UP"
            return res
        if self.n % self.stride_n != 0:
            return res
        x = window_features(np.vstack(self.buf), 1.0 / self.f_sample)
        self.last_features = x
        self.score = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        decision = self.score >= self.threshold
        state, turned_on = self.alarm.update(decision)
        self.status = "ANOMALY ALARM" if state else ("SUSPICIOUS" if decision else "NORMAL")
        if self.diagnosis:
            p2, mult = predict_windows(self.diagnosis, x)
            self.diag = self.tracker.update(decision, state, p2[0], mult[0])
        res.update(updated=True, score=self.score, decision=decision, alarm=state,
                   turned_on=turned_on, status=self.status, diagnosis=self.diag)
        return res
