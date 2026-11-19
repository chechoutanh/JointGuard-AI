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
from .config import load_config
from .features import FEATURE_NAMES, INPUT_COLUMNS, window_features, window_sizes
from .residual import RESIDUAL_FEATURE_NAMES, history_samples, window_residual_from_block


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
    def __init__(self, bundle, diagnosis=None, payload_declared=0.0, cfg=None):
        """payload_declared and cfg are only used by models with physics-residual features."""
        self.bundle = bundle
        self.diagnosis = diagnosis
        self.feature_names = list(bundle.get("feature_names", FEATURE_NAMES))
        self.residual = bool(set(self.feature_names) & set(RESIDUAL_FEATURE_NAMES))
        self.cfg = (cfg or load_config()) if self.residual else cfg
        self.payload_declared = float(payload_declared)
        # residual models without a trained diagnosis stage report the friction ratio directly
        self.physics_diag = self.residual and not diagnosis
        if self.physics_diag:
            self.tracker = DiagnosisTracker(20)
        if diagnosis:
            for m in (diagnosis["joint_model"], diagnosis["mult_model"]):
                m.n_jobs = 1
            self.tracker = DiagnosisTracker(diagnosis["max_windows"])
        self.model = bundle["model"]
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1   # single-sample inference: thread start-up costs more than it saves
        self.f_sample = bundle["f_sample"]
        self.window_n, self.stride_n = window_sizes(bundle["window_s"], bundle["stride_s"], self.f_sample)
        self.history_n = history_samples() if self.residual else 0
        self.monitor_start_s = bundle["monitor_start_s"]
        self.threshold = bundle["threshold"]
        self.alarm = AlarmLogic(bundle["n_on"], bundle["n_off"])
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=self.window_n + self.history_n)
        self.n = 0
        self.alarm.reset()
        if self.diagnosis or self.physics_diag:
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
        if len(self.buf) < self.buf.maxlen or e / self.f_sample < self.monitor_start_s - 1e-9:
            self.status = res["status"] = "WARMING UP"
            return res
        if self.n % self.stride_n != 0:
            return res
        block = np.vstack(self.buf)
        x_base = window_features(block[-self.window_n:], 1.0 / self.f_sample)
        values = dict(zip(FEATURE_NAMES, x_base))
        if self.residual:
            values.update(zip(RESIDUAL_FEATURE_NAMES,
                              window_residual_from_block(block, self.payload_declared, self.cfg)))
        x = np.array([values[n] for n in self.feature_names])
        self.last_features = x
        self.score = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        decision = self.score >= self.threshold
        state, turned_on = self.alarm.update(decision)
        self.status = "ANOMALY ALARM" if state else ("SUSPICIOUS" if decision else "NORMAL")
        if self.diagnosis:
            p2, mult = predict_windows(self.diagnosis, x_base)  # diagnosis models use the original features
            self.diag = self.tracker.update(decision, state, p2[0], mult[0])
        elif self.physics_diag:
            r1, r2 = values["j1_fric_ratio"], values["j2_fric_ratio"]
            self.diag = self.tracker.update(decision, state, 1.0 if r2 > r1 else 0.0, max(r1, r2))
            if self.diag:
                self.diag["source"] = "physics"
        res.update(updated=True, score=self.score, decision=decision, alarm=state,
                   turned_on=turned_on, status=self.status, diagnosis=self.diag)
        return res
