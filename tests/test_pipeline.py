"""Data / ML integrity checks: online-offline consistency, no leakage, alarm/event rules."""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.alarm import alarm_sequence
from src.config import load_config
from src.dataset import simulate_spec, spec_from_manifest_row
from src.evaluation import per_run_events
from src.features import FEATURE_NAMES, INPUT_COLUMNS, run_features, window_sizes
from src.online import OnlineDetector, load_bundle, ModelNotFoundError
from src.scenarios import sample_run_specs
from src.simulator import SIGNAL_COLUMNS
from src.splits import make_split, check_split

CFG = load_config()


@pytest.fixture(scope="module")
def one_run():
    spec = [s for s in sample_run_specs(CFG, "smoke", "indomain") if s["scenario"] == "severe_j2"][0]
    spec, sig, truth, info = simulate_spec((spec, CFG))
    return spec, pd.DataFrame(sig, columns=SIGNAL_COLUMNS), truth


def _toy_bundle(window_s=0.5):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURE_NAMES)))
    y = (X[:, 5] + X[:, 17] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    return {"model": rf, "feature_names": FEATURE_NAMES, "window_s": window_s, "stride_s": 0.1,
            "monitor_start_s": 2.0, "f_sample": 100, "threshold": 0.5, "n_on": 3, "n_off": 5}


@pytest.mark.parametrize("window_s", [0.5, 1.0])
def test_online_offline_features_and_predictions_identical(one_run, window_s):
    _, sig, _ = one_run
    bundle = _toy_bundle(window_s)
    X_off, ends = run_features(sig, window_s, 0.1, 100, 2.0)
    det = OnlineDetector(bundle)
    X_on, scores_on, idx_on = [], [], []
    for i, row in enumerate(sig[INPUT_COLUMNS].to_numpy()):
        r = det.push(row)
        if r["updated"]:
            X_on.append(det.last_features); scores_on.append(r["score"]); idx_on.append(i)
    assert idx_on == ends
    np.testing.assert_array_equal(np.array(X_on), X_off)
    np.testing.assert_array_equal(np.array(scores_on), bundle["model"].predict_proba(X_off)[:, 1])


def test_features_do_not_use_time_or_metadata(one_run):
    _, sig, _ = one_run
    forbidden = {"t", "run_id", "label", "scenario", "fric", "payload", "true", "onset", "seed"}
    for name in FEATURE_NAMES + INPUT_COLUMNS:
        assert not any(f == name or f in name.split("_") for f in forbidden), name
    # shifting absolute time must not change the features
    shifted = sig.copy(); shifted["t"] += 1000.0
    X1, _ = run_features(sig, 0.5, 0.1, 100, 2.0)
    X2, _ = run_features(shifted, 0.5, 0.1, 100, 2.0)
    np.testing.assert_array_equal(X1, X2)


def test_features_are_causal(one_run):
    """Changing future samples must not change features of earlier windows."""
    _, sig, _ = one_run
    X1, ends = run_features(sig, 0.5, 0.1, 100, 2.0)
    mod = sig.copy()
    cut = ends[40] + 1
    mod.loc[cut:, "tau1_applied"] += 50.0
    X2, _ = run_features(mod, 0.5, 0.1, 100, 2.0)
    np.testing.assert_array_equal(X1[:41], X2[:41])
    assert not np.allclose(X1[41:], X2[41:])


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python -m scripts"):
        load_bundle(tmp_path)


def test_split_keeps_groups_together():
    specs = sample_run_specs(CFG, "standard", "indomain") + sample_run_specs(CFG, "standard", "stress")
    man = pd.DataFrame([{k: s[k] for k in ("run_id", "group_id", "subset", "scenario")} for s in specs])
    split = make_split(man, CFG)
    assert check_split(split)
    assert set(split.loc[man["subset"] == "stress", "split"]) == {"stress"}
    for scen, df in split[split["split"] != "stress"].groupby("scenario"):
        assert set(df["split"]) == {"train", "val", "test"}, scen


def test_noise_level_independent_of_scenario():
    specs = pd.DataFrame(sample_run_specs(CFG, "standard", "indomain"))
    lo, hi = CFG["sensor"]["noise_q_range"]
    for _, df in specs.groupby("fault_kind"):
        assert df["noise_q"].between(lo, hi).all()
    # same sampling distribution: mean within a loose band for healthy vs faulty
    h = specs[specs.fault_kind == "none"]["noise_q"].mean()
    f = specs[specs.fault_kind != "none"]["noise_q"].mean()
    assert abs(h - f) < 0.15 * (hi - lo)


def test_fault_onsets_vary():
    specs = pd.DataFrame(sample_run_specs(CFG, "standard", "indomain"))
    on = specs.loc[specs.fault_kind == "abrupt", "t_onset"]
    assert on.nunique() == len(on) and on.std() > 1.0


def test_dataset_is_reproducible_from_manifest_spec(one_run):
    spec, sig, _ = one_run
    row = pd.Series({**{k: v for k, v in spec.items() if k != "traj"}, "traj_json": __import__("json").dumps(spec["traj"])})
    spec2 = spec_from_manifest_row(row)
    _, sig2, _, _ = simulate_spec((spec2, CFG))
    np.testing.assert_array_equal(sig.to_numpy(), sig2)


def test_alarm_logic_counts_transitions():
    d = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
    st, on = alarm_sequence(d, n_on=3, n_off=5)
    assert list(np.where(on)[0]) == [3, 12]
    assert st[3] and st[8] and not st[9]


def _events(times, states, onsets, scen="severe_j1", kind="abrupt", t_onset=5.0, t_label=5.0):
    win = pd.DataFrame({"run_id": "X", "t_end": times, "label": (times >= t_label).astype(int),
                        "state": states, "onset": onsets})
    man = pd.DataFrame([{"run_id": "X", "scenario": scen, "severity": "severe", "fault_joint": 1,
                         "fault_kind": kind, "t_onset": t_onset, "t_label": t_label}])
    return per_run_events(win, man, 0.1).iloc[0]


def test_pre_existing_false_alarm_is_not_a_detection():
    t = np.round(np.arange(2.0, 10.0, 0.1), 3)
    states = t >= 4.0                       # alarm turned on before the fault and never turned off
    onsets = np.isclose(t, 4.0)
    r = _events(t, states, onsets)
    assert r["false_alarms"] == 1 and not r["detected"]


def test_delay_measured_from_defined_fault_time():
    t = np.round(np.arange(2.0, 10.0, 0.1), 3)
    states = t >= 6.3
    onsets = np.isclose(t, 6.3)
    r = _events(t, states, onsets)
    assert r["detected"] and r["delay"] == pytest.approx(1.3) and r["false_alarms"] == 0


def test_gradual_early_alarm_counts_with_zero_delay():
    t = np.round(np.arange(2.0, 14.0, 0.1), 3)
    states = t >= 7.0
    onsets = np.isclose(t, 7.0)
    r = _events(t, states, onsets, scen="gradual_j1", kind="gradual", t_onset=5.0, t_label=8.0)
    assert r["detected"] and r["delay"] == 0.0 and r["detect_type"] == "early_active"
    assert r["false_alarms"] == 0 and r["early_alarms"] == 1


def test_diagnosis_only_while_alarm_and_cleared_after():
    from src.diagnosis import DiagnosisTracker
    tr = DiagnosisTracker(max_windows=3)
    assert tr.update(True, False, 0.9, 3.0) is None          # positive but alarm not on yet
    d = tr.update(True, True, 0.7, 2.0)                      # alarm on: mean of both positives
    assert d["joint"] == 2 and d["mult"] == pytest.approx(2.5) and d["joint_confidence"] == pytest.approx(0.8)
    tr.update(True, True, 0.1, 1.6); d = tr.update(True, True, 0.1, 1.6)   # window of 3 slides
    assert d["n_windows"] == 3 and d["joint"] == 1
    assert tr.update(False, False, 0.9, 4.0) is None and not tr.buf     # alarm off -> cleared
