"""Evaluate the trained Random Forest and the baseline on the independent TEST split and,
separately, on the held-out STRESS split. Nothing here feeds back into training.

  python -m scripts.evaluate --data data/standard --model models/standard --tag standard
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ensure_new_dir, PROJECT_ROOT
from src.dataset import load_manifest
from src.features import FEATURE_NAMES
from src.training import attach_split, baseline_scores, evaluate_detector, load_feature_table, rf_scores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    bundle = joblib.load(Path(a.model) / "model_bundle.joblib")
    baseline = json.load(open(Path(a.model) / "baseline.json"))
    split = pd.read_csv(Path(a.data) / "split_manifest.csv")
    manifest = load_manifest(a.data)
    out = ensure_new_dir(PROJECT_ROOT / "results" / a.tag, a.overwrite)
    feat_rf = attach_split(load_feature_table(a.data, bundle["window_s"]), split)
    feat_bl = attach_split(load_feature_table(a.data, baseline["window_s"]), split)
    metrics = {"model_dir": str(a.model), "data_dir": str(a.data),
               "rf": {k: bundle[k] for k in ("window_s", "threshold", "n_on", "n_off", "rf_params")},
               "baseline": baseline}
    for part in ("val", "test", "stress"):
        metrics[part] = {}
        for name, feat, scorer, p in (("rf", feat_rf, lambda d: rf_scores(bundle["model"], d), bundle),
                                      ("baseline", feat_bl, lambda d: baseline_scores(baseline, d), baseline)):
            d = feat[feat["split"] == part].reset_index(drop=True)
            if not len(d):
                continue
            w, runs, summ = evaluate_detector(d, scorer(d), p["threshold"], p["n_on"], p["n_off"],
                                              manifest, bundle["stride_s"])
            metrics[part][name] = summ
            w.to_csv(out / f"windows_{part}_{name}.csv.gz", index=False, float_format="%.6g")
            runs.to_csv(out / f"runs_{part}_{name}.csv", index=False)
            s = summ
            print(f"[{part:6s}] {name:8s} F1={s['window_alarm_state']['f1']:.3f} "
                  f"P={s['window_alarm_state']['precision']:.3f} R={s['window_alarm_state']['recall']:.3f} "
                  f"FA/min={s['false_alarms_per_healthy_min']:.3f} det={s['n_detected']}/{s['n_fault_runs']} "
                  f"delay_med={s['delay_median']:.2f}s")
    metrics["note"] = ("'val' is shown only for reference (used for selection); 'test' and 'stress' "
                       "were not used for any training or tuning decision.")
    imp = pd.DataFrame({"feature": FEATURE_NAMES, "importance": bundle["model"].feature_importances_})
    imp.sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"saved results to {out}")


if __name__ == "__main__":
    main()
