"""Evaluate the diagnosis stage on TEST and STRESS, the way it is shown online: only for
faulty runs the detector caught, using the DiagnosisTracker output at the detection
window and 2 s later. Window-level metrics on all label-1 windows are reported too.
Requires the detector results (scripts.evaluate) in --results.

  python -m scripts.evaluate_diagnosis --data data/standard --model models/standard --results results/standard
Output: <results>/diagnosis_metrics.json, <results>/diagnosis_runs_<part>.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.dataset import load_manifest
from src.diagnosis import (DiagnosisTracker, attach_true_multipliers, diagnosis_targets, predict_windows,
                           window_metrics)
from src.features import FEATURE_NAMES
from src.training import attach_split, load_feature_table

LATER_S = 2.0


def run_diagnoses(bundle, feat, win, runs):
    """Replay the tracker over each detected faulty run."""
    p2, m = predict_windows(bundle, feat[FEATURE_NAMES].to_numpy())
    f = feat[["run_id", "t_end", "mult1", "mult2"]].assign(p2=p2, mult_hat=m)
    f = f.merge(win[["run_id", "t_end", "decision", "state"]], on=["run_id", "t_end"])
    rows = []
    for _, r in runs[runs["is_fault_run"] & runs["detected"]].iterrows():
        df = f[f["run_id"] == r["run_id"]].sort_values("t_end")
        tracker = DiagnosisTracker(bundle["max_windows"])
        out = {"run_id": r["run_id"], "scenario": r["scenario"], "fault_joint": int(r["fault_joint"]),
               "fault_kind": r["fault_kind"], "t_detect": r["t_detect"]}
        for w in df.itertuples():
            diag = tracker.update(w.decision, w.state, w.p2, w.mult_hat)
            for tag, t_eval in (("at_alarm", r["t_detect"]), ("later", r["t_detect"] + LATER_S)):
                if f"{tag}_t" not in out and w.t_end >= t_eval - 1e-6:
                    true_m = w.mult1 if r["fault_joint"] == 1 else w.mult2
                    out.update({f"{tag}_t": w.t_end, f"{tag}_joint": diag["joint"] if diag else np.nan,
                                f"{tag}_mult": diag["mult"] if diag else np.nan, f"{tag}_mult_true": true_m})
        rows.append(out)
    return pd.DataFrame(rows)


def summarize(d):
    out = {}
    for tag in ("at_alarm", "later"):
        x = d.dropna(subset=[f"{tag}_joint"]) if f"{tag}_joint" in d else d.iloc[0:0]
        err = (x[f"{tag}_mult"] - x[f"{tag}_mult_true"]).abs() if len(x) else pd.Series(dtype=float)
        out[tag] = {"n_runs": int(len(x)),
                    "joint_correct": int((x[f"{tag}_joint"] == x["fault_joint"]).sum()) if len(x) else 0,
                    "mult_median_abs_error": float(err.median()) if len(x) else float("nan"),
                    "mult_p90_abs_error": float(err.quantile(.9)) if len(x) else float("nan")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    res = Path(a.results)
    if (res / "diagnosis_metrics.json").exists() and not a.overwrite:
        raise SystemExit(f"{res / 'diagnosis_metrics.json'} exists; refusing to overwrite (use --overwrite).")
    bundle = joblib.load(Path(a.model) / "diagnosis_bundle.joblib")
    split = pd.read_csv(Path(a.data) / "split_manifest.csv")
    manifest = load_manifest(a.data)
    feat = attach_split(load_feature_table(a.data, bundle["window_s"]), split)
    metrics = {"note": "Diagnosis trained on train windows only; test and stress are first looked at here.",
               "later_s": LATER_S}
    for part in ("test", "stress"):
        fp = attach_true_multipliers(feat[feat["split"] == part], a.data)
        win = pd.read_csv(res / f"windows_{part}_rf.csv.gz")
        runs = pd.read_csv(res / f"runs_{part}_rf.csv")
        d = run_diagnoses(bundle, fp, win, runs)
        d.to_csv(res / f"diagnosis_runs_{part}.csv", index=False)
        metrics[part] = {"windows": window_metrics(bundle, diagnosis_targets(fp, manifest)), "runs": summarize(d)}
        by = {}
        for scen, g in d.groupby("scenario"):
            by[scen] = summarize(g)
        metrics[part]["by_scenario"] = by
        s = metrics[part]
        print(f"[{part}] windows: joint acc={s['windows']['joint_accuracy']:.3f} mult MAE={s['windows']['mult_mae']:.2f}")
        for tag in ("at_alarm", "later"):
            r = s["runs"][tag]
            print(f"   {tag:8s}: joint {r['joint_correct']}/{r['n_runs']}  mult median|err|={r['mult_median_abs_error']:.2f} "
                  f"p90={r['mult_p90_abs_error']:.2f}")
    with open(res / "diagnosis_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print("saved", res / "diagnosis_metrics.json")


if __name__ == "__main__":
    main()
