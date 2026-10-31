"""Train the diagnosis stage (which joint, how much friction) on TRAIN windows only.
Hyper-parameters are copied from the trained detector; nothing is tuned here.
Validation metrics are reported for reference.

  python -m scripts.train_diagnosis --data data/standard --model models/standard
Output: models/<m>/diagnosis_bundle.joblib, models/<m>/diagnosis_report.json
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config
from src.dataset import load_manifest
from src.diagnosis import attach_true_multipliers, diagnosis_targets, train_diagnoser, window_metrics
from src.training import attach_split, load_feature_table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.model) / "diagnosis_bundle.joblib"
    if out.exists() and not a.overwrite:
        raise SystemExit(f"{out} exists; refusing to overwrite (use --overwrite).")
    cfg = load_config()
    det = joblib.load(Path(a.model) / "model_bundle.joblib")
    split = pd.read_csv(Path(a.data) / "split_manifest.csv")
    manifest = load_manifest(a.data)
    feat = attach_split(load_feature_table(a.data, det["window_s"]), split)
    feat = feat[feat["split"].isin(["train", "val"])]
    d = diagnosis_targets(attach_true_multipliers(feat, a.data), manifest)
    tr, va = d[d["split"] == "train"], d[d["split"] == "val"]
    print(f"train windows: {len(tr)} (joint 1: {(tr['fault_joint'] == 1).sum()}, joint 2: {(tr['fault_joint'] == 2).sum()})")
    clf, reg = train_diagnoser(tr, det["rf_params"], cfg["seed"])
    bundle = {"joint_model": clf, "mult_model": reg, "window_s": det["window_s"],
              "feature_names": det["feature_names"], "rf_params": det["rf_params"], "seed": cfg["seed"],
              "max_windows": 20, "versions": det["versions"],
              "note": "Diagnosis is shown only while the detector's alarm is on. mult = friction "
                      "multiplier of the diagnosed joint relative to its healthy value."}
    report = {"train": window_metrics(bundle, tr), "val": window_metrics(bundle, va),
              "train_mult_range": [float(tr["mult_true"].min()), float(tr["mult_true"].max())]}
    print("val:", report["val"])
    joblib.dump(bundle, out, compress=3)   # ~18 MB instead of ~60 MB; predictions unchanged
    with open(Path(a.model) / "diagnosis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
