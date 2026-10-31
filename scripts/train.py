"""Train the Random Forest and fit the baseline using train/validation only.

  python -m scripts.train --data data/standard --tag standard
Outputs models/<tag>/: model_bundle.joblib, baseline.json, training_report.json, val_operating_points*.csv
"""
import argparse
import json
from pathlib import Path

import joblib

from src.config import load_config, ensure_new_dir, PROJECT_ROOT
from src.dataset import load_manifest
from src.splits import check_split
import pandas as pd
from src.training import train_all


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    split_path = Path(a.data) / "split_manifest.csv"
    if not split_path.exists():
        raise SystemExit("split_manifest.csv not found: run `python -m scripts.make_splits` first.")
    split = pd.read_csv(split_path)
    check_split(split)
    out = ensure_new_dir(PROJECT_ROOT / "models" / a.tag, a.overwrite)
    bundle, baseline, report, tab_rf, tab_base = train_all(a.data, split, load_manifest(a.data), cfg)
    bundle["data_dir"] = str(a.data)
    joblib.dump(bundle, out / "model_bundle.joblib")
    with open(out / "baseline.json", "w") as f:
        json.dump(baseline, f, indent=2)
    card = {k: v for k, v in bundle.items() if k != "model"}
    with open(out / "model_card.json", "w") as f:
        json.dump(card, f, indent=2, default=str)
    with open(out / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    tab_rf.to_csv(out / "val_operating_points_rf.csv", index=False)
    tab_base.to_csv(out / "val_operating_points_baseline.csv", index=False)
    print(f"saved model to {out}")


if __name__ == "__main__":
    main()
