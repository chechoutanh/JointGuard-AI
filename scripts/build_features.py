"""Compute window features for every run and every window length option.

  python -m scripts.build_features --data data/standard
Output: data/<name>/features/w<window>.csv.gz  (meta columns + FEATURE_NAMES)
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.dataset import load_manifest, load_signals, load_truth
from src.features import FEATURE_NAMES, run_features, window_labels, window_sizes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    fc, fs = cfg["features"], cfg["timing"]["f_sample"]
    out_dir = Path(a.data) / "features"
    out_dir.mkdir(exist_ok=True)
    manifest = load_manifest(a.data)
    for window_s in fc["window_s_options"]:
        out = out_dir / f"w{window_s:g}.csv.gz"
        if out.exists() and not a.overwrite:
            raise SystemExit(f"{out} exists; refusing to overwrite (use --overwrite).")
        wn, _ = window_sizes(window_s, fc["stride_s"], fs)
        parts = []
        for run_id in manifest["run_id"]:
            sig, truth = load_signals(a.data, run_id), load_truth(a.data, run_id)
            X, ends = run_features(sig, window_s, fc["stride_s"], fs, fc["monitor_start_s"])
            y, mixed = window_labels(truth, ends, wn)
            df = pd.DataFrame(X, columns=FEATURE_NAMES)
            df.insert(0, "mixed", mixed)
            df.insert(0, "label", y)
            df.insert(0, "t_end", sig["t"].to_numpy()[ends])
            df.insert(0, "run_id", run_id)
            parts.append(df)
        feat = pd.concat(parts, ignore_index=True)
        feat.to_csv(out, index=False, float_format="%.10g")
        print(f"saved {out}: {len(feat)} windows, {int(feat['label'].sum())} anomalous, {int(feat['mixed'].sum())} mixed")


if __name__ == "__main__":
    main()
