"""Create the run-level split manifest BEFORE training.

  python -m scripts.make_splits --data data/standard
"""
import argparse
from pathlib import Path

from src.config import load_config
from src.dataset import load_manifest
from src.splits import make_split, check_split


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.data) / "split_manifest.csv"
    if out.exists() and not a.overwrite:
        raise SystemExit(f"{out} exists; refusing to overwrite (use --overwrite).")
    split = make_split(load_manifest(a.data), load_config(a.config))
    check_split(split)
    split.to_csv(out, index=False)
    print(f"saved {out}")
    print(split.pivot_table(index="scenario", columns="split", values="run_id", aggfunc="count", fill_value=0).to_string())


if __name__ == "__main__":
    main()
