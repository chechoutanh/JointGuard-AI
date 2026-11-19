"""Generate a dataset headlessly (no Pygame).

  python -m scripts.generate_dataset --mode smoke
  python -m scripts.generate_dataset --mode standard --name standard
  python -m scripts.generate_dataset --mode standard --name stress2 --subsets stress --seed 20261120
"""
import argparse

from src.config import load_config
from src.dataset import generate_dataset


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["smoke", "standard"], default="smoke")
    ap.add_argument("--name", default=None, help="output folder under data/ (default = mode)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--subsets", nargs="+", choices=["indomain", "stress"], default=["indomain", "stress"])
    ap.add_argument("--seed", type=int, default=None, help="override the config seed (e.g. a fresh stress set)")
    ap.add_argument("--overwrite", action="store_true", help="explicitly replace an existing dataset")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    out, manifest, failed = generate_dataset(a.name or a.mode, a.mode, cfg, a.overwrite, a.processes,
                                             subsets=a.subsets)
    print(f"Saved {len(manifest)} runs to {out} ({len(failed)} failed, see failed_runs.csv)")
    print(manifest.groupby(["subset", "scenario"]).size().to_string())
    print(f"max saturated fraction: {manifest['saturated_fraction'].max():.4f}")


if __name__ == "__main__":
    main()
