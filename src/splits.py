"""Run-level train/val/test split (Stage D.1), created BEFORE any training.

* Splitting unit = group (base experiment); all replicates of a group share a split.
* Stratified by scenario type: inside each scenario the groups are shuffled with a fixed
  seed and allocated ~60/20/20.
* Stress-test runs (held-out conditions) always get split = "stress".
"""
import numpy as np
import pandas as pd


def make_split(manifest: pd.DataFrame, cfg) -> pd.DataFrame:
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 7]))
    fr = cfg["split"]
    assign = {}
    ind = manifest[manifest["subset"] == "indomain"]
    for scen, df in sorted(ind.groupby("scenario")):
        groups = sorted(df["group_id"].unique())
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, int(round(fr["test"] * n))) if n >= 3 else 0
        n_val = max(1, int(round(fr["val"] * n))) if n >= 2 else 0
        for i, g in enumerate(groups):
            assign[g] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    for g in manifest.loc[manifest["subset"] == "stress", "group_id"].unique():
        assign[g] = "stress"
    out = manifest[["run_id", "group_id", "subset", "scenario"]].copy()
    out["split"] = out["group_id"].map(assign)
    return out


def check_split(split: pd.DataFrame):
    """Every group must belong to exactly one split."""
    per_group = split.groupby("group_id")["split"].nunique()
    if (per_group > 1).any():
        raise AssertionError("group leakage across splits: " + ", ".join(per_group[per_group > 1].index))
    return True
