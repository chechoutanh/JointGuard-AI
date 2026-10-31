"""Create all figures from saved evaluation results.

  python -m scripts.make_figures --results results/standard
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src import plots


def pick_examples(results_dir):
    """Deterministic choice of example runs from the TEST split (first by run_id)."""
    r = pd.read_csv(Path(results_dir) / "runs_test_rf.csv").sort_values("run_id")
    ex = []
    for scen in ("healthy_speed", "mild_j2", "gradual_j1", "severe_j1"):
        s = r[r["scenario"] == scen]
        if len(s):
            ex.append(s.iloc[0]["run_id"])
    missed = r[r["is_fault_run"] & ~r["detected"]]
    if len(missed):
        ex.append(missed.iloc[0]["run_id"])
    fa = r[r["false_alarms"] > 0]
    if len(fa):
        ex.append(fa.iloc[0]["run_id"])
    return list(dict.fromkeys(ex))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args(argv)
    res = Path(a.results)
    metrics = json.load(open(res / "metrics.json"))
    out = res / "figures"
    out.mkdir(exist_ok=True)
    paths = [plots.confusion(metrics, out), plots.comparison(metrics, out), plots.by_scenario(metrics, out),
             plots.false_alarms_by_scenario(metrics, out), plots.delays(res, out), plots.importance(res, out)]
    for run_id in pick_examples(res):
        paths.append(plots.example_run(metrics["data_dir"], res, run_id, "test", out, metrics["rf"]["threshold"]))
    for p in paths:
        print("saved", p)


if __name__ == "__main__":
    main()
