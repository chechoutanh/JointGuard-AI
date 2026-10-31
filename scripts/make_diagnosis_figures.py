"""Figures for the diagnosis stage and the load-vs-friction signature.

  python -m scripts.make_diagnosis_figures --results results/standard --demo web/demo_runs.json
Output: <results>/figures/diagnosis_estimates.png, <results>/figures/signature_load_vs_friction.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def diagnosis_figure(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, part in zip(axes, ("test", "stress")):
        d = pd.read_csv(results / f"diagnosis_runs_{part}.csv").dropna(subset=["at_alarm_joint"])
        ok = d["at_alarm_joint"] == d["fault_joint"]
        ax.plot([1, 4.5], [1, 4.5], color="0.6", lw=1, ls="--", label="perfect estimate")
        ax.scatter(d.loc[ok, "at_alarm_mult_true"], d.loc[ok, "at_alarm_mult"], s=28, color="#2a6cb0",
                   label=f"right joint ({ok.sum()})")
        ax.scatter(d.loc[~ok, "at_alarm_mult_true"], d.loc[~ok, "at_alarm_mult"], s=40, marker="x",
                   color="#c53030", label=f"wrong joint ({(~ok).sum()})")
        ax.set_title(f"{part.upper()} - diagnosis when the alarm turns on")
        ax.set_xlabel("true friction multiplier")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("estimated friction multiplier")
    fig.tight_layout()
    fig.savefig(out / "diagnosis_estimates.png", dpi=160)
    plt.close(fig)


def signature_figure(demo, out):
    D = json.loads(Path(demo).read_text())
    sc = {s["id"]: s for s in D["scenarios"]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, j in zip(axes, (1, 2)):
        tau, edir = (5, 3) if j == 1 else (6, 4)
        groups = [("healthy, 0.5 kg", sc["healthy"]["windows"], "#2a8452"),
                  ("healthy, 2.0 kg", sc["heavy"]["windows"], "#2a6cb0"),
                  (f"friction x3 on joint {j}", [w for w in sc[f"abrupt_j{j}"]["windows"] if w[0] >= 13], "#c53030")]
        for name, W, col in groups:
            ax.scatter([w[tau] for w in W[::2]], [w[edir] * 1000 for w in W[::2]], s=8, alpha=.6, color=col, label=name)
        ax.set_xlabel(f"joint {j} torque RMS [N m]")
        ax.set_ylabel("tracking error along motion [mrad]")
        ax.set_title(f"Joint {j}")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(out / "signature_load_vs_friction.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/standard")
    ap.add_argument("--demo", default="web/demo_runs.json")
    a = ap.parse_args(argv)
    res = Path(a.results)
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    diagnosis_figure(res, out)
    signature_figure(a.demo, out)
    print("saved", out / "diagnosis_estimates.png", "and", out / "signature_load_vs_friction.png")


if __name__ == "__main__":
    main()
