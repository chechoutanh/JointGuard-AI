"""Does the physics-residual detector carry over to a 7-DOF KUKA LBR iiwa? (docs/KUKA_EXPERIMENT.md)

  python -m scripts.kuka_experiment [--processes 8] [--overwrite]

The residual method is frozen from the 2-DOF experiment (acceleration stencil K, prior weight,
moving threshold): nothing about it is re-tuned. Only the alarm operating point (window,
threshold, n_on) is chosen, on the KUKA validation split, with the same criterion as before:
max window F1 with at most 0.5 false alarms per healthy minute.

  val     half of the in-domain groups: operating point of the rule and the baseline
  test    other half: same operating range
  stress  held out, never used for any choice: faster, heavier payload, half point-to-point

Detectors: physics rule  score = max_j fric_ratio_j  (7 joints)
           error baseline score = max_j e_rms_j / ref_j, ref = 95 % quantile on healthy val windows
Outputs: data/kuka/manifest.csv, failed_runs.csv, features/ (git-ignored);
         results/kuka/summary.csv, per_joint.csv, runs_*.csv, metrics.json
"""
import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import kuka
from src.config import PROJECT_ROOT, ensure_new_dir, load_config
from src.simulator import SimulationDivergedError
from src.training import evaluate_detector, select_operating_point

CFG_PATH = PROJECT_ROOT / "configs" / "kuka.json"


def _work(spec):
    cfg = load_config(CFG_PATH)
    t0 = time.time()
    try:
        sig = kuka.simulate(spec, cfg)
    except SimulationDivergedError as e:
        return spec, {"error": str(e)}, None
    feats = {}
    for w in cfg["features"]["window_s_options"]:
        f = kuka.run_features(sig, spec["payload_declared"], cfg, w)
        f.insert(0, "run_id", spec["run_id"])
        feats[w] = f
    return spec, {"t_label": sig["t_label"], "saturated_fraction": sig["saturated_fraction"],
                  "elapsed": time.time() - t0}, feats


def split_groups(manifest, seed):
    """In-domain groups -> val/test alternately within each scenario (shuffled); stress -> stress."""
    rng = np.random.default_rng(seed)
    out = {}
    ind = manifest[manifest["subset"] == "indomain"]
    for _, g in ind.groupby("scenario"):
        groups = list(dict.fromkeys(g["group_id"]))
        rng.shuffle(groups)
        out.update({gid: ("val" if i % 2 == 0 else "test") for i, gid in enumerate(groups)})
    return manifest["group_id"].map(out).fillna("stress")


def rule_scores(df):
    return df[kuka.RATIO_NAMES].to_numpy().max(axis=1)


def baseline_scores(df, ref):
    return (df[[f"j{j}_e_rms" for j in range(1, kuka.N + 1)]].to_numpy() / np.asarray(ref)).max(axis=1)


def _row(part, det, s):
    return {"part": part, "detector": det, "detected": f"{s['n_detected']}/{s['n_fault_runs']}",
            "fa_per_healthy_min": round(s["false_alarms_per_healthy_min"], 3),
            "healthy_runs_with_fa": f"{s['healthy_runs_with_false_alarm']}/{s['n_healthy_runs']}",
            "delay_median_s": round(s["delay_median"], 2), "window_f1": round(s["window_alarm_state"]["f1"], 3)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(CFG_PATH)
    data = ensure_new_dir(PROJECT_ROOT / "data" / "kuka", a.overwrite)
    out = ensure_new_dir(PROJECT_ROOT / "results" / "kuka", a.overwrite)
    specs = kuka.sample_specs(cfg, "indomain") + kuka.sample_specs(cfg, "stress")
    print(f"simulating {len(specs)} runs", flush=True)
    rows, failed, feats = [], [], {w: [] for w in cfg["features"]["window_s_options"]}
    t0 = time.time()
    with Pool(a.processes) as pool:
        for i, (spec, info, f) in enumerate(pool.imap_unordered(_work, specs), 1):
            meta = {k: spec[k] for k in ("run_id", "group_id", "subset", "scenario", "fault_kind", "severity",
                                         "fault_joint", "duration", "payload", "payload_declared", "traj_kind",
                                         "t_onset", "mult_final", "ramp", "noise_seed")}
            if f is None:
                failed.append({**meta, **info})
            else:
                rows.append({**meta, **info})
                for w, df in f.items():
                    feats[w].append(df)
            if i % 20 == 0:
                print(f"  {i}/{len(specs)} ({time.time() - t0:.0f} s)", flush=True)
    manifest = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    manifest["split"] = split_groups(manifest, cfg["seed"])
    manifest.to_csv(data / "manifest.csv", index=False)
    pd.DataFrame(failed).to_csv(data / "failed_runs.csv", index=False)
    (data / "features").mkdir()
    tables = {}
    for w, parts in feats.items():
        t = pd.concat(parts).merge(manifest[["run_id", "split"]], on="run_id")
        t = t.sort_values(["run_id", "t_end"]).reset_index(drop=True)
        t.to_csv(data / "features" / f"w{w}.csv.gz", index=False)
        tables[w] = t
    print(f"{len(manifest)} runs ok, {len(failed)} failed", flush=True)

    tc, stride = cfg["training"], cfg["features"]["stride_s"]
    metrics, summary = {"note": __doc__, "n_runs": len(manifest), "n_failed": len(failed)}, []
    choices = {}
    for det in ("physics_rule", "error_baseline"):
        best = None
        for w, t in tables.items():
            va = t[t["split"] == "val"].reset_index(drop=True)
            if det == "physics_rule":
                ref, scores, grid = None, rule_scores(va), tc["rule_threshold_grid"]
            else:
                healthy = va[va["label"] == 0]
                ref = [float(np.quantile(healthy[f"j{j}_e_rms"], tc["baseline_quantile"])) for j in range(1, kuka.N + 1)]
                scores, grid = baseline_scores(va, ref), tc["baseline_scale_grid"]
            op, _ = select_operating_point(va, scores, grid, tc["n_on_grid"], tc["n_off"], manifest, stride,
                                           tc["max_false_alarms_per_min"])
            key = (op["fa_per_min"] <= tc["max_false_alarms_per_min"], op["f1"])
            if best is None or key > best["key"]:
                best = {"key": key, "window_s": w, "ref": ref, "op": op}
        choices[det] = best
        metrics[det] = {"window_s": best["window_s"], "ref": best["ref"], "operating_point": best["op"]}
        print(f"{det}: window={best['window_s']} op={best['op']}", flush=True)
    per_joint = []
    for part in ("val", "test", "stress"):
        for det, best in choices.items():
            t = tables[best["window_s"]]
            d = t[t["split"] == part].reset_index(drop=True)
            scores = rule_scores(d) if det == "physics_rule" else baseline_scores(d, best["ref"])
            _, runs, s = evaluate_detector(d, scores, best["op"]["threshold"], int(best["op"]["n_on"]), tc["n_off"],
                                           manifest, stride)
            metrics[det][part] = s
            summary.append(_row(part, det, s))
            runs.to_csv(out / f"runs_{part}_{det}.csv", index=False)
            f = runs[runs["is_fault_run"]]
            for j, g in f.groupby("fault_joint"):
                per_joint.append({"part": part, "detector": det, "joint": int(j),
                                  "detected": f"{int(g['detected'].sum())}/{len(g)}",
                                  "delay_median_s": round(float(g["delay"].median()), 2)
                                  if g["detected"].any() else math.nan})
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(per_joint).to_csv(out / "per_joint.csv", index=False)
    metrics["saturated_fraction_max"] = float(manifest["saturated_fraction"].max())
    with open(out / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print("\n" + pd.DataFrame(summary).to_string(index=False))
    print("\n" + pd.DataFrame(per_joint).to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
