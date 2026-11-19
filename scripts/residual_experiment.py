"""Ablation: original features vs physics-residual features vs both (docs/RESIDUAL_EXPERIMENT.md).

  python -m scripts.residual_experiment --data data/standard --fresh data/stress2 data/stress3

For each feature set the SAME selection procedure as scripts.train runs on train/val of
data/standard only. Each model is then scored on:
  test     data/standard test split (inside the training range)
  stress   data/standard stress split -- used to DEVELOP the residual features (development set now)
  stress2  new seed; seen once, with the first (discarded) residual version -- secondary check
  stress3  new seed, generated after the method was frozen -- the real held-out check
A physics-only rule is scored too: score = max_j fric_ratio_j (no machine learning), alarm when
score >= threshold for n_on windows; window, threshold and n_on chosen on validation with the
same criterion as the Random Forest.
Outputs: models/residual_<set>/model_bundle.joblib, results/residual/metrics.json and summary.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, ensure_new_dir, PROJECT_ROOT
from src.dataset import load_manifest
from src.features import FEATURE_NAMES
from src.residual import RESIDUAL_FEATURE_NAMES
from src.splits import check_split
from src.training import (attach_split, baseline_scores, evaluate_detector, load_feature_table,
                          rf_scores, select_operating_point, train_all)

FEATURE_SETS = {"original": FEATURE_NAMES, "residual": RESIDUAL_FEATURE_NAMES,
                "both": FEATURE_NAMES + RESIDUAL_FEATURE_NAMES}


def _row(fs, part, det, s):
    return {"feature_set": fs, "part": part, "detector": det,
            "detected": f"{s['n_detected']}/{s['n_fault_runs']}",
            "fa_per_healthy_min": round(s["false_alarms_per_healthy_min"], 3),
            "healthy_runs_with_fa": f"{s['healthy_runs_with_false_alarm']}/{s['n_healthy_runs']}",
            "delay_median_s": round(s["delay_median"], 2), "window_f1": round(s["window_alarm_state"]["f1"], 3)}


def rule_scores(df):
    return np.maximum(df["j1_fric_ratio"].to_numpy(), df["j2_fric_ratio"].to_numpy())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/standard")
    ap.add_argument("--fresh", nargs="*", default=["data/stress2", "data/stress3"],
                    help="extra stress-only datasets (output tag = folder name)")
    ap.add_argument("--sets", nargs="+", default=list(FEATURE_SETS))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    stride = cfg["features"]["stride_s"]
    split = pd.read_csv(f"{a.data}/split_manifest.csv")
    check_split(split)
    manifest = load_manifest(a.data)
    out = ensure_new_dir(PROJECT_ROOT / "results" / "residual", a.overwrite)
    metrics, rows = {"note": __doc__}, []
    parts = [("test", a.data, split, manifest, "test"), ("stress", a.data, split, manifest, "stress")]
    parts += [(Path(d).name, d, pd.read_csv(f"{d}/split_manifest.csv"), load_manifest(d), "stress") for d in a.fresh]

    # --- physics-only rule, operating point on validation ---
    tcfg = cfg["training"]
    rule_best = None
    for window_s in cfg["features"]["window_s_options"]:
        fv = attach_split(load_feature_table(a.data, window_s), split)
        va = fv[fv["split"] == "val"].reset_index(drop=True)
        op, _ = select_operating_point(va, rule_scores(va), list(np.round(np.arange(1.1, 2.55, 0.1), 2)),
                                       tcfg["n_on_grid"], tcfg["n_off"], manifest, stride,
                                       tcfg["max_false_alarms_per_min"])
        key = (op["fa_per_min"] <= tcfg["max_false_alarms_per_min"], op["f1"])
        if rule_best is None or key > rule_best["key"]:
            rule_best = {"key": key, "window_s": window_s, "op": op}
    metrics["physics_rule"] = {"window_s": rule_best["window_s"], "operating_point": rule_best["op"]}
    print(f"physics rule: window={rule_best['window_s']} op={rule_best['op']}", flush=True)
    for part, ddir, sp, man, key in parts:
        fr_ = attach_split(load_feature_table(ddir, rule_best["window_s"]), sp)
        d = fr_[fr_["split"] == key].reset_index(drop=True)
        _, _, s = evaluate_detector(d, rule_scores(d), rule_best["op"]["threshold"], int(rule_best["op"]["n_on"]),
                                    tcfg["n_off"], man, stride)
        metrics["physics_rule"][part] = s
        rows.append(_row("residual", part, "physics_rule", s))
        print(rows[-1], flush=True)
    for fs in a.sets:
        names = FEATURE_SETS[fs]
        print(f"\n=== feature set: {fs} ({len(names)} features) ===", flush=True)
        bundle, baseline, report, _, _ = train_all(a.data, split, manifest, cfg, log=lambda *_: None,
                                                   feature_names=names)
        mdir = ensure_new_dir(PROJECT_ROOT / "models" / f"residual_{fs}", a.overwrite)
        bundle["data_dir"] = a.data
        joblib.dump(bundle, mdir / "model_bundle.joblib")
        metrics[fs] = {"rf_best": report["rf_best"], "operating_point": report["rf_operating_point"]}
        print(f"selected: {report['rf_best']}  op: thr={bundle['threshold']} n_on={bundle['n_on']}", flush=True)
        for part, ddir, sp, man, key in parts:
            feat = attach_split(load_feature_table(ddir, bundle["window_s"]), sp)
            d = feat[feat["split"] == key].reset_index(drop=True)
            _, runs, s = evaluate_detector(d, rf_scores(bundle["model"], d, names), bundle["threshold"],
                                           bundle["n_on"], bundle["n_off"], man, stride)
            metrics[fs][part] = s
            rows.append(_row(fs, part, "rf", s))
            runs.to_csv(out / f"runs_{part}_{fs}.csv", index=False)
            if fs == a.sets[0]:  # baseline does not depend on the feature set
                fb = attach_split(load_feature_table(ddir, baseline["window_s"]), sp)
                db = fb[fb["split"] == key].reset_index(drop=True)
                _, _, sb = evaluate_detector(db, baseline_scores(baseline, db), baseline["threshold"],
                                             baseline["n_on"], baseline["n_off"], man, stride)
                metrics.setdefault("baseline", {})[part] = sb
                rows.append(_row("-", part, "baseline", sb))
            print(rows[-1] if fs != a.sets[0] else rows[-2], flush=True)
        imp = pd.DataFrame({"feature": names, "importance": bundle["model"].feature_importances_})
        imp.sort_values("importance", ascending=False).to_csv(out / f"feature_importance_{fs}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
