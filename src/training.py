"""Model selection and training (Stage C.2/C.3). Uses ONLY train and validation splits.

Random Forest:
  1. for each window length option and each grid point, fit on train windows
     (mixed-label windows dropped) and score validation average precision;
  2. keep the best (window, hyper-parameters);
  3. choose alarm threshold and persistence n_on on validation: maximise window-level
     F1 of the alarm state subject to false alarms per healthy minute <= limit.
Baseline (tracking error threshold):
  score = max_j  e_rms_j / ref_j ,  ref_j = 95th percentile of e_rms_j on HEALTHY TRAIN windows
  alarm when score >= scale for n_on consecutive windows; window, scale, n_on chosen on
  validation with exactly the same criterion as the Random Forest.
"""
import itertools
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from .alarm import alarm_sequence
from .evaluation import per_run_events, summarize
from .features import FEATURE_NAMES

META_COLUMNS = ["run_id", "t_end", "label", "mixed"]


def load_feature_table(data_dir, window_s):
    return pd.read_csv(f"{data_dir}/features/w{window_s:g}.csv.gz")


def attach_split(feat, split):
    return feat.merge(split[["run_id", "split"]], on="run_id", how="left")


def rf_scores(model, df):
    return model.predict_proba(df[FEATURE_NAMES].to_numpy())[:, 1]


def baseline_scores(params, df):
    e1 = df["j1_e_rms"].to_numpy() / params["ref"][0]
    e2 = df["j2_e_rms"].to_numpy() / params["ref"][1]
    return np.maximum(e1, e2)


def apply_alarm(df, scores, threshold, n_on, n_off):
    """Run alarm logic per run (reset at every run start). Returns window table."""
    w = df[["run_id", "t_end", "label"]].copy()
    w["score"] = scores
    w["decision"] = scores >= threshold
    w["state"] = False
    w["onset"] = False
    for run_id, idx in w.groupby("run_id", sort=False).groups.items():
        sub = w.loc[idx].sort_values("t_end")
        st, on = alarm_sequence(sub["decision"].to_numpy(), n_on, n_off)
        w.loc[sub.index, "state"] = st
        w.loc[sub.index, "onset"] = on
    return w


def evaluate_detector(df, scores, threshold, n_on, n_off, manifest, stride_s):
    w = apply_alarm(df, scores, threshold, n_on, n_off)
    runs = per_run_events(w, manifest, stride_s)
    return w, runs, summarize(w, runs)


def select_operating_point(df, scores, grid_thr, grid_non, n_off, manifest, stride_s, max_fa):
    """Grid search threshold x n_on on validation data. Returns best dict and full table."""
    rows = []
    for thr, non in itertools.product(grid_thr, grid_non):
        _, _, s = evaluate_detector(df, scores, thr, non, n_off, manifest, stride_s)
        rows.append({"threshold": thr, "n_on": non, "f1": s["window_alarm_state"]["f1"],
                     "fa_per_min": s["false_alarms_per_healthy_min"],
                     "detection_rate": s["detection_rate"], "delay_median": s["delay_median"]})
    tab = pd.DataFrame(rows)
    ok = tab[tab["fa_per_min"] <= max_fa]
    best = (ok.sort_values(["f1", "fa_per_min"], ascending=[False, True]).iloc[0] if len(ok)
            else tab.sort_values(["fa_per_min", "f1"], ascending=[True, False]).iloc[0])
    return best.to_dict(), tab


def train_all(data_dir, split, manifest, cfg, log=print):
    tcfg, fcfg = cfg["training"], cfg["features"]
    stride_s = fcfg["stride_s"]
    seed = cfg["seed"]
    selection = []
    best = None
    tables = {}
    for window_s in fcfg["window_s_options"]:
        feat = attach_split(load_feature_table(data_dir, window_s), split)
        tables[window_s] = feat
        tr = feat[(feat["split"] == "train") & (~feat["mixed"])]
        va = feat[(feat["split"] == "val") & (~feat["mixed"])]
        keys = list(tcfg["rf_grid"].keys())
        for values in itertools.product(*[tcfg["rf_grid"][k] for k in keys]):
            params = dict(zip(keys, values))
            rf = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
            rf.fit(tr[FEATURE_NAMES].to_numpy(), tr["label"].to_numpy())
            ap = average_precision_score(va["label"], rf_scores(rf, va))
            selection.append({"window_s": window_s, **{k: str(v) for k, v in params.items()}, "val_ap": ap})
            log(f"  RF w={window_s}s {params} -> val AP={ap:.4f}")
            if best is None or ap > best["val_ap"]:
                best = {"window_s": window_s, "params": params, "val_ap": ap, "model": rf}
    log(f"Best RF: window={best['window_s']} s, {best['params']}, val AP={best['val_ap']:.4f}")

    # --- RF operating point on validation (all val windows, transitions kept) ---
    feat = tables[best["window_s"]]
    val_all = feat[feat["split"] == "val"]
    op_rf, op_tab_rf = select_operating_point(val_all, rf_scores(best["model"], val_all),
                                              tcfg["threshold_grid"], tcfg["n_on_grid"], tcfg["n_off"],
                                              manifest, stride_s, tcfg["max_false_alarms_per_min"])
    log(f"RF operating point: {op_rf}")

    # --- baseline ---
    base_best, base_tabs = None, []
    for window_s, feat_w in tables.items():
        tr_h = feat_w[(feat_w["split"] == "train") & (feat_w["label"] == 0)]
        ref = [float(np.quantile(tr_h[f"j{j}_e_rms"], tcfg["baseline_quantile"])) for j in (1, 2)]
        params = {"window_s": window_s, "ref": ref}
        va_w = feat_w[feat_w["split"] == "val"]
        op, tab = select_operating_point(va_w, baseline_scores(params, va_w), tcfg["baseline_scale_grid"],
                                         tcfg["n_on_grid"], tcfg["n_off"], manifest, stride_s,
                                         tcfg["max_false_alarms_per_min"])
        tab["window_s"] = window_s
        base_tabs.append(tab)
        feasible = op["fa_per_min"] <= tcfg["max_false_alarms_per_min"]
        key = (feasible, op["f1"])
        if base_best is None or key > base_best["key"]:
            base_best = {"key": key, "params": params, "op": op}
    log(f"Baseline: {base_best['params']} op={base_best['op']}")

    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    bundle = {
        "model": best["model"], "feature_names": list(FEATURE_NAMES),
        "window_s": best["window_s"], "stride_s": stride_s,
        "monitor_start_s": fcfg["monitor_start_s"], "f_sample": cfg["timing"]["f_sample"],
        "threshold": float(op_rf["threshold"]), "n_on": int(op_rf["n_on"]), "n_off": int(tcfg["n_off"]),
        "rf_params": best["params"], "seed": seed, "versions": versions,
        "score_note": "predict_proba output = model score, NOT a calibrated probability",
    }
    baseline = {"window_s": base_best["params"]["window_s"], "ref": base_best["params"]["ref"],
                "threshold": float(base_best["op"]["threshold"]), "n_on": int(base_best["op"]["n_on"]),
                "n_off": int(tcfg["n_off"]), "stride_s": stride_s,
                "monitor_start_s": fcfg["monitor_start_s"], "f_sample": cfg["timing"]["f_sample"]}
    report = {"rf_selection": selection, "rf_best": {"window_s": best["window_s"], "params": best["params"],
                                                    "val_ap": best["val_ap"]},
              "rf_operating_point": op_rf, "baseline": baseline, "baseline_operating_point": base_best["op"],
              "versions": versions}
    return bundle, baseline, report, op_tab_rf, pd.concat(base_tabs)
