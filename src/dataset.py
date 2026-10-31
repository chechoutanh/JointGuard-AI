"""Headless batch data generation and I/O (Stage B).

Layout of data/<name>/:
  manifest.csv         one row per successful run: metadata + ground truth (NOT features)
  failed_runs.csv      runs that diverged numerically / left joint limits (excluded)
  signals/<run>.csv.gz observable signals usable by models (see SIGNAL_COLUMNS)
  truth/<run>.csv.gz   simulator ground truth: true state, friction multipliers, label
  generation_info.json config snapshot, library versions, timing
"""
import json
import math
import platform
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, ensure_new_dir
from .faults import FaultSchedule
from .scenarios import sample_run_specs
from .simulator import Simulator, SimulationDivergedError, SIGNAL_COLUMNS, TRUTH_COLUMNS
from .trajectory import trajectory_from_dict


def fault_from_spec(spec, cfg):
    return FaultSchedule(kind=spec["fault_kind"], joint=int(spec["fault_joint"]),
                         t_onset=float(spec["t_onset"]), mult_final=float(spec["mult_final"]),
                         ramp=float(spec["ramp"]),
                         label_threshold=cfg["fault"]["label_threshold_multiplier"])


def simulator_from_spec(spec, cfg):
    traj = trajectory_from_dict(spec["traj"])
    q_start = traj(0.0)[0]
    q0 = (q_start[0] + spec["q0_offset1"], q_start[1] + spec["q0_offset2"])
    return Simulator(cfg, spec["payload"], spec["payload_declared"],
                     (spec["b1"], spec["b2"]), (spec["tau_c1"], spec["tau_c2"]), traj,
                     fault_from_spec(spec, cfg), spec["noise_q"], spec["noise_qd"],
                     spec["noise_seed"], q0=q0)


def simulate_spec(args):
    spec, cfg = args
    t0 = time.time()
    sim = simulator_from_spec(spec, cfg)
    try:
        sig, truth = sim.run(spec["duration"])
    except SimulationDivergedError as e:
        return spec, None, None, {"error": str(e), "elapsed": time.time() - t0}
    fault = sim.fault
    info = {"t_label": fault.t_label, "saturated_fraction": sim.saturated_ticks / max(sim.k, 1),
            "elapsed": time.time() - t0}
    return spec, sig, truth, info


def generate_dataset(name, mode, cfg=None, overwrite=False, processes=None, data_root=None):
    cfg = cfg or load_config()
    out = ensure_new_dir(Path(data_root or Path(__file__).resolve().parents[1] / "data") / name, overwrite)
    (out / "signals").mkdir()
    (out / "truth").mkdir()
    specs = sample_run_specs(cfg, mode, "indomain") + sample_run_specs(cfg, mode, "stress")
    rows, failed = [], []
    t0 = time.time()
    with Pool(processes) as pool:
        for spec, sig, truth, info in pool.imap(simulate_spec, [(s, cfg) for s in specs]):
            if sig is None:
                failed.append({"run_id": spec["run_id"], "scenario": spec["scenario"],
                               "error": info["error"], "spec": json.dumps(spec)})
                print(f"  [FAILED] {spec['run_id']}: {info['error']}")
                continue
            # save after each run
            pd.DataFrame(sig, columns=SIGNAL_COLUMNS).to_csv(
                out / "signals" / f"{spec['run_id']}.csv.gz", index=False, float_format="%.10g")
            tdf = pd.DataFrame(truth, columns=TRUTH_COLUMNS)
            tdf["label"] = tdf["label"].astype(int)
            tdf.to_csv(out / "truth" / f"{spec['run_id']}.csv.gz", index=False, float_format="%.10g")
            row = {k: v for k, v in spec.items() if k != "traj"}
            row["traj_json"] = json.dumps(spec["traj"])
            row["t_label"] = info["t_label"]
            row["saturated_fraction"] = info["saturated_fraction"]
            rows.append(row)
            if len(rows) % 20 == 0:
                print(f"  {len(rows)}/{len(specs)} runs done ({time.time() - t0:.0f} s)")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(failed, columns=["run_id", "scenario", "error", "spec"]).to_csv(out / "failed_runs.csv", index=False)
    import scipy, sklearn
    info = {"name": name, "mode": mode, "seed": cfg["seed"], "n_runs_ok": len(rows),
            "n_runs_failed": len(failed), "elapsed_s": time.time() - t0,
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg}
    with open(out / "generation_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    return out, manifest, failed


def load_manifest(data_dir):
    m = pd.read_csv(Path(data_dir) / "manifest.csv")
    m["t_onset"] = m["t_onset"].astype(float)
    m["t_label"] = m["t_label"].astype(float)
    return m


def load_signals(data_dir, run_id):
    return pd.read_csv(Path(data_dir) / "signals" / f"{run_id}.csv.gz")


def load_truth(data_dir, run_id):
    return pd.read_csv(Path(data_dir) / "truth" / f"{run_id}.csv.gz")


def spec_from_manifest_row(row):
    spec = {k: row[k] for k in row.index if k != "traj_json"}
    spec["traj"] = json.loads(row["traj_json"])
    for k in ("noise_seed", "fault_joint"):
        spec[k] = int(spec[k])
    return spec
