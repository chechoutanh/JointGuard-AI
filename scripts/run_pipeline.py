"""Headless end-to-end pipeline. Every step can also be run on its own (see README).

  python -m scripts.run_pipeline --mode smoke      # quick check (~1-2 min)
  python -m scripts.run_pipeline --mode standard   # full experiment

Existing outputs are never overwritten unless --overwrite is given.
"""
import argparse
import subprocess
import sys


def step(args):
    print("\n>>> python -m " + " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-m", *args])
    if r.returncode:
        sys.exit(f"step failed: {' '.join(args)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "standard"], default="smoke")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(argv)
    ow = ["--overwrite"] if a.overwrite else []
    name = a.mode
    data = f"data/{name}"
    if not a.skip_tests:
        step(["pytest", "-q"])
    step(["scripts.generate_dataset", "--mode", a.mode, "--name", name, *ow])
    step(["scripts.make_splits", "--data", data, *ow])
    step(["scripts.build_features", "--data", data, *ow])
    step(["scripts.train", "--data", data, "--tag", name, *ow])
    step(["scripts.evaluate", "--data", data, "--model", f"models/{name}", "--tag", name, *ow])
    step(["scripts.train_diagnosis", "--data", data, "--model", f"models/{name}", *ow])
    step(["scripts.evaluate_diagnosis", "--data", data, "--model", f"models/{name}",
          "--results", f"results/{name}", *ow])
    step(["scripts.make_figures", "--results", f"results/{name}"])
    for lang in ("vi", "en"):
        step(["scripts.make_report", "--results", f"results/{name}", "--model", f"models/{name}",
              "--out", f"reports/{name}", "--lang", lang, *ow])
    print(f"\nDone. Results: results/{name}  Report: reports/{name}")


if __name__ == "__main__":
    main()
