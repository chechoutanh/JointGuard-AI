"""Open the live Pygame demo.

  python -m scripts.run_demo                                   # physics-residual detector (default)
  python -m scripts.run_demo --model models/standard           # original detector, for comparison
"""
import argparse

from src.gui.app import DemoApp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/residual_residual")
    a = ap.parse_args(argv)
    DemoApp(a.model).run()


if __name__ == "__main__":
    main()
