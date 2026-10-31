"""Open the live Pygame demo.

  python -m scripts.run_demo --model models/standard
"""
import argparse

from src.gui.app import DemoApp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/standard")
    a = ap.parse_args(argv)
    DemoApp(a.model).run()


if __name__ == "__main__":
    main()
