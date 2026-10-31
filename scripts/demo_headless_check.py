"""Automated, headless run of the demo (no window needed) that injects key presses at
given simulated times and saves screenshots. Useful for checking the GUI on a server
or for preparing figures. Uses frame-locked stepping (1/60 s of sim time per frame).

  python -m scripts.demo_headless_check --model models/standard
"""
import argparse
import os

import pygame


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/standard")
    a = ap.parse_args(argv)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from src.gui.app import DemoApp
    app = DemoApp(a.model, fixed_step=True)
    script = [(14.0, pygame.K_3)]
    shots = [(1.0, "headless_01_warmup.png"), (12.0, "headless_02_healthy.png"),
             (14.6, "headless_03_fault_injected.png"), (24.0, "headless_04_after_fault.png")]
    app.run(script=script, max_sim_time=24.5, shots=shots)


if __name__ == "__main__":
    main()
