"""Pygame live demo (Stage E).

Scheduling: physics (1 kHz RK4) and control (100 Hz) run inside Simulator.step();
sampling at 100 Hz; inference every stride (0.1 s) through OnlineDetector; display at
~60 FPS. Each frame advances simulated time by the elapsed wall-clock time, so robot
motion is NOT slowed by a low frame rate. If the computer cannot simulate fast enough,
the real-time factor drops below 1 and a NOT REAL-TIME warning is shown.

The alarm shown in the AI panel comes only from the trained model. Ground truth is
shown in a separate, labelled panel for the experimenter.
"""
import csv
import json
import math
import time
from collections import deque
from pathlib import Path

import pygame

from ..config import load_config, PROJECT_ROOT
from ..dynamics import forward_kinematics
from ..faults import FaultSchedule
from ..features import INPUT_COLUMNS
from ..online import OnlineDetector, load_bundle, load_diagnosis, ModelNotFoundError
from ..simulator import Simulator, SimulationDivergedError
from ..trajectory import SinusoidTrajectory

W, H = 1280, 760
BG = (18, 22, 30); PANEL = (30, 36, 48); GRID = (52, 60, 76); TEXT = (220, 226, 236)
DIM = (140, 150, 168); C1 = (90, 170, 255); C2 = (255, 170, 70); GOOD = (80, 200, 120)
WARN = (240, 200, 60); BAD = (240, 80, 80); TARGET = (150, 150, 170)

PAYLOADS = [0.5, 1.5, 2.0]
SPEEDS = {"normal": (0.30, 0.38), "fast": (0.50, 0.55)}


class DemoApp:
    def __init__(self, model_dir, cfg=None, fixed_step=False, seed=1, log_dir=None):
        self.cfg = cfg or load_config()
        self.fixed_step = fixed_step
        self.seed = seed
        self.log_dir = Path(log_dir or PROJECT_ROOT / "results" / "demo_logs")
        self.shot_dir = PROJECT_ROOT / "results" / "screenshots"
        try:
            self.bundle = load_bundle(model_dir)
            self.model_error = None
        except ModelNotFoundError as e:
            self.bundle, self.model_error = None, str(e)
        self.diag_bundle = load_diagnosis(model_dir) if self.bundle else None
        self.payload_idx, self.speed = 0, "normal"
        self.paused = False
        self.log_rows = []
        self.events = []
        self.reset_count = 0
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("2-DOF robot - joint friction anomaly detection (simulation)")
        self.font = pygame.font.SysFont("dejavusans,arial", 15)
        self.font_b = pygame.font.SysFont("dejavusans,arial", 18, bold=True)
        self.font_big = pygame.font.SysFont("dejavusans,arial", 26, bold=True)
        self.clock = pygame.time.Clock()
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        cfg = self.cfg
        f1, f2 = SPEEDS[self.speed]
        self.traj = SinusoidTrajectory((0.8, -1.25), (0.40, 0.45), (f1, f2), (0.0, 1.2),
                                       cfg["trajectory"]["t_ramp"])
        payload = PAYLOADS[self.payload_idx]
        fr = cfg["friction"]
        self.sim = Simulator(cfg, payload, payload, fr["nominal_b"], fr["nominal_tau_c"], self.traj,
                             FaultSchedule(), 0.0005, 0.005, noise_seed=self.seed + self.reset_count)
        self.reset_count += 1
        self.detector = OnlineDetector(self.bundle, self.diag_bundle) if self.bundle else None
        self.mode_name = "healthy"
        self.hist = deque(maxlen=1000)      # (t, tau1, tau2, e1, e2) at 100 Hz, 10 s
        self.score_hist = deque(maxlen=100)  # (t, score, alarm) at 10 Hz
        self.trail = deque(maxlen=400)
        self.n_alarm_events = 0
        self.first_alarm_after_fault = None
        self.fault_time = None
        self.diverged = None
        self.rtf = 1.0
        self._rtf_acc = [0.0, 0.0]
        self.det = {"status": "WARMING UP" if self.detector else "NO MODEL", "score": None, "alarm": False}

    def set_fault(self, key):
        t = self.sim.t
        lt = self.cfg["fault"]["label_threshold_multiplier"]
        if key == 1:
            self.sim.fault = FaultSchedule(label_threshold=lt); self.mode_name = "healthy"
            self.fault_time = None
        elif key in (2, 3):
            j = 1 if key == 2 else 2
            self.sim.fault = FaultSchedule("abrupt", j, t, 3.0, label_threshold=lt)
            self.mode_name = f"abrupt x3.0 joint {j}"
            self.fault_time = self.sim.fault.t_label
        elif key == 4:
            self.sim.fault = FaultSchedule("gradual", 1, t, 3.5, 10.0, label_threshold=lt)
            self.mode_name = "gradual -> x3.5 joint 1 (10 s)"
            self.fault_time = self.sim.fault.t_label
        self.first_alarm_after_fault = None
        self.events.append({"t": t, "event": "mode", "mode": self.mode_name})

    # ------------------------------------------------------------------ simulation
    def advance(self, sim_dt):
        """Advance simulated time by sim_dt, bounded by a wall-clock budget."""
        T_c = self.sim.T_c
        n = int(round(sim_dt / T_c)) if self.fixed_step else int(sim_dt / T_c + self._carry)
        if not self.fixed_step:
            self._carry = sim_dt / T_c + self._carry - n
        t0 = time.perf_counter()
        done = 0
        for _ in range(n):
            if self.diverged:
                break
            try:
                sig, truth = self.sim.step()
            except SimulationDivergedError as e:
                self.diverged = str(e)
                self.events.append({"t": self.sim.t, "event": "diverged", "error": str(e)})
                break
            done += 1
            if sig is not None:
                self.on_sample(sig, truth)
            if not self.fixed_step and time.perf_counter() - t0 > 0.03:
                self._carry = 0.0   # cannot keep up: drop backlog, report RTF < 1
                break
        return done * T_c

    _carry = 0.0

    def on_sample(self, sig, truth):
        t = sig[0]
        e1, e2 = sig[5] - sig[1], sig[6] - sig[2]
        self.hist.append((t, sig[11], sig[12], e1, e2))
        q1, q2 = truth[1], truth[2]
        self.trail.append(forward_kinematics(q1, q2, self.sim.params)[2:])
        res = None
        if self.detector:
            sample = list(sig[1:9]) + [sig[11], sig[12]]   # INPUT_COLUMNS order
            res = self.detector.push(sample)
            self.det = res
            if res["updated"]:
                self.score_hist.append((t, res["score"], res["alarm"]))
                if res["turned_on"]:
                    self.n_alarm_events += 1
                    self.events.append({"t": t, "event": "alarm_on", "score": res["score"]})
                    if self.fault_time is not None and t >= self.fault_time and self.first_alarm_after_fault is None:
                        self.first_alarm_after_fault = t
        self.log_rows.append([self.reset_count, *sig, truth[5], truth[6], truth[7],
                              (res or {}).get("score"), (res or {}).get("alarm"), self.mode_name])

    # ------------------------------------------------------------------ drawing
    def text(self, s, pos, color=TEXT, font=None):
        self.screen.blit((font or self.font).render(s, True, color), pos)

    def draw_robot(self, rect):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
        self.text("Robot (vertical plane)  -  dotted: target tip path,  solid: actual tip trail (last 4 s)",
                  (rect.x + 12, rect.y + 8), DIM)
        scale = 330.0
        ox, oy = rect.x + rect.w * 0.30, rect.y + rect.h * 0.62

        def to_px(x, y):
            return int(ox + x * scale), int(oy - y * scale)
        # dotted target tip path over the same 4 s as the actual trail
        t_now = self.sim.t
        for k in range(0, 200, 2):
            tk = t_now - 4.0 + k * 0.02
            if tk >= 0:
                q, _ = self.traj(tk)
                pygame.draw.circle(self.screen, TARGET, to_px(*forward_kinematics(q[0], q[1], self.sim.params)[2:]), 2)
        if len(self.trail) > 1:
            pygame.draw.lines(self.screen, (90, 220, 200), False, [to_px(*p) for p in self.trail], 2)
        q1, q2 = self.sim.state[0], self.sim.state[1]
        x1, y1, x2, y2 = forward_kinematics(q1, q2, self.sim.params)
        qd, _ = self.traj(self.sim.t)
        _, _, xd, yd = forward_kinematics(qd[0], qd[1], self.sim.params)
        m1, m2 = self.sim.fault.multipliers(self.sim.t)
        pygame.draw.rect(self.screen, (80, 88, 104), (ox - 30, oy, 60, 22))
        pygame.draw.line(self.screen, C1, to_px(0, 0), to_px(x1, y1), 14)
        pygame.draw.line(self.screen, C2, to_px(x1, y1), to_px(x2, y2), 10)
        for (x, y), m in (((0, 0), m1), ((x1, y1), m2)):
            col = BAD if m >= 1.5 else (WARN if m > 1.0 else TEXT)
            pygame.draw.circle(self.screen, col, to_px(x, y), 11)
            pygame.draw.circle(self.screen, BG, to_px(x, y), 5)
        s = int(8 + 5 * self.sim.model.payload)
        px, py = to_px(x2, y2)
        pygame.draw.rect(self.screen, (200, 200, 210), (px - s // 2, py - s // 2, s, s))
        pygame.draw.circle(self.screen, TARGET, to_px(xd, yd), 6, 1)
        self.text("joint ring colour = injected friction (ground truth view)", (rect.x + 12, rect.bottom - 24), DIM)

    def draw_plot(self, rect, title, series, ylabel, t_now, hline=None, shade=None, ylim=None):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
        self.text(title, (rect.x + 10, rect.y + 6), TEXT, self.font_b)
        inner = pygame.Rect(rect.x + 55, rect.y + 32, rect.w - 70, rect.h - 50)
        pygame.draw.rect(self.screen, (24, 29, 39), inner)
        t0 = t_now - 10.0
        vals = [v for _, pts, _ in series for (_, v) in pts]
        if ylim is None:
            lo, hi = (min(vals), max(vals)) if vals else (-1, 1)
            pad = 0.1 * (hi - lo) + 1e-6
            lo, hi = lo - pad, hi + pad
        else:
            lo, hi = ylim

        def to_px(t, v):
            return (inner.x + (t - t0) / 10.0 * inner.w,
                    inner.bottom - (min(max(v, lo), hi) - lo) / (hi - lo) * inner.h)
        if shade:
            for ta, tb in shade:
                xa, xb = to_px(max(ta, t0), 0)[0], to_px(tb, 0)[0]
                if xb > xa:
                    pygame.draw.rect(self.screen, (70, 30, 34), (xa, inner.y, xb - xa, inner.h))
        for frac in (0.0, 0.5, 1.0):
            v = lo + frac * (hi - lo)
            y = to_px(t0, v)[1]
            pygame.draw.line(self.screen, GRID, (inner.x, y), (inner.right, y), 1)
            self.text(f"{v:.3g}", (rect.x + 4, y - 8), DIM)
        if hline is not None:
            y = to_px(t0, hline)[1]
            for x in range(inner.x, inner.right, 10):
                pygame.draw.line(self.screen, WARN, (x, y), (x + 5, y), 1)
        lx = rect.right - 15
        for name, pts, col in reversed(series):
            if len(pts) > 1:
                pygame.draw.lines(self.screen, col, False, [to_px(t, v) for t, v in pts], 2)
            surf = self.font.render(name, True, col)
            lx -= surf.get_width() + 14
            self.screen.blit(surf, (lx, rect.y + 8))
        self.text(ylabel, (inner.x + 4, inner.y + 2), DIM)

    def draw_panels(self):
        t_now = self.sim.t
        h = list(self.hist)
        right = pygame.Rect(650, 10, 620, 170)
        self.draw_plot(right, "Applied joint torque (simulated signal)",
                       [("tau1", [(r[0], r[1]) for r in h], C1), ("tau2", [(r[0], r[2]) for r in h], C2)],
                       "N*m", t_now)
        self.draw_plot(right.move(0, 180), "Tracking error  q_des - q_meas",
                       [("e1", [(r[0], r[3]) for r in h], C1), ("e2", [(r[0], r[4]) for r in h], C2)],
                       "rad", t_now)
        shade, start = [], None
        for t, s, a in self.score_hist:
            if a and start is None:
                start = t
            if not a and start is not None:
                shade.append((start, t)); start = None
        if start is not None:
            shade.append((start, t_now))
        thr = self.bundle["threshold"] if self.bundle else None
        self.draw_plot(right.move(0, 360), "Model score (RF predict_proba, not calibrated)",
                       [("score", [(t, s) for t, s, _ in self.score_hist], (220, 120, 255))],
                       "", t_now, hline=thr, shade=shade, ylim=(0.0, 1.0))
        # AI status
        box = pygame.Rect(650, 540, 300, 210)
        pygame.draw.rect(self.screen, PANEL, box, border_radius=8)
        self.text("AI PREDICTION", (box.x + 12, box.y + 8), DIM, self.font_b)
        if self.model_error:
            self.text("MODEL NOT FOUND", (box.x + 12, box.y + 40), BAD, self.font_big)
            self.text("No predictions are shown.", (box.x + 12, box.y + 80))
            self.text("Train first:", (box.x + 12, box.y + 104))
            self.text("python -m scripts.run_pipeline", (box.x + 12, box.y + 126), WARN)
            self.text("       --mode standard", (box.x + 12, box.y + 146), WARN)
        else:
            st = self.det["status"]
            col = {"WARMING UP": DIM, "NORMAL": GOOD, "SUSPICIOUS": WARN, "ANOMALY ALARM": BAD}.get(st, TEXT)
            self.text(st, (box.x + 12, box.y + 40), col, self.font_big)
            sc = self.det.get("score")
            self.text(f"score: {sc:.2f}" if sc is not None else "score: -", (box.x + 12, box.y + 82))
            self.text(f"threshold: {self.bundle['threshold']:.2f}   on after {self.bundle['n_on']} windows",
                      (box.x + 12, box.y + 104), DIM)
            self.text(f"window: {self.bundle['window_s']} s, update every {self.bundle['stride_s']} s",
                      (box.x + 12, box.y + 126), DIM)
            self.text(f"alarm events this run: {self.n_alarm_events}", (box.x + 12, box.y + 150))
            dg = self.det.get("diagnosis")
            if dg:
                self.text(f"Joint {dg['joint']} - friction increase: {dg['increase_pct']:.0f}%  (x{dg['mult']:.1f})",
                          (box.x + 12, box.y + 176), BAD)
            elif self.diag_bundle is None:
                self.text("diagnosis: not trained", (box.x + 12, box.y + 176), DIM)
        # ground truth
        gt = pygame.Rect(960, 540, 310, 210)
        pygame.draw.rect(self.screen, (40, 36, 30), gt, border_radius=8)
        self.text("SIMULATOR GROUND TRUTH", (gt.x + 12, gt.y + 8), WARN, self.font_b)
        self.text("(for the experimenter - NOT an AI input)", (gt.x + 12, gt.y + 30), DIM)
        m1, m2 = self.sim.fault.multipliers(t_now)
        lab = self.sim.fault.label(t_now)
        lines = [f"mode: {self.mode_name}",
                 f"friction multiplier  j1: {m1:.2f}   j2: {m2:.2f}",
                 f"label: {'ANOMALY' if lab else 'normal'}",
                 f"payload: {self.sim.model.payload:.1f} kg   speed: {self.speed}"]
        if self.fault_time is not None:
            lines.append(f"label switched at t = {self.fault_time:.1f} s" if t_now >= self.fault_time
                         else f"label will switch at t = {self.fault_time:.1f} s")
            if self.first_alarm_after_fault is not None:
                lines.append(f"first alarm after it: +{self.first_alarm_after_fault - self.fault_time:.1f} s")
        for i, s in enumerate(lines):
            self.text(s, (gt.x + 12, gt.y + 56 + 22 * i), BAD if (s.startswith("label: ANOM")) else TEXT)

    def draw_footer(self):
        r = pygame.Rect(10, 650, 630, 100)
        pygame.draw.rect(self.screen, PANEL, r, border_radius=8)
        rt = "fixed-step render (headless)" if self.fixed_step else f"real-time factor {self.rtf:4.2f}"
        self.text(f"t = {self.sim.t:6.1f} s    FPS {self.clock.get_fps():4.0f}    {rt}", (r.x + 12, r.y + 8))
        if self.paused:
            self.text("PAUSED", (r.x + 470, r.y + 8), WARN, self.font_b)
        elif self.rtf < 0.95 and not self.fixed_step:
            self.text("NOT REAL-TIME", (r.x + 480, r.y + 8), BAD, self.font_b)
        if self.diverged:
            self.text("SIMULATION DIVERGED (logged, not a fault): " + self.diverged[:50], (r.x + 12, r.y + 30), BAD)
        self.text("Space pause | R reset | 1 healthy | 2 friction j1 | 3 friction j2 | 4 gradual j1",
                  (r.x + 12, r.y + 52), DIM)
        self.text("L payload | V speed (both reset) | S screenshot | Esc quit & save log",
                  (r.x + 12, r.y + 74), DIM)

    def draw(self):
        self.screen.fill(BG)
        self.draw_robot(pygame.Rect(10, 10, 630, 630))
        self.draw_panels()
        self.draw_footer()
        pygame.display.flip()

    # ------------------------------------------------------------------ io
    def screenshot(self, name=None):
        self.shot_dir.mkdir(parents=True, exist_ok=True)
        path = self.shot_dir / (name or f"demo_{time.strftime('%Y%m%d_%H%M%S')}_{self.sim.t:05.1f}s.png")
        pygame.image.save(self.screen, str(path))
        return path

    def save_log(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / f"demo_{stamp}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["reset_no", "t", "q1_meas", "q2_meas", "qd1_meas", "qd2_meas", "q1_des", "q2_des",
                        "qd1_des", "qd2_des", "tau1_req", "tau2_req", "tau1_applied", "tau2_applied",
                        "gt_fric_mult1", "gt_fric_mult2", "gt_label", "model_score", "alarm", "mode"])
            w.writerows(self.log_rows)
        with open(self.log_dir / f"demo_{stamp}_events.json", "w") as f:
            json.dump(self.events, f, indent=2)
        return path

    def handle_key(self, key):
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.reset(); self.events.append({"event": "reset"})
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            self.set_fault(int(pygame.key.name(key)))
        elif key == pygame.K_l:
            self.payload_idx = (self.payload_idx + 1) % len(PAYLOADS); self.reset()
        elif key == pygame.K_v:
            self.speed = "fast" if self.speed == "normal" else "normal"; self.reset()
        elif key == pygame.K_s:
            print("screenshot:", self.screenshot())
        elif key == pygame.K_ESCAPE:
            return False
        return True

    def run(self, script=None, max_sim_time=None, shots=None):
        """script: list of (sim_time, key) for headless/automated runs; shots: list of (sim_time, filename)."""
        script = sorted(script or [])
        shots = sorted(shots or [])
        running = True
        last = time.perf_counter()
        while running:
            dt_frame = self.clock.tick(self.cfg["timing"]["fps_display"]) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    running = self.handle_key(ev.key)
            while script and self.sim.t >= script[0][0]:
                running = self.handle_key(script.pop(0)[1]) and running
            if not self.paused:
                want = 1.0 / self.cfg["timing"]["fps_display"] if self.fixed_step else min(dt_frame, 0.1)
                got = self.advance(want)
                self._rtf_acc[0] += got; self._rtf_acc[1] += dt_frame
                if self._rtf_acc[1] > 1.0:
                    self.rtf = self._rtf_acc[0] / self._rtf_acc[1]; self._rtf_acc = [0.0, 0.0]
            self.draw()
            while shots and self.sim.t >= shots[0][0]:
                print("screenshot:", self.screenshot(shots.pop(0)[1]))
            if max_sim_time is not None and self.sim.t >= max_sim_time:
                running = False
        log = self.save_log()
        print("log saved:", log)
        pygame.quit()
        return log
