# Generalisation check: a 7-DOF KUKA LBR iiwa

The rest of the project uses a 2-joint planar arm. This experiment asks one question: **does the
physics-residual detector (see [`RESIDUAL_EXPERIMENT.md`](RESIDUAL_EXPERIMENT.md)) still work on a
7-joint industrial-style arm with full 3-D coupled dynamics, without re-designing it?**

It covers detection only. Diagnosis (which joint, how much) and the Random Forest are not redone here.
The 2-DOF results are unchanged.

Code: `src/kuka.py`, `configs/kuka.json`, `scripts/kuka_experiment.py`, `tests/test_kuka.py`.
Results: `results/kuka/`.

## Set-up

| | 2-DOF arm (main project) | KUKA iiwa (this experiment) |
|---|---|---|
| Dynamics | closed-form M, C, G | pybullet as a dynamics library (mass matrix, inverse dynamics) on `pybullet_data/kuka_iiwa/model.urdf`, plus motor armature |
| Joints | 2, planar | 7, spatial |
| Payload | point mass at the tip | point mass at link 7's centre of mass |
| Plant, controller, sensors | RK4 1 kHz, PD + gravity compensation 100 Hz with declared payload, noisy sensors 100 Hz | same |
| Faults | friction ×1.6–4.0 on one joint, abrupt or gradual, label at ×1.5 | same, on any of the 7 joints (fault groups cycle through joints 1–7) |

The link masses, centres of mass and inertias come from the URDF shipped with pybullet. They are
**approximate values, not KUKA's identified parameters**. Armature, friction (larger on the base
joints, smaller on the wrist) and PD gains were chosen for this experiment. Torque limits follow the
iiwa 7 R800 data sheet. pybullet's own `stepSimulation` is never used: the integration loop is the
same as in `src/simulator.py`.

**What was frozen and what was chosen.** The residual features are the 2-DOF ones extended to n joints:
the same code (`window_residual_features`, now n-joint, bit-identical for 2 joints), acceleration
stencil K = 4, prior weight and moving threshold. Nothing about the method was tuned on the KUKA
data. Only the alarm operating point (window length, threshold, persistence n_on) was chosen, on the
KUKA **validation** split, with the same criterion as before: maximum window F1 with at most 0.5 false
alarms per healthy minute.

**Data.** 186 runs of 25 s, seed 20261128, none diverged and no torque saturation.
- In-domain, 124 runs: payload 0–3 kg, sinusoid frequency 0.15–0.55 Hz. Split by group into
  val (62) and test (62).
- Stress, 62 runs, held out and never used for any choice: payload 3–5 kg, 0.55–0.75 Hz, and half of
  the groups on point-to-point motion, which is never seen in val.

**Detectors.**
- Physics rule: score = max over the 7 joints of `fric_ratio_j`, the payload-corrected residual
  divided by nominal friction. It is about 1 when healthy and about the friction multiplier under a fault.
- Error baseline: score = max_j e_rms_j / ref_j, where ref_j is the 95 % quantile on healthy val windows.

## Results

`results/kuka/summary.csv`, produced by `python -m scripts.kuka_experiment`.
Selected on val: physics rule with 1.0 s window, threshold 1.5, n_on 1. Error baseline with 1.0 s
window, scale 1.6, n_on 1.

| Set | Detector | Faults detected | False alarms / healthy min | Healthy runs w/ FA | Median delay | Window F1 |
|---|---|---|---|---|---|---|
| Test | **Physics rule** | **42/42** | 0.48 | 5/20 | 0.46 s | **0.98** |
| Test | Error baseline | 20/42 | 0.97 | 0/20 | 1.26 s | 0.51 |
| **Stress (held out)** | **Physics rule** | **42/42** | **0.85** | 6/20 | 0.58 s | **0.94** |
| Stress (held out) | Error baseline | 17/42 | 3.55 | 10/20 | 4.17 s | 0.56 |
| Val (used for selection) | Physics rule | 42/42 | 0.08 | 0/20 | 0.39 s | 0.98 |

Per joint (`results/kuka/per_joint.csv`), the physics rule detects 6/6 faults on **every** joint in
test and stress. That includes the wrist joints 5–7, whose friction is only 0.4–0.8 N·m s/rad. The error
baseline misses almost all base-joint faults (joints 1–2: 0/12 on test), because a stiff, heavily loaded
joint barely changes its tracking error when friction rises.

### Where the false alarms come from

- **Stress: all 11 false alarms are on point-to-point motion** (11 in 7.6 healthy minutes), with none on
  sinusoids (5.4 healthy minutes). This is the same failure mode as on the 2-DOF arm. During stops and
  starts friction is barely observable and the acceleration estimate is worst.
- **Test: 5 of the 6 false alarms are in the slow, heavy-load healthy group**, each one alarm, with
  ratios 1.5–1.7, just above the threshold.
- Almost every false alarm is raised by **joint 4 (elbow) or joint 2 (shoulder)**. These joints carry the
  largest gravity and inertia torques relative to their friction, so a small error in the acceleration
  estimate (multiplied by a large inertia) shows up as a large friction ratio.

## What this means

- The method transfers. Extended from 2 to 7 joints with nothing re-tuned except the alarm threshold,
  the physics rule catches every fault on every joint, including on a held-out set with heavier loads
  and faster, unseen motion. The tracking-error baseline catches fewer than half.
- The false-alarm rate on stress, 0.85 per healthy minute, is above the 0.5 target. The weak spots are
  identifiable: point-to-point motion, and the heavily loaded joints 2 and 4. They are the same weak spots
  as on the 2-DOF arm, so they are a property of the method, not of one robot.
- No threshold or feature was changed after seeing the test or stress results.

## Limitations

- Still 100 % simulation, and the same model is used in the plant and in the residual. On a real iiwa
  the nominal model is never exact, so identification would be needed first. The URDF inertias are
  approximate.
- Motors, gearboxes, joint elasticity and torque-sensor dynamics are not modelled. The real iiwa has
  joint torque sensors, which would make the residual easier, not harder.
- Detection only. The friction ratio already points at the faulty joint, but diagnosis was not evaluated.
- 62 test and 62 stress runs, without confidence intervals.
- One idea that was *not* tried, to keep stress held out: normalising the ratio per joint by the torque it
  carries, or ignoring windows with stops. Any such change needs a fresh stress set.

## Reproduce

```
pip install pybullet==3.2.7
python -m scripts.kuka_experiment            # about 5 min on 12 cores; --overwrite to redo
python -m pytest tests/test_kuka.py
```
