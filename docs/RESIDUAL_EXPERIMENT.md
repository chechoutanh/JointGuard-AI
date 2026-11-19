# Follow-up experiment: physics-residual features

The standard run (README, `results/standard/`) showed that the Random Forest mistakes heavier
loads and faster motion for friction on the held-out stress set. This experiment tests whether
features built from the equation of motion separate load from friction, and checks the result
on stress sets generated **after** the method was frozen.

The standard results are not changed. Everything here lives in `src/residual.py`,
`scripts/build_residual_features.py`, `scripts/residual_experiment.py`, `tests/test_residual.py`
and `results/residual/`.

## Method

The controller knows the nominal arm and the payload declared by the operator. Subtracting the
torque this nominal model needs from the applied torque leaves what it cannot explain:

$$r = \tau_{\text{applied}} - \big[M(q)\ddot q + C(q,\dot q)\dot q + G(q)\big]_{m_{\text{declared}}}
    = \tau_f(\dot q) + \Delta m\, y(q,\dot q,\ddot q) + \text{noise}$$

The dynamics are linear in the payload mass (checked by a test), so
$y = \hat\tau(m{=}1) - \hat\tau(m{=}0)$. Friction depends on velocity and a payload error on pose
and acceleration, so in every 1 s window a least-squares fit with unknowns
$[b_1, \tau_{c1}, b_2, \tau_{c2}, \Delta m]$ separates them. A weak Tikhonov prior (5 equivalent
samples per unknown) pulls the estimate towards nominal when a joint barely moves.

Five dimensionless features per window:

| Feature | Meaning |
|---|---|
| `j1_fric_ratio`, `j2_fric_ratio` | payload-corrected residual projected on the motion direction, divided by the nominal friction, over samples where the joint moves (\|qd\| > 0.1 rad/s). ≈ 1 when healthy, ≈ friction multiplier under a fault |
| `dm_hat` | estimated payload error [kg] |
| `j1_moving_frac`, `j2_moving_frac` | fraction of the window in which the joint moves (how much the ratio can be trusted) |

Acceleration comes from a smoothed difference of the measured velocity,
$\ddot q_i = (\dot q_{i+K} - \dot q_{i+1-K}) / ((2K-1)\,\Delta t)$ with $K = 4$. A window ending at
sample $e$ only uses samples up to $e-K$, so the features stay causal. $K$ was chosen on train
runs only (median ratio error 0.049 at $K=4$, against 0.192 at $K=1$).

Two detectors use these features:

- **Physics rule** (no machine learning): score $= \max_j$ `fric_ratio_j`. Window, threshold
  (1.1–2.5) and persistence were chosen on validation with the same criterion as the forest:
  maximise F1 with at most 0.5 false alarms per healthy minute. Result: threshold 1.3, 1 window.
- **Random Forest** on the residual features, trained and tuned exactly like the original one.

## How the method was developed (what was seen when)

| Step | Data looked at | Outcome |
|---|---|---|
| v1: 13 features incl. residual magnitudes, target acceleration | test, old stress, **stress2** (one dry run) | test F1 0.99, but stress false alarms *worse* (11–18 / healthy min) |
| Diagnosis | old stress only | the forest used magnitudes that grow with speed; and the arm lags its target, so using the target acceleration leaves $M\,\delta\,\omega^2\,\dot q$ in the residual, which looks exactly like viscous friction and grows with frequency² |
| v2: dimensionless features only, measured acceleration, $K$ on train | old stress (development) | physics rule 40/40, 0.24 false alarms / healthy min |
| Method frozen, **stress3** generated (seed 20261121) | stress3 | results below |

So the old stress split is now a development set, stress2 was seen once with the discarded v1,
and **stress3 is the only truly held-out check**. All three are reported.

## Results

`results/residual/summary.csv`, produced by `python -m scripts.residual_experiment`.
"Healthy runs w/ FA" = healthy runs with at least one false alarm.

| Set | Detector | Faults detected | False alarms / healthy min | Healthy runs w/ FA | Median delay | Window F1 |
|---|---|---|---|---|---|---|
| Test | RF, original features | 32/32 | 0.21 | 0/16 | 1.48 s | 0.82 |
| Test | RF, residual features | 32/32 | 0.00 | 0/16 | 0.30 s | 0.99 |
| Test | Physics rule | 32/32 | 0.00 | 0/16 | 0.16 s | 0.99 |
| **Stress3 (held out)** | RF, original features | 24/40 | 4.74 | 20/20 | 1.08 s | 0.61 |
| **Stress3 (held out)** | **RF, residual features** | **40/40** | **0.08** | **1/20** | 0.64 s | **0.95** |
| **Stress3 (held out)** | Physics rule | 40/40 | 0.41 | 3/20 | 0.36 s | 0.95 |
| Stress3 (held out) | Error-threshold baseline | 21/40 | 3.52 | 16/20 | 4.03 s | 0.47 |
| Stress2 (seen with v1) | RF, original features | 12/40 | 5.65 | 20/20 | 2.41 s | 0.71 |
| Stress2 (seen with v1) | RF, residual features | 40/40 | 0.42 | 5/20 | 0.35 s | 0.95 |
| Stress2 (seen with v1) | Physics rule | 40/40 | 1.16 | 8/20 | 0.27 s | 0.95 |
| Old stress (development) | RF, residual features | 40/40 | 0.08 | 0/20 | 0.63 s | 0.91 |

The "original features" rows reproduce the published standard numbers on test and old stress
(32/32, 0.21; 26/40, 3.88), so the training code change did not alter the original model.
Combining original and residual features ("both" in `summary.csv`) is no better than residual
features alone.

## What this means

- Built from the equation of motion, the features separate load and speed from friction. On a
  stress set generated after the method was frozen, false alarms drop from 4.7 to 0.08 per healthy
  minute and every fault is caught.
- A one-line physics rule without machine learning already does most of the work. The forest adds
  some robustness (fewer false alarms), not the core idea.
- Every remaining false alarm on stress2 and stress3 happens on **point-to-point** trajectories,
  which are never in the training data. While the arm stops and starts, friction is barely
  observable and the acceleration estimate is worst. stress2 has 16 of 20 healthy runs on p2p
  trajectories, stress3 only 4, which explains most of the gap between them.
- The first version failed, and the failure is reported. It showed that a "physics-informed"
  feature can still leak speed into the detector through the acceleration estimate.

## Limitations

- Still 100 % simulation, with the same arm model in the simulator and in the residual. On a
  real arm the nominal model is never exact; an identification step would be needed first.
- Only the detector was re-done. The diagnosis stage (which joint, how much) was not retrained
  on residual features, although `fric_ratio_j` is already a direct estimate of the multiplier.
- The old stress split and stress2 influenced the method. Only stress3 (60 runs) is fully
  held out, and no confidence intervals are computed.
- The forest is not wired into the desktop or web demo; they still use the standard model.

## Reproduce

```
python -m scripts.build_residual_features --data data/standard
python -m scripts.generate_dataset --mode standard --name stress2 --subsets stress --seed 20261120
python -m scripts.generate_dataset --mode standard --name stress3 --subsets stress --seed 20261121
# for each of stress2, stress3: make_splits, build_features, build_residual_features
python -m scripts.residual_experiment --data data/standard --fresh data/stress2 data/stress3
python -m pytest tests/test_residual.py
```
