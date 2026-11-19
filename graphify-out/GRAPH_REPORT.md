# Graph Report - robot_friction_detection  (2026-11-20)

## Corpus Check
- 128 files · ~69,530 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 70 file(s) not represented in the graph (top: .csv 48, .gz 12, .joblib 4)
- WARNING: 70 of 198 files seen (35%) have no supported extractor and are missing from this graph entirely. It may look complete while representing only part of the codebase.

## Summary
- 645 nodes · 1590 edges · 45 communities (36 shown, 9 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 157 edges (avg confidence: 0.88)
- Token cost: 1,561,381 input · 0 output

## Community Hubs (Navigation)
- Script Entry Points
- Physics-Residual Experiment
- Technical Report Concepts
- Pygame Desktop Demo
- Controller and Web Export
- Alarm and Evaluation Metrics
- Report Generation
- Feature Building Scripts
- Arm Dynamics Model
- Plotting and Figures
- Online Detector and Alarm
- Reference Trajectories
- Demo Screenshot UI
- Physics Tests and Divergence
- Residual Tests and Torque
- Smoke Run Report
- Scenario Sampling
- Example Run R0176
- Data and Evaluation Design
- Closed-Loop Simulator
- RF vs Baseline Figure
- Example Run R0001 (Healthy)
- Feature Importance Figure
- Fault Schedule
- Model Loading Errors
- Study Guide and FAQ
- Example Run R0111
- Example Run R0137
- Example Run R0191
- Project Overview and Reproducibility
- Confusion Matrices Figure
- Detection by Scenario Figure
- False Alarms by Scenario Figure
- Diagnosis Estimates Figure
- Load vs Friction Signature
- Physics Simulation Stage
- Detection Delay Figure
- Pipeline Runner
- Online Residual Computation
- Moving Fraction Feature
- Sign Conventions
- Group Split Convention

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 33 edges
2. `ArmModel` - 30 edges
3. `FaultSchedule` - 28 edges
4. `Simulator` - 27 edges
5. `DemoApp` - 26 edges
6. `SinusoidTrajectory` - 20 edges
7. `load_manifest()` - 18 edges
8. `RobotParams` - 18 edges
9. `OnlineDetector` - 15 edges
10. `sample_run_specs()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `JointGuard AI technical report (PDF)` --references--> `JointGuard AI landing page (web/index.html)`  [INFERRED]
  reports/latex/jointguard_report.pdf → web/index.html
- `fric_ratio_j feature (payload-corrected friction ratio)` --semantically_similar_to--> `Tracking error along direction of motion (e_dir)`  [INFERRED] [semantically similar]
  docs/RESIDUAL_EXPERIMENT.md → README.md
- `stress3 set (seed 20261121, only truly held-out)` --semantically_similar_to--> `Out-of-distribution stress test set`  [INFERRED] [semantically similar]
  docs/RESIDUAL_EXPERIMENT.md → README.md
- `Implemented / verified / assumed status table` --semantically_similar_to--> `Limitations (simulation only, single fault type, uncalibrated scores, not RUL)`  [INFERRED] [semantically similar]
  docs/HUONG_DAN_HOC.md → reports/standard/report_en.md
- `drawFriction()` --implements--> `Friction fault model: multiplier m on viscous + tanh Coulomb friction`  [INFERRED]
  web/index.html → reports/latex/jointguard_report.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Simulation-to-alarm detection pipeline** — readme_two_joint_arm_simulator, readme_pd_gravity_controller, readme_friction_fault_model, readme_window_features, readme_random_forest_detector, readme_alarm_rule, readme_diagnosis_stage [EXTRACTED 1.00]
- **Physics-residual feature method** — docs_residual_experiment_torque_residual, docs_residual_experiment_least_squares_fit, docs_residual_experiment_smoothed_acceleration, docs_residual_experiment_fric_ratio, docs_residual_experiment_dm_hat, docs_residual_experiment_moving_frac [EXTRACTED 1.00]
- **Leakage-free evaluation protocol** — readme_group_based_splits, readme_stress_test_set, readme_event_level_metrics, docs_residual_experiment_stress3, progress_n_on_grid_boundary [INFERRED 0.85]
- **Five-stage pipeline: simulation, data, features/ML, evaluation, GUI** — docs_huong_dan_hoc_stage_a_physics_simulation, docs_huong_dan_hoc_stage_b_data_generation, docs_huong_dan_hoc_stage_c_features_ml, docs_huong_dan_hoc_stage_d_independent_evaluation, docs_huong_dan_hoc_stage_e_pygame_gui [EXTRACTED 1.00]
- **Label-leakage prevention measures** — reports_standard_report_en_anomaly_definition, reports_standard_report_en_sensor_noise_model, reports_standard_report_en_group_split, reports_standard_report_en_signals_truth_separation, reports_standard_report_en_causal_window_features [INFERRED 0.85]
- **Out-of-distribution failure and proposed remedies** — reports_standard_report_en_stress_test_results, reports_standard_report_en_ood_generalisation_failure, reports_standard_report_en_model_residual_feature, reports_standard_report_en_ood_detection, docs_huong_dan_hoc_demo_fast_mode_false_alarms [INFERRED 0.85]
- **JointGuard detection-to-diagnosis pipeline** — reports_latex_jointguard_report_causal_window_features, reports_latex_jointguard_report_random_forest_detector, reports_latex_jointguard_report_alarm_rule, reports_latex_jointguard_report_diagnosis_stage [EXTRACTED 1.00]
- **Physics simulation stack (dynamics, controller, fault model)** — reports_latex_jointguard_report_two_joint_arm_dynamics, reports_latex_jointguard_report_pd_controller, reports_latex_jointguard_report_friction_fault_model [EXTRACTED 1.00]
- **Landing-page figures driven by demo_runs.json** — web_index_buildplayer, web_index_buildstrip, web_index_buildimportance, web_index_drawscatter, web_index_drawschedule, web_index_drawdiag [EXTRACTED 1.00]
- **Detector evaluation metric suite (F1, detection rate, false alarms, delay)** — results_standard_figures_comparison_rf_vs_baseline_window_f1, results_standard_figures_comparison_rf_vs_baseline_run_detection_rate, results_standard_figures_comparison_rf_vs_baseline_false_alarms_per_healthy_min, results_standard_figures_comparison_rf_vs_baseline_median_detection_delay [EXTRACTED 1.00]
- **Detector comparison across TEST and STRESS sets** — results_standard_figures_confusion_matrices_random_forest_detector, results_standard_figures_confusion_matrices_tracking_error_baseline, results_standard_figures_confusion_matrices_test_set, results_standard_figures_confusion_matrices_stress_set, results_standard_figures_confusion_matrices_alarm_state [EXTRACTED 1.00]
- **RF vs Baseline per-scenario detection comparison** — results_standard_figures_detection_by_scenario_random_forest_detector, results_standard_figures_detection_by_scenario_tracking_error_baseline, results_standard_figures_detection_by_scenario_fault_scenarios, results_standard_figures_detection_by_scenario_per_scenario_detection_rate [EXTRACTED 1.00]
- **RF vs Baseline detection-delay comparison across TEST and STRESS** — results_standard_figures_detection_delays_random_forest_detector, results_standard_figures_detection_delays_tracking_error_baseline, results_standard_figures_detection_delays_detection_delay_metric, results_standard_figures_detection_delays_test_set_result, results_standard_figures_detection_delays_stress_set_result [EXTRACTED 1.00]
- **Diagnosis stage outputs evaluated at alarm onset** — results_standard_figures_diagnosis_estimates_alarm_onset, results_standard_figures_diagnosis_estimates_faulty_joint_identification, results_standard_figures_diagnosis_estimates_friction_multiplier_estimation [INFERRED 0.85]
- **Signals feeding friction-fault score in R0001** — results_standard_figures_example_test_r0001_applied_torque, results_standard_figures_example_test_r0001_tracking_error, results_standard_figures_example_test_r0001_random_forest_model, results_standard_figures_example_test_r0001_alarm_threshold [INFERRED 0.75]
- **Signals to score to alarm detection flow (R0111)** — results_standard_figures_example_test_r0111_applied_torque, results_standard_figures_example_test_r0111_tracking_error, results_standard_figures_example_test_r0111_rf_model_score, results_standard_figures_example_test_r0111_alarm_threshold, results_standard_figures_example_test_r0111_detection_delay [INFERRED 0.85]
- **R0137 fault-to-alarm detection flow** — results_standard_figures_example_test_r0137_severe_j1_fault, results_standard_figures_example_test_r0137_applied_torque, results_standard_figures_example_test_r0137_tracking_error, results_standard_figures_example_test_r0137_rf_model_score, results_standard_figures_example_test_r0137_alarm_threshold, results_standard_figures_example_test_r0137_detection_delay [INFERRED 0.85]
- **R0176 signals-to-alarm detection flow** — joint_friction_fault_injection, results_standard_figures_example_test_r0176_applied_torque, results_standard_figures_example_test_r0176_tracking_error, random_forest_fault_detector, alarm_threshold_0_40, detection_delay [INFERRED 0.85]
- **Signals to score to alarm detection flow** — results_standard_figures_example_test_r0191_applied_torque, results_standard_figures_example_test_r0191_tracking_error, results_standard_figures_example_test_r0191_rf_model_score, results_standard_figures_example_test_r0191_alarm_threshold, results_standard_figures_example_test_r0191_detection_delay [INFERRED 0.85]
- **RF vs baseline false-alarm comparison across TEST and STRESS sets** — results_standard_figures_false_alarms_by_scenario_random_forest, results_standard_figures_false_alarms_by_scenario_baseline_tracking_error, results_standard_figures_false_alarms_by_scenario_test_set, results_standard_figures_false_alarms_by_scenario_stress_set, results_standard_figures_false_alarms_by_scenario_false_alarm_rate [EXTRACTED 1.00]
- **Window feature groups ranked by Random Forest** — results_standard_figures_feature_importance_tracking_error_features, results_standard_figures_feature_importance_command_features, results_standard_figures_feature_importance_torque_features [EXTRACTED 1.00]
- **Load vs friction fault signature in feature space** — results_standard_figures_signature_load_vs_friction_torque_rms, results_standard_figures_signature_load_vs_friction_tracking_error_along_motion, results_standard_figures_signature_load_vs_friction_friction_x3_fault, results_standard_figures_signature_load_vs_friction_payload_variation [EXTRACTED 1.00]
- **Signals -> RF score -> debounced alarm -> fault readout** — results_screenshots_headless_04_after_fault_applied_torque_plot, results_screenshots_headless_04_after_fault_tracking_error_plot, results_screenshots_headless_04_after_fault_model_score_plot, results_screenshots_headless_04_after_fault_alarm_debounce, results_screenshots_headless_04_after_fault_ai_prediction_panel [INFERRED 0.85]

## Communities (45 total, 9 thin omitted)

### Community 0 - "Script Entry Points"
Cohesion: 0.07
Nodes (40): main(), run_diagnoses(), summarize(), main(), main(), main(), pick_examples(), main() (+32 more)

### Community 1 - "Physics-Residual Experiment"
Cohesion: 0.05
Nodes (48): dm_hat payload error feature, fric_ratio_j feature (payload-corrected friction ratio), Per-window least-squares fit [b1, tc1, b2, tc2, dm] with Tikhonov prior, Physics rule detector (score = max fric_ratio, threshold 1.3), Remaining false alarms on point-to-point trajectories, Physics-residual features experiment, Random Forest on residual features, Causal smoothed-difference acceleration estimate (K=4) (+40 more)

### Community 2 - "Technical Report Concepts"
Cohesion: 0.07
Nodes (24): 24 causal window features (12 per joint, 1.0 s window every 0.1 s), Detection results: 32/32 test, 3.88 FA/min on stress, Diagnosis stage: joint classifier + friction-multiplier regressor, Error-threshold baseline (max_j RMS(e_j)/r_j), Limitations (simulation only, no generalisation, no RUL), Next step: physics-residual features (measured vs nominal-model torque), Random Forest detector (200 trees, val AP 0.927), Stress test (2-3 kg payload, fast motion, point-to-point trajectory) (+16 more)

### Community 3 - "Pygame Desktop Demo"
Cohesion: 0.12
Nodes (3): main(), main(), DemoApp

### Community 4 - "Controller and Web Export"
Cohesion: 0.14
Nodes (6): PDGravityController, forward_kinematics(), friction_torque(), RobotParams, load_diagnosis(), test_forward_kinematics_known_configs()

### Community 5 - "Alarm and Evaluation Metrics"
Cohesion: 0.15
Nodes (14): alarm_sequence(), _detection_block(), per_run_events(), prf(), summarize(), _events(), test_alarm_logic_counts_transitions(), test_delay_measured_from_defined_fault_time() (+6 more)

### Community 6 - "Report Generation"
Cohesion: 0.16
Nodes (9): main(), build(), row(), Doc, build_en(), row(), write_report_en(), f2() (+1 more)

### Community 7 - "Feature Building Scripts"
Cohesion: 0.20
Nodes (10): main(), main(), load_signals(), load_truth(), run_features(), window_end_indices(), window_labels(), window_sizes() (+2 more)

### Community 8 - "Arm Dynamics Model"
Cohesion: 0.15
Nodes (6): ArmModel, solve_sym2(), test_gravity_matches_potential_energy_gradient(), test_linear_solve_matches_numpy(), test_mass_matrix_symmetric_positive_definite(), test_static_gravity_balance()

### Community 9 - "Plotting and Figures"
Cohesion: 0.24
Nodes (13): diagnosis_figure(), main(), signature_figure(), main(), forward_kinematics_np(), by_scenario(), comparison(), confusion() (+5 more)

### Community 10 - "Online Detector and Alarm"
Cohesion: 0.16
Nodes (4): AlarmLogic, window_features(), OnlineDetector, test_online_residual_features_identical_to_offline()

### Community 11 - "Reference Trajectories"
Cohesion: 0.17
Nodes (4): PointToPointTrajectory, SinusoidTrajectory, _smoothstep(), trajectory_from_dict()

### Community 12 - "Demo Screenshot UI"
Cohesion: 0.17
Nodes (13): Abrupt x3.0 Friction Fault on Joint 2 (switched at t=14.0 s), AI Prediction Panel (ANOMALY ALARM, score 0.99), Alarm Debounce (threshold 0.40, on after 8 windows; 1.0 s window, 0.1 s update), Applied Joint Torque Plot (tau1, tau2), Detection Latency (first alarm +0.8 s after label switch), Fault Localization Readout (Joint 2 friction increase 213%, x3.1), Fixed-Step Headless Render Mode, Keyboard Controls (pause, reset, healthy/friction j1/j2/gradual, payload, speed, screenshot) (+5 more)

### Community 13 - "Physics Tests and Divergence"
Cohesion: 0.22
Nodes (11): SimulationDivergedError, make_sim(), test_controller_holds_static_pose_without_noise(), test_divergence_is_reported(), test_energy_conserved_without_friction_and_torque(), test_fault_changes_physics_not_controller(), test_friction_dissipates_energy(), test_halving_time_step_changes_little() (+3 more)

### Community 14 - "Residual Tests and Torque"
Cohesion: 0.22
Nodes (7): _torque(), window_residual_features(), _sim_features(), test_dynamics_linear_in_payload(), test_friction_fault_shows_on_the_right_joint(), test_least_squares_recovers_known_parameters(), test_payload_error_is_not_read_as_friction()

### Community 15 - "Smoke Run Report"
Cohesion: 0.24
Nodes (13): Stage C: Features and machine learning, Smoke report (VI), Smoke results: RF 9.52 FA/min, F1 0.83 vs baseline 0.81, Smoke RF config (max_depth 14, threshold 0.5, 3-window persistence), Stale qualitative remarks copied from standard run, Alarm persistence rule (8 on / 5 off windows), 24 causal past-window features (1.0 s window, 0.1 s step), Diagnosis stage: joint classifier + friction-multiplier regressor (+5 more)

### Community 16 - "Scenario Sampling"
Cohesion: 0.26
Nodes (8): parse_scenario(), _sample_p2p(), sample_run_specs(), _sample_sinusoid(), _u(), within_limits(), test_fault_onsets_vary(), test_noise_level_independent_of_scenario()

### Community 17 - "Example Run R0176"
Cohesion: 0.24
Nodes (12): Alarm Threshold 0.40, Detection Delay, False Alarm, Joint Friction Fault Injection (severe_j2), Random Forest Fault Detector, Applied Torque (Simulated) Panel, Example Test Run R0176 (severe_j2) Figure, Target vs Measured Joint Angles Panel (+4 more)

### Community 18 - "Data and Evaluation Design"
Cohesion: 0.20
Nodes (7): Stage B: Data generation, Stage D: Independent evaluation, Event-based metrics: false alarms/min, detection delay, Scenarios: healthy_speed, healthy_load, mild, severe, gradual (j1/j2), Research question: detect friction rise without false alarms on speed/load, Observed signals vs ground truth separation (signals/, truth/, manifest.csv), Held-out stress test (2-3 kg, 0.55-0.75 Hz, quintic p2p)

### Community 19 - "Closed-Loop Simulator"
Cohesion: 0.36
Nodes (4): report(), run_scenario(), step_size_check(), Simulator

### Community 20 - "RF vs Baseline Figure"
Cohesion: 0.27
Nodes (9): Random Forest vs Baseline Comparison Figure, False Alarms per Healthy Minute, Median Detection Delay [s], Random Forest Friction Detector, Detection Rate (runs), Stress Test Split, Test Split, Baseline (Tracking Error Threshold) (+1 more)

### Community 21 - "Example Run R0001 (Healthy)"
Cohesion: 0.29
Nodes (10): Alarm Threshold 0.40, Applied Torque (Simulated) Panel tau1/tau2, False Alarms (0 in healthy run), Example Test Run R0001 Figure (healthy_speed), Target vs Measured Joint Angles Panel, RF Model Score and Alarm State Panel, Random Forest Fault Detection Model, Tip Path (Forward Kinematics) Panel (+2 more)

### Community 22 - "Feature Importance Figure"
Cohesion: 0.29
Nodes (9): Commanded-Motion Features (qd_des_rms, qdd_des_rms), Random Forest Top-14 Feature Importances Figure, Impurity-Based Feature Importance (MDI), j1_e_dir_mean (top feature, ~0.138), j2_e_dir_mean (~0.093), j2_e_rms (~0.101), Random Forest Classifier, Torque Features (tau_std, tau_maxabs) (+1 more)

### Community 24 - "Model Loading Errors"
Cohesion: 0.28
Nodes (4): load_bundle(), ModelNotFoundError, history_samples(), test_missing_model_gives_clear_error()

### Community 25 - "Study Guide and FAQ"
Cohesion: 0.25
Nodes (9): HUONG_DAN_HOC (study guide, VI), Common critical questions (FAQ for defence), Demo finding: fast mode causes 2-3 false alarms (domain corner), Stage E: Pygame GUI demo, End-to-end pipeline overview (sim -> features -> RF -> alarm), Model residual / declared payload as feature (next step), Out-of-distribution detection ('uncertain' instead of alarm), Out-of-distribution failure: heavy load/speed mistaken for friction (+1 more)

### Community 26 - "Example Run R0111"
Cohesion: 0.33
Nodes (9): Example Test Run R0111 Figure, Alarm Threshold 0.40, Applied Torque (tau1, tau2), Detection Delay 4.3 s / 0 False Alarms, Target vs Measured Joint Angles, Mild Joint-2 Friction Fault (x1.74, t=7.6 s), Random Forest Model Score, Tip Path (Forward Kinematics) (+1 more)

### Community 27 - "Example Run R0137"
Cohesion: 0.33
Nodes (9): Alarm Threshold 0.40, Applied Joint Torque (tau1, tau2), Detection at 15.9 s (1.1 s delay, 0 false alarms), Example Test Run R0137 Figure, Target vs Measured Joint Angles, Random Forest Model Score (uncalibrated), Severe Joint-1 Friction Fault (x3.61 at t=14.8 s), Tip Path via Forward Kinematics (+1 more)

### Community 28 - "Example Run R0191"
Cohesion: 0.33
Nodes (9): Example Test Run R0191 Figure, Alarm Threshold 0.40, Applied Joint Torque (tau1, tau2), Detection at 9.8 s (delay 3.4 s after label threshold, 0 false alarms), Gradual Joint-1 Friction Fault (x3.40, injected t=4.8 s), Target vs Measured Joint Angles, Random Forest Model Score (uncalibrated), Tip Path via Forward Kinematics (+1 more)

### Community 29 - "Project Overview and Reproducibility"
Cohesion: 0.25
Nodes (8): Implemented / verified / assumed status table, JointGuard AI technical report (PDF), Research question: detect friction without load/speed false alarms, scripts.run_pipeline (referenced), JointGuard AI standard report (EN), JointGuard AI, Limitations (simulation only, single fault type, uncalibrated scores, not RUL), Reproducibility (seed 20261024, run_pipeline)

### Community 30 - "Confusion Matrices Figure"
Cohesion: 0.36
Nodes (6): Window-level Confusion Matrices of Alarm State (figure), Alarm State (window-level prediction), Random Forest Friction Detector, STRESS Set, TEST Set, Tracking-Error Threshold Baseline

### Community 31 - "Detection by Scenario Figure"
Cohesion: 0.36
Nodes (8): Friction Fault Scenarios (mild/severe/gradual x joint1/joint2), Detection Rate by Scenario Figure (TEST vs STRESS), Per-Scenario Detection Rate (detected/fault runs), Random Forest Detector, STRESS Set Result: RF degrades but still beats baseline on mild and severe_j1, STRESS Test Set, TEST Set Result: RF detects all faults, baseline misses mild/severe_j1/gradual_j1, Baseline (Tracking-Error Threshold)

### Community 32 - "False Alarms by Scenario Figure"
Cohesion: 0.32
Nodes (7): False Alarms by Scenario Figure, Baseline (Tracking Error Threshold), False Alarms per Healthy Minute (metric), Fault Scenarios (healthy_speed, healthy_load, mild/severe/gradual j1/j2), Random Forest Detector, STRESS Set Scenarios, TEST Set Scenarios

### Community 33 - "Diagnosis Estimates Figure"
Cohesion: 0.38
Nodes (7): Diagnosis at Alarm Onset, Faulty Joint Identification, Diagnosis Estimates Figure (true vs estimated friction multiplier), Friction Multiplier Estimation, Estimate compression bias (overestimates mild faults ~1.5-2x, underestimates severe faults >3x), STRESS set diagnosis at alarm onset (23 right joint, 3 wrong joint), TEST set diagnosis at alarm onset (30 right joint, 2 wrong joint)

### Community 34 - "Load vs Friction Signature"
Cohesion: 0.43
Nodes (7): Signature Plot: Load vs Friction (torque RMS vs tracking error), Friction x3 fault (per joint), Payload vs friction fault separability, Healthy payload variation (0.5 kg vs 2.0 kg), Joint torque RMS feature, Tracking error along motion feature, Simulated 2-joint robot arm

### Community 35 - "Physics Simulation Stage"
Cohesion: 0.53
Nodes (6): Stage A: Physics simulation, PD controller with gravity compensation (100 Hz, ZOH), Physics checks (check_physics, 17 tests), RK4 integration at 1 ms, Friction model b*q' + tau_c*tanh(q'/eps), 2-DOF planar arm dynamics M(q)q'' + C q' + G + tau_f = tau

### Community 36 - "Detection Delay Figure"
Cohesion: 0.47
Nodes (6): Detection Delay Figure (TEST vs STRESS), Detection Delay After Fault Onset, Random Forest Detector, STRESS Set: RF 26/40 detected (median ~1.8s) vs Baseline 17/40 (median ~3.5s), TEST Set: RF 32/32 detected (median ~1.5s) vs Baseline 16/32 (median ~2.7s), Baseline (Tracking-Error Threshold)

## Ambiguous Edges - Review These
- `Smoke results: RF 9.52 FA/min, F1 0.83 vs baseline 0.81` → `Stale qualitative remarks copied from standard run`  [AMBIGUOUS]
  reports/smoke/report_vi.md · relation: conceptually_related_to

## Knowledge Gaps
- **60 isolated node(s):** `Detect / Diagnose / Estimate Wear problem split`, `Fault types: mild jump, severe jump, gradual ramp`, `Group-based train/val/test splits (60/20/20)`, `Event-level metrics (false alarms, delay, detections)`, `Project web page (jointguardaii.vercel.app)` (+55 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 184 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `JointGuard AI standard report (EN)` connect `Project Overview and Reproducibility` to `Data and Evaluation Design`, `Technical Report Concepts`, `Report Generation`, `Smoke Run Report`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `ArmModel` (e.g. with `PDGravityController` and `_torque()`) actually correct?**
  _`ArmModel` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Detect / Diagnose / Estimate Wear problem split`, `Fault types: mild jump, severe jump, gradual ramp`, `Group-based train/val/test splits (60/20/20)` to the rest of the system?**
  _60 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Script Entry Points` be split into smaller, more focused modules?**
  _Cohesion score 0.07071189679885333 - nodes in this community are weakly interconnected._
- **Why does `JointGuard AI landing page (web/index.html)` connect `Technical Report Concepts` to `Project Overview and Reproducibility`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Should `Physics-Residual Experiment` be split into smaller, more focused modules?**
  _Cohesion score 0.052244897959183675 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Smoke results: RF 9.52 FA/min, F1 0.83 vs baseline 0.81` and `Stale qualitative remarks copied from standard run`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._