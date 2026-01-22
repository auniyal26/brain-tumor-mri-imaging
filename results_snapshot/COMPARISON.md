# Imaging Results Snapshot (Brain Tumor MRI)

## Runs

* Baseline (4-class ResNet18): `results_snapshot/baseline_4class/`
* New (Stage AB + calibrated threshold): `results_snapshot/new_stageAB_calibrated/`
* Run variance v1 (same code, THRESH_NO=0.07): `results_snapshot/run_variance_v1_thresh0p07/`
* Robust Aug v1 (MRI intensity/grayscale, THRESH_NO=0.87): `results_snapshot/robust_aug_v1/`
* Constrained threshold v2 (min tumor recall on VAL, THRESH_NO=0.54): `results_snapshot/thresh_rule_v2_T0p54/`

## Comparison (TEST)

| Run                                                                |    acc | balanced acc | macro F1 | glioma recall | meningioma recall | no_tumor recall | pituitary recall |
| ------------------------------------------------------------------ | -----: | -----------: | -------: | ------------: | ----------------: | --------------: | ---------------: |
| Baseline 4-class                                                   | 0.4467 |       0.4314 |   0.3943 |        0.0900 |            0.3652 |          1.0000 |           0.2703 |
| New Stage AB (calibrated)                                          | 0.5635 |       0.5644 |   0.5401 |        0.1900 |            0.8957 |          0.4286 |           0.7432 |
| Run variance v1 (same code, THRESH_NO=0.07)                        | 0.4264 |       0.4364 |   0.3980 |        0.2000 |            0.1217 |          0.9238 |           0.5000 |
| Robust Aug v1 (MRI intensity/grayscale, THRESH_NO=0.87)            | 0.4924 |       0.5076 |   0.4552 |        0.1700 |            0.2609 |          0.9238 |           0.6757 |
| Constrained threshold v2 (min tumor recall on VAL, THRESH_NO=0.54) | 0.6193 |       0.6205 |   0.5865 |        0.2200 |            0.6696 |          0.8762 |           0.7162 |

## Notes (5 bullets)

* Improved: **Constrained threshold v2** is now best overall on TEST acc/balanced/macro F1 (**0.6193 / 0.6205 / 0.5865**).
* Improved: Stage A gate is no longer “dumb”: Stage A VAL tumor recall **0.9818** and Stage A TEST balanced acc **0.8204** (gate stability improved).
* Didn’t: TEST **glioma recall is still very low** (best so far is only **0.22**), and v2 still collapses glioma specifically.
* Didn’t: This confirms the bottleneck has shifted from **thresholding** to **Stage B glioma generalization** (not a gate threshold issue now).
* Next: run **one Stage B-only lever** (e.g., class-weighted/focal loss *or* balanced sampler) to raise glioma recall without touching Stage A or augmentations.
