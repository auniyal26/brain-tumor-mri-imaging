# Imaging Results Snapshot (Brain Tumor MRI)

## Runs
- Baseline (4-class ResNet18): `results/baseline_4class/`
- New (Stage AB + calibrated threshold): `results/new_stageAB_calibrated/`

## Comparison (TEST)

| Run | acc | balanced acc | macro F1 | glioma recall | meningioma recall | no_tumor recall | pituitary recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline 4-class | 0.4467 | 0.4314 | 0.3943 | 0.09 | 0.3652 | 1.00 | 0.2703 |
| New Stage AB (calibrated) | 0.5635 | 0.5644 | 0.5401 | 0.19 | 0.8957 | 0.4286 | 0.7432 |
| Run variance v1 v1 (MRI intensity/grayscale, THRESH_NO=0.07) | 0.4264 | 0.4364 | 0.3980 | 0.20 | 0.1217 | 0.9238 | 0.50 |

## Notes (5 bullets)
- Improved: Stage AB (calibrated) is clearly better than the baseline on TEST acc and macro F1.
- Improved: calibration can shift class trade-offs (e.g., meningioma/pituitary recall improved in the calibrated run).
- Didn’t: the robustness aug v1 run degraded TEST performance (gate over-called no_tumor; Stage A tumor recall collapsed on TEST).
- Didn’t: glioma recall remains a major failure mode on TEST across runs.
- Next: constrain threshold selection (e.g., require minimum tumor recall on VAL) or replace threshold tuning with temperature scaling; then re-test.

