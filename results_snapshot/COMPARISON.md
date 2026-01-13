# Imaging Results Snapshot (Brain Tumor MRI)

## Runs
- Baseline (4-class ResNet18): `results/baseline_4class/`
- New (Stage AB + calibrated threshold): `results/new_stageAB_calibrated/`

## Comparison (TEST)

| Run | acc | balanced acc | macro F1 | glioma recall | meningioma recall | no_tumor recall | pituitary recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline 4-class | 0.4467 | 0.4314 | 0.3943 | 0.09 | 0.3652 | 1.00 | 0.2703 |
| New Stage AB (calibrated) | 0.5635 | 0.5644 | 0.5401 | 0.19 | 0.8957 | 0.4286 | 0.7432 |

## Notes (5 bullets)
- Improved: overall TEST acc and macro F1 increased substantially vs baseline.
- Improved: meningioma and pituitary recall increased strongly with Stage AB.
- Didn’t: glioma recall is still low (main failure mode on TEST).
- Didn’t: no_tumor recall dropped after calibration (threshold trade-off).
- Next: run ONE robustness lever focused on test shift (MRI intensity/grayscale augmentation) and re-evaluate.
