# Imaging Results Snapshot (Brain Tumor MRI)

## Runs
- Baseline (4-class ResNet18): `results/baseline_4class/`
- New (Stage AB + calibrated threshold): `results/new_stageAB_calibrated/`

## Comparison (TEST)

| Run | acc | balanced acc | macro F1 | glioma recall | meningioma recall | no_tumor recall | pituitary recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline 4-class | 0.4467 | 0.4314 | 0.3943 | 0.09 | 0.3652 | 1.00 | 0.2703 |
| New Stage AB (calibrated) | 0.5635 | 0.5644 | 0.5401 | 0.19 | 0.8957 | 0.4286 | 0.7432 |
| Run variance v1 (same code, THRESH_NO=0.07) | 0.4264 | 0.4364 | 0.3980 | 0.20 | 0.1217 | 0.9238 | 0.50 |
| Robust Aug v1 (MRI intensity/grayscale, THRESH_NO=0.87) | 0.4924 | 0.5076 | 0.4552 | 0.17 | 0.2609 | 0.9238 | 0.6757 |

## Notes (5 bullets)
- Improved: Stage AB (calibrated) is still the best TEST run so far on acc/balanced/macro F1.
- Improved: Robust Aug v1 improved over the variance run on TEST acc/balanced/macro F1, but not enough.
- Didn’t: Robust Aug v1 still generalizes poorly for glioma + meningioma on TEST.
- Didn’t: Stage A remains the bottleneck — when tumor recall drops, Stage AB collapses downstream.
- Next: constrain threshold selection (minimum tumor recall on VAL) and/or make Stage B train on harder “tumor-only” augmentations without touching Stage A.

