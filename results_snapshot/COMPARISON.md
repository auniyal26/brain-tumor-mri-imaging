# Imaging Results Snapshot (Brain Tumor MRI)

## Scoreboard (TEST) — best runs

**Best overall (right now):**
- **Stage B weighted CE v1** → acc **0.6244**, bal **0.6342**, macroF1 **0.6056**
  - recalls: glioma **0.28**, meningioma **0.5739**, no_tumor **0.8857**, pituitary **0.7973**

**Runner-up:**
- **Constrained threshold v2** → acc **0.6193**, bal **0.6205**, macroF1 **0.5865**
  - recalls: glioma **0.22**, meningioma **0.6696**, no_tumor **0.8762**, pituitary **0.7162**

**Today’s controlled lever (threshold selection = macroF1):**
- **Threshold macroF1 v1** → acc **0.5406**, bal **0.5436**, macroF1 **0.5093**
  - recalls: glioma **0.20**, meningioma **0.4957**, no_tumor **0.8571**, pituitary **0.6216**


## Comparison (TEST)

| Run                                                                |    acc | balanced acc | macro F1 | glioma recall | meningioma recall | no_tumor recall | pituitary recall |
| ------------------------------------------------------------------ | -----: | -----------: | -------: | ------------: | ----------------: | --------------: | ---------------: |
| Baseline 4-class                                                   | 0.4467 |       0.4314 |   0.3943 |        0.0900 |            0.3652 |          1.0000 |           0.2703 |
| New Stage AB (calibrated)                                          | 0.5635 |       0.5644 |   0.5401 |        0.1900 |            0.8957 |          0.4286 |           0.7432 |
| Run variance v1 (same code, THRESH_NO=0.07)                        | 0.4264 |       0.4364 |   0.3980 |        0.2000 |            0.1217 |          0.9238 |           0.5000 |
| Robust Aug v1 (MRI intensity/grayscale, THRESH_NO=0.87)            | 0.4924 |       0.5076 |   0.4552 |        0.1700 |            0.2609 |          0.9238 |           0.6757 |
| Constrained threshold v2 (min tumor recall on VAL, THRESH_NO=0.54) | 0.6193 |       0.6205 |   0.5865 |        0.2200 |            0.6696 |          0.8762 |           0.7162 |
| Stage B weighted CE v1                                             | 0.6244 |       0.6342 |   0.6056 |        0.2800 |            0.5739 |          0.8857 |           0.7973 |
| Threshold macroF1 v1 (THRESH_NO=0.92)                              | 0.5406 |       0.5436 |   0.5093 |        0.2000 |            0.4957 |          0.8571 |           0.6216 |

## Notes (5 bullets)

- Improved: Stage AB TEST acc (0.541) and macro F1 (0.509) increased vs Robust Aug v1.

- Improved: meningioma recall improved to 0.496 while keeping no_tumor recall high (0.857).

- Didn’t: glioma recall still low (0.20) — main remaining failure mode.

- Didn’t: balanced acc only slightly improved vs calibrated best; still far from “clinical-ready”.

- Next: keep this threshold rule, then do ONE lever next time on Stage B robustness (tumor-only intensity aug) or Stage B loss (label smoothing only).