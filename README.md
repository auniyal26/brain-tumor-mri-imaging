
```md
# Imaging — Brain Tumor MRI (4-class)

This repo contains a reproducible baseline + a two-stage classifier (Stage A: no_tumor vs tumor, Stage B: tumor subtype).

## Dataset
Expected structure:
```

Imaging/Data/
Training/
glioma_tumor/
meningioma_tumor/
no_tumor/
pituitary_tumor/
Testing/
glioma_tumor/
meningioma_tumor/
no_tumor/
pituitary_tumor/

````

## Setup
Create/select venv and install deps:
```bash
pip install -r requirements.txt
````

## Run (CLI)

Interactive runner:

```bash
python Imaging/scripts/main.py
```

Choose:

* `A` for Stage A only
* `AB` for full pipeline (Stage A + Stage B + 4-class eval)

Outputs are written to:

* `LifeReset/Artifacts/run/` (cleared each run)

## Results snapshot

Versioned, non-overwritten outputs are stored in:

* `LifeReset/results/baseline_4class/`
* `LifeReset/results/new_stageAB_calibrated/`

Comparison table:

* `LifeReset/results/COMPARISON.md`
