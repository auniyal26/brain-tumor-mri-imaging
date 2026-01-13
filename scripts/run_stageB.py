from __future__ import annotations

import os, sys, argparse, csv
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score

# ---- make imports work from anywhere ----
IMAGING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Imaging/
sys.path.insert(0, IMAGING_ROOT)

from src.data import make_datasets_4class, BinaryTransformSubset, TumorTransformSubset, get_transforms
from src.models import build_stageA, build_stageB
from src.train import train_model
from src.eval import metrics_binary, metrics_multiclass, save_confusion_matrix_png, write_report_txt
from src.utils import set_seed, clear_dir, write_json, read_json, get_device


def banner(msg: str) -> None:
    print("\n" + "█" * 80)
    print("█ " + msg)
    print("█" * 80)


def run(config_path: str):
    sys.argv = [sys.argv[0], "--config", config_path]
    main()


@torch.no_grad()
def collect_stageA_pno(modelA: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Returns p_no (float) and y_true (int)."""
    modelA.eval()
    pno, ytrue = [], []
    for x, y in loader:
        x = x.to(device)
        logits = modelA(x)  # [B,2] => index 0 is "no_tumor"
        p_no = torch.softmax(logits, dim=1)[:, 0].detach().cpu().numpy()
        pno.append(p_no)
        ytrue.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(pno), np.concatenate(ytrue)


@torch.no_grad()
def collect_stageB_pred_orig(modelB: nn.Module, loader: DataLoader, device, b_to_orig: list[int]) -> np.ndarray:
    """Runs modelB everywhere and maps {0,1,2} -> original class idx. (No_tumor will be overridden by threshold.)"""
    modelB.eval()
    pred_orig = []
    for x, _ in loader:
        x = x.to(device)
        logits = modelB(x)  # [B,3]
        pred_b = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        mapped = np.array([b_to_orig[i] for i in pred_b], dtype=np.int64)
        pred_orig.append(mapped)
    return np.concatenate(pred_orig)


def per_class_recall_from_cm(cm: np.ndarray, classes: list[str]) -> dict:
    rec = {}
    for i, name in enumerate(classes):
        denom = max(int(cm[i].sum()), 1)
        rec[name] = float(cm[i, i] / denom)
    return rec


def sweep_T_for_stageAB(
    p_no: np.ndarray,
    y_true: np.ndarray,
    predB_orig: np.ndarray,
    no_idx: int,
    classes: list[str],
    ts: np.ndarray | None = None,
) -> tuple[float, list[dict]]:
    """
    For each T:
      pred = no_idx if p_no >= T else predB_orig
    Choose best T by VAL balanced accuracy (multiclass).
    """
    if ts is None:
        ts = np.round(np.linspace(0.05, 0.99, 95), 2)

    rows = []
    best_bal, best_T = -1.0, None

    labels = np.arange(len(classes), dtype=np.int64)

    for T in ts:
        y_pred = np.where(p_no >= T, no_idx, predB_orig).astype(np.int64)

        bal = float(balanced_accuracy_score(y_true, y_pred))
        mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        rec = per_class_recall_from_cm(cm, classes)

        row = {
            "T": float(T),
            "balanced_acc": bal,
            "macro_f1": mf1,
            **{f"recall_{k}": v for k, v in rec.items()},
        }
        rows.append(row)

        if bal > best_bal:
            best_bal, best_T = bal, float(T)

    return best_T, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=os.path.join(IMAGING_ROOT, "configs", "stageAB.json"),
        help="Path to config json",
    )
    args = p.parse_args()

    cfg0 = read_json(args.config)

    SEED = int(cfg0["seed"])
    VAL_FRAC = float(cfg0["val_frac"])
    IMG_SIZE = int(cfg0["img_size"])
    BATCH_SIZE = int(cfg0["batch_size"])
    EPOCHS_A = int(cfg0["epochs_A"])
    EPOCHS_B = int(cfg0["epochs_B"])
    LR = float(cfg0["lr"])
    THRESH_INIT = float(cfg0["threshold_no"])

    TRAIN_DIR = os.path.join(IMAGING_ROOT, "Data", "Training")
    TEST_DIR  = os.path.join(IMAGING_ROOT, "Data", "Testing")

    ART_ROOT  = os.path.abspath(os.path.join(IMAGING_ROOT, "..", "Artifacts"))
    RUN_DIR   = os.path.join(ART_ROOT, "run")
    SPLIT_DIR = os.path.join(ART_ROOT, "splits", f"seed{SEED}_val{VAL_FRAC:.2f}")

    banner("SETUP")
    set_seed(SEED, deterministic=False)
    device = get_device()
    clear_dir(RUN_DIR)
    print("Device:", device)
    print("RUN_DIR:", os.path.abspath(RUN_DIR))

    banner("DATA + SPLIT")
    bundle = make_datasets_4class(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        split_dir=SPLIT_DIR,
        img_size=IMG_SIZE,
        seed=SEED,
        val_frac=VAL_FRAC,
    )

    base_train = bundle.base_train
    targets = bundle.targets
    train_idx = bundle.train_idx
    val_idx = bundle.val_idx
    classes = bundle.classes
    class_to_idx = bundle.class_to_idx
    num_classes = len(classes)

    NO = class_to_idx["no_tumor"]
    GL = class_to_idx["glioma_tumor"]
    ME = class_to_idx["meningioma_tumor"]
    PI = class_to_idx["pituitary_tumor"]

    print("Classes:", classes)
    print("Train/Val sizes:", len(train_idx), len(val_idx))

    train_tf = get_transforms(IMG_SIZE, train=True)
    eval_tf  = get_transforms(IMG_SIZE, train=False)

    val4_loader  = DataLoader(bundle.val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test4_loader = DataLoader(bundle.test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # persist effective config for this run
    cfg = {
        "config_path": os.path.abspath(args.config),
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_A": EPOCHS_A,
        "epochs_B": EPOCHS_B,
        "lr": LR,
        "threshold_no_init": THRESH_INIT,
        "train_dir": os.path.abspath(TRAIN_DIR),
        "test_dir": os.path.abspath(TEST_DIR),
        "split_dir": os.path.abspath(SPLIT_DIR),
        "run_dir": os.path.abspath(RUN_DIR),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(bundle.test_ds)),
        "train_counts": Counter(targets[train_idx].tolist()),
        "val_counts": Counter(targets[val_idx].tolist()),
    }
    write_json(os.path.join(RUN_DIR, "config.json"), cfg)

    # =========================
    # STAGE A (binary gate)
    # =========================
    banner("STAGE A: train (no_tumor vs tumor)")

    trainA_ds = BinaryTransformSubset(base_train, train_idx, train_tf, NO)
    valA_ds   = BinaryTransformSubset(base_train, val_idx,   eval_tf,  NO)
    trainA_loader = DataLoader(trainA_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    valA_loader   = DataLoader(valA_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    modelA = build_stageA(pretrained=True).to(device)

    train_labels_bin = np.array([0 if int(targets[i]) == NO else 1 for i in train_idx], dtype=np.int64)
    countsA = np.bincount(train_labels_bin, minlength=2)
    wA = countsA.sum() / (2 * np.maximum(countsA, 1))
    wA = torch.tensor(wA, dtype=torch.float32, device=device)

    criterionA = nn.CrossEntropyLoss(weight=wA)
    optimizerA = torch.optim.Adam(modelA.parameters(), lr=LR)

    _ = train_model(modelA, trainA_loader, valA_loader, criterionA, optimizerA, device, epochs=EPOCHS_A)
    torch.save(modelA.state_dict(), os.path.join(RUN_DIR, "stageA_resnet18.pt"))

    # =========================
    # STAGE B (tumor subtype)
    # =========================
    banner("STAGE B: train (glioma/meningioma/pituitary only)")

    orig_to_B = {GL: 0, ME: 1, PI: 2}
    B_to_orig = [GL, ME, PI]

    train_idx_B = [i for i in train_idx if int(targets[i]) != NO]
    val_idx_B   = [i for i in val_idx   if int(targets[i]) != NO]

    trainB_ds = TumorTransformSubset(base_train, train_idx_B, train_tf, NO, orig_to_B)
    valB_ds   = TumorTransformSubset(base_train, val_idx_B,   eval_tf,  NO, orig_to_B)

    train_labels_B = np.array([orig_to_B[int(targets[i])] for i in train_idx_B], dtype=np.int64)
    countsB = np.bincount(train_labels_B, minlength=3)
    class_wB = 1.0 / np.maximum(countsB, 1)
    sample_wB = torch.as_tensor(class_wB[train_labels_B], dtype=torch.double)

    samplerB = WeightedRandomSampler(weights=sample_wB, num_samples=len(sample_wB), replacement=True)
    trainB_loader = DataLoader(trainB_ds, batch_size=BATCH_SIZE, sampler=samplerB, num_workers=0)
    valB_loader   = DataLoader(valB_ds,   batch_size=BATCH_SIZE, shuffle=False,  num_workers=0)

    modelB = build_stageB(pretrained=True).to(device)
    criterionB = nn.CrossEntropyLoss()
    optimizerB = torch.optim.Adam(modelB.parameters(), lr=LR)

    _ = train_model(modelB, trainB_loader, valB_loader, criterionB, optimizerB, device, epochs=EPOCHS_B)
    torch.save(modelB.state_dict(), os.path.join(RUN_DIR, "stageB_resnet18.pt"))

    # =========================
    # ONE LEVER: Calibrate T for Stage AB on VAL
    # =========================
    banner("CALIBRATION: sweep T for Stage AB (optimize VAL balanced_acc)")

    pno_val, y_val = collect_stageA_pno(modelA, val4_loader, device)
    predB_val_orig = collect_stageB_pred_orig(modelB, val4_loader, device, B_to_orig)

    best_T, rows = sweep_T_for_stageAB(
        p_no=pno_val,
        y_true=y_val,
        predB_orig=predB_val_orig,
        no_idx=NO,
        classes=classes,
    )

    print("Best T (VAL balanced_acc):", best_T)
    write_json(os.path.join(RUN_DIR, "stageAB_best_threshold.json"), {"best_T_val": best_T})

    sweep_path = os.path.join(RUN_DIR, "stageAB_threshold_sweep_val.csv")
    with open(sweep_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", sweep_path)

    THRESH_NO = best_T

    # =========================
    # FINAL EVALS
    # =========================
    banner("EVAL: Stage A (using tuned T)")
    # Stage A metrics at tuned threshold (consistent with pipeline)
    # Binary loaders:
    valA_true, valA_pred = [], []
    testA_true, testA_pred = [], []

    # valA_loader yields binary labels already
    for x, y in valA_loader:
        pass  # just to ensure loader exists; real eval via util below

    from src.infer import infer_stageA_threshold, infer_stageAB  # keep local to avoid clutter above

    valA_true, valA_pred = infer_stageA_threshold(
        modelA, valA_loader, device=device, threshold_no=THRESH_NO, no_class_idx=NO, y_is_multiclass=False
    )
    testA_true, testA_pred = infer_stageA_threshold(
        modelA, test4_loader, device=device, threshold_no=THRESH_NO, no_class_idx=NO, y_is_multiclass=True
    )

    valA_metrics = metrics_binary(valA_true, valA_pred, ["no_tumor", "tumor"])
    testA_metrics = metrics_binary(testA_true, testA_pred, ["no_tumor", "tumor"])
    write_json(os.path.join(RUN_DIR, "stageA_val_metrics.json"), valA_metrics)
    write_json(os.path.join(RUN_DIR, "stageA_test_metrics.json"), testA_metrics)

    cmA_val  = confusion_matrix(valA_true,  valA_pred,  labels=[0, 1])
    cmA_test = confusion_matrix(testA_true, testA_pred, labels=[0, 1])
    save_confusion_matrix_png(cmA_val,  ["no_tumor", "tumor"], os.path.join(RUN_DIR, "stageA_val_confusion_matrix.png"),  "Stage A (Val)")
    save_confusion_matrix_png(cmA_test, ["no_tumor", "tumor"], os.path.join(RUN_DIR, "stageA_test_confusion_matrix.png"), "Stage A (Test)")

    banner("EVAL: Stage AB (using tuned T)")
    val_true4, val_pred4 = infer_stageAB(
        modelA, modelB, val4_loader, device=device, threshold_no=THRESH_NO, no_class_idx=NO, b_to_orig=B_to_orig
    )
    test_true4, test_pred4 = infer_stageAB(
        modelA, modelB, test4_loader, device=device, threshold_no=THRESH_NO, no_class_idx=NO, b_to_orig=B_to_orig
    )

    val_metrics = metrics_multiclass(val_true4, val_pred4, classes)
    test_metrics = metrics_multiclass(test_true4, test_pred4, classes)
    write_json(os.path.join(RUN_DIR, "stageAB_val_metrics.json"), val_metrics)
    write_json(os.path.join(RUN_DIR, "stageAB_test_metrics.json"), test_metrics)

    cm_val  = confusion_matrix(val_true4,  val_pred4,  labels=np.arange(num_classes))
    cm_test = confusion_matrix(test_true4, test_pred4, labels=np.arange(num_classes))
    save_confusion_matrix_png(cm_val,  classes, os.path.join(RUN_DIR, "stageAB_val_confusion_matrix.png"),  "Stage AB (Val)")
    save_confusion_matrix_png(cm_test, classes, os.path.join(RUN_DIR, "stageAB_test_confusion_matrix.png"), "Stage AB (Test)")

    write_report_txt(val_true4,  val_pred4,  classes, os.path.join(RUN_DIR, "stageAB_val_report.txt"))
    write_report_txt(test_true4, test_pred4, classes, os.path.join(RUN_DIR, "stageAB_test_report.txt"))

    banner("RESULTS")
    print("Tuned THRESH_NO:", THRESH_NO)
    print("Stage A VAL:", valA_metrics)
    print("Stage A TEST:", testA_metrics)
    print("Stage AB VAL:", val_metrics)
    print("Stage AB TEST:", test_metrics)
    print("DONE. Outputs in:", os.path.abspath(RUN_DIR))


if __name__ == "__main__":
    main()
