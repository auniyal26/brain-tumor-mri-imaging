from __future__ import annotations

import os, sys
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# ---- make imports work from anywhere ----
IMAGING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Imaging/
sys.path.insert(0, IMAGING_ROOT)

from src.data import make_datasets_4class, BinaryTransformSubset, get_transforms
from src.models import build_stageA
from src.train import train_model
from src.eval import metrics_binary, save_confusion_matrix_png, write_report_txt
from src.infer import infer_stageA_threshold
from src.utils import set_seed, clear_dir, write_json, read_json, get_device


def banner(msg):
    print("\n" + "█"*80)
    print("█ " + msg)
    print("█"*80)

def run(config_path: str):
    import sys
    sys.argv = [sys.argv[0], "--config", config_path]
    main()


def plot_curves(hist, out_loss: str, out_acc: str, title_prefix: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(hist.train_loss, label="train_loss")
    plt.plot(hist.val_loss, label="val_loss")
    plt.legend()
    plt.title(f"{title_prefix} Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(out_loss, dpi=200)
    plt.close()

    plt.figure()
    plt.plot(hist.train_acc, label="train_acc")
    plt.plot(hist.val_acc, label="val_acc")
    plt.legend()
    plt.title(f"{title_prefix} Acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.tight_layout()
    plt.savefig(out_acc, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(IMAGING_ROOT, "configs", "stageAB.json"),
        help="Path to config json (uses epochs_A, threshold_no, etc.)",
    )
    args = parser.parse_args()

    cfg0 = read_json(args.config)

    SEED = int(cfg0["seed"])
    VAL_FRAC = float(cfg0["val_frac"])
    IMG_SIZE = int(cfg0["img_size"])
    BATCH_SIZE = int(cfg0["batch_size"])
    EPOCHS_A = int(cfg0["epochs_A"])
    LR = float(cfg0["lr"])
    THRESH_NO = float(cfg0["threshold_no"])

    TRAIN_DIR = os.path.join(IMAGING_ROOT, "Data", "Training")
    TEST_DIR  = os.path.join(IMAGING_ROOT, "Data", "Testing")

    ART_ROOT  = os.path.abspath(os.path.join(IMAGING_ROOT, "..", "Artifacts"))  # LifeReset/Artifacts
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

    NO = class_to_idx["no_tumor"]

    print("Classes:", classes)
    print("NO idx:", NO)
    print("Train/Val sizes:", len(train_idx), len(val_idx))
    print("Train counts:", dict(zip(classes, np.bincount(targets[train_idx], minlength=len(classes)).tolist())))
    print("Val counts:  ", dict(zip(classes, np.bincount(targets[val_idx],   minlength=len(classes)).tolist())))

    train_tf = get_transforms(IMG_SIZE, train=True)
    eval_tf  = get_transforms(IMG_SIZE, train=False)

    # loaders for Stage A training (binary labels)
    trainA_ds = BinaryTransformSubset(base_train, train_idx, train_tf, NO)
    valA_ds   = BinaryTransformSubset(base_train, val_idx,   eval_tf,  NO)

    trainA_loader = DataLoader(trainA_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    valA_loader   = DataLoader(valA_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # loaders for Stage A evaluation on original 4-class val/test (mapped inside infer)
    val4_loader  = DataLoader(bundle.val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test4_loader = DataLoader(bundle.test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # write effective config into run dir (so every run is self-contained)
    cfg = {
        "config_path": os.path.abspath(args.config),
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_A": EPOCHS_A,
        "lr": LR,
        "threshold_no": THRESH_NO,
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

    banner("STAGE A: train binary gate (no_tumor vs tumor)")
    modelA = build_stageA(pretrained=True).to(device)

    # weighted CE for binary imbalance
    train_labels_bin = np.array([0 if int(targets[i]) == NO else 1 for i in train_idx], dtype=np.int64)
    countsA = np.bincount(train_labels_bin, minlength=2)  # [no_tumor, tumor]
    wA = countsA.sum() / (2 * np.maximum(countsA, 1))
    wA = torch.tensor(wA, dtype=torch.float32, device=device)
    print("StageA counts [no, tumor]:", countsA.tolist())
    print("StageA class weights:", wA.detach().cpu().numpy().tolist())

    criterionA = nn.CrossEntropyLoss(weight=wA)
    optimizerA = torch.optim.Adam(modelA.parameters(), lr=LR)

    histA = train_model(
        model=modelA,
        train_loader=trainA_loader,
        val_loader=valA_loader,
        criterion=criterionA,
        optimizer=optimizerA,
        device=device,
        epochs=EPOCHS_A,
    )

    # curves
    plot_curves(
        histA,
        out_loss=os.path.join(RUN_DIR, "stageA_loss.png"),
        out_acc=os.path.join(RUN_DIR, "stageA_acc.png"),
        title_prefix="Stage A",
    )

    # eval with strict threshold:
    # - valA_loader already yields binary labels
    # - val4/test4 loaders yield original labels and will be mapped
    banner("STAGE A: eval (strict threshold gate)")
    val_true, val_pred = infer_stageA_threshold(
        modelA, valA_loader, device=device, threshold_no=THRESH_NO, no_class_idx=NO, y_is_multiclass=False
    )
    test_true, test_pred = infer_stageA_threshold(
        modelA, test4_loader, device=device, threshold_no=THRESH_NO, no_class_idx=NO, y_is_multiclass=True
    )

    val_metrics = metrics_binary(val_true, val_pred, ["no_tumor", "tumor"])
    test_metrics = metrics_binary(test_true, test_pred, ["no_tumor", "tumor"])
    write_json(os.path.join(RUN_DIR, "stageA_val_metrics.json"), val_metrics)
    write_json(os.path.join(RUN_DIR, "stageA_test_metrics.json"), test_metrics)

    cm_val  = confusion_matrix(val_true,  val_pred,  labels=[0, 1])
    cm_test = confusion_matrix(test_true, test_pred, labels=[0, 1])

    save_confusion_matrix_png(cm_val,  ["no_tumor", "tumor"], os.path.join(RUN_DIR, "stageA_val_confusion_matrix.png"),  "Stage A (Val)")
    save_confusion_matrix_png(cm_test, ["no_tumor", "tumor"], os.path.join(RUN_DIR, "stageA_test_confusion_matrix.png"), "Stage A (Test)")

    write_report_txt(val_true,  val_pred,  ["no_tumor", "tumor"], os.path.join(RUN_DIR, "stageA_val_report.txt"))
    write_report_txt(test_true, test_pred, ["no_tumor", "tumor"], os.path.join(RUN_DIR, "stageA_test_report.txt"))

    torch.save(modelA.state_dict(), os.path.join(RUN_DIR, "stageA_resnet18.pt"))

    banner("RESULTS")
    print("Stage A VAL:", val_metrics)
    print("Stage A TEST:", test_metrics)
    print("DONE. Outputs in:", os.path.abspath(RUN_DIR))


if __name__ == "__main__":
    main()
