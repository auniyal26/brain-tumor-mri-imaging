from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

import random
import torchvision.transforms.functional as TF


# ----------------------------
# Constants
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ----------------------------
# Dataset wrappers
# ----------------------------
class TransformSubset(Dataset):
    """
    Wraps an ImageFolder with saved indices and applies a transform on-the-fly.
    Base dataset must be created with transform=None so samples yield PIL images.
    """
    def __init__(self, base_dataset: datasets.ImageFolder, indices, transform):
        self.base = base_dataset
        self.indices = list(map(int, indices))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x, y = self.base[self.indices[i]]  # PIL image
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class BinaryTransformSubset(Dataset):
    """
    Stage A: no_tumor vs tumor.
    y_bin = 0 if original == no_class_idx else 1
    """
    def __init__(self, base_dataset: datasets.ImageFolder, indices, transform, no_class_idx: int):
        self.base = base_dataset
        self.indices = list(map(int, indices))
        self.transform = transform
        self.no_class_idx = int(no_class_idx)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x, y = self.base[self.indices[i]]  # PIL
        if self.transform is not None:
            x = self.transform(x)
        y_bin = 0 if int(y) == self.no_class_idx else 1
        return x, y_bin


class TumorTransformSubset(Dataset):
    """
    Stage B: tumor subtype classification on tumor-only indices.
    Maps original {glioma, meningioma, pituitary} -> {0,1,2}.
    """
    def __init__(
        self,
        base_dataset: datasets.ImageFolder,
        indices,
        transform,
        no_class_idx: int,
        orig_to_b: Dict[int, int],
    ):
        self.base = base_dataset
        self.indices = list(map(int, indices))
        self.transform = transform
        self.no_class_idx = int(no_class_idx)
        self.orig_to_b = {int(k): int(v) for k, v in orig_to_b.items()}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x, y = self.base[self.indices[i]]  # PIL
        y = int(y)

        # Safety: Stage B should not see no_tumor
        if y == self.no_class_idx:
            raise ValueError("TumorTransformSubset received a no_tumor sample. Indices are wrong.")

        if y not in self.orig_to_b:
            raise ValueError(f"Class idx {y} not found in orig_to_b mapping.")

        if self.transform is not None:
            x = self.transform(x)

        return x, self.orig_to_b[y]


# ----------------------------
# Transforms (MRI shift targeted)
# ----------------------------
def get_transforms(img_size: int, train: bool) -> transforms.Compose:
    """
    MRI shift-targeted transforms:
    - forces grayscale consistency (MRI often stored as RGB but visually grayscale)
    - adds intensity/histogram augmentation (contrast/brightness/gamma/autocontrast/equalize)
    - keeps ImageNet normalize for pretrained ResNet18
    """

    def rand_gamma(img):
        g = random.uniform(0.7, 1.5)
        return TF.adjust_gamma(img, gamma=g)

    common = [
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
    ]

    if train:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),

            # intensity domain (MRI-relevant)
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.25, contrast=0.25)],
                p=0.70,
            ),
            transforms.RandomAutocontrast(p=0.25),
            transforms.RandomEqualize(p=0.10),
            transforms.RandomApply([transforms.Lambda(rand_gamma)], p=0.50),
        ]
        tail = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        return transforms.Compose(common + aug + tail)

    # eval: deterministic
    return transforms.Compose(common + [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ----------------------------
# Splits + bundle
# ----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_or_make_splits(
    targets: np.ndarray,
    split_dir: Path,
    seed: int,
    val_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Saves/loads:
      split_dir/train_idx.npy
      split_dir/val_idx.npy
    """
    _ensure_dir(split_dir)
    train_idx_path = split_dir / "train_idx.npy"
    val_idx_path = split_dir / "val_idx.npy"

    if train_idx_path.exists() and val_idx_path.exists():
        train_idx = np.load(train_idx_path)
        val_idx = np.load(val_idx_path)
        return train_idx.astype(np.int64), val_idx.astype(np.int64)

    idx = np.arange(len(targets), dtype=np.int64)
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_frac,
        random_state=seed,
        stratify=targets,
    )

    np.save(train_idx_path, train_idx.astype(np.int64))
    np.save(val_idx_path, val_idx.astype(np.int64))
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


@dataclass
class DatasetBundle:
    base_train: datasets.ImageFolder
    targets: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray

    train_ds: Dataset
    val_ds: Dataset
    test_ds: datasets.ImageFolder

    classes: List[str]
    class_to_idx: Dict[str, int]


def make_base_and_splits(
    train_dir: str | Path,
    split_dir: str | Path,
    seed: int,
    val_frac: float,
) -> Tuple[datasets.ImageFolder, np.ndarray, np.ndarray, np.ndarray]:
    train_dir = Path(train_dir)
    split_dir = Path(split_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    base_train = datasets.ImageFolder(str(train_dir), transform=None)
    targets = np.array([y for _, y in base_train.samples], dtype=np.int64)

    train_idx, val_idx = _load_or_make_splits(
        targets=targets,
        split_dir=split_dir,
        seed=seed,
        val_frac=val_frac,
    )

    return base_train, targets, train_idx, val_idx


def make_datasets_4class(
    train_dir: str | Path,
    test_dir: str | Path,
    split_dir: str | Path,
    img_size: int,
    seed: int,
    val_frac: float,
) -> DatasetBundle:
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    split_dir = Path(split_dir)

    if not test_dir.exists():
        raise FileNotFoundError(f"Testing directory not found: {test_dir}")

    base_train, targets, train_idx, val_idx = make_base_and_splits(
        train_dir=train_dir,
        split_dir=split_dir,
        seed=seed,
        val_frac=val_frac,
    )

    train_tf = get_transforms(img_size, train=True)
    eval_tf = get_transforms(img_size, train=False)

    train_ds = TransformSubset(base_train, train_idx, train_tf)
    val_ds = TransformSubset(base_train, val_idx, eval_tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=eval_tf)

    return DatasetBundle(
        base_train=base_train,
        targets=targets,
        train_idx=train_idx,
        val_idx=val_idx,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        classes=list(base_train.classes),
        class_to_idx=dict(base_train.class_to_idx),
    )


# ----------------------------
# Sanity check (optional)
# ----------------------------
if __name__ == "__main__":
    # Run from anywhere; resolve relative to Imaging/src/data.py
    imaging_root = Path(__file__).resolve().parents[1]  # Imaging/
    life_root = imaging_root.parent                    # LifeReset/

    TRAIN_DIR = imaging_root / "Data" / "Training"
    TEST_DIR  = imaging_root / "Data" / "Testing"
    ART_DIR   = life_root / "Artifacts"
    SPLIT_DIR = ART_DIR / "splits" / "seed42_val0.20"

    bundle = make_datasets_4class(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        split_dir=SPLIT_DIR,
        img_size=224,
        seed=42,
        val_frac=0.2,
    )

    print("Classes:", bundle.classes)
    print("Split sizes:", len(bundle.train_idx), len(bundle.val_idx), "Total:", len(bundle.targets))

    tr_counts = np.bincount(bundle.targets[bundle.train_idx], minlength=len(bundle.classes))
    va_counts = np.bincount(bundle.targets[bundle.val_idx],   minlength=len(bundle.classes))

    test_targets = np.array([y for _, y in bundle.test_ds.samples], dtype=np.int64)
    te_counts = np.bincount(test_targets, minlength=len(bundle.classes))

    print("Train counts:", dict(zip(bundle.classes, tr_counts.tolist())))
    print("Val counts:  ", dict(zip(bundle.classes, va_counts.tolist())))
    print("Test counts: ", dict(zip(bundle.test_ds.classes, te_counts.tolist())))
    print("OK: data.py sanity check passed")
