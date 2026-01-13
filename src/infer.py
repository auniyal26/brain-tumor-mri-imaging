from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def infer_stageA_threshold(
    modelA: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold_no: float,
    no_class_idx: int,
    y_is_multiclass: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage A strict gate:
      predict no_tumor only if p(no_tumor) >= threshold_no, else tumor.

    Returns (y_true, y_pred) where y_true/y_pred are binary (0=no_tumor, 1=tumor).
    If y_is_multiclass=True, loader yields original 0..3 labels and we map to binary.
    """
    modelA.eval()
    yt, yp = [], []

    for x, y in loader:
        x = x.to(device)
        logits = modelA(x)                 # [B,2] (0=no_tumor, 1=tumor)
        probs = torch.softmax(logits, dim=1)
        p_no = probs[:, 0].cpu().numpy()   # [B]

        pred = (p_no < threshold_no).astype(np.int64)  # 1=tumor if not confidently no_tumor

        if y_is_multiclass:
            y_np = np.array(y, dtype=np.int64)
            true = (y_np != int(no_class_idx)).astype(np.int64)
        else:
            true = np.array(y, dtype=np.int64)

        yt.append(true)
        yp.append(pred)

    return np.concatenate(yt), np.concatenate(yp)


@torch.no_grad()
def infer_stageAB(
    modelA: torch.nn.Module,
    modelB: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold_no: float,
    no_class_idx: int,
    b_to_orig: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full 2-stage prediction in original 4-class label space:
      - Stage A decides no_tumor vs tumor using strict threshold on p(no_tumor)
      - Stage B assigns subtype (0..2) and we map back to original label idx via b_to_orig

    loader must yield original multiclass labels (0..3).
    Returns (y_true4, y_pred4) as numpy arrays.
    """
    modelA.eval()
    modelB.eval()

    yt4, yp4 = [], []

    for x, y in loader:
        x = x.to(device)

        # Stage A
        logitsA = modelA(x)
        probsA = torch.softmax(logitsA, dim=1)  # [B,2]
        p_no = probsA[:, 0]                     # [B]
        is_tumor = (p_no < threshold_no)        # True => tumor

        # default: no_tumor
        pred4 = torch.full((x.size(0),), int(no_class_idx), dtype=torch.long, device=x.device)

        # Stage B on tumor subset
        if is_tumor.any():
            x_t = x[is_tumor]
            logitsB = modelB(x_t)               # [Nt,3]
            predB = logitsB.argmax(1).cpu().numpy()  # 0..2

            mapped = torch.tensor([int(b_to_orig[i]) for i in predB], dtype=torch.long, device=x.device)
            pred4[is_tumor] = mapped

        yt4.append(np.array(y, dtype=np.int64))
        yp4.append(pred4.cpu().numpy())

    return np.concatenate(yt4), np.concatenate(yp4)


# --------- self-check ----------
if __name__ == "__main__":
    print("OK: infer.py import sanity check passed")
