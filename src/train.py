from __future__ import annotations
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainHistory:
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_acc: List[float]


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> TrainHistory:

    hist = TrainHistory(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    for epoch in range(epochs):
        # ----- train
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = torch.as_tensor(y, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += bs

            pbar.set_postfix(loss=total_loss/max(total_n,1), acc=total_correct/max(total_n,1))

        tr_loss = total_loss / max(total_n, 1)
        tr_acc = total_correct / max(total_n, 1)

        # ----- val
        model.eval()
        v_loss_sum, v_correct, v_n = 0.0, 0, 0

        pbar = tqdm(val_loader, desc=f"Val   {epoch+1}/{epochs}", leave=False)
        with torch.no_grad():
            for x, y in pbar:
                x = x.to(device)
                y = torch.as_tensor(y, dtype=torch.long, device=device)

                logits = model(x)
                loss = criterion(logits, y)

                bs = x.size(0)
                v_loss_sum += loss.item() * bs
                v_correct += (logits.argmax(1) == y).sum().item()
                v_n += bs

                pbar.set_postfix(loss=v_loss_sum/max(v_n,1), acc=v_correct/max(v_n,1))

        va_loss = v_loss_sum / max(v_n, 1)
        va_acc = v_correct / max(v_n, 1)

        hist.train_loss.append(float(tr_loss))
        hist.train_acc.append(float(tr_acc))
        hist.val_loss.append(float(va_loss))
        hist.val_acc.append(float(va_acc))

        # one clean line per epoch
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train {tr_loss:.4f}/{tr_acc:.4f} | "
            f"val {va_loss:.4f}/{va_acc:.4f}"
        )

    return hist

# --------- self-check ----------
if __name__ == "__main__":
    # no full training here; just import sanity
    print("OK: train.py import sanity check passed")
