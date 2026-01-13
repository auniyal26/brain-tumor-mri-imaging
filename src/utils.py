from __future__ import annotations

import os
import json
import shutil
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Basic seeding. If deterministic=True, may require extra env vars on CUDA
    (you already saw the CuBLAS warning in the notebook).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        # faster, nondeterministic is fine for now
        torch.backends.cudnn.benchmark = True


def clear_dir(path: str) -> None:
    """Delete and recreate a directory (your 'single run folder' rule)."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------- self-check ----------
if __name__ == "__main__":
    d = get_device()
    print("Device:", d)

    tmp = os.path.join(".", "_tmp_run")
    clear_dir(tmp)
    write_json(os.path.join(tmp, "a.json"), {"ok": True, "n": 1})
    print("Wrote:", os.path.abspath(os.path.join(tmp, "a.json")))
    print("OK: utils.py sanity check passed")
