from __future__ import annotations

import os
from run_stageA import run as run_stageA
from run_stageB import run as run_stageAB

IMAGING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(IMAGING_ROOT, "configs", "stageAB.json")


def ask_choice() -> str:
    while True:
        s = input("\nRun what? [A]=StageA gate | [AB]=StageAB full (A+B+eval) : ").strip().lower()
        if s in {"a", "stagea"}:
            return "A"
        if s in {"ab", "stageab", "b"}:
            return "AB"
        print("Invalid. Type A or AB.")


def ask_config() -> str:
    s = input(f"Config path (Enter for default): {DEFAULT_CONFIG}\n> ").strip()
    return s if s else DEFAULT_CONFIG


def main():
    choice = ask_choice()
    cfg = ask_config()

    if choice == "A":
        run_stageA(cfg)
    else:
        run_stageAB(cfg)


if __name__ == "__main__":
    main()
