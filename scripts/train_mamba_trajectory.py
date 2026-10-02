#!/usr/bin/env python3
"""Train the target-only MT baseline; accepts only the prespecified layer sweep."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.mamba_jaad_utils import train_baseline

if __name__ == "__main__":
    train_baseline("trajectory_mamba")
