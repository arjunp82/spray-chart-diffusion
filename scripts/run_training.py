#!/usr/bin/env python3
"""CLI: launch diffusion model training."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spray chart diffusion model")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
