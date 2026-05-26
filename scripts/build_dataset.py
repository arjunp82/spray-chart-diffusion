#!/usr/bin/env python3
"""CLI: preprocess raw Statcast CSVs into spray chart images."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import main

if __name__ == "__main__":
    main()
