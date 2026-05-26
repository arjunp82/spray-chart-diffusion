#!/usr/bin/env python3
"""CLI: download Statcast data for one or more seasons."""

import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetch import main

if __name__ == "__main__":
    main()
