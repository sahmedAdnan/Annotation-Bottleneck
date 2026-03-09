"""
experiments/run_density_clustering.py
--------------------------------------
Backwards-compatible wrapper around cli/density.py.

Prefer calling cli/density.py directly:
    python cli/density.py --config configs/eumoth.yaml

This file exists so any existing bookmarks or scripts that call
run_density_clustering.py continue to work without changes.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.density import main

if __name__ == "__main__":
    main()
