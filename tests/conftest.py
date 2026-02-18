from __future__ import annotations

import sys
from pathlib import Path

# Ensure tests can import local packages and benchmark modules without relying
# on environment-specific PYTHONPATH behavior.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)
