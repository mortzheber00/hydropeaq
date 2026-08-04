"""Shared pytest setup.

The pipeline is not an installable package: ``stage1_gait_optimization`` puts
itself on ``sys.path`` from inside each driver script.  Do that once here so
tests can ``import hydro_model`` without inheriting that hack.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE1 = REPO_ROOT / "stage1_gait_optimization"
GOLDEN = Path(__file__).resolve().parent / "data" / "amph_golden.npz"

BODY2_SCRIPTS = REPO_ROOT / "src" / "BODY2" / "scripts"

for _p in (str(STAGE1), str(REPO_ROOT), str(BODY2_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
