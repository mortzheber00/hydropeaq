#!/usr/bin/env python3
"""Load a finished co-design sweep: summary rows paired with their solution files.

Solutions are named ``{gait}_v{v_target:.2f}_T{t_center:.3f}`` with dots
replaced by ``p`` (or by the row's ``npz`` field). The tag is duplicated from
run_codesign.py to avoid importing CasADi/MLflow here. Solutions without
``Xc`` are rejected.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (_ROOT / "stage1_gait_optimization" / "codesign" /
                   "codesign_results")

# Tolerance on |T - t_center| for a period at the edge of its window
BAND_TOL = 1e-3


def tag_of(row: dict) -> str:
    """Unique name of a solve within a sweep (npz basename)."""
    return (f"{row['gait']}_v{row['v_target']:.2f}_T{row['t_center']:.3f}"
            .replace(".", "p"))


def npz_of(row: dict) -> str:
    """Solution file name: the row's ``npz`` field if present, else derived from the tag."""
    return row.get("npz") or f"{tag_of(row)}.npz"


def detect_band(rows: list[dict]) -> float:
    """Free-T half-window of the sweep, estimated as the largest |T - t_center|.

    FREE_T_BAND is not stored in the summary.
    """
    devs = [abs(r["T"] - r["t_center"]) for r in rows if r["feasible"]]
    return max(devs) if devs else 0.0


def is_pinned(row: dict, band: float) -> bool:
    """True if T ended at the edge of its window (so T is not optimal and should be marked)."""
    return abs(row["T"] - row["t_center"]) >= band - BAND_TOL


def load_rows(results_dir: Path = DEFAULT_RESULTS) -> list[dict]:
    """All summary rows except those marked ``excluded`` by replot_pareto.py."""
    summary = Path(results_dir) / "codesign_summary.json"
    if not summary.exists():
        raise SystemExit(f"no sweep summary at {summary}")
    return [r for r in json.loads(summary.read_text()) if not r.get("excluded")]


def load_sweep(results_dir: Path = DEFAULT_RESULTS, *, pareto_only: bool = True,
               gaits=None) -> list[tuple[dict, dict]]:
    """``[(row, meta), ...]`` of feasible solves sorted by speed.

    ``row`` is the summary entry, ``meta`` the loaded solution. By default only
    Pareto-front points are returned.
    """
    # Local import to avoid loading pinocchio for tag_of-only users
    from stage1_gait_optimization.hydro_model.trajectory import load_solution

    results_dir = Path(results_dir)
    rows = [r for r in load_rows(results_dir)
            if r["feasible"] and np.isfinite(r["cot"])]
    if pareto_only:
        rows = [r for r in rows if r.get("pareto")]
    if gaits:
        rows = [r for r in rows if r["gait"] in gaits]
    if not rows:
        raise SystemExit(
            f"no {'Pareto-front ' if pareto_only else ''}solves in "
            f"{results_dir}{' for ' + ', '.join(gaits) if gaits else ''}")

    out = []
    for row in sorted(rows, key=lambda r: r["speed"]):
        path = results_dir / npz_of(row)
        if not path.exists():
            raise SystemExit(
                f"{path.name} is missing, but its summary row is feasible — the "
                f"sweep directory is incomplete")
        meta = load_solution(path)
        if "Xc" not in meta:
            raise SystemExit(
                f"{path.name} has no Xc block, so these figures could only be "
                f"sampled at the grid nodes — where Radau never enforced the "
                f"dynamics or the state bounds.  Re-run the sweep with the Xc "
                f"export.")
        out.append((row, meta))
    return out


def describe(pairs: list[tuple[dict, dict]], band: float) -> None:
    """Print one line per solve."""
    print(f"{'tag':<28s}{'v [m/s]':>9s}{'T [s]':>8s}{'COT':>8s}  pinned")
    for row, _ in pairs:
        print(f"{tag_of(row):<28s}{row['speed']:>9.4f}{row['T']:>8.3f}"
              f"{row['cot']:>8.3f}  {'yes' if is_pinned(row, band) else '-'}")
