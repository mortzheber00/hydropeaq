#!/usr/bin/env python3
"""Reading a finished co-design sweep: summary rows paired with their solutions.

``codesign/run_codesign.py`` writes one ``codesign_summary.json`` holding a row
per solve, plus one ``.npz`` per *feasible* solve named by the tag
``{gait}_v{v_target:.2f}_T{t_center:.3f}`` with dots replaced by ``p`` — or, in a
sweep whose rows carry an ``npz`` field, by whatever that field says.  Three
figures walk that pairing — ``plot_limit_activity``, ``plot_structure_vs_speed``
and ``plot_mechanism_vs_speed`` — so the tag convention, the exclusion rules and
the ``Xc`` requirement are spelled out once here instead of three times.

Every consumer here samples at the collocation points, so a sweep whose files
predate the ``Xc`` export is refused with the tag of the first file missing it
rather than silently measured at the grid nodes.  ``run_codesign`` builds the
same tag inline and ``replot_pareto`` has its own copy; this is a third, kept
separate because stage 3 importing the sweep driver would drag CasADi and
MLflow into a plotting script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (_ROOT / "stage1_gait_optimization" / "codesign" /
                   "codesign_results")

# Tolerance on |T - t_center| for calling the free-T window active.
BAND_TOL = 1e-3


def tag_of(row: dict) -> str:
    """The solve's npz basename — its unique name within a sweep."""
    return (f"{row['gait']}_v{row['v_target']:.2f}_T{row['t_center']:.3f}"
            .replace(".", "p"))


def npz_of(row: dict) -> str:
    """The solve's file name, as recorded rather than re-derived when possible.

    A sweep that writes its own file names into the summary is believed over
    ``tag_of``: the tag rounds ``v_target`` to two decimals, and a sweep written
    with three (``v0p110``, not ``v0p11``) then looks entirely absent.  Rows
    without the field predate it and still have to be named by the tag.
    """
    return row.get("npz") or f"{tag_of(row)}.npz"


def detect_band(rows: list[dict]) -> float:
    """Free-T half-window used by the sweep, read off the data.

    ``FREE_T_BAND`` lives in run_codesign.py and is not written to the summary,
    but every solve that hit the window sits exactly at ``t_center ± band``, so
    the largest observed excursion is the band.
    """
    devs = [abs(r["T"] - r["t_center"]) for r in rows if r["feasible"]]
    return max(devs) if devs else 0.0


def is_pinned(row: dict, band: float) -> bool:
    """Did T end up at an edge of its refine window?

    Such a point's cadence is set by the window, not by an efficiency optimum,
    so it is not a stationary point of anything and every figure that puts T (or
    1/T) on an axis has to mark it.
    """
    return abs(row["T"] - row["t_center"]) >= band - BAND_TOL


def load_rows(results_dir: Path = DEFAULT_RESULTS) -> list[dict]:
    """Every summary row, excluded solves dropped.

    ``replot_pareto.py`` keeps a dropped solve in the summary flagged
    ``excluded`` rather than deleting it, so the record survives; a figure wants
    it gone.
    """
    summary = Path(results_dir) / "codesign_summary.json"
    if not summary.exists():
        raise SystemExit(f"no sweep summary at {summary}")
    return [r for r in json.loads(summary.read_text()) if not r.get("excluded")]


def load_sweep(results_dir: Path = DEFAULT_RESULTS, *, pareto_only: bool = True,
               gaits=None) -> list[tuple[dict, dict]]:
    """``[(row, meta), ...]`` for the solves worth plotting, sorted by speed.

    ``row`` is the summary entry (speed, COT, T, gait, pareto flag) and ``meta``
    the loaded solution file.  ``pareto_only`` keeps the front, which is what a
    figure with speed on its x-axis wants: the other points sit at the same
    speeds with worse COT and would draw as vertical scatter.
    """
    # Local, so importing this module from a script that only needs tag_of does
    # not pay for pinocchio.
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
    """One stdout line per solve: what the figure is about to draw."""
    print(f"{'tag':<28s}{'v [m/s]':>9s}{'T [s]':>8s}{'COT':>8s}  pinned")
    for row, _ in pairs:
        print(f"{tag_of(row):<28s}{row['speed']:>9.4f}{row['T']:>8.3f}"
              f"{row['cot']:>8.3f}  {'yes' if is_pinned(row, band) else '-'}")
