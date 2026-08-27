#!/usr/bin/env python3
"""Outer co-design sweep over (initial gait, cycle period T).

For each of the four initial gaits {LSPG25, LSPG33, TLPG50, Prototype} and each
period on its per-gait T grid, solve the collocation OCP
(``gait_ocp_eval.solve_gait_ocp``).  Every solve self-refines T within a narrow
band, so there is no separate refine stage.  The feasible points are collected
into a speed-vs-efficiency (COT) Pareto front.

This supersedes the sinusoid-only ``run_search.py``.  It is structured so the
sweep driver can later be swapped for a pymoo NSGA-II driver (adding leg
dimensions as co-design variables) while reusing ``solve_gait_ocp`` unchanged.

Run:
    python run_codesign.py
Toggle ``PARALLEL`` below: serial (default) keeps RAM bounded on a laptop;
parallel uses ``N_WORKERS`` processes when you have the memory.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import numpy as np

import solver as ev

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ── Sweep configuration ─────────────────────────────────────────────────────
# Target-speed axis (ε-constraint): the inner solve minimises energy subject to
# speed >= v_target, the floor binds, so each value yields a point at that speed.
# Sweeping it is what spans the speed axis of the Pareto front.
SPEED_TARGETS = np.linspace(0.1, 0.3, 25)  # m/s; 0.025 spacing gives ~10% speed resolution

# Pareto dominance tolerance on speed.  The ε-constraint floor binds, so
# converged speeds cluster at the targets up to solver/float noise (~1e-8).
# Without a tolerance that noise makes points at the *same* target speed look
# like distinct speeds, so higher-COT points survive as spurious front members.
# Treat speeds within SPEED_TOL as equal (well below the 0.025 target spacing).
SPEED_TOL = 1e-4

# Period grid per gait.  At each target speed, T is the efficiency knob: the grid
# (each point free-T-refined) explores cadences and the Pareto filter keeps the
# most efficient one per (gait, speed).  Keep it small — total solves = gaits ×
# speeds × T-centres.  Each gait may override with its own cadence range.
T_GRID_DEFAULT = np.linspace(0.5, 1.4, 7)
GAIT_T_OVERRIDE: dict[str, np.ndarray] = {
    # "LSPG25": np.linspace(0.5, 1.2, 3),
}
FREE_T_BAND = 0.15   # δ: free-T half-window; > grid spacing/2 so bands overlap

PARALLEL = True     # serial keeps RAM bounded; set True to use a process pool
N_WORKERS = 4        # processes when PARALLEL
QUIET_SOLVES = False  # suppress per-solve guess/IPOPT prints for a clean bar

# ── Experiment tracking ─────────────────────────────────────────────────────
USE_MLFLOW = True
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "gait_codesign"
RUN_LABEL = ""       # optional tag appended to the MLflow run name

OUT_DIR = Path(__file__).parent / "codesign_results"


@contextlib.contextmanager
def _maybe_quiet(quiet: bool):
    """Swallow stdout (noisy guess/solver prints) so the progress bar stays clean."""
    if not quiet:
        yield
        return
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def gait_t_centers(gait: str) -> np.ndarray:
    return GAIT_T_OVERRIDE.get(gait, T_GRID_DEFAULT)


def build_tasks() -> list[tuple[str, float, float]]:
    return [
        (g, float(v), float(t))
        for g in ev.GAITS
        for v in SPEED_TARGETS
        for t in gait_t_centers(g)
    ]


def _eval_point(task: tuple[str, float, float]) -> dict:
    """Worker entry: solve one (gait, speed, T) point. Robot/dyn cached per process."""
    gait, v_target, t_center = task
    with _maybe_quiet(QUIET_SOLVES):
        return ev.solve_gait_ocp(gait, t_center, v_target=v_target, free_T_band=FREE_T_BAND)


def pareto_front(rows: list[dict]) -> list[int]:
    """Indices of non-dominated feasible points: maximise speed, minimise COT.

    Speeds within ``SPEED_TOL`` are treated as equal, so float/solver noise at a
    shared target speed doesn't keep higher-COT points as spurious front members.
    """
    feasible = [i for i, r in enumerate(rows) if r["feasible"] and np.isfinite(r["cot"])]
    keep = []
    for i in feasible:
        ri = rows[i]
        dominated = any(
            rows[j]["speed"] >= ri["speed"] - SPEED_TOL
            and rows[j]["cot"] <= ri["cot"]
            and (rows[j]["speed"] > ri["speed"] + SPEED_TOL or rows[j]["cot"] < ri["cot"])
            for j in feasible if j != i
        )
        if not dominated:
            keep.append(i)
    keep.sort(key=lambda i: rows[i]["speed"])
    return keep


def _save(rows: list[dict], pareto_idx: list[int]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    pareto_set = set(pareto_idx)

    summary = []
    for i, r in enumerate(rows):
        summary.append({
            "gait": r["gait"],
            "v_target": r["v_target"],
            "t_center": r["t_center"],
            "T": r["T"],
            "speed": r["speed"],
            "forward_dist": r["forward_dist"],
            "energy": r["energy"],
            "cot": r["cot"],
            "feasible": r["feasible"],
            "pareto": i in pareto_set,
        })
        # Per-point trajectory for feasible solves.
        if r["feasible"]:
            tag = f"{r['gait']}_v{r['v_target']:.2f}_T{r['t_center']:.3f}".replace(".", "p")
            np.savez(
                OUT_DIR / f"{tag}.npz",
                T=r["T"], X=r["X"], U=r["U"], N=r["N"], nq=r["nq"],
            )
    (OUT_DIR / "codesign_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary + per-point npz to {OUT_DIR}")


def _plot(rows: list[dict], pareto_idx: list[int]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scienceplots  # noqa: F401  registers the 'science' matplotlib style
        from matplotlib.lines import Line2D
    except ImportError:
        return
    plt.style.use(["science"])
    plt.rcParams["text.usetex"] = True

    pts = [r for r in rows if r["feasible"] and np.isfinite(r["cot"])]
    if not pts:
        return

    # Marker shape per gait; colour encodes the solved cycle period T.
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    gait_marker = {g: markers[i % len(markers)] for i, g in enumerate(ev.GAITS)}
    t_all = [r["T"] for r in pts]
    vmin, vmax = min(t_all), max(t_all)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sc = None
    for gait in ev.GAITS:
        gpts = [r for r in pts if r["gait"] == gait]
        if not gpts:
            continue
        sc = ax.scatter(
            [r["speed"] for r in gpts], [r["cot"] for r in gpts],
            c=[r["T"] for r in gpts], cmap="viridis", vmin=vmin, vmax=vmax,
            marker=gait_marker[gait], s=55, edgecolors="k", linewidths=0.4,
        )

    # Pareto front as a dashed line; the markers keep their T colour.
    front = sorted((rows[i] for i in pareto_idx), key=lambda r: r["speed"])
    if front:
        ax.plot([r["speed"] for r in front], [r["cot"] for r in front],
                "k--", lw=1.2, zorder=0)

    if sc is not None:
        fig.colorbar(sc, ax=ax).set_label("cycle period T [s]")

    # Gait legend via marker shape (neutral colour, since colour now means T).
    handles = [
        Line2D([], [], marker=gait_marker[g], color="gray", linestyle="",
               markeredgecolor="k", label=g)
        for g in ev.GAITS
    ]
    handles.append(Line2D([], [], color="k", linestyle="--", label="Pareto front"))
    ax.legend(handles=handles, loc="best")

    ax.set_xlabel("forward speed [m/s]")
    ax.set_ylabel("cost of transport [-]")
    ax.set_title("Gait co-design: speed vs. efficiency (colour = T)")
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    # Thesis figures: 300 dpi raster + vector PDF.  pad_inches above the default:
    # the tight bbox under-measures usetex text.
    for suffix in (".png", ".pdf"):
        path = OUT_DIR / f"pareto_front{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
        print(f"Saved Pareto plot to {path}")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _log_mlflow(rows: list[dict], pareto_idx: list[int], n_tasks: int) -> None:
    """Log the whole sweep as one MLflow run: config params, summary metrics,
    and the full codesign_results/ directory (summary.json, png, per-point npz)
    as artifacts.  Inner solves do not log, so there are no nested/concurrent
    runs."""
    if not USE_MLFLOW:
        return
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{stamp}_{RUN_LABEL}" if RUN_LABEL else stamp

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "speed_targets": np.array2string(np.asarray(SPEED_TARGETS), precision=3),
                "t_grid": np.array2string(np.asarray(T_GRID_DEFAULT), precision=3),
                "free_T_band": FREE_T_BAND,
                "N_collocation": ev.N,
                "tau_max": ev.TAU_MAX,
                "f_c": ev.F_C,
                "heading_tol": ev.HEADING_TOL,
                "w_power": ev.W_POWER,
                "w_vel_smooth": ev.W_VEL_SMOOTH,
                "w_drift": ev.W_DRIFT,
                "gaits": ",".join(ev.GAITS),
                "n_tasks": n_tasks,
                "parallel": PARALLEL,
                "git_sha": _git_sha(),
            })

            feasible = [r for r in rows if r["feasible"] and np.isfinite(r["cot"])]
            metrics = {
                "n_total": len(rows),
                "n_feasible": len(feasible),
                "frac_feasible": len(feasible) / len(rows) if rows else 0.0,
                "n_pareto": len(pareto_idx),
            }
            if feasible:
                metrics["min_cot"] = min(r["cot"] for r in feasible)
                metrics["max_speed"] = max(r["speed"] for r in feasible)
            # Best (lowest) COT achieved at each target speed.
            for v in SPEED_TARGETS:
                at_v = [r["cot"] for r in feasible if abs(r["v_target"] - v) < 1e-9]
                if at_v:
                    metrics[f"best_cot_v{float(v):.2f}"] = min(at_v)
            mlflow.log_metrics(metrics)

            # All run artifacts: summary.json, pareto_front.png, per-point npz.
            mlflow.log_artifacts(str(OUT_DIR))
        print(f"Logged sweep to MLflow ({MLFLOW_EXPERIMENT}/{run_name})")
    except Exception as e:  # never let logging kill a finished sweep
        print(f"  (MLflow logging skipped: {e})")


def main() -> None:
    tasks = build_tasks()
    print(f"Co-design sweep: {len(tasks)} (gait, speed, T) points "
          f"[{'parallel x' + str(N_WORKERS) if PARALLEL else 'serial'}]")

    use_bar = tqdm is not None
    if PARALLEL:
        with Pool(N_WORKERS) as pool:
            results = pool.imap_unordered(_eval_point, tasks)
            if use_bar:
                results = tqdm(results, total=len(tasks), desc="co-design sweep")
            rows = list(results)
    else:
        robot, dyn = ev.get_robot_dyn()   # build once, reuse across solves
        rows = []
        bar = tqdm(total=len(tasks), desc="co-design sweep") if use_bar else None
        for i, (gait, v_target, t_center) in enumerate(tasks, 1):
            if bar is not None:
                bar.set_postfix_str(f"{gait} v={v_target:.2f} T={t_center:.3f}")
            else:
                print(f"  [{i}/{len(tasks)}] {gait} @ v={v_target:.2f} T={t_center:.3f} ...",
                      flush=True)
            with _maybe_quiet(QUIET_SOLVES and bar is not None):
                rows.append(ev.solve_gait_ocp(gait, t_center, v_target=v_target,
                                              free_T_band=FREE_T_BAND,
                                              robot=robot, dyn=dyn))
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

    pareto_idx = pareto_front(rows)

    print("\n── Pareto front (speed ↑, COT ↓) ───────────────────────────")
    print(f"  {'gait':<10s}{'T':>8s}{'speed':>10s}{'COT':>10s}")
    for i in pareto_idx:
        r = rows[i]
        print(f"  {r['gait']:<10s}{r['T']:>8.3f}{r['speed']:>10.4f}{r['cot']:>10.4f}")
    n_feasible = sum(r["feasible"] for r in rows)
    print(f"\n  feasible: {n_feasible}/{len(rows)}   pareto points: {len(pareto_idx)}")

    _save(rows, pareto_idx)
    _plot(rows, pareto_idx)
    _log_mlflow(rows, pareto_idx, len(tasks))


if __name__ == "__main__":
    main()
