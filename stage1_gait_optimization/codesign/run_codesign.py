#!/usr/bin/env python3
"""Speed vs. cost-of-transport sweep over initial gait, cycle period and target speed.

Solves one OCP per (gait, T centre, speed target) via ``solver.solve_gait_ocp``
and extracts the Pareto front. Each (gait, T centre) is a chain over ascending
speeds, warm-started link to link, and chains run in parallel. Configure the
sweep with the constants below. Results go to codesign_results/ and MLflow.

Usage:
  python stage1_gait_optimization/codesign/run_codesign.py
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

# --- Sweep configuration ---
# Speed floors; the floor binds, so each target yields a point at that speed.
SPEED_TARGETS = np.linspace(0.1, 0.2, 7)  # [m/s]

# Speeds closer than this count as equal in the Pareto filter, so solver noise
# at a shared target does not keep dominated points.
SPEED_TOL = 1e-4

# T centres per gait (total solves = gaits x speeds x T centres).
T_GRID_DEFAULT = np.linspace(0.8, 1.2, 3)
GAIT_T_OVERRIDE: dict[str, np.ndarray] = {
    # "LSPG25": np.linspace(0.5, 1.2, 3),
}
FREE_T_BAND = 0.2   # free-T half-window; > grid spacing / 2 so bands overlap

PARALLEL = True     # False keeps RAM bounded
N_WORKERS = 4        # processes when PARALLEL
# Warm-start each speed from the previous one in its chain. False makes every
# solve an independent cold start (slower).
WARM_START = True
QUIET_SOLVES = False  # suppress per-solve output

# --- Experiment tracking ---
USE_MLFLOW = True
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "gait_codesign"
RUN_LABEL = ""       # optional tag appended to the MLflow run name

OUT_DIR = Path(__file__).parent / "codesign_results"


@contextlib.contextmanager
def _maybe_quiet(quiet: bool):
    """Suppress stdout if ``quiet``."""
    if not quiet:
        yield
        return
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def gait_t_centers(gait: str) -> np.ndarray:
    return GAIT_T_OVERRIDE.get(gait, T_GRID_DEFAULT)


def build_chains() -> list[tuple[str, float, list[float]]]:
    """One chain per (gait, T centre) over the ascending speed targets.

    Chains never span gaits: the gaits are meant to reach different local minima,
    and warm-starting across them would collapse those into one.
    """
    speeds = sorted(float(v) for v in SPEED_TARGETS)
    return [(g, float(t), speeds) for g in ev.GAITS for t in gait_t_centers(g)]


def _eval_chain(chain: tuple[str, float, list[float]], robot=None, dyn=None) -> list[dict]:
    """Solve one chain; after a failed link the next one starts cold."""
    gait, t_center, speeds = chain
    rows, warm = [], None
    for v in speeds:
        with _maybe_quiet(QUIET_SOLVES):
            r = ev.solve_gait_ocp(gait, t_center, v_target=v,
                                  free_T_band=FREE_T_BAND,
                                  robot=robot, dyn=dyn, warm_start=warm)
        # Pop the large vector so it is not sent back to the parent process.
        nxt = r.pop("warm_start")
        warm = nxt if WARM_START else None
        rows.append(r)
    return rows


def pareto_front(rows: list[dict]) -> list[int]:
    """Indices of non-dominated feasible points (max speed, min COT), sorted by speed."""
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
            # Solver cost, to compare warm and cold starts
            "iterations": r["iterations"],
            "wall_time_s": r["wall_time_s"],
            "status": r["status"],
            "warm_started": r["warm_started"],
        })
        if r["feasible"]:
            tag = f"{r['gait']}_v{r['v_target']:.2f}_T{r['t_center']:.3f}".replace(".", "p")
            np.savez(
                OUT_DIR / f"{tag}.npz",
                T=r["T"], X=r["X"], U=r["U"], N=r["N"], nq=r["nq"],
            )
    (OUT_DIR / "codesign_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary + per-point npz to {OUT_DIR}")


# Legend names where the figure label differs from the gait key.
GAIT_LABELS = {"Prototype": "Inverse-kinematics"}


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

    # Marker = gait, colour = solved period T
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
            zorder=3,
        )

    front = sorted((rows[i] for i in pareto_idx), key=lambda r: r["speed"])
    if front:
        ax.plot([r["speed"] for r in front], [r["cot"] for r in front],
                "k--", lw=1.2, zorder=2)

    if sc is not None:
        fig.colorbar(sc, ax=ax).set_label("cycle period T [s]")

    # Gait legend in a neutral colour, since colour encodes T
    handles = [
        Line2D([], [], marker=gait_marker[g], color="gray", linestyle="",
               markeredgecolor="k", label=GAIT_LABELS.get(g, g))
        for g in ev.GAITS
    ]
    handles.append(Line2D([], [], color="k", linestyle="--", label="Pareto front"))
    ax.legend(handles=handles, loc="best")

    ax.set_xlabel(r"forward speed [m\,s$^{-1}$]")
    ax.set_ylabel("cost of transport [-]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    # Extra padding: the tight bbox under-measures usetex text.
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
    """Log the sweep as a single MLflow run with OUT_DIR as artifacts."""
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
                "N_collocation": ev.CFG.n,
                "d_colloc": ev.CFG.d_colloc,
                "tau_max": ev.CFG.tau_max,
                "f_c": ev.CFG.f_c,
                "heading_tol": ev.CFG.heading_tol,
                "enforce_symmetry": ev.CFG.enforce_symmetry,
                "symmetry_phase": ev.CFG.symmetry_phase,
                "w_power": ev.CFG.w_power,
                "w_vel_smooth": ev.CFG.w_vel_smooth,
                "w_joint_smooth": ev.CFG.w_joint_smooth,
                "w_drift": ev.CFG.w_drift,
                "gaits": ",".join(ev.GAITS),
                "n_tasks": n_tasks,
                "parallel": PARALLEL,
                "warm_start": WARM_START,
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
            warm = [r for r in rows if r["warm_started"]]
            cold = [r for r in rows if not r["warm_started"]]
            metrics["total_iterations"] = sum(r["iterations"] for r in rows)
            metrics["total_solve_wall_s"] = sum(r["wall_time_s"] for r in rows)
            for label, group in (("warm", warm), ("cold", cold)):
                if group:
                    metrics[f"mean_iters_{label}"] = float(
                        np.mean([r["iterations"] for r in group]))
                    metrics[f"mean_wall_s_{label}"] = float(
                        np.mean([r["wall_time_s"] for r in group]))
            # Best COT per target speed
            for v in SPEED_TARGETS:
                at_v = [r["cot"] for r in feasible if abs(r["v_target"] - v) < 1e-9]
                if at_v:
                    metrics[f"best_cot_v{float(v):.2f}"] = min(at_v)
            mlflow.log_metrics(metrics)

            mlflow.log_artifacts(str(OUT_DIR))
        print(f"Logged sweep to MLflow ({MLFLOW_EXPERIMENT}/{run_name})")
    except Exception as e:  # never let logging kill a finished sweep
        print(f"  (MLflow logging skipped: {e})")


def main() -> None:
    chains = build_chains()
    n_solves = sum(len(c[2]) for c in chains)
    print(f"Co-design sweep: {len(chains)} chains x {len(chains[0][2])} speeds "
          f"= {n_solves} solves "
          f"[{'parallel x' + str(N_WORKERS) if PARALLEL else 'serial'}, "
          f"{'warm-started chains' if WARM_START else 'all cold'}]")

    rows = []
    bar = tqdm(total=n_solves, desc="co-design sweep") if tqdm is not None else None
    if PARALLEL:
        with Pool(N_WORKERS) as pool:
            # Parallelise over chains; each chain is sequential.
            for i, chain_rows in enumerate(pool.imap_unordered(_eval_chain, chains), 1):
                rows.extend(chain_rows)
                if bar is not None:
                    bar.update(len(chain_rows))
                else:
                    print(f"  [{i}/{len(chains)}] chain done "
                          f"({chain_rows[0]['gait']} T={chain_rows[0]['t_center']:.3f})",
                          flush=True)
    else:
        robot, dyn = ev.get_robot_dyn()   # build once, reuse across solves
        for i, chain in enumerate(chains, 1):
            if bar is not None:
                bar.set_postfix_str(f"{chain[0]} T={chain[1]:.3f}")
            else:
                print(f"  [{i}/{len(chains)}] {chain[0]} @ T={chain[1]:.3f} ...",
                      flush=True)
            chain_rows = _eval_chain(chain, robot, dyn)
            rows.extend(chain_rows)
            if bar is not None:
                bar.update(len(chain_rows))
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

    warm = [r["iterations"] for r in rows if r["warm_started"]]
    cold = [r["iterations"] for r in rows if not r["warm_started"]]
    if warm and cold:
        print(f"  mean IPOPT iterations: {np.mean(cold):.0f} cold ({len(cold)}) "
              f"vs {np.mean(warm):.0f} warm ({len(warm)})")

    _save(rows, pareto_idx)
    _plot(rows, pareto_idx)
    _log_mlflow(rows, pareto_idx, n_solves)


if __name__ == "__main__":
    main()
