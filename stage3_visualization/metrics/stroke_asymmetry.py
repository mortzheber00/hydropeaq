#!/usr/bin/env python3
"""Stroke-structure numbers for the thesis (results-structure section).

1. Power-stroke duty fraction per leg and number of separate sweeps.
2. Submerged fraction of thigh and calf in power vs recovery, with the spread
   within each phase (more robust than foot depth).
3. Frozen-leg baseline: drag with the leg held at its mean pose and at the best
   pose from a (thigh, calf) grid search. The base motion stays as solved, so
   this attributes thrust but does not compare gaits.
4. Front-vs-hind shape mismatch: best and worst alignment residual.

"Power stroke" always means ``v_foot_x < 0`` in the base frame, not positive
joint power. All averages use Radau weights at the collocation points.

Usage:
  python stage3_visualization/metrics/stroke_asymmetry.py
  python stage3_visualization/metrics/stroke_asymmetry.py --solution <npz> --grid 13
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))

from stage3_visualization.common.collocation import D_COLLOC, collocation_states, quadrature_weights  # noqa: E402
from stage3_visualization.common.drag_model import (  # noqa: E402
    _leg_joint_indices,
    leg_cycle_drag,
    leg_mean_pose,
    link_drag_terms,
    require_supported,
)
from stage3_visualization.thrust.gait_diagnostics import compute_traces  # noqa: E402
from stage3_visualization.metrics.nominal_metrics import MIN_SWEEP, cyclic_runs  # noqa: E402

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402
from stage1_gait_optimization.ocp_common import limits_for  # noqa: E402

_DEFAULT_SOLUTION = (_ROOT / "experiment_results" / "mlruns" / "2" /
                     "69943d2cffc94ca5b78271a7df24c546" / "artifacts" /
                     "TLPG50_v0p180_T1p400.npz")

LEGS = ("Front_Left", "Front_Right", "Hind_Left", "Hind_Right")

# Power-phase fractions of the seed gaits, for comparison
SEED_DUTY = {"LSPG25": 0.25, "LSPG33": 0.33, "TLPG50": 0.50}


def link_submersion(robot, Xc, nq, link_name) -> np.ndarray:
    """Submerged fraction of one link per collocation sample (as used by the drag model)."""
    out = np.empty(Xc.shape[1])
    for t in range(Xc.shape[1]):
        q = Xc[:nq, t]
        robot.forward_kinematics(q)
        out[t] = link_drag_terms(robot, link_name, q)[2]
    return out


def best_frozen_pose(robot, Xc, nq, N, leg, grid: int = 21):
    """Held (thigh, calf) pose with the least rearward drag, via a grid over the joint box.

    The side joint stays at zero. Returns ``(force, (thigh, calf), alpha_calf)``.
    Costs ``grid**2`` cycle evaluations per leg.
    """
    idx = _leg_joint_indices(robot, leg)
    q_lb, q_ub, _, _ = limits_for(robot)
    best = (-np.inf, None)
    for a in np.linspace(q_lb[idx[1]], q_ub[idx[1]], grid):
        for b in np.linspace(q_lb[idx[2]], q_ub[idx[2]], grid):
            f = leg_cycle_drag(robot, Xc, nq, N, leg, hold=[0.0, a, b])
            if f > best[0]:                     # least negative = least drag
                best = (f, (float(a), float(b)))
    q = Xc[:nq, 0].copy()
    q[7 + idx[1]], q[7 + idx[2]] = best[1]
    robot.forward_kinematics(q)
    alpha = link_drag_terms(robot, f"{leg}_Calf_link", q)[2]
    return best[0], best[1], float(alpha)


def measure_leg(robot, Xc, nq, N, T, leg, w) -> dict:
    """Power-stroke fraction, submersion split and speed split for one leg."""
    v_foot_x, v_foot_mag, F_drag_x, _, _ = compute_traces(robot, leg, Xc)
    power = v_foot_x < 0.0

    def split(a):
        """Weighted ``(mean_power, mean_recovery, sd_power, sd_recovery)``."""
        mp = float(np.average(a[power], weights=w[power]))
        mr = float(np.average(a[~power], weights=w[~power]))
        sp = float(np.sqrt(np.average((a[power] - mp) ** 2, weights=w[power])))
        sr = float(np.sqrt(np.average((a[~power] - mr) ** 2, weights=w[~power])))
        return mp, mr, sp, sr

    alpha = {seg: split(link_submersion(robot, Xc, nq, f"{leg}_{seg}_link"))
             for seg in ("Thigh", "Calf")}
    s_pow, s_rec, _, _ = split(v_foot_mag)
    windows = [r for r in cyclic_runs(power) if r >= MIN_SWEEP]
    return {
        "duty": float(w[power].sum() / w.sum()),
        "windows": windows,
        "alpha": alpha,
        "s_pow": s_pow, "s_rec": s_rec, "s_ratio": s_pow / s_rec,
        "F_pow": float(np.average(F_drag_x[power], weights=w[power])),
        "F_rec": float(np.average(F_drag_x[~power], weights=w[~power])),
    }


def pair_mismatch(q_a: np.ndarray, q_b: np.ndarray, upsample: int = 64):
    """Best and worst RMS residual [deg] over all relative phases, with their phases.

    Inputs are ``(n_joints, N)`` in radians. A best close to the worst means
    the shapes do not match at any phase.
    """
    N = q_a.shape[1]
    fb = np.fft.rfft(q_b, axis=1)
    k = np.fft.rfftfreq(N, d=1.0 / N)
    errs = []
    for s in np.arange(N * upsample) / upsample:
        shifted = np.fft.irfft(fb * np.exp(-2j * np.pi * k * s / N), n=N, axis=1)
        errs.append((float(np.sqrt(np.mean((shifted - q_a) ** 2))), s / N))
    errs.sort()
    return (np.degrees(errs[0][0]), errs[0][1],
            np.degrees(errs[-1][0]), errs[-1][1])


def harmonic_phase(trace: np.ndarray, harmonic: int = 1) -> float:
    """Phase of one Fourier component, in cycles, in [0, 1)."""
    F = np.fft.rfft(trace)
    return float((-np.angle(F[harmonic]) / (2.0 * np.pi)) % 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solution", type=Path, default=_DEFAULT_SOLUTION)
    ap.add_argument("--grid", type=int, default=21,
                    help="pose-sweep resolution per joint for the frozen-leg "
                         "baseline (default: %(default)s)")
    args = ap.parse_args()

    meta = load_solution(args.solution)
    require_supported(meta["robot"], "stroke_asymmetry")
    robot = load_robot(meta["robot"])
    X, nq, N, T = meta["X"], meta["nq"], meta["N"], meta["T"]
    Xc, phase, _ = collocation_states(robot, X, meta["Xc"], nq, N, D_COLLOC)
    w, _ = quadrature_weights(N, T)

    print(f"=== {args.solution.name} ===")
    print("\n-- power stroke (KINEMATIC: v_foot_x < 0), Radau-weighted --")
    print(f"  {'leg':<14s}{'duty':>8s}{'wins':>6s}"
          f"{'|v|_pow':>9s}{'|v|_rec':>9s}{'ratio':>7s}")
    per_leg = {}
    for leg in LEGS:
        m = measure_leg(robot, Xc, nq, N, T, leg, w)
        per_leg[leg] = m
        print(f"  {leg:<14s}{100 * m['duty']:>7.1f}%{len(m['windows']):>6d}"
              f"{m['s_pow']:>9.3f}{m['s_rec']:>9.3f}{m['s_ratio']:>7.2f}")
    print("  (|v| is the foot speed relative to the hull, in m/s)")

    print("\n-- submerged fraction per link, power vs recovery --")
    print(f"  {'link':<22s}{'a_pow':>8s}{'a_rec':>8s}{'ratio':>7s}"
          f"{'sd_pow':>8s}{'sd_rec':>8s}   representative?")
    for leg in LEGS:
        for seg in ("Thigh", "Calf"):
            ap, ar, sp, sr = per_leg[leg]["alpha"][seg]
            # Meaningful only if the gap exceeds the within-phase spread
            ok = "yes" if abs(ap - ar) > max(sp, sr) else "NO -- gap < spread"
            print(f"  {leg + '/' + seg:<22s}{ap:>8.3f}{ar:>8.3f}"
                  f"{ap / ar:>7.2f}{sp:>8.3f}{sr:>8.3f}   {ok}")

    duties = np.array([m["duty"] for m in per_leg.values()])
    print(f"\n  duty range {100 * duties.min():.1f}-{100 * duties.max():.1f} %, "
          f"against the seeds " +
          ", ".join(f"{g} {100 * f:.0f} %" for g, f in SEED_DUTY.items()))

    # Front vs hind (left/right are symmetric by constraint)
    names = list(robot.model.names)[1:][-robot.n_actuated:]
    def leg_rows(key):
        return [i for i, n in enumerate(names)
                if key.lower() in n.lower() and "side" not in n.lower()]
    q = X[7:nq, :-1]
    fl, hl = leg_rows("Front_Left"), leg_rows("Hind_Left")
    best, ph_b, worst, ph_w = pair_mismatch(q[fl], q[hl])
    print(f"\n-- frozen-leg counterfactual, {args.grid}x{args.grid} pose sweep --")
    print(f"  {'leg':<13s}{'solved':>9s}{'@mean':>9s}{'@best':>9s}"
          f"{'best pose':>17s}{'gain/mean':>10s}{'gain/best':>10s}")
    for leg in LEGS:
        solved = leg_cycle_drag(robot, Xc, nq, N, leg)
        at_mean = leg_cycle_drag(robot, Xc, nq, N, leg,
                                 hold=leg_mean_pose(robot, Xc, leg))
        at_best, pose, alpha = best_frozen_pose(robot, Xc, nq, N, leg, args.grid)
        print(f"  {leg:<13s}{solved:>+9.3f}{at_mean:>+9.3f}{at_best:>+9.3f}"
              f"{'th%+.0f ca%+.0f' % (np.degrees(pose[0]), np.degrees(pose[1])):>17s}"
              f"{solved - at_mean:>+10.3f}{solved - at_best:>+10.3f}")
        print(f"  {'':13s}calf submersion at the best held pose: {alpha:.3f}")
    print("  (newtons of cycle-mean world-x force; a positive gain means the "
          "leg's\n   own motion is worth the drag it costs, against that baseline)")

    print("\n-- front vs hind shape mismatch (front-left against hind-left) --")
    print(f"  best alignment  {best:6.1f} deg RMS at {ph_b:.3f} cycles")
    print(f"  worst alignment {worst:6.1f} deg RMS at {ph_w:.3f} cycles")
    print(f"  the best alignment removes only "
          f"{100 * (1 - best / worst):.0f} % of the worst-case residual")

    print("\n-- first-harmonic phase relative to Front_Left [cycles] --")
    for kind in ("Thigh", "Calf"):
        row = {}
        ref = None
        for leg in LEGS:
            i = next(i for i, n in enumerate(names)
                     if leg.lower() in n.lower() and kind.lower() in n.lower())
            p = harmonic_phase(q[i])
            ref = p if ref is None else ref
            row[leg] = (p - ref) % 1.0
        print(f"  {kind:<6s} " +
              "  ".join(f"{leg.split('_')[0][0]}{leg.split('_')[1][0]}"
                        f" {v:.3f}" for leg, v in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
