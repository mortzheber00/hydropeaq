#!/usr/bin/env python3
"""The numbers \\cref{sec:results-structure} quotes: how the stroke earns thrust.

``gait_diagnostics.py`` already reports the per-leg impulse split and
``plot_stroke_benefit.py`` the frozen-leg counterfactual; both are figures with a
report attached.  What is missing, and what this script adds, is the three
measurements the thesis argues from that neither of them computes:

  1. **The power-stroke fraction**, Radau-weighted, per leg.  It is the share of
     the cycle in which the foot moves backwards relative to the hull, and the
     thesis compares it against the 25/33/50 % duty fractions of the seed gaits.
     Reported as a share AND as a count of separate windows, because the stroke
     is not one contiguous sweep -- see \\cref{sec:results-nominal}.

  2. **The submersion asymmetry** ``alpha_pow`` against ``alpha_rec``: the mean
     submerged fraction of a leg link over the power and the recovery phase.
     This is the mechanism H3 predicts -- lift the leg on the return so it drags
     less -- and it has to be measured rather than assumed, because the
     optimizer is free to buy the same asymmetry through speed instead.

     **Why the submerged fraction and not the depth of the foot.**  Foot depth
     was the obvious first choice and it is a bad metric here.  The hind foot
     travels between 14 mm and 177 mm below the surface WITHIN its power phase,
     so the phase mean carries a standard deviation of 49 mm while the quantity
     it is being used to compare -- the power-minus-recovery difference -- is
     4 mm.  A number thirteen times smaller than its own scatter cannot support
     a claim.  The submerged fraction has none of that problem: it is bounded in
     [0, 1], it saturates, and it is what the drag law is actually a function
     of.  It is also what explains the hind legs, whose calf sits at alpha ~ 1
     however deep the foot happens to be, so a large vertical excursion buys
     them nothing.  Reported per link, with the within-phase spread, so a mean
     that is not representative shows up as one.

  3. **The best frozen-leg baseline.**  ``plot_stroke_benefit.py`` asks whether
     a leg's motion is worth its drag by freezing it at its CYCLE-MEAN pose.
     That baseline is arbitrary and it flatters the stroke: the cycle mean of a
     trajectory optimized for stroking has no reason to be a good pose to hold,
     and here it is a bad one.  The honest baseline is the BEST pose the leg
     could be held at, which is what a designer choosing to keep a leg still
     would pick.  It is found by sweeping the leg's (thigh, calf) box -- the
     side joint is pinned to zero by the pose constraints -- and taking the pose
     with the least rearward cycle-mean force.

     This changes the conclusion for the hind legs, so it is not a refinement.
     Against the mean pose all four legs gain; against the best pose only the
     front pair does.

     What the counterfactual does NOT say: that retracting the hind legs is a
     better gait.  The base motion is held as solved, and that base motion was
     produced with the hind legs stroking, so this is an attribution at the
     solved trajectory and not a comparison of two gaits.  Answering the design
     question properly means re-solving the OCP with the hind legs pinned.

  4. **The front-versus-hind shape mismatch.**  Within a pair the legs are
     mirror copies by construction (the symmetry constraint), so the only
     asymmetry left to discover is front against hind.  It is quantified by
     aligning the front-left and hind-left joint trajectories at every relative
     phase and reporting the best and the worst residual: if even the best
     alignment leaves a large residual, the two pairs share no stroke shape and
     no front--hind phase offset is meaningful.

**Terminology, which has bitten once.**  "Power phase" here is always the
KINEMATIC one, ``v_foot_x < 0`` in the base frame -- the interval in which the
foot sweeps backwards.  It is NOT the interval of positive joint power
(``tau qdot > 0``), which \\cref{sec:results-nominal} reports and which is a
different set of samples.  The two are never given the same unqualified name.

Everything is sampled at the collocation points and averaged on the Radau
weights, as everywhere else in stage 3: the points are not equally spaced, so an
unweighted mean over half a cycle silently reweights it.

Usage:
  python stroke_asymmetry.py
  python stroke_asymmetry.py --solution <npz>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))

from collocation import D_COLLOC, collocation_states, quadrature_weights  # noqa: E402
from drag_model import (  # noqa: E402
    _leg_joint_indices,
    leg_cycle_drag,
    leg_mean_pose,
    link_drag_terms,
    require_supported,
)
from gait_diagnostics import compute_traces  # noqa: E402
from nominal_metrics import MIN_SWEEP, cyclic_runs  # noqa: E402

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402
from stage1_gait_optimization.ocp_common import limits_for  # noqa: E402

_DEFAULT_SOLUTION = (_ROOT / "experiment_results" / "mlruns" / "2" /
                     "69943d2cffc94ca5b78271a7df24c546" / "artifacts" /
                     "TLPG50_v0p180_T1p400.npz")

LEGS = ("Front_Left", "Front_Right", "Hind_Left", "Hind_Right")

# The duty fractions the seed gaits prescribe, from the source they come from.
# The thesis compares the converged fraction against these: the optimizer was
# handed three of them and is free to keep one.
SEED_DUTY = {"LSPG25": 0.25, "LSPG33": 0.33, "TLPG50": 0.50}


def link_submersion(robot, Xc, nq, link_name) -> np.ndarray:
    """Submerged fraction of one link, per collocation sample.

    ``drag_model.submersion_ratio`` via ``link_drag_terms``, i.e. the same
    quantity the drag law multiplies by -- not a re-derivation of it, so the
    asymmetry reported here is the one the solver actually saw.
    """
    out = np.empty(Xc.shape[1])
    for t in range(Xc.shape[1]):
        q = Xc[:nq, t]
        robot.forward_kinematics(q)
        out[t] = link_drag_terms(robot, link_name, q)[2]
    return out


def best_frozen_pose(robot, Xc, nq, N, leg, grid: int = 21):
    """The held pose costing the least rearward force, and what it costs.

    Swept over the leg's own joint-angle box, so the pose is one the robot can
    actually adopt.  Only thigh and calf are searched: the side joint is pinned
    to zero by the pose constraints, and holding it elsewhere would compare
    against a pose the OCP was never allowed to use.

    Returns ``(force, (thigh, calf), alpha_calf)``.  The submersion of the calf
    at the winning pose comes back with it because it is the explanation: the
    optimum is the leg folded up with its calf clear of the water, and a reader
    should not have to take that on trust.

    Cost is ``grid**2`` cycle evaluations per leg, which is the reason this is a
    coarse sweep and not a solve.  The optimum sits in a corner of the box, so
    resolution buys little; 13 and 21 agree to 1e-3 N here.
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
        """``(mean_power, mean_recovery, sd_power, sd_recovery)``.

        The two standard deviations are returned with every split, not as an
        afterthought: a phase mean is only worth quoting when the phases differ
        by more than the samples inside them scatter, and that comparison is the
        one this script exists to make honest.
        """
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
    """Best and worst RMS residual over all relative phases, in degrees.

    Both are ``(n_joints, N)`` in radians.  The WORST residual is reported
    alongside the best because the claim being tested is "these two legs share
    no stroke shape": a best residual that is only slightly under the worst is
    what says the alignment is meaningless, and the best alone cannot say it.
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
            # A phase mean earns its place only if the gap between the phases
            # exceeds the scatter within them.
            ok = "yes" if abs(ap - ar) > max(sp, sr) else "NO -- gap < spread"
            print(f"  {leg + '/' + seg:<22s}{ap:>8.3f}{ar:>8.3f}"
                  f"{ap / ar:>7.2f}{sp:>8.3f}{sr:>8.3f}   {ok}")

    duties = np.array([m["duty"] for m in per_leg.values()])
    print(f"\n  duty range {100 * duties.min():.1f}-{100 * duties.max():.1f} %, "
          f"against the seeds " +
          ", ".join(f"{g} {100 * f:.0f} %" for g, f in SEED_DUTY.items()))

    # ── front against hind: the only asymmetry the symmetry constraint leaves ──
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
