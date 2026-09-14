#!/usr/bin/env python3
"""Every number \\cref{sec:results-nominal} of the thesis quotes, recomputed.

The subsection reports a nominal gait as a table of scalars, a paragraph of
constraint residuals and a paragraph of limit activity, and then generalises the
limit finding across the sweep.  All of that is measured here rather than read
off an MLflow dashboard, so the thesis and the stored solution cannot drift
apart: the run's own logged ``energy`` and ``cot`` are reproduced as a check on
the state layout, and the script exits non-zero if they disagree.

**The state layout, and why the energy check certifies it.**  ``X`` is the
legacy layout ``[p(3), quat(4), q(n_theta); v(3), w(3), qdot(n_theta)]`` — note
the 7/6 split, so joint angles are rows ``7:nq`` but joint rates are rows
``nq+6:``.  ``Xc`` is stored in *tangent* coordinates about a reference
quaternion the file does not record, so it is passed through
``collocation.collocation_states`` first, which recovers it and checks the
recovery against the grid nodes it shares.  Getting either wrong gives a
mechanical work that misses the logged value by a wide margin, so reproducing
``energy`` to five figures is what says the rows below are the right ones.

**Where the bounds are enforced.**  At the grid nodes and at the interior
collocation points, which together are exactly the columns of ``Xc`` — see the
long note in ``plot_limit_activity.py``.  Activity shares are therefore taken
over ``Xc``, never over ``X``, which would miss two of every three enforced
states.  ``ACTIVE_TOL`` and the normalisation are imported from that script so
the percentages here and the appendix figure cannot disagree.

**Per-joint dwell shares are reported over the grid nodes** (``X``), not over
``Xc``: a share like "this thigh sits at its lower stop for 20.8 % of the cycle"
is a statement about time, and only the grid nodes are equally spaced in time.
The aggregate "some joint is on some bound" shares use ``Xc``, matching the
appendix figure.  The two are labelled distinctly in the report.

Usage:
  python nominal_metrics.py                       # the thesis nominal
  python nominal_metrics.py --solution <npz> --results <sweep artifacts dir>
  python nominal_metrics.py --no-sweep            # nominal only, no sweep pass
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))

import sweep_io  # noqa: E402
from collocation import D_COLLOC, collocation_states  # noqa: E402
from gait_diagnostics import compute_traces  # noqa: E402
from plot_base_motion import base_pose  # noqa: E402
from plot_limit_activity import (  # noqa: E402
    ACTIVE_TOL, DWELL_FRAC, activity, dwell_masks, normalised)

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402
from stage1_gait_optimization.ocp_common import (  # noqa: E402
    collocation_coefficients,
    cost_of_transport,
    cycle_energy,
    limits_for,
)

# The thesis nominal: the v = 0.18 point of the cold co-design front.
_DEFAULT_RESULTS = (_ROOT / "experiment_results" / "mlruns" / "2" /
                    "69943d2cffc94ca5b78271a7df24c546" / "artifacts")
_DEFAULT_SOLUTION = _DEFAULT_RESULTS / "TLPG50_v0p180_T1p400.npz"

GRAVITY = 9.81
# The logged metrics must come back to this many significant figures, or the
# rows this script reads are not the rows the solver wrote.
ENERGY_RTOL = 1e-4

# Two tolerances, because they answer two different questions and conflating
# them is how the thesis draft ended up with two sets of percentages.  Both are
# imported from ``plot_limit_activity``, where they are defined and explained:
# ACTIVE_TOL (1e-3) asks "is this bound ACTIVE at the solution?", DWELL_FRAC
# (1e-2 of the quantity's span) asks "how long does the joint SIT on its stop?".
# DWELL_FRAC is the band the thesis's metric section defines and this prose
# quotes; ACTIVE_TOL is what the trace figure draws.

# How far a competing left/right alignment must sit from the best one before it
# counts as a genuinely different delay, in cycles.  See circular_shift_phase.
RIVAL_GAP = 0.15

# Backward-sweep windows shorter than this share of the cycle are not counted:
# a single collocation sample either side of zero is a sign flip in the trace,
# not a stroke.  Every window that survives here is at least three samples long,
# so the counts are not sensitive to the exact value.
MIN_SWEEP = 0.02


# ── helpers ──────────────────────────────────────────────────────────────────

def circular_shift_phase(a: np.ndarray, b: np.ndarray,
                         upsample: int = 64) -> tuple[float, float]:
    """Cycle fraction by which ``b`` lags ``a``, and the residual it leaves.

    Both are ``(n_joints, N)`` over one period.  A whole-node search quantises
    the phase at ``1/N``, which on this grid is 0.021 cycles — enough to leave a
    several-degree residual on a pair that mirrors exactly, and the thesis draft
    once read that residual as a constraint violation.  So the search is run on
    a band-limited resample: both signals are shifted in the Fourier domain,
    which is exact for the 8-harmonic shapes the symmetry constraint
    parametrises, and the phase comes back at ``1/(N*upsample)``.

    Preferred over ``argmax`` alignment: a thigh rests against its stop for a
    fifth of the cycle, so its maximum is a plateau and ``argmax`` picks an
    arbitrary point of it.

    Returns ``(phase, residual, rival_phase, rival_residual)``.
    """
    N = a.shape[1]
    fb = np.fft.rfft(b, axis=1)
    k = np.fft.rfftfreq(N, d=1.0 / N)
    scan = []
    for s in np.arange(N * upsample) / upsample:
        shifted = np.fft.irfft(fb * np.exp(-2j * np.pi * k * s / N), n=N, axis=1)
        scan.append((float(np.sqrt(np.mean((shifted - a) ** 2))), s / N))
    scan.sort()
    err, phase = scan[0]
    # The best rival delay, at least RIVAL_GAP cycles away.  A trajectory with a
    # strong second harmonic is nearly its own half-cycle shift, and then a
    # measured half-cycle delay is an artifact rather than a finding -- so the
    # margin to the nearest genuinely different alignment comes back with it,
    # and the thesis quotes that margin instead of asking the reader to trust
    # the fit.  Here the margin is three orders of magnitude, so the delay is
    # real; the legs are simply not self-similar under a half-cycle shift.
    rival = next(((e, ph) for e, ph in scan
                  if min(abs(ph - phase), 1.0 - abs(ph - phase)) > RIVAL_GAP),
                 (float("nan"), float("nan")))
    return phase, err, rival[1], rival[0]


def cyclic_runs(mask: np.ndarray) -> list[float]:
    """Lengths, as shares of the cycle, of the contiguous True runs in ``mask``.

    Cyclic: a run spanning the wrap-around is one run, not two.  Used to count
    how many times per cycle a foot sweeps backwards, which is the measurement
    that shows the stroke is not a single sweep.
    """
    n = len(mask)
    if not mask.any() or mask.all():
        return []
    out = []
    for b in (i for i in range(n) if mask[i] and not mask[(i - 1) % n]):
        length = 0
        while mask[(b + length) % n]:
            length += 1
        out.append(length / n)
    return out


def stroke_structure(robot, Xc, nq) -> dict:
    """How many times per cycle each foot sweeps backwards, per leg.

    The draft described this gait as an antiphase left--right pattern and let
    the reader infer one stroke per leg per cycle.  It is not: the calf carries
    its largest Fourier component at TWICE the stride frequency and each foot
    reverses its fore-aft direction several times per cycle.  The left/right
    delay is a relation between the two legs of a pair and says nothing about
    the shape of the stroke, so both are measured and reported together.

    ``v_foot_x`` is the foot velocity relative to the HULL, from
    ``gait_diagnostics.compute_traces`` -- the kinematic quantity, with the base
    twist zeroed.  No drag law is involved, so this stays inside what
    \cref{sec:results-nominal} is allowed to report; which of these sweeps
    actually produces thrust belongs to \cref{sec:results-structure}.
    """
    sweeps = {}
    for leg in ("Front_Left", "Front_Right", "Hind_Left", "Hind_Right"):
        v_foot_x = compute_traces(robot, leg, Xc)[0]
        sweeps[leg] = [r for r in cyclic_runs(v_foot_x < 0.0) if r >= MIN_SWEEP]
    return sweeps


def dominant_harmonic(trace: np.ndarray, n_harmonics: int = 6):
    """``(index of the largest harmonic, its amplitudes)`` of one cycle."""
    n = len(trace)
    amp = 2.0 * np.abs(np.fft.rfft(trace) / n)[1:n_harmonics + 1]
    return 1 + int(np.argmax(amp)), amp


def joint_kinds(names: list[str]) -> list[str]:
    """'Side' / 'Thigh' / 'Calf' per actuated joint, from its URDF name."""
    out = []
    for n in names:
        low = n.lower()
        out.append("Side" if "side" in low else
                   "Thigh" if "thigh" in low else
                   "Calf" if "calf" in low else "Other")
    return out


# ── the nominal point ────────────────────────────────────────────────────────

def measure_solution(path: Path, robot) -> dict:
    """Every scalar, residual and activity share for one stored solve."""
    meta = load_solution(path)
    T, N, nq = meta["T"], meta["N"], meta["nq"]
    X, U = meta["X"], meta["U"]
    d = D_COLLOC
    n_theta = robot.n_actuated
    _, _, _, B = collocation_coefficients(d)

    Xc, _, _ = collocation_states(robot, X, meta["Xc"], nq, N, d)
    q = X[7:nq, :]                       # joint angles, grid nodes
    qd = X[nq + 6:, :]                   # joint rates, grid nodes
    qc = Xc[7:nq, :]                     # joint angles, enforced samples
    qdc = Xc[nq + 6:, :]                 # joint rates, enforced samples

    q_lb, q_ub, v_ub, tau_ub = limits_for(robot)
    names = list(robot.model.names)[1:][-n_theta:]
    kinds = joint_kinds(names)

    # ── scalars ──
    forward = float(X[0, -1] - X[0, 0])
    speed = forward / T
    energy = cycle_energy(U, qdc, B, d, N, T)
    cot = cost_of_transport(energy, robot, forward)

    # Work split by sign of the instantaneous joint power, on the same Radau
    # quadrature the energy uses — so the two halves sum to the |·| total only
    # up to the sign changes inside an interval, which is why both are printed.
    dt = T / N
    p = U[:, :, None] * qdc.reshape(n_theta, N, d)
    w = np.asarray(B, dtype=float)[None, None, :] * dt
    w_pos = float(np.sum(np.where(p > 0, p, 0.0) * w))
    w_neg = float(np.sum(np.where(p < 0, p, 0.0) * w))

    # ── base motion ──
    # Through plot_base_motion.base_pose, not off the raw rows: X's velocity
    # block is the BODY-frame twist, and with the hull pitching by 27 deg over
    # the cycle, reading its x row as a world-frame forward speed is wrong by
    # more than a tenth of the peak.  Sharing the function also guarantees these
    # numbers are the ones fig:results-base-motion draws.
    pos, rpy, vel_world = base_pose(X, nq)
    v_fwd = vel_world[0]
    heave = pos[2]

    # ── constraint residuals ──
    res = {
        "speed": abs(speed - round(speed, 6)),
        "q_periodic": float(np.max(np.abs(q[:, -1] - q[:, 0]))),
        "qd_periodic": float(np.max(np.abs(qd[:, -1] - qd[:, 0]))),
        "heave_periodic": float(abs(heave[-1] - heave[0])),
        "quat_periodic": float(np.max(np.abs(X[3:7, -1] - X[3:7, 0]))),
    }

    # ── left/right phase of each symmetric pair ──
    # Thigh and calf only: the side joints are pinned to zero by the pose
    # constraints, so they carry no phase and would drag the fit to zero.
    pairs = {}
    for label, l_key, r_key in (("front", "Front_Left", "Front_Right"),
                                ("hind", "Hind_Left", "Hind_Right")):
        li = [i for i, n in enumerate(names)
              if l_key.lower() in n.lower() and kinds[i] != "Side"]
        ri = [i for i, n in enumerate(names)
              if r_key.lower() in n.lower() and kinds[i] != "Side"]
        if len(li) == len(ri) == 2:
            phase, resid, rph, rres = circular_shift_phase(q[li, :-1],
                                                           q[ri, :-1])
            pairs[label] = (phase, np.degrees(resid), rph, np.degrees(rres))

    # ── stroke structure and the harmonics of the base motion ──
    # Both exist to keep the prose honest about the SHAPE of the cycle: a
    # half-cycle left/right delay does not imply one sweep per leg, and this is
    # what says so.
    sweeps = stroke_structure(robot, Xc, nq)
    harmonics = {
        "v_x": dominant_harmonic(v_fwd[:-1]),
        "roll": dominant_harmonic(rpy[0, :-1]),
        "pitch": dominant_harmonic(rpy[1, :-1]),
        "yaw": dominant_harmonic(rpy[2, :-1]),
    }
    for i, n in enumerate(names):
        if kinds[i] != "Side":
            harmonics[n] = dominant_harmonic(np.degrees(q[i, :-1]))

    # ── limit activity ──
    # Over the enforced samples and at the solver's own tolerance: this is the
    # share the appendix figure draws, and the number that says whether a bound
    # is part of the solution at all.
    norm, alpha = normalised(robot, Xc, U, nq, N, T, d)
    agg = {k: float((np.abs(v) >= 1.0 - ACTIVE_TOL).any(axis=0).mean())
           for k, v in norm.items()}

    # Per-joint dwell, from the shared masks so that the per-joint shares here
    # and the sweep-wide panel of \cref{fig:results-pareto} are one definition.
    masks = dwell_masks(robot, X, U, nq)
    any_share = activity(masks)
    dwell = {
        "angle_lo": masks["angle_lo"].mean(axis=1),
        "angle_hi": masks["angle_hi"].mean(axis=1),
        "rate": masks["rate"].mean(axis=1),
        "any_angle": any_share["angle"],
        "any_rate": any_share["rate"],
    }

    return {
        "path": path, "T": T, "N": N, "names": names, "kinds": kinds,
        "speed": speed, "forward": forward, "freq": 1.0 / T,
        "energy": energy, "cot": cot, "power": energy / T,
        "w_pos": w_pos, "w_neg": w_neg,
        "tau_peak": float(np.max(np.abs(U))),
        "tau_peak_joint": names[int(np.argmax(np.max(np.abs(U), axis=1)))],
        "tau_peak_per_joint": np.max(np.abs(U), axis=1),
        "tau_rms": float(np.sqrt(np.mean(U ** 2))),
        "tau_ub": tau_ub, "v_ub": v_ub, "q_lb": q_lb, "q_ub": q_ub,
        "tau_util": float(np.max(np.abs(U) / tau_ub[:, None])),
        "rate_peak": float(np.max(np.abs(qdc))),
        "rate_util": float(np.max(np.abs(qdc) / v_ub[:, None])),
        "amplitude": np.degrees(qc.max(axis=1) - qc.min(axis=1)),
        "heave_pp": float(heave.max() - heave.min()) ,
        "heave_mean": float(heave.mean()),
        "v_fwd_peak": float(v_fwd.max()), "v_fwd_min": float(v_fwd.min()),
        "rpy_pp": rpy.max(axis=1) - rpy.min(axis=1),
        "rpy_lo": rpy.min(axis=1), "rpy_hi": rpy.max(axis=1),
        "res": res, "pairs": pairs, "activity": agg, "dwell": dwell,
        "alpha": alpha, "sweeps": sweeps, "harmonics": harmonics,
    }


def report_nominal(m: dict, row: dict | None) -> None:
    names, kinds = m["names"], m["kinds"]
    thigh = [i for i, k in enumerate(kinds) if k == "Thigh"]
    calf = [i for i, k in enumerate(kinds) if k == "Calf"]
    front_t = [i for i in thigh if "front" in names[i].lower()]
    hind_t = [i for i in thigh if "hind" in names[i].lower()]

    print(f"\n=== nominal: {m['path'].name} ===")
    if row is not None:
        print(f"  logged energy / cot        {row['energy']:.4f} / {row['cot']:.4f}")
    print(f"  recomputed energy / cot    {m['energy']:.4f} / {m['cot']:.4f}")
    print(f"  achieved speed             {m['speed']:.6f} m/s")
    print(f"  period T / frequency       {m['T']:.4f} s / {m['freq']:.4f} Hz")
    print(f"  distance per cycle         {m['forward']:.5f} m")
    print(f"  work / mean power          {m['energy']:.3f} J / {m['power']:.3f} W")
    print(f"  positive / negative work   {m['w_pos']:.3f} J / {m['w_neg']:.3f} J")
    print(f"  torque RMS                 {m['tau_rms']:.4f} Nm")
    print(f"  peak torque                {m['tau_peak']:.4f} Nm "
          f"= {100 * m['tau_util']:.1f} % of bound, at {m['tau_peak_joint']}")
    print(f"    thighs                   {m['tau_peak_per_joint'][thigh].min():.2f}"
          f" .. {m['tau_peak_per_joint'][thigh].max():.2f} Nm")
    print(f"    calves                   {m['tau_peak_per_joint'][calf].min():.2f}"
          f" .. {m['tau_peak_per_joint'][calf].max():.2f} Nm")
    print(f"  peak joint rate            {m['rate_peak']:.4f} rad/s "
          f"= {100 * m['rate_util']:.1f} % of bound")
    print(f"  mean / max joint amplitude {m['amplitude'].mean():.1f} deg / "
          f"{m['amplitude'].max():.1f} deg ({names[int(np.argmax(m['amplitude']))]})")
    print(f"  thigh angle box            {m['q_lb'][thigh[0]]:.3f} .. "
          f"{m['q_ub'][thigh[0]]:.3f} rad "
          f"({np.degrees(m['q_ub'][thigh[0]] - m['q_lb'][thigh[0]]):.1f} deg)")
    print(f"  forward base velocity      {m['v_fwd_min']:.4f} .. "
          f"{m['v_fwd_peak']:.4f} m/s")
    print(f"  heave                      {1e3 * m['heave_pp']:.1f} mm p-p about "
          f"{1e3 * m['heave_mean']:.1f} mm")
    for i, ax in enumerate(("roll", "pitch", "yaw")):
        print(f"  {ax:<26s} {m['rpy_pp'][i]:.2f} deg p-p "
              f"({m['rpy_lo'][i]:+.2f} .. {m['rpy_hi'][i]:+.2f})")

    print("  -- constraint residuals --")
    for k, v in m["res"].items():
        print(f"    {k:<24s} {v:.3e}")

    print("  -- left/right phase: cycles the RIGHT leg lags the LEFT --")
    for label, (phase, resid, rph, rres) in m["pairs"].items():
        print(f"    {label:<24s} {phase:.3f} cycles, {resid:.2f} deg RMS "
              f"(best rival {rph:.3f} cyc at {rres:.2f} deg)")

    print("  -- stroke structure: backward foot sweeps per cycle --")
    for leg, runs in m["sweeps"].items():
        print(f"    {leg:<24s} {len(runs)} sweeps, lengths "
              f"{', '.join(f'{r:.3f}' for r in runs)} of the cycle")

    print("  -- dominant harmonic of the cycle (1 = once per cycle) --")
    for key, (h, amp) in m["harmonics"].items():
        print(f"    {key:<24s} h{h}   amplitudes h1..h{len(amp)}: "
              f"{'  '.join(f'{a:.2f}' for a in amp)}")

    print(f"  -- ANY joint on the bound, enforced samples, tol {ACTIVE_TOL:g} --")
    for k, v in m["activity"].items():
        print(f"    {k:<24s} {100 * v:.1f} %")

    print(f"  -- ANY joint on the bound, grid nodes, {100 * DWELL_FRAC:g} % of span --")
    print(f"    {'angle':<24s} {100 * m['dwell']['any_angle']:.1f} %")
    print(f"    {'rate':<24s} {100 * m['dwell']['any_rate']:.1f} %")

    print(f"  -- per-joint dwell, share of the cycle, {100 * DWELL_FRAC:g} % of span --")
    print(f"    {'joint':<22s}{'angle lo':>10s}{'angle hi':>10s}{'rate':>10s}")
    for i, n in enumerate(names):
        print(f"    {n:<22s}{100 * m['dwell']['angle_lo'][i]:>9.1f}%"
              f"{100 * m['dwell']['angle_hi'][i]:>9.1f}%"
              f"{100 * m['dwell']['rate'][i]:>9.1f}%")
    for label, idx in (("front thighs", front_t), ("hind thighs", hind_t)):
        lo = m["dwell"]["angle_lo"][idx]
        hi = m["dwell"]["angle_hi"][idx]
        print(f"    {label:<22s} lower {100 * lo.min():.1f}-{100 * lo.max():.1f} %,"
              f" upper {100 * hi.min():.1f}-{100 * hi.max():.1f} %")
    n_rate = int((m["dwell"]["rate"] > 0).sum())
    print(f"    joints touching the rate bound: {n_rate} of {len(names)}")


# ── the sweep-wide claim ─────────────────────────────────────────────────────

def report_sweep(results: Path, robot) -> None:
    """Is 'kinematically limited, not torque limited' one gait or the platform?

    Reported over every feasible solve and again over the Pareto front alone,
    because the thesis makes the claim about the platform and the front is the
    part of it the rest of the chapter argues from.
    """
    rows = [r for r in sweep_io.load_rows(results)
            if r["feasible"] and np.isfinite(r["cot"])]
    print(f"\n=== sweep: {len(rows)} feasible solves in {results.name} ===")

    recs = []
    for row in sorted(rows, key=lambda r: (r["speed"], r["gait"])):
        path = results / sweep_io.npz_of(row)
        if not path.exists():
            continue
        m = measure_solution(path, robot)
        thigh = [i for i, k in enumerate(m["kinds"]) if k == "Thigh"]
        recs.append({
            "row": row, "speed": row["speed"], "front": bool(row.get("pareto")),
            "tau_util": m["tau_util"], "rate_util": m["rate_util"],
            "tau_peak": m["tau_peak"], "rate_peak": m["rate_peak"],
            # The DWELL share, not the ACTIVE_TOL one: this is the series
            # \cref{sec:results-tradeoff} quotes and the activity panel draws.
            "rate_share": m["dwell"]["any_rate"],
            "angle_share": m["dwell"]["any_angle"],
            # Both ends of the thigh spread: the claim "the thighs traverse
            # their box" is only as strong as its weakest thigh, so the minimum
            # is what the sweep-wide sentence has to be written from.
            "thigh_sweep": float(m["amplitude"][thigh].max()),
            "thigh_sweep_min": float(m["amplitude"][thigh].min()),
            "tag": sweep_io.tag_of(row),
        })

    thigh_box = np.degrees(m["q_ub"][thigh[0]] - m["q_lb"][thigh[0]])
    for label, sel in (("all feasible", recs),
                       ("Pareto front", [r for r in recs if r["front"]])):
        if not sel:
            continue
        tau = np.array([r["tau_util"] for r in sel])
        rate = np.array([r["rate_util"] for r in sel])
        sweep = np.array([r["thigh_sweep"] for r in sel])
        sweep_min = np.array([r["thigh_sweep_min"] for r in sel])
        full = sweep >= thigh_box - 1.0
        full_all = sweep_min >= thigh_box - 1.0
        print(f"\n  -- {label} ({len(sel)} solves) --")
        print(f"    peak torque utilisation   max {100 * tau.max():.1f} % "
              f"({max(sel, key=lambda r: r['tau_util'])['tag']}), "
              f"median {100 * np.median(tau):.1f} %")
        print(f"    peak rate utilisation     min {100 * rate.min():.1f} %, "
              f"at the bound in {int((rate >= 1 - ACTIVE_TOL).sum())} of {len(sel)}")
        print(f"    some thigh sweeps the full {thigh_box:.1f} deg box in "
              f"{int(full.sum())} of {len(sel)}; all four in "
              f"{int(full_all.sum())}")
        for r in sel:
            if r["thigh_sweep"] < thigh_box - 1.0:
                print(f"      exception {r['tag']:<24s} v={r['speed']:.3f} "
                      f"widest thigh {r['thigh_sweep']:.1f} deg")
        narrow = min(sel, key=lambda r: r["thigh_sweep_min"])
        print(f"    narrowest single thigh anywhere: "
              f"{narrow['thigh_sweep_min']:.1f} deg ({narrow['tag']})")

    front = sorted([r for r in recs if r["front"]], key=lambda r: r["speed"])
    if front:
        print(f"\n    share of the cycle on the bound along the front, "
              f"{100 * DWELL_FRAC:g} % of span:")
        print(f"      {'v':>7s}{'rate':>9s}{'angle':>9s}")
        for r in front:
            print(f"      {r['speed']:7.3f}{100 * r['rate_share']:8.1f}%"
                  f"{100 * r['angle_share']:8.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solution", type=Path, default=_DEFAULT_SOLUTION)
    ap.add_argument("--results", type=Path, default=_DEFAULT_RESULTS)
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the sweep-wide pass (it reloads every solve)")
    args = ap.parse_args()

    robot = load_robot("amph")
    m = measure_solution(args.solution, robot)

    row = None
    if args.results.exists():
        for r in sweep_io.load_rows(args.results):
            if r["feasible"] and sweep_io.npz_of(r) == args.solution.name:
                row = r
                break
    report_nominal(m, row)

    if row is not None:
        for key in ("energy", "cot"):
            if not np.isclose(m[key], row[key], rtol=ENERGY_RTOL):
                print(f"\nFAIL: recomputed {key} {m[key]:.6f} does not reproduce "
                      f"the logged {row[key]:.6f}; the state layout read here is "
                      f"not the one the solver wrote.")
                return 1

    if not args.no_sweep and args.results.exists():
        report_sweep(args.results, robot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
