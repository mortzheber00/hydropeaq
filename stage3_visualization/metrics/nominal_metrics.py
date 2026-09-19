#!/usr/bin/env python3
"""Recompute all numbers reported for the nominal gait (thesis, results-nominal).

Prints scalars, constraint residuals, left/right phase, stroke structure,
harmonics and limit activity for one solution, plus sweep-wide limit usage.
Exits non-zero if the recomputed energy/COT do not match the logged values,
which would mean the state layout is read incorrectly.

X layout: [p(3), quat(4), q; v(3), w(3), qdot], i.e. joint angles at rows 7:nq
and joint rates at nq+6:. Active shares use Xc (all enforced points); dwell
shares use the equally spaced grid nodes. Tolerances come from
plot_limit_activity.py.

Usage:
  python stage3_visualization/metrics/nominal_metrics.py
  python stage3_visualization/metrics/nominal_metrics.py --solution <npz> --results <sweep dir>
  python stage3_visualization/metrics/nominal_metrics.py --no-sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_gait_optimization"))

from stage3_visualization.common import sweep_io  # noqa: E402
from stage3_visualization.common.collocation import D_COLLOC, collocation_states  # noqa: E402
from stage3_visualization.thrust.gait_diagnostics import compute_traces  # noqa: E402
from stage3_visualization.gait.plot_base_motion import base_pose  # noqa: E402
from stage3_visualization.gait.plot_limit_activity import (  # noqa: E402
    ACTIVE_TOL, DWELL_FRAC, activity, dwell_masks, normalised)

from stage1_gait_optimization.hydro_model import load_robot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402
from stage1_gait_optimization.ocp_common import (  # noqa: E402
    collocation_coefficients,
    cost_of_transport,
    cycle_energy,
    limits_for,
)

# Thesis nominal: the v = 0.18 m/s point of the co-design front
_DEFAULT_RESULTS = (_ROOT / "experiment_results" / "mlruns" / "2" /
                    "69943d2cffc94ca5b78271a7df24c546" / "artifacts")
_DEFAULT_SOLUTION = _DEFAULT_RESULTS / "TLPG50_v0p180_T1p400.npz"

GRAVITY = 9.81
# Relative tolerance for reproducing the logged energy and COT
ENERGY_RTOL = 1e-4

# Minimum distance [cycles] of an alternative left/right alignment from the best one
RIVAL_GAP = 0.15

# Backward sweeps shorter than this share of the cycle are ignored (sign noise).
MIN_SWEEP = 0.02


# --- Helpers ---

def circular_shift_phase(a: np.ndarray, b: np.ndarray,
                         upsample: int = 64) -> tuple[float, float]:
    """Phase [cycles] by which ``b`` lags ``a``, found by Fourier-domain shifting.

    ``a`` and ``b`` are ``(n_joints, N)`` over one period. Sub-node resolution
    (``1/(N*upsample)``) avoids residuals from phase quantisation. Returns
    ``(phase, residual, rival_phase, rival_residual)``, where the rival is the
    best alignment at least RIVAL_GAP away.
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
    # A close rival (e.g. with strong second harmonics) makes the phase ambiguous.
    rival = next(((e, ph) for e, ph in scan
                  if min(abs(ph - phase), 1.0 - abs(ph - phase)) > RIVAL_GAP),
                 (float("nan"), float("nan")))
    return phase, err, rival[1], rival[0]


def cyclic_runs(mask: np.ndarray) -> list[float]:
    """Lengths (cycle shares) of the True runs in a cyclic ``mask``."""
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
    """Backward foot sweeps per cycle for each leg (lengths as cycle shares).

    Uses the hull-relative foot velocity from ``gait_diagnostics.compute_traces``
    (kinematics only, no drag model).
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
    """Joint type ('Side', 'Thigh', 'Calf' or 'Other') from each URDF name."""
    out = []
    for n in names:
        low = n.lower()
        out.append("Side" if "side" in low else
                   "Thigh" if "thigh" in low else
                   "Calf" if "calf" in low else "Other")
    return out


# --- Nominal solution ---

def measure_solution(path: Path, robot) -> dict:
    """All reported scalars, residuals and activity shares of one solution."""
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

    # Scalars
    forward = float(X[0, -1] - X[0, 0])
    speed = forward / T
    energy = cycle_energy(U, qdc, B, d, N, T)
    cot = cost_of_transport(energy, robot, forward)

    # Positive/negative work on the same quadrature (may differ slightly from
    # the |.| total where the sign changes within an interval)
    dt = T / N
    p = U[:, :, None] * qdc.reshape(n_theta, N, d)
    w = np.asarray(B, dtype=float)[None, None, :] * dt
    w_pos = float(np.sum(np.where(p > 0, p, 0.0) * w))
    w_neg = float(np.sum(np.where(p < 0, p, 0.0) * w))

    # Base motion; base_pose rotates the body-frame twist into the world frame.
    pos, rpy, vel_world = base_pose(X, nq)
    v_fwd = vel_world[0]
    heave = pos[2]

    # Constraint residuals
    res = {
        "speed": abs(speed - round(speed, 6)),
        "q_periodic": float(np.max(np.abs(q[:, -1] - q[:, 0]))),
        "qd_periodic": float(np.max(np.abs(qd[:, -1] - qd[:, 0]))),
        "heave_periodic": float(abs(heave[-1] - heave[0])),
        "quat_periodic": float(np.max(np.abs(X[3:7, -1] - X[3:7, 0]))),
    }

    # Left/right phase per pair (thigh and calf; side joints are pinned)
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

    # Stroke structure and dominant harmonics
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

    # Active shares over the enforced samples (as in the appendix figure)
    norm, alpha = normalised(robot, Xc, U, nq, N, T, d)
    agg = {k: float((np.abs(v) >= 1.0 - ACTIVE_TOL).any(axis=0).mean())
           for k, v in norm.items()}

    # Dwell shares, same definition as the sweep-wide activity figure
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
    """Print the nominal-solution report (``row`` = logged summary row, if found)."""
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


# --- Sweep ---

def report_sweep(results: Path, robot) -> None:
    """Torque/rate utilisation and thigh range over all feasible solves and the Pareto front."""
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
            # Dwell shares, as in the activity figure
            "rate_share": m["dwell"]["any_rate"],
            "angle_share": m["dwell"]["any_angle"],
            # Widest and narrowest thigh range
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
