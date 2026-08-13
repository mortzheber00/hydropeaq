#!/usr/bin/env python3
"""Measure whether BODY2's prescribed legs actually stay assembled in Gazebo.

BODY2's URDF is a 24-joint tree standing in for eight real degrees of freedom,
because URDF cannot express the two loop-closure pins per leg (P6, joining
Link_*2.1 to Link_*1.3, and P8, joining Link_*1.3 to Link_*2.3).  Replay
prescribes all 24 angles from ``Body2CoordinateMap``, which solves the linkage
in closed form, so the pins coincide *by construction* -- as long as the angles
Gazebo ends up holding are the ones the map asked for.

Whether they are is the question this answers, and it is worth answering with a
number rather than by eye.  Only ``base_link`` and the four paddles
(``Link_*2.3``) carry collision geometry, so those are the only bodies the SPH
solver couples to, and each paddle is a plain serial descendant of the base
through Joint_*2_1 -> Joint_*2_2 -> Joint_*2_3.  A small gap at P6/P8 is
therefore cosmetic: it opens on Link_*1.3, which the fluid never touches.  A
*large* gap is not, because it means the prescribed angles have left the
linkage manifold -- and then the paddle is not where the OCP thinks it is
either, which is exactly what the validation rests on.

Three measurements, all self-consistency checks on the logged angles, so none
of them needs the bag and the solution to be time-aligned:

    pin gap        P6 and P8 reconstructed from each of the two links they
                   join, and the distance between the two reconstructions
                   *perpendicular to the pin axis*.  The two links are plates
                   sitting side by side on the pin, so they are separated along
                   it by design -- 4.5 to 7.8 mm depending on the pin -- and
                   that offset is structure, not slack.  What must stay zero is
                   the perpendicular part, which says the pin is still one pin.
                   Nothing measures the axial part, because every joint axis in
                   the tree is parallel to the others: no configuration of it
                   can move a link out of its leg plane, so that number could
                   only ever read zero.

    map residual   the 16 passive angles against what the coordinate map
                   produces from the 8 logged hip angles.  This is directly
                   comparable to the 40 deg RMSE measured when the loops were
                   modelled as SDF joints and driven through ros_control.

    paddle error   Link_*2.3's pose relative to the base, from the logged
                   angles against the map's, which is the quantity the fluid
                   coupling actually sees.

Record a bag first (no fluid, no gravity -- whatever the legs do there is the
prescription alone):

    roslaunch BODY2 joint_prescription.launch \\
        npz_path:=/home/ws/task3_solution_body2.npz \\
        bag_path:=/home/ws/prescription.bag n_repeat:=3
    python3 stage2_sim_validation/check_prescription.py --bag /home/ws/prescription.bag

Exits non-zero if the worst pin gap exceeds ``--tol-mm``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stage1_gait_optimization"))
sys.path.insert(0, str(REPO_ROOT / "src" / "BODY2" / "scripts"))

from hydro_model import load_robot  # noqa: E402
from hydro_model.robots.body2 import LEG_NAMES, urdf_joint  # noqa: E402
from hydro_model.robots.body2_map import JOINT_KEYS  # noqa: E402

# Which two links each cut pin joins, and which link frame the tip belongs to.
# dump_linkage.py freezes each of these points in one owning link only, because
# that is all the cylinder specs need; measuring a gap needs both sides, so the
# second one is derived here from leg_linkage_sim's own geometry.
CUT_PINS = {"P6": ("2.1", "1.3"), "P8": ("1.3", "2.3")}


def local_points(leg_name: str) -> dict:
    """``{pin: {link_key: offset in that link's frame}}`` for one leg.

    Derived from the STL hole fits rather than read from body2_linkage.json --
    the frozen file carries each pin in one link only.  ``tests/
    test_body2_linkage.py`` is what keeps the two derivations agreeing.
    """
    from leg_linkage_sim import Leg

    leg = Leg(leg_name)
    out = {}
    for pin_name, links in CUT_PINS.items():
        p_xz = leg.pins0[pin_name]
        out[pin_name] = {}
        for key in links:
            R, o = leg.frames[f"Link_{leg_name}{key}"]
            # The pin is a line parallel to world Y; take the point on it level
            # with the link frame origin, so no spurious out-of-plane offset.
            p_world = np.array([p_xz[0], o[1], p_xz[1]])
            out[pin_name][key] = R.T @ (p_world - o)
    return out


# ── Bag ──────────────────────────────────────────────────────────────────────

def read_bag_joint_states(bag_path: str, ros_ns: str):
    """Return (times [s, from bag start], {joint name: (M,) angles [rad]})."""
    import rosbag

    times, rows, names = [], [], None
    with rosbag.Bag(bag_path) as bag:
        for _, msg, t in bag.read_messages(topics=[f"/{ros_ns}/joint_states"]):
            if names is None:
                names = list(msg.name)
            rows.append(np.asarray(msg.position, dtype=float))
            times.append(t.to_sec())
    if names is None:
        raise SystemExit(f"{bag_path}: no /{ros_ns}/joint_states messages")

    times = np.asarray(times)
    return times - times[0], dict(zip(names, np.stack(rows, axis=1)))


# ── Configuration vectors ────────────────────────────────────────────────────

def _tree_joints(model):
    """(name, idx_q - 7, nq) per tree joint, skipping the free flyer."""
    return [(model.names[j], model.joints[j].idx_q - 7, model.joints[j].nq)
            for j in range(1, model.njoints) if model.joints[j].idx_q >= 7]


def q_from_angles(robot, angles: dict) -> np.ndarray:
    """Full configuration from a name -> angle mapping.

    The base is left at the identity: every quantity below is measured relative
    to the base, so where the robot floats is irrelevant.
    """
    model = robot.model
    q = pin.neutral(model)
    for name, iq, nq_j in _tree_joints(model):
        a = angles[name]
        # A continuous joint stores (cos, sin) rather than an angle.
        q[7 + iq:7 + iq + nq_j] = (np.cos(a), np.sin(a)) if nq_j == 2 else (a,)
    return q


def map_angles(robot, hips: np.ndarray) -> dict:
    """Tree angles the coordinate map produces from the eight hip angles."""
    q_j = robot.coord_map.expand_numeric(hips)
    out = {}
    for name, iq, nq_j in _tree_joints(robot.model):
        out[name] = (np.arctan2(q_j[iq + 1], q_j[iq]) if nq_j == 2 else q_j[iq])
    return out


# ── Measurements ─────────────────────────────────────────────────────────────

def pin_gaps(robot, q: np.ndarray, points: dict) -> dict:
    """``{(leg, pin): gap [m]}`` perpendicular to the pin, at one configuration.

    ``q``'s base is the identity, so this is measured in the base frame and a
    free-floating base cannot contaminate it.  The pin axes are parallel to the
    base frame's Y: the legs are planar and ``prepare_urdf.py`` keeps their
    planes normal to it, which is the whole reason the base may only be yawed.
    """
    robot.forward_kinematics(q)
    axis = np.array([0.0, 1.0, 0.0])
    out = {}
    for leg in LEG_NAMES:
        for pin_name, links in CUT_PINS.items():
            p = []
            for key in links:
                oMf = robot.frame_placement(robot._frame_id(f"Link_{leg}{key}"))
                p.append(np.asarray(oMf.translation)
                         + np.asarray(oMf.rotation) @ points[leg][pin_name][key])
            d = p[0] - p[1]
            out[(leg, pin_name)] = float(np.linalg.norm(d - (d @ axis) * axis))
    return out


def paddle_pose(robot, q: np.ndarray) -> dict:
    """``{leg: SE3}`` of each paddle relative to the base."""
    robot.forward_kinematics(q)
    base = robot.frame_placement(robot._frame_id(robot.spec.base_link))
    return {leg: base.inverse() * robot.frame_placement(
        robot._frame_id(f"Link_{leg}2.3")) for leg in LEG_NAMES}


def _stats(x: np.ndarray, t: np.ndarray):
    """(rms, max, time of max)."""
    i = int(np.argmax(np.abs(x)))
    return float(np.sqrt(np.mean(x ** 2))), float(abs(x[i])), float(t[i])


# ── Report ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bag", default="/home/ws/prescription.bag")
    ap.add_argument("--tol-mm", type=float, default=1.0,
                    help="worst allowed pin gap before this exits non-zero")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth bag sample (the FK is the slow part)")
    args = ap.parse_args()

    robot = load_robot("body2")
    t, logged = read_bag_joint_states(args.bag, robot.spec.ros)
    t = t[::args.stride]
    logged = {k: v[::args.stride] for k, v in logged.items()}

    tree_names = [n for n, _, _ in _tree_joints(robot.model)]
    missing = [n for n in tree_names if n not in logged]
    if missing:
        raise SystemExit(
            f"{args.bag} is missing {len(missing)} tree joint(s), e.g. {missing[:3]}.\n"
            f"  The URDF's joint_state_publisher must list all {len(tree_names)}."
        )
    hip_names = list(robot.spec.actuated_joint_names)

    print(f"\nBODY2 prescribed-leg integrity")
    print(f"  bag      {args.bag}")
    print(f"  samples  {len(t)} over {t[-1]:.2f} s\n")

    points = {leg: local_points(leg) for leg in LEG_NAMES}

    keys = [(leg, p) for leg in LEG_NAMES for p in CUT_PINS]
    gap = {k: np.zeros(len(t)) for k in keys}
    resid = {n: np.zeros(len(t)) for n in tree_names}
    dpos = {leg: np.zeros(len(t)) for leg in LEG_NAMES}
    drot = {leg: np.zeros(len(t)) for leg in LEG_NAMES}

    for k in range(len(t)):
        angles = {n: logged[n][k] for n in tree_names}
        q_log = q_from_angles(robot, angles)

        for key, perp in pin_gaps(robot, q_log, points).items():
            gap[key][k] = perp
        pose_log = paddle_pose(robot, q_log)

        want = map_angles(robot, np.array([logged[n][k] for n in hip_names]))
        for n in tree_names:
            e = angles[n] - want[n]
            resid[n][k] = np.arctan2(np.sin(e), np.cos(e))
        pose_map = paddle_pose(robot, q_from_angles(robot, want))

        for leg in LEG_NAMES:
            d = pose_map[leg].inverse() * pose_log[leg]
            dpos[leg][k] = np.linalg.norm(d.translation)
            drot[leg][k] = np.linalg.norm(pin.log3(d.rotation))

    # ── pin gap ──────────────────────────────────────────────────────────
    print("Pin gap, perpendicular to the pin [mm]")
    print("                    P6                        P8")
    print("  leg          rms     max   at [s]      rms     max   at [s]")
    worst = 0.0
    for leg in LEG_NAMES:
        cells = []
        for p in ("P6", "P8"):
            r, m, tm = _stats(gap[(leg, p)] * 1e3, t)
            worst = max(worst, m)
            cells.append(f"{r:7.3f} {m:7.3f} {tm:7.2f}")
        print(f"  {leg:<10s} {cells[0]}   {cells[1]}")
    allgap = np.concatenate([gap[k] for k in keys]) * 1e3
    print(f"  {'ALL':<10s} {np.sqrt(np.mean(allgap ** 2)):7.3f} {allgap.max():7.3f}")

    # ── passive-joint residual ───────────────────────────────────────────
    passive = [urdf_joint(leg, k) for leg in LEG_NAMES
               for k in JOINT_KEYS if k not in ("1.1", "2.1")]
    print("\nPassive-joint residual vs the coordinate map [deg]")
    r_all = np.degrees(np.concatenate([resid[n] for n in passive]))
    for n in passive:
        r, m, tm = _stats(np.degrees(resid[n]), t)
        if m > 0.5:                      # only the joints that are actually off
            print(f"  {n:<16s} rms {r:8.3f}   max {m:8.3f} at {tm:.2f} s")
    print(f"  {'ALL':<16s} rms {np.sqrt(np.mean(r_all ** 2)):8.3f} "
          f"  max {np.abs(r_all).max():8.3f}")

    # ── paddle placement ─────────────────────────────────────────────────
    print("\nPaddle pose vs the coordinate map (relative to base)")
    print("  leg        pos rms [mm]  pos max [mm]   rot rms [deg]  rot max [deg]")
    for leg in LEG_NAMES:
        pr, pm, _ = _stats(dpos[leg] * 1e3, t)
        rr, rm, _ = _stats(np.degrees(drot[leg]), t)
        print(f"  {leg:<10s} {pr:12.3f} {pm:13.3f} {rr:15.3f} {rm:14.3f}")

    ok = worst <= args.tol_mm
    print(f"\n{'PASS' if ok else 'FAIL'}: worst pin gap {worst:.3f} mm "
          f"(tolerance {args.tol_mm:.3f} mm)")
    if ok:
        print("  The prescription is on the linkage manifold, so the paddles are\n"
              "  where the OCP puts them and any visible seam at P6/P8 is the\n"
              "  missing pin being drawn, not the leg coming apart.")
    else:
        print("  The prescribed angles have left the manifold.  Check, in order:\n"
              "  SUBDIV_CLOSED_CHAIN in replay_trajectory.py (the plugin lerps the\n"
              "  24 angles independently between waypoints, and the manifold is\n"
              "  nonlinear); the np.unwrap over 24 joints; a _circle_circle branch\n"
              "  flip mid-gait; SetPosition's preserveWorldVelocity leaving link\n"
              "  velocities the solver then integrates for a step.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
