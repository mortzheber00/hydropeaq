"""Post-process BODY2's SolidWorks URDF export in place (run after every CAD export).

Steps:
0. Rename joints ``Joint_FL1.1`` -> ``Joint_FL1_1`` (ROS names cannot contain
   dots; link names and mesh files keep them).
1. Make all leg joints continuous, move their origins onto the pin centres and
   re-home all legs to the same zero pose (link origins are compensated).
2. Move the base frame to the CoM with +x forward (yaw only, so the leg planes
   stay normal to y); the inertia tensor is rotated too.
3. Regenerate body2_linkage.json.

Idempotent. Loop closures are not representable in URDF; see
hydro_model/robots/body2_map.py.

Usage:
  python3 src/BODY2/scripts/prepare_urdf.py
"""

from __future__ import annotations

import importlib
import re

import dump_linkage
import leg_linkage_sim as lls
import numpy as np
import pinocchio as pin

HOME = "FL"                       # leg whose zero pose becomes everyone's home
KEYS = ("1.1", "1.2", "1.3", "2.1", "2.2", "2.3")
PARENT = {"1.1": None, "1.2": "1.1", "1.3": "1.2",
          "2.1": None, "2.2": "2.1", "2.3": "2.2"}
PIN = {"1.1": "P1", "1.2": "P2", "1.3": "P3",
       "2.1": "P4", "2.2": "P5", "2.3": "P7"}
INERTIA = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")

ORIGIN = re.compile(r'(<origin\s+xyz=")([^"]*)("\s+rpy=")([^"]*)(")')
AXIS = re.compile(r'(<axis\s+xyz=")([^"]*)(")')
TYPE = re.compile(r'(type=")([^"]*)(")')


# --- Helpers ---

def read_urdf() -> str:
    with open(lls.URDF, newline="") as f:      # keep CRLF line endings
        return f.read()


def write_urdf(text: str) -> None:
    with open(lls.URDF, "w", newline="") as f:
        f.write(text)


def _block(text: str, tag: str, name: str):
    """``(start, end)`` of the named ``<tag>`` element in ``text``."""
    m = re.search(rf'<{tag}\s+name="{re.escape(name)}"', text)
    if m is None:
        raise KeyError(f"{tag} {name} not found")
    return m.start(), text.index(f"</{tag}>", m.start())


def _fmt(v) -> str:
    # Round values below 1e-12 to zero (noise)
    v = np.where(np.abs(np.asarray(v, dtype=float)) < 1e-12, 0.0, v)
    return " ".join(f"{x:.15g}" for x in np.atleast_1d(v))


def _to_rpy(R: np.ndarray):
    pitch = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    if not np.allclose(lls._rpy(roll, pitch, yaw), R, atol=1e-9):
        raise RuntimeError("rpy extraction failed (gimbal lock?)")
    return roll, pitch, yaw


def _attr(blk: str, key: str) -> str:
    return re.search(rf'{key}="([^"]*)"', blk).group(1)


def _set_attr(blk: str, key: str, value: float) -> str:
    return re.sub(rf'({key}=")([^"]*)(")',
                  lambda m: m.group(1) + _fmt([value]) + m.group(3), blk, count=1)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _rot_xz(th):
    """3D rotation turning the (x, z) plane CCW by ``th``."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


# --- Re-frame the base ---

def _target_frame(model) -> pin.SE3:
    """New base frame (CoM, facing the front legs) in the current base frame."""
    data = model.createData()
    q0 = pin.neutral(model)
    com = np.asarray(pin.centerOfMass(model, data, q0))
    pin.forwardKinematics(model, data, q0)

    def hip_x(prefix):
        return float(np.mean([
            float(data.oMi[model.getJointId(lls.urdf_joint(leg, "1.1"))].translation[0])
            for leg in lls.LEGS if leg.startswith(prefix)
        ]))

    yaw = np.pi if hip_x("F") < hip_x("B") else 0.0
    return pin.SE3(lls._rpy(0.0, 0.0, yaw), com)


def reframe_base(text: str) -> str:
    """Move base_link and its child joint origins into the target frame."""
    model = pin.buildModelFromUrdf(str(lls.URDF), pin.JointModelFreeFlyer())
    M = _target_frame(model)
    yaw = float(np.arctan2(M.rotation[1, 0], M.rotation[0, 0]))
    print(f"  new base frame in the old one: t={np.round(M.translation, 4)} "
          f"yaw={np.degrees(yaw):.1f} deg")
    if np.allclose(M.homogeneous, np.eye(4), atol=1e-12):
        print("  already in the target convention")
        return text

    inv = M.inverse()                                   # old -> new
    R = np.asarray(inv.rotation)

    a, b = _block(text, "link", "base_link")
    blk, seen = text[a:b], []

    def sub(m):
        seen.append(1)
        xyz = np.array([float(v) for v in m.group(2).split()])
        rpy = np.array([float(v) for v in m.group(4).split()])
        new = inv * pin.SE3(lls._rpy(*rpy), xyz)
        return (m.group(1) + _fmt(new.translation) + m.group(3)
                + _fmt(_to_rpy(np.asarray(new.rotation))) + m.group(5))

    blk = ORIGIN.sub(sub, blk)
    if len(seen) != 3:
        raise RuntimeError(f"base_link: expected 3 origins, found {len(seen)}")

    # Inertia is given in the link frame, so rotate it along
    vals = {k: float(_attr(blk, k)) for k in INERTIA}
    I_old = np.array([[vals["ixx"], vals["ixy"], vals["ixz"]],
                      [vals["ixy"], vals["iyy"], vals["iyz"]],
                      [vals["ixz"], vals["iyz"], vals["izz"]]])
    I_new = R @ I_old @ R.T
    for key, v in zip(INERTIA, (I_new[0, 0], I_new[0, 1], I_new[0, 2],
                                I_new[1, 1], I_new[1, 2], I_new[2, 2])):
        blk = _set_attr(blk, key, v)
    text = text[:a] + blk + text[b:]

    for name, j in lls.JOINTS.items():
        if j["parent"] != "base_link":
            continue
        a, b = _block(text, "joint", name)
        blk = text[a:b]
        m = ORIGIN.search(blk)
        xyz = np.array([float(v) for v in m.group(2).split()])
        rpy = np.array([float(v) for v in m.group(4).split()])
        new = inv * pin.SE3(lls._rpy(*rpy), xyz)
        blk = ORIGIN.sub(
            lambda _m: (_m.group(1) + _fmt(new.translation) + _m.group(3)
                        + _fmt(_to_rpy(np.asarray(new.rotation))) + _m.group(5)),
            blk, count=1)
        text = text[:a] + blk + text[b:]
    return text


# --- Fix and re-home the legs ---

def _home_offset(leg, ref):
    """Hip angles putting ``leg`` in the configuration ``ref`` has at its zero."""
    def ang(v):
        return np.arctan2(v[1], v[0])

    q1 = leg.sgn["1.1"] * (ang(ref.pins0["P2"] - ref.pins0["P1"])
                           - ang(leg.pins0["P2"] - leg.pins0["P1"]))
    q2 = leg.sgn["2.1"] * (ang(ref.pins0["P6"] - ref.pins0["P4"])
                           - ang(leg.pins0["P6"] - leg.pins0["P4"]))
    return _wrap(q1), _wrap(q2)


def _leg_update(name, ref, text):
    """New joint and link origins for one leg at its home pose, plus the home offset."""
    leg = lls.Leg(name)
    q1, q2 = _home_offset(leg, ref)
    sol = leg.solve(q1, q2)
    if not sol["ok"]:
        raise RuntimeError(f"{name}: home pose is not assemblable")

    body, frame = {}, {}
    for k in KEYS:
        a0, a1 = leg.pins0[PIN[k]], sol["pins"][PIN[k]]
        A0 = np.array([a0[0], 0.0, a0[1]])
        A1 = np.array([a1[0], 0.0, a1[1]])
        M = _rot_xz(float(sol["theta"][k]))
        R0, p0 = leg.frames[f"Link_{name}{k}"]
        moved = A1 + M @ (p0 - A0)
        body[k] = (M @ R0, moved)
        # Link frame: same orientation, origin moved onto the pin axis
        frame[k] = (M @ R0, np.array([a1[0], moved[1], a1[1]]))

    hip = lls.JOINTS[lls.urdf_joint(name, "1.1")]
    n_world = leg.frames["base_link"][0] @ hip["R"] @ hip["axis"]

    joints, links = {}, {}
    for k in KEYS:
        Rc, oc = frame[k]
        Rp, op = (np.eye(3), np.zeros(3)) if PARENT[k] is None else frame[PARENT[k]]
        jn = lls.urdf_joint(name, k)
        if k in ("1.3", "2.3"):
            n = n_world                    # formerly fixed joints: use the hip axis
        else:
            j = lls.JOINTS[jn]
            n = leg.frames[j["parent"]][0] @ j["R"] @ j["axis"]
        joints[jn] = dict(xyz=Rp.T @ (oc - op), rpy=_to_rpy(Rp.T @ Rc), axis=Rc.T @ n)

        Rb, pb = body[k]
        d = Rc.T @ (pb - oc)               # old link frame in the new one
        link = f"Link_{name}{k}"
        a, b = _block(text, "link", link)
        com = np.array([float(v) for v in ORIGIN.search(text[a:b]).group(2).split()])
        # Compose with any mesh offset from an earlier run
        links[link] = dict(mesh=d + lls._visual_offset(link), com=d + com)
    return joints, links, (q1, q2)


def _apply_legs(text, joints, links):
    """Write the joint and link updates from ``_leg_update`` into the URDF text."""
    for name, upd in joints.items():
        a, b = _block(text, "joint", name)
        blk = text[a:b]
        blk = ORIGIN.sub(lambda m: (m.group(1) + _fmt(upd["xyz"]) + m.group(3)
                                    + _fmt(upd["rpy"]) + m.group(5)), blk, count=1)
        blk = AXIS.sub(lambda m: m.group(1) + _fmt(upd["axis"]) + m.group(3), blk, count=1)
        blk = TYPE.sub(lambda m: m.group(1) + "continuous" + m.group(3), blk, count=1)
        text = text[:a] + blk + text[b:]

    for name, upd in links.items():
        a, b = _block(text, "link", name)
        blk, seen = text[a:b], []

        def sub(m):
            seen.append(1)
            xyz = upd["com"] if len(seen) == 1 else upd["mesh"]   # inertial, visual, collision
            return m.group(1) + _fmt(xyz) + m.group(3) + m.group(4) + m.group(5)

        blk = ORIGIN.sub(sub, blk)
        if len(seen) != 3:
            raise RuntimeError(f"{name}: expected 3 origins, found {len(seen)}")
        text = text[:a] + blk + text[b:]
    return text


def fix_and_home_legs(text: str) -> str:
    """Apply the joint fixes and re-home all legs to the HOME leg's zero pose."""
    ref = lls.Leg(HOME)
    for name in lls.LEGS:
        joints, links, q = _leg_update(name, ref, text)
        text = _apply_legs(text, joints, links)
        print(f"  {name}: home offset q1.1={np.degrees(q[0]):+7.2f} deg  "
              f"q2.1={np.degrees(q[1]):+7.2f} deg")
    return text


# --- Driver ---

def sanitise_joint_names(text: str) -> str:
    """Rename ``Joint_FL1.1`` -> ``Joint_FL1_1`` (joints only; see ``lls.urdf_joint``)."""
    text, n = re.subn(r"Joint_([A-Z]{2})([12])\.([123])", r"Joint_\1\2_\3", text)
    print(f"  renamed {n} dotted joint-name occurrence(s)")
    return text


def main() -> None:
    print(f"preparing {lls.URDF}")

    # First, since everything below looks joints up by their new names
    print("\n[0/4] sanitising joint names for ROS")
    write_urdf(sanitise_joint_names(read_urdf()))
    importlib.reload(lls)

    # Legs before the base, so the CoM is measured at the home pose.
    print("\n[1/4] fixing joint types/origins and homing the legs")
    write_urdf(fix_and_home_legs(read_urdf()))

    # leg_linkage_sim reads the URDF at import time
    importlib.reload(lls)

    print("\n[2/4] re-framing the base")
    write_urdf(reframe_base(read_urdf()))
    importlib.reload(lls)

    print("\n[3/4] refreshing the frozen pin geometry")
    importlib.reload(dump_linkage)
    dump_linkage.main()

    print("\ndone.  Verify with:  python -m pytest tests -m 'not slow'")


if __name__ == "__main__":
    main()
