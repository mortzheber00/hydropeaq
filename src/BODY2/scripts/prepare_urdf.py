"""Turn the raw SolidWorks export of BODY2 into a URDF the pipeline can use.

Run this once after every CAD re-export.  It applies, in order:

1. **Fix the leg joints.**  ``Joint_*1.3`` and ``Joint_*2.3`` are exported as
   ``fixed`` but are real pins, and several joint origins sit up to 2.6 mm off
   the physical hole centres (two of them are not even on the part).  Every leg
   joint becomes ``continuous`` with its origin on the pin it represents, and
   all four legs are re-homed so ``q = 0`` is the same physical pose on each --
   the export freezes whatever crank angle each leg happened to have, which
   differs by up to 13 deg.  Moving a joint origin moves its child link frame,
   so that link's visual, collision and inertial origins are compensated and
   the geometry stays exactly where it was.

2. **Re-frame the base.**  The export leaves ``base_link`` at the CAD origin,
   ~1.1 m from the robot and facing -x, with the handedness mirrored relative
   to amph.  Everything is moved to amph's convention -- origin at the centre
   of mass, +x forward, +y to the robot's left -- by translating to the CoM and
   yawing 180 deg.  A yaw is the only rotation that keeps the leg planes normal
   to world Y, which the closed-form linkage solver relies on.  The base
   inertia tensor is rotated too, not just translated.  This runs *after* the
   legs are homed, so the origin lands on the CoM of the pose the robot will
   actually sit in -- which is also what makes a second run a no-op.

3. **Refresh the frozen pin geometry** in ``hydro_model/robots/body2_linkage.json``,
   which is expressed in world coordinates and so depends on both steps above.

    python src/BODY2/scripts/prepare_urdf.py

Idempotent: every transform is measured from the current file, so re-running
computes identities (bar ~1e-14 of float round-trip through the text).

What this still cannot do: URDF has no way to express the two loop-closure pins
per leg, so the result is a kinematically correct but under-constrained tree
with 6 joints for 2 real DOF.  The loops live in the coordinate map instead --
see ``hydro_model/robots/body2_map.py``.
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


# ------------------------------------------------------------------ helpers

def read_urdf() -> str:
    with open(lls.URDF, newline="") as f:      # newline="" keeps the CRLF endings
        return f.read()


def write_urdf(text: str) -> None:
    with open(lls.URDF, "w", newline="") as f:
        f.write(text)


def _block(text: str, tag: str, name: str):
    m = re.search(rf'<{tag}\s+name="{re.escape(name)}"', text)
    if m is None:
        raise KeyError(f"{tag} {name} not found")
    return m.start(), text.index(f"</{tag}>", m.start())


def _fmt(v) -> str:
    # the smallest meaningful figure here is ~1e-4 m, so 1e-12 is noise
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


# --------------------------------------------------------- 1. re-frame base

def _target_frame(model) -> pin.SE3:
    """The new base frame, expressed in the current one."""
    data = model.createData()
    q0 = pin.neutral(model)
    com = np.asarray(pin.centerOfMass(model, data, q0))
    pin.forwardKinematics(model, data, q0)

    def hip_x(prefix):
        return float(np.mean([
            float(data.oMi[model.getJointId(f"Joint_{leg}1.1")].translation[0])
            for leg in lls.LEGS if leg.startswith(prefix)
        ]))

    yaw = np.pi if hip_x("F") < hip_x("B") else 0.0
    return pin.SE3(lls._rpy(0.0, 0.0, yaw), com)


def reframe_base(text: str) -> str:
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

    # the tensor is given in the link frame, so rotating the frame rotates it
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


# ------------------------------------------------------ 2. fix + home legs

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
        # the link frame keeps that orientation but slides onto the pin axis
        frame[k] = (M @ R0, np.array([a1[0], moved[1], a1[1]]))

    hip = lls.JOINTS[f"Joint_{name}1.1"]
    n_world = leg.frames["base_link"][0] @ hip["R"] @ hip["axis"]

    joints, links = {}, {}
    for k in KEYS:
        Rc, oc = frame[k]
        Rp, op = (np.eye(3), np.zeros(3)) if PARENT[k] is None else frame[PARENT[k]]
        jn = f"Joint_{name}{k}"
        if k in ("1.3", "2.3"):
            n = n_world                    # new joints follow the hip's handedness
        else:
            j = lls.JOINTS[jn]
            n = leg.frames[j["parent"]][0] @ j["R"] @ j["axis"]
        joints[jn] = dict(xyz=Rp.T @ (oc - op), rpy=_to_rpy(Rp.T @ Rc), axis=Rc.T @ n)

        Rb, pb = body[k]
        d = Rc.T @ (pb - oc)               # old link frame, seen from the new one
        link = f"Link_{name}{k}"
        a, b = _block(text, "link", link)
        com = np.array([float(v) for v in ORIGIN.search(text[a:b]).group(2).split()])
        # the mesh may already carry an offset from an earlier run; compose
        links[link] = dict(mesh=d + lls._visual_offset(link), com=d + com)
    return joints, links, (q1, q2)


def _apply_legs(text, joints, links):
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
    ref = lls.Leg(HOME)
    for name in lls.LEGS:
        joints, links, q = _leg_update(name, ref, text)
        text = _apply_legs(text, joints, links)
        print(f"  {name}: home offset q1.1={np.degrees(q[0]):+7.2f} deg  "
              f"q2.1={np.degrees(q[1]):+7.2f} deg")
    return text


# ---------------------------------------------------------------- driver

def main() -> None:
    print(f"preparing {lls.URDF}")

    # Legs first: re-framing measures the centre of mass, and doing it after
    # homing means the origin lands on the CoM of the pose the robot will
    # actually sit in -- which is also what makes a second run a no-op.
    print("\n[1/3] fixing joint types/origins and homing the legs")
    write_urdf(fix_and_home_legs(read_urdf()))

    # leg_linkage_sim caches the URDF at import; it must re-read after a rewrite
    importlib.reload(lls)

    print("\n[2/3] re-framing the base")
    write_urdf(reframe_base(read_urdf()))
    importlib.reload(lls)

    print("\n[3/3] refreshing the frozen pin geometry")
    importlib.reload(dump_linkage)
    dump_linkage.main()

    print("\ndone.  Verify with:  python -m pytest tests -m 'not slow'")


if __name__ == "__main__":
    main()
