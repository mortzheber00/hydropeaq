"""Constrained kinematics + interactive viewer for the BODY2 parallel legs.

The URDF exports each leg as two independent open chains (``*1.1 -> *1.2 -> *1.3``
and ``*2.1 -> *2.2 -> *2.3``), which is not what the hardware does: the leg is a
single closed 2-DOF planar mechanism.  Recovering it from the STL pin holes gives
seven links (ground + 6) joined by eight revolute pins, all with the same axis:

    P1  base   <-> 1.1     hip A          ACTUATED  (Joint_<leg>1.1)
    P2  1.1    <-> 1.2                    passive   (Joint_<leg>1.2)
    P3  1.2    <-> 1.3                    passive   (Joint_<leg>1.3, exported "fixed")
    P4  base   <-> 2.1     hip B          ACTUATED  (Joint_<leg>2.1)
    P5  2.1    <-> 2.2                    passive   (Joint_<leg>2.2)
    P6  2.1    <-> 1.3                    passive   LOOP CLOSURE - not in the URDF
    P7  2.2    <-> 2.3                    passive   (Joint_<leg>2.3, exported "fixed")
    P8  1.3    <-> 2.3                    passive   LOOP CLOSURE - not in the URDF

    DOF = 3*6 - 2*8 = 2, matching the two hip servos.

Loop 1 (P1-P2-P3-P6-P4) is a five-bar; loop 2 (P5-P6-P8-P7) is a parallelogram
that ties the foot orientation to link 2.1.  Both close in one pass:

    hips  ->  P2, P5, P6           (rotate the cranks)
    P3    =  circle(P2, |P2P3|) x circle(P6, |P6P3|)     -> pose of 1.2 and 1.3
    P8    =  rigid on 1.3
    P7    =  circle(P5, |P5P7|) x circle(P8, |P8P7|)     -> pose of 2.2 and 2.3

All pin locations are read from the mesh geometry rather than from the exported
joint origins, which sit up to 2.6 mm off the physical hole centres.

Run with no arguments for the slider viewer; ``Leg("BR").solve(q1, q2)`` gives the
passive joint values for driving a simulator.  The viewer opens a second window
holding the same leg in the optimiser's own coordinates -- see
``_optimiser_margin`` below.  ``--limits`` puts a box constraint on the two hip
angles, the viewer's twin of bounding ``theta`` in the OCP -- ``--spec-limits``
takes that box from stage1's own spec rather than the command line -- and
``--trajectory`` draws a saved solution or initial guess across both windows.

``--export out.pdf`` skips the viewer and writes that coordinate map as a
thesis figure instead, in the shared style of ``stage3_visualization``.  It is
the same painted set the viewer's second window shows -- one function draws
both -- so what is checked interactively is what reaches the page.
"""

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
URDF = PKG / "urdf" / "BODY2.urdf"
MESHES = PKG / "meshes"
LEGS = ["FL", "FR", "BL", "BR"]

# the leg planes are normal to world Y, so all planar maths runs on (x, z)
PLANE = [0, 2]


# ---------------------------------------------------------------- URDF / STL

def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _read_stl(path):
    """Return (triangles (n,3,3), normals (n,3)) from a binary STL."""
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[80:84])[0]
    buf = np.frombuffer(raw[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)
    vals = buf[:, :48].copy().view(np.float32).astype(float)
    return vals[:, 3:].reshape(n, 3, 3), vals[:, :3]


def _joints():
    out = {}
    for j in ET.parse(URDF).getroot().findall("joint"):
        o, a = j.find("origin"), j.find("axis")
        out[j.get("name")] = dict(
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            xyz=np.array([float(v) for v in o.get("xyz").split()]),
            R=_rpy(*[float(v) for v in o.get("rpy").split()]),
            axis=np.array([float(v) for v in a.get("xyz").split()]) if a is not None
            else np.zeros(3))
    return out


def _visual_offsets():
    """Per-link translation from the link frame to its mesh frame.

    A rotated visual origin is rejected, but only when the link is actually
    looked up: the hole fitting below assumes bores run along the mesh frame's
    local z, which a rotation would break.  ``base_link`` legitimately carries
    one after the base is re-framed, and is never consulted here.
    """
    out = {}
    for link in ET.parse(URDF).getroot().findall("link"):
        name = link.get("name")
        o = link.find("visual/origin")
        if o is None:
            out[name] = np.zeros(3)
            continue
        rotated = any(abs(float(v)) > 1e-12 for v in o.get("rpy", "0 0 0").split())
        out[name] = (RuntimeError(f"{name}: rotated visual origin is not supported")
                     if rotated else np.array([float(v) for v in o.get("xyz").split()]))
    return out


def _visual_offset(link: str) -> np.ndarray:
    tv = VISUAL[link]
    if isinstance(tv, Exception):
        raise tv
    return tv


JOINTS = _joints()
VISUAL = _visual_offsets()


def urdf_joint(leg: str, key: str) -> str:
    """URDF joint name for a linkage key: ``("FL", "1.1") -> "Joint_FL1_1"``.

    The pins are named ``1.1``-style throughout this module and the coordinate
    map, because that is the CAD's nomenclature.  The URDF spells the same
    joints with an underscore: ROS graph resource names forbid dots, so a
    dotted joint name is illegal anywhere a name is built out of it -- a
    controller's parameter namespace, a rosparam key, a dynamic_reconfigure
    server.  Link names and mesh files keep the dots; they never become ROS
    names.  ``prepare_urdf.py`` enforces this on every run.

    hydro_model/robots/body2.py carries the same one-liner; neither package is
    on the other's import path.  Only the viewer reaches across, optionally and
    one way, to draw the optimiser's feasible set (``_optimiser_margin``).
    """
    return f"Joint_{leg}{key.replace('.', '_')}"


def _link_frames(leg):
    """World pose of every link of one leg at the URDF zero configuration."""
    T = {"base_link": (np.eye(3), np.zeros(3))}
    for name in (f"Joint_{leg}{c}_{k}" for c in (1, 2) for k in (1, 2, 3)):
        j = JOINTS[name]
        Rp, pp = T[j["parent"]]
        T[j["child"]] = (Rp @ j["R"], pp + Rp @ j["xyz"])
    return T


# ---------------------------------------------------------------- pin holes

def _hole_centres(tri, nrm):
    """Centres of the through-holes of a link, in its own frame (local x, y).

    Every pin bore in these parts runs along the link's local z, so its wall
    triangles are the ones with a normal perpendicular to z.  Those are grouped
    by position and fitted with |p - c| = r, which is linear in (c, r) once the
    inward and outward walls of a boss are separated.
    """
    side = np.abs(nrm[:, 2]) < 0.1
    cen = tri[side].mean(axis=1)[:, :2]
    nn = nrm[side][:, :2]
    nn = nn / np.linalg.norm(nn, axis=1, keepdims=True)

    label = -np.ones(len(cen), int)
    ngroups = 0
    for i in range(len(cen)):
        if label[i] >= 0:
            continue
        stack, label[i] = [i], ngroups
        while stack:
            near = (label < 0) & (np.linalg.norm(cen - cen[stack.pop()], axis=1) < 2.5e-3)
            for t in np.flatnonzero(near):
                label[t] = ngroups
                stack.append(t)
        ngroups += 1

    found = []
    for g in range(ngroups):
        m = label == g
        if m.sum() < 8:
            continue
        pg, ng = cen[m], nn[m]
        inward = ((pg - pg.mean(0)) * ng).sum(1) < 0          # boss and bore overlap
        for sub in (inward, ~inward):
            if sub.sum() < 8:
                continue
            p, v = pg[sub], ng[sub]
            if np.linalg.norm(v.mean(0)) > 0.35:              # not a closed circle
                continue
            sol, *_ = np.linalg.lstsq(np.hstack([v, np.ones((len(p), 1))]),
                                      (p * v).sum(1), rcond=None)
            c, r = sol[:2], abs(sol[2])
            if not 0.7e-3 < r < 6e-3:
                continue
            if np.abs(np.linalg.norm(p - c, axis=1) - r).max() > 0.4e-3:
                continue
            found.append(c)

    uniq = []
    for c in found:
        if not any(np.linalg.norm(c - u) < 1e-3 for u in uniq):
            uniq.append(c)
    return uniq


def _pins(leg):
    """The eight pin positions of one leg, as world (x, z) at the zero pose."""
    T = _link_frames(leg)
    holes = {}
    for k in ("1.1", "1.2", "1.3", "2.1", "2.2"):
        R, p = T[f"Link_{leg}{k}"]
        tri, nrm = _read_stl(MESHES / f"Link_{leg}{k}.STL")
        tv = _visual_offset(f"Link_{leg}{k}")
        holes[k] = [(R @ (np.array([c[0], c[1], 0.0]) + tv) + p)[PLANE]
                    for c in _hole_centres(tri, nrm)]
    for k, want in (("1.1", 1), ("1.2", 2), ("1.3", 3), ("2.1", 2), ("2.2", 2)):
        if len(holes[k]) != want:
            raise RuntimeError(f"Link_{leg}{k}: found {len(holes[k])} pin holes, expected {want}")

    def shared(a, b):
        hits = [ca for ca in holes[a]
                if any(np.linalg.norm(ca - cb) < 1.5e-3 for cb in holes[b])]
        if len(hits) != 1:
            raise RuntimeError(f"{leg}: {len(hits)} shared holes between {a} and {b}")
        return hits[0]

    def other(candidates, *known):
        rest = [c for c in candidates
                if all(np.linalg.norm(c - k) > 1.5e-3 for k in known)]
        if len(rest) != 1:
            raise RuntimeError(f"{leg}: ambiguous pin match ({len(rest)} left)")
        return rest[0]

    P2 = shared("1.1", "1.2")
    P3 = other(holes["1.2"], P2)
    P6 = shared("2.1", "1.3")
    P5 = other(holes["2.1"], P6)
    P7 = other(holes["2.2"], P5)
    P8 = other(holes["1.3"], P3, P6)
    # the hips carry a D-profile servo horn rather than a bore: take the URDF axes
    return dict(P1=T[f"Link_{leg}1.1"][1][PLANE], P2=P2, P3=P3,
                P4=T[f"Link_{leg}2.1"][1][PLANE], P5=P5, P6=P6, P7=P7, P8=P8)


# ---------------------------------------------------------------- planar maths

def _rot2(v, th):
    """Rotate (..., 2) vectors CCW in the (x, z) plane."""
    c, s = np.cos(th), np.sin(th)
    x, y = v[..., 0], v[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def _angle(v):
    return np.arctan2(v[..., 1], v[..., 0])


def _circle_circle(c1, r1, c2, r2, branch):
    """Intersection of two circles; ``branch`` = +-1 selects the assembly mode.

    Returns (point, ok); ``point`` is nan where the circles do not meet.
    """
    d = c2 - c1
    L = np.linalg.norm(d, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (r1 ** 2 - r2 ** 2 + L ** 2) / (2 * L)
        h2 = r1 ** 2 - a ** 2
        ok = h2 > 0
        h = np.sqrt(np.where(ok, h2, np.nan))
        u = d / L[..., None]
    perp = np.stack([-u[..., 1], u[..., 0]], axis=-1)
    return c1 + a[..., None] * u + branch * h[..., None] * perp, ok


class Leg:
    """Closed-loop kinematics of one BODY2 leg."""

    def __init__(self, name):
        self.name = name
        self.frames = _link_frames(name)
        self.pins0 = _pins(name)
        p = self.pins0

        # sign mapping a URDF joint value onto a CCW rotation of the (x, z) plane
        self.sgn = {}
        hip = None
        for c in (1, 2):
            for k in (1, 2, 3):
                j = JOINTS[f"Joint_{name}{c}_{k}"]
                axis = self.frames[j["parent"]][0] @ j["R"] @ j["axis"]
                if np.linalg.norm(axis) < 1e-9:
                    axis = hip          # *.3 still exported as a weld: it is a pin
                if abs(abs(axis[1]) - 1) > 1e-6:
                    raise RuntimeError(f"{name}{c}.{k}: joint axis is not along world Y")
                hip = hip if hip is not None else axis
                self.sgn[f"{c}.{k}"] = -1.0 if axis[1] > 0 else 1.0

        self.L = {  # rigid distances that the loops must preserve
            "P2P3": np.linalg.norm(p["P3"] - p["P2"]),
            "P6P3": np.linalg.norm(p["P3"] - p["P6"]),
            "P5P7": np.linalg.norm(p["P7"] - p["P5"]),
            "P8P7": np.linalg.norm(p["P7"] - p["P8"]),
        }
        # assembly modes that reproduce the zero pose
        self.branch = (self._branch(p["P2"], p["P6"], p["P3"]),
                       self._branch(p["P5"], p["P8"], p["P7"]))

        # meshes and the foot tip, in the frame of the body that carries them
        self.mesh = {}
        for c in (1, 2):
            for k in (1, 2, 3):
                key = f"{c}.{k}"
                R, o = self.frames[f"Link_{name}{key}"]
                tri, _ = _read_stl(MESHES / f"Link_{name}{key}.STL")
                tv = _visual_offset(f"Link_{name}{key}")
                self.mesh[key] = ((tri.reshape(-1, 3) + tv) @ R.T + o)[:, PLANE].reshape(-1, 3, 2)
        v = self.mesh["2.3"].reshape(-1, 2)
        self.tip0 = v[np.argmax(np.linalg.norm(v - p["P7"], axis=1))]

    @staticmethod
    def _branch(c1, c2, pt):
        d, e = c2 - c1, pt - c1
        return np.sign(d[0] * e[1] - d[1] * e[0])

    def solve(self, q1, q2):
        """Close both loops for hip angles ``q1`` (joint 1.1) and ``q2`` (joint 2.1).

        ``q1``/``q2`` are URDF joint values in radians and may be arrays.  Returns a
        dict of pin positions, body rotations and every passive joint value; the
        ``ok`` mask is False where the linkage cannot be assembled.
        """
        q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
        p0, L = self.pins0, self.L
        th1 = self.sgn["1.1"] * q1
        th2 = self.sgn["2.1"] * q2

        P1 = np.broadcast_to(p0["P1"], q1.shape + (2,))
        P4 = np.broadcast_to(p0["P4"], q1.shape + (2,))
        P2 = P1 + _rot2(p0["P2"] - p0["P1"], th1)
        P5 = P4 + _rot2(p0["P5"] - p0["P4"], th2)
        P6 = P4 + _rot2(p0["P6"] - p0["P4"], th2)

        P3, ok1 = _circle_circle(P2, L["P2P3"], P6, L["P6P3"], self.branch[0])
        th12 = _angle(P3 - P2) - _angle(p0["P3"] - p0["P2"])          # link 1.2
        th13 = _angle(P6 - P3) - _angle(p0["P6"] - p0["P3"])          # link 1.3
        P8 = P3 + _rot2(p0["P8"] - p0["P3"], th13)

        P7, ok2 = _circle_circle(P5, L["P5P7"], P8, L["P8P7"], self.branch[1])
        th22 = _angle(P7 - P5) - _angle(p0["P7"] - p0["P5"])          # link 2.2
        th23 = _angle(P8 - P7) - _angle(p0["P8"] - p0["P7"])          # link 2.3
        tip = P7 + _rot2(self.tip0 - p0["P7"], th23)

        def wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        return dict(
            ok=ok1 & ok2,
            pins=dict(P1=P1, P2=P2, P3=P3, P4=P4, P5=P5, P6=P6, P7=P7, P8=P8),
            theta={"1.1": th1, "1.2": th12, "1.3": th13,
                   "2.1": th2, "2.2": th22, "2.3": th23},
            tip=tip,
            # values for the joints the URDF already declares
            joints={urdf_joint(self.name, "1.2"): wrap(self.sgn["1.2"] * (th12 - th1)),
                    urdf_joint(self.name, "1.3"): wrap(self.sgn["1.3"] * (th13 - th12)),
                    urdf_joint(self.name, "2.2"): wrap(self.sgn["2.2"] * (th22 - th2)),
                    urdf_joint(self.name, "2.3"): wrap(self.sgn["2.3"] * (th23 - th22))},
            # the two pins that have no URDF joint at all
            closures={"P6 (2.1<->1.3)": wrap(self.sgn["1.1"] * (th13 - th2)),
                      "P8 (1.3<->2.3)": wrap(self.sgn["1.1"] * (th23 - th13))})

    def place(self, key, sol):
        """Mesh of link ``key`` moved into the pose held in ``sol``."""
        anchor = {"1.1": "P1", "1.2": "P2", "1.3": "P3",
                  "2.1": "P4", "2.2": "P5", "2.3": "P7"}[key]
        a0, a = self.pins0[anchor], sol["pins"][anchor]
        return a + _rot2(self.mesh[key] - a0, sol["theta"][key])


# ------------------------------------------------------- optimiser coordinates

STAGE1 = PKG.parents[1] / "stage1_gait_optimization"


def _optimiser_margin():
    """Loop-closure margin of the coordinate map stage1 optimises through.

    ``hydro_model.robots.body2_map`` drives a leg from the same two hip angles
    this module does, and stage1's OCP additionally holds both circle-circle
    intersections at ``h^2 >= H_MIN^2`` so a trajectory cannot pass through a
    configuration where the loops fall apart.  That constraint, not the raw
    assemblability the viewer draws, is the set the optimiser may plan in.

    Returns ``(margin, H_MIN_mm)`` where ``margin(leg, Q1, Q2)`` is the signed
    half-chord ``sign(h^2) * sqrt(|h^2|)`` of whichever loop binds, in mm.  It
    is monotone in ``h^2``, so its zero and its ``H_MIN`` contour are exactly
    the constraint's, on a scale that can be measured off the leg.

    Returns ``None`` if stage1 or casadi is not importable; nothing else in
    this module needs either.
    """
    import sys

    if str(STAGE1) not in sys.path:
        sys.path.insert(0, str(STAGE1))
    try:
        import casadi as ca
        from hydro_model.robots.body2 import H_MIN, LINKAGE
        from hydro_model.robots.body2_map import _solve_leg
    except ImportError as exc:
        print(f"optimiser-map window disabled ({exc})")
        return None

    cache = {}

    def margin(leg, q1, q2):
        if leg not in cache:
            th = ca.SX.sym("theta", 2)
            _, h_sq = _solve_leg(th[0], th[1], LINKAGE["legs"][leg], LINKAGE["branch"])
            cache[leg] = ca.Function("h_sq", [th], [ca.vertcat(*h_sq)])
        flat = np.vstack([np.ravel(q1).astype(float), np.ravel(q2).astype(float)])
        h2 = np.asarray(cache[leg].map(flat.shape[1])(flat))
        h = np.sign(h2) * np.sqrt(np.abs(h2))
        return h.min(axis=0).reshape(np.shape(q1)) * 1e3

    return margin, H_MIN * 1e3


def _trajectory(path):
    """Reduced hip coordinates from a stage1 solution or initial-guess ``.npz``.

    Both come out of ``hydro_model.trajectory.save_solution`` and share one
    layout: ``X`` is ``(nq + nv, N+1)`` and its rows ``7:nq`` hold ``theta``,
    ordered ``(1.1, 2.1)`` per leg over ``LEG_NAMES``.  The file is read with
    numpy rather than through ``hydro_model`` so a trajectory can be drawn on a
    machine without casadi, and because importing that package for a dict of
    arrays would pull in pinocchio too.

    Returns ``(theta (8, N+1), T)`` in radians and seconds.
    """
    data = np.load(path, allow_pickle=False)
    if "version" not in data.files:
        raise SystemExit(f"{path}: pre-versioning file; those are amph solutions "
                         f"in tree coordinates, not BODY2 hip angles")
    robot, coords = str(data["robot"]), str(data["coords"])
    if robot != "body2":
        raise SystemExit(f"{path}: robot is {robot!r}, not 'body2'")
    if coords != "reduced":
        raise SystemExit(f"{path}: coords are {coords!r}; BODY2 states are written "
                         f"in the reduced actuated coordinates")
    theta = data["X"][7:int(data["nq"])]
    if len(theta) != 2 * len(LEGS):
        raise SystemExit(f"{path}: {len(theta)} reduced coordinates, expected {2 * len(LEGS)}")
    return theta, float(data["T"])


# theta's leg order, i.e. hydro_model.robots.body2.LEG_NAMES.  Spelled out
# rather than reusing LEGS: this one indexes somebody else's array.
THETA_LEGS = ("FL", "FR", "BL", "BR")


def _wrap_break(a, b):
    """Wrap two degree series into (-180, 180] and cut the polyline where either wraps.

    Solutions run in unwrapped angles -- the hips are continuous and the
    coordinate map has no branch cut -- so a path may leave the square the map
    is drawn on.  Without the cut, re-entering on the far side draws a stripe
    straight across the plot that was never part of the trajectory.
    """
    wa, wb = (a + 180) % 360 - 180, (b + 180) % 360 - 180
    cut = np.flatnonzero((np.abs(np.diff(wa)) > 180) | (np.abs(np.diff(wb)) > 180)) + 1
    return np.insert(wa, cut, np.nan), np.insert(wb, cut, np.nan)


def _box(limits):
    """Validate ``--limits`` (four degrees) into ``((q1lo, q1hi), (q2lo, q2hi))``.

    The zero pose has to be inside.  It is the configuration the leg is
    assembled in, and the assembly branch every solve here runs on is read off
    it; a box excluding it would describe a range the leg could only enter by
    first leaving the box, leaving nothing honest to draw.
    """
    box = ((limits[0], limits[1]), (limits[2], limits[3]))
    for joint, (lo, hi) in zip(("x1.1", "x2.1"), box):
        if lo >= hi:
            raise SystemExit(f"--limits: {joint} lower bound {lo:g} is not below {hi:g}")
        if not (-180 <= lo and hi <= 180):
            raise SystemExit(f"--limits: {joint} box must lie within [-180, 180]")
        if not lo <= 0 <= hi:
            raise SystemExit(f"--limits: {joint} box [{lo:g}, {hi:g}] excludes the zero pose")
    return box


def _spec_hip_box(leg):
    """The OCP's own box on this leg's two hips, in degrees, from the spec.

    A solution ``.npz`` records which robot it was solved for but not the
    bounds it was solved under, so there is nothing in the file to read: this
    goes to that robot's spec instead and takes ``HIP_BOX``, the array
    ``ocp_common.limits_for`` hands the OCP as ``theta_lower``/``theta_upper``.
    It is therefore the box as the spec stands now, which is the solution's
    only if the spec has not moved since it was written -- the file carries
    nothing to check that against, the same gap ``hind_workspace.py`` reports
    when it overlays a saved guess.

    Indexed through that module's own ``LEG_NAMES`` rather than this one's
    ``LEGS``, for the reason ``THETA_LEGS`` is spelled out below: the order is
    somebody else's array's, not ours.  The two sides mount mirrored, so their
    boxes negate and there is no single square for the robot -- the box is per
    leg, and a figure drawing one leg's box on another would be drawing the
    wrong constraint.
    """
    import sys

    if str(STAGE1) not in sys.path:
        sys.path.insert(0, str(STAGE1))
    try:
        from hydro_model.robots.body2 import HIP_BOX, LEG_NAMES
    except ImportError as exc:
        raise SystemExit(f"--spec-limits reads the box out of stage1's body2 "
                         f"spec, which is not importable: {exc}")
    i = 2 * LEG_NAMES.index(leg)
    lo1, hi1 = np.degrees(HIP_BOX[i])
    lo2, hi2 = np.degrees(HIP_BOX[i + 1])
    # Through _box, so a spec box that cannot be drawn -- one excluding the zero
    # pose the assembly branch is read off -- fails the way a typed one does.
    return _box((lo1, hi1, lo2, hi2))


def _box_of(limits, spec_limits):
    """Resolve the two box flags into ``leg -> box or None``.

    ``--spec-limits`` varies with the leg; an explicit ``--limits`` is one box
    for whichever leg is drawn, and no flag at all is no box.
    """
    if spec_limits:
        return _spec_hip_box
    fixed = None if limits is None else _box(limits)
    return lambda _: fixed


# ---------------------------------------------------------------- the map

# The thesis palette, i.e. stage3_visualization/common/thesis_style.py's ``PALETTE``,
# by the index each is taken from.  It is spelled out rather than imported for
# the reason ``urdf_joint`` is spelled out twice: importing that module is not
# free, and here it is worse than not free -- it switches matplotlib to LaTeX
# text rendering, which a viewer run has no business paying for.  Keep these in
# step with it; they are the same five colours every figure in the thesis uses.
CHAIN = ("#0173B2", "#B2182B")   # PALETTE[0], PALETTE[2]: the two hip chains
# Not reachable from zero pose: a grey hatch, since it is mechanism, not identity.
UNREACH, UNREACH_HATCH = "0.15", "////"
# PALETTE[0], which is what hind_workspace.py draws its "trajectory" in; a
# trajectory is the same object here, so it is the same colour.
TRACE = "#0173B2"
# PALETTE[2]: the h bound stage1 enforces, the line the trajectory is read
# against.  h = 0 stays black -- a hard limit, and the hatch already marks it.
BOUND = "#B2182B"
# Everything that is mechanism rather than identity stays greyscale, so that
# inside the stick figure colour encodes exactly one thing: which chain a link
# belongs to.  The joints, the ground and the foot are roles, not identities.
FRAME = "0.2"


def _inbox(Q1, Q2, box):
    """Grid cells a ``--limits`` box admits; all of them when there is no box.

    The comparison carries a tolerance far below the grid step, which is not
    about where the edge belongs but about a cell that is *on* it: the grid
    point nearest 66 deg comes out of ``linspace`` at 66.000000000000014, so a
    bare ``<=`` drops it, and a box and its mirror image then keep different
    columns.  --spec-limits makes that easy to walk into, both of its bounds
    being exact multiples of the radian-to-degree conversion.
    """
    if box is None:
        return np.ones(Q1.shape, bool)
    eps = 1e-9
    (a1, b1), (a2, b2) = box
    d1, d2 = np.degrees(Q1), np.degrees(Q2)
    return ((d1 >= a1 - eps) & (d1 <= b1 + eps)
            & (d2 >= a2 - eps) & (d2 <= b2 + eps))


def _reachable(leg, Q1, Q2, grid, inbox):
    """Configurations reachable from the zero pose without taking the leg apart.

    Assemblable (q1, q2) cells split into several islands; only the one holding
    the zero pose can be driven to, so the rest are flooded away.  The hips are
    continuous joints, hence the wrap-around neighbourhood -- a box constraint
    needs no special case there, since wrapping from +180 to -180 has to cross
    cells the box already masks out unless the box spans the full turn.

    Returns the reachable mask over the (q1, q2) grid and the foot tips it
    maps to: the set the viewer draws as the tip cloud in one window and as the
    island boundary in the other.
    """
    sol = leg.solve(Q1, Q2)
    ok = sol["ok"] & np.isfinite(sol["tip"][..., 0]) & inbox
    n = len(grid)
    seed = (int(np.argmin(np.abs(grid))),) * 2
    reach = np.zeros_like(ok)
    if ok[seed]:
        reach[seed] = True
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                nr, nc = nr % n, nc % n
                if ok[nr, nc] and not reach[nr, nc]:
                    reach[nr, nc] = True
                    stack.append((nr, nc))
    return reach, sol["tip"][reach]


def paint_map(ax, Q1, Q2, margin, reach, inbox, box, hmin_mm, traj, scale=1.0):
    """Paint one leg's optimiser coordinate map onto ``ax``; returns the image.

    The grey field is the binding loop's half-chord: dark where the leg cannot
    be assembled at all, light where it can, and the two black contours are the
    assembly limit and the tighter bound stage1 actually constrains.  The
    reachable island is the same set the viewer's other window draws as the
    foot-tip cloud, so the two are readable side by side.

    ``scale`` thins the linework for a figure printed at half the text width,
    where the viewer's weights would be several times too heavy.  Everything
    else is identical in both, which is the point of there being one painter:
    the thesis figure cannot drift from the set that was checked on screen.
    """
    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    deg1, deg2 = np.degrees(Q1), np.degrees(Q2)
    half = (deg1[0, 1] - deg1[0, 0]) / 2
    lim = 180 + half
    vmax = np.nanmax(margin)

    # Greyscale, so the field is mechanism and leaves colour to the island, the
    # chains and the trajectory -- on a red/blue map the palette-blue trajectory
    # vanished into the feasible half.  The black end is cut off: infeasible
    # bottoms out at mid-grey, where the black h = 0 and h = 2 mm contours
    # still read.
    greys = ListedColormap(colormaps["Greys_r"](np.linspace(0.35, 1.0, 256)))
    im = ax.imshow(margin, origin="lower", extent=(-lim, lim, -lim, lim),
                   cmap=greys, vmin=-vmax, vmax=vmax, interpolation="nearest")
    # Hatch what is not reachable from zero pose rather than outlining what is:
    # on all four legs of this build the island's edge is the h = 0 contour, so
    # an outline only doubled that line, and a hatch still marks an h > 0
    # island the leg cannot get to.  Near-black but thin: inside the box the
    # unreachable part is the dark h < 0 field, where a light hatch vanishes.  Matplotlib 3.7 takes the hatch colour when the
    # artist is made but the hatch width from rcParams when the figure is
    # saved, hence the global setting.
    import matplotlib
    matplotlib.rcParams["hatch.linewidth"] = 0.4 * scale
    with matplotlib.rc_context({"hatch.color": UNREACH}):
        ax.contourf(deg1, deg2, reach.astype(float), [-0.5, 0.5],
                    colors="none", hatches=[UNREACH_HATCH])
    ax.contour(deg1, deg2, margin, [0.0], colors="k", linewidths=1.0 * scale)
    ax.contour(deg1, deg2, margin, [hmin_mm], colors=BOUND,
               linewidths=1.0 * scale, linestyles="dashed")
    if box is not None:
        # shade out what the constraint forbids, rather than cropping to it:
        # the excluded structure is exactly what one wants to see when
        # deciding whether the box is drawn in the right place.
        veil = np.ones(Q1.shape + (4,))          # white, i.e. wash out
        veil[..., 3] = np.where(inbox, 0.0, 0.62)
        ax.imshow(veil, origin="lower", extent=(-lim, lim, -lim, lim),
                  interpolation="nearest", zorder=4)
        ax.add_patch(Rectangle((box[0][0], box[1][0]), box[0][1] - box[0][0],
                               box[1][1] - box[1][0], fill=False, ec="k",
                               lw=1.4 * scale, zorder=6))
    if traj is not None:
        # PALETTE[0], as the thesis draws every trajectory; on the grey field it
        # reads everywhere.  The thin white casing only separates it from the
        # black contours where the path hugs the h = 2 mm bound.
        from matplotlib.patheffects import withStroke

        ax.plot(*traj, "-", color=TRACE, lw=1.4 * scale, zorder=7,
                path_effects=[withStroke(linewidth=2.2 * scale, foreground="w")])

    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(range(-180, 181, 90))
    ax.set_yticks(range(-180, 181, 90))
    return im


def _map_trace(theta, leg):
    """The leg's hip path in degrees, cut where it wraps, ready for ``paint_map``."""
    i = THETA_LEGS.index(leg)
    return _wrap_break(np.degrees(theta[2 * i]), np.degrees(theta[2 * i + 1]))


# ------------------------------------------------------------ the stick figure

# The pins each rigid body carries, in order along the bar -- the topology of
# the module docstring, read as a kinematic diagram rather than as two exported
# chains.  Three of the six links are ternary: 1.3 carries P3, P6 and P8, 2.1
# carries P4, P5 and P6, and 2.3 carries P7, P8 and the foot.  On this build the
# first two come out collinear to within a few microns, so every link draws as a
# polyline; a filled triangle would be a sliver on two of the three.  ``TIP``
# stands in for the foot, which is a point on 2.3 and not a pin.
TIP = "tip"
LINK_PINS = {"1.1": ("P1", "P2"), "1.2": ("P2", "P3"), "1.3": ("P3", "P6", "P8"),
             "2.1": ("P4", "P5", "P6"), "2.2": ("P5", "P7"), "2.3": ("P7", "P8", TIP)}
GROUND = ("P1", "P4")        # both hips are mounted on base_link
ACTUATED = ("P1", "P4")      # ... and both are driven; the other six are free


def _hatch(ax, a, b, away_from, colour, scale, n=7):
    """Draw the fixed-frame hatching along the ground link ``a``-``b``.

    The strokes go on the side of the link away from ``away_from``, which is
    the foot: the mechanism hangs off the base, so that is the side with
    nothing on it whichever way round the leg is mounted.
    """
    d = b - a
    L = np.linalg.norm(d)
    u = d / L
    nrm = np.array([-u[1], u[0]])
    if np.dot(nrm, away_from - (a + b) / 2) > 0:
        nrm = -nrm
    # 45 degrees to the link, the usual mark, and swept back along it so the
    # strokes lean the same way rather than fanning.  Both the spacing and the
    # stroke length scale with the link, which on this leg is only 2.8 cm long:
    # a fixed stroke length closes the hatching up into a solid blob there.
    step, run = L / n, 0.22 * L
    for k in range(n + 1):
        p = a + u * (step * k)
        ax.plot(*np.array([p, p + (nrm - u) * run / 2]).T, "-",
                color=colour, lw=0.6 * scale, zorder=2, solid_capstyle="butt")


# The two loops ``_circle_circle`` closes, as (centre, centre, intersection).
# Loop 1 is chain 1's, loop 2 is chain 2's, which is the order CHAIN is in.
LOOPS = (("P2", "P6", "P3"), ("P5", "P8", "P7"))


def _pin_label_spots(P):
    """Where each pin's label goes: off the bars that meet at that pin.

    Each label is pushed away from the mean of the pins it is jointed to,
    rather than radially off the mechanism's centre.  P3, P5 and P6 all sit near
    that centre, where a radial offset is both too short to clear anything and
    very nearly along link 1.3, so all three labels would land on the bar.
    """
    nbr = {k: set() for k in P}
    for pins in LINK_PINS.values():
        for a in pins:
            nbr[a] |= {b for b in pins if b != a}
    nbr["P1"].add("P4")
    nbr["P4"].add("P1")                         # the ground link joins these two
    out = {}
    for name in (k for k in P if k != TIP):
        off = P[name] - np.mean([P[k] for k in nbr[name]], axis=0)
        n = np.linalg.norm(off)
        out[name] = P[name] + (off / n if n > 1e-9 else np.array([1.0, 0.0])) * 0.95
    return out


def _draw_loops(ax, P, scale, chain=CHAIN, avoid=()):
    """Draw each loop's two construction circles and its half-chord ``h``.

    These are the circles the closure actually intersects: one about each
    centre pin, drawn through the intersection pin.  Their radii are therefore
    the rigid link lengths the loop preserves, and can be measured off the pose
    itself rather than passed in -- which is the same fact that makes the
    closure solvable in the first place.

    ``h`` is the perpendicular offset of the intersection pin from the line
    joining the two centres: half the circles' common chord, and the quantity
    the OCP holds away from zero.  It is drawn as that perpendicular, from its
    foot on the centre line out to the pin, so the picture shows what shrinking
    it means -- the two intersections merging as the circles fall tangent, which
    is the only way the mechanism can change assembly branch.

    ``avoid`` holds points the labels must keep clear of, in cm: the pins, their
    labels and samples along the bars.  Each label is called out on a leader
    rather than set beside its segment.  At any pose worth drawing the segment
    is a few millimetres long -- that is what "near the bound" means -- so there
    is no room beside it, and the pins it runs between already carry their own
    labels.  The two loops' labels are numbered, and each one's leader goes to
    its own segment, so they can be told apart when they end up close together.
    """
    from matplotlib.patches import Circle
    from matplotlib.patheffects import withStroke

    avoid = [np.asarray(p) for p in avoid]
    placed = []
    # Candidate spots are kept inside the box the circles span, which is the box
    # export() frames to, so a label never gets pushed off the panel.
    ext = np.array([c + s * np.linalg.norm(P[x] - c)
                    for a, b, x in LOOPS for c in (P[a], P[b]) for s in (-1, 1)])
    lo, hi = ext.min(0), ext.max(0)
    for i, ((a, b, x), colour) in enumerate(zip(LOOPS, chain), start=1):
        c1, c2, pin = P[a], P[b], P[x]
        for c in (c1, c2):
            ax.add_patch(Circle(c, np.linalg.norm(pin - c), fill=False,
                                ec=colour, ls=":", lw=0.8 * scale, alpha=0.55,
                                zorder=1))
        d = c2 - c1
        u = d / np.linalg.norm(d)
        foot = c1 + u * np.dot(pin - c1, u)
        # Kept to two thin strokes: the centre line dashed in the loop's colour,
        # and h as a dark line from it to the pin.  On this leg each half-chord
        # runs almost parallel to a ternary bar (h1 beside P3-P6, h2 beside
        # P7-P8), so any heavier mark merges with the bar into a smudge.  That
        # includes a white casing, end ticks and a right-angle box, all tried.
        # Marking the second intersection too was tried as well: loop 1's lands
        # beside P5 and reads as a ninth pin, and its legend entry costs the
        # fixed-height panel a row.
        ax.plot(*np.array([c1, c2]).T, color=colour, lw=0.5, alpha=0.9,
                ls=(0, (3, 1.5)), zorder=2)
        ax.plot(*np.array([foot, pin]).T, "-", color="0.05", lw=0.9, zorder=6.5,
                solid_capstyle="butt")

        # The best spot is the one whose nearest obstacle is farthest away,
        # searched on rings around the segment's midpoint.  A closer ring wins
        # ties, since a short leader reads as belonging to its segment.
        mid = (foot + pin) / 2
        obstacles = np.array(avoid + placed)
        best, best_score = None, -np.inf
        # Rings start past a label's own width at this scale (~1 cm for 6 pt at
        # half the text width), since anything nearer sits on the pin labels.
        for r in (2.2, 2.8, 3.4, 4.0):
            for ang in np.radians(np.arange(0, 360, 15)):
                q = mid + r * np.array([np.cos(ang), np.sin(ang)])
                if np.any(q < lo) or np.any(q > hi):
                    continue
                score = np.min(np.linalg.norm(obstacles - q, axis=1)) - 0.08 * r
                if score > best_score:
                    best, best_score = q, score
        if best is None:                    # boxed in: fall back beside the segment
            best = mid + u * 2.2
        placed.append(best)
        # The leader is a plain segment stopped short of the label, not an
        # annotate arrow: those did not reach the PDF here at all.
        v = best - mid
        end = best - v / np.linalg.norm(v) * 0.6
        ax.plot(*np.array([mid, end]).T, "-", color=FRAME, lw=0.6, zorder=7,
                solid_capstyle="butt")
        ax.annotate(rf"$h_{i}$", best, fontsize=7.5, zorder=8,
                    ha="center", va="center", color="0.1",
                    path_effects=[withStroke(linewidth=1.6, foreground="w")])


def paint_leg(ax, sol, cloud=None, scale=1.0, chain=CHAIN, labels=True,
              circles=False):
    """Paint one leg as a kinematic stick figure; lengths in cm.

    This is the mechanism the viewer's first window draws as STL outlines,
    reduced to what a thesis figure is actually making a claim about: the bars,
    the eight pins, which two of them are driven, and where the foot is.  The
    meshes carry no kinematic information the pins do not, and at half the text
    width their triangles close up into a grey smear.

    Colour carries one thing only, which chain a link belongs to; the joints,
    the ground and the foot are roles rather than identities and stay in
    ``FRAME`` grey, told apart by marker instead.  ``cloud`` is the reachable
    foot-tip set to stipple behind it, optional and in metres as ``Leg.solve``
    returns it.
    """
    from matplotlib.lines import Line2D

    P = {k: np.asarray(v) * 100.0 for k, v in sol["pins"].items()}
    P[TIP] = np.asarray(sol["tip"]) * 100.0

    if cloud is not None and len(cloud):
        # A swept grid is a point cloud, not a polygon; rasterised so the PDF
        # does not carry one vector dot per sample.
        ax.plot(cloud[:, 0] * 100, cloud[:, 1] * 100, ".", ms=0.7, color="0.88",
                rasterized=True, zorder=0)

    spots = _pin_label_spots(P)
    if circles:
        bars = [P[p] + t * (P[q] - P[p])
                for pins in LINK_PINS.values() for p, q in zip(pins, pins[1:])
                for t in np.linspace(0, 1, 9)]
        _draw_loops(ax, P, scale, chain,
                    avoid=list(P.values()) + (list(spots.values()) if labels else [])
                    + bars)

    _hatch(ax, P["P1"], P["P4"], P[TIP], "0.45", scale)
    ax.plot(*np.array([P["P1"], P["P4"]]).T, "-", color=FRAME, lw=2.2 * scale,
            zorder=3, solid_capstyle="round")
    for name, pins in LINK_PINS.items():
        ax.plot(*np.array([P[k] for k in pins]).T, "-",
                color=chain[0] if name[0] == "1" else chain[1],
                lw=2.2 * scale, zorder=4, solid_capstyle="round",
                solid_joinstyle="round")

    passive = [k for k in P if k not in ACTUATED and k != TIP]
    ax.plot(*np.array([P[k] for k in passive]).T, "o", ms=3.4 * scale, mfc="w",
            mec=FRAME, mew=0.7 * scale, ls="", zorder=5)
    ax.plot(*np.array([P[k] for k in ACTUATED]).T, "o", ms=4.4 * scale,
            color=FRAME, ls="", zorder=6)
    ax.plot(*P[TIP], "D", ms=4.6 * scale, color=FRAME, mec="w",
            mew=0.6 * scale, zorder=7)

    if labels:
        from matplotlib.patheffects import withStroke

        for name, spot in spots.items():
            ax.annotate(name, spot, fontsize=6, zorder=8,
                        ha="center", va="center", color="0.15",
                        path_effects=[withStroke(linewidth=1.3, foreground="w")])

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [cm]")
    ax.set_ylabel(r"$z$ [cm]")
    handles = [
        Line2D([], [], color=chain[0], lw=1.4, label="chain 1"),
        Line2D([], [], color=chain[1], lw=1.4, label="chain 2"),
        Line2D([], [], color=FRAME, marker="o", ls="", ms=3.6, label="actuated"),
        Line2D([], [], color=FRAME, marker="D", ls="", ms=4.0, label="foot"),
    ]
    if circles:
        # Grey, because these two entries name a role: the circles themselves
        # take their loop's chain colour, which the legend already explains.
        handles += [
            Line2D([], [], color="0.45", ls=":", lw=0.9, label="loop circles"),
            Line2D([], [], color="0.05", lw=0.9, label=r"half-chord $h_i$"),
        ]
    return handles


# ---------------------------------------------------------------- viewer

def main(limits=None, trajectory=None, leg="BR", spec_limits=False):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.widgets import RadioButtons, Slider

    legs = {n: Leg(n) for n in LEGS}
    colours = {"1.1": "#d62728", "1.2": "#ff7f0e", "1.3": "#8c564b",
               "2.1": "#1f77b4", "2.2": "#2ca02c", "2.3": "#9467bd"}
    grid = np.linspace(-np.pi, np.pi, 181)
    Q1, Q2 = np.meshgrid(grid, grid)

    # Optional box constraint on the hips, the viewer's twin of bounding theta
    # in the OCP -- under --spec-limits it is that bound itself.  The mask is
    # quantised to the grid, so the reachable set and the tip cloud snap to the
    # nearest 2 deg cell while the drawn box is exact.  It follows the radio
    # buttons, because the sides mount mirrored and their boxes negate.
    box_of = _box_of(limits, spec_limits)

    theta, period = (None, None) if trajectory is None else _trajectory(trajectory)

    fig = plt.figure(figsize=(11, 8))
    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.16, top=0.94)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]  (forward)")
    ax.set_ylabel("z [m]  (up)")

    cloud = ax.plot([], [], ".", ms=1.2, color="0.75", zorder=0,
                    label="foot-tip workspace")[0]
    trace = (ax.plot([], [], "-", color=TRACE, lw=1.4, zorder=1,
                     label="trajectory")[0] if theta is not None else None)
    meshes = {k: LineCollection([], colors=c, lw=0.6, alpha=0.85, zorder=3)
              for k, c in colours.items()}
    for lc in meshes.values():
        ax.add_collection(lc)
    skel = LineCollection([], colors="0.25", lw=1.2, alpha=0.6, zorder=2)
    ax.add_collection(skel)
    pin_dots = ax.plot([], [], "o", ms=6, mfc="w", mec="k", zorder=5)[0]
    tip_dot = ax.plot([], [], "o", ms=9, color="k", zorder=6)[0]
    warn = ax.text(0.5, 0.95, "linkage cannot be assembled", transform=ax.transAxes,
                   ha="center", color="crimson", weight="bold", visible=False)
    read = fig.text(0.02, 0.62, "", family="monospace", fontsize=8, va="top")

    box = box_of(leg)
    (s1lo, s1hi), (s2lo, s2hi) = box if box is not None else ((-180, 180), (-180, 180))
    s1 = Slider(fig.add_axes([0.32, 0.07, 0.6, 0.03]), "Joint x1.1 [deg]", s1lo, s1hi, valinit=0)
    s2 = Slider(fig.add_axes([0.32, 0.02, 0.6, 0.03]), "Joint x2.1 [deg]", s2lo, s2hi, valinit=0)
    radio = RadioButtons(fig.add_axes([0.02, 0.80, 0.12, 0.16]), LEGS, active=LEGS.index(leg))
    state = {"leg": leg, "box": box, "inbox": _inbox(Q1, Q2, box)}

    # second window: the same leg in the coordinates stage1 optimises through
    opt = _optimiser_margin()
    mapax = None
    if opt is not None:
        margin_of, hmin_mm = opt
        mapfig = plt.figure(figsize=(6.8, 6.2))
        mapfig.subplots_adjust(left=0.13, right=0.99, bottom=0.10, top=0.92)
        mapax = mapfig.add_subplot(111)
        mstate = {}

    if theta is not None:
        print(f"{trajectory}: {theta.shape[1]} samples over T = {period:.4f} s")
        if mapax is not None:
            for i, name in enumerate(THETA_LEGS):
                h = margin_of(name, theta[2 * i], theta[2 * i + 1]).min()
                verdict = "inside" if h >= hmin_mm else "VIOLATES"
                print(f"  {name}: min loop half-chord {h:+8.2f} mm  "
                      f"({verdict} the stage1 bound of {hmin_mm:g} mm)")

    def workspace(leg):
        return _reachable(leg, Q1, Q2, grid, state["inbox"])

    def draw_map(name, reach):
        """Redraw the optimiser-coordinate window for a newly picked leg."""
        mapax.clear()
        im = paint_map(mapax, Q1, Q2, margin_of(name, Q1, Q2), reach,
                       state["inbox"], state["box"], hmin_mm,
                       None if theta is None else _map_trace(theta, name))
        mstate["marker"] = mapax.plot([], [], "o", ms=8, mfc="w", mec="k", zorder=5)[0]
        mstate["read"] = mapax.text(0.02, 0.02, "", transform=mapax.transAxes,
                                    family="monospace", fontsize=8, zorder=7,
                                    bbox=dict(fc="w", ec="none", alpha=0.7, pad=2))

        if mstate.get("cbar") is None:
            mstate["cbar"] = mapfig.colorbar(im, ax=mapax, extend="min", pad=0.02)
            mstate["cbar"].set_label("loop half-chord h [mm]  (binding loop)", fontsize=9)
        else:
            mstate["cbar"].update_normal(im)

        mapax.set_xlabel("q1  =  Joint x1.1 [deg]")
        mapax.set_ylabel("q2  =  Joint x2.1 [deg]")
        mapax.set_title(f"BODY2 leg {name} - optimiser coordinate map", weight="bold")
        mapax.legend(handles=[
            Patch(facecolor="none", edgecolor=UNREACH, hatch=UNREACH_HATCH, lw=0.5,
                  label="not reachable from zero pose"),
            Line2D([], [], color="k", lw=1.0, label="assembly limit  h = 0"),
            Line2D([], [], color=BOUND, lw=1.0, ls="--",
                   label=f"stage1 bound  h = {hmin_mm:g} mm"),
            Line2D([], [], color="k", marker="o", ls="", mfc="w", ms=7,
                   label="current pose"),
        ] + ([Line2D([], [], color="k", lw=1.4, label="box constraint")]
             if state["box"] is not None else [])
          + ([Line2D([], [], color=TRACE, lw=1.4, label="trajectory")]
             if theta is not None else []),
            loc="upper right", fontsize=8, framealpha=0.9)

    def draw(_=None):
        leg = legs[state["leg"]]
        if mapax is not None:
            h = float(margin_of(leg.name, np.radians(s1.val), np.radians(s2.val)))
            inside = "inside" if h >= hmin_mm else "OUTSIDE"
            mstate["marker"].set_data([s1.val], [s2.val])
            mstate["read"].set_text(f"h = {h:+8.2f} mm   {inside} the stage1 bound")
            mapfig.canvas.draw_idle()
        sol = leg.solve(np.radians(s1.val), np.radians(s2.val))
        if not sol["ok"]:
            warn.set_visible(True)
            fig.canvas.draw_idle()
            return
        warn.set_visible(False)
        for k, lc in meshes.items():
            t = leg.place(k, sol)
            lc.set_segments(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]))
        P = sol["pins"]
        skel.set_segments([[P[a], P[b]] for a, b in
                           (("P1", "P2"), ("P2", "P3"), ("P3", "P6"), ("P6", "P4"),
                            ("P4", "P1"), ("P4", "P5"), ("P5", "P7"), ("P7", "P8"),
                            ("P8", "P3"))])
        pts = np.array([P[f"P{i}"] for i in range(1, 9)])
        pin_dots.set_data(pts[:, 0], pts[:, 1])
        tip_dot.set_data([sol["tip"][0]], [sol["tip"][1]])
        lines = [f"leg {leg.name}", "", "passive joints [deg]"]
        lines += [f"  {n.split('_')[1]:>8s} {np.degrees(v):+8.2f}"
                  for n, v in sol["joints"].items()]
        lines += ["", "loop-closure pins [deg]"]
        lines += [f"  {n:>16s} {np.degrees(v):+8.2f}" for n, v in sol["closures"].items()]
        lines += ["", f"foot tip  x {sol['tip'][0]:+.4f}", f"          z {sol['tip'][1]:+.4f}"]
        read.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    def clamp(box):
        """Re-range the sliders onto ``box``, pulling a now-outside pose inside."""
        (lo1, hi1), (lo2, hi2) = box if box is not None else ((-180, 180), (-180, 180))
        for s, lo, hi in ((s1, lo1, hi1), (s2, lo2, hi2)):
            s.valmin, s.valmax = lo, hi
            s.ax.set_xlim(lo, hi)
            if not lo <= s.val <= hi:
                s.set_val(min(max(s.val, lo), hi))       # fires draw()

    def pick(label):
        state["leg"] = label
        state["box"] = box_of(label)
        state["inbox"] = _inbox(Q1, Q2, state["box"])
        reach, w = workspace(legs[label])
        if mapax is not None:
            draw_map(label, reach)
        # After draw_map, never before: clamping can fire draw(), which reads the
        # marker that draw_map is what creates.
        clamp(state["box"])
        cloud.set_data(w[:, 0], w[:, 1])
        xs, zs = w[:, 0], w[:, 1]
        if trace is not None:
            i = THETA_LEGS.index(label)
            tip = legs[label].solve(theta[2 * i], theta[2 * i + 1])["tip"]
            trace.set_data(tip[:, 0], tip[:, 1])
            # a guess can leave the reachable set, and does so as nan: keep the
            # frame around whatever of it did land somewhere.
            fin = np.isfinite(tip[:, 0])
            if fin.any():
                xs = np.concatenate([xs, tip[fin, 0]])
                zs = np.concatenate([zs, tip[fin, 1]])
        ax.set_title(f"BODY2 leg {label} - closed-loop kinematics", weight="bold")
        draw()
        m = 0.02
        ax.set_xlim(xs.min() - m, xs.max() + m)
        ax.set_ylim(zs.min() - m, zs.max() + m)
        fig.canvas.draw_idle()

    s1.on_changed(draw)
    s2.on_changed(draw)
    radio.on_clicked(pick)
    ax.legend(loc="upper right", fontsize=8)
    pick(leg)
    plt.show()


# ---------------------------------------------------------------- thesis export

# stage3_visualization/common/thesis_style.py carries the style every figure in the
# thesis is drawn in.  It is reached across the same way STAGE1 is above --
# optionally, one way, and only to draw; a viewer run never imports it, and
# nothing there imports this.
STAGE3 = PKG.parents[1] / "stage3_visualization" / "common"


def _thesis_style():
    """The shared thesis style, activated for a half-text-width figure.

    Importing it switches matplotlib to the 'science' style with real LaTeX
    text, so every label written from here on is TeX.
    """
    import sys

    if str(STAGE3) not in sys.path:
        sys.path.insert(0, str(STAGE3))
    try:
        import thesis_style
    except ImportError as exc:
        raise SystemExit(f"--export needs {STAGE3}/thesis_style.py and its "
                         f"dependencies (scienceplots, a LaTeX install): {exc}")
    thesis_style.half_width()
    return thesis_style


def _suffixed(path, name):
    """``out.pdf`` -> ``out_leg.pdf``, so a row of panels shares one stem."""
    path = Path(path)
    return path.with_name(f"{path.stem}_{name}{path.suffix}")


def _finish(fig, ax, handles):
    """Put ``handles`` in a legend above the axes and lay the figure out.

    The canvas stays the one thesis_style hands out, legend and all: both of
    these panels are bound by the width left over beside their labels rather
    than by height, so the legend's band comes out of height nothing was using.
    Neither thesis_style.legend_row nor a rect reserving the band belongs here
    then -- matplotlib's own tight_layout already makes room for a legend
    hanging off the axes, and both would reserve it twice.

    tight_layout runs once and only once: a colorbar attached with ``ax=ax``
    takes its space out of that axes, so a second pass takes it a second time
    and the panel walks inwards.
    """
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, framealpha=0.0, handlelength=1.6, columnspacing=1.2,
              borderaxespad=0.0)
    fig.tight_layout()


def export(path, leg="BR", limits=None, trajectory=None, resolution=361,
           spec_limits=False, pose=(0.0, 0.0), workspace=False,
           circles=False):
    """Write the two thesis panels for one leg, as ``<path>_leg`` and ``_map``.

    Both are the viewer's two windows with the dressing changed: titles go (a
    thesis figure is named by its caption), the pose marker and readout go
    (there is no slider to move), each legend moves above its axes where it
    cannot sit on the data, and the labels are TeX.  They are written to a
    suffixed pair rather than to ``path`` itself so that a row of two shares one
    stem, the way ``stage3_visualization/thrust/hind_workspace.py`` writes its pair.

    The map goes through ``paint_map``, which also draws the viewer's, so the
    set that reaches the page is the set that was checked on screen.  The leg
    panel does not: the viewer draws STL outlines and this draws the kinematic
    diagram, which is the claim a thesis figure is actually making, and at half
    the text width the meshes close up into a grey smear anyway.

    The sweep is finer than the viewer's 2 deg because a printed contour shows
    the staircase that a screen at a third of the size hides.
    """
    style = _thesis_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if leg not in LEGS:
        raise SystemExit(f"unknown leg {leg!r}; this robot has {LEGS}")
    opt = _optimiser_margin()
    if opt is None:
        raise SystemExit("--export draws the optimiser's own coordinate map, so "
                         "it needs stage1 and casadi importable")
    margin_of, hmin_mm = opt

    box = _box_of(limits, spec_limits)(leg)
    grid = np.linspace(-np.pi, np.pi, resolution)
    Q1, Q2 = np.meshgrid(grid, grid)
    inbox = _inbox(Q1, Q2, box)
    obj = Leg(leg)
    reach, cloud = _reachable(obj, Q1, Q2, grid, inbox)

    # The trajectory is drawn on the map only.  In hip coordinates it is a path
    # through the constraint the map is a picture of, which is a claim; over the
    # stick figure it would only be a second curve in a panel whose subject is
    # the mechanism at one pose.
    traj = None
    if trajectory is not None:
        theta, period = _trajectory(trajectory)
        traj = _map_trace(theta, leg)
        i = THETA_LEGS.index(leg)
        h = margin_of(leg, theta[2 * i], theta[2 * i + 1]).min()
        verdict = "inside" if h >= hmin_mm else "VIOLATES"
        print(f"{trajectory}: {theta.shape[1]} samples over T = {period:.4f} s")
        print(f"  {leg}: min loop half-chord {h:+8.2f} mm  "
              f"({verdict} the stage1 bound of {hmin_mm:g} mm)")

    # ---- the leg, as a kinematic stick figure
    sol = obj.solve(*np.radians(pose))
    if not sol["ok"]:
        raise SystemExit(f"--pose {pose[0]:g} {pose[1]:g}: leg {leg} cannot be "
                         f"assembled there, so there is no mechanism to draw")
    # thesis_style.HALF's width, which is what has to match the LaTeX slot, but
    # taller.  Both panels have an equal aspect with a legend above, so at
    # HALF's own height they are height-bound and the spare width is blank
    # margin; at this height the map fills its width.  The leg panel gets the
    # same canvas so the two still sit in one row at one height.
    canvas = (style.HALF[0], 2.55)
    legfig, legax = plt.subplots(figsize=canvas)
    # Heavier than the map's 0.55: the mechanism is a small object next to the
    # workspace it sweeps, so at the map's weights the bars that are the whole
    # subject of the panel read as thinner than its grid.
    # With --circles the pin labels go.  At half the text width P3 and P6 end up
    # ~12 pt apart on the page, and that gap would have to hold both labels plus
    # h1's segment and its callout.  No placement fits, and each attempt buried
    # the half-chord, which is the point of the variant.  The plain leg figure
    # carries the pin names.
    handles = paint_leg(legax, sol, cloud if workspace else None, scale=0.85,
                        labels=not circles, circles=circles)
    if workspace:
        handles.append(Line2D([], [], color="0.88", marker="s", ls="", ms=4,
                              label="workspace"))

    # Framed on what the panel is about -- the bars -- and on the workspace only
    # when it was asked for.  That cloud is a 13 cm disc once nothing bounds the
    # hips, so framing to it by default would leave the mechanism at a seventh
    # of the panel width, which is not a stick figure any more.  The map panel
    # is where the reachable set is the claim being made; here it is background,
    # and opt-in.
    pts = np.array([np.asarray(p) for p in sol["pins"].values()]
                   + [np.asarray(sol["tip"])]) * 100.0
    if workspace and len(cloud):
        pts = np.vstack([pts, cloud * 100.0])
    if circles:
        # A construction circle reaches a full link length past its centre
        # pin, so framing on the pins alone would crop both of them.
        Pc = {k: np.asarray(v) * 100.0 for k, v in sol["pins"].items()}
        span = []
        for a, b, x in LOOPS:
            for c in (Pc[a], Pc[b]):
                r = np.linalg.norm(Pc[x] - c)
                span += [c + r, c - r]
        pts = np.vstack([pts, np.array(span)])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.12 * max(hi - lo)
    legax.set_xlim(lo[0] - pad, hi[0] + pad)
    legax.set_ylim(lo[1] - pad, hi[1] + pad)
    legax.grid(alpha=0.25, lw=0.4)
    legax.set_axisbelow(True)
    _finish(legfig, legax, handles)
    leg_path = _suffixed(path, "leg")
    legfig.savefig(leg_path, dpi=300)
    print(f"Saved → {leg_path}")

    # ---- the optimiser's coordinate map
    fig, ax = plt.subplots(figsize=canvas)
    im = paint_map(ax, Q1, Q2, margin_of(leg, Q1, Q2), reach, inbox, box,
                   hmin_mm, traj, scale=0.55)
    cbar = fig.colorbar(im, ax=ax, extend="min", pad=0.03, fraction=0.046)
    cbar.set_label(r"$h$ [mm]")
    cbar.ax.tick_params(length=2)
    ax.set_xlabel(r"$q_1$ [deg]")
    ax.set_ylabel(r"$q_2$ [deg]")
    # The square comes out about 1.2 in wide, which is not enough for five
    # labelled ticks: -180 and -90 touch.  The quadrant boundaries stay as
    # minor ticks, so the axis still reads at 90 deg without the collision.
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks(range(-180, 181, 180))
        axis.set_ticks(range(-90, 91, 180), minor=True)

    # Short labels on purpose: at half the text width a spelled-out legend is
    # half the figure.  What each one means belongs in the caption.
    handles = [
        Patch(facecolor="none", edgecolor=UNREACH, hatch=UNREACH_HATCH, lw=0.5,
              label="not reachable"),
        Line2D([], [], color="k", lw=0.6, label=r"$h = 0$"),
        Line2D([], [], color=BOUND, lw=0.6, ls="--", label=rf"$h = {hmin_mm:g}$ mm"),
    ]
    if box is not None:
        handles.append(Line2D([], [], color="k", lw=0.8, label="limits"))
    if traj is not None:
        handles.append(Line2D([], [], color=TRACE, lw=0.8, label="trajectory"))

    _finish(fig, ax, handles)
    map_path = _suffixed(path, "map")
    fig.savefig(map_path, dpi=300)
    print(f"Saved → {map_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Interactive viewer for the BODY2 legs.")
    lim = ap.add_mutually_exclusive_group()
    lim.add_argument("--limits", nargs=4, type=float,
                     metavar=("Q1MIN", "Q1MAX", "Q2MIN", "Q2MAX"),
                     help="box constraint on the two hip angles, in degrees.  The "
                          "sliders are clamped to it, the reachable set is flooded "
                          "inside it only -- so the foot-tip workspace shows what the "
                          "constrained leg can actually reach -- and the map shades "
                          "out the rest.  Must contain the zero pose.")
    lim.add_argument("--spec-limits", action="store_true",
                     help="take that box from stage1's body2 spec instead of "
                          "typing it: HIP_BOX, which is what bounds theta in the "
                          "OCP, so a --trajectory is drawn against the limits it "
                          "was solved under.  Note a solution .npz records no "
                          "limits of its own, so this is the spec as it stands "
                          "now.  The box is per leg -- the sides mount mirrored "
                          "and theirs negate -- and follows the leg on screen.")
    ap.add_argument("--trajectory", metavar="NPZ",
                    help="draw a stage1 solution or initial guess: in the viewer "
                         "the hip path on the coordinate map and the foot path on "
                         "the leg, on --export the hip path only.  Any .npz "
                         "written by hydro_model.trajectory.save_solution for "
                         "body2 in reduced coordinates.")
    ap.add_argument("--leg", default="BR", choices=LEGS,
                    help="leg to show; the viewer opens on it, --export writes it")
    ap.add_argument("--export", metavar="PDF",
                    help="skip the viewer and write thesis figures instead, in "
                         "the shared style of stage3_visualization and each "
                         "sized for a half-text-width slot so the two sit side "
                         "by side.  Writes a suffixed pair, <stem>_leg and "
                         "<stem>_map -- the leg as a kinematic stick figure at "
                         "one pose, and the optimiser's coordinate map, which "
                         "is the panel a --trajectory is drawn on.")
    ap.add_argument("--pose", nargs=2, type=float, default=(0.0, 0.0),
                    metavar=("Q1", "Q2"),
                    help="hip angles in degrees for the pose the --export stick "
                         "figure is drawn at, the twin of the viewer's sliders "
                         "(default 0 0, the assembly pose)")
    ap.add_argument("--workspace", action="store_true",
                    help="draw the reachable foot-tip cloud behind the --export "
                         "stick figure, and widen it to fit.  Off by default: "
                         "unbounded that cloud is a 13 cm disc around a 12 cm "
                         "leg, and framing to it shrinks the mechanism to a "
                         "seventh of the panel.")
    ap.add_argument("--circles", action="store_true",
                    help="on the --export stick figure, draw the two "
                         "construction circles of each loop and its "
                         "half-chord h, the margin the OCP bounds away "
                         "from zero.  Widens the panel to fit them.")
    ap.add_argument("--resolution", type=int, default=361,
                    help="sweep samples per hip axis for --export (default 361, "
                         "i.e. 1 deg; the viewer always runs at 2 deg)")
    args = ap.parse_args()
    if args.export:
        export(args.export, args.leg, args.limits, args.trajectory,
               args.resolution, args.spec_limits, args.pose, args.workspace,
               args.circles)
    else:
        main(args.limits, args.trajectory, args.leg, args.spec_limits)
