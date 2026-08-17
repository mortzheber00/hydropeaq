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
angles, the viewer's twin of bounding ``theta`` in the OCP, and ``--trajectory``
draws a saved solution or initial guess across both windows.
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


# ---------------------------------------------------------------- viewer

def main(limits=None, trajectory=None):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import RadioButtons, Slider

    legs = {n: Leg(n) for n in LEGS}
    colours = {"1.1": "#d62728", "1.2": "#ff7f0e", "1.3": "#8c564b",
               "2.1": "#1f77b4", "2.2": "#2ca02c", "2.3": "#9467bd"}
    grid = np.linspace(-np.pi, np.pi, 181)
    Q1, Q2 = np.meshgrid(grid, grid)

    # Optional box constraint on the hips, the viewer's twin of bounding theta
    # in the OCP.  The mask is quantised to the grid, so the reachable set and
    # the tip cloud snap to the nearest 2 deg cell while the drawn box is exact.
    box = None if limits is None else _box(limits)
    if box is None:
        inbox = np.ones(Q1.shape, bool)
    else:
        (a1, b1), (a2, b2) = box
        d1, d2 = np.degrees(Q1), np.degrees(Q2)
        inbox = (d1 >= a1) & (d1 <= b1) & (d2 >= a2) & (d2 <= b2)

    theta, period = (None, None) if trajectory is None else _trajectory(trajectory)
    TRACE = "#e6007e"                          # reads against RdBu, the island and the meshes

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

    (s1lo, s1hi), (s2lo, s2hi) = box if box is not None else ((-180, 180), (-180, 180))
    s1 = Slider(fig.add_axes([0.32, 0.07, 0.6, 0.03]), "Joint x1.1 [deg]", s1lo, s1hi, valinit=0)
    s2 = Slider(fig.add_axes([0.32, 0.02, 0.6, 0.03]), "Joint x2.1 [deg]", s2lo, s2hi, valinit=0)
    radio = RadioButtons(fig.add_axes([0.02, 0.80, 0.12, 0.16]), LEGS, active=LEGS.index("BR"))
    state = {"leg": "BR"}

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
        """Configurations reachable from the zero pose without taking the leg apart.

        Assemblable (q1, q2) cells split into several islands; only the one holding
        the zero pose can be driven to, so the rest are flooded away.  The hips are
        continuous joints, hence the wrap-around neighbourhood -- a box constraint
        needs no special case there, since wrapping from +180 to -180 has to cross
        cells the box already masks out unless the box spans the full turn.

        Returns the reachable mask over the (q1, q2) grid and the foot tips it
        maps to: the same set drawn once in each of the viewer's two windows.
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

    def draw_map(name, reach):
        """Redraw the optimiser-coordinate window for a newly picked leg.

        The colour field is the binding loop's half-chord: red where the leg
        cannot be assembled at all, blue where it can, and the two contours are
        the assembly limit and the tighter bound stage1 actually constrains.
        The reachable island is the same set the other window draws as the
        foot-tip cloud, so the two are readable side by side.
        """
        margin = margin_of(name, Q1, Q2)
        deg1, deg2 = np.degrees(Q1), np.degrees(Q2)
        half = np.degrees(grid[1] - grid[0]) / 2
        lim = 180 + half
        vmax = np.nanmax(margin)

        mapax.clear()
        im = mapax.imshow(margin, origin="lower", extent=(-lim, lim, -lim, lim),
                          cmap="RdBu", vmin=-vmax, vmax=vmax, interpolation="nearest")
        # On all four legs of this build the assemblable set turns out to be a
        # single island, so the green boundary lies exactly on the h = 0 contour.
        # The island goes down first and thicker: the black line then stays on
        # top with a green fringe instead of being painted over.
        mapax.contour(deg1, deg2, reach.astype(float), [0.5],
                      colors="#2ca02c", linewidths=2.2)
        mapax.contour(deg1, deg2, margin, [0.0], colors="k", linewidths=1.0)
        mapax.contour(deg1, deg2, margin, [hmin_mm], colors="k", linewidths=1.0,
                      linestyles="dashed")
        if box is not None:
            # shade out what the constraint forbids, rather than cropping to it:
            # the excluded structure is exactly what one wants to see when
            # deciding whether the box is drawn in the right place.
            veil = np.ones(Q1.shape + (4,))          # white, i.e. wash out
            veil[..., 3] = np.where(inbox, 0.0, 0.62)
            mapax.imshow(veil, origin="lower", extent=(-lim, lim, -lim, lim),
                         interpolation="nearest", zorder=4)
            mapax.add_patch(Rectangle((box[0][0], box[1][0]), box[0][1] - box[0][0],
                                      box[1][1] - box[1][0], fill=False, ec="k",
                                      lw=1.4, zorder=6))
        if theta is not None:
            i = THETA_LEGS.index(name)
            tq1, tq2 = _wrap_break(np.degrees(theta[2 * i]), np.degrees(theta[2 * i + 1]))
            mapax.plot(tq1, tq2, "-", color=TRACE, lw=1.4, zorder=7)
            mapax.plot(tq1[:1], tq2[:1], "o", color=TRACE, ms=5, zorder=7)
        mstate["marker"] = mapax.plot([], [], "o", ms=8, mfc="w", mec="k", zorder=5)[0]
        mstate["read"] = mapax.text(0.02, 0.02, "", transform=mapax.transAxes,
                                    family="monospace", fontsize=8, zorder=7,
                                    bbox=dict(fc="w", ec="none", alpha=0.7, pad=2))

        if mstate.get("cbar") is None:
            mstate["cbar"] = mapfig.colorbar(im, ax=mapax, extend="min", pad=0.02)
            mstate["cbar"].set_label("loop half-chord h [mm]  (binding loop)", fontsize=9)
        else:
            mstate["cbar"].update_normal(im)

        mapax.set_aspect("equal")
        mapax.set_xlim(-lim, lim)
        mapax.set_ylim(-lim, lim)
        mapax.set_xticks(range(-180, 181, 90))
        mapax.set_yticks(range(-180, 181, 90))
        mapax.set_xlabel("q1  =  Joint x1.1 [deg]")
        mapax.set_ylabel("q2  =  Joint x2.1 [deg]")
        mapax.set_title(f"BODY2 leg {name} - optimiser coordinate map", weight="bold")
        mapax.legend(handles=[
            Line2D([], [], color="#2ca02c", lw=2.2, label="reachable from zero pose"),
            Line2D([], [], color="k", lw=1.0, label="assembly limit  h = 0"),
            Line2D([], [], color="k", lw=1.0, ls="--",
                   label=f"stage1 bound  h = {hmin_mm:g} mm"),
            Line2D([], [], color="k", marker="o", ls="", mfc="w", ms=7,
                   label="current pose"),
        ] + ([Line2D([], [], color="k", lw=1.4, label="box constraint")]
             if box is not None else [])
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

    def pick(label):
        state["leg"] = label
        reach, w = workspace(legs[label])
        if mapax is not None:
            draw_map(label, reach)
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
    pick("BR")
    plt.show()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Interactive viewer for the BODY2 legs.")
    ap.add_argument("--limits", nargs=4, type=float,
                    metavar=("Q1MIN", "Q1MAX", "Q2MIN", "Q2MAX"),
                    help="box constraint on the two hip angles, in degrees.  The "
                         "sliders are clamped to it, the reachable set is flooded "
                         "inside it only -- so the foot-tip workspace shows what the "
                         "constrained leg can actually reach -- and the map shades "
                         "out the rest.  Must contain the zero pose.")
    ap.add_argument("--trajectory", metavar="NPZ",
                    help="draw a stage1 solution or initial guess: the hip path on "
                         "the coordinate map and the foot path on the leg.  Any "
                         ".npz written by hydro_model.trajectory.save_solution for "
                         "body2 in reduced coordinates.")
    args = ap.parse_args()
    main(args.limits, args.trajectory)
