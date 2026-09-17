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

Pin positions are fitted to the STL holes (the exported joint origins are off
by up to 2.6 mm).

Without arguments an interactive viewer opens: the leg with hip sliders, and a
second window with the optimiser's (q1, q2) coordinate map. ``--export`` writes
the stick figure and the map as thesis figures instead.
``Leg("BR").solve(q1, q2)`` gives the passive joint values.

Usage:
  python src/BODY2/scripts/leg_linkage_sim.py --leg BR
  python src/BODY2/scripts/leg_linkage_sim.py --spec-limits --trajectory task3_solution.npz
  python src/BODY2/scripts/leg_linkage_sim.py --export body2.pdf --circles --pose 20 10
"""

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
URDF = PKG / "urdf" / "BODY2.urdf"
MESHES = PKG / "meshes"
LEGS = ["FL", "FR", "BL", "BR"]

# Leg planes are normal to world y, so planar maths uses (x, z).
PLANE = [0, 2]


# --- URDF / STL ---

def _rpy(r, p, y):
    """Rotation matrix from URDF roll/pitch/yaw."""
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
    """URDF joints by name: parent, child, origin and axis."""
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

    Rotated visual origins are stored as an error raised on lookup, since hole
    fitting assumes bores along the mesh z (base_link has one but is never used).
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
    """Mesh offset of ``link``; raises for rotated visual origins."""
    tv = VISUAL[link]
    if isinstance(tv, Exception):
        raise tv
    return tv


JOINTS = _joints()
VISUAL = _visual_offsets()


def urdf_joint(leg: str, key: str) -> str:
    """URDF joint name for a linkage key: ``("FL", "1.1") -> "Joint_FL1_1"``.

    ROS names cannot contain dots. Duplicated in hydro_model/robots/body2.py.
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


# --- Pin holes ---

def _hole_centres(tri, nrm):
    """Centres (local x, y) of a link's pin holes.

    Bore walls are the triangles with normals perpendicular to local z; they
    are clustered and fitted with circles (inner and outer walls separately).
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
    # Hips have a servo horn instead of a bore: use the URDF joint origins
    return dict(P1=T[f"Link_{leg}1.1"][1][PLANE], P2=P2, P3=P3,
                P4=T[f"Link_{leg}2.1"][1][PLANE], P5=P5, P6=P6, P7=P7, P8=P8)


# --- Planar maths ---

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

        # Sign mapping each URDF joint value to a CCW rotation in the (x, z) plane
        self.sgn = {}
        hip = None
        for c in (1, 2):
            for k in (1, 2, 3):
                j = JOINTS[f"Joint_{name}{c}_{k}"]
                axis = self.frames[j["parent"]][0] @ j["R"] @ j["axis"]
                if np.linalg.norm(axis) < 1e-9:
                    axis = hip          # *.3 exported as fixed; use the hip axis
                if abs(abs(axis[1]) - 1) > 1e-6:
                    raise RuntimeError(f"{name}{c}.{k}: joint axis is not along world Y")
                hip = hip if hip is not None else axis
                self.sgn[f"{c}.{k}"] = -1.0 if axis[1] > 0 else 1.0

        self.L = {  # rigid pin distances
            "P2P3": np.linalg.norm(p["P3"] - p["P2"]),
            "P6P3": np.linalg.norm(p["P3"] - p["P6"]),
            "P5P7": np.linalg.norm(p["P7"] - p["P5"]),
            "P8P7": np.linalg.norm(p["P7"] - p["P8"]),
        }
        # Assembly branches of the zero pose
        self.branch = (self._branch(p["P2"], p["P6"], p["P3"]),
                       self._branch(p["P5"], p["P8"], p["P7"]))

        # Planar link meshes at the zero pose and the foot tip (farthest point from P7)
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
        """Side (+-1) of ``pt`` relative to the line c1 -> c2."""
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
            # URDF passive joints
            joints={urdf_joint(self.name, "1.2"): wrap(self.sgn["1.2"] * (th12 - th1)),
                    urdf_joint(self.name, "1.3"): wrap(self.sgn["1.3"] * (th13 - th12)),
                    urdf_joint(self.name, "2.2"): wrap(self.sgn["2.2"] * (th22 - th2)),
                    urdf_joint(self.name, "2.3"): wrap(self.sgn["2.3"] * (th23 - th22))},
            # Loop-closure pins (not in the URDF)
            closures={"P6 (2.1<->1.3)": wrap(self.sgn["1.1"] * (th13 - th2)),
                      "P8 (1.3<->2.3)": wrap(self.sgn["1.1"] * (th23 - th13))})

    def place(self, key, sol):
        """Mesh of link ``key`` moved into the pose held in ``sol``."""
        anchor = {"1.1": "P1", "1.2": "P2", "1.3": "P3",
                  "2.1": "P4", "2.2": "P5", "2.3": "P7"}[key]
        a0, a = self.pins0[anchor], sol["pins"][anchor]
        return a + _rot2(self.mesh[key] - a0, sol["theta"][key])


# --- Optimiser coordinates ---

STAGE1 = PKG.parents[1] / "stage1_gait_optimization"


def _optimiser_margin():
    """Loop half-chord of stage1's coordinate map, for the OCP's ``h >= H_MIN`` constraint.

    Returns ``(margin, H_MIN_mm)`` with ``margin(leg, Q1, Q2)`` the signed
    half-chord [mm] of the binding loop, or None if stage1/casadi cannot be
    imported.
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
    """``(theta (8, N+1), T)`` from a BODY2 solution or guess ``.npz``.

    Read with plain numpy to avoid importing hydro_model (casadi, pinocchio).
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


# Leg order of theta (= hydro_model.robots.body2.LEG_NAMES)
THETA_LEGS = ("FL", "FR", "BL", "BR")


def _wrap_break(a, b):
    """Wrap two angle series [deg] into (-180, 180] and insert NaN breaks at the wraps."""
    wa, wb = (a + 180) % 360 - 180, (b + 180) % 360 - 180
    cut = np.flatnonzero((np.abs(np.diff(wa)) > 180) | (np.abs(np.diff(wb)) > 180)) + 1
    return np.insert(wa, cut, np.nan), np.insert(wb, cut, np.nan)


def _box(limits):
    """Validate four limits [deg] into ``((q1lo, q1hi), (q2lo, q2hi))``.

    The box must contain the zero pose, which defines the assembly branch.
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
    """This leg's hip box [deg] from the current body2 spec (HIP_BOX).

    Solutions do not store their limits, so this is the spec as it is now.
    The box differs per side (mirrored mounting).
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
    # Validate like a --limits box
    return _box((lo1, hi1, lo2, hi2))


def _box_of(limits, spec_limits):
    """Function ``leg -> box or None`` from the --limits / --spec-limits flags."""
    if spec_limits:
        return _spec_hip_box
    fixed = None if limits is None else _box(limits)
    return lambda _: fixed


# --- Coordinate map ---

# Colours from thesis_style.PALETTE, copied to avoid importing it (it enables
# LaTeX rendering). Keep in sync.
CHAIN = ("#0173B2", "#B2182B")   # the two hip chains
UNREACH, UNREACH_HATCH = "0.15", "////"   # not reachable from the zero pose
TRACE = "#0173B2"   # trajectory
BOUND = "#B2182B"   # stage1 h bound
FRAME = "0.2"       # ground, joints and foot (colour is reserved for the chains)


def _inbox(Q1, Q2, box):
    """Grid cells inside ``box`` (all if None).

    A small tolerance keeps cells lying exactly on the edge despite float error.
    """
    if box is None:
        return np.ones(Q1.shape, bool)
    eps = 1e-9
    (a1, b1), (a2, b2) = box
    d1, d2 = np.degrees(Q1), np.degrees(Q2)
    return ((d1 >= a1 - eps) & (d1 <= b1 + eps)
            & (d2 >= a2 - eps) & (d2 <= b2 + eps))


def _reachable(leg, Q1, Q2, grid, inbox):
    """Assemblable cells connected to the zero pose, and their foot tips.

    Flood fill with wrap-around (continuous hips). Returns ``(mask, tips)``.
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
    """Draw one leg's (q1, q2) map: half-chord field, h = 0 and H_MIN contours,
    unreachable hatch, optional box and trajectory. Returns the image.

    Used by both the viewer and --export; ``scale`` thins the lines for print.
    """
    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    deg1, deg2 = np.degrees(Q1), np.degrees(Q2)
    half = (deg1[0, 1] - deg1[0, 0]) / 2
    lim = 180 + half
    vmax = np.nanmax(margin)

    # Greyscale field (without the black end, so the contours stay visible)
    greys = ListedColormap(colormaps["Greys_r"](np.linspace(0.35, 1.0, 256)))
    im = ax.imshow(margin, origin="lower", extent=(-lim, lim, -lim, lim),
                   cmap=greys, vmin=-vmax, vmax=vmax, interpolation="nearest")
    # Hatch the unreachable region. Matplotlib 3.7 reads the hatch width from
    # rcParams at save time, hence the global setting.
    import matplotlib
    matplotlib.rcParams["hatch.linewidth"] = 0.4 * scale
    with matplotlib.rc_context({"hatch.color": UNREACH}):
        ax.contourf(deg1, deg2, reach.astype(float), [-0.5, 0.5],
                    colors="none", hatches=[UNREACH_HATCH])
    ax.contour(deg1, deg2, margin, [0.0], colors="k", linewidths=1.0 * scale)
    ax.contour(deg1, deg2, margin, [hmin_mm], colors=BOUND,
               linewidths=1.0 * scale, linestyles="dashed")
    if box is not None:
        # Fade (not crop) the region outside the box
        veil = np.ones(Q1.shape + (4,))          # white overlay
        veil[..., 3] = np.where(inbox, 0.0, 0.62)
        ax.imshow(veil, origin="lower", extent=(-lim, lim, -lim, lim),
                  interpolation="nearest", zorder=4)
        ax.add_patch(Rectangle((box[0][0], box[1][0]), box[0][1] - box[0][0],
                               box[1][1] - box[1][0], fill=False, ec="k",
                               lw=1.4 * scale, zorder=6))
    if traj is not None:
        # White casing separates the path from nearby contours
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
    """The leg's hip path [deg] for ``paint_map``."""
    i = THETA_LEGS.index(leg)
    return _wrap_break(np.degrees(theta[2 * i]), np.degrees(theta[2 * i + 1]))


# --- Stick figure ---

# Pins carried by each link, in order along the bar (drawn as polylines; the
# ternary links 1.3 and 2.1 are nearly collinear). TIP is the foot point on 2.3.
TIP = "tip"
LINK_PINS = {"1.1": ("P1", "P2"), "1.2": ("P2", "P3"), "1.3": ("P3", "P6", "P8"),
             "2.1": ("P4", "P5", "P6"), "2.2": ("P5", "P7"), "2.3": ("P7", "P8", TIP)}
GROUND = ("P1", "P4")        # pins on base_link
ACTUATED = ("P1", "P4")      # driven pins


def _hatch(ax, a, b, away_from, colour, scale, n=7):
    """Ground hatching along ``a``-``b``, on the side away from ``away_from`` (the foot)."""
    d = b - a
    L = np.linalg.norm(d)
    u = d / L
    nrm = np.array([-u[1], u[0]])
    if np.dot(nrm, away_from - (a + b) / 2) > 0:
        nrm = -nrm
    # 45 deg strokes; spacing and length scale with the (short) link
    step, run = L / n, 0.22 * L
    for k in range(n + 1):
        p = a + u * (step * k)
        ax.plot(*np.array([p, p + (nrm - u) * run / 2]).T, "-",
                color=colour, lw=0.6 * scale, zorder=2, solid_capstyle="butt")


# Loops as (centre, centre, intersection pin), in CHAIN order
LOOPS = (("P2", "P6", "P3"), ("P5", "P8", "P7"))


def _pin_label_spots(P):
    """Label position per pin, offset away from the pins it connects to."""
    nbr = {k: set() for k in P}
    for pins in LINK_PINS.values():
        for a in pins:
            nbr[a] |= {b for b in pins if b != a}
    nbr["P1"].add("P4")
    nbr["P4"].add("P1")                         # ground link
    out = {}
    for name in (k for k in P if k != TIP):
        off = P[name] - np.mean([P[k] for k in nbr[name]], axis=0)
        n = np.linalg.norm(off)
        out[name] = P[name] + (off / n if n > 1e-9 else np.array([1.0, 0.0])) * 0.95
    return out


def _draw_loops(ax, P, scale, chain=CHAIN, avoid=()):
    """Draw each loop's construction circles and its half-chord ``h_i``.

    ``h`` is the distance of the intersection pin from the line between the
    circle centres (the quantity the OCP keeps positive). Labels are placed on
    leaders at the spot farthest from ``avoid`` (points in cm).
    """
    from matplotlib.patches import Circle
    from matplotlib.patheffects import withStroke

    avoid = [np.asarray(p) for p in avoid]
    placed = []
    # Keep labels inside the circles' bounding box (the export frame)
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
        # Thin strokes only; h runs nearly parallel to a bar and heavier marks merge with it.
        ax.plot(*np.array([c1, c2]).T, color=colour, lw=0.5, alpha=0.9,
                ls=(0, (3, 1.5)), zorder=2)
        ax.plot(*np.array([foot, pin]).T, "-", color="0.05", lw=0.9, zorder=6.5,
                solid_capstyle="butt")

        # Search rings around the midpoint for the spot farthest from obstacles
        # (slight preference for short leaders).
        mid = (foot + pin) / 2
        obstacles = np.array(avoid + placed)
        best, best_score = None, -np.inf
        # Radii start beyond the pin labels (~1 cm)
        for r in (2.2, 2.8, 3.4, 4.0):
            for ang in np.radians(np.arange(0, 360, 15)):
                q = mid + r * np.array([np.cos(ang), np.sin(ang)])
                if np.any(q < lo) or np.any(q > hi):
                    continue
                score = np.min(np.linalg.norm(obstacles - q, axis=1)) - 0.08 * r
                if score > best_score:
                    best, best_score = q, score
        if best is None:                    # fallback: beside the segment
            best = mid + u * 2.2
        placed.append(best)
        # Leader as a plain line (annotate arrows did not render in the PDF)
        v = best - mid
        end = best - v / np.linalg.norm(v) * 0.6
        ax.plot(*np.array([mid, end]).T, "-", color=FRAME, lw=0.6, zorder=7,
                solid_capstyle="butt")
        ax.annotate(rf"$h_{i}$", best, fontsize=7.5, zorder=8,
                    ha="center", va="center", color="0.1",
                    path_effects=[withStroke(linewidth=1.6, foreground="w")])


def paint_leg(ax, sol, cloud=None, scale=1.0, chain=CHAIN, labels=True,
              circles=False):
    """Draw one leg as a kinematic diagram in cm; returns legend handles.

    Colour marks the chain; ground, joints and foot are grey with distinct
    markers. ``cloud`` (optional, metres) is the reachable foot-tip set.
    """
    from matplotlib.lines import Line2D

    P = {k: np.asarray(v) * 100.0 for k, v in sol["pins"].items()}
    P[TIP] = np.asarray(sol["tip"]) * 100.0

    if cloud is not None and len(cloud):
        # Rasterised to keep the PDF small
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
        handles += [
            Line2D([], [], color="0.45", ls=":", lw=0.9, label="loop circles"),
            Line2D([], [], color="0.05", lw=0.9, label=r"half-chord $h_i$"),
        ]
    return handles


# --- Interactive viewer ---

def main(limits=None, trajectory=None, leg="BR", spec_limits=False):
    """Slider viewer of one leg plus its coordinate map (if stage1 is importable)."""
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

    # Optional hip box (per leg); the reachable set uses the 2 deg grid.
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

    # Second window: coordinate map
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
        """Reachable mask and tips for the current box."""
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
        """Redraw the leg and readouts for the current slider values."""
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
        """Switch to another leg."""
        state["leg"] = label
        state["box"] = box_of(label)
        state["inbox"] = _inbox(Q1, Q2, state["box"])
        reach, w = workspace(legs[label])
        if mapax is not None:
            draw_map(label, reach)
        # After draw_map: clamp() may call draw(), which needs the map marker.
        clamp(state["box"])
        cloud.set_data(w[:, 0], w[:, 1])
        xs, zs = w[:, 0], w[:, 1]
        if trace is not None:
            i = THETA_LEGS.index(label)
            tip = legs[label].solve(theta[2 * i], theta[2 * i + 1])["tip"]
            trace.set_data(tip[:, 0], tip[:, 1])
            # Unassemblable samples are NaN; frame the finite ones
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


# --- Thesis export ---

# Location of thesis_style.py (imported only for --export)
STAGE3 = PKG.parents[1] / "stage3_visualization" / "common"


def _thesis_style():
    """Import and activate the thesis style (LaTeX text) for half-width figures."""
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
    """``out.pdf`` -> ``out_<name>.pdf``."""
    path = Path(path)
    return path.with_name(f"{path.stem}_{name}{path.suffix}")


def _finish(fig, ax, handles):
    """Legend above the axes, then a single tight_layout.

    Call tight_layout only once: with a colorbar each pass shrinks the axes again.
    """
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, framealpha=0.0, handlelength=1.6, columnspacing=1.2,
              borderaxespad=0.0)
    fig.tight_layout()


def export(path, leg="BR", limits=None, trajectory=None, resolution=361,
           spec_limits=False, pose=(0.0, 0.0), workspace=False,
           circles=False):
    """Write ``<path>_leg`` (stick figure at ``pose``) and ``<path>_map`` (coordinate map).

    The map uses the same painter as the viewer, at a finer default resolution.
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

    # The trajectory goes on the map only.
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

    # --- Stick figure ---
    sol = obj.solve(*np.radians(pose))
    if not sol["ok"]:
        raise SystemExit(f"--pose {pose[0]:g} {pose[1]:g}: leg {leg} cannot be "
                         f"assembled there, so there is no mechanism to draw")
    # Half text width, a bit taller than HALF so the equal-aspect panels fill it
    canvas = (style.HALF[0], 2.55)
    legfig, legax = plt.subplots(figsize=canvas)
    # Heavier lines than the map; pin labels are omitted with --circles (no room).
    handles = paint_leg(legax, sol, cloud if workspace else None, scale=0.85,
                        labels=not circles, circles=circles)
    if workspace:
        handles.append(Line2D([], [], color="0.88", marker="s", ls="", ms=4,
                              label="workspace"))

    # Frame the mechanism (plus the workspace only if requested)
    pts = np.array([np.asarray(p) for p in sol["pins"].values()]
                   + [np.asarray(sol["tip"])]) * 100.0
    if workspace and len(cloud):
        pts = np.vstack([pts, cloud * 100.0])
    if circles:
        # Include the full construction circles
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

    # --- Coordinate map ---
    fig, ax = plt.subplots(figsize=canvas)
    im = paint_map(ax, Q1, Q2, margin_of(leg, Q1, Q2), reach, inbox, box,
                   hmin_mm, traj, scale=0.55)
    cbar = fig.colorbar(im, ax=ax, extend="min", pad=0.03, fraction=0.046)
    cbar.set_label(r"$h$ [mm]")
    cbar.ax.tick_params(length=2)
    ax.set_xlabel(r"$q_1$ [deg]")
    ax.set_ylabel(r"$q_2$ [deg]")
    # Labelled ticks every 180 deg, minor ticks at +-90 (too narrow for more labels)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks(range(-180, 181, 180))
        axis.set_ticks(range(-90, 91, 180), minor=True)

    # Short labels; details go in the caption.
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
