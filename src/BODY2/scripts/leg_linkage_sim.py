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
passive joint values for driving a simulator.
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
    """Per-link translation from the link frame to its mesh frame."""
    out = {}
    for link in ET.parse(URDF).getroot().findall("link"):
        o = link.find("visual/origin")
        if o is None:
            out[link.get("name")] = np.zeros(3)
            continue
        if any(abs(float(v)) > 1e-12 for v in o.get("rpy", "0 0 0").split()):
            raise RuntimeError(f"{link.get('name')}: rotated visual origin is not supported")
        out[link.get("name")] = np.array([float(v) for v in o.get("xyz").split()])
    return out


JOINTS = _joints()
VISUAL = _visual_offsets()


def _link_frames(leg):
    """World pose of every link of one leg at the URDF zero configuration."""
    T = {"base_link": (np.eye(3), np.zeros(3))}
    for name in (f"Joint_{leg}{c}.{k}" for c in (1, 2) for k in (1, 2, 3)):
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
        tv = VISUAL[f"Link_{leg}{k}"]
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
                j = JOINTS[f"Joint_{name}{c}.{k}"]
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
                tv = VISUAL[f"Link_{name}{key}"]
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

        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
        return dict(
            ok=ok1 & ok2,
            pins=dict(P1=P1, P2=P2, P3=P3, P4=P4, P5=P5, P6=P6, P7=P7, P8=P8),
            theta={"1.1": th1, "1.2": th12, "1.3": th13,
                   "2.1": th2, "2.2": th22, "2.3": th23},
            tip=tip,
            # values for the joints the URDF already declares
            joints={f"Joint_{self.name}1.2": wrap(self.sgn["1.2"] * (th12 - th1)),
                    f"Joint_{self.name}1.3": wrap(self.sgn["1.3"] * (th13 - th12)),
                    f"Joint_{self.name}2.2": wrap(self.sgn["2.2"] * (th22 - th2)),
                    f"Joint_{self.name}2.3": wrap(self.sgn["2.3"] * (th23 - th22))},
            # the two pins that have no URDF joint at all
            closures={"P6 (2.1<->1.3)": wrap(self.sgn["1.1"] * (th13 - th2)),
                      "P8 (1.3<->2.3)": wrap(self.sgn["1.1"] * (th23 - th13))})

    def place(self, key, sol):
        """Mesh of link ``key`` moved into the pose held in ``sol``."""
        anchor = {"1.1": "P1", "1.2": "P2", "1.3": "P3",
                  "2.1": "P4", "2.2": "P5", "2.3": "P7"}[key]
        a0, a = self.pins0[anchor], sol["pins"][anchor]
        return a + _rot2(self.mesh[key] - a0, sol["theta"][key])


# ---------------------------------------------------------------- viewer

def main():
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.widgets import RadioButtons, Slider

    legs = {n: Leg(n) for n in LEGS}
    colours = {"1.1": "#d62728", "1.2": "#ff7f0e", "1.3": "#8c564b",
               "2.1": "#1f77b4", "2.2": "#2ca02c", "2.3": "#9467bd"}
    grid = np.linspace(-np.pi, np.pi, 181)

    fig = plt.figure(figsize=(11, 8))
    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.16, top=0.94)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]  (forward)")
    ax.set_ylabel("z [m]  (up)")

    cloud = ax.plot([], [], ".", ms=1.2, color="0.75", zorder=0,
                    label="foot-tip workspace")[0]
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

    s1 = Slider(fig.add_axes([0.32, 0.07, 0.6, 0.03]), "Joint x1.1 [deg]", -180, 180, valinit=0)
    s2 = Slider(fig.add_axes([0.32, 0.02, 0.6, 0.03]), "Joint x2.1 [deg]", -180, 180, valinit=0)
    radio = RadioButtons(fig.add_axes([0.02, 0.80, 0.12, 0.16]), LEGS, active=LEGS.index("BR"))
    state = {"leg": "BR"}

    def workspace(leg):
        """Foot-tip positions reachable from the zero pose without taking the leg apart.

        Assemblable (q1, q2) cells split into several islands; only the one holding
        the zero pose can be driven to, so the rest are flooded away.  The hips are
        continuous joints, hence the wrap-around neighbourhood.
        """
        Q1, Q2 = np.meshgrid(grid, grid)
        sol = leg.solve(Q1, Q2)
        ok = sol["ok"] & np.isfinite(sol["tip"][..., 0])
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
        return sol["tip"][reach]

    def draw(_=None):
        leg = legs[state["leg"]]
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
        w = workspace(legs[label])
        cloud.set_data(w[:, 0], w[:, 1])
        ax.set_title(f"BODY2 leg {label} - closed-loop kinematics", weight="bold")
        draw()
        m = 0.02
        ax.set_xlim(w[:, 0].min() - m, w[:, 0].max() + m)
        ax.set_ylim(w[:, 1].min() - m, w[:, 1].max() + m)
        fig.canvas.draw_idle()

    s1.on_changed(draw)
    s2.on_changed(draw)
    radio.on_clicked(pick)
    ax.legend(loc="upper right", fontsize=8)
    pick("BR")
    plt.show()


if __name__ == "__main__":
    main()
