#!/usr/bin/env python3
"""
Fluid slices through amph's legs over a SPlisHSPlasH run: side or top view.

Reads the VTK export of the Gazebo fluid plugin (``ParticleData_fluid_<k>.vtk``
and ``rb_data_<body>_<k>.vtk``) and shows an animation of slice planes that
move with the robot (base rotation recovered by fitting its vertices against
frame 1):

``--view side`` (default), two panels: the left side (Front Left + Hind Left)
and the right side (Front Right + Hind Right). The plane is the base link's
sagittal plane shifted to pass through the area-weighted surface centroid of
that side's two legs; front and hind leg of a side move in that plane, so it
stays through their middle while the robot rolls, pitches and turns. Axes are
relative to that centroid: vertical = world up projected into the plane (so the
free surface stays roughly level), horizontal = forward.

``--view top``, one panel per ``--depths`` value: a plane parallel to the base's
forward/lateral axes at that body-frame height below the base centre, seen from
above with the heading pointing up (robot's left on the left). It shows the
vortices shed by the sides of the foot paddles and whether the wake spreads or
drifts sideways; the dashed lines mark the two side-view planes. The default
-125 mm is mid power stroke for all four calves (they sweep between about -140
and -105 mm then and rise to about -50 mm during recovery). The plane is fixed
in the body rather than following the feet, so shed vortices stay in it.

For every frame and panel:
  - the particle velocities are interpolated onto a regular grid (``--res``) in
    that plane with VTK's SPH interpolator (quintic kernel; support
    ``--kernel-support``, default 3x the particle spacing measured from frame 1).
    Grid points whose kernel sum is below half the bulk-water value are treated
    as air or robot and left blank, which draws the free surface and the legs'
    footprint;
  - the colour is either the in-plane speed or the out-of-plane vorticity
    (PyVista ``compute_derivative``), with evenly spaced streamlines of the
    in-plane velocity on top (``--stream-spacing`` apart, independent of ``--res``);
  - the legs' and base's cross-sections with the plane are drawn in black/grey.

Vorticity is the component normal to the plane, positive counter-clockwise as
seen in the plot (in the top view: seen from above). Colour scales are fixed
over all loaded frames (99th percentile) and printed, so they can be passed back
with ``--speed-max`` / ``--vorticity-max`` to give stills of single frames the
same scale as the video.

Outputs (in ``--out-dir``, ``<name>`` = ``leg_flow`` or ``leg_flow_top``):
``<name>_<field>.mp4`` (needs ffmpeg), or with ``--save-frames`` one still per
frame, ``<name>_<field>_frame<k>.<format>``.

Body indices follow the plugin's export order for ``amph/worlds/swimming_pool.world``:
0-4 pool, 5 base, then Side/Thigh/Calf/Foot for FL (6-9), FR (10-13),
HL (14-17), HR (18-21). The Foot bodies duplicate part of the calf and are not
used. Frames are exported at 25 fps of simulation time.

Usage:
  python stage3_visualization/sph/leg_flow_slices.py --show   # slider, prev/next, play/pause, field toggle
  python stage3_visualization/sph/leg_flow_slices.py --res 0.002 --field vorticity          # video
  python stage3_visualization/sph/leg_flow_slices.py --res 0.002 --save-frames 55 61 67 \\
      --format pdf --speed-max 0.57 --vorticity-max 25                                      # thesis stills
  python stage3_visualization/sph/leg_flow_slices.py --view top --field vorticity --show
  python stage3_visualization/sph/leg_flow_slices.py --view top --depths -0.105 -0.125 -0.145 \\
      --res 0.002 --save-frames 61 --field vorticity                                        # depth stack
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, Slider
from scipy.ndimage import binary_dilation, distance_transform_edt

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from stage3_visualization.common.thesis_style import TEXT_WIDTH_IN, full_width  # noqa: E402  activates the shared style

full_width()

BASE_BODY = 5
# Side/Thigh/Calf per leg; the Foot bodies (9, 13, 17, 21) duplicate the calf and are skipped.
SIDE_BODIES = {
    "Left": [*range(6, 9), *range(14, 17)],    # Front Left, Hind Left
    "Right": [*range(10, 13), *range(18, 21)],  # Front Right, Hind Right
}
LEG_BODIES = sorted(b for bodies in SIDE_BODIES.values() for b in bodies)
EXPORT_FPS = 25.0
# Region that is entirely bulk water in the first frame (pool 2 x 1 m, filled to z = 0.25 m,
# robot still above the surface); used for the particle spacing and the kernel-sum reference.
BULK_BOX = ((-0.5, 0.5), (-0.3, 0.3), (0.05, 0.2))
FIELDS = {
    "speed": dict(cmap="viridis", label="in-plane speed [m/s]", stream="white"),
    "vorticity": dict(cmap="RdBu_r", label="vorticity (counter-clockwise +) [1/s]", stream="black"),
}


def frame_numbers(vtk_dir: Path) -> list[int]:
    nums = [int(re.search(r"_(\d+)\.vtk$", p.name).group(1))
            for p in vtk_dir.glob("ParticleData_fluid_*.vtk")]
    return sorted(nums)


def base_rotation(base_pts: np.ndarray, base_pts0: np.ndarray) -> np.ndarray:
    """Rotation of the base relative to frame 1 (Kabsch fit on the same vertices)."""
    a = base_pts0 - base_pts0.mean(0)
    b = base_pts - base_pts.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def surface_centroid(meshes: list[pv.DataSet]) -> np.ndarray:
    merged = pv.merge(meshes).compute_cell_sizes(length=False, volume=False)
    area = merged["Area"]
    return (merged.cell_centers().points * area[:, None]).sum(0) / area.sum()


def section_segments(mesh: pv.DataSet, origin, normal, to_plane) -> np.ndarray:
    """Cross-section of a mesh with the plane, as (n, 2, 2) in-plane segments."""
    cut = mesh.slice(normal=normal, origin=origin)
    if cut.n_cells == 0:
        return np.empty((0, 2, 2))
    seg = cut.points[cut.lines.reshape(-1, 3)[:, 1:]]  # slice yields 2-point lines
    return to_plane(seg.reshape(-1, 3)).reshape(-1, 2, 2)


def sph_interpolate(pos: np.ndarray, vel: np.ndarray, probe_pts: np.ndarray, h: float):
    """SPH-interpolated velocity and kernel sum at ``probe_pts``."""
    if len(pos) == 0:  # no water near the plane (e.g. robot still above the surface)
        return np.zeros((len(probe_pts), 3)), np.zeros(len(probe_pts))
    src = pv.PolyData(pos)
    src["velocity"] = vel
    kernel = vtk.vtkSPHQuinticKernel()
    kernel.SetSpatialStep(h)
    interp = vtk.vtkSPHInterpolator()
    interp.SetKernel(kernel)
    interp.SetInputData(pv.PolyData(probe_pts))
    interp.SetSourceData(src)
    interp.SetComputeShepardSum(True)
    interp.SetShepardNormalization(True)
    interp.SetNullPointsStrategyToNullValue()
    interp.SetNullValue(0.0)
    interp.Update()
    out = pv.wrap(interp.GetOutput())
    return np.asarray(out["velocity"]), np.asarray(out["Shepard Summation"])


def streamlines(grid: pv.ImageData, fluid_pt: np.ndarray, spacing: float):
    """Evenly spaced in-plane streamlines (VTK) as polylines, plus one arrow per line.

    ``grid`` holds the in-plane velocity with air/robot zeroed, so lines stop at
    the free surface and the legs. ``spacing`` is the distance between lines [m].
    Returns ``(lines, (xy, dxy))``.
    """
    if not fluid_pt.any():
        return [], (np.empty((0, 2)), np.empty((0, 2)))
    # VTK grows every further line next to an existing one from a single seed, so
    # it cannot pass gaps narrower than the line spacing (e.g. under a foot), and a
    # seed next to air or a leg ends at once. So seed repeatedly at the water point
    # deepest inside the water and furthest from the lines so far, until no water
    # is left more than ~one line spacing from a line. For each extra pass the velocity
    # near existing lines is zeroed, so new lines stop there instead of crossing.
    MAX_PASSES = 20
    nx, nz = grid.dimensions[:2]
    (ox, oy, _), res = grid.origin, grid.spacing[0]
    # In grid cells. VTK's separating distance is in cell diagonals (res * sqrt 2).
    sep_diag = spacing / (res * np.sqrt(2))
    gap, stop, min_depth = spacing / res, 0.35 * spacing / res, max(2.0, 0.5 * spacing / res)
    fluid = fluid_pt.reshape(nz, nx)
    depth = distance_transform_edt(np.pad(fluid, 1))[1:-1, 1:-1]
    work = grid.copy()
    vel0 = np.asarray(grid["velocity"])
    hit = np.zeros((nz, nx), dtype=bool)
    lines = []
    for _ in range(MAX_PASSES):
        dist = distance_transform_edt(~hit) if hit.any() else np.full((nz, nx), np.inf)
        score = np.where(fluid & (dist > gap) & (depth >= min_depth), np.minimum(depth, dist), 0)
        if score.max() == 0:
            break
        iz, ix = np.unravel_index(np.argmax(score), score.shape)
        hit[iz, ix] = True  # never reuse a seed, even if it yields no line
        work["velocity"] = vel0 * (dist.ravel() > stop)[:, None]
        sl = work.streamlines_evenly_spaced_2D(vectors="velocity", start_position=grid.points[iz * nx + ix],
                                               separating_distance=sep_diag, compute_vorticity=False)
        conn, j = sl.lines, 0
        while j < len(conn):
            n = conn[j]
            if n >= 3:
                ln = sl.points[conn[j + 1:j + 1 + n], :2].astype(np.float32)
                lines.append(ln)
                hit[np.clip(np.round((ln[:, 1] - oy) / res).astype(int), 0, nz - 1),
                    np.clip(np.round((ln[:, 0] - ox) / res).astype(int), 0, nx - 1)] = True
            j += n + 1
    mid = [ln[len(ln) // 2] for ln in lines]
    dxy = [ln[len(ln) // 2 + 1] - ln[len(ln) // 2 - 1] for ln in lines]
    return lines, (np.array(mid).reshape(-1, 2), np.array(dxy).reshape(-1, 2))


def near(pos: np.ndarray, c, axes_halfsizes) -> np.ndarray:
    """Mask of particles inside the box centred at ``c`` with the given (axis, half-size) pairs."""
    rel = pos - c
    mask = np.ones(len(pos), dtype=bool)
    for axis, half in axes_halfsizes:
        mask &= np.abs(rel @ axis) < half
    return mask


def kernel_setup(vtk_dir: Path, support: float | None) -> tuple[float, float, float]:
    """``(h, support, w_bulk)`` from the first exported frame.

    The particle spacing comes from the bulk density and sets the default support
    (3x spacing); ``w_bulk`` is the kernel sum in bulk water, and grid points below
    half of it are treated as air or robot.
    """
    first = frame_numbers(vtk_dir)[0]
    pos0 = np.asarray(pv.read(vtk_dir / f"ParticleData_fluid_{first}.vtk").points)
    lo, hi = np.array(BULK_BOX).T
    in_bulk = np.all((pos0 > lo) & (pos0 < hi), axis=1)
    particle_spacing = float((np.prod(hi - lo) / in_bulk.sum()) ** (1 / 3))
    cut = support if support is not None else 3.0 * particle_spacing
    h = cut / 3.0  # the quintic kernel's support is 3 smoothing lengths
    probe = (lo + hi) / 2 + np.random.default_rng(0).uniform(-0.05, 0.05, (200, 3))
    _, w = sph_interpolate(pos0, np.zeros_like(pos0), probe, h)
    return h, cut, float(np.median(w))


def panel_names(view: str, depths: list[float]) -> list[str]:
    return list(SIDE_BODIES) if view == "side" else [f"{d * 1e3:.0f} mm" for d in depths]


def slice_planes(view: str, rot: np.ndarray, base_centre: np.ndarray, meshes: dict,
                 depths: list[float]) -> list[dict]:
    """One plane per panel: origin, in-plane axes ``e1`` (plot right) and ``e2`` (plot up),
    normal ``n = e1 x e2``, the leg bodies to outline, and (top view) the plot-x
    positions of the side-view planes."""
    if view == "side":
        # Normal = the base's lateral axis, so the plane rolls/pitches/yaws with the
        # body and stays in the legs' plane of motion. Screen-up is world up projected
        # into that plane, so the free surface reads roughly level.
        lat = rot[:, 1]
        up = np.array([0.0, 0.0, 1.0]) - lat[2] * lat
        up /= np.linalg.norm(up)
        fwd = np.cross(lat, up)
        return [dict(origin=surface_centroid([meshes[b] for b in bodies]), e1=fwd, e2=up,
                     n=-lat, bodies=bodies, guides=[])
                for bodies in SIDE_BODIES.values()]
    # Top view: seen from above with the heading up, so plot right = robot's right.
    e1, e2, n = -rot[:, 1], rot[:, 0], rot[:, 2]
    guides = [float((surface_centroid([meshes[b] for b in bodies]) - base_centre) @ e1)
              for bodies in SIDE_BODIES.values()]
    return [dict(origin=base_centre + d * n, e1=e1, e2=e2, n=n, bodies=LEG_BODIES, guides=guides)
            for d in depths]


def collect(vtk_dir: Path, frames: list[int], view: str, depths: list[float],
            extent: tuple[float, float, float, float], res: float,
            support: float | None, stream_spacing: float) -> tuple:
    """Per panel, per frame: interpolated speed, vorticity, streamlines and outlines.

    ``extent = (x0, x1, y0, y1)`` of each panel in plot coordinates relative to
    the plane origin. Returns ``(data, support)``; ``support`` defaults to 3x the
    particle spacing.
    """
    first = frame_numbers(vtk_dir)[0]  # references must come from the first export, not frames[0]
    base0 = pv.read(vtk_dir / f"rb_data_{BASE_BODY}_{first}.vtk").points[::20]
    h, cut, w_bulk = kernel_setup(vtk_dir, support)

    x0, x1, y0, y1 = extent
    nx, nz = int(round((x1 - x0) / res)) + 1, int(round((y1 - y0) / res)) + 1
    gu, gv = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, nz))
    grid = pv.ImageData(dimensions=(nx, nz, 1), spacing=(res, res, res), origin=(x0, y0, 0.0))
    names = panel_names(view, depths)
    data = {name: [] for name in names}

    for i, k in enumerate(frames):
        fluid = pv.read(vtk_dir / f"ParticleData_fluid_{k}.vtk")
        pos, vel = np.asarray(fluid.points), np.asarray(fluid["velocity"])
        base = pv.read(vtk_dir / f"rb_data_{BASE_BODY}_{k}.vtk")
        meshes = {b: pv.read(vtk_dir / f"rb_data_{b}_{k}.vtk") for b in LEG_BODIES}
        base_pts = base.points[::20]
        planes = slice_planes(view, base_rotation(base_pts, base0), base_pts.mean(0), meshes, depths)

        for name, pl in zip(names, planes):
            c, e1, e2, n = pl["origin"], pl["e1"], pl["e2"], pl["n"]

            def to_plane(p, c=c, e1=e1, e2=e2):
                return np.stack([(p - c) @ e1, (p - c) @ e2], axis=-1)

            centre = c + (x0 + x1) / 2 * e1 + (y0 + y1) / 2 * e2
            sel = near(pos, centre, [(n, cut), (e1, (x1 - x0) / 2 + cut), (e2, (y1 - y0) / 2 + cut)])
            probe_pts = c + gu.reshape(-1, 1) * e1 + gv.reshape(-1, 1) * e2
            v, w = sph_interpolate(pos[sel], vel[sel], probe_pts, h)
            fluid_pt = w > 0.5 * w_bulk
            uw = np.stack([v @ e1, v @ e2], axis=-1) * fluid_pt[:, None]

            grid["velocity"] = np.column_stack([uw, np.zeros(len(uw))])
            vort = np.asarray(grid.compute_derivative(scalars="velocity", gradient=False,
                                                      vorticity=True)["vorticity"])[:, 2]
            lines, arrows = streamlines(grid, fluid_pt, stream_spacing)
            blank = ~fluid_pt.reshape(nz, nx)
            uw = uw.reshape(nz, nx, 2)
            uw[blank] = np.nan
            vort = vort.reshape(nz, nx)
            # The derivative next to blank cells differences against the zeroed air/robot.
            vort[binary_dilation(blank)] = np.nan

            data[name].append({
                "speed": np.linalg.norm(uw, axis=-1).astype(np.float32),
                "vorticity": vort.astype(np.float32),
                "lines": lines,
                "arrows": arrows,
                "legs": np.concatenate([section_segments(meshes[b], c, n, to_plane)
                                        for b in pl["bodies"]]),
                "base": section_segments(base, c, n, to_plane),
                "guides": pl["guides"],
            })
        print(f"frame {k} ({i + 1}/{len(frames)})", flush=True)
    return data, cut


def animate(data: dict, frames: list[int], view_name: str, extent: tuple, limits: dict, field: str,
            scale: float = 1.0):
    """Figure plus ``update(i)`` to draw frame i and ``set_field(name)`` to switch the colour.

    The canvas is the thesis text width; ``scale`` enlarges it (not the type) for
    the interactive player.
    """
    names = list(data)
    nz, nx = data[names[0]][0]["speed"].shape
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])  # panel height / width
    if view_name == "side":
        # Axes take about 4.9 in of the width next to the colour bar; ~0.55 in per
        # panel for its title and the shared x label.
        height = len(names) * (4.9 * aspect + 0.55)
        fig, axes = plt.subplots(len(names), 1, figsize=(scale * TEXT_WIDTH_IN, scale * height),
                                 sharex=True, constrained_layout=True, squeeze=False)
        axes = axes[:, 0]
        titles = [f"{s} side (front and hind {s.lower()} leg)" for s in names]
        axes[-1].set_xlabel("forward from centroid of both legs [m]")
        for ax in axes:
            ax.set_ylabel("up [m]")
    else:
        panel_w = (TEXT_WIDTH_IN - 1.4) / len(names)  # width left by the colour bar and y label
        height = min(panel_w * aspect, 6.0) + 1.0     # + title, x label and time line
        fig, axes = plt.subplots(1, len(names), figsize=(scale * TEXT_WIDTH_IN, scale * height),
                                 sharey=True, constrained_layout=True, squeeze=False)
        axes = axes[0]
        titles = [f"depth ${float(name.split()[0]):.0f}$ mm" for name in names]
        axes[0].set_ylabel("forward from base centre [m]")
        for ax in axes:
            ax.set_xlabel("right [m]")
    artists = []
    for ax, title_text in zip(axes, titles):
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_title(title_text)
        im = ax.imshow(np.full((nz, nx), np.nan), origin="lower", interpolation="bilinear",
                       extent=extent)
        stream_lc = LineCollection([], linewidths=0.45, zorder=2)
        base_lc = LineCollection([], colors="0.5", linewidths=0.8, zorder=3)
        legs_lc = LineCollection([], colors="k", linewidths=1.1, zorder=3)
        for lc in (stream_lc, base_lc, legs_lc):
            ax.add_collection(lc)
        # Top view: where the side-view planes cut this plane.
        guides = [ax.axvline(0.0, color="0.3", lw=0.8, ls="--", zorder=1, visible=False)
                  for _ in SIDE_BODIES]
        artists.append((im, stream_lc, base_lc, legs_lc, guides))
    cbar = fig.colorbar(artists[0][0], ax=axes, shrink=0.8)
    title = fig.suptitle("")
    heads = [None] * len(names)  # arrow count changes per frame, so the quivers are redrawn
    view = {"field": field, "i": 0}

    def update(i):
        view["i"] = i
        color = FIELDS[view["field"]]["stream"]
        for j, (ax, name, (im, stream_lc, base_lc, legs_lc, guides)) in enumerate(
                zip(axes, names, artists)):
            d = data[name][i]
            im.set_data(d[view["field"]])
            stream_lc.set_segments(d["lines"])
            stream_lc.set_color(color)
            base_lc.set_segments(d["base"])
            legs_lc.set_segments(d["legs"])
            for line, x in zip(guides, d["guides"]):
                line.set_xdata([x, x])
                line.set_visible(True)
            if heads[j] is not None:
                heads[j].remove()
            xy, dxy = d["arrows"]
            dxy = dxy / np.maximum(np.linalg.norm(dxy, axis=1, keepdims=True), 1e-12)
            # Short unit arrows at each streamline's midpoint: only the head is visible.
            heads[j] = ax.quiver(xy[:, 0], xy[:, 1], dxy[:, 0], dxy[:, 1], color=color, zorder=2,
                                 pivot="tip", angles="xy", scale=90, width=0.002,
                                 headwidth=4, headlength=5, headaxislength=4.5)
        t = (frames[i] - 1) / EXPORT_FPS
        title.set_text(f"$t = {t:.2f}$ s (frame {frames[i]})")

    def set_field(name):
        view["field"] = name
        for im, *_ in artists:
            im.set_cmap(FIELDS[name]["cmap"])
            im.set_clim(*limits[name])
        cbar.update_normal(artists[0][0])
        cbar.set_label(FIELDS[name]["label"])
        update(view["i"])

    set_field(field)
    return fig, update, set_field, view


def show_player(fig, update, set_field, view, frames: list[int], fps: float):
    """Interactive window: frame slider, previous/next, play/pause and a field toggle."""
    fig.get_layout_engine().set(rect=(0, 0.1, 1, 0.9))  # free a strip at the bottom
    slider = Slider(fig.add_axes([0.1, 0.04, 0.45, 0.03]), "frame", 0, len(frames) - 1,
                    valinit=0, valstep=1)
    slider.valtext.set_text(str(frames[0]))
    # Plain labels: under usetex "<" and ">" typeset as inverted punctuation.
    b_prev = Button(fig.add_axes([0.60, 0.025, 0.07, 0.05]), "prev")
    b_play = Button(fig.add_axes([0.68, 0.025, 0.08, 0.05]), "play")
    b_next = Button(fig.add_axes([0.77, 0.025, 0.07, 0.05]), "next")
    b_field = Button(fig.add_axes([0.86, 0.025, 0.11, 0.05]), view["field"])

    def on_slide(val):
        i = int(val)
        slider.valtext.set_text(str(frames[i]))  # show the exported frame number
        update(i)
        fig.canvas.draw_idle()

    def step(delta):
        slider.set_val((int(slider.val) + delta) % len(frames))

    timer = fig.canvas.new_timer(interval=int(1000 / fps))
    timer.add_callback(step, 1)
    playing = [False]

    def toggle(_event):
        playing[0] = not playing[0]
        (timer.start if playing[0] else timer.stop)()
        b_play.label.set_text("pause" if playing[0] else "play")
        fig.canvas.draw_idle()

    def switch_field(_event):
        names = list(FIELDS)
        name = names[(names.index(view["field"]) + 1) % len(names)]
        set_field(name)
        b_field.label.set_text(name)
        fig.canvas.draw_idle()

    slider.on_changed(on_slide)
    b_prev.on_clicked(lambda _e: step(-1))
    b_next.on_clicked(lambda _e: step(1))
    b_play.on_clicked(toggle)
    b_field.on_clicked(switch_field)
    # Keep the widgets referenced while the window is open, or they stop responding.
    fig._player = (slider, b_prev, b_play, b_next, b_field, timer)
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vtk-dir", type=Path, default=Path("sph_output/vtk"))
    ap.add_argument("--out-dir", type=Path, default=Path("sph_output"),
                    help="where the video / stills are written")
    ap.add_argument("--field", choices=list(FIELDS), default="speed",
                    help="colour of the saved video or stills / initial colour of the player")
    ap.add_argument("--res", type=float, default=0.004,
                    help="interpolation grid spacing [m]; 0.002 for final figures")
    ap.add_argument("--kernel-support", type=float, default=None,
                    help="SPH kernel support radius [m] (default: 3x particle spacing; "
                         "SPlisHSPlasH itself uses 4x particle radius)")
    ap.add_argument("--stream-spacing", type=float, default=0.017,
                    help="distance between streamlines [m]")
    ap.add_argument("--view", choices=["side", "top"], default="side",
                    help="side: sagittal slice per side; top: horizontal body-frame slice per depth")
    ap.add_argument("--depths", type=float, nargs="+", default=[-0.125], metavar="Z",
                    help="top view: body-frame heights below (-) the base centre, one panel each [m]")
    ap.add_argument("--half-width", type=float, default=0.28,
                    help="side view: horizontal half-size of the window around the leg centroid [m]")
    ap.add_argument("--half-height", type=float, default=0.16,
                    help="side view: vertical half-size of the window around the leg centroid [m]")
    ap.add_argument("--top-extent", type=float, nargs=3, default=[-0.45, 0.30, 0.25],
                    metavar=("BACK", "FRONT", "HALF_LAT"),
                    help="top view: forward range and lateral half-size around the base centre [m]")
    ap.add_argument("--every", type=int, default=1, help="use every n-th exported frame")
    ap.add_argument("--fps", type=float, default=10.0, help="playback rate of the player and video")
    ap.add_argument("--speed-max", type=float, default=None,
                    help="fixed upper colour limit for speed [m/s] (default: 99th percentile)")
    ap.add_argument("--vorticity-max", type=float, default=None,
                    help="fixed +/- colour limit for vorticity [1/s] (default: 99th percentile)")
    ap.add_argument("--save-frames", type=int, nargs="+", default=None, metavar="K",
                    help="save stills of these exported frame numbers instead of a video")
    ap.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf", help="format of the stills")
    ap.add_argument("--dpi", type=int, default=None,
                    help="resolution (default: 300 for stills, 250 for the video)")
    ap.add_argument("--show", action="store_true",
                    help="open an interactive player (slider, prev/next, play/pause, field) instead of saving")
    args = ap.parse_args()
    if not args.show:
        plt.switch_backend("Agg")  # no Qt window needed (avoids Qt's XDG_RUNTIME_DIR warning)
        if args.save_frames is None and not writers.is_available("ffmpeg"):
            ap.error("saving the video needs ffmpeg on PATH (installed by the dev container; "
                     "otherwise `sudo apt install ffmpeg`)")

    all_frames = frame_numbers(args.vtk_dir)
    if args.save_frames is not None:
        missing = sorted(set(args.save_frames) - set(all_frames))
        if missing:
            ap.error(f"frames {missing} are not exported (available: {all_frames[0]}-{all_frames[-1]})")
        frames = sorted(set(args.save_frames))
    else:
        frames = all_frames[::args.every]
    if args.view == "side":
        extent = (-args.half_width, args.half_width, -args.half_height, args.half_height)
    else:
        back, front, half_lat = args.top_extent
        extent = (-half_lat, half_lat, back, front)
    data, support = collect(args.vtk_dir, frames, args.view, args.depths, extent, args.res,
                            args.kernel_support, args.stream_spacing)

    def pooled(key):
        vals = np.concatenate([d[key].ravel() for panel in data.values() for d in panel])
        vals = np.abs(vals[np.isfinite(vals)])
        return float(np.percentile(vals, 99)) if len(vals) else 1.0

    speed_max = args.speed_max if args.speed_max is not None else pooled("speed")
    vort_max = args.vorticity_max if args.vorticity_max is not None else pooled("vorticity")
    limits = {"speed": (0.0, speed_max), "vorticity": (-vort_max, vort_max)}
    print(f"SPH kernel support {support * 1e3:.1f} mm; "
          f"colour limits: --speed-max {speed_max:.3g} --vorticity-max {vort_max:.3g}")

    fig, update, set_field, view = animate(data, frames, args.view, extent, limits, args.field,
                                           scale=1.6 if args.show else 1.0)
    name = "leg_flow" if args.view == "side" else "leg_flow_top"
    if not args.show:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.show:
        show_player(fig, update, set_field, view, frames, args.fps)
    elif args.save_frames is not None:
        for i, k in enumerate(frames):
            update(i)
            out = args.out_dir / f"{name}_{args.field}_frame{k}.{args.format}"
            fig.savefig(out, dpi=args.dpi or 300)
            print(f"wrote {out}")
    else:
        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)
        out = args.out_dir / f"{name}_{args.field}.mp4"
        # H.264 at near-lossless quality; yuv420p keeps it playable everywhere. It needs
        # even pixel sizes, which the text-width canvas does not give at every dpi.
        writer = FFMpegWriter(fps=args.fps, codec="libx264",
                              extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
                                          "-pix_fmt", "yuv420p", "-crf", "18"])
        anim.save(out, writer=writer, dpi=args.dpi or 250)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
