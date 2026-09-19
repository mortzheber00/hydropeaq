#!/usr/bin/env python3
"""Flow slices through amph's legs from a SPlisHSPlasH VTK export.

Particle velocities are SPH-interpolated onto a plane that moves with the robot
and drawn as in-plane speed or normal vorticity with streamlines and the leg
cross-sections on top.

  --view side  one panel per body side, in the sagittal plane through that
               side's two legs
  --view top   one horizontal body-frame plane per --depths value

Output is an mp4 (needs ffmpeg), single stills (--save-frames), or one gait
cycle as 8 snapshots (--cycle). Colour limits default to the 99th percentile
over all loaded frames and are printed so stills can reuse the video's scale.

Usage:
  python stage3_visualization/sph/leg_flow_slices.py --show
  python stage3_visualization/sph/leg_flow_slices.py --res 0.002 --field vorticity
  python stage3_visualization/sph/leg_flow_slices.py --res 0.002 --field vorticity --cycle 1.49 2.49 --side Right
  python stage3_visualization/sph/leg_flow_slices.py --view top --depths -0.105 -0.125 -0.145 --save-frames 61
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import resource
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button, Slider
from scipy.ndimage import binary_dilation, distance_transform_edt

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from stage3_visualization.common.thesis_style import TEXT_WIDTH_IN, full_width  # noqa: E402  activates the shared style

full_width()

# Body indices follow the plugin's export order for amph/worlds/swimming_pool.world:
# 0-4 pool, 5 base, then Side/Thigh/Calf/Foot per leg (FL, FR, HL, HR).
BASE_BODY = 5
# The Foot bodies (9, 13, 17, 21) duplicate part of the calf and are skipped.
SIDE_BODIES = {
    "Left": [*range(6, 9), *range(14, 17)],    # Front Left, Hind Left
    "Right": [*range(10, 13), *range(18, 21)],  # Front Right, Hind Right
}
LEG_BODIES = sorted(b for bodies in SIDE_BODIES.values() for b in bodies)
# Hip link of each leg; anchors the leg label.
LEG_LABELS = {6: "FL", 10: "FR", 14: "HL", 18: "HR"}
EXPORT_FPS = 25.0  # of simulation time
# Box that is pure bulk water in the first frame; reference for particle spacing and kernel sum.
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
    """Evenly spaced in-plane streamlines as ``(lines, (xy, dxy))``, one arrow per line.

    ``grid`` holds the in-plane velocity with air/robot zeroed, so lines stop at
    the free surface and the legs. ``spacing`` is the line distance [m].

    VTK 9.3's ``vtkEvenlySpacedStreamlines2D`` segfaults on some inputs, so each
    attempt runs in a forked child and a crash is retried with slightly
    perturbed parameters.
    """
    empty = [], (np.empty((0, 2)), np.empty((0, 2)))
    if not fluid_pt.any():
        return empty
    attempts = [(1.0, r) for r in (0.5, 0.45, 0.55, 0.4)] + [(s, 0.5) for s in (1.05, 0.95, 1.1, 0.9)]
    for scale, ratio in attempts:
        result = _in_child(_trace_streamlines, grid, fluid_pt, spacing * scale, ratio)
        if result is not None:
            return result
    print("  WARNING: VTK streamline filter crashed on every retry; panel drawn without streamlines")
    return empty


def _in_child(fn, *args):
    """``fn(*args)`` in a forked process; ``None`` if the child crashed."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child must never return into the caller's code
        status = 1
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # crashes are expected; skip core dumps
            with os.fdopen(write_fd, "wb") as f:
                pickle.dump(fn(*args), f)
            status = 0
        finally:
            os._exit(status)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as f:
        payload = f.read()
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
        return pickle.loads(payload)
    return None


def _trace_streamlines(grid: pv.ImageData, fluid_pt: np.ndarray, spacing: float, ratio: float):
    # VTK grows all lines from a single seed, so it cannot cross gaps narrower than
    # the line spacing (e.g. under a foot). Reseed repeatedly at the water point
    # farthest from both the boundary and existing lines until the water is covered;
    # zeroing the velocity near existing lines keeps new lines from crossing them.
    MAX_PASSES = 20
    nx, nz = grid.dimensions[:2]
    (ox, oy, _), res = grid.origin, grid.spacing[0]
    # Distances in grid cells; VTK's separating distance is in cell diagonals.
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
                                               separating_distance=sep_diag,
                                               separating_distance_ratio=ratio, compute_vorticity=False)
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
    """Smoothing length, kernel support and bulk-water kernel sum from the first frame.

    The default support is 3x the particle spacing. Grid points with a kernel sum
    below half of ``w_bulk`` are treated as air or robot.
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
    """One plane per panel.

    Each plane has an origin, in-plane axes ``e1`` (plot right) and ``e2`` (plot
    up), normal ``n``, the bodies to outline and, for the top view, the plot-x
    positions of the side-view planes.
    """
    if view == "side":
        # Normal is the base's lateral axis so the plane follows the legs' plane of
        # motion; plot-up is world up projected into it so the free surface stays level.
        lat = rot[:, 1]
        up = np.array([0.0, 0.0, 1.0]) - lat[2] * lat
        up /= np.linalg.norm(up)
        fwd = np.cross(lat, up)
        return [dict(origin=surface_centroid([meshes[b] for b in bodies]), e1=fwd, e2=up,
                     n=-lat, bodies=bodies, guides=[])
                for bodies in SIDE_BODIES.values()]
    # Top view: seen from above, heading up.
    e1, e2, n = -rot[:, 1], rot[:, 0], rot[:, 2]
    guides = [float((surface_centroid([meshes[b] for b in bodies]) - base_centre) @ e1)
              for bodies in SIDE_BODIES.values()]
    return [dict(origin=base_centre + d * n, e1=e1, e2=e2, n=n, bodies=LEG_BODIES, guides=guides)
            for d in depths]


def collect(vtk_dir: Path, frames: list[int], view: str, depths: list[float],
            extent: tuple[float, float, float, float], res: float,
            support: float | None, stream_spacing: float) -> tuple:
    """Interpolated fields, streamlines and outlines per panel and frame.

    ``extent = (x0, x1, y0, y1)`` is relative to the plane origin. Returns
    ``(data, support)``.
    """
    first = frame_numbers(vtk_dir)[0]  # reference pose is the first export, not frames[0]
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
            # Derivatives next to blank cells difference against zeros; drop them.
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
                "labels": [(LEG_LABELS[b], to_plane(surface_centroid([meshes[b]])))
                           for b in pl["bodies"] if b in LEG_LABELS],
            })
        print(f"frame {k} ({i + 1}/{len(frames)})", flush=True)
    return data, cut


def animate(data: dict, frames: list[int], view_name: str, extent: tuple, limits: dict, field: str,
            scale: float = 1.0):
    """Build the animation figure; returns ``(fig, update, set_field, view)``.

    ``scale`` enlarges the canvas (not the fonts) for the interactive player.
    """
    names = list(data)
    nz, nx = data[names[0]][0]["speed"].shape
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])  # panel height / width
    if view_name == "side":
        # ~4.9 in axes width next to the colour bar, ~0.55 in per panel for title and label.
        height = len(names) * (4.9 * aspect + 0.55)
        fig, axes = plt.subplots(len(names), 1, figsize=(scale * TEXT_WIDTH_IN, scale * height),
                                 sharex=True, constrained_layout=True, squeeze=False)
        axes = axes[:, 0]
        titles = [f"{s} side (front and hind {s.lower()} leg)" for s in names]
        axes[-1].set_xlabel("forward from centroid of both legs [m]")
        for ax in axes:
            ax.set_ylabel("up [m]")
    else:
        panel_w = (TEXT_WIDTH_IN - 1.4) / len(names)  # minus colour bar and y label
        height = min(panel_w * aspect, 6.0) + 1.0     # plus title and x label
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
        # Top view only: traces of the side-view planes.
        guides = [ax.axvline(0.0, color="0.3", lw=0.8, ls="--", zorder=1, visible=False)
                  for _ in SIDE_BODIES]
        artists.append((im, stream_lc, base_lc, legs_lc, guides))
    cbar = fig.colorbar(artists[0][0], ax=axes, shrink=0.8)
    title = fig.suptitle("")
    heads = [None] * len(names)  # quivers are recreated each frame since the arrow count varies
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
            # Tiny unit arrows at each line's midpoint, so only the head shows.
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


def cycle_frames(start: float, end: float, available: list[int]) -> list[tuple[Fraction, int]]:
    """Nearest exported frame for each phase k/8 of the cycle ``[start, end)``."""
    period = end - start
    picks = []
    print(f"cycle T = {period:.3f} s, one picture every T/8 = {period / 8 * 1e3:.0f} ms "
          f"(exports are {1e3 / EXPORT_FPS:.0f} ms apart):")
    for k in range(8):
        t = start + k * period / 8
        frame = int(round(t * EXPORT_FPS)) + 1
        if frame not in available:
            raise ValueError(f"t = {t:.3f} s (frame {frame}) is outside the export "
                             f"(frames {available[0]}-{available[-1]})")
        t_frame = (frame - 1) / EXPORT_FPS
        print(f"  {k}/8 T: target t = {t:.3f} s -> frame {frame} (t = {t_frame:.3f} s, "
              f"{(t_frame - t) * 1e3:+.0f} ms = {(t_frame - t) / period * 100:+.1f} % of T)")
        picks.append((Fraction(k, 8), frame))
    if len({f for _, f in picks}) < 8:
        print("  WARNING: T/8 is shorter than the export spacing, so some pictures repeat a frame")
    return picks


def phase_label(phase: Fraction) -> str:
    if phase == 0:
        return "$t = 0$"
    num = "" if phase.numerator == 1 else str(phase.numerator)
    return f"$t = {num}T/{phase.denominator}$"


def cycle_figure(panels: list[dict], phases: list[Fraction], view_name: str, extent: tuple,
                 limits: dict, field: str):
    """One gait cycle as a 2 x 4 grid of snapshots sharing axes and one colour bar."""
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    panel_w = (TEXT_WIDTH_IN - 0.7) / 4  # minus y labels
    # "compressed" avoids the gaps constrained layout leaves around fixed-aspect axes.
    fig, axes = plt.subplots(2, 4, figsize=(TEXT_WIDTH_IN, 2 * panel_w * aspect + 1.25),
                             sharex=True, sharey=True, layout="compressed")
    spec = FIELDS[field]
    label_y = extent[3] - 0.04 * (extent[3] - extent[2])
    for ax, d, phase in zip(axes.flat, panels, phases):
        im = ax.imshow(d[field], origin="lower", interpolation="bilinear", extent=extent,
                       cmap=spec["cmap"], vmin=limits[field][0], vmax=limits[field][1])
        ax.add_collection(LineCollection(d["lines"], colors=spec["stream"], linewidths=0.3, zorder=2))
        xy, dxy = d["arrows"]
        dxy = dxy / np.maximum(np.linalg.norm(dxy, axis=1, keepdims=True), 1e-12)
        ax.quiver(xy[:, 0], xy[:, 1], dxy[:, 0], dxy[:, 1], color=spec["stream"], zorder=2,
                  pivot="tip", angles="xy", scale=60, width=0.004,
                  headwidth=4, headlength=5, headaxislength=4.5)
        ax.add_collection(LineCollection(d["base"], colors="0.5", linewidths=0.5, zorder=3))
        ax.add_collection(LineCollection(d["legs"], colors="k", linewidths=0.8, zorder=3))
        for x in d["guides"]:
            ax.axvline(x, color="0.3", lw=0.5, ls="--", zorder=1)
        for text, (x, y) in d["labels"]:
            if view_name == "side":  # along the top edge, above the leg
                ax.text(x, label_y, text, ha="center", va="top", fontsize=7, zorder=4)
            else:  # at the hip, which lies inside the top-view frame
                ax.text(x, y, text, ha="center", va="center", fontsize=7, zorder=4,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.8))
        ax.text(0.97, 0.04, phase_label(phase), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.8))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=axes, location="top", shrink=0.6, aspect=40, label=spec["label"])
    if view_name == "side":
        fig.supxlabel("forward from centroid of both legs [m]", fontsize="medium")
        fig.supylabel("up [m]", fontsize="medium")
    else:
        fig.supxlabel("right [m]", fontsize="medium")
        fig.supylabel("forward from base centre [m]", fontsize="medium")
    return fig


def show_player(fig, update, set_field, view, frames: list[int], fps: float):
    """Interactive window: frame slider, previous/next, play/pause and a field toggle."""
    fig.get_layout_engine().set(rect=(0, 0.1, 1, 0.9))  # leave room for the controls
    slider = Slider(fig.add_axes([0.1, 0.04, 0.45, 0.03]), "frame", 0, len(frames) - 1,
                    valinit=0, valstep=1)
    slider.valtext.set_text(str(frames[0]))
    # Word labels: usetex renders "<" and ">" as inverted punctuation.
    b_prev = Button(fig.add_axes([0.60, 0.025, 0.07, 0.05]), "prev")
    b_play = Button(fig.add_axes([0.68, 0.025, 0.08, 0.05]), "play")
    b_next = Button(fig.add_axes([0.77, 0.025, 0.07, 0.05]), "next")
    b_field = Button(fig.add_axes([0.86, 0.025, 0.11, 0.05]), view["field"])

    def on_slide(val):
        i = int(val)
        slider.valtext.set_text(str(frames[i]))  # show the export number, not the index
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
    # Widgets stop responding once garbage-collected.
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
    ap.add_argument("--stream-spacing", type=float, default=None,
                    help="distance between streamlines [m] (default: 0.017; 0.03 for --cycle)")
    ap.add_argument("--view", choices=["side", "top"], default="side",
                    help="side: sagittal slice per side; top: horizontal body-frame slice per depth")
    ap.add_argument("--depths", type=float, nargs="+", default=[-0.125], metavar="Z",
                    help="top view: body-frame heights below (-) the base centre, one panel each [m]")
    ap.add_argument("--half-width", type=float, default=0.28,
                    help="side view: horizontal half-size of the window around the leg centroid [m]")
    ap.add_argument("--up", type=float, default=0.16,
                    help="side view: window height above the leg centroid [m]")
    ap.add_argument("--down", type=float, default=0.22,
                    help="side view: window depth below the leg centroid [m] "
                         "(the feet reach about 0.19 below it)")
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
    ap.add_argument("--cycle", type=float, nargs=2, default=None, metavar=("START", "END"),
                    help="save one gait cycle [START, END) in simulation time [s] as 8 snapshots "
                         "at t = 0, T/8, ..., 7T/8 in a 2x4 figure (START as for validate_sim.py --start)")
    ap.add_argument("--side", choices=list(SIDE_BODIES), default="Left",
                    help="side view with --cycle: which side's leg plane to show")
    ap.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf", help="format of the stills")
    ap.add_argument("--dpi", type=int, default=None,
                    help="resolution (default: 300 for stills, 250 for the video)")
    ap.add_argument("--show", action="store_true",
                    help="open an interactive player (slider, prev/next, play/pause, field) instead of saving")
    args = ap.parse_args()
    if args.cycle is not None:
        if args.show or args.save_frames is not None:
            ap.error("--cycle cannot be combined with --show or --save-frames")
        if args.view == "top" and len(args.depths) != 1:
            ap.error("--cycle with --view top needs a single --depths value")
        if args.cycle[1] <= args.cycle[0]:
            ap.error("--cycle END must be after START")
    if args.stream_spacing is None:
        args.stream_spacing = 0.03 if args.cycle is not None else 0.017
    if not args.show:
        plt.switch_backend("Agg")  # headless; also silences Qt's XDG_RUNTIME_DIR warning
        if args.save_frames is None and args.cycle is None and not writers.is_available("ffmpeg"):
            ap.error("saving the video needs ffmpeg on PATH (installed by the dev container; "
                     "otherwise `sudo apt install ffmpeg`)")

    all_frames = frame_numbers(args.vtk_dir)
    if args.cycle is not None:
        try:
            picks = cycle_frames(*args.cycle, all_frames)
        except ValueError as err:
            ap.error(str(err))
        frames = [frame for _, frame in picks]
    elif args.save_frames is not None:
        missing = sorted(set(args.save_frames) - set(all_frames))
        if missing:
            ap.error(f"frames {missing} are not exported (available: {all_frames[0]}-{all_frames[-1]})")
        frames = sorted(set(args.save_frames))
    else:
        frames = all_frames[::args.every]
    if args.view == "side":
        extent = (-args.half_width, args.half_width, -args.down, args.up)
    else:
        back, front, half_lat = args.top_extent
        extent = (-half_lat, half_lat, back, front)
    data, support = collect(args.vtk_dir, frames, args.view, args.depths, extent, args.res,
                            args.kernel_support, args.stream_spacing)
    if args.cycle is not None:  # colour limits from the shown panel only
        data = {key: panels for key, panels in data.items()
                if key == (args.side if args.view == "side" else next(iter(data)))}

    def pooled(key):
        vals = np.concatenate([d[key].ravel() for panel in data.values() for d in panel])
        vals = np.abs(vals[np.isfinite(vals)])
        return float(np.percentile(vals, 99)) if len(vals) else 1.0

    speed_max = args.speed_max if args.speed_max is not None else pooled("speed")
    vort_max = args.vorticity_max if args.vorticity_max is not None else pooled("vorticity")
    limits = {"speed": (0.0, speed_max), "vorticity": (-vort_max, vort_max)}
    print(f"SPH kernel support {support * 1e3:.1f} mm; "
          f"colour limits: --speed-max {speed_max:.3g} --vorticity-max {vort_max:.3g}")

    name = "leg_flow" if args.view == "side" else "leg_flow_top"
    if not args.show:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.cycle is not None:
        panels = next(iter(data.values()))
        fig = cycle_figure(panels, [phase for phase, _ in picks], args.view, extent, limits, args.field)
        tag = f"_{args.side.lower()}" if args.view == "side" else ""
        out = args.out_dir / f"{name}_{args.field}_cycle{tag}.{args.format}"
        fig.savefig(out, dpi=args.dpi or 300)
        print(f"wrote {out}")
        return

    fig, update, set_field, view = animate(data, frames, args.view, extent, limits, args.field,
                                           scale=1.6 if args.show else 1.0)
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
        # yuv420p for broad player support; it needs even frame sizes, hence the pad.
        writer = FFMpegWriter(fps=args.fps, codec="libx264",
                              extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
                                          "-pix_fmt", "yuv420p", "-crf", "18"])
        anim.save(out, writer=writer, dpi=args.dpi or 250)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
