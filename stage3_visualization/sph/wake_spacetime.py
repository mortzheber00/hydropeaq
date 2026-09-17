#!/usr/bin/env python3
"""Space-time diagram of the wake between amph's front and hind legs (SPH run).

For each body side, the flow is sampled every frame along a body-frame line
through both legs (in the side's leg plane, at a fixed depth) and averaged over
a thin band. x = position along the body, y = time, colour = backward flow
-u_x or vorticity. Black lines are the front and hind calf positions, thick
during the power stroke. Streaks running from the front track towards the hind
track show the wake travelling back. Blank cells are air or robot.

Flow is in the pool frame unless --relative (flow seen from the robot).

Usage:
  python stage3_visualization/sph/wake_spacetime.py
  python stage3_visualization/sph/wake_spacetime.py --field vorticity --save wake_vorticity.pdf
  python stage3_visualization/sph/wake_spacetime.py --relative --depth -0.10 --band 0.02
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.ndimage import binary_dilation

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from stage3_visualization.sph.leg_flow_slices import (  # noqa: E402
    BASE_BODY,
    EXPORT_FPS,
    SIDE_BODIES,
    base_rotation,
    frame_numbers,
    kernel_setup,
    near,
    sph_interpolate,
    surface_centroid,
)
from stage3_visualization.common.thesis_style import TEXT_WIDTH_IN, full_width  # noqa: E402  activates the shared style

full_width()

# Calf bodies (including the foot paddle) of the (front, hind) leg per side
CALVES = {"Left": (8, 16), "Right": (12, 20)}
FIELDS = {
    "backward": "backward flow $-u_x$ [m/s]",
    "vorticity": "vorticity (counter-clockwise +) [1/s]",
}
# Distinguish tracks by dash; red/blue leg colours would vanish in the colormap.
TRACK_STYLES = {"front": "-", "hind": (0, (3, 1.5))}
POWER_SPEED = 0.05  # [m/s] backward calf speed (body frame) that counts as power stroke


def body_tracks(vtk_dir: Path, frames: list[int], times: np.ndarray) -> dict:
    """Base pose, base velocity, leg-plane offsets and calf centroids in the body frame."""
    base0 = pv.read(vtk_dir / f"rb_data_{BASE_BODY}_{frame_numbers(vtk_dir)[0]}.vtk").points[::20]
    rots, origins = [], []
    lateral = {side: np.zeros(len(frames)) for side in SIDE_BODIES}
    calf = {side: np.zeros((len(frames), 2, 3)) for side in SIDE_BODIES}  # [frame, front/hind, xyz]
    for i, k in enumerate(frames):
        bp = pv.read(vtk_dir / f"rb_data_{BASE_BODY}_{k}.vtk").points[::20]
        r, o = base_rotation(bp, base0), bp.mean(0)  # the vertex mean is a body-fixed point
        rots.append(r)
        origins.append(o)
        for side, bodies in SIDE_BODIES.items():
            meshes = [pv.read(vtk_dir / f"rb_data_{b}_{k}.vtk") for b in bodies]
            lateral[side][i] = (surface_centroid(meshes) - o) @ r[:, 1]
            for j, b in enumerate(CALVES[side]):
                calf[side][i, j] = (surface_centroid([meshes[bodies.index(b)]]) - o) @ r
    rots, origins = np.array(rots), np.array(origins)
    base_vel = np.gradient(origins, times, axis=0)
    return {
        "R": rots,
        "o": origins,
        "vx": np.einsum("ij,ij->i", base_vel, rots[:, :, 0]),  # base velocity along body x
        "lateral": lateral,
        "calf": calf,
    }


def power_stroke_depth(calf: np.ndarray, times: np.ndarray) -> float:
    """Median body-frame calf height during the power stroke (where vortices are shed)."""
    z, vx = calf[:, :, 2], np.gradient(calf[:, :, 0], times, axis=0)
    power = vx < -POWER_SPEED
    return float(np.median(z[power] if power.any() else z))


def sample(vtk_dir: Path, frames: list[int], tracks: dict, xs: np.ndarray, depth: dict,
           band: float, res: float, support: float | None, relative: bool) -> dict:
    """Band-averaged backward flow and vorticity along the body line, per side and frame."""
    h, cut, w_bulk = kernel_setup(vtk_dir, support)
    zs = np.arange(-band, band + 1e-9, res)
    gx, gz = np.meshgrid(xs, zs)
    grid = pv.ImageData(dimensions=(len(xs), len(zs), 1), spacing=(res, res, res),
                        origin=(xs[0], zs[0], 0.0))
    half_len = (xs[-1] - xs[0]) / 2
    out = {side: {f: np.full((len(frames), len(xs)), np.nan) for f in FIELDS} for side in SIDE_BODIES}

    for i, k in enumerate(frames):
        fluid = pv.read(vtk_dir / f"ParticleData_fluid_{k}.vtk")
        pos, vel = np.asarray(fluid.points), np.asarray(fluid["velocity"])
        o = tracks["o"][i]
        ex, ey, ez = tracks["R"][i].T  # body forward, lateral (left), up

        for side in SIDE_BODIES:
            y, z0 = tracks["lateral"][side][i], depth[side]
            centre = o + (xs[0] + half_len) * ex + y * ey + z0 * ez
            sel = near(pos, centre, [(ey, cut), (ex, half_len + cut), (ez, band + cut)])
            probe = o + gx.reshape(-1, 1) * ex + y * ey + (z0 + gz.reshape(-1, 1)) * ez
            v, w = sph_interpolate(pos[sel], vel[sel], probe, h)
            wet = w > 0.5 * w_bulk
            u, uz = (v @ ex) * wet, (v @ ez) * wet

            grid["velocity"] = np.column_stack([u, uz, np.zeros_like(u)])
            vort = np.asarray(grid.compute_derivative(scalars="velocity", gradient=False,
                                                      vorticity=True)["vorticity"])[:, 2]
            dry = ~wet.reshape(gz.shape)
            vort = vort.reshape(gz.shape)
            vort[binary_dilation(dry)] = np.nan  # invalid derivatives next to dry cells
            backward = -(u.reshape(gz.shape) - (tracks["vx"][i] if relative else 0.0))
            backward[dry] = np.nan

            with warnings.catch_warnings():  # columns fully out of the water stay NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                out[side]["backward"][i] = np.nanmean(backward, axis=0)
                out[side]["vorticity"][i] = np.nanmean(vort, axis=0)
        print(f"frame {k} ({i + 1}/{len(frames)})", flush=True)
    return out


def plot(out: dict, tracks: dict, times: np.ndarray, xs: np.ndarray, field: str, vmax: float,
         depth: dict, band: float, relative: bool):
    """One space-time panel per side with calf tracks."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 4.8), sharey=True, constrained_layout=True)
    dx, dt = xs[1] - xs[0], (times[1] - times[0]) if len(times) > 1 else 1 / EXPORT_FPS
    for ax, side in zip(axes, SIDE_BODIES):
        im = ax.imshow(out[side][field], origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest",
                       extent=(xs[0] - dx / 2, xs[-1] + dx / 2, times[0] - dt / 2, times[-1] + dt / 2))
        for j, leg in enumerate(TRACK_STYLES):
            x = tracks["calf"][side][:, j, 0]
            pts = np.column_stack([x, times])
            # Threshold ignores jitter of a resting leg
            power = np.diff(x) / np.diff(times) < -POWER_SPEED
            ax.add_collection(LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), colors="black",
                                             linestyles=TRACK_STYLES[leg],
                                             linewidths=np.where(power, 2.0, 0.7)))
        ax.set_xlim(xs[0], xs[-1])
        ax.set_title(f"{side} side (depth ${depth[side] * 1e3:.0f}$ mm)")
        ax.set_xlabel("forward from base centre [m]")
    axes[0].set_ylabel("time [s]")
    frame = "relative to the robot" if relative else "pool frame"
    fig.colorbar(im, ax=axes, shrink=0.8, label=f"{FIELDS[field]}, {frame}")
    handles = [Line2D([], [], color="black", ls=ls, lw=lw, label=f"{leg} calf, {phase}")
               for leg, ls in TRACK_STYLES.items() for lw, phase in ((2.0, "power stroke"), (0.7, "recovery"))]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vtk-dir", type=Path, default=Path("sph_output/vtk"))
    ap.add_argument("--field", choices=list(FIELDS), default="backward")
    ap.add_argument("--relative", action="store_true",
                    help="subtract the base velocity along the body (flow as seen from the robot)")
    ap.add_argument("--depth", type=float, default=None,
                    help="body-frame height of the line below/above the base centre [m] "
                         "(default: median calf centroid height of that side during power strokes)")
    ap.add_argument("--band", type=float, default=0.015,
                    help="half-height of the band averaged around the line [m]")
    ap.add_argument("--x-range", type=float, nargs=2, default=None, metavar=("X0", "X1"),
                    help="body-x extent of the line [m] (default: hind calf - 0.12 to front calf + 0.08)")
    ap.add_argument("--res", type=float, default=0.004, help="sampling spacing along the line [m]")
    ap.add_argument("--kernel-support", type=float, default=None,
                    help="SPH kernel support radius [m] (default: 3x particle spacing)")
    ap.add_argument("--every", type=int, default=1, help="use every n-th exported frame")
    ap.add_argument("--vmax", type=float, default=None,
                    help="symmetric colour limit (default: 99th percentile of |value|)")
    ap.add_argument("--save", type=Path, default=None, help="write the figure instead of showing it")
    args = ap.parse_args()
    if args.save is not None:
        plt.switch_backend("Agg")

    frames = frame_numbers(args.vtk_dir)[::args.every]
    times = (np.array(frames) - 1) / EXPORT_FPS
    tracks = body_tracks(args.vtk_dir, frames, times)

    all_calf = np.concatenate([c for c in tracks["calf"].values()])
    if args.x_range is None:
        x0, x1 = all_calf[:, 1, 0].min() - 0.12, all_calf[:, 0, 0].max() + 0.08
    else:
        x0, x1 = args.x_range
    xs = np.arange(x0, x1 + 1e-9, args.res)
    depth = {side: args.depth if args.depth is not None else power_stroke_depth(tracks["calf"][side], times)
             for side in SIDE_BODIES}

    out = sample(args.vtk_dir, frames, tracks, xs, depth, args.band, args.res,
                 args.kernel_support, args.relative)
    if args.vmax is not None:
        vmax = args.vmax
    else:
        vals = np.abs(np.concatenate([out[s][args.field].ravel() for s in SIDE_BODIES]))
        vals = vals[np.isfinite(vals)]
        vmax = float(np.percentile(vals, 99)) if len(vals) else 1.0
    print(f"colour limit: --vmax {vmax:.3g}")

    fig = plot(out, tracks, times, xs, args.field, vmax, depth, args.band, args.relative)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=300)
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
