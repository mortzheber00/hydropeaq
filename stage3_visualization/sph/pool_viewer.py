#!/usr/bin/env python3
"""
Interactive 3D viewer of the pool scene over a SPlisHSPlasH run.

Reads the same VTK export as ``leg_flow_slices.py`` (``ParticleData_fluid_<k>.vtk``
and ``rb_data_<body>_<k>.vtk``) and shows the pool (floor opaque, walls
translucent), amph's base and legs, and the fluid particles coloured by speed.
Frames are read on demand, so memory stays at one frame (~1.7 M particles).

Controls:
  slider              pick an exported frame
  Left / Right        previous / next frame
  space               play / pause (``--fps``)
  x                   save the current view as ``<out-dir>/pool_view_frame<k>.pdf``
  c                   print the current camera as ``--camera ...`` (also printed on save)
  mouse               rotate / pan / zoom (standard VTK)

``--stride n`` draws every n-th particle for a faster display; ``--cut`` hides
the particles on the camera's side of the robot (y below the base centre) so the
legs under water are visible.

``--save-frame K`` writes the same PDF for frame K without opening a window, with
``--camera`` or else the default view of the whole pool, so a view set up in the
viewer can be reproduced for any frame. PDFs hold a raster image (``--scale`` x
the window size, cropped to the scene, without the time label) at the thesis
text width, with the colour bar drawn below it by matplotlib.

``--save-video mp4|gif`` renders the frames (``--every``, optionally ``--range``)
the same way, with the time next to the colour bar, at ``--fps``.

Usage:
  python stage3_visualization/sph/pool_viewer.py
  python stage3_visualization/sph/pool_viewer.py --cut --stride 2 --every 2
  python stage3_visualization/sph/pool_viewer.py --save-frame 121 --cut
  python stage3_visualization/sph/pool_viewer.py --save-frame 121 \\
      --camera -0.2 -1.6 1.1 -0.4 0.0 0.3 0.0 0.0 1.0   # values printed by 'c'
  python stage3_visualization/sph/pool_viewer.py --save-video mp4 --range 38 63
  python stage3_visualization/sph/pool_viewer.py --save-video gif --scale 1 --every 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.animation import FFMpegWriter, PillowWriter, writers
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from stage3_visualization.common.thesis_style import TEXT_WIDTH_IN  # noqa: E402
from stage3_visualization.sph.leg_flow_slices import (  # noqa: E402
    BASE_BODY,
    EXPORT_FPS,
    LEG_BODIES,
    frame_numbers,
)

POOL_FLOOR = 0
POOL_WALLS = [1, 2, 3, 4]
ROBOT_BODIES = [BASE_BODY, *LEG_BODIES]
# Whole pool from the -y side and above: position, focal point, view up.
DEFAULT_CAMERA = [(0.0, -2.2, 2.6), (0.0, 0.0, 0.1), (0.0, 0.0, 1.0)]
COLORBAR_STRIP_IN = 0.75  # PDF height below the image for the colour bar, its ticks and label


def read_fluid(vtk_dir: Path, k: int, stride: int, cut: bool, base_y: float) -> pv.PolyData:
    fluid = pv.read(vtk_dir / f"ParticleData_fluid_{k}.vtk")
    pos = np.asarray(fluid.points)[::stride]
    speed = np.linalg.norm(np.asarray(fluid["velocity"])[::stride], axis=1)
    if cut:
        keep = pos[:, 1] > base_y
        pos, speed = pos[keep], speed[keep]
    poly = pv.PolyData(pos)
    poly["speed [m/s]"] = speed
    return poly


def capture(pl: pv.Plotter, scale: int, label) -> np.ndarray:
    """Screenshot at ``scale`` x the window size without the time ``label`` and scalar bar.

    Both are drawn by matplotlib in ``compose`` instead: in a scaled screenshot
    VTK renders tiles at window size, and the scalar bar's labels overlap there.
    """
    overlays = [label, *pl.scalar_bars.values()]
    for actor in overlays:
        actor.SetVisibility(False)
    try:
        return pl.screenshot(return_img=True, scale=scale)
    finally:
        for actor in overlays:
            actor.SetVisibility(True)


def crop_box(img: np.ndarray, pad: int):
    """Slices that crop the background around the scene, keeping ``pad`` pixels."""
    rows, cols = np.nonzero(np.any(img != img[0, 0], axis=2))
    return (slice(max(rows.min() - pad, 0), rows.max() + pad + 1),
            slice(max(cols.min() - pad, 0), cols.max() + pad + 1))


def compose(img: np.ndarray, speed_max: float):
    """Text-width figure with ``img`` on top and the colour bar below.

    Returns ``(fig, im, time_text)``; the time text sits left of the colour bar
    and is empty until set. Save at ``dpi = img width / TEXT_WIDTH_IN`` so the
    image is not resampled.
    """
    h, w = img.shape[:2]
    img_h = TEXT_WIDTH_IN * h / w
    fig_h = img_h + COLORBAR_STRIP_IN
    fig = Figure(figsize=(TEXT_WIDTH_IN, fig_h))
    ax = fig.add_axes([0, COLORBAR_STRIP_IN / fig_h, 1, img_h / fig_h])
    im = ax.imshow(img, interpolation="none")
    ax.set_axis_off()
    # Bar 0.12 in high near the top of the strip; ticks and label go below it.
    bar_y = (COLORBAR_STRIP_IN - 0.2) / fig_h
    cax = fig.add_axes([0.2, bar_y, 0.6, 0.12 / fig_h])
    fig.colorbar(ScalarMappable(Normalize(0.0, speed_max), cmap="viridis"), cax=cax,
                 orientation="horizontal", label="speed [m/s]")
    time_text = fig.text(0.02, bar_y + 0.06 / fig_h, "", va="center")
    return fig, im, time_text


def save_pdf(pl: pv.Plotter, out: Path, scale: int, label, speed_max: float):
    """The current view as a raster image in a text-width PDF, cropped to the scene.

    Not ``save_graphic``: its vector export (GL2PS) draws the particles as plain
    squares and would hold ~1.7 M of them.
    """
    img = capture(pl, scale, label)
    img = img[crop_box(img, 5 * scale)]
    fig, _, _ = compose(img, speed_max)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=img.shape[1] / TEXT_WIDTH_IN)
    print(f"wrote {out} ({img.shape[1]} x {img.shape[0]} px)")
    print_camera(pl)


def save_video(pl: pv.Plotter, show, frames: list[int], out: Path, scale: int, label,
               speed_max: float, fps: float):
    """All ``frames`` composed like ``save_pdf``, with the time, as mp4 or gif (by suffix).

    The crop is taken from the first frame and kept, so every video frame has the
    same size; splashes above the first frame's scene may be cut.
    """
    show(0)
    img = capture(pl, scale, label)
    box = crop_box(img, 5 * scale)
    fig, im, time_text = compose(img[box], speed_max)
    if out.suffix == ".mp4":
        # As in leg_flow_slices.py: yuv420p needs even pixel sizes, hence the padding.
        writer = FFMpegWriter(fps=fps, codec="libx264",
                              extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
                                          "-pix_fmt", "yuv420p", "-crf", "18"])
    else:
        writer = PillowWriter(fps=fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(out), dpi=img[box].shape[1] / TEXT_WIDTH_IN):
        for i, k in enumerate(frames):
            if i:
                show(i)
                img = capture(pl, scale, label)
            im.set_data(img[box])
            time_text.set_text(f"$t = {(k - 1) / EXPORT_FPS:.2f}$ s")
            writer.grab_frame()
            print(f"frame {k} ({i + 1}/{len(frames)})", flush=True)
    print(f"wrote {out}")
    print_camera(pl)


def print_camera(pl: pv.Plotter):
    """Current camera as ``--camera`` arguments (position, focal point, view up)."""
    print("--camera " + " ".join(f"{v:.4f}" for vec in pl.camera_position for v in vec))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vtk-dir", type=Path, default=Path("sph_output/vtk"))
    ap.add_argument("--out-dir", type=Path, default=Path("sph_output"), help="where PDFs are written")
    ap.add_argument("--save-frame", type=int, default=None, metavar="K",
                    help="write the view of exported frame K as PDF (no window) and exit")
    ap.add_argument("--save-video", choices=["mp4", "gif"], default=None,
                    help="render the frames (--every, --range) to <out-dir>/pool_view.<format> "
                         "(no window) and exit")
    ap.add_argument("--range", type=int, nargs=2, default=None, metavar=("K0", "K1"),
                    help="with --save-video: only exported frames K0 to K1 (inclusive)")
    ap.add_argument("--scale", type=int, default=3,
                    help="PDF / video resolution as a multiple of the window size "
                         "(use 1 for gifs, which get large)")
    ap.add_argument("--every", type=int, default=1, help="use every n-th exported frame")
    ap.add_argument("--stride", type=int, default=1, help="draw every n-th particle")
    ap.add_argument("--cut", action="store_true",
                    help="hide particles on the camera's side of the robot (y < base centre)")
    ap.add_argument("--speed-max", type=float, default=0.5, help="upper colour limit [m/s]")
    ap.add_argument("--particle-radius", type=float, default=0.0035,
                    help="drawn particle radius [m] (the SPH particles are ~6.6 mm apart)")
    ap.add_argument("--camera", type=float, nargs=9, default=None,
                    metavar=("PX", "PY", "PZ", "FX", "FY", "FZ", "UX", "UY", "UZ"),
                    help="camera position, focal point and view up for --save-frame and the "
                         "viewer's start (default: whole pool); 'c' in the viewer prints them")
    ap.add_argument("--fps", type=float, default=10.0, help="playback rate of the viewer and video")
    args = ap.parse_args()
    if args.save_frame is not None and args.save_video is not None:
        ap.error("--save-frame and --save-video are exclusive")
    if args.range is not None and args.save_video is None:
        ap.error("--range needs --save-video")
    if args.save_video == "mp4" and not writers.is_available("ffmpeg"):
        ap.error("saving an mp4 needs ffmpeg on PATH (otherwise use --save-video gif)")

    all_frames = frame_numbers(args.vtk_dir)
    if not all_frames:
        ap.error(f"no ParticleData_fluid_*.vtk in {args.vtk_dir}")
    if args.save_frame is not None:
        if args.save_frame not in all_frames:
            ap.error(f"frame {args.save_frame} is not exported "
                     f"(available: {all_frames[0]}-{all_frames[-1]})")
        frames = [args.save_frame]
    else:
        frames = all_frames[::args.every]
        if args.range is not None:
            frames = [k for k in frames if args.range[0] <= k <= args.range[1]]
            if not frames:
                ap.error(f"no exported frames in {args.range[0]}-{args.range[1]} "
                         f"(available: {all_frames[0]}-{all_frames[-1]})")

    def rb(b, k):
        # PolyData, so add_mesh draws this object itself (for an UnstructuredGrid it
        # draws a surface copy, and in-place updates would not show).
        return pv.read(args.vtk_dir / f"rb_data_{b}_{k}.vtk").extract_surface()

    k0 = frames[0]
    robot = {b: rb(b, k0) for b in ROBOT_BODIES}
    fluid = read_fluid(args.vtk_dir, k0, args.stride, args.cut, robot[BASE_BODY].center[1])

    pl = pv.Plotter(title="amph pool",
                    off_screen=args.save_frame is not None or args.save_video is not None)
    pl.add_mesh(rb(POOL_FLOOR, k0), color="lightgrey")
    for b in POOL_WALLS:  # the pool does not move, so it is read once
        pl.add_mesh(rb(b, k0), color="lightblue", opacity=0.15)
    for b, mesh in robot.items():
        # No smooth_shading: its normals would also make add_mesh draw a copy.
        pl.add_mesh(mesh, color="dimgrey" if b == BASE_BODY else "orange")
    fluid_actor = pl.add_mesh(fluid, scalars="speed [m/s]", cmap="viridis",
                              clim=(0.0, args.speed_max), render_points_as_spheres=True)

    def keep_radius(_obj, _event):
        # Point sizes are in pixels, so before every render set the one that shows
        # --particle-radius at the focal distance (nearer/farther particles are not
        # scaled). A scaled screenshot narrows the view angle, which enlarges it too.
        cam = pl.camera
        px_per_m = pl.renderer.GetSize()[1] / (
            2 * cam.GetDistance() * np.tan(np.radians(cam.GetViewAngle()) / 2))
        fluid_actor.prop.point_size = 2 * args.particle_radius * px_per_m

    pl.renderer.AddObserver("StartEvent", keep_radius)
    label = pl.add_text("", position="upper_left", font_size=10)
    state = {"i": 0, "playing": False}

    def show(i):
        i = int(round(i)) % len(frames)
        state["i"] = i
        k = frames[i]
        for b, mesh in robot.items():
            mesh.shallow_copy(rb(b, k))  # in place, so the actors keep their meshes
        fluid.shallow_copy(read_fluid(args.vtk_dir, k, args.stride, args.cut,
                                      robot[BASE_BODY].center[1]))
        label.SetText(2, f"t = {(k - 1) / EXPORT_FPS:.2f} s (frame {k})")  # 2 = upper left
        pl.render()

    def aim_camera():
        pl.camera_position = (DEFAULT_CAMERA if args.camera is None
                              else np.reshape(args.camera, (3, 3)).tolist())

    if args.save_frame is not None:
        show(0)
        aim_camera()
        save_pdf(pl, args.out_dir / f"pool_view_frame{args.save_frame}.pdf", args.scale, label,
                 args.speed_max)
        pl.close()
        return
    if args.save_video is not None:
        aim_camera()
        tag = f"_{args.range[0]}-{args.range[1]}" if args.range is not None else ""
        save_video(pl, show, frames, args.out_dir / f"pool_view{tag}.{args.save_video}",
                   args.scale, label, args.speed_max, args.fps)
        pl.close()
        return

    # Calls show(0) once on creation.
    slider = pl.add_slider_widget(show, rng=(0, len(frames) - 1), value=0, title="frame index",
                                  pointa=(0.35, 0.92), pointb=(0.95, 0.92), fmt="%.0f")

    def step(delta):
        i = (state["i"] + delta) % len(frames)
        slider.GetRepresentation().SetValue(i)
        show(i)

    def tick(_obj, _event):
        if state["playing"]:
            step(1)

    def toggle():
        state["playing"] = not state["playing"]

    def export():
        slider.Off()  # keep the controls out of the image
        pl.hide_axes()
        save_pdf(pl, args.out_dir / f"pool_view_frame{frames[state['i']]}.pdf", args.scale, label,
                 args.speed_max)
        slider.On()
        pl.show_axes()

    pl.add_key_event("Right", lambda: step(1))
    pl.add_key_event("Left", lambda: step(-1))
    pl.add_key_event("space", toggle)
    # Not "p": VTK's default key handler also picks the mesh under the mouse on "p"
    # and outlines it in red.
    pl.add_key_event("x", export)
    pl.add_key_event("c", lambda: print_camera(pl))
    pl.add_axes()
    aim_camera()
    # Not pl.add_timer_event: PyVista 0.44 runs all its max_steps inside the first
    # timer event, which blocks the window. One step per tick instead.
    pl.iren.add_observer("TimerEvent", tick)
    pl.iren.create_timer(int(1000 / args.fps), repeating=True)
    pl.show()


if __name__ == "__main__":
    main()
