"""
Debug: visualize all robot link meshes together in their world poses.
Applies the same pinocchio FK + Z-up→Y-up transform used in 03_generate_scene.py.
Exports a single combined OBJ so you can inspect in any viewer (MeshLab, Blender…).
"""
import numpy as np
import pinocchio as pin
import trimesh
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
URDF_PATH = REPO_ROOT / "src" / "amph" / "urdf" / "amph.urdf"
MESH_DIR  = REPO_ROOT / "stage2_gait_validation" / "sph_meshes"
OUT_OBJ   = REPO_ROOT / "stage2_gait_validation" / "debug_robot_combined.obj"

R_ZUP_TO_YUP = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
], dtype=float)


def build_transform_4x4(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def main():
    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    data  = model.createData()

    # Neutral pose, base at origin (no water offset — we just want to see the shape)
    q0 = pin.neutral(model).copy()
    q0[3:7] = [0, 0, 0, 1]
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    available_meshes = {f.stem: f for f in MESH_DIR.iterdir() if f.suffix == '.obj'}

    scene     = trimesh.scene.Scene()
    no_mesh   = []

    for fid, frame in enumerate(model.frames):
        if frame.type != pin.FrameType.BODY:
            continue
        link_name = frame.name
        if link_name not in available_meshes:
            no_mesh.append(link_name)
            continue

        T   = data.oMf[fid]
        R   = R_ZUP_TO_YUP @ T.rotation          # same as scene generator
        pos = R_ZUP_TO_YUP @ T.translation

        mesh = trimesh.load(str(available_meshes[link_name]), force='mesh')
        mesh.apply_transform(build_transform_4x4(R, pos))

        scene.add_geometry(mesh, node_name=link_name)
        print(f"  {link_name:35s}  y={pos[1]:.4f}")

    if no_mesh:
        print(f"\nNo mesh for: {no_mesh}")

    # Export combined OBJ
    combined = trimesh.util.concatenate(list(scene.geometry.values()))
    combined.export(str(OUT_OBJ))
    print(f"\nCombined mesh saved to: {OUT_OBJ}")
    print("Open it in MeshLab or Blender to inspect link alignment.")

    # Also try to open the trimesh viewer (works if a display is available)
    try:
        scene.show()
    except Exception as e:
        print(f"(Viewer not available: {e})")


if __name__ == '__main__':
    main()
