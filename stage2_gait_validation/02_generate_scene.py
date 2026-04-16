"""
Generate SPlisHSPlasH scene.json from the OCP solution and prepared meshes.

Coordinate system: SPlisHSPlasH is Y-up (gravity [0,-9.81,0]), matching
DamBreakModel. Pinocchio uses Z-up. A Rx(-90°) transform is applied to all
link poses when writing to the scene so the robot sits upright in the pool.

Setup:
  - Pool (UnitBox, mapInvert=true) filled with water up to water_depth
  - Robot spawned above the water surface — no fluid inside links at t=0
  - Robot pose: neutral joints, level orientation
"""
import numpy as np
import pinocchio as pin
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
URDF_PATH = REPO_ROOT / "src" / "amph" / "urdf" / "amph.urdf"
MESH_DIR  = REPO_ROOT / "stage2_gait_validation" / "sph_meshes"
SCENE_OUT = REPO_ROOT / "SPlisHSPlasH" / "data" / "Scenes" / "amph_swim.json"
UNIT_BOX  = REPO_ROOT / "SPlisHSPlasH" / "data" / "models" / "UnitBox.obj"

# Rotation that converts Z-up (pinocchio) → Y-up (SPlisHSPlasH):
#   x' = x,  y' = z,  z' = -y
R_ZUP_TO_YUP = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
], dtype=float)


def zup_pos_to_yup(p):
    return R_ZUP_TO_YUP @ np.array(p)


def zup_rot_to_yup(R):
    # Mesh vertices are in Z-up space; compose transforms directly.
    # Do NOT use change-of-basis (R_new @ R @ R_new.T) here.
    return R_ZUP_TO_YUP @ R


def rotation_to_axis_angle(R):
    """Convert rotation matrix to axis-angle (radians) for scene.json."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-8:
        return [0.0, 1.0, 0.0], 0.0
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis.tolist(), float(angle)   # radians, not degrees


def build_initial_config(model, water_depth, drop_height=0.10):
    """
    Neutral joints, level orientation.
    Base placed so the highest link clears the water surface by drop_height.
    All in pinocchio Z-up space — the scene writer converts to Y-up.
    """
    q0       = pin.neutral(model).copy()
    q0[3:7]  = [0, 0, 0, 1]    # identity quaternion (x,y,z,w)
    data_tmp = model.createData()
    pin.forwardKinematics(model, data_tmp, q0)
    # z-extents of all joints at neutral, base at z=0
    link_z_max = max(
        data_tmp.oMi[i].translation[2] for i in range(1, model.njoints)
    )
    # After Rx(-90°), z becomes y in SPlisHSPlasH — so set base z such that
    # all links end up above water_depth once transformed.
    q0[2] = water_depth + drop_height + max(0.0, link_z_max)
    return q0


def generate_scene(
    particle_radius = 0.028,
    pool_x          = 1.5,     # pool length [m], SPlisHSPlasH X
    pool_z          = 1.5,     # pool width  [m], SPlisHSPlasH Z
    water_depth     = 0.4,     # water height [m], SPlisHSPlasH Y
    viscosity       = 0.01,
):
    print("Loading robot model...")
    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    data  = model.createData()

    print("Building drop-in initial pose...")
    q0 = build_initial_config(model, water_depth)
    base_yup = zup_pos_to_yup(q0[:3])
    print(f"  Base pos (pinocchio Z-up): {q0[:3]}")
    print(f"  Base pos (scene Y-up):     {base_yup}")

    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    available_meshes = {f.stem for f in MESH_DIR.iterdir() if f.suffix == '.obj'}
    body_frames = [
        (fid, frame)
        for fid, frame in enumerate(model.frames)
        if frame.type == pin.FrameType.BODY and frame.name in available_meshes
    ]
    print(f"  Robot links with meshes: {len(body_frames)}")

    # ── Pool (Y-up: height is Y axis) ────────────────────────────────────────
    pool_height = water_depth + 0.3    # headroom above water
    pool_wall = {
        "geometryFile":  str(UNIT_BOX),
        "translation":   [0.0, pool_height / 2.0, 0.0],
        "rotationAxis":  [1.0, 0.0, 0.0],
        "rotationAngle": 0.0,
        "scale":         [pool_x, pool_height, pool_z],
        "isDynamic":     False,
        "isWall":        True,
        "mapInvert":     True,
        "mapThickness":  0.0,
        "mapResolution": [40, 30, 40],
    }

    # ── Robot rigid bodies ────────────────────────────────────────────────────
    robot_bodies = []
    for fid, frame in body_frames:
        link_name = frame.name
        T         = data.oMf[fid]

        # Convert pose from Z-up to Y-up
        pos_yup = zup_pos_to_yup(T.translation)
        rot_yup = zup_rot_to_yup(T.rotation)
        axis, ang = rotation_to_axis_angle(rot_yup)

        robot_bodies.append({
            "geometryFile":    str(MESH_DIR / f"{link_name}.obj"),
            "translation":     [float(x) for x in pos_yup],
            "rotationAxis":    [float(x) for x in axis],
            "rotationAngle":   ang,
            "scale":           [1.0, 1.0, 1.0],
            "velocity":        [0.0, 0.0, 0.0],
            "angularVelocity": [0.0, 0.0, 0.0],
            "isDynamic":       False,
            "isAnimated":      True,
            "mapInvert":       False,
            "mapThickness":    0.0,
            "mapResolution":   [20, 20, 20],
            "density":         1000.0,
        })
        print(f"  + {link_name}  y={pos_yup[1]:.3f}  (above water: {pos_yup[1]>water_depth})")

    # ── Fluid block (fills pool to water_depth in Y) ──────────────────────────
    m = particle_radius * 2
    fluid_block = {
        "denseMode": 0,
        "start": [-pool_x / 2 + m,  0.0,           -pool_z / 2 + m],
        "end":   [ pool_x / 2 - m,  water_depth,     pool_z / 2 - m],
    }

    vol         = (pool_x - 2*m) * (water_depth - 2*m) * (pool_z - 2*m)
    n_particles = vol / (2 * particle_radius) ** 3 * 0.64
    print(f"\nPool:  {pool_x:.2f}m × {pool_z:.2f}m × {water_depth:.2f}m water (Y-up)")
    print(f"Estimated fluid particles: {n_particles / 1000:.0f}K at r={particle_radius*1000:.0f}mm")

    scene = {
        "Configuration": {
            "particleRadius":             particle_radius,
            "numberOfStepsPerRenderUpdate": 4,
            "simulationMethod":           4,        # DFSPH
            "gravitation":                [0.0, -9.81, 0.0],
            "cflMethod":                  1,
            "cflFactor":                  1,
            "cflMaxTimeStepSize":         0.005,
            "boundaryHandlingMethod":     0,
            "DFSPH": {
                "minIterations":          2,
                "maxIterations":          100,
                "maxError":               0.05,
                "maxIterationsV":         100,
                "maxErrorV":              0.1,
                "enableDivergenceSolver": True,
            },
            "dataExportFPS":              25,
            "exportVTK":                  True,
            "particleAttributes":         "velocity;density;pressure",
        },
        "Materials": [
            {
                "id":              "Fluid",
                "density0":        1000,
                "colorMapType":    1,
                "viscosityMethod": 1,
                "Standard viscosity": {"viscosity": viscosity},
            }
        ],
        "FluidBlocks": [fluid_block],
        "RigidBodies": [pool_wall] + robot_bodies,
    }

    SCENE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SCENE_OUT, 'w') as f:
        json.dump(scene, f, indent=2)

    print(f"\nScene written to: {SCENE_OUT}")
    print(f"Rigid bodies: 1 pool + {len(robot_bodies)} robot links")


if __name__ == '__main__':
    generate_scene(
        particle_radius = 0.025,
        pool_x          = 1.5,
        pool_z          = 1.5,
        water_depth     = 0.4,
        viscosity       = 0.01,
    )
