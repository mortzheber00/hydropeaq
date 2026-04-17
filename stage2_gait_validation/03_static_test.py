"""
Test static equilibrium — does the robot maintain depth with no gait?
This validates that SPH hydrostatic pressure approximately balances gravity.
Must pass before running the full gait simulation.

Coordinate convention:
  Pinocchio uses Z-up; SPlisHSPlasH uses Y-up.
  Poses are converted Z-up→Y-up before passing to SPlisHSPlasH.
  Forces are converted Y-up→Z-up before passing to pinocchio.
"""
import numpy as np
import pinocchio as pin
import pysplishsplash as sph_lib
from pathlib import Path

REPO_ROOT  = Path(__file__).parent.parent
URDF_PATH  = REPO_ROOT / "src" / "amph" / "urdf" / "amph.urdf"
SCENE_PATH = REPO_ROOT / "SPlisHSPlasH" / "data" / "Scenes" / "amph_swim.json"
MESH_DIR   = REPO_ROOT / "stage2_gait_validation" / "sph_meshes"

# Rx(-90°): maps Z-up → Y-up   (x'=x, y'=z, z'=-y)
R_ZUP_TO_YUP = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
], dtype=float)


def to_yup_pos(p):
    return R_ZUP_TO_YUP @ np.asarray(p)


def to_yup_rot(R):
    return R_ZUP_TO_YUP @ R


def to_zup_vec(v):
    """Inverse transform: Y-up vector → Z-up (for forces from SPlisHSPlasH)."""
    return R_ZUP_TO_YUP.T @ np.asarray(v)


def build_initial_config(model, water_depth=0.4, drop_height=0.10):
    """Neutral joints, level orientation, base above water. Matches 03_generate_scene.py."""
    q0 = pin.neutral(model).copy()
    q0[3:7] = [0, 0, 0, 1]
    data_tmp = model.createData()
    pin.forwardKinematics(model, data_tmp, q0)
    link_z_max = max(
        data_tmp.oMi[i].translation[2] for i in range(1, model.njoints)
    )
    q0[2] = water_depth + drop_height + max(0.0, link_z_max)
    return q0


def run_static_test(
    test_duration    = 2.0,
    dt               = 0.001,
    acceptable_drift = 0.02,
    gui              = False,
):
    print("=" * 50)
    print("STATIC EQUILIBRIUM TEST")
    print("=" * 50)

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    data  = model.createData()

    q_full = build_initial_config(model)
    v_full = np.zeros(model.nv)
    print(f"Initial base z (pinocchio): {q_full[2]:.4f}m")

    q_joints_init = q_full[7:].copy()

    available_meshes = {f.stem for f in MESH_DIR.iterdir() if f.suffix == '.obj'}
    body_frames = [
        (fid, frame)
        for fid, frame in enumerate(model.frames)
        if frame.type == pin.FrameType.BODY and frame.name in available_meshes
    ]

    # Initialize SPlisHSPlasH
    print("\nInitializing SPlisHSPlasH...")
    base = sph_lib.Exec.SimulatorBase()
    base.init(sceneFile=str(SCENE_PATH), useGui=gui)
    if gui:
        gui_obj = sph_lib.GUI.Simulator_GUI_imgui(base)
        base.setGui(gui_obj)
    base.initSimulation()
    sim = sph_lib.Simulation.getCurrent()

    if not gui:
        # Headless: manually complete deferredInit (initBoundaryData +
        # neighbourhood sort + DM velocity update for Koschier2017).
        # In GUI mode runSimulation() calls deferredInit() internally — calling
        # initBoundaryData() a second time would corrupt state and segfault.
        base.getBoundarySimulator().initBoundaryData()
        sim.performNeighborhoodSearchSort()
        base.updateDMVelocity()
        n_bodies = sim.numberOfBoundaryModels()
        print(f"Fluid particles: {sim.getFluidModel(0).numActiveParticles()}")
        print(f"Rigid bodies:    {n_bodies}")
        assert n_bodies == len(body_frames) + 1, (
            f"Scene has {n_bodies} rigid bodies but model has {len(body_frames)} "
            f"links with meshes — regenerate scene with 02_generate_scene.py"
        )
    else:
        print(f"Fluid particles: {sim.getFluidModel(0).numActiveParticles()}")

    def push_poses_to_sph():
        # Boundary model 0 is the pool wall; robot links start at index 1.
        for rb_idx, (fid, frame) in enumerate(body_frames):
            T  = data.oMf[fid]
            v  = pin.getVelocity(
                     model, data, frame.parentJoint,
                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                 )
            rb = sim.getBoundaryModel(rb_idx + 1).getRigidBodyObject()
            rb.setWorldSpacePosition(to_yup_pos(T.translation))
            rb.setWorldSpaceRotation(to_yup_rot(T.rotation))
            rb.setVelocity(to_yup_pos(v.linear))
            rb.setAngularVelocity(to_yup_pos(v.angular))
            rb.updateMeshTransformation()
        # Update the Koschier2017 CFL velocity bound so adaptive stepping
        # accounts for boundary motion.  (No updateBoundaryParticles — that
        # does an Akinci2012 cast and crashes with our boundaryHandlingMethod=1.
        # The density-map queries in computeDensityAndGradient read getPosition()
        # directly each step, so no separate particle-position update is needed.)
        base.updateDMVelocity()

    # Compute initial FK so data is populated before the first callback.
    # In headless mode also push poses to SPlisHSPlasH; in GUI mode the scene
    # JSON already contains the matching initial positions so the push is
    # redundant (and initBoundaryData hasn't run yet anyway).
    pin.forwardKinematics(model, data, q_full, v_full)
    pin.updateFramePlacements(model, data)
    if not gui:
        push_poses_to_sph()

    # Shared mutable state accessed by the timestep callback
    q_init = q_full.copy()
    v_init = v_full.copy()

    state = {
        'q': q_full,
        'v': v_full,
        't': 0,
        'z_log': [],
    }
    z_init = q_full[2]

    def on_reset():
        """Called by SPlisHSPlasH when the user hits Reset in the GUI.
        SPH already restored m_x = m_x0 for all bodies; we must mirror
        that in pinocchio so the next timestep callback uses the same pose.
        """
        state['q'] = q_init.copy()
        state['v'] = v_init.copy()
        state['t'] = 0
        state['z_log'].clear()
        pin.forwardKinematics(model, data, state['q'], state['v'])
        pin.updateFramePlacements(model, data)
        push_poses_to_sph()
        print("  [reset] pinocchio state restored to initial pose")

    def timestep():
        q = state['q']
        v = state['v']

        # Collect SPH forces (Y-up) → convert to Z-up for pinocchio
        f_ext = pin.StdVec_Force()
        for _ in range(model.njoints):
            f_ext.append(pin.Force.Zero())

        f_buf = np.zeros(3, dtype=np.float32)
        t_buf = np.zeros(3, dtype=np.float32)
        total_f_zup = np.zeros(3)
        for rb_idx, (fid, frame) in enumerate(body_frames):
            bm = sim.getBoundaryModel(rb_idx + 1)  # 0 = pool wall
            T  = data.oMf[fid]

            bm.getForceAndTorque(f_buf, t_buf)
            f_yup = f_buf.astype(float)
            t_yup = t_buf.astype(float)
            bm.clearForceAndTorque()

            f_world = to_zup_vec(f_yup)
            t_world = to_zup_vec(t_yup)
            total_f_zup += f_world
            f_ext[frame.parentJoint] = pin.Force(
                T.rotation.T @ f_world,
                T.rotation.T @ t_world,
            )

        tau = np.zeros(model.nv)
        a   = pin.aba(model, data, q, v, tau, f_ext)

        v[:6] += a[:6] * dt
        q      = pin.integrate(model, q, v * dt)
        q[7:]  = q_joints_init
        v[6:]  = 0.0

        pin.forwardKinematics(model, data, q, v)
        pin.updateFramePlacements(model, data)
        push_poses_to_sph()

        state['q'] = q
        state['v'] = v
        state['z_log'].append(q[2])

        t = state['t']
        if t % 100 == 0:
            fz = total_f_zup[2]
            print(f"  t={t*dt:.2f}s  body_z={q[2]:.4f}m  drift={q[2]-z_init:+.4f}m  sph_fz={fz:+.2f}N")
        state['t'] += 1

    if gui:
        base.setTimeStepCB(timestep)
        base.setResetCB(on_reset)
        print(f"\nGUI mode — close the window to finish.")
        base.runSimulation()  # calls deferredInit() then GUI loop
    else:
        print(f"\nRunning {test_duration}s static test...")
        n_steps = int(test_duration / dt)
        for _ in range(n_steps):
            base.timeStepNoGUI()
            timestep()

    z_log       = state['z_log']
    final_drift = abs(z_log[-1] - z_init) if z_log else 0.0

    print(f"\n{'='*50}")
    print(f"STATIC TEST RESULTS")
    print(f"Initial body z:  {z_init:.4f}m")
    print(f"Final body z:    {z_log[-1]:.4f}m")
    print(f"Total drift:     {final_drift:.4f}m")
    print(f"Acceptable:      {acceptable_drift:.4f}m")

    if final_drift <= acceptable_drift:
        print(f"✓ PASS — SPH buoyancy sufficient")
        return True
    else:
        direction = "sinking" if z_log[-1] < z_init else "rising"
        print(f"✗ FAIL — robot is {direction} ({final_drift*100:.1f}cm drift)")
        print(f"  Options:")
        print(f"  1. Reduce particle_radius for better pressure accuracy")
        print(f"  2. Add analytical buoyancy correction")
        print(f"  3. Adjust robot mass in URDF to be more neutrally buoyant")
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Run with SPlisHSPlasH GUI')
    parser.add_argument('--duration', type=float, default=2.0)
    args = parser.parse_args()
    run_static_test(test_duration=args.duration, gui=args.gui)
