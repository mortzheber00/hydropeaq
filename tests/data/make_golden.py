"""Capture a numerical fingerprint of the amph model.

Run this ONCE on unmodified source, before the multi-robot refactor begins.
``tests/test_amph_regression.py`` replays every quantity recorded here and
fails if any of them moves, which is what makes "amph is unchanged" a
checkable claim rather than a hope.

    python tests/data/make_golden.py

Re-run it only when an amph-affecting change is *intended*, and say so in the
commit message.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stage1_gait_optimization"))

from hydro_model import QuadrupedRobot, SymbolicDynamics  # noqa: E402
from ocp_common import legacy_to_tangent, tangent_to_legacy  # noqa: E402

URDF_PATH = REPO_ROOT / "src" / "amph" / "urdf" / "amph.urdf"
OUT = Path(__file__).resolve().parent / "amph_golden.npz"

# Reproduces the boot sequence used verbatim by every driver script, including
# the zeros-not-neutral quirk at trajopt/run_collocation.py:68.
BOOT_Q = "zeros"
N_SAMPLES = 20
SEED = 20240617

CYL_FIELDS = (
    "radius", "length", "volume_displaced",
    "center", "axis_world", "axis_local", "center_local",
)


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict[str, np.ndarray] = {}

    robot = QuadrupedRobot(URDF_PATH)
    robot.forward_kinematics(np.zeros(robot.nq))
    robot.build_cylinders()
    dyn = SymbolicDynamics(robot)

    nq, nv, n_act = robot.nq, robot.nv, robot.n_actuated
    out["dims"] = np.array([nq, nv, n_act, robot.model.njoints, robot.model.nframes])
    out["actuated_joint_names"] = np.array(robot.actuated_joint_names)
    out["link_names"] = np.array(sorted(robot.links))
    out["total_mass"] = np.array(robot.total_mass())

    # --- model geometry: catches _recenter_base_y regressions ---------------
    out["joint_placements"] = np.array(
        [np.asarray(p.translation) for p in robot.model.jointPlacements]
    )
    out["joint_rotations"] = np.array(
        [np.asarray(p.rotation) for p in robot.model.jointPlacements]
    )
    out["base_lever"] = np.asarray(robot.model.inertias[1].lever)
    out["foot_frame_ids"] = np.array(
        [robot.foot_frame_ids[leg] for leg in sorted(robot.foot_frame_ids)]
    )
    out["link_mesh_volumes"] = np.array(
        [robot.link_mesh_volumes[k] for k in sorted(robot.link_mesh_volumes)]
    )

    # --- cylinder primitives ------------------------------------------------
    for name in sorted(robot.links):
        cyl = robot.links[name].cylinder
        if cyl is None:
            continue
        for f in CYL_FIELDS:
            out[f"cyl/{name}/{f}"] = np.asarray(getattr(cyl, f), dtype=float)

    # --- CasADi functions on fixed random inputs ----------------------------
    Q = np.zeros((N_SAMPLES, nq))
    V = rng.normal(scale=0.5, size=(N_SAMPLES, nv))
    A = rng.normal(scale=0.5, size=(N_SAMPLES, nv))
    for i in range(N_SAMPLES):
        q = robot.neutral_config()
        q[0:3] = rng.normal(scale=0.1, size=3)
        quat = rng.normal(size=4)
        q[3:7] = quat / np.linalg.norm(quat)
        q[7:] = rng.uniform(-0.5, 0.5, size=nq - 7)
        Q[i] = q
    out["sample_q"], out["sample_v"], out["sample_a"] = Q, V, A

    for fname, args in (
        ("f_M_rb", ("q",)), ("f_C_rb", ("q", "v")), ("f_g_rb", ("q",)),
        ("f_M_added", ("q",)), ("f_tau_buoyancy", ("q",)), ("f_tau_drag", ("q", "v")),
        ("f_C_A_v", ("q", "v")), ("f_tau_added", ("q", "v", "a")),
        ("f_inverse_dynamics", ("q", "v", "a")),
        ("f_forward_dynamics", ("q", "v", "a")),   # 3rd arg is tau, reuse A
    ):
        fn = getattr(dyn, fname)
        rows = []
        for i in range(N_SAMPLES):
            call = {"q": Q[i], "v": V[i], "a": A[i]}
            rows.append(np.asarray(fn(*[call[a] for a in args])).ravel())
        out[f"fn/{fname}"] = np.array(rows)

    # --- tangent dynamics ---------------------------------------------------
    q_ref_quat = np.array([0.0, 0.0, 0.0, 1.0])
    f_kin, f_inv_dyn = dyn.build_tangent_dynamics(q_ref_quat)
    XT = rng.normal(scale=0.3, size=(N_SAMPLES, 2 * nv))
    AR = rng.normal(scale=0.3, size=(N_SAMPLES, nv))
    out["sample_xt"], out["sample_ar"] = XT, AR
    out["fn/f_kin"] = np.array([np.asarray(f_kin(XT[i])).ravel() for i in range(N_SAMPLES)])
    out["fn/f_inv_dyn"] = np.array(
        [np.asarray(f_inv_dyn(XT[i], AR[i])).ravel() for i in range(N_SAMPLES)]
    )

    # --- tangent <-> legacy round trip --------------------------------------
    X_leg = np.zeros((nq + nv, N_SAMPLES))
    X_leg[:nq, :] = Q.T
    X_leg[nq:, :] = V.T
    X_tan = legacy_to_tangent(X_leg, q_ref_quat, robot.model)
    out["legacy_to_tangent"] = X_tan
    out["tangent_to_legacy"] = tangent_to_legacy(X_tan, q_ref_quat, robot.model)

    # --- trim ---------------------------------------------------------------
    out["trim_state"] = dyn.find_trim_state()

    # --- foot positions -----------------------------------------------------
    robot.forward_kinematics(out["trim_state"])
    fp = robot.foot_positions()
    out["foot_positions"] = np.array([fp[leg] for leg in sorted(fp)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}  ({len(out)} arrays, {OUT.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
