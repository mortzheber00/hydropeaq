"""Turn a prescribed joint stroke into a full initial guess.

Expands the stroke onto the tree via the robot's ``CoordinateMap``, simulates
the floating base and computes the joint torques. Works for serial and
closed-chain robots; the output is in reduced coordinates.
"""

from __future__ import annotations

import numpy as np

from .base_sim import simulate_base_kinematics


def assemble_guess(
    dyn,
    theta: np.ndarray,
    thd: np.ndarray,
    thdd: np.ndarray,
    T: float,
    N: int,
    TAU_MAX: float,
    *,
    n_cycles: int = 20,
    label: str = "prescribed joint kinematics",
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a prescribed stroke into ``(X_guess, U_guess)``.

    Parameters
    ----------
    theta, thd, thdd : (n_theta, N+1)
        Actuated coordinates and their first two time derivatives.
    n_cycles : int
        Gait periods simulated to let the base speed settle.

    Returns
    -------
    X_guess : (nq_reduced + nv_reduced, N+1)
        ``[pos(3); quat(4); theta; v_base(6); thetadot]``.
    U_guess : (n_theta, N)
        Torques in the actuated coordinates, clipped to ``+-TAU_MAX``.
    """
    robot = dyn.robot
    cmap = robot.coord_map
    nq_r, nv_r = robot.nq_reduced, robot.nv_reduced
    dt = T / N

    for name, arr in (("theta", theta), ("thd", thd), ("thdd", thdd)):
        if arr.shape[1] < N + 1:
            raise ValueError(
                f"{name} has {arr.shape[1]} columns, need at least {N + 1}"
            )

    q_tree = np.column_stack([cmap.expand_numeric(theta[:, k]) for k in range(N + 1)])
    v_tree = np.column_stack([cmap.v_numeric(theta[:, k], thd[:, k]) for k in range(N + 1)])
    a_tree = np.column_stack([
        np.asarray(cmap.a_joints(theta[:, k], thd[:, k], thdd[:, k])).ravel()
        for k in range(N + 1)
    ])

    q_trim = dyn.find_trim_state()
    print(f"  Simulating base DOF ({label})...")
    q_base, v_base = simulate_base_kinematics(
        dyn, q_tree, v_tree, a_tree, q_trim, dt, n_cycles=n_cycles
    )

    X_guess = np.zeros((nq_r + nv_r, N + 1))
    X_guess[0:7] = q_base
    X_guess[7:nq_r] = theta[:, : N + 1]
    X_guess[nq_r:nq_r + 6] = v_base
    X_guess[nq_r + 6:] = thd[:, : N + 1]

    # Include the base acceleration; it couples into the joint torques via M[6:, :6].
    U_guess = np.zeros((robot.n_actuated, N))
    for k in range(N):
        a_full = np.concatenate([(v_base[:, k + 1] - v_base[:, k]) / dt, a_tree[:, k]])
        tau_tree = dyn.eval_inverse_dynamics(
            np.concatenate([q_base[:, k], q_tree[:, k]]),
            np.concatenate([v_base[:, k], v_tree[:, k]]),
            a_full,
        )
        tau_theta = np.asarray(cmap.tau_joints(theta[:, k], tau_tree[6:])).ravel()
        U_guess[:, k] = np.clip(tau_theta, -TAU_MAX, TAU_MAX)

    return X_guess, U_guess
