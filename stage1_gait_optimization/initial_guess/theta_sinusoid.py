"""Initial guess for a closed-chain robot, written directly in theta.

The other guess builders in this package solve inverse kinematics to turn a
desired foot path into joint angles.  That step is unnecessary here: for a
robot with a coordinate map, theta *is* the actuated coordinate, so a gait can
be written down in it and differentiated analytically.

The one thing that does need care is staying assemblable.  BODY2's feasible
set is a diagonal band in ``(theta1, theta2)``, roughly +-8 deg wide across but
essentially unbounded along, so the stroke is an ellipse aligned to the band:

    theta(t) = theta_home + A_along * cos(phi) * band + A_across * sin(phi) * across

Sweeping along the band is the power stroke; the small across-band component is
what makes the foot path enclose an area, and therefore what generates thrust.
"""

from __future__ import annotations

import numpy as np

from .assemble import assemble_guess

# Quarter-cycle phase offsets giving a diagonal (trot-like) sequence, in the
# spec's leg order.
DEFAULT_LEG_PHASE = (0.0, 0.5, 0.5, 0.0)


def theta_trajectory(robot, N, T, *, a_along, a_across, leg_phase, phase0=0.0):
    """Analytic ``(theta, thetadot, thetaddot)``, each ``(n_theta, N+1)``.

    Derivatives are exact rather than finite-differenced, which keeps the
    dynamics defects of the guess small.
    """
    from hydro_model.robots.body2 import BAND_DIR

    spec = robot.spec
    n_legs = len(spec.leg_names)
    if len(leg_phase) != n_legs:
        raise ValueError(f"leg_phase has {len(leg_phase)} entries, need {n_legs}")

    across = np.array([-BAND_DIR[1], BAND_DIR[0]])
    home = (np.zeros(robot.n_actuated) if spec.theta_home is None
            else np.asarray(spec.theta_home, dtype=float))

    t = np.linspace(0.0, T, N + 1)
    w = 2.0 * np.pi / T
    theta = np.tile(home[:, None], (1, N + 1))
    thd = np.zeros_like(theta)
    thdd = np.zeros_like(theta)

    for i in range(n_legs):
        phi = w * t + phase0 - 2.0 * np.pi * leg_phase[i]
        s, c = np.sin(phi), np.cos(phi)
        for j, d in enumerate((a_along * BAND_DIR, a_across * across)):
            # j = 0 uses cos(phi), j = 1 uses sin(phi)
            f, fd, fdd = ((c, -w * s, -w * w * c) if j == 0 else (s, w * c, -w * w * s))
            for k in (0, 1):
                theta[2 * i + k] += d[k] * f
                thd[2 * i + k] += d[k] * fd
                thdd[2 * i + k] += d[k] * fdd
    return theta, thd, thdd


def build_theta_sinusoid_guess(
    dyn,
    N: int,
    T: float,
    TAU_MAX: float,
    *,
    a_along: float = np.radians(30.0),
    a_across: float = np.radians(5.0),
    leg_phase=DEFAULT_LEG_PHASE,
    phase0: float = 0.0,
    n_cycles: int = 20,
):
    """Return ``(X_guess, U_guess)`` in the robot's reduced coordinates.

    ``X_guess`` is ``(nq_reduced + nv_reduced, N+1)`` laid out as
    ``[pos(3); quat(4); theta; v_base(6); thetadot]``; ``U_guess`` is
    ``(n_theta, N)``.
    """
    robot = dyn.robot
    cmap = robot.coord_map

    theta, thd, thdd = theta_trajectory(
        robot, N, T, a_along=a_along, a_across=a_across,
        leg_phase=leg_phase, phase0=phase0,
    )

    margin = np.array([np.min(np.asarray(cmap.feasibility(theta[:, k])))
                       for k in range(N + 1)])
    if not np.all(np.isfinite(margin)) or margin.min() <= 0:
        k = int(np.nanargmin(margin))
        raise ValueError(
            f"guess leaves the assemblable set at step {k}/{N} "
            f"(min h^2 = {margin[k]:.3e}); reduce a_along/a_across"
        )
    print(f"  guess assembly margin: h >= {np.sqrt(margin.min()) * 1e3:.2f} mm")

    return assemble_guess(dyn, theta, thd, thdd, T, N, TAU_MAX, n_cycles=n_cycles)
