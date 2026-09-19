"""Initial guesses for the gait OCP.

- ``build_initial_guess``: Fourier-series paddling stroke (Qu et al. 2025)
- ``build_robot_ik_initial_guess``: the firmware's 4-phase IK swim gait

Both return ``(X_guess (nx, N+1), U_guess (n_act, N))``.
"""

from .firmware import build_robot_ik_initial_guess
from .paper import build_initial_guess

__all__ = [
    "build_initial_guess",
    "build_robot_ik_initial_guess",
]
