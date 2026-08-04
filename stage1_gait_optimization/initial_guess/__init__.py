"""Initial guess strategies for the gait OCP.

Three strategies are provided:

1. Fourier-series paddling trajectory (Qu et al. 2025):
   ``build_initial_guess(dyn, gait, N, T_FIXED, TAU_MAX)``

2. Robot firmware IK gait — mirrors ``Robot_Swim_Task_IK`` from the embedded C
   firmware (4-phase state machine: recovery → strike → power → lift):
   ``build_robot_ik_initial_guess(dyn, N, T_FIXED, TAU_MAX, ...)``

3. Sinusoid search (Cui et al. 2026 imitation-learning stage): LHS-samples
   joint-space sinusoid gaits and keeps the best under a simulated
   thrust-minus-lift score:
   ``build_search_initial_guess(dyn, N, T_FIXED, TAU_MAX, ...)``

All return ``(X_guess (nx, N+1), U_guess (n_act, N))``.
"""

from .firmware import build_robot_ik_initial_guess
from .paper import build_initial_guess
from .search import (
    build_search_initial_guess,
    build_sinusoid_guess,
    search_sinusoid_params,
)
from .theta_sinusoid import build_theta_sinusoid_guess

__all__ = [
    "build_initial_guess",
    "build_theta_sinusoid_guess",
    "build_robot_ik_initial_guess",
    "build_search_initial_guess",
    "build_sinusoid_guess",
    "search_sinusoid_params",
]
