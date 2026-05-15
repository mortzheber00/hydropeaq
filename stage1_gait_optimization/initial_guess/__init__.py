"""Initial guess strategies for the gait OCP.

Two strategies are provided:

1. Fourier-series paddling trajectory (Qu et al. 2025):
   ``build_initial_guess(dyn, gait, N, T_FIXED, TAU_MAX)``

2. Robot firmware IK gait — mirrors ``Robot_Swim_Task_IK`` from the embedded C
   firmware (4-phase state machine: recovery → strike → power → lift):
   ``build_robot_ik_initial_guess(dyn, N, T_FIXED, TAU_MAX, ...)``

Both return ``(X_guess (nx, N+1), U_guess (n_act, N))``.
"""

from .firmware import build_robot_ik_initial_guess
from .paper import build_initial_guess

__all__ = ["build_initial_guess", "build_robot_ik_initial_guess"]
