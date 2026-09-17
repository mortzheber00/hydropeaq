"""Hydrodynamic coefficients; the single place to change them.

These are the constructor defaults of ``SymbolicHydrodynamicModel`` and
``SymbolicDynamics`` and the CLI defaults of ``simulate_ocp.py``.

Note: ``CD_A`` and ``CA_A`` sit on the lower bound of the calibration sweep, so
they are not really identified. Widen the sweep bounds before interpreting them.
"""

# --- Quadratic form drag ---
CD_T = 3.25        # transverse (cross-flow) drag coefficient
CD_A = 0.8       # axial (blunt-body) drag coefficient    [at sweep bound]

# --- Added mass ---
CA_T = 1.9        # transverse added-mass coefficient
CA_A = 0.200        # axial added-mass coefficient           [at sweep bound]

# --- Linear (skin-friction) damping, Fossen's D_S ---
CD_LIN_T = 1.6667    # transverse linear damping coefficient
CD_LIN_A = 3.75    # axial linear damping coefficient

# --- Model settings tied to the fit ---
# Linearisation speed of the quadratic drag, D_S = 0.5*rho*Cd_lin*A*v. CD_LIN_*
# were fitted at this value; change them together or re-run the sweep.
V_LINEAR_THRESHOLD = 0.7   # [m/s]

# Drag scale on non-trunk links (wake slip); not part of the fit.
LEG_THRUST_SCALE = 1.0
