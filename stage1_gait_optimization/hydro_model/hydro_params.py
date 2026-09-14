"""Single source of truth for the hydrodynamic coefficients.

``SymbolicHydrodynamicModel`` and ``SymbolicDynamics`` take their constructor
defaults from here, and ``stage2_sim_validation/hydro_calibration/simulate_ocp.py`` uses them as
its CLI defaults.  Every consumer that builds the dynamics without overriding
coefficients -- the OCP (``trajopt/run_collocation.py``), the codesign solver,
the prescribed-joint rollout, the diagnostics scripts -- therefore moves
together when these values change.  Change them here and nowhere else.

Fitted 2026-08-16 against the SPH bag by
``stage2_sim_validation/hydro_calibration/sweep_hydro_params.py --method optimize``.

Caveat on the fit: ``CD_A`` and ``CA_A`` both landed exactly on the lower edge
of the sweep's search range, so they are bound artifacts rather than identified
values -- the data wanted to push them lower still.  Treat the axial
coefficients as "as small as the sweep was allowed to make them", and widen
``BOUNDS`` in the sweep before reading anything physical into them.
"""

# ── Quadratic form drag ───────────────────────────────────────────────────────
CD_T = 3.25        # transverse (cross-flow) drag coefficient
CD_A = 0.8       # axial (blunt-body) drag coefficient    [at sweep bound]

# ── Added mass ────────────────────────────────────────────────────────────────
CA_T = 1.9        # transverse added-mass coefficient
CA_A = 0.200        # axial added-mass coefficient           [at sweep bound]

# ── Linear (skin-friction) damping — Fossen's D_S ─────────────────────────────
CD_LIN_T = 1.6667    # transverse linear damping coefficient
CD_LIN_A = 3.75    # axial linear damping coefficient

# ── Model settings that travel with the coefficients ──────────────────────────
# Speed about which the quadratic drag is linearised: D_S = 0.5*rho*Cd_lin*A*v.
# The sweep's linear-drag unit components are built at this value, so a fitted
# CD_LIN_* is only meaningful together with the V_LINEAR_THRESHOLD it was fitted
# at -- change the two together, or re-run the sweep.
V_LINEAR_THRESHOLD = 0.7   # [m/s]

# Drag scale on non-trunk links (wake slip).  Not part of the fit.
LEG_THRUST_SCALE = 1.0
