#!/usr/bin/env python3
"""Search the sinusoid gait family and save the best parameters.

Runs the Cui-et-al.-style LHS sinusoid search (see
``initial_guess/search.py``), then writes the winning parameters to
``search_best_params.json``.  The OCP scripts can load that file and rebuild
the same warm start without re-running the search:

    import json
    from initial_guess import build_sinusoid_guess
    params = json.load(open("search_best_params.json"))["params"]
    X_guess, U_guess = build_sinusoid_guess(dyn, params, N, T_FIXED, TAU_MAX)
"""

import json
from pathlib import Path

import numpy as np
from hydro_model import SymbolicDynamics, load_robot
from initial_guess import build_sinusoid_guess, search_sinusoid_params

OUT_PATH = Path(__file__).parent / "search_best_params.json"

# ── Must match the OCP horizon the params will be reused with ───────────
N = 64  # shooting intervals
T_FIXED = 1.0  # fixed cycle period [s]
TAU_MAX = 3.5  # joint torque limit [Nm]

# ── Search budget / ranges (see search_sinusoid_params docstring) ───────
SEARCH_KWARGS = dict(
    n_samples=1000,
    seed=0,
    lift_weight=0.5,
    amp_thigh_range=(0.10, 0.55),
    amp_calf_range=(0.10, 0.55),
    phase_range=(0.0, np.pi),
    diag_offset_range=(0.0, 0.5),
    n_cycles=1,
    score_cycles=8,
)

ROBOT = "amph"   # registered robot name; see hydro_model/robots/



def main():
    print("Building robot and symbolic dynamics...")
    robot = load_robot(ROBOT)
    dyn = SymbolicDynamics(robot)

    print("Searching sinusoid gait parameters...")
    params, score = search_sinusoid_params(dyn, N, T_FIXED, **SEARCH_KWARGS)

    # Verify the winner builds a sane guess over the full OCP horizon.
    X_guess, _ = build_sinusoid_guess(dyn, params, N, T_FIXED, TAU_MAX)
    net_dx = float(X_guess[0, -1])

    record = {
        "params": params,
        "score": score,
        "net_dx": net_dx,
        "N": N,
        "T_FIXED": T_FIXED,
        "TAU_MAX": TAU_MAX,
        "search_kwargs": SEARCH_KWARGS,
    }
    OUT_PATH.write_text(json.dumps(record, indent=2))

    print("\n── Best sinusoid parameters ─────────────────────────────")
    for k, v in params.items():
        print(f"  {k:12s} = {v}")
    print(f"  score        = {score:.4f}")
    print(f"  net dx/cycle = {net_dx:.4f} m")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
