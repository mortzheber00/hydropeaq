"""Freeze BODY2's linkage geometry into the JSON the OCP pipeline reads.

``leg_linkage_sim`` derives the pin positions by fitting cylinders to the STL
hole walls.  That is the right way to *establish* the geometry but the wrong
thing to depend on at import time: it carries half a dozen tuned tolerances and
raises on any mismatch, so re-exporting a mesh would silently take the whole
optimisation pipeline down.  This script runs the derivation once and writes the
~70 numbers out; ``tests/test_body2_linkage.py`` re-derives and checks them.

    python src/BODY2/scripts/dump_linkage.py

Re-run after changing the CAD or re-running ``rehome_urdf.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from leg_linkage_sim import LEGS, PLANE, Leg

OUT = (Path(__file__).resolve().parents[3]
       / "stage1_gait_optimization" / "hydro_model" / "robots" / "body2_linkage.json")

# Which link frame carries each pin that no joint origin sits on.  The cylinder
# specs need these as offsets in the owning link's local frame.
LOCAL_POINTS = {"P6": "2.1", "P8": "1.3", "tip": "2.3"}


def _to_local(leg: Leg, link_key: str, p_xz) -> list:
    """A pin, expressed in the local frame of one of the links it belongs to.

    The pin is a line parallel to world Y; we take the point on it level with
    the link frame origin, so the offset has no spurious out-of-plane component.
    """
    R, o = leg.frames[f"Link_{leg.name}{link_key}"]
    p_world = np.array([p_xz[0], o[1], p_xz[1]])
    return (R.T @ (p_world - o)).tolist()


def main() -> None:
    legs = {}
    ref_len = ref_branch = None
    for name in LEGS:
        leg = Leg(name)
        # Lengths are kept per leg rather than shared: each leg's values are
        # derived from that same leg's pins, so the loops close exactly at the
        # zero pose.  They agree across legs to ~1 nm (mesh-fitting noise).
        this_len = {k: float(v) for k, v in leg.L.items()}
        this_branch = [float(b) for b in leg.branch]
        legs[name] = {
            "pins": {k: [float(v[0]), float(v[1])] for k, v in leg.pins0.items()},
            "tip": [float(leg.tip0[0]), float(leg.tip0[1])],
            "sgn": {k: float(v) for k, v in leg.sgn.items()},
            "lengths": this_len,
            "local_points": {
                k: _to_local(leg, link, leg.tip0 if k == "tip" else leg.pins0[k])
                for k, link in LOCAL_POINTS.items()
            },
        }
        if ref_len is None:
            ref_len, ref_branch = this_len, this_branch
        else:
            drift = max(abs(this_len[k] - ref_len[k]) for k in this_len)
            assert drift < 1e-6, f"{name}: link lengths differ from {LEGS[0]} by {drift:.2e} m"
            assert this_branch == ref_branch, f"{name}: assembly branch differs between legs"

    doc = {
        "_generated_by": "src/BODY2/scripts/dump_linkage.py",
        "_note": "world (x, z) at the URDF zero pose; the leg plane is normal to Y",
        "plane_axes": PLANE,
        "branch": ref_branch,
        "legs": legs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    ref = legs[LEGS[0]]["lengths"]
    print(f"  lengths [mm]: { {k: round(v * 1000, 3) for k, v in ref.items()} }")
    print(f"  branch: {ref_branch}")


if __name__ == "__main__":
    main()
