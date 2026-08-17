"""Shared plotting style for the thesis figures.

Importing this activates the ``science`` matplotlib style with real LaTeX text
rendering, and hands out a colour + marker per initial gait.  Both are keyed by
gait *name*, not by the order a figure happens to select them in, so a gait
looks the same in every figure of the thesis.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  registers the 'science' matplotlib style

# Professional thesis style with real LaTeX text rendering (Computer Modern).
plt.style.use(["science"])
plt.rcParams["text.usetex"] = True

# Colourblind-safe categorical palette, verified in OKLab (dE x100): worst
# normal-vision pair 21.1 (floor 15); worst pair under simulated deuteranopia
# 12.9, protanopia 10.8, tritanopia 13.2 (target 8).
PALETTE = ("#0173B2", "#66A61E", "#B2182B", "#CC78BC", "#332288")
# Distinct marker shapes carry the same identity without colour — needed in
# greyscale print and for the weakest CVD pairs.
MARKERS = ("o", "s", "^", "D", "v")

# Slot order for the gaits the pipeline knows about; anything else takes the
# next free slot in first-seen order.
GAIT_ORDER = ["LSPG25", "LSPG33", "TLPG50", "Prototype", "ThetaSinusoid"]


def style_for(gait: str) -> tuple[str, str]:
    """``(colour, marker)`` for one gait, stable across figures and runs."""
    if gait not in GAIT_ORDER:
        GAIT_ORDER.append(gait)
    i = GAIT_ORDER.index(gait)
    return PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]


def tex(s: str) -> str:
    """Escape the characters that break usetex in a run/gait name."""
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
             .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#"))
