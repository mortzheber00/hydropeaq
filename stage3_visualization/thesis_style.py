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


# Canvas for a figure that shares a row with another one: 0.49\textwidth of a
# 15.2 cm text block.  Both figures take this exact canvas and are included
# unscaled, so the point sizes in half_width() are the point sizes on the page.
HALF = (2.94, 2.2)

def half_width() -> None:
    """Type and canvas settings for figures placed side by side.

    Call it once at import in a script whose figures go in a half-width slot.
    Scaling a full-width figure into that slot instead would put its 10 pt tick
    labels on the page at under 6 pt, and thin its 0.5 pt spines to a hairline.
    """
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # 'science' sets this to 'tight', which crops each figure to its own
        # content and so gives two figures two different page sizes — they then
        # scale differently in LaTeX and their type no longer matches.  None
        # keeps the canvas.  Note savefig(bbox_inches=None) does *not* do this:
        # that means "use this rcParam".
        "savefig.bbox": None,
        # Spines and ticks at the default 0.5 print unevenly once this small.
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


def tex(s: str) -> str:
    """Escape the characters that break usetex in a run/gait name."""
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
             .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#"))
