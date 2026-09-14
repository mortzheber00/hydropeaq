"""Shared plotting style for the thesis figures.

Importing this activates the ``science`` matplotlib style with real LaTeX text
rendering, and hands out a colour + marker per initial gait.  Both are keyed by
gait *name*, not by the order a figure happens to select them in, so a gait
looks the same in every figure of the thesis.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import scienceplots
from cycler import cycler  # noqa: F401  registers the 'science' matplotlib style

# Professional thesis style with real LaTeX text rendering (Computer Modern).
plt.style.use(["science"])
plt.rcParams["text.usetex"] = True

# Colourblind-safe categorical palette, verified in OKLab (dE x100): worst
# normal-vision pair 21.1 (floor 15); worst pair under simulated deuteranopia
# 12.9, protanopia 10.8, tritanopia 13.2 (target 8).
PALETTE = ("#0173B2", "#66A61E", "#B2182B", "#CC78BC", "#332288")
# The 'science' style cycles its own colours; implicit "C0", "C1", … draw from
# the thesis palette instead.
plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)
# Distinct marker shapes carry the same identity without colour — needed in
# greyscale print and for the weakest CVD pairs.
MARKERS = ("o", "s", "^", "D", "v")

# One colour per leg, in ``RobotSpec.leg_names`` order (FL, FR, HL/BL, HR/BR),
# so a leg reads the same in every figure.
LEG_COLORS = PALETTE[:4]

# Slot order for the gaits the pipeline knows about; anything else takes the
# next free slot in first-seen order.
GAIT_ORDER = ["LSPG25", "LSPG33", "TLPG50", "Prototype", "ThetaSinusoid"]


def style_for(gait: str) -> tuple[str, str]:
    """``(colour, marker)`` for one gait, stable across figures and runs."""
    if gait not in GAIT_ORDER:
        GAIT_ORDER.append(gait)
    i = GAIT_ORDER.index(gait)
    return PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]


# Canvas widths, both for figures included unscaled so that the point sizes set
# here are the point sizes on the page: the full 15.2 cm text block, and
# 0.49\textwidth of it for a figure sharing a row with another.
TEXT_WIDTH_IN = 5.98
HALF = (2.94, 2.2)

# Height to add to a canvas for one row of legend placed above the axes.  A
# legend outside the axes is not counted by tight_layout when the canvas is
# fixed (savefig.bbox is None), so the row has to be paid for explicitly.
LEGEND_ROW_IN = 0.30


def full_width() -> None:
    """Keep the canvas a figure asked for, for the full-text-width slot.

    'science' sets ``savefig.bbox='tight'``, which crops each figure to its own
    content, so two figures reach the page at two widths and scale differently
    under \\includegraphics — their type then no longer matches.  Note that
    ``savefig(bbox_inches=None)`` does *not* do this: that means "use the
    rcParam".
    """
    plt.rcParams["savefig.bbox"] = None


def legend_row(fig, ax, rows: int = 1) -> None:
    """Grow the canvas by ``rows`` legend rows and pin the axes to the bottom.

    The companion to a ``legend(loc="lower center", bbox_to_anchor=(0.5, 1.0))``
    above the axes, which would otherwise be drawn off the fixed canvas.
    """
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h + rows * LEGEND_ROW_IN)
    ax.set_anchor("S")


def half_width() -> None:
    """Type and canvas settings for figures placed side by side.

    Call it once at import in a script whose figures go in a half-width slot.
    Scaling a full-width figure into that slot instead would put its 10 pt tick
    labels on the page at under 6 pt, and thin its 0.5 pt spines to a hairline.
    """
    full_width()          # the uncropped canvas, for the same reason
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # Spines and ticks at the default 0.5 print unevenly once this small.
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


def tex(s: str) -> str:
    """Escape the characters that break usetex in a run/gait name."""
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
             .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#"))
