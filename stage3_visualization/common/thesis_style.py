"""Shared matplotlib style for the thesis figures.

Importing this activates the ``science`` style with LaTeX text and defines the
palette. Colours and markers are keyed by gait name, so a gait looks the same
in every figure.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  registers the 'science' matplotlib style
from cycler import cycler

plt.style.use(["science"])
plt.rcParams["text.usetex"] = True

# Colourblind-safe categorical palette (checked in OKLab for common CVD types)
PALETTE = ("#0173B2", "#66A61E", "#B2182B", "#CC78BC", "#332288")
plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)
# Markers for greyscale print and weak CVD pairs
MARKERS = ("o", "s", "^", "D", "v")

# One colour per leg, in RobotSpec.leg_names order
LEG_COLORS = PALETTE[:4]

# Palette slot per gait; unknown gaits are appended in first-seen order.
GAIT_ORDER = ["LSPG25", "LSPG33", "TLPG50", "Prototype"]


def style_for(gait: str) -> tuple[str, str]:
    """``(colour, marker)`` for one gait, stable across figures and runs."""
    if gait not in GAIT_ORDER:
        GAIT_ORDER.append(gait)
    i = GAIT_ORDER.index(gait)
    return PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]


# Canvas sizes for figures included unscaled: full text width (15.2 cm) and
# 0.49 textwidth for side-by-side figures.
TEXT_WIDTH_IN = 5.98
HALF = (2.94, 2.2)

# Extra canvas height per legend row above the axes (not covered by tight_layout
# on a fixed canvas)
LEGEND_ROW_IN = 0.30


def full_width() -> None:
    """Disable tight cropping so saved figures keep their canvas size.

    Otherwise figures end up with different widths and font sizes in LaTeX.
    (``savefig(bbox_inches=None)`` would fall back to the rcParam.)
    """
    plt.rcParams["savefig.bbox"] = None


def legend_row(fig, ax, rows: int = 1) -> None:
    """Enlarge the canvas for a legend above the axes and anchor the axes at the bottom.

    Use with ``legend(loc="lower center", bbox_to_anchor=(0.5, 1.0))``.
    """
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h + rows * LEGEND_ROW_IN)
    ax.set_anchor("S")


def half_width() -> None:
    """Font and line settings for half-width (side-by-side) figures; call once at import."""
    full_width()
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # The default 0.5 prints unevenly at this size.
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


def tex(s: str) -> str:
    """Escape LaTeX special characters in a name."""
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
             .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#"))
