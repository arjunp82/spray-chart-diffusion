"""Spray chart plotting utilities."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# MLB-style 7-zone definitions in normalised [-1, 1] space.
# Zones: LF, LC, CF, RC, RF (outfield) + INF (infield) + SHORT (shallow infield)
# Layout mirrors the image the user showed: wedge split into depth × lateral bands.
# ---------------------------------------------------------------------------

# Depth bands (y axis, outfield = positive y)
_Y_SHORT  = (0.00, 0.20)   # short/infield
_Y_INF    = (0.20, 0.45)   # mid infield / shallow outfield
_Y_OUT    = (0.45, 1.05)   # outfield

# Lateral bands for outfield (5 zones)
_X_LF     = (-1.00, -0.50)
_X_LC     = (-0.50, -0.17)
_X_CF     = (-0.17,  0.17)
_X_RC     = ( 0.17,  0.50)
_X_RF     = ( 0.50,  1.00)

# Lateral bands for infield (4 zones matching MLB chart)
_X_INF_3B  = (-1.00, -0.38)   # 3rd base side
_X_INF_LC  = (-0.38,  0.00)   # left-center infield
_X_INF_RC  = ( 0.00,  0.38)   # right-center infield
_X_INF_1B  = ( 0.38,  1.00)   # 1st base side

_IMG = 64
_LIN = np.linspace(-1, 1, _IMG)
_GX, _GY = np.meshgrid(_LIN, _LIN[::-1])


def _make_zone_masks() -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    # Outfield: 5 lateral bands in the deep band
    for name, (x0, x1) in zip(
        ("LF", "LC", "CF", "RC", "RF"),
        (_X_LF, _X_LC, _X_CF, _X_RC, _X_RF),
    ):
        masks[name] = (
            (_GX >= x0) & (_GX < x1) &
            (_GY >= _Y_OUT[0]) & (_GY < _Y_OUT[1])
        )
    # Infield: 4 lateral zones
    inf_depth = (_GY >= _Y_INF[0]) & (_GY < _Y_INF[1])
    for name, (x0, x1) in zip(
        ("INF_3B", "INF_LC", "INF_RC", "INF_1B"),
        (_X_INF_3B, _X_INF_LC, _X_INF_RC, _X_INF_1B),
    ):
        masks[name] = inf_depth & (_GX >= x0) & (_GX < x1)
    # Keep a combined INF mask for backwards compatibility
    masks["INF"] = inf_depth
    # Short/bunt zone
    masks["SHORT"] = (_GY >= _Y_SHORT[0]) & (_GY < _Y_SHORT[1])
    return masks


MLB_ZONE_MASKS: dict[str, np.ndarray] = _make_zone_masks()

# Label positions (x, y) in normalised coords for each zone annotation
_ZONE_LABEL_POS: dict[str, tuple[float, float]] = {
    "LF":    (-0.72,  0.72),
    "LC":    (-0.33,  0.72),
    "CF":    ( 0.00,  0.80),
    "RC":    ( 0.33,  0.72),
    "RF":    ( 0.72,  0.72),
    "INF":   ( 0.00,  0.32),
    "SHORT": ( 0.00,  0.10),
}

# Custom baseball-field colormap: white → light green → dark green → yellow → red
_FIELD_CMAP = LinearSegmentedColormap.from_list(
    "spray", ["#ffffff", "#d4edda", "#28a745", "#ffc107", "#dc3545"]
)


def draw_field_outline(ax: plt.Axes, color: str = "#555555", lw: float = 1.5) -> None:
    """Draw a simple baseball fair-territory wedge and infield diamond."""
    theta = np.linspace(np.pi * 0.25, np.pi * 0.75, 100)
    ax.plot(np.cos(theta), np.sin(theta), color=color, lw=lw)
    # Foul lines
    ax.plot([0, -1.0], [0, 1.0], color=color, lw=lw)
    ax.plot([0, 1.0], [0, 1.0], color=color, lw=lw)
    # Diamond (rough approximation in normalised coords)
    diamond_x = [0, -0.14, 0, 0.14, 0]
    diamond_y = [0, 0.14, 0.28, 0.14, 0]
    ax.plot(diamond_x, diamond_y, color=color, lw=lw * 0.8)
    # Pitcher's mound
    mound = plt.Circle((0, 0.175), 0.025, color=color, fill=False, lw=lw * 0.6)
    ax.add_patch(mound)


def plot_spray_chart(
    density: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "",
    show_field: bool = True,
    cmap=None,
) -> plt.Axes:
    """
    Plot a 64×64 density as a spray chart heat-map.

    density: (64, 64) normalised probability density.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    if cmap is None:
        cmap = _FIELD_CMAP

    nonzero = density[density > 0]
    vmax = float(np.percentile(nonzero, 98)) if nonzero.size > 0 else 1e-6
    ax.imshow(
        density,
        extent=[-1, 1, -1, 1],
        origin="upper",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        aspect="equal",
    )
    if show_field:
        draw_field_outline(ax)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.15, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    return ax


def overlay_zone_pcts(
    density: np.ndarray,
    ax: plt.Axes,
    fontsize: int = 9,
    color: str = "white",
) -> dict[str, float]:
    """
    Compute per-zone integrated probability from a 64×64 density and
    annotate each zone on ax with its percentage.

    Returns the zone -> pct dict for downstream use.
    """
    pcts: dict[str, float] = {}
    for name, mask in MLB_ZONE_MASKS.items():
        pcts[name] = float(density[mask].sum() * 100)

    for name, (lx, ly) in _ZONE_LABEL_POS.items():
        ax.text(
            lx, ly, f"{pcts[name]:.0f}%",
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.35, lw=0),
        )
    return pcts


def plot_comparison(
    generated: np.ndarray,
    actual: np.ndarray,
    batter_label: str = "",
    figsize: tuple[int, int] = (8, 4),
) -> plt.Figure:
    """Side-by-side comparison: generated vs. actual 2023 spray chart."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_spray_chart(generated, ax=axes[0], title=f"Generated — {batter_label}")
    plot_spray_chart(actual, ax=axes[1], title=f"Actual 2023 — {batter_label}")
    fig.tight_layout()
    return fig


def plot_calibration_diagram(
    pa_thresholds: list[int],
    model_coverage: list[float],
    kde_coverage: list[float],
    confidence_level: float = 0.80,
    figsize: tuple[int, int] = (6, 4),
) -> plt.Figure:
    """Reliability diagram: coverage vs. PA count for model and KDE baselines."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(pa_thresholds, model_coverage, "o-", label="Diffusion", color="#0066cc")
    ax.plot(pa_thresholds, kde_coverage, "s--", label="Situational KDE", color="#cc6600")
    ax.axhline(confidence_level, color="gray", linestyle=":", label=f"Target {confidence_level:.0%}")
    ax.set_xlabel("PA count (k)")
    ax.set_ylabel(f"{confidence_level:.0%} credible region coverage")
    ax.set_title("Calibration: coverage vs. available data")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_kl_vs_pa(
    results: dict[str, dict[int, float]],
    figsize: tuple[int, int] = (6, 4),
) -> plt.Figure:
    """
    Plot KL divergence vs. PA count for multiple methods.
    results: {method_name -> {pa_count -> kl_value}}
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#0066cc", "#cc0000", "#cc6600", "#228B22", "#9932CC"]
    for (name, kl_dict), color in zip(results.items(), colors):
        ks = sorted(kl_dict.keys())
        vals = [kl_dict[k] for k in ks]
        ax.plot(ks, vals, "o-", label=name, color=color)
    ax.set_xlabel("Available plate appearances (k)")
    ax.set_ylabel("KL divergence (↓ better)")
    ax.set_title("Sparse-data stress test")
    ax.legend()
    fig.tight_layout()
    return fig
