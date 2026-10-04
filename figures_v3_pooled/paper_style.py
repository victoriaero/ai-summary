"""Shared visual system for the IASEAI paper figures.

Every figure is drawn at its final printed size for the AAAI two-column
template, so text renders at the stated point sizes with no LaTeX scaling:

    single column  ->  \\includegraphics[width=\\columnwidth]{...}
    full width     ->  \\includegraphics[width=\\textwidth]{...}   (figure*)

One palette, with one meaning, in every figure:

    COMPARISON (blue)    query naming the majority group  (e.g. "White people")
    FOCAL      (orange)  query naming the minority group  (e.g. "Black people")
    GENERIC    (grey)    the unmarked query ("people") and whatever is shared
                         with it

Blue and orange pass the dataviz palette validator on white (worst CVD
dE 18.0, normal-vision dE 26.6, contrast >= 3:1). Everything else is ink.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib import transforms
from matplotlib.patches import Circle, Wedge


# ============================================================
# PATHS
# ============================================================

FIGURE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = FIGURE_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"


# ============================================================
# PAGE GEOMETRY (AAAI: \textwidth = 7.0in, \columnsep = 0.375in)
# ============================================================

TEXT_WIDTH = 7.0
COLUMN_WIDTH = 3.3125


# ============================================================
# PALETTE
# ============================================================

COMPARISON = "#3368A0"
COMPARISON_TINT = "#DEE7F0"

FOCAL = "#C8562C"
FOCAL_TINT = "#F6E4DD"

GENERIC = "#7D828A"
GENERIC_TINT = "#E9EAEC"

CONDITION = {
    "majority": {"color": COMPARISON, "tint": COMPARISON_TINT},
    "minority": {"color": FOCAL, "tint": FOCAL_TINT},
    "generic": {"color": GENERIC, "tint": GENERIC_TINT},
}

INK = "#1E2127"        # primary text, statistical overlays
INK_SOFT = "#4A4F57"   # axis labels, tick labels
MUTED = "#7A7F87"      # notes, secondary annotations
RULE = "#B4B8BE"       # axes and reference lines
GRID = "#E7E8EB"       # hairline grid
BAND = "#F3F4F6"       # background bands
WHITE = "#FFFFFF"


# ============================================================
# TYPOGRAPHY (points; body text of the paper is 10 pt)
# ============================================================

FS_TICK = 6.5
FS_LABEL = 7.0
FS_TITLE = 7.5
FS_NOTE = 6.3


def use_paper_style():

    mpl.rcParams.update({

        # Liberation Sans is metric-compatible with Arial and ships as
        # TrueType, so it embeds as Type 42 (AAAI rejects Type 3 fonts).
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Liberation Sans",
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "mathtext.sf": "Liberation Sans",
        "mathtext.cal": "Liberation Sans:italic",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,

        "text.color": INK,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.6,
        "axes.facecolor": "none",
        "axes.unicode_minus": True,
        "axes.labelpad": 3.0,

        "xtick.color": RULE,
        "ytick.color": RULE,
        "xtick.labelcolor": INK_SOFT,
        "ytick.labelcolor": INK_SOFT,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,

        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "figure.dpi": 150,
    })


# ============================================================
# SOCIAL DIMENSIONS
# ============================================================

DIMENSIONS = [
    "Race",
    "Ethnicity",
    "Gender",
    "Disability",
    "Sexual Orientation",
    "Gender Identity",
]

DIMENSION_LABELS = {
    "Race": "Race",
    "Ethnicity": "Ethnicity",
    "Gender": "Gender",
    "Disability": "Disability",
    "Sexual Orientation": "Sexual orientation",
    "Gender Identity": "Gender identity",
}


# ============================================================
# LAYOUT HELPERS
# ============================================================

def add_axes_in(fig, left, bottom, width, height, **kwargs):
    """Add axes positioned in inches from the figure's lower-left corner."""

    fig_w, fig_h = fig.get_size_inches()

    return fig.add_axes(
        [left / fig_w, bottom / fig_h, width / fig_w, height / fig_h],
        **kwargs,
    )


def fig_x(fig, inches):
    """Horizontal position in figure fraction."""

    return inches / fig.get_size_inches()[0]


def fig_y(fig, inches):
    """Vertical position in figure fraction."""

    return inches / fig.get_size_inches()[1]


def fig_x_data_y(fig, ax):
    """Transform: x in figure fraction, y in the data coordinates of `ax`."""

    return transforms.blended_transform_factory(fig.transFigure, ax.transData)


def offset(transform, fig, x_pt=0.0, y_pt=0.0):
    """`transform` shifted by a fixed distance in printed points."""

    return transforms.offset_copy(transform, fig=fig, x=x_pt, y=y_pt, units="points")


def strip_axes(ax, keep=()):
    """Hide every spine not listed in `keep`."""

    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


# ============================================================
# OVERLAP GLYPH (Figures 1 and 3)
#
# A ring in the colour of the condition; the grey wedge, clockwise from
# 12 o'clock, is the share of its cited-source set that overlaps the generic
# query's source set (URL Jaccard similarity). The glyph is sized in printed
# points, so it stays round whatever the axes' aspect ratio.
# ============================================================

def overlap_glyph(ax, x, y, share, condition, diameter=8.0, ring=1.0, zorder=3):

    fig = ax.figure
    colors = CONDITION[condition]

    place = (
        transforms.Affine2D().scale(1 / 72)
        + fig.dpi_scale_trans
        + transforms.ScaledTranslation(x, y, ax.transData)
    )

    radius = diameter / 2

    patches = [
        Circle((0, 0), radius, facecolor=colors["tint"], edgecolor="none"),
    ]

    if share > 0:
        patches.append(
            Wedge(
                (0, 0),
                radius,
                90 - 360 * min(share, 1.0),
                90,
                facecolor=GENERIC,
                edgecolor="none",
            )
        )

    patches.append(
        Circle(
            (0, 0),
            radius,
            facecolor="none",
            edgecolor=colors["color"],
            linewidth=ring,
        )
    )

    for level, patch in enumerate(patches):
        patch.set_transform(place)
        patch.set_zorder(zorder + level * 0.01)
        patch.set_clip_on(False)
        ax.add_patch(patch)


# ============================================================
# STATISTICS HELPERS
# ============================================================

N_BOOTSTRAP = 20000
SEED = 42


def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, seed=SEED):
    """95% percentile CI of a mean, resampling matched outcomes."""

    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    index = rng.integers(0, len(values), size=(n_boot, len(values)))
    low, high = np.percentile(values[index].mean(axis=1), [2.5, 97.5])

    return float(low), float(high)


def format_p(p, label="p"):
    """APA-style p value: no leading zero, '< .001' floor."""

    if p < 0.001:
        return f"{label} < .001"

    return f"{label} = {p:.3f}".replace("0.", ".", 1)


def format_signed(value, decimals):

    return f"{value:+.{decimals}f}".replace("-", "−")


def association_text(r, ci_low, ci_high, p_holm):

    return (
        rf"$r_\mathrm{{blocked}}$ = {r:.2f}, "
        f"95% CI [{ci_low:.2f}, {ci_high:.2f}], "
        + format_p(p_holm, r"$p_\mathrm{Holm}$")
    )


# ============================================================
# OUTPUT
# ============================================================

def save(fig, stem):
    """Write vector PDF (for LaTeX) and a 600 dpi PNG preview."""

    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"

    fig.savefig(pdf_path, metadata={"CreationDate": None})
    fig.savefig(png_path, dpi=600)

    width, height = fig.get_size_inches()

    print(f"\nSaved ({width:.4g} x {height:.4g} in):")
    print(f"  {pdf_path}")
    print(f"  {png_path}")


def save_to(fig, output_dir, stem):
    """Write vector PDF + 600 dpi PNG to an explicit output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, metadata={"CreationDate": None})
    fig.savefig(png_path, dpi=600)
    width, height = fig.get_size_inches()
    print(f"\nSaved ({width:.4g} x {height:.4g} in):")
    print(f"  {pdf_path}")
    print(f"  {png_path}")
    return pdf_path, png_path
