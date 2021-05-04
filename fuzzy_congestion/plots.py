"""Figures for the model: membership functions, rule viewer, inference trace, control surfaces.

Colours: one blue for data marks, an orange accent for crisp input/output values, and a
single-hue blue ramp (light = little congestion, dark = severe) wherever colour encodes LOC.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
# Sequential blue ramp, 7 steps for the 7 LOC levels (light -> dark).
LOC_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LOC_CMAP = LinearSegmentedColormap.from_list("loc", LOC_RAMP)
LOC_LEVELS = ListedColormap(LOC_RAMP)


def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def _peak(mf):
    p = mf.params
    return p[1] if mf.kind == "trimf" else (p[1] + p[2]) / 2


def plot_membership_functions(var, ax=None, value=None, label_terms=True):
    """All terms of one variable; ``value`` marks a crisp input and where it meets each term."""
    ax = ax or plt.gca()
    x = np.linspace(*var.range, 1000)
    for term, mf in var.terms.items():
        mu = mf(x)
        # Hide the flat zero stretches so neighbouring terms do not pile up on the baseline.
        near = np.convolve(mu > 0, np.ones(3), mode="same") > 0
        ax.plot(x, np.where(near, mu, np.nan), color=BLUE, lw=1.6)
        if label_terms:
            ax.text(np.clip(_peak(mf), *var.range), 1.06, term, ha="center", va="bottom", fontsize=8, color=INK)
    if value is not None:
        ax.axvline(value, color=ORANGE, lw=1.4)
        for mf in var.terms.values():
            m = float(mf(value))
            if m > 0:
                ax.plot([value], [m], "o", ms=6, color=ORANGE, mec=SURFACE, mew=1.5, zorder=5)
    ax.set_xlim(*var.range)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("membership μ")
    ax.set_xlabel(f"{var.name} ({var.unit})" if var.unit else var.name)
    ax.grid(axis="y")
    return ax


def plot_rule_viewer(fis, x, rules=None, fig=None):
    """MATLAB-style rule viewer: one row per rule, one column per variable, aggregate at the bottom."""
    res = fis.infer(x)
    rules = list(range(len(fis.rules))) if rules is None else list(rules)
    n_rows = len(rules) + 1
    fig = fig or plt.figure(figsize=(9, 0.26 * n_rows + 0.9))
    gs = fig.add_gridspec(n_rows, 5, width_ratios=[0.25, 1, 1, 1, 1], hspace=0.25, wspace=0.12,
                          left=0.02, right=0.99, top=1 - 0.45 / fig.get_figheight(), bottom=0.35 / fig.get_figheight())
    variables = fis.inputs + [fis.output]
    grids = [np.linspace(*v.range, 400) for v in variables]
    for r, k in enumerate(rules):
        rule = fis.rules[k]
        ax = fig.add_subplot(gs[r, 0])
        ax.axis("off")
        ax.text(1, 0.5, str(k + 1), ha="right", va="center", fontsize=7, color=INK_2)
        for c, var in enumerate(variables):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(GRID)
            ax.set_xlim(*var.range); ax.set_ylim(0, 1.05)
            term = rule.antecedents[c] if c < len(fis.inputs) else rule.consequent
            if term is not None:
                g = grids[c]
                mu = var.terms[term](g)
                level = (res.memberships[c][var.index(term)] if c < len(fis.inputs) else res.strengths[k])
                ax.plot(g, mu, color=MUTED, lw=0.8)
                if level > 0:
                    ax.fill_between(g, np.minimum(mu, level), color=BLUE, alpha=0.45 if c < len(fis.inputs) else 0.85, lw=0)
            if c < len(fis.inputs):
                ax.axvline(x[c], color=ORANGE, lw=0.9)
            if r == 0:
                label = f"{var.name} = {x[c]:g}" if c < len(fis.inputs) else f"{var.name} = {res.output:.1f}"
                ax.set_title(label, fontsize=9, pad=4)
    ax = fig.add_subplot(gs[-1, 4])
    ax.fill_between(res.z, res.aggregated, color=BLUE, alpha=0.85, lw=0, step=None)
    ax.axvline(res.output, color=ORANGE, lw=1.4)
    ax.set_xlim(*fis.output.range); ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("aggregated output and centroid", fontsize=8)
    return fig, res


def plot_surface(fis, ix, iy, ax, azim, n=15, reference=None, zlim=None):
    """Control surface like MATLAB gensurf; ``azim`` uses matplotlib's convention (MATLAB az - 90)."""
    X, Y, Z = fis.surface(ix, iy, n=n, reference=reference)
    ax.set_proj_type("ortho")
    ax.plot_surface(X, Y, Z, cmap=LOC_CMAP, vmin=fis.output.range[0], vmax=fis.output.range[1],
                    edgecolor=INK, linewidth=0.35, antialiased=True, shade=False)
    ax.view_init(elev=30, azim=azim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.988, 0.988, 0.984, 1.0))
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.6
        axis.line.set_color(AXIS)
    ax.tick_params(labelsize=7, pad=1)
    xv, yv = fis.inputs[ix], fis.inputs[iy]
    ax.set_xlabel(f"{xv.name} ({xv.unit})", labelpad=2)
    ax.set_ylabel(f"{yv.name} ({yv.unit})", labelpad=2)
    ax.set_zlabel(fis.output.name, labelpad=2)
    if zlim:
        ax.set_zlim(*zlim)
    return X, Y, Z
