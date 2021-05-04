"""Regenerate every figure of the paper from the Python model.

    python scripts/make_figures.py            # writes PNGs into figures/

Fig. 3 (the network map) is a photograph of a map, not model output, so only the
original image is used for it (figures/original/fig3_network_map.png).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fuzzy_congestion import FLOW, LANE, LENGTH, LOC, MamdaniFIS, MembershipFunction, Rule, Variable, build_model
from fuzzy_congestion.plots import (
    AXIS, BLUE, INK, INK_2, LOC_RAMP, MUTED, ORANGE, SURFACE,
    plot_membership_functions, plot_rule_viewer, plot_surface, use_style,
)

OUT = ROOT / "figures"
SAMPLE = (253, 5.16, 2)  # the worked example of Section V / Fig. 6
DPI = 200


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def _box(ax, x, y, w, h, text, fill=SURFACE, edge=AXIS, size=9, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02", fc=fill, ec=edge, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, weight=weight)


def _arrow(ax, p, q, text=None, color=INK_2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1, color=color, shrinkA=0, shrinkB=0))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.025, text, ha="center", va="bottom", fontsize=7, color=MUTED)


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0.3, 0.68), 0.4, 0.28, fc="none", ec=AXIS, lw=1, ls=(0, (4, 3))))
    ax.text(0.5, 0.92, "Knowledge base", ha="center", va="center", fontsize=9, color=INK_2)
    _box(ax, 0.33, 0.72, 0.15, 0.13, "Fuzzy sets\n(Tables I, II)", fill="#eef4fc", size=8)
    _box(ax, 0.52, 0.72, 0.15, 0.13, "Rule base\n(75 if-then rules)", fill="#eef4fc", size=8)
    _box(ax, 0.12, 0.22, 0.17, 0.2, "Fuzzification", fill="#eef4fc")
    _box(ax, 0.415, 0.22, 0.17, 0.2, "Fuzzy inference\nengine", fill="#eef4fc")
    _box(ax, 0.71, 0.22, 0.17, 0.2, "Defuzzification\n(centroid)", fill="#eef4fc")
    ax.text(0.02, 0.32, "Crisp\ninput", ha="center", va="center", fontsize=8, color=INK_2)
    ax.text(0.98, 0.32, "Crisp\noutput", ha="center", va="center", fontsize=8, color=INK_2)
    _arrow(ax, (0.05, 0.32), (0.12, 0.32))
    _arrow(ax, (0.29, 0.32), (0.415, 0.32), "fuzzy")
    _arrow(ax, (0.585, 0.32), (0.71, 0.32), "fuzzy")
    _arrow(ax, (0.88, 0.32), (0.95, 0.32))
    _arrow(ax, (0.405, 0.72), (0.22, 0.42))
    _arrow(ax, (0.5, 0.68), (0.5, 0.42))
    _arrow(ax, (0.595, 0.72), (0.78, 0.42))
    save(fig, "fig1_fis_architecture.png")


def fig2_max_min():
    """A generic two-rule Mamdani system evaluated with the same engine (Eqs. 3, 4 and 7)."""
    X = Variable("x", (0, 10), {"A1": MembershipFunction("trimf", (1, 3, 5.5)), "A2": MembershipFunction("trapmf", (3, 5, 7, 9.5))})
    Y = Variable("y", (0, 10), {"B1": MembershipFunction("trimf", (2, 4.5, 7)), "B2": MembershipFunction("trapmf", (3.5, 6, 8, 10))})
    Z = Variable("z", (0, 10), {"C1": MembershipFunction("trimf", (0.5, 3, 5.5)), "C2": MembershipFunction("trapmf", (4, 6, 8, 10))})
    fis = MamdaniFIS([X, Y], Z, [Rule(("A1", "B1"), "C1"), Rule(("A2", "B2"), "C2")], n_points=501)
    x0 = (4.4, 5.2)
    res = fis.infer(x0)
    fig = plt.figure(figsize=(8, 5.2))
    gs = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.3)
    for r, rule in enumerate(fis.rules):
        for c, var in enumerate([X, Y, Z]):
            ax = fig.add_subplot(gs[r, c])
            g = np.linspace(*var.range, 500)
            term = rule.antecedents[c] if c < 2 else rule.consequent
            mu = var.terms[term](g)
            ax.plot(g, mu, color=BLUE, lw=1.6)
            if c < 2:
                m = res.memberships[c][var.index(term)]
                ax.axvline(x0[c], color=ORANGE, lw=1.2)
                ax.plot([x0[c]], [m], "o", ms=6, color=ORANGE, mec=SURFACE, mew=1.5, zorder=5)
                ax.axhline(m, color=MUTED, lw=0.8, ls=(0, (3, 2)))
                note = f"μ = {m:.2f}"
            else:
                w = res.strengths[r]
                ax.fill_between(g, np.minimum(mu, w), color=BLUE, alpha=0.35, lw=0)
                ax.axhline(w, color=MUTED, lw=0.8, ls=(0, (3, 2)))
                note = f"clipped at w{r + 1} = min = {w:.2f}"
            ax.set_ylim(0, 1.15); ax.set_xlim(*var.range); ax.set_yticks([0, 1])
            head = f"Rule {r + 1}: {var.name} is {term}" if c == 0 else f"{var.name} is {term}"
            ax.set_title(f"{head}   ({note})", fontsize=8.5, loc="left")
    ax = fig.add_subplot(gs[2, 2])
    ax.fill_between(res.z, res.aggregated, color=BLUE, alpha=0.35, lw=0)
    ax.plot(res.z, res.aggregated, color=BLUE, lw=1.6)
    ax.axvline(res.output, color=ORANGE, lw=1.4)
    ax.text(res.output + 0.2, 1.0, f"z* = {res.output:.2f}", fontsize=8, color=INK, va="top")
    ax.set_ylim(0, 1.15); ax.set_xlim(0, 10); ax.set_yticks([0, 1])
    ax.set_title("Aggregation (max) → centroid", fontsize=8.5, loc="left")
    ax = fig.add_subplot(gs[2, :2]); ax.axis("off")
    ax.text(0, 0.75, "Rule 1: IF x is A1 AND y is B1 THEN z is C1\nRule 2: IF x is A2 AND y is B2 THEN z is C2",
            fontsize=8.5, color=INK, va="top", family="monospace")
    ax.text(0, 0.05, f"inputs x = {x0[0]}, y = {x0[1]}  →  output z* = {res.output:.2f}", fontsize=8.5, color=INK_2)
    save(fig, "fig2_max_min_composition.png")


def fig4_schematic(fis):
    fig = plt.figure(figsize=(8, 3.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for i, var in enumerate(fis.inputs):
        y0 = 0.7 - i * 0.3
        sub = fig.add_axes([0.04, y0 + 0.04, 0.2, 0.16])
        plot_membership_functions(var, sub, label_terms=False)
        sub.set_xticks([]); sub.set_yticks([]); sub.set_xlabel(""); sub.set_ylabel(""); sub.grid(False)
        sub.set_title(f"{var.name} ({len(var.terms)} terms)", fontsize=8, pad=2)
        _arrow(ax, (0.25, y0 + 0.12), (0.4, 0.5))
    n_paper = sum(r.source != "added" for r in fis.rules)
    _box(ax, 0.4, 0.32, 0.2, 0.36,
         f"Mamdani fuzzy\ninference system\n\n{n_paper} rules (+{len(fis.rules) - n_paper} added)\nmin / max / centroid",
         fill="#eef4fc", size=9)
    _arrow(ax, (0.6, 0.5), (0.72, 0.5))
    sub = fig.add_axes([0.73, 0.4, 0.24, 0.2])
    plot_membership_functions(fis.output, sub, label_terms=False)
    sub.set_xticks([]); sub.set_yticks([]); sub.set_xlabel(""); sub.set_ylabel(""); sub.grid(False)
    sub.set_title("Level_of_Congestion (7 terms)", fontsize=8, pad=2)
    save(fig, "fig4_model_schematic.png")


def fig_membership(fis):
    fig, axes = plt.subplots(4, 1, figsize=(8, 8.4))
    for ax, var in zip(axes, fis.inputs + [fis.output]):
        plot_membership_functions(var, ax)
    fig.tight_layout(h_pad=1.2)
    save(fig, "membership_functions.png")


def fig5_rule_base(fis):
    lanes, lengths, flows = list(LANE.terms), list(LENGTH.terms), list(FLOW.terms)
    loc_terms = list(LOC.terms)
    fig, ax = plt.subplots(figsize=(12, 4.4))
    gap = 0.35
    for k, rule in enumerate(fis.rules):
        f, le, la = rule.antecedents
        if k == 1:  # rule 2 duplicates rule 1; the cell shows rule 1
            continue
        col = lanes.index(la) * (len(lengths) + gap) + lengths.index(le)
        row = flows.index(f)
        level = loc_terms.index(rule.consequent)
        ax.add_patch(Rectangle((col + 0.03, row + 0.03), 0.94, 0.94, fc=LOC_RAMP[level], ec="none"))
        mark = {"reconstructed": " †", "added": " +"}.get(rule.source, "")
        ax.text(col + 0.5, row + 0.6, rule.consequent, ha="center", va="center", fontsize=7.5,
                color="#ffffff" if level >= 3 else INK)
        ax.text(col + 0.5, row + 0.3, f"#{k + 1}" + (",2" if k == 0 else "") + mark, ha="center", va="center",
                fontsize=6.5, color="#ffffff" if level >= 3 else INK_2)
    width = len(lanes) * len(lengths) + (len(lanes) - 1) * gap
    ax.set_xlim(0, width); ax.set_ylim(0, len(flows))
    ax.set_yticks(np.arange(len(flows)) + 0.5, [f"Flow {t}" for t in flows])
    xt = [lanes.index(la) * (len(lengths) + gap) + j + 0.5 for la in lanes for j in range(len(lengths))]
    ax.set_xticks(xt, lengths * len(lanes), fontsize=7.5)
    for i, la in enumerate(lanes):
        ax.text(i * (len(lengths) + gap) + len(lengths) / 2, len(flows) + 0.15, f"Lane {la}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Length term (within each lane group)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.text(0, -0.95, "Cell = consequent (LOC term) and rule number.  † reconstructed, not shown in the paper.  "
            "+ added to close the coverage gap.  Rules 1 and 2 are identical.", transform=ax.transData,
            fontsize=7.5, color=INK_2)
    save(fig, "fig5_rule_base.png")


def fig6(fis):
    fig, res = plot_rule_viewer(fis, SAMPLE, rules=range(30))
    save(fig, "fig6_rule_viewer.png")

    fig = plt.figure(figsize=(8, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.6, wspace=0.28)
    for c, var in enumerate(fis.inputs):
        ax = fig.add_subplot(gs[0, c])
        plot_membership_functions(var, ax, value=SAMPLE[c], label_terms=False)
        degrees = ", ".join(f"{t} {m:.2f}" for t, m in zip(var.terms, res.memberships[c]) if m > 0)
        ax.set_title(f"{var.name} = {SAMPLE[c]:g}\n{degrees}", fontsize=8.5, loc="left")
        if c:
            ax.set_ylabel("")
    ax = fig.add_subplot(gs[1, :])
    fired = np.flatnonzero(res.strengths > 0)
    for k in fired:
        ax.plot(res.z, res.clipped[k], color=MUTED, lw=0.9, ls=(0, (3, 2)))
    ax.fill_between(res.z, res.aggregated, color=BLUE, alpha=0.25, lw=0)
    ax.plot(res.z, res.aggregated, color=BLUE, lw=1.8)
    ax.axvline(res.output, color=ORANGE, lw=1.4)
    ax.text(res.output + 15, 0.92, f"centroid z* = {res.output:.1f}  (paper: 281)", fontsize=8.5, color=INK, va="top")
    lines = [f"rule {k + 1}: w = {res.strengths[k]:.2f} → {fis.rules[k].consequent}" for k in fired]
    ax.text(0.99, 0.95, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK_2)
    ax.set_xlim(1, 1000); ax.set_ylim(0, 1.05)
    ax.set_xlabel("LOC (output universe 1–1875, first 1000 shown)"); ax.set_ylabel("membership μ")
    ax.set_title("Clipped consequents (dashed), their max-aggregation (blue) and centroid", fontsize=9, loc="left")
    ax.grid(axis="y")
    save(fig, "fig6_inference_trace.png")


# (x input, y input, matplotlib azimuth) chosen to match the camera of each panel of Fig. 7.
SURFACES = {
    "I": (1, 2, 52.5),     # Length x Lane, Flow held at 1000.5
    "II": (1, 0, -52.5),   # Length x Flow, Lane held at 2.5
    "III": (0, 2, 127.5),  # Flow x Lane, Length held at 9.55
}


def fig7(fis):
    for name, (ix, iy, azim) in SURFACES.items():
        fig = plt.figure(figsize=(6.4, 4.4))
        ax = fig.add_subplot(projection="3d")
        ax.set_box_aspect((1.5, 1.5, 0.9))
        X, Y, Z = plot_surface(fis, ix, iy, ax, azim)
        held = [v for i, v in enumerate(fis.inputs) if i not in (ix, iy)][0]
        ax.set_title(f"{name}: {fis.inputs[ix].name} × {fis.inputs[iy].name}  "
                     f"({held.name} = {np.mean(held.range):g}; LOC {Z.min():.0f}–{Z.max():.0f})", fontsize=9)
        save(fig, f"fig7_surface_{name}.png")


def main():
    use_style()
    OUT.mkdir(exist_ok=True)
    fis = build_model()
    fig1_architecture()
    fig2_max_min()
    fig4_schematic(fis)
    fig_membership(fis)
    fig5_rule_base(fis)
    fig6(fis)
    fig7(fis)


if __name__ == "__main__":
    main()
