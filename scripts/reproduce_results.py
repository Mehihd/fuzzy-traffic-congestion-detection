"""Recompute the paper's numerical results and check the claims made in Section V.

    python scripts/reproduce_results.py

Prints a report and writes the Fig. 7 surface grids to results/fig7_surface_*.csv.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fuzzy_congestion import PAPER_RULES, TERM_NAMES, build_model, classify

OUT = ROOT / "results"


def flow_step_increase(fis, lengths, lanes, flows, step=200):
    """Relative LOC increase when flow rises by ``step``, over a grid of (flow, length, lane)."""
    F, Le, La = (a.ravel() for a in np.meshgrid(flows, lengths, lanes))
    before = fis.evaluate(np.column_stack([F, Le, La]))
    after = fis.evaluate(np.column_stack([F + step, Le, La]))
    return after / before - 1


def main():
    fis = build_model()
    names = TERM_NAMES["LOC"]

    print("1. Worked example (Section V, Fig. 6): Flow = 253, Length = 5.16 km, Lane = 2")
    for flow in (253.0, 253.7):
        y = fis.evaluate([flow, 5.16, 2])
        print(f"   Flow = {flow:<6} -> LOC = {y:7.2f}  ({names[classify(y)]})")
    res = fis.infer([253, 5.16, 2])
    fired = ", ".join(f"{k + 1} (w={res.strengths[k]:.2f})" for k in np.flatnonzero(res.strengths > 0))
    print(f"   fired rules: {fired}   [paper Fig. 6: 8, 9, 23, 24]")
    print("   paper: LOC = 281, 'congestion free'\n")

    print("2. Control surfaces (Fig. 7), other input at mid-range, 15 x 15 grid")
    OUT.mkdir(exist_ok=True)
    surfaces = {"I": (1, 2), "II": (1, 0), "III": (0, 2)}
    for name, (ix, iy) in surfaces.items():
        X, Y, Z = fis.surface(ix, iy)
        xv, yv = fis.inputs[ix].name, fis.inputs[iy].name
        print(f"   {name:<3} {xv} x {yv}: LOC {Z.min():7.1f} .. {Z.max():7.1f}")
        rows = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        np.savetxt(OUT / f"fig7_surface_{name}.csv", rows, delimiter=",", fmt="%.4f",
                   header=f"{xv},{yv},LOC", comments="")
    print("   paper (read off the plots): I ~380..~1530, II ~150..~1700, III ~150..~1450\n")

    print("3. Claims in Section V")
    # a) Length transitions = where neighbouring Length terms overlap.
    transitions = [(1, 2), (4, 6), (8, 12), (15, 17)]
    print("   a) Fig. 7-I (Flow = 1000.5): LOC change across each Length transition   [paper: largest at ~4-6 km, lane 2-3]")
    for lane in (2.0, 2.5, 3.0):
        drops = [fis.evaluate([1000.5, b, lane]) - fis.evaluate([1000.5, a, lane]) for a, b in transitions]
        cells = "  ".join(f"{a}-{b} km: {d:+5.0f}" for (a, b), d in zip(transitions, drops))
        a, b = transitions[int(np.argmin(drops))]
        print(f"      lane {lane}: {cells}   -> largest at {a}-{b} km")

    r = flow_step_increase(fis, np.linspace(1, 6, 26), [2.5], np.linspace(200, 1800, 81))
    print(f"   b) Fig. 7-II (Lane = 2.5), Length 1-6 km, Flow 200-1800: +200 veh/h raises LOC by "
          f"median {100 * np.median(r):.0f}%, >= 50% in {100 * (r >= 0.5).mean():.0f}% of cases, "
          f"at most {100 * r.max():.0f}%   [paper: ~50%]")

    grid = np.stack(np.meshgrid(np.linspace(1, 2000, 80), np.linspace(0.1, 19, 80), np.linspace(3, 4, 11)), -1)
    top = fis.evaluate(grid.reshape(-1, 3)).max()
    print(f"   c) Lanes 3-4: highest LOC anywhere = {top:.0f} ({names[classify(top)]})"
          "   [paper: no severe congestion with 3 or 4 lanes]")

    r = flow_step_increase(fis, [9.55], np.linspace(1, 2, 11), np.linspace(1, 1800, 90))
    print(f"   d) Fig. 7-III (Length = 9.55), Lane 1-2: +200 veh/h raises LOC by median {100 * np.median(r):.0f}%, "
          f">= 50% in {100 * (r >= 0.5).mean():.0f}% of cases, at most {100 * r.max():.0f}%   [paper: can exceed 50%]\n")

    print("4. Rule coverage")
    paper = build_model(PAPER_RULES)
    point = [700, 18, 3.5]
    print(f"   paper's 75 rules at Flow=700, Length=18, Lane=3.5: no rule fires -> LOC = {paper.evaluate(point):.1f}"
          f" (mid-range fallback); with rule 76 -> {fis.evaluate(point):.1f}")


if __name__ == "__main__":
    main()
