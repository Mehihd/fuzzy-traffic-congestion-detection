"""Command-line use.

    python -m fuzzy_congestion 253 5.16 2            # one segment: flow, length (km), lanes
    python -m fuzzy_congestion 253 5.16 2 --trace    # also list the rules that fired
    python -m fuzzy_congestion --csv segments.csv --out loc.csv
    python -m fuzzy_congestion 253 5.16 2 --fis original.fis

The CSV needs the columns ``flow``, ``length_km`` and ``lanes``; the output adds
``loc`` and ``level``. Inputs are clipped to the model's ranges: outside them no
membership function is non-zero, no rule fires and the output would fall back
to the middle of the LOC range.
"""

import argparse
import csv
import sys

import numpy as np

from .fisfile import load_fis
from .model import TERM_NAMES, build_model, classify, rule_text


def clip_to_ranges(fis, X):
    lo, hi = np.array([v.range for v in fis.inputs]).T
    clipped = np.clip(X, lo, hi)
    n = int(np.any(clipped != X, axis=-1).sum())
    if n:
        print(f"note: {n} input row(s) clipped to the model ranges", file=sys.stderr)
    return clipped


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m fuzzy_congestion", description="Fuzzy level-of-congestion model")
    p.add_argument("values", nargs="*", type=float, metavar="FLOW LENGTH LANES")
    p.add_argument("--csv", help="input CSV with columns flow, length_km, lanes")
    p.add_argument("--out", help="output CSV (default: stdout)")
    p.add_argument("--fis", help="use a MATLAB .fis file instead of the reconstructed model")
    p.add_argument("--trace", action="store_true", help="print the rules that fired")
    args = p.parse_args(argv)

    fis = load_fis(args.fis) if args.fis else build_model()

    if args.csv:
        with open(args.csv, newline="") as f:
            rows = list(csv.DictReader(f))
        X = np.array([[float(r["flow"]), float(r["length_km"]), float(r["lanes"])] for r in rows])
        locs = np.atleast_1d(fis.evaluate(clip_to_ranges(fis, X)))
        out = open(args.out, "w", newline="") if args.out else sys.stdout
        writer = csv.DictWriter(out, fieldnames=list(rows[0]) + ["loc", "level"])
        writer.writeheader()
        for r, y in zip(rows, locs):
            writer.writerow({**r, "loc": f"{y:.2f}", "level": TERM_NAMES["LOC"].get(classify(y, fis.output), classify(y, fis.output))})
        if args.out:
            out.close()
        return

    if len(args.values) != 3:
        p.error("give FLOW LENGTH LANES, or --csv")
    res = fis.infer(clip_to_ranges(fis, np.array(args.values)))
    level = classify(res.output, fis.output)
    print(f"LOC = {res.output:.2f}  ->  {TERM_NAMES['LOC'].get(level, level)}")
    if args.trace:
        for k in np.flatnonzero(res.strengths > 0):
            print(f"  w = {res.strengths[k]:.3f}   {rule_text(fis.rules[k], k + 1)}")


if __name__ == "__main__":
    main()
