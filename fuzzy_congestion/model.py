"""The paper's Mamdani model: three inputs, one output, 75 rules (paper Section IV).

Where each number comes from
----------------------------
* Universes: Section V ("flow rate from 1 to 2000 vehicles per 60 minutes, length of
  the segments from 0.1 to 19 km, and number of lanes from 1 to 4 ... level of
  congestion ... in the range of 1 to 1875").
* Membership functions: supports are the "Numerical Ranges" of Tables I and II. The
  shoulders (where a term reaches full membership) are where its neighbours' ranges end,
  e.g. Flow "Low" = trapmf(200, 300, 500, 600) because "Very low" ends at 300 and
  "Average" starts at 500. The Lane terms do not overlap in Table I, so their shapes
  were measured from the rule viewer in Fig. 6. Every breakpoint was checked against
  pixel positions in Fig. 6.
* Rules: rules 1-30 are read from the rule viewer (Fig. 6), rules 50-75 from the rule
  editor (Fig. 5). Rules 31-49 are not shown in the paper; they were reconstructed
  from the regular pattern of the visible rules and checked against the control
  surfaces of Fig. 7 (see README, "Reconstruction notes"). ``Rule.source`` records
  which is which. Rule 76 is an addition that closes a coverage gap (see below).

If the original MATLAB ``.fis`` file is available, ``fisfile.load_fis`` builds an exact
replica of it instead.
"""

from .mamdani import MamdaniFIS, Rule, Variable
from .membership import MembershipFunction as MF

FLOW = Variable(
    "Flow",
    (1, 2000),
    {
        "V.L": MF("trapmf", (1, 1, 200, 300)),
        "LOW": MF("trapmf", (200, 300, 500, 600)),
        "Avg.": MF("trapmf", (500, 600, 800, 1000)),
        "High": MF("trapmf", (800, 1000, 1200, 1400)),
        "V.High": MF("trapmf", (1200, 1400, 2000, 2000)),
    },
    unit="veh/h",
)

LENGTH = Variable(
    "Length",
    (0.1, 19),
    {
        "V.Short": MF("trapmf", (0.1, 0.1, 1, 2)),
        "Short": MF("trapmf", (1, 2, 4, 6)),
        "Avg.": MF("trapmf", (4, 6, 8, 12)),
        "Long": MF("trapmf", (8, 12, 15, 17)),
        "V.Long": MF("trapmf", (15, 17, 19, 19)),
    },
    unit="km",
)

LANE = Variable(
    "Lane",
    (1, 4),
    {
        "Narrow": MF("trapmf", (1, 1, 1.5, 2)),
        "Average": MF("trimf", (1.5, 2, 3)),
        "Wide": MF("trapmf", (2.5, 3, 4, 4)),
    },
    unit="lanes",
)

LOC = Variable(
    "LOC",
    (1, 1875),
    {
        "C.C.free": MF("trapmf", (1, 1, 186, 280)),
        "C.free": MF("trapmf", (186, 280, 470, 580)),
        "Low": MF("trapmf", (470, 580, 750, 940)),
        "Stable": MF("trapmf", (750, 940, 1130, 1200)),
        "Near_C.": MF("trapmf", (1130, 1200, 1400, 1500)),
        "C.": MF("trapmf", (1400, 1500, 1600, 1650)),
        "Severe_C.": MF("trapmf", (1600, 1650, 2000, 2000)),
    },
    unit="veh",
)

INPUTS = (FLOW, LENGTH, LANE)

# Long names used in Tables I and II of the paper, keyed by the MATLAB term labels above.
TERM_NAMES = {
    "Flow": {"V.L": "Very low", "LOW": "Low", "Avg.": "Average", "High": "High", "V.High": "Very high"},
    "Length": {"V.Short": "Very short", "Short": "Short", "Avg.": "Average", "Long": "Long", "V.Long": "Very long"},
    "Lane": {"Narrow": "Narrow", "Average": "Average", "Wide": "Wide"},
    "LOC": {
        "C.C.free": "Completely congestion free",
        "C.free": "Congestion free",
        "Low": "Low",
        "Stable": "Stable",
        "Near_C.": "Near congestion",
        "C.": "Congestion",
        "Severe_C.": "Severe congestion",
    },
}

# The 75 rules in the order of the MATLAB rule list. Columns: flow, length, lane -> LOC, source.
#   fig6 = read from the rule viewer (Fig. 6), fig5 = read from the rule editor (Fig. 5),
#   reconstructed = not shown in the paper (see module docstring).
_RULE_TABLE = [
    # Flow = very low (rules 1-16; rules 1 and 2 are identical in Fig. 6)
    ("V.L", "V.Short", "Narrow", "C.free", "fig6"),
    ("V.L", "V.Short", "Narrow", "C.free", "fig6"),
    ("V.L", "Short", "Narrow", "C.free", "fig6"),
    ("V.L", "Avg.", "Narrow", "C.free", "fig6"),
    ("V.L", "Long", "Narrow", "C.free", "fig6"),
    ("V.L", "V.Long", "Narrow", "C.free", "fig6"),
    ("V.L", "V.Short", "Average", "C.C.free", "fig6"),
    ("V.L", "Short", "Average", "C.C.free", "fig6"),
    ("V.L", "Avg.", "Average", "C.C.free", "fig6"),
    ("V.L", "Long", "Average", "C.C.free", "fig6"),
    ("V.L", "V.Long", "Average", "C.C.free", "fig6"),
    ("V.L", "V.Short", "Wide", "C.C.free", "fig6"),
    ("V.L", "Short", "Wide", "C.C.free", "fig6"),
    ("V.L", "Avg.", "Wide", "C.C.free", "fig6"),
    ("V.L", "Long", "Wide", "C.C.free", "fig6"),
    ("V.L", "V.Long", "Wide", "C.C.free", "fig6"),
    # Flow = low (rules 17-31)
    ("LOW", "V.Short", "Narrow", "Low", "fig6"),
    ("LOW", "Short", "Narrow", "Low", "fig6"),
    ("LOW", "Avg.", "Narrow", "Low", "fig6"),
    ("LOW", "Long", "Narrow", "Low", "fig6"),
    ("LOW", "V.Long", "Narrow", "Low", "fig6"),
    ("LOW", "V.Short", "Average", "C.free", "fig6"),
    ("LOW", "Short", "Average", "C.free", "fig6"),
    ("LOW", "Avg.", "Average", "C.free", "fig6"),
    ("LOW", "Long", "Average", "C.free", "fig6"),
    ("LOW", "V.Long", "Average", "C.free", "fig6"),
    ("LOW", "V.Short", "Wide", "C.C.free", "fig6"),
    ("LOW", "Short", "Wide", "C.C.free", "fig6"),
    ("LOW", "Avg.", "Wide", "C.C.free", "fig6"),
    ("LOW", "Long", "Wide", "C.C.free", "fig6"),
    ("LOW", "V.Long", "Wide", "C.C.free", "reconstructed"),
    # Flow = average (rules 32-45)
    ("Avg.", "V.Short", "Narrow", "Near_C.", "reconstructed"),
    ("Avg.", "Short", "Narrow", "Near_C.", "reconstructed"),
    ("Avg.", "Avg.", "Narrow", "Stable", "reconstructed"),
    ("Avg.", "Long", "Narrow", "Low", "reconstructed"),
    ("Avg.", "V.Long", "Narrow", "Low", "reconstructed"),
    ("Avg.", "V.Short", "Average", "Near_C.", "reconstructed"),
    ("Avg.", "Short", "Average", "Stable", "reconstructed"),
    ("Avg.", "Avg.", "Average", "Low", "reconstructed"),
    ("Avg.", "Long", "Average", "C.free", "reconstructed"),
    ("Avg.", "V.Long", "Average", "C.free", "reconstructed"),
    ("Avg.", "V.Short", "Wide", "Low", "reconstructed"),
    ("Avg.", "Short", "Wide", "C.free", "reconstructed"),
    ("Avg.", "Avg.", "Wide", "C.free", "reconstructed"),
    ("Avg.", "Long", "Wide", "C.C.free", "reconstructed"),
    # Flow = high (rules 46-60)
    ("High", "V.Short", "Narrow", "C.", "reconstructed"),
    ("High", "Short", "Narrow", "C.", "reconstructed"),
    ("High", "Avg.", "Narrow", "Near_C.", "reconstructed"),
    ("High", "Long", "Narrow", "Stable", "reconstructed"),
    ("High", "V.Long", "Narrow", "Stable", "fig5"),
    ("High", "V.Short", "Average", "C.", "fig5"),
    ("High", "Short", "Average", "Near_C.", "fig5"),
    ("High", "Avg.", "Average", "Stable", "fig5"),
    ("High", "Long", "Average", "Low", "fig5"),
    ("High", "V.Long", "Average", "Low", "fig5"),
    ("High", "V.Short", "Wide", "Stable", "fig5"),
    ("High", "Short", "Wide", "Low", "fig5"),
    ("High", "Avg.", "Wide", "Low", "fig5"),
    ("High", "Long", "Wide", "C.free", "fig5"),
    ("High", "V.Long", "Wide", "C.free", "fig5"),
    # Flow = very high (rules 61-75)
    ("V.High", "V.Short", "Narrow", "Severe_C.", "fig5"),
    ("V.High", "Short", "Narrow", "Severe_C.", "fig5"),
    ("V.High", "Avg.", "Narrow", "C.", "fig5"),
    ("V.High", "Long", "Narrow", "Near_C.", "fig5"),
    ("V.High", "V.Long", "Narrow", "Near_C.", "fig5"),
    ("V.High", "V.Short", "Average", "Severe_C.", "fig5"),
    ("V.High", "Short", "Average", "C.", "fig5"),
    ("V.High", "Avg.", "Average", "Near_C.", "fig5"),
    ("V.High", "Long", "Average", "Stable", "fig5"),
    ("V.High", "V.Long", "Average", "Stable", "fig5"),
    ("V.High", "V.Short", "Wide", "Near_C.", "fig5"),
    ("V.High", "Short", "Wide", "Stable", "fig5"),
    ("V.High", "Avg.", "Wide", "Stable", "fig5"),
    ("V.High", "Long", "Wide", "Low", "fig5"),
    ("V.High", "V.Long", "Wide", "Low", "fig5"),
    # Rule 76 is not in the paper. Rules 1 and 2 are identical, so the paper's 75 rules can
    # cover only 74 of the 5 x 5 x 3 = 75 input combinations; without this rule no rule fires
    # for (Avg. flow, V.Long, Wide) and the output falls back to the middle of the range (938).
    ("Avg.", "V.Long", "Wide", "C.C.free", "added"),
]

RULES = tuple(Rule((f, le, la), out, source=src) for f, le, la, out, src in _RULE_TABLE)
PAPER_RULES = RULES[:75]  # exactly the paper's 75 rules, including the coverage gap


def build_model(rules=RULES, n_points=101) -> MamdaniFIS:
    """The congestion model with MATLAB's default 101-point centroid defuzzification."""
    return MamdaniFIS(INPUTS, LOC, rules, n_points=n_points)


def classify(loc_value, output=LOC) -> str:
    """Linguistic level of a crisp LOC value: the output term with the highest membership.

    This is how the paper labels its example (LOC = 281 -> "Congestion free").
    """
    mu = output.fuzzify(loc_value)
    return list(output.terms)[int(mu.argmax())]


def rule_text(rule: Rule, number=None) -> str:
    """A rule in MATLAB rule-editor syntax, e.g. as in Fig. 5."""
    parts = [f"({v.name} is {t})" for v, t in zip(INPUTS, rule.antecedents) if t is not None]
    head = f"{number}. " if number is not None else ""
    return f"{head}If {' and '.join(parts)} then ({LOC.name} is {rule.consequent}) ({rule.weight:g})"
