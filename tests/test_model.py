import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuzzy_congestion import (
    FLOW, LANE, LENGTH, LOC, PAPER_RULES, RULES, MembershipFunction, Rule, Variable, MamdaniFIS,
    build_model, classify, trapmf, trimf,
)
from fuzzy_congestion.fisfile import load_fis, save_fis


# Membership functions (Eqs. 5 and 6)

def test_trimf_matches_eq5():
    x = np.array([0, 1, 2, 3, 4, 5, 6])
    assert np.allclose(trimf(x, 1, 3, 5), [0, 0, 0.5, 1, 0.5, 0, 0])


def test_trapmf_matches_eq6_and_shoulders():
    x = np.array([0, 1, 2, 3, 4, 5, 6])
    assert np.allclose(trapmf(x, 1, 2, 4, 5), [0, 0, 1, 1, 1, 0, 0])
    assert np.allclose(trapmf([1, 200, 250, 300], 1, 1, 200, 300), [1, 1, 0.5, 0])   # left shoulder
    assert np.allclose(trapmf([1400, 1700, 2000], 1200, 1400, 2000, 2000), [1, 1, 1])  # right shoulder


def test_input_terms_cover_their_universe():
    for var in (FLOW, LENGTH, LANE, LOC):
        x = np.linspace(*var.range, 2001)
        assert var.fuzzify(x).max(axis=0).min() > 0, var.name


def test_bad_parameters_are_rejected():
    with pytest.raises(ValueError):
        MembershipFunction("trimf", (3, 2, 1))
    with pytest.raises(ValueError):
        MembershipFunction("gaussmf", (1, 2))


# The worked example of Section V / Fig. 6

def test_paper_example():
    fis = build_model()
    res = fis.infer([253, 5.16, 2])
    assert res.output == pytest.approx(279.59, abs=0.01)
    assert [k + 1 for k in np.flatnonzero(res.strengths > 0)] == [8, 9, 23, 24]
    assert classify(res.output) == "C.free"
    # The rule viewer rounds the displayed input; Flow ~ 253.7 gives the published 281.
    assert fis.evaluate([253.7, 5.16, 2]) == pytest.approx(281, abs=0.5)


# Rule base

def test_rule_sources():
    sources = [r.source for r in RULES]
    assert len(PAPER_RULES) == 75 and len(RULES) == 76
    assert sources[:30] == ["fig6"] * 30
    assert sources[30:49] == ["reconstructed"] * 19
    assert sources[49:75] == ["fig5"] * 26
    assert sources[75] == "added"


@pytest.mark.parametrize("number, antecedents, consequent", [
    (50, ("High", "V.Long", "Narrow"), "Stable"),
    (51, ("High", "V.Short", "Average"), "C."),
    (59, ("High", "Long", "Wide"), "C.free"),
    (61, ("V.High", "V.Short", "Narrow"), "Severe_C."),
    (68, ("V.High", "Avg.", "Average"), "Near_C."),
    (75, ("V.High", "V.Long", "Wide"), "Low"),
])
def test_rules_transcribed_from_fig5(number, antecedents, consequent):
    rule = RULES[number - 1]
    assert rule.antecedents == antecedents and rule.consequent == consequent


def test_rule_coverage():
    combos = set(itertools.product(FLOW.terms, LENGTH.terms, LANE.terms))
    assert {r.antecedents for r in RULES} == combos
    missing = combos - {r.antecedents for r in PAPER_RULES}
    assert missing == {("Avg.", "V.Long", "Wide")}


def test_no_rule_fired_falls_back_to_mid_range():
    paper = build_model(PAPER_RULES)
    assert paper.evaluate([700, 18, 3.5]) == pytest.approx(np.mean(LOC.range))


# Behaviour claimed in Section V

def test_three_or_four_lanes_never_severe():
    fis = build_model()
    g = np.stack(np.meshgrid(np.linspace(1, 2000, 40), np.linspace(0.1, 19, 40), np.linspace(3, 4, 5)), -1)
    loc = fis.evaluate(g.reshape(-1, 3))
    assert all(classify(y) != "Severe_C." for y in loc)


def test_loc_rises_with_flow_for_narrow_short_segments():
    fis = build_model()
    flows = np.linspace(1, 2000, 50)
    loc = fis.evaluate(np.column_stack([flows, np.full(50, 1.0), np.full(50, 1.0)]))
    assert np.all(np.diff(loc) >= -1e-9)


# Engine details

def test_batch_equals_single():
    fis = build_model()
    X = np.random.default_rng(1).uniform([1, 0.1, 1], [2000, 19, 4], (50, 3))
    assert np.allclose(fis.evaluate(X), [fis.evaluate(x) for x in X])


def test_none_antecedent_is_ignored():
    x = Variable("x", (0, 1), {"a": MembershipFunction("trimf", (0, 0, 1))})
    y = Variable("y", (0, 1), {"b": MembershipFunction("trimf", (0, 1, 1))})
    z = Variable("z", (0, 1), {"c": MembershipFunction("trimf", (0, 0.5, 1))})
    fis = MamdaniFIS([x, y], z, [Rule(("a", None), "c")])
    assert fis.firing_strengths([0.25, 0.0])[0, 0] == pytest.approx(0.75)


def test_fis_round_trip(tmp_path):
    fis = build_model()
    path = tmp_path / "model.fis"
    save_fis(fis, path)
    again = load_fis(path)
    X = np.random.default_rng(0).uniform([1, 0.1, 1], [2000, 19, 4], (500, 3))
    assert np.allclose(fis.evaluate(X), again.evaluate(X))
