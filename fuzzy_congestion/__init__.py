"""Python replication of Amini, Hatwagner, Mikulai and Koczy (2021),
"An intelligent traffic congestion detection approach based on fuzzy inference system",
IEEE SACI 2021, doi:10.1109/SACI51354.2021.9465637.
"""

from .mamdani import Inference, MamdaniFIS, Rule, Variable
from .membership import MembershipFunction, trapmf, trimf
from .model import (
    FLOW, INPUTS, LANE, LENGTH, LOC, PAPER_RULES, RULES, TERM_NAMES, build_model, classify, rule_text,
)

__all__ = [
    "FLOW", "INPUTS", "LANE", "LENGTH", "LOC", "PAPER_RULES", "RULES", "TERM_NAMES",
    "Inference", "MamdaniFIS", "MembershipFunction", "Rule", "Variable",
    "build_model", "classify", "rule_text", "trapmf", "trimf",
]
