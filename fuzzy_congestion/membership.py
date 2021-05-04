"""Membership functions used by the model (paper Eqs. 5 and 6).

Both functions follow the MATLAB Fuzzy Logic Toolbox definitions of
``trimf`` and ``trapmf`` exactly, including the degenerate "shoulder"
cases (a == b or c == d), so results can be compared one-to-one with
the original MATLAB model.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def trimf(x, a, b, c):
    """Triangular membership function, paper Eq. (5).

    In the paper's notation a = alpha_min, b = beta, c = alpha_max::

        mu(x) = 0                         x <= alpha_min
                (x - alpha_min)/(beta - alpha_min)   alpha_min < x < beta
                1                         x == beta
                (alpha_max - x)/(alpha_max - beta)   beta < x < alpha_max
                0                         x >= alpha_max
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if a != b:
        m = (a < x) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    if b != c:
        m = (b < x) & (x < c)
        y[m] = (c - x[m]) / (c - b)
    y[x == b] = 1.0
    return y


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function, paper Eq. (6).

    In the paper's notation a = alpha_min, b = beta_1, c = beta_2,
    d = alpha_max. The function is the minimum of a rising ramp
    (a -> b) and a falling ramp (c -> d); when a == b (or c == d) the
    corresponding ramp is a vertical step, which gives the left (right)
    "shoulder" shapes used for the outermost linguistic terms.
    """
    x = np.asarray(x, dtype=float)
    rise = np.ones_like(x)
    rise[x < a] = 0.0
    if a != b:
        m = (a <= x) & (x < b)
        rise[m] = (x[m] - a) / (b - a)
    fall = np.ones_like(x)
    fall[x > d] = 0.0
    if c != d:
        m = (c < x) & (x <= d)
        fall[m] = (d - x[m]) / (d - c)
    return np.minimum(rise, fall)


_KINDS = {"trimf": (trimf, 3), "trapmf": (trapmf, 4)}


@dataclass(frozen=True)
class MembershipFunction:
    """A named-shape membership function, e.g. ``MembershipFunction("trapmf", (1, 1, 200, 300))``."""

    kind: str
    params: Tuple[float, ...]

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported membership function {self.kind!r}")
        if len(self.params) != _KINDS[self.kind][1]:
            raise ValueError(f"{self.kind} needs {_KINDS[self.kind][1]} parameters, got {self.params}")
        if list(self.params) != sorted(self.params):
            raise ValueError(f"{self.kind} parameters must be non-decreasing, got {self.params}")

    def __call__(self, x):
        return _KINDS[self.kind][0](x, *self.params)

    @property
    def support(self):
        return self.params[0], self.params[-1]
