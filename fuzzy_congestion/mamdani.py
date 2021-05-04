"""A small Mamdani fuzzy inference engine (paper Section III, Eqs. 3, 4 and 7).

The engine reproduces MATLAB's default Mamdani settings, which the paper
used (Fuzzy Logic Toolbox R2020b):

    AND method      = min            (combines the antecedents of a rule)
    Implication     = min            (clips the consequent at the firing strength)
    Aggregation     = max            (unions the clipped consequents)
    Defuzzification = centroid (COA) evaluated on 101 points of the output range
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .membership import MembershipFunction


@dataclass
class Variable:
    """A linguistic variable: a name, a numeric universe and ordered linguistic terms."""

    name: str
    range: Tuple[float, float]
    terms: Dict[str, MembershipFunction]
    unit: str = ""

    def fuzzify(self, x):
        """Membership degree of ``x`` in every term, shape ``(len(terms),) + shape(x)``."""
        return np.stack([mf(x) for mf in self.terms.values()])

    def index(self, term):
        return list(self.terms).index(term)


@dataclass(frozen=True)
class Rule:
    """``IF in_1 is A_1 AND ... AND in_r is A_r THEN out is B`` (paper Eq. 3).

    ``antecedents`` holds one term name per input, in input order; ``None``
    means the input is not used by the rule (MATLAB's "none").
    """

    antecedents: Tuple[Optional[str], ...]
    consequent: str
    weight: float = 1.0
    source: str = ""


@dataclass
class Inference:
    """Every intermediate quantity of one Mamdani evaluation, for plotting and explanation."""

    inputs: np.ndarray            # crisp inputs, shape (n_inputs,)
    memberships: list             # per input: membership in each of its terms
    strengths: np.ndarray         # firing strength w_k of every rule, shape (n_rules,)
    z: np.ndarray                 # output sample points, shape (n_points,)
    clipped: np.ndarray           # min(w_k, mu_Bk(z)), shape (n_rules, n_points)
    aggregated: np.ndarray        # max over rules, shape (n_points,)
    output: float                 # centroid of the aggregated set (Eq. 7)


class MamdaniFIS:
    def __init__(self, inputs: Sequence[Variable], output: Variable, rules: Sequence[Rule], n_points: int = 101):
        self.inputs = list(inputs)
        self.output = output
        self.rules = list(rules)
        self.n_points = n_points
        self._check_rules()
        # Antecedent term index per rule and input (-1 = not used), consequent index per rule.
        self._ante = np.array(
            [[-1 if t is None else v.index(t) for v, t in zip(self.inputs, r.antecedents)] for r in self.rules]
        )
        self._cons = np.array([output.index(r.consequent) for r in self.rules])
        self._weights = np.array([r.weight for r in self.rules], dtype=float)
        self.z = np.linspace(*output.range, n_points)
        self._out_mf = output.fuzzify(self.z)  # (n_terms, n_points)

    def _check_rules(self):
        for k, r in enumerate(self.rules, 1):
            if len(r.antecedents) != len(self.inputs):
                raise ValueError(f"rule {k}: expected {len(self.inputs)} antecedents, got {len(r.antecedents)}")
            for v, t in zip(self.inputs, r.antecedents):
                if t is not None and t not in v.terms:
                    raise ValueError(f"rule {k}: {t!r} is not a term of {v.name}")
            if r.consequent not in self.output.terms:
                raise ValueError(f"rule {k}: {r.consequent!r} is not a term of {self.output.name}")

    def firing_strengths(self, X):
        """Firing strength of every rule for every input row: ``min`` over the used antecedents.

        X has shape (n_samples, n_inputs); the result has shape (n_samples, n_rules).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        w = np.ones((X.shape[0], len(self.rules)))
        for i, var in enumerate(self.inputs):
            mu = var.fuzzify(X[:, i])  # (n_terms, n_samples)
            used = self._ante[:, i] >= 0
            w[:, used] = np.minimum(w[:, used], mu[self._ante[used, i]].T)
        return w * self._weights

    def evaluate(self, X):
        """Crisp output for one input vector (returns a float) or many (returns an array)."""
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        w = self.firing_strengths(X)  # (n_samples, n_rules)
        # Implication (min) and aggregation (max), Eq. (4): for each consequent term keep the
        # strongest rule, then clip that term and take the union over terms.
        n_terms = len(self.output.terms)
        level = np.zeros((w.shape[0], n_terms))
        for t in range(n_terms):
            sel = self._cons == t
            if sel.any():
                level[:, t] = w[:, sel].max(axis=1)
        agg = np.minimum(level[:, :, None], self._out_mf[None]).max(axis=1)  # (n_samples, n_points)
        y = self._centroid(agg)
        return float(y[0]) if single else y

    def _centroid(self, agg):
        """Centre of area, Eq. (7), as a discrete sum over the output sample points."""
        area = agg.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            y = (agg * self.z).sum(axis=-1) / area
        # MATLAB returns the middle of the output range when no rule fires.
        return np.where(area > 0, y, np.mean(self.output.range))

    def infer(self, x) -> Inference:
        """Evaluate one input vector and keep every intermediate step."""
        x = np.asarray(x, dtype=float)
        w = self.firing_strengths(x)[0]
        clipped = np.minimum(w[:, None], self._out_mf[self._cons])
        agg = clipped.max(axis=0)
        return Inference(
            inputs=x,
            memberships=[v.fuzzify(xi) for v, xi in zip(self.inputs, x)],
            strengths=w,
            z=self.z,
            clipped=clipped,
            aggregated=agg,
            output=float(self._centroid(agg[None])[0]),
        )

    def surface(self, ix: int, iy: int, n: int = 15, reference=None):
        """Control surface over inputs ``ix`` and ``iy``, like MATLAB ``gensurf``.

        The other inputs are held at ``reference`` (default: the middle of
        their range, which is also MATLAB's default). Returns (X, Y, Z) grids.
        """
        ref = np.array([np.mean(v.range) for v in self.inputs] if reference is None else reference, dtype=float)
        xs = np.linspace(*self.inputs[ix].range, n)
        ys = np.linspace(*self.inputs[iy].range, n)
        X, Y = np.meshgrid(xs, ys)
        pts = np.tile(ref, (X.size, 1))
        pts[:, ix], pts[:, iy] = X.ravel(), Y.ravel()
        return X, Y, self.evaluate(pts).reshape(X.shape)
