"""Read and write MATLAB Fuzzy Logic Toolbox ``.fis`` files.

``save_fis`` lets the reconstructed model be opened in MATLAB (``readfis``,
Fuzzy Logic Designer) for a side-by-side check; ``load_fis`` builds an exact
replica of an original MATLAB model. Only the features this model uses are
supported: Mamdani type, min/max/min/max/centroid, trimf/trapmf, AND rules.
"""

import re
from pathlib import Path

from .mamdani import MamdaniFIS, Rule, Variable
from .membership import MembershipFunction

_DEFAULTS = {"AndMethod": "min", "OrMethod": "max", "ImpMethod": "min", "AggMethod": "max", "DefuzzMethod": "centroid"}


def _fmt(v):
    return f"{v:g}"


def save_fis(fis: MamdaniFIS, path, name="traffic_congestion"):
    lines = [
        "[System]",
        f"Name='{name}'",
        "Type='mamdani'",
        "Version=2.0",
        f"NumInputs={len(fis.inputs)}",
        "NumOutputs=1",
        f"NumRules={len(fis.rules)}",
        *(f"{k}='{v}'" for k, v in _DEFAULTS.items()),
        "",
    ]
    sections = [(f"Input{i}", v) for i, v in enumerate(fis.inputs, 1)] + [("Output1", fis.output)]
    for header, var in sections:
        lines += [f"[{header}]", f"Name='{var.name}'", f"Range=[{_fmt(var.range[0])} {_fmt(var.range[1])}]",
                  f"NumMFs={len(var.terms)}"]
        for j, (term, mf) in enumerate(var.terms.items(), 1):
            lines.append(f"MF{j}='{term}':'{mf.kind}',[{' '.join(_fmt(p) for p in mf.params)}]")
        lines.append("")
    lines.append("[Rules]")
    for r in fis.rules:
        ante = " ".join("0" if t is None else str(v.index(t) + 1) for v, t in zip(fis.inputs, r.antecedents))
        lines.append(f"{ante}, {fis.output.index(r.consequent) + 1} ({_fmt(r.weight)}) : 1")
    Path(path).write_text("\n".join(lines) + "\n")


def _parse_sections(text):
    sections, current = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.fullmatch(r"\[(\w+)\]", line)
        if m:
            current = m.group(1)
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def _variable(lines):
    fields, mfs = {}, []
    for line in lines:
        key, _, value = line.partition("=")
        if key.startswith("MF"):
            m = re.fullmatch(r"'(.*)':'(\w+)',\[(.*)\]", value)
            if not m:
                raise ValueError(f"cannot parse membership function line: {line}")
            term, kind, params = m.groups()
            if kind not in ("trimf", "trapmf"):
                raise NotImplementedError(f"membership function type {kind!r} is not supported")
            mfs.append((term, MembershipFunction(kind, tuple(float(p) for p in params.split()))))
        else:
            fields[key] = value.strip("'")
    lo, hi = (float(v) for v in fields["Range"].strip("[]").split())
    return Variable(fields["Name"], (lo, hi), dict(mfs))


def load_fis(path, n_points=101) -> MamdaniFIS:
    sections = _parse_sections(Path(path).read_text())
    system = dict(line.split("=", 1) for line in sections["System"])
    system = {k: v.strip("'") for k, v in system.items()}
    if system.get("Type", "mamdani").lower() != "mamdani":
        raise NotImplementedError("only Mamdani systems are supported")
    for key, expected in _DEFAULTS.items():
        if system.get(key, expected) != expected:
            raise NotImplementedError(f"{key}={system[key]!r} is not supported (expected {expected!r})")
    inputs = [_variable(sections[f"Input{i}"]) for i in range(1, int(system["NumInputs"]) + 1)]
    output = _variable(sections["Output1"])
    rules = []
    for line in sections.get("Rules", []):
        m = re.fullmatch(r"([-\d\s]+),\s*([-\d]+)\s*\(([\d.eE+-]+)\)\s*:\s*(\d)", line)
        if not m:
            raise ValueError(f"cannot parse rule line: {line}")
        ante, out, weight, conn = m.groups()
        idx = [int(a) for a in ante.split()]
        if conn != "1" or any(i < 0 for i in idx) or int(out) <= 0:
            raise NotImplementedError(f"OR rules and NOT terms are not supported: {line}")
        terms = tuple(None if i == 0 else list(v.terms)[i - 1] for v, i in zip(inputs, idx))
        rules.append(Rule(terms, list(output.terms)[int(out) - 1], float(weight), source="fis"))
    return MamdaniFIS(inputs, output, rules, n_points=n_points)
