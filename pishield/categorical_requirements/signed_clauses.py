from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable


@dataclass(frozen=True)
class SignedLiteral:
    var: int
    values: frozenset[int]

    def __repr__(self) -> str:
        vs = ",".join(str(v) for v in sorted(self.values))
        return "[" + vs + "]" + f":y_{self.var}"


@dataclass(frozen=True)
class SignedClause:
    literals: tuple[SignedLiteral, ...]

    @classmethod
    def from_literals(cls, lits: Iterable[SignedLiteral]) -> "SignedClause":
        """Build a clause, merging duplicate variables (S1 ∪ S2 : Y)."""
        merged: dict[int, frozenset[int]] = {}
        for lit in lits:
            merged[lit.var] = merged.get(lit.var, frozenset()) | lit.values
        items = sorted(merged.items())
        return cls(tuple(SignedLiteral(v, s) for v, s in items))

    @cached_property
    def variables(self) -> frozenset[int]:
        # Cached: the compiler queries this in per-elimination-step scans, so
        # rebuilding the frozenset on every access is quadratic overall.
        # (cached_property stores via instance __dict__, which works on frozen
        # dataclasses; equality and hash still use only `literals`.)
        return frozenset(lit.var for lit in self.literals)

    def get(self, var: int) -> frozenset[int] | None:
        for lit in self.literals:
            if lit.var == var:
                return lit.values
        return None

    def drop(self, var: int) -> "SignedClause":
        return SignedClause(tuple(lit for lit in self.literals if lit.var != var))

    def __repr__(self) -> str:
        if not self.literals:
            return "FALSE"
        return " or ".join(repr(lit) for lit in self.literals)


@dataclass
class ConstraintSet:
    var_names: list[str]
    var_domains: list[int]
    clauses: list[SignedClause]
    ordering: list[int]
    value_base: int = 0


class ParseError(Exception):
    pass


_VAR_RE = re.compile(r"^var\s+([A-Za-z_]\w*)\s+(\d+)\s*$")
_VALUE_BASE_RE = re.compile(r"^value-base\s+(\d+)\s*$")
_ORDER_RE = re.compile(r"^ordering\s+(.+)$")
_LITERAL_RE = re.compile(r"^\s*\[\s*([\d,\s]*)\s*\]\s*:\s*([A-Za-z_]\w*)\s*$")

_RESERVED_VAR_NAMES = frozenset({"or"})

_OR_SPLIT_RE = re.compile(r"\s+or\s+")


def parse_constraints(text: str) -> ConstraintSet:
    var_names: list[str] = []
    var_domains: list[int] = []
    var_index: dict[str, int] = {}
    clauses: list[SignedClause] = []
    ordering: list[int] | None = None
    value_base: int = 0
    value_base_seen: bool = False

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        m = _VALUE_BASE_RE.match(line)
        if m:
            if value_base_seen:
                raise ParseError(f"line {lineno}: value-base already declared")
            if clauses:
                raise ParseError(
                    f"line {lineno}: value-base must appear before any clauses"
                )
            value_base = int(m.group(1))
            value_base_seen = True
            continue

        m = _VAR_RE.match(line)
        if m:
            name, size = m.group(1), int(m.group(2))
            if name in _RESERVED_VAR_NAMES:
                raise ParseError(
                    f"line {lineno}: {name!r} is a reserved keyword and "
                    f"cannot be used as a variable name"
                )
            if name in var_index:
                raise ParseError(f"line {lineno}: variable {name!r} redeclared")
            if size <= 0:
                raise ParseError(
                    f"line {lineno}: variable {name!r} domain size must be positive"
                )
            var_index[name] = len(var_names)
            var_names.append(name)
            var_domains.append(size)
            continue

        m = _ORDER_RE.match(line)
        if m:
            if ordering is not None:
                raise ParseError(f"line {lineno}: ordering already declared")
            names = [tok for tok in re.split(r"[,\s]+", m.group(1).strip()) if tok]
            try:
                ordering = [var_index[n] for n in names]
            except KeyError as e:
                raise ParseError(
                    f"line {lineno}: unknown variable {e.args[0]!r} in ordering"
                )
            continue

        if line.startswith("clause ") or line == "clause":
            raise ParseError(
                f"line {lineno}: `clause` keyword is no longer used; "
                f"write the clause body directly (e.g. `[0]:y_2 or [0]:y_3`)"
            )
        if line.startswith("order ") or line == "order":
            raise ParseError(
                f"line {lineno}: `order` keyword has been renamed to "
                f"`ordering` (e.g. `ordering y_1 y_2 y_3`)"
            )
        if "|" in line:
            raise ParseError(
                f"line {lineno}: `|` is no longer used; write `or` "
                f"between literals (e.g. `[0]:y_2 or [0]:y_3`)"
            )

        lits: list[SignedLiteral] = []
        for piece in _OR_SPLIT_RE.split(line):
            lm = _LITERAL_RE.match(piece)
            if not lm:
                raise ParseError(
                    f"line {lineno}: malformed literal {piece.strip()!r}"
                )
            vals_str, name = lm.group(1), lm.group(2)
            if name not in var_index:
                raise ParseError(f"line {lineno}: unknown variable {name!r}")
            var = var_index[name]
            domain = var_domains[var]
            if vals_str.strip():
                raw_vals = [v for v in re.split(r"[,\s]+", vals_str.strip()) if v]
                parsed = []
                for raw_v in raw_vals:
                    iv = int(raw_v)
                    shifted = iv - value_base
                    if not (0 <= shifted < domain):
                        raise ParseError(
                            f"line {lineno}: value {iv} out of range "
                            f"[{value_base},{value_base + domain}) for {name!r}"
                        )
                    parsed.append(shifted)
                values = frozenset(parsed)
            else:
                values = frozenset()
            lits.append(SignedLiteral(var, values))
        clauses.append(SignedClause.from_literals(lits))

    if not var_names:
        raise ParseError("no variables declared")

    if ordering is None:
        ordering = list(range(len(var_names)))

    if sorted(ordering) != list(range(len(var_names))):
        raise ParseError(
            "ordering must be a permutation of every declared variable"
        )

    return ConstraintSet(
        var_names=var_names,
        var_domains=var_domains,
        clauses=clauses,
        ordering=ordering,
        value_base=value_base,
    )


def clause_satisfied(clause: SignedClause, assignment: dict[int, int]) -> bool:
    """True iff some signed literal of `clause` is satisfied by `assignment`."""
    for lit in clause.literals:
        if lit.var in assignment and assignment[lit.var] in lit.values:
            return True
    return False


def constraints_satisfied(
    clauses: Iterable[SignedClause], assignment: dict[int, int]
) -> bool:
    return all(clause_satisfied(c, assignment) for c in clauses)
