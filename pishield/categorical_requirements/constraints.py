from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from resolution import (
    ClauseFeasibility,
    CompiledConstraints,
    compile_constraints,
    from_json as _resolution_from_json,
)
from signed_clauses import parse_constraints


class Constraints:
    """INFER feasibility object — one type, two construction modes.

    `Constraints.from_file(path)` reads a signed-clause DSL and walks the
    compiled per-variable clauses at runtime; use for clause-shaped
    constraints (Sudoku ALLDIFFERENT, propositional rules).

    `Constraints.from_function(fn, var_domains, ordering=None)` wraps a
    Python callable that evaluates S_i(a_<i) directly; use when the
    signed-CNF representation would explode (MNIST-Sum, MNIST-Add carry chain).
    """

    def __init__(self, _impl=None):
        if _impl is None:
            raise TypeError(
                "Use Constraints.from_file(path) or "
                "Constraints.from_function(fn, var_domains) to construct."
            )
        self._impl = _impl

    @property
    def var_domains(self) -> list[int]:
        return self._impl.var_domains

    @property
    def ordering(self) -> list[int]:
        return self._impl.ordering

    @property
    def neighbours(self) -> dict[int, frozenset[int]]:
        """Variable -> the set of other variables it shares a constraint with."""
        return self._impl.neighbours

    @property
    def ordering_levels(self) -> list[list[int]]:
        """Greedy levelisation of `ordering` under the constraint graph.

        Variables in the same level share no constraint and can be projected
        simultaneously.
        """
        return compute_levels(self.ordering, self.neighbours)

    def feasible_set(
        self, step: int, prior_assignment: dict[int, int]
    ) -> frozenset[int]:
        """Compute S_i(a_<i) ⊆ [h_i]."""
        return self._impl.feasible_set(step, prior_assignment)

    @classmethod
    def from_file(cls, path: str | Path) -> "Constraints":
        """Load from a DSL `.txt` (parsed + compiled) or compiled `.json`."""
        path = Path(path)
        if path.suffix == ".json":
            cc = _resolution_from_json(json.loads(path.read_text()))
        else:
            cs = parse_constraints(path.read_text())
            cc = compile_constraints(cs)
        return cls(_impl=ClauseFeasibility(cc))

    @classmethod
    def from_text(cls, text: str) -> "Constraints":
        """Parse a DSL string in memory and compile."""
        cs = parse_constraints(text)
        cc = compile_constraints(cs)
        return cls(_impl=ClauseFeasibility(cc))

    @classmethod
    def from_compiled(cls, compiled: CompiledConstraints) -> "Constraints":
        """Wrap an already-compiled `CompiledConstraints` object."""
        return cls(_impl=ClauseFeasibility(compiled))

    @classmethod
    def from_compiled_dsl(cls, path: str | Path) -> "Constraints":
        """Load a DSL file that's the flattened output of a prior compile,
        WITHOUT re-running elimination.

        Reconstructs `per_var_clauses` by partitioning each parsed clause into
        the step bucket where it was live: step = max(step_of[v] for v in
        clause.variables), where step_of comes from the file's `ordering`
        directive (ordering[i] is the variable eliminated at step i).
        """
        path = Path(path)
        cs = parse_constraints(path.read_text())
        n = len(cs.var_names)
        step_of = {v: i for i, v in enumerate(cs.ordering)}

        per_var_clauses: list[list] = [[] for _ in range(n)]
        for clause in cs.clauses:
            if not clause.literals:
                continue
            step = max(step_of[v] for v in clause.variables)
            per_var_clauses[step].append(clause)

        cc = CompiledConstraints(
            var_names=list(cs.var_names),
            var_domains=list(cs.var_domains),
            ordering=list(cs.ordering),
            per_var_clauses=per_var_clauses,
            value_base=cs.value_base,
        )
        return cls(_impl=ClauseFeasibility(cc))

    @classmethod
    def from_function(
        cls,
        fn: Callable[[int, dict[int, int], int], "Iterable[int]"],
        var_domains: list[int],
        ordering: list[int] | None = None,
        neighbours: dict[int, "Iterable[int]"] | None = None,
    ) -> "Constraints":
        """Wrap a Python callable that evaluates S_i(a_<i) directly.

        The callable receives `(step, prior_assignment, var_domain)` and
        returns the feasible set.

        `neighbours[v]` lists the other variables whose assignment can affect
        `v`'s feasibility. Omitted -> worst case (every prior affects every
        later variable), which forces a fully-sequential INFER walk.
        """
        return cls(
            _impl=_FunctionFeasibility(fn, var_domains, ordering, neighbours)
        )


class _FunctionFeasibility:
    def __init__(
        self,
        fn: Callable[[int, dict[int, int], int], "Iterable[int]"],
        var_domains: list[int],
        ordering: list[int] | None = None,
        neighbours: dict[int, "Iterable[int]"] | None = None,
    ):
        self.fn = fn
        self.var_domains = list(var_domains)
        self.ordering = (
            list(ordering)
            if ordering is not None
            else list(range(len(var_domains)))
        )
        if neighbours is None:
            self.neighbours = {
                v: frozenset(self.ordering[:i])
                for i, v in enumerate(self.ordering)
            }
        else:
            self.neighbours = {
                v: frozenset(neighbours.get(v, []))
                for v in self.ordering
            }

    def feasible_set(
        self, step: int, prior_assignment: dict[int, int]
    ) -> frozenset[int]:
        var = self.ordering[step]
        return frozenset(self.fn(step, prior_assignment, self.var_domains[var]))


def compute_levels(
    ordering: list[int],
    neighbours: dict[int, "Iterable[int]"],
) -> list[list[int]]:
    """Greedy levelisation respecting `ordering` and the constraint graph.

    Each `v` is placed at `1 + max(level[n] for n in neighbours[v] if placed)`,
    so every earlier-ordered constraint-neighbour of `v` lands in a strictly
    earlier level. Variables sharing a level have no mutual constraint and
    can be projected in parallel.
    """
    levels: list[list[int]] = []
    placed_level: dict[int, int] = {}
    for var in ordering:
        nbrs = neighbours.get(var, ())
        target = 1 + max(
            (placed_level[n] for n in nbrs if n in placed_level),
            default=-1,
        )
        while target >= len(levels):
            levels.append([])
        levels[target].append(var)
        placed_level[var] = target
    return levels
