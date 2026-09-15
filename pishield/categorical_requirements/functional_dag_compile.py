from __future__ import annotations

from itertools import product

from resolution import CompiledConstraints
from signed_clauses import ConstraintSet, SignedClause


class NotFunctionalDAG(ValueError):
    pass


def compile_functional_dag(cs: ConstraintSet) -> CompiledConstraints:
    """Compile a topologically ordered, total functional signed-clause DAG.

    This is a validated fast path for functional-link encodings.  Every
    non-empty inference bucket must encode exactly one output for every tuple
    of its parent values using clauses of the form

        [not a]:A or [not b]:B or [f(a,b)]:Y.

    Such a bucket directly defines Y as a total function of earlier variables.
    Its clauses are already an INFER artifact: eliminating Y can retain no
    non-tautological resolvent.  We verify that structure and totality in time
    linear in the artifact size, then return the correctly bucketed clauses.
    """
    n_vars = len(cs.var_names)
    step_of = {var: step for step, var in enumerate(cs.ordering)}
    buckets: list[list[SignedClause]] = [[] for _ in range(n_vars)]
    for clause in cs.clauses:
        if not clause.literals:
            raise NotFunctionalDAG("empty clause")
        step = max(step_of[lit.var] for lit in clause.literals)
        buckets[step].append(clause)

    for step, clauses in enumerate(buckets):
        if not clauses:
            continue
        output = cs.ordering[step]
        parent_tuple: tuple[int, ...] | None = None
        mapping: dict[tuple[int, ...], int] = {}

        for clause in clauses:
            output_values = clause.get(output)
            if output_values is None or len(output_values) != 1:
                raise NotFunctionalDAG(
                    f"bucket {cs.var_names[output]!r} has a non-singleton output"
                )
            parents = tuple(
                lit.var for lit in clause.literals if lit.var != output
            )
            if parent_tuple is None:
                parent_tuple = parents
            elif parents != parent_tuple:
                raise NotFunctionalDAG(
                    f"bucket {cs.var_names[output]!r} has inconsistent parents"
                )
            if any(step_of[parent] >= step for parent in parents):
                raise NotFunctionalDAG(
                    f"bucket {cs.var_names[output]!r} is not topological"
                )

            trigger = []
            for parent in parents:
                values = clause.get(parent) or frozenset()
                domain = cs.var_domains[parent]
                missing = set(range(domain)) - set(values)
                if len(values) != domain - 1 or len(missing) != 1:
                    raise NotFunctionalDAG(
                        f"bucket {cs.var_names[output]!r} has a non-trigger literal"
                    )
                trigger.append(next(iter(missing)))
            key = tuple(trigger)
            value = next(iter(output_values))
            previous = mapping.setdefault(key, value)
            if previous != value:
                raise NotFunctionalDAG(
                    f"bucket {cs.var_names[output]!r} is not deterministic"
                )

        parents = parent_tuple or ()
        expected = 1
        for parent in parents:
            expected *= cs.var_domains[parent]
        if len(mapping) != expected:
            raise NotFunctionalDAG(
                f"bucket {cs.var_names[output]!r} is not total: "
                f"{len(mapping)} of {expected} input tuples"
            )
        # Checking the exact key set also catches duplicate clauses masking a
        # missing trigger while keeping the count unchanged.
        expected_keys = product(*(range(cs.var_domains[p]) for p in parents))
        if any(key not in mapping for key in expected_keys):
            raise NotFunctionalDAG(
                f"bucket {cs.var_names[output]!r} is not total"
            )

    return CompiledConstraints(
        var_names=list(cs.var_names),
        var_domains=list(cs.var_domains),
        ordering=list(cs.ordering),
        per_var_clauses=buckets,
        value_base=cs.value_base,
    )
