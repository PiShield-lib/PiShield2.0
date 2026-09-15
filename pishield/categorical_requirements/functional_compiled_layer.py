from __future__ import annotations

from itertools import product
from pathlib import Path

import torch
import torch.nn as nn

from constraints import Constraints
from minedit import MinEditL2


class FunctionalCompiledConstraintLayer(nn.Module):
    """Fast path for compiled constraints whose non-empty buckets are functions.

    Each compiled clause bucket is exhaustively converted at load time into a
    small lookup table from its earlier neighbours to its singleton feasible
    value.  Buckets which are not total, functional, or small enough are
    rejected instead of silently changing the constraint semantics.
    """

    def __init__(
        self,
        compiled_path: str | Path,
        variable_ranges: list[int],
        eps: float = 1e-3,
        max_table_entries: int = 100_000,
    ):
        super().__init__()
        self.constraints = Constraints.from_compiled_dsl(compiled_path)
        impl = self.constraints._impl
        compiled = impl.compiled
        domains = list(compiled.var_domains)

        self.n_real_vars = len(variable_ranges)
        if domains[: self.n_real_vars] != list(variable_ranges):
            raise ValueError(
                "compiled artifact leading domains do not match variable_ranges"
            )

        self.domains = domains
        self.ordering = list(compiled.ordering)
        self.real_total = sum(variable_ranges)
        self.total_categories = sum(domains)
        self.var_starts = [0]
        for h in domains[:-1]:
            self.var_starts.append(self.var_starts[-1] + h)

        self._minedits: dict[int, MinEditL2] = {}
        for h in sorted(set(domains)):
            module = MinEditL2(h, eps=eps)
            self._minedits[h] = module
            self.add_module(f"_minedit_h{h}", module)

        step_of = {var: step for step, var in enumerate(self.ordering)}
        self.lookup_specs: dict[int, tuple[tuple[int, ...], str]] = {}
        for step, var in enumerate(self.ordering):
            if not compiled.per_var_clauses[step]:
                continue
            parents = tuple(sorted(
                (
                    n for n in impl.neighbours[var]
                    if step_of[n] < step
                ),
                key=step_of.__getitem__,
            ))
            n_entries = 1
            for parent in parents:
                n_entries *= domains[parent]
            if n_entries > max_table_entries:
                raise ValueError(
                    f"variable {compiled.var_names[var]!r} needs a lookup table "
                    f"with {n_entries} entries (limit {max_table_entries})"
                )

            table = torch.empty(n_entries, dtype=torch.long)
            for values in product(*(range(domains[p]) for p in parents)):
                assignment = dict(zip(parents, values))
                feasible = impl.feasible_set(step, assignment)
                if len(feasible) != 1:
                    raise ValueError(
                        f"variable {compiled.var_names[var]!r} is not a total "
                        f"function at {assignment}: feasible={sorted(feasible)}"
                    )
                flat = 0
                for parent, value in zip(parents, values):
                    flat = flat * domains[parent] + value
                table[flat] = next(iter(feasible))
            buffer_name = f"_lookup_{var}"
            self.register_buffer(buffer_name, table)
            self.lookup_specs[var] = (parents, buffer_name)

    def forward(self, probs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch = probs.shape[0]
        if probs.shape[1] != self.real_total:
            raise ValueError(
                f"expected {self.real_total} probability columns, "
                f"got {probs.shape[1]}"
            )

        slices = []
        offset = 0
        for var, domain in enumerate(self.domains):
            if var < self.n_real_vars:
                slices.append(probs[:, offset : offset + domain])
                offset += domain
            else:
                slices.append(torch.full(
                    (batch, domain), 1.0 / domain,
                    dtype=probs.dtype, device=probs.device,
                ))

        assignments: dict[int, torch.Tensor] = {}
        l2_by_var = [
            torch.zeros(batch, dtype=probs.dtype, device=probs.device)
            for _ in self.domains
        ]
        projected_by_var = [0] * len(self.domains)

        for var in self.ordering:
            current = slices[var]
            predicted = current.argmax(dim=1)
            spec = self.lookup_specs.get(var)
            if spec is None:
                assignments[var] = predicted
                continue

            parents, buffer_name = spec
            flat = torch.zeros(batch, dtype=torch.long, device=probs.device)
            for parent in parents:
                flat = flat * self.domains[parent] + assignments[parent]
            target = getattr(self, buffer_name)[flat]
            needs_fix = predicted != target
            if not bool(needs_fix.any()):
                assignments[var] = predicted
                continue

            adjusted = current
            l2 = l2_by_var[var]
            for target_value in target[needs_fix].unique():
                mask = needs_fix & (target == target_value)
                rows = mask.nonzero(as_tuple=False).squeeze(1)
                projected, distance, _ = self._minedits[self.domains[var]](
                    current[rows],
                    target_category=int(target_value.item()),
                    is_positive=True,
                )
                replacement = torch.zeros_like(current).index_copy(
                    0, rows, projected
                )
                adjusted = torch.where(mask.unsqueeze(1), replacement, adjusted)
                expanded_l2 = torch.zeros_like(l2).index_copy(0, rows, distance)
                l2 = torch.where(mask, expanded_l2, l2)

            slices[var] = adjusted
            l2_by_var[var] = l2
            projected_by_var[var] = int(needs_fix.sum().item())
            assignments[var] = target

        adjusted_real = torch.cat(slices[: self.n_real_vars], dim=1)
        l2_real = torch.stack(l2_by_var[: self.n_real_vars], dim=1)
        l2_per_sample = l2_real.sum(dim=1)
        info = {
            "projection_loss": (
                l2_per_sample.mean() if batch else l2_per_sample.sum()
            ),
            "total_l2_distance": float(l2_per_sample.sum().item()),
            "avg_l2_per_sample": (
                float(l2_per_sample.mean().item()) if batch else 0.0
            ),
            "total_constraints_applied": sum(
                projected_by_var[: self.n_real_vars]
            ),
            "final_assignments": assignments,
        }
        return adjusted_real, info
