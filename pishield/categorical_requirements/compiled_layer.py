from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from pishield.categorical_requirements.casper_layer import CasperLayer
from pishield.categorical_requirements.constraints import Constraints


class CompiledConstraintLayer(nn.Module):
    """Drop-in constraint layer backed by a compiled signed-clause artifact.

    Wraps `CasperLayer(Constraints.from_compiled_dsl(path))` behind the same
    forward contract as the hard-coded MNIST layers (`ConstraintLayer`,
    `CarryChainConstraintLayer`): input and output are `(B, sum(variable_ranges))`
    tensors over the *network's* variables only.

    The compiled constraint set may declare additional auxiliary variables
    after the network variables (e.g. the carry chain's `k_i`). Those have no
    neural head, so the layer feeds them fixed uniform distributions; INFER
    forces them to their (singleton) feasible values, which is how derived
    state such as carries propagates through the chain. Auxiliary edits are
    excluded from every reported metric via CasperLayer's `l2_per_var`.

    The artifact is loaded with `Constraints.from_compiled_dsl` — i.e. it must
    be the *output* of a compile (or a constraint set that is its own compile
    fixpoint, like the generated MNIST-Add DSL); loading with
    `Constraints.from_file` would re-run the whole elimination.
    """

    def __init__(
        self,
        compiled_path: str | Path,
        variable_ranges: list[int],
        eps: float = 1e-3,
        projection_backend: str = "closed_form",
        solver_time_limit: float | None = None,
    ):
        super().__init__()
        self.constraints = Constraints.from_compiled_dsl(compiled_path)
        domains = self.constraints.var_domains

        n_real = len(variable_ranges)
        if len(domains) < n_real or domains[:n_real] != list(variable_ranges):
            raise ValueError(
                f"compiled artifact {compiled_path} declares leading domains "
                f"{domains[:n_real]} but the network expects {list(variable_ranges)}"
            )

        self.n_real_vars = n_real
        self.real_total = sum(variable_ranges)
        self.aux_domains: list[int] = domains[n_real:]
        self.casper = CasperLayer(self.constraints, eps=eps,
                                  projection_backend=projection_backend,
                                  solver_time_limit=solver_time_limit)

    def forward(self, probs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B = probs.shape[0]
        assert probs.shape[1] == self.real_total, (
            f"expected probs with {self.real_total} columns, got {probs.shape[1]}"
        )

        if self.aux_domains:
            aux = [
                torch.full(
                    (B, h), 1.0 / h, device=probs.device, dtype=probs.dtype
                )
                for h in self.aux_domains
            ]
            full = torch.cat([probs] + aux, dim=1)
        else:
            full = probs

        adjusted_full, casper_info = self.casper(full)
        adjusted = adjusted_full[:, : self.real_total]

        l2_real = casper_info["l2_per_var"][:, : self.n_real_vars]
        l2_per_sample = l2_real.sum(dim=1)
        n_projected = sum(casper_info["n_projected_per_var"][: self.n_real_vars])

        info = {
            "projection_loss": l2_per_sample.mean() if B else l2_per_sample.sum(),
            "total_l2_distance": float(l2_per_sample.sum().item()),
            "avg_l2_per_sample": float(l2_per_sample.mean().item()) if B else 0.0,
            "total_constraints_applied": n_projected,
            "final_assignments": casper_info["final_assignments"],
        }
        return adjusted, info
