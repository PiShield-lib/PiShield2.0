from __future__ import annotations

import torch
import torch.nn as nn

from constraints import Constraints
from minedit import MinEditL2


class CasperLayer(nn.Module):
    """Algorithm 1 INFER as a differentiable nn.Module."""

    def __init__(self, constraints: Constraints, eps: float = 1e-3,
                 projection_backend: str = "closed_form",
                 solver_time_limit: float | None = None):
        super().__init__()
        self.constraints = constraints
        self.var_domains: list[int] = list(constraints.var_domains)
        self.ordering: list[int] = list(constraints.ordering)
        self.ordering_levels: list[list[int]] = constraints.ordering_levels
        self._step_of: dict[int, int] = {v: i for i, v in enumerate(self.ordering)}
        self.eps = eps

        # Which solver computes the MinEdit projection. The constraint set is
        # identical either way -- only the projection differs, which is what
        # makes closed_form vs cvxpy a clean single-variable comparison.
        if projection_backend == "cvxpy":
            from cvxpy_minedit import CVXPYMinEditL2

            def _make(h):
                return CVXPYMinEditL2(h, eps=eps,
                                      solver_time_limit=solver_time_limit)
        elif projection_backend == "closed_form":
            def _make(h):
                return MinEditL2(h, eps=eps)
        else:
            raise ValueError(
                f"projection_backend must be 'closed_form' or 'cvxpy', "
                f"got {projection_backend!r}")
        self.projection_backend = projection_backend

        self._minedits: dict = {}
        for h in sorted(set(self.var_domains)):
            m = _make(h)
            self._minedits[h] = m
            self.add_module(f"_minedit_h{h}", m)

        self.var_starts: list[int] = [0]
        for h in self.var_domains[:-1]:
            self.var_starts.append(self.var_starts[-1] + h)
        self.total_categories = sum(self.var_domains)

    def forward(self, probs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Run INFER on a batched, concatenated softmax tensor.

        Args:
            probs: (B, sum(var_domains)) per-variable softmax probs, vars in
                   declaration order.

        Returns:
            adjusted: (B, sum(var_domains)) — post-INFER probs.
            info: diagnostic dict with differentiable `projection_loss`.
        """
        B, total = probs.shape
        assert total == self.total_categories, (
            f"expected probs with {self.total_categories} columns, got {total}"
        )
        device = probs.device
        n_vars = len(self.var_domains)

        slices: list[torch.Tensor] = [
            probs[:, self.var_starts[v] : self.var_starts[v] + self.var_domains[v]]
            for v in range(n_vars)
        ]

        assignments: list[dict[int, int]] = [{} for _ in range(B)]

        l2_cols: list[torch.Tensor] = [
            torch.zeros(B, device=device, dtype=probs.dtype)
            for _ in range(n_vars)
        ]
        n_projected_per_var: list[int] = [0] * n_vars
        total_projected = 0

        for level in self.ordering_levels:
          for var in level:
            step = self._step_of[var]
            h = self.var_domains[var]
            slice_v = slices[var]
            argmax_v = slice_v.argmax(dim=1)

            feasibles: list[frozenset[int]] = []
            for b in range(B):
                f = self.constraints.feasible_set(step, assignments[b])
                if not f:
                    raise RuntimeError(
                        f"empty feasible set at step={step} var={var} "
                        f"sample={b}; constraints inconsistent with prior "
                        f"assignment {assignments[b]}"
                    )
                feasibles.append(f)

            groups: dict[frozenset[int], list[int]] = {}
            for b, f in enumerate(feasibles):
                groups.setdefault(f, []).append(b)

            new_slice = slice_v

            for f, idxs in groups.items():
                if len(f) == h:
                    for b in idxs:
                        assignments[b][var] = int(argmax_v[b].item())
                    continue

                idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
                preds = argmax_v[idx_t]

                in_f = torch.zeros(len(idxs), dtype=torch.bool, device=device)
                for t in f:
                    in_f = in_f | (preds == t)
                for j, b in enumerate(idxs):
                    if bool(in_f[j].item()):
                        assignments[b][var] = int(preds[j].item())

                need_local = (~in_f).nonzero(as_tuple=False).squeeze(1)
                if need_local.numel() == 0:
                    continue

                need_rows = slice_v[idx_t[need_local]]
                minedit = self._minedits[h]

                # Project under each feasible target and pick the per-row
                # argmin over L2 distance.
                best_dist: torch.Tensor | None = None
                best_t: torch.Tensor | None = None
                best_adj: torch.Tensor | None = None
                for t in sorted(f):
                    adj_t, dist_t, _ = minedit(
                        need_rows, target_category=int(t), is_positive=True
                    )
                    if best_dist is None:
                        best_dist = dist_t
                        best_t = torch.full(
                            (need_local.numel(),), int(t),
                            dtype=torch.long, device=device,
                        )
                        best_adj = adj_t
                    else:
                        better = dist_t < best_dist
                        best_dist = torch.where(better, dist_t, best_dist)
                        best_t = torch.where(better, torch.full_like(best_t, int(t)), best_t)
                        best_adj = torch.where(better.unsqueeze(1), adj_t, best_adj)

                global_need = idx_t[need_local]
                expanded = torch.zeros(
                    B, h, device=device, dtype=slice_v.dtype
                ).index_copy(0, global_need, best_adj)
                mask = torch.zeros(B, dtype=torch.bool, device=device)
                mask[global_need] = True
                new_slice = torch.where(mask.unsqueeze(1), expanded, new_slice)

                expanded_l2 = torch.zeros(
                    B, device=device, dtype=probs.dtype
                ).index_copy(0, global_need, best_dist)
                l2_cols[var] = l2_cols[var] + expanded_l2
                n_projected_per_var[var] += int(need_local.numel())
                total_projected += int(need_local.numel())

                for k, local_j in enumerate(need_local.tolist()):
                    global_b = idxs[local_j]
                    assignments[global_b][var] = int(best_t[k].item())

            slices[var] = new_slice

        adjusted = torch.cat(slices, dim=1)
        # (B, n_vars) differentiable per-variable L2 edits; column v is the
        # edit applied to variable v's slice. Lets callers weight or exclude
        # variables (e.g. auxiliary variables with no neural head).
        l2_per_var = torch.stack(l2_cols, dim=1)
        l2_per_sample = l2_per_var.sum(dim=1)
        info = {
            "projection_loss": l2_per_sample.sum() / max(B, 1),
            "total_l2_distance": float(l2_per_sample.sum().item()),
            "avg_l2_per_sample": float(l2_per_sample.mean().item()) if B else 0.0,
            "total_constraints_applied": total_projected,
            "final_assignments": assignments,
            "l2_per_var": l2_per_var,
            "n_projected_per_var": n_projected_per_var,
        }
        return adjusted, info
