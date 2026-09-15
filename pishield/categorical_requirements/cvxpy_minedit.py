import torch
import torch.nn as nn

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


def _build_canonical_layer(n_categories: int, eps: float) -> CvxpyLayer:
    """Compile a CvxpyLayer that solves the L2 projection with target=0."""
    C = n_categories
    p = cp.Parameter(C)
    q = cp.Variable(C)

    margin_lhs = q[0] - q[1:]
    constraints = [margin_lhs >= eps]
    objective = cp.Minimize(cp.sum_squares(q - p))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp(), "CVXPY problem must be DPP for cvxpylayers"
    return CvxpyLayer(problem, parameters=[p], variables=[q])


class CVXPYMinEditL2(nn.Module):
    """L2 MinEdit projection solved as a QP via cvxpylayers + SCS.

    Drop-in for `MinEditL2`. The layer is compiled once with target fixed at
    index 0; at call time samples are permuted so the requested target lands
    at 0. The L2 projection is permutation-equivariant, so this is exact.
    """

    def __init__(
        self,
        n_categories: int,
        eps: float = 1e-3,
        solver: str = "SCS",
        solver_args: dict | None = None,
        solver_time_limit: float | None = None,
    ):
        super().__init__()
        self.n_categories = n_categories
        self.eps = eps
        self.solver = solver
        self.solver_args = dict(solver_args or {})
        if solver_time_limit is not None and solver_time_limit > 0:
            self.solver_args.setdefault("time_limit_secs", float(solver_time_limit))
        self.solver_time_limit = solver_time_limit
        self._layer = _build_canonical_layer(n_categories, eps)

    def _project_canonical(self, p_canon: torch.Tensor) -> torch.Tensor:
        (q_canon,) = self._layer(
            p_canon, solver_args={"solve_method": self.solver, **self.solver_args}
        )
        return q_canon

    def _positive_head_batch(self, p: torch.Tensor, target: int):
        B, C = p.shape
        if target == 0:
            p_canon = p
            q_canon = self._project_canonical(p_canon)
            p_adj = q_canon
        else:
            perm = torch.arange(C, device=p.device).clone()
            perm[0], perm[target] = target, 0
            p_canon = p[:, perm]
            q_canon = self._project_canonical(p_canon)
            p_adj = q_canon[:, perm]

        l2 = torch.linalg.vector_norm(p_adj - p, ord=2, dim=1)
        n_active = torch.zeros(B, dtype=torch.long, device=p.device)
        return p_adj, l2, n_active

    def _negative_head_batch(self, p: torch.Tensor, forbidden: int):
        B, C = p.shape
        current_argmax = p.argmax(dim=1)
        needs_fix = current_argmax == forbidden

        if not needs_fix.any():
            return (
                p.clone(),
                torch.zeros(B, device=p.device),
                torch.zeros(B, dtype=torch.long, device=p.device),
            )

        forbidden_mask = torch.zeros(C, dtype=torch.bool, device=p.device)
        forbidden_mask[forbidden] = True
        p_masked = torch.where(
            forbidden_mask.unsqueeze(0),
            torch.tensor(-float("inf"), device=p.device),
            p,
        )
        runner_up = p_masked.argmax(dim=1)

        p_adj = p
        l2 = torch.zeros(B, device=p.device)
        n_active = torch.zeros(B, dtype=torch.long, device=p.device)

        for target_val in runner_up[needs_fix].unique():
            group_mask = needs_fix & (runner_up == target_val)
            if not group_mask.any():
                continue
            adj, dist, na = self._positive_head_batch(p[group_mask], target_val.item())
            idx = group_mask.nonzero(as_tuple=False).squeeze(1)
            expanded_adj = torch.zeros_like(p).index_copy(0, idx, adj)
            expanded_l2 = torch.zeros(B, device=p.device).index_copy(0, idx, dist)
            expanded_na = torch.zeros(B, dtype=torch.long, device=p.device).index_copy(0, idx, na)
            p_adj = torch.where(group_mask.unsqueeze(1), expanded_adj, p_adj)
            l2 = torch.where(group_mask, expanded_l2, l2)
            n_active = torch.where(group_mask, expanded_na, n_active)

        return p_adj, l2, n_active

    def forward(self, probs: torch.Tensor, target_category: int, is_positive: bool):
        if is_positive:
            return self._positive_head_batch(probs, int(target_category))
        return self._negative_head_batch(probs, int(target_category))
