import torch
import torch.nn as nn


class MinEditL2(nn.Module):
    """L2 MinEdit projection — closed-form single-pass active-set heuristic.

    Given p over C categories and target index t, returns q satisfying
    `q[t] >= q[j] + eps for all j != t` and aims to minimise ||q - p||_2.
    The active set is selected once from p; when that selection is also
    optimal at the projected point this coincides with the exact L2
    projection, otherwise it is an upper bound.

    forward(probs, target_category, is_positive)
        probs           : (B, C) softmax probabilities
        target_category : int
        is_positive     : True enforces argmax == target; False enforces
                          argmax != target (negative head promotes runner-up)

    Returns (p_adj, l2, n_active).
    """

    def __init__(self, n_categories, eps=1e-3):
        super().__init__()
        self.n_categories = n_categories
        self.eps = eps

    def _positive_head_batch(self, p, target):
        """Project so argmax(p_adj) == target with margin eps.

        Closed-form KKT: pool active mass (j != target with p[t] < p[j] + eps)
        with p[t] and redistribute as
            new_target = (pool + n_active * eps) / (1 + n_active)
            new_active = (pool - eps) / (1 + n_active)
        Categories outside the active set are unchanged; mass is conserved.
        """
        B, C = p.shape

        p_target = p[:, target : target + 1]

        target_exclude = torch.ones(C, dtype=torch.bool, device=p.device)
        target_exclude[target] = False
        active_mask = (p_target < p + self.eps) & target_exclude.unsqueeze(0)

        n_active = active_mask.sum(dim=1)
        has_active = n_active > 0

        active_sum = (p * active_mask).sum(dim=1)
        pool = p[:, target] + active_sum

        denom = (1 + n_active).float()
        new_target_val = (pool + n_active.float() * self.eps) / denom
        new_active_val = (pool - self.eps) / denom

        active_replacement = new_active_val.unsqueeze(1).expand_as(p)
        update_mask = active_mask & has_active.unsqueeze(1)
        p_adj = torch.where(update_mask, active_replacement, p)

        target_col = torch.where(has_active, new_target_val, p[:, target])
        target_mask = torch.zeros(C, dtype=torch.bool, device=p.device)
        target_mask[target] = True
        target_mask = target_mask.unsqueeze(0).expand_as(p)
        p_adj = torch.where(target_mask, target_col.unsqueeze(1).expand_as(p), p_adj)

        l2 = torch.linalg.vector_norm(p_adj - p, ord=2, dim=1)
        return p_adj, l2, n_active

    def _negative_head_batch(self, p, forbidden):
        """Project so argmax(p_adj) != forbidden by promoting the runner-up."""
        B, C = p.shape
        current_argmax = p.argmax(dim=1)
        needs_fix = (current_argmax == forbidden)

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
            torch.tensor(-float('inf'), device=p.device),
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

    def forward(self, probs, target_category, is_positive):
        if is_positive:
            return self._positive_head_batch(probs, int(target_category))
        return self._negative_head_batch(probs, int(target_category))
