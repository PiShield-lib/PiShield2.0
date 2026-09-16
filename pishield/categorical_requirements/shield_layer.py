"""The categorical Shield Layer, the main public entry point of this subpackage.

Defines :class:`ShieldLayer`, a differentiable PyTorch module that corrects neural
network predictions over categorical variables so they satisfy a set of categorical
requirements. Each variable is represented in the input/output tensor as a contiguous
block of softmax probabilities (one column per value it can take), and each requirement
restricts, for every variable, which of its values remain feasible given the
already-corrected values of the earlier variables in the ordering.
"""

import random

import torch
from torch import nn

from pishield.categorical_requirements.casper_layer import CasperLayer
from pishield.categorical_requirements.constraints import Constraints
from pishield.categorical_requirements.resolution import compile_constraints
from pishield.categorical_requirements.signed_clauses import parse_constraints


class ShieldLayer(nn.Module):
    """
    Differentiable layer that corrects predictions so they satisfy a set of categorical
    requirements: signed clauses over categorical variables, each restricting a variable
    to a fixed subset of its possible values (e.g. ``[1,2,3]:a_1``), optionally
    conditioned (via disjunction) on the values of other variables.

    The requirements file is compiled ahead of time (signed resolution) into, for each
    variable and each possible prior assignment, the set of values still feasible for it
    (Algorithm 1, INFER; see :mod:`~pishield.categorical_requirements.resolution`).
    `forward` walks the variables in a fixed ordering, and for each one either accepts
    the network's argmax (if already feasible) or projects its predicted distribution
    onto the nearest feasible one (L2 MinEdit) -- so a variable is only ever corrected
    using already-corrected values of the earlier variables, guaranteeing all
    requirements hold on the output.

    Attributes:
        num_variables: The total number of tensor columns, i.e. the sum of the domain
            sizes of all declared categorical variables (matches the input/output shape).
        ordering: The order in which variables are visited during correction.
        constraints: The compiled :class:`~pishield.categorical_requirements.constraints.Constraints`.
        last_info: Diagnostics dict from the most recent `forward` call (e.g. the
            differentiable `projection_loss`); see
            :meth:`~pishield.categorical_requirements.casper_layer.CasperLayer.forward`.
    """

    def __init__(self, num_variables: int,
                 requirements_filepath: str,
                 ordering_choice: str = 'given',
                 eps: float = 1e-3,
                 projection_backend: str = 'closed_form',
                 solver_time_limit: float = None):
        """Build the layer by compiling the requirements and wrapping a CasperLayer.

        Args:
            num_variables: Total number of tensor columns, i.e. the sum of the domain
                sizes of all variables declared in `requirements_filepath` (matches the
                dimension of the tensors to be corrected by the layer -- *not* the count
                of categorical variables).
            requirements_filepath: Path to a ``.txt`` file holding the variable
                declarations (``var name domain_size``), an optional ``ordering`` line,
                and the categorical constraints (signed clauses, e.g.
                ``[1,2,3]:a_1 or [0]:b_1``).
            ordering_choice: ``'given'`` uses the ordering declared in the file (or
                declaration order if none is given); ``'random'`` uses a random
                permutation of the variables instead.
            eps: Margin enforced between the corrected top value and the runner-up in
                the L2 MinEdit projection.
            projection_backend: ``'closed_form'`` (default) or ``'cvxpy'``; only the
                MinEdit projection differs, the enforced constraints are identical.
            solver_time_limit: Optional time limit (seconds), only used by the
                ``'cvxpy'`` projection backend.

        Raises:
            Exception: If the requirements file's total category count doesn't match
                `num_variables`, or if `ordering_choice` is not recognised.

        Example:
            >>> layer = ShieldLayer(num_variables=19,
            ...                     requirements_filepath='constraints.txt',
            ...                     ordering_choice='given')
            >>> corrected = layer(predictions)  # predictions: (batch, 19)
        """
        super().__init__()
        self.num_variables = num_variables
        self.ordering_choice = ordering_choice

        with open(requirements_filepath, 'r') as f:
            constraint_set = parse_constraints(f.read())

        total_categories = sum(constraint_set.var_domains)
        if total_categories != num_variables:
            raise Exception(
                f"num_variables={num_variables} does not match the requirements file: "
                f"{requirements_filepath!r} declares {len(constraint_set.var_domains)} "
                f"variables with domain sizes summing to {total_categories}."
            )

        if ordering_choice == 'random':
            random.shuffle(constraint_set.ordering)
        elif ordering_choice != 'given':
            raise Exception(
                f"Unknown ordering_choice {ordering_choice!r} for categorical "
                f"requirements; use 'given' or 'random'."
            )

        compiled = compile_constraints(constraint_set, verbose=True)
        self.constraints = Constraints.from_compiled(compiled)
        self.ordering = list(self.constraints.ordering)

        self.casper = CasperLayer(self.constraints, eps=eps,
                                  projection_backend=projection_backend,
                                  solver_time_limit=solver_time_limit)
        self.last_info = None

    def forward(self, preds: torch.Tensor):
        """Correct a batch of predictions so they satisfy the requirements.

        Args:
            preds: Predictions tensor of shape ``(B, num_variables)``, the concatenated
                per-variable softmax probabilities in declaration order.

        Returns:
            A tensor of the same shape with predictions corrected to satisfy the
            categorical constraints. Diagnostics from the correction (including the
            differentiable ``projection_loss``) are stored in `self.last_info`.

        Example:
            >>> corrected = layer(preds)
        """
        adjusted, info = self.casper(preds)
        self.last_info = info
        return adjusted

# TODO: Lohith check this file; it's similar to pishield\qflra_requirements\shield_layer.py