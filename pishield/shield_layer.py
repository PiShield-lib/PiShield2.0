"""Top-level entry point for building Shield Layers.

A Shield Layer is a differentiable layer that corrects a model's outputs so
that they are *guaranteed* to satisfy a given set of requirements (constraints),
regardless of the input. This module exposes :func:`build_shield_layer`, which
dispatches to the appropriate backend (linear, QFLRA, propositional, or categorical)
based on the requirements, and :func:`detect_requirements_type`, which infers the
requirement type from a requirements file.
"""

import re
from typing import List

from pishield.linear_requirements.shield_layer import ShieldLayer as LinearConstraintLayer
from pishield.qflra_requirements.shield_layer import ShieldLayer as QFLRAConstraintLayer
from pishield.propositional_requirements.shield_layer import ShieldLayer as PropositionalConstraintLayer
from pishield.categorical_requirements.shield_layer import ShieldLayer as CategoricalConstraintLayer


def build_shield_layer(num_variables: int,
                       requirements_filepath: str,
                       ordering_choice: str = 'given',
                       custom_ordering: List = None,
                       requirements_type='auto'):
    """Build a Shield Layer from the given requirements.

    Selects and constructs the appropriate Shield Layer backend (linear, QFLRA,
    propositional, or categorical) for the supplied requirements. The returned layer is
    differentiable and can be used at both inference and training time to correct
    a model's outputs so that they satisfy the requirements.

    Args:
        num_variables: Total number of variables (e.g. labels or features,
            depending on the task), matching the dimension of the tensors that
            are to be corrected by the layer. For the categorical backend, this is
            the sum of the domain sizes of all declared categorical variables
            (i.e. the number of one-hot/softmax columns), not the count of variables.
        requirements_filepath: Path to a ``.txt`` file containing the requirements.
        ordering_choice: How to order the variables when applying corrections.
            One of ``'given'``, ``'random'``, or a custom ordering implemented by
            the user. If ``'given'``, the ordering is read from
            ``requirements_filepath`` when available, otherwise the ascending
            order of the variables is used. If ``'random'``, a random ordering of
            the variables is used.
        custom_ordering: An explicit ordering of the variables (only used by the
            propositional backend). Defaults to None.
        requirements_type: One of ``'auto'``, ``'linear'``, ``'propositional'``,
            ``'qflra'``, or ``'categorical'``. If ``'auto'``, the appropriate backend
            is detected from the contents of ``requirements_filepath`` via
            :func:`detect_requirements_type`.

    Returns:
        A Shield Layer instance (``LinearConstraintLayer``, ``QFLRAConstraintLayer``,
        ``PropositionalConstraintLayer``, or ``CategoricalConstraintLayer``) that
        corrects model outputs to satisfy the requirements.

    Raises:
        Exception: If ``requirements_type`` is not one of the recognised values.

    Example:
        >>> layer = build_shield_layer(
        ...     num_variables=5,
        ...     requirements_filepath='requirements.txt',
        ... )
        >>> corrected = layer(model_output)  # corrected satisfies the requirements
    """

    if requirements_type == 'linear':
        return LinearConstraintLayer(num_variables, requirements_filepath, ordering_choice)
    elif requirements_type == 'qflra':
        return QFLRAConstraintLayer(num_variables, requirements_filepath, ordering_choice)
    elif requirements_type == 'propositional':
        return PropositionalConstraintLayer(num_variables, requirements_filepath, ordering_choice, custom_ordering=custom_ordering)
    elif requirements_type == 'categorical':
        return CategoricalConstraintLayer(num_variables, requirements_filepath, ordering_choice)
    elif requirements_type == 'auto':
        detected_requirements_type = detect_requirements_type(requirements_filepath)
        return build_shield_layer(num_variables, requirements_filepath, ordering_choice, custom_ordering=custom_ordering,
                                  requirements_type=detected_requirements_type)
    else:
        raise Exception('Unknown requirements type!')


def detect_requirements_type(requirements_filepath: str) -> str:
    """Infer the requirement type from the contents of a requirements file.

    Scans the file and classifies it as ``'categorical'``, ``'propositional'``,
    ``'qflra'``, or ``'linear'`` based on the tokens it contains (see inline comments
    for the exact detection rules).

    Args:
        requirements_filepath: Path to a ``.txt`` file containing the requirements.

    Returns:
        The detected requirement type as one of ``'categorical'``, ``'propositional'``,
        ``'qflra'``, or ``'linear'``, or None if no requirement type could be detected.
    """
    # Categorical requirements are signed clauses of the form '[v1,v2,...]:var', disjoined
    # with 'or', e.g. '[1,...,9]:a_1 or [1,...,9]:b_1 or [0]:s_1'. Every such literal
    # contains a bracketed set of possible values, so a single pair of square brackets
    # anywhere in a line unambiguously marks the file as categorical: no other requirement
    # type uses square brackets. This check therefore comes first.
    #
    # Propositional requirements can be written either as Horn rules ('head :- body') or as
    # disjunctive clauses ('y_0 or not y_1'); both are accepted by the propositional parser.
    # The detection order matters: a ':-' token unambiguously marks a propositional Horn rule.
    # Otherwise, QFLRA and linear requirements both contain inequality signs, so we distinguish
    # them by the boolean operators ('or'/'neg') that only QFLRA uses. A clause-style
    # propositional requirement also uses 'or' but, unlike QFLRA, has no inequality sign.
    #
    # The linear-vs-QFLRA decision is a *whole-file* property: a QFLRA file may mix plain
    # inequalities with disjunctive ones, so we must not conclude 'linear' from an early plain
    # inequality line. We only settle on 'linear' after scanning the whole file without seeing
    # any boolean operator; a single disjunction/negation anywhere makes the file QFLRA.
    inequality_signs = ['>=', '>', '<=', '<']
    bracket_pair = re.compile(r'\[[^\[\]]*\]')
    saw_inequality = False
    with open(requirements_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or 'ordering' in line:
                continue
            if bracket_pair.search(line):
                print('Using auto mode ::: Detected categorical requirements!')
                return 'categorical'
            tokens = line.split()
            if ':-' in tokens:
                print('Using auto mode ::: Detected propositional requirements!')
                return 'propositional'
            has_inequality = any(sign in line for sign in inequality_signs)
            has_boolean_op = 'or' in tokens or 'neg' in tokens
            if has_inequality:
                saw_inequality = True
                if has_boolean_op:
                    print('Using auto mode ::: Detected QFLRA requirements!')
                    return 'qflra'
            elif has_boolean_op:
                print('Using auto mode ::: Detected propositional requirements!')
                return 'propositional'
    if saw_inequality:
        print('Using auto mode ::: Detected linear requirements!')
        return 'linear'
    return None

