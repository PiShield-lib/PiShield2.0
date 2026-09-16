"""Categorical requirements subpackage of PiShield.

This subpackage implements the Shield Layer for *categorical* requirements: signed
clauses over categorical variables, each variable taking one of a fixed, finite set of
values (e.g. ``[1,2,3]:a_1``). A requirement (clause) is a disjunction of such signed
literals, e.g. ``[1,2,3,4,5,6,7,8,9]:a_1 or [1,2,3,4,5,6,7,8,9]:b_1 or [0]:s_1``.

The requirements are compiled, ahead of time, into per-variable feasible-value tables
via signed resolution (see :mod:`pishield.categorical_requirements.resolution`), and
then enforced at inference/training time by
:class:`~pishield.categorical_requirements.casper_layer.CasperLayer`, which projects
each variable's predicted distribution onto its feasible values,
following a fixed variable ordering so that each correction only ever depends on
already-corrected variables.
:class:`~pishield.categorical_requirements.shield_layer.ShieldLayer`
wraps this machinery behind PiShield's standard Shield Layer interface.
"""
# TODO: Lohith please check the above note and shield_layer.py in pishield/categorical_requirements/