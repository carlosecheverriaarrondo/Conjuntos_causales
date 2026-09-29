"""Simulador de conjuntos causales por sprinkling de Poisson."""

from .causal import is_partial_order, minkowski_causal_matrix, partial_order_violations
from .dimension import (
    myrheim_meyer_dimension,
    myrheim_meyer_from_fraction,
    ordering_fraction,
    ordering_fraction_expected,
)
from .regions import CausalDiamond
from .sprinkling import sample_uniform, sprinkle

__all__ = [
    "CausalDiamond",
    "sample_uniform",
    "sprinkle",
    "minkowski_causal_matrix",
    "partial_order_violations",
    "is_partial_order",
    "ordering_fraction",
    "ordering_fraction_expected",
    "myrheim_meyer_from_fraction",
    "myrheim_meyer_dimension",
]
