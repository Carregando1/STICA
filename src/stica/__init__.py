"""STICA: spacetime-inhomogeneous cellular automata in JAX / CAX."""
from .ca import CELL, TIME, STICA, initial_state, rollout, simulate, stica_step
from .export import lattice_to_stl
from .objectives import OBJECTIVES, get_objective, neg_surface, neg_surface_to_volume, surface_area
from .optimize import GAResult, Mutations, init_state, run_ga, run_ga_batch, sample_distinct, srt_shape
from .rules import (neighbor_counts, neighbor_tuples, num_rules, reorder_two_state_rules, rule_index,
                    rules_per_state, validate_srt)

__version__ = "0.1.0"

__all__ = [
    "CELL", "TIME", "STICA", "initial_state", "rollout", "simulate", "stica_step",
    "lattice_to_stl",
    "OBJECTIVES", "get_objective", "neg_surface", "neg_surface_to_volume", "surface_area",
    "GAResult", "Mutations", "init_state", "run_ga", "run_ga_batch", "sample_distinct", "srt_shape",
    "neighbor_counts", "neighbor_tuples", "num_rules", "reorder_two_state_rules", "rule_index",
    "rules_per_state", "validate_srt",
]
