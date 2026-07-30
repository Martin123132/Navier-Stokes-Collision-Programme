"""Audit conservative octree branching and replica-pair interface transfer."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm


def _load_transition_module():
    script = Path(__file__).resolve().with_name(
        "dyadic_gauge_transition_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dyadic_transition_for_branching", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _induced_l1_norm(matrix: np.ndarray) -> float:
    return float(np.max(np.sum(np.abs(matrix), axis=0)))


def _build_interface_generator() -> np.ndarray:
    child_bits = list(itertools.product((0, 1), repeat=3))
    generator = np.zeros((len(child_bits), len(child_bits)))
    for source, source_bits in enumerate(child_bits):
        outgoing_rate = 0.0
        for target, target_bits in enumerate(child_bits):
            differing_coordinates = sum(
                source_bit != target_bit
                for source_bit, target_bit in zip(source_bits, target_bits)
            )
            if differing_coordinates != 1:
                continue
            rate = 0.35 + 0.07 * (source + 1) + 0.03 * (target + 1)
            generator[target, source] = rate
            outgoing_rate += rate
        generator[source, source] = -outgoing_rate
    return generator


def _build_child_weights() -> np.ndarray:
    return np.exp(np.linspace(-0.45, 0.45, 8))


def audit() -> dict[str, object]:
    child_count = 8
    child_probabilities = np.arange(1.0, child_count + 1.0)
    child_probabilities /= np.sum(child_probabilities)
    pair_probabilities = np.kron(
        child_probabilities, child_probabilities
    )

    interface_generator = _build_interface_generator()

    interface_time = 0.7
    interface_semigroup = expm(interface_time * interface_generator)
    identity = np.eye(child_count)
    pair_interface_generator = (
        np.kron(interface_generator, identity)
        + np.kron(identity, interface_generator)
    )
    pair_interface_semigroup = expm(
        interface_time * pair_interface_generator
    )

    child_weights = _build_child_weights()
    weight_matrix = np.diag(child_weights)
    weighted_interface_semigroup = (
        weight_matrix
        @ interface_semigroup
        @ np.diag(1.0 / child_weights)
    )
    pair_weights = np.kron(child_weights, child_weights)
    pair_weight_matrix = np.diag(pair_weights)
    weighted_pair_interface_semigroup = (
        pair_weight_matrix
        @ pair_interface_semigroup
        @ np.diag(1.0 / pair_weights)
    )

    harmonic_parent_weight = float(
        np.dot(child_probabilities, child_weights)
    )
    compatible_single_transfer_norm = float(
        np.dot(child_probabilities, child_weights)
        / harmonic_parent_weight
    )
    compatible_pair_transfer_norm = float(
        np.dot(pair_probabilities, pair_weights)
        / harmonic_parent_weight**2
    )
    unit_parent_single_weighted_factor = float(
        np.dot(child_probabilities, child_weights)
    )
    unit_parent_pair_weighted_factor = float(
        np.dot(pair_probabilities, pair_weights)
    )

    transition_result = _load_transition_module().audit()
    single_true_split_factor = float(
        transition_result["audited_one_history_net_factor"]
    )
    pair_true_split_factor = float(
        transition_result["audited_pair_net_factor"]
    )
    contracted_single_branch = (
        single_true_split_factor * child_probabilities
    )
    contracted_pair_branch = pair_true_split_factor * pair_probabilities

    generation_rows = []
    for generations in (1, 5, 10, 20, 50):
        generation_rows.append(
            {
                "true_split_generations": generations,
                "replica_pair_branch_count": str(64**generations),
                "total_pair_mass": pair_true_split_factor**generations,
                "incorrect_branch_count_bound": (
                    (64.0 * pair_true_split_factor) ** generations
                ),
            }
        )

    single_interface_norm = _induced_l1_norm(interface_semigroup)
    pair_interface_norm = _induced_l1_norm(pair_interface_semigroup)
    weighted_single_interface_norm = _induced_l1_norm(
        weighted_interface_semigroup
    )
    weighted_pair_interface_norm = _induced_l1_norm(
        weighted_pair_interface_semigroup
    )
    pair_split_log_budget = -math.log(pair_true_split_factor)

    result: dict[str, object] = {
        "child_probabilities": child_probabilities.tolist(),
        "single_branch_probability_sum": float(
            np.sum(child_probabilities)
        ),
        "single_branching_is_conservative": bool(
            abs(np.sum(child_probabilities) - 1.0) < 1.0e-14
        ),
        "replica_pair_branch_count": 64,
        "pair_branch_probability_sum": float(np.sum(pair_probabilities)),
        "pair_branching_is_conservative": bool(
            abs(np.sum(pair_probabilities) - 1.0) < 1.0e-14
        ),
        "interface_generator_maximum_column_sum": float(
            np.max(np.abs(np.sum(interface_generator, axis=0)))
        ),
        "interface_semigroup_minimum_entry": float(
            np.min(interface_semigroup)
        ),
        "interface_semigroup_maximum_column_sum_error": float(
            np.max(np.abs(np.sum(interface_semigroup, axis=0) - 1.0))
        ),
        "single_interface_physical_l1_norm": single_interface_norm,
        "single_interface_is_physical_l1_contractive": bool(
            abs(single_interface_norm - 1.0) < 1.0e-12
            and np.min(interface_semigroup) >= -1.0e-14
        ),
        "pair_interface_semigroup_maximum_column_sum_error": float(
            np.max(
                np.abs(np.sum(pair_interface_semigroup, axis=0) - 1.0)
            )
        ),
        "pair_interface_physical_l1_norm": pair_interface_norm,
        "pair_interface_is_physical_l1_contractive": bool(
            abs(pair_interface_norm - 1.0) < 1.0e-12
            and np.min(pair_interface_semigroup) >= -1.0e-14
        ),
        "child_gauge_weights": child_weights.tolist(),
        "harmonic_parent_weight": harmonic_parent_weight,
        "compatible_single_weighted_transfer_norm": (
            compatible_single_transfer_norm
        ),
        "compatible_pair_weighted_transfer_norm": (
            compatible_pair_transfer_norm
        ),
        "harmonic_weight_inheritance_is_nonexpansive": bool(
            abs(compatible_single_transfer_norm - 1.0) < 1.0e-14
            and abs(compatible_pair_transfer_norm - 1.0) < 1.0e-14
        ),
        "unit_parent_single_weighted_factor": (
            unit_parent_single_weighted_factor
        ),
        "unit_parent_pair_weighted_factor": unit_parent_pair_weighted_factor,
        "noncompatible_branch_weights_can_amplify": bool(
            unit_parent_pair_weighted_factor > 1.0
        ),
        "weighted_single_interface_l1_norm": (
            weighted_single_interface_norm
        ),
        "weighted_pair_interface_l1_norm": weighted_pair_interface_norm,
        "nonuniform_weights_can_amplify_interface_transfer": bool(
            weighted_single_interface_norm > 1.0
            and weighted_pair_interface_norm > 1.0
        ),
        "single_true_split_factor": single_true_split_factor,
        "pair_true_split_factor": pair_true_split_factor,
        "contracted_single_branch_l1_norm": float(
            np.sum(np.abs(contracted_single_branch))
        ),
        "contracted_pair_branch_l1_norm": float(
            np.sum(np.abs(contracted_pair_branch))
        ),
        "true_split_contraction_survives_branching": bool(
            abs(
                np.sum(np.abs(contracted_single_branch))
                - single_true_split_factor
            )
            < 1.0e-14
            and abs(
                np.sum(np.abs(contracted_pair_branch))
                - pair_true_split_factor
            )
            < 1.0e-14
        ),
        "pair_true_split_log_mismatch_budget": pair_split_log_budget,
        "generation_rows": generation_rows,
        "branch_count_does_not_create_mass": bool(
            generation_rows[-1]["total_pair_mass"] < 1.0
            and generation_rows[-1]["incorrect_branch_count_bound"] > 1.0
        ),
        "conditional_cycle_criterion": (
            "pair_true_split_factor*M_balance*M_interface*M_pressure*"
            "M_renewal<1"
        ),
        "balance_split_requirement": (
            "balance-only refinements must be conservative in one common "
            "physical norm or charged to a bounded number of true splits"
        ),
        "remaining_weight_gate": (
            "construct partition/gauge weights with harmonic parent-child "
            "inheritance and controlled pressure edge mismatch"
        ),
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
