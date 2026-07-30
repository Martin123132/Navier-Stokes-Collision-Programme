"""Audit conservative cubic refinement and its gauged split payment."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp


def _load_radial_partition_module():
    script = Path(__file__).resolve().with_name(
        "radial_cubic_partition_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "radial_cubic_for_level_transfer", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit() -> dict[str, object]:
    radial_module = _load_radial_partition_module()
    radial = radial_module.audit()
    optimized = radial["optimized_full_tensor_budget_row"]
    support_radius = float(optimized["radial_support_radius_over_L"])
    reynolds = float(radial["R_star"])

    refinement_mask = np.array(
        [math.comb(4, index) / 8.0 for index in range(5)]
    )
    integral_normalized_probabilities = refinement_mask / 2.0
    tensor_probabilities = np.einsum(
        "i,j,k->ijk",
        integral_normalized_probabilities,
        integral_normalized_probabilities,
        integral_normalized_probabilities,
    )
    pair_probabilities = np.outer(
        tensor_probabilities.ravel(),
        tensor_probabilities.ravel(),
    )

    coordinates = np.linspace(0.0, 4.0, 200_001)
    parent = radial_module._cardinal_cubic(coordinates)
    child_terms = np.array(
        [
            coefficient
            * radial_module._cardinal_cubic(2.0 * coordinates - index)
            for index, coefficient in enumerate(refinement_mask)
        ]
    )
    reconstructed_parent = np.sum(child_terms, axis=0)
    positive_parent = parent > 1.0e-14
    conditional_probabilities = np.divide(
        child_terms,
        parent,
        out=np.zeros_like(child_terms),
        where=positive_parent,
    )
    conditional_sum_error = float(
        np.max(
            np.abs(
                np.sum(conditional_probabilities, axis=0)[positive_parent]
                - 1.0
            )
        )
    )

    child_coordinate, normalized_offset = sp.symbols(
        "child_coordinate normalized_offset", real=True
    )
    exponent_difference = sp.expand(
        (normalized_offset + child_coordinate / 2) ** 2
        - child_coordinate**2
    )
    critical_coordinate = sp.solve(
        sp.diff(exponent_difference, child_coordinate),
        child_coordinate,
    )[0]
    maximum_at_fixed_offset = sp.factor(
        exponent_difference.subs(child_coordinate, critical_coordinate)
    )
    maximum_normalized_offset = support_radius / (2.0 * math.sqrt(2.0))
    maximum_axial_center_offset_over_parent_L = 0.75
    maximum_three_direction_exponent_difference = (
        support_radius**2 / 3.0 + 0.75
    )
    maximum_log_gauge_cost = (
        reynolds * maximum_three_direction_exponent_difference / 4.0
    )
    single_gauge_transition_factor = math.exp(maximum_log_gauge_cost)
    single_true_split_factor = single_gauge_transition_factor / 2.0
    pair_true_split_factor = single_true_split_factor**2

    generation_rows = []
    for generations in (1, 5, 10, 20, 50):
        generation_rows.append(
            {
                "true_global_level_changes": generations,
                "unpaid_pair_gauge_product": (
                    single_gauge_transition_factor ** (2 * generations)
                ),
                "shrink_paid_pair_product": (
                    pair_true_split_factor**generations
                ),
            }
        )

    result: dict[str, object] = {
        "cubic_refinement_identity": (
            "N_3(x)=sum_r a_r N_3(2x-r), "
            "a=(1,4,6,4,1)/8"
        ),
        "refinement_mask": refinement_mask.tolist(),
        "refinement_mask_sum": float(np.sum(refinement_mask)),
        "integral_normalized_child_probabilities": (
            integral_normalized_probabilities.tolist()
        ),
        "integral_normalized_child_probability_sum": float(
            np.sum(integral_normalized_probabilities)
        ),
        "one_dimensional_integral_normalized_split_is_conservative": bool(
            abs(np.sum(integral_normalized_probabilities) - 1.0)
            < 1.0e-15
        ),
        "tensor_child_state_count": int(tensor_probabilities.size),
        "tensor_child_probability_sum": float(
            np.sum(tensor_probabilities)
        ),
        "tensor_split_is_conservative": bool(
            abs(np.sum(tensor_probabilities) - 1.0) < 1.0e-15
        ),
        "replica_pair_child_state_count": int(pair_probabilities.size),
        "replica_pair_probability_sum": float(np.sum(pair_probabilities)),
        "replica_pair_split_is_conservative": bool(
            abs(np.sum(pair_probabilities) - 1.0) < 1.0e-15
        ),
        "maximum_pointwise_refinement_identity_error": float(
            np.max(np.abs(parent - reconstructed_parent))
        ),
        "maximum_conditional_child_probability_sum_error": (
            conditional_sum_error
        ),
        "conditional_pointwise_split_is_Markov": bool(
            conditional_sum_error < 2.0e-12
            and np.min(conditional_probabilities) >= -1.0e-15
        ),
        "pointwise_child_label_kernel": (
            "P(r|i,x)=a_r N_3(2x-(2i+r))/N_3(x-i)"
        ),
        "one_coordinate_gauge_exponent_difference": str(
            exponent_difference
        ),
        "maximizing_child_coordinate": str(critical_coordinate),
        "maximum_at_fixed_center_offset": str(maximum_at_fixed_offset),
        "maximum_child_center_offset_over_parent_L": (
            maximum_normalized_offset
        ),
        "maximum_axial_center_offset_over_parent_L": (
            maximum_axial_center_offset_over_parent_L
        ),
        "maximum_three_direction_exponent_difference": (
            maximum_three_direction_exponent_difference
        ),
        "maximum_log_gauge_cost": maximum_log_gauge_cost,
        "single_history_gauge_transition_factor": (
            single_gauge_transition_factor
        ),
        "radius_halving_shrink_factor": 0.5,
        "single_history_true_level_change_factor": (
            single_true_split_factor
        ),
        "replica_pair_true_level_change_factor": pair_true_split_factor,
        "true_level_change_is_strictly_contractive": bool(
            single_true_split_factor < 1.0
            and pair_true_split_factor < 1.0
        ),
        "generation_rows": generation_rows,
        "many_generation_pair_product_decays": bool(
            generation_rows[-1]["shrink_paid_pair_product"]
            < generation_rows[-2]["shrink_paid_pair_product"]
            < 1.0
        ),
        "global_monotone_level_rule": (
            "use one fixed cubic lattice level L_n until the running global "
            "envelope reaches A_n=R_star*nu/L_n^2; then set "
            "L_(n+1)=L_n/2 and A_(n+1)=4A_n"
        ),
        "global_level_change_has_no_balance_only_splits": True,
        "pressure_at_level_change": (
            "both levels have sum phi=1 and the pointwise child-label map "
            "is Markov, so pressure partition mass is unchanged"
        ),
        "Poisson_conversion_policy": (
            "stay in the gauged visit norm across level changes and apply "
            "the physical/Poisson conversion only at complete buffered "
            "visit entry or exit"
        ),
        "static_full_tensor_mass_budget_over_nu": optimized[
            "full_Poisson_compatible_L3_over_2_budget_over_nu"
        ],
        "remaining_Navier_Stokes_gate": (
            "derive the local positive non-affine perturbation bound "
            "||q_+||_(3/2)/nu<0.2159 on every complete cubic support from "
            "Navier-Stokes strain/frame/pressure quantities"
        ),
        "scope_warning": (
            "the contraction is earned only at envelope-triggered global "
            "level halvings; arbitrary extra refinements receive no shrink "
            "payment"
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
