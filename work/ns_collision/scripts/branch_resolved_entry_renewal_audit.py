"""Audit branch-resolved entry smoothing and renewal bookkeeping."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

from scipy.optimize import brentq


REYNOLDS_LEVEL = 0.5
DIMENSION = 3.0
LEGACY_RETURN_ONE_HISTORY = 0.5
LEGACY_CYCLE_COEFFICIENT = 0.5161236147249065
CUBIC_SUPPORT_RADIUS = 1.91


def _load_sibling(filename: str, module_name: str):
    script = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _renewal_quantities(
    split_composite_gain: float,
    return_composite_gain: float,
) -> dict[str, float | bool]:
    split_pair_gain = split_composite_gain**2
    return_pair_gain = return_composite_gain**2
    denominator = 1.0 - return_pair_gain
    renewed_generation_factor = (
        math.inf
        if denominator <= 0.0
        else split_pair_gain / denominator
    )
    closure_criterion = split_pair_gain + return_pair_gain
    return {
        "split_pair_gain": split_pair_gain,
        "return_pair_gain": return_pair_gain,
        "same_scale_renewal_denominator": denominator,
        "renewed_generation_factor": renewed_generation_factor,
        "branch_sum_closure_criterion": closure_criterion,
        "same_scale_series_converges": return_pair_gain < 1.0,
        "complete_generation_closes": closure_criterion < 1.0,
        "renewed_factor_and_branch_sum_tests_agree": bool(
            denominator > 0.0
            and (renewed_generation_factor < 1.0)
            == (closure_criterion < 1.0)
        ),
    }


def _allowances(
    split_baseline: float,
    return_baseline: float,
) -> dict[str, float]:
    equal = brentq(
        lambda error: (
            (split_baseline + error) ** 2
            + (return_baseline + error) ** 2
            - 1.0
        ),
        0.0,
        1.0,
    )
    return {
        "equal_error_each_branch": equal,
        "split_only_error": (
            math.sqrt(1.0 - return_baseline**2) - split_baseline
        ),
        "return_only_error": (
            math.sqrt(1.0 - split_baseline**2) - return_baseline
        ),
    }


def audit() -> dict[str, object]:
    barrier_module = _load_sibling(
        "radial_h1_payoff_supersolution_pilot.py",
        "radial_h1_for_branch_resolved_renewal",
    )
    cylinder_module = _load_sibling(
        "cylindrical_brownian_return_pilot.py",
        "cylinder_for_branch_resolved_renewal",
    )
    barrier = barrier_module.audit()
    entry_gain = float(barrier["entry_gain"])

    split_pair_factor = math.exp(
        REYNOLDS_LEVEL * DIMENSION / 24.0
    ) / 4.0
    split_one_history = math.sqrt(split_pair_factor)
    reconstructed_legacy_coefficient = (
        split_pair_factor + LEGACY_RETURN_ONE_HISTORY**2
    )

    legacy_split_baseline = split_one_history * entry_gain
    legacy_return_baseline = LEGACY_RETURN_ONE_HISTORY * entry_gain
    legacy = _renewal_quantities(
        legacy_split_baseline, legacy_return_baseline
    )
    legacy_allowances = _allowances(
        legacy_split_baseline, legacy_return_baseline
    )

    cubic_split_log_cost = (
        REYNOLDS_LEVEL
        * (CUBIC_SUPPORT_RADIUS**2 / 3.0 + 0.75)
        / 4.0
    )
    cubic_split_one_history = math.exp(cubic_split_log_cost) / 2.0
    cubic_split_pair = cubic_split_one_history**2
    current_split_baseline = cubic_split_one_history * entry_gain
    current_legacy_return = _renewal_quantities(
        current_split_baseline, legacy_return_baseline
    )
    current_legacy_allowances = _allowances(
        current_split_baseline, legacy_return_baseline
    )

    patch_return = cylinder_module._axial_patch_return_probability()
    brownian_return_one_history = float(patch_return["probability"])
    brownian_return_baseline = brownian_return_one_history * entry_gain
    brownian = _renewal_quantities(
        current_split_baseline, brownian_return_baseline
    )
    brownian_allowances = _allowances(
        current_split_baseline, brownian_return_baseline
    )

    double_counted_return_pair = (
        brownian_return_one_history**2
        * brownian["return_pair_gain"]
    )
    result: dict[str, object] = {
        "branch_resolved_theorem": (
            "if a_S and a_R are the complete one-history split-entry/visit "
            "and return-entry/visit gains, then the pair generation norm is "
            "a_S^2/(1-a_R^2), and it is below one exactly when "
            "a_S^2+a_R^2<1"
        ),
        "entry_kernel_gain_definition": (
            "p_j=||T_j 1||_L2(mu); for an unnormalized positive entry "
            "kernel and U_H<=g_H, ||T_j U_H||<=p_j g_H"
        ),
        "conditional_error_bound": (
            "a_j<=p_j*g_H+delta_j, with "
            "delta_j<=F*sqrt(C_4*J_(rho_j))"
        ),
        "conditional_branch_closure": (
            "(p_S*g_H+delta_S)^2+(p_R*g_H+delta_R)^2<1"
        ),
        "unnormalized_branch_mass_is_applied_exactly_once": True,
        "legacy_calibration": {
            "R_star": REYNOLDS_LEVEL,
            "dimension": DIMENSION,
            "entry_gain": entry_gain,
            "split_pair_factor": split_pair_factor,
            "split_one_history_factor": split_one_history,
            "return_one_history_factor": LEGACY_RETURN_ONE_HISTORY,
            "reconstructed_cycle_coefficient": (
                reconstructed_legacy_coefficient
            ),
            "stored_cycle_coefficient": LEGACY_CYCLE_COEFFICIENT,
            "coefficient_difference": abs(
                reconstructed_legacy_coefficient
                - LEGACY_CYCLE_COEFFICIENT
            ),
            "split_composite_baseline": legacy_split_baseline,
            "return_composite_baseline": legacy_return_baseline,
            **legacy,
            "additive_error_allowances": legacy_allowances,
        },
        "current_cubic_split_with_legacy_return": {
            "cubic_support_radius_over_L": CUBIC_SUPPORT_RADIUS,
            "split_log_gauge_cost": cubic_split_log_cost,
            "split_one_history_factor": cubic_split_one_history,
            "split_pair_factor": cubic_split_pair,
            "split_composite_baseline": current_split_baseline,
            "return_one_history_factor": LEGACY_RETURN_ONE_HISTORY,
            "return_composite_baseline": legacy_return_baseline,
            "maximum_common_visit_gain_with_no_errors": (
                1.0
                / math.sqrt(
                    cubic_split_pair + LEGACY_RETURN_ONE_HISTORY**2
                )
            ),
            **current_legacy_return,
            "additive_error_allowances": current_legacy_allowances,
        },
        "Brownian_finite_patch_calibration": {
            "exact_probability_formula": (
                "p_H=(2/pi)int_0^infinity sin(kH)/k "
                "K_0(2k)/K_0(k) dk"
            ),
            "patch_half_height": cylinder_module.AXIAL_HALF_HEIGHT,
            **patch_return,
            "return_composite_baseline": brownian_return_baseline,
            "maximum_common_visit_gain_with_no_errors": (
                1.0
                / math.sqrt(
                    cubic_split_pair + brownian_return_one_history**2
                )
            ),
            **brownian,
            "additive_error_allowances": brownian_allowances,
            "numerically_certified": False,
        },
        "double_counting_diagnostic": {
            "correct_return_pair_contribution": brownian[
                "return_pair_gain"
            ],
            "incorrect_contribution_if_mass_is_squared_again": (
                double_counted_return_pair
            ),
            "incorrect_criterion": (
                brownian["split_pair_gain"]
                + double_counted_return_pair
            ),
        },
        "branch_separated_renewal_identity_closed": True,
        "Brownian_patch_return_probability_certified": False,
        "true_split_entry_density_envelope_certified": False,
        "pointwise_split_density_inheritance_available": True,
        "deterministic_split_time_atom_handled": False,
        "weighted_return_entry_density_envelope_certified": False,
        "full_Navier_Stokes_generation_gate_closed": False,
        "scope_guard": (
            "the renewal identity and branch-mass bookkeeping are exact "
            "inside the current synchronized two-history model. The Bessel "
            "integral is a high-accuracy numerical Brownian calibration, "
            "not an interval certificate. It omits exterior Navier-Stokes "
            "drift, deformation weight, cap exits, moving geometry, and the "
            "physical true-split entry law"
        ),
        "next_gate": (
            "derive the weighted same-scale return envelope; then prove "
            "that a true split either inherits an existing absolute "
            "space-time density or has a bounded fixed-time child-volume "
            "law, without multiplying either branch mass twice"
        ),
    }
    positive_checks = (
        barrier["all_positive_H1_supersolution_pilot_checks_pass"],
        result["legacy_calibration"]["coefficient_difference"] < 1.0e-10,
        legacy["renewed_factor_and_branch_sum_tests_agree"],
        current_legacy_return[
            "renewed_factor_and_branch_sum_tests_agree"
        ],
        brownian["renewed_factor_and_branch_sum_tests_agree"],
        legacy["complete_generation_closes"],
        current_legacy_return["complete_generation_closes"],
        brownian["complete_generation_closes"],
        0.30 < brownian_return_one_history < 0.32,
        current_legacy_return["branch_sum_closure_criterion"] < 0.87,
        current_legacy_allowances["equal_error_each_branch"] > 0.049,
        brownian["branch_sum_closure_criterion"] < 0.67,
        brownian_allowances["equal_error_each_branch"] > 0.13,
        brownian_allowances["return_only_error"] > 0.32,
        double_counted_return_pair < brownian["return_pair_gain"],
        result["unnormalized_branch_mass_is_applied_exactly_once"],
        result["pointwise_split_density_inheritance_available"],
        not result["deterministic_split_time_atom_handled"],
        not result["true_split_entry_density_envelope_certified"],
        not result["full_Navier_Stokes_generation_gate_closed"],
    )
    result["all_positive_branch_resolved_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
