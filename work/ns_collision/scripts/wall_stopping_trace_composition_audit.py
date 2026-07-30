"""Audit branch-resolved trace composition at a migrating wall stop."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import brentq


FORM_FLOOR = 4.832287335665
TRACE_L4_FORM_CONSTANT = 0.6741481379606137
BROWNIAN_CYLINDER_STRESS_J0 = 0.7964751972107911
BROWNIAN_PATCH_RETURN_MASS = 0.3101351513711487
POTENTIAL_L3_OVER_2_FORCING = 0.7989685513198063
DRIFT_L3_FORCING = 3.072840583265365
POTENTIAL_RELATIVE_FORM = 0.2203290376862308


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _positive_kernel_audit() -> dict[str, object]:
    rng = np.random.default_rng(20260720)
    kernel = rng.uniform(0.01, 1.2, size=(7, 9))
    entry_law = rng.uniform(0.01, 1.0, size=7)
    entry_law /= np.sum(entry_law)
    constant_payoff = np.sum(kernel, axis=1)
    norm_squared = float(
        np.dot(entry_law, constant_payoff**2)
    )
    tilted_exit_law = (
        (entry_law * constant_payoff) @ kernel / norm_squared
    )

    ratios = []
    for _ in range(200):
        payoff = rng.normal(size=kernel.shape[1])
        input_norm_squared = float(
            np.dot(tilted_exit_law, payoff**2)
        )
        output = kernel @ payoff
        output_norm_squared = float(np.dot(entry_law, output**2))
        if input_norm_squared > 0.0:
            ratios.append(output_norm_squared / input_norm_squared)

    constant_output_norm_squared = float(
        np.dot(entry_law, (kernel @ np.ones(kernel.shape[1])) ** 2)
    )
    constant_input_norm_squared = float(np.sum(tilted_exit_law))
    return {
        "square_tilted_exit_mass": float(np.sum(tilted_exit_law)),
        "exact_kernel_norm_squared": norm_squared,
        "maximum_random_payoff_norm_ratio_squared": max(ratios),
        "constant_payoff_norm_ratio_squared": (
            constant_output_norm_squared / constant_input_norm_squared
        ),
        "square_tilted_positive_kernel_identity_verified": bool(
            abs(np.sum(tilted_exit_law) - 1.0) < 2.0e-15
            and max(ratios) <= norm_squared * (1.0 + 2.0e-14)
            and abs(
                constant_output_norm_squared / constant_input_norm_squared
                - norm_squared
            )
            < 2.0e-13
        ),
    }


def _density_scaling_audit() -> dict[str, object]:
    mass, trace_constant, normalized_factor = sp.symbols(
        "p C_4 J_norm", positive=True
    )
    raw_factor = mass * normalized_factor
    normalized_response = mass * sp.sqrt(
        trace_constant * normalized_factor
    )
    raw_response = sp.sqrt(
        mass * trace_constant * raw_factor
    )
    return {
        "square_tilted_response_formula": (
            "delta_j<=p_j*F*sqrt(C_4*J_tilted_j(alpha))"
        ),
        "raw_density_response_formula": (
            "delta_j<=F*sqrt(p_j*C_4*J_raw_j(alpha))"
        ),
        "raw_to_normalized_factor_law": "J_raw_j=p_j*J_tilted_j",
        "raw_and_square_tilted_forms_agree": bool(
            sp.simplify(normalized_response - raw_response) == 0
        ),
    }


def _conservative_J_bound(alpha: float) -> float:
    if not 0.0 <= alpha < 1.0:
        return math.inf
    # At the alpha=0 optimizing window, each interval energy term grows
    # by at most (1-alpha)^(-3).
    return BROWNIAN_CYLINDER_STRESS_J0 / (1.0 - alpha) ** 3


def _branch_calibration() -> dict[str, object]:
    residual = _load_module(
        "migrating_core_residual_budget_audit.py",
        "residual_for_trace_composition",
    )
    budget = residual._numerical_budget()
    entry_gain = float(budget["entry_gain"])
    trace_response_at_zero = math.sqrt(
        TRACE_L4_FORM_CONSTANT * BROWNIAN_CYLINDER_STRESS_J0
    )
    rows = []
    for row in budget["angle_rows"]:
        return_baseline = float(row["return_one_history_gain"])
        wall_baseline = float(row["wall_one_history_gain"])
        return_scalar = return_baseline / entry_gain
        wall_scalar = wall_baseline / entry_gain
        return_allowance = float(
            row["return_only_additive_gain_allowance"]
        )

        normalized_return_response = (
            return_scalar * trace_response_at_zero
        )
        drift_threshold = return_allowance / (
            normalized_return_response * DRIFT_L3_FORCING
        )

        potential_upper = 0.999 / POTENTIAL_RELATIVE_FORM

        def potential_gate(potential_mass: float) -> float:
            alpha = POTENTIAL_RELATIVE_FORM * potential_mass
            response = (
                return_scalar
                * POTENTIAL_L3_OVER_2_FORCING
                * potential_mass
                * math.sqrt(
                    TRACE_L4_FORM_CONSTANT
                    * _conservative_J_bound(alpha)
                )
            )
            return (
                (return_baseline + response) ** 2
                + wall_baseline**2
                - 1.0
            )

        potential_threshold = brentq(
            potential_gate, 0.0, potential_upper
        )
        rows.append(
            {
                "angle": row["angle"],
                "return_scalar_branch_gain": return_scalar,
                "wall_scalar_branch_gain": wall_scalar,
                "return_baseline_one_history_gain": return_baseline,
                "wall_baseline_one_history_gain": wall_baseline,
                "return_only_additive_gain_allowance": return_allowance,
                "conditional_normalized_return_response_at_alpha_zero": (
                    normalized_return_response
                ),
                "conditional_return_only_potential_L3_over_2_threshold": (
                    potential_threshold
                ),
                "conditional_return_only_drift_L3_threshold": (
                    drift_threshold
                ),
            }
        )

    potential_worst = min(
        rows,
        key=lambda row: row[
            "conditional_return_only_potential_L3_over_2_threshold"
        ],
    )
    drift_worst = min(
        rows,
        key=lambda row: row[
            "conditional_return_only_drift_L3_threshold"
        ],
    )
    wall_action = math.log(
        float(budget["minimum_wall_only_multiplicative_ceiling"])
    )
    return_action = math.log(
        float(budget["minimum_return_only_multiplicative_ceiling"])
    )
    finite_patch_raw_response = math.sqrt(
        BROWNIAN_PATCH_RETURN_MASS
        * TRACE_L4_FORM_CONSTANT
        * BROWNIAN_CYLINDER_STRESS_J0
    )

    return {
        "working_pair_criterion": budget[
            "maximum_baseline_pair_criterion"
        ],
        "angle_rows": rows,
        "Brownian_cylinder_stress_J_at_alpha_zero": (
            BROWNIAN_CYLINDER_STRESS_J0
        ),
        "Brownian_cylinder_L4_trace_response_at_alpha_zero": (
            trace_response_at_zero
        ),
        "Brownian_finite_patch_return_mass": BROWNIAN_PATCH_RETURN_MASS,
        "Brownian_finite_patch_raw_response_upper_at_alpha_zero": (
            finite_patch_raw_response
        ),
        "conservative_alpha_inflation": (
            "J(alpha)<=J(0)/(1-alpha)^3 at the alpha=0 pilot window"
        ),
        "conditional_minimum_return_only_potential_L3_over_2_threshold": (
            potential_worst[
                "conditional_return_only_potential_L3_over_2_threshold"
            ]
        ),
        "conditional_worst_potential_angle": potential_worst["angle"],
        "conditional_minimum_return_only_drift_L3_threshold": drift_worst[
            "conditional_return_only_drift_L3_threshold"
        ],
        "conditional_worst_drift_angle": drift_worst["angle"],
        "wall_only_integrated_log_action_ceiling": wall_action,
        "return_only_integrated_log_action_ceiling": return_action,
    }


def audit() -> dict[str, object]:
    result = {
        **_positive_kernel_audit(),
        **_density_scaling_audit(),
        **_branch_calibration(),
        "return_branch_H1_target": (
            "same-scale storage return lands on child/core Sigma={r=1}"
        ),
        "raw_wall_branch_target": (
            "migration lands on child outer surface {r=2}, not H1 Sigma"
        ),
        "core_entry_wall_composite_kernel": (
            "B_S_core = B_wall * M_migrate * B_child_return"
        ),
        "raw_wall_flux_can_be_used_as_H1_trace_law": False,
        "migration_residual_is_wall_branch_only": True,
        "return_core_residual_and_migration_residual_must_be_separated": True,
        "conditional_return_trace_theorem": (
            "if the square-tilted return law has spatial-L2 interval factor "
            "J_R(alpha), then K_R=p_R*sqrt(C_4*J_R(alpha))"
        ),
        "conditional_wall_trace_theorem": (
            "K_S=p_S*sqrt(C_4*J_S(alpha)) only after J_S is proved for "
            "the composite wall-migrate-child-return law"
        ),
        "Brownian_cylinder_envelope_certified": False,
        "actual_return_square_tilted_density_envelope_certified": False,
        "composite_wall_to_child_core_density_envelope_certified": False,
        "full_wall_stopping_trace_gate_closed": False,
        "scope_guard": (
            "the positive-kernel and trace-composition formulas are exact. "
            "The numerical critical-norm rows assume that each normalized "
            "return law obeys the stored inflated Brownian-cylinder pilot "
            "envelope; that envelope is not a certificate and does not "
            "cover the weighted affine or Navier-Stokes strip. No K_S is "
            "assigned to the raw wall law because it targets the wrong "
            "surface"
        ),
        "next_gate": (
            "record boundary-resolved time flux for the neutral-strip "
            "return, form its square-tilted spatial-L2 envelope, and compose "
            "the wall flux with migration and one child return before "
            "estimating K_S"
        ),
    }
    checks = (
        result["square_tilted_positive_kernel_identity_verified"],
        result["raw_and_square_tilted_forms_agree"],
        abs(result["working_pair_criterion"] - 0.6721268902914064)
        < 2.0e-12,
        0.72
        < result["Brownian_cylinder_L4_trace_response_at_alpha_zero"]
        < 0.74,
        0.40
        < result["Brownian_finite_patch_raw_response_upper_at_alpha_zero"]
        < 0.42,
        result["wall_only_integrated_log_action_ceiling"] > 0.29,
        result["migration_residual_is_wall_branch_only"],
        result[
            "return_core_residual_and_migration_residual_must_be_separated"
        ],
        not result["raw_wall_flux_can_be_used_as_H1_trace_law"],
        not result["actual_return_square_tilted_density_envelope_certified"],
        not result[
            "composite_wall_to_child_core_density_envelope_certified"
        ],
        not result["full_wall_stopping_trace_gate_closed"],
    )
    result["all_positive_wall_stopping_trace_checks_pass"] = all(checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
