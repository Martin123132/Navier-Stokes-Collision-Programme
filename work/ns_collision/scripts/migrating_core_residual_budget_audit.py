"""Audit the residual budget for a smoothly migrating half-scale core."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import sympy as sp


R_STAR = 0.5
ENTRY_RADIUS = 2.0
WORKING_HALF_WIDTH = 2.1
TARGET_SPACING = 0.075

# Independently validated by radial_h1_payoff_supersolution_pilot.py.
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


def _effective_potential(
    error: sp.Matrix,
    stretching_error: sp.Expr,
    coordinates: tuple[sp.Symbol, sp.Symbol],
    reynolds: sp.Symbol,
) -> sp.Expr:
    y_1, y_2 = coordinates
    divergence = sp.diff(error[0], y_1) + sp.diff(error[1], y_2)
    radial_moment = y_1 * error[0] + y_2 * error[1]
    return sp.expand(
        stretching_error - divergence / 2 - reynolds * radial_moment / 2
    )


def _symbolic_audit() -> dict[str, object]:
    y_1, y_2 = sp.symbols("y_1 y_2", real=True)
    reynolds, scale_rate, entry_radius = sp.symbols(
        "R ell eta", real=True
    )
    amplitude_deficit, angular_rate = sp.symbols(
        "delta_a omega", real=True
    )
    residual_stretching = sp.symbols("delta_s_res", real=True)
    residual_1 = sp.Function("e_res_1")(y_1, y_2)
    residual_2 = sp.Function("e_res_2")(y_1, y_2)
    coordinates = (y_1, y_2)

    geometry_error = sp.Matrix(
        [scale_rate * (entry_radius - y_1), -scale_rate * y_2]
    )
    amplitude_error = amplitude_deficit * sp.Matrix([y_1, y_2])
    rotation_error = sp.Matrix(
        [angular_rate * y_2, -angular_rate * y_1]
    )
    residual_error = sp.Matrix([residual_1, residual_2])

    geometry_potential = sp.factor(
        _effective_potential(
            geometry_error, sp.Integer(0), coordinates, reynolds
        )
    )
    expected_geometry = sp.factor(
        scale_rate
        * (
            1
            + reynolds
            * (y_1**2 + y_2**2 - entry_radius * y_1)
            / 2
        )
    )
    amplitude_potential = sp.factor(
        _effective_potential(
            amplitude_error,
            2 * amplitude_deficit,
            coordinates,
            reynolds,
        )
    )
    rotation_potential = sp.factor(
        _effective_potential(
            rotation_error, sp.Integer(0), coordinates, reynolds
        )
    )
    residual_potential = sp.factor(
        _effective_potential(
            residual_error, residual_stretching, coordinates, reynolds
        )
    )
    total_potential = sp.factor(
        _effective_potential(
            geometry_error
            + amplitude_error
            + rotation_error
            + residual_error,
            2 * amplitude_deficit + residual_stretching,
            coordinates,
            reynolds,
        )
    )
    decomposed_potential = sp.expand(
        geometry_potential
        + amplitude_potential
        + rotation_potential
        + residual_potential
    )

    geometry_bracket = sp.factor(geometry_potential / scale_rate)
    completed_bracket = (
        1
        - reynolds * entry_radius**2 / 8
        + reynolds
        * ((y_1 - entry_radius / 2) ** 2 + y_2**2)
        / 2
    )
    working_bracket = sp.factor(
        geometry_bracket.subs(
            {reynolds: sp.Rational(1, 2), entry_radius: 2}
        )
    )
    working_completed_bracket = (
        completed_bracket.subs(
            {reynolds: sp.Rational(1, 2), entry_radius: 2}
        )
    )

    return {
        "mapped_drift_decomposition": (
            "e=e_res+(a-A)y+ell(eta*n-y)-Omega*y"
        ),
        "stretching_decomposition": (
            "delta_s=2(a-A)+delta_s_res"
        ),
        "geometry_effective_potential": str(geometry_potential),
        "expected_geometry_effective_potential": str(expected_geometry),
        "amplitude_effective_potential": str(amplitude_potential),
        "rotation_effective_potential": str(rotation_potential),
        "residual_effective_potential": str(residual_potential),
        "geometry_bracket": str(geometry_bracket),
        "completed_geometry_bracket": str(completed_bracket),
        "working_geometry_bracket": str(working_bracket),
        "working_completed_geometry_bracket": str(
            working_completed_bracket
        ),
        "working_geometry_bracket_minimum": 0.75,
        "geometry_identity_verified": bool(
            sp.simplify(geometry_potential - expected_geometry) == 0
        ),
        "geometry_square_completion_verified": bool(
            sp.simplify(geometry_bracket - completed_bracket) == 0
        ),
        "amplitude_identity_verified": bool(
            sp.simplify(
                amplitude_potential
                - amplitude_deficit
                * (1 - reynolds * (y_1**2 + y_2**2) / 2)
            )
            == 0
        ),
        "amplitude_deficit_is_nonpositive_for_A_ge_a_R_le_2": True,
        "pure_rotation_cancels_in_radial_gauge": bool(
            rotation_potential == 0
        ),
        "complete_residual_decomposition_verified": bool(
            sp.simplify(total_potential - decomposed_potential) == 0
        ),
    }


def _positive_root(
    return_gain: float,
    wall_gain: float,
    return_response: float,
    wall_response: float,
) -> float:
    quadratic = return_response**2 + wall_response**2
    linear = (
        return_gain * return_response + wall_gain * wall_response
    )
    constant = return_gain**2 + wall_gain**2 - 1.0
    if quadratic == 0.0:
        return math.inf if constant < 0.0 else 0.0
    discriminant = linear**2 - quadratic * constant
    return (-linear + math.sqrt(max(0.0, discriminant))) / quadratic


def _numerical_budget() -> dict[str, object]:
    migration = _load_module(
        "geometry_triggered_migrating_child_pilot.py",
        "migration_for_residual_budget",
    )
    resolvent = _load_module(
        "neutral_strip_branch_resolvent_pilot.py",
        "resolvent_for_residual_budget",
    )
    axial = _load_module(
        "neutral_strip_axial_patch_branch_pilot.py",
        "axial_for_residual_budget",
    )
    wall_row = migration._full_wall_transition_row(
        resolvent,
        axial,
        WORKING_HALF_WIDTH,
        target_spacing=TARGET_SPACING,
    )

    conversion = math.exp(R_STAR / 4.0)
    tracking = 2.0 ** (-0.75)
    rows = []
    for angle_row in wall_row["angle_rows"]:
        return_gain = (
            resolvent.H1_ENTRY_GAIN
            * angle_row["axial_patch_return_gain"]
        )
        wall_gain = (
            resolvent.H1_ENTRY_GAIN
            * conversion
            * tracking
            * angle_row["unweighted_wall_gain"]
        )
        criterion = return_gain**2 + wall_gain**2
        common_additive = _positive_root(
            return_gain, wall_gain, 1.0, 1.0
        )
        rows.append(
            {
                "angle": angle_row["angle"],
                "return_one_history_gain": return_gain,
                "wall_one_history_gain": wall_gain,
                "pair_criterion": criterion,
                "common_multiplicative_ceiling": 1.0 / math.sqrt(criterion),
                "common_log_action_ceiling": -0.5 * math.log(criterion),
                "common_additive_gain_allowance": common_additive,
                "return_only_additive_gain_allowance": _positive_root(
                    return_gain, wall_gain, 1.0, 0.0
                ),
                "wall_only_additive_gain_allowance": _positive_root(
                    return_gain, wall_gain, 0.0, 1.0
                ),
                "return_only_multiplicative_ceiling": math.sqrt(
                    (1.0 - wall_gain**2) / return_gain**2
                ),
                "wall_only_multiplicative_ceiling": math.sqrt(
                    (1.0 - return_gain**2) / wall_gain**2
                ),
            }
        )

    worst_criterion = max(rows, key=lambda row: row["pair_criterion"])
    common_additive = min(
        rows, key=lambda row: row["common_additive_gain_allowance"]
    )
    return_additive = min(
        rows, key=lambda row: row["return_only_additive_gain_allowance"]
    )
    wall_additive = min(
        rows, key=lambda row: row["wall_only_additive_gain_allowance"]
    )
    return_multiplier = min(
        rows, key=lambda row: row["return_only_multiplicative_ceiling"]
    )
    wall_multiplier = min(
        rows, key=lambda row: row["wall_only_multiplicative_ceiling"]
    )
    allowance = common_additive["common_additive_gain_allowance"]

    return {
        "working_strip_half_width": WORKING_HALF_WIDTH,
        "working_mesh_spacing": wall_row["spacing"],
        "entry_gain": resolvent.H1_ENTRY_GAIN,
        "physical_to_gauge_conversion": conversion,
        "smooth_tracking_factor": tracking,
        "angle_rows": rows,
        "maximum_baseline_pair_criterion": worst_criterion[
            "pair_criterion"
        ],
        "maximum_baseline_pair_criterion_angle": worst_criterion["angle"],
        "source_wall_criterion": wall_row[
            "maximum_smooth_tracking_wall_with_conversion_criterion"
        ],
        "source_criterion_recovered": bool(
            abs(
                worst_criterion["pair_criterion"]
                - wall_row[
                    "maximum_smooth_tracking_wall_with_conversion_criterion"
                ]
            )
            < 2.0e-13
        ),
        "common_one_history_multiplicative_ceiling": worst_criterion[
            "common_multiplicative_ceiling"
        ],
        "common_one_history_integrated_log_action_ceiling": worst_criterion[
            "common_log_action_ceiling"
        ],
        "pair_integrated_log_margin": -math.log(
            worst_criterion["pair_criterion"]
        ),
        "minimum_common_additive_gain_allowance": allowance,
        "minimum_common_additive_gain_allowance_angle": common_additive[
            "angle"
        ],
        "minimum_return_only_additive_gain_allowance": return_additive[
            "return_only_additive_gain_allowance"
        ],
        "minimum_wall_only_additive_gain_allowance": wall_additive[
            "wall_only_additive_gain_allowance"
        ],
        "minimum_return_only_multiplicative_ceiling": return_multiplier[
            "return_only_multiplicative_ceiling"
        ],
        "minimum_wall_only_multiplicative_ceiling": wall_multiplier[
            "wall_only_multiplicative_ceiling"
        ],
        "conditional_boundary_response_gate": (
            "(a_R+K_R*F)^2+(a_S+K_S*F)^2<1"
        ),
        "conditional_sharp_F_ceiling": (
            "min_theta positive_root[(a_R+K_R F)^2+"
            "(a_S+K_S F)^2=1]"
        ),
        "H1_global_energy_forcing_coefficients": {
            "potential_L3_over_2": POTENTIAL_L3_OVER_2_FORCING,
            "drift_L3": DRIFT_L3_FORCING,
            "potential_relative_form": POTENTIAL_RELATIVE_FORM,
        },
        "unit_common_response_combined_F_ceiling": allowance,
        "unit_common_response_pure_potential_L3_over_2_ceiling": (
            allowance / POTENTIAL_L3_OVER_2_FORCING
        ),
        "unit_common_response_pure_drift_L3_ceiling": (
            allowance / DRIFT_L3_FORCING
        ),
        "unit_potential_relative_form_alpha_at_pure_ceiling": (
            POTENTIAL_RELATIVE_FORM
            * allowance
            / POTENTIAL_L3_OVER_2_FORCING
        ),
    }


def audit() -> dict[str, object]:
    result = {
        **_symbolic_audit(),
        **_numerical_budget(),
        "common_log_action_gate": (
            "if each branch gain is multiplied by exp(E), require "
            "E<-.5*log(C_base)"
        ),
        "branch_resolved_log_action_gate": (
            "a_R^2 exp(2E_R)+a_S^2 exp(2E_S)<1"
        ),
        "comoving_tracking_center_law": (
            "c'=v_c-L*ell*eta*O*n"
        ),
        "constant_Reynolds_reference_law": (
            "A*L^2/nu=R_star, so R_star'=0"
        ),
        "boundary_response_constants_certified": False,
        "actual_Navier_Stokes_residual_norms_certified": False,
        "adapted_moving_core_PDE_localization_certified": False,
        "full_Navier_Stokes_migrating_residual_gate_closed": False,
        "scope_guard": (
            "the decomposition and scalar allowances are exact or "
            "reproducible. The unit-response L3/2 and L3 numbers are "
            "calibrations only: a theorem must still derive the branch "
            "response constants K_R,K_S and bound the actual swept-core "
            "residual without assuming coherence"
        ),
        "next_gate": (
            "derive K_R,K_S from the existing space-time trace theorem for "
            "the wall-stopping law, then construct a mollified adapted "
            "center/frame whose q_res and e_res obey the resulting bound"
        ),
    }
    checks = (
        result["geometry_identity_verified"],
        result["geometry_square_completion_verified"],
        result["amplitude_identity_verified"],
        result["amplitude_deficit_is_nonpositive_for_A_ge_a_R_le_2"],
        result["pure_rotation_cancels_in_radial_gauge"],
        result["complete_residual_decomposition_verified"],
        result["source_criterion_recovered"],
        result["maximum_baseline_pair_criterion"] < 0.68,
        result["common_one_history_integrated_log_action_ceiling"] > 0.19,
        result["minimum_common_additive_gain_allowance"] > 0.0,
        result["unit_common_response_pure_potential_L3_over_2_ceiling"] > 0.0,
        result["unit_common_response_pure_drift_L3_ceiling"] > 0.0,
        not result["boundary_response_constants_certified"],
        not result["actual_Navier_Stokes_residual_norms_certified"],
        not result["adapted_moving_core_PDE_localization_certified"],
        not result["full_Navier_Stokes_migrating_residual_gate_closed"],
    )
    result["all_positive_migrating_residual_checks_pass"] = all(checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
