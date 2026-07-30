"""Audit a compact cubic partition fitted to the radial Poisson collar."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp


def _load_script(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cardinal_cubic(argument: np.ndarray) -> np.ndarray:
    value = np.zeros_like(argument, dtype=float)
    first = (argument >= 0.0) & (argument < 1.0)
    second = (argument >= 1.0) & (argument < 2.0)
    third = (argument >= 2.0) & (argument < 3.0)
    fourth = (argument >= 3.0) & (argument <= 4.0)
    value[first] = argument[first] ** 3 / 6.0
    local = argument[second] - 1.0
    value[second] = (
        -3.0 * local**3 + 3.0 * local**2 + 3.0 * local + 1.0
    ) / 6.0
    local = 3.0 - argument[third]
    value[third] = (
        -3.0 * local**3 + 3.0 * local**2 + 3.0 * local + 1.0
    ) / 6.0
    value[fourth] = (4.0 - argument[fourth]) ** 3 / 6.0
    return value


def _cardinal_cubic_derivative(argument: np.ndarray) -> np.ndarray:
    value = np.zeros_like(argument, dtype=float)
    first = (argument >= 0.0) & (argument < 1.0)
    second = (argument >= 1.0) & (argument < 2.0)
    third = (argument >= 2.0) & (argument < 3.0)
    fourth = (argument >= 3.0) & (argument <= 4.0)
    value[first] = argument[first] ** 2 / 2.0
    local = argument[second] - 1.0
    value[second] = (-3.0 * local**2 + 2.0 * local + 1.0) / 2.0
    local = 3.0 - argument[third]
    value[third] = -(
        -3.0 * local**2 + 2.0 * local + 1.0
    ) / 2.0
    value[fourth] = -(4.0 - argument[fourth]) ** 2 / 2.0
    return value


def _uniform_partition_audit(grid_points: int = 200_001) -> dict[str, object]:
    coordinates = np.linspace(0.0, 1.0, grid_points, endpoint=False)
    shifts = range(-3, 1)
    values = np.array(
        [_cardinal_cubic(coordinates - shift) for shift in shifts]
    )
    derivatives = np.array(
        [
            _cardinal_cubic_derivative(coordinates - shift)
            for shift in shifts
        ]
    )
    fisher_ims = 0.25 * np.sum(
        np.divide(
            derivatives**2,
            values,
            out=np.zeros_like(values),
            where=values > 1.0e-15,
        ),
        axis=0,
    )

    x = sp.symbols("x", real=True)
    pieces = (
        (1 - x) ** 3 / 6,
        (3 * x**3 - 6 * x**2 + 4) / 6,
        (-3 * x**3 + 3 * x**2 + 3 * x + 1) / 6,
        x**3 / 6,
    )
    exact_fisher = sp.factor(
        sum(sp.diff(piece, x) ** 2 / piece for piece in pieces) / 4
    )
    rational_upper = sp.Rational(157, 200)
    difference = sp.factor(rational_upper - exact_fisher)
    numerator, denominator = sp.together(difference).as_numer_denom()
    numerator_roots = sp.Poly(numerator, x).count_roots(0, 1)
    denominator_roots = sp.Poly(denominator, x).count_roots(0, 1)
    midpoint_difference = difference.subs(x, sp.Rational(1, 2))

    refinement_mask = np.array(
        [math.comb(4, index) / 8.0 for index in range(5)]
    )
    refinement_coordinates = np.linspace(-1.0, 5.0, grid_points)
    coarse = _cardinal_cubic(refinement_coordinates)
    refined = sum(
        coefficient
        * _cardinal_cubic(2.0 * refinement_coordinates - index)
        for index, coefficient in enumerate(refinement_mask)
    )

    return {
        "maximum_partition_sum_error": float(
            np.max(np.abs(np.sum(values, axis=0) - 1.0))
        ),
        "sampled_maximum_one_dimensional_IMS_at_unit_knot_spacing": (
            float(np.max(fisher_ims))
        ),
        "exact_one_cell_Fisher_expression": str(exact_fisher),
        "rational_one_dimensional_IMS_upper": float(rational_upper),
        "rational_upper_numerator_has_no_roots_on_unit_cell": bool(
            numerator_roots == 0
        ),
        "rational_upper_denominator_has_no_roots_on_unit_cell": bool(
            denominator_roots == 0
        ),
        "rational_upper_has_positive_midpoint_difference": bool(
            midpoint_difference > 0
        ),
        "rational_IMS_upper_verified": bool(
            numerator_roots == 0
            and denominator_roots == 0
            and midpoint_difference > 0
        ),
        "cubic_refinement_mask": refinement_mask.tolist(),
        "maximum_two_scale_refinement_identity_error": float(
            np.max(np.abs(coarse - refined))
        ),
        "two_scale_refinement_identity_verified": bool(
            np.max(np.abs(coarse - refined)) < 2.0e-12
        ),
    }


def _budget_row(
    support_radius: float,
    poisson_alpha: float,
    poisson_condition_number: float,
    dimensionless_margin: float,
    sobolev_constant: float,
) -> dict[str, float | bool]:
    # A cubic spline has support width 4h. Its transverse support square is
    # inside rho<=rho_s L when 2 sqrt(2) h=rho_s L.
    knot_spacing_over_L = support_radius / (2.0 * math.sqrt(2.0))
    ims_cost = (
        2.0
        * (157.0 / 200.0)
        / knot_spacing_over_L**2
    )
    remaining_margin = dimensionless_margin - ims_cost
    if remaining_margin > 0.0:
        unit_form_mass_budget = remaining_margin / (
            sobolev_constant * (remaining_margin + 1.0)
        )
        final_mass_budget = poisson_alpha * unit_form_mass_budget
    else:
        unit_form_mass_budget = 0.0
        final_mass_budget = 0.0
    axial_knot_spacing_over_L = 0.75
    axial_ims_cost = (157.0 / 200.0) / axial_knot_spacing_over_L**2
    full_ims_cost = ims_cost + axial_ims_cost
    full_remaining_margin = dimensionless_margin - full_ims_cost
    if full_remaining_margin > 0.0:
        full_unit_form_mass_budget = full_remaining_margin / (
            sobolev_constant * (full_remaining_margin + 1.0)
        )
        full_final_mass_budget = (
            poisson_alpha * full_unit_form_mass_budget
        )
    else:
        full_unit_form_mass_budget = 0.0
        full_final_mass_budget = 0.0
    return {
        "radial_support_radius_over_L": support_radius,
        "cubic_knot_spacing_over_L": knot_spacing_over_L,
        "support_square_circumradius_over_L": (
            2.0 * math.sqrt(2.0) * knot_spacing_over_L
        ),
        "Poisson_condition_number": poisson_condition_number,
        "Poisson_relative_form_alpha_budget": poisson_alpha,
        "dimensionless_two_direction_IMS_cost": ims_cost,
        "dimensionless_spectral_margin_after_IMS": remaining_margin,
        "IMS_cost_is_absorbable": bool(remaining_margin > 0.0),
        "unit_relative_form_L3_over_2_budget_after_IMS": (
            unit_form_mass_budget
        ),
        "Poisson_compatible_L3_over_2_budget_over_nu": final_mass_budget,
        "axial_cubic_knot_spacing_over_L": axial_knot_spacing_over_L,
        "dimensionless_axial_IMS_cost": axial_ims_cost,
        "dimensionless_full_three_direction_IMS_cost": full_ims_cost,
        "dimensionless_full_spectral_margin_after_IMS": (
            full_remaining_margin
        ),
        "full_three_direction_IMS_cost_is_absorbable": bool(
            full_remaining_margin > 0.0
        ),
        "full_unit_relative_form_L3_over_2_budget_after_IMS": (
            full_unit_form_mass_budget
        ),
        "full_Poisson_compatible_L3_over_2_budget_over_nu": (
            full_final_mass_budget
        ),
    }


def audit() -> dict[str, object]:
    uniform = _uniform_partition_audit()
    poisson = _load_script(
        "poisson_cutoff_form_transfer_audit.py",
        "poisson_cutoff_for_radial_cubic_partition",
    ).audit()
    localized = _load_script(
        "localized_strain_tube_audit.py",
        "localized_strain_for_radial_cubic_partition",
    ).audit()

    reynolds = 0.5
    localized_row = next(
        row
        for row in localized["spectral_rows"]
        if row["tube_reynolds"] == reynolds
    )
    dimensionless_margin = (
        localized_row["single_history_perturbation_budget_over_a"]
        * reynolds
    )
    sobolev_constant = 4.0 ** (2.0 / 3.0) / (
        3.0 * math.pi ** (4.0 / 3.0)
    )
    candidate_profiles = [
        row
        for row in poisson["profile_rows"]
        if row["cutoff_taper_radius"] == 2.0
        and row["perturbation_support_radius"] >= 1.5
    ]
    budget_rows = [
        _budget_row(
            float(row["perturbation_support_radius"]),
            float(row["allowable_relative_form_alpha"]),
            float(row["Poisson_cutoff_condition_number"]),
            dimensionless_margin,
            sobolev_constant,
        )
        for row in candidate_profiles
    ]
    optimized_transverse_row = max(
        budget_rows,
        key=lambda row: row[
            "Poisson_compatible_L3_over_2_budget_over_nu"
        ],
    )
    optimized_full_row = max(
        budget_rows,
        key=lambda row: row[
            "full_Poisson_compatible_L3_over_2_budget_over_nu"
        ],
    )
    row_1p5 = next(
        row
        for row in budget_rows
        if row["radial_support_radius_over_L"] == 1.5
    )

    tensor_points = np.linspace(0.0, 1.0, 31, endpoint=False)
    one_dimensional_values = np.array(
        [
            _cardinal_cubic(tensor_points - shift)
            for shift in range(-3, 1)
        ]
    )
    tensor_sum = np.zeros((31, 31))
    for first, second in itertools.product(range(4), repeat=2):
        tensor_sum += np.outer(
            one_dimensional_values[first],
            one_dimensional_values[second],
        )
    maximum_tensor_partition_error = float(
        np.max(np.abs(tensor_sum - 1.0))
    )

    result: dict[str, object] = {
        "partition_weights": (
            "phi_jk=N_3(x/h-j)N_3(y/h-k), chi_jk=sqrt(phi_jk)"
        ),
        "radial_support_geometry": (
            "each support is a square of half-width 2h and is contained "
            "in rho<=rho_s L when 2*sqrt(2)*h=rho_s L"
        ),
        "fixed_cylinder_axial_scope": (
            "the transverse result needs no axial cutoff inside one finite "
            "cylinder, but a global tensor partition pays the separate "
            "axial cubic IMS cost"
        ),
        **uniform,
        "maximum_two_dimensional_partition_sum_error": (
            maximum_tensor_partition_error
        ),
        "two_dimensional_partition_identity_verified": bool(
            maximum_tensor_partition_error < 2.0e-13
        ),
        "maximum_pointwise_overlap_count": 16,
        "R_star": reynolds,
        "dimensionless_transverse_form_margin": dimensionless_margin,
        "candidate_budget_rows": budget_rows,
        "optimized_budget_row": optimized_transverse_row,
        "optimized_transverse_budget_row": optimized_transverse_row,
        "optimized_full_tensor_budget_row": optimized_full_row,
        "radius_1p5_tensor_square_fails_IMS_gate": bool(
            not row_1p5["IMS_cost_is_absorbable"]
        ),
        "optimized_radius_is_1p75": bool(
            optimized_transverse_row["radial_support_radius_over_L"] == 1.75
        ),
        "optimized_transverse_partition_has_positive_mass_budget": bool(
            optimized_transverse_row[
                "Poisson_compatible_L3_over_2_budget_over_nu"
            ]
            > 0.59
        ),
        "optimized_full_tensor_radius_is_1p91": bool(
            optimized_full_row["radial_support_radius_over_L"] == 1.91
        ),
        "optimized_full_tensor_has_positive_mass_budget": bool(
            optimized_full_row[
                "full_Poisson_compatible_L3_over_2_budget_over_nu"
            ]
            > 0.21
        ),
        "pressure_compatibility": (
            "sum_jk phi_jk=1, so the linear pressure partition and exact "
            "neighbor-flux cancellation are retained"
        ),
        "dyadic_refinement_identity": (
            "N_3(x)=sum_(r=0)^4 binom(4,r)/8*N_3(2x-r)"
        ),
        "static_cover_status": (
            "the compact fixed-frame three-dimensional tensor partition, "
            "IMS absorption, pressure identity, and optimized Poisson "
            "budget are closed"
        ),
        "orientation_scope_warning": (
            "a spatially varying cylinder axis is not supplied for free; "
            "frame variation remains part of the non-affine perturbation"
        ),
        "remaining_time_coherence_gate": (
            "turn the positive cubic two-scale identity into a common-norm "
            "parent/child visit transfer at monotone level changes, without "
            "paying a fixed gauge or Poisson conversion per generation"
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
