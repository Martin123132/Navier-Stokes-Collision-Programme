"""Audit the IMS cost of quadratic dyadic localization."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np


def _load_script(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _partition_grid_audit(grid_points: int = 41) -> dict[str, float | bool]:
    coordinates = np.linspace(0.0, 1.0, grid_points)
    theta = 0.5 * math.pi * coordinates
    factors = (np.cos(theta), np.sin(theta))
    factor_derivatives = (
        -0.5 * math.pi * np.sin(theta),
        0.5 * math.pi * np.cos(theta),
    )
    partition_sum = np.zeros((grid_points,) * 3)
    gradient_square_sum = np.zeros((grid_points,) * 3)
    for bits in itertools.product((0, 1), repeat=3):
        values = np.ones((grid_points,) * 3)
        gradient_components = []
        for direction in range(3):
            reshape = [1, 1, 1]
            reshape[direction] = grid_points
            values *= factors[bits[direction]].reshape(reshape)
        for direction in range(3):
            component = np.ones((grid_points,) * 3)
            for factor_direction in range(3):
                reshape = [1, 1, 1]
                reshape[factor_direction] = grid_points
                selected = (
                    factor_derivatives[bits[factor_direction]]
                    if factor_direction == direction
                    else factors[bits[factor_direction]]
                )
                component *= selected.reshape(reshape)
            gradient_components.append(component)
        partition_sum += values**2
        gradient_square_sum += sum(
            component**2 for component in gradient_components
        )
    expected_ims_density = 3.0 * math.pi**2 / 4.0
    return {
        "maximum_quadratic_partition_error": float(
            np.max(np.abs(partition_sum - 1.0))
        ),
        "maximum_IMS_density_error": float(
            np.max(np.abs(gradient_square_sum - expected_ims_density))
        ),
        "quadratic_partition_identity_verified": bool(
            np.max(np.abs(partition_sum - 1.0)) < 1.0e-14
        ),
        "tensor_IMS_identity_verified": bool(
            np.max(
                np.abs(gradient_square_sum - expected_ims_density)
            )
            < 1.0e-13
        ),
    }


def _budget_row(
    active_directions: int,
    transition_width_over_L: float,
    dimensionless_margin: float,
    reynolds: float,
    sobolev_constant: float,
    poisson_alpha_budget: float,
) -> dict[str, float | int | bool]:
    ims_cost = (
        active_directions
        * math.pi**2
        / (4.0 * transition_width_over_L**2)
    )
    remaining_margin = dimensionless_margin - ims_cost
    if remaining_margin <= 0.0:
        unit_form_mass_budget = 0.0
        final_mass_budget = 0.0
    else:
        unit_form_mass_budget = remaining_margin / (
            sobolev_constant * (remaining_margin + 2.0 * reynolds)
        )
        final_mass_budget = poisson_alpha_budget * unit_form_mass_budget
    return {
        "simultaneously_active_split_directions": active_directions,
        "transition_width_over_L": transition_width_over_L,
        "dimensionless_IMS_cost": ims_cost,
        "dimensionless_spectral_margin_before_IMS": dimensionless_margin,
        "dimensionless_spectral_margin_after_IMS": remaining_margin,
        "IMS_cost_is_absorbable": bool(remaining_margin > 0.0),
        "unit_relative_form_L3_over_2_budget_after_IMS": (
            unit_form_mass_budget
        ),
        "Poisson_compatible_L3_over_2_budget_over_nu": final_mass_budget,
    }


def audit() -> dict[str, object]:
    localized = _load_script(
        "localized_strain_tube_audit.py",
        "localized_strain_for_ims",
    ).audit()
    poisson = _load_script(
        "poisson_cutoff_form_transfer_audit.py",
        "poisson_cutoff_for_ims",
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
    poisson_profile = next(
        row
        for row in poisson["profile_rows"]
        if row["perturbation_support_radius"] == 1.5
        and row["cutoff_taper_radius"] == 2.0
    )
    poisson_alpha_budget = poisson_profile[
        "allowable_relative_form_alpha"
    ]

    transition_widths = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    budget_rows = [
        _budget_row(
            active_directions,
            transition_width,
            dimensionless_margin,
            reynolds,
            sobolev_constant,
            poisson_alpha_budget,
        )
        for active_directions in (1, 2, 3)
        for transition_width in transition_widths
    ]
    minimum_width_rows = [
        {
            "simultaneously_active_split_directions": active_directions,
            "strict_minimum_transition_width_over_L": (
                0.5
                * math.pi
                * math.sqrt(active_directions / dimensionless_margin)
            ),
        }
        for active_directions in (1, 2, 3)
    ]
    unit_width_rows = [
        row
        for row in budget_rows
        if row["transition_width_over_L"] == 1.0
    ]
    tensor_wide_row = next(
        row
        for row in budget_rows
        if row["simultaneously_active_split_directions"] == 3
        and row["transition_width_over_L"] == 1.5
    )
    sequential_unit_row = next(
        row
        for row in budget_rows
        if row["simultaneously_active_split_directions"] == 1
        and row["transition_width_over_L"] == 1.0
    )

    grid_audit = _partition_grid_audit()
    result: dict[str, object] = {
        "quadratic_partition": (
            "chi_bits=product_j cos(theta_j) or sin(theta_j), "
            "sum_bits chi_bits^2=1"
        ),
        "weighted_IMS_identity": (
            "sum_j a_0[chi_j u]=a_0[u]+"
            "integral w*sum_j|grad chi_j|^2*|u|^2"
        ),
        "tensor_partition_IMS_density": (
            "d*pi^2/(4*omega^2*L^2) for d active directions "
            "and transition width omega*L"
        ),
        **grid_audit,
        "R_star": reynolds,
        "dimensionless_transverse_form_margin": dimensionless_margin,
        "Poisson_relative_form_alpha_budget": poisson_alpha_budget,
        "minimum_transition_width_rows": minimum_width_rows,
        "budget_rows": budget_rows,
        "unit_width_rows": unit_width_rows,
        "unit_width_one_direction_is_absorbable": bool(
            unit_width_rows[0]["IMS_cost_is_absorbable"]
        ),
        "unit_width_two_directions_are_absorbable": bool(
            unit_width_rows[1]["IMS_cost_is_absorbable"]
        ),
        "unit_width_three_direction_octree_is_not_absorbable": bool(
            not unit_width_rows[2]["IMS_cost_is_absorbable"]
        ),
        "wide_tensor_octree_row": tensor_wide_row,
        "width_1p5_tensor_octree_is_absorbable": bool(
            tensor_wide_row["IMS_cost_is_absorbable"]
        ),
        "sequential_unit_split_row": sequential_unit_row,
        "single_direction_stage_has_larger_nominal_mass_budget": bool(
            sequential_unit_row[
                "Poisson_compatible_L3_over_2_budget_over_nu"
            ]
            > tensor_wide_row[
                "Poisson_compatible_L3_over_2_budget_over_nu"
            ]
        ),
        "same_time_sequential_splits_do_not_reset_IMS_margin": True,
        "sequential_staging_requires_new_hierarchical_states_or_visits": True,
        "pressure_compatibility": (
            "phi_j=chi_j^2 is a linear partition of unity, so the exact "
            "neighbor pressure-flux cancellation remains available"
        ),
        "interpretation": (
            "a simultaneous unit-width octree localization spends more "
            "than the conservative spectral margin; a wider simultaneous "
            "overlap is currently justified, while nominally sequential "
            "splits save IMS cost only if parent-buffer states or complete "
            "intermediate visits genuinely prevent the three costs from "
            "being summed"
        ),
        "remaining_cover_gate": (
            "construct a time-coherent width-at-least-1.5L quadratic cover, "
            "or separately prove a hierarchical parent-buffer partition "
            "whose split stages do not recombine into the tensor IMS cost"
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
