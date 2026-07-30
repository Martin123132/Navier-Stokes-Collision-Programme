"""Estimate the internal Green-block condition number of the weighted cylinder."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded


def _load_script(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _radial_form_matrix(
    reynolds: float,
    axial_eigenvalue: float,
    buffer_ratio: float,
    radial_intervals: int,
    angular_mode: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    spacing = buffer_ratio / radial_intervals
    diagonal = np.zeros(radial_intervals)
    off_diagonal = np.zeros(radial_intervals - 1)
    gauss = 1.0 / math.sqrt(3.0)

    for element in range(radial_intervals):
        left = element * spacing
        right = (element + 1) * spacing
        weight_integral = 0.0
        potential_mass = np.zeros((2, 2))
        for gauss_coordinate in (-gauss, gauss):
            radius = (
                0.5 * (left + right)
                + 0.5 * spacing * gauss_coordinate
            )
            quadrature_weight = 0.5 * spacing
            radial_weight = radius * math.exp(
                0.5
                * reynolds
                * min(radius**2, 1.0)
            )
            potential = radial_weight * (
                axial_eigenvalue
                - (2.0 * reynolds if radius < 1.0 else 0.0)
                + angular_mode**2 / radius**2
            )
            shape = np.array(
                [(right - radius) / spacing, (radius - left) / spacing]
            )
            weight_integral += quadrature_weight * radial_weight
            potential_mass += (
                quadrature_weight * potential * np.outer(shape, shape)
            )

        local_matrix = (
            weight_integral
            / spacing**2
            * np.array([[1.0, -1.0], [-1.0, 1.0]])
            + potential_mass
        )
        diagonal[element] += local_matrix[0, 0]
        if element + 1 < radial_intervals:
            diagonal[element + 1] += local_matrix[1, 1]
            off_diagonal[element] += local_matrix[0, 1]

    if angular_mode > 0:
        # Regular m-not-zero modes vanish on the symmetry axis.
        return diagonal[1:], off_diagonal[1:]
    return diagonal, off_diagonal


def _mode_green_blocks(
    reynolds: float,
    axial_eigenvalues: np.ndarray,
    outer_surfaces: tuple[float, ...],
    buffer_ratio: float,
    radial_intervals: int,
    inner_surface: float = 1.0,
    angular_mode: int = 0,
) -> dict[float, list[tuple[float, float, float]]]:
    spacing = buffer_ratio / radial_intervals
    surface_radii = (inner_surface,) + outer_surfaces
    surface_indices = [int(round(radius / spacing)) for radius in surface_radii]
    if any(
        abs(index * spacing - radius) > 1.0e-12
        for index, radius in zip(surface_indices, surface_radii)
    ):
        raise ValueError("all trace surfaces must lie on radial grid nodes")

    if angular_mode > 0:
        surface_indices = [index - 1 for index in surface_indices]

    result = {radius: [] for radius in outer_surfaces}
    for axial_eigenvalue in axial_eigenvalues:
        diagonal, off_diagonal = _radial_form_matrix(
            reynolds,
            float(axial_eigenvalue),
            buffer_ratio,
            radial_intervals,
            angular_mode,
        )
        unknown_count = len(diagonal)
        banded = np.zeros((3, unknown_count))
        banded[0, 1:] = off_diagonal
        banded[1, :] = diagonal
        banded[2, :-1] = off_diagonal
        right_hand_sides = np.zeros(
            (unknown_count, len(surface_indices))
        )
        for column, index in enumerate(surface_indices):
            right_hand_sides[index, column] = 1.0
        green_columns = solve_banded((1, 1), banded, right_hand_sides)
        inner_index = surface_indices[0]
        inner_diagonal = float(green_columns[inner_index, 0])
        for column, outer_surface in enumerate(outer_surfaces, start=1):
            outer_index = surface_indices[column]
            result[outer_surface].append(
                (
                    inner_diagonal,
                    float(green_columns[outer_index, column]),
                    float(green_columns[inner_index, column]),
                )
            )
    return result


def _angular_mode_stress_rows(
    perturbation,
    reynolds: float,
    half_height: float,
    radial_intervals: int = 400,
    buffer_ratio: float = 2.0,
) -> list[dict[str, float | int | bool]]:
    axial_basis = perturbation._axial_basis(
        reynolds,
        half_height,
        grid_points=401,
        mode_count=8,
    )
    axial_eigenvalues = np.asarray(axial_basis["eigenvalues"])
    outer_surface = 1.5
    rows = []
    for angular_mode in range(7):
        blocks = _mode_green_blocks(
            reynolds,
            axial_eigenvalues,
            (outer_surface,),
            buffer_ratio,
            radial_intervals,
            angular_mode=angular_mode,
        )[outer_surface]
        inner_values = np.array([block[0] for block in blocks])
        outer_values = np.array([block[1] for block in blocks])
        cross_values = np.array([block[2] for block in blocks])
        rows.append(
            {
                "angular_mode_absolute_value": angular_mode,
                "maximizing_axial_mode_for_inner_diagonal": int(
                    np.argmax(inner_values)
                ),
                "maximizing_axial_mode_for_outer_diagonal": int(
                    np.argmax(outer_values)
                ),
                "maximizing_axial_mode_for_cross_block": int(
                    np.argmax(cross_values)
                ),
                "inner_diagonal_norm_at_this_angular_mode": float(
                    np.max(inner_values)
                ),
                "outer_diagonal_norm_at_this_angular_mode": float(
                    np.max(outer_values)
                ),
                "cross_norm_at_this_angular_mode": float(
                    np.max(cross_values)
                ),
                "all_sampled_blocks_are_positive": bool(
                    np.all(inner_values > 0.0)
                    and np.all(outer_values > 0.0)
                    and np.all(cross_values > 0.0)
                ),
            }
        )
    return rows


def _allowable_alpha(
    generation_criterion: float, condition_number: float
) -> float:
    excess_multiplier = 1.0 / math.sqrt(generation_criterion) - 1.0
    return excess_multiplier / (condition_number + excess_multiplier)


def _geometry_rows(
    perturbation,
    reynolds: float,
    half_height: float,
    generation_criterion: float,
    radial_intervals: int,
    mode_count: int = 41,
    buffer_ratio: float = 2.0,
) -> list[dict[str, float | bool | int]]:
    axial_basis = perturbation._axial_basis(
        reynolds,
        half_height,
        grid_points=401,
        mode_count=mode_count,
    )
    axial_eigenvalues = np.asarray(axial_basis["eigenvalues"])
    outer_surfaces = (1.1, 1.25, 1.5, 1.75, 1.9)
    blocks = _mode_green_blocks(
        reynolds,
        axial_eigenvalues,
        outer_surfaces,
        buffer_ratio,
        radial_intervals,
    )
    rows = []
    for outer_surface in outer_surfaces:
        mode_blocks = blocks[outer_surface]
        inner_diagonals = np.array([block[0] for block in mode_blocks])
        outer_diagonals = np.array([block[1] for block in mode_blocks])
        cross_values = np.array([block[2] for block in mode_blocks])
        inner_mode = int(np.argmax(inner_diagonals))
        outer_mode = int(np.argmax(outer_diagonals))
        cross_mode = int(np.argmax(cross_values))
        condition_number = math.sqrt(
            float(inner_diagonals[inner_mode])
            * float(outer_diagonals[outer_mode])
        ) / float(cross_values[cross_mode])
        rows.append(
            {
                "outer_internal_surface_radius": outer_surface,
                "unperturbed_outer_collar_width": (
                    buffer_ratio - outer_surface
                ),
                "inner_diagonal_Green_norm": float(
                    inner_diagonals[inner_mode]
                ),
                "outer_diagonal_Green_norm": float(
                    outer_diagonals[outer_mode]
                ),
                "cross_Green_norm": float(cross_values[cross_mode]),
                "diagonal_to_cross_condition_number": condition_number,
                "allowable_relative_form_alpha": _allowable_alpha(
                    generation_criterion, condition_number
                ),
                "inner_norm_maximizing_axial_mode": inner_mode,
                "outer_norm_maximizing_axial_mode": outer_mode,
                "cross_norm_maximizing_axial_mode": cross_mode,
                "all_norms_are_principal_mode": bool(
                    inner_mode == outer_mode == cross_mode == 0
                ),
                "all_retained_cross_Green_values_are_positive": bool(
                    np.all(cross_values > 0.0)
                ),
            }
        )
    return rows


def audit() -> dict[str, object]:
    perturbation = _load_script(
        "finite_cylinder_perturbation_margin_audit.py",
        "finite_cylinder_perturbation_for_buffer_condition",
    )
    boundary_l2 = _load_script(
        "gaussian_boundary_l2_transfer_audit.py",
        "gaussian_boundary_l2_for_buffer_condition",
    ).audit()
    geometry_data = [
        (
            row["R_star"],
            row["half_height_over_L"],
            row["Gaussian_L2_complete_generation_criterion"],
        )
        for row in boundary_l2["visit_rows"]
    ]

    geometry_rows = []
    for reynolds, half_height, criterion in geometry_data:
        geometry_rows.append(
            {
                "R_star": reynolds,
                "half_height_over_L": half_height,
                "baseline_Gaussian_L2_generation_criterion": criterion,
                "surface_rows": _geometry_rows(
                    perturbation,
                    reynolds,
                    half_height,
                    criterion,
                    radial_intervals=400,
                ),
            }
        )

    convergence_rows = []
    working_reynolds, working_height, working_criterion = geometry_data[0]
    for radial_intervals in (200, 400, 800):
        rows = _geometry_rows(
            perturbation,
            working_reynolds,
            working_height,
            working_criterion,
            radial_intervals=radial_intervals,
        )
        convergence_rows.append(
            {
                "radial_intervals": radial_intervals,
                "surface_rows": rows,
            }
        )

    working_limit_rows = convergence_rows[-1]["surface_rows"]
    angular_mode_rows = _angular_mode_stress_rows(
        perturbation,
        working_reynolds,
        working_height,
    )
    condition_sequences = {
        str(surface_row["outer_internal_surface_radius"]): [
            convergence["surface_rows"][index][
                "diagonal_to_cross_condition_number"
            ]
            for convergence in convergence_rows
        ]
        for index, surface_row in enumerate(working_limit_rows)
    }
    result: dict[str, object] = {
        "radial_reversible_weight": (
            "w(rho)=rho*exp[(R_star/2)*min(rho^2,1)]"
        ),
        "radial_mode_operator": (
            "A_zeta=-w^(-1)(w f')'+zeta*f-"
            "2*R_star*1_(rho<1)*f"
        ),
        "radial_boundary_conditions": (
            "regular natural condition at rho=0 and Dirichlet at rho=eta"
        ),
        "internal_trace_surfaces": (
            "rho_i=1 and rho_o<eta; D_i=G(rho_i,rho_i), "
            "D_o=G(rho_o,rho_o), B=G(rho_i,rho_o)"
        ),
        "geometry_rows": geometry_rows,
        "all_retained_norms_are_principal_axial_mode": all(
            row["all_norms_are_principal_mode"]
            for geometry in geometry_rows
            for row in geometry["surface_rows"]
        ),
        "all_retained_cross_Green_values_are_positive": all(
            row["all_retained_cross_Green_values_are_positive"]
            for geometry in geometry_rows
            for row in geometry["surface_rows"]
        ),
        "working_geometry_angular_mode_rows": angular_mode_rows,
        "all_angular_blocks_are_positive": all(
            row["all_sampled_blocks_are_positive"]
            for row in angular_mode_rows
        ),
        "all_angular_block_norms_are_maximized_at_m_zero_n_zero": bool(
            all(
                row["maximizing_axial_mode_for_inner_diagonal"] == 0
                and row["maximizing_axial_mode_for_outer_diagonal"] == 0
                and row["maximizing_axial_mode_for_cross_block"] == 0
                for row in angular_mode_rows
            )
            and angular_mode_rows[0][
                "inner_diagonal_norm_at_this_angular_mode"
            ]
            == max(
                row["inner_diagonal_norm_at_this_angular_mode"]
                for row in angular_mode_rows
            )
            and angular_mode_rows[0][
                "outer_diagonal_norm_at_this_angular_mode"
            ]
            == max(
                row["outer_diagonal_norm_at_this_angular_mode"]
                for row in angular_mode_rows
            )
            and angular_mode_rows[0]["cross_norm_at_this_angular_mode"]
            == max(
                row["cross_norm_at_this_angular_mode"]
                for row in angular_mode_rows
            )
        ),
        "angular_form_order_reason": (
            "mode |m| adds the nonnegative form potential m^2/rho^2, "
            "while higher axial modes add killing; positivity and form "
            "ordering put every baseline block norm in m=0,n=0"
        ),
        "working_geometry_convergence_rows": convergence_rows,
        "working_geometry_condition_sequences": condition_sequences,
        "all_working_condition_numbers_converge": all(
            abs(sequence[-1] - sequence[-2]) < 1.0e-6
            for sequence in condition_sequences.values()
        ),
        "condition_number_increases_toward_Dirichlet_boundary": all(
            later["diagonal_to_cross_condition_number"]
            > earlier["diagonal_to_cross_condition_number"]
            for earlier, later in zip(
                working_limit_rows[:-1], working_limit_rows[1:]
            )
        ),
        "working_geometry_half_radius_surface": next(
            row
            for row in working_limit_rows
            if row["outer_internal_surface_radius"] == 1.5
        ),
        "working_geometry_quarter_collar_surface": next(
            row
            for row in working_limit_rows
            if row["outer_internal_surface_radius"] == 1.75
        ),
        "interpretation": (
            "the weighted internal Green-block condition is finite and "
            "moderate with a positive outer collar, but becomes singular "
            "as the trace surface approaches the Dirichlet payoff boundary"
        ),
        "scope": (
            "the baseline full-cylinder block norms are controlled by the "
            "axisymmetric principal mode, so arbitrary interior mode "
            "coupling is covered by the abstract form bound; perturbation "
            "in the outer collar remains a separate gate"
        ),
        "next_full_3d_gate": (
            "factor the outer Dirichlet-to-internal collar map and assign "
            "its non-affine error to a separate localized estimate"
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
