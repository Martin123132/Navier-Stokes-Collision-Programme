"""Audit a cutoff-energy form bound for the full Poisson visit map."""

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


def _full_radial_form_matrix(
    reynolds: float,
    axial_eigenvalue: float,
    angular_mode: int,
    buffer_ratio: float,
    radial_intervals: int,
) -> tuple[np.ndarray, np.ndarray]:
    spacing = buffer_ratio / radial_intervals
    diagonal = np.zeros(radial_intervals + 1)
    off_diagonal = np.zeros(radial_intervals)
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
                0.5 * reynolds * min(radius**2, 1.0)
            )
            potential = radial_weight * (
                axial_eigenvalue
                + angular_mode**2 / radius**2
                - (2.0 * reynolds if radius < 1.0 else 0.0)
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
        diagonal[element + 1] += local_matrix[1, 1]
        off_diagonal[element] += local_matrix[0, 1]
    return diagonal, off_diagonal


def _poisson_solution_and_form(
    reynolds: float,
    axial_eigenvalue: float,
    angular_mode: int,
    buffer_ratio: float,
    radial_intervals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    full_diagonal, full_off_diagonal = _full_radial_form_matrix(
        reynolds,
        axial_eigenvalue,
        angular_mode,
        buffer_ratio,
        radial_intervals,
    )
    spacing = buffer_ratio / radial_intervals
    if angular_mode == 0:
        radii = spacing * np.arange(radial_intervals)
        diagonal = full_diagonal[:-1]
        off_diagonal = full_off_diagonal[:-1]
    else:
        radii = spacing * np.arange(1, radial_intervals)
        diagonal = full_diagonal[1:-1]
        off_diagonal = full_off_diagonal[1:-1]

    right_hand_side = np.zeros(len(diagonal))
    right_hand_side[-1] = -full_off_diagonal[-1]
    banded = np.zeros((3, len(diagonal)))
    banded[0, 1:] = off_diagonal
    banded[1, :] = diagonal
    banded[2, :-1] = off_diagonal
    solution = solve_banded((1, 1), banded, right_hand_side)
    return radii, diagonal, off_diagonal, solution


def _cutoff_values(
    radii: np.ndarray, support_radius: float, taper_radius: float
) -> np.ndarray:
    values = np.ones_like(radii)
    values[radii >= taper_radius] = 0.0
    transition = (radii > support_radius) & (radii < taper_radius)
    values[transition] = (
        taper_radius - radii[transition]
    ) / (taper_radius - support_radius)
    return values


def _form_energy(
    diagonal: np.ndarray, off_diagonal: np.ndarray, vector: np.ndarray
) -> float:
    return float(
        np.dot(diagonal, vector**2)
        + 2.0 * np.dot(off_diagonal, vector[:-1] * vector[1:])
    )


def _mode_energy_rows(
    perturbation,
    reynolds: float,
    half_height: float,
    cutoff_profiles: tuple[tuple[float, float], ...],
    radial_intervals: int,
    axial_mode_count: int,
    maximum_angular_mode: int,
    buffer_ratio: float = 2.0,
) -> tuple[dict[tuple[float, float], dict[str, float | int]], dict[str, float]]:
    axial_basis = perturbation._axial_basis(
        reynolds,
        half_height,
        grid_points=max(401, 5 * axial_mode_count),
        mode_count=axial_mode_count,
    )
    axial_eigenvalues = np.asarray(axial_basis["eigenvalues"])
    maxima = {
        profile: {
            "cutoff_Poisson_energy": -math.inf,
            "maximizing_angular_mode": -1,
            "maximizing_axial_mode": -1,
        }
        for profile in cutoff_profiles
    }
    principal_data: dict[str, float] = {}

    for angular_mode in range(maximum_angular_mode + 1):
        for axial_mode, axial_eigenvalue in enumerate(axial_eigenvalues):
            radii, diagonal, off_diagonal, poisson_solution = (
                _poisson_solution_and_form(
                    reynolds,
                    float(axial_eigenvalue),
                    angular_mode,
                    buffer_ratio,
                    radial_intervals,
                )
            )
            if angular_mode == 0 and axial_mode == 0:
                spacing = buffer_ratio / radial_intervals
                inner_index = int(round(1.0 / spacing))
                principal_data["finite_element_visit_gain"] = float(
                    poisson_solution[inner_index]
                )
                right_hand_side = np.zeros(len(diagonal))
                right_hand_side[inner_index] = 1.0
                banded = np.zeros((3, len(diagonal)))
                banded[0, 1:] = off_diagonal
                banded[1, :] = diagonal
                banded[2, :-1] = off_diagonal
                green_column = solve_banded(
                    (1, 1), banded, right_hand_side
                )
                principal_data["inner_diagonal_Green_norm"] = float(
                    green_column[inner_index]
                )

            for profile in cutoff_profiles:
                cutoff = _cutoff_values(radii, *profile)
                cutoff_solution = cutoff * poisson_solution
                energy = _form_energy(
                    diagonal, off_diagonal, cutoff_solution
                )
                if energy > maxima[profile]["cutoff_Poisson_energy"]:
                    maxima[profile] = {
                        "cutoff_Poisson_energy": energy,
                        "maximizing_angular_mode": angular_mode,
                        "maximizing_axial_mode": axial_mode,
                    }
    return maxima, principal_data


def _allowable_alpha(
    generation_criterion: float, condition_number: float
) -> float:
    excess_multiplier = 1.0 / math.sqrt(generation_criterion) - 1.0
    return excess_multiplier / (condition_number + excess_multiplier)


def audit() -> dict[str, object]:
    perturbation = _load_script(
        "finite_cylinder_perturbation_margin_audit.py",
        "finite_cylinder_perturbation_for_poisson_cutoff",
    )
    finite = _load_script(
        "finite_cylinder_mode_audit.py",
        "finite_cylinder_for_poisson_cutoff",
    )
    axial = finite._load_axial_module()
    boundary_l2 = _load_script(
        "gaussian_boundary_l2_transfer_audit.py",
        "gaussian_boundary_l2_for_poisson_cutoff",
    ).audit()
    leray_gate = _load_script(
        "three_dimensional_leray_gate_audit.py",
        "three_dimensional_leray_for_poisson_cutoff",
    ).audit()

    reynolds = 0.5
    half_height = 1.5
    buffer_ratio = 2.0
    generation_criterion = boundary_l2["visit_rows"][0][
        "Gaussian_L2_complete_generation_criterion"
    ]
    exact_visit_gain = boundary_l2["visit_rows"][0][
        "Gaussian_L2_visit_operator_norm"
    ]
    unit_relative_form_mass_budget = next(
        row["allowed_q_l3_over_2_over_nu"]
        for row in leray_gate["spectral_budget_rows"]
        if row["tube_reynolds"] == reynolds
    )
    cutoff_profiles = (
        (1.0, 1.5),
        (1.25, 1.5),
        (1.25, 1.75),
        (1.5, 1.75),
        (1.5, 1.9),
        (1.5, 2.0),
        (1.6, 2.0),
        (1.7, 2.0),
        (1.75, 2.0),
        (1.8, 2.0),
        (1.9, 2.0),
        (1.91, 2.0),
        (1.92, 2.0),
        (1.93, 2.0),
        (1.95, 2.0),
    )
    maxima, principal_data = _mode_energy_rows(
        perturbation,
        reynolds,
        half_height,
        cutoff_profiles,
        radial_intervals=400,
        axial_mode_count=61,
        maximum_angular_mode=16,
        buffer_ratio=buffer_ratio,
    )
    inner_diagonal = principal_data["inner_diagonal_Green_norm"]
    profile_rows = []
    for support_radius, taper_radius in cutoff_profiles:
        maximum = maxima[(support_radius, taper_radius)]
        condition_number = math.sqrt(
            inner_diagonal * maximum["cutoff_Poisson_energy"]
        ) / exact_visit_gain
        allowable_alpha = _allowable_alpha(
            generation_criterion, condition_number
        )
        profile_rows.append(
            {
                "perturbation_support_radius": support_radius,
                "cutoff_taper_radius": taper_radius,
                "strictly_unperturbed_outer_collar_width": (
                    buffer_ratio - support_radius
                ),
                "cutoff_transition_width": (
                    taper_radius - support_radius
                ),
                **maximum,
                "Poisson_cutoff_condition_number": condition_number,
                "allowable_relative_form_alpha": allowable_alpha,
                "conservative_L3_over_2_mass_budget_over_nu": (
                    allowable_alpha * unit_relative_form_mass_budget
                ),
            }
        )

    convergence_profiles = ((1.25, 1.75), (1.5, 2.0), (1.91, 2.0))
    convergence_rows = []
    for radial_intervals in (200, 400, 800):
        convergence_maxima, convergence_principal = _mode_energy_rows(
            perturbation,
            reynolds,
            half_height,
            convergence_profiles,
            radial_intervals=radial_intervals,
            axial_mode_count=41,
            maximum_angular_mode=10,
            buffer_ratio=buffer_ratio,
        )
        rows = []
        for profile in convergence_profiles:
            maximum = convergence_maxima[profile]
            condition_number = math.sqrt(
                convergence_principal["inner_diagonal_Green_norm"]
                * maximum["cutoff_Poisson_energy"]
            ) / exact_visit_gain
            rows.append(
                {
                    "profile": str(profile),
                    **maximum,
                    "Poisson_cutoff_condition_number": condition_number,
                }
            )
        convergence_rows.append(
            {
                "radial_intervals": radial_intervals,
                "rows": rows,
                "finite_element_visit_gain": convergence_principal[
                    "finite_element_visit_gain"
                ],
            }
        )

    condition_sequences = {
        str(profile): [
            convergence["rows"][index][
                "Poisson_cutoff_condition_number"
            ]
            for convergence in convergence_rows
        ]
        for index, profile in enumerate(convergence_profiles)
    }
    exact_principal_axial = perturbation._axial_basis(
        reynolds, half_height, grid_points=801, mode_count=1
    )["eigenvalues"][0]
    exact_radial_gain = axial._constant_killing_visit_gain(
        reynolds, buffer_ratio, float(exact_principal_axial)
    )
    result: dict[str, object] = {
        "same_boundary_Poisson_problems": (
            "A_0 u_0=0 and (A_0-Q)u_q=0 with equal outer data"
        ),
        "zero_boundary_difference_equation": (
            "w=u_q-u_0=(A_0-Q)^(-1)Q u_0"
        ),
        "cutoff_form_hypothesis": (
            "Q is supported where zeta=1 and "
            "<v,Qv><=alpha<a_0 v,v> for zero-boundary v"
        ),
        "cutoff_Poisson_energy": (
            "E_zeta=sup_||f||=1 a_0[zeta P_0f,zeta P_0f]"
        ),
        "Poisson_transfer_bound": (
            "||B_q-B_0||<=alpha/(1-alpha)*sqrt(||D_i||*E_zeta)"
        ),
        "Poisson_condition_number": (
            "chi_P=sqrt(||D_i||*E_zeta)/||B_0||"
        ),
        "working_geometry": (
            "R_star=0.5, half-height=1.5, eta=2"
        ),
        "exact_principal_visit_gain": exact_radial_gain,
        "finite_element_principal_visit_gain": principal_data[
            "finite_element_visit_gain"
        ],
        "finite_element_visit_gain_matches_exact": bool(
            abs(
                principal_data["finite_element_visit_gain"]
                - exact_radial_gain
            )
            < 1.0e-5
        ),
        "inner_diagonal_Green_norm": inner_diagonal,
        "unit_relative_form_L3_over_2_mass_budget_over_nu": (
            unit_relative_form_mass_budget
        ),
        "mass_translation": (
            "Q/nu<alpha*m/[S3*(m+2a)] with the conservative "
            "transverse margin at R_star=0.5"
        ),
        "profile_rows": profile_rows,
        "all_profile_budgets_are_positive": all(
            row["allowable_relative_form_alpha"] > 0.0
            for row in profile_rows
        ),
        "all_mode_maximizers_are_inside_retained_range": all(
            row["maximizing_angular_mode"] < 16
            and row["maximizing_axial_mode"] < 60
            for row in profile_rows
        ),
        "convergence_rows": convergence_rows,
        "condition_sequences": condition_sequences,
        "cutoff_condition_numbers_converge": all(
            abs(sequence[-1] - sequence[-2]) < 2.0e-4
            for sequence in condition_sequences.values()
        ),
        "interpretation": (
            "an unperturbed radial collar makes the outer L2 payoff "
            "compatible with critical interior form control; the cutoff "
            "energy, rather than a divergent boundary-source self-energy, "
            "is the correct conversion constant"
        ),
        "remaining_PDE_gate": (
            "localize the actual non-affine Navier-Stokes error inside such "
            "a collar while controlling cutoff and pressure commutators"
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
