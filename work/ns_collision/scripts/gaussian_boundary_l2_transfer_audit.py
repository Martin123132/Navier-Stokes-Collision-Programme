"""Audit the Gaussian boundary L2 norm for visits and Markov transfer."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm, null_space, svdvals


def _load_script(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _weighted_l2_operator_norm(
    backward_operator: np.ndarray,
    source_measure: np.ndarray,
    target_measure: np.ndarray,
) -> float:
    conjugated = (
        np.diag(np.sqrt(source_measure))
        @ backward_operator
        @ np.diag(1.0 / np.sqrt(target_measure))
    )
    return float(svdvals(conjugated)[0])


def _visit_row(
    perturbation,
    axial,
    reynolds: float,
    half_height: float,
    buffer_ratio: float = 2.0,
) -> dict[str, object]:
    basis = perturbation._axial_basis(
        reynolds, half_height, grid_points=801, mode_count=81
    )
    eigenvalues = np.asarray(basis["eigenvalues"])
    mode_gains = np.array(
        [
            axial._constant_killing_visit_gain(
                reynolds, buffer_ratio, float(eigenvalue)
            )
            for eigenvalue in eigenvalues
        ]
    )
    principal_gain = float(mode_gains[0])

    coefficients = np.asarray(basis["coefficients"])
    axial_grid = np.asarray(basis["axial_grid"])
    transformed_constant = np.exp(-reynolds * axial_grid**2 / 2.0)
    retained_input_norm = float(np.linalg.norm(coefficients))
    grid_input_norm = float(np.linalg.norm(transformed_constant))
    output_norm = float(np.linalg.norm(coefficients * mode_gains))
    constant_input_gain = output_norm / grid_input_norm
    retained_boundary_fraction = retained_input_norm / grid_input_norm

    pointwise_visit = perturbation._visit(
        basis,
        reynolds,
        buffer_ratio,
        core_potential=0.0,
        shell_potential=0.0,
    )
    pair_return = buffer_ratio ** (-2.0)
    true_split = math.exp(reynolds * 3.0 / 24.0) / 4.0
    l2_criterion = principal_gain**2 * (true_split + pair_return)
    pointwise_criterion = float(
        pointwise_visit["maximum_visit_gain"]
    ) ** 2 * (true_split + pair_return)
    allowed_measure_mismatch = 1.0 / math.sqrt(l2_criterion)

    return {
        "R_star": reynolds,
        "half_height_over_L": half_height,
        "principal_axial_eigenvalue": float(eigenvalues[0]),
        "principal_radial_multiplier": principal_gain,
        "second_radial_multiplier": float(mode_gains[1]),
        "second_to_principal_multiplier_ratio": float(
            mode_gains[1] / mode_gains[0]
        ),
        "all_retained_mode_multipliers_are_positive": bool(
            np.all(mode_gains > 0.0)
        ),
        "mode_multipliers_strictly_decrease": bool(
            np.all(np.diff(mode_gains) < 0.0)
        ),
        "Gaussian_L2_visit_operator_norm": principal_gain,
        "Gaussian_L2_pair_visit_operator_norm": principal_gain**2,
        "constant_boundary_Gaussian_L2_gain": constant_input_gain,
        "retained_constant_boundary_norm_fraction": (
            retained_boundary_fraction
        ),
        "pointwise_constant_boundary_gain": pointwise_visit[
            "maximum_visit_gain"
        ],
        "Gaussian_L2_complete_generation_criterion": l2_criterion,
        "pointwise_complete_generation_criterion": pointwise_criterion,
        "Gaussian_L2_criterion_is_smaller_than_pointwise": bool(
            l2_criterion < pointwise_criterion
        ),
        "maximum_one_history_entry_exit_measure_mismatch": (
            allowed_measure_mismatch
        ),
        "measure_mismatch_boundary_reproduces_closure_threshold": bool(
            abs(l2_criterion * allowed_measure_mismatch**2 - 1.0)
            < 1.0e-13
        ),
    }


def audit() -> dict[str, object]:
    perturbation = _load_script(
        "finite_cylinder_perturbation_margin_audit.py",
        "finite_cylinder_perturbation_for_boundary_l2",
    )
    finite = _load_script(
        "finite_cylinder_mode_audit.py",
        "finite_cylinder_for_boundary_l2",
    )
    axial = finite._load_axial_module()
    branching = _load_script(
        "branching_transfer_operator_audit.py",
        "branching_for_boundary_l2",
    )

    geometries = (
        (0.5, 1.5),
        (0.5, 1.75),
        (0.5, 2.0),
        (1.0, 1.0),
        (1.0, 1.2),
    )
    visit_rows = [
        _visit_row(perturbation, axial, reynolds, half_height)
        for reynolds, half_height in geometries
    ]

    generator = branching._build_interface_generator()
    semigroup = expm(0.7 * generator)
    source_measure = np.arange(2.0, 10.0)
    source_measure /= np.sum(source_measure)
    target_measure = semigroup @ source_measure
    backward_semigroup = semigroup.T
    dynamic_single_norm = _weighted_l2_operator_norm(
        backward_semigroup, source_measure, target_measure
    )

    pair_semigroup = np.kron(semigroup, semigroup)
    pair_source_measure = np.kron(source_measure, source_measure)
    pair_target_measure = np.kron(target_measure, target_measure)
    dynamic_pair_norm = _weighted_l2_operator_norm(
        pair_semigroup.T, pair_source_measure, pair_target_measure
    )

    stationary_basis = null_space(generator)
    stationary_measure = np.abs(stationary_basis[:, 0])
    stationary_measure /= np.sum(stationary_measure)
    stationary_single_norm = _weighted_l2_operator_norm(
        backward_semigroup, stationary_measure, stationary_measure
    )

    child_probabilities = np.arange(1.0, 9.0)
    child_probabilities /= np.sum(child_probabilities)
    child_backward_map = child_probabilities.reshape(1, -1)
    child_l2_norm = _weighted_l2_operator_norm(
        child_backward_map,
        np.ones(1),
        child_probabilities,
    )
    pair_child_probabilities = np.kron(
        child_probabilities, child_probabilities
    )
    pair_child_l2_norm = _weighted_l2_operator_norm(
        pair_child_probabilities.reshape(1, -1),
        np.ones(1),
        pair_child_probabilities,
    )

    stationary_to_child_density = float(
        np.max(child_probabilities / stationary_measure)
    )
    child_to_stationary_density = float(
        np.max(stationary_measure / child_probabilities)
    )
    synthetic_one_history_round_trip_mismatch = math.sqrt(
        stationary_to_child_density * child_to_stationary_density
    )
    synthetic_pair_round_trip_mismatch = (
        synthetic_one_history_round_trip_mismatch**2
    )

    result: dict[str, object] = {
        "global_reversible_density": (
            "dm=exp[-R_star*y^2+(R_star/2)*min(rho^2,1)] dx"
        ),
        "symmetric_drift_identity": (
            "Delta+grad(log dm) dot grad equals the affine core drift "
            "and the Brownian-radial shell drift"
        ),
        "Gaussian_boundary_measure": "dmu_h proportional exp(-R_star*y^2)dy",
        "visit_spectral_action": "B phi_n=U(zeta_n) phi_n",
        "visit_rows": visit_rows,
        "all_visit_mode_checks_pass": all(
            row["all_retained_mode_multipliers_are_positive"]
            and row["mode_multipliers_strictly_decrease"]
            and row["Gaussian_L2_criterion_is_smaller_than_pointwise"]
            and row[
                "measure_mismatch_boundary_reproduces_closure_threshold"
            ]
            for row in visit_rows
        ),
        "all_constant_boundary_norms_are_resolved": all(
            row["retained_constant_boundary_norm_fraction"] > 0.998
            for row in visit_rows
        ),
        "dynamic_measure_single_Markov_L2_norm": dynamic_single_norm,
        "dynamic_measure_pair_Markov_L2_norm": dynamic_pair_norm,
        "stationary_measure_single_Markov_L2_norm": (
            stationary_single_norm
        ),
        "dynamic_Markov_maps_are_L2_contractive": bool(
            dynamic_single_norm <= 1.0 + 1.0e-12
            and dynamic_pair_norm <= 1.0 + 1.0e-12
            and stationary_single_norm <= 1.0 + 1.0e-12
        ),
        "dynamic_measure_contraction_identity": (
            "nu=P mu implies ||P^T f||_(L2(mu))<=||f||_(L2(nu))"
        ),
        "single_child_expectation_L2_norm": child_l2_norm,
        "pair_child_expectation_L2_norm": pair_child_l2_norm,
        "child_branching_is_L2_contractive": bool(
            abs(child_l2_norm - 1.0) < 1.0e-12
            and abs(pair_child_l2_norm - 1.0) < 1.0e-12
        ),
        "synthetic_child_to_stationary_density_bound": (
            child_to_stationary_density
        ),
        "synthetic_stationary_to_child_density_bound": (
            stationary_to_child_density
        ),
        "synthetic_one_history_round_trip_measure_mismatch": (
            synthetic_one_history_round_trip_mismatch
        ),
        "synthetic_pair_round_trip_measure_mismatch": (
            synthetic_pair_round_trip_mismatch
        ),
        "synthetic_fixed_measure_conversion_is_expansive": bool(
            synthetic_pair_round_trip_mismatch > 1.0
        ),
        "two_norm_cycle": (
            "track actual probability laws through all Markov phases; "
            "convert to the local Gaussian boundary measure only once on "
            "entry to and once on exit from a complete buffered visit"
        ),
        "remaining_measure_gate": (
            "bound the Radon-Nikodym entry/exit mismatch between actual "
            "hitting laws and the Gaussian axial boundary measure within "
            "the tabulated one-history allowance"
        ),
        "remaining_perturbation_gate": (
            "propagate the critical L^(3/2) interior form estimate to the "
            "Gaussian L2 boundary visit operator"
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
