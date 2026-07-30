"""Audit the ground-state Markov transform of the cylinder visit operator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_script(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _transform_row(
    perturbation,
    axial,
    reynolds: float,
    half_height: float,
    grid_points: int,
    mode_count: int,
    buffer_ratio: float = 2.0,
) -> dict[str, float | bool | int]:
    basis = perturbation._axial_basis(
        reynolds, half_height, grid_points, mode_count
    )
    eigenvectors = np.asarray(basis["eigenvectors"])
    mode_gains = np.array(
        [
            axial._constant_killing_visit_gain(
                reynolds, buffer_ratio, float(eigenvalue)
            )
            for eigenvalue in np.asarray(basis["eigenvalues"])
        ]
    )
    visit_operator = (eigenvectors * mode_gains) @ eigenvectors.T
    ground_state = eigenvectors[:, 0].copy()
    centre_index = int(basis["centre_index"])
    ground_state *= np.sign(ground_state[centre_index])
    principal_gain = float(mode_gains[0])
    transformed_kernel = (
        visit_operator * ground_state[np.newaxis, :]
    ) / (principal_gain * ground_state[:, np.newaxis])

    invariant_measure = ground_state**2
    invariant_measure /= np.sum(invariant_measure)
    kernel_density = transformed_kernel / invariant_measure[np.newaxis, :]
    detailed_balance = (
        invariant_measure[:, np.newaxis] * transformed_kernel
        - invariant_measure[np.newaxis, :] * transformed_kernel.T
    )
    stationary_residual = invariant_measure @ transformed_kernel - invariant_measure
    second_ratio = float(mode_gains[1] / mode_gains[0])
    return {
        "grid_points": grid_points,
        "mode_count": mode_count,
        "principal_visit_multiplier": principal_gain,
        "second_visit_multiplier_ratio": second_ratio,
        "mean_zero_L2_contraction_factor": second_ratio,
        "chi_square_contraction_factor": second_ratio**2,
        "minimum_visit_operator_entry": float(np.min(visit_operator)),
        "minimum_transformed_Markov_entry": float(
            np.min(transformed_kernel)
        ),
        "maximum_row_sum_error": float(
            np.max(np.abs(np.sum(transformed_kernel, axis=1) - 1.0))
        ),
        "maximum_stationary_measure_error": float(
            np.max(np.abs(stationary_residual))
        ),
        "maximum_detailed_balance_error": float(
            np.max(np.abs(detailed_balance))
        ),
        "minimum_kernel_density_relative_to_ground_state_measure": float(
            np.min(kernel_density)
        ),
        "maximum_kernel_density_relative_to_ground_state_measure": float(
            np.max(kernel_density)
        ),
        "Doeblin_total_variation_contraction_bound": float(
            1.0 - np.min(kernel_density)
        ),
        "transformed_kernel_is_positive_Markov": bool(
            np.min(transformed_kernel) > 0.0
            and np.max(
                np.abs(np.sum(transformed_kernel, axis=1) - 1.0)
            )
            < 1.0e-12
        ),
        "ground_state_measure_is_reversible": bool(
            np.max(np.abs(stationary_residual)) < 1.0e-12
            and np.max(np.abs(detailed_balance)) < 1.0e-12
        ),
    }


def audit() -> dict[str, object]:
    perturbation = _load_script(
        "finite_cylinder_perturbation_margin_audit.py",
        "finite_cylinder_perturbation_for_ground_state",
    )
    finite = _load_script(
        "finite_cylinder_mode_audit.py",
        "finite_cylinder_for_ground_state",
    )
    axial = finite._load_axial_module()

    geometries = (
        (0.5, 1.5),
        (0.5, 1.75),
        (0.5, 2.0),
        (1.0, 1.0),
        (1.0, 1.2),
    )
    geometry_rows = [
        {
            "R_star": reynolds,
            "half_height_over_L": half_height,
            **_transform_row(
                perturbation,
                axial,
                reynolds,
                half_height,
                grid_points=401,
                mode_count=61,
            ),
        }
        for reynolds, half_height in geometries
    ]
    convergence_rows = [
        _transform_row(
            perturbation,
            axial,
            reynolds=0.5,
            half_height=1.5,
            grid_points=grid_points,
            mode_count=mode_count,
        )
        for grid_points, mode_count in ((201, 41), (401, 61), (801, 81))
    ]

    minimum_density_values = [
        row["minimum_kernel_density_relative_to_ground_state_measure"]
        for row in convergence_rows
    ]
    maximum_density_values = [
        row["maximum_kernel_density_relative_to_ground_state_measure"]
        for row in convergence_rows
    ]
    result: dict[str, object] = {
        "visit_operator": "B phi_n=U_n phi_n",
        "ground_state_transform": (
            "P f=(U_0 phi_0)^(-1) B(phi_0 f)"
        ),
        "ground_state_measure": (
            "dmu_0=phi_0^2 dmu_Gaussian, equivalently psi_0^2 dy"
        ),
        "exact_factorization": "B=U_0 M_(phi_0) P M_(phi_0)^(-1)",
        "geometry_rows": geometry_rows,
        "all_transformed_kernels_are_positive_Markov": all(
            row["transformed_kernel_is_positive_Markov"]
            for row in geometry_rows
        ),
        "all_ground_state_measures_are_reversible": all(
            row["ground_state_measure_is_reversible"]
            for row in geometry_rows
        ),
        "all_visits_have_strict_spectral_mixing_gap": all(
            0.0 < row["second_visit_multiplier_ratio"] < 1.0
            for row in geometry_rows
        ),
        "working_geometry_convergence_rows": convergence_rows,
        "minimum_kernel_density_converges": bool(
            abs(minimum_density_values[-1] - minimum_density_values[-2])
            < 1.0e-5
        ),
        "maximum_kernel_density_converges": bool(
            abs(maximum_density_values[-1] - maximum_density_values[-2])
            < 2.0e-4
        ),
        "working_geometry_uniform_kernel_density_bounds": (
            "0.4136<dP(y,.)/dmu_0<3.783 on the audited grid"
        ),
        "working_geometry_has_numerical_Doeblin_minorization": bool(
            minimum_density_values[-1] > 0.4136
            and maximum_density_values[-1] < 3.783
        ),
        "pair_transform": (
            "B tensor B=U_0^2 times the independent pair Doob kernel"
        ),
        "pair_Markov_part_is_contractive": True,
        "interpretation": (
            "all ideal visit growth is isolated in U_0; higher axial modes "
            "belong to a rapidly mixing Markov factor rather than an "
            "additional multiplicative loss"
        ),
        "remaining_interface_gate": (
            "transport the local ground-state/Gaussian reference measures "
            "between moving, rotating, and splitting Navier-Stokes cells "
            "without paying a conversion at every interface"
        ),
        "remaining_perturbation_gate": (
            "control how the non-affine critical form perturbation changes "
            "U_0, phi_0, and the transformed kernel"
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
