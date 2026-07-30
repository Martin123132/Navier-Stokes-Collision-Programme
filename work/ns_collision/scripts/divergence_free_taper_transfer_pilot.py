"""Nonsymmetric Poisson pilot for the divergence-free shell taper."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, eig, eigvalsh, solve_triangular, svdvals
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.special import hyp1f1, iv, kv


SHARP_SOBOLEV_CONSTANT = 4.0 ** (2.0 / 3.0) / (
    3.0 * math.pi ** (4.0 / 3.0)
)


def _load_script(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node(ring: int, angle: int, angle_count: int) -> int:
    if ring == 0:
        return 0
    return 1 + (ring - 1) * angle_count + angle % angle_count


def _radial_mesh(
    taper_radius: float,
    outer_radius: float,
    core_steps: int,
    taper_steps: int,
    collar_steps: int,
) -> np.ndarray:
    return np.concatenate(
        [
            np.linspace(0.0, 1.0, core_steps + 1),
            np.linspace(1.0, taper_radius, taper_steps + 1)[1:],
            np.linspace(
                taper_radius, outer_radius, collar_steps + 1
            )[1:],
        ]
    )


def _uniform_boundary_mass(angle_count: int) -> np.ndarray:
    angle_step = 2.0 * math.pi / angle_count
    mass = np.zeros((angle_count, angle_count))
    local_mass = angle_step / (2.0 * math.pi) * np.array(
        [[2.0, 1.0], [1.0, 2.0]]
    ) / 6.0
    for angle in range(angle_count):
        indices = (angle, (angle + 1) % angle_count)
        mass[np.ix_(indices, indices)] += local_mass
    return mass


def _taper_values(
    taper_polynomial,
    radius: float,
    taper_radius: float,
) -> tuple[float, float, float]:
    if radius <= 1.0:
        return 1.0, 0.0, 0.0
    if radius >= taper_radius:
        return 0.0, 0.0, 0.0
    width = taper_radius - 1.0
    normalized = (radius - 1.0) / width
    return (
        float(taper_polynomial(normalized)),
        float(taper_polynomial.deriv(1)(normalized) / width),
        float(taper_polynomial.deriv(2)(normalized) / width**2),
    )


def _drift_and_stretching(
    taper_polynomial,
    x: float,
    y: float,
    t_parameter: float,
    taper_radius: float,
) -> tuple[np.ndarray, float, float]:
    radius = math.hypot(x, y)
    taper, first, second = _taper_values(
        taper_polynomial, radius, taper_radius
    )
    strength = t_parameter + 0.5
    if radius > 0.0:
        anisotropic_x = x * taper + x * y**2 * first / radius
        anisotropic_y = -y * taper - x**2 * y * first / radius
        angle = math.atan2(y, x)
    else:
        anisotropic_x = anisotropic_y = 0.0
        angle = 0.0
    drift = np.array(
        [
            0.5 * x + strength * anisotropic_x,
            0.5 * y + strength * anisotropic_y,
        ]
    )
    centre = taper + 0.875 * radius * first + 0.125 * radius**2 * second
    oscillation = (radius**2 * second - radius * first) / 8.0
    mixed = centre - oscillation * math.cos(4.0 * angle)
    diagonal = -oscillation * math.sin(4.0 * angle)
    strain_amplification = math.hypot(mixed, diagonal)
    maximum_forward_stretching = max(
        1.0, -0.5 + strength * strain_amplification
    )
    return drift, maximum_forward_stretching, strain_amplification


def _assemble_operator(
    taper_polynomial,
    t_parameter: float,
    axial_eigenvalue: float,
    taper_radius: float,
    outer_radius: float,
    radii: np.ndarray,
    angle_count: int,
) -> tuple[csc_matrix, csc_matrix, np.ndarray, dict[str, float]]:
    ring_count = len(radii) - 1
    node_count = 1 + ring_count * angle_count
    coordinates = np.zeros((node_count, 2))
    for ring in range(1, ring_count + 1):
        radius = radii[ring]
        for angle_index in range(angle_count):
            angle = 2.0 * math.pi * angle_index / angle_count
            coordinates[_node(ring, angle_index, angle_count)] = (
                radius * math.cos(angle),
                radius * math.sin(angle),
            )

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    cutoff_rows: list[int] = []
    cutoff_columns: list[int] = []
    cutoff_data: list[float] = []
    maximum_stretching = 1.0
    maximum_strain_amplification = 1.0
    barycentric_points = (
        np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]),
        np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]),
        np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]),
    )

    def add_triangle(indices: tuple[int, int, int]) -> None:
        nonlocal maximum_stretching, maximum_strain_amplification
        triangle = coordinates[np.asarray(indices)]
        affine = np.column_stack([np.ones(3), triangle])
        area = abs(float(np.linalg.det(affine))) / 2.0
        gradients = np.linalg.inv(affine)[1:, :].T
        local = np.zeros((3, 3))
        local_cutoff = np.zeros((3, 3))
        for shape in barycentric_points:
            point = shape @ triangle
            drift, stretching, strain_amplification = (
                _drift_and_stretching(
                    taper_polynomial,
                    float(point[0]),
                    float(point[1]),
                    t_parameter,
                    taper_radius,
                )
            )
            maximum_stretching = max(maximum_stretching, stretching)
            maximum_strain_amplification = max(
                maximum_strain_amplification, strain_amplification
            )
            quadrature_weight = area / 3.0 / (2.0 * math.pi)
            for test_index in range(3):
                for trial_index in range(3):
                    local[test_index, trial_index] += quadrature_weight * (
                        float(
                            gradients[test_index]
                            @ gradients[trial_index]
                        )
                        - shape[test_index]
                        * float(drift @ gradients[trial_index])
                        + (axial_eigenvalue - stretching)
                        * shape[test_index]
                        * shape[trial_index]
                    )
                    radius = math.hypot(float(point[0]), float(point[1]))
                    if taper_radius < radius < outer_radius:
                        local_cutoff[test_index, trial_index] += (
                            quadrature_weight
                            / (outer_radius - taper_radius) ** 2
                            * shape[test_index]
                            * shape[trial_index]
                        )
        for local_row, global_row in enumerate(indices):
            for local_column, global_column in enumerate(indices):
                rows.append(global_row)
                columns.append(global_column)
                data.append(local[local_row, local_column])
                cutoff_rows.append(global_row)
                cutoff_columns.append(global_column)
                cutoff_data.append(
                    local_cutoff[local_row, local_column]
                )

    for angle_index in range(angle_count):
        add_triangle(
            (
                0,
                _node(1, angle_index, angle_count),
                _node(1, angle_index + 1, angle_count),
            )
        )
    for ring in range(2, ring_count + 1):
        for angle_index in range(angle_count):
            inner_left = _node(ring - 1, angle_index, angle_count)
            outer_left = _node(ring, angle_index, angle_count)
            outer_right = _node(ring, angle_index + 1, angle_count)
            inner_right = _node(ring - 1, angle_index + 1, angle_count)
            add_triangle((inner_left, outer_left, outer_right))
            add_triangle((inner_left, outer_right, inner_right))
    operator = coo_matrix(
        (data, (rows, columns)), shape=(node_count, node_count)
    ).tocsc()
    cutoff_form = coo_matrix(
        (cutoff_data, (cutoff_rows, cutoff_columns)),
        shape=(node_count, node_count),
    ).tocsc()
    return operator, cutoff_form, coordinates, {
        "quadrature_maximum_forward_stretching": maximum_stretching,
        "quadrature_maximum_strain_amplification": (
            maximum_strain_amplification
        ),
    }


def _weighted_operator_norm(
    operator: np.ndarray,
    target_mass: np.ndarray,
    source_mass: np.ndarray,
) -> float:
    target_cholesky = cholesky(target_mass, lower=True)
    source_cholesky = cholesky(source_mass, lower=True)
    source_inverse_transpose = solve_triangular(
        source_cholesky.T,
        np.eye(len(source_mass)),
        lower=False,
    )
    conjugated = (
        target_cholesky.T @ operator @ source_inverse_transpose
    )
    return float(svdvals(conjugated)[0])


def _largest_generalized_energy(
    energy: np.ndarray, boundary_mass: np.ndarray
) -> float:
    cholesky_factor = cholesky(boundary_mass, lower=True)
    transformed = solve_triangular(
        cholesky_factor, energy, lower=True
    )
    transformed = solve_triangular(
        cholesky_factor, transformed.T, lower=True
    ).T
    transformed = 0.5 * (transformed + transformed.T)
    return float(eigvalsh(transformed)[-1])


def _constant_energy_trace_norm(
    mass_shift: float, outer_radius: float
) -> float:
    if mass_shift <= 0.0:
        raise ValueError("the constant lower energy must have positive mass")
    root = math.sqrt(mass_shift)
    inner_i = iv(0, root)
    inner_right = kv(0, root) - (
        kv(0, root * outer_radius)
        / iv(0, root * outer_radius)
        * inner_i
    )
    return float(inner_i * inner_right)


def _transfer_row(
    taper_polynomial,
    t_parameter: float,
    axial_eigenvalue: float,
    taper_radius: float,
    outer_radius: float,
    radii: np.ndarray,
    angle_count: int,
    dense_strain_amplification: float,
) -> dict[str, float | bool]:
    operator, cutoff_form, _, diagnostics = _assemble_operator(
        taper_polynomial,
        t_parameter,
        axial_eigenvalue,
        taper_radius,
        outer_radius,
        radii,
        angle_count,
    )
    ring_count = len(radii) - 1
    node_count = operator.shape[0]
    outer_indices = np.array(
        [_node(ring_count, angle, angle_count) for angle in range(angle_count)]
    )
    interior_mask = np.ones(node_count, dtype=bool)
    interior_mask[outer_indices] = False
    interior_indices = np.flatnonzero(interior_mask)
    interior_positions = {
        int(node): position
        for position, node in enumerate(interior_indices)
    }
    inner_ring = int(np.flatnonzero(np.isclose(radii, 1.0))[0])
    inner_positions = np.array(
        [
            interior_positions[_node(inner_ring, angle, angle_count)]
            for angle in range(angle_count)
        ]
    )
    interior_operator = operator[interior_indices, :][:, interior_indices]
    boundary_coupling = operator[interior_indices, :][:, outer_indices]
    factorization = splu(interior_operator)
    poisson_interior = factorization.solve(-boundary_coupling.toarray())
    visit_operator = poisson_interior[inner_positions, :]
    boundary_mass = _uniform_boundary_mass(angle_count)
    visit_norm = _weighted_operator_norm(
        visit_operator, boundary_mass, boundary_mass
    )
    eigenvalues, left_vectors, right_vectors = eig(
        visit_operator, left=True, right=True
    )
    principal_index = int(np.argmax(eigenvalues.real))
    principal_multiplier = float(eigenvalues[principal_index].real)
    right_ground = np.asarray(
        right_vectors[:, principal_index].real
    )
    left_ground = np.asarray(left_vectors[:, principal_index].real)
    if np.mean(right_ground) < 0.0:
        right_ground *= -1.0
    if np.mean(left_ground) < 0.0:
        left_ground *= -1.0
    right_ground /= float(np.mean(right_ground))
    left_ground /= float(left_ground @ right_ground)
    markov_kernel = (
        np.diag(1.0 / right_ground)
        @ visit_operator
        @ np.diag(right_ground)
        / principal_multiplier
    )
    stationary_measure = left_ground * right_ground
    stationary_measure /= float(np.sum(stationary_measure))
    markov_conjugated = (
        np.diag(np.sqrt(stationary_measure))
        @ markov_kernel
        @ np.diag(1.0 / np.sqrt(stationary_measure))
    )
    markov_l2_norm = float(svdvals(markov_conjugated)[0])
    natural_observable_measure = left_ground / right_ground
    natural_observable_measure /= float(
        np.sum(natural_observable_measure)
    )
    natural_conjugated_visit = (
        np.diag(np.sqrt(natural_observable_measure))
        @ visit_operator
        @ np.diag(1.0 / np.sqrt(natural_observable_measure))
    )
    natural_visit_norm = float(svdvals(natural_conjugated_visit)[0])
    uniform_node_measure = np.full(angle_count, 1.0 / angle_count)
    natural_to_uniform_density = float(
        np.max(natural_observable_measure / uniform_node_measure)
    )
    uniform_to_natural_density = float(
        np.max(uniform_node_measure / natural_observable_measure)
    )
    natural_uniform_round_trip_mismatch = math.sqrt(
        natural_to_uniform_density * uniform_to_natural_density
    )
    full_poisson = np.zeros((node_count, angle_count))
    full_poisson[interior_indices, :] = poisson_interior
    full_poisson[outer_indices, :] = np.eye(angle_count)
    cutoff_energy_matrix = full_poisson.T @ (
        cutoff_form @ full_poisson
    )
    cutoff_energy = _largest_generalized_energy(
        cutoff_energy_matrix, boundary_mass
    )
    strength = t_parameter + 0.5
    stretching_upper_bound = max(
        1.0, -0.5 + strength * dense_strain_amplification
    )
    lower_energy_mass_shift = (
        axial_eigenvalue + 0.5 - stretching_upper_bound
    )
    trace_norm_squared = _constant_energy_trace_norm(
        lower_energy_mass_shift, outer_radius
    )
    sector_condition_number = (
        math.sqrt(trace_norm_squared * cutoff_energy) / visit_norm
    )
    true_split = math.exp(0.5 * 3.0 / 24.0) / 4.0
    generation_criterion = visit_norm**2 * (
        true_split + outer_radius**-2
    )
    natural_generation_criterion = principal_multiplier**2 * (
        true_split + outer_radius**-2
    )
    excess_multiplier = 1.0 / math.sqrt(generation_criterion) - 1.0
    if excess_multiplier > 0.0:
        allowable_alpha = excess_multiplier / (
            sector_condition_number + excess_multiplier
        )
    else:
        allowable_alpha = 0.0
    return {
        "t_parameter": t_parameter,
        "axial_eigenvalue": axial_eigenvalue,
        "angle_count": angle_count,
        "visit_operator_uniform_angular_L2_norm": visit_norm,
        "minimum_visit_matrix_entry": float(np.min(visit_operator)),
        "Perron_principal_visit_multiplier": principal_multiplier,
        "Doob_Markov_maximum_row_sum_error": float(
            np.max(np.abs(np.sum(markov_kernel, axis=1) - 1.0))
        ),
        "Doob_stationary_residual": float(
            np.max(
                np.abs(
                    stationary_measure @ markov_kernel
                    - stationary_measure
                )
            )
        ),
        "Doob_stationary_Markov_L2_norm": markov_l2_norm,
        "natural_observable_visit_L2_norm": natural_visit_norm,
        "natural_visit_norm_matches_Perron_multiplier": bool(
            abs(natural_visit_norm - principal_multiplier) < 1.0e-10
        ),
        "natural_to_uniform_density_bound": natural_to_uniform_density,
        "uniform_to_natural_density_bound": uniform_to_natural_density,
        "natural_uniform_round_trip_mismatch": (
            natural_uniform_round_trip_mismatch
        ),
        "complete_generation_criterion": generation_criterion,
        "natural_Doob_generation_criterion": natural_generation_criterion,
        "maximum_one_history_measure_mismatch_for_natural_cycle": (
            1.0 / math.sqrt(natural_generation_criterion)
        ),
        "unperturbed_generation_closes": bool(generation_criterion < 1.0),
        "dense_stretching_upper_bound": stretching_upper_bound,
        "lower_symmetric_energy_mass_shift": lower_energy_mass_shift,
        "constant_lower_energy_trace_norm_squared": trace_norm_squared,
        "cutoff_Poisson_energy": cutoff_energy,
        "sector_Poisson_condition_number": sector_condition_number,
        "allowable_relative_form_alpha": allowable_alpha,
        "diagnostic_L3_over_2_mass_budget_over_nu": (
            allowable_alpha / SHARP_SOBOLEV_CONSTANT
        ),
        **diagnostics,
    }


def _exact_axisymmetric_gain(
    axial_eigenvalue: float, outer_radius: float
) -> float:
    kummer_parameter = 1.0 - axial_eigenvalue
    return float(
        hyp1f1(kummer_parameter, 1.0, -0.25)
        / hyp1f1(
            kummer_parameter,
            1.0,
            -0.25 * outer_radius**2,
        )
    )


def audit() -> dict[str, object]:
    taper_module = _load_script(
        "divergence_free_shell_taper_audit.py",
        "divergence_free_taper_for_transfer",
    )
    affine_module = _load_script(
        "anisotropic_poisson_transfer_pilot.py",
        "anisotropic_poisson_for_divergence_free_taper",
    )
    taper_radius = 2.65
    outer_radius = 2.75
    taper_polynomial, taper_diagnostics = taper_module.optimize_taper(
        taper_radius,
        maximum_degree=16,
        constraint_points=10_001,
        validation_points=200_001,
    )

    convergence_rows = []
    working_half_height = 1.2
    working_axial_eigenvalue = affine_module._axial_principal_eigenvalue(
        working_half_height
    )
    for core_steps, taper_steps, collar_steps, angle_count in (
        (16, 28, 8, 32),
        (24, 42, 12, 48),
        (32, 56, 16, 64),
    ):
        radii = _radial_mesh(
            taper_radius,
            outer_radius,
            core_steps,
            taper_steps,
            collar_steps,
        )
        row = _transfer_row(
            taper_polynomial,
            -0.5,
            working_axial_eigenvalue,
            taper_radius,
            outer_radius,
            radii,
            angle_count,
            taper_diagnostics["dense_validated_strain_amplification"],
        )
        exact_gain = _exact_axisymmetric_gain(
            working_axial_eigenvalue, outer_radius
        )
        convergence_rows.append(
            {
                "core_steps": core_steps,
                "taper_steps": taper_steps,
                "collar_steps": collar_steps,
                **row,
                "exact_axisymmetric_visit_gain": exact_gain,
                "axisymmetric_visit_gain_error": abs(
                    row["visit_operator_uniform_angular_L2_norm"]
                    - exact_gain
                ),
            }
        )

    radii = _radial_mesh(taper_radius, outer_radius, 24, 42, 12)
    spectrum_rows = [
        _transfer_row(
            taper_polynomial,
            float(t_parameter),
            working_axial_eigenvalue,
            taper_radius,
            outer_radius,
            radii,
            48,
            taper_diagnostics["dense_validated_strain_amplification"],
        )
        for t_parameter in np.linspace(-0.5, 1.0, 7)
    ]
    height_rows = []
    for half_height in (0.85, 1.0, 1.2, 1.5):
        axial_eigenvalue = affine_module._axial_principal_eigenvalue(
            half_height
        )
        row = _transfer_row(
            taper_polynomial,
            1.0,
            axial_eigenvalue,
            taper_radius,
            outer_radius,
            radii,
            48,
            taper_diagnostics["dense_validated_strain_amplification"],
        )
        height_rows.append(
            {"half_height_over_L": half_height, **row}
        )

    result: dict[str, object] = {
        "status": "nonsymmetric finite-element pilot; not a proof",
        "backward_drift": (
            "b_perp=(x,y)/2+(t+1/2)(partial_y(chi*x*y),"
            "-partial_x(chi*x*y)), b_z=-z"
        ),
        "taper_radius": taper_radius,
        "outer_buffer_radius": outer_radius,
        "taper_dense_strain_amplification": taper_diagnostics[
            "dense_validated_strain_amplification"
        ],
        "boundary_norm": (
            "uniform angular L2 at r=1 and r=eta, Gaussian axial mode"
        ),
        "working_half_height_over_L": working_half_height,
        "working_axial_eigenvalue": working_axial_eigenvalue,
        "axisymmetric_convergence_rows": convergence_rows,
        "working_height_spectrum_rows": spectrum_rows,
        "worst_spectrum_height_rows": height_rows,
        "axisymmetric_visit_calibration_converges": bool(
            convergence_rows[-1]["axisymmetric_visit_gain_error"]
            < convergence_rows[0]["axisymmetric_visit_gain_error"]
        ),
        "working_height_all_sampled_spectra_close": bool(
            all(row["unperturbed_generation_closes"] for row in spectrum_rows)
        ),
        "working_height_worst_sample_is_t_1": bool(
            spectrum_rows[-1]["complete_generation_criterion"]
            == max(
                row["complete_generation_criterion"]
                for row in spectrum_rows
            )
        ),
        "all_sampled_visit_matrices_are_strictly_positive": bool(
            all(row["minimum_visit_matrix_entry"] > 0.0 for row in spectrum_rows)
        ),
        "all_sampled_Doob_transforms_are_Markov_L2_contractions": bool(
            all(
                row["Doob_Markov_maximum_row_sum_error"] < 1.0e-12
                and row["Doob_stationary_residual"] < 1.0e-12
                and row["Doob_stationary_Markov_L2_norm"]
                <= 1.0 + 1.0e-12
                and row["natural_visit_norm_matches_Perron_multiplier"]
                for row in spectrum_rows
            )
        ),
        "all_sampled_natural_cycles_improve_uniform_L2_criterion": bool(
            all(
                row["natural_Doob_generation_criterion"]
                <= row["complete_generation_criterion"] + 1.0e-12
                for row in spectrum_rows
            )
        ),
        "all_sampled_natural_uniform_conversions_retain_closure": bool(
            all(
                row["natural_uniform_round_trip_mismatch"] ** 2
                * row["natural_Doob_generation_criterion"]
                < 1.0
                for row in spectrum_rows
            )
        ),
        "quadrature_stretching_stays_below_1p0001": bool(
            max(
                row["quadrature_maximum_forward_stretching"]
                for row in spectrum_rows
            )
            < 1.0001
        ),
        "rigorous_nonsymmetric_transfer_certified": False,
        "remaining_gate": (
            "replace uniform boundary L2 by controlled physical hitting "
            "measures, enclose the finite-element constants, and control "
            "first-order drift error relative to the fitted taper"
        ),
    }
    positive_checks = (
        result["axisymmetric_visit_calibration_converges"],
        result["working_height_all_sampled_spectra_close"],
        result["working_height_worst_sample_is_t_1"],
        result["quadrature_stretching_stays_below_1p0001"],
        result["all_sampled_visit_matrices_are_strictly_positive"],
        result[
            "all_sampled_Doob_transforms_are_Markov_L2_contractions"
        ],
        result[
            "all_sampled_natural_cycles_improve_uniform_L2_criterion"
        ],
        result[
            "all_sampled_natural_uniform_conversions_retain_closure"
        ],
    )
    result["all_positive_transfer_checks_pass"] = all(positive_checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
