"""Finite-element pilot for a constantly rotating affine strain.

This is a numerical counterexample search, not a certified bound. Rotation
about the cylinder axis turns the nonautonomous affine drift into a constant
nonsymmetric drift in rotating coordinates.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _load_affine_module():
    script = Path(__file__).resolve().with_name(
        "anisotropic_poisson_transfer_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "anisotropic_poisson_for_rotating_visit", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assemble_rotating_form(
    affine,
    t_parameter: float,
    angular_rate: float,
    axial_eigenvalue: float,
    radii: np.ndarray,
    angle_count: int,
):
    ring_count = len(radii) - 1
    node_count = 1 + ring_count * angle_count
    coordinates = np.zeros((node_count, 2))
    for ring in range(1, ring_count + 1):
        radius = radii[ring]
        for angle_index in range(angle_count):
            angle = 2.0 * math.pi * angle_index / angle_count
            coordinates[affine._node(ring, angle_index, angle_count)] = (
                radius * math.cos(angle),
                radius * math.sin(angle),
            )

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    barycentric_points = (
        np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]),
        np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]),
        np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]),
    )
    base_drift = np.diag([1.0 + t_parameter, -t_parameter])
    rotation_generator = np.array([[0.0, -1.0], [1.0, 0.0]])
    drift_matrix = base_drift - angular_rate * rotation_generator

    def add_triangle(indices: tuple[int, int, int]) -> None:
        triangle = coordinates[np.asarray(indices)]
        affine_map = np.column_stack([np.ones(3), triangle])
        area = abs(float(np.linalg.det(affine_map))) / 2.0
        gradients = np.linalg.inv(affine_map)[1:, :].T
        local = np.zeros((3, 3))
        for shape in barycentric_points:
            point = shape @ triangle
            drift = drift_matrix @ point
            trial_advection = gradients @ drift
            local += area / 3.0 * (
                gradients @ gradients.T
                - np.outer(shape, trial_advection)
                + (axial_eigenvalue - 1.0) * np.outer(shape, shape)
            )
        for local_row, global_row in enumerate(indices):
            for local_column, global_column in enumerate(indices):
                rows.append(global_row)
                columns.append(global_column)
                data.append(local[local_row, local_column])

    for angle_index in range(angle_count):
        add_triangle(
            (
                0,
                affine._node(1, angle_index, angle_count),
                affine._node(1, angle_index + 1, angle_count),
            )
        )
    for ring in range(2, ring_count + 1):
        for angle_index in range(angle_count):
            inner_left = affine._node(
                ring - 1, angle_index, angle_count
            )
            outer_left = affine._node(ring, angle_index, angle_count)
            outer_right = affine._node(
                ring, angle_index + 1, angle_count
            )
            inner_right = affine._node(
                ring - 1, angle_index + 1, angle_count
            )
            add_triangle((inner_left, outer_left, outer_right))
            add_triangle((inner_left, outer_right, inner_right))

    return coo_matrix(
        (data, (rows, columns)), shape=(node_count, node_count)
    ).tocsc()


def _visit_row(
    affine,
    t_parameter: float,
    angular_rate: float,
    axial_eigenvalue: float,
    radii: np.ndarray,
    angle_count: int,
) -> dict[str, float]:
    form = _assemble_rotating_form(
        affine,
        t_parameter,
        angular_rate,
        axial_eigenvalue,
        radii,
        angle_count,
    )
    ring_count = len(radii) - 1
    outer_indices = np.array(
        [
            affine._node(ring_count, angle, angle_count)
            for angle in range(angle_count)
        ]
    )
    interior_mask = np.ones(form.shape[0], dtype=bool)
    interior_mask[outer_indices] = False
    interior_indices = np.flatnonzero(interior_mask)
    interior_positions = {
        int(node): position
        for position, node in enumerate(interior_indices)
    }
    inner_ring = int(np.flatnonzero(np.isclose(radii, 1.0))[0])
    inner_positions = np.array(
        [
            interior_positions[
                affine._node(inner_ring, angle, angle_count)
            ]
            for angle in range(angle_count)
        ]
    )
    interior_form = form[interior_indices, :][:, interior_indices]
    boundary_coupling = form[interior_indices, :][:, outer_indices]
    poisson = splu(interior_form).solve(-boundary_coupling.toarray())
    visit_operator = poisson[inner_positions, :]

    uniform_mass = affine._boundary_mass(0.0, angle_count)
    uniform_norm = affine._weighted_norm(
        visit_operator, uniform_mass, uniform_mass
    )
    constant_payoff = visit_operator @ np.ones(angle_count)
    anisotropy = (1.0 + 2.0 * t_parameter) / 4.0
    inner_static_mass = affine._boundary_mass(anisotropy, angle_count)
    outer_static_mass = affine._boundary_mass(
        4.0 * anisotropy, angle_count
    )
    static_weighted_norm = affine._weighted_norm(
        visit_operator, inner_static_mass, outer_static_mass
    )
    return {
        "t_parameter": t_parameter,
        "angular_rate": angular_rate,
        "uniform_angular_L2_visit_norm": uniform_norm,
        "static_Gaussian_weighted_visit_norm_diagnostic": (
            static_weighted_norm
        ),
        "constant_outer_payoff_maximum": float(np.max(constant_payoff)),
        "constant_outer_payoff_minimum": float(np.min(constant_payoff)),
    }


def _exact_linear_transition(
    drift_matrix: np.ndarray, time_step: float
) -> tuple[np.ndarray, np.ndarray]:
    block = np.zeros((6, 6))
    block[:3, :3] = drift_matrix
    block[:3, 3:] = 2.0 * np.eye(3)
    block[3:, 3:] = -drift_matrix.T
    exponential = expm(block * time_step)
    transition = exponential[:3, :3]
    covariance = exponential[:3, 3:] @ transition.T
    covariance = 0.5 * (covariance + covariance.T)
    return transition, np.linalg.cholesky(covariance)


def _constant_payoff_monte_carlo_row(
    angular_rate: float,
    rotation_axis: str,
    path_count: int,
    time_step: float,
    maximum_time: float,
    seed: int,
) -> dict[str, float | str]:
    base_drift = np.diag([2.0, -1.0, -1.0])
    if rotation_axis == "cylinder":
        generator = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
    elif rotation_axis == "tilting":
        generator = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        )
    else:
        raise ValueError("unknown rotation axis")
    transition, covariance_cholesky = _exact_linear_transition(
        base_drift - angular_rate * generator, time_step
    )
    rng = np.random.default_rng(seed)
    positions = np.zeros((path_count, 3))
    positions[:, 0] = 1.0
    active = np.ones(path_count, dtype=bool)
    payoffs = np.zeros(path_count)

    for step in range(int(maximum_time / time_step)):
        active_indices = np.flatnonzero(active)
        if len(active_indices) == 0:
            break
        old = positions[active_indices].copy()
        new = (
            old @ transition.T
            + rng.standard_normal((len(active_indices), 3))
            @ covariance_cholesky.T
        )
        positions[active_indices] = new
        radial_exit = np.sum(new[:, :2] ** 2, axis=1) >= 4.0
        cap_exit = np.abs(new[:, 2]) >= 0.75
        exited = radial_exit | cap_exit
        if not np.any(exited):
            continue

        local_exit = np.flatnonzero(exited)
        old_exit = old[local_exit]
        new_exit = new[local_exit]
        displacement = new_exit - old_exit
        radial_fraction = np.full(len(local_exit), np.inf)
        cap_fraction = np.full(len(local_exit), np.inf)

        radial_mask = radial_exit[local_exit]
        if np.any(radial_mask):
            transverse_displacement = displacement[radial_mask, :2]
            transverse_old = old_exit[radial_mask, :2]
            quadratic = np.sum(transverse_displacement**2, axis=1)
            linear = 2.0 * np.sum(
                transverse_old * transverse_displacement, axis=1
            )
            constant = np.sum(transverse_old**2, axis=1) - 4.0
            discriminant = np.maximum(
                0.0, linear**2 - 4.0 * quadratic * constant
            )
            radial_fraction[radial_mask] = (
                -linear + np.sqrt(discriminant)
            ) / (2.0 * quadratic)

        cap_mask = cap_exit[local_exit]
        if np.any(cap_mask):
            cap_target = np.sign(new_exit[cap_mask, 2]) * 0.75
            cap_fraction[cap_mask] = (
                cap_target - old_exit[cap_mask, 2]
            ) / displacement[cap_mask, 2]

        radial_first = radial_fraction < cap_fraction
        exit_fraction = np.clip(
            np.minimum(radial_fraction, cap_fraction), 0.0, 1.0
        )
        global_exit = active_indices[local_exit]
        radial_global = global_exit[radial_first]
        payoffs[radial_global] = np.exp(
            (step + exit_fraction[radial_first]) * time_step
        )
        active[global_exit] = False

    payoff_mean = float(np.mean(payoffs))
    payoff_standard_error = float(
        np.std(payoffs, ddof=1) / math.sqrt(path_count)
    )
    return {
        "rotation_axis": rotation_axis,
        "angular_rate": angular_rate,
        "constant_outer_payoff_mean": payoff_mean,
        "payoff_standard_error": payoff_standard_error,
        "radial_exit_probability": float(np.mean(payoffs > 0.0)),
        "unresolved_path_fraction_at_maximum_time": float(np.mean(active)),
    }


def audit() -> dict[str, object]:
    affine = _load_affine_module()
    half_height = 0.75
    axial_eigenvalue = affine._axial_principal_eigenvalue(half_height)
    radii = affine._radial_mesh(20, 24, 24)
    angle_count = 48
    angular_rates = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    spectrum_parameters = (-0.5, 0.0, 0.5, 1.0)
    rows = [
        _visit_row(
            affine,
            t_parameter,
            angular_rate,
            axial_eigenvalue,
            radii,
            angle_count,
        )
        for t_parameter in spectrum_parameters
        for angular_rate in angular_rates
    ]
    grouped_rows = []
    for t_parameter in spectrum_parameters:
        spectrum_rows = [
            row for row in rows if row["t_parameter"] == t_parameter
        ]
        static_uniform = spectrum_rows[0]["uniform_angular_L2_visit_norm"]
        worst = max(
            spectrum_rows,
            key=lambda row: row["uniform_angular_L2_visit_norm"],
        )
        grouped_rows.append(
            {
                "t_parameter": t_parameter,
                "static_uniform_visit_norm": static_uniform,
                "maximum_sampled_uniform_visit_norm": worst[
                    "uniform_angular_L2_visit_norm"
                ],
                "maximizing_sampled_angular_rate": worst["angular_rate"],
                "maximum_to_static_uniform_ratio": worst[
                    "uniform_angular_L2_visit_norm"
                ]
                / static_uniform,
            }
        )

    static_t1 = next(
        row
        for row in rows
        if row["t_parameter"] == 1.0 and row["angular_rate"] == 0.0
    )
    independent_static_t1 = affine._transfer_row(
        1.0,
        axial_eigenvalue,
        radii,
        angle_count,
        continuation="full_affine",
    )
    calibration_error = abs(
        static_t1["static_Gaussian_weighted_visit_norm_diagnostic"]
        - independent_static_t1["visit_operator_norm"]
    )
    monte_carlo_path_count = 60_000
    monte_carlo_time_step = 0.00125
    monte_carlo_maximum_time = 4.0
    monte_carlo_specs = [("cylinder", 0.0)] + [
        (axis, rate)
        for axis in ("cylinder", "tilting")
        for rate in (1.0, 2.0, 4.0, 8.0)
    ]
    monte_carlo_rows = [
        _constant_payoff_monte_carlo_row(
            angular_rate=rate,
            rotation_axis=axis,
            path_count=monte_carlo_path_count,
            time_step=monte_carlo_time_step,
            maximum_time=monte_carlo_maximum_time,
            seed=19_072_026 + index,
        )
        for index, (axis, rate) in enumerate(monte_carlo_specs)
    ]
    monte_carlo_baseline = monte_carlo_rows[0]
    rotating_monte_carlo_rows = monte_carlo_rows[1:]
    result: dict[str, object] = {
        "status": "finite-element stress test; not a rigorous enclosure",
        "rotating_frame_drift": (
            "B_omega=diag(1+t,-t)-omega*J in the transverse disk"
        ),
        "rotation_axis": (
            "the cylinder axis; axial OU separation is retained"
        ),
        "compact_half_height_over_L": half_height,
        "axial_principal_eigenvalue": axial_eigenvalue,
        "mesh": {
            "core_steps": 20,
            "shell_steps": 24,
            "collar_steps": 24,
            "angle_count": angle_count,
        },
        "rows": rows,
        "spectrum_summaries": grouped_rows,
        "static_t1_weighted_norm": static_t1[
            "static_Gaussian_weighted_visit_norm_diagnostic"
        ],
        "independent_static_t1_weighted_norm": independent_static_t1[
            "visit_operator_norm"
        ],
        "static_formulation_calibration_error": calibration_error,
        "sampled_rotation_uniform_norm_increase_detected": any(
            summary["maximum_to_static_uniform_ratio"] > 1.001
            for summary in grouped_rows
        ),
        "tilting_constant_payoff_monte_carlo": {
            "path_count_per_row": monte_carlo_path_count,
            "time_step": monte_carlo_time_step,
            "maximum_time": monte_carlo_maximum_time,
            "initial_point": [1.0, 0.0, 0.0],
            "stretching_potential": 1.0,
            "rows": monte_carlo_rows,
            "sampled_rotation_payoff_increase_detected": any(
                row["constant_outer_payoff_mean"]
                > monte_carlo_baseline["constant_outer_payoff_mean"]
                + 3.0
                * math.sqrt(
                    row["payoff_standard_error"] ** 2
                    + monte_carlo_baseline["payoff_standard_error"] ** 2
                )
                for row in rotating_monte_carlo_rows
            ),
        },
        "nonautonomous_boundary_visit_certified": False,
        "scope_guard": (
            "the uniform angular boundary norm is a diagnostic and has "
            "not yet been identified with the evolving physical hitting "
            "law. The tilting rows use exact linear-SDE steps but discrete "
            "boundary detection, so they are statistical diagnostics, not "
            "upper bounds"
        ),
        "next_gate": (
            "stress-test piecewise switching histories and identify the "
            "evolving boundary measure, then seek a comparison theorem "
            "showing that the static endpoint is worst"
        ),
    }
    positive_checks = (
        axial_eigenvalue > 3.9,
        calibration_error < 0.02,
        all(row["uniform_angular_L2_visit_norm"] > 0.0 for row in rows),
        all(
            row["constant_outer_payoff_maximum"]
            >= row["constant_outer_payoff_minimum"]
            for row in rows
        ),
        all(
            row["unresolved_path_fraction_at_maximum_time"] == 0.0
            for row in monte_carlo_rows
        ),
        not result["tilting_constant_payoff_monte_carlo"]
        ["sampled_rotation_payoff_increase_detected"],
    )
    result["all_positive_rotating_visit_pilot_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
