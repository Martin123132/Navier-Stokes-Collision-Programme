"""Finite-element pilot for anisotropic affine Poisson transfer.

This is a converged numerical stress test, not a computer-assisted proof.
It retains axial separability but resolves the coupled transverse angular
problem created by a general symmetric trace-free affine core.
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.linalg import cholesky, eigvalsh, solve_triangular, svdvals
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.special import hyp1f1, i0


SUPPORT_RADIUS = 1.91
OUTER_RADIUS = 2.0
AXIAL_PRINCIPAL_EIGENVALUE = 0.66930259
AXIAL_EIGENVALUES = (
    0.669302586721769,
    4.0446798718669275,
    9.54448306701239,
    17.22619846383246,
    27.09781252705468,
)
AXISYMMETRIC_VISIT_GAIN = 0.8568168256166799
AXISYMMETRIC_INNER_GREEN_NORM = 0.51630576
AXISYMMETRIC_CONDITION_NUMBER = 4.356026632998257
BASE_GENERATION_CRITERION = 0.37890444740087703
FULL_AFFINE_AXISYMMETRIC_VISIT_GAIN = 1.2221476896875563
UNIFORM_GENERAL_AFFINE_MARGIN = 5.283185962946783
TRANSVERSE_CUBIC_IMS_COST = 3.4428880787259133
SHARP_SOBOLEV_CONSTANT = 4.0 ** (2.0 / 3.0) / (
    3.0 * math.pi ** (4.0 / 3.0)
)


def _node(ring: int, angle: int, angle_count: int) -> int:
    if ring == 0:
        return 0
    return 1 + (ring - 1) * angle_count + angle % angle_count


def _radial_mesh(
    core_steps: int, shell_steps: int, collar_steps: int
) -> np.ndarray:
    pieces = (
        np.linspace(0.0, 1.0, core_steps + 1),
        np.linspace(1.0, SUPPORT_RADIUS, shell_steps + 1)[1:],
        np.linspace(
            SUPPORT_RADIUS, OUTER_RADIUS, collar_steps + 1
        )[1:],
    )
    return np.concatenate(pieces)


def _boundary_mass(anisotropy: float, angle_count: int) -> np.ndarray:
    mass = np.zeros((angle_count, angle_count))
    angle_step = 2.0 * math.pi / angle_count
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(4)
    normalization = 2.0 * math.pi * i0(anisotropy)
    for angle_index in range(angle_count):
        indices = (angle_index, (angle_index + 1) % angle_count)
        for gauss_node, gauss_weight in zip(gauss_nodes, gauss_weights):
            local = 0.5 * (1.0 + gauss_node)
            angle = (angle_index + local) * angle_step
            shape = np.array([1.0 - local, local])
            density = math.exp(
                anisotropy * math.cos(2.0 * angle)
            ) / normalization
            mass[np.ix_(indices, indices)] += (
                0.5
                * angle_step
                * gauss_weight
                * density
                * np.outer(shape, shape)
            )
    return mass


def _axial_principal_eigenvalue(half_height: float) -> float:
    reynolds = 0.5
    boundary_argument = reynolds * half_height**2

    def boundary_value(axial_eigenvalue: float) -> float:
        return float(
            hyp1f1(
                -axial_eigenvalue / (4.0 * reynolds),
                0.5,
                boundary_argument,
            )
        )

    upper = max(1.0, math.pi**2 / (4.0 * half_height**2))
    while boundary_value(upper) > 0.0:
        upper *= 1.5
    return float(brentq(boundary_value, 0.0, upper, xtol=1.0e-13))


def _assemble_forms(
    t_parameter: float,
    axial_eigenvalue: float,
    radii: np.ndarray,
    angle_count: int,
    continuation: str = "ray_constant",
) -> tuple[csc_matrix, csc_matrix, np.ndarray]:
    if continuation not in {"ray_constant", "full_affine"}:
        raise ValueError("unknown affine shell continuation")
    anisotropy = (1.0 + 2.0 * t_parameter) / 4.0
    ring_count = len(radii) - 1
    node_count = 1 + ring_count * angle_count
    coordinates = np.zeros((node_count, 2))
    node_radii = np.zeros(node_count)
    for ring in range(1, ring_count + 1):
        radius = radii[ring]
        for angle_index in range(angle_count):
            angle = 2.0 * math.pi * angle_index / angle_count
            index = _node(ring, angle_index, angle_count)
            coordinates[index] = (
                radius * math.cos(angle),
                radius * math.sin(angle),
            )
            node_radii[index] = radius

    form_rows: list[int] = []
    form_columns: list[int] = []
    form_data: list[float] = []
    cutoff_rows: list[int] = []
    cutoff_columns: list[int] = []
    cutoff_data: list[float] = []
    barycentric_points = (
        np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]),
        np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]),
        np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]),
    )

    def add_triangle(indices: tuple[int, int, int]) -> None:
        triangle = coordinates[np.asarray(indices)]
        affine = np.column_stack([np.ones(3), triangle])
        determinant = float(np.linalg.det(affine))
        area = abs(determinant) / 2.0
        gradients = np.linalg.inv(affine)[1:, :].T
        local_form = np.zeros((3, 3))
        local_cutoff = np.zeros((3, 3))
        for shape in barycentric_points:
            point = shape @ triangle
            radius_squared = float(point @ point)
            if continuation == "full_affine":
                radial_factor = radius_squared
                stretching = 1.0
            else:
                radial_factor = min(radius_squared, 1.0)
                stretching = float(radius_squared < 1.0)
            angle = math.atan2(point[1], point[0])
            weight = math.exp(
                0.25 * radial_factor
                + anisotropy
                * radial_factor
                * math.cos(2.0 * angle)
            )
            potential = axial_eigenvalue - stretching
            quadrature_weight = area / 3.0 / (2.0 * math.pi)
            local_form += quadrature_weight * weight * (
                gradients @ gradients.T
                + potential * np.outer(shape, shape)
            )
            radius = math.sqrt(radius_squared)
            if SUPPORT_RADIUS < radius < OUTER_RADIUS:
                cutoff_gradient_squared = 1.0 / (
                    OUTER_RADIUS - SUPPORT_RADIUS
                ) ** 2
                local_cutoff += (
                    quadrature_weight
                    * weight
                    * cutoff_gradient_squared
                    * np.outer(shape, shape)
                )
        for local_row, global_row in enumerate(indices):
            for local_column, global_column in enumerate(indices):
                form_rows.append(global_row)
                form_columns.append(global_column)
                form_data.append(local_form[local_row, local_column])
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

    form = coo_matrix(
        (form_data, (form_rows, form_columns)),
        shape=(node_count, node_count),
    ).tocsc()
    cutoff_form = coo_matrix(
        (cutoff_data, (cutoff_rows, cutoff_columns)),
        shape=(node_count, node_count),
    ).tocsc()
    return form, cutoff_form, node_radii


def _weighted_norm(
    operator: np.ndarray,
    target_mass: np.ndarray,
    source_mass: np.ndarray | None = None,
) -> float:
    if source_mass is None:
        source_mass = target_mass
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


def _transfer_row(
    t_parameter: float,
    axial_eigenvalue: float,
    radii: np.ndarray,
    angle_count: int,
    continuation: str = "ray_constant",
) -> dict[str, float]:
    form, cutoff_form, _ = _assemble_forms(
        t_parameter,
        axial_eigenvalue,
        radii,
        angle_count,
        continuation,
    )
    ring_count = len(radii) - 1
    node_count = form.shape[0]
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

    interior_form = form[interior_indices, :][:, interior_indices]
    boundary_coupling = form[interior_indices, :][:, outer_indices]
    factorization = splu(interior_form)
    poisson_interior = factorization.solve(
        -boundary_coupling.toarray()
    )
    visit_operator = poisson_interior[inner_positions, :]

    anisotropy = (1.0 + 2.0 * t_parameter) / 4.0
    inner_boundary_mass = _boundary_mass(anisotropy, angle_count)
    outer_anisotropy = (
        4.0 * anisotropy
        if continuation == "full_affine"
        else anisotropy
    )
    outer_boundary_mass = _boundary_mass(
        outer_anisotropy, angle_count
    )
    visit_norm = _weighted_norm(
        visit_operator, inner_boundary_mass, outer_boundary_mass
    )

    inner_load = np.zeros((len(interior_indices), angle_count))
    inner_load[inner_positions, :] = inner_boundary_mass
    inner_green = factorization.solve(inner_load)[inner_positions, :]
    inner_green_norm = _weighted_norm(
        inner_green, inner_boundary_mass
    )

    full_poisson = np.zeros((node_count, angle_count))
    full_poisson[interior_indices, :] = poisson_interior
    full_poisson[outer_indices, :] = np.eye(angle_count)
    cutoff_energy = full_poisson.T @ (cutoff_form @ full_poisson)
    cutoff_energy_norm = _largest_generalized_energy(
        cutoff_energy, outer_boundary_mass
    )
    condition_number = (
        math.sqrt(inner_green_norm * cutoff_energy_norm) / visit_norm
    )
    generation_ratio = BASE_GENERATION_CRITERION * (
        visit_norm / AXISYMMETRIC_VISIT_GAIN
    ) ** 2
    excess_multiplier = 1.0 / math.sqrt(generation_ratio) - 1.0
    if excess_multiplier > 0.0:
        allowable_alpha = excess_multiplier / (
            condition_number + excess_multiplier
        )
    else:
        allowable_alpha = 0.0
    return {
        "t_parameter": t_parameter,
        "continuation": continuation,
        "anisotropy_kappa": anisotropy,
        "axial_eigenvalue": axial_eigenvalue,
        "visit_operator_norm": visit_norm,
        "inner_diagonal_Green_norm": inner_green_norm,
        "cutoff_Poisson_energy": cutoff_energy_norm,
        "Poisson_cutoff_condition_number": condition_number,
        "baseline_generation_criterion": generation_ratio,
        "unperturbed_generation_closes": bool(generation_ratio < 1.0),
        "allowable_relative_form_alpha": allowable_alpha,
    }


def audit() -> dict[str, object]:
    mesh_specs = (
        (16, 20, 18, 32),
        (24, 30, 30, 48),
        (32, 40, 45, 64),
    )
    convergence_rows = []
    for core_steps, shell_steps, collar_steps, angle_count in mesh_specs:
        radii = _radial_mesh(core_steps, shell_steps, collar_steps)
        row = _transfer_row(
            -0.5,
            AXIAL_PRINCIPAL_EIGENVALUE,
            radii,
            angle_count,
        )
        convergence_rows.append(
            {
                "core_steps": core_steps,
                "shell_steps": shell_steps,
                "collar_steps": collar_steps,
                "angle_count": angle_count,
                **row,
                "visit_gain_error_from_exact_axisymmetric_value": abs(
                    row["visit_operator_norm"]
                    - AXISYMMETRIC_VISIT_GAIN
                ),
                "inner_Green_error_from_axisymmetric_value": abs(
                    row["inner_diagonal_Green_norm"]
                    - AXISYMMETRIC_INNER_GREEN_NORM
                ),
                "condition_error_from_axisymmetric_value": abs(
                    row["Poisson_cutoff_condition_number"]
                    - AXISYMMETRIC_CONDITION_NUMBER
                ),
            }
        )

    core_steps, shell_steps, collar_steps, angle_count = mesh_specs[-1]
    stress_radii = _radial_mesh(core_steps, shell_steps, collar_steps)
    spectrum_rows = [
        _transfer_row(
            float(t_parameter),
            AXIAL_PRINCIPAL_EIGENVALUE,
            stress_radii,
            angle_count,
        )
        for t_parameter in np.linspace(-0.5, 1.0, 7)
    ]
    axial_stress_radii = _radial_mesh(24, 30, 30)
    axial_mode_stress_rows = []
    for t_parameter in (-0.5, 1.0):
        mode_rows = [
            {
                "axial_mode": axial_mode,
                **_transfer_row(
                    t_parameter,
                    axial_eigenvalue,
                    axial_stress_radii,
                    48,
                ),
            }
            for axial_mode, axial_eigenvalue in enumerate(AXIAL_EIGENVALUES)
        ]
        axial_mode_stress_rows.append(
            {"t_parameter": t_parameter, "mode_rows": mode_rows}
        )
    full_affine_radii = _radial_mesh(24, 30, 30)
    full_affine_rows = [
        _transfer_row(
            float(t_parameter),
            AXIAL_EIGENVALUES[0],
            full_affine_radii,
            48,
            continuation="full_affine",
        )
        for t_parameter in np.linspace(-0.5, 1.0, 7)
    ]
    compact_geometry_rows = []
    for half_height in np.linspace(0.75, 1.30, 12):
        axial_eigenvalue = _axial_principal_eigenvalue(
            float(half_height)
        )
        row = _transfer_row(
            1.0,
            axial_eigenvalue,
            full_affine_radii,
            48,
            continuation="full_affine",
        )
        axial_knot_spacing = float(half_height) / 2.0
        axial_ims_cost = 0.785 / axial_knot_spacing**2
        post_ims_margin = (
            UNIFORM_GENERAL_AFFINE_MARGIN
            + axial_eigenvalue
            - TRANSVERSE_CUBIC_IMS_COST
            - axial_ims_cost
        )
        if post_ims_margin > 0.0:
            unit_mass_budget = post_ims_margin / (
                SHARP_SOBOLEV_CONSTANT * (post_ims_margin + 1.0)
            )
        else:
            unit_mass_budget = 0.0
        compact_geometry_rows.append(
            {
                "half_height_over_L": float(half_height),
                "axial_principal_eigenvalue": axial_eigenvalue,
                "axial_cubic_knot_spacing_over_L": axial_knot_spacing,
                "dimensionless_axial_cubic_IMS_cost": axial_ims_cost,
                "post_IMS_uniform_spectral_margin": post_ims_margin,
                "unit_relative_form_mass_budget": unit_mass_budget,
                "diagnostic_final_mass_budget_over_nu": (
                    row["allowable_relative_form_alpha"]
                    * unit_mass_budget
                ),
                **row,
            }
        )
    optimized_compact_row = max(
        compact_geometry_rows,
        key=lambda row: row["diagnostic_final_mass_budget_over_nu"],
    )
    optimized_spectrum_rows = [
        _transfer_row(
            float(t_parameter),
            optimized_compact_row["axial_principal_eigenvalue"],
            full_affine_radii,
            48,
            continuation="full_affine",
        )
        for t_parameter in np.linspace(-0.5, 1.0, 7)
    ]
    result: dict[str, object] = {
        "status": "converged numerical pilot; not a rigorous enclosure",
        "trace_free_spectrum": "(-1-t,t,1), -1/2<=t<=1",
        "transverse_weight": (
            "exp[(1/4)min(r^2,1)+kappa min(r^2,1)cos(2theta)]"
        ),
        "anisotropy_parameter": "kappa=(1+2t)/4",
        "shell_continuation": (
            "the transverse affine potential is held constant along "
            "radial rays for 1<=r<=2"
        ),
        "shell_model_caveat": (
            "the ray-constant shell is reversible and axially separable, "
            "but it is not a divergence-free Navier-Stokes shell; this is "
            "a transfer stress test inside the existing ideal architecture"
        ),
        "cutoff_energy_identity": (
            "a(zeta u,zeta u)=integral |grad zeta|^2 u^2 dm"
        ),
        "axisymmetric_convergence_rows": convergence_rows,
        "spectrum_stress_rows": spectrum_rows,
        "axial_mode_stress_rows": axial_mode_stress_rows,
        "forced_full_affine_continuation_rows": full_affine_rows,
        "compact_full_affine_geometry_rows_worst_spectrum_t_1": (
            compact_geometry_rows
        ),
        "optimized_compact_full_affine_geometry_row": (
            optimized_compact_row
        ),
        "optimized_compact_full_affine_spectrum_rows": (
            optimized_spectrum_rows
        ),
        "full_affine_axisymmetric_exact_visit_gain": (
            FULL_AFFINE_AXISYMMETRIC_VISIT_GAIN
        ),
        "full_affine_axisymmetric_visit_calibration_error": abs(
            full_affine_rows[0]["visit_operator_norm"]
            - FULL_AFFINE_AXISYMMETRIC_VISIT_GAIN
        ),
        "axisymmetric_visit_calibration_converges": bool(
            convergence_rows[-1][
                "visit_gain_error_from_exact_axisymmetric_value"
            ]
            < convergence_rows[0][
                "visit_gain_error_from_exact_axisymmetric_value"
            ]
        ),
        "axisymmetric_condition_calibration_converges": bool(
            convergence_rows[-1][
                "condition_error_from_axisymmetric_value"
            ]
            < convergence_rows[0][
                "condition_error_from_axisymmetric_value"
            ]
        ),
        "sampled_visit_norm_is_worst_at_axisymmetric_endpoint": bool(
            spectrum_rows[0]["visit_operator_norm"]
            == max(row["visit_operator_norm"] for row in spectrum_rows)
        ),
        "sampled_allowable_alpha_remains_positive": bool(
            min(row["allowable_relative_form_alpha"] for row in spectrum_rows)
            > 0.0
        ),
        "sampled_diagonal_and_cutoff_norms_are_principal_axial_mode": bool(
            all(
                mode_stress["mode_rows"][0][
                    "inner_diagonal_Green_norm"
                ]
                == max(
                    row["inner_diagonal_Green_norm"]
                    for row in mode_stress["mode_rows"]
                )
                and mode_stress["mode_rows"][0][
                    "cutoff_Poisson_energy"
                ]
                == max(
                    row["cutoff_Poisson_energy"]
                    for row in mode_stress["mode_rows"]
                )
                and mode_stress["mode_rows"][0][
                    "visit_operator_norm"
                ]
                == max(
                    row["visit_operator_norm"]
                    for row in mode_stress["mode_rows"]
                )
                for mode_stress in axial_mode_stress_rows
            )
        ),
        "full_affine_axisymmetric_visit_calibrates": bool(
            abs(
                full_affine_rows[0]["visit_operator_norm"]
                - FULL_AFFINE_AXISYMMETRIC_VISIT_GAIN
            )
            < 1.0e-3
        ),
        "forced_full_affine_working_height_fails_for_some_spectra": bool(
            any(
                not row["unperturbed_generation_closes"]
                for row in full_affine_rows
            )
        ),
        "optimized_compact_full_affine_all_sampled_spectra_close": bool(
            all(
                row["unperturbed_generation_closes"]
                for row in optimized_spectrum_rows
            )
        ),
        "optimized_compact_full_affine_worst_sample_is_t_1": bool(
            optimized_spectrum_rows[-1]["baseline_generation_criterion"]
            == max(
                row["baseline_generation_criterion"]
                for row in optimized_spectrum_rows
            )
        ),
        "rigorous_general_affine_Poisson_transfer_certified": False,
        "next_gate": (
            "refine the compact t-family, verify higher axial modes, then "
            "replace convergence evidence by certified form comparisons "
            "or eigenvalue enclosures"
        ),
    }
    positive_checks = (
        result["axisymmetric_visit_calibration_converges"],
        result["axisymmetric_condition_calibration_converges"],
        result["sampled_visit_norm_is_worst_at_axisymmetric_endpoint"],
        result["sampled_allowable_alpha_remains_positive"],
        result[
            "sampled_diagonal_and_cutoff_norms_are_principal_axial_mode"
        ],
        result["full_affine_axisymmetric_visit_calibrates"],
        result[
            "forced_full_affine_working_height_fails_for_some_spectra"
        ],
        result[
            "optimized_compact_full_affine_all_sampled_spectra_close"
        ],
        result[
            "optimized_compact_full_affine_worst_sample_is_t_1"
        ],
    )
    result["all_positive_pilot_checks_pass"] = all(positive_checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
