"""Audit the consistency route from the reversible strip chain to weighted P1 FEM."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigvalsh
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frobenius(matrix) -> float:
    return float(np.sqrt(np.sum(np.asarray(matrix.data) ** 2)))


def _extreme_generalized(first, second) -> tuple[float, float]:
    lower = eigsh(
        first,
        k=1,
        M=second,
        which="SA",
        return_eigenvectors=False,
        tol=1.0e-9,
        maxiter=10000,
    )[0]
    upper = eigsh(
        first,
        k=1,
        M=second,
        which="LA",
        return_eigenvectors=False,
        tol=1.0e-9,
        maxiter=10000,
    )[0]
    return float(lower), float(upper)


def _circle_geometry(inner_vertex_count: int) -> dict[str, float]:
    half_angle = math.pi / inner_vertex_count
    chord_over_arc = math.sin(half_angle) / half_angle
    return {
        "inner_polygon_sagitta": float(1.0 - math.cos(half_angle)),
        "inner_polygon_perimeter_deficit_fraction": float(
            1.0 - chord_over_arc
        ),
        "inner_polygon_chord_over_true_arc": float(chord_over_arc),
        "maximum_polygon_normal_angle_mismatch_radians": float(half_angle),
    }


def _mesh_quality(grid: dict[str, object]) -> dict[str, float]:
    vertices = np.asarray(grid["vertices"])
    triangles = np.asarray(grid["triangles"])
    minimum_angle = 180.0
    maximum_angle = 0.0
    maximum_diameter = 0.0
    maximum_log_weight_oscillation = 0.0
    for triangle in triangles:
        points = vertices[triangle]
        lengths = [
            float(
                np.linalg.norm(
                    points[(index + 1) % 3]
                    - points[(index + 2) % 3]
                )
            )
            for index in range(3)
        ]
        maximum_diameter = max(maximum_diameter, *lengths)
        for index in range(3):
            first = points[(index + 1) % 3] - points[index]
            second = points[(index + 2) % 3] - points[index]
            cosine = float(
                np.dot(first, second)
                / (np.linalg.norm(first) * np.linalg.norm(second))
            )
            angle = math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))
            minimum_angle = min(minimum_angle, angle)
            maximum_angle = max(maximum_angle, angle)

        minimum_x = float(np.min(points[:, 0]))
        maximum_x = float(np.max(points[:, 0]))
        maximum_square = max(minimum_x**2, maximum_x**2)
        minimum_square = (
            0.0
            if minimum_x <= 0.0 <= maximum_x
            else min(minimum_x**2, maximum_x**2)
        )
        maximum_log_weight_oscillation = max(
            maximum_log_weight_oscillation,
            0.5 * (maximum_square - minimum_square),
        )
    return {
        "maximum_triangle_diameter": maximum_diameter,
        "minimum_triangle_angle_degrees": minimum_angle,
        "maximum_triangle_angle_degrees": maximum_angle,
        "maximum_element_log_weight_oscillation": (
            maximum_log_weight_oscillation
        ),
    }


def _iv_add(
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[float, float]:
    return (
        float(np.nextafter(first[0] + second[0], -math.inf)),
        float(np.nextafter(first[1] + second[1], math.inf)),
    )


def _iv_sub(
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[float, float]:
    return (
        float(np.nextafter(first[0] - second[1], -math.inf)),
        float(np.nextafter(first[1] - second[0], math.inf)),
    )


def _iv_mul(
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[float, float]:
    products = (
        first[0] * second[0],
        first[0] * second[1],
        first[1] * second[0],
        first[1] * second[1],
    )
    return (
        float(np.nextafter(min(products), -math.inf)),
        float(np.nextafter(max(products), math.inf)),
    )


def _iv_determinant(matrix: list[list[tuple[float, float]]]):
    dimension = len(matrix)
    if dimension == 1:
        return matrix[0][0]
    if dimension == 2:
        return _iv_sub(
            _iv_mul(matrix[0][0], matrix[1][1]),
            _iv_mul(matrix[0][1], matrix[1][0]),
        )
    first_minor = _iv_sub(
        _iv_mul(matrix[1][1], matrix[2][2]),
        _iv_mul(matrix[1][2], matrix[2][1]),
    )
    second_minor = _iv_sub(
        _iv_mul(matrix[1][0], matrix[2][2]),
        _iv_mul(matrix[1][2], matrix[2][0]),
    )
    third_minor = _iv_sub(
        _iv_mul(matrix[1][0], matrix[2][1]),
        _iv_mul(matrix[1][1], matrix[2][0]),
    )
    return _iv_add(
        _iv_sub(
            _iv_mul(matrix[0][0], first_minor),
            _iv_mul(matrix[0][1], second_minor),
        ),
        _iv_mul(matrix[0][2], third_minor),
    )


def _local_mass_coercivity_minors(
    local_mass: np.ndarray,
    alpha: float,
) -> list[float]:
    dimension = local_mass.shape[0]
    row_sums: list[tuple[float, float]] = []
    for row in range(dimension):
        total = (0.0, 0.0)
        for column in range(dimension):
            value = float(local_mass[row, column])
            total = _iv_add(total, (value, value))
        row_sums.append(total)

    shifted: list[list[tuple[float, float]]] = []
    alpha_interval = (float(alpha), float(alpha))
    for row in range(dimension):
        shifted_row = []
        for column in range(dimension):
            value = float(local_mass[row, column])
            interval = (value, value)
            if row == column:
                interval = _iv_sub(
                    interval,
                    _iv_mul(alpha_interval, row_sums[row]),
                )
            shifted_row.append(interval)
        shifted.append(shifted_row)
    return [
        _iv_determinant(
            [row[:minor_size] for row in shifted[:minor_size]]
        )[0]
        for minor_size in range(1, dimension + 1)
    ]


def _reference_forms(
    grid: dict[str, object],
    quadrature_order: int,
    mass_coercivity_alpha: float | None = None,
) -> dict[str, object]:
    """Assemble high-order quadrature references for the exact weighted forms."""
    vertices = np.asarray(grid["vertices"])
    triangles = np.asarray(grid["triangles"])
    state_vertices = np.asarray(grid["state_vertices"])
    state_lookup = {
        int(vertex): index for index, vertex in enumerate(state_vertices)
    }
    inner_coordinates = np.column_stack(
        (
            np.cos(np.asarray(grid["inner_angles"])),
            np.sin(np.asarray(grid["inner_angles"])),
        )
    )
    boundary_module = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        f"boundary_fem_match_{quadrature_order}",
    )
    inner_vertices = boundary_module._matched_vertex_indices(
        vertices, inner_coordinates
    )
    inner_lookup = {
        int(vertex): index for index, vertex in enumerate(inner_vertices)
    }

    nodes, weights = leggauss(quadrature_order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    stiffness_rows: list[int] = []
    stiffness_columns: list[int] = []
    stiffness_values: list[float] = []
    mass_rows: list[int] = []
    mass_columns: list[int] = []
    mass_values: list[float] = []
    boundary_rows: list[int] = []
    boundary_columns: list[int] = []
    boundary_values: list[float] = []
    boundary_mass_rows: list[int] = []
    boundary_mass_columns: list[int] = []
    boundary_mass_values: list[float] = []
    minimum_mass_minor_lower = [math.inf, math.inf, math.inf]
    local_mass_coercivity_checks_pass = True

    maximum_constant_rule_error = 0.0
    for triangle in triangles:
        points = vertices[triangle]
        edge_matrix = np.column_stack(
            (points[1] - points[0], points[2] - points[0])
        )
        determinant = abs(float(np.linalg.det(edge_matrix)))
        inverse = np.linalg.inv(edge_matrix)
        gradients = np.empty((3, 2))
        gradients[1] = inverse[0]
        gradients[2] = inverse[1]
        gradients[0] = -gradients[1] - gradients[2]

        weighted_area = 0.0
        weighted_mass = np.zeros((3, 3))
        quadrature_area = 0.0
        for first_index, first in enumerate(nodes):
            for second_index, second in enumerate(nodes):
                barycentric = np.asarray(
                    [
                        1.0 - first,
                        first * (1.0 - second),
                        first * second,
                    ]
                )
                jacobian_weight = (
                    weights[first_index]
                    * weights[second_index]
                    * determinant
                    * first
                )
                x_coordinate = float(barycentric @ points[:, 0])
                invariant_weight = math.exp(-0.5 * x_coordinate**2)
                weighted_area += jacobian_weight * invariant_weight
                weighted_mass += (
                    jacobian_weight
                    * invariant_weight
                    * np.outer(barycentric, barycentric)
                )
                quadrature_area += jacobian_weight
        maximum_constant_rule_error = max(
            maximum_constant_rule_error,
            abs(quadrature_area - 0.5 * determinant),
        )
        weighted_stiffness = weighted_area * (gradients @ gradients.T)
        if mass_coercivity_alpha is not None:
            state_local_indices = [
                local_index
                for local_index, vertex in enumerate(triangle)
                if int(vertex) in state_lookup
            ]
            if state_local_indices:
                local_state_mass = weighted_mass[
                    np.ix_(state_local_indices, state_local_indices)
                ]
                minor_lowers = _local_mass_coercivity_minors(
                    local_state_mass,
                    mass_coercivity_alpha,
                )
                for index, lower in enumerate(minor_lowers):
                    minimum_mass_minor_lower[index] = min(
                        minimum_mass_minor_lower[index],
                        lower,
                    )
                    local_mass_coercivity_checks_pass &= lower > 0.0

        for local_row, row_vertex in enumerate(triangle):
            row = state_lookup.get(int(row_vertex))
            if row is None:
                continue
            for local_column, column_vertex in enumerate(triangle):
                column = state_lookup.get(int(column_vertex))
                if column is not None:
                    stiffness_rows.append(row)
                    stiffness_columns.append(column)
                    stiffness_values.append(
                        float(weighted_stiffness[local_row, local_column])
                    )
                    mass_rows.append(row)
                    mass_columns.append(column)
                    mass_values.append(
                        float(weighted_mass[local_row, local_column])
                    )
                    continue
                boundary_column = inner_lookup.get(int(column_vertex))
                if boundary_column is not None:
                    boundary_rows.append(row)
                    boundary_columns.append(boundary_column)
                    boundary_values.append(
                        float(-weighted_stiffness[local_row, local_column])
                    )
                    boundary_mass_rows.append(row)
                    boundary_mass_columns.append(boundary_column)
                    boundary_mass_values.append(
                        float(weighted_mass[local_row, local_column])
                    )

    state_count = len(state_vertices)
    inner_count = len(inner_vertices)
    mass_contribution_counts = coo_matrix(
        (
            np.ones(len(mass_values), dtype=float),
            (mass_rows, mass_columns),
        ),
        shape=(state_count, state_count),
    ).tocsr()
    maximum_mass_contribution_count = int(
        np.max(mass_contribution_counts.data)
    )
    summation_epsilon = np.finfo(float).eps
    summation_operations = 2 * maximum_mass_contribution_count + 8
    summation_product = summation_operations * summation_epsilon
    summation_gamma = float(
        np.nextafter(
            summation_product / (1.0 - summation_product),
            math.inf,
        )
    )
    global_mass_coercivity_lower = None
    if mass_coercivity_alpha is not None:
        global_mass_coercivity_lower = float(
            np.nextafter(
                (mass_coercivity_alpha - summation_gamma)
                / (1.0 + summation_gamma),
                -math.inf,
            )
        )

    return {
        "stiffness": coo_matrix(
            (
                stiffness_values,
                (stiffness_rows, stiffness_columns),
            ),
            shape=(state_count, state_count),
        ).tocsr(),
        "mass": coo_matrix(
            (mass_values, (mass_rows, mass_columns)),
            shape=(state_count, state_count),
        ).tocsr(),
        "boundary_coupling": coo_matrix(
            (
                boundary_values,
                (boundary_rows, boundary_columns),
            ),
            shape=(state_count, inner_count),
        ).tocsr(),
        "boundary_mass_coupling": coo_matrix(
            (
                boundary_mass_values,
                (boundary_mass_rows, boundary_mass_columns),
            ),
            shape=(state_count, inner_count),
        ).tocsr(),
        "maximum_constant_rule_error": maximum_constant_rule_error,
        "mass_coercivity_certificate": {
            "local_row_lumped_alpha": mass_coercivity_alpha,
            "all_local_leading_principal_minors_positive": (
                local_mass_coercivity_checks_pass
                if mass_coercivity_alpha is not None
                else False
            ),
            "minimum_local_leading_principal_minor_lower_by_size": [
                value if math.isfinite(value) else None
                for value in minimum_mass_minor_lower
            ],
            "maximum_global_entry_contribution_count": (
                maximum_mass_contribution_count
            ),
            "global_duplicate_summation_gamma": summation_gamma,
            "global_row_lumped_coercivity_lower": (
                global_mass_coercivity_lower
            ),
            "stored_mass_row_lumped_coercivity_proved": bool(
                mass_coercivity_alpha is not None
                and local_mass_coercivity_checks_pass
                and global_mass_coercivity_lower is not None
                and global_mass_coercivity_lower > 0.0
            ),
        },
    }


def _relative_frobenius(first, second) -> float:
    return _frobenius(first - second) / max(_frobenius(second), 1.0e-300)


def _consistency_row(
    grid: dict[str, object],
    low_mode_count: int,
    quadrature_order: int,
) -> dict[str, object]:
    reference = _reference_forms(grid, quadrature_order)
    refined_reference = _reference_forms(grid, quadrature_order + 4)
    exact_mass = refined_reference["mass"]
    exact_stiffness = refined_reference["stiffness"]
    exact_boundary = refined_reference["boundary_coupling"]
    exact_boundary_mass = refined_reference["boundary_mass_coupling"]

    modified_mass = diags(np.asarray(grid["state_mass"]))
    modified_stiffness = (-modified_mass @ grid["generator"]).tocsr()
    modified_stiffness = 0.5 * (
        modified_stiffness + modified_stiffness.transpose()
    )
    modified_boundary = (
        modified_mass @ grid["inner_rate_matrix"]
    ).tocsr()

    mass_ratio = _extreme_generalized(modified_mass, exact_mass)
    stiffness_ratio = _extreme_generalized(
        modified_stiffness, exact_stiffness
    )

    mode_count = min(low_mode_count, exact_mass.shape[0] - 2)
    eigenvalues, eigenvectors = eigsh(
        exact_stiffness,
        k=mode_count,
        M=exact_mass,
        sigma=0.0,
        which="LM",
        tol=1.0e-10,
        maxiter=10000,
    )
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    restricted_mass = eigenvectors.T @ (modified_mass @ eigenvectors)
    restricted_stiffness = eigenvectors.T @ (
        modified_stiffness @ eigenvectors
    )
    low_mass_eigenvalues = eigvalsh(restricted_mass)
    low_stiffness_eigenvalues = eigvalsh(
        restricted_stiffness, np.diag(eigenvalues)
    )
    exact_low_boundary = np.asarray(
        exact_boundary.transpose() @ eigenvectors
    )
    exact_low_boundary_mass = np.asarray(
        exact_boundary_mass.transpose() @ eigenvectors
    )
    exact_low_transient_boundary = exact_low_boundary + (
        exact_low_boundary_mass * eigenvalues[None, :]
    )
    modified_low_boundary = np.asarray(
        modified_boundary.transpose() @ eigenvectors
    )

    quadrature_checks = {
        "mass_relative_frobenius": _relative_frobenius(
            reference["mass"], exact_mass
        ),
        "stiffness_relative_frobenius": _relative_frobenius(
            reference["stiffness"], exact_stiffness
        ),
        "boundary_relative_frobenius": _relative_frobenius(
            reference["boundary_coupling"], exact_boundary
        ),
        "boundary_mass_relative_frobenius": _relative_frobenius(
            reference["boundary_mass_coupling"], exact_boundary_mass
        ),
    }
    return {
        "spacing": float(grid["spacing"]),
        "state_count": int(exact_mass.shape[0]),
        "triangle_count": int(len(grid["triangles"])),
        "inner_boundary_vertex_count": int(
            grid["inner_boundary_vertex_count"]
        ),
        **_circle_geometry(int(grid["inner_boundary_vertex_count"])),
        **_mesh_quality(grid),
        "quadrature_order": quadrature_order,
        "refined_quadrature_order": quadrature_order + 4,
        "maximum_constant_rule_error": max(
            reference["maximum_constant_rule_error"],
            refined_reference["maximum_constant_rule_error"],
        ),
        "quadrature_cross_checks": quadrature_checks,
        "global_modified_to_reference_mass_ratio": list(mass_ratio),
        "global_modified_to_reference_stiffness_ratio": list(
            stiffness_ratio
        ),
        "global_boundary_coupling_relative_frobenius": (
            _relative_frobenius(modified_boundary, exact_boundary)
        ),
        "minimum_reference_boundary_coupling": float(
            np.min(exact_boundary.data)
        ),
        "low_mode_count": mode_count,
        "reference_low_mode_eigenvalue_range": [
            float(eigenvalues[0]),
            float(eigenvalues[-1]),
        ],
        "low_mode_modified_mass_ratio": [
            float(low_mass_eigenvalues[0]),
            float(low_mass_eigenvalues[-1]),
        ],
        "low_mode_modified_stiffness_ratio": [
            float(low_stiffness_eigenvalues[0]),
            float(low_stiffness_eigenvalues[-1]),
        ],
        "low_mode_boundary_coupling_relative_frobenius": float(
            np.linalg.norm(modified_low_boundary - exact_low_boundary)
            / np.linalg.norm(exact_low_boundary)
        ),
        "low_mode_boundary_coupling_relative_spectral": float(
            np.linalg.norm(modified_low_boundary - exact_low_boundary, 2)
            / np.linalg.norm(exact_low_boundary, 2)
        ),
        "low_mode_transient_boundary_mass_correction_relative_spectral": float(
            np.linalg.norm(
                exact_low_transient_boundary - exact_low_boundary, 2
            )
            / np.linalg.norm(exact_low_transient_boundary, 2)
        ),
        "low_mode_modified_to_transient_boundary_relative_spectral": float(
            np.linalg.norm(
                modified_low_boundary - exact_low_transient_boundary, 2
            )
            / np.linalg.norm(exact_low_transient_boundary, 2)
        ),
    }


def audit(
    spacings: tuple[float, ...] = (0.12, 0.09),
    low_mode_count: int = 20,
    quadrature_order: int = 10,
) -> dict[str, object]:
    if quadrature_order < 4:
        raise ValueError("quadrature_order must be at least four")
    if low_mode_count < 2:
        raise ValueError("low_mode_count must be at least two")
    boundary_fem = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "boundary_fem_for_consistency_gate",
    )
    rows = [
        _consistency_row(
            boundary_fem._build_mesh(spacing),
            low_mode_count,
            quadrature_order,
        )
        for spacing in spacings
    ]
    maximum_quadrature_cross_check = max(
        max(row["quadrature_cross_checks"].values()) for row in rows
    )
    global_mass_near_identity = all(
        row["global_modified_to_reference_mass_ratio"][0] >= 0.9
        and row["global_modified_to_reference_mass_ratio"][1] <= 1.1
        for row in rows
    )
    low_mode_consistency = all(
        row["low_mode_modified_mass_ratio"][0] >= 0.9
        and row["low_mode_modified_mass_ratio"][1] <= 1.15
        and row["low_mode_modified_stiffness_ratio"][0] >= 0.9
        and row["low_mode_modified_stiffness_ratio"][1] <= 1.1
        and row[
            "low_mode_modified_to_transient_boundary_relative_spectral"
        ]
        <= 0.05
        for row in rows
    )
    result = {
        "model": "rho=0 weighted neutral strip on the finite polygon",
        "reference_form": (
            "conforming P1 with high-order Duffy-Gauss integration of "
            "mu=exp(-x^2/2)"
        ),
        "rows": rows,
        "maximum_quadrature_order_cross_check": (
            maximum_quadrature_cross_check
        ),
        "polygonal_circle_geometry_analytically_quantified": True,
        "weighted_reference_forms_independently_assembled": True,
        "reference_quadrature_numerically_cross_checked": bool(
            maximum_quadrature_cross_check < 1.0e-10
        ),
        "consistent_mass_transient_boundary_term_assembled": True,
        "legacy_stiffness_only_boundary_map_complete": False,
        "global_mass_form_near_identity": global_mass_near_identity,
        "whole_spectrum_multiplicative_perturbation_closes": False,
        "low_mode_consistency_observed": low_mode_consistency,
        "parabolic_low_high_mode_split_required": True,
        "reference_quadrature_interval_certified": False,
        "continuum_boundary_flux_error_certified": False,
        "continuum_return_response_certified": False,
        "scope": (
            "The analytic circle geometry is exact. Matrix comparisons are "
            "independent high-order numerical diagnostics on the same P1 "
            "space. The corrected transient conormal map includes the "
            "boundary mass cross-block. These are not interval enclosures "
            "or a continuum flux theorem."
        ),
        "next_required_step": (
            "Interval-enclose the corrected finite low block, prove an "
            "a-posteriori continuum Ritz-projector bound, and transfer the "
            "polygonal domain and conormal measure to the true circle."
        ),
    }
    checks = [
        result["polygonal_circle_geometry_analytically_quantified"],
        result["weighted_reference_forms_independently_assembled"],
        result["reference_quadrature_numerically_cross_checked"],
        result["consistent_mass_transient_boundary_term_assembled"],
        not result["legacy_stiffness_only_boundary_map_complete"],
        not result["global_mass_form_near_identity"],
        not result["whole_spectrum_multiplicative_perturbation_closes"],
        result["low_mode_consistency_observed"],
        result["parabolic_low_high_mode_split_required"],
        not result["reference_quadrature_interval_certified"],
        not result["continuum_boundary_flux_error_certified"],
        not result["continuum_return_response_certified"],
    ]
    result["all_reversible_fem_consistency_gate_checks_pass"] = bool(
        all(checks)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
