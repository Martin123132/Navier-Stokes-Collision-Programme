"""Stationary axisymmetric finite-element pilot for the collar trace norm."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


OUTER_RADIUS = 2.0
HALF_HEIGHT = 0.75


def _load_cutoff_module():
    script = Path(__file__).resolve().with_name(
        "radial_barrier_cutoff_energy_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "radial_cutoff_for_collar_trace", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _local_matrices(
    radial_left: float,
    axial_bottom: float,
    dr: float,
    dz: float,
    transverse_strain: float,
) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(3)
    operator = np.zeros((4, 4))
    mass = np.zeros((4, 4))
    signs = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
    jacobian = dr * dz / 4.0
    for xi, weight_xi in zip(points, weights):
        for eta, weight_eta in zip(points, weights):
            shapes = np.array(
                [0.25 * (1.0 + sx * xi) * (1.0 + sz * eta) for sx, sz in signs]
            )
            radial_derivatives = np.array(
                [0.5 * sx * (1.0 + sz * eta) / dr for sx, sz in signs]
            )
            axial_derivatives = np.array(
                [0.5 * sz * (1.0 + sx * xi) / dz for sx, sz in signs]
            )
            radial = radial_left + 0.5 * dr * (1.0 + xi)
            axial = axial_bottom + 0.5 * dz * (1.0 + eta)
            drift = np.array(
                [
                    transverse_strain * radial,
                    -2.0 * transverse_strain * axial,
                ]
            )
            trial_advection = (
                drift[0] * radial_derivatives
                + drift[1] * axial_derivatives
            )
            quadrature_weight = (
                4.0
                * math.pi
                * radial
                * jacobian
                * weight_xi
                * weight_eta
            )
            local_mass = quadrature_weight * np.outer(shapes, shapes)
            mass += local_mass
            operator += quadrature_weight * (
                np.outer(radial_derivatives, radial_derivatives)
                + np.outer(axial_derivatives, axial_derivatives)
                - np.outer(shapes, trial_advection)
                - np.outer(shapes, shapes)
            )
    return operator, mass


def _assemble_collar(
    radial_cells: int,
    axial_cells: int,
    collar_distance: float,
    transverse_strain: float,
):
    radial_grid = np.linspace(0.0, OUTER_RADIUS, radial_cells + 1)
    axial_grid = np.linspace(0.0, HALF_HEIGHT, axial_cells + 1)
    dr = OUTER_RADIUS / radial_cells
    dz = HALF_HEIGHT / axial_cells
    radial_support = 1.0 - collar_distance
    axial_support = HALF_HEIGHT - collar_distance
    rows: list[int] = []
    columns: list[int] = []
    operator_data: list[float] = []
    mass_data: list[float] = []

    def node(radial_index: int, axial_index: int) -> int:
        return radial_index * (axial_cells + 1) + axial_index

    for radial_index in range(radial_cells):
        radial_right = radial_grid[radial_index + 1]
        for axial_index in range(axial_cells):
            axial_top = axial_grid[axial_index + 1]
            if (
                radial_right <= radial_support + 1.0e-13
                and axial_top <= axial_support + 1.0e-13
            ):
                continue
            local_operator, local_mass = _local_matrices(
                radial_grid[radial_index],
                axial_grid[axial_index],
                dr,
                dz,
                transverse_strain,
            )
            indices = (
                node(radial_index, axial_index),
                node(radial_index + 1, axial_index),
                node(radial_index, axial_index + 1),
                node(radial_index + 1, axial_index + 1),
            )
            for local_row, global_row in enumerate(indices):
                for local_column, global_column in enumerate(indices):
                    rows.append(global_row)
                    columns.append(global_column)
                    operator_data.append(
                        local_operator[local_row, local_column]
                    )
                    mass_data.append(local_mass[local_row, local_column])

    node_count = (radial_cells + 1) * (axial_cells + 1)
    operator = coo_matrix(
        (operator_data, (rows, columns)),
        shape=(node_count, node_count),
    ).tocsc()
    mass = coo_matrix(
        (mass_data, (rows, columns)),
        shape=(node_count, node_count),
    ).tocsc()
    return operator, mass, radial_grid, axial_grid


def _trace_row(
    radial_cells: int,
    axial_cells: int,
    collar_distance: float,
    transverse_strain: float,
    cutoff_factor: float,
) -> dict[str, float | int | bool]:
    operator, mass, radial_grid, axial_grid = _assemble_collar(
        radial_cells,
        axial_cells,
        collar_distance,
        transverse_strain,
    )
    radial_mesh, axial_mesh = np.meshgrid(
        radial_grid, axial_grid, indexing="ij"
    )
    radial_support = 1.0 - collar_distance
    axial_support = HALF_HEIGHT - collar_distance
    tolerance = 1.0e-12
    strict_core = (
        (radial_mesh < radial_support - tolerance)
        & (axial_mesh < axial_support - tolerance)
    )
    interface = (
        (
            np.isclose(radial_mesh, radial_support, atol=tolerance)
            & (axial_mesh <= axial_support + tolerance)
        )
        | (
            np.isclose(axial_mesh, axial_support, atol=tolerance)
            & (radial_mesh <= radial_support + tolerance)
        )
    )
    absorbing_boundary = np.zeros_like(strict_core)
    absorbing_boundary[-1, :] = True
    absorbing_boundary[:, -1] = True
    collar_unknown = ~(strict_core | absorbing_boundary)
    free = np.flatnonzero((collar_unknown & ~interface).ravel())
    boundary = np.flatnonzero(interface.ravel())
    unknown = np.concatenate([free, boundary])

    free_operator = operator[free][:, free]
    interface_operator = operator[free][:, boundary]
    factorization = splu(free_operator)
    free_poisson = factorization.solve(-interface_operator.toarray())
    poisson = np.vstack(
        [free_poisson, np.eye(len(boundary), dtype=float)]
    )
    collar_mass = mass[unknown][:, unknown]
    boundary_gram = poisson.T @ (collar_mass @ poisson)

    radial_point = int(round(1.0 / (OUTER_RADIUS / radial_cells)))
    point_global = radial_point * (axial_cells + 1)
    point_position = int(np.flatnonzero(free == point_global)[0])
    evaluation = free_poisson[point_position]
    riesz = solve(
        boundary_gram,
        evaluation,
        assume_a="pos",
        check_finite=False,
    )
    trace_norm_squared = float(evaluation @ riesz)
    trace_norm = math.sqrt(trace_norm_squared)
    optimizing_boundary = riesz / trace_norm_squared
    optimizing_solution = poisson @ optimizing_boundary
    return {
        "radial_cells": radial_cells,
        "axial_cells": axial_cells,
        "collar_distance": collar_distance,
        "transverse_strain": transverse_strain,
        "free_collar_unknown_count": len(free),
        "interface_unknown_count": len(boundary),
        "stationary_axisymmetric_L2_to_point_trace_norm": trace_norm,
        "cutoff_factor": cutoff_factor,
        "stationary_axisymmetric_chi_pilot": trace_norm * cutoff_factor,
        "optimizer_boundary_minimum": float(np.min(optimizing_boundary)),
        "optimizer_solution_minimum": float(np.min(optimizing_solution)),
        "optimizer_is_nonnegative_to_tolerance": bool(
            np.min(optimizing_solution) > -1.0e-8
        ),
    }


def audit() -> dict[str, object]:
    cutoff = _load_cutoff_module().audit()
    cutoff_factors = {
        row["collar_distance"]: row[
            "sqrt_energy_over_m0_over_barrier_gain"
        ]
        for row in cutoff["fine_rows"]
    }
    mesh_specs = ((80, 60), (120, 90))
    distances = (0.10, 0.20, 0.30, 0.40)
    transverse_strains = (-1.0, 0.0, 0.5)
    rows = []
    for radial_cells, axial_cells in mesh_specs:
        for distance in distances:
            for transverse_strain in transverse_strains:
                rows.append(
                    _trace_row(
                        radial_cells,
                        axial_cells,
                        distance,
                        transverse_strain,
                        cutoff_factors[distance],
                    )
                )

    fine_rows = [
        row for row in rows if row["radial_cells"] == mesh_specs[-1][0]
    ]
    convergence_rows = []
    for distance in distances:
        for transverse_strain in transverse_strains:
            pair = [
                row
                for row in rows
                if row["collar_distance"] == distance
                and row["transverse_strain"] == transverse_strain
            ]
            coarse, fine = pair
            convergence_rows.append(
                {
                    "collar_distance": distance,
                    "transverse_strain": transverse_strain,
                    "coarse_trace_norm": coarse[
                        "stationary_axisymmetric_L2_to_point_trace_norm"
                    ],
                    "fine_trace_norm": fine[
                        "stationary_axisymmetric_L2_to_point_trace_norm"
                    ],
                    "relative_change": abs(
                        fine[
                            "stationary_axisymmetric_L2_to_point_trace_norm"
                        ]
                        - coarse[
                            "stationary_axisymmetric_L2_to_point_trace_norm"
                        ]
                    )
                    / fine[
                        "stationary_axisymmetric_L2_to_point_trace_norm"
                    ],
                }
            )

    worst_by_distance = []
    for distance in distances:
        candidates = [
            row for row in fine_rows if row["collar_distance"] == distance
        ]
        worst_by_distance.append(
            max(
                candidates,
                key=lambda row: row["stationary_axisymmetric_chi_pilot"],
            )
        )
    result: dict[str, object] = {
        "status": (
            "stationary axisymmetric finite-element operator pilot; "
            "not an enclosure and not a nonautonomous theorem"
        ),
        "collar_problem": (
            "solve (-Delta-B_a y.grad-1)w=0 on D\\E_d with arbitrary "
            "inner-interface data and zero absorbing-boundary data"
        ),
        "trace_norm": (
            "sup |w(r=1,z=0)|/||w||_(L2(D\\E_d)) over the discrete "
            "stationary collar solution space"
        ),
        "axisymmetric_affine_family": (
            "B_a=diag(a,a,-2a), -1<=a<=1/2"
        ),
        "mesh_rows": rows,
        "fine_rows": fine_rows,
        "convergence_rows": convergence_rows,
        "worst_fine_row_by_distance": worst_by_distance,
        "all_coarse_fine_changes_below_two_percent": all(
            row["relative_change"] < 0.02 for row in convergence_rows
        ),
        "all_discrete_optimizers_nonnegative": all(
            row["optimizer_is_nonnegative_to_tolerance"] for row in rows
        ),
        "all_sampled_stationary_chi_below_two": all(
            row["stationary_axisymmetric_chi_pilot"] < 2.0
            for row in fine_rows
        ),
        "all_sampled_d0p2_or_wider_stationary_chi_below_two": all(
            row["stationary_axisymmetric_chi_pilot"] < 2.0
            for row in fine_rows
            if row["collar_distance"] >= 0.20
        ),
        "scope_guard": (
            "the unrestricted interface norm is an upper benchmark for "
            "stationary axisymmetric nonnegative collar solutions, but "
            "measurable time dependence, non-axisymmetric affine controls, "
            "finite-element enclosure, and physical support localization "
            "remain open"
        ),
    }
    positive_checks = (
        result["all_coarse_fine_changes_below_two_percent"],
        result[
            "all_sampled_d0p2_or_wider_stationary_chi_below_two"
        ],
        len(worst_by_distance) == len(distances),
        all(
            row["stationary_axisymmetric_L2_to_point_trace_norm"] > 0.0
            for row in rows
        ),
    )
    result["all_positive_stationary_collar_pilot_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
