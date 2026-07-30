"""Axisymmetric finite-element pilot for the optimal barrier cutoff energy."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


OUTER_RADIUS = 2.0
HALF_HEIGHT = 0.75
BASELINE_DECAY = 4.832287335665
BARRIER_GAIN = 1.3428786845671419


def _barrier(radial: np.ndarray, axial: np.ndarray) -> np.ndarray:
    radial_fraction = radial / 2.0
    radial_layer = np.maximum(1.0 - radial_fraction**2, 0.0)
    axial_layer = np.maximum(np.cos(2.0 * math.pi * axial / 3.0), 0.0)
    return (
        0.89945 * radial_fraction**2
        + 0.10055 * radial_fraction**16
        + 1.3479 * radial_layer ** (13.0 / 20.0)
        * axial_layer ** (7.0 / 20.0)
    )


def _local_form(radial_left: float, dr: float, dz: float) -> np.ndarray:
    points, weights = np.polynomial.legendre.leggauss(3)
    local = np.zeros((4, 4))
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
            local += (
                4.0
                * math.pi
                * radial
                * jacobian
                * weight_xi
                * weight_eta
                * (
                    np.outer(radial_derivatives, radial_derivatives)
                    + np.outer(axial_derivatives, axial_derivatives)
                    - np.outer(shapes, shapes)
                )
            )
    return local


def _assemble_form(radial_cells: int, axial_cells: int):
    radial_grid = np.linspace(0.0, OUTER_RADIUS, radial_cells + 1)
    axial_grid = np.linspace(0.0, HALF_HEIGHT, axial_cells + 1)
    dr = OUTER_RADIUS / radial_cells
    dz = HALF_HEIGHT / axial_cells
    entry_count = radial_cells * axial_cells * 16
    rows = np.empty(entry_count, dtype=np.int64)
    columns = np.empty(entry_count, dtype=np.int64)
    data = np.empty(entry_count)
    cursor = 0

    def node(radial_index: int, axial_index: int) -> int:
        return radial_index * (axial_cells + 1) + axial_index

    for radial_index in range(radial_cells):
        local = _local_form(radial_grid[radial_index], dr, dz)
        for axial_index in range(axial_cells):
            indices = np.array(
                [
                    node(radial_index, axial_index),
                    node(radial_index + 1, axial_index),
                    node(radial_index, axial_index + 1),
                    node(radial_index + 1, axial_index + 1),
                ]
            )
            block = slice(cursor, cursor + 16)
            rows[block] = np.repeat(indices, 4)
            columns[block] = np.tile(indices, 4)
            data[block] = local.ravel()
            cursor += 16
    node_count = (radial_cells + 1) * (axial_cells + 1)
    form = coo_matrix(
        (data, (rows, columns)), shape=(node_count, node_count)
    ).tocsr()
    return form, radial_grid, axial_grid


def _minimum_extension_row(
    form,
    radial_grid: np.ndarray,
    axial_grid: np.ndarray,
    collar_distance: float,
) -> dict[str, float | int]:
    radial_cells = len(radial_grid) - 1
    axial_cells = len(axial_grid) - 1
    radial_mesh, axial_mesh = np.meshgrid(
        radial_grid, axial_grid, indexing="ij"
    )
    radial_support = 1.0 - collar_distance
    axial_support = HALF_HEIGHT - collar_distance
    support_mask = (
        (radial_mesh <= radial_support + 1.0e-13)
        & (axial_mesh <= axial_support + 1.0e-13)
    )
    outer_boundary = np.zeros_like(support_mask)
    outer_boundary[-1, :] = True
    outer_boundary[:, -1] = True
    fixed_mask = support_mask | outer_boundary
    fixed_values = np.zeros_like(radial_mesh)
    fixed_values[support_mask] = _barrier(
        radial_mesh[support_mask], axial_mesh[support_mask]
    )

    fixed = np.flatnonzero(fixed_mask.ravel())
    free = np.flatnonzero(~fixed_mask.ravel())
    values = fixed_values.ravel().copy()
    free_form = form[free][:, free]
    coupling = form[free][:, fixed]
    values[free] = spsolve(free_form, -(coupling @ values[fixed]))
    energy = float(values @ (form @ values))
    cutoff_factor = math.sqrt(energy / BASELINE_DECAY) / BARRIER_GAIN
    return {
        "radial_cells": radial_cells,
        "axial_cells": axial_cells,
        "collar_distance": collar_distance,
        "support_radius": radial_support,
        "support_half_height": axial_support,
        "free_unknown_count": len(free),
        "minimum_axisymmetric_cutoff_energy": energy,
        "sqrt_energy_over_m0_over_barrier_gain": cutoff_factor,
        "maximum_C_col_for_chi_one": 1.0 / cutoff_factor,
        "maximum_C_col_for_chi_two": 2.0 / cutoff_factor,
        "solution_minimum": float(np.min(values)),
        "solution_maximum": float(np.max(values)),
    }


def audit() -> dict[str, object]:
    mesh_specs = ((80, 60), (160, 120), (240, 180))
    collar_distances = (0.10, 0.20, 0.30, 0.40)
    rows = []
    for radial_cells, axial_cells in mesh_specs:
        form, radial_grid, axial_grid = _assemble_form(
            radial_cells, axial_cells
        )
        for distance in collar_distances:
            rows.append(
                _minimum_extension_row(
                    form, radial_grid, axial_grid, distance
                )
            )

    convergence = []
    for distance in collar_distances:
        distance_rows = [
            row for row in rows if row["collar_distance"] == distance
        ]
        coarse, medium, fine = distance_rows
        convergence.append(
            {
                "collar_distance": distance,
                "coarse_energy": coarse[
                    "minimum_axisymmetric_cutoff_energy"
                ],
                "medium_energy": medium[
                    "minimum_axisymmetric_cutoff_energy"
                ],
                "fine_energy": fine[
                    "minimum_axisymmetric_cutoff_energy"
                ],
                "medium_to_fine_relative_change": abs(
                    fine["minimum_axisymmetric_cutoff_energy"]
                    - medium["minimum_axisymmetric_cutoff_energy"]
                )
                / fine["minimum_axisymmetric_cutoff_energy"],
                "fine_cutoff_factor": fine[
                    "sqrt_energy_over_m0_over_barrier_gain"
                ],
                "fine_maximum_C_col_for_chi_two": fine[
                    "maximum_C_col_for_chi_two"
                ],
            }
        )

    fine_rows = [
        row for row in rows if row["radial_cells"] == mesh_specs[-1][0]
    ]
    result: dict[str, object] = {
        "status": "axisymmetric finite-element pilot; not an enclosure",
        "variational_problem": (
            "minimize h[v]=int_D(|grad v|^2-v^2) over even axisymmetric "
            "v with v=U on E_d and v=0 on the absorbing boundary"
        ),
        "protected_support": (
            "E_d={r<=1-d, |z|<=0.75-d}"
        ),
        "why_this_is_optimal": (
            "every cutoff zeta=1 on E_d gives v=zeta*U in this affine "
            "constraint class, so the minimizer is a lower benchmark for "
            "any explicit cutoff energy"
        ),
        "mesh_rows": rows,
        "convergence_rows": convergence,
        "fine_rows": fine_rows,
        "all_minimizers_nonnegative": all(
            row["solution_minimum"] > -1.0e-10 for row in rows
        ),
        "all_medium_fine_changes_below_one_percent": all(
            row["medium_to_fine_relative_change"] < 0.01
            for row in convergence
        ),
        "energy_decreases_with_larger_collar": all(
            later["minimum_axisymmetric_cutoff_energy"]
            < earlier["minimum_axisymmetric_cutoff_energy"]
            for earlier, later in zip(fine_rows, fine_rows[1:])
        ),
        "scope_guard": (
            "the solve calibrates only the best axisymmetric cutoff energy "
            "for the stated protected supports. It does not certify the "
            "parabolic trace constant, realize the support from a Leray "
            "solution, or prove Navier-Stokes regularity."
        ),
    }
    positive_checks = (
        result["all_minimizers_nonnegative"],
        result["all_medium_fine_changes_below_one_percent"],
        result["energy_decreases_with_larger_collar"],
        len(fine_rows) == len(collar_distances),
    )
    result["all_positive_cutoff_energy_pilot_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
