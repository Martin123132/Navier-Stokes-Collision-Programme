"""Finite-state pilot for the two neutral-strip stopping branches."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu


TARGET_RADIUS = 1.0
ENTRY_RADIUS = 2.0
STRIP_HALF_WIDTH = 2.1
DEFAULT_X_HALF_WIDTH = 4.2
H1_ENTRY_GAIN = 1.1456141449982071
CUBIC_SPLIT_FACTOR = 0.6392926080189678


def _build_generator(
    y_intervals: int,
    rho: float,
    x_half_width: float = DEFAULT_X_HALF_WIDTH,
    fit_inner_boundary: bool = True,
    strip_half_width: float = STRIP_HALF_WIDTH,
) -> dict[str, object]:
    spacing = 2.0 * strip_half_width / y_intervals
    x_intervals = int(round(2.0 * x_half_width / spacing))
    x_half_width = x_intervals * spacing / 2.0
    xs = np.linspace(-x_half_width, x_half_width, x_intervals + 1)
    ys = np.linspace(
        -strip_half_width,
        strip_half_width,
        y_intervals + 1,
    )

    node_lookup: dict[tuple[int, int], int] = {}
    coordinates: list[tuple[float, float]] = []
    for x_index, x_value in enumerate(xs):
        for y_index, y_value in enumerate(ys):
            if (
                abs(y_value) < strip_half_width - 1.0e-12
                and x_value**2 + y_value**2
                > TARGET_RADIUS**2 + 1.0e-12
            ):
                node_lookup[(x_index, y_index)] = len(coordinates)
                coordinates.append((float(x_value), float(y_value)))

    row_indices: list[int] = []
    column_indices: list[int] = []
    entries: list[float] = []
    inner_rates = np.zeros(len(coordinates))
    wall_rates = np.zeros(len(coordinates))
    inner_boundary_edges: list[tuple[int, float, float, float]] = []
    wall_boundary_edges: list[tuple[int, float, float, float]] = []
    inverse_spacing_squared = 1.0 / spacing**2

    def classify_neighbor(
        x_index: int,
        y_index: int,
        x_step: int,
        y_step: int,
    ) -> tuple[str, int | None, float]:
        neighbor_x_index = x_index + x_step
        neighbor_y_index = y_index + y_step
        if neighbor_x_index < 0 or neighbor_x_index > x_intervals:
            return "reflect", None, spacing
        if (
            neighbor_y_index < 0
            or neighbor_y_index > y_intervals
            or abs(float(ys[neighbor_y_index]))
            >= strip_half_width - 1.0e-12
        ):
            return "wall", None, spacing
        neighbor_x = float(xs[neighbor_x_index])
        neighbor_y = float(ys[neighbor_y_index])
        if (
            neighbor_x**2 + neighbor_y**2
            <= TARGET_RADIUS**2 + 1.0e-12
        ):
            distance = spacing
            if fit_inner_boundary:
                x_value = float(xs[x_index])
                y_value = float(ys[y_index])
                directional_projection = x_value * x_step + y_value * y_step
                radius_squared = x_value**2 + y_value**2
                discriminant = (
                    directional_projection**2
                    - (radius_squared - TARGET_RADIUS**2)
                )
                roots = (
                    -directional_projection - math.sqrt(max(discriminant, 0.0)),
                    -directional_projection + math.sqrt(max(discriminant, 0.0)),
                )
                positive_roots = [root for root in roots if root > 1.0e-13]
                if not positive_roots:
                    raise RuntimeError("circle intersection was not forward")
                distance = min(positive_roots)
                if distance > spacing * (1.0 + 1.0e-10):
                    raise RuntimeError("circle intersection exceeds grid step")
            return "inner", None, distance
        return "interior", node_lookup[(neighbor_x_index, neighbor_y_index)], spacing

    for (x_index, y_index), node in node_lookup.items():
        x_value = float(xs[x_index])
        y_value = float(ys[y_index])
        drift_x = -x_value
        drift_y = -rho * y_value
        total_rate = 0.0
        for plus_step, minus_step, drift in (
            ((1, 0), (-1, 0), drift_x),
            ((0, 1), (0, -1), drift_y),
        ):
            plus = classify_neighbor(
                x_index, y_index, plus_step[0], plus_step[1]
            )
            minus = classify_neighbor(
                x_index, y_index, minus_step[0], minus_step[1]
            )
            if plus[0] == "reflect" or minus[0] == "reflect":
                # The artificial x boundary is far into the inward OU tail.
                # Omitting its outward jump gives a reflecting finite-state
                # approximation whose width dependence is audited below.
                active = minus if plus[0] == "reflect" else plus
                active_step = (
                    minus_step if plus[0] == "reflect" else plus_step
                )
                sign = -1.0 if plus[0] == "reflect" else 1.0
                rate = inverse_spacing_squared + max(sign * drift, 0.0) / spacing
                directional_rows = ((active, rate, active_step),)
            else:
                plus_distance = plus[2]
                minus_distance = minus[2]
                distance_sum = plus_distance + minus_distance
                plus_rate = (
                    2.0 / (plus_distance * distance_sum)
                    + drift
                    * minus_distance
                    / (plus_distance * distance_sum)
                )
                minus_rate = (
                    2.0 / (minus_distance * distance_sum)
                    - drift
                    * plus_distance
                    / (minus_distance * distance_sum)
                )
                directional_rows = (
                    (plus, plus_rate, plus_step),
                    (minus, minus_rate, minus_step),
                )
            for target, rate, step in directional_rows:
                if rate <= 0.0:
                    raise ValueError(
                        "grid is too coarse for monotone unequal-step rates"
                    )
                if target[0] == "inner":
                    inner_rates[node] += rate
                    hit_x = x_value + step[0] * target[2]
                    hit_y = y_value + step[1] * target[2]
                    inner_boundary_edges.append(
                        (node, rate, hit_x, hit_y)
                    )
                elif target[0] == "wall":
                    wall_rates[node] += rate
                    hit_x = x_value + step[0] * target[2]
                    hit_y = y_value + step[1] * target[2]
                    wall_boundary_edges.append(
                        (node, rate, hit_x, hit_y)
                    )
                elif target[0] == "interior":
                    row_indices.append(node)
                    column_indices.append(target[1])
                    entries.append(rate)
                else:
                    raise RuntimeError("unexpected reflected target")
                total_rate += rate
        row_indices.append(node)
        column_indices.append(node)
        entries.append(-total_rate)

    generator = coo_matrix(
        (entries, (row_indices, column_indices)),
        shape=(len(coordinates), len(coordinates)),
    ).tocsc()
    return {
        "generator": generator,
        "inner_rates": inner_rates,
        "wall_rates": wall_rates,
        "inner_boundary_edges": inner_boundary_edges,
        "wall_boundary_edges": wall_boundary_edges,
        "xs": xs,
        "ys": ys,
        "node_lookup": node_lookup,
        "coordinates": np.asarray(coordinates),
        "spacing": spacing,
        "x_half_width": x_half_width,
        "strip_half_width": strip_half_width,
        "inner_boundary_scheme": (
            "coordinate-line-fitted Shortley-Weller"
            if fit_inner_boundary
            else "staircase neighbor-center"
        ),
    }


def _corner_value(
    values: np.ndarray,
    branch_index: int,
    x_index: int,
    y_index: int,
    grid: dict[str, object],
) -> float:
    lookup = grid["node_lookup"]
    if (x_index, y_index) in lookup:
        return float(values[lookup[(x_index, y_index)], branch_index])
    xs = grid["xs"]
    ys = grid["ys"]
    x_value = float(xs[x_index])
    y_value = float(ys[y_index])
    if x_value**2 + y_value**2 <= TARGET_RADIUS**2 + 1.0e-12:
        return 1.0 if branch_index == 0 else 0.0
    if abs(y_value) >= grid["strip_half_width"] - 1.0e-12:
        return 1.0 if branch_index == 1 else 0.0
    raise RuntimeError("unclassified interpolation corner")


def _interpolate_entry(
    values: np.ndarray,
    angle: float,
    grid: dict[str, object],
) -> np.ndarray:
    xs = grid["xs"]
    ys = grid["ys"]
    x_value = ENTRY_RADIUS * math.cos(angle)
    y_value = ENTRY_RADIUS * math.sin(angle)
    x_left = int(np.searchsorted(xs, x_value, side="right") - 1)
    y_bottom = int(np.searchsorted(ys, y_value, side="right") - 1)
    x_left = min(max(x_left, 0), len(xs) - 2)
    y_bottom = min(max(y_bottom, 0), len(ys) - 2)
    x_fraction = (x_value - xs[x_left]) / (xs[x_left + 1] - xs[x_left])
    y_fraction = (y_value - ys[y_bottom]) / (
        ys[y_bottom + 1] - ys[y_bottom]
    )
    result = np.zeros(2)
    for x_offset, x_weight in (
        (0, 1.0 - x_fraction),
        (1, x_fraction),
    ):
        for y_offset, y_weight in (
            (0, 1.0 - y_fraction),
            (1, y_fraction),
        ):
            for branch_index in (0, 1):
                result[branch_index] += x_weight * y_weight * _corner_value(
                    values,
                    branch_index,
                    x_left + x_offset,
                    y_bottom + y_offset,
                    grid,
                )
    return result


def _solve_row(
    y_intervals: int,
    rho: float,
    x_half_width: float = DEFAULT_X_HALF_WIDTH,
    angle_count: int = 64,
    fit_inner_boundary: bool = True,
) -> dict[str, object]:
    grid = _build_generator(
        y_intervals, rho, x_half_width, fit_inner_boundary
    )
    generator = grid["generator"]
    boundary_rates = np.column_stack(
        [grid["inner_rates"], grid["wall_rates"]]
    )
    probability_factor = splu(-generator)
    probabilities = probability_factor.solve(boundary_rates)

    residual_rate = (1.0 - rho) / 2.0
    if residual_rate == 0.0:
        residual_moments = probabilities.copy()
    else:
        moment_matrix = -generator - residual_rate * eye(
            generator.shape[0], format="csc"
        )
        residual_moments = splu(moment_matrix).solve(boundary_rates)

    angle_rows = []
    for angle in np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False):
        probability = _interpolate_entry(probabilities, angle, grid)
        moment = _interpolate_entry(residual_moments, angle, grid)
        probability_criterion = H1_ENTRY_GAIN**2 * (
            probability[0] ** 2
            + (CUBIC_SPLIT_FACTOR * probability[1]) ** 2
        )
        residual_criterion = H1_ENTRY_GAIN**2 * (
            moment[0] ** 2
            + (CUBIC_SPLIT_FACTOR * moment[1]) ** 2
        )
        angle_rows.append(
            {
                "angle": float(angle),
                "return_probability": float(probability[0]),
                "wall_probability": float(probability[1]),
                "return_residual_moment": float(moment[0]),
                "wall_residual_moment": float(moment[1]),
                "unweighted_scalar_stress_criterion": float(
                    probability_criterion
                ),
                "residual_scalar_stress_criterion": float(
                    residual_criterion
                ),
            }
        )

    worst_probability = max(
        angle_rows,
        key=lambda row: row["unweighted_scalar_stress_criterion"],
    )
    worst_residual = max(
        angle_rows,
        key=lambda row: row["residual_scalar_stress_criterion"],
    )
    top_entry = min(
        angle_rows, key=lambda row: abs(row["angle"] - math.pi / 2.0)
    )
    axis_entry = min(angle_rows, key=lambda row: abs(row["angle"]))
    return {
        "y_intervals": y_intervals,
        "spacing": grid["spacing"],
        "x_half_width": grid["x_half_width"],
        "interior_state_count": generator.shape[0],
        "rho": rho,
        "inner_boundary_scheme": grid["inner_boundary_scheme"],
        "residual_rate": residual_rate,
        "maximum_grid_probability_partition_error": float(
            np.max(np.abs(np.sum(probabilities, axis=1) - 1.0))
        ),
        "minimum_resolvent_value": float(
            min(np.min(probabilities), np.min(residual_moments))
        ),
        "maximum_unweighted_scalar_stress_criterion": worst_probability[
            "unweighted_scalar_stress_criterion"
        ],
        "maximum_unweighted_stress_angle": worst_probability["angle"],
        "maximum_residual_scalar_stress_criterion": worst_residual[
            "residual_scalar_stress_criterion"
        ],
        "maximum_residual_stress_angle": worst_residual["angle"],
        "axis_entry": axis_entry,
        "top_entry": top_entry,
    }


def audit() -> dict[str, object]:
    mesh_intervals = (30, 40, 50)
    rho_values = (0.0, 0.25, 0.5, 0.75, 1.0)
    convergence_rows = [
        _solve_row(intervals, rho)
        for intervals in mesh_intervals
        for rho in rho_values
    ]
    finest_rows = {
        row["rho"]: row
        for row in convergence_rows
        if row["y_intervals"] == mesh_intervals[-1]
    }
    middle_rows = {
        row["rho"]: row
        for row in convergence_rows
        if row["y_intervals"] == mesh_intervals[-2]
    }
    maximum_mesh_change = max(
        abs(
            finest_rows[rho]["maximum_residual_scalar_stress_criterion"]
            - middle_rows[rho][
                "maximum_residual_scalar_stress_criterion"
            ]
        )
        for rho in rho_values
    )

    truncation_rows = [
        _solve_row(40, rho, x_half_width)
        for rho in (0.0, 1.0)
        for x_half_width in (3.15, 4.2, 5.25)
    ]
    truncation_groups = {
        rho: [row for row in truncation_rows if row["rho"] == rho]
        for rho in (0.0, 1.0)
    }
    maximum_x_truncation_spread = max(
        max(
            row["maximum_residual_scalar_stress_criterion"]
            for row in rows
        )
        - min(
            row["maximum_residual_scalar_stress_criterion"]
            for row in rows
        )
        for rows in truncation_groups.values()
    )

    finest_summary = []
    for rho in rho_values:
        row = finest_rows[rho]
        finest_summary.append(
            {
                "rho": rho,
                "maximum_unweighted_scalar_stress_criterion": row[
                    "maximum_unweighted_scalar_stress_criterion"
                ],
                "maximum_residual_scalar_stress_criterion": row[
                    "maximum_residual_scalar_stress_criterion"
                ],
                "maximum_residual_stress_angle": row[
                    "maximum_residual_stress_angle"
                ],
                "axis_entry": row["axis_entry"],
                "top_entry": row["top_entry"],
            }
        )

    boundary_scheme_rows = [
        _solve_row(intervals, 0.0, fit_inner_boundary=fit_boundary)
        for intervals in (40, 60, 80)
        for fit_boundary in (False, True)
    ]

    result: dict[str, object] = {
        "stopped_domain": "Omega_Y={r>1, |y|<2.1}",
        "static_generator": "L_rho=Delta-x*partial_x-rho*y*partial_y",
        "branch_resolvents": (
            "p_j=E[1_j], m_j=E[exp((1-rho)tau/2)1_j]"
        ),
        "finite_state_scheme": (
            "centered monotone nearest-neighbor generator, absorbing inner "
            "circle and y walls, reflecting artificial x truncation"
        ),
        "current_scalar_stress_formula": (
            "g_H^2[m_R^2+(s_cubic*m_S)^2]"
        ),
        "H1_entry_gain": H1_ENTRY_GAIN,
        "cubic_split_factor": CUBIC_SPLIT_FACTOR,
        "convergence_rows": convergence_rows,
        "finest_summary": finest_summary,
        "x_truncation_rows": truncation_rows,
        "inner_boundary_scheme_rows": boundary_scheme_rows,
        "maximum_middle_to_fine_residual_criterion_change": (
            maximum_mesh_change
        ),
        "maximum_x_truncation_residual_criterion_spread": (
            maximum_x_truncation_spread
        ),
        "probability_partition_verified_on_grids": all(
            row["maximum_grid_probability_partition_error"] < 1.0e-10
            for row in convergence_rows + truncation_rows
        ),
        "all_resolvents_nonnegative_on_grids": all(
            row["minimum_resolvent_value"] >= -1.0e-11
            for row in convergence_rows + truncation_rows
        ),
        "raw_all_z_scalar_stress_closes_uniformly": all(
            row["maximum_residual_scalar_stress_criterion"] < 1.0
            for row in finest_rows.values()
        ),
        "boundary_flux_space_time_L2_gains_computed": False,
        "finite_axial_patch_included_in_scalar_return_transform": False,
        "outer_wall_exit_identified_with_physical_true_split": False,
        "full_Navier_Stokes_branch_gate_closed": False,
        "interpretation": (
            "the stopped strip separates the two event masses, but the raw "
            "all-z residual scalar stress fails near the strongly returning "
            "axis. The finite axial patch and the actual dynamic L2 entry "
            "norm are essential, not optional refinements"
        ),
        "scope_guard": (
            "this is a converging finite-state resolvent pilot, not an "
            "interval or finite-element certificate. Its scalar stress "
            "formula is diagnostic: it does not equal the required "
            "space-time boundary-density operator norm, does not include "
            "the finite axial return patch, and assumes wall mass may carry "
            "the cubic split factor"
        ),
        "next_gate": (
            "compute the time-resolved inner and wall boundary fluxes, "
            "compose the inner flux with the exact outward-OU axial patch, "
            "and measure both physical space-time L2 density gains before "
            "revisiting renewal closure"
        ),
    }
    positive_checks = (
        result["probability_partition_verified_on_grids"],
        result["all_resolvents_nonnegative_on_grids"],
        maximum_mesh_change < 0.08,
        maximum_x_truncation_spread < 0.03,
        finest_rows[0.0]["maximum_residual_scalar_stress_criterion"] > 1.2,
        finest_rows[0.0]["top_entry"][
            "residual_scalar_stress_criterion"
        ]
        < 0.7,
        not result["raw_all_z_scalar_stress_closes_uniformly"],
        not result["boundary_flux_space_time_L2_gains_computed"],
        not result["full_Navier_Stokes_branch_gate_closed"],
    )
    result["all_positive_branch_resolvent_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
