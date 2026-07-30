"""Pilot the composite wall-migration-child-return trace density."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.special import erf


PATCH_HALF_HEIGHT = 0.75
TRACE_L4_FORM_CONSTANT = 0.6741481379606137
POTENTIAL_L3_OVER_2_FORCING = 0.7989685513198063
DRIFT_L3_FORCING = 3.072840583265365
POTENTIAL_RELATIVE_FORM = 0.2203290376862308
R_STAR = 0.5
MIGRATION_FACTOR = math.exp(R_STAR / 4.0) * 2.0 ** (-0.75)


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wall_migration_rate_matrix(
    grid: dict[str, object], angle_bins: int
):
    rows = []
    columns = []
    values = []
    for node, rate, hit_x, hit_y in grid["wall_boundary_edges"]:
        child_angle = math.atan2(hit_y, hit_x) % (2.0 * math.pi)
        angle_bin = min(
            int(child_angle * angle_bins / (2.0 * math.pi)),
            angle_bins - 1,
        )
        rows.append(node)
        columns.append(angle_bin)
        values.append(rate)
    return coo_matrix(
        (values, (rows, columns)),
        shape=(grid["generator"].shape[0], angle_bins),
    ).tocsc()


def _entry_state_and_wall_atom(
    grid: dict[str, object], entry_angles: np.ndarray, angle_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    xs = grid["xs"]
    ys = grid["ys"]
    lookup = grid["node_lookup"]
    state = np.zeros((grid["generator"].shape[0], len(entry_angles)))
    wall_atom = np.zeros((len(entry_angles), angle_bins))
    for angle_index, angle in enumerate(entry_angles):
        x_value = 2.0 * math.cos(float(angle))
        y_value = 2.0 * math.sin(float(angle))
        x_left = int(np.searchsorted(xs, x_value, side="right") - 1)
        y_bottom = int(np.searchsorted(ys, y_value, side="right") - 1)
        x_left = min(max(x_left, 0), len(xs) - 2)
        y_bottom = min(max(y_bottom, 0), len(ys) - 2)
        x_fraction = (x_value - xs[x_left]) / (
            xs[x_left + 1] - xs[x_left]
        )
        y_fraction = (y_value - ys[y_bottom]) / (
            ys[y_bottom + 1] - ys[y_bottom]
        )
        for x_offset, x_weight in (
            (0, 1.0 - x_fraction),
            (1, x_fraction),
        ):
            for y_offset, y_weight in (
                (0, 1.0 - y_fraction),
                (1, y_fraction),
            ):
                weight = x_weight * y_weight
                key = (x_left + x_offset, y_bottom + y_offset)
                if key in lookup:
                    state[lookup[key], angle_index] += weight
                    continue
                corner_x = float(xs[key[0]])
                corner_y = float(ys[key[1]])
                if (
                    abs(corner_y)
                    >= grid["strip_half_width"] - 1.0e-12
                ):
                    child_angle = math.atan2(corner_y, corner_x) % (
                        2.0 * math.pi
                    )
                    angle_bin = min(
                        int(child_angle * angle_bins / (2.0 * math.pi)),
                        angle_bins - 1,
                    )
                    wall_atom[angle_index, angle_bin] += weight
    return state, wall_atom


def _uniform_flux_trajectory(
    generator,
    initial_state: np.ndarray,
    boundary_rates,
    maximum_time: float,
    time_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    point_count = int(round(maximum_time / time_step)) + 1
    times = np.linspace(0.0, maximum_time, point_count)
    trajectory = expm_multiply(
        generator.transpose().tocsc(),
        initial_state,
        start=0.0,
        stop=maximum_time,
        num=point_count,
        endpoint=True,
    )
    flux = np.asarray(
        [
            np.asarray(boundary_rates.transpose() @ snapshot).T
            for snapshot in trajectory
        ]
    )
    return times, flux


def _axial_patch_factors(variance: float) -> tuple[float, float]:
    if variance <= 0.0:
        return 0.0, 0.0
    patch_mass = float(
        erf(PATCH_HALF_HEIGHT / math.sqrt(2.0 * variance))
    )
    squared_density_mass = (
        float(erf(PATCH_HALF_HEIGHT / math.sqrt(variance)))
        / (2.0 * math.sqrt(math.pi) * math.sqrt(variance))
    )
    return patch_mass, math.sqrt(squared_density_mass)


def _compose(
    wall_times: np.ndarray,
    wall_flux: np.ndarray,
    wall_atom: np.ndarray,
    return_times: np.ndarray,
    return_flux: np.ndarray,
    angle_bins: int,
) -> dict[str, object]:
    wall_step = float(wall_times[1] - wall_times[0])
    return_step = float(return_times[1] - return_times[0])
    child_time_step = return_step
    if abs(4.0 * wall_step - child_time_step) > 1.0e-12:
        raise ValueError("wall and child time grids are not scale aligned")
    total_count = len(wall_times) + len(return_times) - 1
    total_times = np.arange(total_count, dtype=float) * child_time_step
    entry_count = wall_flux.shape[1]
    raw_l2_upper = np.zeros((total_count, entry_count))
    scalar_density = np.zeros((total_count, entry_count))
    angular_bin_width = 2.0 * math.pi / angle_bins

    # The wall component of bilinear entry data is a time-zero atom. One
    # child return immediately smooths it before the H1 trace is evaluated.
    for child_index in range(1, len(return_times)):
        child_time = float(return_times[child_index])
        angular = wall_atom @ return_flux[child_index]
        angular_l2 = np.sqrt(
            np.sum(angular**2, axis=1) / angular_bin_width
        )
        angular_mass = np.sum(angular, axis=1)
        variance = math.expm1(2.0 * child_time)
        patch_mass, axial_l2 = _axial_patch_factors(variance)
        deformation = MIGRATION_FACTOR * math.exp(child_time)
        raw_l2_upper[child_index] += (
            deformation * axial_l2 * angular_l2
        )
        scalar_density[child_index] += (
            deformation * patch_mass * angular_mass
        )

    wall_quadrature = np.full(len(wall_times), wall_step)
    wall_quadrature[[0, -1]] *= 0.5
    for wall_index, wall_time in enumerate(wall_times):
        if not np.any(wall_flux[wall_index]):
            continue
        first_variance = math.expm1(2.0 * float(wall_time))
        for child_index in range(1, len(return_times)):
            child_time = float(return_times[child_index])
            total_index = wall_index + child_index
            angular = wall_flux[wall_index] @ return_flux[child_index]
            angular_l2 = np.sqrt(
                np.sum(angular**2, axis=1) / angular_bin_width
            )
            angular_mass = np.sum(angular, axis=1)
            variance = (
                4.0 * math.exp(2.0 * child_time) * first_variance
                + math.expm1(2.0 * child_time)
            )
            patch_mass, axial_l2 = _axial_patch_factors(variance)
            deformation = MIGRATION_FACTOR * math.exp(
                float(wall_time) + child_time
            )
            weight = wall_quadrature[wall_index]
            raw_l2_upper[total_index] += (
                weight * deformation * axial_l2 * angular_l2
            )
            scalar_density[total_index] += (
                weight * deformation * patch_mass * angular_mass
            )

    scalar_gain = np.trapezoid(
        scalar_density, total_times, axis=0
    )
    return {
        "times": total_times,
        "raw_l2_upper": raw_l2_upper,
        "scalar_density": scalar_density,
        "scalar_gain": scalar_gain,
    }


def _row(
    resolvent,
    return_density,
    y_intervals: int,
    angle_bins: int = 16,
) -> dict[str, object]:
    grid = resolvent._build_generator(y_intervals, 0.0)
    entry_angles = np.linspace(
        0.0, 2.0 * math.pi, angle_bins, endpoint=False
    )
    initial_state, wall_atom = _entry_state_and_wall_atom(
        grid, entry_angles, angle_bins
    )
    wall_rates = _wall_migration_rate_matrix(grid, angle_bins)
    return_rates = return_density._inner_rate_matrix(grid, angle_bins)
    wall_times, wall_flux = _uniform_flux_trajectory(
        grid["generator"], initial_state, wall_rates, 4.0, 0.025
    )
    child_initial = return_density._entry_matrix(grid, entry_angles)
    return_times, child_return_flux = _uniform_flux_trajectory(
        grid["generator"], child_initial, return_rates, 8.0, 0.1
    )
    composite = _compose(
        wall_times,
        wall_flux,
        wall_atom,
        return_times,
        child_return_flux,
        angle_bins,
    )

    angle_rows = []
    for angle_index, angle in enumerate(entry_angles):
        interval = return_density._sampled_interval_factor(
            composite["times"][1:],
            composite["raw_l2_upper"][1:, angle_index],
        )
        scalar_gain = float(composite["scalar_gain"][angle_index])
        response = math.sqrt(
            scalar_gain
            * TRACE_L4_FORM_CONSTANT
            * interval["sampled_stressed_factor"]
        )
        angle_rows.append(
            {
                "angle": float(angle),
                "composite_scalar_gain": scalar_gain,
                "wall_atom_mass": float(np.sum(wall_atom[angle_index])),
                "peak_raw_spatial_L2_upper": float(
                    np.max(composite["raw_l2_upper"][:, angle_index])
                ),
                "peak_child_time": float(
                    composite["times"][
                        int(
                            np.argmax(
                                composite["raw_l2_upper"][:, angle_index]
                            )
                        )
                    ]
                ),
                "raw_interval_factor_upper": interval[
                    "sampled_stressed_factor"
                ],
                "composite_trace_response_at_alpha_zero": response,
                "optimal_sampled_window": interval[
                    "optimal_sampled_window"
                ],
                "fitted_stressed_tail_decay": interval[
                    "fitted_stressed_tail_decay"
                ],
            }
        )
    return {
        "y_intervals": y_intervals,
        "spacing": grid["spacing"],
        "state_count": grid["generator"].shape[0],
        "angle_bin_count": angle_bins,
        "wall_time_step_parent_units": 0.025,
        "child_time_step_child_units": 0.1,
        "maximum_child_time": float(composite["times"][-1]),
        "angle_rows": angle_rows,
        "maximum_composite_scalar_gain": max(
            row["composite_scalar_gain"] for row in angle_rows
        ),
        "maximum_raw_interval_factor_upper": max(
            row["raw_interval_factor_upper"] for row in angle_rows
        ),
        "maximum_composite_trace_response_at_alpha_zero": max(
            row["composite_trace_response_at_alpha_zero"]
            for row in angle_rows
        ),
    }


def _threshold_rows(
    composite_row: dict[str, object], residual_budget: dict[str, object]
) -> list[dict[str, float]]:
    rows = []
    for composite_angle in composite_row["angle_rows"]:
        source = min(
            residual_budget["angle_rows"],
            key=lambda row: abs(row["angle"] - composite_angle["angle"]),
        )
        response_zero = composite_angle[
            "composite_trace_response_at_alpha_zero"
        ]
        wall_allowance = source["wall_only_additive_gain_allowance"]
        drift_threshold = wall_allowance / (
            response_zero * DRIFT_L3_FORCING
        )

        def potential_gate(potential_mass: float) -> float:
            alpha = POTENTIAL_RELATIVE_FORM * potential_mass
            response = (
                response_zero
                * POTENTIAL_L3_OVER_2_FORCING
                * potential_mass
                / (1.0 - alpha) ** 1.5
            )
            return (
                source["return_one_history_gain"] ** 2
                + (source["wall_one_history_gain"] + response) ** 2
                - 1.0
            )

        potential_threshold = brentq(
            potential_gate,
            0.0,
            0.999 / POTENTIAL_RELATIVE_FORM,
        )
        rows.append(
            {
                "angle": composite_angle["angle"],
                "conditional_wall_core_potential_L3_over_2_threshold": (
                    potential_threshold
                ),
                "conditional_wall_core_drift_L3_threshold": drift_threshold,
            }
        )
    return rows


def audit() -> dict[str, object]:
    resolvent = _load_module(
        "neutral_strip_branch_resolvent_pilot.py",
        "resolvent_for_wall_child_composite",
    )
    return_density = _load_module(
        "neutral_strip_return_density_pilot.py",
        "return_density_for_wall_child_composite",
    )
    residual = _load_module(
        "migrating_core_residual_budget_audit.py",
        "residual_for_wall_child_composite",
    )
    mesh_rows = [
        _row(resolvent, return_density, intervals)
        for intervals in (24, 30, 36)
    ]
    working = mesh_rows[-1]
    residual_budget = residual._numerical_budget()
    threshold_rows = _threshold_rows(working, residual_budget)
    potential_worst = min(
        threshold_rows,
        key=lambda row: row[
            "conditional_wall_core_potential_L3_over_2_threshold"
        ],
    )
    drift_worst = min(
        threshold_rows,
        key=lambda row: row[
            "conditional_wall_core_drift_L3_threshold"
        ],
    )
    result = {
        "composite_kernel": (
            "B_S_core=B_wall*M_migrate*B_child_return"
        ),
        "child_normalized_time": "u=4*t_wall+t_child",
        "child_axial_initial_coordinate": "z_child=2*z_wall",
        "composite_axial_variance": (
            "4*exp(2*t_child)*(exp(2*t_wall)-1)"
            "+exp(2*t_child)-1"
        ),
        "migration_factor": MIGRATION_FACTOR,
        "L2_wall_time_mixture_bound": "Minkowski upper bound",
        "time_zero_wall_atom_propagated_through_child_return": True,
        "mesh_rows": mesh_rows,
        "maximum_mesh_scalar_gain_change": abs(
            mesh_rows[-1]["maximum_composite_scalar_gain"]
            - mesh_rows[-2]["maximum_composite_scalar_gain"]
        ),
        "maximum_mesh_interval_factor_change": abs(
            mesh_rows[-1]["maximum_raw_interval_factor_upper"]
            - mesh_rows[-2]["maximum_raw_interval_factor_upper"]
        ),
        "maximum_mesh_response_change": abs(
            mesh_rows[-1][
                "maximum_composite_trace_response_at_alpha_zero"
            ]
            - mesh_rows[-2][
                "maximum_composite_trace_response_at_alpha_zero"
            ]
        ),
        "threshold_rows": threshold_rows,
        "conditional_minimum_wall_core_potential_L3_over_2_threshold": (
            potential_worst[
                "conditional_wall_core_potential_L3_over_2_threshold"
            ]
        ),
        "conditional_worst_potential_angle": potential_worst["angle"],
        "conditional_minimum_wall_core_drift_L3_threshold": drift_worst[
            "conditional_wall_core_drift_L3_threshold"
        ],
        "conditional_worst_drift_angle": drift_worst["angle"],
        "finite_state_composite_K_S_computed": True,
        "continuum_composite_K_S_certified": False,
        "nonaffine_Navier_Stokes_composite_K_S_certified": False,
        "full_wall_stopping_trace_gate_closed": False,
        "scope_guard": (
            "the scale/time map, Gaussian variance, positive semigroup "
            "composition, and Minkowski direction are exact. Boundary/time "
            "binning, finite horizons, fitted tails, and mesh convergence "
            "are pilots. The calculation covers the static rho=0 model, "
            "uses one child return, and does not certify continuum or "
            "Navier-Stokes coefficients"
        ),
        "next_gate": (
            "certify the two component flux envelopes and their convolution "
            "uniformly in rho, then insert actual migration q_res; no further "
            "stage is started in this session"
        ),
    }
    checks = (
        0.67 < result["migration_factor"] < 0.68,
        result["time_zero_wall_atom_propagated_through_child_return"],
        working["maximum_composite_scalar_gain"] > 0.0,
        working["maximum_raw_interval_factor_upper"] > 0.0,
        working["maximum_composite_trace_response_at_alpha_zero"] > 0.0,
        result["conditional_minimum_wall_core_potential_L3_over_2_threshold"]
        > 0.0,
        result["conditional_minimum_wall_core_drift_L3_threshold"] > 0.0,
        result["finite_state_composite_K_S_computed"],
        result["maximum_mesh_scalar_gain_change"] < 0.02,
        result["maximum_mesh_interval_factor_change"] < 0.03,
        result["maximum_mesh_response_change"] < 0.005,
        not result["continuum_composite_K_S_certified"],
        not result["nonaffine_Navier_Stokes_composite_K_S_certified"],
        not result["full_wall_stopping_trace_gate_closed"],
    )
    result["all_positive_wall_child_composite_checks_pass"] = all(checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
