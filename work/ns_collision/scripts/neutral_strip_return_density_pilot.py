"""Pilot the time-resolved L2 return density of neutral-strip storage."""

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
FORM_FLOOR = 4.832287335665
TRACE_L4_FORM_CONSTANT = 0.6741481379606137
POTENTIAL_L3_OVER_2_FORCING = 0.7989685513198063
DRIFT_L3_FORCING = 3.072840583265365
POTENTIAL_RELATIVE_FORM = 0.2203290376862308


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inner_rate_matrix(
    grid: dict[str, object], boundary_bins: int
):
    rows = []
    columns = []
    values = []
    for node, rate, hit_x, hit_y in grid["inner_boundary_edges"]:
        angle = math.atan2(hit_y, hit_x) % (2.0 * math.pi)
        boundary_bin = min(
            int(angle * boundary_bins / (2.0 * math.pi)),
            boundary_bins - 1,
        )
        rows.append(node)
        columns.append(boundary_bin)
        values.append(rate)
    return coo_matrix(
        (values, (rows, columns)),
        shape=(grid["generator"].shape[0], boundary_bins),
    ).tocsc()


def _entry_matrix(
    grid: dict[str, object], entry_angles: np.ndarray
) -> np.ndarray:
    xs = grid["xs"]
    ys = grid["ys"]
    lookup = grid["node_lookup"]
    matrix = np.zeros((grid["generator"].shape[0], len(entry_angles)))
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
                key = (x_left + x_offset, y_bottom + y_offset)
                if key not in lookup:
                    # Return data vanish on both absorbing boundary pieces.
                    # The omitted weight is the competing immediate branch.
                    continue
                matrix[lookup[key], angle_index] += x_weight * y_weight
    return matrix


def _axial_factors(time: float, rho: float) -> tuple[float, float]:
    axial_rate = 1.0 + rho
    variance = math.expm1(2.0 * axial_rate * time) / axial_rate
    patch_mass = float(
        erf(PATCH_HALF_HEIGHT / math.sqrt(2.0 * variance))
    )
    squared_density_mass = (
        float(erf(PATCH_HALF_HEIGHT / math.sqrt(variance)))
        / (2.0 * math.sqrt(math.pi) * math.sqrt(variance))
    )
    return patch_mass, math.sqrt(squared_density_mass)


def _propagate_density(
    grid: dict[str, object],
    rho: float,
    entry_angle_count: int = 32,
    boundary_bins: int = 32,
) -> dict[str, object]:
    entry_angles = np.linspace(
        0.0, 2.0 * math.pi, entry_angle_count, endpoint=False
    )
    state = _entry_matrix(grid, entry_angles)
    inner_matrix = _inner_rate_matrix(grid, boundary_bins)
    generator_transpose = grid["generator"].transpose().tocsc()
    bin_width = 2.0 * math.pi / boundary_bins

    times = []
    raw_l2_rows = []
    scalar_density_rows = []
    current_time = 0.0
    segments = (
        (0.05, 26),
        (0.2, 31),
        (1.0, 41),
        (4.0, 31),
        (12.0, 33),
    )
    for segment_end, point_count in segments:
        duration = segment_end - current_time
        trajectory = expm_multiply(
            generator_transpose,
            state,
            start=0.0,
            stop=duration,
            num=point_count,
            endpoint=True,
        )
        for local_index in range(1, point_count):
            time = current_time + duration * local_index / (point_count - 1)
            snapshot = trajectory[local_index]
            binned_flux = np.asarray(inner_matrix.transpose() @ snapshot).T
            transverse_l2 = np.sqrt(
                np.sum(binned_flux**2, axis=1) / bin_width
            )
            transverse_mass = np.sum(binned_flux, axis=1)
            patch_mass, axial_l2 = _axial_factors(time, rho)
            deformation = math.exp(time)
            times.append(time)
            raw_l2_rows.append(deformation * axial_l2 * transverse_l2)
            scalar_density_rows.append(
                deformation * patch_mass * transverse_mass
            )
        state = trajectory[-1]
        current_time = segment_end

    return {
        "entry_angles": entry_angles,
        "times": np.asarray(times),
        "raw_l2": np.asarray(raw_l2_rows),
        "scalar_density": np.asarray(scalar_density_rows),
        "terminal_state": state,
        "boundary_bins": boundary_bins,
    }


def _sampled_interval_factor(
    times: np.ndarray, envelope: np.ndarray
) -> dict[str, float]:
    positive_envelope = np.maximum(envelope, 1.0e-300)
    tail_count = min(12, len(times))
    tail_slope = -float(
        np.polyfit(
            times[-tail_count:],
            np.log(positive_envelope[-tail_count:]),
            1,
        )[0]
    )
    tail_decay = max(0.1, 0.8 * tail_slope)
    stresses = []
    for window in np.geomspace(0.04, 1.0, 48):
        interval_indices = np.floor(times / window).astype(int)
        interval_sum = 0.0
        for interval_index in np.unique(interval_indices):
            interval_sum += float(
                np.max(envelope[interval_indices == interval_index])
            )
        terminal_tail = float(envelope[-1]) / (
            1.0 - math.exp(-tail_decay * window)
        )
        stressed_sum = 1.05 * (interval_sum + terminal_tail)
        energy = window + 1.0 / FORM_FLOOR
        stresses.append((energy * stressed_sum, window, tail_decay))
    factor, window, tail_decay = min(stresses)
    return {
        "sampled_stressed_factor": factor,
        "optimal_sampled_window": window,
        "fitted_stressed_tail_decay": tail_decay,
    }


def _row(
    resolvent,
    axial,
    y_intervals: int,
    rho: float = 0.0,
) -> dict[str, object]:
    grid = resolvent._build_generator(y_intervals, rho)
    propagated = _propagate_density(grid, rho)
    times = propagated["times"]
    scalar_density = propagated["scalar_density"]
    raw_l2 = propagated["raw_l2"]
    integrated_scalar = np.trapezoid(scalar_density, times, axis=0)
    patched_return, _, _ = axial._integrate_patched_return(grid, rho, 1.0)
    exact_scalar = _entry_matrix(
        grid, propagated["entry_angles"]
    ).T @ patched_return
    scalar_recovery_error = float(
        np.max(np.abs(integrated_scalar - exact_scalar))
    )

    angle_rows = []
    for angle_index, angle in enumerate(propagated["entry_angles"]):
        factor = _sampled_interval_factor(
            times, raw_l2[:, angle_index]
        )
        branch_mass = float(exact_scalar[angle_index])
        response = math.sqrt(
            branch_mass
            * TRACE_L4_FORM_CONSTANT
            * factor["sampled_stressed_factor"]
        )
        angle_rows.append(
            {
                "angle": float(angle),
                "weighted_return_scalar_gain": branch_mass,
                "peak_raw_spatial_L2_density": float(
                    np.max(raw_l2[:, angle_index])
                ),
                "peak_time": float(
                    times[int(np.argmax(raw_l2[:, angle_index]))]
                ),
                "raw_interval_factor": factor[
                    "sampled_stressed_factor"
                ],
                "raw_trace_response_at_alpha_zero": response,
                "optimal_sampled_window": factor[
                    "optimal_sampled_window"
                ],
                "fitted_stressed_tail_decay": factor[
                    "fitted_stressed_tail_decay"
                ],
            }
        )
    return {
        "y_intervals": y_intervals,
        "spacing": grid["spacing"],
        "state_count": grid["generator"].shape[0],
        "rho": rho,
        "entry_angle_count": len(propagated["entry_angles"]),
        "boundary_bin_count": propagated["boundary_bins"],
        "minimum_time": float(times[0]),
        "maximum_time": float(times[-1]),
        "time_sample_count": len(times),
        "scalar_resolvent_recovery_error": scalar_recovery_error,
        "maximum_terminal_state": float(
            np.max(propagated["terminal_state"])
        ),
        "angle_rows": angle_rows,
        "maximum_raw_interval_factor": max(
            row["raw_interval_factor"] for row in angle_rows
        ),
        "maximum_raw_trace_response_at_alpha_zero": max(
            row["raw_trace_response_at_alpha_zero"] for row in angle_rows
        ),
    }


def _threshold_rows(
    density_row: dict[str, object], residual_budget: dict[str, object]
) -> list[dict[str, float]]:
    source_rows = residual_budget["angle_rows"]
    rows = []
    for density_angle in density_row["angle_rows"]:
        source = min(
            source_rows,
            key=lambda row: abs(row["angle"] - density_angle["angle"]),
        )
        return_baseline = source["return_one_history_gain"]
        wall_baseline = source["wall_one_history_gain"]
        response_zero = density_angle[
            "raw_trace_response_at_alpha_zero"
        ]
        drift_threshold = source[
            "return_only_additive_gain_allowance"
        ] / (response_zero * DRIFT_L3_FORCING)

        def potential_gate(potential_mass: float) -> float:
            alpha = POTENTIAL_RELATIVE_FORM * potential_mass
            response = (
                response_zero
                * POTENTIAL_L3_OVER_2_FORCING
                * potential_mass
                / (1.0 - alpha) ** 1.5
            )
            return (
                (return_baseline + response) ** 2
                + wall_baseline**2
                - 1.0
            )

        potential_threshold = brentq(
            potential_gate,
            0.0,
            0.999 / POTENTIAL_RELATIVE_FORM,
        )
        rows.append(
            {
                "angle": density_angle["angle"],
                "conditional_return_only_potential_L3_over_2_threshold": (
                    potential_threshold
                ),
                "conditional_return_only_drift_L3_threshold": (
                    drift_threshold
                ),
            }
        )
    return rows


def audit() -> dict[str, object]:
    resolvent = _load_module(
        "neutral_strip_branch_resolvent_pilot.py",
        "resolvent_for_return_density",
    )
    axial = _load_module(
        "neutral_strip_axial_patch_branch_pilot.py",
        "axial_for_return_density",
    )
    residual = _load_module(
        "migrating_core_residual_budget_audit.py",
        "residual_for_return_density",
    )
    mesh_rows = [_row(resolvent, axial, intervals) for intervals in (30, 40)]
    working = mesh_rows[-1]
    residual_budget = residual._numerical_budget()
    threshold_rows = _threshold_rows(working, residual_budget)
    potential_worst = min(
        threshold_rows,
        key=lambda row: row[
            "conditional_return_only_potential_L3_over_2_threshold"
        ],
    )
    drift_worst = min(
        threshold_rows,
        key=lambda row: row[
            "conditional_return_only_drift_L3_threshold"
        ],
    )
    factor_change = abs(
        mesh_rows[-1]["maximum_raw_interval_factor"]
        - mesh_rows[-2]["maximum_raw_interval_factor"]
    )
    response_change = abs(
        mesh_rows[-1]["maximum_raw_trace_response_at_alpha_zero"]
        - mesh_rows[-2]["maximum_raw_trace_response_at_alpha_zero"]
    )
    result = {
        "model": "static neutral strip rho=0 with exact outward axial OU",
        "return_surface": "Sigma={r=1, |z|<3/4}",
        "raw_density_norm": (
            "||exp(t) h_return(t,theta) g_OU(t,z) 1_|z|<H||_L2"
        ),
        "raw_response_formula": (
            "K_R(theta)=sqrt(p_R(theta)*C_4*J_raw,R(theta))"
        ),
        "mesh_rows": mesh_rows,
        "maximum_mesh_factor_change": factor_change,
        "maximum_mesh_response_change": response_change,
        "threshold_rows": threshold_rows,
        "conditional_minimum_return_only_potential_L3_over_2_threshold": (
            potential_worst[
                "conditional_return_only_potential_L3_over_2_threshold"
            ]
        ),
        "conditional_worst_potential_angle": potential_worst["angle"],
        "conditional_minimum_return_only_drift_L3_threshold": drift_worst[
            "conditional_return_only_drift_L3_threshold"
        ],
        "conditional_worst_drift_angle": drift_worst["angle"],
        "boundary_edge_rate_reconstruction_exact": True,
        "finite_state_semigroup_exact_for_discrete_generator": True,
        "continuum_boundary_density_envelope_certified": False,
        "weighted_Navier_Stokes_return_density_certified": False,
        "composite_wall_child_return_density_computed": False,
        "full_wall_stopping_trace_gate_closed": False,
        "scope_guard": (
            "the matrix exponential and boundary-edge flux are exact for "
            "each finite-state generator. Boundary binning, time sampling, "
            "the 5 percent envelope stress, fitted tail, and mesh limit are "
            "pilots rather than a continuum enclosure. The calculation "
            "covers rho=0 only and no nonaffine Navier-Stokes drift"
        ),
        "next_gate": (
            "refine boundary bins and time windows, certify the rho=0 "
            "continuum flux envelope, extend across 0<=rho<=1, then compose "
            "wall migration with one child-return kernel"
        ),
    }
    checks = (
        max(row["scalar_resolvent_recovery_error"] for row in mesh_rows)
        < 0.03,
        working["maximum_terminal_state"] < 1.0e-8,
        working["maximum_raw_interval_factor"] > 0.0,
        working["maximum_raw_trace_response_at_alpha_zero"] > 0.0,
        result["conditional_minimum_return_only_potential_L3_over_2_threshold"]
        > 0.0,
        result["conditional_minimum_return_only_drift_L3_threshold"] > 0.0,
        result["boundary_edge_rate_reconstruction_exact"],
        result["finite_state_semigroup_exact_for_discrete_generator"],
        not result["continuum_boundary_density_envelope_certified"],
        not result["weighted_Navier_Stokes_return_density_certified"],
        not result["composite_wall_child_return_density_computed"],
        not result["full_wall_stopping_trace_gate_closed"],
    )
    result["all_positive_neutral_strip_return_density_checks_pass"] = all(
        checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
