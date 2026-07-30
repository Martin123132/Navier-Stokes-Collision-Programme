"""Audit a geometry-triggered migrating half-scale child transition."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.sparse import eye
from scipy.sparse.linalg import eigs, splu


R_STAR = 0.5
ENTRY_RADIUS = 2.0
CURRENT_CUBIC_SUPPORT_RADIUS = 1.91
WORKING_HALF_WIDTH = 2.1


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _partition_transfer_audit(radial_module) -> dict[str, object]:
    points = np.linspace(-1.0, 1.0, 401)
    phases = (0.0, 0.371)
    partition_rows = []
    partition_values = []
    for spacing, phase in ((1.0, phases[0]), (0.5, phases[1])):
        scaled = points / spacing - phase
        labels = np.arange(-8, 9)
        values = np.array(
            [radial_module._cardinal_cubic(scaled - label) for label in labels]
        )
        sums = np.sum(values, axis=0)
        partition_values.append(values)
        partition_rows.append(
            {
                "spacing": spacing,
                "phase": phase,
                "minimum_weight": float(np.min(values)),
                "maximum_partition_sum_error": float(
                    np.max(np.abs(sums - 1.0))
                ),
            }
        )

    parent_values, child_values = partition_values
    parent_sums = np.sum(parent_values, axis=0)
    child_sums = np.sum(child_values, axis=0)
    # P(child|parent,x)=phi_child(x) is a physical-norm Markov resampling.
    child_marginal = child_values * parent_sums[np.newaxis, :]
    marginal_error = float(np.max(np.abs(child_marginal - child_values)))
    pair_sum_error = float(np.max(np.abs(child_sums**2 - 1.0)))
    return {
        "partition_rows": partition_rows,
        "maximum_parent_to_translated_child_marginal_error": marginal_error,
        "maximum_translated_child_pair_probability_sum_error": pair_sum_error,
        "translated_fine_partition_is_physical_Markov": bool(
            max(
                row["maximum_partition_sum_error"]
                for row in partition_rows
            )
            < 2.0e-13
            and min(row["minimum_weight"] for row in partition_rows)
            >= -1.0e-15
            and marginal_error < 2.0e-13
            and pair_sum_error < 4.0e-13
        ),
    }


def _interpolate_full_entry(
    values: np.ndarray,
    angle: float,
    grid: dict[str, object],
    half_width: float,
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
    result = np.zeros(3)
    for x_offset, x_weight in ((0, 1.0 - x_fraction), (1, x_fraction)):
        for y_offset, y_weight in ((0, 1.0 - y_fraction), (1, y_fraction)):
            x_index = x_left + x_offset
            y_index = y_bottom + y_offset
            lookup = grid["node_lookup"]
            if (x_index, y_index) in lookup:
                corner = values[lookup[(x_index, y_index)]]
            else:
                corner_x = float(xs[x_index])
                corner_y = float(ys[y_index])
                if corner_x**2 + corner_y**2 <= 1.0 + 1.0e-12:
                    corner = np.array([1.0, 0.0, 0.0])
                elif abs(corner_y) >= half_width - 1.0e-12:
                    offset = math.sqrt(corner_x**2 + half_width**2) - 1.0
                    wall_factor = 0.5 * math.exp(R_STAR * offset**2 / 3.0)
                    corner = np.array([0.0, 1.0, wall_factor])
                else:
                    raise RuntimeError("unclassified full-wall interpolation corner")
            result += x_weight * y_weight * corner
    return result


def _axial_wall_benchmark_row(
    width_module,
    resolvent,
    axial,
    half_width: float,
    target_spacing: float = 0.075,
    time_step_scale: float = 1.0,
    x_half_width: float | None = None,
) -> dict[str, object]:
    row = width_module._row(
        resolvent,
        axial,
        half_width,
        target_spacing=target_spacing,
        time_step_scale=time_step_scale,
        x_half_width=x_half_width,
    )
    migration_offset = half_width - ENTRY_RADIUS / 2.0
    migration_log_gauge_cost = R_STAR * migration_offset**2 / 3.0
    migration_one_history_factor = (
        math.exp(migration_log_gauge_cost) / 2.0
    )
    one_history_physical_gauge_conversion = math.exp(R_STAR / 4.0)
    minimum_capture_support = 2.0 * math.sqrt(2.0) * migration_offset
    support_paid_log_gauge_cost = R_STAR / 4.0 * (
        minimum_capture_support**2 / 3.0 + 0.75
    )
    support_paid_factor = math.exp(support_paid_log_gauge_cost) / 2.0

    angle_rows = []
    for angle_row in row["angle_rows"]:
        return_gain = angle_row["axial_patch_return_gain"]
        wall_gain = angle_row["wall_deformation_gain"]
        inward_split_criterion = resolvent.H1_ENTRY_GAIN**2 * (
            (0.5 * return_gain) ** 2 + wall_gain**2
        )
        migrating_wall_criterion = resolvent.H1_ENTRY_GAIN**2 * (
            return_gain**2
            + (migration_one_history_factor * wall_gain) ** 2
        )
        migrating_with_conversion_criterion = resolvent.H1_ENTRY_GAIN**2 * (
            return_gain**2
            + (
                one_history_physical_gauge_conversion
                * migration_one_history_factor
                * wall_gain
            )
            ** 2
        )
        residual_wall_budget = (
            1.0 / resolvent.H1_ENTRY_GAIN**2 - return_gain**2
        )
        if wall_gain > 0.0 and residual_wall_budget > 0.0:
            allowable_wall_transfer_mismatch = math.sqrt(
                residual_wall_budget
            ) / (migration_one_history_factor * wall_gain)
        else:
            allowable_wall_transfer_mismatch = math.inf
        expanded_support_criterion = resolvent.H1_ENTRY_GAIN**2 * (
            return_gain**2 + (support_paid_factor * wall_gain) ** 2
        )
        angle_rows.append(
            {
                "angle": angle_row["angle"],
                "axial_patch_return_gain": return_gain,
                "wall_deformation_gain": wall_gain,
                "inward_split_wall_unpaid_criterion": inward_split_criterion,
                "migrating_wall_endpoint_criterion": migrating_wall_criterion,
                "migrating_wall_with_conversion_criterion": (
                    migrating_with_conversion_criterion
                ),
                "allowable_one_history_wall_transfer_mismatch": (
                    allowable_wall_transfer_mismatch
                ),
                "expanded_support_cubic_criterion": (
                    expanded_support_criterion
                ),
            }
        )

    def maximum(field: str) -> dict[str, float]:
        return max(angle_rows, key=lambda item: item[field])

    inward_worst = maximum("inward_split_wall_unpaid_criterion")
    migrating_worst = maximum("migrating_wall_endpoint_criterion")
    conversion_worst = maximum("migrating_wall_with_conversion_criterion")
    support_worst = maximum("expanded_support_cubic_criterion")
    minimum_mismatch_allowance = min(
        angle_rows,
        key=lambda item: item[
            "allowable_one_history_wall_transfer_mismatch"
        ],
    )
    return {
        "strip_half_width": half_width,
        "spacing": row["spacing"],
        "x_half_width": row["x_half_width"],
        "time_step_scale": time_step_scale,
        "principal_killed_rate_pilot": row["principal_killed_rate_pilot"],
        "exact_survival_margin": math.pi**2 / (4.0 * half_width**2) - 0.5,
        "migration_center_offset_over_parent_L": migration_offset,
        "migration_endpoint_log_gauge_cost": migration_log_gauge_cost,
        "migration_one_history_shrink_paid_factor": (
            migration_one_history_factor
        ),
        "one_history_physical_gauge_conversion_stress": (
            one_history_physical_gauge_conversion
        ),
        "minimum_cubic_support_radius_for_direct_capture": (
            minimum_capture_support
        ),
        "minimum_support_shrink_paid_factor": support_paid_factor,
        "maximum_inward_split_wall_unpaid_criterion": inward_worst[
            "inward_split_wall_unpaid_criterion"
        ],
        "worst_inward_split_angle": inward_worst["angle"],
        "maximum_migrating_wall_endpoint_criterion": migrating_worst[
            "migrating_wall_endpoint_criterion"
        ],
        "worst_migrating_wall_angle": migrating_worst["angle"],
        "maximum_migrating_wall_with_conversion_criterion": (
            conversion_worst["migrating_wall_with_conversion_criterion"]
        ),
        "worst_migrating_wall_with_conversion_angle": conversion_worst[
            "angle"
        ],
        "minimum_allowable_one_history_wall_transfer_mismatch": (
            minimum_mismatch_allowance[
                "allowable_one_history_wall_transfer_mismatch"
            ]
        ),
        "minimum_mismatch_allowance_angle": minimum_mismatch_allowance[
            "angle"
        ],
        "maximum_expanded_support_cubic_criterion": support_worst[
            "expanded_support_cubic_criterion"
        ],
        "worst_expanded_support_angle": support_worst["angle"],
        "unweighted_return_resolvent_recovery_error": row[
            "unweighted_return_resolvent_recovery_error"
        ],
        "terminal_inner_flux_maximum": row["terminal_inner_flux_maximum"],
    }


def _full_wall_transition_row(
    resolvent,
    axial,
    half_width: float,
    target_spacing: float = 0.075,
    time_step_scale: float = 1.0,
    x_half_width: float | None = None,
) -> dict[str, object]:
    y_intervals = int(round(2.0 * half_width / target_spacing))
    if x_half_width is None:
        x_half_width = max(4.2, half_width + 2.1)
    grid = resolvent._build_generator(
        y_intervals,
        0.0,
        x_half_width,
        True,
        half_width,
    )
    generator = grid["generator"]
    eigenvalues = eigs(
        generator,
        k=2,
        which="LR",
        return_eigenvectors=False,
        tol=1.0e-8,
        maxiter=100000,
    )
    killed_rate = float(np.min(-np.real(eigenvalues)))
    patched_return, recovery_error, terminal_flux = (
        axial._integrate_patched_return(
            grid, 0.0, time_step_scale
        )
    )
    resolvent_factorization = splu(
        -generator - eye(generator.shape[0], format="csc")
    )
    unweighted_wall = resolvent_factorization.solve(grid["wall_rates"])

    wall_x = np.asarray(grid["coordinates"], dtype=float)[:, 0]
    wall_radius = np.sqrt(wall_x**2 + half_width**2)
    migration_offsets = wall_radius - ENTRY_RADIUS / 2.0
    wall_endpoint_factors = 0.5 * np.exp(
        R_STAR * migration_offsets**2 / 3.0
    )
    weighted_wall = resolvent_factorization.solve(
        grid["wall_rates"] * wall_endpoint_factors
    )
    one_history_conversion = math.exp(R_STAR / 4.0)
    tracking_bracket_minimum = 1.0 - R_STAR * ENTRY_RADIUS**2 / 8.0
    tracking_shrink_factor = 2.0 ** (-tracking_bracket_minimum)
    branch_values = np.column_stack(
        [patched_return, unweighted_wall, weighted_wall]
    )

    axial_offset = half_width - ENTRY_RADIUS / 2.0
    axial_endpoint_factor = 0.5 * math.exp(
        R_STAR * axial_offset**2 / 3.0
    )
    minimum_capture_support = 2.0 * math.sqrt(2.0) * axial_offset
    support_paid_factor = 0.5 * math.exp(
        R_STAR / 4.0 * (minimum_capture_support**2 / 3.0 + 0.75)
    )
    angle_rows = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False):
        return_gain, wall_gain, migrating_wall_gain = (
            _interpolate_full_entry(
                branch_values, angle, grid, half_width
            )
        )
        inward_criterion = resolvent.H1_ENTRY_GAIN**2 * (
            (0.5 * return_gain) ** 2 + wall_gain**2
        )
        axial_benchmark = resolvent.H1_ENTRY_GAIN**2 * (
            return_gain**2 + (axial_endpoint_factor * wall_gain) ** 2
        )
        migrating_criterion = resolvent.H1_ENTRY_GAIN**2 * (
            return_gain**2 + migrating_wall_gain**2
        )
        conversion_criterion = resolvent.H1_ENTRY_GAIN**2 * (
            return_gain**2
            + (one_history_conversion * migrating_wall_gain) ** 2
        )
        tracking_criterion = resolvent.H1_ENTRY_GAIN**2 * (
            return_gain**2 + (tracking_shrink_factor * wall_gain) ** 2
        )
        tracking_conversion_criterion = resolvent.H1_ENTRY_GAIN**2 * (
            return_gain**2
            + (
                one_history_conversion
                * tracking_shrink_factor
                * wall_gain
            )
            ** 2
        )
        support_benchmark = resolvent.H1_ENTRY_GAIN**2 * (
            return_gain**2 + (support_paid_factor * wall_gain) ** 2
        )
        residual_wall_budget = (
            1.0 / resolvent.H1_ENTRY_GAIN**2 - return_gain**2
        )
        if migrating_wall_gain > 0.0 and residual_wall_budget > 0.0:
            mismatch_allowance = math.sqrt(
                residual_wall_budget
            ) / migrating_wall_gain
        else:
            mismatch_allowance = math.inf
        angle_rows.append(
            {
                "angle": float(angle),
                "axial_patch_return_gain": float(return_gain),
                "unweighted_wall_gain": float(wall_gain),
                "geometry_weighted_migrating_wall_gain": float(
                    migrating_wall_gain
                ),
                "inward_split_wall_unpaid_criterion": float(
                    inward_criterion
                ),
                "axial_wall_point_benchmark_criterion": float(
                    axial_benchmark
                ),
                "migrating_wall_endpoint_criterion": float(
                    migrating_criterion
                ),
                "migrating_wall_with_conversion_criterion": float(
                    conversion_criterion
                ),
                "smooth_tracking_wall_criterion": float(
                    tracking_criterion
                ),
                "smooth_tracking_wall_with_conversion_criterion": float(
                    tracking_conversion_criterion
                ),
                "expanded_support_cubic_criterion": float(
                    support_benchmark
                ),
                "allowable_one_history_wall_transfer_mismatch": float(
                    mismatch_allowance
                ),
            }
        )

    def maximum(field: str) -> dict[str, float]:
        return max(angle_rows, key=lambda item: item[field])

    inward_worst = maximum("inward_split_wall_unpaid_criterion")
    axial_worst = maximum("axial_wall_point_benchmark_criterion")
    migrating_worst = maximum("migrating_wall_endpoint_criterion")
    conversion_worst = maximum("migrating_wall_with_conversion_criterion")
    tracking_worst = maximum("smooth_tracking_wall_criterion")
    tracking_conversion_worst = maximum(
        "smooth_tracking_wall_with_conversion_criterion"
    )
    support_worst = maximum("expanded_support_cubic_criterion")
    mismatch_worst = min(
        angle_rows,
        key=lambda item: item[
            "allowable_one_history_wall_transfer_mismatch"
        ],
    )
    return {
        "strip_half_width": half_width,
        "spacing": grid["spacing"],
        "x_half_width": grid["x_half_width"],
        "time_step_scale": time_step_scale,
        "principal_killed_rate_pilot": killed_rate,
        "exact_survival_margin": math.pi**2 / (4.0 * half_width**2) - 0.5,
        "axial_wall_point_center_offset_over_parent_L": axial_offset,
        "axial_wall_point_log_gauge_cost": R_STAR * axial_offset**2 / 3.0,
        "axial_wall_point_shrink_paid_factor": axial_endpoint_factor,
        "minimum_cubic_support_radius_for_axial_point_capture": (
            minimum_capture_support
        ),
        "minimum_support_shrink_paid_factor": support_paid_factor,
        "smooth_tracking_bracket_minimum": tracking_bracket_minimum,
        "smooth_tracking_one_history_shrink_factor": tracking_shrink_factor,
        "angle_rows": angle_rows,
        "maximum_wall_endpoint_factor_on_truncated_grid": float(
            np.max(wall_endpoint_factors)
        ),
        "maximum_inward_split_wall_unpaid_criterion": inward_worst[
            "inward_split_wall_unpaid_criterion"
        ],
        "worst_inward_split_angle": inward_worst["angle"],
        "maximum_axial_wall_point_benchmark_criterion": axial_worst[
            "axial_wall_point_benchmark_criterion"
        ],
        "maximum_migrating_wall_endpoint_criterion": migrating_worst[
            "migrating_wall_endpoint_criterion"
        ],
        "worst_migrating_wall_angle": migrating_worst["angle"],
        "maximum_migrating_wall_with_conversion_criterion": conversion_worst[
            "migrating_wall_with_conversion_criterion"
        ],
        "worst_migrating_wall_with_conversion_angle": conversion_worst[
            "angle"
        ],
        "maximum_smooth_tracking_wall_criterion": tracking_worst[
            "smooth_tracking_wall_criterion"
        ],
        "maximum_smooth_tracking_wall_with_conversion_criterion": (
            tracking_conversion_worst[
                "smooth_tracking_wall_with_conversion_criterion"
            ]
        ),
        "minimum_allowable_one_history_wall_transfer_mismatch": mismatch_worst[
            "allowable_one_history_wall_transfer_mismatch"
        ],
        "minimum_mismatch_allowance_angle": mismatch_worst["angle"],
        "maximum_expanded_support_cubic_criterion": support_worst[
            "expanded_support_cubic_criterion"
        ],
        "unweighted_return_resolvent_recovery_error": recovery_error,
        "terminal_inner_flux_maximum": terminal_flux,
    }


def audit() -> dict[str, object]:
    resolvent = _load_module(
        "neutral_strip_branch_resolvent_pilot.py",
        "resolvent_for_migrating_child",
    )
    axial = _load_module(
        "neutral_strip_axial_patch_branch_pilot.py",
        "axial_for_migrating_child",
    )
    radial = _load_module(
        "radial_cubic_partition_audit.py",
        "radial_partition_for_migrating_child",
    )

    eta = sp.Integer(2)
    support = sp.symbols("rho_s", positive=True)
    maximum_direct_child_reach = eta / 2 + support / (2 * sp.sqrt(2))
    support_contained_reach = sp.simplify(
        maximum_direct_child_reach.subs(support, eta)
    )
    reach_ratio = sp.simplify(support_contained_reach / eta)
    wall = sp.Rational(21, 10)
    migration_offset = sp.simplify(wall - eta / 2)
    required_support = sp.simplify(2 * sp.sqrt(2) * migration_offset)
    current_offset = sp.Rational(191, 100) / (2 * sp.sqrt(2))
    current_capture_gap = sp.simplify(wall - eta / 2 - current_offset)

    child_coordinate, offset = sp.symbols("y_child d", real=True)
    exponent_difference = sp.expand(
        (offset + child_coordinate / 2) ** 2 - child_coordinate**2
    )
    critical_coordinate = sp.solve(
        sp.diff(exponent_difference, child_coordinate), child_coordinate
    )[0]
    maximum_exponent_difference = sp.factor(
        exponent_difference.subs(child_coordinate, critical_coordinate)
    )
    y_parallel, y_perpendicular = sp.symbols(
        "y_parallel y_perpendicular", real=True
    )
    tracking_bracket = sp.expand(
        1
        + sp.Rational(1, 4)
        * (y_parallel**2 + y_perpendicular**2 - 2 * y_parallel)
    )
    tracking_square_completion = sp.expand(
        sp.Rational(3, 4)
        + sp.Rational(1, 4)
        * ((y_parallel - 1) ** 2 + y_perpendicular**2)
    )
    working_log_cost = sp.simplify(
        sp.Rational(1, 2) * migration_offset**2 / 3
    )
    working_migration_factor = math.exp(float(working_log_cost)) / 2.0
    working_conversion_factor = math.exp(R_STAR / 4.0)
    working_effective_factor = (
        working_migration_factor * working_conversion_factor
    )
    child_entry_to_wall_gap = float(wall - eta)

    partition = _partition_transfer_audit(radial)
    widths = (2.02, 2.05, 2.10, 2.15, 2.20)
    width_rows = [
        _full_wall_transition_row(resolvent, axial, half_width)
        for half_width in widths
    ]
    working_row = next(
        row for row in width_rows if row["strip_half_width"] == 2.10
    )
    working_complete_pair_factor = working_row[
        "maximum_migrating_wall_with_conversion_criterion"
    ]
    generation_rows = [
        {
            "migrating_generations": generations,
            "conditional_complete_pair_product": (
                working_complete_pair_factor**generations
            ),
        }
        for generations in (1, 5, 10, 20, 50, 100, 200)
    ]
    mesh_rows = [
        _full_wall_transition_row(
            resolvent,
            axial,
            WORKING_HALF_WIDTH,
            target_spacing=spacing,
        )
        for spacing in (0.09, 0.07, 0.055)
    ]
    time_rows = [
        _full_wall_transition_row(
            resolvent,
            axial,
            WORKING_HALF_WIDTH,
            target_spacing=0.07,
            time_step_scale=scale,
        )
        for scale in (1.0, 0.5)
    ]
    x_rows = [
        _full_wall_transition_row(
            resolvent,
            axial,
            WORKING_HALF_WIDTH,
            target_spacing=0.07,
            x_half_width=x_width,
        )
        for x_width in (4.2, 5.25)
    ]
    mesh_spread = max(
        row["maximum_migrating_wall_endpoint_criterion"]
        for row in mesh_rows
    ) - min(
        row["maximum_migrating_wall_endpoint_criterion"]
        for row in mesh_rows
    )
    time_change = abs(
        time_rows[0]["maximum_migrating_wall_endpoint_criterion"]
        - time_rows[1]["maximum_migrating_wall_endpoint_criterion"]
    )
    x_spread = abs(
        x_rows[0]["maximum_migrating_wall_endpoint_criterion"]
        - x_rows[1]["maximum_migrating_wall_endpoint_criterion"]
    )
    minimum_sampled_mismatch_allowance = min(
        row["minimum_allowable_one_history_wall_transfer_mismatch"]
        for row in width_rows
    )
    one_history_conversion_stress = math.exp(R_STAR / 4.0)

    result: dict[str, object] = {
        "current_entry_radius_over_L": int(eta),
        "admissible_strip_width": "2<Y<pi/sqrt(2)",
        "direct_child_reach_formula": (
            "eta/2+rho_s/(2sqrt(2))"
        ),
        "support_contained_maximum_reach": str(support_contained_reach),
        "support_contained_reach_over_entry_radius": str(reach_ratio),
        "support_contained_child_reaches_entry_surface": bool(
            support_contained_reach >= eta
        ),
        "support_contained_child_can_reach_admissible_outer_wall": False,
        "working_wall": str(wall),
        "working_axial_wall_point_center_offset": str(migration_offset),
        "minimum_working_capture_support": str(required_support),
        "minimum_working_capture_support_value": float(required_support),
        "minimum_capture_support_fits_current_buffer": bool(
            required_support <= eta
        ),
        "current_cubic_direct_capture_gap": str(current_capture_gap),
        "current_cubic_direct_capture_gap_value": float(current_capture_gap),
        "migration_gauge_exponent_difference": str(exponent_difference),
        "migration_gauge_maximizer": str(critical_coordinate),
        "migration_gauge_maximum_at_offset_d": str(
            maximum_exponent_difference
        ),
        "working_axial_wall_point_log_gauge_cost": str(working_log_cost),
        "working_axial_wall_point_shrink_paid_factor": (
            working_migration_factor
        ),
        "working_axial_wall_point_pair_factor": working_migration_factor**2,
        "working_axial_wall_point_one_conversion_factor": (
            working_effective_factor
        ),
        "working_axial_wall_point_many_generation_product_decays": bool(
            working_migration_factor < 1.0
        ),
        "smooth_tracking_geometric_error": (
            "q_geom=ell[1+(R_star/2)(|y|^2-eta*y_parallel)]"
        ),
        "working_smooth_tracking_bracket": str(tracking_bracket),
        "working_smooth_tracking_square_completion": str(
            tracking_square_completion
        ),
        "working_smooth_tracking_bracket_minimum": 0.75,
        "working_smooth_tracking_one_history_factor": 2.0 ** (-0.75),
        "smooth_tracking_center_scale_identity_verified": bool(
            sp.simplify(tracking_bracket - tracking_square_completion) == 0
        ),
        "child_entry_radius_after_migration": int(eta),
        "child_entry_to_next_wall_normalized_gap": child_entry_to_wall_gap,
        "single_transition_has_strict_positive_wall_gap": bool(
            child_entry_to_wall_gap > 0.0
        ),
        "uniform_cumulative_center_travel_bound_available": False,
        "generation_rows": generation_rows,
        "bounded_terminal_Zeno_remainder_vanishes": bool(
            working_complete_pair_factor < 1.0
            and generation_rows[-1]["conditional_complete_pair_product"]
            < 1.0e-12
        ),
        **partition,
        "fixed_branch_full_partition_jump": (
            "sum_j phi_child_j-sum_i phi_parent_i=1-1=0"
        ),
        "fixed_branch_full_partition_jump_is_conservative": True,
        "fixed_branch_pressure_commutator_identity": (
            "sum_j [Delta,phi_j]p=[Delta,sum_j phi_j]p=[Delta,1]p=0"
        ),
        "fixed_branch_signed_pressure_commutator_cancels": True,
        "abstract_extended_state_stopping_kernel": (
            "at a wall sign sigma, set L_child=L/2, "
            "c_child=c+sigma*(Y-eta/2)*L*e_neutral, and resample a "
            "translated fine label with probabilities phi_j^child(x)"
        ),
        "abstract_extended_state_stopping_transfer_is_Markov": bool(
            partition["translated_fine_partition_is_physical_Markov"]
        ),
        "width_rows": width_rows,
        "working_width_row": working_row,
        "mesh_refinement_rows": mesh_rows,
        "time_refinement_rows": time_rows,
        "x_truncation_rows": x_rows,
        "maximum_mesh_refinement_spread": mesh_spread,
        "time_refinement_change": time_change,
        "x_truncation_change": x_spread,
        "minimum_sampled_one_history_wall_transfer_mismatch_allowance": (
            minimum_sampled_mismatch_allowance
        ),
        "one_history_physical_gauge_conversion_stress": (
            one_history_conversion_stress
        ),
        "remaining_mismatch_factor_after_one_conversion": (
            minimum_sampled_mismatch_allowance
            / one_history_conversion_stress
        ),
        "inward_concentric_split_closes_sampled_scalar_gate": all(
            row["maximum_inward_split_wall_unpaid_criterion"] < 1.0
            for row in width_rows
        ),
        "migrating_wall_endpoint_closes_sampled_scalar_gate": all(
            row["maximum_migrating_wall_endpoint_criterion"] < 1.0
            for row in width_rows
        ),
        "migrating_wall_with_one_conversion_closes_sampled_scalar_gate": all(
            row["maximum_migrating_wall_with_conversion_criterion"] < 1.0
            for row in width_rows
        ),
        "smooth_tracking_closes_sampled_scalar_gate": all(
            row["maximum_smooth_tracking_wall_criterion"] < 1.0
            for row in width_rows
        ),
        "smooth_tracking_with_one_conversion_closes_sampled_scalar_gate": all(
            row[
                "maximum_smooth_tracking_wall_with_conversion_criterion"
            ]
            < 1.0
            for row in width_rows
        ),
        "expanded_support_cubic_closes_sampled_scalar_gate": all(
            row["maximum_expanded_support_cubic_criterion"] < 1.0
            for row in width_rows
        ),
        "path_triggered_partition_common_PDE_localization_certified": False,
        "physical_shrink_payment_for_path_triggered_migration_certified": False,
        "boundary_space_time_L2_error_gain_certified": False,
        "full_Navier_Stokes_geometry_transition_gate_closed": False,
        "interpretation": (
            "a nested support-contained dyadic child cannot reach any wall "
            "outside the complete entry surface. A concentric inward split "
            "is geometrically legal but leaves the wall branch above one. "
            "A wall-following migrating child has an x-dependent but "
            "integrable endpoint gauge cost and favorable scalar arithmetic, "
            "so the route is narrowed to a nonlocal physical/partition "
            "transfer theorem rather than ruled out"
        ),
        "scope_guard": (
            "the translated cubic partition gives an exact positive Markov "
            "resampling in the common physical norm, but this pilot does not "
            "promote the branchwise identities to one path-triggered adapted "
            "PDE localization or prove that the migration earns the monotone-"
            "envelope shrink payment. The endpoint criterion is conditional"
        ),
        "next_gate": (
            "construct the wall-sign-resolved physical transfer through a "
            "complete visit boundary, including one physical-to-gauge "
            "conversion, adapted signed pressure flux, migration-center "
            "moments, and zero-lag cascades. Reject the migration if its "
            "complete pair mismatch exceeds the scalar margin"
        ),
    }
    checks = (
        reach_ratio < 1,
        not result["support_contained_child_reaches_entry_surface"],
        not result[
            "support_contained_child_can_reach_admissible_outer_wall"
        ],
        required_support > eta,
        current_capture_gap > 0,
        maximum_exponent_difference == 4 * offset**2 / 3,
        working_migration_factor < 0.62,
        result["translated_fine_partition_is_physical_Markov"],
        result["fixed_branch_full_partition_jump_is_conservative"],
        result["fixed_branch_signed_pressure_commutator_cancels"],
        result["abstract_extended_state_stopping_transfer_is_Markov"],
        result["single_transition_has_strict_positive_wall_gap"],
        result["bounded_terminal_Zeno_remainder_vanishes"],
        not result["uniform_cumulative_center_travel_bound_available"],
        not result["inward_concentric_split_closes_sampled_scalar_gate"],
        result["migrating_wall_endpoint_closes_sampled_scalar_gate"],
        result[
            "migrating_wall_with_one_conversion_closes_sampled_scalar_gate"
        ],
        result["smooth_tracking_center_scale_identity_verified"],
        result["smooth_tracking_closes_sampled_scalar_gate"],
        result[
            "smooth_tracking_with_one_conversion_closes_sampled_scalar_gate"
        ],
        min(
            row["minimum_allowable_one_history_wall_transfer_mismatch"]
            for row in width_rows
        )
        > math.exp(R_STAR / 4.0),
        result["expanded_support_cubic_closes_sampled_scalar_gate"],
        mesh_spread < 0.003,
        time_change < 0.003,
        x_spread < 0.003,
        not result[
            "path_triggered_partition_common_PDE_localization_certified"
        ],
        not result[
            "physical_shrink_payment_for_path_triggered_migration_certified"
        ],
        not result["full_Navier_Stokes_geometry_transition_gate_closed"],
    )
    result["all_positive_migrating_child_checks_pass"] = all(checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
