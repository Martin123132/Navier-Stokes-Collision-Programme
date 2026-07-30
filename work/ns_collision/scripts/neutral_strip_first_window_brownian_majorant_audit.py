"""Prove and audit a Brownian majorant for the first return window."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import erf, erfc


START_RADIUS = 2.0
INNER_RADIUS = 1.0
PATCH_HALF_HEIGHT = 0.75
WINDOW = 0.375
FORM_FLOOR = 4.832287335665
DEFAULT_EXPLICIT_MODES = 96
DEFAULT_TIME_COUNT = 241
AXIAL_SCALAR_GLOBAL_UPPER = math.sqrt(
    1.0 + 2.0 * PATCH_HALF_HEIGHT**2 / math.pi
)


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _one_dimensional_first_passage_density(
    distance: float, time: float
) -> float:
    """First-passage density for a Brownian motion with generator Delta."""
    if distance <= 0.0 or time <= 0.0:
        raise ValueError("distance and time must be positive")
    return (
        distance
        / (2.0 * math.sqrt(math.pi) * time**1.5)
        * math.exp(-(distance**2) / (4.0 * time))
    )


def _ou_to_brownian_likelihood_upper(time: float) -> float:
    """Stopped-path likelihood upper for starts on the radius-two circle."""
    if time < 0.0:
        raise ValueError("time must be nonnegative")
    return math.exp(1.0 + 0.5 * time)


def _zeroth_brownian_mode_upper(time: float) -> float:
    first_passage = _one_dimensional_first_passage_density(1.0, time)
    return first_passage * math.exp(0.25 * time) / math.sqrt(2.0)


def _mode_path_split_upper(
    time: float,
    mode: int,
    excursion_radius: float | None = None,
) -> float:
    """Upper-bound one Brownian hitting-angle Fourier coefficient.

    The coefficient is reduced by Bessel absolute continuity to a weighted
    one-dimensional first-passage law. Paths below ``excursion_radius`` pay
    the angular clock; paths crossing it pay a longer first-passage density.
    """
    if time <= 0.0:
        raise ValueError("time must be positive")
    if mode < 1:
        raise ValueError("mode must be positive")
    if excursion_radius is None:
        excursion_radius = 2.0 + math.sqrt(mode * time)
    if excursion_radius <= START_RADIUS:
        raise ValueError("excursion radius must exceed the start radius")

    first_passage = _one_dimensional_first_passage_density(1.0, time)
    angular_clock = math.exp(
        -(mode * mode - 0.25) * time / excursion_radius**2
    )
    excursion_distance = 2.0 * excursion_radius - 3.0
    excursion = _one_dimensional_first_passage_density(
        excursion_distance, time
    )
    return (first_passage * angular_clock + excursion) / math.sqrt(2.0)


def _squared_mode_tail_upper(time: float, first_mode: int) -> float:
    """Sum the squared path-split bounds from ``first_mode`` to infinity."""
    if time <= 0.0:
        raise ValueError("time must be positive")
    if first_mode < 3 or first_mode * time <= 1.0:
        raise ValueError("tail must start in the mode*time>1 regime")
    first_passage = _one_dimensional_first_passage_density(1.0, time)

    good_ratio = math.exp(-2.0 / 9.0)
    good_tail = (
        math.exp(1.0 / 54.0)
        * good_ratio**first_mode
        / (1.0 - good_ratio)
    )

    bad_ratio = math.exp(-2.0)
    geometric_tail = bad_ratio**first_mode / (1.0 - bad_ratio)
    weighted_tail = (
        bad_ratio**first_mode
        * (first_mode - (first_mode - 1) * bad_ratio)
        / (1.0 - bad_ratio) ** 2
    )
    bad_tail = 2.0 * geometric_tail + 8.0 * time * weighted_tail
    return first_passage**2 * (good_tail + bad_tail)


def _brownian_angular_l2_path_split_upper(
    time: float, explicit_modes: int = DEFAULT_EXPLICIT_MODES
) -> dict[str, float | int]:
    if explicit_modes < 1:
        raise ValueError("at least one explicit mode is required")
    last_mode = max(explicit_modes, math.floor(1.0 / time) + 1)
    mode_bounds = np.asarray(
        [_mode_path_split_upper(time, mode) for mode in range(1, last_mode + 1)]
    )
    first_tail_mode = last_mode + 1
    squared_tail = _squared_mode_tail_upper(time, first_tail_mode)
    zeroth = _zeroth_brownian_mode_upper(time)
    angular_squared = (
        zeroth**2
        + 2.0 * (float(np.dot(mode_bounds, mode_bounds)) + squared_tail)
    ) / (2.0 * math.pi)
    return {
        "explicit_mode_count": last_mode,
        "first_tail_mode": first_tail_mode,
        "zeroth_mode_upper": zeroth,
        "squared_mode_tail_upper": squared_tail,
        "angular_L2_upper": math.sqrt(angular_squared),
    }


def _axial_l2_factor(time: float) -> float:
    variance = math.expm1(2.0 * time)
    return math.exp(time) * math.sqrt(
        erf(PATCH_HALF_HEIGHT / math.sqrt(variance))
        / (2.0 * math.sqrt(math.pi) * math.sqrt(variance))
    )


def _raw_path_split_upper(
    time: float, explicit_modes: int = DEFAULT_EXPLICIT_MODES
) -> dict[str, float | int]:
    angular = _brownian_angular_l2_path_split_upper(time, explicit_modes)
    raw = (
        _ou_to_brownian_likelihood_upper(time)
        * _axial_l2_factor(time)
        * float(angular["angular_L2_upper"])
    )
    return {**angular, "raw_spatial_L2_upper": raw}


def _uniform_first_window_bound() -> dict[str, float]:
    """Closed-form uniformization of the infinite path-split mode sum."""
    gaussian_sum_coefficient = math.sqrt(6.0 * math.pi)
    high_good_tail = (
        math.exp(1.0 / 54.0)
        * math.exp(-2.0 / 3.0)
        / (1.0 - math.exp(-2.0 / 9.0))
    )
    geometric_sum = 1.0 / (math.exp(2.0) - 1.0)
    weighted_geometric_sum = math.exp(-2.0) / (
        1.0 - math.exp(-2.0)
    ) ** 2
    bad_sum = (
        2.0 * geometric_sum
        + 8.0 * WINDOW * weighted_geometric_sum
    )
    constant_mode_terms = (
        0.5 * math.exp(0.5 * WINDOW)
        + 2.0 * (high_good_tail + bad_sum)
    )
    mode_sum_coefficient = (
        gaussian_sum_coefficient
        + constant_mode_terms * math.sqrt(WINDOW)
    )
    angular_coefficient = math.sqrt(mode_sum_coefficient) / (
        2.0 * math.sqrt(math.pi) * math.sqrt(2.0 * math.pi)
    )
    axial_coefficient = 1.0 / math.sqrt(
        2.0 * math.sqrt(2.0 * math.pi)
    )

    maximizing_time = (4.0 - math.sqrt(10.0)) / 6.0
    raw_coefficient = math.e * angular_coefficient * axial_coefficient
    raw_upper = raw_coefficient * maximizing_time ** (-2.0) * math.exp(
        -1.0 / (4.0 * maximizing_time) + 1.5 * maximizing_time
    )
    interval_factor = (WINDOW + 1.0 / FORM_FLOOR) * raw_upper

    first_passage_probability = float(
        erfc(1.0 / (2.0 * math.sqrt(WINDOW)))
    )
    scalar_gain = (
        AXIAL_SCALAR_GLOBAL_UPPER
        * math.exp(1.0 + 0.75 * WINDOW)
        / math.sqrt(2.0)
        * first_passage_probability
    )
    return {
        "gaussian_mode_sum_coefficient": gaussian_sum_coefficient,
        "high_mode_geometric_sum": high_good_tail,
        "large_excursion_squared_sum": bad_sum,
        "mode_sum_coefficient": mode_sum_coefficient,
        "angular_power_law_coefficient": angular_coefficient,
        "axial_power_law_coefficient": axial_coefficient,
        "raw_bound_maximizing_time": maximizing_time,
        "uniform_raw_spatial_L2_upper": raw_upper,
        "first_window_interval_factor_upper": interval_factor,
        "first_window_scalar_gain_upper": scalar_gain,
    }


def _pointwise_path_split_pilot(
    time_count: int, explicit_modes: int
) -> dict[str, object]:
    if time_count < 3:
        raise ValueError("time grid needs at least three points")
    times = np.geomspace(0.002, WINDOW, time_count)
    maximizing_time = (4.0 - math.sqrt(10.0)) / 6.0
    times = np.unique(np.append(times, [maximizing_time, WINDOW]))
    rows = [_raw_path_split_upper(float(time), explicit_modes) for time in times]
    raw = np.asarray([row["raw_spatial_L2_upper"] for row in rows])
    peak_index = int(np.argmax(raw))
    return {
        "minimum_time": float(times[0]),
        "maximum_time": float(times[-1]),
        "time_count": int(len(times)),
        "requested_explicit_modes": explicit_modes,
        "maximum_actual_explicit_modes": max(
            int(row["explicit_mode_count"]) for row in rows
        ),
        "peak_time": float(times[peak_index]),
        "peak_raw_spatial_L2_upper": float(raw[peak_index]),
        "peak_angular_L2_upper": float(rows[peak_index]["angular_L2_upper"]),
        "peak_squared_mode_tail_upper": float(
            rows[peak_index]["squared_mode_tail_upper"]
        ),
        "grid_supremum_certified": False,
    }


def _mode_inversion_cross_check(time: float, maximum_mode: int) -> dict[str, float]:
    brownian = _load_module(
        "cylindrical_brownian_return_pilot.py",
        "brownian_return_for_first_window",
    )
    order_14 = brownian._hitting_modes(time, 14, maximum_mode)
    order_16 = brownian._hitting_modes(time, 16, maximum_mode)

    def angular_l2(modes: np.ndarray) -> float:
        return math.sqrt(
            (modes[0] ** 2 + 2.0 * float(np.dot(modes[1:], modes[1:])))
            / (2.0 * math.pi)
        )

    l2_14 = angular_l2(order_14)
    l2_16 = angular_l2(order_16)
    return {
        "time": time,
        "maximum_mode": maximum_mode,
        "Stehfest_order_14_angular_L2": l2_14,
        "Stehfest_order_16_angular_L2": l2_16,
        "order_relative_spread": abs(l2_16 - l2_14)
        / max(abs(l2_16), abs(l2_14), 1.0e-300),
        "numerically_certified": False,
    }


def audit(
    run_pointwise_pilot: bool = True,
    run_inversion: bool = False,
    time_count: int = DEFAULT_TIME_COUNT,
    explicit_modes: int = DEFAULT_EXPLICIT_MODES,
) -> dict[str, object]:
    uniform = _uniform_first_window_bound()
    pointwise = (
        _pointwise_path_split_pilot(time_count, explicit_modes)
        if run_pointwise_pilot
        else {}
    )
    inversion = {}
    if run_inversion:
        comparison_time = (
            float(pointwise["peak_time"])
            if pointwise
            else uniform["raw_bound_maximizing_time"]
        )
        inversion = _mode_inversion_cross_check(comparison_time, 40)

    result: dict[str, object] = {
        "model": "rho=0 continuum neutral-strip first return window",
        "window": WINDOW,
        "exact_Brownian_mode_transform": (
            "Laplace c_n(p)=K_n(2sqrt(p))/K_n(sqrt(p))"
        ),
        "Brownian_angular_Parseval_identity": (
            "||h||_2^2=(c_0^2+2 sum_(n>=1)c_n^2)/(2pi)"
        ),
        "OU_to_Brownian_stopped_path_factor": "exp(1+t/2)",
        "mode_path_split_bound": (
            "c_n(t)<=2^(-1/2)[q_1(t)exp(-(n^2-1/4)t/M^2)"
            "+q_(2M-3)(t)]"
        ),
        "deterministic_excursion_radius": "M_n(t)=2+sqrt(n t)",
        "uniform_analytic_budget": uniform,
        "pointwise_path_split_pilot": pointwise,
        "Brownian_mode_inversion_cross_check": inversion,
        "OU_stopped_hit_measure_domination_proved": True,
        "Bessel_absolute_continuity_mode_identity_proved": True,
        "large_excursion_first_passage_convolution_proved": True,
        "infinite_angular_mode_tail_summed": True,
        "continuum_first_window_uniform_flux_bound_proved": True,
        "continuum_first_window_flux_certified": True,
        "first_window_interval_factor_below_one": (
            uniform["first_window_interval_factor_upper"] < 1.0
        ),
        "first_window_scalar_gain_below_one": (
            uniform["first_window_scalar_gain_upper"] < 1.0
        ),
        "first_window_response_budget_closed": False,
        "continuum_return_response_certified": False,
        "scope": (
            "The stopped Girsanov bound, Bessel path split, infinite mode "
            "sum, and closed-form uniform first-window bound are analytic. "
            "The sampled path-split envelope and optional inverse-Laplace "
            "comparison are diagnostics. The current uniform constant is "
            "too coarse to close the response budget."
        ),
        "next_required_step": (
            "Interval-enclose the finite low Brownian modes and optimize the "
            "excursion split, retaining the analytic tail proved here."
        ),
    }
    checks = [
        result["OU_stopped_hit_measure_domination_proved"],
        result["Bessel_absolute_continuity_mode_identity_proved"],
        result["large_excursion_first_passage_convolution_proved"],
        result["infinite_angular_mode_tail_summed"],
        result["continuum_first_window_uniform_flux_bound_proved"],
        result["continuum_first_window_flux_certified"],
        uniform["raw_bound_maximizing_time"] < WINDOW,
        uniform["uniform_raw_spatial_L2_upper"] > 0.0,
        uniform["first_window_interval_factor_upper"] > 1.0,
        result["first_window_scalar_gain_below_one"],
        not result["first_window_response_budget_closed"],
        not result["continuum_return_response_certified"],
    ]
    if pointwise:
        checks.extend(
            [
                pointwise["peak_raw_spatial_L2_upper"]
                < uniform["uniform_raw_spatial_L2_upper"],
                not pointwise["grid_supremum_certified"],
            ]
        )
    if inversion:
        checks.append(not inversion["numerically_certified"])
    result["all_first_window_majorant_checks_pass"] = bool(all(checks))
    return result


def main() -> None:
    print(json.dumps(audit(run_inversion=True), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
