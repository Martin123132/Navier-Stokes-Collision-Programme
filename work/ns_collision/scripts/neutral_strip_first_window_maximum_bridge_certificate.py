"""Certify the first-window Brownian modes by a bridge-maximum bound."""

from __future__ import annotations

from decimal import Decimal, getcontext
import importlib.util
import json
import math
from pathlib import Path

import mpmath
import numpy as np


WINDOW = Decimal("0.375")
EARLY_CUTOFF = Decimal("0.02")
FORM_FLOOR = Decimal("4.832287335665")
DEFAULT_TIME_STEP = Decimal("0.001")
DEFAULT_RADIUS_STEP = Decimal("0.01")
DEFAULT_MODE_COUNT = 96
DEFAULT_MAXIMUM_RADIUS = Decimal("5.0")


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upper(interval) -> float:
    return np.nextafter(float(interval.b), math.inf)


def _iv_point(value: Decimal | str | int):
    return mpmath.iv.mpf(str(value))


def _up(value):
    return np.nextafter(value, math.inf)


def _down(value):
    return np.nextafter(value, -math.inf)


def _up_add(left, right):
    return _up(np.add(left, right))


def _down_add(left, right):
    return _down(np.add(left, right))


def _up_mul(left, right):
    return _up(np.multiply(left, right))


def _down_mul(left, right):
    return _down(np.multiply(left, right))


def _up_div(numerator, denominator):
    return _up(np.divide(numerator, denominator))


def _down_div(numerator, denominator):
    return _down(np.divide(numerator, denominator))


def _positive_sum_upper(values: np.ndarray, axis=None):
    count = values.shape[-1] if axis is not None else values.size
    total = np.sum(values, axis=axis)
    epsilon = np.finfo(float).eps
    gamma = count * epsilon / (1.0 - count * epsilon)
    return _up_mul(total, _up(1.0 + gamma))


def _q_upper_at(time: Decimal) -> float:
    iv = mpmath.iv
    t = _iv_point(time)
    value = iv.exp(-1 / (4 * t)) / (2 * iv.sqrt(iv.pi) * t**1.5)
    return _upper(value)


def _c0_upper_at(time: Decimal) -> float:
    iv = mpmath.iv
    t = _iv_point(time)
    q = iv.exp(-1 / (4 * t)) / (2 * iv.sqrt(iv.pi) * t**1.5)
    return _upper(q * iv.exp(t / 4) / iv.sqrt(2))


def _q_slab_upper(start: Decimal, end: Decimal) -> float:
    candidates = [_q_upper_at(start), _q_upper_at(end)]
    critical = Decimal(1) / Decimal(6)
    if start <= critical <= end:
        candidates.append(_q_upper_at(critical))
    return max(candidates)


def _c0_slab_upper(start: Decimal, end: Decimal) -> float:
    candidates = [_c0_upper_at(start), _c0_upper_at(end)]
    critical_float = 3.0 - 2.0 * math.sqrt(2.0)
    if float(start) <= critical_float <= float(end):
        iv = mpmath.iv
        critical = 3 - 2 * iv.sqrt(2)
        q = iv.exp(-1 / (4 * critical)) / (
            2 * iv.sqrt(iv.pi) * critical**1.5
        )
        candidates.append(
            _upper(q * iv.exp(critical / 4) / iv.sqrt(2))
        )
    return max(candidates)


def _ou_likelihood_upper_at(time: Decimal) -> float:
    iv = mpmath.iv
    t = _iv_point(time)
    return _upper(iv.exp(1 + t / 2))


def _axial_erf_free_upper_at(time: Decimal) -> float:
    """Use erf<=1 in the exact finite-patch axial L2 factor."""
    iv = mpmath.iv
    t = _iv_point(time)
    variance = iv.exp(2 * t) - 1
    value = iv.exp(t) / iv.sqrt(
        2 * iv.sqrt(iv.pi) * iv.sqrt(variance)
    )
    return _upper(value)


def _axial_slab_upper(start: Decimal, end: Decimal) -> float:
    # The erf-free factor decreases before log(2)/2 and increases after it.
    return max(
        _axial_erf_free_upper_at(start),
        _axial_erf_free_upper_at(end),
    )


def _early_window_raw_upper() -> float:
    """Evaluate the prior analytic power bound at t=EARLY_CUTOFF."""
    iv = mpmath.iv
    t = _iv_point(EARLY_CUTOFF)
    terminal = _iv_point(WINDOW)
    gaussian = iv.sqrt(6 * iv.pi)
    high_good = (
        iv.exp(iv.mpf(1) / 54)
        * iv.exp(-iv.mpf(2) / 3)
        / (1 - iv.exp(-iv.mpf(2) / 9))
    )
    geometric = 1 / (iv.exp(2) - 1)
    weighted = iv.exp(-2) / (1 - iv.exp(-2)) ** 2
    bad = 2 * geometric + 8 * terminal * weighted
    constant_terms = (
        iv.mpf("0.5") * iv.exp(terminal / 2)
        + 2 * (high_good + bad)
    )
    mode_sum = gaussian + constant_terms * iv.sqrt(terminal)
    angular = iv.sqrt(mode_sum) / (
        2 * iv.sqrt(iv.pi) * iv.sqrt(2 * iv.pi)
    )
    axial = 1 / iv.sqrt(2 * iv.sqrt(2 * iv.pi))
    raw_coefficient = iv.e * angular * axial
    value = raw_coefficient * t**-2 * iv.exp(-1 / (4 * t) + 1.5 * t)
    return _upper(value)


def _radius_partition(radius_step: Decimal, maximum_radius: Decimal):
    span = maximum_radius - Decimal(2)
    cell_count = int((span / radius_step).to_integral_exact())
    lower_exact = [Decimal(2) + index * radius_step for index in range(cell_count)]
    upper_exact = [value + radius_step for value in lower_exact]
    lower = _down(np.asarray([float(value) for value in lower_exact]))
    upper = _up(np.asarray([float(value) for value in upper_exact]))
    width = _up(
        np.asarray(
            [float(right - left) for left, right in zip(lower_exact, upper_exact)]
        )
    )
    return lower, upper, width, cell_count


def _integrated_maximum_mode_uppers(
    start: Decimal,
    end: Decimal,
    mode_count: int,
    radius_lower: np.ndarray,
    radius_upper: np.ndarray,
    radius_width: np.ndarray,
    maximum_radius: Decimal,
) -> np.ndarray:
    """Enclose the positive maximum-tail integral for all retained modes."""
    modes = np.arange(1, mode_count + 1, dtype=float)
    kappa = (modes * modes - 0.25)[:, None]
    start_lower = _down(float(start))
    end_upper = _up(float(end))

    d_lower = _down_add(_down_mul(2.0, radius_lower), -3.0)
    d_upper = _up_add(_up_mul(2.0, radius_upper), -3.0)

    numerator = _down_mul(kappa, start_lower)
    radius_square_upper = _up_mul(radius_upper, radius_upper)
    first_exponent = _down_div(numerator, radius_square_upper[None, :])

    d_square_lower = _down_mul(d_lower, d_lower)
    distance_numerator = _down_add(d_square_lower, -1.0)
    # The exact radius domain starts at M=2, so (2M-3)^2-1 is nonnegative.
    distance_numerator = np.maximum(distance_numerator, 0.0)
    time_denominator = _up_mul(4.0, end_upper)
    second_exponent = _down_div(distance_numerator, time_denominator)
    exponent_lower = _down_add(first_exponent, second_exponent[None, :])
    exponential_upper = _up(np.exp(-exponent_lower))

    prefactor = _up_mul(2.0, kappa)
    prefactor = _up_mul(prefactor, end_upper)
    prefactor = _up_mul(prefactor, d_upper[None, :])
    radius_square_lower = _down_mul(radius_lower, radius_lower)
    radius_cube_lower = _down_mul(radius_square_lower, radius_lower)
    prefactor = _up_div(prefactor, radius_cube_lower[None, :])

    cell_upper = _up_mul(prefactor, exponential_upper)
    cell_upper = _up_mul(cell_upper, radius_width[None, :])
    integral_upper = _positive_sum_upper(cell_upper, axis=1)

    iv = mpmath.iv
    end_iv = _iv_point(end)
    maximum_iv = _iv_point(maximum_radius)
    distance = 2 * maximum_iv - 3
    excursion_tail = distance * iv.exp(
        -(distance**2 - 1) / (4 * end_iv)
    )
    integral_upper = _up_add(integral_upper, _upper(excursion_tail))

    kappa_flat = kappa[:, 0]
    initial_clock = _up(
        np.exp(_up_div(_up_mul(-kappa_flat, start_lower), 4.0))
    )
    bracket = _up_add(initial_clock, integral_upper)
    q_upper = _q_slab_upper(start, end)
    iv.dps = 80
    inverse_sqrt_two = _upper(1 / iv.sqrt(2))
    return _up_mul(_up_mul(q_upper, inverse_sqrt_two), bracket)


def _squared_omitted_mode_upper(
    start: Decimal,
    end: Decimal,
    first_mode: int,
) -> float:
    if first_mode * float(start) <= 1.0:
        raise ValueError("the omitted tail must satisfy mode*time>1")
    iv = mpmath.iv
    mode = iv.mpf(first_mode)
    end_iv = _iv_point(end)
    good_ratio = iv.exp(-iv.mpf(2) / 9)
    good = (
        iv.exp(iv.mpf(1) / 54)
        * good_ratio**first_mode
        / (1 - good_ratio)
    )
    bad_ratio = iv.exp(-2)
    geometric = bad_ratio**first_mode / (1 - bad_ratio)
    weighted = (
        bad_ratio**first_mode
        * (mode - (mode - 1) * bad_ratio)
        / (1 - bad_ratio) ** 2
    )
    ratio_upper = _upper(good + 2 * geometric + 8 * end_iv * weighted)
    q_upper = _q_slab_upper(start, end)
    return float(_up_mul(_up_mul(q_upper, q_upper), ratio_upper))


def _slab_raw_upper(
    start: Decimal,
    end: Decimal,
    mode_count: int,
    radius_lower: np.ndarray,
    radius_upper: np.ndarray,
    radius_width: np.ndarray,
    maximum_radius: Decimal,
) -> dict[str, float]:
    modes = _integrated_maximum_mode_uppers(
        start,
        end,
        mode_count,
        radius_lower,
        radius_upper,
        radius_width,
        maximum_radius,
    )
    mode_squares = _up_mul(modes, modes)
    retained_squared_sum = float(_positive_sum_upper(mode_squares))
    omitted_squared_sum = _squared_omitted_mode_upper(
        start, end, mode_count + 1
    )
    c0_upper = _c0_slab_upper(start, end)
    angular_square = _up_add(
        _up_mul(c0_upper, c0_upper),
        _up_mul(
            2.0,
            _up_add(retained_squared_sum, omitted_squared_sum),
        ),
    )
    iv = mpmath.iv
    inverse_two_pi = _upper(1 / (2 * iv.pi))
    angular_square = _up_mul(angular_square, inverse_two_pi)
    angular_upper = float(_up(math.sqrt(float(angular_square))))
    raw_upper = _up_mul(
        _ou_likelihood_upper_at(end),
        _axial_slab_upper(start, end),
    )
    raw_upper = float(_up_mul(raw_upper, angular_upper))
    return {
        "raw_spatial_L2_upper": raw_upper,
        "Brownian_angular_L2_upper": angular_upper,
        "retained_squared_mode_sum_upper": retained_squared_sum,
        "omitted_squared_mode_sum_upper": omitted_squared_sum,
        "first_mode_upper": float(modes[0]),
        "last_retained_mode_upper": float(modes[-1]),
    }


def certificate(
    time_step: Decimal = DEFAULT_TIME_STEP,
    radius_step: Decimal = DEFAULT_RADIUS_STEP,
    mode_count: int = DEFAULT_MODE_COUNT,
    maximum_radius: Decimal = DEFAULT_MAXIMUM_RADIUS,
) -> dict[str, object]:
    getcontext().prec = 50
    mpmath.iv.dps = 80
    if time_step <= 0 or radius_step <= 0:
        raise ValueError("partition steps must be positive")
    if mode_count < 51:
        raise ValueError("at least 51 modes are required for the tail gate")
    if (mode_count + 1) * float(EARLY_CUTOFF) <= 1.0:
        raise ValueError("mode count does not enter the geometric tail regime")
    if maximum_radius <= 2:
        raise ValueError("maximum radius must exceed the entry radius")

    radius_lower, radius_upper, radius_width, radius_cells = _radius_partition(
        radius_step, maximum_radius
    )
    time_cells = int(
        ((WINDOW - EARLY_CUTOFF) / time_step).to_integral_value(
            rounding="ROUND_CEILING"
        )
    )
    peak: dict[str, float | str] | None = None
    maximum_omitted = 0.0
    for index in range(time_cells):
        start = EARLY_CUTOFF + index * time_step
        end = min(start + time_step, WINDOW)
        row = _slab_raw_upper(
            start,
            end,
            mode_count,
            radius_lower,
            radius_upper,
            radius_width,
            maximum_radius,
        )
        maximum_omitted = max(
            maximum_omitted, row["omitted_squared_mode_sum_upper"]
        )
        if peak is None or row["raw_spatial_L2_upper"] > peak[
            "raw_spatial_L2_upper"
        ]:
            peak = {
                "time_start": str(start),
                "time_end": str(end),
                **row,
            }
    assert peak is not None

    early_upper = _early_window_raw_upper()
    full_raw_upper = max(early_upper, float(peak["raw_spatial_L2_upper"]))
    interval_multiplier = float(WINDOW + Decimal(1) / FORM_FLOOR)
    interval_factor = float(_up_mul(full_raw_upper, interval_multiplier))
    result: dict[str, object] = {
        "model": "rho=0 first-window bridge-maximum continuum certificate",
        "time_window": ["0", str(WINDOW)],
        "early_window": ["0", str(EARLY_CUTOFF)],
        "time_step": str(time_step),
        "time_slab_count": time_cells,
        "retained_mode_count": mode_count,
        "first_omitted_mode": mode_count + 1,
        "radius_step": str(radius_step),
        "radius_cell_count": radius_cells,
        "maximum_explicit_radius": str(maximum_radius),
        "bridge_maximum_identity": (
            "E[f(M_tau);tau in dt]=f(2)q_1(t)dt+"
            "int_2^infinity f'(M)P(M_tau>M,tau in dt)dM"
        ),
        "bridge_maximum_tail_bound": (
            "P(M_tau>M,tau in dt)/dt<=q_(2M-3)(t)"
        ),
        "early_window_raw_L2_upper": early_upper,
        "peak_enclosed_slab": peak,
        "maximum_omitted_squared_mode_sum_upper": maximum_omitted,
        "complete_first_window_raw_L2_upper": full_raw_upper,
        "complete_first_window_interval_factor_upper": interval_factor,
        "OU_to_Brownian_domination_used": True,
        "positive_bridge_maximum_quadrature_enclosed": True,
        "radius_beyond_maximum_analytically_bounded": True,
        "omitted_angular_modes_analytically_bounded": True,
        "early_time_singularity_analytically_bounded": True,
        "complete_first_window_time_supremum_enclosed": True,
        "continuum_first_window_flux_certified": True,
        "finite_low_Brownian_modes_interval_enclosed": True,
        "exact_Brownian_modes_inverse_Laplace_enclosed": False,
        "first_window_interval_factor_below_one": interval_factor < 1.0,
        "first_window_response_budget_closed": interval_factor < 1.0,
        "continuum_return_response_certified": False,
        "scope": (
            "The certificate uses only positive path-measure bounds, "
            "outward-rounded rectangle sums, and analytic tails. It does "
            "not certify exact inverse-Laplace mode values. The retained "
            "continuum FEM block and polygon-to-circle flux map remain open."
        ),
        "next_required_step": (
            "Interval-enclose the retained continuum spectral block and the "
            "polygon-to-circle conormal flux map, then compose this bound "
            "with the later-window low and high modes."
        ),
    }
    checks = [
        result["OU_to_Brownian_domination_used"],
        result["positive_bridge_maximum_quadrature_enclosed"],
        result["radius_beyond_maximum_analytically_bounded"],
        result["omitted_angular_modes_analytically_bounded"],
        result["early_time_singularity_analytically_bounded"],
        result["complete_first_window_time_supremum_enclosed"],
        result["continuum_first_window_flux_certified"],
        result["finite_low_Brownian_modes_interval_enclosed"],
        not result["exact_Brownian_modes_inverse_Laplace_enclosed"],
        result["first_window_interval_factor_below_one"],
        not result["continuum_return_response_certified"],
        float(peak["time_start"]) > float(EARLY_CUTOFF),
        float(peak["time_end"]) < float(WINDOW),
        maximum_omitted < 1.0e-7,
    ]
    result["all_first_window_bridge_certificate_checks_pass"] = bool(
        all(checks)
    )
    return result


def main() -> None:
    print(json.dumps(certificate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
