#!/usr/bin/env python3
"""Certify Chebyshev coefficients for the 3/80 time-slab step."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import mpmath
import numpy as np
from scipy.special import ive


DEFAULT_SCALING_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_chebyshev_scaling_coefficients_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_within_window_substep_coefficients_v1.json"
)
SUBSTEP = 0.0375
DEGREE = 112
SCALING_LOWER = 1.9
SCALING_UPPER = 8000.0
TAIL_DIRECT_END = 360
INTERVAL_DPS = 80
UNIT_ROUNDOFF = 2.0**-53


def _set_below_normal_priority() -> bool:
    try:
        import psutil

        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return process.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
    except Exception:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _gamma(operation_count: int) -> float:
    product = operation_count * UNIT_ROUNDOFF
    if product >= 0.01:
        raise ArithmeticError("roundoff operation count is too large")
    return float(np.nextafter(product / (1.0 - product), math.inf))


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _down(value: float) -> float:
    return float(np.nextafter(float(value), -math.inf))


def _iv_bounds(value) -> tuple[float, float]:
    return _down(float(value.a)), _up(float(value.b))


def _iv_nonnegative_upper(value):
    upper = _up(float(value.b))
    return mpmath.iv.mpf(["0.0", repr(upper)])


def _scaled_bessel_positive_series(order: int, argument):
    """Enclose exp(-a) I_order(a) by a positive power series."""
    term = (
        mpmath.iv.exp(-argument)
        * (argument / 2) ** order
        / mpmath.iv.factorial(order)
    )
    total = term
    index = 0
    while True:
        ratio = argument**2 / (
            4 * (index + 1) * (index + order + 1)
        )
        term *= ratio
        index += 1
        total += term
        next_ratio = argument**2 / (
            4 * (index + 1) * (index + order + 1)
        )
        if float(next_ratio.b) < 0.5 and float(term.b) < 1.0e-115:
            first_omitted = term * next_ratio
            remainder = first_omitted / (1 - next_ratio)
            total += _iv_nonnegative_upper(remainder)
            return total, index, _iv_bounds(remainder)[1]
        if index > 10000:
            raise ArithmeticError("positive Bessel series did not terminate")


def _scaled_bessel_coefficients(argument):
    first, first_terms, first_remainder = (
        _scaled_bessel_positive_series(0, argument)
    )
    second, second_terms, second_remainder = (
        _scaled_bessel_positive_series(1, argument)
    )
    values = [first, second]
    for order in range(1, DEGREE):
        following = (
            values[order - 1]
            - (2 * order / argument) * values[order]
        )
        if float(following.a) <= 0.0:
            raise ArithmeticError(
                f"Bessel forward interval lost positivity at {order + 1}"
            )
        values.append(following)
    return values, {
        "order_0_terms": first_terms,
        "order_1_terms": second_terms,
        "order_0_series_remainder_upper": first_remainder,
        "order_1_series_remainder_upper": second_remainder,
    }


def _scaled_bessel_order_tail(argument):
    total = mpmath.iv.mpf(["0.0", "0.0"])
    term_counts: list[int] = []
    maximum_series_remainder = 0.0
    last_value = None
    for order in range(DEGREE + 1, TAIL_DIRECT_END + 1):
        value, term_count, remainder_upper = (
            _scaled_bessel_positive_series(order, argument)
        )
        total += value
        term_counts.append(term_count)
        maximum_series_remainder = max(
            maximum_series_remainder, remainder_upper
        )
        last_value = value
    if last_value is None:
        raise AssertionError("empty direct Bessel tail")

    ratio_bound = argument / (2 * (TAIL_DIRECT_END + 1))
    if float(ratio_bound.b) >= 1.0:
        raise ArithmeticError("terminal Bessel-order ratio is not geometric")
    geometric_remainder = last_value * ratio_bound / (1 - ratio_bound)
    total += _iv_nonnegative_upper(geometric_remainder)
    return total, {
        "first_direct_order": DEGREE + 1,
        "last_direct_order": TAIL_DIRECT_END,
        "minimum_series_term_count": min(term_counts),
        "maximum_series_term_count": max(term_counts),
        "maximum_individual_series_remainder_upper": (
            maximum_series_remainder
        ),
        "post_direct_ratio_upper": _iv_bounds(ratio_bound)[1],
        "post_direct_geometric_remainder_upper": _iv_bounds(
            geometric_remainder
        )[1],
    }


def audit(scaling_result_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    scaling_result = json.loads(
        scaling_result_path.read_text(encoding="ascii")
    )
    if not scaling_result["all_scaling_coefficient_checks_pass"]:
        raise RuntimeError("production scaling premise is not certified")
    matrix_scaling = scaling_result["matrix_scaling"]
    if not matrix_scaling[
        "exact_spectrum_inside_scaling_interval_certified"
    ]:
        raise RuntimeError("stored spectrum is not certified")
    if matrix_scaling["scaling_interval"] != [
        SCALING_LOWER,
        SCALING_UPPER,
    ]:
        raise RuntimeError("Chebyshev scaling interval changed")

    mpmath.iv.dps = INTERVAL_DPS
    time_interval = mpmath.iv.mpf(["0.0375", "0.0375"])
    lower_interval = mpmath.iv.mpf(["1.9", "1.9"])
    upper_interval = mpmath.iv.mpf(["8000.0", "8000.0"])
    argument = time_interval * (upper_interval - lower_interval) / 2
    damping = mpmath.iv.exp(-time_interval * lower_interval)
    scaled_bessel, base_series = _scaled_bessel_coefficients(argument)

    scipy_coefficients = (
        np.exp(-SUBSTEP * SCALING_LOWER)
        * ive(
            np.arange(DEGREE + 1),
            SUBSTEP * (SCALING_UPPER - SCALING_LOWER) / 2,
        )
    )
    scipy_coefficients[1:] *= 2.0
    scipy_coefficients[1::2] *= -1.0

    rows = []
    implementation_errors: list[float] = []
    maximum_interval_width = 0.0
    all_scipy_contained = True
    for order, scaled_value in enumerate(scaled_bessel):
        coefficient = damping * scaled_value
        if order:
            coefficient *= 2
        if order % 2:
            coefficient = -coefficient
        lower, upper = _iv_bounds(coefficient)
        central = float(scipy_coefficients[order])
        contained = bool(lower <= central <= upper)
        all_scipy_contained = all_scipy_contained and contained
        implementation_error = _up(
            max(abs(central - lower), abs(central - upper))
        )
        implementation_errors.append(implementation_error)
        maximum_interval_width = max(
            maximum_interval_width, upper - lower
        )
        rows.append(
            {
                "order": order,
                "lower": lower,
                "upper": upper,
                "scipy_central": central,
                "scipy_central_contained": contained,
                "scipy_to_exact_coefficient_error_upper": (
                    implementation_error
                ),
            }
        )

    coefficient_l1_error = _up(
        math.fsum(implementation_errors)
        / (1.0 - _gamma(len(implementation_errors) + 8))
    )
    scaled_tail, tail_details = _scaled_bessel_order_tail(argument)
    coefficient_tail = 2 * damping * scaled_tail
    tail_lower, tail_upper = _iv_bounds(coefficient_tail)
    all_intervals_ordered = all(
        row["lower"] <= row["upper"] for row in rows
    )
    implementation_discrepancy_certified = bool(
        coefficient_l1_error < 1.0e-14
    )
    coefficients_certified = bool(
        all_intervals_ordered
        and implementation_discrepancy_certified
        and maximum_interval_width < 1.0e-15
        and tail_upper < 3.0e-19
    )
    checks = [
        priority_set,
        matrix_scaling[
            "exact_spectrum_inside_scaling_interval_certified"
        ],
        len(rows) == DEGREE + 1,
        all_intervals_ordered,
        implementation_discrepancy_certified,
        coefficient_l1_error < 1.0e-14,
        maximum_interval_width < 1.0e-15,
        tail_upper < 3.0e-19,
        coefficients_certified,
    ]
    return {
        "kind": (
            "neutral_strip_within_window_substep_coefficients_certificate"
        ),
        "model": (
            "directed degree-112 Chebyshev-Bessel coefficients for a "
            "3/80 stored-chain semigroup substep"
        ),
        "below_normal_priority_set": priority_set,
        "premise_artifacts": {
            "production_scaling_result": str(scaling_result_path),
            "production_scaling_result_sha256": _sha256_file(
                scaling_result_path
            ),
        },
        "matrix_scaling": {
            "scaling_interval": [SCALING_LOWER, SCALING_UPPER],
            "exact_spectrum_inside_scaling_interval_certified": True,
            "exact_stored_normalized_generator_gershgorin_upper": (
                matrix_scaling[
                    "exact_stored_normalized_generator_gershgorin_upper"
                ]
            ),
            "normalized_generator_construction_error_upper": (
                matrix_scaling[
                    "normalized_generator_construction_error_upper"
                ]
            ),
        },
        "coefficient_intervals": {
            "interval_decimal_digits": INTERVAL_DPS,
            "time": SUBSTEP,
            "degree": DEGREE,
            "substeps_per_3_over_8_window": 10,
            "finite_window_count": 15,
            "total_substeps_through_time_6": 160,
            "bessel_argument_interval": _iv_bounds(argument),
            "damping_interval": _iv_bounds(damping),
            "base_positive_series": base_series,
            "rows": rows,
            "all_scipy_central_values_contained": all_scipy_contained,
            "all_exact_coefficient_intervals_ordered": (
                all_intervals_ordered
            ),
            "scipy_implementation_discrepancy_certified": (
                implementation_discrepancy_certified
            ),
            "scipy_coefficient_l1_error_upper": coefficient_l1_error,
            "maximum_binary64_interval_width": maximum_interval_width,
            "tail": {
                **tail_details,
                "lower": tail_lower,
                "upper": tail_upper,
            },
            "degree_112_exact_coefficients_and_infinite_tail_certified": (
                coefficients_certified
            ),
        },
        "sparse_recurrence_roundoff_enclosed": False,
        "within_window_suprema_certified": False,
        "post_terminal_time_tail_certified": False,
        "screen_updated": False,
        "checks": checks,
        "all_substep_coefficient_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_step": (
            "Use the universal Chebyshev U-polynomial recurrence bound for "
            "the degree-112 substep, then propagate the 112 actual sources "
            "on the 160-point grid with atomic checkpoints."
        ),
        "scope": (
            "This certificate closes only the scalar coefficients and tail "
            "for the within-window substep. It does not certify the sparse "
            "matrix action, time-slab suprema, terminal tail, continuum "
            "transfer, or Navier-Stokes regularity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scaling-result",
        type=Path,
        default=DEFAULT_SCALING_RESULT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = audit(arguments.scaling_result)
    _atomic_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_substep_coefficient_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
