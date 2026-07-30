#!/usr/bin/env python3
"""Certify Chebyshev scaling and degree-320 coefficient intervals."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import mpmath
import numpy as np
from scipy.sparse import diags
from scipy.special import ive


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EIGEN_CACHE = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k241_reference_eigensystem_v1.npz"
)
DEFAULT_TWO_BLOCK_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_modified_two_block_leakage_v1.json"
)
DEFAULT_PILOT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_boundary_leakage_chebyshev_pilot_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_chebyshev_scaling_coefficients_v1.json"
)
WINDOW = 0.375
DEGREE = 320
SCALING_LOWER = 1.9
SCALING_UPPER = 8000.0
TAIL_DIRECT_END = 752
INTERVAL_DPS = 80
UNIT_ROUNDOFF = 2.0**-53


def _load_module(filename: str, module_name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    return float(
        np.nextafter(product / (1.0 - product), math.inf)
    )


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
    """Enclose exp(-a) I_order(a) by its positive power series."""
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
        if (
            index > 1000
            and float(next_ratio.b) < 0.5
            and float(term.b) < 1.0e-115
        ):
            first_omitted = term * next_ratio
            remainder = first_omitted / (1 - next_ratio)
            total += _iv_nonnegative_upper(remainder)
            return total, index, _iv_bounds(remainder)[1]


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
    return (
        values,
        {
            "order_0_terms": first_terms,
            "order_1_terms": second_terms,
            "order_0_series_remainder_upper": first_remainder,
            "order_1_series_remainder_upper": second_remainder,
        },
    )


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

    # From the positive I_k series,
    # I_(k+1)(a)/I_k(a) is a weighted average of
    # a/(2(j+k+1)), hence is at most a/(2(k+1)).
    ratio_bound = argument / (2 * (TAIL_DIRECT_END + 1))
    if float(ratio_bound.b) >= 1.0:
        raise ArithmeticError("terminal Bessel order ratio is not geometric")
    geometric_remainder = (
        last_value * ratio_bound / (1 - ratio_bound)
    )
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


def _global_two_block_floor(
    low_floor: float,
    high_floor: float,
    coupling: float,
) -> float:
    midpoint = 0.5 * (low_floor + high_floor)
    half_gap = 0.5 * (high_floor - low_floor)
    return midpoint - math.hypot(half_gap, coupling)


def audit(
    eigen_cache: Path,
    two_block_result_path: Path,
    pilot_result_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    two_block = json.loads(
        two_block_result_path.read_text(encoding="ascii")
    )
    pilot = json.loads(pilot_result_path.read_text(encoding="ascii"))
    if not two_block["all_modified_two_block_leakage_checks_pass"]:
        raise RuntimeError("two-block premise is not certified")
    if (
        pilot["status"] != "complete"
        or not pilot["all_pilot_integrity_checks_pass"]
    ):
        raise RuntimeError("completed Chebyshev pilot is unavailable")
    if (
        pilot["premise_artifacts"]["two_block_result_sha256"]
        != _sha256_file(two_block_result_path)
    ):
        raise RuntimeError("pilot/two-block premise hash mismatch")

    base_module = _load_module(
        "neutral_strip_modified_complement_inertia_schur_certificate.py",
        "chebyshev_scaling_coefficient_base",
    )
    mass, stiffness, _, matrix_metadata = (
        base_module._assemble_modified_pencil(0.06, eigen_cache)
    )
    mass_diagonal = np.asarray(mass.diagonal(), dtype=np.float64)
    inverse_square_root_mass = 1.0 / np.sqrt(mass_diagonal)
    normalized_generator = (
        diags(inverse_square_root_mass)
        @ stiffness
        @ diags(inverse_square_root_mass)
    ).tocsr()
    normalized_generator = (
        0.5 * (normalized_generator + normalized_generator.transpose())
    ).tocsr()
    maximum_row_nonzeros = int(
        np.max(np.diff(normalized_generator.indptr))
    )
    central_row_sums = np.asarray(
        abs(normalized_generator).sum(axis=1)
    ).ravel()
    central_gershgorin = float(np.max(central_row_sums))

    entry_relative_error = _gamma(20)
    row_sum_error = _gamma(maximum_row_nonzeros + 8)
    exact_gershgorin_upper = _up(
        central_gershgorin
        / (1.0 - row_sum_error)
        / (1.0 - entry_relative_error)
    )
    normalized_matrix_error_upper = _up(
        exact_gershgorin_upper * entry_relative_error
    )

    parameters = two_block["certified_parameters"]
    global_floor = _global_two_block_floor(
        float(parameters["low_block_floor"]),
        float(parameters["high_block_floor"]),
        float(parameters["off_block_coupling_upper"]),
    )
    scaling_interval_certified = bool(
        global_floor > SCALING_LOWER
        and exact_gershgorin_upper < SCALING_UPPER
        and matrix_metadata["mass_diagonal_strictly_positive"]
        and matrix_metadata["stiffness_exactly_symmetric"]
    )

    mpmath.iv.dps = INTERVAL_DPS
    time_interval = mpmath.iv.mpf(["0.375", "0.375"])
    lower_interval = mpmath.iv.mpf(["1.9", "1.9"])
    upper_interval = mpmath.iv.mpf(["8000.0", "8000.0"])
    radius = (upper_interval - lower_interval) / 2
    argument = time_interval * radius
    damping = mpmath.iv.exp(-time_interval * lower_interval)
    scaled_bessel, base_series = _scaled_bessel_coefficients(argument)
    coefficient_rows = []
    scipy_coefficients = (
        np.exp(-WINDOW * SCALING_LOWER)
        * ive(
            np.arange(DEGREE + 1),
            WINDOW * (SCALING_UPPER - SCALING_LOWER) / 2,
        )
    )
    scipy_coefficients[1:] *= 2.0
    scipy_coefficients[1::2] *= -1.0
    all_scipy_contained = True
    scipy_coefficient_error_terms: list[float] = []
    maximum_interval_width = 0.0
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
        implementation_error_upper = _up(
            max(abs(central - lower), abs(central - upper))
        )
        scipy_coefficient_error_terms.append(
            implementation_error_upper
        )
        maximum_interval_width = max(
            maximum_interval_width, upper - lower
        )
        coefficient_rows.append(
            {
                "order": order,
                "lower": lower,
                "upper": upper,
                "scipy_central": central,
                "scipy_central_contained": contained,
                "scipy_to_exact_coefficient_error_upper": (
                    implementation_error_upper
                ),
            }
        )
    scipy_coefficient_l1_error_upper = _up(
        float(sum(scipy_coefficient_error_terms))
        / (1.0 - _gamma(len(scipy_coefficient_error_terms) + 8))
    )

    scaled_order_tail, tail_details = _scaled_bessel_order_tail(
        argument
    )
    coefficient_tail = 2 * damping * scaled_order_tail
    tail_lower, tail_upper = _iv_bounds(coefficient_tail)
    pilot_tail = float(
        pilot["chebyshev_diagnostics"]["sampled_scalar_tail"]
    )
    pilot_tail_distance = _up(
        max(tail_lower - pilot_tail, pilot_tail - tail_upper, 0.0)
    )
    coefficient_intervals_certified = bool(
        scipy_coefficient_l1_error_upper < 7.1e-16
        and tail_upper < 7.1e-17
        and all(
            row["lower"] <= row["upper"]
            for row in coefficient_rows
        )
    )
    checks = [
        priority_set,
        matrix_metadata["mass_diagonal_strictly_positive"],
        matrix_metadata["stiffness_exactly_symmetric"],
        scaling_interval_certified,
        exact_gershgorin_upper < 7988.0,
        normalized_matrix_error_upper < 1.0e-9,
        scipy_coefficient_l1_error_upper < 7.1e-16,
        maximum_interval_width < 1.0e-16,
        pilot_tail_distance < 1.0e-29,
        tail_upper < 7.1e-17,
        coefficient_intervals_certified,
    ]
    return {
        "kind": "neutral_strip_chebyshev_scaling_coefficients_certificate",
        "model": (
            "directed spectral scaling and interval degree-320 "
            "Chebyshev-Bessel coefficients for one 3/8 semigroup step"
        ),
        "below_normal_priority_set": priority_set,
        "premise_artifacts": {
            "eigen_cache": str(eigen_cache),
            "eigen_cache_sha256": _sha256_file(eigen_cache),
            "two_block_result": str(two_block_result_path),
            "two_block_result_sha256": _sha256_file(
                two_block_result_path
            ),
            "pilot_result": str(pilot_result_path),
            "pilot_result_sha256": _sha256_file(pilot_result_path),
        },
        "matrix_scaling": {
            "state_count": int(normalized_generator.shape[0]),
            "maximum_row_nonzeros": maximum_row_nonzeros,
            "central_binary64_gershgorin_upper": central_gershgorin,
            "entry_relative_roundoff_upper": entry_relative_error,
            "row_sum_roundoff_upper": row_sum_error,
            "exact_stored_normalized_generator_gershgorin_upper": (
                exact_gershgorin_upper
            ),
            "normalized_generator_construction_error_upper": (
                normalized_matrix_error_upper
            ),
            "certified_two_block_global_floor_lower": global_floor,
            "scaling_interval": [SCALING_LOWER, SCALING_UPPER],
            "exact_spectrum_inside_scaling_interval_certified": (
                scaling_interval_certified
            ),
            "roundoff_model": (
                "IEEE binary64 gamma_n bounds; stored mass and stiffness "
                "entries are treated as exact binary inputs"
            ),
        },
        "coefficient_intervals": {
            "interval_decimal_digits": INTERVAL_DPS,
            "time": WINDOW,
            "degree": DEGREE,
            "bessel_argument_interval": _iv_bounds(argument),
            "damping_interval": _iv_bounds(damping),
            "base_positive_series": base_series,
            "rows": coefficient_rows,
            "all_scipy_central_values_contained": all_scipy_contained,
            "scipy_coefficient_l1_error_upper": (
                scipy_coefficient_l1_error_upper
            ),
            "maximum_binary64_interval_width": maximum_interval_width,
            "tail": {
                **tail_details,
                "lower": tail_lower,
                "upper": tail_upper,
                "pilot_sampled_value": pilot_tail,
                "pilot_sampled_value_contained": bool(
                    tail_lower <= pilot_tail <= tail_upper
                ),
                "pilot_sampled_value_distance_to_interval": (
                    pilot_tail_distance
                ),
            },
            "degree_320_exact_coefficients_and_infinite_tail_certified": (
                coefficient_intervals_certified
            ),
        },
        "sparse_recurrence_roundoff_enclosed": False,
        "reduced_semigroup_roundoff_enclosed": False,
        "boundary_output_roundoff_enclosed": False,
        "within_window_suprema_certified": False,
        "post_terminal_time_tail_certified": False,
        "screen_updated": False,
        "checks": checks,
        "all_scaling_coefficient_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_step": (
            "Use Chebyshev U-polynomial stability to accumulate directed "
            "sparse-recurrence local errors for all 112 source columns, then "
            "enclose the reduced action and boundary multiplication."
        ),
        "scope": (
            "This certificate closes only the finite stored-chain scaling "
            "interval and scalar Chebyshev coefficients. It does not certify "
            "the computed matrix action, time-window suprema, continuum "
            "transfer, polygon-circle transfer, or Navier-Stokes regularity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eigen-cache", type=Path, default=DEFAULT_EIGEN_CACHE)
    parser.add_argument(
        "--two-block-result",
        type=Path,
        default=DEFAULT_TWO_BLOCK_RESULT,
    )
    parser.add_argument(
        "--pilot-result", type=Path, default=DEFAULT_PILOT_RESULT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = audit(
        arguments.eigen_cache,
        arguments.two_block_result,
        arguments.pilot_result,
    )
    _atomic_json(arguments.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
