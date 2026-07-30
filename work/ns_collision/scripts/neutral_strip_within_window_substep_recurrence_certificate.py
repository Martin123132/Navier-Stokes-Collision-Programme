#!/usr/bin/env python3
"""Certify the universal degree-112 time-slab step operator error."""

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

import numpy as np


DEFAULT_COEFFICIENT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_within_window_substep_coefficients_v1.json"
)
DEFAULT_PRODUCTION_RECURRENCE_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_chebyshev_recurrence_roundoff_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_within_window_substep_recurrence_v1.json"
)
SUBSTEP = 0.0375
DEGREE = 112
TOTAL_SUBSTEPS = 160
SCALING_LOWER = 1.9
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


def audit(
    coefficient_result_path: Path,
    production_recurrence_result_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    coefficient = json.loads(
        coefficient_result_path.read_text(encoding="ascii")
    )
    production = json.loads(
        production_recurrence_result_path.read_text(encoding="ascii")
    )
    if not coefficient["all_substep_coefficient_checks_pass"]:
        raise RuntimeError("substep coefficient premise is not certified")
    if not production["all_recurrence_roundoff_checks_pass"]:
        raise RuntimeError("production recurrence premise is not certified")

    intervals = coefficient["coefficient_intervals"]
    if intervals["degree"] != DEGREE or intervals["time"] != SUBSTEP:
        raise RuntimeError("substep coefficient parameters changed")
    rows = intervals["rows"]
    if len(rows) != DEGREE + 1:
        raise RuntimeError("substep coefficient row count changed")

    operator = production["operator"]
    absolute_scaled_norm = _up(
        float(operator["central_scaled_absolute_row_sum_upper"])
        / (1.0 - _gamma(20))
    )
    sparse_action_relative_error = _up(
        _gamma(2 * int(operator["maximum_scaled_row_nonzeros"]) + 8)
        * absolute_scaled_norm
    )
    recurrence_arithmetic_gamma = _gamma(6)
    local_errors = np.zeros(DEGREE + 1, dtype=np.float64)
    state_errors = np.zeros(DEGREE + 1, dtype=np.float64)
    state_errors[1] = sparse_action_relative_error
    for order in range(2, DEGREE + 1):
        local_errors[order] = _up(
            2.0
            * sparse_action_relative_error
            * (1.0 + state_errors[order - 1])
            + recurrence_arithmetic_gamma
            * (
                2.0
                * (1.0 + sparse_action_relative_error)
                * (1.0 + state_errors[order - 1])
                + 1.0
                + state_errors[order - 2]
            )
        )
        weighted = order * sparse_action_relative_error
        for local_order in range(2, order + 1):
            weighted += (
                order - local_order + 1
            ) * local_errors[local_order]
        state_errors[order] = _up(
            weighted / (1.0 - _gamma(2 * order + 16))
        )

    coefficients = np.asarray(
        [abs(float(row["scipy_central"])) for row in rows],
        dtype=np.float64,
    )
    recurrence_error = _up(
        float(np.sum(coefficients * state_errors))
        / (1.0 - _gamma(2 * (DEGREE + 1) + 16))
    )
    accumulation_error = _up(
        _gamma(2 * (DEGREE + 1) + 16)
        * float(np.sum(coefficients * (1.0 + state_errors)))
    )
    coefficient_error = float(
        intervals["scipy_coefficient_l1_error_upper"]
    )
    tail_error = float(intervals["tail"]["upper"])
    generator_error = _up(
        SUBSTEP
        * float(
            operator["exact_to_computational_generator_error_upper"]
        )
        * math.exp(-SCALING_LOWER * SUBSTEP)
    )
    total_step_error = _up(
        recurrence_error
        + accumulation_error
        + coefficient_error
        + tail_error
        + generator_error
    )
    exact_contraction = _up(
        math.exp(-SCALING_LOWER * SUBSTEP)
    )
    computational_norm = _up(exact_contraction + total_step_error)
    repeated_errors = [
        _up(
            step
            * total_step_error
            * computational_norm ** (step - 1)
        )
        for step in range(1, TOTAL_SUBSTEPS + 1)
    ]
    maximum_repeated_error = max(repeated_errors)
    maximizing_step = 1 + int(np.argmax(repeated_errors))
    terminal_repeated_error = repeated_errors[-1]

    checks = [
        priority_set,
        sparse_action_relative_error < 4.0e-15,
        recurrence_error < 2.0e-12,
        accumulation_error < 1.0e-13,
        coefficient_error < 1.0e-14,
        tail_error < 3.0e-19,
        generator_error < 1.3e-12,
        total_step_error < 3.0e-12,
        computational_norm < 0.932,
        maximum_repeated_error < 2.0e-11,
        terminal_repeated_error < 1.0e-12,
    ]
    return {
        "kind": (
            "neutral_strip_within_window_substep_recurrence_certificate"
        ),
        "model": (
            "universal binary64 degree-112 Chebyshev action bound for one "
            "3/80 stored-chain semigroup substep"
        ),
        "below_normal_priority_set": priority_set,
        "premise_artifacts": {
            "substep_coefficient_result": str(coefficient_result_path),
            "substep_coefficient_result_sha256": _sha256_file(
                coefficient_result_path
            ),
            "production_recurrence_result": str(
                production_recurrence_result_path
            ),
            "production_recurrence_result_sha256": _sha256_file(
                production_recurrence_result_path
            ),
        },
        "operator": {
            "substep": SUBSTEP,
            "degree": DEGREE,
            "total_substeps_through_time_6": TOTAL_SUBSTEPS,
            "absolute_scaled_operator_norm_upper": absolute_scaled_norm,
            "sparse_action_relative_error_upper": (
                sparse_action_relative_error
            ),
            "maximum_chebyshev_state_error_upper": _up(
                float(np.max(state_errors))
            ),
            "maximum_local_recurrence_error_upper": _up(
                float(np.max(local_errors))
            ),
            "recurrence_polynomial_operator_error_upper": (
                recurrence_error
            ),
            "polynomial_accumulation_operator_error_upper": (
                accumulation_error
            ),
            "coefficient_operator_error_upper": coefficient_error,
            "tail_operator_error_upper": tail_error,
            "generator_transfer_operator_error_upper": generator_error,
            "total_one_substep_operator_error_upper": total_step_error,
            "exact_substep_contraction_upper": exact_contraction,
            "computational_substep_operator_norm_upper": (
                computational_norm
            ),
        },
        "repeated_propagation": {
            "formula": "n*epsilon*max(||P||,||S||)^(n-1)",
            "maximum_operator_error_upper": maximum_repeated_error,
            "maximum_at_substep": maximizing_step,
            "terminal_operator_error_upper": terminal_repeated_error,
        },
        "sparse_recurrence_roundoff_enclosed": bool(all(checks)),
        "within_window_grid_propagated": False,
        "within_window_suprema_certified": False,
        "post_terminal_time_tail_certified": False,
        "screen_updated": False,
        "checks": checks,
        "all_substep_recurrence_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_step": (
            "Propagate the 112 actual entry sources over the 160 substeps "
            "with atomic per-window checkpoints, enclose each boundary "
            "endpoint evaluation, and combine adjacent values with a "
            "certified second-derivative interpolation charge."
        ),
        "scope": (
            "This is an arbitrary-state stored-chain action certificate. It "
            "does not yet certify the computed 160-point source grid, any "
            "within-window supremum, the terminal tail, continuum transfer, "
            "or Navier-Stokes regularity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coefficient-result",
        type=Path,
        default=DEFAULT_COEFFICIENT_RESULT,
    )
    parser.add_argument(
        "--production-recurrence-result",
        type=Path,
        default=DEFAULT_PRODUCTION_RECURRENCE_RESULT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = audit(
        arguments.coefficient_result,
        arguments.production_recurrence_result,
    )
    _atomic_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_substep_recurrence_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
