#!/usr/bin/env python3
"""Certify all 16 stored boundary-leakage endpoint values."""

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
    "neutral_strip_h006_chebyshev_scaling_coefficients_v1.json"
)
DEFAULT_RECURRENCE_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_chebyshev_recurrence_roundoff_v1.json"
)
DEFAULT_FIRST_ENDPOINT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_first_endpoint_boundary_leakage_v1.json"
)
DEFAULT_PILOT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_boundary_leakage_chebyshev_pilot_v1.json"
)
DEFAULT_PROJECTED_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_projected_interval_two_block_transfer_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_all_endpoint_boundary_leakage_v1.json"
)
WINDOW = 0.375
DEGREE = 320
LOW_FLOOR = 2.36
SCALING_LOWER = 1.9
FORM_FLOOR = 4.832287335665
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
    return float(
        np.nextafter(product / (1.0 - product), math.inf)
    )


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _one_step_operator_error(
    coefficient_result: dict[str, Any],
    recurrence_result: dict[str, Any],
) -> dict[str, float]:
    operator = recurrence_result["operator"]
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
        [
            abs(float(row["scipy_central"]))
            for row in coefficient_result["coefficient_intervals"]["rows"]
        ],
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
        coefficient_result["coefficient_intervals"][
            "scipy_coefficient_l1_error_upper"
        ]
    )
    tail_error = float(
        coefficient_result["coefficient_intervals"]["tail"]["upper"]
    )
    generator_error = _up(
        WINDOW
        * float(
            operator["exact_to_computational_generator_error_upper"]
        )
        * math.exp(-SCALING_LOWER * WINDOW)
    )
    total = _up(
        recurrence_error
        + accumulation_error
        + coefficient_error
        + tail_error
        + generator_error
    )
    return {
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
        "recurrence_polynomial_operator_error_upper": recurrence_error,
        "polynomial_accumulation_operator_error_upper": (
            accumulation_error
        ),
        "coefficient_operator_error_upper": coefficient_error,
        "tail_operator_error_upper": tail_error,
        "generator_transfer_operator_error_upper": generator_error,
        "total_one_step_operator_error_upper": total,
    }


def audit(
    coefficient_result_path: Path,
    recurrence_result_path: Path,
    first_endpoint_result_path: Path,
    pilot_result_path: Path,
    projected_result_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    coefficient = json.loads(
        coefficient_result_path.read_text(encoding="ascii")
    )
    recurrence = json.loads(
        recurrence_result_path.read_text(encoding="ascii")
    )
    first_endpoint = json.loads(
        first_endpoint_result_path.read_text(encoding="ascii")
    )
    pilot = json.loads(pilot_result_path.read_text(encoding="ascii"))
    projected = json.loads(
        projected_result_path.read_text(encoding="ascii")
    )
    if not coefficient["all_scaling_coefficient_checks_pass"]:
        raise RuntimeError("coefficient premise is not certified")
    if not recurrence["all_recurrence_roundoff_checks_pass"]:
        raise RuntimeError("recurrence premise is not certified")
    if not first_endpoint["all_first_endpoint_boundary_checks_pass"]:
        raise RuntimeError("first-endpoint premise is not certified")
    if pilot["status"] != "complete":
        raise RuntimeError("complete endpoint pilot is unavailable")

    one_step = _one_step_operator_error(coefficient, recurrence)
    step_error = float(one_step["total_one_step_operator_error_upper"])
    exact_step_contraction = _up(
        math.exp(-SCALING_LOWER * WINDOW)
    )
    computational_step_norm = _up(
        exact_step_contraction + step_error
    )
    maximum_source_norm = float(
        recurrence["operator"]["maximum_source_state_norm"]
    )
    source_construction_relative_error = _gamma(6)

    reduced_transfer_first = float(
        first_endpoint["reduced_galerkin_transfer"][
            "reduced_form_transfer_state_error_upper"
        ]
    )
    reduced_dense_first = float(
        first_endpoint["reduced_galerkin_transfer"][
            "reduced_dense_arithmetic_state_error_upper"
        ]
    )
    output_norm = float(
        first_endpoint["boundary_operator"]["exact_operator_norm_upper"]
    )
    output_error = float(
        first_endpoint["boundary_operator"][
            "exact_operator_difference_upper"
        ]
    )
    output_product_error = float(
        first_endpoint["boundary_operator"][
            "maximum_sparse_product_roundoff_upper"
        ]
    )
    low_step_contraction = _up(math.exp(-LOW_FLOOR * WINDOW))

    pilot_rows = pilot["endpoint_rows"]
    if len(pilot_rows) != 16:
        raise RuntimeError("pilot endpoint count changed")
    endpoint_rows = []
    finite_weighted_sum = 0.0
    previous_upper = math.inf
    all_monotone = True
    for index, pilot_row in enumerate(pilot_rows, start=1):
        time_value = float(pilot_row["time"])
        full_repeated_error = _up(
            maximum_source_norm
            * (
                computational_step_norm**index
                * source_construction_relative_error
                + index
                * step_error
                * computational_step_norm ** (index - 1)
            )
        )
        reduced_form_error = _up(
            reduced_transfer_first
            * index
            * low_step_contraction ** (index - 1)
        )
        reduced_dense_error = _up(
            reduced_dense_first
            * index
            * low_step_contraction ** (index - 1)
        )
        total_state_error = _up(
            full_repeated_error
            + reduced_form_error
            + reduced_dense_error
        )
        central_boundary = float(
            pilot_row["maximum_boundary_l2_difference"]
        )
        central_state_difference = float(
            pilot_row["maximum_state_l2_difference"]
        )
        norm_evaluation_error = _up(
            _gamma(2 * 112 + 8) * central_boundary
        )
        boundary_error = _up(
            output_norm * total_state_error
            + output_error * central_state_difference
            + output_product_error
            + norm_evaluation_error
        )
        boundary_upper = _up(central_boundary + boundary_error)
        all_monotone = all_monotone and boundary_upper < previous_upper
        previous_upper = boundary_upper
        axial_upper = float(pilot_row["axial_l2_upper"])
        if index <= 15:
            finite_weighted_sum = _up(
                finite_weighted_sum + axial_upper * boundary_upper
            )
        endpoint_rows.append(
            {
                "step": index,
                "time": time_value,
                "central_boundary_l2_difference": central_boundary,
                "full_repeated_state_error_upper": full_repeated_error,
                "reduced_form_state_error_upper": reduced_form_error,
                "reduced_dense_state_error_upper": reduced_dense_error,
                "total_state_difference_error_upper": total_state_error,
                "boundary_l2_error_upper": boundary_error,
                "boundary_l2_difference_upper": boundary_upper,
                "maximizing_entry_index": int(
                    pilot_row["maximizing_entry_index"]
                ),
                "axial_l2_upper": axial_upper,
            }
        )

    endpoint_only_charge = _up(
        (WINDOW + 1.0 / FORM_FLOOR) * finite_weighted_sum
    )
    existing_screen = float(
        projected["exact_form_boundary_and_source_transfer"][
            "upgraded_complete_screen_upper"
        ]
    )
    endpoint_only_combined = _up(existing_screen + endpoint_only_charge)
    checks = [
        priority_set,
        step_error < 1.0e-11,
        computational_step_norm < 0.491,
        len(endpoint_rows) == 16,
        endpoint_rows[0]["boundary_l2_difference_upper"] < 6.6e-4,
        endpoint_rows[-1]["boundary_l2_difference_upper"] < 1.0e-8,
        all_monotone,
        endpoint_only_charge < 4.0e-4,
        endpoint_only_combined < 0.971,
    ]
    return {
        "kind": "neutral_strip_all_endpoint_boundary_leakage_certificate",
        "model": (
            "certified stored-chain source-oriented boundary leakage at "
            "all 16 multiples of the 3/8 production window"
        ),
        "below_normal_priority_set": priority_set,
        "premise_artifacts": {
            "coefficient_result": str(coefficient_result_path),
            "coefficient_result_sha256": _sha256_file(
                coefficient_result_path
            ),
            "recurrence_result": str(recurrence_result_path),
            "recurrence_result_sha256": _sha256_file(
                recurrence_result_path
            ),
            "first_endpoint_result": str(first_endpoint_result_path),
            "first_endpoint_result_sha256": _sha256_file(
                first_endpoint_result_path
            ),
            "pilot_result": str(pilot_result_path),
            "pilot_result_sha256": _sha256_file(pilot_result_path),
            "projected_result": str(projected_result_path),
            "projected_result_sha256": _sha256_file(
                projected_result_path
            ),
        },
        "one_step_operator_error": one_step,
        "exact_step_contraction_upper": exact_step_contraction,
        "computational_step_operator_norm_upper": (
            computational_step_norm
        ),
        "endpoint_rows": endpoint_rows,
        "endpoint_uppers_strictly_decrease": all_monotone,
        "finite_endpoint_only_weighted_sum_upper": finite_weighted_sum,
        "finite_endpoint_only_screen_charge_upper": endpoint_only_charge,
        "existing_certified_screen_upper": existing_screen,
        "endpoint_only_combined_screen_diagnostic_upper": (
            endpoint_only_combined
        ),
        "all_sixteen_boundary_endpoints_certified": bool(all(checks)),
        "within_window_suprema_certified": False,
        "post_terminal_time_tail_certified": False,
        "screen_updated": False,
        "checks": checks,
        "all_endpoint_boundary_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_step": (
            "Replace the first fifteen endpoint values by certified suprema "
            "over their following 3/8 windows. Use the sixteenth endpoint "
            "at t=6 as the anchor for a separate post-time-6 tail before "
            "adding any leakage charge to the production screen."
        ),
        "scope": (
            "This certificate closes the 16 stored finite-chain endpoint "
            "values only. Endpoint monotonicity does not prove between-time "
            "monotonicity. Continuum and domain transfer remain open."
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
        "--recurrence-result",
        type=Path,
        default=DEFAULT_RECURRENCE_RESULT,
    )
    parser.add_argument(
        "--first-endpoint-result",
        type=Path,
        default=DEFAULT_FIRST_ENDPOINT_RESULT,
    )
    parser.add_argument(
        "--pilot-result", type=Path, default=DEFAULT_PILOT_RESULT
    )
    parser.add_argument(
        "--projected-result", type=Path, default=DEFAULT_PROJECTED_RESULT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = audit(
        arguments.coefficient_result,
        arguments.recurrence_result,
        arguments.first_endpoint_result,
        arguments.pilot_result,
        arguments.projected_result,
    )
    _atomic_json(arguments.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
