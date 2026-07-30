#!/usr/bin/env python3
"""Certify a bounded directed-LDL prefix of the hypercircle pencil."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

import numpy as np
import scipy
from scipy.sparse import csc_matrix, csr_matrix, diags, tril
from scipy.sparse.linalg import splu


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPLETE_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_positive_exponential_complete_assembly_v1.json"
)
DEFAULT_MATRICES = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_positive_exponential_assembly_matrices_v1.npz"
)
DEFAULT_GAUSSIAN_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
)
DEFAULT_GAUSSIAN_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_checkpoint_v1.npz"
)
DEFAULT_CENTRAL_AUDIT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_central_factorization_audit_v1.json"
)
DEFAULT_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_directed_ldl_prefix_checkpoint_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_directed_ldl_prefix_audit_v1.json"
)

ALGORITHM_VERSION = 1
DEFAULT_MAXIMUM_PIVOTS = 1024
DEFAULT_PRECISIONS = (50, 80, 120)
DEFAULT_CHECKPOINT_BATCH = 64
RUIZ_ITERATIONS = 10
DAYTIME_BASELINE_CPU_LIMIT = 60.0
DAYTIME_PARK_CPU_LIMIT = 75.0
EXPECTED_DIMENSION = 123816
EXPECTED_CENTRAL_NNZ = 798384
EXPECTED_RADIUS_NNZ = 612660
EXPECTED_SCALE_SHA256 = (
    "bef73a763a5ee24b85651a3c761b940b0fde2f8e04294a7d7e9b0c085c7ca9c2"
)
EXPECTED_RAW_PERMUTATION_SHA256 = (
    "fc613374bffd7bba84293e3c302e56d0ef945a0530443b04dde5ba079adb36db"
)
EXPECTED_ORDER_SHA256 = (
    "5f8adf72fa6fe8e3ea62d716c6a2f34df7252fa0a45bd5cafb55fbf7001d5f13"
)
EXPECTED_FACTOR_PATTERN_SHA256 = (
    "dc941205f68286d3f318a58670fd0ddf14bf63afdaf48202dcc9e0291238103b"
)
EXPECTED_U_DIAGONAL_SHA256 = (
    "c3cf0559f421039b208855c6ba2a3bab9d887d4a8459ffc71c76b381bdfbf140"
)

Interval = tuple[Decimal, Decimal]


def _load_module(filename: str, module_name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    compact: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="ascii",
            newline="\n",
        ) as stream:
            options: dict[str, Any] = {
                "allow_nan": False,
                "sort_keys": True,
            }
            if compact:
                options["separators"] = (",", ":")
            else:
                options["indent"] = 2
            json.dump(payload, stream, **options)
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


def _set_below_normal_priority() -> bool:
    try:
        import psutil

        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return process.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
    except Exception:
        return False


def _sample_cpu_baseline(seconds: int = 5) -> list[float]:
    try:
        import psutil
    except Exception:
        return []
    return [
        float(psutil.cpu_percent(interval=1.0))
        for _ in range(max(5, seconds))
    ]


class DirectedDecimal:
    """Outward-rounded Decimal interval operations at one precision."""

    def __init__(self, precision: int):
        if precision < 8:
            raise ValueError("directed Decimal precision must be at least 8")
        self.precision = precision
        self.lower = Context(prec=precision, rounding=ROUND_FLOOR)
        self.upper = Context(prec=precision, rounding=ROUND_CEILING)
        self.nearest = Context(
            prec=precision + 4,
            rounding=ROUND_HALF_EVEN,
        )
        self.zero = Decimal(0)
        self.one = Decimal(1)
        self.two = Decimal(2)

    def add(self, first: Interval, second: Interval) -> Interval:
        return (
            self.lower.add(first[0], second[0]),
            self.upper.add(first[1], second[1]),
        )

    def subtract(self, first: Interval, second: Interval) -> Interval:
        return (
            self.lower.subtract(first[0], second[1]),
            self.upper.subtract(first[1], second[0]),
        )

    def multiply(self, first: Interval, second: Interval) -> Interval:
        lower_products = [
            self.lower.multiply(left, right)
            for left in first
            for right in second
        ]
        upper_products = [
            self.upper.multiply(left, right)
            for left in first
            for right in second
        ]
        return min(lower_products), max(upper_products)

    def divide(self, numerator: Interval, denominator: Interval) -> Interval:
        if denominator[0] <= self.zero <= denominator[1]:
            raise ZeroDivisionError("pivot interval contains zero")
        reciprocal = (
            self.lower.divide(self.one, denominator[1]),
            self.upper.divide(self.one, denominator[0]),
        )
        return self.multiply(numerator, reciprocal)

    def midpoint_radius(self, interval: Interval) -> tuple[Decimal, Decimal]:
        midpoint = self.nearest.divide(
            self.nearest.add(interval[0], interval[1]),
            self.two,
        )
        radius = self.upper.divide(
            self.upper.subtract(interval[1], interval[0]),
            self.two,
        )
        return midpoint, radius

    def absolute_upper(self, interval: Interval) -> Decimal:
        return max(abs(interval[0]), abs(interval[1]))

    def width(self, interval: Interval) -> Decimal:
        return self.upper.subtract(interval[1], interval[0])


@dataclass
class PrefixProblem:
    center: csr_matrix
    radius: csr_matrix
    scale: np.ndarray
    order: np.ndarray
    positions: np.ndarray
    lower: csc_matrix
    central_pivots: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.center.shape[0])


def _sparse_value(matrix: csr_matrix, row: int, column: int) -> float:
    start = int(matrix.indptr[row])
    stop = int(matrix.indptr[row + 1])
    indices = matrix.indices[start:stop]
    offset = int(np.searchsorted(indices, column))
    if offset < len(indices) and int(indices[offset]) == column:
        return float(matrix.data[start + offset])
    return 0.0


def _factor_pointer(
    lower: csc_matrix,
    row: int,
    column: int,
) -> int | None:
    start = int(lower.indptr[column])
    stop = int(lower.indptr[column + 1])
    indices = lower.indices[start:stop]
    offset = int(np.searchsorted(indices, row))
    if offset < len(indices) and int(indices[offset]) == row:
        return start + offset
    return None


def _descendants(lower: csc_matrix, column: int) -> np.ndarray:
    start = int(lower.indptr[column])
    stop = int(lower.indptr[column + 1])
    rows = lower.indices[start:stop]
    if not np.any(rows == column):
        raise RuntimeError(f"factor column {column} lacks its diagonal")
    if np.any(rows < column):
        raise RuntimeError(f"factor column {column} is not lower triangular")
    return rows[rows > column].astype(np.int64, copy=False)


def _input_neighbors_after(
    problem: PrefixProblem,
    pivot: int,
) -> set[int]:
    original_row = int(problem.order[pivot])
    neighbors: set[int] = set()
    for matrix in (problem.center, problem.radius):
        start = int(matrix.indptr[original_row])
        stop = int(matrix.indptr[original_row + 1])
        for pointer in range(start, stop):
            if matrix.data[pointer] == 0.0:
                continue
            original_column = int(matrix.indices[pointer])
            elimination_column = int(problem.positions[original_column])
            if elimination_column > pivot:
                neighbors.add(elimination_column)
    return neighbors


def _scan_symbolic_prefix(
    problem: PrefixProblem,
    maximum_pivots: int,
    checkpoints: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Scan exact potential fill without performing interval arithmetic."""
    if not 0 < maximum_pivots <= problem.dimension:
        raise ValueError("symbolic scan pivot count is out of range")
    checkpoint_set = set(checkpoints)
    if any(
        checkpoint <= 0 or checkpoint > maximum_pivots
        for checkpoint in checkpoint_set
    ):
        raise ValueError("symbolic scan checkpoint is out of range")

    row_columns: dict[int, set[int]] = {}
    column_rows: dict[int, tuple[int, ...]] = {}
    first_by_diagonal_term_count: dict[int, int] = {}
    first_by_descendant_count: dict[int, int] = {}
    first_by_off_diagonal_common_term_count: dict[int, int] = {}
    checkpoint_rows: dict[str, dict[str, int]] = {}
    maximum_diagonal_terms = 0
    maximum_descendants = 0
    maximum_off_diagonal_common_terms = 0
    symbolic_lower_entries = 0
    total_diagonal_terms = 0
    total_off_diagonal_common_terms = 0
    reference_product_pair_terms = 0

    for pivot in range(maximum_pivots):
        prior_columns = row_columns.get(pivot, set())
        required = _input_neighbors_after(problem, pivot)
        for prior in prior_columns:
            required.update(
                row
                for row in column_rows[prior]
                if row > pivot
            )
        descendants = tuple(sorted(required))
        column_rows[pivot] = descendants
        maximum_common_terms = 0
        for row in descendants:
            common_terms = len(
                prior_columns.intersection(
                    row_columns.get(row, set())
                )
            )
            maximum_common_terms = max(
                maximum_common_terms,
                common_terms,
            )
            total_off_diagonal_common_terms += common_terms
            row_columns.setdefault(row, set()).add(pivot)

        diagonal_term_count = len(prior_columns)
        descendant_count = len(descendants)
        symbolic_lower_entries += descendant_count
        total_diagonal_terms += diagonal_term_count
        factor_column_entries = descendant_count + 1
        reference_product_pair_terms += (
            factor_column_entries * (factor_column_entries + 1) // 2
        )
        maximum_diagonal_terms = max(
            maximum_diagonal_terms,
            diagonal_term_count,
        )
        maximum_descendants = max(
            maximum_descendants,
            descendant_count,
        )
        maximum_off_diagonal_common_terms = max(
            maximum_off_diagonal_common_terms,
            maximum_common_terms,
        )
        first_by_diagonal_term_count.setdefault(
            diagonal_term_count,
            pivot,
        )
        first_by_descendant_count.setdefault(
            descendant_count,
            pivot,
        )
        first_by_off_diagonal_common_term_count.setdefault(
            maximum_common_terms,
            pivot,
        )
        if pivot + 1 in checkpoint_set:
            checkpoint_rows[str(pivot + 1)] = {
                "maximum_diagonal_term_count": maximum_diagonal_terms,
                "maximum_descendant_count": maximum_descendants,
                "maximum_off_diagonal_common_term_count": (
                    maximum_off_diagonal_common_terms
                ),
                "symbolic_lower_entry_count": symbolic_lower_entries,
                "total_diagonal_term_count": total_diagonal_terms,
                "total_off_diagonal_common_term_count": (
                    total_off_diagonal_common_terms
                ),
                "reference_product_pair_term_count": (
                    reference_product_pair_terms
                ),
            }

    return {
        "maximum_pivots": maximum_pivots,
        "first_pivot_by_diagonal_term_count": {
            str(count): pivot
            for count, pivot in sorted(
                first_by_diagonal_term_count.items()
            )
        },
        "first_pivot_by_descendant_count": {
            str(count): pivot
            for count, pivot in sorted(
                first_by_descendant_count.items()
            )
        },
        "first_pivot_by_off_diagonal_common_term_count": {
            str(count): pivot
            for count, pivot in sorted(
                first_by_off_diagonal_common_term_count.items()
            )
        },
        "maximum_diagonal_term_count": maximum_diagonal_terms,
        "maximum_descendant_count": maximum_descendants,
        "maximum_off_diagonal_common_term_count": (
            maximum_off_diagonal_common_terms
        ),
        "symbolic_lower_entry_count": symbolic_lower_entries,
        "total_diagonal_term_count": total_diagonal_terms,
        "total_off_diagonal_common_term_count": (
            total_off_diagonal_common_terms
        ),
        "reference_product_pair_term_count": (
            reference_product_pair_terms
        ),
        "checkpoints": checkpoint_rows,
        "arithmetic_signs_certified": False,
    }


def _input_interval(
    problem: PrefixProblem,
    arithmetic: DirectedDecimal,
    row: int,
    column: int,
) -> Interval:
    original_row = int(problem.order[row])
    original_column = int(problem.order[column])
    center = Decimal.from_float(
        _sparse_value(problem.center, original_row, original_column)
    )
    radius = Decimal.from_float(
        _sparse_value(problem.radius, original_row, original_column)
    )
    if radius < arithmetic.zero:
        raise RuntimeError("input radius is negative")
    interval = (
        arithmetic.lower.subtract(center, radius),
        arithmetic.upper.add(center, radius),
    )
    row_scale = Decimal.from_float(float(problem.scale[original_row]))
    column_scale = Decimal.from_float(float(problem.scale[original_column]))
    if row_scale <= arithmetic.zero or column_scale <= arithmetic.zero:
        raise RuntimeError("congruence scale is not positive")
    interval = arithmetic.multiply(
        interval,
        (row_scale, row_scale),
    )
    return arithmetic.multiply(
        interval,
        (column_scale, column_scale),
    )


def _same_structure(
    first: csc_matrix | csr_matrix,
    second: csc_matrix | csr_matrix,
) -> bool:
    left = first.tocsr()
    right = second.tocsr()
    left.sort_indices()
    right.sort_indices()
    return bool(
        np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
    )


def _pattern_missing_count(
    required: csc_matrix | csr_matrix,
    available: csc_matrix | csr_matrix,
) -> int:
    required_pattern = required.copy().tocsr()
    available_pattern = available.copy().tocsr()
    required_pattern.eliminate_zeros()
    available_pattern.eliminate_zeros()
    required_pattern.data = np.ones_like(required_pattern.data)
    available_pattern.data = np.ones_like(available_pattern.data)
    missing = required_pattern - required_pattern.multiply(available_pattern)
    missing.eliminate_zeros()
    return int(missing.nnz)


def _interval_strings(interval: Interval) -> list[str]:
    return [str(interval[0]), str(interval[1])]


def _attempt_state(precision: int) -> dict[str, Any]:
    return {
        "precision": precision,
        "next_pivot": 0,
        "pivot_lower_decimal": [],
        "pivot_upper_decimal": [],
        "pivot_sign": [],
        "pivot_diagnostics": [],
        "lower_entries": [],
        "failed_pivot": None,
    }


def _checkpoint_payload(
    contract: dict[str, Any],
    precision_schedule: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "checkpoint_version": 1,
        "kind": "hypercircle-directed-ldl-prefix-checkpoint",
        "contract": contract,
        "overall_status": "running",
        "precision_schedule": list(precision_schedule),
        "attempt_summaries": [],
        "current_attempt_index": 0,
        "current_attempt": _attempt_state(precision_schedule[0]),
    }


def _write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    material = {
        key: value
        for key, value in checkpoint.items()
        if key != "state_sha256"
    }
    payload = dict(material)
    payload["state_sha256"] = _canonical_sha256(material)
    _atomic_json(path, payload, compact=True)
    checkpoint.clear()
    checkpoint.update(payload)


def _load_checkpoint(
    path: Path,
    contract: dict[str, Any],
    precision_schedule: tuple[int, ...],
) -> dict[str, Any]:
    checkpoint = json.loads(path.read_text(encoding="ascii"))
    recorded_hash = checkpoint.pop("state_sha256", None)
    if recorded_hash != _canonical_sha256(checkpoint):
        raise RuntimeError("directed-LDL checkpoint state hash mismatch")
    checkpoint["state_sha256"] = recorded_hash
    if checkpoint.get("checkpoint_version") != 1:
        raise RuntimeError("directed-LDL checkpoint version mismatch")
    if checkpoint.get("contract") != contract:
        raise RuntimeError("directed-LDL checkpoint contract mismatch")
    if checkpoint.get("precision_schedule") != list(precision_schedule):
        raise RuntimeError("directed-LDL checkpoint precision mismatch")
    return checkpoint


def _restore_row_entries(
    problem: PrefixProblem,
    attempt: dict[str, Any],
    maximum_pivots: int,
) -> tuple[
    dict[int, dict[int, Interval]],
    dict[int, tuple[int, ...]],
]:
    next_pivot = int(attempt["next_pivot"])
    if not 0 <= next_pivot <= maximum_pivots:
        raise RuntimeError("checkpoint next pivot is out of range")
    lower_rows = attempt["lower_entries"]
    if not (
        len(attempt["pivot_lower_decimal"])
        == len(attempt["pivot_upper_decimal"])
        == len(attempt["pivot_sign"])
        == len(attempt["pivot_diagnostics"])
        == next_pivot
    ):
        raise RuntimeError("checkpoint pivot arrays are incomplete")

    row_entries: dict[int, dict[int, Interval]] = {}
    column_rows: dict[int, tuple[int, ...]] = {}
    zero = Decimal(0)
    offset = 0
    for column in range(next_pivot):
        required = _input_neighbors_after(problem, column)
        for prior in row_entries.get(column, {}):
            required.update(
                row
                for row in column_rows[prior]
                if row > column
            )
        expected_rows = tuple(sorted(required))
        stop = offset + len(expected_rows)
        records = lower_rows[offset:stop]
        actual_coordinates = [
            (int(record[0]), int(record[1]))
            for record in records
        ]
        expected_coordinates = [
            (row, column) for row in expected_rows
        ]
        if actual_coordinates != expected_coordinates:
            raise RuntimeError(
                "checkpoint symbolic lower coordinates are incomplete"
            )
        for record in records:
            row = int(record[0])
            interval = (Decimal(record[2]), Decimal(record[3]))
            if (
                not interval[0].is_finite()
                or not interval[1].is_finite()
                or interval[0] > interval[1]
            ):
                raise RuntimeError("checkpoint lower interval is invalid")
            row_entries.setdefault(row, {})[column] = interval
        column_rows[column] = expected_rows
        offset = stop
    if offset != len(lower_rows):
        raise RuntimeError("checkpoint has surplus lower entries")
    for index in range(next_pivot):
        interval = (
            Decimal(attempt["pivot_lower_decimal"][index]),
            Decimal(attempt["pivot_upper_decimal"][index]),
        )
        sign = int(attempt["pivot_sign"][index])
        if (
            not interval[0].is_finite()
            or not interval[1].is_finite()
            or interval[0] > interval[1]
            or interval[0] <= zero <= interval[1]
            or sign not in (-1, 1)
            or (sign < 0) != (interval[1] < zero)
        ):
            raise RuntimeError("checkpoint pivot interval is invalid")
    return row_entries, column_rows


def _summarize_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    diagnostics = attempt["pivot_diagnostics"]
    signs = [int(value) for value in attempt["pivot_sign"]]
    margins = [
        Decimal(row["pivot_margin_decimal"])
        for row in diagnostics
    ]
    relative_radii = [
        Decimal(row["pivot_radius_to_margin_upper_decimal"])
        for row in diagnostics
    ]
    cancellation = [
        Decimal(row["cancellation_charge_upper_decimal"])
        for row in diagnostics
    ]
    lower_widths = [
        Decimal(row["maximum_lower_interval_width_decimal"])
        for row in diagnostics
    ]
    update_rows = [
        row for row in diagnostics if int(row["diagonal_term_count"]) > 0
    ]
    minimum_interaction_margin_row = (
        min(
            update_rows,
            key=lambda row: Decimal(row["pivot_margin_decimal"]),
        )
        if update_rows
        else None
    )
    maximum_interaction_radius_row = (
        max(
            update_rows,
            key=lambda row: Decimal(
                row["pivot_radius_to_margin_upper_decimal"]
            ),
        )
        if update_rows
        else None
    )
    maximum_interaction_cancellation_row = (
        max(
            update_rows,
            key=lambda row: Decimal(
                row["cancellation_charge_upper_decimal"]
            ),
        )
        if update_rows
        else None
    )
    maximum_interaction_lower_width_row = (
        max(
            update_rows,
            key=lambda row: Decimal(
                row["maximum_lower_interval_width_decimal"]
            ),
        )
        if update_rows
        else None
    )
    minimum_margin_row = (
        min(
            diagnostics,
            key=lambda row: Decimal(row["pivot_margin_decimal"]),
        )
        if diagnostics
        else None
    )
    maximum_relative_radius_row = (
        max(
            diagnostics,
            key=lambda row: Decimal(
                row["pivot_radius_to_margin_upper_decimal"]
            ),
        )
        if diagnostics
        else None
    )
    maximum_cancellation_row = (
        max(
            diagnostics,
            key=lambda row: Decimal(
                row["cancellation_charge_upper_decimal"]
            ),
        )
        if diagnostics
        else None
    )
    symbolic_coordinates = [
        [int(record[0]), int(record[1])]
        for record in attempt["lower_entries"]
    ]
    return {
        "precision": int(attempt["precision"]),
        "completed_pivot_count": int(attempt["next_pivot"]),
        "negative_pivot_count": signs.count(-1),
        "positive_pivot_count": signs.count(1),
        "minimum_pivot_margin_decimal": (
            str(min(margins)) if margins else None
        ),
        "minimum_pivot_margin_index": (
            int(minimum_margin_row["index"])
            if minimum_margin_row is not None
            else None
        ),
        "maximum_pivot_radius_to_margin_upper_decimal": (
            str(max(relative_radii)) if relative_radii else None
        ),
        "maximum_pivot_radius_to_margin_index": (
            int(maximum_relative_radius_row["index"])
            if maximum_relative_radius_row is not None
            else None
        ),
        "maximum_cancellation_charge_upper_decimal": (
            str(max(cancellation)) if cancellation else None
        ),
        "maximum_cancellation_charge_index": (
            int(maximum_cancellation_row["index"])
            if maximum_cancellation_row is not None
            else None
        ),
        "maximum_lower_interval_width_decimal": (
            str(max(lower_widths)) if lower_widths else None
        ),
        "maximum_symbolic_descendant_count": (
            max(
                int(row["symbolic_descendant_count"])
                for row in diagnostics
            )
            if diagnostics
            else 0
        ),
        "symbolic_lower_entry_count": len(
            attempt["lower_entries"]
        ),
        "symbolic_lower_coordinates_sha256": _canonical_sha256(
            {"coordinates": symbolic_coordinates}
        ),
        "lower_interval_entries_sha256": _canonical_sha256(
            {"entries": attempt["lower_entries"]}
        ),
        "pivot_intervals_sha256": _canonical_sha256(
            {
                "lower": attempt["pivot_lower_decimal"],
                "upper": attempt["pivot_upper_decimal"],
                "sign": signs,
            }
        ),
        "central_superlu_pivots_contained_count": sum(
            int(row["central_superlu_pivot_contained"])
            for row in diagnostics
        ),
        "central_factor_envelope_missing_count": sum(
            int(row["central_factor_envelope_missing_count"])
            for row in diagnostics
        ),
        "pivots_with_nonzero_diagonal_term_count": len(update_rows),
        "first_pivot_with_nonzero_diagonal_term": (
            int(update_rows[0]["index"]) if update_rows else None
        ),
        "total_diagonal_recurrence_term_count": sum(
            int(row["diagonal_term_count"]) for row in diagnostics
        ),
        "total_off_diagonal_recurrence_term_count": sum(
            int(row["off_diagonal_recurrence_term_count"])
            for row in diagnostics
        ),
        "interaction_profile": {
            "pivot_count": len(update_rows),
            "first_pivot": (
                int(update_rows[0]["index"]) if update_rows else None
            ),
            "last_pivot": (
                int(update_rows[-1]["index"]) if update_rows else None
            ),
            "negative_pivot_count": sum(
                int(row["sign"]) < 0 for row in update_rows
            ),
            "positive_pivot_count": sum(
                int(row["sign"]) > 0 for row in update_rows
            ),
            "minimum_pivot_margin_decimal": (
                minimum_interaction_margin_row["pivot_margin_decimal"]
                if minimum_interaction_margin_row is not None
                else None
            ),
            "minimum_pivot_margin_index": (
                int(minimum_interaction_margin_row["index"])
                if minimum_interaction_margin_row is not None
                else None
            ),
            "maximum_pivot_radius_to_margin_upper_decimal": (
                maximum_interaction_radius_row[
                    "pivot_radius_to_margin_upper_decimal"
                ]
                if maximum_interaction_radius_row is not None
                else None
            ),
            "maximum_pivot_radius_to_margin_index": (
                int(maximum_interaction_radius_row["index"])
                if maximum_interaction_radius_row is not None
                else None
            ),
            "maximum_cancellation_charge_upper_decimal": (
                maximum_interaction_cancellation_row[
                    "cancellation_charge_upper_decimal"
                ]
                if maximum_interaction_cancellation_row is not None
                else None
            ),
            "maximum_cancellation_charge_index": (
                int(maximum_interaction_cancellation_row["index"])
                if maximum_interaction_cancellation_row is not None
                else None
            ),
            "maximum_lower_interval_width_decimal": (
                maximum_interaction_lower_width_row[
                    "maximum_lower_interval_width_decimal"
                ]
                if maximum_interaction_lower_width_row is not None
                else None
            ),
            "maximum_lower_interval_width_index": (
                int(maximum_interaction_lower_width_row["index"])
                if maximum_interaction_lower_width_row is not None
                else None
            ),
            "maximum_diagonal_term_count": (
                max(
                    int(row["diagonal_term_count"])
                    for row in update_rows
                )
                if update_rows
                else 0
            ),
            "maximum_off_diagonal_recurrence_term_count": (
                max(
                    int(row["off_diagonal_recurrence_term_count"])
                    for row in update_rows
                )
                if update_rows
                else 0
            ),
            "maximum_symbolic_descendant_count": (
                max(
                    int(row["symbolic_descendant_count"])
                    for row in update_rows
                )
                if update_rows
                else 0
            ),
        },
        "failed_pivot": attempt["failed_pivot"],
    }


def _continue_attempt(
    problem: PrefixProblem,
    attempt: dict[str, Any],
    maximum_pivots: int,
    pivot_limit: int,
    checkpoint_batch: int,
    checkpoint_callback: Callable[[], None],
    cpu_park_callback: Callable[[], bool] | None,
) -> tuple[str, int]:
    arithmetic = DirectedDecimal(int(attempt["precision"]))
    row_entries, column_rows = _restore_row_entries(
        problem,
        attempt,
        maximum_pivots,
    )

    completed_this_call = 0
    while int(attempt["next_pivot"]) < maximum_pivots:
        pivot = int(attempt["next_pivot"])
        required = _input_neighbors_after(problem, pivot)
        for prior in row_entries.get(pivot, {}):
            required.update(
                row
                for row in column_rows[prior]
                if row > pivot
            )
        pivot_descendants = tuple(sorted(required))

        diagonal_input = _input_interval(
            problem,
            arithmetic,
            pivot,
            pivot,
        )
        diagonal_sum: Interval = (arithmetic.zero, arithmetic.zero)
        cancellation_numerator = arithmetic.absolute_upper(
            diagonal_input
        )
        diagonal_term_count = 0
        for prior, lower_value in sorted(
            row_entries.get(pivot, {}).items()
        ):
            term = arithmetic.multiply(lower_value, lower_value)
            prior_pivot = (
                Decimal(attempt["pivot_lower_decimal"][prior]),
                Decimal(attempt["pivot_upper_decimal"][prior]),
            )
            term = arithmetic.multiply(term, prior_pivot)
            diagonal_sum = arithmetic.add(diagonal_sum, term)
            cancellation_numerator = arithmetic.upper.add(
                cancellation_numerator,
                arithmetic.absolute_upper(term),
            )
            diagonal_term_count += 1
        pivot_interval = arithmetic.subtract(
            diagonal_input,
            diagonal_sum,
        )
        pivot_center, pivot_radius = arithmetic.midpoint_radius(
            pivot_interval
        )
        input_center, input_radius = arithmetic.midpoint_radius(
            diagonal_input
        )
        sum_center, sum_radius = arithmetic.midpoint_radius(diagonal_sum)
        central_pivot = Decimal.from_float(
            float(problem.central_pivots[pivot])
        )

        if (
            pivot_interval[0] <= arithmetic.zero <= pivot_interval[1]
        ):
            attempt["failed_pivot"] = {
                "index": pivot,
                "failure_kind": "zero_containing_pivot",
                "pivot_interval_decimal": _interval_strings(
                    pivot_interval
                ),
                "pivot_center_decimal": str(pivot_center),
                "pivot_radius_decimal": str(pivot_radius),
                "input_diagonal_interval_decimal": _interval_strings(
                    diagonal_input
                ),
                "diagonal_update_interval_decimal": _interval_strings(
                    diagonal_sum
                ),
                "diagonal_term_count": diagonal_term_count,
                "cancellation_numerator_upper_decimal": str(
                    cancellation_numerator
                ),
                "central_superlu_pivot": float(
                    problem.central_pivots[pivot]
                ),
            }
            return "precision_failure", completed_this_call

        sign = -1 if pivot_interval[1] < arithmetic.zero else 1
        margin = min(abs(pivot_interval[0]), abs(pivot_interval[1]))
        relative_radius = arithmetic.upper.divide(
            pivot_radius,
            margin,
        )
        cancellation_charge = arithmetic.upper.divide(
            cancellation_numerator,
            margin,
        )
        central_pivot_contained = bool(
            pivot_interval[0] <= central_pivot <= pivot_interval[1]
        )

        maximum_lower_width = arithmetic.zero
        maximum_lower_relative_radius = arithmetic.zero
        off_diagonal_term_count = 0
        central_lower_contained_count = 0
        central_lower_comparison_count = 0
        central_factor_envelope_missing_count = 0
        for row_value in pivot_descendants:
            row = int(row_value)
            off_diagonal_sum: Interval = (
                arithmetic.zero,
                arithmetic.zero,
            )
            row_prior = row_entries.get(row, {})
            pivot_prior = row_entries.get(pivot, {})
            common = sorted(set(row_prior).intersection(pivot_prior))
            for prior in common:
                term = arithmetic.multiply(
                    row_prior[prior],
                    pivot_prior[prior],
                )
                prior_pivot = (
                    Decimal(attempt["pivot_lower_decimal"][prior]),
                    Decimal(attempt["pivot_upper_decimal"][prior]),
                )
                term = arithmetic.multiply(term, prior_pivot)
                off_diagonal_sum = arithmetic.add(
                    off_diagonal_sum,
                    term,
                )
                off_diagonal_term_count += 1
            numerator = arithmetic.subtract(
                _input_interval(
                    problem,
                    arithmetic,
                    row,
                    pivot,
                ),
                off_diagonal_sum,
            )
            lower_interval = arithmetic.divide(
                numerator,
                pivot_interval,
            )
            row_entries.setdefault(row, {})[pivot] = lower_interval
            attempt["lower_entries"].append(
                [
                    row,
                    pivot,
                    str(lower_interval[0]),
                    str(lower_interval[1]),
                ]
            )
            lower_width = arithmetic.width(lower_interval)
            maximum_lower_width = max(
                maximum_lower_width,
                lower_width,
            )
            lower_scale = arithmetic.absolute_upper(lower_interval)
            lower_relative_radius = (
                arithmetic.upper.divide(
                    arithmetic.midpoint_radius(lower_interval)[1],
                    lower_scale,
                )
                if lower_scale != arithmetic.zero
                else arithmetic.zero
            )
            maximum_lower_relative_radius = max(
                maximum_lower_relative_radius,
                lower_relative_radius,
            )
            pointer = _factor_pointer(problem.lower, row, pivot)
            if pointer is None:
                central_factor_envelope_missing_count += 1
            else:
                central_lower = Decimal.from_float(
                    float(problem.lower.data[pointer])
                )
                central_lower_comparison_count += 1
                central_lower_contained_count += int(
                    lower_interval[0]
                    <= central_lower
                    <= lower_interval[1]
                )

        column_rows[pivot] = pivot_descendants
        attempt["pivot_lower_decimal"].append(str(pivot_interval[0]))
        attempt["pivot_upper_decimal"].append(str(pivot_interval[1]))
        attempt["pivot_sign"].append(sign)
        attempt["pivot_diagnostics"].append(
            {
                "index": pivot,
                "sign": sign,
                "pivot_interval_decimal": _interval_strings(
                    pivot_interval
                ),
                "pivot_center_decimal": str(pivot_center),
                "pivot_radius_decimal": str(pivot_radius),
                "pivot_margin_decimal": str(margin),
                "pivot_radius_to_margin_upper_decimal": str(
                    relative_radius
                ),
                "input_diagonal_interval_decimal": _interval_strings(
                    diagonal_input
                ),
                "input_diagonal_center_decimal": str(input_center),
                "input_diagonal_radius_decimal": str(input_radius),
                "diagonal_update_interval_decimal": _interval_strings(
                    diagonal_sum
                ),
                "diagonal_update_center_decimal": str(sum_center),
                "diagonal_update_radius_decimal": str(sum_radius),
                "diagonal_term_count": diagonal_term_count,
                "cancellation_numerator_upper_decimal": str(
                    cancellation_numerator
                ),
                "cancellation_charge_upper_decimal": str(
                    cancellation_charge
                ),
                "symbolic_descendant_count": len(pivot_descendants),
                "symbolic_required_count": len(required),
                "off_diagonal_recurrence_term_count": (
                    off_diagonal_term_count
                ),
                "maximum_lower_interval_width_decimal": str(
                    maximum_lower_width
                ),
                "maximum_lower_relative_radius_decimal": str(
                    maximum_lower_relative_radius
                ),
                "central_superlu_pivot": float(
                    problem.central_pivots[pivot]
                ),
                "central_superlu_pivot_contained": (
                    central_pivot_contained
                ),
                "central_superlu_lower_entries_contained": (
                    central_lower_contained_count
                ),
                "central_superlu_lower_entry_count": (
                    central_lower_comparison_count
                ),
                "central_factor_envelope_missing_count": (
                    central_factor_envelope_missing_count
                ),
            }
        )
        attempt["next_pivot"] = pivot + 1
        completed_this_call += 1

        at_batch = (
            int(attempt["next_pivot"]) % checkpoint_batch == 0
            or int(attempt["next_pivot"]) == maximum_pivots
        )
        if at_batch:
            checkpoint_callback()
            if (
                cpu_park_callback is not None
                and cpu_park_callback()
            ):
                return "cpu_park", completed_this_call
        if pivot_limit > 0 and completed_this_call >= pivot_limit:
            checkpoint_callback()
            return "pivot_budget_park", completed_this_call
    return "complete", completed_this_call


def _run_adaptive_prefix(
    problem: PrefixProblem,
    maximum_pivots: int,
    precision_schedule: tuple[int, ...],
    checkpoint_batch: int,
    checkpoint_path: Path,
    contract: dict[str, Any],
    pivot_budget: int = 0,
    cpu_park_callback: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if checkpoint_path.exists():
        checkpoint = _load_checkpoint(
            checkpoint_path,
            contract,
            precision_schedule,
        )
    else:
        checkpoint = _checkpoint_payload(contract, precision_schedule)
        _write_checkpoint(checkpoint_path, checkpoint)

    if checkpoint["overall_status"] == "completed":
        _restore_row_entries(
            problem,
            checkpoint["current_attempt"],
            maximum_pivots,
        )
        return checkpoint
    if checkpoint["overall_status"] in (
        "failed_symbolic_pattern",
        "failed_precision_schedule",
    ):
        return checkpoint
    checkpoint["overall_status"] = "running"

    remaining_budget = pivot_budget
    while True:
        attempt = checkpoint["current_attempt"]

        def flush() -> None:
            _write_checkpoint(checkpoint_path, checkpoint)

        outcome, completed = _continue_attempt(
            problem,
            attempt,
            maximum_pivots,
            remaining_budget,
            checkpoint_batch,
            flush,
            cpu_park_callback,
        )
        if remaining_budget > 0:
            remaining_budget -= completed
        if outcome == "complete":
            checkpoint["overall_status"] = "completed"
            _write_checkpoint(checkpoint_path, checkpoint)
            return checkpoint
        if outcome in ("cpu_park", "pivot_budget_park"):
            checkpoint["overall_status"] = "parked"
            checkpoint["park_reason"] = outcome
            _write_checkpoint(checkpoint_path, checkpoint)
            return checkpoint
        checkpoint["attempt_summaries"].append(
            _summarize_attempt(attempt)
        )
        next_index = int(checkpoint["current_attempt_index"]) + 1
        if next_index >= len(precision_schedule):
            checkpoint["overall_status"] = "failed_precision_schedule"
            _write_checkpoint(checkpoint_path, checkpoint)
            return checkpoint
        checkpoint["current_attempt_index"] = next_index
        checkpoint["current_attempt"] = _attempt_state(
            precision_schedule[next_index]
        )
        _write_checkpoint(checkpoint_path, checkpoint)
        if remaining_budget == 0 and pivot_budget > 0:
            checkpoint["overall_status"] = "parked"
            checkpoint["park_reason"] = "pivot_budget_park"
            _write_checkpoint(checkpoint_path, checkpoint)
            return checkpoint


def _prepare_production_problem(
    complete_result_path: Path,
    matrices_path: Path,
    gaussian_result_path: Path,
    gaussian_checkpoint_path: Path,
) -> tuple[PrefixProblem, dict[str, Any]]:
    base = _load_module(
        "neutral_strip_weighted_hypercircle_central_factorization_audit.py",
        "directed_ldl_central_base",
    )
    center, radius, inventory = base._load_complete_pencil(
        complete_result_path,
        matrices_path,
        gaussian_result_path,
        gaussian_checkpoint_path,
    )
    scale, scale_inventory = base._symmetric_ruiz_scaling(
        center,
        RUIZ_ITERATIONS,
    )
    scaled_center = base._symmetric_scale(center, scale)
    factor = splu(
        scaled_center,
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    lower = factor.L.tocsc()
    upper = factor.U.tocsc()
    lower.sort_indices()
    upper.sort_indices()
    raw_permutation = np.asarray(factor.perm_r)
    order = np.argsort(raw_permutation).astype(np.int64)
    positions = np.empty(len(order), dtype=np.int64)
    positions[order] = np.arange(len(order), dtype=np.int64)
    central_pivots = np.asarray(upper.diagonal(), dtype=float)

    permuted = scaled_center[order, :][:, order].tocsr()
    lu_residual = (permuted - lower @ upper).tocsr()
    lu_residual.eliminate_zeros()
    matrix_norm = base._row_sum_norm(permuted)
    permutation_semantics_relative_residual = (
        base._row_sum_norm(lu_residual) / matrix_norm
    )
    ldl_defect = (
        upper - diags(central_pivots, format="csc") @ lower.transpose()
    ).tocsr()
    ldl_defect.eliminate_zeros()
    ldl_relation_relative_residual = (
        base._row_sum_norm(ldl_defect)
        / max(base._row_sum_norm(upper), np.finfo(float).tiny)
    )

    radius_missing = _pattern_missing_count(radius, center)
    lower_pattern = lower.copy()
    upper_transpose_pattern = upper.transpose().tocsc()
    lower_pattern.data = np.ones_like(lower_pattern.data)
    upper_transpose_pattern.data = np.ones_like(
        upper_transpose_pattern.data
    )
    factor_envelope_pattern = lower_pattern.maximum(
        upper_transpose_pattern
    ).tocsc()
    factor_envelope_pattern.sort_indices()
    upper_only_pattern = (
        upper_transpose_pattern
        - upper_transpose_pattern.multiply(lower_pattern)
    ).tocsc()
    upper_only_pattern.eliminate_zeros()
    upper_derived_lower = (
        diags(1.0 / central_pivots, format="csc") @ upper
    ).transpose().tocsc()
    central_lower_envelope = (
        lower + upper_derived_lower.multiply(upper_only_pattern)
    ).tocsc()
    central_lower_envelope.sort_indices()
    union_pattern = (abs(center) + abs(radius)).tocsc()
    union_pattern.eliminate_zeros()
    permuted_union = union_pattern[order, :][:, order]
    required_lower = tril(permuted_union, format="csc")
    strict_required_lower = tril(permuted_union, k=-1, format="coo")
    first_input_dependent_pivot = (
        int(np.min(strict_required_lower.row))
        if strict_required_lower.nnz
        else None
    )
    input_missing = _pattern_missing_count(
        required_lower,
        factor_envelope_pattern,
    )
    factor_structures_transpose = _same_structure(
        upper,
        lower.transpose(),
    )

    hashes = {
        "scale_sha256": _sha256_arrays(scale),
        "raw_permutation_sha256": _sha256_arrays(raw_permutation),
        "order_sha256": _sha256_arrays(order),
        "factor_pattern_sha256": _sha256_arrays(
            lower.indptr,
            lower.indices,
            upper.indptr,
            upper.indices,
        ),
        "U_diagonal_sha256": _sha256_arrays(central_pivots),
    }
    frozen_hashes_match = bool(
        hashes["scale_sha256"] == EXPECTED_SCALE_SHA256
        and hashes["raw_permutation_sha256"]
        == EXPECTED_RAW_PERMUTATION_SHA256
        and hashes["order_sha256"] == EXPECTED_ORDER_SHA256
        and hashes["factor_pattern_sha256"]
        == EXPECTED_FACTOR_PATTERN_SHA256
        and hashes["U_diagonal_sha256"]
        == EXPECTED_U_DIAGONAL_SHA256
    )
    preparation_checks = {
        "dimension_matches": center.shape[0] == EXPECTED_DIMENSION,
        "central_nnz_matches": center.nnz == EXPECTED_CENTRAL_NNZ,
        "radius_nnz_matches": radius.nnz == EXPECTED_RADIUS_NNZ,
        "row_and_column_permutations_equal": bool(
            np.array_equal(factor.perm_r, factor.perm_c)
        ),
        "radius_pattern_contained_in_center": radius_missing == 0,
        "input_pattern_contained_in_factor_envelope": input_missing == 0,
        "argsort_permutation_semantics_verified": (
            permutation_semantics_relative_residual < 1.0e-10
        ),
        "central_LDL_relation_verified": (
            ldl_relation_relative_residual < 1.0e-10
        ),
        "frozen_hashes_match": frozen_hashes_match,
    }
    if not all(preparation_checks.values()):
        raise RuntimeError(
            "production directed-LDL preparation checks did not close: "
            + json.dumps(preparation_checks, sort_keys=True)
        )

    problem = PrefixProblem(
        center=center.tocsr(),
        radius=radius.tocsr(),
        scale=scale,
        order=order,
        positions=positions,
        lower=central_lower_envelope,
        central_pivots=central_pivots,
    )
    problem.center.sort_indices()
    problem.radius.sort_indices()
    return problem, {
        "matrix_inventory": inventory,
        "scale_inventory": scale_inventory,
        "hashes": hashes,
        "preparation_checks": preparation_checks,
        "factor_pattern_diagnostics": {
            "factor_structures_are_transposes": (
                factor_structures_transpose
            ),
            "L_entries_missing_from_U_transpose": (
                _pattern_missing_count(
                    lower_pattern,
                    upper_transpose_pattern,
                )
            ),
            "U_transpose_entries_missing_from_L": int(
                upper_only_pattern.nnz
            ),
            "symmetric_factor_envelope_nnz": int(
                factor_envelope_pattern.nnz
            ),
            "symmetric_factor_envelope_sha256": _sha256_arrays(
                factor_envelope_pattern.indptr,
                factor_envelope_pattern.indices,
            ),
            "numerical_factor_pattern_is_not_used_for_symbolic_certification": (
                True
            ),
        },
        "radius_pattern_entries_missing_from_center": radius_missing,
        "input_pattern_entries_missing_from_factor_envelope": (
            input_missing
        ),
        "first_input_graph_dependent_pivot": (
            first_input_dependent_pivot
        ),
        "initial_input_graph_independent_pivot_count": (
            first_input_dependent_pivot
        ),
        "permutation_semantics_relative_infinity_residual": (
            permutation_semantics_relative_residual
        ),
        "central_LDL_relation_relative_infinity_residual": (
            ldl_relation_relative_residual
        ),
        "factor_L_nnz": int(lower.nnz),
        "factor_U_nnz": int(upper.nnz),
    }


def run_audit(
    complete_result_path: Path = DEFAULT_COMPLETE_RESULT,
    matrices_path: Path = DEFAULT_MATRICES,
    gaussian_result_path: Path = DEFAULT_GAUSSIAN_RESULT,
    gaussian_checkpoint_path: Path = DEFAULT_GAUSSIAN_CHECKPOINT,
    central_audit_path: Path = DEFAULT_CENTRAL_AUDIT,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    maximum_pivots: int = DEFAULT_MAXIMUM_PIVOTS,
    precision_schedule: tuple[int, ...] = DEFAULT_PRECISIONS,
    checkpoint_batch: int = DEFAULT_CHECKPOINT_BATCH,
    pivot_budget: int = 0,
    enforce_cpu_policy: bool = True,
) -> dict[str, Any]:
    if not 0 < maximum_pivots <= EXPECTED_DIMENSION:
        raise ValueError("maximum pivots is out of range")
    if (
        not precision_schedule
        or tuple(sorted(set(precision_schedule))) != precision_schedule
    ):
        raise ValueError("precisions must be unique and increasing")
    if checkpoint_batch <= 0:
        raise ValueError("checkpoint batch must be positive")
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    baseline_samples = (
        _sample_cpu_baseline(5) if enforce_cpu_policy else []
    )
    baseline_mean = (
        sum(baseline_samples) / len(baseline_samples)
        if baseline_samples
        else None
    )
    if (
        baseline_mean is not None
        and baseline_mean > DAYTIME_BASELINE_CPU_LIMIT
    ):
        raise RuntimeError(
            "daytime baseline CPU exceeds the one-worker launch limit: "
            f"{baseline_mean:.3f}%"
        )

    problem, preparation = _prepare_production_problem(
        complete_result_path,
        matrices_path,
        gaussian_result_path,
        gaussian_checkpoint_path,
    )
    central_audit = json.loads(
        central_audit_path.read_text(encoding="ascii")
    )
    if (
        central_audit["recommended_scale_sha256"]
        != EXPECTED_SCALE_SHA256
        or central_audit["recommended_permutation_sha256"]
        != EXPECTED_RAW_PERMUTATION_SHA256
        or central_audit["recommended_factor_pattern_sha256"]
        != EXPECTED_FACTOR_PATTERN_SHA256
    ):
        raise RuntimeError("central audit recommendations changed")

    contract = {
        "algorithm_version": ALGORITHM_VERSION,
        "maximum_pivots": maximum_pivots,
        "checkpoint_batch": checkpoint_batch,
        "precision_schedule": list(precision_schedule),
        "dimension": problem.dimension,
        "congruence": "ten-step symmetric Ruiz scaling",
        "ordering": "MMD_AT_PLUS_A",
        "scale_sha256": preparation["hashes"]["scale_sha256"],
        "raw_permutation_sha256": preparation["hashes"][
            "raw_permutation_sha256"
        ],
        "order_sha256": preparation["hashes"]["order_sha256"],
        "factor_pattern_sha256": preparation["hashes"][
            "factor_pattern_sha256"
        ],
        "U_diagonal_sha256": preparation["hashes"][
            "U_diagonal_sha256"
        ],
        "complete_assembly_result_sha256": _sha256_file(
            complete_result_path
        ),
        "matrix_archive_sha256": _sha256_file(matrices_path),
        "Gaussian_result_sha256": _sha256_file(
            gaussian_result_path
        ),
        "Gaussian_checkpoint_sha256": _sha256_file(
            gaussian_checkpoint_path
        ),
        "central_audit_sha256": _sha256_file(central_audit_path),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }

    cpu_samples: list[float] = []

    def cpu_park() -> bool:
        if not enforce_cpu_policy:
            return False
        try:
            import psutil

            cpu_samples.append(float(psutil.cpu_percent(interval=0.25)))
        except Exception:
            return False
        return bool(
            len(cpu_samples) >= 2
            and all(
                value > DAYTIME_PARK_CPU_LIMIT
                for value in cpu_samples[-2:]
            )
        )

    checkpoint = _run_adaptive_prefix(
        problem,
        maximum_pivots,
        precision_schedule,
        checkpoint_batch,
        checkpoint_path,
        contract,
        pivot_budget=pivot_budget,
        cpu_park_callback=cpu_park,
    )
    replayed = _load_checkpoint(
        checkpoint_path,
        contract,
        precision_schedule,
    )
    checkpoint_replays = replayed == checkpoint
    attempt = checkpoint["current_attempt"]
    final_summary = _summarize_attempt(attempt)
    edge_count = int(
        preparation["matrix_inventory"]["edge_count"]
    )
    triangle_count = int(
        preparation["matrix_inventory"]["triangle_count"]
    )
    state_count = int(
        preparation["matrix_inventory"]["state_count"]
    )
    block_boundaries = (
        ("edge_metric", 0, edge_count),
        (
            "triangle_constraint",
            edge_count,
            edge_count + triangle_count,
        ),
        (
            "state",
            edge_count + triangle_count,
            edge_count + triangle_count + state_count,
        ),
        (
            "source_triangle",
            edge_count + triangle_count + state_count,
            problem.dimension,
        ),
    )
    pivot_block_counts = {
        name: int(
            np.count_nonzero(
                (
                    problem.order[
                        : final_summary["completed_pivot_count"]
                    ]
                    >= start
                )
                & (
                    problem.order[
                        : final_summary["completed_pivot_count"]
                    ]
                    < stop
                )
            )
        )
        for name, start, stop in block_boundaries
    }
    interaction_indices = np.asarray(
        [
            int(row["index"])
            for row in attempt["pivot_diagnostics"]
            if int(row["diagonal_term_count"]) > 0
        ],
        dtype=np.int64,
    )
    interaction_original_indices = (
        problem.order[interaction_indices]
        if len(interaction_indices)
        else np.empty(0, dtype=np.int64)
    )
    interaction_block_counts = {
        name: int(
            np.count_nonzero(
                (interaction_original_indices >= start)
                & (interaction_original_indices < stop)
            )
        )
        for name, start, stop in block_boundaries
    }
    interaction_profile = dict(final_summary["interaction_profile"])
    interaction_profile["block_counts"] = interaction_block_counts
    completed = checkpoint["overall_status"] == "completed"
    structural_failure = (
        checkpoint["overall_status"] == "failed_symbolic_pattern"
    )
    integrity_checks = {
        **preparation["preparation_checks"],
        "checkpoint_state_hash_replays": checkpoint_replays,
        "checkpoint_semantics_validate": True,
        "no_symbolic_pattern_failure": not structural_failure,
        "full_inertia_claim_remains_false": True,
    }
    all_stage_checks = bool(all(integrity_checks.values()))
    status = {
        "completed": "certified_bounded_prefix",
        "parked": "parked_with_valid_checkpoint",
        "failed_precision_schedule": "bounded_prefix_not_certified",
        "failed_symbolic_pattern": "symbolic_pattern_failure",
    }[checkpoint["overall_status"]]

    return {
        "kind": "hypercircle-directed-ldl-prefix-audit",
        "status": status,
        "all_current_stage_checks_pass": all_stage_checks,
        "contract": contract,
        "checks": integrity_checks,
        "preparation": preparation,
        "directed_LDL_prefix": {
            "arithmetic": (
                "componentwise outward-rounded Decimal interval recurrence"
            ),
            "input_interval": (
                "stored binary64 center plus/minus stored nonnegative "
                "binary64 radius, followed by positive Ruiz congruence"
            ),
            "requested_pivot_count": maximum_pivots,
            "completed_pivot_count": final_summary[
                "completed_pivot_count"
            ],
            "precision_used": final_summary["precision"],
            "negative_pivot_count": final_summary[
                "negative_pivot_count"
            ],
            "positive_pivot_count": final_summary[
                "positive_pivot_count"
            ],
            "minimum_pivot_margin_decimal": final_summary[
                "minimum_pivot_margin_decimal"
            ],
            "minimum_pivot_margin_index": final_summary[
                "minimum_pivot_margin_index"
            ],
            "maximum_pivot_radius_to_margin_upper_decimal": (
                final_summary[
                    "maximum_pivot_radius_to_margin_upper_decimal"
                ]
            ),
            "maximum_pivot_radius_to_margin_index": final_summary[
                "maximum_pivot_radius_to_margin_index"
            ],
            "maximum_cancellation_charge_upper_decimal": (
                final_summary[
                    "maximum_cancellation_charge_upper_decimal"
                ]
            ),
            "maximum_cancellation_charge_index": final_summary[
                "maximum_cancellation_charge_index"
            ],
            "maximum_lower_interval_width_decimal": final_summary[
                "maximum_lower_interval_width_decimal"
            ],
            "maximum_symbolic_descendant_count": final_summary[
                "maximum_symbolic_descendant_count"
            ],
            "symbolic_lower_entry_count": final_summary[
                "symbolic_lower_entry_count"
            ],
            "symbolic_lower_coordinates_sha256": final_summary[
                "symbolic_lower_coordinates_sha256"
            ],
            "lower_interval_entries_sha256": final_summary[
                "lower_interval_entries_sha256"
            ],
            "pivot_intervals_sha256": final_summary[
                "pivot_intervals_sha256"
            ],
            "central_superlu_pivots_contained_count": final_summary[
                "central_superlu_pivots_contained_count"
            ],
            "central_factor_envelope_missing_count": final_summary[
                "central_factor_envelope_missing_count"
            ],
            "pivots_with_nonzero_diagonal_term_count": final_summary[
                "pivots_with_nonzero_diagonal_term_count"
            ],
            "first_pivot_with_nonzero_diagonal_term": final_summary[
                "first_pivot_with_nonzero_diagonal_term"
            ],
            "total_diagonal_recurrence_term_count": final_summary[
                "total_diagonal_recurrence_term_count"
            ],
            "total_off_diagonal_recurrence_term_count": final_summary[
                "total_off_diagonal_recurrence_term_count"
            ],
            "pivot_block_counts": pivot_block_counts,
            "interaction_profile": interaction_profile,
            "failed_pivot": attempt["failed_pivot"],
            "prior_precision_attempts": checkpoint[
                "attempt_summaries"
            ],
            "pivot_diagnostics": attempt["pivot_diagnostics"],
            "checkpoint_state_sha256": checkpoint["state_sha256"],
        },
        "certification_scope": (
            "When the bounded-prefix flag is true, the reported signs enclose "
            "the exact one-by-one LDL pivots for every symmetric matrix in "
            "the stored entrywise interval family after the fixed positive "
            "Ruiz congruence and fixed elimination order. This certifies only "
            "the requested leading pivot prefix, not the remaining pivots, "
            "the full matrix inertia, or the continuum spectral statement."
        ),
        "certification_flags": {
            "bounded_prefix_directed_LDL_certified": completed,
            "full_123816_pivot_inertia_certified": False,
            "reversible_weighted_hypercircle_full_inertia_certified": False,
            "continuum_spectrum_below_60_captured": False,
        },
        "runtime": {
            "below_normal_priority_set": priority_set,
            "baseline_cpu_samples_percent": baseline_samples,
            "baseline_cpu_mean_percent": baseline_mean,
            "periodic_cpu_samples_percent": cpu_samples,
            "elapsed_seconds": time.perf_counter() - started,
            "pivot_budget": pivot_budget,
        },
        "artifacts": {
            "complete_assembly_result": str(
                complete_result_path
            ).replace("\\", "/"),
            "matrix_archive": str(matrices_path).replace("\\", "/"),
            "Gaussian_result": str(gaussian_result_path).replace("\\", "/"),
            "Gaussian_checkpoint": str(
                gaussian_checkpoint_path
            ).replace("\\", "/"),
            "central_factorization_audit": str(
                central_audit_path
            ).replace("\\", "/"),
            "checkpoint": str(checkpoint_path).replace("\\", "/"),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
        },
        "next_required_step": (
            f"Independently replay the bounded {maximum_pivots}-pivot "
            "certificate and inspect its pivot-margin, interval-radius, "
            "cancellation, and interaction profiles. Only then choose "
            "another bounded structural transition; do not launch all "
            "123816 pivots."
            if completed
            else "Inspect the recorded fail-closed pivot or resume the "
            "hash-bound checkpoint under the daytime resource policy."
        ),
    }


def _parse_precisions(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "precisions must be comma-separated integers"
        ) from error
    if not result or any(value < 8 for value in result):
        raise argparse.ArgumentTypeError(
            "every Decimal precision must be at least 8"
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complete-result",
        type=Path,
        default=DEFAULT_COMPLETE_RESULT,
    )
    parser.add_argument(
        "--matrices",
        type=Path,
        default=DEFAULT_MATRICES,
    )
    parser.add_argument(
        "--gaussian-result",
        type=Path,
        default=DEFAULT_GAUSSIAN_RESULT,
    )
    parser.add_argument(
        "--gaussian-checkpoint",
        type=Path,
        default=DEFAULT_GAUSSIAN_CHECKPOINT,
    )
    parser.add_argument(
        "--central-audit",
        type=Path,
        default=DEFAULT_CENTRAL_AUDIT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--maximum-pivots",
        type=int,
        default=DEFAULT_MAXIMUM_PIVOTS,
    )
    parser.add_argument(
        "--precisions",
        type=_parse_precisions,
        default=DEFAULT_PRECISIONS,
    )
    parser.add_argument(
        "--checkpoint-batch",
        type=int,
        default=DEFAULT_CHECKPOINT_BATCH,
    )
    parser.add_argument("--pivot-budget", type=int, default=0)
    parser.add_argument("--skip-cpu-policy", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_audit(
        complete_result_path=args.complete_result,
        matrices_path=args.matrices,
        gaussian_result_path=args.gaussian_result,
        gaussian_checkpoint_path=args.gaussian_checkpoint,
        central_audit_path=args.central_audit,
        checkpoint_path=args.checkpoint,
        maximum_pivots=args.maximum_pivots,
        precision_schedule=args.precisions,
        checkpoint_batch=args.checkpoint_batch,
        pivot_budget=args.pivot_budget,
        enforce_cpu_policy=not args.skip_cpu_policy,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
