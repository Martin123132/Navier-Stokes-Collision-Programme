"""Certify indexed stored-pencil eigenvalues by directed sparse LDL inertia."""

from __future__ import annotations

import argparse
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import splu


ASSEMBLY_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_checkpoint_v1.npz"
)
EIGENSYSTEM_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_eigensystem_residual_audit_v1.json"
)
PIVOT_CACHE = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_sparse_inertia_pivots_v1.npz"
)
ROW_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_sparse_inertia_row_checkpoint_v1.json"
)
ROW_CHECKPOINT_CACHE = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_sparse_inertia_row_checkpoint_v1.npz"
)


Interval = tuple[float, float]


def _set_below_normal_priority() -> bool:
    try:
        import psutil

        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return process.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
    except Exception:
        return False


def _down(value: float) -> float:
    return float(np.nextafter(float(value), -math.inf))


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _iv_add(first: Interval, second: Interval) -> Interval:
    return _down(first[0] + second[0]), _up(first[1] + second[1])


def _iv_sub(first: Interval, second: Interval) -> Interval:
    return _down(first[0] - second[1]), _up(first[1] - second[0])


def _iv_mul(first: Interval, second: Interval) -> Interval:
    products = (
        first[0] * second[0],
        first[0] * second[1],
        first[1] * second[0],
        first[1] * second[1],
    )
    return _down(min(products)), _up(max(products))


def _iv_div(first: Interval, second: Interval) -> Interval:
    if second[0] <= 0.0 <= second[1]:
        raise ZeroDivisionError("pivot interval contains zero")
    reciprocal = (_down(1.0 / second[1]), _up(1.0 / second[0]))
    return _iv_mul(first, reciprocal)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text_rows(*rows: list[Decimal]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for value in row:
            encoded = str(value).encode("ascii")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def _load_forms(path: Path) -> tuple[csr_matrix, csr_matrix]:
    with np.load(path, allow_pickle=False) as checkpoint:
        next_triangle = int(
            checkpoint["next_selected_position"].item()
        )
        if next_triangle != 30954:
            raise RuntimeError("assembly checkpoint is incomplete")
        state_count = 15211
        shape = (state_count, state_count)
        mass = coo_matrix(
            (
                checkpoint["mass_values"],
                (
                    checkpoint["mass_rows"],
                    checkpoint["mass_columns"],
                ),
            ),
            shape=shape,
        ).tocsr()
        stiffness = coo_matrix(
            (
                checkpoint["stiffness_values"],
                (
                    checkpoint["stiffness_rows"],
                    checkpoint["stiffness_columns"],
                ),
            ),
            shape=shape,
        ).tocsr()
    mass.sort_indices()
    stiffness.sort_indices()
    return mass, stiffness


def _sparse_entry(matrix: csr_matrix, row: int, column: int) -> float:
    start = matrix.indptr[row]
    stop = matrix.indptr[row + 1]
    indices = matrix.indices[start:stop]
    offset = int(np.searchsorted(indices, column))
    if offset < len(indices) and int(indices[offset]) == column:
        return float(matrix.data[start + offset])
    return 0.0


def _pencil_entry(
    stiffness: csr_matrix,
    mass: csr_matrix,
    shift: float,
    row: int,
    column: int,
) -> Interval:
    stiffness_value = _sparse_entry(stiffness, row, column)
    mass_value = _sparse_entry(mass, row, column)
    shifted_mass = _iv_mul(
        (shift, shift),
        (mass_value, mass_value),
    )
    return _iv_sub(
        (stiffness_value, stiffness_value),
        shifted_mass,
    )


def _structure_matches(first: csr_matrix, second: csr_matrix) -> bool:
    return bool(
        np.array_equal(first.indptr, second.indptr)
        and np.array_equal(first.indices, second.indices)
    )


def _is_exactly_symmetric(matrix: csr_matrix) -> bool:
    transpose = matrix.transpose().tocsr()
    transpose.sort_indices()
    return bool(
        _structure_matches(matrix, transpose)
        and np.array_equal(matrix.data, transpose.data)
    )


def _symbolic_ldl_pattern_closed(
    lower_by_column,
    lower_factor_positions: dict[tuple[int, int], int],
) -> tuple[bool, dict[str, int] | None, int]:
    checked_pairs = 0
    for column in range(lower_by_column.shape[1]):
        descendants = lower_by_column.indices[
            lower_by_column.indptr[column]:
            lower_by_column.indptr[column + 1]
        ]
        descendants = descendants[descendants > column]
        for first_offset in range(len(descendants)):
            first = int(descendants[first_offset])
            for second_offset in range(first_offset):
                second = int(descendants[second_offset])
                checked_pairs += 1
                if (max(first, second), min(first, second)) not in (
                    lower_factor_positions
                ):
                    return (
                        False,
                        {
                            "elimination_column": column,
                            "first_descendant": first,
                            "second_descendant": second,
                        },
                        checked_pairs,
                    )
    return True, None, checked_pairs


def _exact_pattern_contained(
    stiffness: csr_matrix,
    mass: csr_matrix,
    lower_factor_positions: dict[tuple[int, int], int],
) -> bool:
    combined = (abs(stiffness) + abs(mass)).tocsr()
    combined.eliminate_zeros()
    for row in range(combined.shape[0]):
        for pointer in range(
            combined.indptr[row],
            combined.indptr[row + 1],
        ):
            column = int(combined.indices[pointer])
            lower_row = max(row, column)
            lower_column = min(row, column)
            if (lower_row, lower_column) not in lower_factor_positions:
                return False
    return True


def _directed_ldl(
    stiffness: csr_matrix,
    mass: csr_matrix,
    shift: float,
    maximum_pivots: int,
    ordering: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    dimension = mass.shape[0]
    central_pencil = (stiffness - shift * mass).tocsc()
    factor = splu(
        central_pencil,
        permc_spec=ordering,
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    permutations_equal = bool(
        np.array_equal(factor.perm_r, factor.perm_c)
    )
    order = np.argsort(factor.perm_r)
    permuted_stiffness = stiffness[order, :][:, order].tocsr()
    permuted_mass = mass[order, :][:, order].tocsr()
    permuted_stiffness.sort_indices()
    permuted_mass.sort_indices()

    lower = factor.L.tocsr()
    lower.sort_indices()
    lower_by_column = lower.tocsc()
    lower_by_column.sort_indices()
    upper = factor.U.tocsr()
    upper.sort_indices()
    lower_transpose = lower.transpose().tocsr()
    lower_transpose.sort_indices()
    factor_structures_transpose = _structure_matches(
        upper,
        lower_transpose,
    )

    positions = {
        (row, int(lower.indices[pointer])): pointer
        for row in range(dimension)
        for pointer in range(
            lower.indptr[row],
            lower.indptr[row + 1],
        )
    }
    exact_pattern_contained = _exact_pattern_contained(
        permuted_stiffness,
        permuted_mass,
        positions,
    )
    (
        symbolic_pattern_closed,
        symbolic_pattern_failure,
        symbolic_closure_pair_count,
    ) = _symbolic_ldl_pattern_closed(lower_by_column, positions)

    lower_intervals = np.zeros((2, lower.nnz), dtype=float)
    pivot_lower = np.empty(dimension)
    pivot_upper = np.empty(dimension)
    central_pivots = factor.U.diagonal()
    pivot_count = min(
        dimension,
        maximum_pivots if maximum_pivots > 0 else dimension,
    )

    negative_count = 0
    positive_count = 0
    failed_pivot = None
    minimum_pivot_margin = math.inf
    maximum_relative_pivot_width = 0.0
    maximum_lower_interval_width = 0.0
    all_central_pivots_contained = True
    all_central_lower_entries_contained = True
    started = time.perf_counter()

    for pivot_index in range(pivot_count):
        diagonal_sum = (0.0, 0.0)
        for pointer in range(
            lower.indptr[pivot_index],
            lower.indptr[pivot_index + 1],
        ):
            prior = int(lower.indices[pointer])
            if prior >= pivot_index:
                break
            lower_value = (
                float(lower_intervals[0, pointer]),
                float(lower_intervals[1, pointer]),
            )
            term = _iv_mul(lower_value, lower_value)
            term = _iv_mul(
                term,
                (
                    float(pivot_lower[prior]),
                    float(pivot_upper[prior]),
                ),
            )
            diagonal_sum = _iv_add(diagonal_sum, term)
        exact_diagonal = _pencil_entry(
            permuted_stiffness,
            permuted_mass,
            shift,
            pivot_index,
            pivot_index,
        )
        pivot_interval = _iv_sub(exact_diagonal, diagonal_sum)
        pivot_lower[pivot_index] = pivot_interval[0]
        pivot_upper[pivot_index] = pivot_interval[1]
        central_pivot = float(central_pivots[pivot_index])
        all_central_pivots_contained &= (
            pivot_interval[0]
            <= central_pivot
            <= pivot_interval[1]
        )
        if pivot_interval[0] <= 0.0 <= pivot_interval[1]:
            failed_pivot = {
                "index": pivot_index,
                "interval": list(pivot_interval),
                "central_superlu_pivot": central_pivot,
            }
            break
        if pivot_interval[1] < 0.0:
            negative_count += 1
        else:
            positive_count += 1
        minimum_pivot_margin = min(
            minimum_pivot_margin,
            abs(pivot_interval[0]),
            abs(pivot_interval[1]),
        )
        maximum_relative_pivot_width = max(
            maximum_relative_pivot_width,
            (pivot_interval[1] - pivot_interval[0])
            / max(
                abs(pivot_interval[0]),
                abs(pivot_interval[1]),
                1.0e-300,
            ),
        )

        diagonal_pointer = positions[(pivot_index, pivot_index)]
        lower_intervals[:, diagonal_pointer] = 1.0
        for column_pointer in range(
            lower_by_column.indptr[pivot_index],
            lower_by_column.indptr[pivot_index + 1],
        ):
            row = int(lower_by_column.indices[column_pointer])
            if row <= pivot_index:
                continue
            row_pointer = lower.indptr[row]
            row_stop = lower.indptr[row + 1]
            pivot_pointer = lower.indptr[pivot_index]
            pivot_stop = lower.indptr[pivot_index + 1]
            off_diagonal_sum = (0.0, 0.0)
            while row_pointer < row_stop and pivot_pointer < pivot_stop:
                row_column = int(lower.indices[row_pointer])
                pivot_column = int(lower.indices[pivot_pointer])
                if (
                    row_column >= pivot_index
                    or pivot_column >= pivot_index
                ):
                    break
                if row_column == pivot_column:
                    term = _iv_mul(
                        (
                            float(lower_intervals[0, row_pointer]),
                            float(lower_intervals[1, row_pointer]),
                        ),
                        (
                            float(lower_intervals[0, pivot_pointer]),
                            float(lower_intervals[1, pivot_pointer]),
                        ),
                    )
                    term = _iv_mul(
                        term,
                        (
                            float(pivot_lower[row_column]),
                            float(pivot_upper[row_column]),
                        ),
                    )
                    off_diagonal_sum = _iv_add(
                        off_diagonal_sum,
                        term,
                    )
                    row_pointer += 1
                    pivot_pointer += 1
                elif row_column < pivot_column:
                    row_pointer += 1
                else:
                    pivot_pointer += 1

            exact_entry = _pencil_entry(
                permuted_stiffness,
                permuted_mass,
                shift,
                row,
                pivot_index,
            )
            numerator = _iv_sub(exact_entry, off_diagonal_sum)
            lower_interval = _iv_div(numerator, pivot_interval)
            storage_pointer = positions[(row, pivot_index)]
            lower_intervals[0, storage_pointer] = lower_interval[0]
            lower_intervals[1, storage_pointer] = lower_interval[1]
            central_lower = float(lower.data[storage_pointer])
            all_central_lower_entries_contained &= (
                lower_interval[0]
                <= central_lower
                <= lower_interval[1]
            )
            maximum_lower_interval_width = max(
                maximum_lower_interval_width,
                lower_interval[1] - lower_interval[0],
            )

    completed_pivots = (
        pivot_count if failed_pivot is None else failed_pivot["index"]
    )
    complete = completed_pivots == dimension
    pivot_lower_output = pivot_lower[:completed_pivots].copy()
    pivot_upper_output = pivot_upper[:completed_pivots].copy()
    signs = np.sign(
        0.5 * (pivot_lower_output + pivot_upper_output)
    ).astype(np.int8)

    central_reconstruction = (
        lower @ diags(central_pivots) @ lower.transpose()
    ).tocsr()
    permuted_central_pencil = central_pencil[
        order, :
    ][:, order].tocsr()
    central_residual = (
        permuted_central_pencil - central_reconstruction
    ).tocsr()
    central_residual.eliminate_zeros()
    central_residual_inf = float(
        np.max(np.asarray(abs(central_residual).sum(axis=1)).reshape(-1))
    )

    checks = [
        permutations_equal,
        factor_structures_transpose,
        exact_pattern_contained,
        symbolic_pattern_closed,
        failed_pivot is None,
        all_central_pivots_contained,
        all_central_lower_entries_contained,
    ]
    if complete:
        checks.extend(
            [
                negative_count + positive_count == dimension,
                len(pivot_lower_output) == dimension,
            ]
        )
    return (
        {
            "shift": shift,
            "ordering": ordering,
            "dimension": dimension,
            "requested_pivot_count": pivot_count,
            "completed_pivot_count": completed_pivots,
            "complete_inertia": complete,
            "negative_pivot_count": negative_count,
            "positive_pivot_count": positive_count,
            "failed_pivot": failed_pivot,
            "minimum_pivot_interval_margin": (
                minimum_pivot_margin
                if math.isfinite(minimum_pivot_margin)
                else None
            ),
            "maximum_relative_pivot_interval_width": (
                maximum_relative_pivot_width
            ),
            "maximum_lower_factor_interval_width": (
                maximum_lower_interval_width
            ),
            "all_superlu_central_pivots_contained": (
                all_central_pivots_contained
            ),
            "all_superlu_central_lower_entries_contained": (
                all_central_lower_entries_contained
            ),
            "row_and_column_permutations_equal": permutations_equal,
            "factor_structures_are_transposes": (
                factor_structures_transpose
            ),
            "exact_pencil_pattern_contained_in_symbolic_factor": (
                exact_pattern_contained
            ),
            "symbolic_ldl_pattern_closed": symbolic_pattern_closed,
            "symbolic_ldl_pattern_failure": symbolic_pattern_failure,
            "symbolic_closure_pair_count": symbolic_closure_pair_count,
            "factor_nonzero_count": int(lower.nnz),
            "central_factorization_residual_inf": (
                central_residual_inf
            ),
            "permutation_sha256": _sha256_arrays(order),
            "pivot_intervals_sha256": _sha256_arrays(
                pivot_lower_output,
                pivot_upper_output,
                signs,
            ),
            "all_directed_ldl_checks_pass": bool(all(checks)),
            "elapsed_seconds": time.perf_counter() - started,
        },
        {
            "shift": np.asarray(shift),
            "permutation": order.astype(np.int64),
            "pivot_lower": pivot_lower_output,
            "pivot_upper": pivot_upper_output,
            "pivot_sign": signs,
        },
    )


def _directed_ldl_decimal(
    stiffness: csr_matrix,
    mass: csr_matrix,
    shift: float,
    maximum_pivots: int,
    ordering: str,
    precision: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    dimension = mass.shape[0]
    central_pencil = (stiffness - shift * mass).tocsc()
    factor = splu(
        central_pencil,
        permc_spec=ordering,
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    permutations_equal = bool(
        np.array_equal(factor.perm_r, factor.perm_c)
    )
    order = np.argsort(factor.perm_r)
    permuted_stiffness = stiffness[order, :][:, order].tocsr()
    permuted_mass = mass[order, :][:, order].tocsr()
    permuted_stiffness.sort_indices()
    permuted_mass.sort_indices()

    lower = factor.L.tocsr()
    lower.sort_indices()
    lower_by_column = lower.tocsc()
    lower_by_column.sort_indices()
    upper = factor.U.tocsr()
    upper.sort_indices()
    lower_transpose = lower.transpose().tocsr()
    lower_transpose.sort_indices()
    factor_structures_transpose = _structure_matches(
        upper,
        lower_transpose,
    )
    positions = {
        (row, int(lower.indices[pointer])): pointer
        for row in range(dimension)
        for pointer in range(
            lower.indptr[row],
            lower.indptr[row + 1],
        )
    }
    exact_pattern_contained = _exact_pattern_contained(
        permuted_stiffness,
        permuted_mass,
        positions,
    )
    (
        symbolic_pattern_closed,
        symbolic_pattern_failure,
        symbolic_closure_pair_count,
    ) = _symbolic_ldl_pattern_closed(lower_by_column, positions)

    lower_context = Context(prec=precision, rounding=ROUND_FLOOR)
    upper_context = Context(prec=precision, rounding=ROUND_CEILING)
    zero = Decimal(0)
    one = Decimal(1)
    shift_decimal = Decimal.from_float(shift)

    def interval_add(
        first: tuple[Decimal, Decimal],
        second: tuple[Decimal, Decimal],
    ) -> tuple[Decimal, Decimal]:
        return (
            lower_context.add(first[0], second[0]),
            upper_context.add(first[1], second[1]),
        )

    def interval_subtract(
        first: tuple[Decimal, Decimal],
        second: tuple[Decimal, Decimal],
    ) -> tuple[Decimal, Decimal]:
        return (
            lower_context.subtract(first[0], second[1]),
            upper_context.subtract(first[1], second[0]),
        )

    def interval_multiply(
        first: tuple[Decimal, Decimal],
        second: tuple[Decimal, Decimal],
    ) -> tuple[Decimal, Decimal]:
        lower_products = [
            lower_context.multiply(first_value, second_value)
            for first_value in first
            for second_value in second
        ]
        upper_products = [
            upper_context.multiply(first_value, second_value)
            for first_value in first
            for second_value in second
        ]
        return min(lower_products), max(upper_products)

    def interval_divide(
        first: tuple[Decimal, Decimal],
        second: tuple[Decimal, Decimal],
    ) -> tuple[Decimal, Decimal]:
        if second[0] <= zero <= second[1]:
            raise ZeroDivisionError("pivot interval contains zero")
        reciprocal = (
            lower_context.divide(one, second[1]),
            upper_context.divide(one, second[0]),
        )
        return interval_multiply(first, reciprocal)

    def exact_pencil_entry(row: int, column: int):
        stiffness_value = Decimal.from_float(
            _sparse_entry(permuted_stiffness, row, column)
        )
        mass_value = Decimal.from_float(
            _sparse_entry(permuted_mass, row, column)
        )
        shifted_mass = interval_multiply(
            (shift_decimal, shift_decimal),
            (mass_value, mass_value),
        )
        return interval_subtract(
            (stiffness_value, stiffness_value),
            shifted_mass,
        )

    lower_interval_lower = [zero] * lower.nnz
    lower_interval_upper = [zero] * lower.nnz
    pivot_interval_lower = [zero] * dimension
    pivot_interval_upper = [zero] * dimension
    central_pivots = factor.U.diagonal()
    pivot_count = min(
        dimension,
        maximum_pivots if maximum_pivots > 0 else dimension,
    )

    negative_count = 0
    positive_count = 0
    failed_pivot = None
    minimum_pivot_margin = Decimal("Infinity")
    maximum_relative_pivot_width = zero
    maximum_lower_interval_width = zero
    all_central_pivots_contained = True
    all_central_lower_entries_contained = True
    started = time.perf_counter()

    for pivot_index in range(pivot_count):
        diagonal_sum = (zero, zero)
        for pointer in range(
            lower.indptr[pivot_index],
            lower.indptr[pivot_index + 1],
        ):
            prior = int(lower.indices[pointer])
            if prior >= pivot_index:
                break
            lower_value = (
                lower_interval_lower[pointer],
                lower_interval_upper[pointer],
            )
            term = interval_multiply(lower_value, lower_value)
            term = interval_multiply(
                term,
                (
                    pivot_interval_lower[prior],
                    pivot_interval_upper[prior],
                ),
            )
            diagonal_sum = interval_add(diagonal_sum, term)
        pivot_interval = interval_subtract(
            exact_pencil_entry(pivot_index, pivot_index),
            diagonal_sum,
        )
        pivot_interval_lower[pivot_index] = pivot_interval[0]
        pivot_interval_upper[pivot_index] = pivot_interval[1]
        central_pivot = Decimal.from_float(
            float(central_pivots[pivot_index])
        )
        all_central_pivots_contained &= (
            pivot_interval[0]
            <= central_pivot
            <= pivot_interval[1]
        )
        if pivot_interval[0] <= zero <= pivot_interval[1]:
            failed_pivot = {
                "index": pivot_index,
                "interval_decimal": [
                    str(pivot_interval[0]),
                    str(pivot_interval[1]),
                ],
                "central_superlu_pivot": float(
                    central_pivots[pivot_index]
                ),
            }
            break
        if pivot_interval[1] < zero:
            negative_count += 1
        else:
            positive_count += 1
        minimum_pivot_margin = min(
            minimum_pivot_margin,
            abs(pivot_interval[0]),
            abs(pivot_interval[1]),
        )
        interval_width = upper_context.subtract(
            pivot_interval[1],
            pivot_interval[0],
        )
        relative_width = upper_context.divide(
            interval_width,
            max(abs(pivot_interval[0]), abs(pivot_interval[1])),
        )
        maximum_relative_pivot_width = max(
            maximum_relative_pivot_width,
            relative_width,
        )

        diagonal_pointer = positions[(pivot_index, pivot_index)]
        lower_interval_lower[diagonal_pointer] = one
        lower_interval_upper[diagonal_pointer] = one
        for column_pointer in range(
            lower_by_column.indptr[pivot_index],
            lower_by_column.indptr[pivot_index + 1],
        ):
            row = int(lower_by_column.indices[column_pointer])
            if row <= pivot_index:
                continue
            row_pointer = lower.indptr[row]
            row_stop = lower.indptr[row + 1]
            pivot_pointer = lower.indptr[pivot_index]
            pivot_stop = lower.indptr[pivot_index + 1]
            off_diagonal_sum = (zero, zero)
            while row_pointer < row_stop and pivot_pointer < pivot_stop:
                row_column = int(lower.indices[row_pointer])
                pivot_column = int(lower.indices[pivot_pointer])
                if (
                    row_column >= pivot_index
                    or pivot_column >= pivot_index
                ):
                    break
                if row_column == pivot_column:
                    term = interval_multiply(
                        (
                            lower_interval_lower[row_pointer],
                            lower_interval_upper[row_pointer],
                        ),
                        (
                            lower_interval_lower[pivot_pointer],
                            lower_interval_upper[pivot_pointer],
                        ),
                    )
                    term = interval_multiply(
                        term,
                        (
                            pivot_interval_lower[row_column],
                            pivot_interval_upper[row_column],
                        ),
                    )
                    off_diagonal_sum = interval_add(
                        off_diagonal_sum,
                        term,
                    )
                    row_pointer += 1
                    pivot_pointer += 1
                elif row_column < pivot_column:
                    row_pointer += 1
                else:
                    pivot_pointer += 1

            numerator = interval_subtract(
                exact_pencil_entry(row, pivot_index),
                off_diagonal_sum,
            )
            lower_interval = interval_divide(
                numerator,
                pivot_interval,
            )
            storage_pointer = positions[(row, pivot_index)]
            lower_interval_lower[storage_pointer] = lower_interval[0]
            lower_interval_upper[storage_pointer] = lower_interval[1]
            central_lower = Decimal.from_float(
                float(lower.data[storage_pointer])
            )
            all_central_lower_entries_contained &= (
                lower_interval[0]
                <= central_lower
                <= lower_interval[1]
            )
            maximum_lower_interval_width = max(
                maximum_lower_interval_width,
                upper_context.subtract(
                    lower_interval[1],
                    lower_interval[0],
                ),
            )

    completed_pivots = (
        pivot_count if failed_pivot is None else failed_pivot["index"]
    )
    complete = completed_pivots == dimension
    pivot_lower_output = pivot_interval_lower[:completed_pivots]
    pivot_upper_output = pivot_interval_upper[:completed_pivots]
    signs = np.asarray(
        [
            -1 if upper < zero else 1
            for upper in pivot_upper_output
        ],
        dtype=np.int8,
    )

    central_reconstruction = (
        lower @ diags(central_pivots) @ lower.transpose()
    ).tocsr()
    permuted_central_pencil = central_pencil[
        order, :
    ][:, order].tocsr()
    central_residual = (
        permuted_central_pencil - central_reconstruction
    ).tocsr()
    central_residual.eliminate_zeros()
    central_residual_inf = float(
        np.max(np.asarray(abs(central_residual).sum(axis=1)).reshape(-1))
    )

    checks = [
        permutations_equal,
        factor_structures_transpose,
        exact_pattern_contained,
        symbolic_pattern_closed,
        failed_pivot is None,
    ]
    if complete:
        checks.extend(
            [
                negative_count + positive_count == dimension,
                len(pivot_lower_output) == dimension,
            ]
        )
    return (
        {
            "shift": shift,
            "ordering": ordering,
            "arithmetic": "directed Decimal interval recurrence",
            "decimal_precision": precision,
            "dimension": dimension,
            "requested_pivot_count": pivot_count,
            "completed_pivot_count": completed_pivots,
            "complete_inertia": complete,
            "negative_pivot_count": negative_count,
            "positive_pivot_count": positive_count,
            "failed_pivot": failed_pivot,
            "minimum_pivot_interval_margin": (
                float(minimum_pivot_margin)
                if minimum_pivot_margin.is_finite()
                else None
            ),
            "minimum_pivot_interval_margin_decimal": (
                str(minimum_pivot_margin)
                if minimum_pivot_margin.is_finite()
                else None
            ),
            "maximum_relative_pivot_interval_width": float(
                maximum_relative_pivot_width
            ),
            "maximum_relative_pivot_interval_width_decimal": str(
                maximum_relative_pivot_width
            ),
            "maximum_lower_factor_interval_width": float(
                maximum_lower_interval_width
            ),
            "maximum_lower_factor_interval_width_decimal": str(
                maximum_lower_interval_width
            ),
            "all_superlu_central_pivots_contained": (
                all_central_pivots_contained
            ),
            "all_superlu_central_lower_entries_contained": (
                all_central_lower_entries_contained
            ),
            "superlu_numeric_values_used_for_certification": False,
            "row_and_column_permutations_equal": permutations_equal,
            "factor_structures_are_transposes": (
                factor_structures_transpose
            ),
            "exact_pencil_pattern_contained_in_symbolic_factor": (
                exact_pattern_contained
            ),
            "symbolic_ldl_pattern_closed": symbolic_pattern_closed,
            "symbolic_ldl_pattern_failure": symbolic_pattern_failure,
            "symbolic_closure_pair_count": symbolic_closure_pair_count,
            "factor_nonzero_count": int(lower.nnz),
            "central_factorization_residual_inf": (
                central_residual_inf
            ),
            "permutation_sha256": _sha256_arrays(order),
            "pivot_intervals_sha256": _sha256_text_rows(
                pivot_lower_output,
                pivot_upper_output,
            ),
            "all_directed_ldl_checks_pass": bool(all(checks)),
            "elapsed_seconds": time.perf_counter() - started,
        },
        {
            "shift": np.asarray(shift),
            "permutation": order.astype(np.int64),
            "pivot_lower_decimal": np.asarray(
                [str(value) for value in pivot_lower_output]
            ),
            "pivot_upper_decimal": np.asarray(
                [str(value) for value in pivot_upper_output]
            ),
            "pivot_sign": signs,
        },
    )


def audit(
    assembly_checkpoint: Path,
    eigensystem_result: Path,
    maximum_pivots: int,
    ordering: str,
    decimal_precision: int,
    row_checkpoint: Path | None,
    row_checkpoint_cache: Path | None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    started = time.perf_counter()
    below_normal_priority_set = _set_below_normal_priority()
    mass, stiffness = _load_forms(assembly_checkpoint)
    mass_exactly_symmetric = _is_exactly_symmetric(mass)
    stiffness_exactly_symmetric = _is_exactly_symmetric(stiffness)
    with eigensystem_result.open("r", encoding="utf-8") as handle:
        eigensystem = json.load(handle)
    pair_rows = eigensystem["reference_generalized_eigensystem"][
        "pair_rows"
    ]
    if len(pair_rows) != 241:
        raise RuntimeError("the inertia audit requires 241 proximity rows")

    retained_upper = float(pair_rows[239]["proximity_interval_upper"])
    omitted_lower = float(pair_rows[240]["proximity_interval_lower"])
    omitted_upper = float(pair_rows[240]["proximity_interval_upper"])
    retained_gap_shift = 0.5 * (retained_upper + omitted_lower)
    post_241_shift = omitted_upper + 0.01
    shifts = (
        ("retained_gap", retained_gap_shift, 240),
        ("post_241_interval", post_241_shift, 241),
    )

    checkpoint_signature = {
        "assembly_checkpoint": str(assembly_checkpoint),
        "assembly_checkpoint_sha256": _sha256_file(assembly_checkpoint),
        "eigensystem_result": str(eigensystem_result),
        "eigensystem_result_sha256": _sha256_file(eigensystem_result),
        "state_count": int(mass.shape[0]),
        "maximum_pivots": maximum_pivots,
        "ordering": ordering,
        "decimal_precision": decimal_precision,
        "shifts": [
            {
                "name": name,
                "shift": shift,
                "expected_negative_pivot_count": expected_count,
            }
            for name, shift, expected_count in shifts
        ],
    }
    if (row_checkpoint is None) != (row_checkpoint_cache is None):
        raise ValueError(
            "row checkpoint JSON and cache paths must be supplied together"
        )

    rows: dict[str, dict[str, object]] = {}
    cache_arrays: dict[str, np.ndarray] = {
        "cache_version": np.asarray(2, dtype=np.int64),
    }
    if row_checkpoint is not None and row_checkpoint.exists():
        if not row_checkpoint_cache.exists():
            raise RuntimeError("row checkpoint cache is missing")
        with row_checkpoint.open("r", encoding="utf-8") as handle:
            saved_checkpoint = json.load(handle)
        if saved_checkpoint.get("signature") != checkpoint_signature:
            raise RuntimeError("row checkpoint signature does not match run")
        if (
            saved_checkpoint.get("cache_sha256")
            != _sha256_file(row_checkpoint_cache)
        ):
            raise RuntimeError("row checkpoint cache hash does not match")
        rows = saved_checkpoint["inertia_rows"]
        allowed_names = {name for name, _, _ in shifts}
        if not set(rows).issubset(allowed_names):
            raise RuntimeError("row checkpoint contains an unknown shift")
        with np.load(row_checkpoint_cache, allow_pickle=False) as checkpoint:
            cache_arrays = {
                name: checkpoint[name].copy()
                for name in checkpoint.files
            }
        if int(cache_arrays["cache_version"].item()) != 2:
            raise RuntimeError("row checkpoint cache version is unsupported")

    for name, shift, expected_count in shifts:
        if name in rows:
            continue
        if decimal_precision > 0:
            row, arrays = _directed_ldl_decimal(
                stiffness,
                mass,
                shift,
                maximum_pivots,
                ordering,
                decimal_precision,
            )
        else:
            row, arrays = _directed_ldl(
                stiffness,
                mass,
                shift,
                maximum_pivots,
                ordering,
            )
        row["expected_negative_pivot_count"] = expected_count
        row["expected_count_matches"] = bool(
            row["complete_inertia"]
            and row["negative_pivot_count"] == expected_count
        )
        rows[name] = row
        for array_name, array in arrays.items():
            cache_arrays[f"{name}_{array_name}"] = array
        if row_checkpoint is not None:
            _atomic_write_npz(row_checkpoint_cache, cache_arrays)
            _atomic_write_json(
                row_checkpoint,
                {
                    "checkpoint_version": 1,
                    "signature": checkpoint_signature,
                    "completed_shift_names": list(rows),
                    "inertia_rows": rows,
                    "cache_sha256": _sha256_file(row_checkpoint_cache),
                },
            )
        print(
            f"completed and checkpointed inertia row {name}",
            flush=True,
        )

    full_run_requested = (
        maximum_pivots <= 0 or maximum_pivots >= mass.shape[0]
    )
    complete = bool(
        full_run_requested
        and all(row["complete_inertia"] for row in rows.values())
    )
    retained_intervals_below_gap = all(
        float(pair_rows[index]["proximity_interval_upper"])
        < retained_gap_shift
        for index in range(240)
    )
    omitted_interval_above_gap = (
        float(pair_rows[240]["proximity_interval_lower"])
        > retained_gap_shift
    )
    all_intervals_below_post_shift = all(
        float(row["proximity_interval_upper"]) < post_241_shift
        for row in pair_rows
    )
    mass_coercivity_proved = bool(
        eigensystem["reference_generalized_eigensystem"][
            "mass_coercivity"
        ]["stored_mass_row_lumped_coercivity_proved"]
    )
    all_241_distinct = bool(
        eigensystem["distinct_reference_eigenvalue_proximity_intervals_proved"]
    )
    indexed_first_240 = bool(
        complete
        and rows["retained_gap"]["all_directed_ldl_checks_pass"]
        and rows["retained_gap"]["expected_count_matches"]
        and retained_intervals_below_gap
        and omitted_interval_above_gap
        and mass_coercivity_proved
        and all_241_distinct
    )
    indexed_all_241 = bool(
        indexed_first_240
        and rows["post_241_interval"]["all_directed_ldl_checks_pass"]
        and rows["post_241_interval"]["expected_count_matches"]
        and all_intervals_below_post_shift
    )
    checks = [
        complete,
        mass_exactly_symmetric,
        stiffness_exactly_symmetric,
        mass_coercivity_proved,
        all_241_distinct,
        retained_intervals_below_gap,
        omitted_interval_above_gap,
        all_intervals_below_post_shift,
    ]
    if complete:
        checks.extend(
            [
                rows["retained_gap"]["all_directed_ldl_checks_pass"],
                rows["retained_gap"]["expected_count_matches"],
                rows["post_241_interval"][
                    "all_directed_ldl_checks_pass"
                ],
                rows["post_241_interval"]["expected_count_matches"],
                indexed_first_240,
                indexed_all_241,
            ]
        )
    return (
        {
            "model": (
                "directed sparse LDL inertia of the stored q12 "
                "generalized pencil"
            ),
            "assembly_checkpoint": str(assembly_checkpoint),
            "eigensystem_result": str(eigensystem_result),
            "state_count": int(mass.shape[0]),
            "proximity_interval_count": len(pair_rows),
            "maximum_pivots": maximum_pivots,
            "ordering": ordering,
            "arithmetic": (
                "directed Decimal interval recurrence"
                if decimal_precision > 0
                else "directed binary64 interval recurrence"
            ),
            "decimal_precision": (
                decimal_precision if decimal_precision > 0 else None
            ),
            "complete_production_audit": complete,
            "below_normal_priority_set": below_normal_priority_set,
            "stored_mass_exactly_symmetric": mass_exactly_symmetric,
            "stored_stiffness_exactly_symmetric": (
                stiffness_exactly_symmetric
            ),
            "retained_interval_upper": retained_upper,
            "omitted_interval_lower": omitted_lower,
            "omitted_interval_upper": omitted_upper,
            "retained_cutoff_gap": omitted_lower - retained_upper,
            "retained_gap_shift": retained_gap_shift,
            "post_241_interval_shift": post_241_shift,
            "mass_coercivity_proved": mass_coercivity_proved,
            "distinct_eigenvalue_existence_in_all_241_intervals_proved": (
                all_241_distinct
            ),
            "retained_240_intervals_below_gap_shift": (
                retained_intervals_below_gap
            ),
            "omitted_interval_above_gap_shift": (
                omitted_interval_above_gap
            ),
            "all_241_intervals_below_post_interval_shift": (
                all_intervals_below_post_shift
            ),
            "inertia_rows": rows,
            "first_240_stored_generalized_eigenvalues_indexed": (
                indexed_first_240
            ),
            "all_241_stored_generalized_eigenvalue_intervals_indexed": (
                indexed_all_241
            ),
            "stored_complement_generalized_eigenvalue_lower_bound": (
                retained_gap_shift if indexed_first_240 else None
            ),
            "all_sparse_inertia_audit_checks_pass": bool(all(checks)),
            "elapsed_seconds": time.perf_counter() - started,
        },
        cache_arrays,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assembly-checkpoint",
        type=Path,
        default=ASSEMBLY_CHECKPOINT,
    )
    parser.add_argument(
        "--eigensystem-result",
        type=Path,
        default=EIGENSYSTEM_RESULT,
    )
    parser.add_argument(
        "--maximum-pivots",
        type=int,
        default=0,
        help="positive values run a non-certifying prefix pilot",
    )
    parser.add_argument(
        "--ordering",
        choices=("MMD_AT_PLUS_A", "MMD_ATA", "COLAMD", "NATURAL"),
        default="MMD_AT_PLUS_A",
    )
    parser.add_argument(
        "--decimal-precision",
        type=int,
        default=220,
        help=(
            "Decimal digits for directed interval recurrence; "
            "use 0 for the binary64 diagnostic backend"
        ),
    )
    parser.add_argument("--pivot-cache", type=Path, default=PIVOT_CACHE)
    parser.add_argument(
        "--row-checkpoint",
        type=Path,
        default=ROW_CHECKPOINT,
    )
    parser.add_argument(
        "--row-checkpoint-cache",
        type=Path,
        default=ROW_CHECKPOINT_CACHE,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result, cache_arrays = audit(
        args.assembly_checkpoint,
        args.eigensystem_result,
        args.maximum_pivots,
        args.ordering,
        args.decimal_precision,
        args.row_checkpoint if args.output is not None else None,
        args.row_checkpoint_cache if args.output is not None else None,
    )
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _atomic_write_npz(args.pivot_cache, cache_arrays)
        _atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
