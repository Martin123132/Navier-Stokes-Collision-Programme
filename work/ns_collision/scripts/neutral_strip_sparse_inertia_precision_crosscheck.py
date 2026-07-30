"""Cross-check every sparse-inertia pivot at two Decimal precisions."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

import numpy as np


PRIMARY_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_sparse_inertia_audit_v1.json"
)
PRIMARY_PIVOTS = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_sparse_inertia_pivots_v1.npz"
)
CROSSCHECK_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_sparse_inertia_p260_crosscheck_v1.json"
)
CROSSCHECK_PIVOTS = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_sparse_inertia_p260_crosscheck_pivots_v1.npz"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def crosscheck(
    primary_result_path: Path,
    primary_pivots_path: Path,
    crosscheck_result_path: Path,
    crosscheck_pivots_path: Path,
) -> dict[str, object]:
    with primary_result_path.open("r", encoding="utf-8") as handle:
        primary_result = json.load(handle)
    with crosscheck_result_path.open("r", encoding="utf-8") as handle:
        crosscheck_result = json.load(handle)

    primary_precision = int(primary_result["decimal_precision"])
    crosscheck_precision = int(crosscheck_result["decimal_precision"])
    rows: dict[str, dict[str, object]] = {}

    with (
        np.load(primary_pivots_path, allow_pickle=False) as primary,
        np.load(crosscheck_pivots_path, allow_pickle=False) as crosscheck_cache,
    ):
        for name in ("retained_gap", "post_241_interval"):
            primary_lower = primary[f"{name}_pivot_lower_decimal"]
            primary_upper = primary[f"{name}_pivot_upper_decimal"]
            crosscheck_lower = crosscheck_cache[
                f"{name}_pivot_lower_decimal"
            ]
            crosscheck_upper = crosscheck_cache[
                f"{name}_pivot_upper_decimal"
            ]
            primary_sign = primary[f"{name}_pivot_sign"]
            crosscheck_sign = crosscheck_cache[f"{name}_pivot_sign"]
            primary_permutation = primary[f"{name}_permutation"]
            crosscheck_permutation = crosscheck_cache[
                f"{name}_permutation"
            ]

            lengths_equal = bool(
                len(primary_lower)
                == len(primary_upper)
                == len(primary_sign)
                == len(crosscheck_lower)
                == len(crosscheck_upper)
                == len(crosscheck_sign)
                == 15211
            )
            nested = True
            first_nesting_failure = None
            all_primary_intervals_exclude_zero = True
            all_crosscheck_intervals_exclude_zero = True
            for index, (first_lo, first_hi, second_lo, second_hi) in (
                enumerate(
                    zip(
                        primary_lower,
                        primary_upper,
                        crosscheck_lower,
                        crosscheck_upper,
                    )
                )
            ):
                first_interval = (
                    Decimal(str(first_lo)),
                    Decimal(str(first_hi)),
                )
                second_interval = (
                    Decimal(str(second_lo)),
                    Decimal(str(second_hi)),
                )
                all_primary_intervals_exclude_zero &= not (
                    first_interval[0] <= 0 <= first_interval[1]
                )
                all_crosscheck_intervals_exclude_zero &= not (
                    second_interval[0] <= 0 <= second_interval[1]
                )
                if not (
                    first_interval[0]
                    <= second_interval[0]
                    <= second_interval[1]
                    <= first_interval[1]
                ):
                    nested = False
                    first_nesting_failure = index
                    break

            primary_row = primary_result["inertia_rows"][name]
            crosscheck_row = crosscheck_result["inertia_rows"][name]
            signs_equal = bool(
                np.array_equal(primary_sign, crosscheck_sign)
            )
            permutations_equal = bool(
                np.array_equal(
                    primary_permutation,
                    crosscheck_permutation,
                )
            )
            shifts_equal = bool(
                np.array_equal(
                    primary[f"{name}_shift"],
                    crosscheck_cache[f"{name}_shift"],
                )
            )
            negative_count = int(np.count_nonzero(crosscheck_sign < 0))
            positive_count = int(np.count_nonzero(crosscheck_sign > 0))
            counts_match_results = bool(
                negative_count
                == primary_row["negative_pivot_count"]
                == crosscheck_row["negative_pivot_count"]
                and positive_count
                == primary_row["positive_pivot_count"]
                == crosscheck_row["positive_pivot_count"]
            )
            row_checks = [
                lengths_equal,
                nested,
                all_primary_intervals_exclude_zero,
                all_crosscheck_intervals_exclude_zero,
                signs_equal,
                permutations_equal,
                shifts_equal,
                counts_match_results,
                primary_row["complete_inertia"],
                crosscheck_row["complete_inertia"],
                primary_row["all_directed_ldl_checks_pass"],
                crosscheck_row["all_directed_ldl_checks_pass"],
            ]
            rows[name] = {
                "pivot_count": len(primary_sign),
                "negative_pivot_count": negative_count,
                "positive_pivot_count": positive_count,
                "lengths_equal": lengths_equal,
                "all_crosscheck_intervals_nested_in_primary": nested,
                "first_nesting_failure": first_nesting_failure,
                "all_primary_intervals_exclude_zero": (
                    all_primary_intervals_exclude_zero
                ),
                "all_crosscheck_intervals_exclude_zero": (
                    all_crosscheck_intervals_exclude_zero
                ),
                "all_pivot_signs_equal": signs_equal,
                "permutations_equal": permutations_equal,
                "shifts_bitwise_equal": shifts_equal,
                "counts_match_result_json": counts_match_results,
                "primary_maximum_relative_pivot_interval_width": (
                    primary_row[
                        "maximum_relative_pivot_interval_width_decimal"
                    ]
                ),
                "crosscheck_maximum_relative_pivot_interval_width": (
                    crosscheck_row[
                        "maximum_relative_pivot_interval_width_decimal"
                    ]
                ),
                "all_row_crosscheck_checks_pass": bool(all(row_checks)),
            }

    top_checks = [
        primary_result["all_sparse_inertia_audit_checks_pass"],
        crosscheck_result["all_sparse_inertia_audit_checks_pass"],
        primary_result[
            "all_241_stored_generalized_eigenvalue_intervals_indexed"
        ],
        crosscheck_result[
            "all_241_stored_generalized_eigenvalue_intervals_indexed"
        ],
        primary_precision < crosscheck_precision,
        all(
            row["all_row_crosscheck_checks_pass"]
            for row in rows.values()
        ),
    ]
    return {
        "model": (
            "independent-precision comparison of every directed sparse "
            "LDL pivot interval"
        ),
        "primary_result": str(primary_result_path),
        "primary_result_sha256": _sha256_file(primary_result_path),
        "primary_pivots": str(primary_pivots_path),
        "primary_pivots_sha256": _sha256_file(primary_pivots_path),
        "primary_decimal_precision": primary_precision,
        "crosscheck_result": str(crosscheck_result_path),
        "crosscheck_result_sha256": _sha256_file(
            crosscheck_result_path
        ),
        "crosscheck_pivots": str(crosscheck_pivots_path),
        "crosscheck_pivots_sha256": _sha256_file(
            crosscheck_pivots_path
        ),
        "crosscheck_decimal_precision": crosscheck_precision,
        "rows": rows,
        "all_30422_pivot_signs_reproduced": bool(
            all(row["all_pivot_signs_equal"] for row in rows.values())
        ),
        "all_30422_crosscheck_intervals_nested": bool(
            all(
                row["all_crosscheck_intervals_nested_in_primary"]
                for row in rows.values()
            )
        ),
        "all_sparse_inertia_precision_crosscheck_checks_pass": bool(
            all(top_checks)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-result",
        type=Path,
        default=PRIMARY_RESULT,
    )
    parser.add_argument(
        "--primary-pivots",
        type=Path,
        default=PRIMARY_PIVOTS,
    )
    parser.add_argument(
        "--crosscheck-result",
        type=Path,
        default=CROSSCHECK_RESULT,
    )
    parser.add_argument(
        "--crosscheck-pivots",
        type=Path,
        default=CROSSCHECK_PIVOTS,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = crosscheck(
        args.primary_result,
        args.primary_pivots,
        args.crosscheck_result,
        args.crosscheck_pivots,
    )
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
