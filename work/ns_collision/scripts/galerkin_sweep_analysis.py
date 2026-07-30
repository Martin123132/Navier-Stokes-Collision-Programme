"""Deduplicate and classify the collision-defect Galerkin sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


KEY_FIELDS = (
    "maximum_mode",
    "reynolds",
    "viscosity",
    "heat_scale",
    "sign",
    "final_time",
    "samples",
)


def _key(record: dict[str, object]) -> tuple[object, ...]:
    return tuple(record[field] for field in KEY_FIELDS)


def analyze(input_path: Path) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    duplicate_rows_are_identical = True
    for record in records:
        key = _key(record)
        if key in unique and record != unique[key]:
            duplicate_rows_are_identical = False
        unique.setdefault(key, record)

    rows = list(unique.values())
    grouped: dict[tuple[int, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["sign"]), float(row["reynolds"])), []).append(
            row
        )

    positive_table = []
    for (sign, reynolds), candidates in sorted(grouped.items()):
        if sign != 1 or reynolds not in (0.25, 0.5, 1.0, 2.0, 4.0):
            continue
        ordered = sorted(candidates, key=lambda row: int(row["maximum_mode"]))
        highest = ordered[-1]
        previous = ordered[-2] if len(ordered) > 1 else None
        convergence = None
        if previous is not None:
            convergence = abs(
                float(highest["integrated_defect"])
                - float(previous["integrated_defect"])
            ) / max(abs(float(highest["integrated_defect"])), 1.0e-30)
        boundary = float(highest["maximum_boundary_energy_fraction"])
        resolved = bool(
            convergence is not None and convergence < 0.03 and boundary < 1.0e-3
        )
        marginal = bool(
            not resolved
            and convergence is not None
            and convergence < 0.05
            and boundary < 2.0e-3
        )
        positive_table.append(
            {
                "reynolds": reynolds,
                "maximum_mode": int(highest["maximum_mode"]),
                "integrated_defect": float(highest["integrated_defect"]),
                "defect_to_viscous_palinstrophy_ratio": float(
                    highest["defect_to_viscous_palinstrophy_ratio"]
                ),
                "boundary_energy_fraction": boundary,
                "relative_change_from_previous_mode": convergence,
                "defect_sign_changes": int(highest["defect_sign_changes"]),
                "resolved": resolved,
                "marginal": marginal,
            }
        )

    negative_mode_three = sorted(
        (
            row
            for row in rows
            if int(row["sign"]) == -1
            and int(row["maximum_mode"]) == 3
            and float(row["final_time"]) == 2.0
        ),
        key=lambda row: float(row["reynolds"]),
    )
    transition_bracket = None
    for left, right in zip(negative_mode_three, negative_mode_three[1:]):
        if (
            float(left["integrated_defect"]) <= 0
            and float(right["integrated_defect"]) > 0
        ):
            transition_bracket = {
                "lower_reynolds": float(left["reynolds"]),
                "lower_integrated_defect": float(left["integrated_defect"]),
                "lower_endpoint_primitive": float(left["final_primitive"])
                - float(left["initial_primitive"]),
                "lower_integrated_transfer_from_balance": float(
                    left["final_primitive"]
                )
                - float(left["initial_primitive"])
                + float(left["viscosity"]) * float(left["integrated_defect"]),
                "upper_reynolds": float(right["reynolds"]),
                "upper_integrated_defect": float(right["integrated_defect"]),
                "upper_endpoint_primitive": float(right["final_primitive"])
                - float(right["initial_primitive"]),
                "upper_integrated_transfer_from_balance": float(
                    right["final_primitive"]
                )
                - float(right["initial_primitive"])
                + float(right["viscosity"]) * float(right["integrated_defect"]),
            }

    heat_scale = 0.5
    x = math.exp(-heat_scale)
    p = x**3 + 2 * x**2 + 3 * x
    weak_ratio_coefficient = (1 - x) ** 2 * (29 * p + 61) / 960
    for row in positive_table:
        reynolds = float(row["reynolds"])
        row["ratio_over_weak_prediction"] = float(
            row["defect_to_viscous_palinstrophy_ratio"]
        ) / (weak_ratio_coefficient * reynolds**2)

    maximum_energy_residual = max(
        float(row["maximum_relative_energy_balance_residual"]) for row in rows
    )
    maximum_identity_residual = max(
        float(row["maximum_identity_residual"]) for row in rows
    )
    maximum_primitive_balance_residual = max(
        abs(float(row["primitive_balance_residual"])) for row in rows
    )
    result: dict[str, object] = {
        "raw_record_count": len(records),
        "unique_record_count": len(rows),
        "duplicate_record_count": len(records) - len(rows),
        "duplicate_rows_are_identical": duplicate_rows_are_identical,
        "positive_channel_table": positive_table,
        "negative_channel_transition_bracket": transition_bracket,
        "weak_ratio_coefficient": weak_ratio_coefficient,
        "maximum_relative_energy_balance_residual": maximum_energy_residual,
        "maximum_differential_identity_residual": maximum_identity_residual,
        "maximum_primitive_balance_residual": maximum_primitive_balance_residual,
        "validation_checks_pass": bool(
            duplicate_rows_are_identical
            and maximum_energy_residual < 1.0e-10
            and maximum_identity_residual < 1.0e-10
            and maximum_primitive_balance_residual < 1.0e-5
        ),
        "resolved_positive_reynolds": [
            row["reynolds"] for row in positive_table if row["resolved"]
        ],
        "marginal_positive_reynolds": [
            row["reynolds"]
            for row in positive_table
            if row["marginal"]
        ],
        "all_sampled_positive_channel_integrals_are_positive": all(
            float(row["integrated_defect"]) > 0 for row in positive_table
        ),
        "negative_channel_loses_cumulative_sign": bool(
            transition_bracket is not None
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.input)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
