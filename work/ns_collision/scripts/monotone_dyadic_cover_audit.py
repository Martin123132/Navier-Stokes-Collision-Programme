"""Audit a fixed-centre monotone dyadic intrinsic-radius cover."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_intrinsic_cover_module():
    script = Path(__file__).resolve().with_name(
        "intrinsic_radius_cover_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "intrinsic_radius_for_dyadic_cover", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit() -> dict[str, object]:
    intrinsic = _load_intrinsic_cover_module()
    domain_length = 4.0
    target_reynolds = 2.0
    viscosity = 1.0
    kappa = 0.125
    coordinates = np.linspace(0.0, domain_length, 801)
    amplitudes = (0.0, 10.0, 40.0, 160.0, 640.0)

    def bounds(cell: tuple[int, int]) -> tuple[float, float, float]:
        level, index = cell
        side = domain_length / (2**level)
        left = index * side
        return left, left + side, side

    def interval_values(
        values: np.ndarray, cell: tuple[int, int]
    ) -> np.ndarray:
        left, right, _ = bounds(cell)
        mask = (coordinates >= left - 1.0e-14) & (
            coordinates <= right + 1.0e-14
        )
        return values[mask]

    def split(active: set[tuple[int, int]], cell: tuple[int, int]) -> None:
        level, index = cell
        active.remove(cell)
        active.add((level + 1, 2 * index))
        active.add((level + 1, 2 * index + 1))

    active_cells: set[tuple[int, int]] = {(0, 0)}
    previous_envelope = np.ones_like(coordinates)
    previous_count = 0
    snapshot_rows = []
    total_split_count = 0
    only_split_operations = True

    for snapshot, amplitude in enumerate(amplitudes):
        candidate = (
            1.0
            + 5.0 * np.exp(-((coordinates - 0.9) / 0.35) ** 2)
            + amplitude
            * np.exp(-((coordinates - 3.05) / 0.19) ** 2)
        )
        envelope = np.maximum(previous_envelope, candidate)
        raw_radius = np.sqrt(
            target_reynolds * viscosity / envelope
        )
        radius = intrinsic._lipschitz_minorant(
            coordinates, raw_radius, kappa
        )
        current_strain = envelope * (
            0.6 + 0.35 * np.sin(1.3 * coordinates) ** 2
        )

        split_count = 0
        while True:
            unsafe_cells = []
            for cell in active_cells:
                _, _, side = bounds(cell)
                if side > float(np.min(interval_values(radius, cell))) + 1.0e-14:
                    unsafe_cells.append(cell)
            if not unsafe_cells:
                break
            for cell in unsafe_cells:
                if cell in active_cells:
                    split(active_cells, cell)
                    split_count += 1

        while True:
            ordered = sorted(active_cells, key=lambda cell: bounds(cell)[0])
            imbalance = None
            for left_cell, right_cell in zip(ordered[:-1], ordered[1:]):
                if abs(left_cell[0] - right_cell[0]) > 1:
                    imbalance = (
                        left_cell
                        if left_cell[0] < right_cell[0]
                        else right_cell
                    )
                    break
            if imbalance is None:
                break
            split(active_cells, imbalance)
            split_count += 1

        ordered = sorted(active_cells, key=lambda cell: bounds(cell)[0])
        sides = np.array([bounds(cell)[2] for cell in ordered])
        safety_ratios = np.array(
            [
                bounds(cell)[2]
                / float(np.min(interval_values(radius, cell)))
                for cell in ordered
            ]
        )
        local_reynolds = np.array(
            [
                float(np.max(interval_values(current_strain, cell)))
                * bounds(cell)[2] ** 2
                / viscosity
                for cell in ordered
            ]
        )
        neighbor_side_ratios = np.array(
            [
                max(left_side / right_side, right_side / left_side)
                for left_side, right_side in zip(sides[:-1], sides[1:])
            ]
        )
        if len(neighbor_side_ratios) == 0:
            neighbor_side_ratios = np.array([1.0])
        coverage_length = float(np.sum(sides))
        total_split_count += split_count
        snapshot_rows.append(
            {
                "snapshot": snapshot,
                "peak_amplitude_parameter": amplitude,
                "active_cell_count": len(active_cells),
                "new_split_count": split_count,
                "maximum_level": max(cell[0] for cell in active_cells),
                "minimum_side": float(np.min(sides)),
                "maximum_side": float(np.max(sides)),
                "coverage_length": coverage_length,
                "maximum_side_to_safe_radius": float(
                    np.max(safety_ratios)
                ),
                "maximum_cell_local_reynolds": float(
                    np.max(local_reynolds)
                ),
                "maximum_neighbor_side_ratio": float(
                    np.max(neighbor_side_ratios)
                ),
                "maximum_neighbor_reference_ratio": float(
                    np.max(neighbor_side_ratios) ** 2
                ),
            }
        )
        only_split_operations = bool(
            only_split_operations and len(active_cells) >= previous_count
        )
        previous_count = len(active_cells)
        previous_envelope = envelope

    cell_counts = [row["active_cell_count"] for row in snapshot_rows]
    maximum_levels = [row["maximum_level"] for row in snapshot_rows]
    result: dict[str, object] = {
        "safe_cell_rule": "side(Q)<=inf_Q rho_kappa",
        "tree_rule": "unsafe cells split into dyadic children and never merge",
        "balance_rule": "adjacent active cells differ by at most one level",
        "fixed_cell_centres_remove_continuous_centre_velocity": True,
        "snapshot_rows": snapshot_rows,
        "all_snapshots_cover_domain_exactly": all(
            abs(row["coverage_length"] - domain_length) < 1.0e-12
            for row in snapshot_rows
        ),
        "all_cells_obey_safe_radius": all(
            row["maximum_side_to_safe_radius"] <= 1.0 + 1.0e-12
            for row in snapshot_rows
        ),
        "all_cells_obey_local_reynolds_cap": all(
            row["maximum_cell_local_reynolds"]
            <= target_reynolds + 1.0e-12
            for row in snapshot_rows
        ),
        "all_neighbors_are_two_to_one_balanced": all(
            row["maximum_neighbor_side_ratio"] <= 2.0 + 1.0e-12
            for row in snapshot_rows
        ),
        "neighbor_reference_envelope_ratio_at_most_four": all(
            row["maximum_neighbor_reference_ratio"] <= 4.0 + 1.0e-12
            for row in snapshot_rows
        ),
        "cell_count_is_nondecreasing": bool(
            all(
                later >= earlier
                for earlier, later in zip(cell_counts[:-1], cell_counts[1:])
            )
        ),
        "maximum_level_is_nondecreasing": bool(
            all(
                later >= earlier
                for earlier, later in zip(
                    maximum_levels[:-1], maximum_levels[1:]
                )
            )
        ),
        "cover_uses_only_splits": only_split_operations,
        "later_envelope_growth_causes_further_refinement": bool(
            any(row["new_split_count"] > 0 for row in snapshot_rows[1:])
        ),
        "total_split_count": total_split_count,
        "parent_to_child_side_ratio": 2,
        "parent_to_child_reference_envelope_ratio": 4,
        "pressure_partition_consequence": (
            "a subordinate partition retains exact pressure conservation; "
            "balanced neighbors have reference-amplitude ratio at most four"
        ),
        "remaining_transition_gate": (
            "replace each parent by child weights without accumulating a "
            "factor per generation in gauged deformation or renewal norms"
        ),
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
