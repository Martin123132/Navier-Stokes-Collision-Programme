"""Audit pressure-shell conservation across a smooth partition of unity."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np


def _load_pairing_module():
    script = Path(__file__).resolve().with_name(
        "pressure_frame_pairing_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pressure_frame_pairing_for_partition", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit() -> dict[str, object]:
    pairing = _load_pairing_module()
    pairing_result = pairing.audit()
    fields = pairing._build_spectral_fields()
    center = np.array(pairing_result["optimized_point_mod_2pi"])
    grid_size = pairing.GRID_SIZE
    coordinates = (
        2
        * np.pi
        * np.indices((grid_size, grid_size, grid_size))
        / grid_size
    )
    displacement = coordinates - center[:, None, None, None]

    plus_weights = (1 + np.cos(displacement)) / 2
    minus_weights = (1 - np.cos(displacement)) / 2
    plus_derivatives = -0.5 * np.sin(displacement)
    minus_derivatives = 0.5 * np.sin(displacement)

    cells = []
    for bits in itertools.product((0, 1), repeat=3):
        factors = [
            plus_weights[direction]
            if bit == 0
            else minus_weights[direction]
            for direction, bit in enumerate(bits)
        ]
        derivatives = [
            plus_derivatives[direction]
            if bit == 0
            else minus_derivatives[direction]
            for direction, bit in enumerate(bits)
        ]
        localization = np.prod(factors, axis=0)
        gradient = np.empty((3, grid_size, grid_size, grid_size))
        for direction in range(3):
            other_directions = [
                index for index in range(3) if index != direction
            ]
            gradient[direction] = derivatives[direction] * np.prod(
                [factors[index] for index in other_directions], axis=0
            )
        cell = {
            "bits": bits,
            "label": "".join(str(bit) for bit in bits),
            "localization": localization,
            "gradient": gradient,
        }
        cells.append(cell)

    partition_sum = np.sum(
        [cell["localization"] for cell in cells], axis=0
    )
    partition_gradient_sum = np.sum(
        [cell["gradient"] for cell in cells], axis=0
    )
    velocity_gradient = fields["velocity_gradient_grid"]
    strain = fields["strain_grid"]
    cell_coefficients = np.array(
        [1.0, -2.0, 0.5, 3.0, -1.5, 0.25, 2.5, -0.75]
    )
    weighted_localization = np.sum(
        [
            coefficient * cell["localization"]
            for coefficient, cell in zip(cell_coefficients, cells)
        ],
        axis=0,
    )
    weighted_gradient = np.sum(
        [
            coefficient * cell["gradient"]
            for coefficient, cell in zip(cell_coefficients, cells)
        ],
        axis=0,
    )

    split_results = []
    cell_piece_works: dict[str, list[float]] = {}
    for label, pressure_name, potential_gradient_name in (
        ("full", "pressure_grid", "pressure_potential_gradient_grid"),
        (
            "low",
            "pressure_low_grid",
            "pressure_potential_low_gradient_grid",
        ),
        (
            "collision_defect",
            "pressure_defect_grid",
            "pressure_potential_defect_gradient_grid",
        ),
    ):
        pressure = fields[pressure_name]
        potential_gradient = fields[potential_gradient_name]
        pressure_strain = np.einsum(
            "ijxyz,ijxyz->xyz", pressure, strain
        )
        works = []
        boundaries = []
        for cell in cells:
            transposed_gradient_times_cutoff = np.einsum(
                "ijxyz,ixyz->jxyz",
                velocity_gradient,
                cell["gradient"],
            )
            works.append(
                float(np.mean(cell["localization"] * pressure_strain))
            )
            boundaries.append(
                float(
                    -np.mean(
                        np.einsum(
                            "jxyz,jxyz->xyz",
                            potential_gradient,
                            transposed_gradient_times_cutoff,
                        )
                    )
                )
            )

        weighted_velocity_gradient = np.einsum(
            "ijxyz,ixyz->jxyz", velocity_gradient, weighted_gradient
        )
        weighted_work = float(
            np.sum(cell_coefficients * np.array(works))
        )
        weighted_boundary = float(
            np.sum(cell_coefficients * np.array(boundaries))
        )
        direct_weighted_work = float(
            np.mean(weighted_localization * pressure_strain)
        )
        direct_weighted_boundary = float(
            -np.mean(
                np.einsum(
                    "jxyz,jxyz->xyz",
                    potential_gradient,
                    weighted_velocity_gradient,
                )
            )
        )
        shifted_weighted_work = float(
            np.sum((cell_coefficients + 7.0) * np.array(works))
        )
        coefficient_by_bits = {
            cell["bits"]: coefficient
            for cell, coefficient in zip(cells, cell_coefficients)
        }
        edge_rows = []
        edge_weighted_boundary = 0.0
        for direction in range(3):
            other_directions = [
                index for index in range(3) if index != direction
            ]
            for other_bits in itertools.product((0, 1), repeat=2):
                minus_bits = [0, 0, 0]
                plus_bits = [0, 0, 0]
                minus_bits[direction] = 0
                plus_bits[direction] = 1
                for index, other_direction in enumerate(other_directions):
                    minus_bits[other_direction] = other_bits[index]
                    plus_bits[other_direction] = other_bits[index]
                minus_bits_tuple = tuple(minus_bits)
                plus_bits_tuple = tuple(plus_bits)
                other_factor = np.prod(
                    [
                        plus_weights[other_direction]
                        if other_bits[index] == 0
                        else minus_weights[other_direction]
                        for index, other_direction in enumerate(
                            other_directions
                        )
                    ],
                    axis=0,
                )
                edge_derivative = (
                    plus_derivatives[direction] * other_factor
                )
                edge_velocity_gradient = (
                    velocity_gradient[direction] * edge_derivative
                )
                edge_flux = float(
                    -np.mean(
                        np.einsum(
                            "jxyz,jxyz->xyz",
                            potential_gradient,
                            edge_velocity_gradient,
                        )
                    )
                )
                coefficient_difference = float(
                    coefficient_by_bits[minus_bits_tuple]
                    - coefficient_by_bits[plus_bits_tuple]
                )
                edge_weighted_boundary += coefficient_difference * edge_flux
                edge_rows.append(
                    {
                        "direction": direction,
                        "minus_cell": "".join(
                            str(bit) for bit in minus_bits_tuple
                        ),
                        "plus_cell": "".join(
                            str(bit) for bit in plus_bits_tuple
                        ),
                        "coefficient_difference": coefficient_difference,
                        "edge_flux": edge_flux,
                    }
                )
        cell_piece_works[label] = works
        split_results.append(
            {
                "piece": label,
                "cell_works": works,
                "cell_boundaries": boundaries,
                "maximum_cell_identity_residual": max(
                    abs(work - boundary)
                    for work, boundary in zip(works, boundaries)
                ),
                "partition_total_work": float(np.sum(works)),
                "partition_total_boundary": float(np.sum(boundaries)),
                "maximum_absolute_cell_work": float(
                    np.max(np.abs(works))
                ),
                "weighted_work": weighted_work,
                "weighted_boundary": weighted_boundary,
                "direct_weighted_work": direct_weighted_work,
                "direct_weighted_boundary": direct_weighted_boundary,
                "weighted_representation_residual": max(
                    abs(weighted_work - direct_weighted_work),
                    abs(weighted_boundary - direct_weighted_boundary),
                    abs(weighted_work - weighted_boundary),
                ),
                "edge_rows": edge_rows,
                "edge_weighted_boundary": edge_weighted_boundary,
                "edge_representation_residual": abs(
                    edge_weighted_boundary - weighted_boundary
                ),
                "constant_coefficient_shift_residual": (
                    shifted_weighted_work - weighted_work
                ),
            }
        )

    cell_heat_split_residuals = [
        full - low - defect
        for full, low, defect in zip(
            cell_piece_works["full"],
            cell_piece_works["low"],
            cell_piece_works["collision_defect"],
        )
    ]
    result: dict[str, object] = {
        "partition": (
            "eight tensor-product cells from (1+cos)/2 and (1-cos)/2"
        ),
        "maximum_partition_sum_error": float(
            np.max(np.abs(partition_sum - 1.0))
        ),
        "maximum_partition_gradient_sum": float(
            np.max(np.abs(partition_gradient_sum))
        ),
        "partition_of_unity_verified": bool(
            np.max(np.abs(partition_sum - 1.0)) < 1.0e-14
            and np.max(np.abs(partition_gradient_sum)) < 1.0e-14
        ),
        "split_results": split_results,
        "all_cell_boundary_identities_verified": all(
            row["maximum_cell_identity_residual"] < 1.0e-10
            for row in split_results
        ),
        "all_partition_pressure_fluxes_cancel": all(
            abs(row["partition_total_work"]) < 1.0e-10
            and abs(row["partition_total_boundary"]) < 1.0e-10
            for row in split_results
        ),
        "individual_cells_have_nonzero_pressure_transfer": all(
            row["maximum_absolute_cell_work"] > 0.1
            for row in split_results
        ),
        "all_weighted_representations_verified": all(
            row["weighted_representation_residual"] < 1.0e-10
            for row in split_results
        ),
        "all_neighbor_edge_representations_verified": all(
            len(row["edge_rows"]) == 12
            and row["edge_representation_residual"] < 1.0e-10
            for row in split_results
        ),
        "constant_weight_shifts_are_invisible": all(
            abs(row["constant_coefficient_shift_residual"]) < 1.0e-10
            for row in split_results
        ),
        "cellwise_heat_split_recombines": bool(
            max(abs(value) for value in cell_heat_split_residuals) < 1.0e-10
        ),
        "pressure_interpretation": (
            "pressure shell terms are conservative transfers between cells"
        ),
        "weighted_partition_gate": (
            "only coefficient differences survive; control neighboring "
            "envelope/gauge weight mismatch instead of absolute pressure tails"
        ),
        "edge_flux_identity": (
            "weighted pressure=sum_edges (w_minus-w_plus)*edge_flux"
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
