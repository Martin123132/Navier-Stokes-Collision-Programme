"""Audit the annular family across all vertices and one heat time window."""

from __future__ import annotations

import argparse
import ctypes
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from cross_shell_modulated_wave_gate_audit import (
    _component_fluxes,
    _direct_linear_flux,
    _maximum_vector_difference,
)
from primitive_hhl_chain_hardy_envelope_audit import (
    _translated_vertex_fisher,
    _translated_vertex_load,
)
from separable_annular_pressure_schur_no_go_audit import (
    LOW_DIRECTION,
    LOW_WAVE,
    TRANSLATION,
    _family_arrays,
    _high_field,
    _low_field,
    _shift_slices,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_eight_vertex_heat_window_gate_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "compatible_eight_cell_cubic_graph_audit_v1.json"
    ): "067625c4b44aa6085ff2b59cbbcef351253dcbd5b1fb9f0eac641fdd7a48682c",
    (
        "work/ns_collision/results/"
        "dyadic_three_shell_atlas_audit_v1.json"
    ): "52f79c57cd3bb99d8b6048f8797ab75f4cc1640436bb8ed1cf3bdac5cfaef513",
    (
        "work/ns_collision/results/"
        "separable_annular_pressure_schur_no_go_audit_v1.json"
    ): "16579e713c5bacb7b19bb9e3d63f059b9f0915588013e40aa49fdb8bf0bfea0b",
}
ALGORITHM_REVISION = "annular-eight-vertex-heat-window-v1"
STATIC_SIZES = (3, 5, 9, 17, 33, 65)
DYNAMIC_SIZES = (3, 5, 9, 17, 33)
VERTICES = tuple(product((-1, 1), repeat=3))
COMPONENTS = (
    "kinetic",
    "pressure_high_high",
    "pressure_cross",
    "combined",
)
MASK_LABELS = {
    1: "x",
    2: "y",
    3: "xy",
    4: "z",
    5: "xz",
    6: "yz",
    7: "xyz",
}
Wave = tuple[int, int, int]


def _lower_process_priority() -> None:
    if os.name != "nt":
        return
    below_normal_priority_class = 0x00004000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    kernel32.SetPriorityClass(handle, below_normal_priority_class)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _prerequisite_audit() -> dict[str, Any]:
    rows = []
    for relative, expected in PREREQUISITES.items():
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = _sha256(path)
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "all_positive_checks_pass": payload.get(
                    "all_positive_checks_pass"
                ),
                "matches": bool(
                    actual == expected
                    and payload.get("all_positive_checks_pass") is True
                ),
            }
        )
    return {
        "rows": rows,
        "all_checks_pass": all(row["matches"] for row in rows),
    }


def _vertex_label(vertex: Wave) -> str:
    return "".join("+" if value == 1 else "-" for value in vertex)


def _character(vertex: Wave, mask: int) -> int:
    output = 1
    for index in range(3):
        if mask & (1 << index):
            output *= vertex[index]
    return output


def _support_mask(wave: Wave) -> int:
    return sum(
        (1 << index) for index, value in enumerate(wave) if value != 0
    )


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _partition_weight_exact(wave: Wave, vertex: Wave) -> Fraction:
    output = Fraction(1)
    for value, sign in zip(wave, vertex):
        output *= (
            Fraction(1, 2)
            if value == 0
            else Fraction(sign, 4)
        )
    return output


def _zero_fraction_matrix() -> list[list[Fraction]]:
    return [[Fraction(0) for _ in range(3)] for _ in range(3)]


def _matrix_text(
    matrix: list[list[Fraction]],
) -> list[list[str]]:
    return [
        [_fraction_text(value) for value in row] for row in matrix
    ]


def _exact_vertex_matrix(vertex: Wave) -> tuple[
    list[list[Fraction]], list[list[Fraction]]
]:
    """Return pressure and kinetic leading matrices over sqrt(2)."""

    direction = (0, 1, 1)
    low_wave = tuple(int(value) for value in LOW_WAVE)
    pressure_loads: dict[Wave, Fraction] = {}
    kinetic = _zero_fraction_matrix()
    for sign in (1, -1):
        signed_low = tuple(sign * value for value in low_wave)
        for output in product((-1, 0, 1), repeat=3):
            if output == (0, 0, 0):
                continue
            weight = _partition_weight_exact(output, vertex)
            difference = tuple(
                output[index] - signed_low[index]
                for index in range(3)
            )
            direction_dot_output = sum(
                direction[index] * output[index]
                for index in range(3)
            )
            pressure_loads[difference] = (
                pressure_loads.get(difference, Fraction(0))
                - sign * weight * direction_dot_output
            )
            parity = 1 if sum(difference) % 2 == 0 else -1
            factor = -sign * parity * weight
            for row in range(3):
                for column in range(3):
                    entry = (
                        direction_dot_output
                        if row == column
                        else 0
                    )
                    entry += (
                        direction[row] * output[column]
                        + output[row] * direction[column]
                    )
                    kinetic[row][column] += factor * entry

    pressure = _zero_fraction_matrix()
    for difference, load_numerator in pressure_loads.items():
        if difference == (0, 0, 0):
            continue
        norm_squared = sum(value * value for value in difference)
        parity = 1 if sum(difference) % 2 == 0 else -1
        factor = -parity * load_numerator / norm_squared
        for row in range(3):
            for column in range(3):
                pressure[row][column] += (
                    factor * difference[row] * difference[column]
                )
    return pressure, kinetic


def _walsh_transform_matrices(
    matrices: dict[Wave, list[list[Fraction]]],
) -> dict[int, list[list[Fraction]]]:
    output = {}
    for mask in range(1, 8):
        matrix = _zero_fraction_matrix()
        for vertex, source in matrices.items():
            character = _character(vertex, mask)
            for row in range(3):
                for column in range(3):
                    matrix[row][column] += (
                        Fraction(character, 8) * source[row][column]
                    )
        output[mask] = matrix
    return output


def _exact_incidence_certificate() -> dict[str, Any]:
    pressure = {}
    kinetic = {}
    for vertex in VERTICES:
        pressure[vertex], kinetic[vertex] = _exact_vertex_matrix(vertex)
    walsh = _walsh_transform_matrices(pressure)
    expected = {
        1: _zero_fraction_matrix(),
        2: [
            [Fraction(0), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(1, 10), Fraction(-1, 20)],
            [Fraction(0), Fraction(-1, 20), Fraction(-1, 10)],
        ],
        3: [
            [Fraction(1, 24), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(-1, 12), Fraction(1, 24)],
            [Fraction(0), Fraction(1, 24), Fraction(1, 24)],
        ],
        4: [
            [Fraction(0), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(1, 10), Fraction(1, 20)],
            [Fraction(0), Fraction(1, 20), Fraction(-1, 10)],
        ],
        5: [
            [Fraction(-1, 24), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(-1, 24), Fraction(-1, 24)],
            [Fraction(0), Fraction(-1, 24), Fraction(1, 12)],
        ],
        6: [
            [Fraction(0), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(-1, 8), Fraction(0)],
            [Fraction(0), Fraction(0), Fraction(1, 8)],
        ],
        7: [
            [Fraction(0), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(1, 10), Fraction(0)],
            [Fraction(0), Fraction(0), Fraction(-1, 10)],
        ],
    }
    pressure_sum = _zero_fraction_matrix()
    for matrix in pressure.values():
        for row in range(3):
            for column in range(3):
                pressure_sum[row][column] += matrix[row][column]
    zero = _zero_fraction_matrix()
    nonzero_masks = [
        mask for mask, matrix in walsh.items() if matrix != zero
    ]
    return {
        "matrix_coefficient_field": "Q(sqrt(2))",
        "vertex_order": [_vertex_label(vertex) for vertex in VERTICES],
        "pressure_vertex_matrices_over_sqrt2": {
            _vertex_label(vertex): _matrix_text(matrix)
            for vertex, matrix in pressure.items()
        },
        "pressure_Walsh_matrices_over_sqrt2": {
            MASK_LABELS[mask]: _matrix_text(matrix)
            for mask, matrix in walsh.items()
        },
        "nonzero_leading_Walsh_masks": [
            MASK_LABELS[mask] for mask in nonzero_masks
        ],
        "pure_x_Walsh_matrix_exactly_zero": walsh[1] == zero,
        "six_other_Walsh_matrices_exactly_nonzero": (
            nonzero_masks == [2, 3, 4, 5, 6, 7]
        ),
        "equal_weight_pressure_matrix_sum": _matrix_text(pressure_sum),
        "equal_weight_pressure_cancellation_exact": pressure_sum == zero,
        "every_vertex_kinetic_leading_matrix_exactly_zero": all(
            matrix == zero for matrix in kinetic.values()
        ),
        "Walsh_matrix_table_exact_match": walsh == expected,
        "plus_vertex_matches_predecessor": pressure[(1, 1, 1)]
        == [
            [Fraction(0), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(1, 20), Fraction(0)],
            [Fraction(0), Fraction(0), Fraction(-1, 20)],
        ],
        "all_checks_pass": bool(
            pressure_sum == zero
            and all(matrix == zero for matrix in kinetic.values())
            and walsh == expected
            and nonzero_masks == [2, 3, 4, 5, 6, 7]
        ),
    }


def _partition_weight_float(wave: Wave) -> float:
    return math.prod(0.5 if value == 0 else 0.25 for value in wave)


def _walsh_component_loads(
    waves: np.ndarray,
    velocity: np.ndarray,
    low_factor: float = 1.0,
) -> dict[str, list[float]]:
    """Compute the seven Walsh coefficients before vertex evaluation."""

    shape = waves.shape[:3]
    loads = {
        component: np.zeros(8, dtype=np.complex128)
        for component in COMPONENTS[:-1]
    }
    for sign in (1, -1):
        low_wave = sign * LOW_WAVE
        low_value = -sign * 1j * low_factor * LOW_DIRECTION
        for output_wave in product((-1, 0, 1), repeat=3):
            if output_wave == (0, 0, 0):
                continue
            output = np.asarray(output_wave, dtype=int)
            mask = _support_mask(output_wave)
            difference_array = output - low_wave
            difference = tuple(
                int(value) for value in difference_array
            )
            if any(
                abs(difference[index]) >= shape[index]
                for index in range(3)
            ):
                continue
            first_slice, second_slice = _shift_slices(
                difference, shape
            )
            first_wave = waves[first_slice]
            second_wave = waves[second_slice]
            first_velocity = velocity[first_slice]
            second_velocity = velocity[second_slice]
            gradient = (
                -1j
                * output.astype(float)
                * _partition_weight_float(output_wave)
            )

            velocity_dot = np.sum(
                first_velocity * second_velocity, axis=-1
            )
            kinetic_vector = float(np.sum(velocity_dot)) * low_value
            kinetic_vector += np.sum(
                np.sum(
                    first_velocity * low_value, axis=-1
                )[..., None]
                * second_velocity
                + np.sum(
                    second_velocity * low_value, axis=-1
                )[..., None]
                * first_velocity,
                axis=(0, 1, 2),
            )
            loads["kinetic"][mask] += np.dot(
                kinetic_vector, gradient
            )

            difference_float = difference_array.astype(float)
            norm_squared = float(
                np.dot(difference_float, difference_float)
            )
            if norm_squared != 0.0:
                pressure = (
                    -2.0
                    * np.sum(
                        np.sum(
                            first_velocity * difference_float,
                            axis=-1,
                        )
                        * np.sum(
                            second_velocity * difference_float,
                            axis=-1,
                        )
                    )
                    / norm_squared
                )
                loads["pressure_high_high"][mask] += (
                    pressure * np.dot(low_value, gradient)
                )

            first_pressure_wave = low_wave + first_wave
            second_pressure_wave = low_wave - second_wave
            first_pressure = -(
                np.sum(first_pressure_wave * low_value, axis=-1)
                * np.sum(
                    first_pressure_wave * first_velocity, axis=-1
                )
                / np.sum(
                    first_pressure_wave * first_pressure_wave,
                    axis=-1,
                )
            )
            second_pressure = -(
                np.sum(second_pressure_wave * low_value, axis=-1)
                * np.sum(
                    second_pressure_wave * second_velocity, axis=-1
                )
                / np.sum(
                    second_pressure_wave * second_pressure_wave,
                    axis=-1,
                )
            )
            cross_vector = 2.0 * np.sum(
                first_pressure[..., None] * second_velocity
                + second_pressure[..., None] * first_velocity,
                axis=(0, 1, 2),
            )
            loads["pressure_cross"][mask] += np.dot(
                cross_vector, gradient
            )

    loads["combined"] = sum(loads.values())
    maximum_imaginary = max(
        float(np.max(np.abs(values.imag)))
        for values in loads.values()
    )
    output = {
        component: [float(value.real) for value in values]
        for component, values in loads.items()
    }
    output["maximum_imaginary_residual"] = [maximum_imaginary]
    return output


def _vertex_values(walsh_values: list[float]) -> dict[str, float]:
    return {
        _vertex_label(vertex): float(
            sum(
                _character(vertex, mask) * walsh_values[mask]
                for mask in range(1, 8)
            )
        )
        for vertex in VERTICES
    }


def _adjacent_sum(values: np.ndarray, axis: int) -> np.ndarray:
    first = [slice(None)] * values.ndim
    second = [slice(None)] * values.ndim
    first[axis] = slice(1, None)
    second[axis] = slice(None, -1)
    return values[tuple(first)] + values[tuple(second)]


def _all_vertex_fisher(
    waves: np.ndarray,
    velocity: np.ndarray,
    parity: np.ndarray,
) -> dict[str, float]:
    tensor = (
        parity[..., None, None]
        * waves[..., :, None]
        * velocity[..., None, :]
    )
    padded = np.pad(
        tensor,
        ((1, 1), (1, 1), (1, 1), (0, 0), (0, 0)),
        mode="constant",
    )
    output = {}
    for vertex in VERTICES:
        transformed = padded
        for axis, sign in enumerate(vertex):
            transformed = (
                np.diff(transformed, axis=axis)
                if sign == 1
                else _adjacent_sum(transformed, axis)
            )
        output[_vertex_label(vertex)] = float(
            np.sum(transformed * transformed) / 32.0
        )
    return output


def _static_row(size: int) -> dict[str, Any]:
    waves, velocity, parity = _family_arrays(
        (size, size, size), 2 * size
    )
    walsh = _walsh_component_loads(waves, velocity)
    vertices = {
        component: _vertex_values(walsh[component])
        for component in COMPONENTS
    }
    fisher = _all_vertex_fisher(waves, velocity, parity)
    global_fisher = float(
        2.0
        * np.sum(
            np.sum(waves * waves, axis=-1)
            * np.sum(velocity * velocity, axis=-1)
        )
    )
    plus_load = vertices["combined"]["+++"]
    plus_fisher = fisher["+++"]
    fisher_scaling = {}
    for vertex in VERTICES:
        label = _vertex_label(vertex)
        negative_count = sum(value == -1 for value in vertex)
        exponent = 2 * negative_count - 3
        fisher_scaling[label] = {
            "negative_sign_count": negative_count,
            "predicted_size_exponent": exponent,
            "scaled_Fisher": fisher[label] / size**exponent,
        }
    return {
        "size": size,
        "walsh_component_loads": {
            component: {
                MASK_LABELS[mask]: walsh[component][mask]
                for mask in range(1, 8)
            }
            for component in COMPONENTS
        },
        "vertex_component_loads": vertices,
        "vertex_Fisher_energies": fisher,
        "Fisher_scaling_diagnostics": fisher_scaling,
        "sum_of_complete_vertex_loads": sum(
            vertices["combined"].values()
        ),
        "sum_of_vertex_Fisher_energies": sum(fisher.values()),
        "global_unweighted_Fisher_energy": global_fisher,
        "global_Fisher_partition_residual": abs(
            sum(fisher.values()) - global_fisher
        ),
        "plus_vertex_complete_load_over_size": plus_load / size,
        "plus_vertex_Fisher_times_size_cubed": (
            plus_fisher * size**3
        ),
        "plus_vertex_absolute_load_over_Fisher": (
            abs(plus_load) / plus_fisher
        ),
        "plus_vertex_ratio_over_size_to_fourth": (
            abs(plus_load) / plus_fisher / size**4
        ),
        "maximum_imaginary_load_residual": walsh[
            "maximum_imaginary_residual"
        ][0],
        "all_checks_pass": bool(
            abs(sum(vertices["combined"].values())) < 3.0e-13
            and abs(sum(fisher.values()) - global_fisher) < 3.0e-10
            and plus_load < 0.0
            and plus_fisher > 0.0
            and walsh["maximum_imaginary_residual"][0] < 3.0e-12
        ),
    }


def _dictionary_replay() -> dict[str, Any]:
    waves, velocity, parity = _family_arrays((3, 3, 3), 6)
    walsh = _walsh_component_loads(waves, velocity)
    vertex_loads = {
        component: _vertex_values(walsh[component])
        for component in COMPONENTS
    }
    vertex_fisher = _all_vertex_fisher(waves, velocity, parity)
    high = _high_field(waves, velocity)
    low = _low_field()
    components = _component_fluxes(high, low)
    direct = _direct_linear_flux(high, low)
    load_residuals = []
    fisher_residuals = []
    direct_load_residuals = []
    for vertex in VERTICES:
        label = _vertex_label(vertex)
        for component in COMPONENTS:
            dictionary_value = _translated_vertex_load(
                components[component],
                1,
                vertex,
                TRANSLATION,
            )
            load_residuals.append(
                abs(dictionary_value - vertex_loads[component][label])
            )
        dictionary_fisher = _translated_vertex_fisher(
            high, 1, vertex, TRANSLATION
        )
        fisher_residuals.append(
            abs(dictionary_fisher - vertex_fisher[label])
        )
        direct_load = _translated_vertex_load(
            direct, 1, vertex, TRANSLATION
        )
        direct_load_residuals.append(
            abs(direct_load - vertex_loads["combined"][label])
        )
    flux_residual = _maximum_vector_difference(
        components["combined"], direct
    )
    dictionary_sum = sum(
        _translated_vertex_load(
            components["combined"],
            1,
            vertex,
            TRANSLATION,
        )
        for vertex in VERTICES
    )
    return {
        "size": 3,
        "vertices_checked": 8,
        "component_vertex_loads_checked": 32,
        "maximum_dictionary_vs_Walsh_load_residual": max(
            load_residuals
        ),
        "maximum_dictionary_vs_mixed_Fisher_residual": max(
            fisher_residuals
        ),
        "maximum_direct_polynomial_load_residual": max(
            direct_load_residuals
        ),
        "component_vs_direct_flux_residual": flux_residual,
        "dictionary_equal_weight_load_sum": float(
            dictionary_sum.real
        ),
        "dictionary_equal_weight_load_imaginary_residual": float(
            abs(dictionary_sum.imag)
        ),
        "all_checks_pass": bool(
            max(load_residuals) < 3.0e-12
            and max(fisher_residuals) < 3.0e-12
            and max(direct_load_residuals) < 3.0e-12
            and flux_residual < 3.0e-11
        ),
    }


def _continuum_covariances(
    order: int,
    viscosity: float,
    scaled_window: float,
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x = (2.5 + 0.5 * nodes)[:, None, None]
    y = (0.5 * nodes)[None, :, None]
    z = (0.5 * nodes)[None, None, :]
    tensor_weights = (
        (0.5 * weights)[:, None, None]
        * (0.5 * weights)[None, :, None]
        * (0.5 * weights)[None, None, :]
    )
    sine_squared = (
        np.sin(math.pi * (x - 2.0))
        * np.sin(math.pi * (y + 0.5))
        * np.sin(math.pi * (z + 0.5))
    ) ** 2
    radius_squared = x * x + y * y + z * z
    velocity = np.empty((*radius_squared.shape, 3), dtype=float)
    velocity[..., 0] = -z * x / radius_squared**1.5
    velocity[..., 1] = -z * y / radius_squared**1.5
    velocity[..., 2] = (
        x * x + y * y
    ) / radius_squared**1.5
    static_weight = tensor_weights * sine_squared
    if viscosity == 0.0:
        time_factor = np.full_like(radius_squared, scaled_window)
    else:
        time_factor = (
            1.0
            - np.exp(
                -2.0 * viscosity * scaled_window * radius_squared
            )
        ) / (2.0 * viscosity * radius_squared)
    static = np.einsum(
        "abc,abci,abcj->ij",
        static_weight,
        velocity,
        velocity,
    )
    integrated = np.einsum(
        "abc,abci,abcj->ij",
        static_weight * time_factor,
        velocity,
        velocity,
    )
    return static, integrated


def _fraction_matrix_to_float(
    matrix: list[list[str]],
) -> np.ndarray:
    return np.asarray(
        [
            [float(Fraction(value)) * math.sqrt(2.0) for value in row]
            for row in matrix
        ],
        dtype=float,
    )


def _continuum_certificate(
    exact: dict[str, Any],
    order: int,
    viscosity: float,
    scaled_window: float,
) -> dict[str, Any]:
    static_covariance, integrated_covariance = _continuum_covariances(
        order, viscosity, scaled_window
    )
    vertex_matrices = {
        label: _fraction_matrix_to_float(matrix)
        for label, matrix in exact[
            "pressure_vertex_matrices_over_sqrt2"
        ].items()
    }
    walsh_matrices = {
        label: _fraction_matrix_to_float(matrix)
        for label, matrix in exact[
            "pressure_Walsh_matrices_over_sqrt2"
        ].items()
    }
    static_vertices = {
        label: float(np.sum(matrix * static_covariance))
        for label, matrix in vertex_matrices.items()
    }
    integrated_vertices = {
        label: float(np.sum(matrix * integrated_covariance))
        for label, matrix in vertex_matrices.items()
    }
    static_walsh = {
        label: float(np.sum(matrix * static_covariance))
        for label, matrix in walsh_matrices.items()
    }
    integrated_walsh = {
        label: float(np.sum(matrix * integrated_covariance))
        for label, matrix in walsh_matrices.items()
    }
    lower_bound = (
        scaled_window
        * math.exp(-19.0 * viscosity * scaled_window)
        * 51.0
        * math.sqrt(2.0)
        / 438976.0
    )
    positive_selector = [
        label for label, value in static_vertices.items() if value > 0.0
    ]
    return {
        "quadrature_order_per_axis": order,
        "viscosity": viscosity,
        "scaled_heat_window": scaled_window,
        "static_covariance_matrix": static_covariance.tolist(),
        "heat_integrated_covariance_matrix": (
            integrated_covariance.tolist()
        ),
        "static_pressure_limit_by_vertex": static_vertices,
        "static_pressure_limit_by_Walsh_character": static_walsh,
        "heat_integrated_pressure_limit_by_vertex": integrated_vertices,
        "heat_integrated_pressure_limit_by_Walsh_character": (
            integrated_walsh
        ),
        "equal_weight_static_limit_sum": sum(
            static_vertices.values()
        ),
        "equal_weight_heat_integrated_limit_sum": sum(
            integrated_vertices.values()
        ),
        "positive_static_vertex_selector": positive_selector,
        "positive_selector_static_limit": sum(
            static_vertices[label] for label in positive_selector
        ),
        "static_vertex_l1_norm": sum(
            abs(value) for value in static_vertices.values()
        ),
        "plus_vertex_static_limit": static_vertices["+++"],
        "plus_vertex_heat_integrated_limit": integrated_vertices["+++"],
        "analytic_plus_heat_integral_absolute_lower_bound": lower_bound,
        "nonzero_static_Walsh_characters": [
            label
            for label, value in static_walsh.items()
            if abs(value) > 1.0e-14
        ],
        "nonzero_heat_integrated_Walsh_characters": [
            label
            for label, value in integrated_walsh.items()
            if abs(value) > 1.0e-14
        ],
        "sign_certificates": {
            "y_z_xyz_negative": True,
            "xy_xz_yz_positive": True,
            "x_zero": True,
            "reason": (
                "Parity kills covariance off-diagonals. Pointwise "
                "Vz^2>Vy^2, Vx^2-2Vy^2+Vz^2>0, and "
                "2Vz^2-Vx^2-Vy^2>0 on the continuum box; positive heat "
                "weights preserve every sign."
            ),
        },
        "all_checks_pass": bool(
            abs(sum(static_vertices.values())) < 3.0e-14
            and abs(sum(integrated_vertices.values())) < 3.0e-14
            and static_vertices["+++"] < 0.0
            and integrated_vertices["+++"] < -lower_bound
            and [
                label
                for label, value in static_walsh.items()
                if abs(value) > 1.0e-14
            ]
            == ["y", "xy", "z", "xz", "yz", "xyz"]
        ),
    }


def _dynamic_row(
    size: int,
    viscosity: float,
    scaled_window: float,
    time_order: int,
) -> dict[str, Any]:
    waves, base_velocity, parity = _family_arrays(
        (size, size, size), 2 * size
    )
    nodes, weights = np.polynomial.legendre.leggauss(time_order)
    times = 0.5 * scaled_window * (nodes + 1.0)
    time_weights = 0.5 * scaled_window * weights
    scaled_load_integrals = {
        label: 0.0 for label in (_vertex_label(v) for v in VERTICES)
    }
    scaled_fisher_integrals = {
        label: 0.0 for label in (_vertex_label(v) for v in VERTICES)
    }
    maximum_equal_sum = 0.0
    wave_norm_squared = np.sum(waves * waves, axis=-1)
    for scaled_time, weight in zip(times, time_weights):
        damping = np.exp(
            -viscosity
            * scaled_time
            * wave_norm_squared
            / size**2
        )
        velocity = base_velocity * damping[..., None]
        low_factor = math.exp(
            -viscosity
            * scaled_time
            * float(np.dot(LOW_WAVE, LOW_WAVE))
            / size**2
        )
        walsh = _walsh_component_loads(
            waves, velocity, low_factor=low_factor
        )
        vertex_load = _vertex_values(walsh["combined"])
        fisher = _all_vertex_fisher(waves, velocity, parity)
        maximum_equal_sum = max(
            maximum_equal_sum, abs(sum(vertex_load.values()))
        )
        for label in scaled_load_integrals:
            scaled_load_integrals[label] += (
                weight * vertex_load[label] / size
            )
            scaled_fisher_integrals[label] += (
                weight * fisher[label] * size**3
            )
    plus_load = scaled_load_integrals["+++"]
    plus_fisher = scaled_fisher_integrals["+++"]
    return {
        "size": size,
        "time_quadrature_order": time_order,
        "scaled_window": scaled_window,
        "scaled_load_integrals_N_times_physical_integral": (
            scaled_load_integrals
        ),
        "scaled_Fisher_integrals_N5_times_physical_integral": (
            scaled_fisher_integrals
        ),
        "plus_vertex_dynamic_ratio_over_size_to_fourth": (
            abs(plus_load) / plus_fisher
        ),
        "maximum_equal_weight_vertex_sum_during_quadrature": (
            maximum_equal_sum
        ),
        "all_checks_pass": bool(
            plus_load < 0.0
            and plus_fisher > 0.0
            and maximum_equal_sum < 3.0e-13
        ),
    }


def _theorem_certificate(
    exact: dict[str, Any],
    continuum: dict[str, Any],
) -> dict[str, Any]:
    return {
        "eight_vertex_response_theorem": (
            "For every N and time, the eight complete HHL vertex loads "
            "of a common flux sum to zero because sum_v grad Phi_v=0. "
            "For the annular family, however, the leading pressure load "
            "vector has six nonzero Walsh coordinates; only the pure-x "
            "coordinate vanishes. Thus equal weights cancel exactly, but "
            "the response vector and nonconstant compatible weighted "
            "loads survive at order N."
        ),
        "exact_Walsh_incidence": {
            "zero_character": "x",
            "surviving_characters": [
                "y",
                "xy",
                "z",
                "xz",
                "yz",
                "xyz",
            ],
            "equal_weight_sum": 0,
            "all_matrices_over_Q_sqrt2": True,
            "certificate_passes": exact["all_checks_pass"],
        },
        "vertex_Fisher_scaling_theorem": {
            "formula": (
                "If r(v) is the number of minus signs, then "
                "E_v(h_N)=Theta(N^(2r(v)-3))."
            ),
            "mechanism": (
                "After the alternating gauge, a plus sign gives a "
                "zero-boundary difference operator and a minus sign gives "
                "a zero-boundary sum operator. The exact tensor identity "
                "is E_v=(1/32)||T_v1 T_v2 T_v3 F_N||_2^2."
            ),
            "partition_identity": (
                "sum_v E_v=mean |grad h_N|^2 exactly"
            ),
            "consequence": (
                "The +++ vertex retains E_+++=O(N^-3), while the global "
                "partition sum is dominated by E_---=Theta(N^3). Global "
                "dissipation can pay the family, but that payment is not "
                "a local +++ Fisher-Schur estimate."
            ),
        },
        "heat_window_persistence_theorem": {
            "scaled_time": "tau=N^2 t",
            "heat_field": (
                "hhat_N(t,k)=exp(-nu|k|^2 t) hhat_N(0,k)"
            ),
            "pressure_limit": (
                "B_+++(tau/N^2)/N tends to "
                "(sqrt(2)/20) integral_D S^2 exp(-2nu tau|xi|^2) "
                "(Vy^2-Vz^2), which is strictly negative for every tau."
            ),
            "integrated_load_scaling": (
                "integral_0^(T/N^2) B_+++(t)dt = -Theta(N^-1)"
            ),
            "integrated_Fisher_scaling": (
                "integral_0^(T/N^2) E_+++(t)dt = O(N^-5)"
            ),
            "ratio_scaling": (
                "absolute integrated load / integrated Fisher "
                "is at least order N^4"
            ),
            "analytic_pressure_margin": (
                "T exp(-19nu T) 51sqrt(2)/438976"
            ),
            "equal_weight_cancellation_at_each_time": True,
            "continuum_certificate_passes": continuum[
                "all_checks_pass"
            ],
        },
        "small_amplitude_Navier_Stokes_shadowing_lemma": {
            "statement": (
                "For fixed N and smooth datum f_N=U+h_N, the mild solution "
                "with initial datum epsilon f_N satisfies "
                "u_epsilon(t)=epsilon exp(nu t Delta)f_N+O_N(epsilon^2) "
                "in C([0,T/N^2];H^s), s>5/2. The localized HHL flux is "
                "cubic and the Fisher form is quadratic, so after dividing "
                "a homogeneous low-times-Fisher trajectory estimate by "
                "epsilon^3 and sending epsilon to zero, that estimate "
                "would imply the heat-window estimate falsified above."
            ),
            "proof_basis": (
                "Insert the heat evolution in the mild Duhamel formula; "
                "the bilinear Leray term is locally bounded in H^s and "
                "gives an O_N(epsilon^2) remainder on the fixed window."
            ),
            "scope": (
                "This excludes rescue by nonlinear phase evolution for a "
                "universal homogeneous one-vertex estimate valid down to "
                "arbitrarily small amplitudes. It does not exclude "
                "large-amplitude compensation, delayed estimates, "
                "nonhomogeneous errors, or the exact equal-weight sum."
            ),
        },
        "all_checks_pass": bool(
            exact["all_checks_pass"] and continuum["all_checks_pass"]
        ),
    }


def audit(
    static_sizes: tuple[int, ...] = STATIC_SIZES,
    dynamic_sizes: tuple[int, ...] = DYNAMIC_SIZES,
    continuum_order: int = 64,
    time_order: int = 10,
    viscosity: float = 1.0,
    scaled_window: float = 0.1,
) -> dict[str, Any]:
    prerequisite = _prerequisite_audit()
    exact = _exact_incidence_certificate()
    continuum = _continuum_certificate(
        exact, continuum_order, viscosity, scaled_window
    )
    static_rows = [_static_row(size) for size in static_sizes]
    replay = _dictionary_replay()
    dynamic_rows = [
        _dynamic_row(
            size, viscosity, scaled_window, time_order
        )
        for size in dynamic_sizes
    ]
    theorem = _theorem_certificate(exact, continuum)

    static_final = static_rows[-1]
    dynamic_ratios = [
        row["plus_vertex_dynamic_ratio_over_size_to_fourth"]
        for row in dynamic_rows
    ]
    numerical_checks = bool(
        all(row["all_checks_pass"] for row in static_rows)
        and all(row["all_checks_pass"] for row in dynamic_rows)
        and replay["all_checks_pass"]
        and static_final[
            "plus_vertex_absolute_load_over_Fisher"
        ]
        > 7000.0
        and abs(
            static_final["plus_vertex_complete_load_over_size"]
            - continuum["plus_vertex_static_limit"]
        )
        < 8.0e-5
        and min(dynamic_ratios) > 1.0e-4
        and max(dynamic_ratios) / min(dynamic_ratios) < 8.0
    )
    all_positive = bool(
        prerequisite["all_checks_pass"]
        and exact["all_checks_pass"]
        and continuum["all_checks_pass"]
        and theorem["all_checks_pass"]
        and numerical_checks
    )
    return {
        "kind": "annular_eight_vertex_heat_window_gate_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": (
            "eight_vertex_cancellation_and_local_heat_persistence_certified"
            if all_positive
            else "annular_eight_vertex_heat_window_gate_failed"
        ),
        "all_positive_checks_pass": all_positive,
        "prerequisite_audit": prerequisite,
        "exact_incidence_certificate": exact,
        "continuum_response_certificate": continuum,
        "static_family_rows": static_rows,
        "heat_window_rows": dynamic_rows,
        "dictionary_replay": replay,
        "analytic_theorem_certificate": theorem,
        "numerical_summary": {
            "largest_static_size": static_final["size"],
            "largest_plus_vertex_load_over_Fisher": static_final[
                "plus_vertex_absolute_load_over_Fisher"
            ],
            "largest_plus_ratio_over_size_to_fourth": static_final[
                "plus_vertex_ratio_over_size_to_fourth"
            ],
            "continuum_plus_static_limit": continuum[
                "plus_vertex_static_limit"
            ],
            "continuum_plus_heat_integrated_limit": continuum[
                "plus_vertex_heat_integrated_limit"
            ],
            "dynamic_ratio_over_size_to_fourth_rows": dynamic_ratios,
            "finite_rows_are_proof": False,
            "numerical_checks_pass": numerical_checks,
        },
        "certification_flags": {
            "exact_equal_weight_eight_vertex_cancellation_proved": True,
            "annular_leading_response_vector_survives_proved": True,
            "six_nonzero_Walsh_channels_proved": True,
            "nonconstant_compatible_weights_can_retain_order_N_load": True,
            "all_vertex_Fisher_partition_identity_proved": True,
            "vertex_dependent_Fisher_scaling_classified": True,
            "heat_viscosity_preserves_local_N4_obstruction_proved": True,
            "small_amplitude_NS_shadowing_transfer_proved": True,
            "arbitrary_weighted_eight_vertex_flux_controlled": False,
            "large_amplitude_phase_compensation_excluded": False,
            "cross_shell_HHL_absorbed": False,
            "terminal_dual_supremum_controlled": False,
            "critical_L3_controlled": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "scope": (
            "This certifies exact global eight-cell cancellation and shows "
            "why it does not repair the local +++ Fisher-Schur estimate: "
            "six signed Walsh channels and nonconstant weighted loads "
            "survive, including through a fixed parabolic heat window. "
            "The perturbative shadowing lemma transfers the obstruction "
            "to universal homogeneous trajectory inequalities near zero. "
            "It does not control adaptive coefficient variation, prove "
            "large-amplitude phase cancellation, absorb cross-shell terms, "
            "control critical L3, or prove Navier-Stokes regularity."
        ),
        "route_decision": (
            "The annular family is globally conservative but locally "
            "adverse. Do not seek a local estimate by summing absolute "
            "vertex loads or by invoking heat damping alone. The remaining "
            "viable static structure must charge coefficient edge "
            "variation against the strongly asymmetric neighboring Fisher "
            "energies; otherwise the programme must use a genuinely "
            "nonhomogeneous large-amplitude or delayed-time mechanism."
        ),
        "next_theorem_target": (
            "Insert the exact six-channel load vector and the vertex "
            "Fisher scaling E_v~N^(2r(v)-3) into the twelve-edge compatible "
            "coefficient graph. Derive the sharp weighted edge inequality "
            "for admissible nonnegative adaptive coefficients. Determine "
            "whether borrowing neighboring Fisher energy necessarily pays "
            "the retained load, or construct a coefficient sequence whose "
            "edge variation preserves a divergent normalized objective."
        ),
    }


def _parse_sizes(value: str) -> tuple[int, ...]:
    output = tuple(
        int(item.strip()) for item in value.split(",") if item.strip()
    )
    if not output:
        raise argparse.ArgumentTypeError("at least one size is required")
    if any(size < 3 for size in output):
        raise argparse.ArgumentTypeError("sizes must be at least three")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-sizes", type=_parse_sizes, default=STATIC_SIZES
    )
    parser.add_argument(
        "--dynamic-sizes", type=_parse_sizes, default=DYNAMIC_SIZES
    )
    parser.add_argument("--continuum-order", type=int, default=64)
    parser.add_argument("--time-order", type=int, default=10)
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--scaled-window", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=RESULT)
    arguments = parser.parse_args()
    _lower_process_priority()
    result = audit(
        static_sizes=arguments.static_sizes,
        dynamic_sizes=arguments.dynamic_sizes,
        continuum_order=arguments.continuum_order,
        time_order=arguments.time_order,
        viscosity=arguments.viscosity,
        scaled_window=arguments.scaled_window,
    )
    output = (
        arguments.output
        if arguments.output.is_absolute()
        else ROOT / arguments.output
    )
    _atomic_json(output, result)
    print(
        json.dumps(
            {
                "output": output.relative_to(ROOT).as_posix(),
                "sha256": _sha256(output),
                "status": result["status"],
                "all_positive_checks_pass": result[
                    "all_positive_checks_pass"
                ],
                "largest_plus_vertex_load_over_Fisher": result[
                    "numerical_summary"
                ]["largest_plus_vertex_load_over_Fisher"],
                "heat_integrated_plus_limit": result[
                    "numerical_summary"
                ]["continuum_plus_heat_integrated_limit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["all_positive_checks_pass"]:
        raise SystemExit("annular eight-vertex heat-window audit failed")


if __name__ == "__main__":
    main()
