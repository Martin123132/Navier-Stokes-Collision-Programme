"""Audit joint primitive HHL incidence blocks against one Fisher matrix."""

from __future__ import annotations

import argparse
import ctypes
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
    _vertex_weight_hat,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "joint_primitive_hhl_incidence_schur_gate_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "primitive_hhl_chain_hardy_envelope_audit_v1.json"
    ): "89d5cee5520acead1deba0231bed2cc7e4e740a673223ff5ca733c4a8375d18a",
}
ALGORITHM_REVISION = "joint-incidence-witness-v1"
WINDOWS = (
    {
        "label": "axial_4",
        "carrier": 8,
        "longitudinal_length": 4,
        "transverse_y_radius": 0,
        "transverse_z_radius": 0,
    },
    {
        "label": "axial_8",
        "carrier": 8,
        "longitudinal_length": 8,
        "transverse_y_radius": 0,
        "transverse_z_radius": 0,
    },
    {
        "label": "strip_4x3",
        "carrier": 8,
        "longitudinal_length": 4,
        "transverse_y_radius": 1,
        "transverse_z_radius": 0,
    },
    {
        "label": "strip_4x5",
        "carrier": 8,
        "longitudinal_length": 4,
        "transverse_y_radius": 2,
        "transverse_z_radius": 0,
    },
    {
        "label": "slab_4x3x3",
        "carrier": 8,
        "longitudinal_length": 4,
        "transverse_y_radius": 1,
        "transverse_z_radius": 1,
    },
    {
        "label": "slab_4x5x3",
        "carrier": 8,
        "longitudinal_length": 4,
        "transverse_y_radius": 2,
        "transverse_z_radius": 1,
    },
    {
        "label": "slab_6x3x3",
        "carrier": 8,
        "longitudinal_length": 6,
        "transverse_y_radius": 1,
        "transverse_z_radius": 1,
    },
    {
        "label": "slab_8x3x3",
        "carrier": 8,
        "longitudinal_length": 8,
        "transverse_y_radius": 1,
        "transverse_z_radius": 1,
    },
)
Wave = tuple[int, int, int]
VectorField = dict[Wave, np.ndarray]
VERTEX: Wave = (1, 1, 1)
TRANSLATION = np.zeros(3, dtype=float)


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


def _negate(wave: Wave) -> Wave:
    return tuple(-value for value in wave)  # type: ignore[return-value]


def _canonical_cube_representatives() -> tuple[Wave, ...]:
    representatives = []
    for wave in product((-1, 0, 1), repeat=3):
        if wave == (0, 0, 0):
            continue
        first_nonzero = next(value for value in wave if value != 0)
        if first_nonzero > 0:
            representatives.append(wave)
    return tuple(representatives)


def _perpendicular_frame(wave: Wave) -> tuple[np.ndarray, np.ndarray]:
    wave_array = np.asarray(wave, dtype=float)
    unit = wave_array / np.linalg.norm(wave_array)
    coordinate = int(np.argmin(np.abs(unit)))
    seed = np.zeros(3, dtype=float)
    seed[coordinate] = 1.0
    first = seed - unit * np.dot(seed, unit)
    first /= np.linalg.norm(first)
    second = np.cross(unit, first)
    second /= np.linalg.norm(second)
    return first, second


def _low_components() -> tuple[dict[str, Any], ...]:
    components = []
    for wave in _canonical_cube_representatives():
        for polarization_index, polarization in enumerate(
            _perpendicular_frame(wave)
        ):
            for phase_label, phase in (
                ("cosine", 1.0 + 0.0j),
                ("sine", 0.0 + 1.0j),
            ):
                coefficient = phase * polarization
                components.append(
                    {
                        "wave": wave,
                        "polarization_index": polarization_index,
                        "phase": phase_label,
                        "coefficient": coefficient.astype(
                            np.complex128
                        ),
                    }
                )
    return tuple(components)


def _high_modes(specification: dict[str, Any]) -> tuple[Wave, ...]:
    carrier = int(specification["carrier"])
    length = int(specification["longitudinal_length"])
    y_radius = int(specification["transverse_y_radius"])
    z_radius = int(specification["transverse_z_radius"])
    return tuple(
        (carrier + offset, y_value, z_value)
        for offset in range(length)
        for y_value in range(-y_radius, y_radius + 1)
        for z_value in range(-z_radius, z_radius + 1)
    )


def _high_variables(
    specification: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    variables = []
    for mode_index, wave in enumerate(_high_modes(specification)):
        norm = float(np.linalg.norm(wave))
        for polarization_index, polarization in enumerate(
            _perpendicular_frame(wave)
        ):
            variables.append(
                {
                    "mode_index": mode_index,
                    "wave": wave,
                    "polarization_index": polarization_index,
                    "velocity": polarization / norm,
                }
            )
    return tuple(variables)


def _low_field(component: dict[str, Any]) -> VectorField:
    wave = component["wave"]
    coefficient = component["coefficient"]
    return {
        wave: coefficient,
        _negate(wave): np.conjugate(coefficient),
    }


def _combined_low_field(
    low_components: tuple[dict[str, Any], ...],
    coordinates: np.ndarray,
) -> VectorField:
    field: VectorField = {}
    for coordinate, component in zip(coordinates, low_components):
        for wave, value in _low_field(component).items():
            field[wave] = field.get(
                wave, np.zeros(3, dtype=np.complex128)
            ) + coordinate * value
    return field


def _high_field(
    variables: tuple[dict[str, Any], ...],
    coefficients: np.ndarray,
) -> VectorField:
    positive: VectorField = {}
    for variable, coefficient in zip(variables, coefficients):
        wave = variable["wave"]
        positive[wave] = positive.get(
            wave, np.zeros(3, dtype=np.complex128)
        ) + coefficient * variable["velocity"]
    field = dict(positive)
    for wave, coefficient in positive.items():
        field[_negate(wave)] = np.conjugate(coefficient)
    return field


def _is_vertex_output(wave: Wave) -> bool:
    return wave != (0, 0, 0) and all(
        abs(value) <= 1 for value in wave
    )


def _pressure_coefficient(
    first_wave: np.ndarray,
    first_value: np.ndarray,
    second_wave: np.ndarray,
    second_value: np.ndarray,
) -> complex:
    output_wave = first_wave + second_wave
    norm_squared = float(np.dot(output_wave, output_wave))
    if norm_squared == 0.0:
        return 0.0j
    return complex(
        -np.dot(output_wave, first_value)
        * np.dot(output_wave, second_value)
        / norm_squared
    )


def _complete_ordered_pair_symbol(
    positive_wave: np.ndarray,
    positive_value: np.ndarray,
    negative_wave: np.ndarray,
    negative_value: np.ndarray,
    low_wave: np.ndarray,
    low_value: np.ndarray,
) -> np.ndarray:
    """Sum the two ordered HHL symbols for c_pos conjugate(c_neg)."""

    high_high_pressure = _pressure_coefficient(
        positive_wave,
        positive_value,
        negative_wave,
        negative_value,
    )
    positive_cross_pressure = _pressure_coefficient(
        low_wave,
        low_value,
        positive_wave,
        positive_value,
    )
    negative_cross_pressure = _pressure_coefficient(
        low_wave,
        low_value,
        negative_wave,
        negative_value,
    )
    return (
        (
            np.dot(positive_value, negative_value)
            + 2.0 * high_high_pressure
        )
        * low_value
        + np.dot(low_value, positive_value) * negative_value
        + np.dot(low_value, negative_value) * positive_value
        + 2.0 * positive_cross_pressure * negative_value
        + 2.0 * negative_cross_pressure * positive_value
    )


def _incidence_map(
    low_components: tuple[dict[str, Any], ...],
) -> dict[Wave, tuple[dict[str, Any], ...]]:
    incidence: dict[Wave, tuple[dict[str, Any], ...]] = {}
    for difference in product(range(-2, 3), repeat=3):
        entries = []
        for low_index, component in enumerate(low_components):
            base_wave = component["wave"]
            base_value = component["coefficient"]
            for sign in (1, -1):
                low_wave = tuple(sign * value for value in base_wave)
                low_value = (
                    base_value
                    if sign == 1
                    else np.conjugate(base_value)
                )
                output_wave = tuple(
                    difference[index] + low_wave[index]
                    for index in range(3)
                )
                if not _is_vertex_output(output_wave):
                    continue
                gradient_wave = _negate(output_wave)
                weight = _vertex_weight_hat(
                    gradient_wave,
                    1,
                    VERTEX,
                    TRANSLATION,
                )
                gradient = (
                    1j
                    * np.asarray(gradient_wave, dtype=float)
                    * weight
                )
                entries.append(
                    {
                        "low_index": low_index,
                        "low_wave": np.asarray(low_wave, dtype=float),
                        "low_value": low_value,
                        "gradient": gradient,
                    }
                )
        incidence[difference] = tuple(entries)
    return incidence


def _assemble_fisher(
    variables: tuple[dict[str, Any], ...],
) -> np.ndarray:
    dimension = len(variables)
    fisher = np.zeros((dimension, dimension), dtype=np.complex128)
    for row, row_variable in enumerate(variables):
        row_wave = np.asarray(row_variable["wave"], dtype=float)
        row_value = row_variable["velocity"]
        for column, column_variable in enumerate(variables):
            column_wave = np.asarray(
                column_variable["wave"], dtype=float
            )
            difference_array = row_wave - column_wave
            difference = tuple(
                int(round(value)) for value in difference_array
            )
            weight = _vertex_weight_hat(
                difference,
                1,
                VERTEX,
                TRANSLATION,
            )
            if weight == 0.0:
                continue
            fisher[row, column] = (
                2.0
                * np.dot(row_wave, column_wave)
                * weight
                * np.dot(
                    row_value,
                    column_variable["velocity"],
                )
            )
    return fisher


def _assemble_load_blocks(
    variables: tuple[dict[str, Any], ...],
    low_components: tuple[dict[str, Any], ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    dimension = len(variables)
    loads = np.zeros(
        (len(low_components), dimension, dimension),
        dtype=np.complex128,
    )
    incidence = _incidence_map(low_components)
    active_variable_pairs = 0
    scalar_incidence_count = 0
    maximum_incidence_per_variable_pair = 0
    for row, negative_variable in enumerate(variables):
        negative_base_wave = np.asarray(
            negative_variable["wave"], dtype=float
        )
        negative_wave = -negative_base_wave
        negative_value = negative_variable["velocity"]
        for column, positive_variable in enumerate(variables):
            positive_wave = np.asarray(
                positive_variable["wave"], dtype=float
            )
            difference = tuple(
                int(round(value))
                for value in positive_wave - negative_base_wave
            )
            entries = incidence.get(difference, ())
            if not entries:
                continue
            active_variable_pairs += 1
            scalar_incidence_count += len(entries)
            maximum_incidence_per_variable_pair = max(
                maximum_incidence_per_variable_pair,
                len(entries),
            )
            positive_value = positive_variable["velocity"]
            for entry in entries:
                symbol = _complete_ordered_pair_symbol(
                    positive_wave,
                    positive_value,
                    negative_wave,
                    negative_value,
                    entry["low_wave"],
                    entry["low_value"],
                )
                loads[entry["low_index"], row, column] += np.dot(
                    symbol,
                    entry["gradient"],
                )
    return loads, {
        "active_ordered_variable_pairs": active_variable_pairs,
        "scalar_low_incidence_count": scalar_incidence_count,
        "maximum_scalar_low_incidence_per_variable_pair": (
            maximum_incidence_per_variable_pair
        ),
    }


def _maximum_hermitian_residual(matrices: np.ndarray) -> float:
    return max(
        (
            float(np.linalg.norm(matrix - np.conjugate(matrix.T), ord=2))
            for matrix in matrices
        ),
        default=0.0,
    )


def _direct_reconstruction(
    variables: tuple[dict[str, Any], ...],
    low_components: tuple[dict[str, Any], ...],
    fisher: np.ndarray,
    loads: np.ndarray,
    low_indices: tuple[int, ...],
) -> dict[str, Any]:
    rng = np.random.default_rng(20260728 + len(variables))
    coefficient_trials = tuple(
        rng.normal(size=len(variables))
        + 1j * rng.normal(size=len(variables))
        for _ in range(2)
    )
    fisher_residuals = []
    load_residuals = []
    load_imaginary_residuals = []
    component_residuals = []
    direct_flux_residuals = []
    divergence_residuals = []
    for coefficients in coefficient_trials:
        high = _high_field(variables, coefficients)
        direct_fisher = _translated_vertex_fisher(
            high,
            1,
            VERTEX,
            TRANSLATION,
        )
        matrix_fisher = np.vdot(
            coefficients, fisher @ coefficients
        )
        fisher_residuals.append(abs(direct_fisher - matrix_fisher))
        divergence_residuals.append(
            max(
                abs(np.dot(np.asarray(wave, dtype=float), value))
                for wave, value in high.items()
            )
        )
        for low_index in low_indices:
            low = _low_field(low_components[low_index])
            components = _component_fluxes(high, low)
            direct = _direct_linear_flux(high, low)
            component_load = _translated_vertex_load(
                components["combined"],
                1,
                VERTEX,
                TRANSLATION,
            )
            direct_load = _translated_vertex_load(
                direct,
                1,
                VERTEX,
                TRANSLATION,
            )
            matrix_load = np.vdot(
                coefficients,
                loads[low_index] @ coefficients,
            )
            load_residuals.append(abs(component_load - matrix_load))
            load_imaginary_residuals.extend(
                (
                    abs(component_load.imag),
                    abs(matrix_load.imag),
                    abs(direct_load.imag),
                )
            )
            component_residuals.append(
                abs(component_load - direct_load)
            )
            direct_flux_residuals.append(
                _maximum_vector_difference(
                    components["combined"], direct
                )
            )
    return {
        "coefficient_trials": len(coefficient_trials),
        "low_coordinates_checked": list(low_indices),
        "maximum_Fisher_matrix_residual": max(
            fisher_residuals, default=0.0
        ),
        "maximum_load_matrix_residual": max(
            load_residuals, default=0.0
        ),
        "maximum_load_imaginary_residual": max(
            load_imaginary_residuals, default=0.0
        ),
        "maximum_component_vs_direct_load_residual": max(
            component_residuals, default=0.0
        ),
        "maximum_component_vs_direct_flux_residual": max(
            direct_flux_residuals, default=0.0
        ),
        "maximum_divergence_residual": max(
            divergence_residuals, default=0.0
        ),
    }


def _largest_absolute_eigenpair(
    matrix: np.ndarray,
) -> tuple[float, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    index = int(np.argmax(np.abs(eigenvalues)))
    return float(eigenvalues[index]), eigenvectors[:, index]


def _joint_numerical_radius_lower(
    normalized_loads: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    low_dimension = normalized_loads.shape[0]
    rng = np.random.default_rng(81173 + normalized_loads.shape[1])
    starts = []
    coordinate_norms = []
    for low_index, matrix in enumerate(normalized_loads):
        eigenvalue, _ = _largest_absolute_eigenpair(matrix)
        coordinate_norms.append(abs(eigenvalue))
    for low_index in np.argsort(coordinate_norms)[-6:]:
        direction = np.zeros(low_dimension, dtype=float)
        direction[int(low_index)] = 1.0
        starts.append(direction)
    for _ in range(10):
        direction = rng.normal(size=low_dimension)
        direction /= np.linalg.norm(direction)
        starts.append(direction)

    best_value = 0.0
    best_iterations = 0
    best_direction = starts[0]
    best_vector = np.zeros(
        normalized_loads.shape[1], dtype=np.complex128
    )
    for start in starts:
        direction = start
        previous = -1.0
        for iteration in range(80):
            combined = np.tensordot(
                direction, normalized_loads, axes=(0, 0)
            )
            _, vector = _largest_absolute_eigenpair(combined)
            quadratic_values = np.asarray(
                [
                    np.vdot(vector, matrix @ vector).real
                    for matrix in normalized_loads
                ],
                dtype=float,
            )
            value = float(np.linalg.norm(quadratic_values))
            if value == 0.0:
                break
            direction = quadratic_values / value
            if abs(value - previous) <= 2.0e-13 * max(1.0, value):
                break
            previous = value
        if value > best_value:
            best_value = value
            best_iterations = iteration + 1
            best_direction = direction.copy()
            best_vector = vector.copy()
    return (
        {
            "deterministic_starts": len(starts),
            "maximum_iterations": 80,
            "best_lower_bound": best_value,
            "iterations_for_best_start": best_iterations,
        },
        best_direction,
        best_vector,
    )


def _joint_witness(
    variables: tuple[dict[str, Any], ...],
    low_components: tuple[dict[str, Any], ...],
    fisher: np.ndarray,
    loads: np.ndarray,
    transform: np.ndarray,
    normalized_loads: np.ndarray,
    low_direction: np.ndarray,
    normalized_high_vector: np.ndarray,
) -> dict[str, Any]:
    coefficients = transform @ normalized_high_vector
    high = _high_field(variables, coefficients)
    low = _combined_low_field(low_components, low_direction)
    components = _component_fluxes(high, low)
    direct = _direct_linear_flux(high, low)
    component_loads = {
        key: _translated_vertex_load(
            value,
            1,
            VERTEX,
            TRANSLATION,
        )
        for key, value in components.items()
    }
    direct_load = _translated_vertex_load(
        direct,
        1,
        VERTEX,
        TRANSLATION,
    )
    fisher_energy = np.vdot(coefficients, fisher @ coefficients)
    coordinate_quadratics = np.asarray(
        [
            np.vdot(
                normalized_high_vector,
                matrix @ normalized_high_vector,
            ).real
            for matrix in normalized_loads
        ],
        dtype=float,
    )
    matrix_load = float(np.dot(low_direction, coordinate_quadratics))
    low_order = np.argsort(np.abs(low_direction))[::-1]
    top_low_coordinates = []
    for index in low_order[:12]:
        component = low_components[int(index)]
        top_low_coordinates.append(
            {
                "coordinate_index": int(index),
                "value": float(low_direction[index]),
                "absolute_value": float(abs(low_direction[index])),
                "wave": component["wave"],
                "polarization_index": component[
                    "polarization_index"
                ],
                "phase": component["phase"],
            }
        )

    mode_mass: dict[int, float] = {}
    for variable, coefficient in zip(variables, coefficients):
        mode_index = int(variable["mode_index"])
        mode_mass[mode_index] = mode_mass.get(mode_index, 0.0) + float(
            abs(coefficient) ** 2
        )
    modes = _high_modes(
        {
            "carrier": min(variable["wave"][0] for variable in variables),
            "longitudinal_length": (
                max(variable["wave"][0] for variable in variables)
                - min(variable["wave"][0] for variable in variables)
                + 1
            ),
            "transverse_y_radius": max(
                abs(variable["wave"][1]) for variable in variables
            ),
            "transverse_z_radius": max(
                abs(variable["wave"][2]) for variable in variables
            ),
        }
    )
    total_mode_mass = sum(mode_mass.values())
    top_mode_indices = sorted(
        mode_mass, key=mode_mass.__getitem__, reverse=True
    )[:12]
    top_high_modes = [
        {
            "mode_index": index,
            "wave": modes[index],
            "coefficient_l2_mass": mode_mass[index],
            "mass_fraction": (
                mode_mass[index] / total_mode_mass
                if total_mode_mass > 0.0
                else 0.0
            ),
        }
        for index in top_mode_indices
    ]
    low_divergence = max(
        (
            abs(np.dot(np.asarray(wave, dtype=float), value))
            for wave, value in low.items()
        ),
        default=0.0,
    )
    return {
        "Fisher_energy": float(fisher_energy.real),
        "Fisher_energy_imaginary_residual": float(
            abs(fisher_energy.imag)
        ),
        "low_coordinate_l2_norm": float(
            np.linalg.norm(low_direction)
        ),
        "low_coordinate_l1_norm": float(
            np.linalg.norm(low_direction, ord=1)
        ),
        "matrix_complete_load": matrix_load,
        "direct_complete_load": float(direct_load.real),
        "direct_complete_load_imaginary_residual": float(
            abs(direct_load.imag)
        ),
        "matrix_vs_direct_complete_load_residual": float(
            abs(matrix_load - direct_load)
        ),
        "component_loads": {
            key: float(value.real)
            for key, value in component_loads.items()
        },
        "maximum_component_load_imaginary_residual": max(
            abs(value.imag) for value in component_loads.values()
        ),
        "component_vs_direct_flux_residual": (
            _maximum_vector_difference(components["combined"], direct)
        ),
        "maximum_low_divergence_residual": float(low_divergence),
        "high_coefficient_l2_norm": float(
            np.linalg.norm(coefficients)
        ),
        "maximum_high_coefficient": float(
            np.max(np.abs(coefficients))
        ),
        "top_low_coordinates": top_low_coordinates,
        "top_high_modes_by_coefficient_mass": top_high_modes,
    }


def _normalized_spectra(
    fisher: np.ndarray,
    loads: np.ndarray,
    variables: tuple[dict[str, Any], ...],
    low_components: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    fisher_hermitian = 0.5 * (fisher + np.conjugate(fisher.T))
    loads_hermitian = 0.5 * (
        loads + np.conjugate(np.swapaxes(loads, 1, 2))
    )
    fisher_eigenvalues, fisher_eigenvectors = np.linalg.eigh(
        fisher_hermitian
    )
    maximum_fisher = float(np.max(fisher_eigenvalues))
    quotient_tolerance = (
        4096.0
        * np.finfo(float).eps
        * fisher.shape[0]
        * max(1.0, maximum_fisher)
    )
    retained = fisher_eigenvalues > quotient_tolerance
    transform = (
        fisher_eigenvectors[:, retained]
        / np.sqrt(fisher_eigenvalues[retained])[None, :]
    )
    normalized_loads = np.asarray(
        [
            np.conjugate(transform.T) @ matrix @ transform
            for matrix in loads_hermitian
        ]
    )
    coordinate_norms = np.asarray(
        [
            max(abs(np.linalg.eigvalsh(matrix)))
            for matrix in normalized_loads
        ],
        dtype=float,
    )
    schur_square = np.zeros(
        normalized_loads.shape[1:],
        dtype=np.complex128,
    )
    for matrix in normalized_loads:
        schur_square += matrix @ matrix
    schur_square = 0.5 * (
        schur_square + np.conjugate(schur_square.T)
    )
    schur_eigenvalues = np.linalg.eigvalsh(schur_square)
    square_function_upper = math.sqrt(
        max(float(schur_eigenvalues[-1]), 0.0)
    )
    lower, low_direction, normalized_high_vector = (
        _joint_numerical_radius_lower(normalized_loads)
    )
    witness = _joint_witness(
        variables,
        low_components,
        fisher_hermitian,
        loads_hermitian,
        transform,
        normalized_loads,
        low_direction,
        normalized_high_vector,
    )
    return {
        "Fisher_dimension": fisher.shape[0],
        "Fisher_rank_after_exact_null_quotient": int(
            np.count_nonzero(retained)
        ),
        "Fisher_numerical_nullity": int(
            np.count_nonzero(~retained)
        ),
        "Fisher_quotient_tolerance": quotient_tolerance,
        "Fisher_minimum_eigenvalue": float(fisher_eigenvalues[0]),
        "Fisher_minimum_retained_eigenvalue": float(
            np.min(fisher_eigenvalues[retained])
        ),
        "Fisher_maximum_eigenvalue": maximum_fisher,
        "Fisher_condition_number_on_quotient": float(
            maximum_fisher / np.min(fisher_eigenvalues[retained])
        ),
        "maximum_single_low_coordinate_norm": float(
            np.max(coordinate_norms)
        ),
        "root_sum_square_coordinate_norm_upper": float(
            np.linalg.norm(coordinate_norms)
        ),
        "Schur_square_function_spectrum_minimum": float(
            schur_eigenvalues[0]
        ),
        "Schur_square_function_spectrum_maximum": float(
            schur_eigenvalues[-1]
        ),
        "Schur_square_function_upper": square_function_upper,
        "joint_l2_numerical_radius_lower": lower,
        "joint_l2_witness": witness,
        "lower_to_Schur_upper_ratio": (
            lower["best_lower_bound"] / square_function_upper
            if square_function_upper > 0.0
            else 0.0
        ),
    }


def _window_row(
    specification: dict[str, Any],
    low_components: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    variables = _high_variables(specification)
    fisher = _assemble_fisher(variables)
    loads, incidence = _assemble_load_blocks(
        variables, low_components
    )
    fisher_hermitian_residual = float(
        np.linalg.norm(
            fisher - np.conjugate(fisher.T), ord=2
        )
    )
    load_hermitian_residual = _maximum_hermitian_residual(loads)
    validation_indices = (
        tuple(range(len(low_components)))
        if len(variables) <= 8
        else (0, 1, 7, 19, 31, 43, 51)
    )
    reconstruction = _direct_reconstruction(
        variables,
        low_components,
        fisher,
        loads,
        validation_indices,
    )
    spectra = _normalized_spectra(
        fisher,
        loads,
        variables,
        low_components,
    )
    mode_count = len(_high_modes(specification))
    positive_negative_alias_count = sum(
        1
        for first in _high_modes(specification)
        for second in _high_modes(specification)
        for low in _canonical_cube_representatives()
        for sign in (-1, 1)
        if _is_vertex_output(
            tuple(
                first[index] + second[index] + sign * low[index]
                for index in range(3)
            )
        )
    )
    row = {
        **specification,
        "high_mode_count": mode_count,
        "complex_high_variable_dimension": len(variables),
        "real_high_dimension": 2 * len(variables),
        "low_wave_representative_count": len(
            _canonical_cube_representatives()
        ),
        "real_low_coordinate_count": len(low_components),
        "same_sign_high_alias_count": positive_negative_alias_count,
        "incidence": incidence,
        "Fisher_hermitian_residual": fisher_hermitian_residual,
        "maximum_load_block_hermitian_residual": (
            load_hermitian_residual
        ),
        "direct_reconstruction": reconstruction,
        "normalized_spectra": spectra,
    }
    row["all_checks_pass"] = bool(
        positive_negative_alias_count == 0
        and fisher_hermitian_residual < 3.0e-12
        and load_hermitian_residual < 3.0e-12
        and spectra["Fisher_minimum_eigenvalue"]
        > -spectra["Fisher_quotient_tolerance"]
        and spectra["Fisher_rank_after_exact_null_quotient"] > 0
        and reconstruction["maximum_Fisher_matrix_residual"]
        < 3.0e-11
        and reconstruction["maximum_load_matrix_residual"] < 3.0e-11
        and reconstruction[
            "maximum_component_vs_direct_load_residual"
        ]
        < 3.0e-11
        and reconstruction[
            "maximum_component_vs_direct_flux_residual"
        ]
        < 3.0e-11
        and reconstruction["maximum_load_imaginary_residual"]
        < 3.0e-11
        and reconstruction["maximum_divergence_residual"] < 3.0e-11
        and spectra["joint_l2_numerical_radius_lower"][
            "best_lower_bound"
        ]
        <= spectra["Schur_square_function_upper"] + 3.0e-11
        and abs(
            spectra["joint_l2_witness"]["Fisher_energy"] - 1.0
        )
        < 3.0e-10
        and abs(
            spectra["joint_l2_witness"]["low_coordinate_l2_norm"]
            - 1.0
        )
        < 3.0e-10
        and spectra["joint_l2_witness"][
            "matrix_vs_direct_complete_load_residual"
        ]
        < 3.0e-10
        and spectra["joint_l2_witness"][
            "component_vs_direct_flux_residual"
        ]
        < 3.0e-10
        and spectra["joint_l2_witness"][
            "maximum_component_load_imaginary_residual"
        ]
        < 3.0e-10
        and spectra["joint_l2_witness"][
            "maximum_low_divergence_residual"
        ]
        < 3.0e-10
    )
    return row


def _result_payload(
    prerequisite: dict[str, Any],
    rows: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    upper_values = [
        row["normalized_spectra"]["Schur_square_function_upper"]
        for row in rows
    ]
    lower_values = [
        row["normalized_spectra"][
            "joint_l2_numerical_radius_lower"
        ]["best_lower_bound"]
        for row in rows
    ]
    by_label = {row["label"]: row for row in rows}
    slab_length_rows = [
        by_label[label]
        for label in (
            "slab_4x3x3",
            "slab_6x3x3",
            "slab_8x3x3",
        )
        if label in by_label
    ]
    slab_lengths = np.asarray(
        [row["longitudinal_length"] for row in slab_length_rows],
        dtype=float,
    )
    slab_lowers = np.asarray(
        [
            row["normalized_spectra"][
                "joint_l2_numerical_radius_lower"
            ]["best_lower_bound"]
            for row in slab_length_rows
        ],
        dtype=float,
    )
    if len(slab_lengths) >= 2:
        slope, intercept = np.polyfit(slab_lengths, slab_lowers, 1)
        fitted = slope * slab_lengths + intercept
        total_square = float(
            np.sum((slab_lowers - np.mean(slab_lowers)) ** 2)
        )
        residual_square = float(np.sum((slab_lowers - fitted) ** 2))
        linear_r_squared = (
            1.0 - residual_square / total_square
            if total_square > 0.0
            else 1.0
        )
    else:
        slope = intercept = linear_r_squared = float("nan")

    def lower(label: str) -> float | None:
        row = by_label.get(label)
        if row is None:
            return None
        return float(
            row["normalized_spectra"][
                "joint_l2_numerical_radius_lower"
            ]["best_lower_bound"]
        )

    def ratio(numerator: str, denominator: str) -> float | None:
        top = lower(numerator)
        bottom = lower(denominator)
        if top is None or bottom is None or bottom == 0.0:
            return None
        return top / bottom

    pressure_fractions = {}
    for label, row in by_label.items():
        witness = row["normalized_spectra"]["joint_l2_witness"]
        complete = witness["matrix_complete_load"]
        pressure_fractions[label] = (
            witness["component_loads"]["pressure_high_high"] / complete
            if complete != 0.0
            else 0.0
        )
    all_rows_pass = bool(rows) and all(
        row["all_checks_pass"] for row in rows
    )
    all_positive = bool(
        complete
        and prerequisite["all_checks_pass"]
        and all_rows_pass
        and len(rows) == len(WINDOWS)
    )
    return {
        "kind": "joint_primitive_hhl_incidence_schur_gate_audit",
        "schema_version": 1,
        "algorithm_revision": ALGORITHM_REVISION,
        "run_state": "complete" if complete else "checkpointed_incomplete",
        "status": (
            "finite_window_joint_pressure_growth_witnesses_validated"
            if all_positive
            else (
                "checkpointed_incomplete"
                if not complete
                else "audit_failed"
            )
        ),
        "all_positive_checks_pass": all_positive,
        "prerequisites": prerequisite,
        "geometry": {
            "partition_scale": 1,
            "vertex": VERTEX,
            "translation": TRANSLATION.tolist(),
            "high_velocity_basis": (
                "two real orthonormal divergence-free polarizations "
                "divided by |k|, so each complex coordinate is "
                "gradient-normalized"
            ),
            "low_basis": (
                "thirteen cube waves modulo sign, two real transverse "
                "polarizations, and cosine/sine Fourier phases"
            ),
            "complete_ordered_HHL_symbol": (
                "(ha dot hb)U/2+(U dot ha)hb+p[ha,hb]U"
                "+2p[U,ha]hb, summed over both ordered high legs"
            ),
        },
        "window_rows": rows,
        "growth_summary": {
            "completed_window_count": len(rows),
            "Schur_upper_values": upper_values,
            "joint_lower_values": lower_values,
            "largest_to_smallest_Schur_upper_ratio": (
                max(upper_values) / min(upper_values)
                if upper_values and min(upper_values) > 0.0
                else None
            ),
            "last_to_first_Schur_upper_ratio": (
                upper_values[-1] / upper_values[0]
                if upper_values and upper_values[0] > 0.0
                else None
            ),
            "directional_comparisons": {
                "axial_length_8_over_4_joint_lower_ratio": ratio(
                    "axial_8", "axial_4"
                ),
                "strip_width_5_over_3_joint_lower_ratio": ratio(
                    "strip_4x5", "strip_4x3"
                ),
                "slab_width_5_over_3_joint_lower_ratio": ratio(
                    "slab_4x5x3", "slab_4x3x3"
                ),
                "slab_length_rows": [
                    {
                        "length": int(length),
                        "joint_lower": float(value),
                    }
                    for length, value in zip(
                        slab_lengths, slab_lowers
                    )
                ],
                "slab_length_linear_fit_slope": float(slope),
                "slab_length_linear_fit_intercept": float(intercept),
                "slab_length_linear_fit_R_squared": float(
                    linear_r_squared
                ),
                "pressure_high_high_fraction_by_window": (
                    pressure_fractions
                ),
            },
            "observed_route_signal": (
                "Axial length is nearly saturated over 4 to 8 modes, "
                "while transverse widening and three-dimensional slab "
                "length produce strongly increasing direct witnesses "
                "dominated by the high-high pressure term."
            ),
            "interpretation": (
                "Finite-window spectra diagnose the next proof route. "
                "They do not prove asymptotic divergence or a "
                "window-uniform bound."
            ),
        },
        "certification_flags": {
            "complete_HHL_blocks_assembled_jointly": all_positive,
            "shared_physical_Fisher_charged_once": all_positive,
            "all_52_real_low_coordinates_included": all_positive,
            "finite_windows_pass_direct_reconstruction": all_rows_pass,
            "finite_window_pressure_growth_witnesses_validated": (
                all_positive
            ),
            "analytic_unbounded_pressure_family_proved": False,
            "window_uniform_joint_Schur_bound_proved": False,
            "all_primitive_steps_and_residue_chains_controlled": False,
            "all_cross_shell_HHL_absorbed": False,
            "terminal_dual_supremum_controlled": False,
            "critical_L3_controlled": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "scope": (
            "This is a finite-window joint incidence calculation, not an "
            "analytic uniform theorem. It removes the repeated-Fisher "
            "bookkeeping error inside the tested windows and exposes the "
            "normalized graph whose growth must next be bounded or "
            "converted into a rigorous obstruction."
        ),
        "next_theorem_target": (
            "Extract a separable divergence-free high-carrier family from "
            "the pressure-dominated slab witnesses. Compute its weighted "
            "Fisher energy and high-high pressure HHL load analytically as "
            "the longitudinal and transverse Dirichlet windows grow. "
            "Either prove an explicit divergent lower ratio, which closes "
            "the joint-Schur route as a no-go, or identify the cancellation "
            "missing from the finite numerical witness."
        ),
    }


def audit(
    *,
    checkpoint_path: Path | None = None,
    windows: tuple[dict[str, Any], ...] = WINDOWS,
) -> dict[str, Any]:
    prerequisite = _prerequisite_audit()
    low_components = _low_components()
    rows: list[dict[str, Any]] = []
    reusable_rows: dict[str, dict[str, Any]] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            previous = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            previous = {}
        if (
            previous.get("kind")
            == "joint_primitive_hhl_incidence_schur_gate_audit"
            and previous.get("algorithm_revision")
            == ALGORITHM_REVISION
            and previous.get("prerequisites") == prerequisite
        ):
            reusable_rows = {
                row["label"]: row
                for row in previous.get("window_rows", ())
                if row.get("all_checks_pass") is True
            }
    for index, specification in enumerate(windows):
        reusable = reusable_rows.get(specification["label"])
        if reusable is not None and all(
            reusable.get(key) == value
            for key, value in specification.items()
        ):
            rows.append(reusable)
            print(
                "reused "
                f"{specification['label']} ({index + 1}/{len(windows)})",
                flush=True,
            )
            continue
        print(
            "assembling "
            f"{specification['label']} ({index + 1}/{len(windows)})",
            flush=True,
        )
        rows.append(_window_row(specification, low_components))
        if checkpoint_path is not None:
            _atomic_json(
                checkpoint_path,
                _result_payload(prerequisite, rows, complete=False),
            )
        print(
            "completed "
            f"{specification['label']}: "
            "Schur upper="
            f"{rows[-1]['normalized_spectra']['Schur_square_function_upper']:.12g}, "
            "joint lower="
            f"{rows[-1]['normalized_spectra']['joint_l2_numerical_radius_lower']['best_lower_bound']:.12g}",
            flush=True,
        )
    return _result_payload(prerequisite, rows, complete=True)


def main() -> None:
    _lower_process_priority()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smallest-only",
        action="store_true",
        help="run only the axial four-mode validation window",
    )
    arguments = parser.parse_args()
    windows = WINDOWS[:1] if arguments.smallest_only else WINDOWS
    result = audit(
        checkpoint_path=None if arguments.check_only else arguments.output,
        windows=windows,
    )
    if not arguments.check_only:
        _atomic_json(arguments.output, result)
    if not result["all_positive_checks_pass"]:
        if arguments.smallest_only and all(
            row["all_checks_pass"] for row in result["window_rows"]
        ):
            return
        raise SystemExit("joint primitive HHL incidence-Schur audit failed")
    print(json.dumps(result["growth_summary"], indent=2))


if __name__ == "__main__":
    main()
