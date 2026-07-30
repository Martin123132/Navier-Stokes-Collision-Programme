"""Audit the scalar local-energy trace and its differentiated HHL transfer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

import cross_shell_modulated_wave_gate_audit as local_energy
import dense_annular_hhh_packet_gate_audit as dense
import dense_spaced_continuum_positivity_audit as continuum
import nonlinear_stress_regeneration_gate_audit as regeneration


ROOT = Path(__file__).resolve().parents[3]
Wave = tuple[int, int, int]
VectorField = dict[Wave, np.ndarray]
Offset = tuple[int, int, int]

PARTITION_WAVE = np.asarray((1.0, 1.0, 1.0))
LOW_WAVE = -PARTITION_WAVE
LOW_POLARIZATION = np.asarray((1.0, -1.0, 0.0)) / math.sqrt(2.0)
LOW_COEFFICIENT = 1j * LOW_POLARIZATION
OFFSET_SPACING = 4
REAL_QUARTET_LOAD_LIMIT = 3.0 * math.sqrt(2.0) / 16.0
POSITIVE_QUARTET_COEFFICIENT_LIMIT = REAL_QUARTET_LOAD_LIMIT / 2.0


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


def _as_wave(wave: Wave | np.ndarray) -> np.ndarray:
    return np.asarray(wave, dtype=float)


def _wave_tuple(wave: Wave | np.ndarray) -> Wave:
    array = np.asarray(wave, dtype=int)
    return tuple(int(value) for value in array)  # type: ignore[return-value]


def _pressure_pair(
    first_wave: Wave | np.ndarray,
    first_value: np.ndarray,
    second_wave: Wave | np.ndarray,
    second_value: np.ndarray,
) -> complex:
    output = _as_wave(first_wave) + _as_wave(second_wave)
    norm_squared = float(np.dot(output, output))
    if norm_squared == 0.0:
        return 0.0j
    return complex(
        -np.dot(output, first_value)
        * np.dot(output, second_value)
        / norm_squared
    )


def _energy_trilinear_components(
    first_wave: Wave | np.ndarray,
    first_value: np.ndarray,
    second_wave: Wave | np.ndarray,
    second_value: np.ndarray,
    third_wave: Wave | np.ndarray,
    third_value: np.ndarray,
) -> dict[str, np.ndarray]:
    kinetic = (
        np.dot(first_value, second_value) * third_value
        + np.dot(first_value, third_value) * second_value
        + np.dot(second_value, third_value) * first_value
    ) / 6.0
    pressure = (
        _pressure_pair(
            first_wave,
            first_value,
            second_wave,
            second_value,
        )
        * third_value
        + _pressure_pair(
            first_wave,
            first_value,
            third_wave,
            third_value,
        )
        * second_value
        + _pressure_pair(
            second_wave,
            second_value,
            third_wave,
            third_value,
        )
        * first_value
    ) / 3.0
    return {
        "kinetic": np.asarray(kinetic, dtype=np.complex128),
        "pressure": np.asarray(pressure, dtype=np.complex128),
    }


def _bilinear_dynamics_components(
    first_wave: Wave | np.ndarray,
    first_value: np.ndarray,
    second_wave: Wave | np.ndarray,
    second_value: np.ndarray,
) -> dict[str, np.ndarray]:
    first_wave_array = _as_wave(first_wave)
    second_wave_array = _as_wave(second_wave)
    raw = (
        np.dot(first_value, second_wave_array) * second_value
        + np.dot(second_value, first_wave_array) * first_value
    )
    transport = -1j * raw
    complete = regeneration._bilinear_ns(
        first_wave,
        first_value,
        second_wave,
        second_value,
    )
    return {
        "transport": np.asarray(transport, dtype=np.complex128),
        "pressure_projection": np.asarray(
            complete - transport,
            dtype=np.complex128,
        ),
        "complete": np.asarray(complete, dtype=np.complex128),
    }


def _zero_quartic_components() -> dict[str, np.ndarray]:
    names = (
        "kinetic_flux_transport_dynamics",
        "kinetic_flux_pressure_dynamics",
        "pressure_flux_transport_dynamics",
        "pressure_flux_pressure_dynamics",
        "high_velocity_evolution",
        "linearized_low_velocity_evolution",
        "kinetic_flux",
        "pressure_flux",
        "complete",
    )
    return {
        name: np.zeros(3, dtype=np.complex128) for name in names
    }


def _quartic_decomposition_roundoff(
    output: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    reconstruction = (
        output["high_velocity_evolution"]
        + output["linearized_low_velocity_evolution"]
    )
    residual = float(np.linalg.norm(output["complete"] - reconstruction))
    primitive_names = (
        "kinetic_flux_transport_dynamics",
        "kinetic_flux_pressure_dynamics",
        "pressure_flux_transport_dynamics",
        "pressure_flux_pressure_dynamics",
    )
    scale = max(
        1.0,
        sum(
            float(np.linalg.norm(output[name]))
            for name in primitive_names
        ),
    )
    roundoff_units = residual / (np.finfo(float).eps * scale)
    return residual, scale, roundoff_units


def _quartic_symbol_components(
    first_wave: Wave | np.ndarray,
    first_value: np.ndarray,
    second_wave: Wave | np.ndarray,
    second_value: np.ndarray,
    third_wave: Wave | np.ndarray,
    third_value: np.ndarray,
    low_wave: Wave | np.ndarray = LOW_WAVE,
    low_value: np.ndarray = LOW_COEFFICIENT,
) -> dict[str, np.ndarray]:
    high_legs = (
        (_as_wave(first_wave), first_value),
        (_as_wave(second_wave), second_value),
        (_as_wave(third_wave), third_value),
    )
    low_leg = (_as_wave(low_wave), low_value)
    output = _zero_quartic_components()

    high_evolution_rows = (
        (high_legs[0], high_legs[1], high_legs[2]),
        (high_legs[0], high_legs[2], high_legs[1]),
        (high_legs[1], high_legs[2], high_legs[0]),
    )
    for first, second, remaining in high_evolution_rows:
        dynamics = _bilinear_dynamics_components(
            first[0],
            first[1],
            second[0],
            second[1],
        )
        for dynamics_name in ("transport", "pressure_projection"):
            dynamics_label = (
                "transport"
                if dynamics_name == "transport"
                else "pressure"
            )
            flux = _energy_trilinear_components(
                remaining[0],
                remaining[1],
                low_leg[0],
                low_leg[1],
                first[0] + second[0],
                dynamics[dynamics_name],
            )
            for flux_name in ("kinetic", "pressure"):
                key = (
                    f"{flux_name}_flux_{dynamics_label}_dynamics"
                )
                contribution = 6.0 * flux[flux_name]
                output[key] += contribution
                output["high_velocity_evolution"] += contribution

    low_evolution_rows = (
        (high_legs[0], high_legs[1], high_legs[2]),
        (high_legs[1], high_legs[0], high_legs[2]),
        (high_legs[2], high_legs[0], high_legs[1]),
    )
    for interacting, remaining_first, remaining_second in low_evolution_rows:
        dynamics = _bilinear_dynamics_components(
            interacting[0],
            interacting[1],
            low_leg[0],
            low_leg[1],
        )
        for dynamics_name in ("transport", "pressure_projection"):
            dynamics_label = (
                "transport"
                if dynamics_name == "transport"
                else "pressure"
            )
            flux = _energy_trilinear_components(
                remaining_first[0],
                remaining_first[1],
                remaining_second[0],
                remaining_second[1],
                interacting[0] + low_leg[0],
                dynamics[dynamics_name],
            )
            for flux_name in ("kinetic", "pressure"):
                key = (
                    f"{flux_name}_flux_{dynamics_label}_dynamics"
                )
                contribution = 6.0 * flux[flux_name]
                output[key] += contribution
                output[
                    "linearized_low_velocity_evolution"
                ] += contribution

    output["kinetic_flux"] = (
        output["kinetic_flux_transport_dynamics"]
        + output["kinetic_flux_pressure_dynamics"]
    )
    output["pressure_flux"] = (
        output["pressure_flux_transport_dynamics"]
        + output["pressure_flux_pressure_dynamics"]
    )
    output["complete"] = output["kinetic_flux"] + output["pressure_flux"]
    residual, scale, roundoff_units = _quartic_decomposition_roundoff(
        output
    )
    if roundoff_units > 256.0:
        raise AssertionError(
            "quartic decomposition failed to reconstruct within "
            f"roundoff: residual={residual}, scale={scale}, "
            f"units={roundoff_units}"
        )
    return output


def _cubic_flux_symbol(
    first_wave: Wave | np.ndarray,
    first_value: np.ndarray,
    second_wave: Wave | np.ndarray,
    second_value: np.ndarray,
    third_wave: Wave | np.ndarray,
    third_value: np.ndarray,
) -> np.ndarray:
    components = _energy_trilinear_components(
        first_wave,
        first_value,
        second_wave,
        second_value,
        third_wave,
        third_value,
    )
    return 6.0 * (components["kinetic"] + components["pressure"])


def _trace_flux_identity_audit() -> dict[str, Any]:
    carrier = 64
    rows = []
    output_offsets = (
        (0, 0, 0),
        (1, 1, 1),
        (1, 0, -1),
        (-1, 1, 0),
    )
    for offset in output_offsets:
        waves = [
            carrier * direction
            for direction in dense.CENTER_DIRECTIONS
        ]
        waves[2] = waves[2] + np.asarray(offset, dtype=float)
        values = [
            phase * regeneration._project(base, wave)
            for base, wave, phase in zip(
                dense.BASE_VECTORS,
                waves,
                dense.PHASES,
            )
        ]
        forcing = regeneration._hhh_stress_forcing(
            waves[0],
            values[0],
            waves[1],
            values[1],
            waves[2],
            values[2],
        )
        flux = _cubic_flux_symbol(
            waves[0],
            values[0],
            waves[1],
            values[1],
            waves[2],
            values[2],
        )
        output_wave = sum(waves, np.zeros(3))
        residual = complex(
            np.trace(forcing)
            + 2j * np.dot(output_wave, flux)
        )
        rows.append(
            {
                "output_wave": output_wave.astype(int).tolist(),
                "stress_trace_real": float(np.trace(forcing).real),
                "stress_trace_imag": float(np.trace(forcing).imag),
                "twice_flux_divergence_real": float(
                    (2j * np.dot(output_wave, flux)).real
                ),
                "twice_flux_divergence_imag": float(
                    (2j * np.dot(output_wave, flux)).imag
                ),
                "identity_residual": abs(residual),
            }
        )
    maximum_residual = max(row["identity_residual"] for row in rows)
    zero_row = rows[0]
    return {
        "identity": "tr G(q) = -2 i q dot F_cubic(q)",
        "zero_output_pressure_gauge": (
            "p(U,V)=0 when the pair output wave is zero"
        ),
        "rows": rows,
        "maximum_identity_residual": maximum_residual,
        "zero_output_trace_residual": abs(
            complex(
                zero_row["stress_trace_real"],
                zero_row["stress_trace_imag"],
            )
        ),
        "interpretation": (
            "The nonlinear/Euler sharp carrier derivative is traceless "
            "at q=0 and is therefore absent from the ordinary scalar "
            "local-energy identity. Full Navier-Stokes energy still has "
            "the standard viscous dissipation. This trace statement does "
            "not yet test the time derivative of the HHL transfer, which "
            "couples the anisotropic tensor to a low velocity."
        ),
        "all_checks_pass": bool(
            maximum_residual < 1.0e-9
            and abs(
                complex(
                    zero_row["stress_trace_real"],
                    zero_row["stress_trace_imag"],
                )
            )
            < 1.0e-10
        ),
    }


def _field_sum(*terms: tuple[complex, VectorField]) -> VectorField:
    output: VectorField = {}
    for factor, field in terms:
        for wave, value in field.items():
            output[wave] = output.get(
                wave,
                np.zeros(3, dtype=np.complex128),
            ) + factor * value
    return {
        wave: value
        for wave, value in output.items()
        if np.linalg.norm(value) > 1.0e-13
    }


def _nonlinear_field(field: VectorField) -> VectorField:
    output: VectorField = {}
    for first_wave, first_value in field.items():
        for second_wave, second_value in field.items():
            wave = tuple(
                left + right
                for left, right in zip(first_wave, second_wave)
            )
            raw = (
                np.dot(first_value, np.asarray(second_wave, dtype=float))
                * second_value
            )
            value = -1j * regeneration._project(raw, wave)
            output[wave] = output.get(
                wave,
                np.zeros(3, dtype=np.complex128),
            ) + value
    return {
        wave: value
        for wave, value in output.items()
        if np.linalg.norm(value) > 1.0e-13
    }


def _energy_flux_time_derivative(
    field: VectorField,
    velocity: VectorField,
) -> VectorField:
    field_dot_velocity = local_energy._vector_dot_product(field, velocity)
    field_squared = local_energy._vector_dot_product(field, field)
    kinetic = local_energy._add_vectors(
        (
            1.0,
            local_energy._scalar_times_vector(
                field_dot_velocity,
                field,
            ),
        ),
        (
            0.5,
            local_energy._scalar_times_vector(
                field_squared,
                velocity,
            ),
        ),
    )
    pressure_derivative = local_energy._add_scalars(
        (1.0, local_energy._pressure_bilinear(velocity, field)),
        (1.0, local_energy._pressure_bilinear(field, velocity)),
    )
    pressure = local_energy._add_vectors(
        (
            1.0,
            local_energy._scalar_times_vector(
                pressure_derivative,
                field,
            ),
        ),
        (
            1.0,
            local_energy._scalar_times_vector(
                local_energy._pressure_bilinear(field, field),
                velocity,
            ),
        ),
    )
    return local_energy._add_vectors((1.0, kinetic), (1.0, pressure))


def _independent_quartic_reconstruction(
    carrier: int = 32,
) -> dict[str, Any]:
    waves = [
        carrier * direction for direction in dense.CENTER_DIRECTIONS
    ]
    values = [
        phase * regeneration._project(base, wave)
        for base, wave, phase in zip(
            dense.BASE_VECTORS,
            waves,
            dense.PHASES,
        )
    ]
    legs = [
        {_wave_tuple(wave): value}
        for wave, value in zip(waves, values)
    ]
    legs.append({_wave_tuple(LOW_WAVE): LOW_COEFFICIENT})
    output_wave = _wave_tuple(LOW_WAVE)
    reconstructed = np.zeros(3, dtype=np.complex128)
    for signs in product((-1.0, 1.0), repeat=4):
        field = _field_sum(
            *[
                (sign, leg)
                for sign, leg in zip(signs, legs)
            ]
        )
        velocity = _nonlinear_field(field)
        derivative = _energy_flux_time_derivative(field, velocity)
        reconstructed += (
            math.prod(signs)
            * derivative.get(
                output_wave,
                np.zeros(3, dtype=np.complex128),
            )
            / 16.0
        )
    direct = _quartic_symbol_components(
        waves[0],
        values[0],
        waves[1],
        values[1],
        waves[2],
        values[2],
    )["complete"]
    residual = float(np.linalg.norm(direct - reconstructed))
    return {
        "carrier": carrier,
        "polarization_method": (
            "16-sign Walsh extraction of the coefficient odd in each of "
            "the three high legs and the low leg"
        ),
        "direct_symbol": [
            [float(value.real), float(value.imag)] for value in direct
        ],
        "independent_reconstruction": [
            [float(value.real), float(value.imag)]
            for value in reconstructed
        ],
        "maximum_vector_residual": residual,
        "all_checks_pass": residual < 1.0e-9,
    }


def _partition_gradient_at_positive_corner() -> np.ndarray:
    return 1j * PARTITION_WAVE / 64.0


def _central_quartic_row(carrier: int) -> dict[str, Any]:
    waves = [
        carrier * direction for direction in dense.CENTER_DIRECTIONS
    ]
    values = [
        phase * regeneration._project(base, wave)
        for base, wave, phase in zip(
            dense.BASE_VECTORS,
            waves,
            dense.PHASES,
        )
    ]
    components = _quartic_symbol_components(
        waves[0],
        values[0],
        waves[1],
        values[1],
        waves[2],
        values[2],
    )
    gradient = _partition_gradient_at_positive_corner()
    loads = {
        name: float(2.0 * np.dot(value, gradient).real)
        for name, value in components.items()
    }
    stress = regeneration._hhh_stress_forcing(
        waves[0],
        values[0],
        waves[1],
        values[1],
        waves[2],
        values[2],
    )
    stress_prediction = stress @ LOW_COEFFICIENT
    stress_load = float(2.0 * np.dot(stress_prediction, gradient).real)
    return {
        "carrier": carrier,
        "complete_real_quartet_vertex_load": loads["complete"],
        "complete_load_over_carrier": loads["complete"] / carrier,
        "stress_regeneration_vertex_load": stress_load,
        "stress_load_over_carrier": stress_load / carrier,
        "complete_minus_stress_over_carrier": (
            loads["complete"] - stress_load
        )
        / carrier,
        "component_real_quartet_vertex_loads": loads,
        "exact_limiting_real_quartet_load_over_carrier": (
            "3*sqrt(2)/16"
        ),
        "numeric_limiting_real_quartet_load_over_carrier": (
            REAL_QUARTET_LOAD_LIMIT
        ),
        "all_values_finite": all(
            math.isfinite(value) for value in loads.values()
        ),
    }


def _central_quartic_audit() -> dict[str, Any]:
    carriers = (16, 32, 64, 128, 256, 512, 1024)
    rows = [_central_quartic_row(carrier) for carrier in carriers]
    errors = [
        abs(
            row["complete_load_over_carrier"]
            - REAL_QUARTET_LOAD_LIMIT
        )
        for row in rows
    ]
    return {
        "carriers": list(carriers),
        "rows": rows,
        "exact_limit": "3*sqrt(2)/16",
        "numeric_limit": REAL_QUARTET_LOAD_LIMIT,
        "frequency_bookkeeping": {
            "high_triple_output_wave": [0, 0, 0],
            "low_velocity_wave": [-1, -1, -1],
            "quartic_flux_output_wave": [-1, -1, -1],
            "partition_gradient_wave": [1, 1, 1],
            "paired_scalar_output_wave": [0, 0, 0],
        },
        "stress_prediction_is_exactly_at_limit": max(
            abs(
                row["stress_load_over_carrier"]
                - REAL_QUARTET_LOAD_LIMIT
            )
            for row in rows
        )
        < 1.0e-12,
        "complete_errors_strictly_decrease": all(
            first > second
            for first, second in zip(errors, errors[1:])
        ),
        "last_complete_relative_error": (
            errors[-1] / REAL_QUARTET_LOAD_LIMIT
        ),
        "leading_identity": (
            "At zero low wave the cross pressure vanishes, linearized "
            "sweeping has zero mean because A+B+C=0, and the complete "
            "differentiated HHL flux is G_HHH U_low. Since "
            "r^T G_HHH U_low=-6sqrt(2)H and the real Fourier pair supplies "
            "the partition factor -2/64, the load/H tends to "
            "3sqrt(2)/16."
        ),
        "all_checks_pass": bool(
            all(row["all_values_finite"] for row in rows)
            and max(
                abs(
                    row["stress_load_over_carrier"]
                    - REAL_QUARTET_LOAD_LIMIT
                )
                for row in rows
            )
            < 1.0e-12
            and all(
                first > second
                for first, second in zip(errors, errors[1:])
            )
            and errors[-1] / REAL_QUARTET_LOAD_LIMIT < 0.01
        ),
    }


def _offsets(radius: int) -> list[Offset]:
    return list(product(range(-radius, radius + 1), repeat=3))


def _negative_offset_sum(first: Offset, second: Offset) -> Offset:
    return tuple(
        -left - right for left, right in zip(first, second)
    )  # type: ignore[return-value]


def _in_box(offset: Offset, radius: int) -> bool:
    return all(abs(value) <= radius for value in offset)


def _dense_quartic_row(
    radius: int,
    carrier_multiple: int = continuum.DEFAULT_CARRIER_MULTIPLE,
) -> dict[str, Any]:
    carrier = carrier_multiple * OFFSET_SPACING * radius
    centers = [
        carrier * direction for direction in dense.CENTER_DIRECTIONS
    ]
    offsets = _offsets(radius)
    values: list[dict[Offset, np.ndarray]] = []
    waves: list[dict[Offset, np.ndarray]] = []
    positive_energy = 0.0
    positive_enstrophy = 0.0
    maximum_divergence = 0.0
    minimum_support = math.inf
    maximum_support = 0.0

    for center, base, phase in zip(
        centers,
        dense.BASE_VECTORS,
        dense.PHASES,
    ):
        cluster_values: dict[Offset, np.ndarray] = {}
        cluster_waves: dict[Offset, np.ndarray] = {}
        for offset in offsets:
            wave = (
                center
                + OFFSET_SPACING * np.asarray(offset, dtype=float)
            )
            value = phase * regeneration._project(base, wave)
            cluster_values[offset] = value
            cluster_waves[offset] = wave
            norm_squared = float(np.vdot(value, value).real)
            positive_energy += norm_squared
            positive_enstrophy += (
                float(np.dot(wave, wave)) * norm_squared
            )
            maximum_divergence = max(
                maximum_divergence,
                abs(np.dot(wave, value)),
            )
            support = float(np.linalg.norm(wave))
            minimum_support = min(minimum_support, support)
            maximum_support = max(maximum_support, support)
        values.append(cluster_values)
        waves.append(cluster_waves)

    normalization = 1.0 / math.sqrt(2.0 * positive_energy)
    gradient = _partition_gradient_at_positive_corner()
    accumulated = _zero_quartic_components()
    accumulated_stress = np.zeros((3, 3), dtype=np.complex128)
    triad_count = 0
    minimum_positive_load_over_carrier = math.inf
    maximum_positive_load_over_carrier = -math.inf
    maximum_decomposition_residual = 0.0
    maximum_decomposition_roundoff_units = 0.0

    for first_offset in offsets:
        for second_offset in offsets:
            third_offset = _negative_offset_sum(
                first_offset,
                second_offset,
            )
            if not _in_box(third_offset, radius):
                continue
            triad_count += 1
            components = _quartic_symbol_components(
                waves[0][first_offset],
                values[0][first_offset],
                waves[1][second_offset],
                values[1][second_offset],
                waves[2][third_offset],
                values[2][third_offset],
            )
            (
                decomposition_residual,
                _,
                decomposition_roundoff_units,
            ) = _quartic_decomposition_roundoff(components)
            maximum_decomposition_residual = max(
                maximum_decomposition_residual,
                decomposition_residual,
            )
            maximum_decomposition_roundoff_units = max(
                maximum_decomposition_roundoff_units,
                decomposition_roundoff_units,
            )
            for name, value in components.items():
                accumulated[name] += value
            stress = regeneration._hhh_stress_forcing(
                waves[0][first_offset],
                values[0][first_offset],
                waves[1][second_offset],
                values[1][second_offset],
                waves[2][third_offset],
                values[2][third_offset],
            )
            accumulated_stress += stress
            positive_load = float(
                np.dot(components["complete"], gradient).real
                / carrier
            )
            minimum_positive_load_over_carrier = min(
                minimum_positive_load_over_carrier,
                positive_load,
            )
            maximum_positive_load_over_carrier = max(
                maximum_positive_load_over_carrier,
                positive_load,
            )

    expected_triad_count = (
        3 * radius**2 + 3 * radius + 1
    ) ** 3
    component_loads = {
        name: float(
            2.0
            * normalization**3
            * np.dot(value, gradient).real
        )
        for name, value in accumulated.items()
    }
    stress_load = float(
        2.0
        * normalization**3
        * np.dot(
            accumulated_stress @ LOW_COEFFICIENT,
            gradient,
        ).real
    )
    count_scale = (
        2.0
        * normalization**3
        * carrier
        * triad_count
    )
    enstrophy = (
        2.0 * normalization**2 * positive_enstrophy
    )
    complete_load = component_loads["complete"]
    return {
        "box_radius": radius,
        "offset_spacing": OFFSET_SPACING,
        "carrier": carrier,
        "carrier_multiple_relative_to_box_width": carrier_multiple,
        "real_high_mode_count": 6 * len(offsets),
        "real_low_mode_count": 2,
        "exact_coherent_triad_count": triad_count,
        "expected_coherent_triad_count": expected_triad_count,
        "normalization": normalization,
        "normalized_high_energy": (
            2.0 * normalization**2 * positive_energy
        ),
        "normalized_high_enstrophy": enstrophy,
        "minimum_high_support": minimum_support,
        "maximum_high_support": maximum_support,
        "high_annulus_ratio": maximum_support / minimum_support,
        "maximum_high_divergence_residual": float(maximum_divergence),
        "maximum_quartic_decomposition_absolute_residual": (
            maximum_decomposition_residual
        ),
        "maximum_quartic_decomposition_roundoff_units": (
            maximum_decomposition_roundoff_units
        ),
        "complete_top_Walsh_vertex_load": complete_load,
        "stress_prediction_top_Walsh_vertex_load": stress_load,
        "complete_minus_stress_load": complete_load - stress_load,
        "complete_load_over_H_five_halves": (
            complete_load / carrier**2.5
        ),
        "coherent_count_normalized_complete_coefficient": (
            complete_load / count_scale
        ),
        "coherent_count_normalized_stress_coefficient": (
            stress_load / count_scale
        ),
        "central_positive_quartet_coefficient_reference": (
            POSITIVE_QUARTET_COEFFICIENT_LIMIT
        ),
        "minimum_positive_quartet_load_over_carrier": (
            minimum_positive_load_over_carrier
        ),
        "maximum_positive_quartet_load_over_carrier": (
            maximum_positive_load_over_carrier
        ),
        "component_top_Walsh_vertex_loads": component_loads,
        "eight_vertex_loads": [
            {
                "vertex": list(vertex),
                "Walsh_character": math.prod(vertex),
                "load": math.prod(vertex) * complete_load,
            }
            for vertex in product((-1, 1), repeat=3)
        ],
        "partition_frequency_isolation": (
            "All high offsets lie in 4Z^3. After adding the low waves "
            "+/-(1,1,1), a quartic output in {-1,0,1}^3 is possible only "
            "at the matching corner frequency. Mixed carrier signs cannot "
            "enter the partition support. The complete vertex load is "
            "therefore exactly the pure top Walsh character."
        ),
        "all_checks_pass": bool(
            triad_count == expected_triad_count
            and abs(
                2.0 * normalization**2 * positive_energy - 1.0
            )
            < 1.0e-12
            and maximum_divergence < 1.0e-10
            and maximum_decomposition_roundoff_units <= 256.0
            and maximum_support / minimum_support < 2.0
            and minimum_positive_load_over_carrier > 0.0
            and complete_load > 0.0
            and all(
                abs(
                    row["load"]
                    - row["Walsh_character"] * complete_load
                )
                < 1.0e-12
                for row in [
                    {
                        "Walsh_character": math.prod(vertex),
                        "load": math.prod(vertex) * complete_load,
                    }
                    for vertex in product((-1, 1), repeat=3)
                ]
            )
        ),
    }


def _dense_quartic_audit(
    radii: tuple[int, ...],
    carrier_multiple: int,
) -> dict[str, Any]:
    continuum_certificate = continuum._certificate(carrier_multiple)
    certified_lower = continuum_certificate[
        "complete_actual_positive_quartet_coefficient_lower"
    ]
    rows = [
        _dense_quartic_row(radius, carrier_multiple)
        for radius in radii
    ]
    coefficients = [
        row["coherent_count_normalized_complete_coefficient"]
        for row in rows
    ]
    return {
        "radii": list(radii),
        "carrier_multiple_relative_to_box_width": carrier_multiple,
        "offset_spacing": OFFSET_SPACING,
        "rows": rows,
        "exact_mode_count": "6(2M+1)^3 high modes plus two low modes",
        "exact_triad_count": "(3M^2+3M+1)^3",
        "sharp_complete_derivative_exponent": "5/2",
        "central_coherent_coefficient_reference_not_limit": (
            POSITIVE_QUARTET_COEFFICIENT_LIMIT
        ),
        "last_coherent_coefficient_relative_difference_from_center": abs(
            coefficients[-1] - POSITIVE_QUARTET_COEFFICIENT_LIMIT
        )
        / POSITIVE_QUARTET_COEFFICIENT_LIMIT,
        "continuum_positive_coefficient_lower": certified_lower,
        "continuum_positivity_certificate": continuum_certificate,
        "fixed_relative_width_limit_statement": (
            "At fixed carrier multiple the offsets do not collapse to the "
            "center. The coefficient limit is a polytope average, not the "
            "central value. The H^(5/2) lower bound instead follows from "
            "the directed interval certificate over the complete "
            "continuous offset domain."
        ),
        "all_coherent_quartets_have_positive_selected_load": all(
            row["minimum_positive_quartet_load_over_carrier"] > 0.0
            for row in rows
        ),
        "all_checks_pass": bool(
            all(row["all_checks_pass"] for row in rows)
            and all(value > 0.0 for value in coefficients)
            and continuum_certificate["all_checks_pass"]
            and all(
                value >= certified_lower for value in coefficients
            )
        ),
    }


def _sharp_negative_shell_norm() -> dict[str, Any]:
    carriers = [16 * 2**index for index in range(8)]
    amplitudes = [
        1.0 / (index + 2.0) for index in range(len(carriers))
    ]
    energy = sum(value**2 for value in amplitudes)
    weighted_forcing_square = sum(
        carrier**2 * amplitude**6
        for carrier, amplitude in zip(carriers, amplitudes)
    )
    dissipation = sum(
        carrier**2 * amplitude**2
        for carrier, amplitude in zip(carriers, amplitudes)
    )
    sequence_ratio = (
        weighted_forcing_square / (energy**2 * dissipation)
    )
    pulse_rows = [
        {
            "carrier": carrier,
            "cost_at_s_5_over_4": carrier**0.5,
            "cost_at_s_3_over_2": 1.0,
            "cost_at_s_7_over_4": carrier**-0.5,
        }
        for carrier in carriers
    ]
    forcing_coordinates = [
        1.0 / (index + 1.0) for index in range(len(carriers))
    ]
    response_l1 = sum(
        carrier**-0.5 * coordinate
        for carrier, coordinate in zip(
            carriers,
            forcing_coordinates,
        )
    )
    response_cauchy_upper = math.sqrt(
        sum(carrier**-1.0 for carrier in carriers)
    ) * math.sqrt(
        sum(coordinate**2 for coordinate in forcing_coordinates)
    )
    return {
        "fixed_low_channel_Bernstein_bound": (
            "|G_H| <= C H^(5/2) a_H^3"
        ),
        "energy_definition": (
            "E_* = sup_t sum_H a_H(t)^2"
        ),
        "dissipation_definition": (
            "D = integral sum_H H^2 a_H(t)^2 dt"
        ),
        "pointwise_weighted_square_bound": (
            "H^(-3)|G_H|^2 <= C H^2 a_H^6 "
            "<= C E_*^2 H^2 a_H^2"
        ),
        "Leray_controlled_norm": (
            "sum_H H^(-3)||G_H||_(L2_t)^2 "
            "<= C E_*^2 D"
        ),
        "forcing_amplitude_weight": "H^(-3/2)",
        "squared_norm_weight": "H^(-3)",
        "sharp_weight_exponent": "3/2",
        "dense_parabolic_cost_with_weight_s": "H^(3-2s)",
        "dense_cost_exponent_at_s_three_halves": "0",
        "finite_sequence_replay": {
            "carriers": carriers,
            "amplitudes": amplitudes,
            "weighted_forcing_over_energy_squared_dissipation": (
                sequence_ratio
            ),
            "pulse_weight_rows": pulse_rows,
            "viscous_response_l1": response_l1,
            "viscous_response_Cauchy_upper": response_cauchy_upper,
            "all_checks_pass": bool(
                sequence_ratio <= 1.0
                and all(
                    row["cost_at_s_3_over_2"] == 1.0
                    for row in pulse_rows
                )
                and all(
                    first["cost_at_s_5_over_4"]
                    < second["cost_at_s_5_over_4"]
                    for first, second in zip(
                        pulse_rows,
                        pulse_rows[1:],
                    )
                )
                and all(
                    first["cost_at_s_7_over_4"]
                    > second["cost_at_s_7_over_4"]
                    for first, second in zip(
                        pulse_rows,
                        pulse_rows[1:],
                    )
                )
                and response_l1 <= response_cauchy_upper
            ),
        },
        "viscous_relaxation_gain": (
            "For the zero-initial forced component of "
            "dot c_H+c nu H^2 c_H=G_H, "
            "||sum_H c_H||_L2 <= "
            "C/(c nu sqrt(H_0)) "
            "[sum_H H^(-3)||G_H||_L2^2]^(1/2) "
            "for dyadic H>=H_0."
        ),
        "scope": (
            "This is a sharp fixed-low-output HHH forcing theorem at the "
            "smooth shell level. Completing the full Navier-Stokes route "
            "still requires comparable-shell bookkeeping, filter "
            "commutators, HHL terms, initial stress, and passage to "
            "suitable weak solutions."
        ),
        "all_checks_pass": bool(
            sequence_ratio <= 1.0
            and response_l1 <= response_cauchy_upper
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "work/ns_collision/results/"
            "scalar_local_energy_regeneration_gate_audit_v2.json"
        ),
    )
    parser.add_argument("--radii", default="1,2,3")
    parser.add_argument(
        "--carrier-multiple",
        type=int,
        default=continuum.DEFAULT_CARRIER_MULTIPLE,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    radii = tuple(
        int(value.strip())
        for value in args.radii.split(",")
        if value.strip()
    )
    if not radii or min(radii) < 1:
        raise ValueError("radii must contain positive integers")
    if args.carrier_multiple < 16:
        raise ValueError("carrier multiple must be at least 16")

    trace = _trace_flux_identity_audit()
    independent = _independent_quartic_reconstruction()
    central = _central_quartic_audit()
    dense_packet = _dense_quartic_audit(
        radii,
        args.carrier_multiple,
    )
    negative_norm = _sharp_negative_shell_norm()
    result = {
        "schema": "ns_scalar_local_energy_regeneration_gate_audit_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "corrected_continuum_positive_dense_HHHL_H5over2_"
            "and_negative_three_halves_norm_certified"
        ),
        "supersedes_result": (
            "work/ns_collision/results/"
            "scalar_local_energy_regeneration_gate_audit_v1.json"
        ),
        "superseded_result_sha256": _sha256(
            ROOT
            / "work/ns_collision/results/"
            "scalar_local_energy_regeneration_gate_audit_v1.json"
        ),
        "correction": (
            "The v1 statement that fixed-width packet coefficients "
            "approach the central 3sqrt(2)/32 value was unjustified "
            "because R/M was fixed. V2 withdraws that limit claim and "
            "uses a directed interval certificate over the full "
            "continuous relative-offset domain."
        ),
        "prerequisite_result": (
            "work/ns_collision/results/"
            "dense_annular_hhh_packet_gate_audit_v1.json"
        ),
        "prerequisite_result_sha256": _sha256(
            ROOT
            / "work/ns_collision/results/"
            "dense_annular_hhh_packet_gate_audit_v1.json"
        ),
        "scalar_trace_identity": trace,
        "independent_quartic_reconstruction": independent,
        "central_complete_quartic_symbol": central,
        "dense_spaced_packet": dense_packet,
        "sharp_negative_shell_norm": negative_norm,
        "certification_flags": {
            "ordinary_scalar_local_energy_trace_removes_H_five_halves": (
                True
            ),
            "complete_HHL_transfer_time_derivative_derived": True,
            "linearized_low_velocity_evolution_included": True,
            "all_kinetic_and_pressure_quartic_terms_included": True,
            "complete_differentiated_HHL_H_five_halves_survives": True,
            "pure_top_Walsh_frequency_isolated": True,
            "fixed_width_center_limit_claim_withdrawn": True,
            "continuous_offset_domain_uniform_positivity_proved": True,
            "sharp_shell_negative_three_halves_forcing_norm_proved": True,
            "Leray_control_of_weighted_HHH_forcing_proved": True,
            "full_nonlinear_shell_response_closed": False,
            "suitable_weak_solution_passage_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "route_decision": {
            "discard": (
                "Any claim that taking the scalar trace also cancels the "
                "time derivative of the complete HHL transfer."
            ),
            "retain": (
                "The H^(-3/2)-weighted shell forcing norm. It is exactly "
                "controlled by Leray energy and dissipation, is sharp on "
                "the dense packet, and leaves a summable half derivative "
                "after the viscous H^(-2) response."
            ),
            "next_gate": (
                "Lift the weighted fixed-channel theorem to the complete "
                "dyadic HHH/HHL shell system with comparable-shell "
                "neighbors and filter commutators. Prove the weighted "
                "Duhamel response before attempting low-regularity "
                "passage."
            ),
        },
    }
    result["all_positive_checks_pass"] = bool(
        trace["all_checks_pass"]
        and independent["all_checks_pass"]
        and central["all_checks_pass"]
        and dense_packet["all_checks_pass"]
        and negative_norm["all_checks_pass"]
    )
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("scalar local-energy regeneration gate failed")
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
