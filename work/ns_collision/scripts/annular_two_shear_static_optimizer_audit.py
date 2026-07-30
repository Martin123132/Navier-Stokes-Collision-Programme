"""Audit the static optimizer for the modified two-shear annular witness."""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sympy as sp

from annular_rho_zero_continuum_convolution_quadrature import (
    _lower_process_priority,
)
from annular_two_shear_square_gate_audit import (
    XY_SHEAR,
    YZ_SHEAR,
    _modified_finite_packet,
)
from compatible_eight_cell_cubic_graph_audit import (
    VERTICES,
    _cubic_energy,
)
from compatible_edge_annular_escape_audit import (
    _delta_weights,
)
from cross_shell_modulated_wave_gate_audit import (
    _add_vectors,
    _energy_flux,
    _field_sum,
    _pressure_bilinear,
    _scalar_times_vector,
)
from primitive_hhl_chain_hardy_envelope_audit import (
    _translated_vertex_fisher,
    _translated_vertex_load,
)
from separable_annular_pressure_schur_no_go_audit import (
    TRANSLATION,
    _high_field,
    _mixed_difference_fisher,
    _shift_slices,
    _vertex_weight_float,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_two_shear_static_optimizer_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "annular_two_shear_square_gate_audit_v1.json"
    ): "7aca1d4c57ce970db5872979449368e65fb6add48c4ccd39b8f875cc1669f923",
    (
        "work/ns_collision/results/"
        "annular_two_shear_full_c1_port_audit_v1.json"
    ): "af0039698cdbd5442be629b23ea259e97556c978a0f0d91e94e9e0d658b1f32f",
    (
        "work/ns_collision/results/"
        "compatible_edge_annular_escape_audit_v1.json"
    ): "fffa314fc9fa516dc0c8f6ac010392d438845912f6d4bc2d16cc1f2dc02b83e0",
    (
        "work/ns_collision/results/"
        "deficit_retaining_annular_restart_gate_audit_v1.json"
    ): "2f32255887eb18ec0aa22dadfacf681b930434e73f0c457041d65a66e8c04e6d",
}
ALGORITHM_REVISION = "annular-two-shear-static-optimizer-v1"
DEFAULT_SIZES = (3, 5, 9, 13, 17, 25, 33, 49)
PLUS_VERTEX = (1, 1, 1)
DELTA_CUBIC_ENERGY = Fraction(75, 256)
Wave = tuple[int, int, int]
FloatField = dict[Wave, np.ndarray]
SymVector = tuple[sp.Expr, sp.Expr, sp.Expr]
SymScalarField = dict[Wave, sp.Expr]
SymVectorField = dict[Wave, SymVector]


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


def _vertex_label(vertex: Wave) -> str:
    return "".join("+" if value == 1 else "-" for value in vertex)


def _sym_text(value: sp.Expr) -> str:
    return sp.sstr(sp.simplify(value))


def _add_wave(first: Wave, second: Wave) -> Wave:
    return tuple(first[index] + second[index] for index in range(3))


def _negate_wave(wave: Wave) -> Wave:
    return tuple(-value for value in wave)


def _sym_zero_vector() -> SymVector:
    return (sp.S.Zero, sp.S.Zero, sp.S.Zero)


def _sym_vector_add(first: SymVector, second: SymVector) -> SymVector:
    return tuple(
        sp.expand(first[index] + second[index]) for index in range(3)
    )


def _sym_vector_scale(value: sp.Expr, vector: SymVector) -> SymVector:
    return tuple(sp.expand(value * component) for component in vector)


def _sym_vector_dot(first: SymVector, second: SymVector) -> sp.Expr:
    return sp.expand(
        sum(first[index] * second[index] for index in range(3))
    )


def _sym_add_vector_fields(
    *terms: tuple[sp.Expr, SymVectorField],
) -> SymVectorField:
    output: SymVectorField = {}
    for factor, field in terms:
        for wave, vector in field.items():
            output[wave] = _sym_vector_add(
                output.get(wave, _sym_zero_vector()),
                _sym_vector_scale(factor, vector),
            )
    return {
        wave: tuple(sp.simplify(component) for component in vector)
        for wave, vector in output.items()
        if any(sp.simplify(component) != 0 for component in vector)
    }


def _sym_vector_dot_product(
    first: SymVectorField,
    second: SymVectorField,
) -> SymScalarField:
    output: SymScalarField = {}
    for first_wave, first_value in first.items():
        for second_wave, second_value in second.items():
            wave = _add_wave(first_wave, second_wave)
            output[wave] = output.get(wave, sp.S.Zero) + _sym_vector_dot(
                first_value,
                second_value,
            )
    return {
        wave: sp.simplify(value)
        for wave, value in output.items()
        if sp.simplify(value) != 0
    }


def _sym_scalar_times_vector(
    scalar: SymScalarField,
    vector: SymVectorField,
) -> SymVectorField:
    output: SymVectorField = {}
    for scalar_wave, scalar_value in scalar.items():
        for vector_wave, vector_value in vector.items():
            wave = _add_wave(scalar_wave, vector_wave)
            output[wave] = _sym_vector_add(
                output.get(wave, _sym_zero_vector()),
                _sym_vector_scale(scalar_value, vector_value),
            )
    return {
        wave: tuple(sp.simplify(component) for component in value)
        for wave, value in output.items()
        if any(sp.simplify(component) != 0 for component in value)
    }


def _sym_pressure_bilinear(
    first: SymVectorField,
    second: SymVectorField,
) -> SymScalarField:
    output: SymScalarField = {}
    for first_wave, first_value in first.items():
        for second_wave, second_value in second.items():
            wave = _add_wave(first_wave, second_wave)
            norm_squared = sum(component * component for component in wave)
            if norm_squared == 0:
                continue
            wave_vector = tuple(sp.Integer(component) for component in wave)
            value = -(
                _sym_vector_dot(wave_vector, first_value)
                * _sym_vector_dot(wave_vector, second_value)
                / norm_squared
            )
            output[wave] = output.get(wave, sp.S.Zero) + value
    return {
        wave: sp.simplify(value)
        for wave, value in output.items()
        if sp.simplify(value) != 0
    }


def _sym_weight(wave: Wave, vertex: Wave) -> sp.Rational:
    if any(abs(component) > 1 for component in wave):
        return sp.S.Zero
    value = sp.S.One
    for index, component in enumerate(wave):
        value *= (
            sp.Rational(1, 2)
            if component == 0
            else sp.Rational(vertex[index], 4)
        )
    return value


def _sym_load(flux: SymVectorField, vertex: Wave) -> sp.Expr:
    value = sp.S.Zero
    for wave, coefficient in flux.items():
        gradient_wave = _negate_wave(wave)
        weight = _sym_weight(gradient_wave, vertex)
        if weight == 0:
            continue
        gradient = tuple(
            sp.I * gradient_wave[index] * weight for index in range(3)
        )
        value += _sym_vector_dot(coefficient, gradient)
    return sp.simplify(value)


def _sym_fisher(field: SymVectorField, vertex: Wave) -> sp.Expr:
    value = sp.S.Zero
    for first_wave, first_value in field.items():
        for second_wave, second_value in field.items():
            difference = tuple(
                second_wave[index] - first_wave[index]
                for index in range(3)
            )
            weight = _sym_weight(difference, vertex)
            if weight == 0:
                continue
            wave_dot = sum(
                first_wave[index] * second_wave[index]
                for index in range(3)
            )
            conjugate_second = tuple(
                sp.conjugate(component) for component in second_value
            )
            value += (
                wave_dot
                * weight
                * _sym_vector_dot(first_value, conjugate_second)
            )
    return sp.simplify(value)


def _sym_l2_mass(field: SymVectorField) -> sp.Expr:
    return sp.simplify(
        sum(
            _sym_vector_dot(
                value,
                tuple(sp.conjugate(component) for component in value),
            )
            for value in field.values()
        )
    )


def _sym_shear(wave: Wave, scaled_direction: Wave) -> SymVectorField:
    direction = tuple(
        sp.Integer(component) / sp.sqrt(2)
        for component in scaled_direction
    )
    positive = tuple(-sp.I * component for component in direction)
    negative = tuple(sp.I * component for component in direction)
    return {
        wave: positive,
        _negate_wave(wave): negative,
    }


def _symbolic_low_field_certificate() -> dict[str, Any]:
    yz = _sym_shear(*YZ_SHEAR)
    xy = _sym_shear(*XY_SHEAR)
    combined = _sym_add_vector_fields((sp.S.One, yz), (sp.S.One, xy))

    low_fields = {"yz": yz, "xy": xy, "combined": combined}
    components: dict[str, dict[str, SymVectorField]] = {}
    for label, field in low_fields.items():
        kinetic = _sym_scalar_times_vector(
            _sym_vector_dot_product(field, field),
            field,
        )
        pressure = _sym_scalar_times_vector(
            _sym_pressure_bilinear(field, field),
            field,
        )
        components[label] = {
            "kinetic": _sym_add_vector_fields(
                (sp.Rational(1, 2), kinetic)
            ),
            "pressure": pressure,
            "complete": _sym_add_vector_fields(
                (sp.Rational(1, 2), kinetic),
                (sp.S.One, pressure),
            ),
        }

    by_vertex: dict[str, Any] = {}
    for vertex in VERTICES:
        vertex_label = _vertex_label(vertex)
        fisher = {
            label: _sym_fisher(field, vertex)
            for label, field in low_fields.items()
        }
        loads = {
            component: _sym_load(
                components["combined"][component],
                vertex,
            )
            for component in ("kinetic", "pressure", "complete")
        }
        by_vertex[vertex_label] = {
            "yz_Fisher": _sym_text(fisher["yz"]),
            "xy_Fisher": _sym_text(fisher["xy"]),
            "combined_Fisher": _sym_text(fisher["combined"]),
            "cross_Fisher": _sym_text(
                fisher["combined"] - fisher["yz"] - fisher["xy"]
            ),
            "kinetic_flux_load": _sym_text(loads["kinetic"]),
            "pressure_flux_load": _sym_text(loads["pressure"]),
            "complete_flux_load": _sym_text(loads["complete"]),
        }

    plus_fisher = _sym_fisher(combined, PLUS_VERTEX)
    plus_kinetic = _sym_load(
        components["combined"]["kinetic"], PLUS_VERTEX
    )
    plus_pressure = _sym_load(
        components["combined"]["pressure"], PLUS_VERTEX
    )
    plus_complete = _sym_load(
        components["combined"]["complete"], PLUS_VERTEX
    )
    yz_complete = _sym_load(components["yz"]["complete"], PLUS_VERTEX)
    xy_complete = _sym_load(components["xy"]["complete"], PLUS_VERTEX)
    l2_mass = _sym_l2_mass(combined)
    expected_fisher = sp.Rational(17, 16)
    expected_kinetic = -sp.sqrt(2) / 16
    expected_pressure = -sp.sqrt(2) / 48
    expected_complete = -sp.sqrt(2) / 12
    checks = {
        "each_shear_has_zero_complete_flux": bool(
            sp.simplify(yz_complete) == 0
            and sp.simplify(xy_complete) == 0
        ),
        "combined_plus_Fisher_exact": bool(
            sp.simplify(plus_fisher - expected_fisher) == 0
        ),
        "combined_plus_kinetic_load_exact": bool(
            sp.simplify(plus_kinetic - expected_kinetic) == 0
        ),
        "combined_plus_pressure_load_exact": bool(
            sp.simplify(plus_pressure - expected_pressure) == 0
        ),
        "combined_plus_complete_load_exact": bool(
            sp.simplify(plus_complete - expected_complete) == 0
        ),
        "complete_is_kinetic_plus_pressure": bool(
            sp.simplify(plus_complete - plus_kinetic - plus_pressure) == 0
        ),
        "combined_L2_mass_exact": bool(
            sp.simplify(l2_mass - 4) == 0
        ),
    }
    return {
        "Fourier_convention": (
            "Uhat(+ell)=-i*d and Uhat(-ell)=+i*d, with |d|=1"
        ),
        "combined_low_field": "U_*=U_yz+U_xy",
        "combined_L2_mass": _sym_text(l2_mass),
        "by_vertex": by_vertex,
        "plus_vertex_exact": {
            "weighted_Fisher_mass": _sym_text(plus_fisher),
            "kinetic_flux_load": _sym_text(plus_kinetic),
            "pressure_flux_load": _sym_text(plus_pressure),
            "complete_flux_load": _sym_text(plus_complete),
            "favorable_complete_cubic_coefficient": (
                _sym_text(-plus_complete)
            ),
            "favorable_pressure_cubic_coefficient": (
                _sym_text(-plus_pressure)
            ),
        },
        "checks": checks,
        "all_symbolic_checks_pass": all(checks.values()),
    }


def _float_shear(wave: Wave, scaled_direction: Wave) -> FloatField:
    direction = np.asarray(scaled_direction, dtype=float) / math.sqrt(2.0)
    return {
        wave: -1j * direction,
        _negate_wave(wave): 1j * direction,
    }


def _float_low_field() -> FloatField:
    field: FloatField = {}
    for shear in (
        _float_shear(*YZ_SHEAR),
        _float_shear(*XY_SHEAR),
    ):
        for wave, value in shear.items():
            field[wave] = field.get(
                wave, np.zeros(3, dtype=np.complex128)
            ) + value
    return field


def _resonant_loads_for_shear(
    waves: np.ndarray,
    velocity: np.ndarray,
    low_wave_tuple: Wave,
    scaled_direction: Wave,
) -> dict[str, float]:
    shape = waves.shape[:3]
    low_wave_base = np.asarray(low_wave_tuple, dtype=int)
    low_direction = (
        np.asarray(scaled_direction, dtype=float) / math.sqrt(2.0)
    )
    loads = {
        "kinetic": 0.0j,
        "pressure_high_high": 0.0j,
        "pressure_cross": 0.0j,
    }
    for sign in (1, -1):
        low_wave = sign * low_wave_base
        low_value = -sign * 1j * low_direction
        for output_wave in product((-1, 0, 1), repeat=3):
            if output_wave == (0, 0, 0):
                continue
            output = np.asarray(output_wave, dtype=int)
            difference_array = output - low_wave
            difference = tuple(int(value) for value in difference_array)
            if any(
                abs(difference[index]) >= shape[index]
                for index in range(3)
            ):
                continue
            first_slice, second_slice = _shift_slices(difference, shape)
            first_wave = waves[first_slice]
            second_wave = waves[second_slice]
            first_velocity = velocity[first_slice]
            second_velocity = velocity[second_slice]
            gradient = (
                -1j
                * output.astype(float)
                * _vertex_weight_float(output_wave)
            )

            velocity_dot = np.sum(
                first_velocity * second_velocity, axis=-1
            )
            kinetic_vector = float(np.sum(velocity_dot)) * low_value
            kinetic_vector += np.sum(
                np.sum(first_velocity * low_value, axis=-1)[..., None]
                * second_velocity
                + np.sum(second_velocity * low_value, axis=-1)[..., None]
                * first_velocity,
                axis=(0, 1, 2),
            )
            loads["kinetic"] += np.dot(kinetic_vector, gradient)

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
                loads["pressure_high_high"] += (
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
            loads["pressure_cross"] += np.dot(cross_vector, gradient)

    loads["pressure"] = (
        loads["pressure_high_high"] + loads["pressure_cross"]
    )
    loads["complete"] = loads["kinetic"] + loads["pressure"]
    return {
        key: float(value.real) for key, value in loads.items()
    } | {
        "maximum_imaginary_residual": max(
            abs(value.imag) for value in loads.values()
        )
    }


def _combined_resonant_loads(
    waves: np.ndarray,
    velocity: np.ndarray,
) -> dict[str, float]:
    rows = [
        _resonant_loads_for_shear(
            waves,
            velocity,
            *shear,
        )
        for shear in (YZ_SHEAR, XY_SHEAR)
    ]
    keys = (
        "kinetic",
        "pressure_high_high",
        "pressure_cross",
        "pressure",
        "complete",
    )
    return {
        key: sum(row[key] for row in rows) for key in keys
    } | {
        "maximum_imaginary_residual": max(
            row["maximum_imaginary_residual"] for row in rows
        )
    }


def _pressure_flux(field: FloatField) -> FloatField:
    return _scalar_times_vector(
        _pressure_bilinear(field, field),
        field,
    )


def _full_field_support_replay(size: int = 3) -> dict[str, Any]:
    waves, velocity, parity = _modified_finite_packet(size)
    high = _high_field(waves, velocity)
    low = _float_low_field()
    hhl = _combined_resonant_loads(waves, velocity)
    high_fisher = _mixed_difference_fisher(waves, velocity, parity)
    low_fisher = float(
        _translated_vertex_fisher(
            low,
            1,
            PLUS_VERTEX,
            TRANSLATION,
        ).real
    )
    low_complete = float(
        _translated_vertex_load(
            _energy_flux(low),
            1,
            PLUS_VERTEX,
            TRANSLATION,
        ).real
    )
    low_pressure = float(
        _translated_vertex_load(
            _pressure_flux(low),
            1,
            PLUS_VERTEX,
            TRANSLATION,
        ).real
    )

    amplitudes = (-2.0, -1.0, 0.0, 1.0, 2.0)
    rows = []
    for amplitude in amplitudes:
        field = _field_sum(high, low, amplitude)
        complete_load = float(
            _translated_vertex_load(
                _energy_flux(field),
                1,
                PLUS_VERTEX,
                TRANSLATION,
            ).real
        )
        pressure_load = float(
            _translated_vertex_load(
                _pressure_flux(field),
                1,
                PLUS_VERTEX,
                TRANSLATION,
            ).real
        )
        fisher = float(
            _translated_vertex_fisher(
                field,
                1,
                PLUS_VERTEX,
                TRANSLATION,
            ).real
        )
        expected_complete = (
            amplitude * hhl["complete"]
            + amplitude**3 * low_complete
        )
        expected_pressure = (
            amplitude * hhl["pressure"]
            + amplitude**3 * low_pressure
        )
        expected_fisher = high_fisher + amplitude**2 * low_fisher
        rows.append(
            {
                "low_field_factor": amplitude,
                "complete_load": complete_load,
                "expected_complete_load": expected_complete,
                "complete_residual": abs(
                    complete_load - expected_complete
                ),
                "pressure_load": pressure_load,
                "expected_pressure_load": expected_pressure,
                "pressure_residual": abs(
                    pressure_load - expected_pressure
                ),
                "weighted_Fisher": fisher,
                "expected_weighted_Fisher": expected_fisher,
                "Fisher_residual": abs(fisher - expected_fisher),
            }
        )

    maximum_complete_residual = max(
        row["complete_residual"] for row in rows
    )
    maximum_pressure_residual = max(
        row["pressure_residual"] for row in rows
    )
    maximum_fisher_residual = max(
        row["Fisher_residual"] for row in rows
    )
    return {
        "replay_size": size,
        "positive_high_first_coordinate_interval": (
            f"[{2 * size},{3 * size - 1}]"
        ),
        "low_first_coordinate_set": "[-1,0,1]",
        "support_gaps": {
            "HHH": (
                "Every mixed-sign HHH output has |k_x|>=N+1>1; "
                "same-sign outputs are farther away."
            ),
            "HLL": "|k_x|>=2N-2>1 for N>=3.",
            "high_low_Fisher": "|Delta k_x|>=2N-1>1 for N>=3.",
        },
        "unit_low_complete_load": low_complete,
        "unit_low_pressure_load": low_pressure,
        "unit_low_weighted_Fisher": low_fisher,
        "complete_HHL_components": hhl,
        "high_weighted_Fisher": high_fisher,
        "amplitude_rows": rows,
        "maximum_complete_polynomial_residual": (
            maximum_complete_residual
        ),
        "maximum_pressure_polynomial_residual": (
            maximum_pressure_residual
        ),
        "maximum_Fisher_polynomial_residual": maximum_fisher_residual,
        "all_support_replay_checks_pass": bool(
            hhl["complete"] < 0.0
            and hhl["pressure"] < 0.0
            and low_complete < 0.0
            and low_pressure < 0.0
            and abs(low_fisher - 17.0 / 16.0) < 2.0e-15
            and maximum_complete_residual < 2.0e-12
            and maximum_pressure_residual < 2.0e-12
            and maximum_fisher_residual < 2.0e-12
        ),
    }


def _coefficient_penalty_certificate() -> dict[str, Any]:
    energies = {
        _vertex_label(vertex): _cubic_energy(_delta_weights(vertex))
        for vertex in VERTICES
    }
    return {
        "formula": "Q(w)=sum_j mean[H_j D_j^2]",
        "delta_vertex_energies": {
            label: (
                str(value.numerator)
                if value.denominator == 1
                else f"{value.numerator}/{value.denominator}"
            )
            for label, value in energies.items()
        },
        "common_exact_value": "75/256",
        "objective_penalty": "-(nu/16)t^3 Q(delta_+++)",
        "all_penalty_checks_pass": all(
            value == DELTA_CUBIC_ENERGY for value in energies.values()
        ),
    }


def _coefficient_scale_optimum(
    margin: float,
    viscosity: float,
) -> dict[str, float | bool]:
    q = float(DELTA_CUBIC_ENERGY)
    if margin <= 0.0:
        return {
            "positive_margin": False,
            "optimal_coefficient_scale": 0.0,
            "optimized_objective": 0.0,
        }
    scale = math.sqrt(16.0 * margin / (3.0 * viscosity * q))
    maximum = 2.0 * margin * scale / 3.0
    return {
        "positive_margin": True,
        "optimal_coefficient_scale": scale,
        "optimized_objective": maximum,
    }


def _finite_row(size: int, viscosity: float) -> dict[str, Any]:
    waves, velocity, parity = _modified_finite_packet(size)
    hhl = _combined_resonant_loads(waves, velocity)
    high_fisher = _mixed_difference_fisher(waves, velocity, parity)
    low_fisher = 17.0 / 16.0
    complete_gamma = math.sqrt(2.0) / 12.0
    pressure_gamma = math.sqrt(2.0) / 48.0
    amplitude = float(size)
    complete_margin = (
        -amplitude * hhl["complete"]
        + complete_gamma * amplitude**3
        - viscosity
        * (high_fisher + low_fisher * amplitude**2)
    )
    pressure_margin = (
        -amplitude * hhl["pressure"]
        + pressure_gamma * amplitude**3
        - viscosity
        * (high_fisher + low_fisher * amplitude**2)
    )
    divergence = np.sum(waves * velocity, axis=-1)
    return {
        "size": size,
        "positive_mode_count": int(size**3),
        "HHL_components": hhl,
        "complete_HHL_load_over_N": hhl["complete"] / size,
        "pressure_HHL_load_over_N": hhl["pressure"] / size,
        "pressure_HH_load_over_N": (
            hhl["pressure_high_high"] / size
        ),
        "high_plus_Fisher": high_fisher,
        "high_plus_Fisher_times_N_cubed": (
            high_fisher * size**3
        ),
        "maximum_divergence_residual": float(
            np.max(np.abs(divergence))
        ),
        "amplitude_equals_N_replay": {
            "amplitude": amplitude,
            "complete_linear_margin": complete_margin,
            "pressure_linear_margin": pressure_margin,
            "complete_coefficient_optimization": (
                _coefficient_scale_optimum(
                    complete_margin,
                    viscosity,
                )
            ),
            "pressure_coefficient_optimization": (
                _coefficient_scale_optimum(
                    pressure_margin,
                    viscosity,
                )
            ),
        },
        "all_finite_checks_pass": bool(
            hhl["complete"] < 0.0
            and hhl["pressure"] < 0.0
            and hhl["pressure_high_high"] < 0.0
            and high_fisher > 0.0
            and hhl["maximum_imaginary_residual"] < 2.0e-12
            and np.max(np.abs(divergence)) < 2.0e-12
        ),
    }


def _prerequisite_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    payloads = {}
    for relative, expected_hash in PREREQUISITES.items():
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[relative] = payload
        actual_hash = _sha256(path)
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "matches": actual_hash == expected_hash,
            }
        )
    return (
        {
            "rows": rows,
            "all_prerequisite_checks_pass": all(
                row["matches"] for row in rows
            ),
        },
        payloads,
    )


def _optimizer_and_scaling_certificate(
    square_payload: dict[str, Any],
    c1_payload: dict[str, Any],
    viscosity: float,
) -> dict[str, Any]:
    beta_signed = float(
        square_payload["finite_static_packet_replay"][
            "positive_packet_continuum_reference_gauss80"
        ]
    )
    q = float(DELTA_CUBIC_ENERGY)
    complete_gamma = math.sqrt(2.0) / 12.0
    pressure_gamma = math.sqrt(2.0) / 48.0
    complete_scale_constant = math.sqrt(
        16.0 * complete_gamma / (3.0 * viscosity * q)
    )
    pressure_scale_constant = math.sqrt(
        16.0 * pressure_gamma / (3.0 * viscosity * q)
    )
    complete_objective_constant = (
        2.0 * complete_gamma * complete_scale_constant / 3.0
    )
    pressure_objective_constant = (
        2.0 * pressure_gamma * pressure_scale_constant / 3.0
    )
    profile_port = c1_payload["fixed_output_convergence_port"]
    checks = {
        "continuum_HHL_reference_negative": beta_signed < 0.0,
        "profile_convergence_port_available": bool(
            all(profile_port["checks"].values())
        ),
        "complete_gamma_positive": complete_gamma > 0.0,
        "pressure_gamma_positive": pressure_gamma > 0.0,
        "coefficient_penalty_positive": q > 0.0,
        "pressure_constants_are_complete_constants_divided_by_four": (
            abs(pressure_gamma - complete_gamma / 4.0) < 1.0e-16
            and abs(
                pressure_scale_constant
                - complete_scale_constant / 2.0
            )
            < 1.0e-16
            and abs(
                pressure_objective_constant
                - complete_objective_constant / 8.0
            )
            < 1.0e-16
        ),
    }
    return {
        "favorable_orientation": "u_N=h_N-a U_*, a>=0",
        "exact_complete_objective": (
            "J_comp,N(a,t)=t[-a B_comp,N+(sqrt(2)/12)a^3"
            "-nu(D_N+(17/16)a^2)]"
            "-(nu/16)(75/256)t^3"
        ),
        "exact_rho_zero_pressure_objective": (
            "J_p,N(a,t)=t[-a B_p,N+(sqrt(2)/48)a^3"
            "-nu(D_N+(17/16)a^2)]"
            "-(nu/16)(75/256)t^3"
        ),
        "finite_N_joint_optimization": {
            "result": (
                "For every fixed N>=3 and every fixed t>0, both "
                "objectives tend to +infinity as a tends to +infinity."
            ),
            "complete_leading_term": (
                "t*(sqrt(2)/12)*a^3"
            ),
            "pressure_leading_term": (
                "t*(sqrt(2)/48)*a^3"
            ),
            "joint_supremum": "+infinity",
            "finite_stationary_optimizer_exists": False,
        },
        "fixed_amplitude_coefficient_optimization": {
            "margin_definition": (
                "A_N(a) is the square bracket multiplying t."
            ),
            "positive_margin_optimum": (
                "t_*(a)=sqrt(16 A_N(a)/(3nu Q)), "
                "max_t J=(2/3)A_N(a)t_*(a)"
            ),
            "complete_large_a_scale": (
                "t_*(a)~sqrt(1024sqrt(2)/(675nu))*a^(3/2)"
            ),
            "pressure_large_a_scale": (
                "t_*(a)~sqrt(256sqrt(2)/(675nu))*a^(3/2)"
            ),
            "complete_objective_scale": (
                "max_t J_comp,N=Theta(a^(9/2))"
            ),
            "pressure_objective_scale": (
                "max_t J_p,N=Theta(a^(9/2))"
            ),
            "complete_scale_numeric_constant": (
                complete_scale_constant
            ),
            "pressure_scale_numeric_constant": (
                pressure_scale_constant
            ),
            "complete_objective_numeric_constant": (
                complete_objective_constant
            ),
            "pressure_objective_numeric_constant": (
                pressure_objective_constant
            ),
        },
        "carrier_power_table": {
            "assumption": "a_N=kappa N^alpha with fixed kappa>0",
            "HHL_power_in_margin": "N^(alpha+1)",
            "low_Fisher_power_in_margin": "N^(2alpha)",
            "low_self_flux_power_in_margin": "N^(3alpha)",
            "high_Fisher_power_in_margin": "N^(-3)",
            "HHL_dominates_self_flux_only_if": "alpha<1/2",
            "HHL_and_self_flux_same_power_if": "alpha=1/2",
            "self_flux_dominates_HHL_if": "alpha>1/2",
            "old_amplitude_alpha_one": {
                "margin": "Theta(N^3), dominated by low self-flux",
                "optimized_t": "Theta(N^(3/2))",
                "optimized_objective": "Theta(N^(9/2))",
                "bounded_t_objective": "Theta(N^3)",
                "old_t_Theta_N_objective": "Theta(N^4)",
            },
        },
        "continuum_HHL_certificate": {
            "pressure_HH_limit": beta_signed,
            "formula": (
                "B_p,N/N -> -(sqrt(2)/20)||b||_L2(D)^2<0"
            ),
            "profile_error_bound": profile_port[
                "profile_error_bound"
            ],
            "role": (
                "The HHL sign remains favorable, but it is not the "
                "leading large-amplitude mechanism after the two-shear "
                "replacement."
            ),
        },
        "restart_scaling_decision": {
            "old_finite_optimizer_ports_unchanged": False,
            "old_a_and_t_Theta_N_scaling_ports_unchanged": False,
            "old_Omega_N3_reset_tax_claim_ports_unchanged": False,
            "old_Omega_N5_average_generator_gate_ports_unchanged": False,
            "reason": (
                "The two-shear low field is not a plane shear. Its exact "
                "self-flux adds a favorable cubic in a, destroys the "
                "finite low-amplitude optimizer, and changes the "
                "coefficient scale before any restart deficit is applied."
            ),
            "two_shear_low_L2_mass": "4",
            "available_norm_only_deficit_bound": (
                "Delta_s>=(1/2)(sqrt(||h_N||_2^2+4a^2)"
                "-5t/16)_+^3"
            ),
            "next_required_gate": (
                "Determine whether a phase/polarization modification can "
                "cancel the low self-flux while preserving the strict "
                "two-shear four-high square. If not, port the exact "
                "backward-adjoint restart with the self-flux and actual "
                "low-mode evolution retained."
            ),
        },
        "checks": checks,
        "all_optimizer_scaling_checks_pass": all(checks.values()),
    }


def audit(
    sizes: Iterable[int] = DEFAULT_SIZES,
    viscosity: float = 1.0,
) -> dict[str, Any]:
    clean_sizes = tuple(int(size) for size in sizes)
    if (
        not clean_sizes
        or any(size < 3 or size % 2 == 0 for size in clean_sizes)
        or tuple(sorted(set(clean_sizes))) != clean_sizes
    ):
        raise ValueError(
            "sizes must be distinct increasing odd integers at least 3"
        )
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive")

    prerequisite, payloads = _prerequisite_audit()
    symbolic = _symbolic_low_field_certificate()
    support = _full_field_support_replay()
    penalty = _coefficient_penalty_certificate()
    rows = [_finite_row(size, viscosity) for size in clean_sizes]
    square_key = (
        "work/ns_collision/results/"
        "annular_two_shear_square_gate_audit_v1.json"
    )
    c1_key = (
        "work/ns_collision/results/"
        "annular_two_shear_full_c1_port_audit_v1.json"
    )
    optimizer = _optimizer_and_scaling_certificate(
        payloads[square_key],
        payloads[c1_key],
        viscosity,
    )
    complete_positive_sizes = [
        row["size"]
        for row in rows
        if row["amplitude_equals_N_replay"][
            "complete_linear_margin"
        ]
        > 0.0
    ]
    pressure_positive_sizes = [
        row["size"]
        for row in rows
        if row["amplitude_equals_N_replay"][
            "pressure_linear_margin"
        ]
        > 0.0
    ]
    all_checks = bool(
        prerequisite["all_prerequisite_checks_pass"]
        and symbolic["all_symbolic_checks_pass"]
        and support["all_support_replay_checks_pass"]
        and penalty["all_penalty_checks_pass"]
        and optimizer["all_optimizer_scaling_checks_pass"]
        and all(row["all_finite_checks_pass"] for row in rows)
        and complete_positive_sizes
        and pressure_positive_sizes
    )
    return {
        "kind": "annular_two_shear_static_optimizer_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": (
            "complete_two_shear_static_optimizer_route_guard"
            if all_checks
            else "two_shear_static_optimizer_audit_failed"
        ),
        "scope": (
            "The complete compatible +++ static bracket and the "
            "pressure-only rho=0 restart generator for the modified "
            "annular high packet plus the exact two-shear low field."
        ),
        "prerequisite_audit": prerequisite,
        "exact_symbolic_low_field_certificate": symbolic,
        "full_field_support_replay": support,
        "coefficient_penalty_certificate": penalty,
        "finite_annular_rows": rows,
        "finite_amplitude_equals_N_summary": {
            "complete_positive_sizes": complete_positive_sizes,
            "pressure_positive_sizes": pressure_positive_sizes,
            "first_complete_positive_size": (
                complete_positive_sizes[0]
                if complete_positive_sizes
                else None
            ),
            "first_pressure_positive_size": (
                pressure_positive_sizes[0]
                if pressure_positive_sizes
                else None
            ),
        },
        "optimizer_and_scaling_certificate": optimizer,
        "theorem": (
            "The modified two-shear low field has exact +++ weighted "
            "Fisher mass 17/16 and exact complete and pressure-only "
            "self-flux loads -sqrt(2)/12 and -sqrt(2)/48. Consequently, "
            "for u_N=h_N-aU_* and every fixed N>=3, both static objectives "
            "are unbounded above in a for every fixed coefficient t>0. "
            "The former finite a,t=Theta(N) optimizer and its restart "
            "scaling do not port unchanged."
        ),
        "route_decision": (
            "Do not import the old optimizer, reset tax, or N^5 restart "
            "gate. First test whether phase or polarization freedom can "
            "remove the low self-flux without losing the strict "
            "four-high square; otherwise retain the self-flux and exact "
            "low-mode dynamics in the backward-adjoint restart."
        ),
        "certification_flags": {
            "exact_two_shear_low_Fisher_enumerated": True,
            "exact_two_shear_complete_self_flux_enumerated": True,
            "exact_two_shear_pressure_self_flux_enumerated": True,
            "complete_HHL_finite_replay_included": True,
            "pressure_HHL_finite_replay_included": True,
            "HLL_and_high_low_Fisher_support_gaps_checked": True,
            "exact_coefficient_penalty_included": True,
            "finite_static_optimizer_exists": False,
            "joint_static_supremum_is_infinite": True,
            "old_restart_scaling_ports_unchanged": False,
            "phase_cancellation_gate_proved": False,
            "complete_finite_first_jet_ported": False,
            "complete_finite_second_jet_ported": False,
            "critical_L3_control_proved": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "next_theorem_target": (
            "Classify the phase and relative-polarization dependence of "
            "the two-shear low self-flux jointly with the static and "
            "four-high sign matrices. Seek an exact self-flux-zero point "
            "inside the strict negative-square region."
        ),
        "all_positive_checks_pass": all_checks,
    }


def _parse_sizes(value: str) -> tuple[int, ...]:
    return tuple(
        int(item.strip()) for item in value.split(",") if item.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SIZES,
        help="comma-separated increasing odd packet sizes",
    )
    parser.add_argument("--viscosity", type=float, default=1.0)
    arguments = parser.parse_args()
    _lower_process_priority()
    result = audit(arguments.sizes, arguments.viscosity)
    _atomic_json(RESULT, result)
    print(
        json.dumps(
            {
                "result": RESULT.relative_to(ROOT).as_posix(),
                "sha256": _sha256(RESULT),
                "status": result["status"],
                "all_positive_checks_pass": result[
                    "all_positive_checks_pass"
                ],
                "finite_amplitude_equals_N_summary": result[
                    "finite_amplitude_equals_N_summary"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["all_positive_checks_pass"]:
        raise SystemExit("two-shear static optimizer audit failed")


if __name__ == "__main__":
    main()
