"""Audit the cross-shell high-high-to-low modulated-wave channel."""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
Wave = tuple[int, int, int]
VectorField = dict[Wave, np.ndarray]
ScalarField = dict[Wave, complex]


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


def _add_wave(first: Wave, second: Wave) -> Wave:
    return tuple(  # type: ignore[return-value]
        left + right for left, right in zip(first, second)
    )


def _negate_wave(wave: Wave) -> Wave:
    return tuple(-value for value in wave)  # type: ignore[return-value]


def _wave_norm(wave: Wave) -> float:
    return math.sqrt(sum(value * value for value in wave))


def _project(vector: np.ndarray, wave: Wave) -> np.ndarray:
    wave_array = np.asarray(wave, dtype=float)
    return vector - wave_array * (
        np.dot(wave_array, vector) / np.dot(wave_array, wave_array)
    )


def _add_real_mode(
    field: VectorField,
    wave: Wave,
    coefficient: np.ndarray,
) -> None:
    field[wave] = np.asarray(coefficient, dtype=np.complex128)
    field[_negate_wave(wave)] = np.conjugate(coefficient)


def _high_field(carrier: int) -> VectorField:
    first_wave = (1, 1, carrier)
    second_wave = (0, 0, carrier)
    first_value = _project(
        np.asarray((1.0, 0.0, 0.0)),
        first_wave,
    )
    first_value /= np.linalg.norm(first_value)
    second_value = np.asarray((1.0, 0.0, 0.0))
    field: VectorField = {}
    _add_real_mode(field, first_wave, first_value)
    _add_real_mode(field, second_wave, second_value)
    return field


def _low_field() -> VectorField:
    wave = (-2, -2, -1)
    stencil = np.asarray((1.0, 1.0, 1.0))
    coefficient = 1j * _project(stencil, wave)
    field: VectorField = {}
    _add_real_mode(field, wave, coefficient)
    return field


def _clean_scalar(field: ScalarField) -> ScalarField:
    return {
        wave: value
        for wave, value in field.items()
        if abs(value) > 1.0e-14
    }


def _clean_vector(field: VectorField) -> VectorField:
    return {
        wave: value
        for wave, value in field.items()
        if np.linalg.norm(value) > 1.0e-14
    }


def _add_vectors(
    *terms: tuple[complex, VectorField],
) -> VectorField:
    output: VectorField = {}
    for factor, field in terms:
        for wave, value in field.items():
            output[wave] = output.get(
                wave,
                np.zeros(3, dtype=np.complex128),
            ) + factor * value
    return _clean_vector(output)


def _add_scalars(
    *terms: tuple[complex, ScalarField],
) -> ScalarField:
    output: ScalarField = {}
    for factor, field in terms:
        for wave, value in field.items():
            output[wave] = output.get(wave, 0.0j) + factor * value
    return _clean_scalar(output)


def _vector_dot_product(
    first: VectorField,
    second: VectorField,
) -> ScalarField:
    output: ScalarField = {}
    for first_wave, first_value in first.items():
        for second_wave, second_value in second.items():
            wave = _add_wave(first_wave, second_wave)
            output[wave] = output.get(wave, 0.0j) + np.dot(
                first_value,
                second_value,
            )
    return _clean_scalar(output)


def _scalar_times_vector(
    scalar: ScalarField,
    vector: VectorField,
) -> VectorField:
    output: VectorField = {}
    for scalar_wave, scalar_value in scalar.items():
        for vector_wave, vector_value in vector.items():
            wave = _add_wave(scalar_wave, vector_wave)
            output[wave] = output.get(
                wave,
                np.zeros(3, dtype=np.complex128),
            ) + scalar_value * vector_value
    return _clean_vector(output)


def _pressure_bilinear(
    first: VectorField,
    second: VectorField,
) -> ScalarField:
    output: ScalarField = {}
    for first_wave, first_value in first.items():
        for second_wave, second_value in second.items():
            wave = _add_wave(first_wave, second_wave)
            norm_squared = sum(value * value for value in wave)
            if norm_squared == 0:
                continue
            wave_array = np.asarray(wave, dtype=float)
            value = -(
                np.dot(wave_array, first_value)
                * np.dot(wave_array, second_value)
                / norm_squared
            )
            output[wave] = output.get(wave, 0.0j) + value
    return _clean_scalar(output)


def _partition_gradient(
    vertex: tuple[int, int, int],
) -> VectorField:
    output: VectorField = {}
    for wave in product((-1, 0, 1), repeat=3):
        if wave == (0, 0, 0):
            continue
        coefficient = 1.0
        for coordinate, frequency in enumerate(wave):
            coefficient *= (
                0.5 if frequency == 0 else 0.25 * vertex[coordinate]
            )
        output[wave] = (
            1j * np.asarray(wave, dtype=float) * coefficient
        )
    return output


def _load(
    flux: VectorField,
    vertex: tuple[int, int, int],
) -> complex:
    gradient = _partition_gradient(vertex)
    value = 0.0j
    for wave, coefficient in flux.items():
        gradient_value = gradient.get(_negate_wave(wave))
        if gradient_value is not None:
            value += np.dot(coefficient, gradient_value)
    return value


def _energy_flux(field: VectorField) -> VectorField:
    kinetic = _scalar_times_vector(
        _vector_dot_product(field, field),
        field,
    )
    pressure = _scalar_times_vector(
        _pressure_bilinear(field, field),
        field,
    )
    return _add_vectors((0.5, kinetic), (1.0, pressure))


def _field_sum(
    first: VectorField,
    second: VectorField,
    second_factor: complex = 1.0,
) -> VectorField:
    return _add_vectors((1.0, first), (second_factor, second))


def _maximum_vector_difference(
    first: VectorField,
    second: VectorField,
) -> float:
    waves = set(first) | set(second)
    zero = np.zeros(3, dtype=np.complex128)
    return max(
        (
            np.linalg.norm(first.get(wave, zero) - second.get(wave, zero))
            for wave in waves
        ),
        default=0.0,
    )


def _component_fluxes(
    high: VectorField,
    low: VectorField,
) -> dict[str, VectorField]:
    high_high_pressure = _pressure_bilinear(high, high)
    cross_pressure = _add_scalars(
        (1.0, _pressure_bilinear(low, high)),
        (1.0, _pressure_bilinear(high, low)),
    )
    kinetic_scalar = _vector_dot_product(high, high)
    low_high_scalar = _vector_dot_product(low, high)
    kinetic = _add_vectors(
        (0.5, _scalar_times_vector(kinetic_scalar, low)),
        (1.0, _scalar_times_vector(low_high_scalar, high)),
    )
    pressure_high_high = _scalar_times_vector(
        high_high_pressure,
        low,
    )
    pressure_cross = _scalar_times_vector(cross_pressure, high)
    combined = _add_vectors(
        (1.0, kinetic),
        (1.0, pressure_high_high),
        (1.0, pressure_cross),
    )
    return {
        "kinetic": kinetic,
        "pressure_high_high": pressure_high_high,
        "pressure_cross": pressure_cross,
        "combined": combined,
    }


def _direct_linear_flux(
    high: VectorField,
    low: VectorField,
) -> VectorField:
    plus = _energy_flux(_field_sum(high, low))
    minus = _energy_flux(_field_sum(high, low, -1.0))
    low_only = _energy_flux(low)
    return _add_vectors(
        (0.5, plus),
        (-0.5, minus),
        (-1.0, low_only),
    )


def _carrier_row(carrier: int) -> dict[str, Any]:
    high = _high_field(carrier)
    low = _low_field()
    components = _component_fluxes(high, low)
    direct = _direct_linear_flux(high, low)
    vertex = (1, 1, 1)
    loads = {
        key: _load(value, vertex)
        for key, value in components.items()
    }
    direct_load = _load(direct, vertex)
    high_high_pressure = _pressure_bilinear(high, high)
    low_output_wave = (1, 1, 0)
    pressure_coefficient = high_high_pressure[low_output_wave]
    first_wave = (1, 1, carrier)
    first_polarization = high[first_wave]
    exact_polarization_factor = (
        carrier**2
        / math.sqrt((carrier**2 + 1) * (carrier**2 + 2))
    )
    high_norms = [_wave_norm(wave) for wave in high]
    divergence_residual = max(
        abs(np.dot(np.asarray(wave, dtype=float), value))
        for wave, value in {**high, **low}.items()
    )
    reality_residual = max(
        np.linalg.norm(
            high.get(_negate_wave(wave), np.zeros(3))
            - np.conjugate(value)
        )
        for wave, value in high.items()
    )
    imaginary_residual = max(
        abs(value.imag) for value in [*loads.values(), direct_load]
    )
    pressure_limit = 1.0 / 144.0
    return {
        "carrier": carrier,
        "minimum_high_mode": min(high_norms),
        "maximum_high_mode": max(high_norms),
        "high_shell_ratio": max(high_norms) / min(high_norms),
        "first_polarization": [
            float(value.real) for value in first_polarization
        ],
        "exact_q_dot_first_polarization": exact_polarization_factor,
        "low_pressure_coefficient_at_q": float(
            pressure_coefficient.real
        ),
        "pressure_only_load": float(
            loads["pressure_high_high"].real
        ),
        "kinetic_load": float(loads["kinetic"].real),
        "cross_pressure_load": float(
            loads["pressure_cross"].real
        ),
        "combined_HHL_load": float(loads["combined"].real),
        "direct_polynomial_linear_load": float(direct_load.real),
        "pressure_load_error_from_limit": abs(
            loads["pressure_high_high"].real - pressure_limit
        ),
        "combined_load_error_from_limit": abs(
            loads["combined"].real - pressure_limit
        ),
        "component_vs_direct_flux_residual": (
            _maximum_vector_difference(components["combined"], direct)
        ),
        "maximum_divergence_residual": float(divergence_residual),
        "maximum_reality_residual": float(reality_residual),
        "maximum_imaginary_load_residual": float(imaginary_residual),
        "all_checks_pass": bool(
            max(high_norms) / min(high_norms) < 1.02
            and abs(
                pressure_coefficient.real + exact_polarization_factor
            )
            < 1.0e-14
            and abs(pressure_coefficient.imag) < 1.0e-14
            and abs(
                loads["combined"].real - direct_load.real
            )
            < 1.0e-12
            and _maximum_vector_difference(
                components["combined"],
                direct,
            )
            < 1.0e-12
            and divergence_residual < 1.0e-12
            and reality_residual < 1.0e-12
            and imaginary_residual < 1.0e-12
        ),
    }


def _asymptotic_replay() -> dict[str, Any]:
    carriers = (8, 16, 32, 64, 128, 256, 512, 1024)
    rows = [_carrier_row(carrier) for carrier in carriers]
    pressure_loads = [row["pressure_only_load"] for row in rows]
    combined_loads = [row["combined_HHL_load"] for row in rows]
    kinetic_loads = [abs(row["kinetic_load"]) for row in rows]
    cross_loads = [abs(row["cross_pressure_load"]) for row in rows]
    pressure_errors = [
        row["pressure_load_error_from_limit"] for row in rows
    ]
    combined_errors = [
        row["combined_load_error_from_limit"] for row in rows
    ]
    scaled_kinetic = [
        carrier * value
        for carrier, value in zip(carriers, kinetic_loads)
    ]
    scaled_cross = [
        carrier**2 * value
        for carrier, value in zip(carriers, cross_loads)
    ]
    limit = 1.0 / 144.0
    return {
        "carriers": list(carriers),
        "rows": rows,
        "analytic_pressure_load_limit": limit,
        "analytic_combined_HHL_load_limit": limit,
        "pressure_last_over_limit": pressure_loads[-1] / limit,
        "combined_last_over_limit": combined_loads[-1] / limit,
        "maximum_kinetic_load": max(kinetic_loads),
        "maximum_H_times_kinetic_load": max(scaled_kinetic),
        "maximum_H_squared_times_cross_pressure_load": max(
            scaled_cross
        ),
        "pressure_errors_strictly_decrease": all(
            first > second
            for first, second in zip(
                pressure_errors,
                pressure_errors[1:],
            )
        ),
        "combined_errors_strictly_decrease_from_H64": all(
            first > second
            for first, second in zip(
                combined_errors[3:],
                combined_errors[4:],
            )
        ),
        "interpretation": (
            "The pressure-only load and the complete cubic HHL local-energy "
            "flux both converge to 1/144. Kinetic transport is O(1/H) "
            "and cross pressure is O(1/H^2) for this geometry. Frequency "
            "separation therefore supplies no positive power of H decay "
            "for the complete channel."
        ),
        "all_checks_pass": bool(
            all(row["all_checks_pass"] for row in rows)
            and min(pressure_loads[-3:]) > 0.99 * limit
            and min(combined_loads[-3:]) > 0.99 * limit
            and all(
                first > second
                for first, second in zip(
                    pressure_errors,
                    pressure_errors[1:],
                )
            )
            and all(
                first > second
                for first, second in zip(
                    combined_errors[3:],
                    combined_errors[4:],
                )
            )
            and max(scaled_kinetic) < 0.02
            and max(scaled_cross) < 0.25
        ),
    }


def _analytic_no_go() -> dict[str, Any]:
    low_wave = (1, 1, 0)
    testing_wave = (-2, -2, -1)
    stencil_wave = (1, 1, 1)
    polarization = (
        Fraction(-1, 9),
        Fraction(-1, 9),
        Fraction(4, 9),
    )
    resonance_sum = tuple(
        low + testing + stencil
        for low, testing, stencil in zip(
            low_wave,
            testing_wave,
            stencil_wave,
        )
    )
    divergence_pairing = sum(
        Fraction(wave) * value
        for wave, value in zip(testing_wave, polarization)
    )
    stencil_pairing = sum(
        Fraction(wave) * value
        for wave, value in zip(stencil_wave, polarization)
    )
    kinetic_limit_pairing = stencil_pairing + 2 * polarization[0]
    pressure_limit = 2 * stencil_pairing / 64
    anisotropic_flux_limit = -4 * polarization[0] / 64
    exact_checks = {
        "q_plus_k_plus_r": list(resonance_sum),
        "k_dot_C": str(divergence_pairing),
        "r_dot_C": str(stencil_pairing),
        "limiting_kinetic_vertex_pairing": str(
            kinetic_limit_pairing
        ),
        "pressure_load_limit": str(pressure_limit),
        "anisotropic_flux_load_limit": str(anisotropic_flux_limit),
        "all_checks_pass": bool(
            resonance_sum == (0, 0, 0)
            and divergence_pairing == 0
            and stencil_pairing == Fraction(2, 9)
            and kinetic_limit_pairing == 0
            and pressure_limit == Fraction(1, 144)
            and anisotropic_flux_limit == Fraction(1, 144)
        ),
    }
    return {
        "high_waves": (
            "a_H=(1,1,H), b_H=(0,0,H), with real conjugates."
        ),
        "high_polarizations": (
            "B=e_1 and A_H=P_(a_H perpendicular)e_1/"
            "|P_(a_H perpendicular)e_1|, so A_H->e_1."
        ),
        "low_wave_and_stencil": (
            "q=a_H-b_H=(1,1,0), k=(-2,-2,-1), "
            "r=(1,1,1), and q+k+r=0."
        ),
        "low_velocity": (
            "Uhat(k)=iC, C=P_(k perpendicular)r="
            "(-1/9,-1/9,4/9)."
        ),
        "low_Reynolds_stress_limit": (
            "Rhat_H(q)=A_H tensor e_1+e_1 tensor A_H "
            "->2e_1 tensor e_1."
        ),
        "exact_low_pressure_coefficient": (
            "phat_HH(q)=-H^2/sqrt((H^2+1)(H^2+2))->-1."
        ),
        "pressure_load_limit": (
            "For the all-cosine vertex, the q,k,r term and its conjugate "
            "give B_HH,L^p ->1/144."
        ),
        "complete_HHL_flux": (
            "F_HHL=(|w_H|^2/2)U+(U dot w_H)w_H"
            "+p[w_H,w_H]U+(p[U,w_H]+p[w_H,U])w_H."
        ),
        "cross_pressure_decay": (
            "The high output cross-pressure symbol has one contraction "
            "with a polarization transverse to its carrier, hence its "
            "contribution is O(1/H)."
        ),
        "complete_flux_limit": (
            "At q, the limiting kinetic scalar U cancels p_HH U, while "
            "the anisotropic Reynolds term 2e_1(e_1 dot U) remains. "
            "Its vertex load, including the conjugate, is 1/144."
        ),
        "falsified_statement": (
            "No estimate for either the pressure-only HHL load or the "
            "complete signed cubic HHL local-energy flux can contain a "
            "universal factor (L/H)^alpha with alpha>0 while all fixed "
            "low amplitudes and high Fourier coefficients remain bounded."
        ),
        "scope": (
            "This rules out carrier-separation decay for the isolated "
            "instantaneous HHL channel. It does not rule out dyadic "
            "summation using shell amplitudes, time integration, "
            "inter-shell telescoping, or a different coupled-cell norm."
        ),
        "exact_rational_checks": exact_checks,
        "all_checks_pass": exact_checks["all_checks_pass"],
    }


def audit() -> dict[str, Any]:
    theorem = _analytic_no_go()
    replay = _asymptotic_replay()
    positive_checks = {
        "analytic_modulated_wave_no_go_passes": theorem[
            "all_checks_pass"
        ],
        "finite_mode_asymptotic_replay_passes": replay[
            "all_checks_pass"
        ],
    }
    return {
        "kind": "cross_shell_modulated_wave_gate_audit",
        "schema_version": 1,
        "status": "cross_shell_carrier_decay_falsified",
        "assumption_scope": (
            "Smooth finite-Fourier divergence-free fields on T^3, one "
            "fixed low velocity mode, and two sidebands in one annular "
            "high shell."
        ),
        "analytic_no_go": theorem,
        "finite_mode_asymptotic_replay": replay,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "pressure_only_cross_shell_H_decay_falsified": True,
            "complete_signed_HHL_flux_H_decay_falsified": True,
            "cross_pressure_leading_order_cancellation_proved": True,
            "anisotropic_Reynolds_stress_survives_in_flux": True,
            "self_shell_pressure_closure_preserved": True,
            "dyadic_amplitude_summation_proved": False,
            "inter_shell_telescoping_proved": False,
            "time_integrated_compensation_proved": False,
            "critical_signed_bound_proved": False,
            "low_regularity_passage_proved": False,
            "exceptional_set_upgrade_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "route_decision": (
            "Do not seek a positive carrier-separation power for the "
            "cross-shell HHL pressure or its complete instantaneous local "
            "energy flux. Build the exact dyadic interaction atlas next "
            "and retain shell amplitudes. Test whether conservative "
            "telescoping across adjacent scale boundaries, time-integrated "
            "viscous payment, or a coupled eight-cell Carleson norm can "
            "sum the O(1) Reynolds-stress channel."
        ),
        "next_theorem_target": (
            "Derive the exact dyadic three-shell flux identity with the "
            "two largest frequencies comparable. Separate self-shell "
            "terms already closed from HHL Reynolds-stress transfers. "
            "Before absolute values, test telescoping in the low-shell "
            "index and across the eight partition vertices; quantify the "
            "remaining term in critical shell amplitudes."
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
            "cross_shell_modulated_wave_gate_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("cross-shell modulated-wave audit failed")
    _atomic_json(arguments.output, result)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": _sha256(arguments.output),
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
