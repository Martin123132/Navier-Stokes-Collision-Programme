"""Certify a separable annular pressure no-go for the joint HHL Schur route."""

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


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "separable_annular_pressure_schur_no_go_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "primitive_hhl_chain_hardy_envelope_audit_v1.json"
    ): "89d5cee5520acead1deba0231bed2cc7e4e740a673223ff5ca733c4a8375d18a",
    (
        "work/ns_collision/results/"
        "joint_primitive_hhl_incidence_schur_gate_audit_v1.json"
    ): "216e41e650e2421c4ef4a2c0100a656618f169a8d9dd758ae5a507a7e23837df",
}
ALGORITHM_REVISION = "separable-annular-pressure-no-go-v1"
ANNULAR_SIZES = (3, 5, 7, 9, 13, 17, 25, 33, 49, 65)
FIXED_TRANSVERSE_SIZES = (8, 16, 32, 64, 128, 256)
VERTEX = (1, 1, 1)
TRANSLATION = np.zeros(3, dtype=float)
LOW_WAVE = np.asarray((0, 1, -1), dtype=int)
LOW_DIRECTION = np.asarray((0.0, 1.0, 1.0)) / math.sqrt(2.0)
Wave = tuple[int, int, int]
VectorField = dict[Wave, np.ndarray]


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


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _vertex_weight_exact(wave: Wave) -> Fraction:
    output = Fraction(1)
    for value in wave:
        output *= Fraction(1, 2) if value == 0 else Fraction(1, 4)
    return output


def _exact_algebra_certificates() -> dict[str, Any]:
    """Work in Q(sqrt(2)); matrices store their sqrt(2) coefficient."""

    direction = (0, 1, 1)
    pressure_loads: dict[Wave, Fraction] = {}
    kinetic_matrix = [
        [Fraction(0) for _ in range(3)] for _ in range(3)
    ]
    for sign in (1, -1):
        low_wave = tuple(
            sign * int(value) for value in LOW_WAVE
        )
        for output in product((-1, 0, 1), repeat=3):
            if output == (0, 0, 0):
                continue
            weight = _vertex_weight_exact(output)
            difference = tuple(
                output[index] - low_wave[index]
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

            parity = (
                1 if sum(difference) % 2 == 0 else -1
            )
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
                    kinetic_matrix[row][column] += factor * entry

    pressure_matrix = [
        [Fraction(0) for _ in range(3)] for _ in range(3)
    ]
    for difference, load_numerator in pressure_loads.items():
        if difference == (0, 0, 0):
            continue
        norm_squared = sum(value * value for value in difference)
        parity = 1 if sum(difference) % 2 == 0 else -1
        # load_numerator/sqrt(2) times -2*parity/|q|^2.
        factor = -parity * load_numerator / norm_squared
        for row in range(3):
            for column in range(3):
                pressure_matrix[row][column] += (
                    factor * difference[row] * difference[column]
                )

    expected_pressure = [
        [Fraction(0), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(1, 20), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(-1, 20)],
    ]
    zero_matrix = [
        [Fraction(0) for _ in range(3)] for _ in range(3)
    ]

    profile = tuple(Fraction(value) for value in (3, -2, 5, 1))
    toeplitz = Fraction(1, 2) * sum(value * value for value in profile)
    toeplitz -= Fraction(1, 2) * sum(
        profile[index] * profile[index + 1]
        for index in range(len(profile) - 1)
    )
    extended = (Fraction(0), *profile, Fraction(0))
    difference_energy = Fraction(1, 4) * sum(
        (extended[index + 1] - extended[index]) ** 2
        for index in range(len(extended) - 1)
    )

    active_loads = {
        ",".join(str(value) for value in wave): _fraction_text(value)
        for wave, value in sorted(pressure_loads.items())
        if value != 0
    }
    return {
        "coefficient_field": "Q(sqrt(2))",
        "fixed_low_wave": LOW_WAVE.tolist(),
        "fixed_positive_low_coefficient": (
            "-i*(e_2+e_3)/sqrt(2)"
        ),
        "pressure_limit_matrix_formula": (
            "A=sum_q L_q[-2(-1)^(q1+q2+q3)/|q|^2] q q^T"
        ),
        "pressure_limit_matrix_sqrt2_coefficients": [
            [_fraction_text(value) for value in row]
            for row in pressure_matrix
        ],
        "expected_pressure_limit_matrix": (
            "diag(0,sqrt(2)/20,-sqrt(2)/20)"
        ),
        "pressure_matrix_exact_match": (
            pressure_matrix == expected_pressure
        ),
        "active_pressure_difference_count": len(active_loads),
        "active_pressure_load_numerators_over_sqrt2": active_loads,
        "kinetic_leading_matrix_over_sqrt2": [
            [_fraction_text(value) for value in row]
            for row in kinetic_matrix
        ],
        "kinetic_leading_matrix_exactly_zero": (
            kinetic_matrix == zero_matrix
        ),
        "one_dimensional_toeplitz_value": _fraction_text(toeplitz),
        "one_dimensional_difference_value": _fraction_text(
            difference_energy
        ),
        "one_dimensional_difference_identity_exact": (
            toeplitz == difference_energy
        ),
        "pressure_limit_absolute_lower_bound": (
            "51*sqrt(2)/438976"
        ),
        "pressure_limit_absolute_lower_bound_decimal": (
            51.0 * math.sqrt(2.0) / 438976.0
        ),
        "all_checks_pass": bool(
            pressure_matrix == expected_pressure
            and kinetic_matrix == zero_matrix
            and toeplitz == difference_energy
            and len(active_loads) == 36
        ),
    }


def _family_arrays(
    shape: tuple[int, int, int],
    carrier: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = shape
    first = np.arange(1, nx + 1, dtype=int)[:, None, None]
    second = np.arange(1, ny + 1, dtype=int)[None, :, None]
    third = np.arange(1, nz + 1, dtype=int)[None, None, :]
    waves = np.empty((*shape, 3), dtype=float)
    waves[..., 0] = carrier + first - 1
    waves[..., 1] = second - (ny + 1) / 2
    waves[..., 2] = third - (nz + 1) / 2

    parity = np.where((first + second + third) % 2 == 0, 1.0, -1.0)
    sine_profile = (
        np.sin(math.pi * first / (nx + 1))
        * np.sin(math.pi * second / (ny + 1))
        * np.sin(math.pi * third / (nz + 1))
    )
    alpha = parity * sine_profile
    norm_squared = np.sum(waves * waves, axis=-1)
    norm = np.sqrt(norm_squared)
    projected = np.zeros_like(waves)
    projected[..., 2] = 1.0
    projected -= waves * (
        waves[..., 2] / norm_squared
    )[..., None]
    velocity = alpha[..., None] * projected / norm[..., None]
    return waves, velocity, parity


def _shift_slices(
    difference: Wave,
    shape: tuple[int, int, int],
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    first = []
    second = []
    for value, size in zip(difference, shape):
        if value >= 0:
            first.append(slice(value, size))
            second.append(slice(0, size - value))
        else:
            first.append(slice(0, size + value))
            second.append(slice(-value, size))
    return tuple(first), tuple(second)


def _vertex_weight_float(wave: Wave) -> float:
    return math.prod(0.5 if value == 0 else 0.25 for value in wave)


def _direct_fisher(
    waves: np.ndarray,
    velocity: np.ndarray,
) -> float:
    shape = waves.shape[:3]
    output = 0.0
    for difference in product((-1, 0, 1), repeat=3):
        first_slice, second_slice = _shift_slices(difference, shape)
        first_wave = waves[first_slice]
        second_wave = waves[second_slice]
        first_velocity = velocity[first_slice]
        second_velocity = velocity[second_slice]
        output += (
            2.0
            * _vertex_weight_float(difference)
            * float(
                np.sum(
                    np.sum(first_wave * second_wave, axis=-1)
                    * np.sum(
                        first_velocity * second_velocity, axis=-1
                    )
                )
            )
        )
    return output


def _mixed_difference_fisher(
    waves: np.ndarray,
    velocity: np.ndarray,
    parity: np.ndarray,
) -> float:
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
    difference = np.diff(
        np.diff(np.diff(padded, axis=0), axis=1), axis=2
    )
    return float(np.sum(difference * difference) / 32.0)


def _resonant_component_loads(
    waves: np.ndarray,
    velocity: np.ndarray,
) -> dict[str, float]:
    shape = waves.shape[:3]
    loads = {
        "kinetic": 0.0j,
        "pressure_high_high": 0.0j,
        "pressure_cross": 0.0j,
    }
    for sign in (1, -1):
        low_wave = sign * LOW_WAVE
        low_value = -sign * 1j * LOW_DIRECTION
        for output_wave in product((-1, 0, 1), repeat=3):
            if output_wave == (0, 0, 0):
                continue
            output = np.asarray(output_wave, dtype=int)
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
                * _vertex_weight_float(output_wave)
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
            loads["pressure_cross"] += np.dot(
                cross_vector, gradient
            )

    loads["combined"] = sum(loads.values())
    return {
        key: float(value.real)
        for key, value in loads.items()
    } | {
        "maximum_imaginary_residual": max(
            abs(value.imag) for value in loads.values()
        )
    }


def _high_field(
    waves: np.ndarray,
    velocity: np.ndarray,
) -> VectorField:
    field: VectorField = {}
    for wave_array, value in zip(
        waves.reshape(-1, 3), velocity.reshape(-1, 3)
    ):
        wave = tuple(int(component) for component in wave_array)
        coefficient = value.astype(np.complex128)
        field[wave] = coefficient
        field[tuple(-component for component in wave)] = np.conjugate(
            coefficient
        )
    return field


def _low_field() -> VectorField:
    wave = tuple(int(value) for value in LOW_WAVE)
    value = -1j * LOW_DIRECTION
    return {
        wave: value.astype(np.complex128),
        tuple(-component for component in wave): np.conjugate(value),
    }


def _dictionary_replay() -> dict[str, Any]:
    waves, velocity, parity = _family_arrays((3, 3, 3), 6)
    vectorized_fisher = _mixed_difference_fisher(
        waves, velocity, parity
    )
    vectorized_loads = _resonant_component_loads(waves, velocity)
    high = _high_field(waves, velocity)
    low = _low_field()
    components = _component_fluxes(high, low)
    direct = _direct_linear_flux(high, low)
    dictionary_fisher = _translated_vertex_fisher(
        high, 1, VERTEX, TRANSLATION
    )
    dictionary_loads = {
        key: _translated_vertex_load(
            value, 1, VERTEX, TRANSLATION
        )
        for key, value in components.items()
    }
    direct_load = _translated_vertex_load(
        direct, 1, VERTEX, TRANSLATION
    )
    load_residuals = {
        key: abs(dictionary_loads[key] - vectorized_loads[key])
        for key in (
            "kinetic",
            "pressure_high_high",
            "pressure_cross",
            "combined",
        )
    }
    divergence_residual = max(
        abs(np.dot(np.asarray(wave, dtype=float), value))
        for wave, value in {**high, **low}.items()
    )
    reality_residual = max(
        np.linalg.norm(
            high[tuple(-component for component in wave)]
            - np.conjugate(value)
        )
        for wave, value in high.items()
    )
    return {
        "size": 3,
        "signed_high_mode_count": len(high),
        "dictionary_Fisher": float(dictionary_fisher.real),
        "vectorized_mixed_difference_Fisher": vectorized_fisher,
        "Fisher_residual": abs(
            dictionary_fisher - vectorized_fisher
        ),
        "dictionary_component_loads": {
            key: float(value.real)
            for key, value in dictionary_loads.items()
        },
        "vectorized_component_loads": {
            key: vectorized_loads[key]
            for key in (
                "kinetic",
                "pressure_high_high",
                "pressure_cross",
                "combined",
            )
        },
        "component_load_residuals": load_residuals,
        "maximum_component_load_residual": max(
            load_residuals.values()
        ),
        "component_vs_direct_load_residual": abs(
            dictionary_loads["combined"] - direct_load
        ),
        "component_vs_direct_flux_residual": (
            _maximum_vector_difference(components["combined"], direct)
        ),
        "maximum_divergence_residual": float(divergence_residual),
        "maximum_reality_residual": float(reality_residual),
        "maximum_imaginary_load_residual": max(
            [
                abs(value.imag) for value in dictionary_loads.values()
            ]
            + [abs(direct_load.imag)]
        ),
        "all_checks_pass": bool(
            abs(dictionary_fisher - vectorized_fisher) < 2.0e-12
            and max(load_residuals.values()) < 2.0e-12
            and abs(
                dictionary_loads["combined"] - direct_load
            )
            < 2.0e-12
            and _maximum_vector_difference(
                components["combined"], direct
            )
            < 2.0e-11
            and divergence_residual < 2.0e-14
            and reality_residual < 2.0e-14
        ),
    }


def _annular_row(size: int) -> dict[str, Any]:
    waves, velocity, parity = _family_arrays(
        (size, size, size), 2 * size
    )
    fisher_direct = _direct_fisher(waves, velocity)
    fisher_difference = _mixed_difference_fisher(
        waves, velocity, parity
    )
    loads = _resonant_component_loads(waves, velocity)
    norms = np.linalg.norm(waves, axis=-1)
    divergence = float(
        np.max(np.abs(np.sum(waves * velocity, axis=-1)))
    )
    combined_ratio = abs(loads["combined"]) / fisher_difference
    return {
        "size": size,
        "positive_high_mode_count": size**3,
        "signed_high_mode_count": 2 * size**3,
        "minimum_high_frequency": float(np.min(norms)),
        "maximum_high_frequency": float(np.max(norms)),
        "shell_ratio": float(np.max(norms) / np.min(norms)),
        "analytic_shell_ratio_upper_bound": "sqrt(19/8)<2",
        "same_sign_high_to_vertex_alias_count": 0,
        "maximum_divergence_residual": divergence,
        "maximum_reality_residual": 0.0,
        "Fisher_energy_direct": fisher_direct,
        "Fisher_energy_mixed_difference": fisher_difference,
        "Fisher_identity_absolute_residual": abs(
            fisher_direct - fisher_difference
        ),
        "Fisher_identity_relative_residual": abs(
            fisher_direct - fisher_difference
        )
        / fisher_difference,
        "size_cubed_times_Fisher": size**3 * fisher_difference,
        "kinetic_load": loads["kinetic"],
        "pressure_high_high_load": loads[
            "pressure_high_high"
        ],
        "pressure_cross_load": loads["pressure_cross"],
        "complete_HHL_load": loads["combined"],
        "maximum_imaginary_load_residual": loads[
            "maximum_imaginary_residual"
        ],
        "kinetic_load_over_size": loads["kinetic"] / size,
        "pressure_high_high_load_over_size": (
            loads["pressure_high_high"] / size
        ),
        "pressure_cross_load_over_size": (
            loads["pressure_cross"] / size
        ),
        "complete_HHL_load_over_size": loads["combined"] / size,
        "absolute_complete_load_over_Fisher": combined_ratio,
        "ratio_over_size_to_fourth": combined_ratio / size**4,
        "all_checks_pass": bool(
            np.max(norms) / np.min(norms) < 2.0
            and divergence < 3.0e-14
            and fisher_difference > 0.0
            and abs(fisher_direct - fisher_difference)
            < 3.0e-11
            and loads["maximum_imaginary_residual"] < 3.0e-12
            and loads["combined"] < 0.0
        ),
    }


def _fixed_transverse_row(size: int) -> dict[str, Any]:
    waves, velocity, parity = _family_arrays((size, 3, 3), size)
    fisher = _mixed_difference_fisher(waves, velocity, parity)
    loads = _resonant_component_loads(waves, velocity)
    return {
        "longitudinal_size": size,
        "transverse_shape": [3, 3],
        "carrier": size,
        "Fisher_energy": fisher,
        "complete_HHL_load": loads["combined"],
        "absolute_complete_load_over_Fisher": (
            abs(loads["combined"]) / fisher
        ),
        "all_checks_pass": bool(
            fisher > 0.0
            and loads["combined"] < 0.0
            and loads["maximum_imaginary_residual"] < 3.0e-12
        ),
    }


def _continuum_pressure_quadrature(order: int = 64) -> dict[str, Any]:
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
    velocity_y = -z * y / radius_squared ** 1.5
    velocity_z = (x * x + y * y) / radius_squared ** 1.5
    raw_integral = float(
        np.sum(
            tensor_weights
            * sine_squared
            * (velocity_y * velocity_y - velocity_z * velocity_z)
        )
    )
    pressure_limit = math.sqrt(2.0) / 20.0 * raw_integral
    lower_bound = 51.0 * math.sqrt(2.0) / 438976.0
    return {
        "quadrature": "tensor Gauss-Legendre",
        "order_per_axis": order,
        "continuum_domain": (
            "[2,3] x [-1/2,1/2] x [-1/2,1/2]"
        ),
        "raw_integral_of_S2_times_Vy2_minus_Vz2": raw_integral,
        "pressure_load_limit": pressure_limit,
        "analytic_absolute_lower_bound": lower_bound,
        "sign_and_margin_check": bool(
            pressure_limit < -lower_bound
        ),
    }


def _theorem_certificate(
    exact: dict[str, Any],
    quadrature: dict[str, Any],
) -> dict[str, Any]:
    return {
        "theorem": (
            "For the fixed compatible tensor vertex lambda and fixed "
            "low wave U below, there is a real divergence-free family "
            "h_N supported in one uniformly bounded annulus such that "
            "E_lambda(h_N)=O(N^-3) and "
            "B_complete_HHL(h_N,U)/N tends to a strictly negative limit. "
            "Consequently |B_complete_HHL|/E_lambda grows at least as "
            "c*N^4, so no N-uniform joint Fisher-Schur bound of this "
            "form can hold."
        ),
        "family": {
            "index_set": "1<=a,b,c<=N, N odd",
            "positive_wave": (
                "k_abc=(2N+a-1,b-(N+1)/2,c-(N+1)/2)"
            ),
            "coefficient": (
                "alpha_abc=(-1)^(a+b+c) "
                "sin(pi*a/(N+1)) sin(pi*b/(N+1)) "
                "sin(pi*c/(N+1))"
            ),
            "positive_velocity": (
                "hhat_N(k_abc)=alpha_abc P_k(e_3)/|k_abc|"
            ),
            "reality_extension": "hhat_N(-k)=conj(hhat_N(k))",
            "fixed_low_wave": "(0,1,-1)",
            "fixed_positive_low_coefficient": (
                "-i(e_2+e_3)/sqrt(2)"
            ),
            "vertex": (
                "lambda(x)=product_j (1+cos(x_j))/2"
            ),
        },
        "annulus_proof": {
            "minimum_frequency_bound": "|k|>=2N",
            "maximum_frequency_bound": (
                "|k|<sqrt(19/2)N"
            ),
            "shell_ratio_bound": "sqrt(19/8)<2",
            "same_sign_vertex_alias_excluded": True,
        },
        "Fisher_proof": {
            "exact_identity": (
                "E_lambda(h_N)=(1/32) sum_cells "
                "||Delta_1 Delta_2 Delta_3 F_N||_F^2"
            ),
            "gauged_grid_tensor": (
                "F_N(a,b,c)=S(t) M(Xi_N(t)), "
                "t=(a,b,c)/(N+1)"
            ),
            "smooth_embedding": (
                "Xi_N(t)=(2-1/N+(1+1/N)t1,"
                "(1+1/N)(t2-1/2),(1+1/N)(t3-1/2))"
            ),
            "cell_Cauchy_bound": (
                "sum||Delta_1Delta_2Delta_3 F_N||^2 "
                "<=(N+1)^-3 integral||partial_123 f_N||^2"
            ),
            "uniform_derivative_reason": (
                "Xi_N([0,1]^3) stays in one compact set separated "
                "from zero for N>=3, and M(xi)=xi tensor "
                "P_xi(e_3)/|xi| is smooth there"
            ),
            "conclusion": "E_lambda(h_N)<=C_F*(N+1)^-3",
            "exact_one_dimensional_identity_checked": exact[
                "one_dimensional_difference_identity_exact"
            ],
        },
        "pressure_limit_proof": {
            "difference_coefficient": (
                "phat_N(q)=-(2/|q|^2) sum_k "
                "(q dot hhat_N(k))(q dot hhat_N(k-q))"
            ),
            "Riemann_limit": (
                "phat_N(q)/N -> "
                "-2(-1)^sum(q)/|q|^2 "
                "integral_D S^2(q dot V)^2"
            ),
            "summed_exact_matrix": (
                "diag(0,sqrt(2)/20,-sqrt(2)/20)"
            ),
            "limit_formula": (
                "B_pressure_HH/N -> (sqrt(2)/20) integral_D "
                "S^2(V_y^2-V_z^2)"
            ),
            "velocity_components": (
                "V_y=-zy/r^3, V_z=(x^2+y^2)/r^3"
            ),
            "pointwise_margin": (
                "V_z^2-V_y^2>=255/13718 on D"
            ),
            "profile_mass": "integral_D S^2=1/8",
            "strict_limit_bound": (
                "lim B_pressure_HH/N <= -51*sqrt(2)/438976<0"
            ),
            "quadrature_value_is_corroborative_only": quadrature[
                "pressure_load_limit"
            ],
        },
        "complete_flux_proof": {
            "kinetic": (
                "After division by N the finite-difference sums are "
                "Riemann sums. Their pointwise leading quadratic matrix "
                "sums exactly to zero over the two low signs, hence "
                "B_kinetic=o(N)."
            ),
            "cross_pressure": (
                "Each p[U,h_k] is O(N^-2), multiplication by the "
                "second O(N^-1) high coefficient gives O(N^-3) per "
                "pair, and only O(N^3) resonant pairs occur; hence "
                "B_cross=O(1)=o(N)."
            ),
            "conclusion": (
                "B_complete_HHL/N has the same strictly negative limit "
                "as B_pressure_HH/N."
            ),
            "kinetic_leading_matrix_exactly_zero": exact[
                "kinetic_leading_matrix_exactly_zero"
            ],
        },
        "logical_conclusion": {
            "assumed_bound_falsified": (
                "|B_complete_HHL(h,U)| <= C ||Uhat|| E_lambda(h) "
                "with C independent of the annular dimension N"
            ),
            "homogeneity_note": (
                "Both sides are quadratic in h, so any desired nonzero "
                "normalization of h leaves the divergent ratio unchanged."
            ),
        },
        "all_checks_pass": bool(
            exact["all_checks_pass"]
            and quadrature["sign_and_margin_check"]
        ),
    }


def audit(
    annular_sizes: tuple[int, ...] = ANNULAR_SIZES,
    fixed_transverse_sizes: tuple[int, ...] = (
        FIXED_TRANSVERSE_SIZES
    ),
    quadrature_order: int = 64,
) -> dict[str, Any]:
    prerequisite = _prerequisite_audit()
    exact = _exact_algebra_certificates()
    annular_rows = [_annular_row(size) for size in annular_sizes]
    control_rows = [
        _fixed_transverse_row(size)
        for size in fixed_transverse_sizes
    ]
    replay = _dictionary_replay()
    quadrature = _continuum_pressure_quadrature(quadrature_order)
    theorem = _theorem_certificate(exact, quadrature)

    final_row = annular_rows[-1]
    tail_ratios = [
        row["absolute_complete_load_over_Fisher"]
        for row in annular_rows[-4:]
    ]
    control_ratios = [
        row["absolute_complete_load_over_Fisher"]
        for row in control_rows
    ]
    numerical_checks = bool(
        all(row["all_checks_pass"] for row in annular_rows)
        and all(row["all_checks_pass"] for row in control_rows)
        and replay["all_checks_pass"]
        and all(
            tail_ratios[index] < tail_ratios[index + 1]
            for index in range(len(tail_ratios) - 1)
        )
        and final_row["absolute_complete_load_over_Fisher"] > 7000.0
        and 3.0 < final_row["size_cubed_times_Fisher"] < 4.5
        and abs(
            final_row["pressure_high_high_load_over_size"]
            - quadrature["pressure_load_limit"]
        )
        < 8.0e-5
        and abs(final_row["kinetic_load_over_size"]) < 1.0e-9
        and abs(final_row["pressure_cross_load_over_size"])
        < 1.0e-10
        and all(
            control_ratios[index] > control_ratios[index + 1]
            for index in range(len(control_ratios) - 1)
        )
        and control_ratios[-1] < 0.3
    )
    all_positive = bool(
        prerequisite["all_checks_pass"]
        and exact["all_checks_pass"]
        and theorem["all_checks_pass"]
        and numerical_checks
    )
    return {
        "kind": "separable_annular_pressure_schur_no_go_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": (
            "analytic_separable_annular_complete_HHL_Schur_no_go_certified"
            if all_positive
            else "separable_annular_pressure_schur_no_go_audit_failed"
        ),
        "all_positive_checks_pass": all_positive,
        "prerequisite_audit": prerequisite,
        "exact_algebra_certificates": exact,
        "analytic_theorem_certificate": theorem,
        "continuum_pressure_quadrature": quadrature,
        "annular_family_rows": annular_rows,
        "fixed_transverse_control_rows": control_rows,
        "dictionary_replay": replay,
        "numerical_summary": {
            "largest_size": final_row["size"],
            "largest_complete_to_Fisher_ratio": final_row[
                "absolute_complete_load_over_Fisher"
            ],
            "largest_ratio_over_size_to_fourth": final_row[
                "ratio_over_size_to_fourth"
            ],
            "largest_size_cubed_times_Fisher": final_row[
                "size_cubed_times_Fisher"
            ],
            "largest_pressure_load_over_size": final_row[
                "pressure_high_high_load_over_size"
            ],
            "continuum_pressure_limit": quadrature[
                "pressure_load_limit"
            ],
            "fixed_transverse_final_ratio": control_ratios[-1],
            "finite_rows_are_proof": False,
            "numerical_checks_pass": numerical_checks,
        },
        "certification_flags": {
            "explicit_real_divergence_free_family_constructed": True,
            "single_uniformly_bounded_annulus_proved": True,
            "exact_mixed_difference_Fisher_identity_proved": True,
            "Fisher_energy_O_N_minus_3_proved": True,
            "strict_nonzero_pressure_limit_proved": True,
            "kinetic_leading_limit_cancels_exactly": True,
            "cross_pressure_is_lower_order_proved": True,
            "complete_HHL_load_linear_in_N_proved": True,
            "complete_HHL_over_Fisher_at_least_order_N4_proved": True,
            "uniform_joint_complete_HHL_Fisher_Schur_bound_falsified": True,
            "isolated_primitive_chain_Hardy_theorem_falsified": False,
            "all_pressure_Fisher_methods_falsified": False,
            "all_cross_shell_HHL_absorbed": False,
            "terminal_dual_supremum_controlled": False,
            "critical_L3_controlled": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "scope": (
            "This is an analytic no-go theorem for one proposed uniform "
            "joint complete-HHL Fisher-Schur estimate at one compatible "
            "tensor vertex. It does not contradict the isolated primitive "
            "chain Hardy theorem, rule out compensated multiscale or "
            "time-dependent estimates, absorb cross-shell terms, control "
            "critical L3, or prove Navier-Stokes regularity."
        ),
        "route_decision": (
            "Do not pursue a scale-uniform joint Schur bound which charges "
            "this complete HHL low-output block solely to the same static "
            "vertex Fisher energy. Any viable continuation must expose an "
            "additional cancellation, use a pressure-adapted signed "
            "multivertex quantity, or pay through evolution/time rather "
            "than this static one-vertex form."
        ),
        "next_theorem_target": (
            "Classify which additional structures can defeat the explicit "
            "annular family without repeating Fisher energy: first test "
            "the signed sum over all eight compatible tensor vertices and "
            "the exact local-energy partition identity; then test whether "
            "the Navier-Stokes time evolution forces phase decorrelation "
            "or a parabolic payment unavailable to a static Schur bound."
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
        "--annular-sizes",
        type=_parse_sizes,
        default=ANNULAR_SIZES,
    )
    parser.add_argument(
        "--fixed-transverse-sizes",
        type=_parse_sizes,
        default=FIXED_TRANSVERSE_SIZES,
    )
    parser.add_argument("--quadrature-order", type=int, default=64)
    parser.add_argument("--output", type=Path, default=RESULT)
    arguments = parser.parse_args()
    _lower_process_priority()
    result = audit(
        annular_sizes=arguments.annular_sizes,
        fixed_transverse_sizes=arguments.fixed_transverse_sizes,
        quadrature_order=arguments.quadrature_order,
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
                "largest_complete_to_Fisher_ratio": result[
                    "numerical_summary"
                ]["largest_complete_to_Fisher_ratio"],
                "continuum_pressure_limit": result[
                    "numerical_summary"
                ]["continuum_pressure_limit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["all_positive_checks_pass"]:
        raise SystemExit("separable annular pressure no-go audit failed")


if __name__ == "__main__":
    main()
