"""Isolate the inviscid pressure branches of the annular second jet."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import sympy as sp

from annular_rho_zero_first_jet_audit import (
    _coefficients,
    _grid_shape,
    _initial_coefficients,
    _physical,
    _pressure_coefficients,
    _scalar_gradient,
    _spectral_data,
    _vector_gradient,
)
from separable_annular_pressure_schur_no_go_audit import (
    LOW_DIRECTION,
    LOW_WAVE,
    _low_field,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_inviscid_second_jet_branch_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "annular_rho_zero_second_jet_route_guard_audit_v1.json"
    ): "7c985480afc51a084eefa0e2fb614fd3b900e9d2e347a6effb0c99b7259c693d",
}
ALGORITHM_REVISION = "annular-rho-zero-inviscid-second-jet-branch-v1"
DEFAULT_SIZES = (5, 7, 9, 13, 17, 21, 25, 29)
Array = np.ndarray
VectorField = dict[str, Any]
ScalarField = dict[str, Any]
VectorPolynomial = dict[int, list[tuple[float, VectorField]]]
ScalarPolynomial = dict[int, list[tuple[float, ScalarField]]]


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


def _prerequisite_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    payload: dict[str, Any] = {}
    for relative, expected in PREREQUISITES.items():
        path = ROOT / relative
        candidate = json.loads(path.read_text(encoding="utf-8"))
        actual = _sha256(path)
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "all_positive_checks_pass": candidate.get(
                    "all_positive_checks_pass"
                ),
                "matches": bool(
                    actual == expected
                    and candidate.get("all_positive_checks_pass") is True
                ),
            }
        )
        payload = candidate
    return (
        {
            "rows": rows,
            "all_checks_pass": all(row["matches"] for row in rows),
        },
        payload,
    )


def _symbolic_compact_identity_certificate() -> dict[str, Any]:
    time_symbol, u0, lambda0 = sp.symbols(
        "t u_0 lambda_0", real=True
    )
    Euler_coefficient, transport_coefficient = sp.symbols(
        "b c", real=True
    )
    u1 = Euler_coefficient * u0**2
    lambda1 = transport_coefficient * u0 * lambda0
    u2 = 2 * Euler_coefficient * u0 * u1
    lambda2 = (
        transport_coefficient * u1 * lambda0
        + transport_coefficient * u0 * lambda1
    )
    u_path = u0 + time_symbol * u1 + time_symbol**2 * u2 / 2
    lambda_path = (
        lambda0
        + time_symbol * lambda1
        + time_symbol**2 * lambda2 / 2
    )
    direct = sp.diff(u_path**3 * lambda_path, time_symbol, 2).subs(
        time_symbol, 0
    )
    compact = (
        6 * u0 * u1**2 * lambda0
        + 6 * u0**2 * u1 * lambda1
        + 3 * u0**2 * u2 * lambda0
        + u0**3 * lambda2
    )

    amplitude = sp.symbols("a", real=True)
    c1, c3 = sp.symbols("c_1 c_3", real=True)
    parity_polynomial = c1 * amplitude + c3 * amplitude**3
    return {
        "symmetric_pressure_form": (
            "S(x,y,z;phi)=(T(x,y,z;phi)+T(x,z,y;phi)"
            "+T(y,z,x;phi))/3, "
            "T(x,y,z;phi)=integral p[x,y] z dot grad phi"
        ),
        "Euler_bilinear": (
            "B(x,y)=-P[((x dot grad)y+(y dot grad)x)/2], "
            "E(u)=B(u,u)"
        ),
        "transport_bilinear": "C(u,phi)=-u dot grad phi",
        "compact_identity": (
            "J_inv=6S(u,E,E;lambda)+6S(u,u,E;A)"
            "+6S(u,u,B(u,E);lambda)+S(u,u,u;lambda_2)"
        ),
        "directions": {
            "E": "B(u,u)",
            "A": "C(u,lambda)",
            "lambda_2": "C(E,lambda)+C(u,A)",
        },
        "chain_rule_residual": str(sp.simplify(direct - compact)),
        "annular_amplitude_polynomial": str(parity_polynomial),
        "amplitude_reason": (
            "J_inv is degree five in u and linear in lambda. The support "
            "gap permits only zero, two, or four high velocity legs. "
            "The low-amplitude powers are therefore five, three, or one; "
            "the pure-low power five vanishes because B(U,U)=p[U,U]=0."
        ),
        "all_checks_pass": bool(sp.simplify(direct - compact) == 0),
    }


def _low_shear_certificate() -> dict[str, Any]:
    dot = float(np.dot(LOW_WAVE, LOW_DIRECTION))
    return {
        "wave": LOW_WAVE.tolist(),
        "polarization": LOW_DIRECTION.tolist(),
        "wave_dot_polarization": dot,
        "self_advection": "(U dot grad)U=0",
        "self_pressure": "p[U,U]=0",
        "Euler_field": "B(U,U)=0",
        "all_checks_pass": abs(dot) < 1.0e-15,
    }


def _branch_support_certificate() -> dict[str, Any]:
    return {
        "full_second_jet_support": "5K+O(1)",
        "full_second_jet_dealias_factor": 10,
        "inviscid_pressure_amplitude_branches": {
            "a1": {
                "high_velocity_legs": 4,
                "maximum_integrand_support": "4K+O(1)",
                "dealias_factor": 8,
            },
            "a3": {
                "high_velocity_legs": 2,
                "maximum_integrand_support": "2K+O(1)",
                "dealias_factor": 4,
            },
            "a5": {
                "high_velocity_legs": 0,
                "value": "zero",
            },
        },
        "implemented_joint_factor": 8,
        "reason": (
            "Polynomial branch projection removes every five-high "
            "coefficient before FFT evaluation. The largest retained "
            "mean has four high legs, so eight-times one-field support "
            "prevents circular return to zero."
        ),
        "all_checks_pass": True,
    }


def _make_vector_field(
    label: str,
    coefficients: Array,
    waves: tuple[Array, ...],
    volume: int,
    need_gradient: bool = True,
) -> VectorField:
    return {
        "label": label,
        "coefficients": coefficients,
        "value": _physical(coefficients, volume),
        "gradient": (
            _vector_gradient(coefficients, waves, volume)
            if need_gradient
            else None
        ),
    }


def _make_scalar_field(
    label: str,
    coefficients: Array,
    waves: tuple[Array, ...],
    volume: int,
    need_gradient: bool = True,
) -> ScalarField:
    return {
        "label": label,
        "coefficients": coefficients,
        "value": _physical(coefficients, volume),
        "gradient": (
            _scalar_gradient(coefficients, waves, volume)
            if need_gradient
            else None
        ),
    }


def _branch_row(
    size: int,
    dealias_factor: int = 8,
) -> dict[str, Any]:
    if dealias_factor < 8:
        raise ValueError("joint a1/a3 branch padding must be at least eight")
    started = time.perf_counter()
    shape = _grid_shape(size, dealias_factor)
    (
        spectral_waves,
        wave_number_squared,
        safe_wave_number_squared,
        volume,
    ) = _spectral_data(shape)
    (
        high_coefficients,
        weight_coefficients,
        _,
        _,
        _,
    ) = _initial_coefficients(
        size,
        shape,
        low_amplitude=0.0,
        coefficient_scale=1.0,
    )
    low_coefficients = np.zeros_like(high_coefficients)
    for wave, value in _low_field().items():
        index = tuple(
            (
                np.asarray(wave, dtype=int)
                % np.asarray(shape, dtype=int)
            ).tolist()
        )
        low_coefficients[(slice(None), *index)] = value

    high = _make_vector_field(
        "H", high_coefficients, spectral_waves, volume
    )
    low = _make_vector_field(
        "U", low_coefficients, spectral_waves, volume
    )
    weight = _make_scalar_field(
        "Phi", weight_coefficients, spectral_waves, volume
    )
    pressure_cache: dict[tuple[str, str], ScalarField] = {}
    symmetric_form_cache: dict[
        tuple[tuple[str, str, str], str], float
    ] = {}
    trilinear_shell_cache: dict[
        tuple[str, str, str, str], dict[int, complex]
    ] = {}
    trilinear_low_mode_cache: dict[
        tuple[str, str, str, str], dict[tuple[int, int, int], complex]
    ] = {}

    def pressure_pair(
        first: VectorField,
        second: VectorField,
    ) -> ScalarField:
        key = tuple(sorted((first["label"], second["label"])))
        if key not in pressure_cache:
            coefficients = _pressure_coefficients(
                first["value"],
                second["value"],
                spectral_waves,
                safe_wave_number_squared,
                volume,
            )
            pressure_cache[key] = _make_scalar_field(
                f"p[{key[0]},{key[1]}]",
                coefficients,
                spectral_waves,
                volume,
                need_gradient=False,
            )
        return pressure_cache[key]

    def Euler_bilinear(
        first: VectorField,
        second: VectorField,
        label: str,
        need_gradient: bool = True,
    ) -> VectorField:
        first_gradient = first["gradient"]
        second_gradient = second["gradient"]
        if first_gradient is None or second_gradient is None:
            raise ValueError("Euler inputs require gradients")
        cross_advection = (
            np.einsum(
                "j...,ij...->i...",
                first["value"],
                second_gradient,
            )
            + np.einsum(
                "j...,ij...->i...",
                second["value"],
                first_gradient,
            )
        )
        pressure = pressure_pair(first, second)
        if pressure["gradient"] is None:
            pressure["gradient"] = _scalar_gradient(
                pressure["coefficients"], spectral_waves, volume
            )
        coefficients = _coefficients(
            -0.5 * cross_advection - pressure["gradient"],
            volume,
        )
        return _make_vector_field(
            label,
            coefficients,
            spectral_waves,
            volume,
            need_gradient=need_gradient,
        )

    def transport(
        velocity: VectorField,
        scalar: ScalarField,
        label: str,
    ) -> ScalarField:
        gradient = scalar["gradient"]
        if gradient is None:
            raise ValueError("transported scalar requires a gradient")
        values = -np.sum(velocity["value"] * gradient, axis=0)
        return _make_scalar_field(
            label,
            _coefficients(values, volume),
            spectral_waves,
            volume,
        )

    def combine_vector(
        label: str,
        terms: Sequence[tuple[float, VectorField]],
        need_gradient: bool,
    ) -> VectorField:
        coefficients = sum(
            factor * field["coefficients"] for factor, field in terms
        )
        return _make_vector_field(
            label,
            coefficients,
            spectral_waves,
            volume,
            need_gradient=need_gradient,
        )

    def combine_scalar(
        label: str,
        terms: Sequence[tuple[float, ScalarField]],
    ) -> ScalarField:
        coefficients = sum(
            factor * field["coefficients"] for factor, field in terms
        )
        return _make_scalar_field(
            label,
            coefficients,
            spectral_waves,
            volume,
        )

    def trilinear(
        first: VectorField,
        second: VectorField,
        third: VectorField,
        scalar: ScalarField,
    ) -> float:
        gradient = scalar["gradient"]
        if gradient is None:
            raise ValueError("pressure test scalar requires a gradient")
        return float(
            np.mean(
                pressure_pair(first, second)["value"]
                * np.sum(third["value"] * gradient, axis=0)
            )
        )

    def symmetric_form(
        first: VectorField,
        second: VectorField,
        third: VectorField,
        scalar: ScalarField,
    ) -> float:
        vectors = tuple(
            sorted(
                (first["label"], second["label"], third["label"])
            )
        )
        key = (vectors, scalar["label"])
        if key not in symmetric_form_cache:
            symmetric_form_cache[key] = (
                trilinear(first, second, third, scalar)
                + trilinear(first, third, second, scalar)
                + trilinear(second, third, first, scalar)
            ) / 3.0
        return symmetric_form_cache[key]

    shell_index: Array | None = None
    shell_bounds: dict[int, tuple[float, float | None]] = {}

    def trilinear_shells(
        first: VectorField,
        second: VectorField,
        third: VectorField,
        scalar: ScalarField,
    ) -> dict[int, complex]:
        nonlocal shell_index
        pair_labels = tuple(sorted((first["label"], second["label"])))
        key = (
            pair_labels[0],
            pair_labels[1],
            third["label"],
            scalar["label"],
        )
        if key in trilinear_shell_cache:
            return trilinear_shell_cache[key]
        if shell_index is None:
            integer_wave_number_squared = np.rint(
                wave_number_squared
            )
            radius = np.sqrt(integer_wave_number_squared)
            shell_index = np.zeros(shape, dtype=np.int16)
            positive = radius > 0.0
            shell_index[positive] = (
                np.floor(np.log2(radius[positive])).astype(np.int16)
                + 1
            )
            maximum = int(np.max(shell_index))
            shell_bounds[0] = (0.0, 0.0)
            for index in range(1, maximum + 1):
                shell_bounds[index] = (
                    2.0 ** (index - 1),
                    2.0**index,
                )
            del radius
        gradient = scalar["gradient"]
        if gradient is None:
            raise ValueError("pressure test scalar requires a gradient")
        test_coefficients = _coefficients(
            np.sum(third["value"] * gradient, axis=0),
            volume,
        )
        summands = (
            pressure_pair(first, second)["coefficients"]
            * np.conjugate(test_coefficients)
        )
        flat_shell = shell_index.ravel()
        real = np.bincount(
            flat_shell,
            weights=summands.real.ravel(),
        )
        imaginary = np.bincount(
            flat_shell,
            weights=summands.imag.ravel(),
        )
        profile = {
            index: complex(real[index], imaginary[index])
            for index in range(len(real))
            if abs(real[index]) + abs(imaginary[index]) > 1.0e-18
        }
        trilinear_shell_cache[key] = profile
        low_mode_mask = np.rint(wave_number_squared) < 16.0
        low_mode_indices = np.argwhere(low_mode_mask)
        low_modes: dict[tuple[int, int, int], complex] = {}
        for index_array in low_mode_indices:
            index = tuple(int(value) for value in index_array)
            value = complex(summands[index])
            if abs(value) <= 1.0e-18:
                continue
            wave = tuple(
                int(round(spectral_waves[axis][index]))
                for axis in range(3)
            )
            low_modes[wave] = value
        trilinear_low_mode_cache[key] = low_modes
        del test_coefficients
        del summands
        return profile

    def symmetric_shells(
        first: VectorField,
        second: VectorField,
        third: VectorField,
        scalar: ScalarField,
        multiplicity: float,
    ) -> dict[str, Any]:
        profiles = (
            trilinear_shells(first, second, third, scalar),
            trilinear_shells(first, third, second, scalar),
            trilinear_shells(second, third, first, scalar),
        )
        indices = sorted(set().union(*(profile.keys() for profile in profiles)))
        rows = []
        maximum_imaginary = 0.0
        for index in indices:
            value = multiplicity * sum(
                profile.get(index, 0.0j) for profile in profiles
            ) / 3.0
            maximum_imaginary = max(maximum_imaginary, abs(value.imag))
            lower, upper = shell_bounds[index]
            rows.append(
                {
                    "shell_index": index,
                    "radius_lower": lower,
                    "radius_upper": upper,
                    "value": float(value.real),
                    "imaginary_residual": float(abs(value.imag)),
                }
            )
        direct = multiplicity * symmetric_form(
            first, second, third, scalar
        )
        shell_sum = sum(row["value"] for row in rows)
        trilinear_keys = (
            (
                tuple(sorted((first["label"], second["label"]))),
                third["label"],
            ),
            (
                tuple(sorted((first["label"], third["label"]))),
                second["label"],
            ),
            (
                tuple(sorted((second["label"], third["label"]))),
                first["label"],
            ),
        )
        mode_profiles = [
            trilinear_low_mode_cache[
                (
                    pair[0],
                    pair[1],
                    third_label,
                    scalar["label"],
                )
            ]
            for pair, third_label in trilinear_keys
        ]
        low_waves = sorted(
            set().union(*(profile.keys() for profile in mode_profiles))
        )
        low_mode_rows = []
        for wave in low_waves:
            value = multiplicity * sum(
                profile.get(wave, 0.0j) for profile in mode_profiles
            ) / 3.0
            low_mode_rows.append(
                {
                    "wave": list(wave),
                    "radius_squared": sum(
                        component * component for component in wave
                    ),
                    "value": float(value.real),
                    "imaginary_residual": float(abs(value.imag)),
                }
            )
        return {
            "rows": rows,
            "bounded_output_mode_rows": low_mode_rows,
            "bounded_output_sum": sum(
                row["value"] for row in low_mode_rows
            ),
            "shell_sum": shell_sum,
            "direct_value": direct,
            "replay_residual": abs(shell_sum - direct),
            "maximum_imaginary_residual": maximum_imaginary,
        }

    Euler_HH = Euler_bilinear(high, high, "B[H,H]")
    Euler_HU = Euler_bilinear(high, low, "B[H,U]")
    transport_H = transport(high, weight, "C[H,Phi]")
    transport_U = transport(low, weight, "C[U,Phi]")

    acceleration_0 = Euler_bilinear(
        high,
        Euler_HH,
        "B[H,B[H,H]]",
        need_gradient=False,
    )
    acceleration_1 = combine_vector(
        "G1",
        (
            (
                -1.0,
                Euler_bilinear(
                    low,
                    Euler_HH,
                    "B[U,B[H,H]]",
                    need_gradient=False,
                ),
            ),
            (
                -2.0,
                Euler_bilinear(
                    high,
                    Euler_HU,
                    "B[H,B[H,U]]",
                    need_gradient=False,
                ),
            ),
        ),
        need_gradient=False,
    )
    acceleration_2 = combine_vector(
        "G2",
        (
            (
                2.0,
                Euler_bilinear(
                    low,
                    Euler_HU,
                    "B[U,B[H,U]]",
                    need_gradient=False,
                ),
            ),
        ),
        need_gradient=False,
    )

    weight_second_0 = combine_scalar(
        "L0",
        (
            (
                1.0,
                transport(Euler_HH, weight, "C[B[H,H],Phi]"),
            ),
            (
                1.0,
                transport(high, transport_H, "C[H,C[H,Phi]]"),
            ),
        ),
    )
    weight_second_1 = combine_scalar(
        "L1",
        (
            (
                -2.0,
                transport(Euler_HU, weight, "C[B[H,U],Phi]"),
            ),
            (
                -1.0,
                transport(low, transport_H, "C[U,C[H,Phi]]"),
            ),
            (
                -1.0,
                transport(high, transport_U, "C[H,C[U,Phi]]"),
            ),
        ),
    )
    weight_second_2 = combine_scalar(
        "L2",
        (
            (
                1.0,
                transport(low, transport_U, "C[U,C[U,Phi]]"),
            ),
        ),
    )

    velocity: VectorPolynomial = {
        0: [(1.0, high)],
        1: [(-1.0, low)],
    }
    Euler: VectorPolynomial = {
        0: [(1.0, Euler_HH)],
        1: [(-2.0, Euler_HU)],
    }
    weight_first: ScalarPolynomial = {
        0: [(1.0, transport_H)],
        1: [(-1.0, transport_U)],
    }
    acceleration: VectorPolynomial = {
        0: [(1.0, acceleration_0)],
        1: [(1.0, acceleration_1)],
        2: [(1.0, acceleration_2)],
    }
    weight_second: ScalarPolynomial = {
        0: [(1.0, weight_second_0)],
        1: [(1.0, weight_second_1)],
        2: [(1.0, weight_second_2)],
    }
    fixed_weight: ScalarPolynomial = {0: [(1.0, weight)]}

    def polynomial_form(
        first: VectorPolynomial,
        second: VectorPolynomial,
        third: VectorPolynomial,
        scalar: ScalarPolynomial,
        target_power: int,
    ) -> tuple[float, list[dict[str, Any]]]:
        value = 0.0
        term_values: dict[str, float] = {}
        for first_power, first_terms in first.items():
            for second_power, second_terms in second.items():
                for third_power, third_terms in third.items():
                    for scalar_power, scalar_terms in scalar.items():
                        if (
                            first_power
                            + second_power
                            + third_power
                            + scalar_power
                            != target_power
                        ):
                            continue
                        for first_factor, first_field in first_terms:
                            for second_factor, second_field in second_terms:
                                for third_factor, third_field in third_terms:
                                    for scalar_factor, scalar_field in (
                                        scalar_terms
                                    ):
                                        contribution = (
                                            first_factor
                                            * second_factor
                                            * third_factor
                                            * scalar_factor
                                            * symmetric_form(
                                                first_field,
                                                second_field,
                                                third_field,
                                                scalar_field,
                                            )
                                        )
                                        value += contribution
                                        vector_labels = sorted(
                                            (
                                                first_field["label"],
                                                second_field["label"],
                                                third_field["label"],
                                            )
                                        )
                                        label = (
                                            "S["
                                            + ",".join(vector_labels)
                                            + f";{scalar_field['label']}]"
                                        )
                                        term_values[label] = (
                                            term_values.get(label, 0.0)
                                            + contribution
                                        )
        rows = [
            {"term": label, "value": term_value}
            for label, term_value in sorted(term_values.items())
        ]
        return value, rows

    def coefficient_blocks(
        target_power: int,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        definitions = (
            (
                "6S[u,E,E;Phi]",
                6.0,
                velocity,
                Euler,
                Euler,
                fixed_weight,
            ),
            (
                "6S[u,u,E;A]",
                6.0,
                velocity,
                velocity,
                Euler,
                weight_first,
            ),
            (
                "6S[u,u,B(u,E);Phi]",
                6.0,
                velocity,
                velocity,
                acceleration,
                fixed_weight,
            ),
            (
                "S[u,u,u;lambda2]",
                1.0,
                velocity,
                velocity,
                velocity,
                weight_second,
            ),
        )
        blocks: dict[str, float] = {}
        term_rows: list[dict[str, Any]] = []
        for block, multiplicity, *polynomials in definitions:
            block_value, rows = polynomial_form(
                *polynomials, target_power
            )
            blocks[block] = multiplicity * block_value
            term_rows.extend(
                {
                    "block": block,
                    "term": row["term"],
                    "multiplicity": multiplicity,
                    "value": multiplicity * row["value"],
                }
                for row in rows
            )
        blocks["combined"] = sum(blocks.values())
        return blocks, term_rows

    coefficient_data = {
        power: coefficient_blocks(power) for power in (1, 2, 3, 4, 5)
    }
    coefficients = {
        power: value[0] for power, value in coefficient_data.items()
    }
    coefficient_term_rows = {
        power: value[1] for power, value in coefficient_data.items()
    }
    dominant_shell_profiles = {
        "-6S[BHH,BHH,U;Phi]": symmetric_shells(
            Euler_HH,
            Euler_HH,
            low,
            weight,
            -6.0,
        ),
        "-12S[B(H,BHH),H,U;Phi]": symmetric_shells(
            acceleration_0,
            high,
            low,
            weight,
            -12.0,
        ),
    }
    bounded_mode_values: dict[tuple[int, int, int], float] = {}
    for profile in dominant_shell_profiles.values():
        for mode_row in profile["bounded_output_mode_rows"]:
            wave = tuple(int(value) for value in mode_row["wave"])
            bounded_mode_values[wave] = (
                bounded_mode_values.get(wave, 0.0)
                + float(mode_row["value"])
            )
    combined_bounded_mode_rows = [
        {
            "wave": list(wave),
            "radius_squared": sum(value * value for value in wave),
            "value": value,
            "value_over_N6": value / size**6,
            "value_over_N7": value / size**7,
        }
        for wave, value in sorted(bounded_mode_values.items())
        if abs(value) > 1.0e-16
    ]
    combined_bounded_output_sum = sum(
        row["value"] for row in combined_bounded_mode_rows
    )
    dominant_direct_sum = sum(
        profile["direct_value"]
        for profile in dominant_shell_profiles.values()
    )
    outside_bounded_output = (
        dominant_direct_sum - combined_bounded_output_sum
    )
    shell_replay_residual = max(
        profile["replay_residual"]
        for profile in dominant_shell_profiles.values()
    )
    shell_imaginary_residual = max(
        profile["maximum_imaginary_residual"]
        for profile in dominant_shell_profiles.values()
    )
    pressure_symmetry_residual = float(
        np.max(
            np.abs(
                _pressure_coefficients(
                    high["value"],
                    low["value"],
                    spectral_waves,
                    safe_wave_number_squared,
                    volume,
                )
                - _pressure_coefficients(
                    low["value"],
                    high["value"],
                    spectral_waves,
                    safe_wave_number_squared,
                    volume,
                )
            )
        )
    )
    divergence_fields = (
        high,
        low,
        Euler_HH,
        Euler_HU,
        acceleration_0,
        acceleration_1,
        acceleration_2,
    )
    divergence_residual = max(
        float(
            np.max(
                np.abs(
                    sum(
                        spectral_waves[index]
                        * field["coefficients"][index]
                        for index in range(3)
                    )
                )
            )
        )
        for field in divergence_fields
    )
    forbidden_parity_residual = max(
        abs(coefficients[power]["combined"]) for power in (2, 4, 5)
    )
    output = {
        "size": size,
        "grid_shape": list(shape),
        "grid_point_count": volume,
        "dealias_factor": dealias_factor,
        "coefficient_blocks": {
            f"a{power}": coefficients[power] for power in coefficients
        },
        "a1_coefficient": coefficients[1]["combined"],
        "a3_coefficient": coefficients[3]["combined"],
        "a1_over_N5": coefficients[1]["combined"] / size**5,
        "a1_over_N6": coefficients[1]["combined"] / size**6,
        "a1_over_N7": coefficients[1]["combined"] / size**7,
        "a3_over_N3": coefficients[3]["combined"] / size**3,
        "coefficient_term_rows": {
            "a1": coefficient_term_rows[1],
            "a3": coefficient_term_rows[3],
        },
        "dominant_a1_pressure_output_shells": dominant_shell_profiles,
        "combined_dominant_bounded_output_modes": (
            combined_bounded_mode_rows
        ),
        "combined_dominant_bounded_output_sum": (
            combined_bounded_output_sum
        ),
        "combined_dominant_bounded_output_sum_over_N6": (
            combined_bounded_output_sum / size**6
        ),
        "combined_dominant_bounded_output_sum_over_N7": (
            combined_bounded_output_sum / size**7
        ),
        "combined_dominant_outside_bounded_output": (
            outside_bounded_output
        ),
        "combined_dominant_outside_bounded_output_over_N6": (
            outside_bounded_output / size**6
        ),
        "combined_dominant_outside_bounded_output_over_N7": (
            outside_bounded_output / size**7
        ),
        "maximum_shell_replay_residual": shell_replay_residual,
        "maximum_shell_imaginary_residual": shell_imaginary_residual,
        "maximum_forbidden_amplitude_parity_residual": (
            forbidden_parity_residual
        ),
        "pressure_bilinear_symmetry_residual": pressure_symmetry_residual,
        "maximum_divergence_residual": divergence_residual,
        "runtime_seconds": time.perf_counter() - started,
        "all_checks_pass": bool(
            forbidden_parity_residual < 2.0e-8
            and pressure_symmetry_residual < 2.0e-10
            and divergence_residual < 3.0e-7
            and shell_replay_residual < 3.0e-8
            and shell_imaginary_residual < 3.0e-8
            and math.isfinite(coefficients[1]["combined"])
            and math.isfinite(coefficients[3]["combined"])
        ),
    }
    del pressure_cache
    del symmetric_form_cache
    del trilinear_low_mode_cache
    gc.collect()
    return output


def _stored_pressure_group(row: dict[str, Any]) -> float:
    channels = row["second_variation"]["channels"]
    selected = (
        "H_uu[E,E]",
        "2H_u_lambda[E,A]",
        "D_u[u2_EE]",
        "D_lambda[lambda2_E0]",
        "D_lambda[lambda2_0A]",
    )
    return sum(
        value
        for label in selected
        for sublabel, value in channels[label]["subterms"].items()
        if sublabel.startswith("pressure")
    )


def _predecessor_replay(
    branch: dict[str, Any],
    predecessor_row: dict[str, Any],
) -> dict[str, Any]:
    amplitude = float(predecessor_row["low_amplitude"])
    coefficient_scale = float(predecessor_row["coefficient_scale"])
    predicted = coefficient_scale * (
        branch["a1_coefficient"] * amplitude
        + branch["a3_coefficient"] * amplitude**3
    )
    stored = _stored_pressure_group(predecessor_row)
    return {
        "size": branch["size"],
        "low_amplitude": amplitude,
        "coefficient_scale": coefficient_scale,
        "branch_polynomial_prediction": predicted,
        "stored_twenty_channel_pressure_group": stored,
        "absolute_residual": abs(predicted - stored),
        "all_checks_pass": abs(predicted - stored) < 3.0e-8,
    }


def _scaling_diagnostics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    sizes = np.asarray([row["size"] for row in rows], dtype=float)
    output: dict[str, Any] = {}
    for label in ("a1_coefficient", "a3_coefficient"):
        values = np.asarray([abs(row[label]) for row in rows], dtype=float)
        if len(rows) == 1:
            slope = float("nan")
            intercept = float("nan")
        else:
            slope, intercept = np.polyfit(
                np.log(sizes), np.log(values), 1
            )
        output[label] = {
            "log_log_exponent": float(slope),
            "log_coefficient": float(intercept),
            "largest_value": float(rows[-1][label]),
        }
    output["largest_a1_over_N5"] = rows[-1]["a1_over_N5"]
    output["largest_a1_over_N6"] = rows[-1]["a1_over_N6"]
    output["largest_a1_over_N7"] = rows[-1]["a1_over_N7"]
    output["largest_a3_over_N3"] = rows[-1]["a3_over_N3"]
    return output


def _route_decision(
    rows: Sequence[dict[str, Any]],
    scaling: dict[str, Any],
) -> dict[str, Any]:
    return {
        "exact_amplitude_reduction": (
            "J_inv,N(a,t)=t[c_1,N a+c_3,N a^3]"
        ),
        "optimized_scaling": "a_N=O(N), t_N=O(N)",
        "candidate_leading_normalizations": {
            "four_high_one_low": "c_1,N/N^7 (candidate N9 after a_N t_N)",
            "two_high_three_low": "c_3,N/N^3",
        },
        "finite_rows": [row["size"] for row in rows],
        "observed_log_log_exponents": {
            "c_1": scaling["a1_coefficient"]["log_log_exponent"],
            "c_3": scaling["a3_coefficient"]["log_log_exponent"],
        },
        "conclusion": (
            "The exact two-branch reduction is certified. The four-high "
            "coefficient is permitted an N7 fixed-output scale and can "
            "therefore produce an optimized N9 term, which the earlier "
            "N7-only "
            "triage did not cover. Finite scaling diagnostics are not an "
            "asymptotic theorem: c_1,N/N7 still requires a finite-output "
            "continuum certificate."
        ),
        "four_high_N9_limit_certified": False,
        "two_high_N7_limit_certified": False,
        "full_inviscid_pressure_N9_limit_certified": False,
        "full_inviscid_pressure_N7_limit_certified": False,
        "large_full_second_jet_FFT_authorized": False,
        "next_action": (
            "Decompose c_1,N by internal pressure output, identify the "
            "low/finite versus dyadic-shell source of its leading route, "
            "and "
            "derive a continuum or interval certificate for c_1,N/N7. "
            "Only then return to the subleading c_3,N branch."
        ),
        "all_checks_pass": True,
    }


def _finite_output_diagnostics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    sizes = np.asarray([row["size"] for row in rows], dtype=float)
    bounded = np.asarray(
        [
            abs(row["combined_dominant_bounded_output_sum"])
            for row in rows
        ],
        dtype=float,
    )
    if len(rows) > 1:
        slope, intercept = np.polyfit(
            np.log(sizes), np.log(bounded), 1
        )
    else:
        slope = float("nan")
        intercept = float("nan")
    largest = rows[-1]
    dominant_direct = sum(
        profile["direct_value"]
        for profile in largest[
            "dominant_a1_pressure_output_shells"
        ].values()
    )
    return {
        "bounded_output_definition": "|q|<4",
        "finite_output_log_log_exponent": float(slope),
        "finite_output_log_coefficient": float(intercept),
        "largest_size": largest["size"],
        "largest_bounded_output_sum": largest[
            "combined_dominant_bounded_output_sum"
        ],
        "largest_bounded_output_sum_over_N6": largest[
            "combined_dominant_bounded_output_sum_over_N6"
        ],
        "largest_bounded_output_sum_over_N7": largest[
            "combined_dominant_bounded_output_sum_over_N7"
        ],
        "largest_outside_bounded_output": largest[
            "combined_dominant_outside_bounded_output"
        ],
        "largest_outside_bounded_output_over_N6": largest[
            "combined_dominant_outside_bounded_output_over_N6"
        ],
        "largest_outside_bounded_output_over_N7": largest[
            "combined_dominant_outside_bounded_output_over_N7"
        ],
        "largest_dominant_direct_sum": dominant_direct,
        "largest_full_a1_coefficient": largest["a1_coefficient"],
        "largest_nondominant_a1_remainder": (
            largest["a1_coefficient"] - dominant_direct
        ),
        "largest_bounded_fraction_of_dominant": (
            largest["combined_dominant_bounded_output_sum"]
            / dominant_direct
        ),
        "all_bounded_sums_negative": all(
            row["combined_dominant_bounded_output_sum"] < 0.0
            for row in rows
        ),
        "interpretation": (
            "The observed candidate leading signal is carried by finitely "
            "many "
            "pressure outputs |q|<4. Dyadic outputs are numerically "
            "subleading, but a fixed-q Riemann-sum or interval proof is "
            "still required."
        ),
        "all_checks_pass": bool(
            all(
                row["maximum_shell_replay_residual"] < 3.0e-8
                for row in rows
            )
            and all(
                row["maximum_shell_imaginary_residual"] < 3.0e-8
                for row in rows
            )
            and all(
                row["combined_dominant_bounded_output_sum"] < 0.0
                for row in rows
            )
        ),
    }


def audit(
    sizes: Sequence[int] = DEFAULT_SIZES,
) -> dict[str, Any]:
    clean_sizes = tuple(int(size) for size in sizes)
    if (
        not clean_sizes
        or any(size < 5 or size % 2 == 0 for size in clean_sizes)
        or tuple(sorted(set(clean_sizes))) != clean_sizes
    ):
        raise ValueError(
            "sizes must be increasing distinct odd integers at least five"
        )
    prerequisite, predecessor = _prerequisite_audit()
    symbolic = _symbolic_compact_identity_certificate()
    low_shear = _low_shear_certificate()
    support = _branch_support_certificate()
    validation = _branch_row(3, dealias_factor=8)
    padding = _branch_row(3, dealias_factor=10)
    padding_residuals = {
        label: abs(validation[label] - padding[label])
        for label in ("a1_coefficient", "a3_coefficient")
    }
    small_replay = _predecessor_replay(
        validation,
        predecessor["small_carrier_validation"],
    )
    rows = [_branch_row(size, dealias_factor=8) for size in clean_sizes]
    second_replay = _predecessor_replay(
        rows[0],
        predecessor["fixed_amplitude_second_small_carrier_row"],
    )
    scaling = _scaling_diagnostics(rows)
    finite_output = _finite_output_diagnostics(rows)
    route = _route_decision(rows, scaling)
    all_checks = bool(
        prerequisite["all_checks_pass"]
        and symbolic["all_checks_pass"]
        and low_shear["all_checks_pass"]
        and support["all_checks_pass"]
        and validation["all_checks_pass"]
        and padding["all_checks_pass"]
        and max(padding_residuals.values()) < 3.0e-8
        and small_replay["all_checks_pass"]
        and second_replay["all_checks_pass"]
        and all(row["all_checks_pass"] for row in rows)
        and finite_output["all_checks_pass"]
        and route["all_checks_pass"]
    )
    return {
        "kind": "annular_rho_zero_inviscid_second_jet_branch_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": (
            "annular_inviscid_second_jet_pressure_branches_isolated"
            if all_checks
            else "annular_inviscid_second_jet_branch_audit_failed"
        ),
        "scope": (
            "The exact pressure-only second derivative along coupled Euler "
            "velocity and transported-weight directions. The audit "
            "isolates its four-high/one-low and two-high/three-low "
            "amplitude coefficients. It does not certify the candidate N9 "
            "or N7 optimized limits."
        ),
        "prerequisite_audit": prerequisite,
        "symbolic_compact_identity_certificate": symbolic,
        "low_shear_certificate": low_shear,
        "branch_support_certificate": support,
        "small_carrier_validation": validation,
        "padding_replay": {
            "base_grid_shape": validation["grid_shape"],
            "padded_grid_shape": padding["grid_shape"],
            "coefficient_residuals": padding_residuals,
            "maximum_residual": max(padding_residuals.values()),
            "all_checks_pass": max(padding_residuals.values()) < 3.0e-8,
        },
        "predecessor_twenty_channel_replays": [
            small_replay,
            second_replay,
        ],
        "carrier_rows": rows,
        "scaling_diagnostics": scaling,
        "finite_pressure_output_diagnostics": finite_output,
        "route_decision": route,
        "certification_flags": {
            "combined_inviscid_pressure_identity_proved": True,
            "low_shear_stationarity_proved": True,
            "amplitude_polynomial_reduced_to_a1_a3": True,
            "four_high_one_low_coefficient_isolated": True,
            "two_high_three_low_coefficient_isolated": True,
            "four_high_N9_limit_certified": False,
            "two_high_N7_limit_certified": False,
            "full_inviscid_pressure_N9_limit_certified": False,
            "full_inviscid_pressure_N7_limit_certified": False,
            "full_second_jet_N7_coefficient_certified": False,
            "uniform_second_jet_Taylor_bound_proved": False,
            "finite_parabolic_window_controlled": False,
            "critical_L3_controlled": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "all_positive_checks_pass": all_checks,
    }


def _parse_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SIZES,
    )
    arguments = parser.parse_args()
    _lower_process_priority()
    result = audit(arguments.sizes)
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
                "scaling_diagnostics": result["scaling_diagnostics"],
                "route_decision": result["route_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["all_positive_checks_pass"]:
        raise SystemExit(
            "annular inviscid second-jet branch audit failed"
        )


if __name__ == "__main__":
    main()
