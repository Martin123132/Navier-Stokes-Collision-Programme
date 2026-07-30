"""Audit an exact two-shear square replacement for the annular sign gate."""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from annular_rho_zero_continuum_convolution_quadrature import (
    _divergence_residual,
    _euler_cross,
    _euler_quadratic,
    _frequency_axes,
    _lower_process_priority,
    _physical,
    _sha256,
)
from annular_rho_zero_direct_continuum_quadrature import (
    _component_square_sum,
    _grid_shape,
    _pair_sum,
    _profile_samples,
)
from separable_annular_pressure_schur_no_go_audit import (
    _family_arrays,
    _shift_slices,
    _vertex_weight_float,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_two_shear_square_gate_audit_v1.json"
)
FIXED_OUTPUT_PREREQUISITE = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_fixed_output_continuum_gate_audit_v1.json"
)
TAIL_PREREQUISITE = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_full_c1_tail_ledger_audit_v1.json"
)
ALGORITHM_REVISION = "annular-two-shear-square-gate-v1"
DEFAULT_SIZES = (8, 16, 32)
DEFAULT_FINITE_SIZES = (5, 9, 13, 17, 25)
ALPHA = math.sqrt(2.0) / 20.0
Array = np.ndarray
Wave = tuple[int, int, int]
Matrix = list[list[Fraction]]

# Each direction is divided by sqrt(2). Both pairs are divergence free.
YZ_SHEAR = ((0, 1, -1), (0, 1, 1))
XY_SHEAR = ((1, -1, 0), (-1, -1, 0))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _weight_coefficient(wave: Wave) -> Fraction:
    if any(abs(component) > 1 for component in wave):
        return Fraction(0)
    value = Fraction(1)
    for component in wave:
        value *= Fraction(1, 2) if component == 0 else Fraction(1, 4)
    return value


def _add(first: Wave, second: Wave, factor: int = 1) -> Wave:
    return tuple(
        first[index] + factor * second[index] for index in range(3)
    )


def _parity(wave: Wave) -> int:
    return 1 if sum(wave) % 2 == 0 else -1


def _active_stencil(low_wave: Wave, scaled_direction: Wave) -> dict[Wave, Fraction]:
    candidates = {
        tuple(
            sign * low_wave[index] + shift[index]
            for index in range(3)
        )
        for sign in (-1, 1)
        for shift in product((-1, 0, 1), repeat=3)
    }
    active: dict[Wave, Fraction] = {}
    for wave in candidates:
        alpha = Fraction(
            sum(
                scaled_direction[index] * wave[index]
                for index in range(3)
            )
        ) * (
            _weight_coefficient(_add(wave, low_wave, factor=-1))
            - _weight_coefficient(_add(wave, low_wave, factor=1))
        )
        if alpha:
            active[wave] = alpha
    return active


def _projector_matrix(active: dict[Wave, Fraction]) -> Matrix:
    matrix = [[Fraction(0) for _ in range(3)] for _ in range(3)]
    for wave, alpha in active.items():
        norm_squared = sum(component * component for component in wave)
        for row in range(3):
            for column in range(3):
                matrix[row][column] += (
                    Fraction(
                        _parity(wave)
                        * wave[row]
                        * wave[column],
                        norm_squared,
                    )
                    * alpha
                )
    return matrix


def _matrix_add(first: Matrix, second: Matrix) -> Matrix:
    return [
        [
            first[row][column] + second[row][column]
            for column in range(3)
        ]
        for row in range(3)
    ]


def _matrix_payload(matrix: Matrix) -> list[list[str]]:
    return [
        [_fraction_text(value) for value in row]
        for row in matrix
    ]


def _strain_matrix(low_wave: Wave, scaled_direction: Wave) -> Matrix:
    return [
        [
            Fraction(
                low_wave[row] * scaled_direction[column]
                + scaled_direction[row] * low_wave[column],
                2,
            )
            for column in range(3)
        ]
        for row in range(3)
    ]


def _matrix_scale(matrix: Matrix, factor: Fraction) -> Matrix:
    return [
        [factor * value for value in row]
        for row in matrix
    ]


def _combine_stencils(
    first: dict[Wave, Fraction],
    second: dict[Wave, Fraction],
) -> dict[Wave, Fraction]:
    combined = dict(first)
    for wave, value in second.items():
        combined[wave] = combined.get(wave, Fraction(0)) + value
    return {wave: value for wave, value in combined.items() if value}


def _stencil_audit() -> dict[str, Any]:
    yz_active = _active_stencil(*YZ_SHEAR)
    xy_active = _active_stencil(*XY_SHEAR)
    combined_active = _combine_stencils(yz_active, xy_active)
    yz_matrix = _projector_matrix(yz_active)
    xy_matrix = _projector_matrix(xy_active)
    combined_matrix = _projector_matrix(combined_active)
    yz_strain = _strain_matrix(*YZ_SHEAR)
    xy_strain = _strain_matrix(*XY_SHEAR)
    combined_strain = _matrix_add(yz_strain, xy_strain)

    expected_yz = [
        [Fraction(0), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(-1, 20), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1, 20)],
    ]
    expected_xy = [
        [Fraction(1, 20), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(-1, 20), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(0)],
    ]
    expected_combined = [
        [Fraction(1, 20), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(-1, 10), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1, 20)],
    ]
    expected_yz_strain = [
        [Fraction(0), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(-1)],
    ]
    expected_xy_strain = [
        [Fraction(-1), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(0)],
    ]
    expected_combined_strain = [
        [Fraction(-1), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(2), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(-1)],
    ]
    maximum_radius_squared = max(
        sum(component * component for component in wave)
        for wave in combined_active
    )
    combined_l1 = sum(abs(value) for value in combined_active.values())
    checks = {
        "each_low_direction_is_divergence_free": all(
            sum(wave[index] * direction[index] for index in range(3)) == 0
            for wave, direction in (YZ_SHEAR, XY_SHEAR)
        ),
        "each_low_wave_has_even_coordinate_sum": all(
            sum(wave) % 2 == 0 for wave, _ in (YZ_SHEAR, XY_SHEAR)
        ),
        "yz_matrix_exact": yz_matrix == expected_yz,
        "xy_matrix_exact": xy_matrix == expected_xy,
        "combined_matrix_exact": combined_matrix == expected_combined,
        "yz_static_strain_exact": yz_strain == expected_yz_strain,
        "xy_static_strain_exact": xy_strain == expected_xy_strain,
        "combined_static_strain_exact": (
            combined_strain == expected_combined_strain
        ),
        "fixed_output_matrix_is_negative_static_strain_over_20": (
            combined_matrix
            == _matrix_scale(combined_strain, Fraction(-1, 20))
        ),
        "combined_stencil_count_exact": len(combined_active) == 58,
        "combined_stencil_l1_exact": combined_l1 == Fraction(3),
        "maximum_radius_squared_exact": maximum_radius_squared == 6,
    }
    return {
        "weight": (
            "w_s=product_j(1/2 if s_j=0 else 1/4), "
            "s in {-1,0,1}^3"
        ),
        "phase": "sigma_q=(-1)^(q_x+q_y+q_z)",
        "low_shears": [
            {
                "label": "yz_original",
                "wave": list(YZ_SHEAR[0]),
                "direction": f"{list(YZ_SHEAR[1])}/sqrt(2)",
                "active_output_count": len(yz_active),
                "sqrt2_times_projector_matrix": _matrix_payload(yz_matrix),
                "static_strain_after_removing_sqrt2": _matrix_payload(
                    yz_strain
                ),
                "actual_projector_matrix": (
                    "(sqrt(2)/40) diag(0,-1,1)"
                ),
            },
            {
                "label": "xy_sign_flipped",
                "wave": list(XY_SHEAR[0]),
                "direction": f"{list(XY_SHEAR[1])}/sqrt(2)",
                "active_output_count": len(xy_active),
                "sqrt2_times_projector_matrix": _matrix_payload(xy_matrix),
                "static_strain_after_removing_sqrt2": _matrix_payload(
                    xy_strain
                ),
                "actual_projector_matrix": (
                    "(sqrt(2)/40) diag(1,-1,0)"
                ),
            },
        ],
        "combined_active_output_count": len(combined_active),
        "combined_active_output_l1_before_dividing_by_sqrt2": (
            _fraction_text(combined_l1)
        ),
        "combined_maximum_radius_squared": maximum_radius_squared,
        "combined_sqrt2_times_projector_matrix": _matrix_payload(
            combined_matrix
        ),
        "combined_static_strain_after_removing_sqrt2": _matrix_payload(
            combined_strain
        ),
        "exact_matrix_relation": "sqrt(2)*Q_*=-S_*/20",
        "combined_actual_projector_matrix": (
            "Q_*=(sqrt(2)/40) diag(1,-2,1)"
        ),
        "checks": checks,
        "all_exact_stencil_checks_pass": all(checks.values()),
    }


def _dominant_profile_samples(
    size: int,
) -> tuple[Array, Array, Array, Array]:
    x, y, z, original = _profile_samples(size)
    profile = np.zeros_like(original)
    profile[2] = original[2]
    profile[0] = (
        -(z[None, None, :] / x[:, None, None]) * profile[2]
    )
    return x, y, z, profile


def _profile_coefficients(
    size: int,
    shape: tuple[int, int, int],
    profile: Array,
) -> Array:
    coefficients = np.zeros((3, *shape), dtype=np.complex128)
    kx = np.arange(2 * size, 3 * size + 1, dtype=int)
    ky = np.arange(-size // 2, size // 2 + 1, dtype=int)
    kz = ky.copy()
    shape_array = np.asarray(shape, dtype=int)
    positive = (
        (kx % shape_array[0])[:, None, None],
        (ky % shape_array[1])[None, :, None],
        (kz % shape_array[2])[None, None, :],
    )
    negative = (
        ((-kx) % shape_array[0])[:, None, None],
        ((-ky) % shape_array[1])[None, :, None],
        ((-kz) % shape_array[2])[None, None, :],
    )
    coefficients[(slice(None), *positive)] = profile
    coefficients[(slice(None), *negative)] = profile
    return coefficients


def _project(vector: Array, wave: Array) -> Array:
    norm_squared = float(np.dot(wave, wave))
    return vector - wave * float(np.dot(wave, vector)) / norm_squared


def _quadrature_row(size: int) -> dict[str, Any]:
    shape = _grid_shape(size)
    volume = int(np.prod(shape))
    frequencies = _frequency_axes(shape)
    wave_number_squared = sum(
        frequency * frequency for frequency in frequencies
    )
    wave_number_squared[0, 0, 0] = 1.0

    x, y, z, profile = _dominant_profile_samples(size)
    sample_divergence = (
        x[:, None, None] * profile[0]
        + y[None, :, None] * profile[1]
        + z[None, None, :] * profile[2]
    )
    profile_coefficients = _profile_coefficients(size, shape, profile)
    profile_values = np.stack(
        [
            _physical(profile_coefficients[component], volume)
            for component in range(3)
        ],
        axis=0,
    )
    velocity_coefficients = _euler_quadratic(
        profile_coefficients,
        profile_values,
        frequencies,
        wave_number_squared,
        volume,
    )
    velocity_values = np.stack(
        [
            _physical(velocity_coefficients[component], volume)
            for component in range(3)
        ],
        axis=0,
    )
    acceleration_coefficients = _euler_cross(
        profile_coefficients,
        profile_values,
        velocity_coefficients,
        velocity_values,
        frequencies,
        wave_number_squared,
        volume,
    )

    h11 = float(size) ** -11
    velocity_energy = np.asarray(
        [
            h11 * _component_square_sum(velocity_coefficients[component])
            for component in range(3)
        ]
    )
    acceleration_pair = np.asarray(
        [
            h11
            * _pair_sum(
                acceleration_coefficients[component],
                profile_coefficients[component],
            )
            for component in range(3)
        ]
    )
    component_curvature = velocity_energy + 2.0 * acceleration_pair
    yz_functional = ALPHA * (
        component_curvature[2] - component_curvature[1]
    )
    xy_functional = ALPHA * (
        component_curvature[0] - component_curvature[1]
    )
    combined_functional = yz_functional + xy_functional
    exact_square_replay = -3.0 * ALPHA * velocity_energy[1]

    # The static HHL limit uses the positive packet D. The four-high
    # convolution and its covariance use the even set K=D union (-D).
    profile_energy = (
        1.0
        / size**3
        * np.asarray(
            [
                np.sum(profile[component] * profile[component])
                for component in range(3)
            ]
        )
    )
    static_yz = ALPHA * (profile_energy[1] - profile_energy[2])
    static_xy = ALPHA * (profile_energy[1] - profile_energy[0])
    static_combined = static_yz + static_xy
    static_negative_norm_replay = -ALPHA * float(np.sum(profile_energy))

    covariance = (
        2.0
        / size**3
        * np.einsum(
            "iabc,jabc->ij",
            profile,
            profile,
            optimize=True,
        )
    )
    integer_direction = np.asarray((0, 1, 1), dtype=int)
    rho = integer_direction.astype(float) / size
    index = tuple(
        (
            integer_direction
            % np.asarray(shape, dtype=int)
        ).tolist()
    )
    quadrature_velocity = (
        -velocity_coefficients[(slice(None), *index)].imag
        / size**4
    )
    leading_velocity = _project(covariance @ rho, rho)
    near_zero_residual = quadrature_velocity - leading_velocity

    trace_scale = max(float(np.sum(velocity_energy)), 1.0)
    energy_trace_relative_residual = (
        abs(float(np.sum(component_curvature))) / trace_scale
    )
    functional_replay_error = abs(
        combined_functional - exact_square_replay
    )
    static_replay_error = abs(
        static_combined - static_negative_norm_replay
    )
    row = {
        "size": size,
        "grid_shape": list(shape),
        "profile_formula": (
            "b=S*(x^2+y^2)/(x*r^3)*(-z,0,x)"
        ),
        "maximum_sample_divergence": float(
            np.max(np.abs(sample_divergence))
        ),
        "maximum_spectral_divergence": _divergence_residual(
            profile_coefficients,
            frequencies,
        ),
        "maximum_profile_y_component": float(
            np.max(np.abs(profile[1]))
        ),
        "positive_packet_profile_energy_components": (
            profile_energy.tolist()
        ),
        "static_yz_shear": float(static_yz),
        "static_xy_shear": float(static_xy),
        "static_combined": float(static_combined),
        "static_negative_norm_replay": float(
            static_negative_norm_replay
        ),
        "static_replay_error": float(static_replay_error),
        "velocity_energy_components": velocity_energy.tolist(),
        "acceleration_pair_components": acceleration_pair.tolist(),
        "component_energy_curvatures": component_curvature.tolist(),
        "energy_trace_relative_residual": float(
            energy_trace_relative_residual
        ),
        "yz_fixed_output_functional": float(yz_functional),
        "xy_fixed_output_functional": float(xy_functional),
        "combined_fixed_output_functional": float(
            combined_functional
        ),
        "negative_square_replay": float(exact_square_replay),
        "functional_replay_error": float(functional_replay_error),
        "covariance_matrix": covariance.tolist(),
        "near_zero_nonvanishing_replay": {
            "rho": rho.tolist(),
            "quadrature_velocity": quadrature_velocity.tolist(),
            "leading_velocity": leading_velocity.tolist(),
            "absolute_residual": float(np.linalg.norm(near_zero_residual)),
            "quadrature_y_component_negative": bool(
                quadrature_velocity[1] < 0.0
            ),
            "leading_y_component_negative": bool(
                leading_velocity[1] < 0.0
            ),
        },
    }
    row["all_numerical_checks_pass"] = bool(
        row["maximum_sample_divergence"] < 1.0e-14
        and row["maximum_spectral_divergence"] < 1.0e-9
        and row["maximum_profile_y_component"] == 0.0
        and static_combined < 0.0
        and static_replay_error < 1.0e-18
        and velocity_energy[1] > 0.0
        and combined_functional < 0.0
        and functional_replay_error < 1.0e-18
        and energy_trace_relative_residual < 1.0e-10
        and row["near_zero_nonvanishing_replay"][
            "quadrature_y_component_negative"
        ]
        and row["near_zero_nonvanishing_replay"][
            "leading_y_component_negative"
        ]
    )
    return row


def _modified_finite_packet(size: int) -> tuple[Array, Array, Array]:
    if size < 3 or size % 2 == 0:
        raise ValueError("finite packet sizes must be odd integers at least 3")
    waves, _, parity = _family_arrays((size, size, size), 2 * size)
    first = np.arange(1, size + 1, dtype=float)[:, None, None]
    second = np.arange(1, size + 1, dtype=float)[None, :, None]
    third = np.arange(1, size + 1, dtype=float)[None, None, :]
    sine_profile = (
        np.sin(math.pi * first / (size + 1))
        * np.sin(math.pi * second / (size + 1))
        * np.sin(math.pi * third / (size + 1))
    )
    scalar = parity * sine_profile
    x = waves[..., 0]
    y = waves[..., 1]
    z = waves[..., 2]
    radius_squared = np.sum(waves * waves, axis=-1)
    factor = (
        (x * x + y * y)
        / (x * radius_squared ** 1.5)
    )
    velocity = np.zeros_like(waves)
    velocity[..., 0] = -scalar * factor * z
    velocity[..., 2] = scalar * factor * x
    return waves, velocity, parity


def _static_hhl_load(
    waves: Array,
    velocity: Array,
    low_wave_tuple: Wave,
    scaled_direction: Wave,
) -> tuple[float, float]:
    shape = waves.shape[:3]
    low_wave_base = np.asarray(low_wave_tuple, dtype=int)
    low_direction = np.asarray(scaled_direction, dtype=float) / math.sqrt(2.0)
    load = 0.0j
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
            first_velocity = velocity[first_slice]
            second_velocity = velocity[second_slice]
            difference_float = difference_array.astype(float)
            norm_squared = float(
                np.dot(difference_float, difference_float)
            )
            if norm_squared == 0.0:
                continue
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
            gradient = (
                -1j
                * output.astype(float)
                * _vertex_weight_float(output_wave)
            )
            load += pressure * np.dot(low_value, gradient)
    return float(load.real), float(abs(load.imag))


def _finite_static_row(size: int) -> dict[str, Any]:
    waves, velocity, parity = _modified_finite_packet(size)
    yz_load, yz_imaginary = _static_hhl_load(
        waves,
        velocity,
        *YZ_SHEAR,
    )
    xy_load, xy_imaginary = _static_hhl_load(
        waves,
        velocity,
        *XY_SHEAR,
    )
    total = yz_load + xy_load
    divergence = np.sum(waves * velocity, axis=-1)
    row = {
        "size": size,
        "positive_mode_count": int(size**3),
        "coefficient_scale": "Theta(N^-1)",
        "wave_box": {
            "x": [2 * size, 3 * size - 1],
            "y": [-(size - 1) // 2, (size - 1) // 2],
            "z": [-(size - 1) // 2, (size - 1) // 2],
        },
        "yz_pressure_hhl_load": yz_load,
        "xy_pressure_hhl_load": xy_load,
        "combined_pressure_hhl_load": total,
        "combined_pressure_hhl_load_over_N": total / size,
        "maximum_divergence_residual": float(
            np.max(np.abs(divergence))
        ),
        "maximum_imaginary_residual": max(
            yz_imaginary,
            xy_imaginary,
        ),
        "parity_values": sorted(
            float(value) for value in np.unique(parity)
        ),
    }
    row["all_finite_static_checks_pass"] = bool(
        total < 0.0
        and row["combined_pressure_hhl_load_over_N"] < 0.0
        and row["maximum_divergence_residual"] < 1.0e-12
        and row["maximum_imaginary_residual"] < 1.0e-12
        and row["parity_values"] == [-1.0, 1.0]
    )
    return row


def _static_continuum_reference(order: int = 80) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x = 2.5 + 0.5 * nodes[:, None, None]
    y = 0.5 * nodes[None, :, None]
    z = 0.5 * nodes[None, None, :]
    tensor_weights = (
        0.5
        * weights[:, None, None]
        * 0.5
        * weights[None, :, None]
        * 0.5
        * weights[None, None, :]
    )
    radius_squared = x * x + y * y + z * z
    scalar = (
        np.sin(math.pi * (x - 2.0))
        * np.cos(math.pi * y)
        * np.cos(math.pi * z)
    )
    b_z = scalar * (x * x + y * y) / radius_squared ** 1.5
    b_x = -(z / x) * b_z
    energy = float(
        np.sum(tensor_weights * (b_x * b_x + b_z * b_z))
    )
    return -ALPHA * energy


def _parse_sizes(text: str) -> tuple[int, ...]:
    sizes = tuple(int(value.strip()) for value in text.split(",") if value)
    if not sizes or any(size < 2 or size % 2 for size in sizes):
        raise ValueError("sizes must be even integers at least 2")
    return sizes


def _parse_finite_sizes(text: str) -> tuple[int, ...]:
    sizes = tuple(int(value.strip()) for value in text.split(",") if value)
    if not sizes or any(size < 3 or size % 2 == 0 for size in sizes):
        raise ValueError("finite sizes must be odd integers at least 3")
    return sizes


def _write_result(
    sizes: Iterable[int],
    finite_sizes: Iterable[int],
) -> dict[str, Any]:
    stencil = _stencil_audit()
    rows = [_quadrature_row(size) for size in sizes]
    finite_static_rows = [
        _finite_static_row(size) for size in finite_sizes
    ]
    static_reference = _static_continuum_reference()
    for row in finite_static_rows:
        row["continuum_reference"] = static_reference
        row["normalized_error_from_reference"] = (
            row["combined_pressure_hhl_load_over_N"]
            - static_reference
        )
    payload = {
        "algorithm_revision": ALGORITHM_REVISION,
        "prerequisites": {
            "fixed_output_gate": (
                FIXED_OUTPUT_PREREQUISITE.relative_to(ROOT).as_posix()
            ),
            "fixed_output_gate_sha256": _sha256(
                FIXED_OUTPUT_PREREQUISITE
            ),
            "full_c1_tail_ledger": (
                TAIL_PREREQUISITE.relative_to(ROOT).as_posix()
            ),
            "full_c1_tail_ledger_sha256": _sha256(TAIL_PREREQUISITE),
        },
        "exact_low_stencil": stencil,
        "modified_high_profile": {
            "positive_domain": "D=[2,3]x[-1/2,1/2]^2",
            "scalar": (
                "S=sin(pi(x-2))*cos(pi*y)*cos(pi*z)"
            ),
            "formula": (
                "b=S*(x^2+y^2)/(x*(x^2+y^2+z^2)^(3/2))"
                "*(-z,0,x)"
            ),
            "even_extension_to_negative_domain": True,
            "zero_extension": True,
            "pointwise_divergence_identity": (
                "xi dot b=(-x*z+x*z)"
                "*S*(x^2+y^2)/(x*r^3)=0"
            ),
            "middle_component_identically_zero": True,
            "face_vanishing": True,
            "zero_extension_is_H1": True,
        },
        "analytic_square_reduction": {
            "component_tensor": (
                "C_j=int |v_j|^2+2 int gamma_j*b_j, "
                "where gamma is the Euler second Taylor coefficient"
            ),
            "energy_trace_identity": "C_x+C_y+C_z=0",
            "missing_component_identity": (
                "b_y=0 implies C_y=int |v_y|^2"
            ),
            "combined_functional": (
                "L_*=alpha[(C_z-C_y)+(C_x-C_y)]"
            ),
            "exact_reduction": (
                "L_*=-3*alpha*int |v_y|^2, alpha=sqrt(2)/20"
            ),
            "covariance_argument": (
                "M=int b tensor b=diag(m_x,0,m_z), m_z>0. "
                "For rho=t(0,1,1), "
                "(P_rho M rho)_y=-m_z*t/2. Since "
                "A(rho)=M+O(|rho|^2), "
                "v_y(t(0,1,1))=-m_z*t/2+O(t^3), so v_y is "
                "not identically zero."
            ),
            "strict_nonvanishing_is_analytic": True,
            "continuum_sign_conclusion": (
                "L_*=-(3*sqrt(2)/20)*||v_y||_2^2<0"
            ),
            "static_load_reduction": (
                "b_0,*=alpha[(E_y-E_z)+(E_y-E_x)]"
                "=-alpha*||b||_L2(D)^2<0"
            ),
        },
        "finite_static_packet_replay": {
            "packet_formula": (
                "hhat_N(k)=parity*sine_N(k)"
                "*(k_x^2+k_y^2)/(k_x*|k|^3)*(-k_z,0,k_x)"
            ),
            "positive_packet_continuum_reference_gauss80": (
                static_reference
            ),
            "continuum_reference_is_sign_replay_not_error_bound": True,
            "rows": finite_static_rows,
            "all_finite_static_rows_pass": all(
                row["all_finite_static_checks_pass"]
                for row in finite_static_rows
            ),
        },
        "rows": rows,
        "certification": {
            "low_stencil_matrix_exact_rational": True,
            "modified_profile_divergence_free_analytic": True,
            "modified_static_continuum_sign_analytic": True,
            "modified_four_high_continuum_sign_analytic": True,
            "strict_nonzero_analytic": True,
            "modified_finite_static_packet_defined": True,
            "modified_finite_static_hhl_replayed": True,
            "modified_static_limit_quantitative_remainder_ported": False,
            "fft_rows_are_replays_not_sign_evidence": True,
            "original_single_shear_interval_branch_superseded": False,
            "modified_finite_c1_tail_ledger_ported": False,
            "modified_first_and_second_jet_optimizer_ported": False,
            "navier_stokes_clay_problem_solved": False,
        },
        "remaining_obligation": (
            "Port the finite packet definition, static optimizer, full c1 "
            "tail ledger, and parabolic-window bounds to the modified "
            "divergence-free profile and two-mode low field. The exact "
            "continuum sign no longer needs interval quadrature on this "
            "branch, but all finite-N and time-window estimates must be "
            "rechecked before it can replace the original witness."
        ),
    }
    payload["all_route_guard_checks_pass"] = bool(
        stencil["all_exact_stencil_checks_pass"]
        and all(row["all_numerical_checks_pass"] for row in rows)
        and payload["finite_static_packet_replay"][
            "all_finite_static_rows_pass"
        ]
        and payload["certification"][
            "modified_four_high_continuum_sign_analytic"
        ]
        and not payload["certification"]["navier_stokes_clay_problem_solved"]
    )
    _atomic_json(RESULT, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default=",".join(str(size) for size in DEFAULT_SIZES),
    )
    parser.add_argument(
        "--finite-sizes",
        default=",".join(str(size) for size in DEFAULT_FINITE_SIZES),
    )
    arguments = parser.parse_args()
    _lower_process_priority()
    payload = _write_result(
        _parse_sizes(arguments.sizes),
        _parse_finite_sizes(arguments.finite_sizes),
    )
    print(
        json.dumps(
            {
                "result": RESULT.relative_to(ROOT).as_posix(),
                "result_sha256": _sha256(RESULT),
                "sizes": [row["size"] for row in payload["rows"]],
                "finite_sizes": [
                    row["size"]
                    for row in payload["finite_static_packet_replay"]["rows"]
                ],
                "combined_values": [
                    row["combined_fixed_output_functional"]
                    for row in payload["rows"]
                ],
                "all_route_guard_checks_pass": payload[
                    "all_route_guard_checks_pass"
                ],
                "analytic_continuum_sign": payload["certification"][
                    "modified_four_high_continuum_sign_analytic"
                ],
                "finite_tail_ported": payload["certification"][
                    "modified_finite_c1_tail_ledger_ported"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
