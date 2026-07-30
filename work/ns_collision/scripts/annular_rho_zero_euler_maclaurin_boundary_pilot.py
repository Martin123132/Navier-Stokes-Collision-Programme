"""Pilot the explicit h^2 Euler-Maclaurin face correction for L_EE."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from annular_rho_zero_continuum_convolution_quadrature import (
    _atomic_json,
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
    _profile_coefficients,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_euler_maclaurin_boundary_pilot_v1.json"
)
PREREQUISITE = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_direct_continuum_quadrature_v1.json"
)
ALGORITHM_REVISION = "annular-rho-zero-euler-maclaurin-boundary-pilot-v1"
DEFAULT_SIZES = (8, 16, 32)
Array = np.ndarray


def _boundary_source_samples(size: int) -> Array:
    x = 2.0 + np.arange(size + 1, dtype=float) / size
    y = -0.5 + np.arange(size + 1, dtype=float) / size
    z = y.copy()
    sine_x = np.sin(math.pi * (x - 2.0))
    cosine_y = np.cos(math.pi * y)
    cosine_z = np.cos(math.pi * z)
    sine_x[[0, -1]] = 0.0
    cosine_y[[0, -1]] = 0.0
    cosine_z[[0, -1]] = 0.0

    xx = x[:, None, None]
    yy = y[None, :, None]
    zz = z[None, None, :]
    radius_squared = xx * xx + yy * yy + zz * zz
    inverse_radius_cubed = radius_squared ** -1.5
    geometric = np.stack(
        (
            -xx * zz * inverse_radius_cubed
            + np.zeros((size + 1, size + 1, size + 1)),
            -yy * zz * inverse_radius_cubed
            + np.zeros((size + 1, size + 1, size + 1)),
            (xx * xx + yy * yy) * inverse_radius_cubed
            + np.zeros((size + 1, size + 1, size + 1)),
        ),
        axis=0,
    )
    source = np.zeros_like(geometric)

    derivative_x = (
        math.pi
        * np.cos(math.pi * (x - 2.0))[:, None, None]
        * cosine_y[None, :, None]
        * cosine_z[None, None, :]
    )
    derivative_y = (
        -math.pi
        * sine_x[:, None, None]
        * np.sin(math.pi * y)[None, :, None]
        * cosine_z[None, None, :]
    )
    derivative_z = (
        -math.pi
        * sine_x[:, None, None]
        * cosine_y[None, :, None]
        * np.sin(math.pi * z)[None, None, :]
    )
    source[:, 0, :, :] -= (
        derivative_x[0, :, :] * geometric[:, 0, :, :] / 12.0
    )
    source[:, -1, :, :] += (
        derivative_x[-1, :, :] * geometric[:, -1, :, :] / 12.0
    )
    source[:, :, 0, :] -= (
        derivative_y[:, 0, :] * geometric[:, :, 0, :] / 12.0
    )
    source[:, :, -1, :] += (
        derivative_y[:, -1, :] * geometric[:, :, -1, :] / 12.0
    )
    source[:, :, :, 0] -= (
        derivative_z[:, :, 0] * geometric[:, :, :, 0] / 12.0
    )
    source[:, :, :, -1] += (
        derivative_z[:, :, -1] * geometric[:, :, :, -1] / 12.0
    )
    return source


def _sample_coefficients(
    samples: Array,
    size: int,
    shape: tuple[int, int, int],
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
    coefficients[(slice(None), *positive)] = samples
    coefficients[(slice(None), *negative)] = samples
    return coefficients


def _boundary_source_coefficients(
    size: int,
    shape: tuple[int, int, int],
) -> Array:
    # A surface measure h^2 sum b_face is represented in the volume
    # convention h^3 sum b_grid by b_grid=b_face/h=N*b_face.
    source = size * _boundary_source_samples(size)
    return _sample_coefficients(source, size, shape)


def _geometric_profile(
    x: Array,
    y: Array,
    z: Array,
) -> tuple[Array, Array]:
    xx = x[:, None, None]
    yy = y[None, :, None]
    zz = z[None, None, :]
    radius_squared = xx * xx + yy * yy + zz * zz
    inverse_radius_cubed = radius_squared ** -1.5
    geometric = np.stack(
        (
            -xx * zz * inverse_radius_cubed
            + np.zeros((x.size, y.size, z.size)),
            -yy * zz * inverse_radius_cubed
            + np.zeros((x.size, y.size, z.size)),
            (xx * xx + yy * yy) * inverse_radius_cubed
            + np.zeros((x.size, y.size, z.size)),
        ),
        axis=0,
    )
    return geometric, radius_squared


def _geometric_second_derivative(
    geometric: Array,
    radius_squared: Array,
    x: Array,
    y: Array,
    z: Array,
    axis: int,
) -> Array:
    coordinates = (
        x[:, None, None],
        y[None, :, None],
        z[None, None, :],
    )
    coordinate = coordinates[axis]
    xx, _, zz = coordinates
    inverse_r3 = radius_squared ** -1.5
    inverse_r5 = radius_squared ** -2.5
    inverse_r7 = radius_squared ** -3.5
    output = np.empty_like(geometric)
    for component in range(3):
        delta_component_z = 1.0 if component == 2 else 0.0
        component_coordinate = coordinates[component]
        product = component_coordinate * zz
        product_first = (
            (zz if component == axis else 0.0)
            + (component_coordinate if axis == 2 else 0.0)
        )
        product_second = (
            2.0 if component == axis and axis == 2 else 0.0
        )
        output[component] = (
            delta_component_z
            * (-inverse_r3 + 3.0 * coordinate**2 * inverse_r5)
            - product_second * inverse_r3
            + 6.0 * product_first * coordinate * inverse_r5
            + 3.0 * product * inverse_r5
            - 15.0 * product * coordinate**2 * inverse_r7
        )
    return output


def _sixth_order_corrected_samples(size: int) -> Array:
    x = 2.0 + np.arange(size + 1, dtype=float) / size
    y = -0.5 + np.arange(size + 1, dtype=float) / size
    z = y.copy()
    sine_x = np.sin(math.pi * (x - 2.0))
    cosine_y = np.cos(math.pi * y)
    cosine_z = np.cos(math.pi * z)
    sine_x[[0, -1]] = 0.0
    cosine_y[[0, -1]] = 0.0
    cosine_z[[0, -1]] = 0.0
    scalar_factors = (sine_x, cosine_y, cosine_z)
    scalar_first = (
        math.pi * np.cos(math.pi * (x - 2.0)),
        -math.pi * np.sin(math.pi * y),
        -math.pi * np.sin(math.pi * z),
    )
    geometric, radius_squared = _geometric_profile(x, y, z)
    scalar = (
        sine_x[:, None, None]
        * cosine_y[None, :, None]
        * cosine_z[None, None, :]
    )
    corrected = scalar[None, :, :, :] * geometric
    h = 1.0 / size

    # Remove h^2 A_j and h^4 C_j from each one-dimensional
    # Euler-Maclaurin factor. At a zero face,
    # partial_j^3(s_j q)=s_j'*(-pi^2 q+3 partial_j^2 q).
    first_face_source = _boundary_source_samples(size)
    corrected -= h * first_face_source
    for axis in range(3):
        geometric_second = _geometric_second_derivative(
            geometric,
            radius_squared,
            x,
            y,
            z,
            axis,
        )
        other_axes = [value for value in range(3) if value != axis]
        derivative_scalar = np.ones(
            (x.size, y.size, z.size),
            dtype=float,
        )
        axis_shape = [1, 1, 1]
        axis_shape[axis] = scalar_first[axis].size
        derivative_scalar *= scalar_first[axis].reshape(axis_shape)
        for other_axis in other_axes:
            factor = scalar_factors[other_axis]
            reshape = [1, 1, 1]
            reshape[other_axis] = factor.size
            derivative_scalar *= factor.reshape(reshape)
        third_derivative = derivative_scalar[None, :, :, :] * (
            -math.pi**2 * geometric + 3.0 * geometric_second
        )
        for index, orientation in ((0, -1.0), (-1, 1.0)):
            face = [slice(None), slice(None), slice(None), slice(None)]
            face[axis + 1] = index
            face_tuple = tuple(face)
            corrected[face_tuple] += (
                h**3
                * orientation
                * third_derivative[face_tuple]
                / 720.0
            )

    # Tensoring the corrected one-dimensional rules contributes the
    # positive h^4 A_j A_k edge terms.
    for first_axis, second_axis in ((0, 1), (0, 2), (1, 2)):
        remaining_axis = (
            {0, 1, 2} - {first_axis, second_axis}
        ).pop()
        remaining_factor = scalar_factors[remaining_axis]
        remaining_shape = [1, 1, 1]
        remaining_shape[remaining_axis] = remaining_factor.size
        first_shape = [1, 1, 1]
        first_shape[first_axis] = scalar_first[first_axis].size
        second_shape = [1, 1, 1]
        second_shape[second_axis] = scalar_first[second_axis].size
        mixed_scalar = (
            scalar_first[first_axis].reshape(first_shape)
            * scalar_first[second_axis].reshape(second_shape)
            * remaining_factor.reshape(remaining_shape)
        )
        mixed_derivative = (
            mixed_scalar[None, :, :, :] * geometric
        )
        for first_index, first_orientation in ((0, -1.0), (-1, 1.0)):
            for second_index, second_orientation in (
                (0, -1.0),
                (-1, 1.0),
            ):
                edge = [
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                ]
                edge[first_axis + 1] = first_index
                edge[second_axis + 1] = second_index
                edge_tuple = tuple(edge)
                corrected[edge_tuple] += (
                    h**2
                    * first_orientation
                    * second_orientation
                    * mixed_derivative[edge_tuple]
                    / 144.0
                )
    return corrected


def _add_fields(*fields: Array) -> Array:
    output = fields[0].copy()
    for field in fields[1:]:
        output += field
    return output


def _functional_value(
    coefficients: Array,
    values: Array,
    frequencies: tuple[Array, Array, Array],
    wave_number_squared: Array,
    volume: int,
    normalization: float,
) -> tuple[float, float, float]:
    velocity = _euler_quadratic(
        coefficients,
        values,
        frequencies,
        wave_number_squared,
        volume,
    )
    velocity_values = np.stack(
        [_physical(velocity[j], volume) for j in range(3)],
        axis=0,
    )
    acceleration_code = _euler_cross(
        coefficients,
        values,
        velocity,
        velocity_values,
        frequencies,
        wave_number_squared,
        volume,
    )
    alpha = math.sqrt(2.0) / 20.0
    velocity_energy = [
        _component_square_sum(velocity[j]) for j in range(3)
    ]
    acceleration_pairs = [
        _pair_sum(acceleration_code[j], coefficients[j])
        for j in range(3)
    ]
    first = (
        alpha
        * normalization
        * (velocity_energy[2] - velocity_energy[1])
    )
    second = (
        2.0
        * alpha
        * normalization
        * (acceleration_pairs[2] - acceleration_pairs[1])
    )
    del acceleration_code
    del velocity_values
    del velocity
    gc.collect()
    return first, second, first + second


def _row(size: int) -> dict[str, Any]:
    started = time.perf_counter()
    shape = _grid_shape(size)
    volume = int(np.prod(shape))
    frequencies = _frequency_axes(shape)
    wave_number_squared = sum(
        frequency * frequency for frequency in frequencies
    )
    wave_number_squared[0, 0, 0] = 1.0

    profile = _profile_coefficients(size, shape)
    boundary = _boundary_source_coefficients(size, shape)
    profile_values = np.stack(
        [_physical(profile[j], volume) for j in range(3)],
        axis=0,
    )
    boundary_values = np.stack(
        [_physical(boundary[j], volume) for j in range(3)],
        axis=0,
    )

    velocity = _euler_quadratic(
        profile,
        profile_values,
        frequencies,
        wave_number_squared,
        volume,
    )
    cross_profile_boundary = _euler_cross(
        profile,
        profile_values,
        boundary,
        boundary_values,
        frequencies,
        wave_number_squared,
        volume,
    )
    velocity_derivative = 2.0 * cross_profile_boundary
    velocity_values = np.stack(
        [_physical(velocity[j], volume) for j in range(3)],
        axis=0,
    )
    velocity_derivative_values = np.stack(
        [
            _physical(velocity_derivative[j], volume)
            for j in range(3)
        ],
        axis=0,
    )

    acceleration_code = _euler_cross(
        profile,
        profile_values,
        velocity,
        velocity_values,
        frequencies,
        wave_number_squared,
        volume,
    )
    acceleration_boundary_velocity = _euler_cross(
        boundary,
        boundary_values,
        velocity,
        velocity_values,
        frequencies,
        wave_number_squared,
        volume,
    )
    acceleration_profile_velocity_derivative = _euler_cross(
        profile,
        profile_values,
        velocity_derivative,
        velocity_derivative_values,
        frequencies,
        wave_number_squared,
        volume,
    )
    acceleration_derivative = _add_fields(
        acceleration_boundary_velocity,
        acceleration_profile_velocity_derivative,
    )

    normalization = float(size) ** -11
    alpha = math.sqrt(2.0) / 20.0
    velocity_energy = [
        _component_square_sum(velocity[j]) for j in range(3)
    ]
    acceleration_pairs = [
        _pair_sum(acceleration_code[j], profile[j])
        for j in range(3)
    ]
    direct_first = (
        alpha
        * normalization
        * (velocity_energy[2] - velocity_energy[1])
    )
    direct_second = (
        2.0
        * alpha
        * normalization
        * (acceleration_pairs[2] - acceleration_pairs[1])
    )
    first_derivative = (
        2.0
        * alpha
        * normalization
        * (
            _pair_sum(velocity[2], velocity_derivative[2])
            - _pair_sum(velocity[1], velocity_derivative[1])
        )
    )
    pair_derivatives = [
        (
            _pair_sum(acceleration_derivative[j], profile[j])
            + _pair_sum(acceleration_code[j], boundary[j])
        )
        for j in range(3)
    ]
    second_derivative = (
        2.0
        * alpha
        * normalization
        * (pair_derivatives[2] - pair_derivatives[1])
    )
    correction = first_derivative + second_derivative

    trace_derivative = (
        2.0
        * sum(
            _pair_sum(velocity[j], velocity_derivative[j])
            for j in range(3)
        )
        + 2.0 * sum(pair_derivatives)
    )
    trace_derivative_scale = max(
        2.0
        * sum(
            abs(_pair_sum(velocity[j], velocity_derivative[j]))
            for j in range(3)
        ),
        2.0 * sum(abs(value) for value in pair_derivatives),
        1.0,
    )
    direct = direct_first + direct_second

    del acceleration_derivative
    del acceleration_profile_velocity_derivative
    del acceleration_boundary_velocity
    del acceleration_code
    del velocity_derivative_values
    del velocity_values
    del velocity_derivative
    del cross_profile_boundary
    del velocity
    del boundary_values
    del profile_values
    gc.collect()

    corrected_profile = profile - boundary / size**2
    corrected_profile_values = np.stack(
        [
            _physical(corrected_profile[j], volume)
            for j in range(3)
        ],
        axis=0,
    )
    (
        corrected_measure_first,
        corrected_measure_second,
        corrected_measure_value,
    ) = _functional_value(
        corrected_profile,
        corrected_profile_values,
        frequencies,
        wave_number_squared,
        volume,
        normalization,
    )
    sixth_order_profile = _sample_coefficients(
        _sixth_order_corrected_samples(size),
        size,
        shape,
    )
    sixth_order_profile_values = np.stack(
        [
            _physical(sixth_order_profile[j], volume)
            for j in range(3)
        ],
        axis=0,
    )
    (
        sixth_order_first,
        sixth_order_second,
        sixth_order_value,
    ) = _functional_value(
        sixth_order_profile,
        sixth_order_profile_values,
        frequencies,
        wave_number_squared,
        volume,
        normalization,
    )
    row = {
        "size": size,
        "grid_shape": list(shape),
        "direct_quadrature": direct,
        "face_correction_first_component": first_derivative,
        "face_correction_second_component": second_derivative,
        "face_correction_c2": correction,
        "face_corrected_value": direct - correction / size**2,
        "corrected_measure_first_component": corrected_measure_first,
        "corrected_measure_second_component": corrected_measure_second,
        "corrected_measure_quartic_value": corrected_measure_value,
        "sixth_order_corrected_first_component": sixth_order_first,
        "sixth_order_corrected_second_component": sixth_order_second,
        "sixth_order_corrected_quartic_value": sixth_order_value,
        "linearized_vs_full_corrected_measure_difference": (
            direct
            - correction / size**2
            - corrected_measure_value
        ),
        "energy_trace_directional_relative_residual": (
            abs(trace_derivative) / trace_derivative_scale
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    row["all_pilot_checks_pass"] = bool(
        direct < 0.0
        and correction > 0.0
        and row["face_corrected_value"] < 0.0
        and corrected_measure_value < 0.0
        and sixth_order_value < 0.0
        and row["energy_trace_directional_relative_residual"] < 1.0e-9
    )

    del sixth_order_profile_values
    del sixth_order_profile
    del corrected_profile_values
    del corrected_profile
    del boundary
    del profile
    del wave_number_squared
    gc.collect()
    return row


def _write(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["size"]))
    payload = {
        "algorithm_revision": ALGORITHM_REVISION,
        "kind": "annular_rho_zero_euler_maclaurin_boundary_pilot",
        "prerequisite": {
            "path": str(PREREQUISITE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(PREREQUISITE),
        },
        "identity": {
            "one_dimensional_rule": (
                "h sum f(kh)=integral f+h^2/12*(f'(b)-f'(a))+O(h^4)"
            ),
            "face_measure": (
                "mu_2=(1/12) sum_faces orientation*partial_normal(a)"
            ),
            "quartic_correction": "c_2=D L[a](mu_2)",
            "corrected_measure": "mu_h_star=mu_h-h^2*mu_2,h",
            "sixth_order_measure": (
                "tensor product of S-h^2*A-h^4*C, including "
                "the h^4*A_j*A_k edge terms"
            ),
        },
        "rows": ordered,
        "certification": {
            "face_correction_derived": True,
            "h4_remainder_interval_certified": False,
            "continuum_sign_interval_certified": False,
            "reason": (
                "The explicit h^2 coefficient is a structural pilot. "
                "A directed enclosure of the h^4 remainder and FFT "
                "roundoff is still required."
            ),
        },
        "all_pilot_checks_pass": bool(
            ordered and all(row["all_pilot_checks_pass"] for row in ordered)
        ),
    }
    _atomic_json(RESULT, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default=",".join(str(value) for value in DEFAULT_SIZES),
    )
    parser.add_argument("--fresh", action="store_true")
    arguments = parser.parse_args()
    sizes = sorted(
        {
            int(value.strip())
            for value in arguments.sizes.split(",")
            if value.strip()
        }
    )
    if any(size < 2 or size % 2 for size in sizes):
        raise ValueError("sizes must be even integers at least 2")
    _lower_process_priority()

    cached: dict[int, dict[str, Any]] = {}
    if RESULT.exists() and not arguments.fresh:
        existing = json.loads(RESULT.read_text(encoding="utf-8"))
        if existing.get("algorithm_revision") == ALGORITHM_REVISION:
            cached = {
                int(row["size"]): row for row in existing.get("rows", [])
            }
    for size in sizes:
        if size in cached:
            print(f"N={size}: cached", flush=True)
            continue
        print(f"N={size}: computing", flush=True)
        cached[size] = _row(size)
        _write(list(cached.values()))
        print(
            f"N={size}: c2={cached[size]['face_correction_c2']:.16e}, "
            f"corrected={cached[size]['face_corrected_value']:.16e}",
            flush=True,
        )
    _write(list(cached.values()))


if __name__ == "__main__":
    main()
