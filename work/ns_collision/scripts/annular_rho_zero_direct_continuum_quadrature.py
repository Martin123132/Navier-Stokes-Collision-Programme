"""Direct tensor-trapezoid audit of the annular continuum functional."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from scipy.fft import next_fast_len

from annular_rho_zero_continuum_convolution_quadrature import (
    _atomic_json,
    _divergence_residual,
    _euler_cross,
    _euler_quadratic,
    _frequency_axes,
    _lower_process_priority,
    _physical,
    _sha256,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_direct_continuum_quadrature_v1.json"
)
PREREQUISITE = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_full_c1_tail_ledger_audit_v1.json"
)
ALGORITHM_REVISION = "annular-rho-zero-direct-continuum-quadrature-v1"
DEFAULT_SIZES = (8, 16, 32)
Array = np.ndarray


def _grid_shape(size: int) -> tuple[int, int, int]:
    if size < 2 or size % 2:
        raise ValueError("size must be an even integer at least 2")
    return (
        next_fast_len(12 * size + 1),
        next_fast_len(2 * size + 1),
        next_fast_len(2 * size + 1),
    )


def _profile_samples(size: int) -> tuple[Array, Array, Array, Array]:
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
    scalar = (
        sine_x[:, None, None]
        * cosine_y[None, :, None]
        * cosine_z[None, None, :]
    )
    profile = np.stack(
        (
            -scalar * xx * zz * inverse_radius_cubed,
            -scalar * yy * zz * inverse_radius_cubed,
            scalar * (xx * xx + yy * yy) * inverse_radius_cubed,
        ),
        axis=0,
    )
    return x, y, z, profile


def _profile_coefficients(
    size: int,
    shape: tuple[int, int, int],
) -> Array:
    _, _, _, profile = _profile_samples(size)
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


def _component_square_sum(values: Array) -> float:
    return float(
        np.sum(values.real * values.real + values.imag * values.imag)
    )


def _pair_sum(left: Array, right: Array) -> float:
    return float(
        np.sum((left * np.conjugate(right)).real)
    )


def _origin_cusp_replay(
    size: int,
    shape: tuple[int, int, int],
    velocity_coefficients: Array,
) -> dict[str, Any]:
    _, _, _, profile = _profile_samples(size)
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
    directions = (
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
    )
    rows = []
    shape_array = np.asarray(shape, dtype=int)
    for direction in directions:
        integer_wave = np.asarray(direction, dtype=int)
        rho = integer_wave.astype(float) / size
        radius_squared = float(np.dot(rho, rho))
        projected_covariance = covariance @ rho
        projected_covariance -= (
            rho
            * float(np.dot(rho, projected_covariance))
            / radius_squared
        )
        index = tuple((integer_wave % shape_array).tolist())
        quadrature_velocity = (
            -velocity_coefficients[(slice(None), *index)].imag
            / size**4
        )
        residual = quadrature_velocity - projected_covariance
        radius = math.sqrt(radius_squared)
        rows.append(
            {
                "integer_direction": list(direction),
                "rho": rho.tolist(),
                "quadrature_velocity": quadrature_velocity.tolist(),
                "leading_cusp_velocity": projected_covariance.tolist(),
                "absolute_residual": float(np.linalg.norm(residual)),
                "residual_over_rho_cubed": (
                    float(np.linalg.norm(residual)) / radius**3
                ),
            }
        )
    off_diagonal = covariance - np.diag(np.diag(covariance))
    return {
        "covariance_matrix_trapezoid": covariance.tolist(),
        "maximum_covariance_off_diagonal": float(
            np.max(np.abs(off_diagonal))
        ),
        "identity": (
            "v(rho)=P_rho*M*rho+O(|rho|^3), "
            "M=integral a tensor a"
        ),
        "rows": rows,
        "maximum_residual_over_rho_cubed": max(
            row["residual_over_rho_cubed"] for row in rows
        ),
    }


def _sector_diagnostic(
    density: Array,
    x_frequencies: Array,
    size: int,
) -> dict[str, Any]:
    x_values = x_frequencies.reshape(-1)
    sectors = {
        "mixed_sign_output": np.abs(x_values) <= size,
        "positive_same_sign_output": (
            (x_values >= 4 * size) & (x_values <= 6 * size)
        ),
        "negative_same_sign_output": (
            (x_values <= -4 * size) & (x_values >= -6 * size)
        ),
        "positive_packet_output": (
            (x_values >= 2 * size) & (x_values <= 3 * size)
        ),
        "negative_packet_output": (
            (x_values <= -2 * size) & (x_values >= -3 * size)
        ),
    }
    output: dict[str, Any] = {}
    for label, mask in sectors.items():
        values = density[mask, :, :]
        positive = values[values > 0.0]
        negative = values[values < 0.0]
        output[label] = {
            "signed_sum": float(np.sum(values)),
            "positive_mass": float(np.sum(positive)),
            "negative_mass": float(np.sum(negative)),
            "positive_count": int(positive.size),
            "negative_count": int(negative.size),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return output


def _quadrature_row(size: int) -> dict[str, Any]:
    started = time.perf_counter()
    shape = _grid_shape(size)
    volume = int(np.prod(shape))
    frequencies = _frequency_axes(shape)
    wave_number_squared = sum(
        frequency * frequency for frequency in frequencies
    )
    wave_number_squared[0, 0, 0] = 1.0

    profile_coefficients = _profile_coefficients(size, shape)
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
    acceleration_code_coefficients = _euler_cross(
        profile_coefficients,
        profile_values,
        velocity_coefficients,
        velocity_values,
        frequencies,
        wave_number_squared,
        volume,
    )

    velocity_energy_components = [
        _component_square_sum(velocity_coefficients[component])
        for component in range(3)
    ]
    acceleration_code_pair_components = [
        _pair_sum(
            acceleration_code_coefficients[component],
            profile_coefficients[component],
        )
        for component in range(3)
    ]
    normalization = float(size) ** -11
    alpha = math.sqrt(2.0) / 20.0
    first_component = (
        alpha
        * normalization
        * (
            velocity_energy_components[2]
            - velocity_energy_components[1]
        )
    )
    # The cross routine returns -g for the real even acceleration
    # coefficient. Thus pair_z-pair_y equals integral(g_y a_y-g_z a_z).
    second_component = (
        2.0
        * alpha
        * normalization
        * (
            acceleration_code_pair_components[2]
            - acceleration_code_pair_components[1]
        )
    )

    energy_trace_residual = (
        sum(velocity_energy_components)
        + 2.0 * sum(acceleration_code_pair_components)
    )
    energy_trace_scale = max(
        sum(velocity_energy_components),
        2.0 * abs(sum(acceleration_code_pair_components)),
        1.0,
    )

    density = (
        alpha
        * normalization
        * (
            (
                velocity_coefficients[2].real ** 2
                + velocity_coefficients[2].imag ** 2
            )
            - (
                velocity_coefficients[1].real ** 2
                + velocity_coefficients[1].imag ** 2
            )
        )
    )
    density += (
        2.0
        * alpha
        * normalization
        * (
            (
                acceleration_code_coefficients[2]
                * np.conjugate(profile_coefficients[2])
            ).real
            - (
                acceleration_code_coefficients[1]
                * np.conjugate(profile_coefficients[1])
            ).real
        )
    )
    combined_component = first_component + second_component
    density_sum = float(np.sum(density))
    divergence_residuals = {
        "a": _divergence_residual(
            profile_coefficients,
            frequencies,
        ),
        "minus_i_v": _divergence_residual(
            velocity_coefficients,
            frequencies,
        ),
        "minus_g": _divergence_residual(
            acceleration_code_coefficients,
            frequencies,
        ),
    }
    row = {
        "size": size,
        "mesh_width": 1.0 / size,
        "grid_shape": list(shape),
        "grid_point_count": volume,
        "first_form_continuum_quadrature": first_component,
        "second_form_continuum_quadrature": second_component,
        "combined_continuum_quadrature": combined_component,
        "density_sum": density_sum,
        "density_sum_absolute_replay_error": abs(
            density_sum - combined_component
        ),
        "velocity_energy_components_times_h11": [
            normalization * value
            for value in velocity_energy_components
        ],
        "minus_g_pair_components_times_h11": [
            normalization * value
            for value in acceleration_code_pair_components
        ],
        "energy_trace_relative_residual": (
            abs(energy_trace_residual) / energy_trace_scale
        ),
        "maximum_divergence_residual": max(
            divergence_residuals.values()
        ),
        "divergence_residuals": divergence_residuals,
        "combined_density_sectors": _sector_diagnostic(
            density,
            frequencies[0],
            size,
        ),
        "origin_leray_cusp_replay": _origin_cusp_replay(
            size,
            shape,
            velocity_coefficients,
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    row["all_numerical_checks_pass"] = bool(
        first_component > 0.0
        and second_component < 0.0
        and combined_component < 0.0
        and row["energy_trace_relative_residual"] < 1.0e-10
        and row["maximum_divergence_residual"] < 1.0e-9
        and row["density_sum_absolute_replay_error"] < 1.0e-18
    )

    del density
    del acceleration_code_coefficients
    del velocity_values
    del velocity_coefficients
    del profile_values
    del profile_coefficients
    del wave_number_squared
    gc.collect()
    return row


def _richardson_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_size = {int(row["size"]): row for row in rows}
    output = []
    for size in sorted(by_size):
        if 2 * size not in by_size:
            continue
        coarse = float(
            by_size[size]["combined_continuum_quadrature"]
        )
        fine = float(
            by_size[2 * size]["combined_continuum_quadrature"]
        )
        output.append(
            {
                "coarse_size": size,
                "fine_size": 2 * size,
                "second_order_extrapolate": (4.0 * fine - coarse) / 3.0,
                "coarse_fine_difference": coarse - fine,
                "certified_error_bound": None,
            }
        )
    return output


def _write_result(rows: Sequence[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["size"]))
    result = {
        "algorithm_revision": ALGORITHM_REVISION,
        "kind": "annular_rho_zero_direct_continuum_quadrature",
        "scope": (
            "Direct exact-box tensor-trapezoid evaluation of L_EE, "
            "including output-sector diagnostics"
        ),
        "prerequisite": {
            "path": str(PREREQUISITE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(PREREQUISITE),
        },
        "continuum_profile": {
            "positive_domain": (
                "[2,3] x [-1/2,1/2] x [-1/2,1/2]"
            ),
            "formula": (
                "a=S*(-xz,-yz,x^2+y^2)/(x^2+y^2+z^2)^(3/2)"
            ),
            "scalar_profile": (
                "S=sin(pi(x-2))*cos(pi*y)*cos(pi*z)"
            ),
            "extension": "a(-xi)=a(xi), zero outside D union -D",
        },
        "normalization": {
            "mesh_width": "h=1/N",
            "quartic_lattice_factor": "h^11",
            "reason": (
                "three independent three-dimensional sums contribute "
                "h^9 and the two integer Euler symbols contribute h^-2"
            ),
        },
        "sign_convention": {
            "quadratic_code_field": "-i v",
            "cross_code_field": "-g",
            "second_component": (
                "sqrt(2)/10*h^11*(pair_z-pair_y)"
            ),
        },
        "rows": ordered,
        "second_order_richardson_diagnostics": _richardson_rows(ordered),
        "certification": {
            "continuum_sign_numerically_stable": bool(
                ordered
                and all(
                    row["all_numerical_checks_pass"] for row in ordered
                )
            ),
            "continuum_sign_interval_certified": False,
            "reason": (
                "No directed-rounding FFT bound or analytic "
                "tensor-trapezoid remainder has yet been supplied."
            ),
            "next_action": (
                "Use the sector diagnostics to construct a deterministic "
                "joint enclosure whose upper endpoint is negative."
            ),
        },
        "all_numerical_checks_pass": bool(
            ordered
            and all(row["all_numerical_checks_pass"] for row in ordered)
        ),
    }
    _atomic_json(RESULT, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default=",".join(str(value) for value in DEFAULT_SIZES),
        help="comma-separated even mesh sizes",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard matching cached rows",
    )
    arguments = parser.parse_args()
    sizes = tuple(
        sorted(
            {
                int(value.strip())
                for value in arguments.sizes.split(",")
                if value.strip()
            }
        )
    )
    if not sizes:
        raise ValueError("at least one size is required")
    if any(size < 2 or size % 2 for size in sizes):
        raise ValueError("all sizes must be even integers at least 2")

    _lower_process_priority()
    cached_rows: dict[int, dict[str, Any]] = {}
    if RESULT.exists() and not arguments.fresh:
        existing = json.loads(RESULT.read_text(encoding="utf-8"))
        if existing.get("algorithm_revision") == ALGORITHM_REVISION:
            cached_rows = {
                int(row["size"]): row for row in existing.get("rows", [])
            }

    rows = dict(cached_rows)
    for size in sizes:
        if size in rows:
            print(f"N={size}: cached", flush=True)
            continue
        print(f"N={size}: computing", flush=True)
        rows[size] = _quadrature_row(size)
        _write_result(list(rows.values()))
        print(
            f"N={size}: "
            f"{rows[size]['combined_continuum_quadrature']:.16e}",
            flush=True,
        )
    _write_result(list(rows.values()))


if __name__ == "__main__":
    main()
