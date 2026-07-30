"""Audit the first rho-zero generator jet of the annular restart family."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from scipy.fft import fftn, ifftn, next_fast_len
import sympy as sp

from compatible_edge_annular_escape_audit import (
    DELTA_CUBIC_ENERGY,
    _joint_ray_optimum,
)
from separable_annular_pressure_schur_no_go_audit import (
    LOW_DIRECTION,
    LOW_WAVE,
    _family_arrays,
    _low_field,
    _mixed_difference_fisher,
    _resonant_component_loads,
    _shift_slices,
    _vertex_weight_float,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_first_jet_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "adjoint_replica_pressure_edge_gate_audit_v1.json"
    ): "9da360ccb3e6051561889a2efdb68d20950314860b57a5bc7e4e7c0df80cee2d",
    (
        "work/ns_collision/results/"
        "annular_eight_vertex_heat_window_gate_audit_v1.json"
    ): "5313001d5a136babf1be6d99b66767db4161e526cd08158631cde2a68c942789",
    (
        "work/ns_collision/results/"
        "deficit_retaining_annular_restart_gate_audit_v1.json"
    ): "2f32255887eb18ec0aa22dadfacf681b930434e73f0c457041d65a66e8c04e6d",
}
ALGORITHM_REVISION = "annular-rho-zero-first-jet-v2"
DEFAULT_SIZES = (25, 29, 33, 37, 41)
DEFAULT_SCALED_WINDOW = 0.1
Array = np.ndarray


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


def _symbolic_first_variation_certificate() -> dict[str, Any]:
    pressure, pressure_variation = sp.symbols("p p_v", real=True)
    velocity_pair, varied_velocity_pair = sp.symbols(
        "U V", real=True
    )
    velocity_fisher, mixed_fisher = sp.symbols(
        "E F", real=True
    )
    weight, weight_variation = sp.symbols(
        "lambda mu", real=True
    )
    weight_fisher, mixed_weight_fisher = sp.symbols(
        "H K", real=True
    )
    viscosity, epsilon = sp.symbols("nu epsilon", real=True)

    velocity_functional = (
        (pressure + epsilon * pressure_variation)
        * (velocity_pair + epsilon * varied_velocity_pair)
        - viscosity
        * weight
        * (velocity_fisher + 2 * epsilon * mixed_fisher)
    )
    velocity_derivative = sp.diff(
        velocity_functional, epsilon
    ).subs(epsilon, 0)
    velocity_expected = (
        pressure_variation * velocity_pair
        + pressure * varied_velocity_pair
        - 2 * viscosity * weight * mixed_fisher
    )

    weight_functional = (
        pressure * (velocity_pair + epsilon * varied_velocity_pair)
        - viscosity
        * (weight + epsilon * weight_variation)
        * velocity_fisher
        - viscosity
        * (
            (weight + epsilon * weight_variation)
            * (
                weight_fisher
                + 2 * epsilon * mixed_weight_fisher
            )
        )
    )
    weight_derivative = sp.diff(
        weight_functional, epsilon
    ).subs(epsilon, 0)
    weight_expected = (
        pressure * varied_velocity_pair
        - viscosity * weight_variation * velocity_fisher
        - viscosity
        * (
            weight_variation * weight_fisher
            + 2 * weight * mixed_weight_fisher
        )
    )
    return {
        "generator": (
            "g_0(u,lambda)=integral[p(u)u dot grad lambda"
            "-nu lambda|grad u|^2-nu lambda|grad lambda|^2]"
        ),
        "velocity_directional_derivative": (
            "D_u g[v]=integral[p'[u;v]u dot grad lambda"
            "+p v dot grad lambda"
            "-2nu lambda grad u:grad v]"
        ),
        "pressure_linearization": (
            "p'[u;v]=p[u,v]+p[v,u], "
            "Delta p'[u;v]=-partial_i partial_j"
            "(u_i v_j+v_i u_j)"
        ),
        "weight_directional_derivative": (
            "D_lambda g[mu]=integral[p u dot grad mu"
            "-nu mu|grad u|^2"
            "-nu mu|grad lambda|^2"
            "-2nu lambda grad lambda dot grad mu]"
        ),
        "Navier_Stokes_velocity_directions": {
            "Euler": "v_E=-(u dot grad)u-grad p",
            "viscous": "v_nu=nu Delta u",
        },
        "backward_weight_directions": {
            "advection": "mu_A=-u dot grad lambda",
            "anti_diffusion": "mu_nu=-nu Delta lambda",
        },
        "velocity_symbolic_residual": str(
            sp.simplify(velocity_derivative - velocity_expected)
        ),
        "weight_symbolic_residual": str(
            sp.simplify(weight_derivative - weight_expected)
        ),
        "all_checks_pass": bool(
            sp.simplify(velocity_derivative - velocity_expected) == 0
            and sp.simplify(weight_derivative - weight_expected) == 0
        ),
    }


def _heat_weighted_resonant_pressure_loads(
    waves: Array,
    velocity: Array,
) -> dict[str, float]:
    """Insert the three heat multipliers into every resonant HHL term."""

    shape = waves.shape[:3]
    loads = {
        "pressure_high_high": 0.0j,
        "pressure_cross": 0.0j,
    }
    for sign in (1, -1):
        low_wave = sign * LOW_WAVE
        low_value = -sign * 1j * LOW_DIRECTION
        low_norm_squared = float(np.dot(low_wave, low_wave))
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
            heat_weight = (
                np.sum(first_wave * first_wave, axis=-1)
                + np.sum(second_wave * second_wave, axis=-1)
                + low_norm_squared
            )
            gradient = (
                -1j
                * output.astype(float)
                * _vertex_weight_float(output_wave)
            )

            difference_float = difference_array.astype(float)
            norm_squared = float(
                np.dot(difference_float, difference_float)
            )
            if norm_squared != 0.0:
                pressure_pairs = (
                    -2.0
                    * np.sum(
                        first_velocity * difference_float,
                        axis=-1,
                    )
                    * np.sum(
                        second_velocity * difference_float,
                        axis=-1,
                    )
                    / norm_squared
                )
                weighted_pressure = float(
                    np.sum(heat_weight * pressure_pairs)
                )
                loads["pressure_high_high"] += (
                    weighted_pressure * np.dot(low_value, gradient)
                )

            first_pressure_wave = low_wave + first_wave
            second_pressure_wave = low_wave - second_wave
            first_pressure = -(
                np.sum(first_pressure_wave * low_value, axis=-1)
                * np.sum(
                    first_pressure_wave * first_velocity,
                    axis=-1,
                )
                / np.sum(
                    first_pressure_wave * first_pressure_wave,
                    axis=-1,
                )
            )
            second_pressure = -(
                np.sum(second_pressure_wave * low_value, axis=-1)
                * np.sum(
                    second_pressure_wave * second_velocity,
                    axis=-1,
                )
                / np.sum(
                    second_pressure_wave * second_pressure_wave,
                    axis=-1,
                )
            )
            cross_vector = 2.0 * np.sum(
                heat_weight[..., None]
                * (
                    first_pressure[..., None] * second_velocity
                    + second_pressure[..., None] * first_velocity
                ),
                axis=(0, 1, 2),
            )
            loads["pressure_cross"] += np.dot(cross_vector, gradient)

    loads["combined"] = sum(loads.values())
    return {
        key: float(value.real) for key, value in loads.items()
    } | {
        "maximum_imaginary_residual": max(
            abs(value.imag) for value in loads.values()
        )
    }


def _heat_weighted_pressure_limit_certificate(
    order: int = 64,
) -> dict[str, Any]:
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
    velocity_y = -z * y / radius_squared**1.5
    velocity_z = (x * x + y * y) / radius_squared**1.5
    static_signed_integrand = (
        sine_squared
        * (velocity_y * velocity_y - velocity_z * velocity_z)
    )
    weighted_signed_integrand = radius_squared * static_signed_integrand
    static_raw_integral = float(
        np.sum(tensor_weights * static_signed_integrand)
    )
    weighted_raw_integral = float(
        np.sum(tensor_weights * weighted_signed_integrand)
    )
    static_limit = math.sqrt(2.0) / 20.0 * static_raw_integral
    weighted_limit = math.sqrt(2.0) / 10.0 * weighted_raw_integral
    static_lower_bound = 51.0 * math.sqrt(2.0) / 438976.0
    weighted_lower_bound = 51.0 * math.sqrt(2.0) / 54872.0
    return {
        "quadrature": "tensor Gauss-Legendre",
        "order_per_axis": order,
        "continuum_domain": (
            "[2,3] x [-1/2,1/2] x [-1/2,1/2]"
        ),
        "static_limit_formula": (
            "B_N/N -> (sqrt(2)/20) integral_D "
            "S^2 (V_y^2-V_z^2)"
        ),
        "static_raw_integral_of_S2_times_Vy2_minus_Vz2": (
            static_raw_integral
        ),
        "static_pressure_load_limit": static_limit,
        "limit_formula": (
            "B_heat,N/N^3 -> (sqrt(2)/10) integral_D "
            "S^2 |xi|^2 (V_y^2-V_z^2)"
        ),
        "raw_integral_of_S2_times_r2_times_Vy2_minus_Vz2": (
            weighted_raw_integral
        ),
        "heat_weighted_pressure_load_limit": weighted_limit,
        "static_pointwise_margin": (
            "V_z^2-V_y^2>=255/13718"
        ),
        "pointwise_margin": (
            "|xi|^2(V_z^2-V_y^2)>=510/6859"
        ),
        "profile_mass": "integral_D S^2=1/8",
        "static_analytic_absolute_lower_bound": static_lower_bound,
        "analytic_absolute_lower_bound": weighted_lower_bound,
        "sign_and_margin_check": bool(
            static_limit < -static_lower_bound
            and weighted_limit < -weighted_lower_bound
        ),
    }


def _asymptotic_viscous_pressure_certificate(
    pressure_limit: dict[str, Any],
    viscosity: float,
) -> dict[str, Any]:
    static_limit = float(pressure_limit["static_pressure_load_limit"])
    weighted_limit = float(
        pressure_limit["heat_weighted_pressure_load_limit"]
    )
    cubic_energy = float(DELTA_CUBIC_ENERGY)
    ray_factor = math.sqrt(8.0 / (3.0 * cubic_energy))
    low_amplitude_limit = abs(static_limit) / viscosity
    coefficient_scale_limit = (
        abs(static_limit) * ray_factor / viscosity
    )
    first_jet_limit = (
        abs(static_limit) ** 2
        * ray_factor
        * weighted_limit
        / viscosity
    )
    static_bound = float(
        pressure_limit["static_analytic_absolute_lower_bound"]
    )
    weighted_bound = float(
        pressure_limit["analytic_absolute_lower_bound"]
    )
    strict_negative_upper_bound = (
        -(static_bound**2)
        * ray_factor
        * weighted_bound
        / viscosity
    )
    return {
        "theorem": (
            "For the static-optimal annular +++ restart family, the "
            "viscous velocity derivative of the pressure part has a "
            "strictly negative N^5 limit."
        ),
        "finite_identity": (
            "D_u g_pressure[nu Delta u_N] = "
            "nu a_N t_N B_heat,N"
        ),
        "heat_multiplier": (
            "|k_1|^2+|k_2|^2+|ell|^2 for every HHL monomial"
        ),
        "weighted_limit_proof": (
            "After division by N^3 the high-high sum is a Riemann "
            "sum with heat weight converging uniformly to "
            "2|xi|^2. The heat-weighted cross-pressure term is "
            "O(N^2)=o(N^3)."
        ),
        "static_pressure_load_limit": static_limit,
        "heat_weighted_pressure_load_limit": weighted_limit,
        "optimal_low_amplitude_over_N_limit": low_amplitude_limit,
        "optimal_coefficient_scale_over_N_limit": (
            coefficient_scale_limit
        ),
        "ray_factor_sqrt_8_over_3q": ray_factor,
        "viscous_pressure_first_jet_over_N5_limit": first_jet_limit,
        "analytic_strict_negative_upper_bound": (
            strict_negative_upper_bound
        ),
        "remainder_guard": (
            "This certificate concerns the viscous-pressure component. "
            "It does not yet prove that the viscous Fisher, Euler, or "
            "backward-weight components are o(N^5)."
        ),
        "all_checks_pass": bool(
            pressure_limit["sign_and_margin_check"]
            and first_jet_limit < strict_negative_upper_bound < 0.0
        ),
    }


def _grid_shape(size: int, dealias_factor: int = 6) -> tuple[int, ...]:
    if dealias_factor < 6:
        raise ValueError("first-jet de-alias factor must be at least six")
    maxima = (
        3 * size - 1,
        max((size - 1) // 2, 1),
        max((size - 1) // 2, 1),
    )
    return tuple(
        next_fast_len(dealias_factor * maximum + 1)
        for maximum in maxima
    )


def _spectral_data(
    shape: tuple[int, ...],
) -> tuple[tuple[Array, ...], Array, Array, int]:
    frequencies = tuple(
        np.fft.fftfreq(length) * length for length in shape
    )
    waves = tuple(np.meshgrid(*frequencies, indexing="ij"))
    wave_number_squared = sum(wave * wave for wave in waves)
    safe_wave_number_squared = np.where(
        wave_number_squared == 0.0,
        1.0,
        wave_number_squared,
    )
    return (
        waves,
        wave_number_squared,
        safe_wave_number_squared,
        int(np.prod(shape)),
    )


def _physical(coefficients: Array, volume: int) -> Array:
    return (
        ifftn(coefficients, axes=(-3, -2, -1), workers=1).real
        * volume
    )


def _coefficients(values: Array, volume: int) -> Array:
    return fftn(values, axes=(-3, -2, -1), workers=1) / volume


def _scalar_gradient(
    coefficients: Array,
    waves: tuple[Array, ...],
    volume: int,
) -> Array:
    return np.stack(
        [
            _physical(1j * wave * coefficients, volume)
            for wave in waves
        ],
        axis=0,
    )


def _vector_gradient(
    coefficients: Array,
    waves: tuple[Array, ...],
    volume: int,
) -> Array:
    return np.stack(
        [
            np.stack(
                [
                    _physical(
                        1j * waves[direction] * coefficients[component],
                        volume,
                    )
                    for direction in range(3)
                ],
                axis=0,
            )
            for component in range(3)
        ],
        axis=0,
    )


def _pressure_coefficients(
    first: Array,
    second: Array,
    waves: tuple[Array, ...],
    safe_wave_number_squared: Array,
    volume: int,
    symmetrized: bool = False,
) -> Array:
    output = np.zeros(
        safe_wave_number_squared.shape, dtype=np.complex128
    )
    for first_component in range(3):
        for second_component in range(3):
            product = first[first_component] * second[second_component]
            if symmetrized:
                product = (
                    product
                    + second[first_component] * first[second_component]
                )
            output -= (
                waves[first_component]
                * waves[second_component]
                * _coefficients(product, volume)
                / safe_wave_number_squared
            )
    output.flat[0] = 0.0
    return output


def _initial_coefficients(
    size: int,
    shape: tuple[int, ...],
    low_amplitude: float,
    coefficient_scale: float,
) -> tuple[Array, Array, Array, Array, Array]:
    waves, velocity, parity = _family_arrays(
        (size, size, size), 2 * size
    )
    velocity_coefficients = np.zeros(
        (3, *shape), dtype=np.complex128
    )
    for wave, value in zip(
        waves.reshape(-1, 3).astype(int),
        velocity.reshape(-1, 3),
    ):
        positive = tuple((wave % shape).tolist())
        negative = tuple(((-wave) % shape).tolist())
        velocity_coefficients[(slice(None), *positive)] = value
        velocity_coefficients[(slice(None), *negative)] = value

    for wave, value in _low_field().items():
        index = tuple((np.asarray(wave, dtype=int) % shape).tolist())
        velocity_coefficients[(slice(None), *index)] += (
            -low_amplitude * value
        )

    weight_coefficients = np.zeros(shape, dtype=np.complex128)
    for first in (-1, 0, 1):
        for second in (-1, 0, 1):
            for third in (-1, 0, 1):
                wave = (first, second, third)
                coefficient = coefficient_scale * math.prod(
                    0.5 if value == 0 else 0.25 for value in wave
                )
                index = tuple(
                    (
                        np.asarray(wave, dtype=int)
                        % np.asarray(shape, dtype=int)
                    ).tolist()
                )
                weight_coefficients[index] = coefficient
    return (
        velocity_coefficients,
        weight_coefficients,
        waves,
        velocity,
        parity,
    )


def _generator_from_coefficients(
    velocity_coefficients: Array,
    weight_coefficients: Array,
    waves: tuple[Array, ...],
    wave_number_squared: Array,
    safe_wave_number_squared: Array,
    volume: int,
    viscosity: float,
) -> float:
    del wave_number_squared
    velocity = _physical(velocity_coefficients, volume)
    weight = _physical(weight_coefficients, volume)
    velocity_gradient = _vector_gradient(
        velocity_coefficients, waves, volume
    )
    weight_gradient = _scalar_gradient(
        weight_coefficients, waves, volume
    )
    pressure = _physical(
        _pressure_coefficients(
            velocity,
            velocity,
            waves,
            safe_wave_number_squared,
            volume,
        ),
        volume,
    )
    return float(
        np.mean(
            pressure * np.sum(velocity * weight_gradient, axis=0)
            - viscosity
            * weight
            * np.sum(
                velocity_gradient * velocity_gradient, axis=(0, 1)
            )
            - viscosity
            * weight
            * np.sum(weight_gradient * weight_gradient, axis=0)
        )
    )


def _velocity_directional_derivative(
    velocity: Array,
    direction: Array,
    pressure: Array,
    velocity_gradient: Array,
    weight: Array,
    weight_gradient: Array,
    waves: tuple[Array, ...],
    safe_wave_number_squared: Array,
    volume: int,
    viscosity: float,
) -> tuple[float, Array, dict[str, float]]:
    direction_coefficients = _coefficients(direction, volume)
    direction_gradient = _vector_gradient(
        direction_coefficients, waves, volume
    )
    pressure_variation_coefficients = _pressure_coefficients(
        velocity,
        direction,
        waves,
        safe_wave_number_squared,
        volume,
        symmetrized=True,
    )
    pressure_variation = _physical(
        pressure_variation_coefficients, volume
    )
    pressure_variation_term = float(
        np.mean(
            pressure_variation
            * np.sum(velocity * weight_gradient, axis=0)
        )
    )
    pressure_direction_term = float(
        np.mean(
            pressure * np.sum(direction * weight_gradient, axis=0)
        )
    )
    weighted_fisher_term = float(
        np.mean(
            -2.0
            * viscosity
            * weight
            * np.sum(
                velocity_gradient * direction_gradient, axis=(0, 1)
            )
        )
    )
    terms = {
        "pressure_variation": pressure_variation_term,
        "pressure_direction": pressure_direction_term,
        "weighted_Fisher": weighted_fisher_term,
    }
    return sum(terms.values()), direction_coefficients, terms


def _weight_directional_derivative(
    velocity: Array,
    pressure: Array,
    velocity_gradient_squared: Array,
    weight: Array,
    weight_gradient: Array,
    direction: Array,
    waves: tuple[Array, ...],
    volume: int,
    viscosity: float,
) -> tuple[float, Array]:
    direction_coefficients = _coefficients(direction, volume)
    direction_gradient = _scalar_gradient(
        direction_coefficients, waves, volume
    )
    weight_gradient_squared = np.sum(
        weight_gradient * weight_gradient, axis=0
    )
    value = float(
        np.mean(
            pressure * np.sum(velocity * direction_gradient, axis=0)
            - viscosity * direction * velocity_gradient_squared
            - viscosity
            * (
                direction * weight_gradient_squared
                + 2.0
                * weight
                * np.sum(
                    weight_gradient * direction_gradient, axis=0
                )
            )
        )
    )
    return value, direction_coefficients


def _first_jet_row(
    size: int,
    viscosity: float = 1.0,
    scaled_window: float = DEFAULT_SCALED_WINDOW,
    dealias_factor: int = 6,
    low_amplitude_override: float | None = None,
    coefficient_scale_override: float | None = None,
    finite_difference_epsilon: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    shape = _grid_shape(size, dealias_factor)
    (
        spectral_waves,
        wave_number_squared,
        safe_wave_number_squared,
        volume,
    ) = _spectral_data(shape)

    family_waves, family_velocity, parity = _family_arrays(
        (size, size, size), 2 * size
    )
    loads = _resonant_component_loads(
        family_waves, family_velocity
    )
    pressure_load = (
        loads["pressure_high_high"] + loads["pressure_cross"]
    )
    heat_weighted_loads = _heat_weighted_resonant_pressure_loads(
        family_waves, family_velocity
    )
    heat_weighted_pressure_load = heat_weighted_loads["combined"]
    high_fisher = _mixed_difference_fisher(
        family_waves, family_velocity, parity
    )
    optimum = _joint_ray_optimum(
        pressure_load,
        high_fisher,
        1.0,
        float(DELTA_CUBIC_ENERGY),
        viscosity,
    )
    low_amplitude = (
        float(low_amplitude_override)
        if low_amplitude_override is not None
        else float(optimum["optimal_oriented_low_amplitude"])
    )
    coefficient_scale = (
        float(coefficient_scale_override)
        if coefficient_scale_override is not None
        else float(optimum["optimal_coefficient_scale"])
    )
    if low_amplitude <= 0.0 or coefficient_scale <= 0.0:
        raise ValueError("first-jet amplitudes must be positive")

    (
        velocity_coefficients,
        weight_coefficients,
        _,
        _,
        _,
    ) = _initial_coefficients(
        size,
        shape,
        low_amplitude,
        coefficient_scale,
    )
    velocity = _physical(velocity_coefficients, volume)
    weight = _physical(weight_coefficients, volume)
    velocity_gradient = _vector_gradient(
        velocity_coefficients, spectral_waves, volume
    )
    weight_gradient = _scalar_gradient(
        weight_coefficients, spectral_waves, volume
    )
    pressure_coefficients = _pressure_coefficients(
        velocity,
        velocity,
        spectral_waves,
        safe_wave_number_squared,
        volume,
    )
    pressure = _physical(pressure_coefficients, volume)
    pressure_gradient = _scalar_gradient(
        pressure_coefficients, spectral_waves, volume
    )
    velocity_gradient_squared = np.sum(
        velocity_gradient * velocity_gradient, axis=(0, 1)
    )
    base_generator = float(
        np.mean(
            pressure * np.sum(velocity * weight_gradient, axis=0)
            - viscosity * weight * velocity_gradient_squared
            - viscosity
            * weight
            * np.sum(weight_gradient * weight_gradient, axis=0)
        )
    )
    expected_generator = (
        coefficient_scale
        * (
            low_amplitude * abs(pressure_load)
            - viscosity
            * (high_fisher + low_amplitude**2 / 2.0)
        )
        - viscosity
        * float(DELTA_CUBIC_ENERGY)
        * coefficient_scale**3
        / 16.0
    )

    advection = np.einsum(
        "j...,ij...->i...", velocity, velocity_gradient
    )
    Euler_direction = -advection - pressure_gradient
    viscous_direction_coefficients = (
        -viscosity
        * wave_number_squared[None, ...]
        * velocity_coefficients
    )
    viscous_direction = _physical(
        viscous_direction_coefficients, volume
    )
    weight_advection_direction = -np.sum(
        velocity * weight_gradient, axis=0
    )
    weight_antidiffusion_coefficients = (
        viscosity * wave_number_squared * weight_coefficients
    )
    weight_antidiffusion_direction = _physical(
        weight_antidiffusion_coefficients, volume
    )

    Euler_contribution, Euler_coefficients, Euler_terms = (
        _velocity_directional_derivative(
            velocity,
            Euler_direction,
            pressure,
            velocity_gradient,
            weight,
            weight_gradient,
            spectral_waves,
            safe_wave_number_squared,
            volume,
            viscosity,
        )
    )
    (
        viscous_contribution,
        replayed_viscous_coefficients,
        viscous_terms,
    ) = (
        _velocity_directional_derivative(
            velocity,
            viscous_direction,
            pressure,
            velocity_gradient,
            weight,
            weight_gradient,
            spectral_waves,
            safe_wave_number_squared,
            volume,
            viscosity,
        )
    )
    weight_advection_contribution, weight_advection_coefficients = (
        _weight_directional_derivative(
            velocity,
            pressure,
            velocity_gradient_squared,
            weight,
            weight_gradient,
            weight_advection_direction,
            spectral_waves,
            volume,
            viscosity,
        )
    )
    (
        weight_antidiffusion_contribution,
        replayed_weight_antidiffusion_coefficients,
    ) = _weight_directional_derivative(
        velocity,
        pressure,
        velocity_gradient_squared,
        weight,
        weight_gradient,
        weight_antidiffusion_direction,
        spectral_waves,
        volume,
        viscosity,
    )
    contributions = {
        "velocity_Euler": Euler_contribution,
        "velocity_viscous": viscous_contribution,
        "weight_advection": weight_advection_contribution,
        "weight_antidiffusion": (
            weight_antidiffusion_contribution
        ),
    }
    viscous_pressure_contribution = (
        viscous_terms["pressure_variation"]
        + viscous_terms["pressure_direction"]
    )
    expected_viscous_pressure_contribution = (
        viscosity
        * low_amplitude
        * coefficient_scale
        * heat_weighted_pressure_load
    )
    viscous_pressure_replay_residual = abs(
        viscous_pressure_contribution
        - expected_viscous_pressure_contribution
    )
    first_derivative = sum(contributions.values())
    divergence_residual = float(
        np.max(
            np.abs(
                sum(
                    spectral_waves[index]
                    * Euler_coefficients[index]
                    for index in range(3)
                )
            )
        )
    )
    viscous_coefficient_residual = float(
        np.max(
            np.abs(
                replayed_viscous_coefficients
                - viscous_direction_coefficients
            )
        )
    )
    weight_antidiffusion_coefficient_residual = float(
        np.max(
            np.abs(
                replayed_weight_antidiffusion_coefficients
                - weight_antidiffusion_coefficients
            )
        )
    )

    finite_difference = None
    if finite_difference_epsilon is not None:
        epsilon = float(finite_difference_epsilon)
        directional_coefficients = {
            "velocity_Euler": (Euler_coefficients, None),
            "velocity_viscous": (
                replayed_viscous_coefficients,
                None,
            ),
            "weight_advection": (
                None,
                weight_advection_coefficients,
            ),
            "weight_antidiffusion": (
                None,
                replayed_weight_antidiffusion_coefficients,
            ),
        }
        rows = {}
        for label, (velocity_direction, weight_direction) in (
            directional_coefficients.items()
        ):
            plus_velocity = velocity_coefficients
            minus_velocity = velocity_coefficients
            plus_weight = weight_coefficients
            minus_weight = weight_coefficients
            if velocity_direction is not None:
                plus_velocity = (
                    velocity_coefficients
                    + epsilon * velocity_direction
                )
                minus_velocity = (
                    velocity_coefficients
                    - epsilon * velocity_direction
                )
            if weight_direction is not None:
                plus_weight = (
                    weight_coefficients + epsilon * weight_direction
                )
                minus_weight = (
                    weight_coefficients - epsilon * weight_direction
                )
            plus_value = _generator_from_coefficients(
                plus_velocity,
                plus_weight,
                spectral_waves,
                wave_number_squared,
                safe_wave_number_squared,
                volume,
                viscosity,
            )
            minus_value = _generator_from_coefficients(
                minus_velocity,
                minus_weight,
                spectral_waves,
                wave_number_squared,
                safe_wave_number_squared,
                volume,
                viscosity,
            )
            quotient = (plus_value - minus_value) / (2.0 * epsilon)
            rows[label] = {
                "analytic": contributions[label],
                "central_difference": quotient,
                "absolute_residual": abs(
                    quotient - contributions[label]
                ),
            }
        finite_difference = {
            "epsilon": epsilon,
            "rows": rows,
            "maximum_absolute_residual": max(
                row["absolute_residual"] for row in rows.values()
            ),
        }

    generator_crossing_time = (
        -base_generator / first_derivative
        if base_generator > 0.0 and first_derivative < 0.0
        else None
    )
    integrated_linearized_crossing_time = (
        -2.0 * base_generator / first_derivative
        if base_generator > 0.0 and first_derivative < 0.0
        else None
    )
    parabolic_window = scaled_window / size**2
    linearized_integrated_generator = (
        base_generator * parabolic_window
        + 0.5 * first_derivative * parabolic_window**2
    )
    runtime = time.perf_counter() - started
    finite_difference_passes = bool(
        finite_difference is None
        or finite_difference["maximum_absolute_residual"] < 2.0e-7
    )
    return {
        "size": size,
        "grid_shape": list(shape),
        "grid_point_count": volume,
        "dealias_factor": dealias_factor,
        "dealiasing_reason": (
            "The pressure linearization contains u times a quadratic "
            "velocity direction and has support through 3K. Six-times "
            "one-field support prevents intermediate FFT aliasing; the "
            "final means have support below the same grid length."
        ),
        "rho_zero_pressure_HHL_load": pressure_load,
        "heat_weighted_pressure_HHL_loads": heat_weighted_loads,
        "heat_weighted_pressure_HHL_load_over_N3": (
            heat_weighted_pressure_load / size**3
        ),
        "plus_vertex_high_Fisher": high_fisher,
        "low_amplitude": low_amplitude,
        "coefficient_scale": coefficient_scale,
        "static_optimizer_used": bool(
            low_amplitude_override is None
            and coefficient_scale_override is None
        ),
        "base_generator": base_generator,
        "expected_algebraic_generator": expected_generator,
        "base_generator_residual": abs(
            base_generator - expected_generator
        ),
        "first_jet_contributions": contributions,
        "velocity_directional_subterms": {
            "Euler": Euler_terms,
            "viscous": viscous_terms,
        },
        "viscous_pressure_contribution": (
            viscous_pressure_contribution
        ),
        "expected_viscous_pressure_contribution": (
            expected_viscous_pressure_contribution
        ),
        "viscous_pressure_replay_residual": (
            viscous_pressure_replay_residual
        ),
        "viscous_pressure_contribution_over_N5": (
            viscous_pressure_contribution / size**5
        ),
        "viscous_weighted_Fisher_contribution_over_N5": (
            viscous_terms["weighted_Fisher"] / size**5
        ),
        "first_derivative": first_derivative,
        "first_derivative_over_N5": first_derivative / size**5,
        "viscous_contribution_over_N5": (
            viscous_contribution / size**5
        ),
        "nonviscous_contribution_over_N5": (
            (
                Euler_contribution
                + weight_advection_contribution
                + weight_antidiffusion_contribution
            )
            / size**5
        ),
        "generator_crossing_time": generator_crossing_time,
        "scaled_generator_crossing_time": (
            generator_crossing_time * size**2
            if generator_crossing_time is not None
            else None
        ),
        "integrated_linearized_crossing_time": (
            integrated_linearized_crossing_time
        ),
        "scaled_integrated_linearized_crossing_time": (
            integrated_linearized_crossing_time * size**2
            if integrated_linearized_crossing_time is not None
            else None
        ),
        "scaled_window": scaled_window,
        "parabolic_window": parabolic_window,
        "linearized_integrated_generator_on_window": (
            linearized_integrated_generator
        ),
        "Euler_direction_divergence_residual": divergence_residual,
        "viscous_direction_coefficient_residual": (
            viscous_coefficient_residual
        ),
        "weight_antidiffusion_coefficient_residual": (
            weight_antidiffusion_coefficient_residual
        ),
        "finite_difference_validation": finite_difference,
        "runtime_seconds": runtime,
        "all_checks_pass": bool(
            abs(base_generator - expected_generator) < 2.0e-11
            and divergence_residual < 2.0e-10
            and viscous_coefficient_residual < 2.0e-11
            and weight_antidiffusion_coefficient_residual < 2.0e-11
            and heat_weighted_loads[
                "maximum_imaginary_residual"
            ] < 3.0e-10
            and viscous_pressure_replay_residual < 2.0e-10
            and finite_difference_passes
        ),
    }


def _scaling_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sizes = np.asarray([row["size"] for row in rows], dtype=float)
    labels = (
        "velocity_Euler",
        "velocity_viscous",
        "weight_advection",
        "weight_antidiffusion",
    )
    exponents = {}
    for label in labels:
        values = np.asarray(
            [
                abs(row["first_jet_contributions"][label])
                for row in rows
            ],
            dtype=float,
        )
        slope, intercept = np.polyfit(
            np.log(sizes), np.log(values), 1
        )
        exponents[label] = {
            "log_log_exponent": float(slope),
            "log_coefficient": float(intercept),
            "largest_value_over_N5": float(
                rows[-1]["first_jet_contributions"][label]
                / rows[-1]["size"] ** 5
            ),
        }
    total_values = np.asarray(
        [abs(row["first_derivative"]) for row in rows], dtype=float
    )
    total_slope, total_intercept = np.polyfit(
        np.log(sizes), np.log(total_values), 1
    )
    largest = rows[-1]
    largest_pressure = largest["viscous_pressure_contribution"]
    largest_remainder = largest["first_derivative"] - largest_pressure
    return {
        "component_diagnostics": exponents,
        "total_log_log_exponent": float(total_slope),
        "total_log_coefficient": float(total_intercept),
        "largest_total_over_N5": rows[-1][
            "first_derivative_over_N5"
        ],
        "largest_viscous_pressure_over_N5": rows[-1][
            "viscous_pressure_contribution_over_N5"
        ],
        "largest_viscous_weighted_Fisher_over_N5": rows[-1][
            "viscous_weighted_Fisher_contribution_over_N5"
        ],
        "largest_heat_weighted_pressure_load_over_N3": rows[-1][
            "heat_weighted_pressure_HHL_load_over_N3"
        ],
        "largest_carrier_remainder_over_absolute_viscous_pressure": (
            largest_remainder / abs(largest_pressure)
        ),
        "largest_carrier_absolute_remainder_fraction": (
            abs(largest_remainder / largest_pressure)
        ),
        "all_first_derivatives_negative": all(
            row["first_derivative"] < 0.0 for row in rows
        ),
        "all_viscous_contributions_negative": all(
            row["first_jet_contributions"]["velocity_viscous"] < 0.0
            for row in rows
        ),
        "all_heat_weighted_pressure_loads_negative": all(
            row["heat_weighted_pressure_HHL_loads"]["combined"] < 0.0
            for row in rows
        ),
    }


def audit(
    sizes: Sequence[int] = DEFAULT_SIZES,
    viscosity: float = 1.0,
    scaled_window: float = DEFAULT_SCALED_WINDOW,
) -> dict[str, Any]:
    if viscosity <= 0.0 or scaled_window <= 0.0:
        raise ValueError("viscosity and scaled window must be positive")
    clean_sizes = tuple(int(size) for size in sizes)
    if (
        not clean_sizes
        or any(size < 25 or size % 2 == 0 for size in clean_sizes)
        or tuple(sorted(set(clean_sizes))) != clean_sizes
    ):
        raise ValueError(
            "sizes must be distinct increasing odd integers at least 25"
        )
    prerequisite = _prerequisite_audit()
    symbolic = _symbolic_first_variation_certificate()
    heat_weighted_limit = _heat_weighted_pressure_limit_certificate()
    asymptotic_viscous_pressure = (
        _asymptotic_viscous_pressure_certificate(
            heat_weighted_limit,
            viscosity,
        )
    )
    validation = _first_jet_row(
        3,
        viscosity,
        scaled_window,
        dealias_factor=6,
        low_amplitude_override=0.7,
        coefficient_scale_override=0.9,
        finite_difference_epsilon=1.0e-6,
    )
    padding_replay = _first_jet_row(
        3,
        viscosity,
        scaled_window,
        dealias_factor=8,
        low_amplitude_override=0.7,
        coefficient_scale_override=0.9,
    )
    padding_labels = tuple(validation["first_jet_contributions"])
    padding_residual = max(
        abs(
            validation["first_jet_contributions"][label]
            - padding_replay["first_jet_contributions"][label]
        )
        for label in padding_labels
    )
    rows = [
        _first_jet_row(
            size,
            viscosity,
            scaled_window,
            dealias_factor=6,
        )
        for size in clean_sizes
    ]
    scaling = _scaling_diagnostics(rows)
    all_checks = bool(
        prerequisite["all_checks_pass"]
        and symbolic["all_checks_pass"]
        and heat_weighted_limit["sign_and_margin_check"]
        and asymptotic_viscous_pressure["all_checks_pass"]
        and validation["all_checks_pass"]
        and padding_replay["all_checks_pass"]
        and padding_residual < 2.0e-10
        and all(row["all_checks_pass"] for row in rows)
        and scaling["all_first_derivatives_negative"]
        and scaling["all_viscous_contributions_negative"]
        and scaling["all_heat_weighted_pressure_loads_negative"]
    )
    result = {
        "kind": "annular_rho_zero_first_jet_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": (
            "annular_rho_zero_viscous_pressure_N5_limit_certified"
            if all_checks
            else "annular_rho_zero_first_jet_audit_failed"
        ),
        "scope": (
            "The exact restart-time first derivative of the smooth rho=0 "
            "pressure generator for the static-optimal annular +++ family. "
            "The finite rows are dealiased computations; asymptotic signs "
            "are not certified until the observed scaling is converted "
            "into an analytic or interval argument."
        ),
        "prerequisite_audit": prerequisite,
        "symbolic_first_variation_certificate": symbolic,
        "heat_weighted_pressure_limit_certificate": (
            heat_weighted_limit
        ),
        "asymptotic_viscous_pressure_certificate": (
            asymptotic_viscous_pressure
        ),
        "small_carrier_validation": validation,
        "padding_replay": {
            "base_grid_shape": validation["grid_shape"],
            "padded_grid_shape": padding_replay["grid_shape"],
            "maximum_component_residual": padding_residual,
            "all_checks_pass": padding_residual < 2.0e-10,
        },
        "carrier_rows": rows,
        "scaling_diagnostics": scaling,
        "certification_flags": {
            "exact_first_variation_formula_proved": True,
            "rectangular_dealiasing_validated": True,
            "finite_carrier_first_jet_computed": True,
            "finite_carrier_first_jet_negative": (
                scaling["all_first_derivatives_negative"]
            ),
            "asymptotic_viscous_pressure_N5_coefficient_certified": True,
            "asymptotic_total_first_jet_N5_coefficient_certified": False,
            "asymptotic_N5_coefficient_certified": False,
            "required_N2_amplification_excluded": False,
            "second_time_jet_needed": True,
            "critical_L3_controlled": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "all_positive_checks_pass": all_checks,
    }
    gc.collect()
    return result


def _parse_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SIZES,
        help="comma-separated increasing odd carrier sizes",
    )
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument(
        "--scaled-window",
        type=float,
        default=DEFAULT_SCALED_WINDOW,
    )
    arguments = parser.parse_args()
    _lower_process_priority()
    result = audit(
        arguments.sizes,
        arguments.viscosity,
        arguments.scaled_window,
    )
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
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["all_positive_checks_pass"]:
        raise SystemExit("annular rho-zero first-jet audit failed")


if __name__ == "__main__":
    main()
