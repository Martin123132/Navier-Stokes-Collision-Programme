"""Audit the exact second jet of the annular rho-zero restart generator."""

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
import sympy as sp

from annular_rho_zero_first_jet_audit import (
    _coefficients,
    _generator_from_coefficients,
    _grid_shape,
    _initial_coefficients,
    _physical,
    _pressure_coefficients,
    _scalar_gradient,
    _spectral_data,
    _vector_gradient,
)
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
    "annular_rho_zero_second_jet_route_guard_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "annular_rho_zero_first_jet_audit_v1.json"
    ): "e07d6511f0ca52484065ba58674594bd9b0a828f4b0525e26caa136153ebcdaf",
    (
        "work/ns_collision/results/"
        "annular_rho_zero_first_jet_remainder_gate_audit_v1.json"
    ): "582a6a4997928b8cd7b67f1d9fd58b5fef6326ee6ffa42bede33a5d9854f36c9",
}
ALGORITHM_REVISION = "annular-rho-zero-second-jet-route-guard-v1"
DEFAULT_HEAT_LOAD_SIZES = (25, 33, 49, 65)
Array = np.ndarray
Field = dict[str, Any]


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
    payloads: dict[str, Any] = {}
    for relative, expected in PREREQUISITES.items():
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = _sha256(path)
        key = Path(relative).stem
        payloads[key] = payload
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
    return (
        {
            "rows": rows,
            "all_checks_pass": all(row["matches"] for row in rows),
        },
        payloads,
    )


def _symbolic_second_variation_certificate() -> dict[str, Any]:
    epsilon, eta, viscosity = sp.symbols(
        "epsilon eta nu", real=True
    )
    pressure, pressure_v, pressure_w, pressure_vw = sp.symbols(
        "p p_v p_w p_vw", real=True
    )
    velocity_pair, pair_v, pair_w = sp.symbols(
        "U V W", real=True
    )
    fisher, fisher_v, fisher_w, fisher_vw = sp.symbols(
        "E F_v F_w J_vw", real=True
    )
    weight = sp.symbols("lambda", real=True)

    velocity_integrand = (
        (
            pressure
            + epsilon * pressure_v
            + eta * pressure_w
            + epsilon * eta * pressure_vw
        )
        * (
            velocity_pair
            + epsilon * pair_v
            + eta * pair_w
        )
        - viscosity
        * weight
        * (
            fisher
            + 2 * epsilon * fisher_v
            + 2 * eta * fisher_w
            + 2 * epsilon * eta * fisher_vw
        )
    )
    velocity_cross = sp.diff(
        velocity_integrand, epsilon, eta
    ).subs({epsilon: 0, eta: 0})
    velocity_expected = (
        pressure_vw * velocity_pair
        + pressure_v * pair_w
        + pressure_w * pair_v
        - 2 * viscosity * weight * fisher_vw
    )

    weight_0, weight_mu, weight_eta = sp.symbols(
        "lambda mu eta_weight", real=True
    )
    weight_fisher, weight_mu_pair, weight_eta_pair = sp.symbols(
        "H K_mu K_eta", real=True
    )
    mixed_weight_fisher = sp.symbols("K_mu_eta", real=True)
    weight_integrand = -viscosity * (
        weight_0 + epsilon * weight_mu + eta * weight_eta
    ) * (
        weight_fisher
        + 2 * epsilon * weight_mu_pair
        + 2 * eta * weight_eta_pair
        + 2 * epsilon * eta * mixed_weight_fisher
    )
    weight_cross = sp.diff(
        weight_integrand, epsilon, eta
    ).subs({epsilon: 0, eta: 0})
    weight_expected = -2 * viscosity * (
        weight_mu * weight_eta_pair
        + weight_eta * weight_mu_pair
        + weight_0 * mixed_weight_fisher
    )

    mixed_pressure, mixed_pair = sp.symbols(
        "p_v M_v", real=True
    )
    base_pressure, base_mixed_pair = sp.symbols(
        "p_0 M_0", real=True
    )
    base_velocity_fisher, mixed_velocity_fisher = sp.symbols(
        "E_0 F_0v", real=True
    )
    mixed_weight = sp.symbols("mu", real=True)
    mixed_integrand = (
        (base_pressure + epsilon * mixed_pressure)
        * (
            velocity_pair
            + epsilon * pair_v
            + eta * base_mixed_pair
            + epsilon * eta * mixed_pair
        )
        - viscosity
        * (weight_0 + eta * mixed_weight)
        * (
            base_velocity_fisher
            + 2 * epsilon * mixed_velocity_fisher
        )
    )
    mixed_cross = sp.diff(
        mixed_integrand, epsilon, eta
    ).subs({epsilon: 0, eta: 0})
    mixed_expected = (
        mixed_pressure * base_mixed_pair
        + base_pressure * mixed_pair
        - 2 * viscosity * mixed_weight * mixed_velocity_fisher
    )

    residuals = {
        "velocity_Hessian": str(
            sp.simplify(velocity_cross - velocity_expected)
        ),
        "mixed_Hessian": str(
            sp.simplify(mixed_cross - mixed_expected)
        ),
        "weight_Hessian": str(
            sp.simplify(weight_cross - weight_expected)
        ),
    }
    return {
        "generator": (
            "g(u,lambda)=integral[p(u)u dot grad lambda"
            "-nu lambda|grad u|^2-nu lambda|grad lambda|^2]"
        ),
        "second_chain_rule": (
            "g''=D_uu g[u_1,u_1]+2D_u_lambda g[u_1,lambda_1]"
            "+D_lambda_lambda g[lambda_1,lambda_1]"
            "+D_u g[u_2]+D_lambda g[lambda_2]"
        ),
        "velocity_Hessian": (
            "D_uu g[v,w]=integral[p''[v,w]u dot grad lambda"
            "+p'[u;v]w dot grad lambda"
            "+p'[u;w]v dot grad lambda"
            "-2nu lambda grad v:grad w]"
        ),
        "mixed_Hessian": (
            "D_u_lambda g[v,mu]=integral[p'[u;v]u dot grad mu"
            "+p v dot grad mu-2nu mu grad u:grad v]"
        ),
        "weight_Hessian": (
            "D_lambda_lambda g[mu,eta]=-2nu integral["
            "mu grad lambda dot grad eta"
            "+eta grad lambda dot grad mu"
            "+lambda grad mu dot grad eta]"
        ),
        "pressure_second_variation": (
            "p''[v,w]=p[v,w]+p[w,v]"
        ),
        "coupled_accelerations": {
            "velocity": (
                "u_2=-P[(u_1 dot grad)u+(u dot grad)u_1]"
                "+nu Delta u_1"
            ),
            "weight": (
                "lambda_2=-u_1 dot grad lambda-u dot grad lambda_1"
                "-nu Delta lambda_1"
            ),
        },
        "symbolic_residuals": residuals,
        "all_checks_pass": all(value == "0" for value in residuals.values()),
    }


def _heat_power_weighted_resonant_pressure_loads(
    waves: Array,
    velocity: Array,
    power: int,
) -> dict[str, float]:
    """Insert a power of the total HHL heat multiplier."""

    if power < 0:
        raise ValueError("heat power must be nonnegative")
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
            heat_sum = (
                np.sum(first_wave * first_wave, axis=-1)
                + np.sum(second_wave * second_wave, axis=-1)
                + low_norm_squared
            )
            heat_weight = heat_sum**power
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


def _second_heat_pressure_limit_certificate(
    predecessor: dict[str, Any],
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
    profile_squared = (
        np.sin(math.pi * (x - 2.0))
        * np.sin(math.pi * (y + 0.5))
        * np.sin(math.pi * (z + 0.5))
    ) ** 2
    radius_squared = x * x + y * y + z * z
    velocity_y = -z * y / radius_squared**1.5
    velocity_z = (x * x + y * y) / radius_squared**1.5
    signed_density = (
        profile_squared
        * (velocity_y * velocity_y - velocity_z * velocity_z)
    )

    raw_integrals = {
        power: float(
            np.sum(
                tensor_weights
                * radius_squared**power
                * signed_density
            )
        )
        for power in (0, 1, 2)
    }
    limits = {
        power: (
            math.sqrt(2.0)
            / 20.0
            * 2.0**power
            * raw_integrals[power]
        )
        for power in raw_integrals
    }
    asymptotic = predecessor[
        "asymptotic_viscous_pressure_certificate"
    ]
    predecessor_limits = {
        0: float(asymptotic["static_pressure_load_limit"]),
        1: float(asymptotic["heat_weighted_pressure_load_limit"]),
    }
    replay_residuals = {
        power: abs(limits[power] - predecessor_limits[power])
        for power in predecessor_limits
    }
    analytic_bounds = {
        0: 51.0 * math.sqrt(2.0) / 438976.0,
        1: 51.0 * math.sqrt(2.0) / 54872.0,
        2: 51.0 * math.sqrt(2.0) / 6859.0,
    }
    return {
        "quadrature": "tensor Gauss-Legendre",
        "order_per_axis": order,
        "continuum_domain": (
            "[2,3] x [-1/2,1/2] x [-1/2,1/2]"
        ),
        "general_limit_formula": (
            "B_(m),N/N^(2m+1) -> (sqrt(2)/20) 2^m "
            "integral_D S^2 |xi|^(2m)(V_y^2-V_z^2)"
        ),
        "raw_integrals": {
            "static_m0": raw_integrals[0],
            "first_heat_m1": raw_integrals[1],
            "second_heat_m2": raw_integrals[2],
        },
        "pressure_load_limits": {
            "static_B0_over_N": limits[0],
            "first_heat_B1_over_N3": limits[1],
            "second_heat_B2_over_N5": limits[2],
        },
        "predecessor_replay_residuals": {
            "static": replay_residuals[0],
            "first_heat": replay_residuals[1],
        },
        "pointwise_sign": (
            "V_z^2-V_y^2>=255/13718 and |xi|^4>=16 on D"
        ),
        "profile_mass": "integral_D S^2=1/8",
        "analytic_absolute_lower_bounds": {
            "static": analytic_bounds[0],
            "first_heat": analytic_bounds[1],
            "second_heat": analytic_bounds[2],
        },
        "second_heat_strict_negative_upper_bound": -analytic_bounds[2],
        "all_checks_pass": bool(
            max(replay_residuals.values()) < 2.0e-16
            and all(
                limits[power] < -analytic_bounds[power]
                for power in (0, 1, 2)
            )
        ),
    }


def _second_heat_asymptotic_certificate(
    limit: dict[str, Any],
    predecessor: dict[str, Any],
    viscosity: float,
) -> dict[str, Any]:
    del viscosity
    pressure_limits = limit["pressure_load_limits"]
    static_limit = float(pressure_limits["static_B0_over_N"])
    second_heat_limit = float(
        pressure_limits["second_heat_B2_over_N5"]
    )
    ray_factor = float(
        predecessor["asymptotic_viscous_pressure_certificate"][
            "ray_factor_sqrt_8_over_3q"
        ]
    )
    coefficient = (
        -(abs(static_limit) ** 2)
        * ray_factor
        * second_heat_limit
    )
    bounds = limit["analytic_absolute_lower_bounds"]
    strict_positive_lower_bound = (
        float(bounds["static"]) ** 2
        * ray_factor
        * float(bounds["second_heat"])
    )
    first_coefficient = float(
        predecessor["asymptotic_viscous_pressure_certificate"][
            "viscous_pressure_first_jet_over_N5_limit"
        ]
    )
    quadratic_turnaround_scale = -first_coefficient / coefficient
    return {
        "theorem": (
            "The pure velocity-heat pressure component of the second "
            "restart jet has a strictly positive N^7 limit."
        ),
        "finite_identity": (
            "D_uu g_pressure[v_nu,v_nu]"
            "+D_u g_pressure[nu Delta v_nu]"
            "=-nu^2 a_N t_N B_(2),N"
        ),
        "second_heat_multiplier": (
            "(|k_1|^2+|k_2|^2+|ell|^2)^2 for every HHL monomial"
        ),
        "optimizer_limits": {
            "a_N_over_N": "|b_0|/nu",
            "t_N_over_N": "|b_0| sqrt(8/(3q))/nu",
        },
        "second_heat_pressure_load_over_N5_limit": second_heat_limit,
        "pure_heat_pressure_second_jet_over_N7_limit": coefficient,
        "analytic_strict_positive_lower_bound": (
            strict_positive_lower_bound
        ),
        "first_jet_N5_limit": first_coefficient,
        "pure_heat_quadratic_slope_turnaround_scale_N2t": (
            quadratic_turnaround_scale
        ),
        "turnaround_guard": (
            "The turnaround scale uses only the negative first coefficient "
            "and positive pure-heat curvature. It is not a prediction for "
            "the full flow because nonlinear second-jet channels, including "
            "a possible scale above N^7, have not yet been bounded."
        ),
        "all_checks_pass": bool(
            limit["all_checks_pass"]
            and second_heat_limit < 0.0
            and coefficient > strict_positive_lower_bound > 0.0
            and first_coefficient < 0.0
            and quadratic_turnaround_scale > 0.0
        ),
    }


def _support_ledger_certificate() -> dict[str, Any]:
    rows = [
        {
            "field": "u",
            "support_radius": "K",
        },
        {
            "field": "lambda",
            "support_radius": "L=1",
        },
        {
            "field": "u_E",
            "support_radius": "2K",
        },
        {
            "field": "u_nu",
            "support_radius": "K",
        },
        {
            "field": "lambda_A",
            "support_radius": "K+L",
        },
        {
            "field": "lambda_nu",
            "support_radius": "L",
        },
        {
            "field": "D E[u_E]",
            "support_radius": "3K",
        },
        {
            "field": "D E[u_nu]",
            "support_radius": "2K",
        },
        {
            "field": "nu Delta u_E",
            "support_radius": "2K",
        },
        {
            "field": "nu Delta u_nu",
            "support_radius": "K",
        },
        {
            "field": "lambda_2",
            "support_radius": "2K+L",
        },
    ]
    return {
        "field_support_rows": rows,
        "maximum_velocity_acceleration_support": "3K",
        "maximum_weight_acceleration_support": "2K+L",
        "maximum_second_jet_integrand_support": "5K+O(L)",
        "required_rectangular_grid_rule": (
            "each grid length exceeds ten times the corresponding "
            "one-field carrier maximum"
        ),
        "implemented_dealias_factor": 10,
        "reason": (
            "Every epsilon^2 coefficient contains either two first "
            "directions or one acceleration. Pressure contributes one "
            "bilinear convolution, so no exact second-jet mean exceeds "
            "five carrier radii. A ten-times one-field grid separates "
            "positive and negative support before the zero-mode mean."
        ),
        "all_checks_pass": True,
    }


def _vector_field(
    label: str,
    coefficients: Array,
    waves: tuple[Array, ...],
    volume: int,
) -> Field:
    return {
        "label": label,
        "coefficients": coefficients,
        "value": _physical(coefficients, volume),
        "gradient": _vector_gradient(coefficients, waves, volume),
    }


def _scalar_field(
    label: str,
    coefficients: Array,
    waves: tuple[Array, ...],
    volume: int,
) -> Field:
    return {
        "label": label,
        "coefficients": coefficients,
        "value": _physical(coefficients, volume),
        "gradient": _scalar_gradient(coefficients, waves, volume),
    }


def _state_and_flow_jets(
    velocity_coefficients: Array,
    weight_coefficients: Array,
    waves: tuple[Array, ...],
    wave_number_squared: Array,
    safe_wave_number_squared: Array,
    volume: int,
    viscosity: float,
) -> dict[str, Any]:
    velocity = _vector_field(
        "u", velocity_coefficients, waves, volume
    )
    weight = _scalar_field(
        "lambda", weight_coefficients, waves, volume
    )
    pressure_coefficients = _pressure_coefficients(
        velocity["value"],
        velocity["value"],
        waves,
        safe_wave_number_squared,
        volume,
    )
    pressure = _scalar_field(
        "p[u,u]", pressure_coefficients, waves, volume
    )

    advection = np.einsum(
        "j...,ij...->i...",
        velocity["value"],
        velocity["gradient"],
    )
    Euler_coefficients = _coefficients(
        -advection - pressure["gradient"], volume
    )
    viscous_coefficients = (
        -viscosity
        * wave_number_squared[None, ...]
        * velocity_coefficients
    )
    velocity_directions = {
        "E": _vector_field(
            "u_E", Euler_coefficients, waves, volume
        ),
        "V": _vector_field(
            "u_nu", viscous_coefficients, waves, volume
        ),
    }

    weight_advection = -np.sum(
        velocity["value"] * weight["gradient"], axis=0
    )
    weight_advection_coefficients = _coefficients(
        weight_advection, volume
    )
    weight_antidiffusion_coefficients = (
        viscosity * wave_number_squared * weight_coefficients
    )
    weight_directions = {
        "A": _scalar_field(
            "lambda_A",
            weight_advection_coefficients,
            waves,
            volume,
        ),
        "D": _scalar_field(
            "lambda_nu",
            weight_antidiffusion_coefficients,
            waves,
            volume,
        ),
    }

    def linearized_Euler(direction: Field, label: str) -> Field:
        pressure_variation_coefficients = _pressure_coefficients(
            velocity["value"],
            direction["value"],
            waves,
            safe_wave_number_squared,
            volume,
            symmetrized=True,
        )
        pressure_variation_gradient = _scalar_gradient(
            pressure_variation_coefficients, waves, volume
        )
        linearized_advection = (
            np.einsum(
                "j...,ij...->i...",
                direction["value"],
                velocity["gradient"],
            )
            + np.einsum(
                "j...,ij...->i...",
                velocity["value"],
                direction["gradient"],
            )
        )
        coefficients = _coefficients(
            -linearized_advection - pressure_variation_gradient,
            volume,
        )
        return _vector_field(label, coefficients, waves, volume)

    def velocity_heat(direction: Field, label: str) -> Field:
        coefficients = (
            -viscosity
            * wave_number_squared[None, ...]
            * direction["coefficients"]
        )
        return _vector_field(label, coefficients, waves, volume)

    velocity_accelerations = {
        "EE": linearized_Euler(
            velocity_directions["E"], "D_E[u_E]"
        ),
        "EV": linearized_Euler(
            velocity_directions["V"], "D_E[u_nu]"
        ),
        "VE": velocity_heat(
            velocity_directions["E"], "nu_Delta_u_E"
        ),
        "VV": velocity_heat(
            velocity_directions["V"], "nu_Delta_u_nu"
        ),
    }

    def scalar_product_field(label: str, values: Array) -> Field:
        return _scalar_field(
            label, _coefficients(values, volume), waves, volume
        )

    weight_accelerations = {
        "E0": scalar_product_field(
            "-u_E_dot_grad_lambda",
            -np.sum(
                velocity_directions["E"]["value"]
                * weight["gradient"],
                axis=0,
            ),
        ),
        "V0": scalar_product_field(
            "-u_nu_dot_grad_lambda",
            -np.sum(
                velocity_directions["V"]["value"]
                * weight["gradient"],
                axis=0,
            ),
        ),
        "0A": scalar_product_field(
            "-u_dot_grad_lambda_A",
            -np.sum(
                velocity["value"]
                * weight_directions["A"]["gradient"],
                axis=0,
            ),
        ),
        "0D": scalar_product_field(
            "-u_dot_grad_lambda_nu",
            -np.sum(
                velocity["value"]
                * weight_directions["D"]["gradient"],
                axis=0,
            ),
        ),
        "DA": _scalar_field(
            "-nu_Delta_lambda_A",
            viscosity
            * wave_number_squared
            * weight_directions["A"]["coefficients"],
            waves,
            volume,
        ),
        "DD": _scalar_field(
            "-nu_Delta_lambda_nu",
            viscosity
            * wave_number_squared
            * weight_directions["D"]["coefficients"],
            waves,
            volume,
        ),
    }
    return {
        "velocity": velocity,
        "weight": weight,
        "pressure": pressure,
        "velocity_directions": velocity_directions,
        "weight_directions": weight_directions,
        "velocity_accelerations": velocity_accelerations,
        "weight_accelerations": weight_accelerations,
    }


def _scaled_terms(terms: dict[str, float], factor: float) -> dict[str, float]:
    return {label: factor * value for label, value in terms.items()}


def _term_record(
    terms: dict[str, float],
    factor: float = 1.0,
) -> dict[str, Any]:
    scaled = _scaled_terms(terms, factor)
    return {
        "multiplicity": factor,
        "subterms": scaled,
        "value": sum(scaled.values()),
    }


def _second_variation_decomposition(
    jets: dict[str, Any],
    waves: tuple[Array, ...],
    safe_wave_number_squared: Array,
    volume: int,
    viscosity: float,
) -> dict[str, Any]:
    velocity = jets["velocity"]
    weight = jets["weight"]
    pressure = jets["pressure"]
    velocity_directions = jets["velocity_directions"]
    weight_directions = jets["weight_directions"]
    velocity_accelerations = jets["velocity_accelerations"]
    weight_accelerations = jets["weight_accelerations"]
    pressure_variations: dict[str, Array] = {}
    pressure_pairs: dict[tuple[str, str], Array] = {}

    def pressure_variation(direction: Field) -> Array:
        label = direction["label"]
        if label not in pressure_variations:
            coefficients = _pressure_coefficients(
                velocity["value"],
                direction["value"],
                waves,
                safe_wave_number_squared,
                volume,
                symmetrized=True,
            )
            pressure_variations[label] = _physical(
                coefficients, volume
            )
        return pressure_variations[label]

    def pressure_pair(first: Field, second: Field) -> Array:
        key = tuple(sorted((first["label"], second["label"])))
        if key not in pressure_pairs:
            coefficients = _pressure_coefficients(
                first["value"],
                second["value"],
                waves,
                safe_wave_number_squared,
                volume,
                symmetrized=True,
            )
            pressure_pairs[key] = _physical(coefficients, volume)
        return pressure_pairs[key]

    def D_u(direction: Field) -> dict[str, float]:
        return {
            "pressure_variation": float(
                np.mean(
                    pressure_variation(direction)
                    * np.sum(
                        velocity["value"] * weight["gradient"],
                        axis=0,
                    )
                )
            ),
            "pressure_direction": float(
                np.mean(
                    pressure["value"]
                    * np.sum(
                        direction["value"] * weight["gradient"],
                        axis=0,
                    )
                )
            ),
            "weighted_Fisher": float(
                np.mean(
                    -2.0
                    * viscosity
                    * weight["value"]
                    * np.sum(
                        velocity["gradient"]
                        * direction["gradient"],
                        axis=(0, 1),
                    )
                )
            ),
        }

    velocity_gradient_squared = np.sum(
        velocity["gradient"] * velocity["gradient"], axis=(0, 1)
    )
    weight_gradient_squared = np.sum(
        weight["gradient"] * weight["gradient"], axis=0
    )

    def D_weight(direction: Field) -> dict[str, float]:
        return {
            "pressure": float(
                np.mean(
                    pressure["value"]
                    * np.sum(
                        velocity["value"] * direction["gradient"],
                        axis=0,
                    )
                )
            ),
            "velocity_Fisher": float(
                np.mean(
                    -viscosity
                    * direction["value"]
                    * velocity_gradient_squared
                )
            ),
            "weight_Fisher_direction": float(
                np.mean(
                    -viscosity
                    * direction["value"]
                    * weight_gradient_squared
                )
            ),
            "weight_Fisher_cross": float(
                np.mean(
                    -2.0
                    * viscosity
                    * weight["value"]
                    * np.sum(
                        weight["gradient"] * direction["gradient"],
                        axis=0,
                    )
                )
            ),
        }

    def H_uu(first: Field, second: Field) -> dict[str, float]:
        return {
            "pressure_second_variation": float(
                np.mean(
                    pressure_pair(first, second)
                    * np.sum(
                        velocity["value"] * weight["gradient"],
                        axis=0,
                    )
                )
            ),
            "pressure_first_variation_first": float(
                np.mean(
                    pressure_variation(first)
                    * np.sum(
                        second["value"] * weight["gradient"],
                        axis=0,
                    )
                )
            ),
            "pressure_first_variation_second": float(
                np.mean(
                    pressure_variation(second)
                    * np.sum(
                        first["value"] * weight["gradient"],
                        axis=0,
                    )
                )
            ),
            "weighted_Fisher": float(
                np.mean(
                    -2.0
                    * viscosity
                    * weight["value"]
                    * np.sum(
                        first["gradient"] * second["gradient"],
                        axis=(0, 1),
                    )
                )
            ),
        }

    def H_u_weight(
        velocity_direction: Field,
        weight_direction: Field,
    ) -> dict[str, float]:
        return {
            "pressure_variation": float(
                np.mean(
                    pressure_variation(velocity_direction)
                    * np.sum(
                        velocity["value"]
                        * weight_direction["gradient"],
                        axis=0,
                    )
                )
            ),
            "pressure_direction": float(
                np.mean(
                    pressure["value"]
                    * np.sum(
                        velocity_direction["value"]
                        * weight_direction["gradient"],
                        axis=0,
                    )
                )
            ),
            "weighted_Fisher": float(
                np.mean(
                    -2.0
                    * viscosity
                    * weight_direction["value"]
                    * np.sum(
                        velocity["gradient"]
                        * velocity_direction["gradient"],
                        axis=(0, 1),
                    )
                )
            ),
        }

    def H_weight_weight(
        first: Field,
        second: Field,
    ) -> dict[str, float]:
        return {
            "first_weight_direction": float(
                np.mean(
                    -2.0
                    * viscosity
                    * first["value"]
                    * np.sum(
                        weight["gradient"] * second["gradient"],
                        axis=0,
                    )
                )
            ),
            "second_weight_direction": float(
                np.mean(
                    -2.0
                    * viscosity
                    * second["value"]
                    * np.sum(
                        weight["gradient"] * first["gradient"],
                        axis=0,
                    )
                )
            ),
            "mixed_weight_gradient": float(
                np.mean(
                    -2.0
                    * viscosity
                    * weight["value"]
                    * np.sum(
                        first["gradient"] * second["gradient"],
                        axis=0,
                    )
                )
            ),
        }

    channels: dict[str, dict[str, Any]] = {}
    channels["H_uu[E,E]"] = _term_record(
        H_uu(velocity_directions["E"], velocity_directions["E"])
    )
    channels["2H_uu[E,V]"] = _term_record(
        H_uu(velocity_directions["E"], velocity_directions["V"]),
        2.0,
    )
    channels["H_uu[V,V]"] = _term_record(
        H_uu(velocity_directions["V"], velocity_directions["V"])
    )
    for velocity_label, velocity_direction in (
        velocity_directions.items()
    ):
        for weight_label, weight_direction in (
            weight_directions.items()
        ):
            channels[
                f"2H_u_lambda[{velocity_label},{weight_label}]"
            ] = _term_record(
                H_u_weight(velocity_direction, weight_direction),
                2.0,
            )
    channels["H_lambda_lambda[A,A]"] = _term_record(
        H_weight_weight(
            weight_directions["A"], weight_directions["A"]
        )
    )
    channels["2H_lambda_lambda[A,D]"] = _term_record(
        H_weight_weight(
            weight_directions["A"], weight_directions["D"]
        ),
        2.0,
    )
    channels["H_lambda_lambda[D,D]"] = _term_record(
        H_weight_weight(
            weight_directions["D"], weight_directions["D"]
        )
    )
    for label, acceleration in velocity_accelerations.items():
        channels[f"D_u[u2_{label}]"] = _term_record(D_u(acceleration))
    for label, acceleration in weight_accelerations.items():
        channels[f"D_lambda[lambda2_{label}]"] = _term_record(
            D_weight(acceleration)
        )

    def combined_vector(
        label: str, fields: Sequence[Field]
    ) -> Field:
        return _vector_field(
            label,
            sum(field["coefficients"] for field in fields),
            waves,
            volume,
        )

    def combined_scalar(
        label: str, fields: Sequence[Field]
    ) -> Field:
        return _scalar_field(
            label,
            sum(field["coefficients"] for field in fields),
            waves,
            volume,
        )

    velocity_first = combined_vector(
        "u_1", tuple(velocity_directions.values())
    )
    weight_first = combined_scalar(
        "lambda_1", tuple(weight_directions.values())
    )
    velocity_second = combined_vector(
        "u_2", tuple(velocity_accelerations.values())
    )
    weight_second = combined_scalar(
        "lambda_2", tuple(weight_accelerations.values())
    )
    direct_blocks = {
        "D_uu": _term_record(H_uu(velocity_first, velocity_first)),
        "2D_u_lambda": _term_record(
            H_u_weight(velocity_first, weight_first), 2.0
        ),
        "D_lambda_lambda": _term_record(
            H_weight_weight(weight_first, weight_first)
        ),
        "D_u_acceleration": _term_record(D_u(velocity_second)),
        "D_lambda_acceleration": _term_record(
            D_weight(weight_second)
        ),
    }
    expanded_value = sum(
        channel["value"] for channel in channels.values()
    )
    direct_value = sum(
        block["value"] for block in direct_blocks.values()
    )
    pure_heat_Hessian = channels["H_uu[V,V]"]["subterms"]
    pure_heat_acceleration = channels["D_u[u2_VV]"]["subterms"]
    pure_heat_pressure = sum(
        value
        for label, value in pure_heat_Hessian.items()
        if label.startswith("pressure")
    ) + sum(
        value
        for label, value in pure_heat_acceleration.items()
        if label.startswith("pressure")
    )
    return {
        "channels": channels,
        "direct_chain_rule_blocks": direct_blocks,
        "expanded_second_derivative": expanded_value,
        "direct_second_derivative": direct_value,
        "decomposition_residual": abs(expanded_value - direct_value),
        "pure_velocity_heat_pressure_second_derivative": (
            pure_heat_pressure
        ),
        "combined_fields": {
            "velocity_first": velocity_first,
            "weight_first": weight_first,
            "velocity_second": velocity_second,
            "weight_second": weight_second,
        },
        "velocity_direction_divergence_residuals": {
            label: float(
                np.max(
                    np.abs(
                        sum(
                            waves[index]
                            * field["coefficients"][index]
                            for index in range(3)
                        )
                    )
                )
            )
            for label, field in (
                list(velocity_directions.items())
                + list(velocity_accelerations.items())
            )
        },
    }


def _optimal_amplitudes(
    size: int,
    viscosity: float,
) -> tuple[float, float, float, float]:
    waves, velocity, parity = _family_arrays(
        (size, size, size), 2 * size
    )
    loads = _resonant_component_loads(waves, velocity)
    pressure_load = (
        loads["pressure_high_high"] + loads["pressure_cross"]
    )
    Fisher = _mixed_difference_fisher(waves, velocity, parity)
    optimum = _joint_ray_optimum(
        pressure_load,
        Fisher,
        1.0,
        float(DELTA_CUBIC_ENERGY),
        viscosity,
    )
    return (
        float(optimum["optimal_oriented_low_amplitude"]),
        float(optimum["optimal_coefficient_scale"]),
        float(pressure_load),
        float(Fisher),
    )


def _second_jet_row(
    size: int,
    viscosity: float = 1.0,
    dealias_factor: int = 10,
    low_amplitude_override: float | None = None,
    coefficient_scale_override: float | None = None,
    finite_difference_epsilon: float | None = None,
) -> dict[str, Any]:
    if dealias_factor < 10:
        raise ValueError("second-jet de-alias factor must be at least ten")
    started = time.perf_counter()
    shape = _grid_shape(size, dealias_factor)
    (
        spectral_waves,
        wave_number_squared,
        safe_wave_number_squared,
        volume,
    ) = _spectral_data(shape)
    (
        optimal_low_amplitude,
        optimal_coefficient_scale,
        pressure_load,
        high_Fisher,
    ) = _optimal_amplitudes(size, viscosity)
    low_amplitude = (
        float(low_amplitude_override)
        if low_amplitude_override is not None
        else optimal_low_amplitude
    )
    coefficient_scale = (
        float(coefficient_scale_override)
        if coefficient_scale_override is not None
        else optimal_coefficient_scale
    )
    if low_amplitude <= 0.0 or coefficient_scale <= 0.0:
        raise ValueError("second-jet amplitudes must be positive")
    (
        velocity_coefficients,
        weight_coefficients,
        family_waves,
        family_velocity,
        _,
    ) = _initial_coefficients(
        size,
        shape,
        low_amplitude,
        coefficient_scale,
    )
    jets = _state_and_flow_jets(
        velocity_coefficients,
        weight_coefficients,
        spectral_waves,
        wave_number_squared,
        safe_wave_number_squared,
        volume,
        viscosity,
    )
    decomposition = _second_variation_decomposition(
        jets,
        spectral_waves,
        safe_wave_number_squared,
        volume,
        viscosity,
    )
    second_heat_loads = (
        _heat_power_weighted_resonant_pressure_loads(
            family_waves, family_velocity, 2
        )
    )
    expected_pure_heat_pressure = (
        -(viscosity**2)
        * low_amplitude
        * coefficient_scale
        * second_heat_loads["combined"]
    )
    pure_heat_replay_residual = abs(
        decomposition[
            "pure_velocity_heat_pressure_second_derivative"
        ]
        - expected_pure_heat_pressure
    )

    fields = decomposition.pop("combined_fields")
    base_generator = _generator_from_coefficients(
        velocity_coefficients,
        weight_coefficients,
        spectral_waves,
        wave_number_squared,
        safe_wave_number_squared,
        volume,
        viscosity,
    )
    finite_difference = None
    if finite_difference_epsilon is not None:
        epsilon = float(finite_difference_epsilon)

        def quotient(step: float) -> float:
            plus_velocity = (
                velocity_coefficients
                + step * fields["velocity_first"]["coefficients"]
                + 0.5
                * step**2
                * fields["velocity_second"]["coefficients"]
            )
            minus_velocity = (
                velocity_coefficients
                - step * fields["velocity_first"]["coefficients"]
                + 0.5
                * step**2
                * fields["velocity_second"]["coefficients"]
            )
            plus_weight = (
                weight_coefficients
                + step * fields["weight_first"]["coefficients"]
                + 0.5
                * step**2
                * fields["weight_second"]["coefficients"]
            )
            minus_weight = (
                weight_coefficients
                - step * fields["weight_first"]["coefficients"]
                + 0.5
                * step**2
                * fields["weight_second"]["coefficients"]
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
            return (plus_value - 2.0 * base_generator + minus_value) / (
                step**2
            )

        coarse = quotient(epsilon)
        fine = quotient(epsilon / 2.0)
        Richardson = (4.0 * fine - coarse) / 3.0
        analytic = decomposition["direct_second_derivative"]
        finite_difference = {
            "epsilon": epsilon,
            "coarse_central_second_difference": coarse,
            "fine_central_second_difference": fine,
            "Richardson_extrapolation": Richardson,
            "analytic_second_derivative": analytic,
            "coarse_to_fine_change": abs(fine - coarse),
            "absolute_residual": abs(Richardson - analytic),
            "relative_residual": abs(Richardson - analytic)
            / max(abs(analytic), 1.0),
        }

    divergence_residual = max(
        decomposition["velocity_direction_divergence_residuals"].values()
    )
    finite_difference_passes = bool(
        finite_difference is None
        or finite_difference["relative_residual"] < 2.0e-7
    )
    row = {
        "size": size,
        "grid_shape": list(shape),
        "grid_point_count": volume,
        "dealias_factor": dealias_factor,
        "low_amplitude": low_amplitude,
        "coefficient_scale": coefficient_scale,
        "static_optimizer_used": bool(
            low_amplitude_override is None
            and coefficient_scale_override is None
        ),
        "rho_zero_pressure_HHL_load": pressure_load,
        "plus_vertex_high_Fisher": high_Fisher,
        "base_generator": base_generator,
        "second_heat_weighted_pressure_loads": second_heat_loads,
        "second_heat_pressure_load_over_N5": (
            second_heat_loads["combined"] / size**5
        ),
        "expected_pure_velocity_heat_pressure_second_derivative": (
            expected_pure_heat_pressure
        ),
        "pure_heat_pressure_replay_residual": pure_heat_replay_residual,
        "pure_heat_pressure_second_derivative_over_N7": (
            decomposition[
                "pure_velocity_heat_pressure_second_derivative"
            ]
            / size**7
        ),
        "second_variation": decomposition,
        "finite_difference_validation": finite_difference,
        "maximum_velocity_divergence_residual": divergence_residual,
        "runtime_seconds": time.perf_counter() - started,
        "all_checks_pass": bool(
            decomposition["decomposition_residual"] < 2.0e-8
            and pure_heat_replay_residual < 2.0e-7
            and second_heat_loads["maximum_imaginary_residual"] < 3.0e-7
            and divergence_residual < 3.0e-7
            and finite_difference_passes
        ),
    }
    del jets
    del fields
    gc.collect()
    return row


def _heat_load_row(
    size: int,
    viscosity: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    waves, velocity, _ = _family_arrays(
        (size, size, size), 2 * size
    )
    low_amplitude, coefficient_scale, _, _ = _optimal_amplitudes(
        size, viscosity
    )
    loads = _heat_power_weighted_resonant_pressure_loads(
        waves, velocity, 2
    )
    contribution = (
        -(viscosity**2)
        * low_amplitude
        * coefficient_scale
        * loads["combined"]
    )
    return {
        "size": size,
        "second_heat_weighted_pressure_loads": loads,
        "second_heat_pressure_load_over_N5": (
            loads["combined"] / size**5
        ),
        "optimal_low_amplitude": low_amplitude,
        "optimal_coefficient_scale": coefficient_scale,
        "pure_heat_pressure_second_jet": contribution,
        "pure_heat_pressure_second_jet_over_N7": (
            contribution / size**7
        ),
        "runtime_seconds": time.perf_counter() - started,
        "all_checks_pass": bool(
            loads["combined"] < 0.0
            and loads["maximum_imaginary_residual"] < 3.0e-7
            and contribution > 0.0
        ),
    }


def _power_route_guard_certificate() -> dict[str, Any]:
    rows = [
        {
            "channel_group": "double_velocity_heat_pressure",
            "exact_components": (
                "H_uu[V,V]_pressure + D_u[u2_VV]_pressure"
            ),
            "route_power": 7,
            "status": "certified_positive_N7_limit",
            "can_affect_full_N7_coefficient": True,
        },
        {
            "channel_group": "double_velocity_heat_weighted_Fisher",
            "exact_components": (
                "H_uu[V,V]_Fisher + D_u[u2_VV]_Fisher"
            ),
            "route_power": 5,
            "status": (
                "subcritical_route_from_compatible_triple-difference "
                "count; full bound deferred"
            ),
            "can_affect_full_N7_coefficient": False,
        },
        {
            "channel_group": "one_heat_one_Euler_velocity",
            "exact_components": (
                "2H_uu[E,V] + D_u[u2_EV] + D_u[u2_VE]"
            ),
            "route_power": 6,
            "status": (
                "predecessor O(N4) route plus one high Laplacian; "
                "projector-stencil proof deferred"
            ),
            "can_affect_full_N7_coefficient": False,
        },
        {
            "channel_group": "velocity_heat_weight_transport",
            "exact_components": (
                "2H_u_lambda[V,A or D] and corresponding lambda2 terms"
            ),
            "route_power": 6,
            "status": (
                "one high Laplacian route; compatible-stencil proof "
                "deferred"
            ),
            "can_affect_full_N7_coefficient": False,
        },
        {
            "channel_group": "pure_nonlinear_velocity_pressure",
            "exact_components": (
                "H_uu[E,E] + D_u[u2_EE], pressure subterms"
            ),
            "route_power": 9,
            "status": (
                "unresolved route guard: the four-high/one-low amplitude "
                "branch is permitted c_1,N at fixed-output N7 scale, "
                "producing optimized N9 after a_N t_N; a finite-output "
                "proof is "
                "required"
            ),
            "can_affect_full_N7_coefficient": True,
        },
        {
            "channel_group": "pure_weight_transport_and_mixed_pressure",
            "exact_components": (
                "H_u_lambda[E,A], H_lambda_lambda[A,A], "
                "and lambda2 transport terms"
            ),
            "route_power": 7,
            "status": (
                "unresolved route guard: low-output incidence and "
                "compatible differences must be counted"
            ),
            "can_affect_full_N7_coefficient": True,
        },
        {
            "channel_group": "fixed_weight_antidiffusion_only",
            "exact_components": (
                "D-weight directions without velocity heat or transport"
            ),
            "route_power": 5,
            "status": (
                "the weight has fixed Fourier support, so "
                "-nu Delta lambda supplies no carrier N2"
            ),
            "can_affect_full_N7_coefficient": False,
        },
    ]
    unresolved = [
        row["channel_group"]
        for row in rows
        if row["can_affect_full_N7_coefficient"]
        and not row["status"].startswith("certified")
    ]
    return {
        "rule": (
            "A high velocity Laplacian can add two carrier powers; "
            "backward anti-diffusion of the fixed weight cannot. Route "
            "powers are triage flags, not proved asymptotic bounds. In "
            "particular, differentiating the four-high Euler pressure "
            "branch is not controlled by adding two powers to the prior "
            "first-jet total bound."
        ),
        "rows": rows,
        "certified_N7_channel": "double_velocity_heat_pressure",
        "unresolved_possible_N7_channel_groups": unresolved,
        "full_N7_coefficient_certified": False,
        "all_channels_above_N7_excluded": False,
        "large_carrier_FFT_authorized": False,
        "next_required_proof": (
            "Project the combined inviscid pressure/transport sector onto "
            "its low-amplitude branches. Resolve the four-high coefficient "
            "at its candidate N7 fixed-amplitude scale before assigning "
            "any full second-jet leading power."
        ),
        "all_checks_pass": bool(
            len(unresolved) == 2
            and any(
                row["status"] == "certified_positive_N7_limit"
                for row in rows
            )
        ),
    }


def audit(
    heat_load_sizes: Sequence[int] = DEFAULT_HEAT_LOAD_SIZES,
    viscosity: float = 1.0,
) -> dict[str, Any]:
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive")
    clean_sizes = tuple(int(size) for size in heat_load_sizes)
    if (
        not clean_sizes
        or any(size < 25 or size % 2 == 0 for size in clean_sizes)
        or tuple(sorted(set(clean_sizes))) != clean_sizes
    ):
        raise ValueError(
            "heat-load sizes must be increasing distinct odd integers "
            "at least 25"
        )
    prerequisite, payloads = _prerequisite_audit()
    predecessor = payloads["annular_rho_zero_first_jet_audit_v1"]
    symbolic = _symbolic_second_variation_certificate()
    support = _support_ledger_certificate()
    limit = _second_heat_pressure_limit_certificate(predecessor)
    asymptotic = _second_heat_asymptotic_certificate(
        limit, predecessor, viscosity
    )
    validation = _second_jet_row(
        3,
        viscosity,
        dealias_factor=10,
        low_amplitude_override=0.7,
        coefficient_scale_override=0.9,
        finite_difference_epsilon=2.0e-4,
    )
    padding_replay = _second_jet_row(
        3,
        viscosity,
        dealias_factor=12,
        low_amplitude_override=0.7,
        coefficient_scale_override=0.9,
    )
    channel_labels = tuple(
        validation["second_variation"]["channels"]
    )
    padding_channel_residual = max(
        abs(
            validation["second_variation"]["channels"][label]["value"]
            - padding_replay["second_variation"]["channels"][label]["value"]
        )
        for label in channel_labels
    )
    padding_total_residual = abs(
        validation["second_variation"]["direct_second_derivative"]
        - padding_replay["second_variation"]["direct_second_derivative"]
    )
    fixed_small_carrier = _second_jet_row(
        5,
        viscosity,
        dealias_factor=10,
        low_amplitude_override=0.6,
        coefficient_scale_override=0.8,
    )
    heat_rows = [
        _heat_load_row(size, viscosity) for size in clean_sizes
    ]
    route_guard = _power_route_guard_certificate()
    all_checks = bool(
        prerequisite["all_checks_pass"]
        and symbolic["all_checks_pass"]
        and support["all_checks_pass"]
        and limit["all_checks_pass"]
        and asymptotic["all_checks_pass"]
        and validation["all_checks_pass"]
        and padding_replay["all_checks_pass"]
        and fixed_small_carrier["all_checks_pass"]
        and padding_channel_residual < 2.0e-7
        and padding_total_residual < 2.0e-7
        and all(row["all_checks_pass"] for row in heat_rows)
        and route_guard["all_checks_pass"]
    )
    result = {
        "kind": "annular_rho_zero_second_jet_route_guard_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": (
            "annular_rho_zero_second_jet_route_guard_certified"
            if all_checks
            else "annular_rho_zero_second_jet_route_guard_failed"
        ),
        "scope": (
            "The exact restart-time second derivative of the rho=0 "
            "generator, including Navier-Stokes acceleration, pressure "
            "Hessians, and backward-weight acceleration. The pure "
            "velocity-heat pressure N7 limit is certified. The full N7 "
            "coefficient and any finite parabolic window remain open."
        ),
        "prerequisite_audit": prerequisite,
        "symbolic_second_variation_certificate": symbolic,
        "second_jet_support_ledger": support,
        "second_heat_pressure_limit_certificate": limit,
        "pure_heat_pressure_asymptotic_certificate": asymptotic,
        "small_carrier_validation": validation,
        "padding_replay": {
            "base_grid_shape": validation["grid_shape"],
            "padded_grid_shape": padding_replay["grid_shape"],
            "maximum_channel_residual": padding_channel_residual,
            "total_second_derivative_residual": padding_total_residual,
            "all_checks_pass": bool(
                padding_channel_residual < 2.0e-7
                and padding_total_residual < 2.0e-7
            ),
        },
        "fixed_amplitude_second_small_carrier_row": fixed_small_carrier,
        "finite_second_heat_load_rows": heat_rows,
        "second_jet_power_route_guard": route_guard,
        "certification_flags": {
            "exact_second_variation_formula_proved": True,
            "Navier_Stokes_acceleration_retained": True,
            "pressure_Hessian_retained": True,
            "backward_weight_second_derivative_retained": True,
            "tenfold_second_jet_dealiasing_validated": True,
            "pure_heat_pressure_N7_limit_certified": True,
            "pure_heat_pressure_N7_coefficient_positive": True,
            "full_second_jet_N7_coefficient_certified": False,
            "all_second_jet_channels_above_N7_excluded": False,
            "uniform_second_jet_Taylor_bound_proved": False,
            "required_N2_amplification_excluded": False,
            "finite_parabolic_window_controlled": False,
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
        "--heat-load-sizes",
        type=_parse_sizes,
        default=DEFAULT_HEAT_LOAD_SIZES,
    )
    parser.add_argument("--viscosity", type=float, default=1.0)
    arguments = parser.parse_args()
    _lower_process_priority()
    result = audit(arguments.heat_load_sizes, arguments.viscosity)
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
                "pure_heat_pressure_asymptotic_certificate": result[
                    "pure_heat_pressure_asymptotic_certificate"
                ],
                "second_jet_power_route_guard": result[
                    "second_jet_power_route_guard"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["all_positive_checks_pass"]:
        raise SystemExit("annular rho-zero second-jet route guard failed")


if __name__ == "__main__":
    main()
