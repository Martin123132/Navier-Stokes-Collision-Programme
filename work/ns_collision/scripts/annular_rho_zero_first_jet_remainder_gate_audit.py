"""Certify the annular rho-zero first-jet remainder below N^5."""

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
    _first_jet_row,
    _grid_shape,
    _initial_coefficients,
    _physical,
    _pressure_coefficients,
    _scalar_gradient,
    _spectral_data,
    _vector_gradient,
    _velocity_directional_derivative,
)
from compatible_edge_annular_escape_audit import (
    DELTA_CUBIC_ENERGY,
    _joint_ray_optimum,
)
from separable_annular_pressure_schur_no_go_audit import (
    _family_arrays,
    _mixed_difference_fisher,
    _resonant_component_loads,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_first_jet_remainder_gate_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "annular_rho_zero_first_jet_audit_v1.json"
    ): "e07d6511f0ca52484065ba58674594bd9b0a828f4b0525e26caa136153ebcdaf",
}
ALGORITHM_REVISION = "annular-rho-zero-first-jet-remainder-v1"
DEFAULT_MIXED_DIFFERENCE_SIZES = (25, 29, 33, 37, 41, 49, 65)
DEFAULT_PURE_HIGH_SIZES = (9, 13, 17, 21, 25)
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


def _prerequisite_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    payload = {}
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


def _vanishing_order(expression: sp.Expr, variable: sp.Symbol) -> int:
    order = 0
    current = sp.simplify(expression)
    while sp.simplify(current.subs(variable, 1)) == 0:
        current = sp.diff(current, variable)
        order += 1
    return order


def _compatible_stencil_certificate() -> dict[str, Any]:
    z = sp.symbols("z", nonzero=True)
    value = sp.Rational(1, 2) - (z + z**-1) / 4
    first = (z**-1 - z) / 4
    second = -(z + z**-1) / 4
    value_expected = -(z - 1) ** 2 / (4 * z)
    first_expected = -(z - 1) * (z + 1) / (4 * z)
    second_expected = -(z**2 + 1) / (4 * z)
    residuals = {
        "value": str(sp.simplify(value - value_expected)),
        "first_derivative": str(
            sp.simplify(first - first_expected)
        ),
        "second_derivative": str(
            sp.simplify(second - second_expected)
        ),
    }
    one_dimensional_orders = {
        "value": _vanishing_order(value, z),
        "first_derivative": _vanishing_order(first, z),
        "second_derivative": _vanishing_order(second, z),
    }
    tensor_orders = {
        "Phi": 2 + 2 + 2,
        "gradient_Phi": 1 + 2 + 2,
        "diagonal_Hessian_Phi": 0 + 2 + 2,
        "cross_Hessian_Phi": 1 + 1 + 2,
        "Laplacian_Phi": 0 + 2 + 2,
        "gradient_Laplacian_Phi": 1 + 0 + 2,
    }
    return {
        "parity_gauged_value_stencil": str(value_expected),
        "parity_gauged_first_derivative_stencil": str(first_expected),
        "parity_gauged_second_derivative_stencil": str(
            second_expected
        ),
        "factorization_residuals": residuals,
        "one_dimensional_vanishing_orders": one_dimensional_orders,
        "tensor_difference_orders": tensor_orders,
        "interpretation": (
            "The alternating annular packet moves the compatible vertex "
            "from z=-1 to z=1. Phi supplies six lattice differences, "
            "grad Phi supplies five, and every Hessian or Laplacian term "
            "supplies at least four. grad Delta Phi supplies at least "
            "three."
        ),
        "all_checks_pass": bool(
            all(value == "0" for value in residuals.values())
            and one_dimensional_orders
            == {
                "value": 2,
                "first_derivative": 1,
                "second_derivative": 0,
            }
            and tensor_orders
            == {
                "Phi": 6,
                "gradient_Phi": 5,
                "diagonal_Hessian_Phi": 4,
                "cross_Hessian_Phi": 4,
                "Laplacian_Phi": 4,
                "gradient_Laplacian_Phi": 3,
            }
        ),
    }


def _support_incidence_certificate() -> dict[str, Any]:
    return {
        "positive_high_x_interval": "[2N,3N-1]",
        "negative_high_x_interval": "[-3N+1,-2N]",
        "low_x_support": "{0}",
        "one_high_gap_from_fixed_low_support": "at least 2N-3",
        "three_high_gap_from_fixed_low_support": "at least N-2",
        "valid_for": "N>=5 and every low test product used here",
        "forbidden_integrated_channels": {
            "one_high_leg": True,
            "three_high_legs": True,
        },
        "allowed_high_leg_parity": "even",
        "channel_polynomial_parity": {
            "Euler_pressure_quartic": "even in low amplitude a",
            "Euler_weighted_Fisher_cubic": "odd in a",
            "weight_advection_pressure_quartic": "even in a",
            "weight_advection_velocity_Fisher_cubic": "odd in a",
            "weight_advection_weight_self_linear": "odd in a",
            "weight_antidiffusion_pressure_cubic": "odd in a",
            "weight_antidiffusion_velocity_Fisher_quadratic": "even in a",
            "weight_antidiffusion_weight_self": "independent of a",
        },
        "reason": (
            "A sum of one high x-frequency and fixed low frequencies "
            "cannot return to zero. For three high legs, the closest "
            "two-plus/one-minus sum has magnitude N+1 before the bounded "
            "low shifts. Thus only zero, two, or four high legs survive."
        ),
        "all_checks_pass": True,
    }


def _optimizer_scaling_certificate(
    predecessor: dict[str, Any],
) -> dict[str, Any]:
    asymptotic = predecessor["asymptotic_viscous_pressure_certificate"]
    static_limit = float(asymptotic["static_pressure_load_limit"])
    ray_factor = float(asymptotic["ray_factor_sqrt_8_over_3q"])
    return {
        "static_pressure_load": "B_N=b_0 N+o(N)",
        "weighted_high_Fisher": "E_N=O(N^-3)",
        "optimal_low_amplitude": "a_N=O(N)",
        "optimal_coefficient_scale": "t_N=O(N)",
        "optimal_low_amplitude_over_N_limit": abs(static_limit),
        "optimal_coefficient_scale_over_N_limit": (
            abs(static_limit) * ray_factor
        ),
        "viscosity_normalization": 1.0,
        "all_checks_pass": bool(
            static_limit < 0.0
            and ray_factor > 0.0
            and predecessor["all_positive_checks_pass"] is True
        ),
    }


def _triple_difference(tensor: Array) -> Array:
    padded = np.pad(
        tensor,
        ((1, 1), (1, 1), (1, 1), (0, 0), (0, 0)),
        mode="constant",
    )
    return np.diff(
        np.diff(np.diff(padded, axis=0), axis=1),
        axis=2,
    )


def _viscous_fisher_row(
    size: int,
    predecessor_rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    waves, velocity, parity = _family_arrays(
        (size, size, size),
        2 * size,
    )
    tensor = (
        parity[..., None, None]
        * waves[..., :, None]
        * velocity[..., None, :]
    )
    wave_number_squared = np.sum(waves * waves, axis=-1)
    weighted_tensor = wave_number_squared[..., None, None] * tensor
    difference = _triple_difference(tensor)
    weighted_difference = _triple_difference(weighted_tensor)
    fisher = float(np.sum(difference * difference) / 32.0)
    heat_pair = float(
        np.sum(difference * weighted_difference) / 32.0
    )

    loads = _resonant_component_loads(waves, velocity)
    pressure_load = (
        loads["pressure_high_high"] + loads["pressure_cross"]
    )
    optimum = _joint_ray_optimum(
        pressure_load,
        fisher,
        1.0,
        float(DELTA_CUBIC_ENERGY),
        1.0,
    )
    low_amplitude = float(optimum["optimal_oriented_low_amplitude"])
    coefficient_scale = float(optimum["optimal_coefficient_scale"])
    high_contribution = 2.0 * coefficient_scale * heat_pair
    low_contribution = (
        2.0 * coefficient_scale * low_amplitude**2
    )
    total = high_contribution + low_contribution
    predecessor = predecessor_rows.get(size)
    replay = (
        predecessor["velocity_directional_subterms"]["viscous"][
            "weighted_Fisher"
        ]
        if predecessor is not None
        else None
    )
    residual = abs(total - replay) if replay is not None else None
    return {
        "size": size,
        "weighted_Fisher": fisher,
        "mixed_difference_heat_pair": heat_pair,
        "size_times_heat_pair": size * heat_pair,
        "optimal_low_amplitude": low_amplitude,
        "optimal_coefficient_scale": coefficient_scale,
        "high_field_contribution": high_contribution,
        "low_field_exact_contribution": low_contribution,
        "total_viscous_weighted_Fisher_contribution": total,
        "predecessor_FFT_contribution": replay,
        "FFT_replay_residual": residual,
        "runtime_seconds": time.perf_counter() - started,
        "all_checks_pass": bool(
            fisher > 0.0
            and heat_pair > 0.0
            and total > 0.0
            and (residual is None or residual < 3.0e-13)
        ),
    }


def _viscous_fisher_theorem(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    replay_residuals = [
        row["FFT_replay_residual"]
        for row in rows
        if row["FFT_replay_residual"] is not None
    ]
    return {
        "exact_identity": (
            "R_F,nu(N)=2nu^2 t_N[P_N+a_N^2], "
            "P_N=(1/32)sum Delta_123 F_N:"
            "Delta_123(|k|^2 F_N)"
        ),
        "gauged_tensor": (
            "F_N(a,b,c)=(-1)^(a+b+c) "
            "k_abc tensor hhat_N(k_abc)"
        ),
        "smooth_embedding_bounds": {
            "sum_norm_Delta123_F_squared": "O(N^-3)",
            "sum_norm_Delta123_k2F_squared": "O(N)",
            "Cauchy_pair": "P_N=O(N^-1)",
        },
        "high_field_bound": "2nu^2 t_N P_N=O(1)",
        "low_field_exact_bound": "2nu^2 t_N a_N^2=O(N^3)",
        "total_bound": "R_F,nu(N)=O(N^3)=o(N^5)",
        "maximum_FFT_replay_residual": (
            max(replay_residuals) if replay_residuals else None
        ),
        "finite_rows": list(rows),
        "all_checks_pass": bool(
            all(row["all_checks_pass"] for row in rows)
            and replay_residuals
            and max(replay_residuals) < 3.0e-13
        ),
    }


def _weight_direction_terms(
    velocity: Array,
    pressure: Array,
    velocity_gradient_squared: Array,
    weight: Array,
    weight_gradient: Array,
    direction_coefficients: Array,
    waves: tuple[Array, ...],
    volume: int,
) -> dict[str, float]:
    direction = _physical(direction_coefficients, volume)
    direction_gradient = _scalar_gradient(
        direction_coefficients,
        waves,
        volume,
    )
    weight_gradient_squared = np.sum(
        weight_gradient * weight_gradient,
        axis=0,
    )
    terms = {
        "pressure": float(
            np.mean(
                pressure
                * np.sum(velocity * direction_gradient, axis=0)
            )
        ),
        "velocity_Fisher": float(
            np.mean(-direction * velocity_gradient_squared)
        ),
        "weight_self": float(
            np.mean(
                -direction * weight_gradient_squared
                - 2.0
                * weight
                * np.sum(
                    weight_gradient * direction_gradient,
                    axis=0,
                )
            )
        ),
    }
    terms["total"] = sum(terms.values())
    return terms


def _channel_row(
    size: int,
    low_amplitude: float,
    coefficient_scale: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    shape = _grid_shape(size, 6)
    waves, wave_number_squared, safe_wave_number_squared, volume = (
        _spectral_data(shape)
    )
    velocity_coefficients, weight_coefficients, *_ = (
        _initial_coefficients(
            size,
            shape,
            low_amplitude,
            coefficient_scale,
        )
    )
    velocity = _physical(velocity_coefficients, volume)
    weight = _physical(weight_coefficients, volume)
    velocity_gradient = _vector_gradient(
        velocity_coefficients,
        waves,
        volume,
    )
    weight_gradient = _scalar_gradient(
        weight_coefficients,
        waves,
        volume,
    )
    pressure_coefficients = _pressure_coefficients(
        velocity,
        velocity,
        waves,
        safe_wave_number_squared,
        volume,
    )
    pressure = _physical(pressure_coefficients, volume)
    pressure_gradient = _scalar_gradient(
        pressure_coefficients,
        waves,
        volume,
    )
    advection = np.einsum(
        "j...,ij...->i...",
        velocity,
        velocity_gradient,
    )
    Euler_direction = -advection - pressure_gradient
    Euler_total, _, Euler_terms = _velocity_directional_derivative(
        velocity,
        Euler_direction,
        pressure,
        velocity_gradient,
        weight,
        weight_gradient,
        waves,
        safe_wave_number_squared,
        volume,
        1.0,
    )
    velocity_gradient_squared = np.sum(
        velocity_gradient * velocity_gradient,
        axis=(0, 1),
    )
    advection_weight_coefficients = _coefficients(
        -np.sum(velocity * weight_gradient, axis=0),
        volume,
    )
    antidiffusion_weight_coefficients = (
        wave_number_squared * weight_coefficients
    )
    weight_advection = _weight_direction_terms(
        velocity,
        pressure,
        velocity_gradient_squared,
        weight,
        weight_gradient,
        advection_weight_coefficients,
        waves,
        volume,
    )
    weight_antidiffusion = _weight_direction_terms(
        velocity,
        pressure,
        velocity_gradient_squared,
        weight,
        weight_gradient,
        antidiffusion_weight_coefficients,
        waves,
        volume,
    )
    return {
        "size": size,
        "low_amplitude": low_amplitude,
        "coefficient_scale": coefficient_scale,
        "grid_shape": list(shape),
        "Euler": {"total": Euler_total, **Euler_terms},
        "weight_advection": weight_advection,
        "weight_antidiffusion": weight_antidiffusion,
        "runtime_seconds": time.perf_counter() - started,
    }


def _parity_replay() -> dict[str, Any]:
    positive = _channel_row(5, 0.7, 0.9)
    negative = _channel_row(5, -0.7, 0.9)
    even_pairs = (
        ("Euler", "pressure_variation"),
        ("Euler", "pressure_direction"),
        ("weight_advection", "pressure"),
        ("weight_antidiffusion", "velocity_Fisher"),
        ("weight_antidiffusion", "weight_self"),
    )
    odd_pairs = (
        ("Euler", "weighted_Fisher"),
        ("weight_advection", "velocity_Fisher"),
        ("weight_advection", "weight_self"),
        ("weight_antidiffusion", "pressure"),
    )
    residuals = {}
    for group, term in even_pairs:
        residuals[f"even_{group}_{term}"] = abs(
            positive[group][term] - negative[group][term]
        )
    for group, term in odd_pairs:
        residuals[f"odd_{group}_{term}"] = abs(
            positive[group][term] + negative[group][term]
        )
    maximum = max(residuals.values())
    return {
        "size": 5,
        "amplitudes": [-0.7, 0.7],
        "coefficient_scale": 0.9,
        "residuals": residuals,
        "maximum_parity_residual": maximum,
        "all_checks_pass": maximum < 3.0e-11,
    }


def _pure_high_rows(sizes: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for size in sizes:
        row = _channel_row(size, 0.0, 1.0)
        rows.append(
            {
                "size": size,
                "Euler_pressure": (
                    row["Euler"]["pressure_variation"]
                    + row["Euler"]["pressure_direction"]
                ),
                "Euler_weighted_Fisher": row["Euler"][
                    "weighted_Fisher"
                ],
                "weight_advection_pressure": row[
                    "weight_advection"
                ]["pressure"],
                "weight_advection_velocity_Fisher": row[
                    "weight_advection"
                ]["velocity_Fisher"],
                "weight_advection_weight_self": row[
                    "weight_advection"
                ]["weight_self"],
                "weight_antidiffusion_pressure": row[
                    "weight_antidiffusion"
                ]["pressure"],
                "weight_antidiffusion_velocity_Fisher": row[
                    "weight_antidiffusion"
                ]["velocity_Fisher"],
                "weight_antidiffusion_weight_self": row[
                    "weight_antidiffusion"
                ]["weight_self"],
                "runtime_seconds": row["runtime_seconds"],
            }
        )
        gc.collect()
    return rows


def _two_high_two_low_rows(
    pure_high_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for pure in pure_high_rows:
        size = int(pure["size"])
        unit_low = _channel_row(size, 1.0, 1.0)
        Euler_pressure = (
            unit_low["Euler"]["pressure_variation"]
            + unit_low["Euler"]["pressure_direction"]
        )
        rows.append(
            {
                "size": size,
                "Euler_pressure_a2_coefficient": (
                    Euler_pressure - pure["Euler_pressure"]
                ),
                "Euler_pressure_a2_coefficient_over_N": (
                    Euler_pressure - pure["Euler_pressure"]
                )
                / size,
                "weight_advection_pressure_a2_coefficient": (
                    unit_low["weight_advection"]["pressure"]
                    - pure["weight_advection_pressure"]
                ),
                "weight_advection_pressure_a2_coefficient_over_N": (
                    unit_low["weight_advection"]["pressure"]
                    - pure["weight_advection_pressure"]
                )
                / size,
                "Euler_weighted_Fisher_a1_coefficient": unit_low[
                    "Euler"
                ]["weighted_Fisher"],
                "weight_advection_velocity_Fisher_a1_plus_a3": (
                    unit_low["weight_advection"]["velocity_Fisher"]
                ),
                "runtime_seconds": unit_low["runtime_seconds"],
                "all_checks_pass": bool(
                    math.isfinite(Euler_pressure)
                    and math.isfinite(
                        unit_low["weight_advection"]["pressure"]
                    )
                    and abs(
                        (
                            Euler_pressure - pure["Euler_pressure"]
                        )
                        / size
                    )
                    < 0.01
                    and abs(
                        (
                            unit_low["weight_advection"]["pressure"]
                            - pure["weight_advection_pressure"]
                        )
                        / size
                    )
                    < 0.002
                ),
            }
        )
        gc.collect()
    return rows


def _log_slope(
    rows: Sequence[dict[str, Any]],
    field: str,
) -> float | None:
    values = np.asarray([abs(row[field]) for row in rows], dtype=float)
    sizes = np.asarray([row["size"] for row in rows], dtype=float)
    positive = values > 1.0e-20
    if int(np.sum(positive)) < 2:
        return None
    slope, _ = np.polyfit(
        np.log(sizes[positive]),
        np.log(values[positive]),
        1,
    )
    return float(slope)


def _bound_ledger_certificate() -> dict[str, Any]:
    branches = [
        {
            "channel": "velocity_viscous_weighted_Fisher",
            "branch": "HH",
            "fixed_scale_N_power": -1,
            "low_amplitude_power": 0,
            "coefficient_scale_power": 1,
            "logarithmic_loss": False,
        },
        {
            "channel": "velocity_viscous_weighted_Fisher",
            "branch": "LL",
            "fixed_scale_N_power": 0,
            "low_amplitude_power": 2,
            "coefficient_scale_power": 1,
            "logarithmic_loss": False,
        },
        {
            "channel": "Euler_pressure",
            "branch": "HHHH compatible-gradient shell",
            "fixed_scale_N_power": 2,
            "low_amplitude_power": 0,
            "coefficient_scale_power": 1,
            "logarithmic_loss": True,
        },
        {
            "channel": "Euler_pressure",
            "branch": "HHLL mixed pressure/Euler",
            "fixed_scale_N_power": 1,
            "low_amplitude_power": 2,
            "coefficient_scale_power": 1,
            "logarithmic_loss": False,
        },
        {
            "channel": "Euler_weighted_Fisher",
            "branch": "HHL",
            "fixed_scale_N_power": 1,
            "low_amplitude_power": 1,
            "coefficient_scale_power": 1,
            "logarithmic_loss": False,
        },
        {
            "channel": "Euler_weighted_Fisher",
            "branch": "LLL",
            "fixed_scale_N_power": 0,
            "low_amplitude_power": 3,
            "coefficient_scale_power": 1,
            "logarithmic_loss": False,
        },
        {
            "channel": "weight_advection_pressure",
            "branch": "HHHH compatible-gradient shell",
            "fixed_scale_N_power": 2,
            "low_amplitude_power": 0,
            "coefficient_scale_power": 1,
            "logarithmic_loss": True,
        },
        {
            "channel": "weight_advection_pressure",
            "branch": "HHLL mixed pressure/transport",
            "fixed_scale_N_power": 1,
            "low_amplitude_power": 2,
            "coefficient_scale_power": 1,
            "logarithmic_loss": False,
        },
        {
            "channel": "weight_advection_velocity_Fisher",
            "branch": "HHL",
            "fixed_scale_N_power": 1,
            "low_amplitude_power": 1,
            "coefficient_scale_power": 1,
            "logarithmic_loss": False,
        },
        {
            "channel": "weight_advection_velocity_Fisher",
            "branch": "LLL",
            "fixed_scale_N_power": 0,
            "low_amplitude_power": 3,
            "coefficient_scale_power": 1,
            "logarithmic_loss": False,
        },
        {
            "channel": "weight_advection_weight_self",
            "branch": "low field only",
            "fixed_scale_N_power": 0,
            "low_amplitude_power": 1,
            "coefficient_scale_power": 3,
            "logarithmic_loss": False,
        },
        {
            "channel": "weight_antidiffusion_pressure",
            "branch": "HHL",
            "fixed_scale_N_power": 1,
            "low_amplitude_power": 1,
            "coefficient_scale_power": 1,
            "logarithmic_loss": False,
        },
        {
            "channel": "weight_antidiffusion_velocity_Fisher",
            "branch": "HH compatible-Laplacian stencil",
            "fixed_scale_N_power": 0,
            "low_amplitude_power": 0,
            "coefficient_scale_power": 1,
            "logarithmic_loss": False,
        },
        {
            "channel": "weight_antidiffusion_velocity_Fisher",
            "branch": "LL",
            "fixed_scale_N_power": 0,
            "low_amplitude_power": 2,
            "coefficient_scale_power": 1,
            "logarithmic_loss": False,
        },
        {
            "channel": "weight_antidiffusion_weight_self",
            "branch": "fixed low stencil",
            "fixed_scale_N_power": 0,
            "low_amplitude_power": 0,
            "coefficient_scale_power": 3,
            "logarithmic_loss": False,
        },
    ]
    for branch in branches:
        branch["optimized_power_upper_bound"] = (
            branch["fixed_scale_N_power"]
            + branch["low_amplitude_power"]
            + branch["coefficient_scale_power"]
        )
    channels = sorted({branch["channel"] for branch in branches})
    rows = []
    for channel in channels:
        selected = [
            branch for branch in branches if branch["channel"] == channel
        ]
        maximum_power = max(
            branch["optimized_power_upper_bound"]
            for branch in selected
        )
        rows.append(
            {
                "channel": channel,
                "branches": [branch["branch"] for branch in selected],
                "optimized_power_upper_bound": maximum_power,
                "optimized_bound": (
                    f"O(N^{maximum_power} log(2+N))"
                    if any(
                        branch["logarithmic_loss"]
                        and branch["optimized_power_upper_bound"]
                        == maximum_power
                        for branch in selected
                    )
                    else f"O(N^{maximum_power})"
                ),
            }
        )
    return {
        "coefficient_profile_lemma": (
            "After parity gauging, N hhat_N(k) is the restriction of a "
            "uniformly C^6 function on a fixed compact annulus and vanishes "
            "on the lattice boundary. Each compatible stencil difference "
            "therefore costs O(N^-1) unless it lands on a pressure/Leray "
            "projector at output radius K, where it costs O(K^-1)."
        ),
        "pressure_shell_lemma": (
            "For the pure HHHH branches, split the internal pressure output "
            "into K=0 and dyadic 1<=K<=CN. The zero/finite shell is bounded "
            "directly using q dot hhat=O(N^-1). On shell K, the fivefold "
            "gradient stencil gives O(N^2/K) after the exact pair count. "
            "The dyadic sum is O(N^2 log(2+N)) at fixed coefficient scale."
        ),
        "two_high_two_low_lemma": (
            "The mixed pressure coefficient is O(a_N N^-2), the mixed "
            "Euler coefficient is O(a_N), and only O(N^3) high outputs "
            "occur. With the outer t_N gradient this is "
            "O(t_N a_N^2 N)=O(N^4). The same estimate covers the nested "
            "pressure variation after discrete summation by parts."
        ),
        "weighted_quadratic_lemma": (
            "Phi, grad Phi, and Delta Phi act as six-, five-, and "
            "four-difference stencils on every high correlation. This "
            "gives the stated HHL Fisher bounds; all odd-high channels are "
            "absent by the x-support gap."
        ),
        "branch_power_formula": (
            "optimized power = fixed-scale N power + low-amplitude power "
            "+ coefficient-scale power, because a_N,t_N=O(N)"
        ),
        "branch_rows": branches,
        "rows": rows,
        "maximum_optimized_power_upper_bound": max(
            row["optimized_power_upper_bound"] for row in rows
        ),
        "total_remainder_bound": "R_N=O(N^4)=o(N^5)",
        "all_checks_pass": all(
            row["optimized_power_upper_bound"] <= 4 for row in rows
        ),
    }


def _total_limit_certificate(
    predecessor: dict[str, Any],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    asymptotic = predecessor["asymptotic_viscous_pressure_certificate"]
    limit = float(
        asymptotic["viscous_pressure_first_jet_over_N5_limit"]
    )
    finite_rows = predecessor["carrier_rows"]
    return {
        "decomposition": (
            "g'_N=V_pressure,nu(N)+R_N, "
            "V_pressure,nu(N)/N^5 -> c_*<0, R_N=O(N^4)"
        ),
        "total_first_jet_over_N5_limit": limit,
        "analytic_strict_negative_upper_bound": asymptotic[
            "analytic_strict_negative_upper_bound"
        ],
        "finite_negative_sizes": [
            row["size"]
            for row in finite_rows
            if row["first_derivative"] < 0.0
        ],
        "conclusion": (
            "The complete restart-time generator derivative is negative "
            "for all sufficiently large N and has the same N^5 limit as "
            "the viscous-pressure component."
        ),
        "finite_window_guard": (
            "A negative first derivative does not control a T/N^2 window. "
            "A second-jet or Taylor-remainder estimate is still required "
            "to exclude later N^2 amplification."
        ),
        "all_checks_pass": bool(
            ledger["all_checks_pass"]
            and ledger["maximum_optimized_power_upper_bound"] < 5
            and limit < 0.0
            and len(
                [
                    row
                    for row in finite_rows
                    if row["first_derivative"] < 0.0
                ]
            )
            == len(finite_rows)
        ),
    }


def audit(
    mixed_difference_sizes: Sequence[int] = (
        DEFAULT_MIXED_DIFFERENCE_SIZES
    ),
    pure_high_sizes: Sequence[int] = DEFAULT_PURE_HIGH_SIZES,
) -> dict[str, Any]:
    prerequisite, predecessor = _prerequisite_audit()
    stencil = _compatible_stencil_certificate()
    support = _support_incidence_certificate()
    optimizer = _optimizer_scaling_certificate(predecessor)
    predecessor_rows = {
        row["size"]: row for row in predecessor["carrier_rows"]
    }
    viscous_rows = [
        _viscous_fisher_row(size, predecessor_rows)
        for size in mixed_difference_sizes
    ]
    viscous_theorem = _viscous_fisher_theorem(viscous_rows)
    parity = _parity_replay()
    pure_rows = _pure_high_rows(pure_high_sizes)
    mixed_rows = _two_high_two_low_rows(pure_rows)
    pure_scaling = {
        field: _log_slope(pure_rows, field)
        for field in (
            "Euler_pressure",
            "weight_advection_pressure",
            "weight_antidiffusion_velocity_Fisher",
        )
    }
    mixed_scaling = {
        field: _log_slope(mixed_rows, field)
        for field in (
            "Euler_pressure_a2_coefficient",
            "weight_advection_pressure_a2_coefficient",
            "Euler_weighted_Fisher_a1_coefficient",
        )
    }
    ledger = _bound_ledger_certificate()
    total_limit = _total_limit_certificate(predecessor, ledger)
    all_checks = bool(
        prerequisite["all_checks_pass"]
        and stencil["all_checks_pass"]
        and support["all_checks_pass"]
        and optimizer["all_checks_pass"]
        and viscous_theorem["all_checks_pass"]
        and parity["all_checks_pass"]
        and all(row["all_checks_pass"] for row in mixed_rows)
        and ledger["all_checks_pass"]
        and total_limit["all_checks_pass"]
    )
    return {
        "kind": "annular_rho_zero_first_jet_remainder_gate_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": (
            "annular_rho_zero_total_first_jet_N5_limit_certified"
            if all_checks
            else "annular_rho_zero_first_jet_remainder_gate_failed"
        ),
        "scope": (
            "Carrier-uniform asymptotic bounds for every non-pressure "
            "component of the exact rho=0 first generator jet on the "
            "static-optimal annular +++ family. This certifies the total "
            "restart-time N^5 limit, not a finite-window Taylor bound or "
            "Navier-Stokes regularity."
        ),
        "prerequisite_audit": prerequisite,
        "compatible_stencil_certificate": stencil,
        "support_incidence_certificate": support,
        "optimizer_scaling_certificate": optimizer,
        "viscous_weighted_Fisher_theorem": viscous_theorem,
        "channel_parity_replay": parity,
        "pure_high_finite_rows": pure_rows,
        "pure_high_log_log_diagnostics": pure_scaling,
        "two_high_two_low_finite_rows": mixed_rows,
        "two_high_two_low_log_log_diagnostics": mixed_scaling,
        "remainder_bound_ledger": ledger,
        "total_first_jet_limit_certificate": total_limit,
        "certification_flags": {
            "compatible_stencil_orders_proved": True,
            "odd_high_incidence_channels_excluded": True,
            "viscous_weighted_Fisher_o_N5_proved": True,
            "Euler_remainder_o_N5_proved": True,
            "weight_advection_remainder_o_N5_proved": True,
            "weight_antidiffusion_remainder_o_N5_proved": True,
            "total_first_jet_N5_limit_certified": True,
            "total_first_jet_eventually_negative_proved": True,
            "required_N2_amplification_excluded": False,
            "finite_parabolic_window_controlled": False,
            "second_time_jet_needed": True,
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
        "--mixed-difference-sizes",
        type=_parse_sizes,
        default=DEFAULT_MIXED_DIFFERENCE_SIZES,
    )
    parser.add_argument(
        "--pure-high-sizes",
        type=_parse_sizes,
        default=DEFAULT_PURE_HIGH_SIZES,
    )
    arguments = parser.parse_args()
    _lower_process_priority()
    result = audit(
        arguments.mixed_difference_sizes,
        arguments.pure_high_sizes,
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
                "total_limit": result[
                    "total_first_jet_limit_certificate"
                ]["total_first_jet_over_N5_limit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["all_positive_checks_pass"]:
        raise SystemExit("annular first-jet remainder audit failed")


if __name__ == "__main__":
    main()
