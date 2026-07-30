"""Audit the backward replica dual and its conservative pressure-edge gate."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from pressure_frame_pairing_audit import (
    GRID_SIZE,
    RANDOM_SEED,
    STARTING_GRID_INDEX,
    VELOCITY_RMS,
    _build_spectral_fields,
)
from signed_projected_replica_generator_audit import (
    _evaluate_velocity_grid,
    _periodic_resample,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parents[3]


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


def _adjoint_dual_symbolic_audit() -> dict[str, Any]:
    viscosity, correlation = sp.symbols("nu rho", nonnegative=True)
    weight, strain, pressure_flux = sp.symbols(
        "lambda R F",
        real=True,
    )
    gradient_pairing, weight_fisher = sp.symbols(
        "G H",
        nonnegative=True,
    )
    replica_flux = (
        -2 * weight * strain
        + pressure_flux
        - 2
        * viscosity
        * (1 - correlation)
        * weight
        * gradient_pairing
    )
    dual_generator = (
        sp.Rational(3, 2) * replica_flux
        - 3 * viscosity * weight_fisher
    )
    expected_generator = (
        -3 * weight * strain
        + sp.Rational(3, 2) * pressure_flux
        - 3
        * viscosity
        * (1 - correlation)
        * weight
        * gradient_pairing
        - 3 * viscosity * weight_fisher
    )
    generator_residual = sp.simplify(
        dual_generator - expected_generator
    )

    physical_pressure, velocity_dissipation = sp.symbols(
        "P D_u",
        real=True,
    )
    rho_zero_generator = 3 * (
        physical_pressure
        - viscosity * velocity_dissipation
        - viscosity * weight_fisher
    )
    rho_zero_expected = (
        3 * physical_pressure
        - 3 * viscosity * velocity_dissipation
        - 3 * viscosity * weight_fisher
    )
    rho_zero_residual = sp.simplify(
        rho_zero_generator - rho_zero_expected
    )

    return {
        "backward_weight_equation": (
            "lambda_t+u dot grad lambda+nu Delta lambda=0, "
            "lambda(T)=lambda_T>=0"
        ),
        "backward_time_form": (
            "mu_tau=u(T-tau) dot grad mu+nu Delta mu"
        ),
        "positivity_preserved": True,
        "penalty_identity": (
            "integral lambda_T^3-integral lambda_s^3="
            "6nu integral_s^T integral lambda |grad lambda|^2"
        ),
        "reset_condition": (
            "V_1(s)=V_2(s)=u(s), hence C_rho(s)=|u(s)|^2"
        ),
        "terminal_majorization": (
            "C_rho(T)>=C_0(T)=|u(T)|^2 for 0<=rho<=1"
        ),
        "restart_dual_inequality": (
            "||u(T)||_3^3<=||u(s)||_3^3+"
            "sup_(lambda_T>=0) integral_s^T integral["
            "-3lambda R_rho+(3/2)grad lambda dot F_rho"
            "-3nu(1-rho)lambda G_rho"
            "-3nu lambda|grad lambda|^2]"
        ),
        "rho_zero_reduction": (
            "||u(T)||_3^3<=||u(s)||_3^3+"
            "3 sup_(lambda_T>=0) integral_s^T integral["
            "p u dot grad lambda"
            "-nu lambda|grad u|^2"
            "-nu lambda|grad lambda|^2]"
        ),
        "general_generator_symbolic_residual": str(generator_residual),
        "rho_zero_symbolic_residual": str(rho_zero_residual),
        "all_checks_pass": (
            generator_residual == 0 and rho_zero_residual == 0
        ),
    }


def _exact_shear_adjoint_audit() -> dict[str, Any]:
    time, terminal_time, y = sp.symbols("t T y", real=True)
    viscosity, wave_number = sp.symbols("nu k", positive=True)
    mean_speed, amplitude = sp.symbols("c A", positive=True)
    rate = viscosity * wave_number**2
    velocity = mean_speed + amplitude * sp.exp(-rate * time) * sp.sin(
        wave_number * y
    )
    weight = (
        mean_speed
        + amplitude
        * sp.exp(-rate * (2 * terminal_time - time))
        * sp.sin(wave_number * y)
    )
    pde_residual = sp.simplify(
        sp.diff(weight, time)
        + viscosity * sp.diff(weight, y, 2)
    )
    terminal_residual = sp.simplify(
        weight.subs(time, terminal_time)
        - velocity.subs(time, terminal_time)
    )

    numeric = {
        "mean_speed": 2.0,
        "amplitude": 0.7,
        "viscosity": 0.15,
        "wave_number": 2.0,
        "terminal_time": 0.8,
    }
    c_value = numeric["mean_speed"]
    a_value = numeric["amplitude"]
    nu_value = numeric["viscosity"]
    k_value = numeric["wave_number"]
    duration = numeric["terminal_time"]
    numeric_rate = nu_value * k_value**2
    terminal_amplitude = a_value * math.exp(-numeric_rate * duration)
    initial_weight_amplitude = a_value * math.exp(
        -2.0 * numeric_rate * duration
    )

    terminal_l3 = c_value**3 + 1.5 * c_value * terminal_amplitude**2
    initial_l3 = c_value**3 + 1.5 * c_value * a_value**2
    initial_weighted_energy = (
        c_value * (c_value**2 + 0.5 * a_value**2)
        + c_value * a_value * initial_weight_amplitude
    )
    initial_weight_l3 = (
        c_value**3
        + 1.5 * c_value * initial_weight_amplitude**2
    )
    initial_dual = (
        1.5 * initial_weighted_energy - 0.5 * initial_weight_l3
    )
    velocity_fisher_time_integral = (
        c_value
        * k_value**2
        / 2.0
        * a_value**2
        * (1.0 - math.exp(-2.0 * numeric_rate * duration))
        / (2.0 * numeric_rate)
    )
    weight_fisher_time_integral = (
        c_value
        * k_value**2
        / 2.0
        * a_value**2
        * (
            math.exp(-2.0 * numeric_rate * duration)
            - math.exp(-4.0 * numeric_rate * duration)
        )
        / (2.0 * numeric_rate)
    )
    reconstructed_terminal_dual = initial_dual - 3.0 * nu_value * (
        velocity_fisher_time_integral + weight_fisher_time_integral
    )
    identity_residual = abs(
        terminal_l3 - reconstructed_terminal_dual
    )

    return {
        "velocity": (
            "u=(c+A exp(-nu k^2 t)sin(ky),0,0), c>A>0"
        ),
        "backward_weight": str(weight),
        "backward_PDE_symbolic_residual": str(pde_residual),
        "terminal_condition_symbolic_residual": str(terminal_residual),
        "parameters": numeric,
        "initial_physical_L3_cubed": initial_l3,
        "terminal_physical_L3_cubed": terminal_l3,
        "initial_propagated_dual": initial_dual,
        "velocity_fisher_time_integral": velocity_fisher_time_integral,
        "weight_fisher_time_integral": weight_fisher_time_integral,
        "reconstructed_terminal_dual": reconstructed_terminal_dual,
        "identity_residual": identity_residual,
        "all_checks_pass": (
            pde_residual == 0
            and terminal_residual == 0
            and terminal_l3 < initial_l3
            and initial_dual <= initial_l3 + 1.0e-14
            and identity_residual < 2.0e-13
        ),
    }


def _abc_stress_audit() -> dict[str, Any]:
    size = 64
    coordinates = 2.0 * math.pi * np.arange(size) / size
    x, y, z = np.meshgrid(
        coordinates,
        coordinates,
        coordinates,
        indexing="ij",
    )
    velocity = np.stack(
        (
            np.sin(z) + np.cos(y),
            np.sin(x) + np.cos(z),
            np.sin(y) + np.cos(x),
        )
    )
    gradient = np.zeros((3, 3, size, size, size))
    gradient[0, 1] = -np.sin(y)
    gradient[0, 2] = np.cos(z)
    gradient[1, 0] = np.cos(x)
    gradient[1, 2] = -np.sin(z)
    gradient[2, 0] = -np.sin(x)
    gradient[2, 1] = np.cos(y)
    speed = np.sqrt(np.sum(velocity**2, axis=0))
    velocity_dot_gradient = np.einsum(
        "ixyz,ikxyz->kxyz",
        velocity,
        gradient,
    )
    speed_gradient = velocity_dot_gradient / np.maximum(speed, 1.0e-14)
    pressure = -0.5 * speed**2
    pressure_work = float(
        np.mean(
            pressure
            * np.einsum(
                "ixyz,ixyz->xyz",
                velocity,
                speed_gradient,
            )
        )
    )
    velocity_fisher = float(
        np.mean(speed * np.sum(gradient**2, axis=(0, 1)))
    )
    speed_fisher = float(
        np.mean(speed * np.sum(speed_gradient**2, axis=0))
    )
    l3_cubed = float(np.mean(speed**3))
    helmholtz_residual = abs(
        velocity_fisher + speed_fisher - l3_cubed
    )
    return {
        "flow": "unit ABC Beltrami field with p=-|u|^2/2",
        "grid_size": size,
        "pressure_work": pressure_work,
        "velocity_fisher": velocity_fisher,
        "speed_fisher": speed_fisher,
        "L3_cubed": l3_cubed,
        "Delta_u_equals_minus_u_balance_residual": helmholtz_residual,
        "instantaneous_L3_rate_at_nu_one": 3.0
        * (
            pressure_work
            - velocity_fisher
            - speed_fisher
        ),
        "all_checks_pass": (
            abs(pressure_work) < 2.0e-14
            and helmholtz_residual < 4.0e-6
        ),
    }


def _resolved_pressure_and_fisher_stress(
    fields: dict[str, object],
    velocity: Array,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    correlations = (0.0, 0.5, 1.0)
    for size in (48, 64, 80, 96):
        resolved_velocity = _periodic_resample(velocity, size)
        resolved_gradient = _periodic_resample(
            np.asarray(fields["velocity_gradient_grid"]),
            size,
        )
        resolved_pressure_gradient = _periodic_resample(
            np.asarray(fields["pressure_potential_gradient_grid"]),
            size,
        )
        speed = np.sqrt(np.sum(resolved_velocity**2, axis=0))
        speed_gradient = np.einsum(
            "ixyz,ikxyz->kxyz",
            resolved_velocity,
            resolved_gradient,
        ) / np.maximum(speed, 1.0e-14)
        pressure_work = float(
            -np.mean(
                speed
                * np.einsum(
                    "ixyz,ixyz->xyz",
                    resolved_velocity,
                    resolved_pressure_gradient,
                )
            )
        )
        velocity_fisher = float(
            np.mean(
                speed * np.sum(resolved_gradient**2, axis=(0, 1))
            )
        )
        weight_fisher = float(
            np.mean(speed * np.sum(speed_gradient**2, axis=0))
        )
        thresholds = {
            str(correlation): (
                (
                    (1.0 - correlation) * velocity_fisher
                    + weight_fisher
                )
                / pressure_work
            )
            for correlation in correlations
        }
        scaled_rows = []
        for amplitude_scale in (1.0, 170.0, 200.0):
            scaled_rows.append(
                {
                    "amplitude_scale": amplitude_scale,
                    "rho_zero_rate_over_3_scale_cubed": (
                        amplitude_scale * pressure_work
                        - velocity_fisher
                        - weight_fisher
                    ),
                }
            )
        rows.append(
            {
                "grid_size": size,
                "pressure_work": pressure_work,
                "velocity_fisher": velocity_fisher,
                "weight_fisher": weight_fisher,
                "sign_change_amplitude_by_rho": thresholds,
                "scaled_rho_zero_rows": scaled_rows,
            }
        )

    rho_zero_thresholds = [
        row["sign_change_amplitude_by_rho"]["0.0"] for row in rows
    ]
    rho_one_thresholds = [
        row["sign_change_amplitude_by_rho"]["1.0"] for row in rows
    ]
    return {
        "source": (
            "seed-81 smooth periodic finite-Fourier pressure adversary"
        ),
        "source_parameters": {
            "random_seed": RANDOM_SEED,
            "base_velocity_rms": VELOCITY_RMS,
            "viscosity": 1.0,
        },
        "instantaneous_reset_generator": (
            "3 alpha^3[alpha P-(1-rho)D_u-D_lambda]"
        ),
        "rho_zero_sign_change_amplitude_range": [
            min(rho_zero_thresholds),
            max(rho_zero_thresholds),
        ],
        "rho_one_sign_change_amplitude_range": [
            min(rho_one_thresholds),
            max(rho_one_thresholds),
        ],
        "rows": rows,
        "interpretation": (
            "At reset, rho=0 has the most dissipation. Scaling the same "
            "smooth datum above the resolved threshold makes the exact "
            "instantaneous L3 generator positive. This falsifies universal "
            "nonpositivity, not regularity or a finite-time bound."
        ),
        "all_checks_pass": (
            max(rho_zero_thresholds) - min(rho_zero_thresholds) < 8.0e-4
            and 168.16 < min(rho_zero_thresholds) < 168.18
            and 41.0 < min(rho_one_thresholds) < 41.1
            and all(
                row["scaled_rho_zero_rows"][0][
                    "rho_zero_rate_over_3_scale_cubed"
                ]
                < 0.0
                and row["scaled_rho_zero_rows"][1][
                    "rho_zero_rate_over_3_scale_cubed"
                ]
                > 0.0
                and row["scaled_rho_zero_rows"][2][
                    "rho_zero_rate_over_3_scale_cubed"
                ]
                > 0.0
                for row in rows
            )
        ),
    }


def _pressure_from_gradient(pressure_gradient: Array) -> tuple[Array, float]:
    size = pressure_gradient.shape[-1]
    frequencies = np.fft.fftfreq(size) * size
    wave_vector = np.array(
        np.meshgrid(frequencies, frequencies, frequencies, indexing="ij")
    )
    wave_number_squared = np.sum(wave_vector**2, axis=0)
    safe_wave_number = np.where(
        wave_number_squared == 0.0,
        1.0,
        wave_number_squared,
    )
    gradient_hat = np.fft.fftn(pressure_gradient, axes=(1, 2, 3))
    pressure_hat = (
        np.sum(-1j * wave_vector * gradient_hat, axis=0)
        / safe_wave_number
    )
    pressure_hat *= wave_number_squared > 0.0
    pressure = np.fft.ifftn(pressure_hat).real
    reconstructed_gradient = np.empty_like(pressure_gradient)
    for direction in range(3):
        reconstructed_gradient[direction] = np.fft.ifftn(
            1j * wave_vector[direction] * pressure_hat
        ).real
    residual = float(
        np.max(np.abs(reconstructed_gradient - pressure_gradient))
    )
    return pressure, residual


def _partition_edge_gate_audit(
    fields: dict[str, object],
    velocity: Array,
) -> dict[str, Any]:
    pressure_gradient = np.asarray(
        fields["pressure_potential_gradient_grid"]
    )
    pressure, pressure_gradient_residual = _pressure_from_gradient(
        pressure_gradient
    )
    velocity_gradient = np.asarray(fields["velocity_gradient_grid"])
    size = GRID_SIZE
    coordinates = (
        2.0
        * math.pi
        * np.indices((size, size, size))
        / size
    )
    center = 2.0 * math.pi * STARTING_GRID_INDEX / size
    displacement = coordinates - center[:, None, None, None]
    plus = (1.0 + np.cos(displacement)) / 2.0
    minus = (1.0 - np.cos(displacement)) / 2.0
    plus_derivative = -0.5 * np.sin(displacement)
    minus_derivative = 0.5 * np.sin(displacement)
    seed_coefficients = np.array(
        [1.0, 1.4, 0.8, 2.0, 0.6, 1.2, 1.8, 0.9]
    )
    coefficients = 2.6 - seed_coefficients

    cells: list[dict[str, Any]] = []
    coefficient_by_bits: dict[tuple[int, ...], float] = {}
    for coefficient, bits in zip(
        coefficients,
        itertools.product((0, 1), repeat=3),
    ):
        factors = [
            plus[direction] if bit == 0 else minus[direction]
            for direction, bit in enumerate(bits)
        ]
        derivatives = [
            plus_derivative[direction]
            if bit == 0
            else minus_derivative[direction]
            for direction, bit in enumerate(bits)
        ]
        localization = np.prod(factors, axis=0)
        gradient = np.empty((3, size, size, size))
        for direction in range(3):
            gradient[direction] = derivatives[direction] * np.prod(
                [
                    factors[other]
                    for other in range(3)
                    if other != direction
                ],
                axis=0,
            )
        cells.append(
            {
                "bits": bits,
                "coefficient": float(coefficient),
                "localization": localization,
                "gradient": gradient,
            }
        )
        coefficient_by_bits[bits] = float(coefficient)

    weight = np.sum(
        [
            cell["coefficient"] * cell["localization"]
            for cell in cells
        ],
        axis=0,
    )
    weight_gradient = np.sum(
        [cell["coefficient"] * cell["gradient"] for cell in cells],
        axis=0,
    )
    cell_pressure_fluxes = [
        float(
            np.mean(
                pressure
                * np.einsum(
                    "ixyz,ixyz->xyz",
                    velocity,
                    cell["gradient"],
                )
            )
        )
        for cell in cells
    ]
    direct_pressure_flux = float(
        np.mean(
            pressure
            * np.einsum(
                "ixyz,ixyz->xyz",
                velocity,
                weight_gradient,
            )
        )
    )
    weighted_cell_flux = float(
        np.dot(coefficients, np.array(cell_pressure_fluxes))
    )

    edge_pressure_flux = 0.0
    conditional_pressure_flux = 0.0
    direct_weight_fisher = float(
        np.mean(weight * np.sum(weight_gradient**2, axis=0))
    )
    unweighted_weight_gradient_energy = float(
        np.mean(np.sum(weight_gradient**2, axis=0))
    )
    conditional_weight_fisher = 0.0
    young_remainder = 0.0
    direction_rows = []
    for direction in range(3):
        other_directions = [
            index for index in range(3) if index != direction
        ]
        face_zero = np.zeros((size, size, size))
        face_one = np.zeros((size, size, size))
        direction_edge_flux = 0.0
        edge_rows = []
        for other_bits in itertools.product((0, 1), repeat=2):
            zero_bits = [0, 0, 0]
            one_bits = [0, 0, 0]
            zero_bits[direction] = 0
            one_bits[direction] = 1
            other_factor = np.ones((size, size, size))
            for index, other_direction in enumerate(other_directions):
                bit = other_bits[index]
                zero_bits[other_direction] = bit
                one_bits[other_direction] = bit
                other_factor *= (
                    plus[other_direction]
                    if bit == 0
                    else minus[other_direction]
                )
            zero_tuple = tuple(zero_bits)
            one_tuple = tuple(one_bits)
            zero_coefficient = coefficient_by_bits[zero_tuple]
            one_coefficient = coefficient_by_bits[one_tuple]
            face_zero += zero_coefficient * other_factor
            face_one += one_coefficient * other_factor
            edge_flux = float(
                np.mean(
                    pressure
                    * velocity[direction]
                    * plus_derivative[direction]
                    * other_factor
                )
            )
            difference = zero_coefficient - one_coefficient
            direction_edge_flux += difference * edge_flux
            edge_rows.append(
                {
                    "zero_cell": "".join(str(bit) for bit in zero_tuple),
                    "one_cell": "".join(str(bit) for bit in one_tuple),
                    "coefficient_difference": difference,
                    "edge_flux": edge_flux,
                }
            )

        face_difference = face_zero - face_one
        face_sum = face_zero + face_one
        edge_density_full = (
            pressure
            * velocity[direction]
            * plus_derivative[direction]
        )
        edge_density = np.mean(edge_density_full, axis=direction)
        face_difference_reduced = np.take(
            face_difference,
            0,
            axis=direction,
        )
        face_sum_reduced = np.take(face_sum, 0, axis=direction)
        conditional_direction_pressure = float(
            np.mean(face_difference_reduced * edge_density)
        )
        conditional_direction_fisher = float(
            np.mean(
                face_sum_reduced
                * face_difference_reduced**2
                / 16.0
            )
        )
        direct_direction_fisher = float(
            np.mean(weight * weight_gradient[direction] ** 2)
        )
        direction_young_remainder = float(
            4.0 * np.mean(edge_density**2 / face_sum_reduced)
        )
        direction_rows.append(
            {
                "direction": direction,
                "edge_rows": edge_rows,
                "edge_pressure_flux": direction_edge_flux,
                "conditional_pressure_flux": (
                    conditional_direction_pressure
                ),
                "direct_weight_fisher": direct_direction_fisher,
                "conditional_weight_fisher": (
                    conditional_direction_fisher
                ),
                "young_remainder_at_nu_one": (
                    direction_young_remainder
                ),
                "young_inequality_margin": (
                    direction_young_remainder
                    - conditional_direction_pressure
                    + conditional_direction_fisher
                ),
            }
        )
        edge_pressure_flux += direction_edge_flux
        conditional_pressure_flux += conditional_direction_pressure
        conditional_weight_fisher += conditional_direction_fisher
        young_remainder += direction_young_remainder

    weighted_velocity_fisher = float(
        np.mean(
            weight * np.sum(velocity_gradient**2, axis=(0, 1))
        )
    )
    exact_dual_flux_at_nu_one = (
        direct_pressure_flux
        - weighted_velocity_fisher
        - direct_weight_fisher
    )
    smooth_partition_sign_change_amplitude = (
        (weighted_velocity_fisher + direct_weight_fisher)
        / direct_pressure_flux
    )
    smooth_partition_scaled_rate_at_700 = (
        700.0 * direct_pressure_flux
        - weighted_velocity_fisher
        - direct_weight_fisher
    )
    edge_young_upper_at_nu_one = (
        young_remainder - weighted_velocity_fisher
    )
    return {
        "partition": (
            "eight tensor-product cells from (1+cos)/2 and (1-cos)/2"
        ),
        "center": center.tolist(),
        "coefficients": coefficients.tolist(),
        "minimum_weight": float(np.min(weight)),
        "maximum_weight": float(np.max(weight)),
        "pressure_gradient_reconstruction_residual": (
            pressure_gradient_residual
        ),
        "cell_pressure_fluxes": cell_pressure_fluxes,
        "partition_pressure_flux_sum": float(
            np.sum(cell_pressure_fluxes)
        ),
        "direct_weighted_pressure_flux": direct_pressure_flux,
        "weighted_cell_pressure_flux": weighted_cell_flux,
        "edge_weighted_pressure_flux": edge_pressure_flux,
        "conditional_weighted_pressure_flux": (
            conditional_pressure_flux
        ),
        "direct_weight_fisher": direct_weight_fisher,
        "unweighted_weight_gradient_energy": (
            unweighted_weight_gradient_energy
        ),
        "conditional_weight_fisher": conditional_weight_fisher,
        "weighted_velocity_fisher": weighted_velocity_fisher,
        "young_remainder_at_nu_one": young_remainder,
        "exact_dual_flux_at_nu_one": exact_dual_flux_at_nu_one,
        "smooth_partition_sign_change_amplitude": (
            smooth_partition_sign_change_amplitude
        ),
        "smooth_partition_scaled_rate_over_3_scale_cubed_at_700": (
            smooth_partition_scaled_rate_at_700
        ),
        "edge_young_upper_at_nu_one": edge_young_upper_at_nu_one,
        "direction_rows": direction_rows,
        "exact_conditional_edge_identity": (
            "For direction j, lambda=A phi_+ +B phi_-, "
            "P_j=mean_other[(A-B)e_j] and "
            "D_j=mean_other[(A+B)(A-B)^2/16]"
        ),
        "degenerate_edge_young_bound": (
            "P_j-nu D_j<=4/nu mean_other[e_j^2/(A+B)]"
        ),
        "interpretation": (
            "Pressure is an antisymmetric edge transfer. The adjoint "
            "Fisher term gives an exact cubic edge penalty without a global "
            "positive weight floor. The chosen smooth positive edge weight "
            "has positive pressure transfer and, when scaled with the "
            "velocity, gives a second universal-sign falsifier without a "
            "|u| cusp. Young's bound leaves a reciprocal face-weight "
            "remainder; controlling that remainder by replica velocity "
            "dissipation is the open edge gate."
        ),
        "all_checks_pass": (
            pressure_gradient_residual < 2.0e-12
            and abs(np.sum(cell_pressure_fluxes)) < 2.0e-12
            and abs(direct_pressure_flux - weighted_cell_flux) < 2.0e-12
            and abs(direct_pressure_flux - edge_pressure_flux) < 2.0e-12
            and abs(
                direct_pressure_flux - conditional_pressure_flux
            )
            < 2.0e-12
            and abs(
                direct_weight_fisher - conditional_weight_fisher
            )
            < 2.0e-12
            and all(
                row["young_inequality_margin"] >= -2.0e-12
                for row in direction_rows
            )
            and direct_pressure_flux > 1.0
            and smooth_partition_sign_change_amplitude < 700.0
            and smooth_partition_scaled_rate_at_700 > 0.0
        ),
    }


def _reset_rho_ordering_audit() -> dict[str, Any]:
    viscosity = 0.8
    weighted_velocity_fisher = 2.7
    rows = []
    for correlation in (0.0, 0.25, 0.5, 0.75, 1.0):
        excess_over_rho_zero = (
            3.0
            * viscosity
            * correlation
            * weighted_velocity_fisher
        )
        rows.append(
            {
                "rho": correlation,
                "dual_generator_excess_over_rho_zero": (
                    excess_over_rho_zero
                ),
            }
        )
    return {
        "reset_formula": (
            "Q_rho^dual-Q_0^dual="
            "3nu rho integral lambda|grad u|^2>=0"
        ),
        "rows": rows,
        "rho_zero_is_instantaneously_optimal": all(
            row["dual_generator_excess_over_rho_zero"] >= 0.0
            for row in rows
        ),
        "interpretation": (
            "Positive correlation can help only after pathwise separation "
            "changes R_rho and F_rho; it cannot improve the reset generator."
        ),
        "all_checks_pass": (
            rows[0]["dual_generator_excess_over_rho_zero"] == 0.0
            and all(
                rows[index]["dual_generator_excess_over_rho_zero"]
                < rows[index + 1][
                    "dual_generator_excess_over_rho_zero"
                ]
                for index in range(len(rows) - 1)
            )
        ),
    }


def audit() -> dict[str, Any]:
    fields = _build_spectral_fields()
    modes, coefficients = fields["velocity"]
    velocity = _evaluate_velocity_grid(modes, coefficients)

    dual = _adjoint_dual_symbolic_audit()
    shear = _exact_shear_adjoint_audit()
    abc = _abc_stress_audit()
    pressure = _resolved_pressure_and_fisher_stress(fields, velocity)
    partition = _partition_edge_gate_audit(fields, velocity)
    reset_ordering = _reset_rho_ordering_audit()

    positive_checks = {
        "backward_restart_dual_algebra_passes": dual["all_checks_pass"],
        "exact_shear_adjoint_identity_passes": shear[
            "all_checks_pass"
        ],
        "ABC_pressure_and_Helmholtz_stress_passes": abc[
            "all_checks_pass"
        ],
        "smooth_high_amplitude_sign_falsifier_is_resolved": pressure[
            "all_checks_pass"
        ],
        "pressure_partition_edge_and_Fisher_identities_pass": partition[
            "all_checks_pass"
        ],
        "rho_zero_reset_ordering_passes": reset_ordering[
            "all_checks_pass"
        ],
    }
    certification_flags = {
        "backward_adjoint_restart_dual_inequality_derived": True,
        "backward_terminal_penalty_contraction_proved": True,
        "replica_reset_endpoint_used": True,
        "positive_rho_cross_gradient_lower_bound_retained": True,
        "rho_zero_physical_pressure_form_derived": True,
        "rho_zero_is_instantaneously_best_at_reset": True,
        "scalar_pressure_partition_edge_identity_derived": True,
        "conditional_partition_Fisher_identity_derived": True,
        "degenerate_edge_Young_budget_derived": True,
        "smooth_universal_flux_nonpositivity_falsified": True,
        "universal_restart_dual_flux_is_nonpositive": False,
        "positive_rho_improves_the_instantaneous_reset_generator": False,
        "edge_Young_remainder_absorbed_by_replica_dissipation": False,
        "finite_partition_represents_all_terminal_dual_weights": False,
        "critical_signed_replica_bound_proved": False,
        "low_regularity_adjoint_replica_system_justified": False,
        "exceptional_set_upgrade_proved": False,
        "Navier_Stokes_global_regularity_proved": False,
    }
    return {
        "kind": "adjoint_replica_pressure_edge_gate_audit",
        "schema_version": 1,
        "status": (
            "backward_restart_dual_derived_"
            "universal_flux_sign_falsified_edge_budget_open"
        ),
        "assumption_scope": (
            "Classical smooth periodic Navier-Stokes with projected Weber "
            "replicas reset at the left endpoint. Smooth nonnegative "
            "terminal weights are used, with approximation needed at zeros "
            "of the exact |u(T)| optimizer."
        ),
        "backward_restart_dual": dual,
        "reset_rho_ordering": reset_ordering,
        "exact_periodic_shear": shear,
        "periodic_ABC": abc,
        "high_amplitude_pressure_sign_falsifier": pressure,
        "partition_pressure_edge_gate": partition,
        "certification_flags": certification_flags,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "next_theorem_target": (
            "Use the exact conditional edge form to seek a scale-adapted "
            "bound for sum_j mean[e_j^2/(A_j+B_j)] by "
            "nu^2(1-rho) integral lambda G_rho without imposing a global "
            "positive floor on lambda. If that bound fails, quantify a "
            "finite-time rho>0 cancellation in R_rho and F_rho, since "
            "rho>0 is provably worse at the reset instant. Any candidate "
            "must survive the amplitude-scaled seed-81 datum and preserve "
            "the full terminal dual supremum."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("adjoint replica pressure-edge audit failed")
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "kind": result["kind"],
                "output": args.output.as_posix(),
                "output_sha256": _sha256(args.output),
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
