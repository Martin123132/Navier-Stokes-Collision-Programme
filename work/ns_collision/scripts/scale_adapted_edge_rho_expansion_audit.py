"""Audit scale-adapted pressure edges and the short-time rho correction."""

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

from adjoint_replica_pressure_edge_gate_audit import _pressure_from_gradient
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
COEFFICIENTS = np.array([1.6, 1.2, 1.8, 0.6, 2.0, 1.4, 0.8, 1.7])


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


def _spectral_data(size: int) -> tuple[Array, Array, Array]:
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
    return wave_vector, wave_number_squared, safe_wave_number


def _partition_weight(
    size: int,
    frequency: int,
) -> dict[str, Any]:
    coordinates = (
        2.0
        * math.pi
        * np.indices((size, size, size))
        / size
    )
    center = 2.0 * math.pi * STARTING_GRID_INDEX / GRID_SIZE
    displacement = coordinates - center[:, None, None, None]
    plus = (1.0 + np.cos(frequency * displacement)) / 2.0
    minus = (1.0 - np.cos(frequency * displacement)) / 2.0
    plus_derivative = (
        -0.5 * frequency * np.sin(frequency * displacement)
    )
    minus_derivative = -plus_derivative

    weight = np.zeros((size, size, size))
    gradient = np.zeros((3, size, size, size))
    coefficient_by_bits: dict[tuple[int, ...], float] = {}
    for coefficient, bits in zip(
        COEFFICIENTS,
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
        weight += coefficient * localization
        for direction in range(3):
            gradient[direction] += coefficient * derivatives[
                direction
            ] * np.prod(
                [
                    factors[other]
                    for other in range(3)
                    if other != direction
                ],
                axis=0,
            )
        coefficient_by_bits[bits] = float(coefficient)

    return {
        "weight": weight,
        "gradient": gradient,
        "plus": plus,
        "minus": minus,
        "plus_derivative": plus_derivative,
        "coefficient_by_bits": coefficient_by_bits,
        "center": center,
    }


def _frequency_edge_row(
    frequency: int,
    velocity: Array,
    velocity_gradient: Array,
    pressure: Array,
) -> dict[str, Any]:
    size = velocity.shape[-1]
    partition = _partition_weight(size, frequency)
    weight = partition["weight"]
    weight_gradient = partition["gradient"]
    plus = partition["plus"]
    minus = partition["minus"]
    plus_derivative = partition["plus_derivative"]
    coefficient_by_bits = partition["coefficient_by_bits"]

    direct_pressure = float(
        np.mean(
            pressure
            * np.einsum(
                "ixyz,ixyz->xyz",
                velocity,
                weight_gradient,
            )
        )
    )
    direct_weight_fisher = float(
        np.mean(weight * np.sum(weight_gradient**2, axis=0))
    )
    weighted_velocity_fisher = float(
        np.mean(
            weight * np.sum(velocity_gradient**2, axis=(0, 1))
        )
    )

    conditional_pressure = 0.0
    conditional_weight_fisher = 0.0
    scale_adapted_young_remainder = 0.0
    direction_rows = []
    for direction in range(3):
        other_directions = [
            index for index in range(3) if index != direction
        ]
        face_zero = np.zeros((size, size, size))
        face_one = np.zeros((size, size, size))
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
            face_zero += (
                coefficient_by_bits[tuple(zero_bits)] * other_factor
            )
            face_one += (
                coefficient_by_bits[tuple(one_bits)] * other_factor
            )

        face_difference = np.take(
            face_zero - face_one,
            0,
            axis=direction,
        )
        face_sum = np.take(
            face_zero + face_one,
            0,
            axis=direction,
        )
        edge_density = np.mean(
            pressure
            * velocity[direction]
            * plus_derivative[direction],
            axis=direction,
        )
        direction_pressure = float(
            np.mean(face_difference * edge_density)
        )
        direction_fisher = float(
            frequency**2
            * np.mean(face_sum * face_difference**2)
            / 16.0
        )
        direction_remainder = float(
            4.0
            * np.mean(edge_density**2 / face_sum)
            / frequency**2
        )
        conditional_pressure += direction_pressure
        conditional_weight_fisher += direction_fisher
        scale_adapted_young_remainder += direction_remainder
        direction_rows.append(
            {
                "direction": direction,
                "conditional_pressure": direction_pressure,
                "direct_weight_fisher": float(
                    np.mean(
                        weight * weight_gradient[direction] ** 2
                    )
                ),
                "conditional_weight_fisher": direction_fisher,
                "young_remainder": direction_remainder,
                "young_margin": (
                    direction_remainder
                    - direction_pressure
                    + direction_fisher
                ),
            }
        )

    exact_sign_threshold = None
    if direct_pressure > 1.0e-10:
        exact_sign_threshold = (
            weighted_velocity_fisher + direct_weight_fisher
        ) / direct_pressure
    edge_absorption_amplitude = None
    if scale_adapted_young_remainder > 1.0e-20:
        edge_absorption_amplitude = math.sqrt(
            weighted_velocity_fisher
            / scale_adapted_young_remainder
        )
    return {
        "partition_frequency": frequency,
        "minimum_weight": float(np.min(weight)),
        "maximum_weight": float(np.max(weight)),
        "direct_pressure": direct_pressure,
        "conditional_pressure": conditional_pressure,
        "direct_weight_fisher": direct_weight_fisher,
        "conditional_weight_fisher": conditional_weight_fisher,
        "weighted_velocity_fisher": weighted_velocity_fisher,
        "scale_adapted_young_remainder": (
            scale_adapted_young_remainder
        ),
        "exact_sign_amplitude_threshold": exact_sign_threshold,
        "edge_absorption_amplitude_threshold": (
            edge_absorption_amplitude
        ),
        "direction_rows": direction_rows,
        "representation_residual": max(
            abs(direct_pressure - conditional_pressure),
            abs(direct_weight_fisher - conditional_weight_fisher),
        ),
        "all_checks_pass": (
            abs(direct_pressure - conditional_pressure) < 2.0e-10
            and abs(
                direct_weight_fisher - conditional_weight_fisher
            )
            < 2.0e-10
            and all(
                row["young_margin"] >= -2.0e-10
                for row in direction_rows
            )
        ),
    }


def _scale_homogeneity_audit() -> dict[str, Any]:
    amplitude, frequency, viscosity = sp.symbols(
        "a m nu",
        positive=True,
    )
    pressure_scale = amplitude**4 * frequency
    fisher_scale = viscosity * amplitude**3 * frequency**2
    exact_ratio = sp.simplify(pressure_scale / fisher_scale)
    edge_remainder_scale = amplitude**5 / viscosity
    edge_ratio = sp.simplify(
        edge_remainder_scale / fisher_scale
    )
    local_reynolds = amplitude / (viscosity * frequency)
    exact_residual = sp.simplify(exact_ratio - local_reynolds)
    edge_residual = sp.simplify(
        edge_ratio - local_reynolds**2
    )
    return {
        "coscaled_fields": (
            "u_(a,m)=a u(mx), p_(a,m)=a^2 p(mx), "
            "lambda_(a,m)=a lambda(mx)"
        ),
        "pressure_flux_scale": "a^4 m",
        "velocity_and_weight_Fisher_scale": "nu a^3 m^2",
        "exact_pressure_to_Fisher_ratio": str(exact_ratio),
        "Young_remainder_scale": "a^5/nu",
        "Young_remainder_to_Fisher_ratio": str(edge_ratio),
        "local_Reynolds_number": "Re_cell=a/(nu m)",
        "exact_ratio_symbolic_residual": str(exact_residual),
        "edge_ratio_symbolic_residual": str(edge_residual),
        "fixed_scale_universal_absorption_possible": False,
        "necessary_adaptation": (
            "partition frequency m must grow at least proportionally "
            "to a/nu before a scale-independent edge estimate is possible"
        ),
        "all_checks_pass": (
            exact_residual == 0 and edge_residual == 0
        ),
    }


def _spectral_gradient(
    values: Array,
    wave_vector: Array,
) -> Array:
    transform = np.fft.fftn(values, axes=(-3, -2, -1))
    tensor_shape = values.shape[:-3]
    result = np.empty(
        tensor_shape + (3,) + values.shape[-3:],
        dtype=float,
    )
    for direction in range(3):
        result[..., direction, :, :, :] = np.fft.ifftn(
            1j * wave_vector[direction] * transform,
            axes=(-3, -2, -1),
        ).real
    return result


def _spectral_laplacian(
    values: Array,
    wave_number_squared: Array,
) -> Array:
    transform = np.fft.fftn(values, axes=(-3, -2, -1))
    return np.fft.ifftn(
        -wave_number_squared * transform,
        axes=(-3, -2, -1),
    ).real


def _replica_pressure(
    velocity: Array,
    velocity_gradient: Array,
    replica: Array,
    replica_gradient: Array,
    wave_vector: Array,
    safe_wave_number: Array,
    nonzero_wave_number: Array,
) -> Array:
    transport = np.einsum(
        "jxyz,ijxyz->ixyz",
        velocity,
        replica_gradient,
    )
    stretch = np.einsum(
        "jixyz,jxyz->ixyz",
        velocity_gradient,
        replica,
    )
    vector = transport + stretch
    vector_hat = np.fft.fftn(vector, axes=(1, 2, 3))
    divergence_hat = 1j * np.sum(wave_vector * vector_hat, axis=0)
    pressure_hat = (
        divergence_hat / safe_wave_number * nonzero_wave_number
    )
    return np.fft.ifftn(pressure_hat).real


def _short_time_rho_row(
    size: int,
    base_velocity: Array,
    base_velocity_gradient: Array,
    base_pressure_gradient: Array,
) -> dict[str, Any]:
    viscosity = 1.0
    velocity = _periodic_resample(base_velocity, size)
    velocity_gradient = _periodic_resample(
        base_velocity_gradient,
        size,
    )
    pressure_gradient = _periodic_resample(
        base_pressure_gradient,
        size,
    )
    pressure, pressure_residual = _pressure_from_gradient(
        pressure_gradient
    )
    partition = _partition_weight(size, 1)
    weight = partition["weight"]
    weight_gradient = partition["gradient"]
    wave_vector, wave_number_squared, safe_wave_number = _spectral_data(
        size
    )
    nonzero_wave_number = wave_number_squared > 0.0

    hessian = _spectral_gradient(velocity_gradient, wave_vector)
    velocity_laplacian = _spectral_laplacian(
        velocity,
        wave_number_squared,
    )
    advection = np.einsum(
        "jxyz,ijxyz->ixyz",
        velocity,
        velocity_gradient,
    )
    velocity_time_derivative = (
        -advection - pressure_gradient + viscosity * velocity_laplacian
    )
    velocity_gradient_time_derivative = _spectral_gradient(
        velocity_time_derivative,
        wave_vector,
    )
    weight_laplacian = _spectral_laplacian(
        weight,
        wave_number_squared,
    )
    weight_time_derivative = (
        -np.einsum(
            "ixyz,ixyz->xyz",
            velocity,
            weight_gradient,
        )
        - viscosity * weight_laplacian
    )

    gradient_energy_density = np.sum(
        velocity_gradient**2,
        axis=(0, 1),
    )
    weighted_gradient_energy = float(
        np.mean(weight * gradient_energy_density)
    )
    weighted_gradient_energy_derivative = float(
        np.mean(
            weight_time_derivative * gradient_energy_density
            + 2.0
            * weight
            * np.sum(
                velocity_gradient
                * velocity_gradient_time_derivative,
                axis=(0, 1),
            )
        )
    )

    strain = (
        velocity_gradient + velocity_gradient.swapaxes(0, 1)
    ) / 2.0
    strain_correction_density = np.zeros_like(weight)
    pressure_correction_vector = np.zeros_like(velocity)
    hessian_energy_density = np.zeros_like(weight)
    replica_pressure_rows = []
    for noise_direction in range(3):
        replica = velocity_gradient[:, noise_direction]
        replica_gradient = hessian[:, :, noise_direction]
        replica_pressure = _replica_pressure(
            velocity,
            velocity_gradient,
            replica,
            replica_gradient,
            wave_vector,
            safe_wave_number,
            nonzero_wave_number,
        )
        strain_correction_density += 2.0 * viscosity * np.einsum(
            "ixyz,ijxyz,jxyz->xyz",
            replica,
            strain,
            replica,
        )
        pressure_correction_vector += (
            4.0 * viscosity * replica_pressure * replica
        )
        hessian_energy_density += (
            2.0
            * viscosity
            * np.sum(replica_gradient**2, axis=(0, 1))
        )
        replica_pressure_rows.append(
            {
                "noise_direction": noise_direction,
                "pressure_rms": float(
                    np.sqrt(np.mean(replica_pressure**2))
                ),
            }
        )

    stochastic_strain_term = float(
        -3.0 * np.mean(weight * strain_correction_density)
    )
    stochastic_pressure_term = float(
        1.5
        * np.mean(
            np.einsum(
                "ixyz,ixyz->xyz",
                weight_gradient,
                pressure_correction_vector,
            )
        )
    )
    stochastic_gradient_term = float(
        -3.0
        * viscosity
        * np.mean(weight * hessian_energy_density)
    )
    leading_reset_loss = (
        3.0 * viscosity * weighted_gradient_energy
    )
    first_time_coefficient = (
        3.0
        * viscosity
        * weighted_gradient_energy_derivative
        + stochastic_strain_term
        + stochastic_pressure_term
        + stochastic_gradient_term
    )
    derivative_crossover = None
    integrated_crossover = None
    if first_time_coefficient < 0.0:
        derivative_crossover = (
            -leading_reset_loss / first_time_coefficient
        )
        integrated_crossover = (
            -2.0 * leading_reset_loss / first_time_coefficient
        )

    mean_pressure = float(np.mean(pressure))
    physical_replica_pressure = _replica_pressure(
        velocity,
        velocity_gradient,
        velocity,
        velocity_gradient,
        wave_vector,
        safe_wave_number,
        nonzero_wave_number,
    )
    expected_physical_pressure = (
        pressure - 0.5 * np.sum(velocity**2, axis=0)
    )
    expected_physical_pressure -= np.mean(expected_physical_pressure)
    physical_pressure_residual = float(
        np.max(
            np.abs(
                physical_replica_pressure - expected_physical_pressure
            )
        )
    )
    return {
        "grid_size": size,
        "pressure_mean": mean_pressure,
        "pressure_gradient_reconstruction_residual": pressure_residual,
        "replica_pressure_gauge_residual": physical_pressure_residual,
        "weighted_gradient_energy": weighted_gradient_energy,
        "weighted_gradient_energy_time_derivative": (
            weighted_gradient_energy_derivative
        ),
        "leading_reset_rho_loss": leading_reset_loss,
        "stochastic_strain_coefficient": stochastic_strain_term,
        "stochastic_pressure_coefficient": stochastic_pressure_term,
        "stochastic_gradient_coefficient": stochastic_gradient_term,
        "first_time_coefficient": first_time_coefficient,
        "formal_derivative_crossover_time": derivative_crossover,
        "formal_integrated_crossover_time": integrated_crossover,
        "replica_pressure_rows": replica_pressure_rows,
    }


def _short_time_rho_audit(
    base_velocity: Array,
    fields: dict[str, object],
) -> dict[str, Any]:
    rows = [
        _short_time_rho_row(
            size,
            base_velocity,
            np.asarray(fields["velocity_gradient_grid"]),
            np.asarray(fields["pressure_potential_gradient_grid"]),
        )
        for size in (48, 64, 80)
    ]
    leading_values = [row["leading_reset_rho_loss"] for row in rows]
    first_coefficients = [row["first_time_coefficient"] for row in rows]
    return {
        "first_chaos_expansion": (
            "V_r=u-sqrt(2nu) partial_k u Delta W_r^k+O(tau)"
        ),
        "rho_derivative_at_reset": (
            "partial_rho Q_rho|_(rho=0,t=s)="
            "3nu integral lambda|grad u|^2>0"
        ),
        "integrated_short_time_ordering": (
            "integral_s^(s+h)(Q_rho-Q_0)="
            "3nu rho h integral lambda|grad u|^2+O(rho h^2)"
        ),
        "first_time_coefficient_formula": (
            "K1=3nu D_t-3 integral lambda R_rho,1"
            "+(3/2)integral grad lambda dot F_rho,1"
            "-3nu integral lambda G_rho,1"
        ),
        "pressure_linearization": (
            "Delta Pi_u[V]=-div((u dot grad)V+(grad u)^T V)"
        ),
        "rows": rows,
        "leading_reset_loss_range": [
            min(leading_values),
            max(leading_values),
        ],
        "first_time_coefficient_range": [
            min(first_coefficients),
            max(first_coefficients),
        ],
        "interpretation": (
            "Positive rho is strictly worse over every sufficiently short "
            "restart window with nonzero weighted gradient energy. The "
            "reported crossover times come only from the quadratic Taylor "
            "truncation and are diagnostics, not sign certificates."
        ),
        "all_checks_pass": (
            max(leading_values) - min(leading_values) < 2.0e-8
            and min(leading_values) > 2000.0
            and max(first_coefficients) - min(first_coefficients) < 2.0e-5
            and all(
                row["pressure_gradient_reconstruction_residual"]
                < 2.0e-11
                and row["replica_pressure_gauge_residual"] < 2.0e-10
                for row in rows
            )
        ),
    }


def audit() -> dict[str, Any]:
    fields = _build_spectral_fields()
    modes, coefficients = fields["velocity"]
    base_velocity = _evaluate_velocity_grid(modes, coefficients)

    scale = _scale_homogeneity_audit()
    resolved_size = 96
    velocity = _periodic_resample(base_velocity, resolved_size)
    velocity_gradient = _periodic_resample(
        np.asarray(fields["velocity_gradient_grid"]),
        resolved_size,
    )
    pressure_gradient = _periodic_resample(
        np.asarray(fields["pressure_potential_gradient_grid"]),
        resolved_size,
    )
    pressure, pressure_residual = _pressure_from_gradient(
        pressure_gradient
    )
    frequency_rows = [
        _frequency_edge_row(
            frequency,
            velocity,
            velocity_gradient,
            pressure,
        )
        for frequency in range(1, 13)
    ]
    frequency_audit = {
        "grid_size": resolved_size,
        "pressure_gradient_reconstruction_residual": pressure_residual,
        "rows": frequency_rows,
        "positive_pressure_frequencies": [
            row["partition_frequency"]
            for row in frequency_rows
            if row["direct_pressure"] > 1.0e-10
        ],
        "spectrally_silent_frequencies": [
            row["partition_frequency"]
            for row in frequency_rows
            if abs(row["direct_pressure"]) < 1.0e-10
        ],
        "all_checks_pass": (
            pressure_residual < 2.0e-11
            and all(row["all_checks_pass"] for row in frequency_rows)
        ),
    }
    short_time = _short_time_rho_audit(base_velocity, fields)

    positive_checks = {
        "scale_homogeneity_algebra_passes": scale["all_checks_pass"],
        "frequency_edge_identities_pass": frequency_audit[
            "all_checks_pass"
        ],
        "short_time_rho_expansion_checks_pass": short_time[
            "all_checks_pass"
        ],
    }
    certification_flags = {
        "scale_adapted_edge_identity_derived": True,
        "edge_Young_ratio_equals_local_Reynolds_squared": True,
        "fixed_scale_universal_edge_absorption_falsified_by_scaling": True,
        "partition_frequency_must_track_local_amplitude": True,
        "first_short_time_rho_correction_derived": True,
        "positive_rho_is_worse_on_sufficiently_short_restart_windows": True,
        "replica_pressure_linearization_implemented": True,
        "scale_adapted_edge_remainder_absorbed": False,
        "finite_time_positive_rho_advantage_proved": False,
        "Taylor_crossover_is_a_sign_certificate": False,
        "critical_signed_replica_bound_proved": False,
        "low_regularity_scale_adapted_partition_justified": False,
        "exceptional_set_upgrade_proved": False,
        "Navier_Stokes_global_regularity_proved": False,
    }
    return {
        "kind": "scale_adapted_edge_rho_expansion_audit",
        "schema_version": 1,
        "status": (
            "fixed_scale_edge_absorption_falsified_"
            "short_time_positive_rho_advantage_excluded"
        ),
        "assumption_scope": (
            "Smooth periodic Navier-Stokes and smooth reset projected "
            "replicas. Frequency sweeps are resolved deterministic "
            "diagnostics; Taylor crossover times are not rigorous beyond "
            "the displayed expansion."
        ),
        "scale_homogeneity": scale,
        "partition_frequency_sweep": frequency_audit,
        "short_time_rho_expansion": short_time,
        "certification_flags": certification_flags,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "next_theorem_target": (
            "Replace fixed partition scale by an intrinsic local-Reynolds "
            "scale m comparable to amplitude/nu and derive a pressure-tail "
            "estimate uniform under that adaptation. Since rho>0 is worse "
            "through first order on short restart windows, test correlation "
            "only on a nonperturbative finite window and require measured "
            "R_rho/F_rho improvement to exceed the exact reset dissipation "
            "loss. Do not treat the quadratic Taylor crossover as evidence "
            "of such an advantage."
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
        raise RuntimeError("scale-adapted edge/rho expansion audit failed")
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
