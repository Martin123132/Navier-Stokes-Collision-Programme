"""Audit the high-carrier weighted-Fisher gate at partition zero faces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp


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


def _unweighted_highpass_coercivity() -> dict[str, Any]:
    carrier, partition, load = sp.symbols("K m B", positive=True)
    riesz_constant, sobolev_constant = sp.symbols(
        "C_R C_S",
        positive=True,
    )
    gradient_bound = sp.sqrt(3) * partition / 2
    enstrophy = sp.symbols("E", positive=True)
    load_upper = (
        riesz_constant
        * gradient_bound
        * sobolev_constant ** sp.Rational(3, 2)
        * carrier ** sp.Rational(-3, 2)
        * enstrophy ** sp.Rational(3, 2)
    )
    enstrophy_lower = sp.simplify(
        carrier
        / sobolev_constant
        * (load / (riesz_constant * gradient_bound))
        ** sp.Rational(2, 3)
    )
    recovered_load = sp.simplify(
        load_upper.subs(enstrophy, enstrophy_lower)
    )
    return {
        "assumptions": (
            "u is smooth, mean zero, divergence free on the normalized "
            "three-torus, and u_hat(k)=0 for |k|<K. Pressure is "
            "p=-R_iR_j(u_i u_j)."
        ),
        "load": "b_Phi=mean[p u dot grad Phi]",
        "holder_riesz_step": (
            "|b_Phi|<=C_R||grad Phi||_infinity||u||_3^3"
        ),
        "highpass_interpolation_step": (
            "||u||_3^3<=C_S^(3/2)K^(-3/2)||grad u||_2^3"
        ),
        "vertex_partition_gradient_bound": (
            "||grad Phi_v,m||_infinity<=sqrt(3)m/2"
        ),
        "load_upper_bound": str(load_upper),
        "enstrophy_lower_bound": str(enstrophy_lower),
        "carrier_exponent_for_fixed_load": 1,
        "scope": (
            "This proves linear carrier coercivity for unweighted "
            "enstrophy of a pure high-pass velocity. It does not control "
            "vertex-weighted enstrophy and does not yet treat mixed "
            "low/high velocity interactions."
        ),
        "all_checks_pass": (
            recovered_load == load
            and sp.simplify(
                carrier
                * sp.diff(sp.log(enstrophy_lower), carrier)
            )
            == 1
        ),
    }


def _square_factor_highpass_bridge() -> dict[str, Any]:
    carrier, partition = sp.symbols("K m", positive=True)
    weighted_mass, weighted_fisher = sp.symbols(
        "M_v E_v",
        positive=True,
    )
    shift_radius = sp.sqrt(3) * partition / 2
    shifted_floor = carrier - shift_radius
    denominator = sp.factor(
        shifted_floor**2 - shift_radius**2
    )
    weighted_mass_upper = sp.simplify(
        weighted_fisher / denominator
    )
    gradient_factor = sp.simplify(
        1
        + sp.sqrt(
            1
            + shift_radius**2 / denominator
        )
    )
    twice_intrinsic_partition = 2 * sp.sqrt(3) * partition
    factor_at_twice = sp.simplify(
        gradient_factor.subs(carrier, twice_intrinsic_partition)
    )
    return {
        "factorization": (
            "Phi_v,m=psi_v,m^2, where each one-dimensional factor of "
            "psi is sin(mx_j/2) or cos(mx_j/2)"
        ),
        "half_lattice_support": (
            "psi_hat is supported on the eight shifts "
            "(+/-m/2,+/-m/2,+/-m/2), each of magnitude sqrt(3)m/2"
        ),
        "eigenfunction_identity": (
            "Delta psi_v,m=-(3m^2/4)psi_v,m"
        ),
        "ground_state_identity": (
            "||grad(psi u)||_2^2"
            "=mean[Phi|grad u|^2]+(3m^2/4)||psi u||_2^2"
        ),
        "shifted_carrier_floor": str(shifted_floor),
        "weighted_mass_symbol": "M_v=||psi_v,m u||_2^2",
        "weighted_Fisher_symbol": (
            "E_v=mean[Phi_v,m|grad u|^2]"
        ),
        "coercive_denominator": str(denominator),
        "validity_threshold": "K>sqrt(3)m",
        "weighted_mass_upper_bound": str(weighted_mass_upper),
        "weighted_zero_face_gradient_bound": (
            "||u grad psi||_2<="
            "[1+sqrt(1+(3m^2/4)/(K(K-sqrt(3)m)))]sqrt(E_v)"
        ),
        "gradient_factor": str(gradient_factor),
        "gradient_factor_at_K_equals_2sqrt3m": str(factor_at_twice),
        "interpretation": (
            "Zero faces can hide unweighted mass, but they do not hide "
            "the correctly weighted mass psi u from weighted Fisher once "
            "the velocity is spectrally separated. The remaining pure "
            "high-pass obstruction is the pressure multiplier commutator."
        ),
        "all_checks_pass": (
            sp.simplify(
                denominator
                - carrier * (carrier - sp.sqrt(3) * partition)
            )
            == 0
            and sp.simplify(
                (
                    shifted_floor**2 * weighted_mass
                    - (
                        weighted_fisher
                        + shift_radius**2 * weighted_mass
                    )
                ).subs(
                    weighted_mass,
                    weighted_mass_upper,
                )
            )
            == 0
            and factor_at_twice == 1 + 3 * sp.sqrt(2) / 4
        ),
    }


def _sine_window_packet(carrier: int) -> dict[str, Any]:
    indices = np.arange(1, carrier + 1, dtype=float)
    modes = carrier + indices
    coefficients = math.sqrt(2.0 / (carrier + 1.0)) * np.sin(
        math.pi * indices / (carrier + 1.0)
    )
    derivative = modes * coefficients
    l2_squared = float(np.sum(coefficients**2))
    unweighted = float(np.sum(derivative**2))
    weighted_toeplitz = float(
        0.5 * np.sum(derivative**2)
        - 0.5 * np.sum(derivative[:-1] * derivative[1:])
    )
    extended = np.concatenate(([0.0], derivative, [0.0]))
    weighted_difference = float(0.25 * np.sum(np.diff(extended) ** 2))
    analytic_bound = float(
        (
            5 * sp.pi**2
            + 2 * (1 + 2 * sp.pi) ** 2
        )
        / 4
    )
    return {
        "carrier": carrier,
        "minimum_mode": int(modes[0]),
        "maximum_mode": int(modes[-1]),
        "L2_squared": l2_squared,
        "unweighted_Dirichlet": unweighted,
        "weighted_Dirichlet": weighted_toeplitz,
        "difference_form_Dirichlet": weighted_difference,
        "weighted_over_unweighted": weighted_toeplitz / unweighted,
        "analytic_uniform_upper_bound": analytic_bound,
        "all_checks_pass": bool(
            abs(l2_squared - 1.0) < 1.0e-12
            and abs(weighted_toeplitz - weighted_difference) < 1.0e-10
            and unweighted >= (carrier + 1) ** 2
            and weighted_toeplitz <= analytic_bound
        ),
    }


def _zero_face_uncertainty_packet() -> dict[str, Any]:
    rows = [
        _sine_window_packet(carrier)
        for carrier in (8, 16, 32, 64, 128, 256)
    ]
    weighted = np.asarray(
        [row["weighted_Dirichlet"] for row in rows],
        dtype=float,
    )
    ratios = np.asarray(
        [row["weighted_over_unweighted"] for row in rows],
        dtype=float,
    )
    return {
        "weight": "phi_-(x)=sin(x/2)^2",
        "packet": (
            "f_N=sum_(j=1)^N sqrt(2/(N+1))"
            "sin(pi j/(N+1)) exp(i(N+j)x)"
        ),
        "exact_difference_identity": (
            "mean[phi_-|f_N'|^2]"
            "=(1/4)sum_k|d_k-d_(k-1)|^2, d_k=k f_hat_N(k)"
        ),
        "uniform_bound_proof": (
            "The sine window vanishes at both spectral endpoints. "
            "|d_(k+1)-d_k| is at most "
            "sqrt(2/(N+1))(1+2pi), while the two endpoint terms are "
            "bounded by the same elementary sine estimate. Hence the "
            "displayed analytic bound is independent of N."
        ),
        "divergence_free_embedding": (
            "For N>=2, u_N(x)=(0,sqrt(2) Re f_N(x_1),0) is real and "
            "divergence free; positive and negative frequency blocks do "
            "not couple through the frequency-one weight, so it has the "
            "same normalized weighted Dirichlet form. Its induced pressure "
            "is zero, making this a support-only weighted-coercivity no-go, "
            "not a pressure-load counterexample."
        ),
        "rows": rows,
        "maximum_weighted_Dirichlet": float(np.max(weighted)),
        "ratio_drop": float(ratios[0] / ratios[-1]),
        "conclusion": (
            "A partition zero face can hide arbitrarily high Fourier "
            "carrier from vertex-weighted Fisher energy. Fourier support "
            "alone therefore cannot promote the unweighted carrier lower "
            "bound to a weighted one."
        ),
        "all_checks_pass": bool(
            all(row["all_checks_pass"] for row in rows)
            and np.max(weighted)
            < rows[0]["analytic_uniform_upper_bound"]
            and np.all(np.diff(ratios) < 0.0)
            and ratios[-1] < 1.0e-4
        ),
    }


def _concentration_scaling() -> dict[str, Any]:
    delta, amplitude, viscosity, target = sp.symbols(
        "delta A nu B",
        positive=True,
    )
    rows = []
    for localized_dimensions in (1, 2, 3):
        load = amplitude**3 * delta ** (localized_dimensions + 1)
        weighted_fisher = (
            amplitude**2 * delta**localized_dimensions
        )
        fixed_load_amplitude = (
            target ** sp.Rational(1, 3)
            * delta
            ** sp.Rational(-(localized_dimensions + 1), 3)
        )
        fixed_load_weighted = sp.simplify(
            weighted_fisher.subs(amplitude, fixed_load_amplitude)
        )
        unweighted_fisher = (
            amplitude**2
            * delta ** (localized_dimensions - 2)
        )
        fixed_load_unweighted = sp.simplify(
            unweighted_fisher.subs(
                amplitude,
                fixed_load_amplitude,
            )
        )
        ratio = sp.simplify(
            load / (viscosity * weighted_fisher)
        )
        rows.append(
            {
                "localized_dimensions": localized_dimensions,
                "unit_amplitude_load_scaling": str(load),
                "unit_amplitude_weighted_Fisher_scaling": str(
                    weighted_fisher
                ),
                "fixed_load_amplitude": str(fixed_load_amplitude),
                "fixed_load_weighted_Fisher": str(
                    fixed_load_weighted
                ),
                "fixed_load_weighted_delta_exponent": str(
                    sp.Rational(localized_dimensions - 2, 3)
                ),
                "fixed_load_unweighted_Fisher": str(
                    fixed_load_unweighted
                ),
                "pressure_to_weighted_Fisher_ratio": str(ratio),
            }
        )
    generic_unweighted_fixed_load = sp.simplify(
        (
            amplitude**2 * delta
        ).subs(
            amplitude,
            target ** sp.Rational(1, 3) / delta,
        )
    )
    return {
        "model": (
            "A divergence-free band-limited packet of amplitude A and "
            "width delta is centered O(delta) from a quadratic zero face. "
            "It is localized in d coordinate directions."
        ),
        "rows": rows,
        "intrinsic_ratio": (
            "load/(nu weighted_Fisher)=A delta/nu=A/(nu K)"
        ),
        "generic_positive_weight_three_dimensional_fixed_load": str(
            generic_unweighted_fixed_load
        ),
        "interpretation": (
            "Full three-dimensional concentration can make the weighted "
            "Fisher cost of a fixed zero-face load scale as delta^(1/3), "
            "while its unweighted Fisher cost grows as delta^(-5/3). "
            "This does not defeat an intrinsic threshold K comparable to "
            "A/nu: the adverse ratio is exactly A/(nu K)."
        ),
        "all_checks_pass": (
            rows[0]["fixed_load_weighted_delta_exponent"] == "-1/3"
            and rows[1]["fixed_load_weighted_delta_exponent"] == "0"
            and rows[2]["fixed_load_weighted_delta_exponent"] == "1/3"
            and all(
                row["pressure_to_weighted_Fisher_ratio"]
                == "A*delta/nu"
                for row in rows
            )
            and generic_unweighted_fixed_load
            == target ** sp.Rational(2, 3) / delta
        ),
    }


def _frequency_grid(size: int) -> tuple[np.ndarray, ...]:
    frequencies = np.fft.fftfreq(size) * size
    return np.meshgrid(
        frequencies,
        frequencies,
        frequencies,
        indexing="ij",
        sparse=True,
    )


def _fejer_peak_one(value: np.ndarray, order: int) -> np.ndarray:
    denominator = np.sin(value / 2.0)
    numerator = np.sin(order * value / 2.0)
    result = np.empty_like(value)
    near_zero = np.abs(denominator) < 1.0e-13
    result[near_zero] = 1.0
    result[~near_zero] = (
        numerator[~near_zero]
        / (order * denominator[~near_zero])
    ) ** 2
    return result


def _pde_packet_row(order: int) -> dict[str, Any]:
    size = 32 * order
    coordinate = 2.0 * math.pi * np.arange(size) / size
    x = coordinate[:, None, None]
    y = coordinate[None, :, None]
    z = coordinate[None, None, :]
    center = (1.0 / order, 0.0, 0.0)
    dx = x - center[0]
    dy = y - center[1]
    dz = z - center[2]
    envelope = (
        _fejer_peak_one(dx, order)
        * _fejer_peak_one(dy, order)
        * _fejer_peak_one(dz, order)
    )

    phases = (
        4 * order * dx,
        4 * order * dy,
        -4 * order * (dx + dy),
    )
    polarizations = (
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((1.0, -1.0, 0.0)) / math.sqrt(2.0),
    )
    velocity = np.zeros((3, size, size, size), dtype=float)
    for phase, polarization in zip(phases, polarizations):
        wave = np.cos(phase)
        for component in range(3):
            velocity[component] += (
                envelope * polarization[component] * wave
            )

    wave_vectors = _frequency_grid(size)
    wave_squared = sum(value**2 for value in wave_vectors)
    nonzero_wave_squared = np.where(wave_squared == 0.0, 1.0, wave_squared)
    velocity_hat = np.fft.fftn(velocity, axes=(1, 2, 3))
    longitudinal = sum(
        wave_vectors[component] * velocity_hat[component]
        for component in range(3)
    )
    for component in range(3):
        velocity_hat[component] -= (
            wave_vectors[component]
            * longitudinal
            / nonzero_wave_squared
        )
    velocity_hat *= wave_squared > 0.0
    velocity = np.fft.ifftn(
        velocity_hat,
        axes=(1, 2, 3),
    ).real

    pressure_hat = np.zeros((size, size, size), dtype=np.complex128)
    for first in range(3):
        for second in range(3):
            product_hat = np.fft.fftn(
                velocity[first] * velocity[second]
            )
            pressure_hat -= (
                wave_vectors[first]
                * wave_vectors[second]
                / nonzero_wave_squared
                * product_hat
            )
    pressure_hat *= wave_squared > 0.0
    pressure = np.fft.ifftn(pressure_hat).real

    phi_minus_x = (1.0 - np.cos(x)) / 2.0
    phi_plus_y = (1.0 + np.cos(y)) / 2.0
    phi_plus_z = (1.0 + np.cos(z)) / 2.0
    weight = phi_minus_x * phi_plus_y * phi_plus_z
    weight_gradient = (
        np.sin(x) / 2.0 * phi_plus_y * phi_plus_z,
        -phi_minus_x * np.sin(y) / 2.0 * phi_plus_z,
        -phi_minus_x * phi_plus_y * np.sin(z) / 2.0,
    )
    pressure_load_density = pressure * sum(
        velocity[component] * weight_gradient[component]
        for component in range(3)
    )
    base_load = float(np.mean(pressure_load_density))

    gradient_squared = np.zeros_like(pressure)
    divergence_hat = np.zeros_like(pressure_hat)
    for component in range(3):
        divergence_hat += (
            1j
            * wave_vectors[component]
            * velocity_hat[component]
        )
        for direction in range(3):
            derivative = np.fft.ifftn(
                1j
                * wave_vectors[direction]
                * velocity_hat[component]
            ).real
            gradient_squared += derivative**2
    unweighted_fisher = float(np.mean(gradient_squared))
    weighted_fisher = float(np.mean(weight * gradient_squared))
    velocity_linf = float(
        np.max(np.sqrt(np.sum(velocity**2, axis=0)))
    )

    coefficient_size = np.sqrt(
        np.sum(np.abs(velocity_hat) ** 2, axis=0)
    )
    occupied = coefficient_size > (
        1.0e-10 * float(np.max(coefficient_size))
    )
    occupied_wave_squared = np.broadcast_to(
        wave_squared,
        occupied.shape,
    )[occupied]
    minimum_mode = float(np.sqrt(np.min(occupied_wave_squared)))
    maximum_mode = float(np.sqrt(np.max(occupied_wave_squared)))
    divergence_residual = float(
        np.max(np.abs(divergence_hat))
        / max(float(np.max(coefficient_size)), 1.0)
    )

    amplitude = abs(base_load) ** (-1.0 / 3.0)
    scaled_load = amplitude**3 * base_load
    return {
        "order": order,
        "grid_size": size,
        "base_pressure_load": base_load,
        "normalizing_amplitude": amplitude,
        "normalized_pressure_load": scaled_load,
        "minimum_velocity_mode": minimum_mode,
        "maximum_velocity_mode": maximum_mode,
        "minimum_mode_over_order": minimum_mode / order,
        "maximum_relative_divergence_residual": divergence_residual,
        "normalized_weighted_Fisher": (
            amplitude**2 * weighted_fisher
        ),
        "normalized_unweighted_Fisher": (
            amplitude**2 * unweighted_fisher
        ),
        "normalized_velocity_L_infinity": amplitude * velocity_linf,
        "intrinsic_Reynolds_proxy": (
            amplitude * velocity_linf / minimum_mode
        ),
        "scope": (
            "Alias-free finite-Fourier FFT pilot for the displayed "
            "Fejer-windowed carrier triad. Values are binary64 diagnostics, "
            "not interval enclosures or an asymptotic proof."
        ),
        "all_checks_pass": bool(
            abs(base_load) > 1.0e-12
            and abs(abs(scaled_load) - 1.0) < 1.0e-10
            and minimum_mode >= 3.0 * order
            and maximum_mode < 9.0 * order
            and divergence_residual < 1.0e-10
        ),
    }


def _pde_zero_face_packet_pilot() -> dict[str, Any]:
    rows = [_pde_packet_row(order) for order in (2, 3, 4, 5)]
    orders = np.asarray([row["order"] for row in rows], dtype=float)

    def slope(field: str) -> float:
        values = np.asarray([row[field] for row in rows], dtype=float)
        return float(np.polyfit(np.log(orders), np.log(values), 1)[0])

    weighted_slope = slope("normalized_weighted_Fisher")
    unweighted_slope = slope("normalized_unweighted_Fisher")
    amplitude_slope = slope("normalizing_amplitude")
    intrinsic_slope = slope("intrinsic_Reynolds_proxy")
    return {
        "construction": (
            "A real three-carrier triad is multiplied by a peak-one "
            "three-dimensional Fejer window centered one packet width from "
            "the zero face of Phi=(-,+,+), then projected modewise onto "
            "divergence-free fields. Grid size 32N makes the cubic load "
            "integration alias free."
        ),
        "rows": rows,
        "fitted_log_slopes": {
            "normalizing_amplitude": amplitude_slope,
            "weighted_Fisher": weighted_slope,
            "unweighted_Fisher": unweighted_slope,
            "intrinsic_Reynolds_proxy": intrinsic_slope,
        },
        "predicted_full_concentration_slopes": {
            "normalizing_amplitude": 4.0 / 3.0,
            "weighted_Fisher": -1.0 / 3.0,
            "unweighted_Fisher": 5.0 / 3.0,
            "intrinsic_Reynolds_proxy": 1.0 / 3.0,
        },
        "interpretation": (
            "This tests whether a strict high-pass divergence-free field "
            "and its actual Poisson pressure can carry a fixed compatible "
            "zero-face load while weighted and unweighted Fisher costs "
            "separate. Three small carriers diagnose the mechanism; they "
            "do not certify the limiting exponents."
        ),
        "all_checks_pass": bool(
            all(row["all_checks_pass"] for row in rows)
            and rows[-1]["normalized_weighted_Fisher"]
            < rows[0]["normalized_weighted_Fisher"]
            and rows[-1]["normalized_unweighted_Fisher"]
            > rows[0]["normalized_unweighted_Fisher"]
            and rows[-1]["intrinsic_Reynolds_proxy"]
            > rows[0]["intrinsic_Reynolds_proxy"]
        ),
    }


def audit() -> dict[str, Any]:
    unweighted = _unweighted_highpass_coercivity()
    square_factor = _square_factor_highpass_bridge()
    uncertainty = _zero_face_uncertainty_packet()
    scaling = _concentration_scaling()
    pde_pilot = _pde_zero_face_packet_pilot()
    positive_checks = {
        "unweighted_highpass_coercivity_passes": unweighted[
            "all_checks_pass"
        ],
        "square_factor_highpass_bridge_passes": square_factor[
            "all_checks_pass"
        ],
        "zero_face_uncertainty_packet_passes": uncertainty[
            "all_checks_pass"
        ],
        "concentration_scaling_passes": scaling["all_checks_pass"],
        "PDE_zero_face_packet_pilot_passes": pde_pilot[
            "all_checks_pass"
        ],
    }
    return {
        "kind": "high_carrier_weighted_fisher_gate_audit",
        "schema_version": 1,
        "status": (
            "highpass_linear_unweighted_and_square_factor_"
            "bridges_certified"
        ),
        "assumption_scope": (
            "Smooth periodic pure high-pass velocities for the rigorous "
            "unweighted theorem; an exact scalar/divergence-free shear "
            "packet for the support-only weighted no-go; symbolic "
            "zero-face concentration scaling; and a binary64 "
            "finite-Fourier pressure-packet pilot."
        ),
        "unweighted_highpass_coercivity": unweighted,
        "square_factor_highpass_bridge": square_factor,
        "zero_face_uncertainty_packet": uncertainty,
        "zero_face_concentration_scaling": scaling,
        "PDE_zero_face_packet_pilot": pde_pilot,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "pure_highpass_unweighted_H1_least_cost_linear_coercivity_proved": (
                True
            ),
            "vertex_square_factor_highpass_mass_bridge_proved": True,
            "vertex_zero_face_gradient_controlled_by_weighted_Fisher": True,
            "global_quadratic_carrier_coercivity_proved": False,
            "weighted_Fisher_carrier_coercivity_from_support_alone": False,
            "zero_face_uncertainty_mechanism_exactly_realized": True,
            "PDE_pressure_packet_mechanism_numerically_realized": True,
            "PDE_pressure_packet_asymptotic_counterexample_proved": False,
            "intrinsic_ratio_A_over_nuK_survives_scaling": True,
            "general_intrinsic_high_carrier_absorption_proved": False,
            "mixed_low_high_velocity_remainder_controlled": False,
            "critical_signed_bound_proved": False,
            "low_regularity_passage_proved": False,
            "exceptional_set_upgrade_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "route_decision": (
            "Use the rigorous linear carrier lower bound as the universal "
            "baseline; do not promote the explicit block family's "
            "quadratic growth globally without a separate proof. Do not "
            "transfer unweighted mass coercivity directly to a vertex "
            "weight: zero faces admit exact uncertainty packets with "
            "uniform weighted Dirichlet cost. Instead use the exact "
            "square-factor bridge, which controls psi_v u and "
            "u grad psi_v by vertex-weighted Fisher. The concentration "
            "calculation and actual pressure pilot preserve A/(nu K), so "
            "the remaining theorem is a high-output pressure multiplier "
            "commutator estimate at amplitude-relative carrier."
        ),
        "next_theorem_target": (
            "Prove or falsify the remaining high-output commutator bound "
            "for T=Q_H R_iR_j: control psi_v T(u_i u_j) by "
            "||u||_infinity times ||psi_v u|| plus a term bounded by "
            "||u||_infinity/K times ||u grad psi_v||. Combined with the "
            "certified square-factor bridge, this would give floor-free "
            "pure-high-pass absorption for K>=C||u||_infinity/nu. Only "
            "then add mixed low/high paraproduct terms."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "work/ns_collision/results/"
            "high_carrier_weighted_fisher_gate_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("high-carrier weighted-Fisher gate audit failed")
    _atomic_json(arguments.output, result)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": _sha256(arguments.output),
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
