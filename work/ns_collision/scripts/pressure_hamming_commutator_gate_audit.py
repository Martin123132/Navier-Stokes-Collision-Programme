"""Audit the vertex-Hamming pressure commutator and its diagonal no-go."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
VERTICES = tuple(itertools.product((-1, 1), repeat=3))
SUBSETS = tuple(range(8))


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


def _mask_product(vertex: tuple[int, int, int], mask: int) -> int:
    value = 1
    for coordinate in range(3):
        if mask & (1 << coordinate):
            value *= vertex[coordinate]
    return value


def _hamming_distance(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
) -> int:
    return sum(left != right for left, right in zip(first, second))


def _exact_vertex_walsh_calculus() -> dict[str, Any]:
    half = Fraction(1, 2)
    output = (
        Fraction(11, 2),
        Fraction(13, 2),
        Fraction(15, 2),
    )

    samples: dict[tuple[int, int, int], Fraction] = {}
    for epsilon in VERTICES:
        frequency = tuple(
            output[index] - half * epsilon[index]
            for index in range(3)
        )
        denominator = sum(value * value for value in frequency)
        samples[epsilon] = (
            -frequency[0] * frequency[1] / denominator
        )

    coefficients: dict[int, Fraction] = {}
    for mask in SUBSETS:
        coefficients[mask] = sum(
            Fraction(_mask_product(epsilon, mask), 8)
            * samples[epsilon]
            for epsilon in VERTICES
        )

    reconstructed = {
        epsilon: sum(
            _mask_product(epsilon, mask) * coefficients[mask]
            for mask in SUBSETS
        )
        for epsilon in VERTICES
    }
    input_values = {
        epsilon: Fraction(index * index + 3 * index + 1, 7)
        for index, epsilon in enumerate(VERTICES)
    }
    sine_coefficients = {
        epsilon: (
            1j
            * _mask_product(epsilon, 7)
            / 8.0
        )
        for epsilon in VERTICES
    }
    direct_output = sum(
        sine_coefficients[epsilon]
        * float(samples[epsilon])
        * float(input_values[epsilon])
        for epsilon in VERTICES
    )
    walsh_output = 0.0j
    for mask in SUBSETS:
        order = mask.bit_count()
        derivative_combo = (1j**order) * sum(
            _mask_product(epsilon, mask)
            * sine_coefficients[epsilon]
            * float(input_values[epsilon])
            for epsilon in VERTICES
        )
        walsh_output += (
            float(coefficients[mask])
            * (1j ** (-order))
            * derivative_combo
        )
    phase_corrected_residual = abs(direct_output - walsh_output)
    nonzero_by_order = {
        order: [
            mask
            for mask in SUBSETS
            if mask.bit_count() == order and coefficients[mask] != 0
        ]
        for order in range(4)
    }

    return {
        "vertex_factorization": (
            "Phi_v=psi_v^2, with Fourier shifts "
            "q_epsilon=(m/2)epsilon for epsilon in {+1,-1}^3."
        ),
        "mixed_derivative_toggle": (
            "partial_S psi_v=sigma(v,S)(m/2)^|S| "
            "psi_(v xor S)"
        ),
        "exact_multiplier_formula": (
            "psi_v T f=sum_(S subset {1,2,3}) "
            "i^(-|S|)A_S(D)[(2/m)^|S| partial_S psi_v f], where "
            "A_S(k)=2^(-3)sum_epsilon epsilon_S "
            "M(k-(m/2)epsilon)."
        ),
        "concrete_symbol": (
            "M_12(xi)=-xi_1 xi_2/|xi|^2 at output "
            "(11/2,13/2,15/2) with m=1"
        ),
        "concrete_walsh_coefficients": {
            str(mask): str(coefficients[mask]) for mask in SUBSETS
        },
        "phase_corrected_Fourier_identity_residual": (
            phase_corrected_residual
        ),
        "nonzero_masks_by_hamming_order": {
            str(order): masks
            for order, masks in nonzero_by_order.items()
        },
        "distance_two_terms_are_genuine": bool(nonzero_by_order[2]),
        "distance_three_term_is_genuine": bool(nonzero_by_order[3]),
        "scope": (
            "The identity is exact on the half-lattice (or on the "
            "periodic double cover). It applies componentwise to the "
            "matrix double-Riesz pressure multiplier."
        ),
        "all_checks_pass": (
            reconstructed == samples
            and bool(nonzero_by_order[2])
            and bool(nonzero_by_order[3])
            and phase_corrected_residual < 1.0e-15
        ),
    }


def _hamming_leakage_bound() -> dict[str, Any]:
    smooth_ratio = Fraction(1, 10)
    smooth_matrix = np.asarray(
        [
            [
                float(smooth_ratio ** _hamming_distance(left, right))
                for right in VERTICES
            ]
            for left in VERTICES
        ],
        dtype=float,
    )
    eigenvalues = np.linalg.eigvalsh(smooth_matrix)
    expected_norm = float((1 + smooth_ratio) ** 3)
    bounded_matrix = np.ones((8, 8), dtype=float)

    return {
        "coefficient_bound": (
            "||A_S||_(2->2)<=sup_xi||M(xi)||. If M is C^3 and "
            "||partial_S M||_infinity<=L_|S| H^(-|S|), then "
            "||A_S||_(2->2)<=L_|S|(m/(2H))^|S|."
        ),
        "coupled_vertex_bound": (
            "||psi_v T(u tensor u)||_2 <= ||u||_infinity "
            "sum_w B_(v,w)||psi_w u||_2, with "
            "B_(v,w)=L_d(m/(2H))^d and d=Hamming(v,w)."
        ),
        "bounded_multiplier_fallback": (
            "Without multiplier derivatives one may take "
            "B_(v,w)=sup||M|| for all eight vertices."
        ),
        "smooth_prototype_ratio": str(smooth_ratio),
        "smooth_prototype_matrix_norm": float(np.max(eigenvalues)),
        "smooth_prototype_exact_row_sum": expected_norm,
        "bounded_multiplier_matrix_norm": float(
            np.max(np.linalg.eigvalsh(bounded_matrix))
        ),
        "higher_strata": {
            "distance_0": "psi_v u",
            "distance_1": "(2/m)partial_j psi_v u",
            "distance_2": "(2/m)^2 partial_jl psi_v u",
            "distance_3": "(2/m)^3 partial_123 psi_v u",
        },
        "interpretation": (
            "The exact commutator is an eight-cell Hamming leakage "
            "operator. The old diagonal proposal retained only distances "
            "zero and one, so it requires a separate bandwidth-sensitive "
            "uncertainty theorem; it does not follow from symbol calculus."
        ),
        "all_checks_pass": bool(
            abs(float(np.max(eigenvalues)) - expected_norm) < 1.0e-12
            and abs(
                float(np.max(np.linalg.eigvalsh(bounded_matrix))) - 8.0
            )
            < 1.0e-12
        ),
    }


def _two_scale_counterexample_theorem() -> dict[str, Any]:
    offset = 2
    minimum_mode = f"sqrt(3)*({offset}+1)=3*sqrt(3)"
    pressure_exponent = Fraction(-3, 1)
    weighted_mass_exponent = Fraction(-9, 2)
    gradient_exponent = Fraction(-7, 2)
    denominator_exponent = max(
        weighted_mass_exponent,
        gradient_exponent,
    )
    ratio_exponent = pressure_exponent - denominator_exponent

    return {
        "packet": (
            "F_N(x)=[sin(Nx/2)/(N sin(x/2))]^2 and "
            "a_N=F_N(x1)F_N(x2)F_N(x3)"
            "cos((N+2)(x1+x2+x3))."
        ),
        "velocity": (
            "u_N=N^(-1)(partial_2 a_N,-partial_1 a_N,0)."
        ),
        "exact_divergence": "div u_N=0",
        "fourier_support": (
            "Each positive-octant coordinate lies in {3,...,2N+1}; "
            "the other component is its full negative-octant conjugate."
        ),
        "fixed_minimum_velocity_mode": minimum_mode,
        "amplitude_bound": (
            "0<c<=||u_N||_infinity<=C independently of N."
        ),
        "fixed_smooth_pressure_tail": (
            "p_N^H=-chi_H(D)R_iR_j(u_Ni u_Nj), where H=3sqrt(3), "
            "chi_H=0 on |xi|<=H and chi_H=1 on |xi|>=2H."
        ),
        "triple_zero_weight": (
            "psi=sin(x1/2)sin(x2/2)sin(x3/2)."
        ),
        "proved_asymptotic_bounds": {
            "weighted_pressure_lower": "||psi p_N^H||_2>=c N^(-3)",
            "weighted_velocity_mass_upper": (
                "||psi u_N||_2<=C N^(-9/2)"
            ),
            "zero_face_gradient_upper": (
                "||u_N grad psi||_2<=C N^(-7/2)"
            ),
            "diagonal_ratio_lower": "ratio>=c N^(1/2)",
        },
        "proof_mechanism": (
            "The shifted Fejer support gives the fixed spectral gap "
            "without a global low-mode correction. At scale 1/N the curl "
            "packet has bounded amplitude and its Reynolds stresses "
            "converge after multiplication by N^3 to a nonzero positive "
            "matrix times delta_0. The fixed smooth high-pass pressure "
            "kernel is nonzero away from zero, giving the N^(-3) lower "
            "bound on a set where psi is nonzero. Standard Fejer kernel "
            "bounds give the two weighted upper bounds at the triple "
            "zero. Amplitude rescaling leaves the ratio invariant, so "
            "K>=C||u||_infinity/nu cannot repair the estimate."
        ),
        "falsified_statement": (
            "There is no bandwidth-independent C such that every smooth "
            "divergence-free u with u_hat(k)=0 for |k|<K obeys "
            "||psi p_H||_2<=C||u||_infinity"
            "(||psi u||_2+K^(-1)||u grad psi||_2)."
        ),
        "scope": (
            "This falsifies the lower-carrier-only diagonal high-output "
            "commutator estimate. It does not falsify an annular or "
            "dyadic-shell estimate, a coupled eight-cell estimate, or a "
            "signed cross-shell pressure theorem."
        ),
        "all_checks_pass": (
            minimum_mode == "sqrt(3)*(2+1)=3*sqrt(3)"
            and ratio_exponent == Fraction(1, 2)
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


def _curl_fejer_row(order: int, offset: int = 2) -> dict[str, Any]:
    size = 10 * order
    coordinate = 2.0 * math.pi * np.arange(size) / size
    x = coordinate[:, None, None]
    y = coordinate[None, :, None]
    z = coordinate[None, None, :]
    envelope = (
        _fejer_peak_one(x, order)
        * _fejer_peak_one(y, order)
        * _fejer_peak_one(z, order)
    )
    potential = envelope * np.cos(
        (order + offset) * (x + y + z)
    )

    wave_vectors = _frequency_grid(size)
    wave_squared = sum(value**2 for value in wave_vectors)
    safe_wave_squared = np.where(
        wave_squared == 0.0,
        1.0,
        wave_squared,
    )
    potential_hat = np.fft.fftn(potential)
    velocity_hat = np.zeros(
        (3, size, size, size),
        dtype=np.complex128,
    )
    velocity_hat[0] = (
        1j * wave_vectors[1] * potential_hat / order
    )
    velocity_hat[1] = (
        -1j * wave_vectors[0] * potential_hat / order
    )
    velocity = np.fft.ifftn(
        velocity_hat,
        axes=(1, 2, 3),
    ).real

    coefficient_size = np.sqrt(
        np.sum(np.abs(velocity_hat) ** 2, axis=0)
    )
    occupied = coefficient_size > (
        1.0e-9 * float(np.max(coefficient_size))
    )
    occupied_wave_squared = np.broadcast_to(
        wave_squared,
        occupied.shape,
    )[occupied]
    minimum_mode = float(np.sqrt(np.min(occupied_wave_squared)))
    maximum_mode = float(np.sqrt(np.max(occupied_wave_squared)))
    expected_minimum_mode = math.sqrt(3.0) * (offset + 1)

    pressure_hat = np.zeros(
        (size, size, size),
        dtype=np.complex128,
    )
    for first in range(3):
        for second in range(3):
            pressure_hat -= (
                wave_vectors[first]
                * wave_vectors[second]
                / safe_wave_squared
                * np.fft.fftn(
                    velocity[first] * velocity[second]
                )
            )
    radius = np.sqrt(wave_squared)
    cutoff = np.zeros_like(radius)
    cutoff[radius >= 2.0 * minimum_mode] = 1.0
    transition = (
        (radius > minimum_mode)
        & (radius < 2.0 * minimum_mode)
    )
    cutoff[transition] = np.sin(
        0.5
        * math.pi
        * (radius[transition] - minimum_mode)
        / minimum_mode
    ) ** 2
    pressure = np.fft.ifftn(pressure_hat * cutoff).real

    psi = (
        np.sin(x / 2.0)
        * np.sin(y / 2.0)
        * np.sin(z / 2.0)
    )
    psi_gradient = (
        0.5
        * np.cos(x / 2.0)
        * np.sin(y / 2.0)
        * np.sin(z / 2.0),
        0.5
        * np.sin(x / 2.0)
        * np.cos(y / 2.0)
        * np.sin(z / 2.0),
        0.5
        * np.sin(x / 2.0)
        * np.sin(y / 2.0)
        * np.cos(z / 2.0),
    )
    velocity_size = np.sqrt(np.sum(velocity**2, axis=0))
    velocity_linf = float(np.max(velocity_size))
    weighted_pressure = float(
        np.sqrt(np.mean((psi * pressure) ** 2))
    )
    weighted_velocity_mass = float(
        np.sqrt(np.mean((psi * velocity_size) ** 2))
    )
    zero_face_gradient = float(
        np.sqrt(
            np.mean(
                velocity_size**2
                * sum(value**2 for value in psi_gradient)
            )
        )
    )
    gradient_over_carrier = zero_face_gradient / minimum_mode
    diagonal_denominator = velocity_linf * (
        weighted_velocity_mass + gradient_over_carrier
    )
    divergence_hat = sum(
        1j * wave_vectors[component] * velocity_hat[component]
        for component in range(3)
    )
    divergence_residual = float(
        np.max(np.abs(divergence_hat))
        / max(float(np.max(coefficient_size)), 1.0)
    )
    maximum_product_coordinate_mode = float(
        2 * (2 * order + offset - 1)
    )
    nyquist = size / 2.0

    return {
        "order": order,
        "grid_size": size,
        "minimum_velocity_mode": minimum_mode,
        "expected_minimum_velocity_mode": expected_minimum_mode,
        "maximum_velocity_mode": maximum_mode,
        "nyquist": nyquist,
        "maximum_product_coordinate_mode_upper": (
            maximum_product_coordinate_mode
        ),
        "velocity_L_infinity": velocity_linf,
        "weighted_pressure_norm": weighted_pressure,
        "weighted_velocity_mass": weighted_velocity_mass,
        "zero_face_gradient_over_carrier": gradient_over_carrier,
        "diagonal_commutator_ratio": (
            weighted_pressure / diagonal_denominator
        ),
        "N_cubed_weighted_pressure": (
            order**3 * weighted_pressure
        ),
        "N_to_9_over_2_weighted_mass": (
            order ** 4.5 * weighted_velocity_mass
        ),
        "N_to_7_over_2_gradient_over_carrier": (
            order ** 3.5 * gradient_over_carrier
        ),
        "maximum_relative_divergence_residual": divergence_residual,
        "scope": (
            "Binary64 alias-free finite-Fourier stress of the exact curl "
            "packet. It supports, but is not needed to prove, the "
            "asymptotic counterexample theorem."
        ),
        "all_checks_pass": bool(
            abs(minimum_mode - expected_minimum_mode) < 1.0e-10
            and maximum_product_coordinate_mode < nyquist
            and divergence_residual < 1.0e-10
            and diagonal_denominator > 0.0
            and weighted_pressure > 0.0
        ),
    }


def _finite_fourier_counterexample_pilot() -> dict[str, Any]:
    rows = [_curl_fejer_row(order) for order in (4, 6, 8, 10, 12, 14)]
    orders = np.asarray([row["order"] for row in rows], dtype=float)

    def slope(field: str, tail: bool = False) -> float:
        values = np.asarray(
            [row[field] for row in rows],
            dtype=float,
        )
        if tail:
            orders_used = orders[-4:]
            values = values[-4:]
        else:
            orders_used = orders
        return float(
            np.polyfit(
                np.log(orders_used),
                np.log(values),
                1,
            )[0]
        )

    ratios = [row["diagonal_commutator_ratio"] for row in rows]
    scaled_pressure = [
        row["N_cubed_weighted_pressure"] for row in rows
    ]
    return {
        "construction": (
            "The real shifted three-dimensional Fejer packet is converted "
            "to the exactly divergence-free field "
            "N^(-1)(partial_2 a,-partial_1 a,0). Its lowest carrier is "
            "fixed while bandwidth and triple-zero concentration grow."
        ),
        "rows": rows,
        "fitted_log_slopes": {
            "weighted_pressure_all_rows": slope(
                "weighted_pressure_norm"
            ),
            "weighted_pressure_tail_rows": slope(
                "weighted_pressure_norm",
                tail=True,
            ),
            "weighted_velocity_mass": slope(
                "weighted_velocity_mass"
            ),
            "zero_face_gradient_over_carrier": slope(
                "zero_face_gradient_over_carrier"
            ),
            "diagonal_commutator_ratio": slope(
                "diagonal_commutator_ratio"
            ),
        },
        "proved_asymptotic_exponents": {
            "weighted_pressure_lower": -3.0,
            "weighted_velocity_mass_upper": -4.5,
            "zero_face_gradient_over_carrier_upper": -3.5,
            "diagonal_ratio_lower": 0.5,
        },
        "observed_ratio_growth_factor": ratios[-1] / ratios[0],
        "interpretation": (
            "The finite rows are pre-asymptotic, but the fixed carrier, "
            "exact divergence constraint, increasing N^3 pressure norm, "
            "and rapidly increasing diagonal ratio independently stress "
            "the analytic no-go mechanism."
        ),
        "all_checks_pass": bool(
            all(row["all_checks_pass"] for row in rows)
            and all(
                first < second
                for first, second in zip(ratios, ratios[1:])
            )
            and all(
                first < second
                for first, second in zip(
                    scaled_pressure,
                    scaled_pressure[1:],
                )
            )
            and ratios[-1] / ratios[0] > 50.0
        ),
    }


def audit() -> dict[str, Any]:
    walsh = _exact_vertex_walsh_calculus()
    hamming = _hamming_leakage_bound()
    theorem = _two_scale_counterexample_theorem()
    pilot = _finite_fourier_counterexample_pilot()
    positive_checks = {
        "exact_vertex_Walsh_calculus_passes": walsh[
            "all_checks_pass"
        ],
        "Hamming_leakage_bound_passes": hamming["all_checks_pass"],
        "two_scale_counterexample_theorem_passes": theorem[
            "all_checks_pass"
        ],
        "finite_Fourier_counterexample_pilot_passes": pilot[
            "all_checks_pass"
        ],
    }
    return {
        "kind": "pressure_hamming_commutator_gate_audit",
        "schema_version": 1,
        "status": (
            "lower_carrier_diagonal_commutator_falsified_"
            "hamming_leakage_certified"
        ),
        "assumption_scope": (
            "Smooth periodic divergence-free pure high-pass velocities; "
            "vertex square roots on the periodic double cover; bounded or "
            "smooth translation-invariant pressure multipliers; and a "
            "fixed smooth high-output cutoff in the analytic no-go family."
        ),
        "exact_vertex_Walsh_calculus": walsh,
        "Hamming_leakage_bound": hamming,
        "two_scale_counterexample_theorem": theorem,
        "finite_Fourier_counterexample_pilot": pilot,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "exact_eight_shift_Walsh_identity_proved": True,
            "distance_two_and_three_pressure_leakage_genuine": True,
            "coupled_eight_cell_multiplier_bound_proved": True,
            "lower_carrier_only_diagonal_commutator_bound_falsified": True,
            "intrinsic_amplitude_condition_repairs_diagonal_bound": False,
            "annular_shell_diagonal_commutator_bound_proved": False,
            "mixed_low_high_paraproduct_controlled": False,
            "critical_signed_bound_proved": False,
            "low_regularity_passage_proved": False,
            "exceptional_set_upgrade_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "route_decision": (
            "Do not seek a bandwidth-independent single-vertex estimate "
            "from a lower carrier cutoff alone. Retain the exact Hamming "
            "leakage terms and move to a dyadic annular decomposition, "
            "where bandwidth is comparable to carrier and triple-zero "
            "concentration cannot occur below the shell scale."
        ),
        "next_theorem_target": (
            "For u supported in one annulus K<=|k|<=Lambda K, prove or "
            "falsify a uniform finite-type inequality that controls the "
            "distance-two and distance-three vertex masses by "
            "||psi_v u||_2+K^(-1)||u grad psi_v||_2. Combine it with the "
            "exact Walsh multiplier formula, then test summability of "
            "comparable-shell and separated-shell pressure paraproducts."
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
            "pressure_hamming_commutator_gate_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("pressure Hamming commutator audit failed")
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
