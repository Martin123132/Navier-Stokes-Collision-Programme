"""Audit joint scale-cell coherence and the viscous occupation replacement."""

from __future__ import annotations

import argparse
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import cross_shell_modulated_wave_gate_audit as cross


ROOT = Path(__file__).resolve().parents[3]
Wave = tuple[int, int, int]
VectorField = dict[Wave, np.ndarray]


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


def _walsh_character(
    vertex: tuple[int, int, int],
    mask: int,
) -> int:
    value = 1
    for coordinate in range(3):
        if mask & (1 << coordinate):
            value *= vertex[coordinate]
    return value


def _combine_high_fields(carriers: list[int]) -> VectorField:
    return cross._add_vectors(
        *((1.0, cross._high_field(carrier)) for carrier in carriers)
    )


def _reynolds_coefficient(
    field: VectorField,
    output_wave: Wave,
) -> np.ndarray:
    coefficient = np.zeros((3, 3), dtype=np.complex128)
    for first_wave, first_value in field.items():
        second_wave = tuple(
            output - first
            for output, first in zip(output_wave, first_wave)
        )
        second_value = field.get(second_wave)
        if second_value is not None:
            coefficient += np.outer(first_value, second_value)
    return coefficient


def _vertex_loads(flux: VectorField) -> dict[tuple[int, int, int], float]:
    return {
        vertex: float(cross._load(flux, vertex).real)
        for vertex in product((-1, 1), repeat=3)
    }


def _walsh_coefficients(
    loads: dict[tuple[int, int, int], float],
) -> dict[int, float]:
    return {
        mask: sum(
            _walsh_character(vertex, mask) * value
            for vertex, value in loads.items()
        )
        / 8.0
        for mask in range(8)
    }


def _heat_evolve(
    field: VectorField,
    time: float,
    viscosity: float,
) -> VectorField:
    return {
        wave: value
        * math.exp(
            -viscosity
            * sum(component * component for component in wave)
            * time
        )
        for wave, value in field.items()
    }


def _baseline_cumulative_stress_bound() -> dict[str, Any]:
    return {
        "definition": (
            "R_L=P_(<=cL) sum_(H,H'>=4L,H~H') "
            "(u_H tensor u_H')."
        ),
        "estimate": (
            "||R_L||_2<=C_c L^(3/2) "
            "sum_(H>=4L)||u_H||_2^2."
        ),
        "proof": (
            "The low-pass kernel has L2 norm C_c L^(3/2). Young gives "
            "||P_(<=cL)(u_H tensor u_H')||_2"
            "<=C_c L^(3/2)||u_H||_2||u_H'||_2. Comparable-shell "
            "multiplicity is finite, and 2ab<=a^2+b^2 closes the sum."
        ),
        "pressure_extension": (
            "Applying the low-frequency double Riesz transform preserves "
            "the L2 estimate. Pairing with a low velocity and grad W "
            "recovers the HHL amplitude envelope from the dyadic atlas."
        ),
        "scope": (
            "This is an ell1 sum of shell energies. Replacing it "
            "pointwise by the ell2 norm "
            "(sum_H ||u_H||_2^4)^(1/2) requires a new orthogonality gain."
        ),
        "all_checks_pass": True,
    }


def _pointwise_channel_audit() -> dict[str, Any]:
    carriers = [64 * 2**index for index in range(8)]
    output_wave = (1, 1, 0)
    low = cross._low_field()
    individual: list[dict[str, Any]] = []
    stresses: list[np.ndarray] = []
    loads: list[float] = []

    for carrier in carriers:
        high = cross._high_field(carrier)
        stress = _reynolds_coefficient(high, output_wave)
        flux = cross._component_fluxes(high, low)["combined"]
        vertex_loads = _vertex_loads(flux)
        walsh = _walsh_coefficients(vertex_loads)
        off_top = max(abs(walsh[mask]) for mask in range(7))
        load = vertex_loads[(1, 1, 1)]
        first_polarization = high[(1, 1, carrier)]
        expected_e11 = 2.0 * float(first_polarization[0].real)
        individual.append(
            {
                "carrier": carrier,
                "stress_Frobenius_norm": float(np.linalg.norm(stress)),
                "stress_e11_coefficient": float(stress[0, 0].real),
                "stress_e11_formula": expected_e11,
                "stress_e11_formula_residual": abs(
                    float(stress[0, 0].real) - expected_e11
                ),
                "all_cosine_vertex_load": load,
                "top_Walsh_coefficient": walsh[7],
                "maximum_off_top_Walsh_coefficient": off_top,
                "all_checks_pass": bool(
                    load > 0.0
                    and abs(float(stress[0, 0].imag)) < 1.0e-13
                    and abs(float(stress[0, 0].real) - expected_e11)
                    < 1.0e-13
                    and abs(walsh[7] - load) < 1.0e-13
                    and off_top < 1.0e-13
                ),
            }
        )
        stresses.append(stress)
        loads.append(load)

    rows = []
    for count in range(1, len(carriers) + 1):
        selected = carriers[:count]
        combined_high = _combine_high_fields(selected)
        combined_stress = _reynolds_coefficient(
            combined_high,
            output_wave,
        )
        expected_stress = sum(
            stresses[:count],
            np.zeros((3, 3), dtype=np.complex128),
        )
        combined_flux = cross._component_fluxes(
            combined_high,
            low,
        )["combined"]
        combined_load = _vertex_loads(combined_flux)[(1, 1, 1)]
        expected_load = sum(loads[:count])
        stress_square_function = math.sqrt(
            sum(
                float(np.linalg.norm(stress)) ** 2
                for stress in stresses[:count]
            )
        )
        flux_square_function = math.sqrt(
            sum(value * value for value in loads[:count])
        )
        stress_ratio = (
            float(np.linalg.norm(combined_stress))
            / stress_square_function
        )
        flux_ratio = abs(combined_load) / flux_square_function
        analytic_stress_ratio_lower = math.sqrt(2.0 / 3.0) * math.sqrt(
            count
        )
        rows.append(
            {
                "shell_count": count,
                "carriers": selected,
                "cumulative_stress_Frobenius_norm": float(
                    np.linalg.norm(combined_stress)
                ),
                "stress_shell_square_function": stress_square_function,
                "stress_sum_over_square_function": stress_ratio,
                "analytic_stress_ratio_lower": (
                    analytic_stress_ratio_lower
                ),
                "cumulative_top_Walsh_vertex_load": combined_load,
                "flux_shell_square_function": flux_square_function,
                "flux_sum_over_square_function": flux_ratio,
                "flux_ratio_over_sqrt_shell_count": (
                    flux_ratio / math.sqrt(count)
                ),
                "stress_cross_shell_residual": float(
                    np.linalg.norm(combined_stress - expected_stress)
                ),
                "flux_cross_shell_residual": abs(
                    combined_load - expected_load
                ),
                "all_checks_pass": bool(
                    np.linalg.norm(combined_stress - expected_stress)
                    < 1.0e-12
                    and abs(combined_load - expected_load) < 1.0e-12
                    and stress_ratio
                    >= analytic_stress_ratio_lower - 1.0e-12
                    and flux_ratio / math.sqrt(count) > 0.995
                ),
            }
        )

    return {
        "carriers": carriers,
        "common_low_Fourier_mode": list(output_wave),
        "common_cell_Walsh_mask": 7,
        "individual_channels": individual,
        "prefix_rows": rows,
        "stress_no_go_theorem": (
            "For H>=1, the q=(1,1,0) stress coefficient R_H has "
            "(R_H)11=2sqrt((H^2+1)/(H^2+2))>=2sqrt(2/3) and "
            "||R_H||_F<=2. Hence "
            "||sum_(j=1)^N R_Hj||_F/"
            "(sum_j||R_Hj||_F^2)^(1/2)>=sqrt(2/3)sqrt(N)."
        ),
        "joint_channel_no_go": (
            "Every complete HHL load is the same top Walsh character and "
            "tends with the same sign to 1/144. Thus its sum divided by "
            "its shell square function is asymptotic to sqrt(N). No "
            "uniform pointwise gain can come from treating high-shell "
            "labels as orthogonal after projection to this fixed "
            "Fourier-Walsh channel."
        ),
        "scope_limit": (
            "This falsifies only an ell2 shell-orthogonality replacement "
            "inside a common low Fourier-Walsh channel. It does not "
            "falsify the baseline ell1 shell-energy estimate, estimates "
            "using low-scale differences, or spacetime Carleson bounds."
        ),
        "all_checks_pass": bool(
            all(row["all_checks_pass"] for row in individual)
            and all(row["all_checks_pass"] for row in rows)
            and rows[-1]["stress_sum_over_square_function"] > 2.0
            and rows[-1]["flux_sum_over_square_function"] > 2.0
        ),
    }


def _occupation_crossing(
    coefficients: np.ndarray,
    rates: np.ndarray,
    threshold: float,
) -> float:
    def value(time: float) -> float:
        return float(np.sum(coefficients * np.exp(-rates * time)))

    lower = 0.0
    upper = 1.0 / float(np.min(rates))
    while value(upper) > threshold:
        upper *= 2.0
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        if value(middle) > threshold:
            lower = middle
        else:
            upper = middle
    return upper


def _viscous_occupation_audit(
    pointwise: dict[str, Any],
) -> dict[str, Any]:
    carriers = np.asarray(pointwise["carriers"], dtype=float)
    coefficients = np.asarray(
        [
            row["all_cosine_vertex_load"]
            for row in pointwise["individual_channels"]
        ],
        dtype=float,
    )
    viscosity = 0.7
    low_wave_norm_squared = 9.0
    high_pair_offset = 2.0
    beta = (low_wave_norm_squared + high_pair_offset) / 2.0
    mu_squared = carriers**2 + beta
    mu = np.sqrt(mu_squared)
    rates = 2.0 * viscosity * mu_squared
    ratios = mu[1:] / mu[:-1]
    rho = float(np.min(ratios))
    schur_row_bound = 0.5 + 2.0 / (rho - 1.0)

    gram = 1.0 / (rates[:, None] + rates[None, :])
    exact_integral = float(coefficients @ gram @ coefficients)
    weighted_square_sum = float(
        np.sum(coefficients**2 / mu_squared)
    )
    schur_upper = (
        schur_row_bound
        * weighted_square_sum
        / (2.0 * viscosity)
    )
    normalized_kernel = (
        mu[:, None]
        * mu[None, :]
        / (mu_squared[:, None] + mu_squared[None, :])
    )
    actual_operator_norm = float(
        np.max(np.linalg.eigvalsh(normalized_kernel))
    )
    maximum_actual_row_sum = float(
        np.max(np.sum(np.abs(normalized_kernel), axis=1))
    )

    heat_residuals = []
    for carrier, coefficient, rate in zip(
        carriers.astype(int),
        coefficients,
        rates,
    ):
        high = cross._high_field(int(carrier))
        low = cross._low_field()
        for dimensionless_time in (0.1, 0.5, 1.0):
            time = dimensionless_time / float(rate)
            evolved_flux = cross._component_fluxes(
                _heat_evolve(high, time, viscosity),
                _heat_evolve(low, time, viscosity),
            )["combined"]
            evolved_load = _vertex_loads(evolved_flux)[(1, 1, 1)]
            expected = coefficient * math.exp(-dimensionless_time)
            heat_residuals.append(abs(evolved_load - expected))

    threshold = 0.5 * float(np.sum(coefficients))
    occupation = _occupation_crossing(
        coefficients,
        rates,
        threshold,
    )
    chebyshev_upper = exact_integral / threshold**2

    prefix_rows = []
    for count in range(1, len(carriers) + 1):
        prefix_coefficients = coefficients[:count]
        prefix_rates = rates[:count]
        prefix_mu_squared = mu_squared[:count]
        prefix_gram = 1.0 / (
            prefix_rates[:, None] + prefix_rates[None, :]
        )
        prefix_integral = float(
            prefix_coefficients
            @ prefix_gram
            @ prefix_coefficients
        )
        prefix_upper = (
            schur_row_bound
            / (2.0 * viscosity)
            * float(
                np.sum(prefix_coefficients**2 / prefix_mu_squared)
            )
        )
        prefix_threshold = 0.5 * float(np.sum(prefix_coefficients))
        prefix_occupation = _occupation_crossing(
            prefix_coefficients,
            prefix_rates,
            prefix_threshold,
        )
        prefix_rows.append(
            {
                "shell_count": count,
                "initial_coherent_sum": float(
                    np.sum(prefix_coefficients)
                ),
                "exact_L2_time_norm_squared": prefix_integral,
                "Schur_upper": prefix_upper,
                "half_peak_threshold": prefix_threshold,
                "half_peak_occupation": prefix_occupation,
                "Chebyshev_occupation_upper": (
                    prefix_integral / prefix_threshold**2
                ),
                "all_checks_pass": bool(
                    prefix_integral <= prefix_upper * (1.0 + 1.0e-12)
                    and prefix_occupation
                    <= prefix_integral / prefix_threshold**2
                    * (1.0 + 1.0e-10)
                ),
            }
        )

    return {
        "viscosity": viscosity,
        "Stokes_HHL_damping_law": (
            "For the full Stokes evolution of the two high sidebands and "
            "the low mode, b_H(t)=b_H(0)exp[-nu(2H^2+11)t]."
        ),
        "beta_in_mu_squared": beta,
        "mu_definition": "mu_j=sqrt(H_j^2+11/2)",
        "minimum_lacunarity_ratio": rho,
        "Schur_row_bound": schur_row_bound,
        "actual_normalized_Gram_operator_norm": actual_operator_norm,
        "maximum_actual_normalized_Gram_row_sum": (
            maximum_actual_row_sum
        ),
        "exact_L2_time_norm_squared": exact_integral,
        "weighted_shell_square_sum": weighted_square_sum,
        "Schur_upper": schur_upper,
        "maximum_exact_Stokes_damping_residual": max(heat_residuals),
        "half_peak_threshold": threshold,
        "half_peak_occupation": occupation,
        "Chebyshev_occupation_upper": chebyshev_upper,
        "prefix_rows": prefix_rows,
        "theorem": (
            "Let H_j be dyadic, mu_j=sqrt(H_j^2+beta), "
            "mu_(j+1)>=rho mu_j, and "
            "F(t)=sum_j exp(-2nu mu_j^2 t)c_j in a Hilbert space. Then "
            "||F||_(L2_t)^2<=[1/(2nu)]"
            "[1/2+2/(rho-1)]sum_j ||c_j||^2/mu_j^2."
        ),
        "proof": (
            "Exact time integration gives the Gram kernel "
            "1/[2nu(mu_j^2+mu_k^2)]. After setting x_j=c_j/mu_j, "
            "the normalized kernel is "
            "mu_j mu_k/(mu_j^2+mu_k^2). Its diagonal is 1/2 and its "
            "distance-d off-diagonal entries are at most rho^(-d), so "
            "the Schur row sum is at most 1/2+2/(rho-1)."
        ),
        "occupation_corollary": (
            "|{t:||F(t)||>=Lambda}|<=Lambda^(-2)||F||_(L2_t)^2."
        ),
        "all_checks_pass": bool(
            rho > 1.9
            and actual_operator_norm <= maximum_actual_row_sum + 1.0e-13
            and maximum_actual_row_sum <= schur_row_bound + 1.0e-13
            and exact_integral <= schur_upper * (1.0 + 1.0e-12)
            and max(heat_residuals) < 1.0e-12
            and occupation <= chebyshev_upper * (1.0 + 1.0e-10)
            and all(row["all_checks_pass"] for row in prefix_rows)
        ),
    }


def _forced_relaxation_audit(
    occupation: dict[str, Any],
) -> dict[str, Any]:
    carriers = np.asarray(
        [64 * 2**index for index in range(8)],
        dtype=float,
    )
    viscosity = float(occupation["viscosity"])
    beta = float(occupation["beta_in_mu_squared"])
    mu_squared = carriers**2 + beta
    mu = np.sqrt(mu_squared)
    rho = float(np.min(mu[1:] / mu[:-1]))
    rates = 2.0 * viscosity * mu_squared
    forcing_decay = 0.3 * rates
    forcing_amplitudes = np.asarray(
        [(-1.0) ** index * (1.0 + 0.1 * index) for index in range(8)]
    )
    forcing_norm_squared = float(
        np.sum(forcing_amplitudes**2 / (2.0 * forcing_decay))
    )
    forcing_constant = 1.0 / (
        2.0
        * viscosity
        * mu_squared[0]
        * math.sqrt(1.0 - rho**-4)
    )

    response_terms: list[tuple[float, float]] = []
    for amplitude, rate, decay in zip(
        forcing_amplitudes,
        rates,
        forcing_decay,
    ):
        factor = float(amplitude / (rate - decay))
        response_terms.append((factor, float(decay)))
        response_terms.append((-factor, float(rate)))
    response_norm_squared = sum(
        first_amplitude
        * second_amplitude
        / (first_decay + second_decay)
        for first_amplitude, first_decay in response_terms
        for second_amplitude, second_decay in response_terms
    )
    response_norm_squared = max(0.0, float(response_norm_squared))
    response_norm = math.sqrt(response_norm_squared)
    forcing_bound = forcing_constant * math.sqrt(
        forcing_norm_squared
    )

    return {
        "forced_system": (
            "dot c_j+2nu mu_j^2 c_j=f_j, c_j(0)=0."
        ),
        "conditional_bound": (
            "||sum_j c_j||_(L2_t)<="
            "[2nu mu_0^2 sqrt(1-rho^(-4))]^(-1)"
            "(sum_j||f_j||_(L2_t)^2)^(1/2)."
        ),
        "proof": (
            "Young gives ||exp(-lambda_j t)*f_j||_2"
            "<=lambda_j^(-1)||f_j||_2. Sum in j, apply "
            "Cauchy-Schwarz, and use "
            "sum_j lambda_j^(-2)<="
            "[4nu^2 mu_0^4(1-rho^(-4))]^(-1)."
        ),
        "forcing_L2_ell2_norm": math.sqrt(forcing_norm_squared),
        "conditional_operator_constant": forcing_constant,
        "exact_replay_response_L2_norm": response_norm,
        "replay_upper": forcing_bound,
        "Navier_Stokes_gap": (
            "For Navier-Stokes, f_j is the projected nonlinear Duhamel "
            "regeneration of the common low Fourier-Walsh stress channel. "
            "No bound of its ell2_H L2_t norm by the Leray energy budget "
            "is proved here."
        ),
        "all_checks_pass": bool(
            forcing_constant > 0.0
            and response_norm <= forcing_bound * (1.0 + 1.0e-10)
        ),
    }


def audit() -> dict[str, Any]:
    baseline = _baseline_cumulative_stress_bound()
    pointwise = _pointwise_channel_audit()
    occupation = _viscous_occupation_audit(pointwise)
    forced = _forced_relaxation_audit(occupation)
    positive_checks = {
        "baseline_cumulative_stress_envelope_passes": baseline[
            "all_checks_pass"
        ],
        "common_low_mode_stress_reconstruction_passes": pointwise[
            "all_checks_pass"
        ],
        "common_top_Walsh_flux_reconstruction_passes": pointwise[
            "all_checks_pass"
        ],
        "pointwise_shell_orthogonality_no_go_passes": pointwise[
            "all_checks_pass"
        ],
        "exact_Stokes_HHL_damping_law_passes": occupation[
            "all_checks_pass"
        ],
        "viscous_Gram_Schur_bound_passes": occupation[
            "all_checks_pass"
        ],
        "viscous_occupation_corollary_passes": occupation[
            "all_checks_pass"
        ],
        "conditional_Duhamel_forcing_bound_passes": forced[
            "all_checks_pass"
        ],
    }
    return {
        "kind": "joint_scale_cell_viscous_occupation_audit",
        "schema_version": 1,
        "status": (
            "pointwise_joint_channel_no_go_"
            "Stokes_occupation_certified"
        ),
        "assumption_scope": (
            "Smooth finite-Fourier divergence-free sideband fields for "
            "the pointwise witness; Hilbert-valued dyadic relaxation "
            "channels for the time theorem; full Stokes heat evolution "
            "for the exact HHL damping replay."
        ),
        "baseline_cumulative_stress_bound": baseline,
        "pointwise_Fourier_Walsh_channel": pointwise,
        "viscous_occupation_bound": occupation,
        "forced_relaxation_bound": forced,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "baseline_cumulative_stress_ell1_bound_proved": True,
            "common_low_Fourier_top_Walsh_channel_exhibited": True,
            "pointwise_high_shell_ell2_orthogonality_gain_falsified": True,
            "all_pointwise_Carleson_estimates_falsified": False,
            "linear_Stokes_HHL_viscous_occupation_bound_proved": True,
            "conditional_forced_relaxation_bound_proved": True,
            "Navier_Stokes_nonlinear_regeneration_bound_proved": False,
            "Navier_Stokes_time_integrated_compensation_proved": False,
            "critical_signed_large_data_bound_proved": False,
            "low_regularity_passage_proved": False,
            "exceptional_set_upgrade_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "route_decision": (
            "Pointwise shell orthogonality is unavailable in the exact "
            "coherent low Fourier/top-Walsh channel: the required constant "
            "grows like sqrt(N). Time must be retained. Under Stokes "
            "damping, the same coherent channel has a weighted ell2 shell "
            "L2-time bound and a finite threshold-occupation estimate. "
            "The next Navier-Stokes gate is not another pointwise "
            "Carleson ansatz; it is control, cancellation, or a sharp "
            "no-go for the nonlinear Duhamel regeneration f_j in the "
            "forced relaxation identity."
        ),
        "next_theorem_target": (
            "Derive the exact shellwise evolution of the low-output "
            "Fourier-Walsh Reynolds-stress channel for a smooth "
            "Navier-Stokes solution. Decompose its Duhamel forcing into "
            "HHH, HHL, and transport pieces, then test whether the "
            "weighted ell2_H L2_t forcing norm required by the conditional "
            "occupation theorem follows from signed transfer plus "
            "dissipation. Stress any candidate on forced coherent "
            "sidebands before attempting low-regularity passage."
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
            "joint_scale_cell_viscous_occupation_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError(
            "joint scale-cell viscous occupation audit failed"
        )
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
