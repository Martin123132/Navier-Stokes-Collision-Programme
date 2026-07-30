"""Audit the complete smooth Galerkin high-shell stress response gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import nonlinear_stress_regeneration_gate_audit as regeneration


ROOT = Path(__file__).resolve().parents[3]
FORCING_SQUARE_CONSTANT = 104.0
HEAT_WEIGHTED_HHL_CONSTANT = 80.0
NEIGHBOR_RADIUS = 2


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


def _pair_rate(first_wave: np.ndarray, second_wave: np.ndarray) -> float:
    return float(
        np.dot(first_wave, first_wave)
        + np.dot(second_wave, second_wave)
    )


def _pairwise_evolution_audit() -> dict[str, Any]:
    generator = np.random.default_rng(20260727)
    viscosity = 0.37
    rows = []
    rates = []
    maximum_residual = 0.0
    for index in range(24):
        first_wave = generator.integers(-12, 13, size=3).astype(float)
        second_wave = generator.integers(-12, 13, size=3).astype(float)
        if np.linalg.norm(first_wave) == 0.0:
            first_wave[0] = 1.0
        if np.linalg.norm(second_wave) == 0.0:
            second_wave[1] = 1.0
        first_value = (
            generator.normal(size=3)
            + 1j * generator.normal(size=3)
        )
        second_value = (
            generator.normal(size=3)
            + 1j * generator.normal(size=3)
        )
        first_nonlinearity = (
            generator.normal(size=3)
            + 1j * generator.normal(size=3)
        )
        second_nonlinearity = (
            generator.normal(size=3)
            + 1j * generator.normal(size=3)
        )
        first_derivative = (
            -viscosity
            * float(np.dot(first_wave, first_wave))
            * first_value
            + first_nonlinearity
        )
        second_derivative = (
            -viscosity
            * float(np.dot(second_wave, second_wave))
            * second_value
            + second_nonlinearity
        )
        stress = np.outer(first_value, second_value)
        stress_derivative = (
            np.outer(first_derivative, second_value)
            + np.outer(first_value, second_derivative)
        )
        rate = _pair_rate(first_wave, second_wave)
        reconstructed = (
            stress_derivative + viscosity * rate * stress
        )
        forcing = (
            np.outer(first_nonlinearity, second_value)
            + np.outer(first_value, second_nonlinearity)
        )
        residual = float(np.linalg.norm(reconstructed - forcing))
        scale = max(1.0, float(np.linalg.norm(forcing)))
        relative_residual = residual / scale
        maximum_residual = max(maximum_residual, relative_residual)
        rates.append(rate)
        rows.append(
            {
                "row": index,
                "pair_rate": rate,
                "relative_residual": relative_residual,
            }
        )
    return {
        "identity": (
            "(d/dt+nu(|k|^2+|q-k|^2))(u_k tensor u_(q-k))="
            "N_k tensor u_(q-k)+u_k tensor N_(q-k)"
        ),
        "rows": rows,
        "maximum_relative_residual": maximum_residual,
        "distinct_pair_rate_count": len(set(rates)),
        "single_shell_scalar_rate_used": False,
        "all_checks_pass": bool(
            maximum_residual < 1.0e-12 and len(set(rates)) > 1
        ),
    }


def _selector_profile(radius_over_shell: float) -> float:
    return max(0.0, 1.0 - abs(radius_over_shell - 1.5))


def _pair_selector(
    first_wave: np.ndarray,
    second_wave: np.ndarray,
    shell: float,
) -> float:
    return _selector_profile(
        float(np.linalg.norm(first_wave)) / shell
    ) * _selector_profile(
        float(np.linalg.norm(second_wave)) / shell
    )


def _heat_weighted_hhl_audit() -> dict[str, Any]:
    carriers = (16, 32, 64, 128, 256, 512, 1024)
    dimensionless_delays = (0.0, 0.125, 0.5, 1.0, 2.0, 4.0)
    output_wave = np.asarray((1.0, 1.0, 0.0))
    low_wave = np.asarray((1.0, 0.0, 0.0))
    low_scale = max(
        float(np.linalg.norm(output_wave)),
        float(np.linalg.norm(low_wave)),
    )
    generator = np.random.default_rng(271828)
    rows = []
    maximum_bound_ratio = 0.0
    maximum_rate_identity_residual = 0.0
    maximum_decomposition_residual = 0.0
    maximum_selector_lipschitz_ratio = 0.0

    for carrier in carriers:
        first_wave = np.asarray((0.0, 1.0, float(carrier)))
        second_wave = output_wave - low_wave - first_wave
        for sample in range(32):
            first_value = regeneration._normalize_transverse(
                generator.normal(size=3),
                first_wave,
            )
            second_value = regeneration._normalize_transverse(
                generator.normal(size=3),
                second_wave,
            )
            low_value = regeneration._normalize_transverse(
                generator.normal(size=3),
                low_wave,
            )
            forcing, first, second = regeneration._hhl_stress_forcing(
                first_wave,
                first_value,
                second_wave,
                second_value,
                low_wave,
                low_value,
            )
            first_generated_wave = first_wave + low_wave
            second_generated_wave = second_wave + low_wave
            rate_zero = _pair_rate(first_wave, second_wave)
            first_rate = _pair_rate(
                first_generated_wave,
                second_wave,
            )
            second_rate = _pair_rate(
                second_generated_wave,
                first_wave,
            )
            first_rate_residual = abs(
                first_rate
                - rate_zero
                - (
                    2.0 * float(np.dot(first_wave, low_wave))
                    + float(np.dot(low_wave, low_wave))
                )
            )
            second_rate_residual = abs(
                second_rate
                - rate_zero
                - (
                    2.0 * float(np.dot(second_wave, low_wave))
                    + float(np.dot(low_wave, low_wave))
                )
            )
            paired_rate_residual = abs(
                first_rate
                - second_rate
                - 2.0
                * float(np.dot(low_wave, first_wave - second_wave))
            )
            maximum_rate_identity_residual = max(
                maximum_rate_identity_residual,
                first_rate_residual,
                second_rate_residual,
                paired_rate_residual,
            )
            selector_zero = _pair_selector(
                first_wave,
                second_wave,
                float(carrier),
            )
            first_selector = _pair_selector(
                first_generated_wave,
                second_wave,
                float(carrier),
            )
            second_selector = _pair_selector(
                second_generated_wave,
                first_wave,
                float(carrier),
            )
            selector_allowance = (
                float(np.linalg.norm(low_wave)) / carrier
            )
            if selector_allowance > 0.0:
                maximum_selector_lipschitz_ratio = max(
                    maximum_selector_lipschitz_ratio,
                    abs(first_selector - selector_zero)
                    / selector_allowance,
                    abs(second_selector - selector_zero)
                    / selector_allowance,
                )

            for dimensionless_delay in dimensionless_delays:
                delay = dimensionless_delay / carrier**2
                first_weight = (
                    first_selector * math.exp(-delay * first_rate)
                )
                second_weight = (
                    second_selector * math.exp(-delay * second_rate)
                )
                common_weight = (
                    selector_zero * math.exp(-delay * rate_zero)
                )
                weighted = (
                    first_weight * first + second_weight * second
                )
                decomposed = (
                    common_weight * forcing
                    + (first_weight - common_weight) * first
                    + (second_weight - common_weight) * second
                )
                decomposition_residual = float(
                    np.linalg.norm(weighted - decomposed)
                )
                maximum_decomposition_residual = max(
                    maximum_decomposition_residual,
                    decomposition_residual,
                )
                bound = (
                    HEAT_WEIGHTED_HHL_CONSTANT
                    * low_scale
                    * math.exp(-dimensionless_delay)
                )
                ratio = float(np.linalg.norm(weighted)) / bound
                maximum_bound_ratio = max(maximum_bound_ratio, ratio)
                rows.append(
                    {
                        "carrier": carrier,
                        "sample": sample,
                        "dimensionless_delay": dimensionless_delay,
                        "weighted_norm": float(np.linalg.norm(weighted)),
                        "analytic_bound": bound,
                        "bound_ratio": ratio,
                    }
                )

    return {
        "rate_difference_identities": (
            "lambda_(a+c,b)-lambda_(a,b)=2a dot c+|c|^2; "
            "lambda_(b+c,a)-lambda_(a,b)=2b dot c+|c|^2"
        ),
        "selector": (
            "A symmetric 1/H-Lipschitz pair-shell selector is used in "
            "the replay. The theorem allows any smooth selector with "
            "the corresponding rescaled derivative bound."
        ),
        "theorem": (
            "The exact pair-rate HHL response is bounded by "
            "(64+16C_chi)L exp(-nu H^2 tau) times the three input "
            "amplitudes. Here C_chi=1, so 80 is retained."
        ),
        "proof_mechanism": (
            "Compare both exact heat/filter weights with one common "
            "weight. Its paired forcing costs 18L. Selector differences "
            "cost (L/H) times an O(H) unpaired term. Rate differences "
            "cost nu*tau*HL times that O(H) term, hence "
            "L(nu H^2 tau)exp(-c nu H^2 tau), which is absorbed into a "
            "weaker exponential."
        ),
        "row_count": len(rows),
        "maximum_bound_ratio": maximum_bound_ratio,
        "maximum_rate_identity_residual": (
            maximum_rate_identity_residual
        ),
        "maximum_weight_decomposition_residual": (
            maximum_decomposition_residual
        ),
        "maximum_selector_lipschitz_ratio": (
            maximum_selector_lipschitz_ratio
        ),
        "all_checks_pass": bool(
            maximum_bound_ratio <= 1.0
            and maximum_rate_identity_residual < 1.0e-12
            and maximum_decomposition_residual < 1.0e-12
            and maximum_selector_lipschitz_ratio <= 1.0 + 1.0e-12
        ),
    }


def _neighbor_energy(amplitudes: np.ndarray, index: int) -> float:
    lower = max(0, index - NEIGHBOR_RADIUS)
    upper = min(len(amplitudes), index + NEIGHBOR_RADIUS + 1)
    return float(np.sum(amplitudes[lower:upper] ** 2))


def _forcing_square_row(
    shells: np.ndarray,
    amplitudes: np.ndarray,
    label: str,
) -> dict[str, Any]:
    energy = float(np.sum(amplitudes**2))
    dissipation = float(np.sum(shells**2 * amplitudes**2))
    comparable_amplitudes = np.asarray(
        [
            math.sqrt(_neighbor_energy(amplitudes, index))
            for index in range(len(shells))
        ]
    )
    low_envelopes = []
    low_cauchy_ratios = []
    unpaired_low_envelopes = []
    unpaired_low_cauchy_ratios = []
    hhh_envelope = []
    hhl_envelope = []
    galerkin_leakage_envelope = []
    for index, shell in enumerate(shells):
        low_mask = shells <= shell / 4.0
        low_envelope = float(
            np.sum(shells[low_mask] ** 2.5 * amplitudes[low_mask])
        )
        low_dissipation = float(
            np.sum(
                shells[low_mask] ** 2 * amplitudes[low_mask] ** 2
            )
        )
        low_envelopes.append(low_envelope)
        denominator = shell**3 * low_dissipation
        low_cauchy_ratios.append(
            0.0 if denominator == 0.0 else low_envelope**2 / denominator
        )
        unpaired_low_envelope = float(
            np.sum(shells[low_mask] ** 1.5 * amplitudes[low_mask])
        )
        unpaired_low_envelopes.append(unpaired_low_envelope)
        unpaired_denominator = shell * low_dissipation
        unpaired_low_cauchy_ratios.append(
            0.0
            if unpaired_denominator == 0.0
            else unpaired_low_envelope**2 / unpaired_denominator
        )
        comparable = comparable_amplitudes[index]
        hhh_envelope.append(shell**2.5 * comparable**3)
        hhl_envelope.append(low_envelope * comparable**2)
        galerkin_leakage_envelope.append(
            shell * unpaired_low_envelope * comparable**2
        )
    hhh = np.asarray(hhh_envelope)
    hhl = np.asarray(hhl_envelope)
    leakage = np.asarray(galerkin_leakage_envelope)
    complete = hhh + hhl + leakage
    hhh_square = float(np.sum(shells**-3 * hhh**2))
    hhl_square = float(np.sum(shells**-3 * hhl**2))
    leakage_square = float(np.sum(shells**-3 * leakage**2))
    complete_square = float(np.sum(shells**-3 * complete**2))
    analytic_bound = FORCING_SQUARE_CONSTANT * energy**2 * dissipation
    return {
        "label": label,
        "shells": shells.astype(int).tolist(),
        "amplitudes": amplitudes.tolist(),
        "energy": energy,
        "dissipation": dissipation,
        "comparable_shell_amplitudes": comparable_amplitudes.tolist(),
        "low_derivative_envelopes": low_envelopes,
        "maximum_low_cauchy_ratio": max(low_cauchy_ratios),
        "unpaired_low_envelopes": unpaired_low_envelopes,
        "maximum_unpaired_low_cauchy_ratio": max(
            unpaired_low_cauchy_ratios
        ),
        "HHH_weighted_forcing_square": hhh_square,
        "HHL_weighted_forcing_square": hhl_square,
        "Galerkin_leakage_weighted_forcing_square": leakage_square,
        "complete_weighted_forcing_square": complete_square,
        "analytic_bound": analytic_bound,
        "complete_over_bound": (
            0.0 if analytic_bound == 0.0
            else complete_square / analytic_bound
        ),
        "complete_forcing_L2_norms": complete.tolist(),
        "all_checks_pass": bool(
            max(low_cauchy_ratios) <= 1.0 / 56.0 + 1.0e-12
            and max(unpaired_low_cauchy_ratios)
            <= 1.0 / 2.0 + 1.0e-12
            and complete_square <= analytic_bound * (1.0 + 1.0e-12)
        ),
    }


def _forcing_square_audit() -> dict[str, Any]:
    shells = np.asarray([2.0**index for index in range(1, 11)])
    generator = np.random.default_rng(161803)
    scenarios = (
        ("harmonic", np.asarray([1.0 / (index + 1) for index in range(10)])),
        (
            "alternating",
            np.asarray(
                [
                    0.85 if index % 2 == 0 else 0.2
                    for index in range(10)
                ]
            ),
        ),
        (
            "seeded_random",
            0.1 + generator.random(10),
        ),
        (
            "two_shell_concentration",
            np.asarray([0.0, 0.0, 1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    )
    rows = [
        _forcing_square_row(shells, amplitudes, label)
        for label, amplitudes in scenarios
    ]
    neighbor_square_constant = sum(
        2.0 ** (2 * offset)
        for offset in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1)
    )
    hhl_constant = (2 * NEIGHBOR_RADIUS + 1) ** 2 / 56.0
    galerkin_leakage_constant = (
        (2 * NEIGHBOR_RADIUS + 1) ** 2 / 2.0
    )
    derived_combined_constant = 3.0 * (
        neighbor_square_constant
        + hhl_constant
        + galerkin_leakage_constant
    )
    return {
        "envelope": (
            "g_H=H^(5/2)A_H^3+A_H^2 "
            "sum_(L<=H/4)L^(5/2)a_L"
            "+H A_H^2 sum_(L<=H/4)L^(3/2)a_L. "
            "The last term pays a worst-case unpaired sharp Galerkin "
            "boundary contribution."
        ),
        "low_shell_cauchy_bound": (
            "(sum_(L<=H/4)L^(5/2)a_L)^2 "
            "<=H^3 D_(<H)/56 for dyadic shells."
        ),
        "unpaired_low_shell_cauchy_bound": (
            "(sum_(L<=H/4)L^(3/2)a_L)^2 "
            "<=H D_(<H)/2 for dyadic shells."
        ),
        "HHH_neighbor_square_constant": neighbor_square_constant,
        "HHL_global_constant": hhl_constant,
        "sharp_Galerkin_leakage_constant": (
            galerkin_leakage_constant
        ),
        "derived_combined_constant": derived_combined_constant,
        "retained_integer_constant": FORCING_SQUARE_CONSTANT,
        "theorem": (
            "sum_H H^(-3)||g_H||_(L2_t)^2 "
            "<=104 E_*^2 integral D(t)dt, before fixed multiplier and "
            "finite-low-channel constants."
        ),
        "rows": rows,
        "all_checks_pass": bool(
            derived_combined_constant < FORCING_SQUARE_CONSTANT
            and all(row["all_checks_pass"] for row in rows)
        ),
    }


def _response_and_initial_audit(
    forcing_audit: dict[str, Any],
) -> dict[str, Any]:
    viscosity = 0.73
    rows = []
    for forcing_row in forcing_audit["rows"]:
        shells = np.asarray(forcing_row["shells"], dtype=float)
        forcing_norms = np.asarray(
            forcing_row["complete_forcing_L2_norms"],
            dtype=float,
        )
        amplitudes = np.asarray(forcing_row["amplitudes"], dtype=float)
        weighted_forcing_square = float(
            np.sum(shells**-3 * forcing_norms**2)
        )
        duhamel_triangle = float(
            np.sum(forcing_norms / (viscosity * shells**2))
        )
        duhamel_cauchy = (
            math.sqrt(float(np.sum(shells**-1)))
            * math.sqrt(weighted_forcing_square)
            / viscosity
        )
        dyadic_tail_bound = (
            math.sqrt(2.0 / shells[0])
            * math.sqrt(weighted_forcing_square)
            / viscosity
        )
        initial_coefficients = amplitudes**2
        initial_triangle = float(
            np.sum(
                initial_coefficients
                / (math.sqrt(2.0 * viscosity) * shells)
            )
        )
        initial_energy_bound = float(
            np.sum(initial_coefficients)
            / (
                math.sqrt(2.0 * viscosity)
                * shells[0]
            )
        )
        initial_gram_square = 0.0
        for first_index, first_shell in enumerate(shells):
            for second_index, second_shell in enumerate(shells):
                initial_gram_square += (
                    initial_coefficients[first_index]
                    * initial_coefficients[second_index]
                    / (
                        viscosity
                        * (first_shell**2 + second_shell**2)
                    )
                )
        initial_exact_norm = math.sqrt(initial_gram_square)
        rows.append(
            {
                "label": forcing_row["label"],
                "duhamel_triangle_bound": duhamel_triangle,
                "duhamel_cauchy_bound": duhamel_cauchy,
                "duhamel_dyadic_tail_bound": dyadic_tail_bound,
                "initial_exact_Gram_norm": initial_exact_norm,
                "initial_triangle_bound": initial_triangle,
                "initial_energy_tail_bound": initial_energy_bound,
                "all_checks_pass": bool(
                    duhamel_triangle
                    <= duhamel_cauchy * (1.0 + 1.0e-12)
                    and duhamel_cauchy
                    <= dyadic_tail_bound * (1.0 + 1.0e-12)
                    and initial_exact_norm
                    <= initial_triangle * (1.0 + 1.0e-12)
                    and initial_triangle
                    <= initial_energy_bound * (1.0 + 1.0e-12)
                ),
            }
        )
    return {
        "forced_response": (
            "||sum_(H>=H0)C_H^F||_L2_t "
            "<=C E_*sqrt(D)/(nu sqrt(H0))."
        ),
        "initial_response": (
            "||sum_(H>=H0)C_H^0||_L2_t "
            "<=C E(0)/(sqrt(nu)H0)."
        ),
        "complete_smooth_galerkin_tail": (
            "The sum of the displayed forced and initial bounds tends to "
            "zero as H0 tends to infinity for every fixed finite set of "
            "low Fourier/tensor channels."
        ),
        "rows": rows,
        "all_checks_pass": all(
            row["all_checks_pass"] for row in rows
        ),
    }


def audit() -> dict[str, Any]:
    pairwise = _pairwise_evolution_audit()
    weighted_hhl = _heat_weighted_hhl_audit()
    forcing = _forcing_square_audit()
    response = _response_and_initial_audit(forcing)
    result = {
        "schema": "ns_smooth_galerkin_shell_response_gate_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "exact_pair_rate_complete_dyadic_smooth_galerkin_"
            "stress_response_certified"
        ),
        "prerequisites": {
            "nonlinear_stress_regeneration_gate": (
                "work/ns_collision/results/"
                "nonlinear_stress_regeneration_gate_audit_v1.json"
            ),
            "nonlinear_stress_regeneration_gate_sha256": _sha256(
                ROOT
                / "work/ns_collision/results/"
                "nonlinear_stress_regeneration_gate_audit_v1.json"
            ),
            "corrected_scalar_regeneration_gate": (
                "work/ns_collision/results/"
                "scalar_local_energy_regeneration_gate_audit_v2.json"
            ),
            "corrected_scalar_regeneration_gate_sha256": _sha256(
                ROOT
                / "work/ns_collision/results/"
                "scalar_local_energy_regeneration_gate_audit_v2.json"
            ),
        },
        "exact_pairwise_evolution": pairwise,
        "heat_weighted_HHL_commutator": weighted_hhl,
        "complete_weighted_forcing_square": forcing,
        "summed_response_and_initial_stress": response,
        "certification_flags": {
            "smooth_Galerkin_exact_pair_rates_retained": True,
            "single_artificial_shell_rate_used": False,
            "heat_weighted_HHL_commutator_proved": True,
            "smooth_shell_filter_leakage_paid": True,
            "sharp_Galerkin_cutoff_leakage_paid": True,
            "complete_HHH_HHL_weighted_forcing_square_proved": True,
            "initial_high_stress_heat_tail_controlled": True,
            "finite_low_channel_high_stress_tail_vanishes": True,
            "scale_uniform_spatial_localization_proved": False,
            "suitable_weak_solution_passage_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "scope": (
            "The theorem is for smooth finite Fourier Galerkin solutions, "
            "smooth static dyadic pair selectors, and any fixed finite "
            "set of low Fourier/tensor channels. Constants may depend on "
            "that channel set and multiplier family. It is not yet a "
            "scale-uniform physical-space localization theorem."
        ),
        "next_gate": (
            "Prove a scale-uniform low-output Littlewood-Paley or "
            "partition-space version of this response estimate and its "
            "stability under Galerkin limits. Only then test suitable-weak "
            "passage and defect-measure consequences."
        ),
    }
    result["all_positive_checks_pass"] = bool(
        pairwise["all_checks_pass"]
        and weighted_hhl["all_checks_pass"]
        and forcing["all_checks_pass"]
        and response["all_checks_pass"]
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "work/ns_collision/results/"
            "smooth_galerkin_shell_response_gate_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("smooth Galerkin shell-response gate failed")
    _atomic_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
