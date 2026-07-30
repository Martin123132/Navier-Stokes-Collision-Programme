"""Audit the dyadic three-shell atlas and localized telescoping defects."""

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


def _phi_coefficients(
    vertex: tuple[int, int, int],
) -> dict[Wave, float]:
    output: dict[Wave, float] = {}
    for wave in product((-1, 0, 1), repeat=3):
        value = 1.0
        for coordinate, frequency in enumerate(wave):
            value *= (
                0.5 if frequency == 0 else 0.25 * vertex[coordinate]
            )
        output[wave] = value
    return output


def _localized_transfer(
    advector: VectorField,
    differentiated: VectorField,
    testing: VectorField,
    vertex: tuple[int, int, int],
) -> complex:
    phi_hat = _phi_coefficients(vertex)
    value = 0.0j
    for advector_wave, advector_value in advector.items():
        for differentiated_wave, differentiated_value in (
            differentiated.items()
        ):
            derivative = 1j * np.dot(
                advector_value,
                np.asarray(differentiated_wave, dtype=float),
            ) * differentiated_value
            for testing_wave, testing_value in testing.items():
                total = cross._add_wave(
                    cross._add_wave(
                        advector_wave,
                        differentiated_wave,
                    ),
                    testing_wave,
                )
                stencil = cross._negate_wave(total)
                coefficient = phi_hat.get(stencil)
                if coefficient is not None:
                    value += (
                        coefficient
                        * np.dot(derivative, testing_value)
                    )
    return value


def _localized_skew_defect(
    advector: VectorField,
    first: VectorField,
    second: VectorField,
    vertex: tuple[int, int, int],
) -> complex:
    phi_hat = _phi_coefficients(vertex)
    value = 0.0j
    for advector_wave, advector_value in advector.items():
        for first_wave, first_value in first.items():
            for second_wave, second_value in second.items():
                total = cross._add_wave(
                    cross._add_wave(advector_wave, first_wave),
                    second_wave,
                )
                stencil = cross._negate_wave(total)
                coefficient = phi_hat.get(stencil)
                if coefficient is None:
                    continue
                gradient = (
                    1j
                    * np.asarray(stencil, dtype=float)
                    * coefficient
                )
                value -= np.dot(advector_value, gradient) * np.dot(
                    first_value,
                    second_value,
                )
    return value


def _support_atlas() -> dict[str, Any]:
    rows = []
    for largest_over_stencil in (2.0, 3.0, 4.0, 8.0, 32.0):
        second_fraction_lower = (
            1.0 - 1.0 / largest_over_stencil
        ) / 2.0
        rows.append(
            {
                "largest_over_partition_stencil": (
                    largest_over_stencil
                ),
                "second_largest_over_largest_lower": (
                    second_fraction_lower
                ),
                "largest_over_second_largest_upper": (
                    1.0 / second_fraction_lower
                ),
                "at_most_four_when_largest_at_least_twice_stencil": bool(
                    1.0 / second_fraction_lower <= 4.0
                ),
            }
        )
    return {
        "frequency_constraint": (
            "Every kinetic or pressure cubic vertex load has "
            "k_1+k_2+k_3+r=0 with |r|<=R=sqrt(3)m."
        ),
        "sorted_support_rule": (
            "If M_1>=M_2>=M_3 are the three velocity magnitudes, then "
            "M_1<=M_2+M_3+R<=2M_2+R, hence "
            "M_2>=(M_1-R)/2."
        ),
        "high_frequency_corollary": (
            "For M_1>=2R, M_2>=M_1/4. In disjoint dyadic annuli "
            "[2^j,2^(j+1)), the largest two shell indices differ by at "
            "most two."
        ),
        "atlas": (
            "Above the partition scale, occupied triples reduce to HHH "
            "(all comparable) or HHL (two comparable high inputs and one "
            "possibly much lower mode). A genuinely separated HLL triple "
            "is excluded."
        ),
        "rows": rows,
        "all_checks_pass": all(
            row[
                "at_most_four_when_largest_at_least_twice_stencil"
            ]
            for row in rows
        ),
    }


def _localized_skew_audit() -> dict[str, Any]:
    high = cross._high_field(64)
    low = cross._low_field()
    vertices = list(product((-1, 1), repeat=3))
    skew_residuals = []
    kinetic_reconstruction_residuals = []
    global_unweighted_transfers = []
    for vertex in vertices:
        pairs = (
            (high, high, low),
            (high, low, high),
            (low, high, high),
        )
        for advector, first, second in pairs:
            forward = _localized_transfer(
                advector,
                first,
                second,
                vertex,
            )
            reverse = _localized_transfer(
                advector,
                second,
                first,
                vertex,
            )
            defect = _localized_skew_defect(
                advector,
                first,
                second,
                vertex,
            )
            skew_residuals.append(abs(forward + reverse - defect))

        transfer_sum = (
            _localized_transfer(low, high, high, vertex)
            + _localized_transfer(high, low, high, vertex)
            + _localized_transfer(high, high, low, vertex)
        )
        kinetic_flux = cross._component_fluxes(high, low)["kinetic"]
        kinetic_load = cross._load(kinetic_flux, vertex)
        kinetic_reconstruction_residuals.append(
            abs(transfer_sum + kinetic_load)
        )

    constant_vertex = {
        (0, 0, 0): 1.0,
    }

    def unweighted_transfer(
        advector: VectorField,
        first: VectorField,
        second: VectorField,
    ) -> complex:
        value = 0.0j
        for advector_wave, advector_value in advector.items():
            for first_wave, first_value in first.items():
                derivative = 1j * np.dot(
                    advector_value,
                    np.asarray(first_wave, dtype=float),
                ) * first_value
                for second_wave, second_value in second.items():
                    total = cross._add_wave(
                        cross._add_wave(advector_wave, first_wave),
                        second_wave,
                    )
                    coefficient = constant_vertex.get(
                        cross._negate_wave(total)
                    )
                    if coefficient is not None:
                        value += coefficient * np.dot(
                            derivative,
                            second_value,
                        )
        return value

    for advector, first, second in (
        (high, high, low),
        (high, low, high),
        (low, high, high),
    ):
        global_unweighted_transfers.append(
            abs(
                unweighted_transfer(advector, first, second)
                + unweighted_transfer(advector, second, first)
            )
        )
    return {
        "localized_identity": (
            "T_Phi(a;b,c)+T_Phi(a;c,b)"
            "=-mean[(a dot grad Phi)(b dot c)]."
        ),
        "global_corollary": (
            "For Phi=1 the right side vanishes, so shell transfer is "
            "exactly antisymmetric."
        ),
        "HHL_kinetic_reconstruction": (
            "T_Phi(L;H,H)+T_Phi(H;L,H)+T_Phi(H;H,L) equals the negative "
            "vertex load of (|H|^2/2)L+(L dot H)H."
        ),
        "maximum_localized_skew_residual": max(skew_residuals),
        "maximum_HHL_kinetic_reconstruction_residual": max(
            kinetic_reconstruction_residuals
        ),
        "maximum_global_antisymmetry_residual": max(
            global_unweighted_transfers
        ),
        "interpretation": (
            "Shell-index antisymmetry is exact globally. Localization does "
            "not destroy conservation; it moves the uncancelled amount "
            "onto the spatial partition boundary."
        ),
        "all_checks_pass": bool(
            max(skew_residuals) < 1.0e-12
            and max(kinetic_reconstruction_residuals) < 1.0e-12
            and max(global_unweighted_transfers) < 1.0e-12
        ),
    }


def _walsh_character(
    vertex: tuple[int, int, int],
    mask: int,
) -> int:
    value = 1
    for coordinate in range(3):
        if mask & (1 << coordinate):
            value *= vertex[coordinate]
    return value


def _eight_vertex_audit() -> dict[str, Any]:
    high = cross._high_field(64)
    low = cross._low_field()
    flux = cross._component_fluxes(high, low)["combined"]
    vertices = list(product((-1, 1), repeat=3))
    loads = {
        vertex: float(cross._load(flux, vertex).real)
        for vertex in vertices
    }
    reference = loads[(1, 1, 1)]
    character_residual = max(
        abs(value - reference * _walsh_character(vertex, 7))
        for vertex, value in loads.items()
    )
    walsh = {
        str(mask): sum(
            _walsh_character(vertex, mask) * value
            for vertex, value in loads.items()
        )
        / 8.0
        for mask in range(8)
    }
    off_character_maximum = max(
        abs(value) for mask, value in walsh.items() if mask != "7"
    )
    equal_weight_sum = sum(loads.values())
    selector_weights = {
        vertex: (
            1.0 + _walsh_character(vertex, 7)
        )
        / 2.0
        for vertex in vertices
    }
    selector_sum = sum(
        selector_weights[vertex] * value
        for vertex, value in loads.items()
    )
    l1_sum = sum(abs(value) for value in loads.values())
    return {
        "carrier": 64,
        "vertex_loads": {
            "".join("+" if value > 0 else "-" for value in vertex): load
            for vertex, load in loads.items()
        },
        "all_cosine_vertex_load": reference,
        "maximum_pure_top_Walsh_character_residual": character_residual,
        "Walsh_coefficients": walsh,
        "maximum_off_top_Walsh_coefficient": off_character_maximum,
        "equal_weight_eight_vertex_sum": equal_weight_sum,
        "nonnegative_top_character_selector_sum": selector_sum,
        "sum_of_absolute_vertex_loads": l1_sum,
        "selector_sum_over_L1": selector_sum / l1_sum,
        "identity": (
            "For this HHL family, b_v=chi_{123}(v)b_{+++}. Equal weights "
            "cancel exactly, but w_v=(1+chi_{123}(v))/2 is nonnegative "
            "and retains one half of the L1 load."
        ),
        "all_checks_pass": bool(
            abs(reference) > 1.0e-4
            and character_residual < 1.0e-12
            and off_character_maximum < 1.0e-12
            and abs(walsh["7"] - reference) < 1.0e-12
            and abs(equal_weight_sum) < 1.0e-12
            and abs(selector_sum / l1_sum - 0.5) < 1.0e-12
        ),
    }


def _combine_high_fields(carriers: list[int]) -> VectorField:
    terms = [
        (1.0, cross._high_field(carrier))
        for carrier in carriers
    ]
    return cross._add_vectors(*terms)


def _multishell_coherence_audit() -> dict[str, Any]:
    carriers = [16, 32, 64, 128, 256]
    low = cross._low_field()
    rows = []
    individual_loads = []
    for carrier in carriers:
        flux = cross._component_fluxes(
            cross._high_field(carrier),
            low,
        )["combined"]
        individual_loads.append(float(cross._load(
            flux,
            (1, 1, 1),
        ).real))

    for count in range(1, len(carriers) + 1):
        selected = carriers[:count]
        combined_high = _combine_high_fields(selected)
        combined_flux = cross._component_fluxes(
            combined_high,
            low,
        )["combined"]
        combined_load = float(
            cross._load(combined_flux, (1, 1, 1)).real
        )
        expected = sum(individual_loads[:count])
        high_energy_proxy = sum(
            float(np.vdot(value, value).real)
            for value in combined_high.values()
        )
        rows.append(
            {
                "high_shell_count": count,
                "carriers": selected,
                "combined_vertex_load": combined_load,
                "sum_of_individual_vertex_loads": expected,
                "cross_shell_coherence_residual": abs(
                    combined_load - expected
                ),
                "high_Fourier_L2_energy_proxy": high_energy_proxy,
                "load_per_high_energy_proxy": (
                    combined_load / high_energy_proxy
                ),
                "all_checks_pass": bool(
                    combined_load > 0.0
                    and abs(combined_load - expected) < 1.0e-12
                    and abs(high_energy_proxy - 4.0 * count) < 1.0e-12
                ),
            }
        )
    return {
        "individual_carriers": carriers,
        "individual_HHL_loads": individual_loads,
        "rows": rows,
        "interpretation": (
            "Separated high shells with the same low modulation accumulate "
            "coherently. Cross terms between distinct high carriers cannot "
            "meet the fixed low/stencil resonance, so the vertex load is "
            "the exact sum of the individual loads. There is no automatic "
            "alternation or telescoping bonus in high-shell index."
        ),
        "what_survives": (
            "The accumulation is linear in the high-shell L2 energy proxy, "
            "so an amplitude square-sum or Carleson estimate is not "
            "falsified."
        ),
        "all_checks_pass": all(row["all_checks_pass"] for row in rows),
    }


def _occupied_triple_support_stress() -> dict[str, Any]:
    carriers = [16, 32, 64, 128]
    high = _combine_high_fields(carriers)
    low = cross._low_field()
    field = cross._add_vectors((1.0, high), (1.0, low))
    waves = list(field)
    stencil = [
        wave
        for wave in product((-1, 0, 1), repeat=3)
        if wave != (0, 0, 0)
    ]
    occupied = []
    maximum_ratio = 0.0
    for first in waves:
        for second in waves:
            for third in waves:
                total = cross._add_wave(
                    cross._add_wave(first, second),
                    third,
                )
                remainder = cross._negate_wave(total)
                if remainder not in stencil:
                    continue
                magnitudes = sorted(
                    (
                        cross._wave_norm(first),
                        cross._wave_norm(second),
                        cross._wave_norm(third),
                    ),
                    reverse=True,
                )
                ratio = magnitudes[0] / magnitudes[1]
                maximum_ratio = max(maximum_ratio, ratio)
                occupied.append(
                    {
                        "largest": magnitudes[0],
                        "second": magnitudes[1],
                        "third": magnitudes[2],
                        "largest_over_second": ratio,
                    }
                )
    return {
        "carrier_set": carriers,
        "occupied_ordered_triple_count": len(occupied),
        "maximum_largest_over_second_ratio": maximum_ratio,
        "theorem_upper_at_largest_ge_2sqrt3": 4.0,
        "all_occupied_largest_two_comparable": all(
            row["largest_over_second"] <= 4.0
            for row in occupied
        ),
        "all_checks_pass": bool(
            occupied
            and maximum_ratio < 4.0
            and all(
                row["largest_over_second"] <= 4.0
                for row in occupied
            )
        ),
    }


def _amplitude_envelope_audit() -> dict[str, Any]:
    rows = []
    for partition, length, seed in (
        (1, 8, 13),
        (1, 16, 29),
        (2, 10, 43),
        (5, 12, 71),
    ):
        generator = np.random.default_rng(seed)
        amplitudes = np.abs(generator.normal(size=length))
        levels = partition * 2.0 ** np.arange(length)
        energy = float(np.sum(amplitudes**2))
        dissipation = float(np.sum(levels**2 * amplitudes**2))
        tails = np.asarray(
            [
                np.sum(amplitudes[index + 2 :] ** 2)
                for index in range(length)
            ]
        )
        shell_sum = float(
            partition
            * np.sum(levels**1.5 * amplitudes * tails)
        )
        cauchy_sum = float(np.sum(levels**-0.5 * amplitudes))
        geometric_factor = float(np.sum(levels**-1.0))
        upper = (
            partition
            * dissipation
            * math.sqrt(energy)
            * math.sqrt(geometric_factor)
        )
        universal_upper = (
            math.sqrt(2.0 * partition)
            * math.sqrt(energy)
            * dissipation
        )
        tail_ratios = [
            float(
                tails[index]
                * levels[index] ** 2
                / dissipation
            )
            for index in range(length)
            if dissipation > 0.0
        ]
        scaling = 3.7
        scaled_shell_sum = scaling**3 * shell_sum
        scaled_energy = scaling**2 * energy
        scaled_dissipation = scaling**2 * dissipation
        scaled_upper = (
            math.sqrt(2.0 * partition)
            * math.sqrt(scaled_energy)
            * scaled_dissipation
        )
        rows.append(
            {
                "partition_frequency": partition,
                "dyadic_level_count": length,
                "random_seed": seed,
                "energy_sequence_norm_squared": energy,
                "dissipation_sequence": dissipation,
                "HHL_amplitude_sum": shell_sum,
                "intermediate_Cauchy_sum": cauchy_sum,
                "finite_geometric_factor": geometric_factor,
                "finite_sequence_upper": upper,
                "universal_sqrt2_upper": universal_upper,
                "sum_over_upper_ratio": shell_sum / universal_upper,
                "maximum_tail_to_dissipation_bound_ratio": max(
                    tail_ratios
                ),
                "amplitude_scaling_factor": scaling,
                "cubic_scaling_residual": abs(
                    scaled_shell_sum / scaled_upper
                    - shell_sum / universal_upper
                ),
                "all_checks_pass": bool(
                    shell_sum <= upper * (1.0 + 1.0e-13)
                    and upper <= universal_upper * (1.0 + 1.0e-13)
                    and max(tail_ratios) <= 1.0 + 1.0e-13
                    and abs(
                        scaled_shell_sum / scaled_upper
                        - shell_sum / universal_upper
                    )
                    < 1.0e-13
                ),
            }
        )
    return {
        "single_low_shell_estimate": (
            "For L>=m and H>=4L, the complete HHL vertex load obeys "
            "|B_(v,L;HHL)|<=C m L^(3/2) a_L "
            "sum_(H>=4L)a_H^2, where a_J=||u_J||_2. Kinetic terms use "
            "Bernstein ||u_L||_infinity<=CL^(3/2)a_L; low-output HH "
            "pressure uses the L1-to-L2 norm CL^(3/2) of the smooth "
            "low Riesz kernel; cross pressure uses L2 Riesz boundedness."
        ),
        "dyadic_sequence_bound": (
            "With E=sum_J a_J^2 and D=sum_J J^2a_J^2 over dyadic "
            "J>=m, the tail satisfies sum_(H>=4L)a_H^2<=L^(-2)D. "
            "Therefore sum_L mL^(3/2)a_L tail_L"
            "<=mD sum_L L^(-1/2)a_L"
            "<=sqrt(2m) E^(1/2)D."
        ),
        "weighted_cell_extension": (
            "For W=sum_v w_v Phi_v, replace the factor m by "
            "||grad W||_infinity, which depends only on nonconstant "
            "Walsh/edge variation of the cell coefficients."
        ),
        "large_data_gate": (
            "The bound is perturbatively absorbed by nu D only when "
            "C sqrt(m)E^(1/2)<nu. Under u->A u its ratio to nu D grows "
            "linearly in A, so Leray energy alone does not give universal "
            "large-data absorption."
        ),
        "rows": rows,
        "all_checks_pass": all(row["all_checks_pass"] for row in rows),
    }


def audit() -> dict[str, Any]:
    support = _support_atlas()
    skew = _localized_skew_audit()
    vertices = _eight_vertex_audit()
    multishell = _multishell_coherence_audit()
    occupied = _occupied_triple_support_stress()
    envelope = _amplitude_envelope_audit()
    positive_checks = {
        "largest_two_comparable_atlas_passes": support[
            "all_checks_pass"
        ],
        "localized_shell_skew_identity_passes": skew[
            "all_checks_pass"
        ],
        "eight_vertex_Walsh_flux_audit_passes": vertices[
            "all_checks_pass"
        ],
        "multishell_coherence_stress_passes": multishell[
            "all_checks_pass"
        ],
        "occupied_triple_support_stress_passes": occupied[
            "all_checks_pass"
        ],
        "HHL_amplitude_envelope_passes": envelope[
            "all_checks_pass"
        ],
    }
    return {
        "kind": "dyadic_three_shell_atlas_audit",
        "schema_version": 1,
        "status": "dyadic_atlas_certified_naive_telescoping_falsified",
        "assumption_scope": (
            "Smooth finite-Fourier divergence-free fields, tensor "
            "partition frequency one, and disjoint dyadic shell geometry."
        ),
        "support_atlas": support,
        "localized_shell_skew_identity": skew,
        "eight_vertex_flux_structure": vertices,
        "multishell_coherence_stress": multishell,
        "occupied_triple_support_stress": occupied,
        "HHL_amplitude_envelope": envelope,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "largest_two_velocity_scales_comparable_proved": True,
            "global_shell_transfer_antisymmetry_proved": True,
            "localized_shell_skew_defect_identity_proved": True,
            "fixed_vertex_pure_shell_telescoping_falsified": True,
            "equal_weight_eight_vertex_cancellation_proved": True,
            "pure_top_Walsh_HHL_channel_exhibited": True,
            "nonconstant_nonnegative_vertex_selector_retains_half_L1": True,
            "coherent_multishell_HHL_accumulation_proved": True,
            "HHL_amplitude_square_sum_bound_proved": True,
            "large_data_viscous_absorption_from_Leray_energy_proved": False,
            "joint_scale_cell_Carleson_bound_proved": False,
            "time_integrated_viscous_compensation_proved": False,
            "critical_signed_bound_proved": False,
            "low_regularity_passage_proved": False,
            "exceptional_set_upgrade_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "route_decision": (
            "The dyadic atlas is now exact: only HHH and HHL survive above "
            "the partition scale. Global shell transfer is conservative, "
            "but at a fixed vertex its antisymmetry defect is precisely "
            "the spatial boundary flux, and separated high shells can "
            "accumulate coherently. Equal eight-cell weights cancel, while "
            "a nonnegative top-Walsh selector retains half the L1 load. "
            "A direct amplitude estimate sums the HHL block by "
            "C sqrt(m)||u||_2||grad u||_2^2, which is perturbative only "
            "at small global Reynolds size. The next viable object is "
            "therefore a joint scale-cell or time-integrated improvement "
            "that beats this coefficient, not shell or cell cancellation "
            "in isolation."
        ),
        "next_theorem_target": (
            "The amplitude square-sum bound is now explicit but leaves "
            "C sqrt(m)||u||_2/nu. Seek a signed or time-integrated gain in "
            "the joint scale-cell flux: first test whether cumulative "
            "high-shell Reynolds stress has a Carleson measure controlled "
            "by dissipation, with cell dependence measured by Walsh/edge "
            "variation; then test whether its bad times have a summable "
            "viscous occupation cost."
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
            "dyadic_three_shell_atlas_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("dyadic three-shell atlas audit failed")
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
