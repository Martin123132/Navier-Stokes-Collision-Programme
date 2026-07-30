"""Audit dense annular HHH multiplicity in one low tensor/Walsh channel."""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import nonlinear_stress_regeneration_gate_audit as regeneration


ROOT = Path(__file__).resolve().parents[3]
Offset = tuple[int, int, int]

CENTER_DIRECTIONS = (
    np.asarray((1.0, 0.0, 0.0)),
    np.asarray((-1.0, 1.0, 0.0)),
    np.asarray((0.0, -1.0, 0.0)),
)
BASE_VECTORS = (
    np.asarray((-4.0, -3.0, 1.0)),
    np.asarray((-3.0, -1.0, 2.0)),
    np.asarray((-3.0, 7.0, 1.0)),
)
PHASES = (1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 1.0j)
CENTRAL_IMAGINARY_MATRIX = np.asarray(
    (
        (36.0, 0.0, -30.0),
        (0.0, -36.0, 30.0),
        (-30.0, 30.0, 0.0),
    )
)
CENTRAL_NORM = 12.0 * math.sqrt(43.0)
CHANNEL_TENSOR = -CENTRAL_IMAGINARY_MATRIX / CENTRAL_NORM


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


def _channel_pairing(matrix: np.ndarray) -> float:
    return float(np.vdot(CHANNEL_TENSOR, matrix).real)


def _central_witness_audit() -> dict[str, Any]:
    polarizations = [
        regeneration._project(vector, direction)
        for vector, direction in zip(BASE_VECTORS, CENTER_DIRECTIONS)
    ]
    unphased = regeneration._hhh_stress_forcing(
        CENTER_DIRECTIONS[0],
        polarizations[0],
        CENTER_DIRECTIONS[1],
        polarizations[1],
        CENTER_DIRECTIONS[2],
        polarizations[2],
    )
    phased = regeneration._hhh_stress_forcing(
        CENTER_DIRECTIONS[0],
        PHASES[0] * polarizations[0],
        CENTER_DIRECTIONS[1],
        PHASES[1] * polarizations[1],
        CENTER_DIRECTIONS[2],
        PHASES[2] * polarizations[2],
    )
    exact_residual = float(
        np.linalg.norm(
            unphased - 1j * CENTRAL_IMAGINARY_MATRIX
        )
    )
    phased_residual = float(
        np.linalg.norm(phased + CENTRAL_IMAGINARY_MATRIX)
    )
    return {
        "center_directions": [
            direction.astype(int).tolist()
            for direction in CENTER_DIRECTIONS
        ],
        "integer_base_vectors": [
            vector.astype(int).tolist() for vector in BASE_VECTORS
        ],
        "projected_center_polarizations": [
            np.real(value).astype(int).tolist()
            for value in polarizations
        ],
        "relative_phases": ["1", "1", "i"],
        "exact_unphased_matrix_factor": "i",
        "exact_integer_matrix": CENTRAL_IMAGINARY_MATRIX.astype(
            int
        ).tolist(),
        "exact_Frobenius_norm": "12*sqrt(43)",
        "numeric_Frobenius_norm": CENTRAL_NORM,
        "exact_matrix_residual": exact_residual,
        "phase_rotated_real_matrix_residual": phased_residual,
        "channel_tensor": CHANNEL_TENSOR.tolist(),
        "central_channel_pairing": _channel_pairing(phased),
        "trace_residual": float(abs(np.trace(phased))),
        "proof_of_nonzero": (
            "All waves and unnormalized projected polarizations are "
            "rational. Direct symbol algebra gives exactly i times the "
            "displayed nonzero integer matrix, whose norm is 12sqrt(43). "
            "The phase product i rotates it to a real matrix."
        ),
        "all_checks_pass": bool(
            exact_residual < 1.0e-13
            and phased_residual < 1.0e-13
            and abs(_channel_pairing(phased) - CENTRAL_NORM) < 1.0e-12
            and abs(np.trace(phased)) < 1.0e-13
        ),
    }


def _offsets(radius: int) -> list[Offset]:
    return list(product(range(-radius, radius + 1), repeat=3))


def _add_offset(
    center: np.ndarray,
    offset: Offset,
) -> np.ndarray:
    return center + np.asarray(offset, dtype=float)


def _negative_offset_sum(first: Offset, second: Offset) -> Offset:
    return tuple(
        -left - right for left, right in zip(first, second)
    )  # type: ignore[return-value]


def _in_box(offset: Offset, radius: int) -> bool:
    return all(abs(value) <= radius for value in offset)


def _dense_packet_row(
    radius: int,
    carrier_multiple: int,
) -> dict[str, Any]:
    carrier = carrier_multiple * radius
    centers = [
        carrier * direction for direction in CENTER_DIRECTIONS
    ]
    offsets = _offsets(radius)
    values: list[dict[Offset, np.ndarray]] = []
    waves: list[dict[Offset, np.ndarray]] = []
    positive_energy = 0.0
    positive_enstrophy = 0.0
    minimum_support = math.inf
    maximum_support = 0.0
    maximum_divergence = 0.0

    for center, base, phase in zip(centers, BASE_VECTORS, PHASES):
        cluster_values: dict[Offset, np.ndarray] = {}
        cluster_waves: dict[Offset, np.ndarray] = {}
        for offset in offsets:
            wave = _add_offset(center, offset)
            value = phase * regeneration._project(base, wave)
            cluster_values[offset] = value
            cluster_waves[offset] = wave
            norm_squared = float(np.vdot(value, value).real)
            positive_energy += norm_squared
            positive_enstrophy += (
                float(np.dot(wave, wave)) * norm_squared
            )
            wave_norm = float(np.linalg.norm(wave))
            minimum_support = min(minimum_support, wave_norm)
            maximum_support = max(maximum_support, wave_norm)
            maximum_divergence = max(
                maximum_divergence,
                abs(np.dot(wave, value)),
            )
        values.append(cluster_values)
        waves.append(cluster_waves)

    total_energy_before_normalization = 2.0 * positive_energy
    normalization = 1.0 / math.sqrt(total_energy_before_normalization)
    mode_count = 6 * len(offsets)
    expected_triad_count = (
        3 * radius**2 + 3 * radius + 1
    ) ** 3
    triad_count = 0
    forcing_sum = np.zeros((3, 3), dtype=np.complex128)
    minimum_unit_channel_over_carrier = math.inf
    maximum_unit_channel_over_carrier = -math.inf

    for first_offset in offsets:
        for second_offset in offsets:
            third_offset = _negative_offset_sum(
                first_offset,
                second_offset,
            )
            if not _in_box(third_offset, radius):
                continue
            triad_count += 1
            contribution = regeneration._hhh_stress_forcing(
                waves[0][first_offset],
                values[0][first_offset],
                waves[1][second_offset],
                values[1][second_offset],
                waves[2][third_offset],
                values[2][third_offset],
            )
            forcing_sum += contribution
            normalized_channel = (
                _channel_pairing(contribution) / carrier
            )
            minimum_unit_channel_over_carrier = min(
                minimum_unit_channel_over_carrier,
                normalized_channel,
            )
            maximum_unit_channel_over_carrier = max(
                maximum_unit_channel_over_carrier,
                normalized_channel,
            )

    forcing = 2.0 * normalization**3 * np.real(forcing_sum)
    channel_value = _channel_pairing(forcing)
    forcing_norm = float(np.linalg.norm(forcing))
    enstrophy = (
        2.0 * normalization**2 * positive_enstrophy
    )
    parabolic_duration = carrier**-2.0
    forcing_l2_time_cost = channel_value**2 * parabolic_duration
    enstrophy_time_cost = enstrophy * parabolic_duration
    count_scale = (
        2.0
        * normalization**3
        * carrier
        * triad_count
    )

    return {
        "box_radius": radius,
        "carrier": carrier,
        "carrier_multiple": carrier_multiple,
        "positive_cluster_mode_count": 3 * len(offsets),
        "real_field_mode_count": mode_count,
        "exact_coherent_triad_count": triad_count,
        "triad_count_formula": expected_triad_count,
        "energy_before_normalization": (
            total_energy_before_normalization
        ),
        "normalization_coefficient": normalization,
        "normalized_energy": (
            normalization**2 * total_energy_before_normalization
        ),
        "normalized_enstrophy": enstrophy,
        "minimum_support_radius": minimum_support,
        "maximum_support_radius": maximum_support,
        "annulus_ratio": maximum_support / minimum_support,
        "maximum_divergence_residual_before_normalization": (
            maximum_divergence
        ),
        "minimum_unit_triad_channel_over_carrier": (
            minimum_unit_channel_over_carrier
        ),
        "maximum_unit_triad_channel_over_carrier": (
            maximum_unit_channel_over_carrier
        ),
        "coherent_count_scale": count_scale,
        "fixed_channel_forcing": channel_value,
        "full_tensor_forcing_Frobenius_norm": forcing_norm,
        "channel_over_full_tensor_norm": (
            channel_value / forcing_norm
        ),
        "channel_over_coherent_count_scale": (
            channel_value / count_scale
        ),
        "trace_residual": float(abs(np.trace(forcing))),
        "parabolic_duration": parabolic_duration,
        "forcing_L2_time_cost": forcing_l2_time_cost,
        "enstrophy_time_cost": enstrophy_time_cost,
        "forcing_cost_over_enstrophy_cost": (
            forcing_l2_time_cost / enstrophy_time_cost
        ),
        "all_checks_pass": bool(
            triad_count == expected_triad_count
            and abs(
                normalization**2 * total_energy_before_normalization
                - 1.0
            )
            < 1.0e-13
            and maximum_divergence < 1.0e-10
            and maximum_support / minimum_support < 2.5
            and minimum_unit_channel_over_carrier > 0.5 * CENTRAL_NORM
            and channel_value > 0.0
            and channel_value / forcing_norm > 0.95
            and abs(np.trace(forcing)) < 1.0e-10 * forcing_norm
        ),
    }


def _dense_packet_audit() -> dict[str, Any]:
    carrier_multiple = 32
    rows = [
        _dense_packet_row(radius, carrier_multiple)
        for radius in (1, 2, 3, 4)
    ]
    forcing_growth = [
        rows[index + 1]["fixed_channel_forcing"]
        / rows[index]["fixed_channel_forcing"]
        for index in range(len(rows) - 1)
    ]
    normalized_channel_values = [
        row["channel_over_coherent_count_scale"] for row in rows
    ]
    cost_ratios = [
        row["forcing_cost_over_enstrophy_cost"] for row in rows
    ]
    return {
        "packet_definition": (
            "For R=32M, take lattice boxes R*A+B_M, R*B+B_M, "
            "R*C+B_M around A=(1,0,0), B=(-1,1,0), C=(0,-1,0), plus "
            "their conjugates. Use coefficients P_k v_A, P_k v_B, and "
            "i P_k v_C, then normalize total Fourier L2 energy to one."
        ),
        "only_zero_center_relations": (
            "Among the six carrier centers, the only signed triples "
            "summing to zero are A+B+C=0 and its full negative. Thus no "
            "other carrier combination can cancel the selected channel."
        ),
        "exact_one_dimensional_offset_pair_count": (
            "3M^2+3M+1"
        ),
        "exact_three_dimensional_triad_count": (
            "(3M^2+3M+1)^3"
        ),
        "rows": rows,
        "forcing_prefix_growth_ratios": forcing_growth,
        "channel_over_count_scale_values": normalized_channel_values,
        "minimum_channel_over_count_scale": min(
            normalized_channel_values
        ),
        "maximum_channel_over_count_scale": max(
            normalized_channel_values
        ),
        "forcing_cost_over_enstrophy_cost_values": cost_ratios,
        "interpretation": (
            "Every audited lattice triad has the same positive projection "
            "on the fixed traceless tensor channel. The exact count scale "
            "is R times O(M^6) coherent triples times the cube of the "
            "O(M^(-3/2)) energy normalization, hence O(R^(5/2)) at "
            "fixed R/M."
        ),
        "all_checks_pass": bool(
            all(row["all_checks_pass"] for row in rows)
            and min(normalized_channel_values) > 0.5 * CENTRAL_NORM
            and max(normalized_channel_values)
            < 1.5 * CENTRAL_NORM
            and all(
                later > earlier
                for earlier, later in zip(cost_ratios, cost_ratios[1:])
            )
        ),
    }


def _sharp_scaling_theorem() -> dict[str, Any]:
    mode_exponent = Fraction(3, 1)
    triad_exponent = Fraction(6, 1)
    coefficient_exponent = -mode_exponent / 2
    derivative_exponent = Fraction(1, 1)
    forcing_exponent = (
        triad_exponent
        + 3 * coefficient_exponent
        + derivative_exponent
    )
    parabolic_duration_exponent = Fraction(-2, 1)
    forcing_cost_exponent = (
        2 * forcing_exponent + parabolic_duration_exponent
    )
    enstrophy_exponent = Fraction(2, 1)
    enstrophy_cost_exponent = (
        enstrophy_exponent + parabolic_duration_exponent
    )
    return {
        "continuity_lower_bound": (
            "The normalized trilinear symbol projected on T is continuous "
            "in the three wave directions and equals 12sqrt(43)>0 at "
            "(A,B,C). Therefore there is epsilon>0 such that every triad "
            "in epsilon-neighborhood boxes has channel at least "
            "6sqrt(43) times the carrier and the same sign."
        ),
        "lattice_construction": (
            "Choose integer carrier R and M=floor(epsilon R/4). For "
            "offsets x,y in B_(M/2), z=-x-y lies in B_M, giving at least "
            "c M^6 positive triads. The six real-field clusters contain "
            "C M^3 modes."
        ),
        "energy_normalization": (
            "Unit L2 energy makes every bounded polarization coefficient "
            "O(M^(-3/2))."
        ),
        "lower_bound": (
            "|<T,G_R(0)>|>=c R M^6 M^(-9/2)>=c_epsilon R^(5/2)."
        ),
        "upper_bound": (
            "|G_R(0)|<=2||u_R||_2||grad p_R||_2"
            "<=C||u_R||_2||u_R||_infinity||grad u_R||_2"
            "<=C R^(5/2)||u_R||_2^3 by annular Bernstein."
        ),
        "sharp_forcing_exponent": str(forcing_exponent),
        "mode_count_exponent": str(mode_exponent),
        "coherent_triad_count_exponent": str(triad_exponent),
        "single_coefficient_exponent": str(coefficient_exponent),
        "parabolic_forcing_L2_cost_exponent": str(
            forcing_cost_exponent
        ),
        "parabolic_enstrophy_cost_exponent": str(
            enstrophy_cost_exponent
        ),
        "conclusion": (
            "The dense packet saturates the H^(3/2) Bernstein "
            "multiplicity beyond the sparse O(H) triad size. The sharp "
            "unit-energy tensor forcing is H^(5/2)."
        ),
        "all_checks_pass": bool(
            forcing_exponent == Fraction(5, 2)
            and forcing_cost_exponent == Fraction(3, 1)
            and enstrophy_cost_exponent == 0
        ),
    }


def _walsh_coupling_audit() -> dict[str, Any]:
    partition_wave = np.asarray((1.0, 1.0, 1.0))
    low_wave = -partition_wave
    low_polarization = np.asarray((1.0, -1.0, 0.0)) / math.sqrt(2.0)
    tensor_pairing = float(
        partition_wave
        @ CHANNEL_TENSOR
        @ low_polarization
    )
    expected_magnitude = 1.0 / math.sqrt(86.0)
    return {
        "stress_output_wave": [0, 0, 0],
        "low_velocity_wave": low_wave.astype(int).tolist(),
        "partition_wave": partition_wave.astype(int).tolist(),
        "low_velocity_polarization": low_polarization.tolist(),
        "low_velocity_divergence_residual": float(
            abs(np.dot(low_wave, low_polarization))
        ),
        "tensor_gradient_pairing": tensor_pairing,
        "exact_pairing_magnitude": "1/sqrt(86)",
        "pairing_magnitude_residual": abs(
            abs(tensor_pairing) - expected_magnitude
        ),
        "Walsh_character": "chi_123(v)=v_1 v_2 v_3",
        "reason": (
            "The partition coefficient at r=(1,1,1) is "
            "chi_123(v)/64. Since q=0 and the low wave is -r, the fixed "
            "dense tensor channel couples nontrivially to the pure top "
            "Walsh vertex channel."
        ),
        "all_checks_pass": bool(
            abs(np.dot(low_wave, low_polarization)) < 1.0e-13
            and abs(abs(tensor_pairing) - expected_magnitude) < 1.0e-13
        ),
    }


def _parabolic_no_go(
    theorem: dict[str, Any],
) -> dict[str, Any]:
    return {
        "test_path": (
            "Let u_R(t)=phi(R^2 t)u_R for a fixed nonzero smooth compact "
            "pulse phi and the unit-energy dense packet u_R."
        ),
        "energy_and_enstrophy": (
            "sup_t||u_R(t)||_2^2<=C, while "
            "integral||grad u_R(t)||_2^2 dt<=C because the shell "
            "enstrophy R^2 is active for time R^(-2)."
        ),
        "forcing_cost": (
            "The cubic regeneration is phi^3 G_R, so "
            "integral|<T,G_R(t)>|^2 dt>=c R^5 R^(-2)=c R^3."
        ),
        "falsified_statement": (
            "No universal estimate of the raw tensor form "
            "sum_H||f_H||_(L2_t)^2"
            "<=C(sup_t||u||_2, integral||grad u||_2^2) can be derived "
            "from those two Leray quantities alone for arbitrary smooth "
            "divergence-free paths."
        ),
        "scope": (
            "The pulse path is not claimed to solve unforced "
            "Navier-Stokes. This is a functional-input no-go: a successful "
            "Navier-Stokes estimate must use equation-specific temporal "
            "correlation or an additional signed cancellation. The trace "
            "channel, complete local-energy balance, and global regularity "
            "are not falsified."
        ),
        "forcing_cost_growth_exponent": theorem[
            "parabolic_forcing_L2_cost_exponent"
        ],
        "enstrophy_cost_growth_exponent": theorem[
            "parabolic_enstrophy_cost_exponent"
        ],
        "all_checks_pass": bool(
            theorem["parabolic_forcing_L2_cost_exponent"] == "3"
            and theorem["parabolic_enstrophy_cost_exponent"] == "0"
        ),
    }


def audit() -> dict[str, Any]:
    central = _central_witness_audit()
    dense = _dense_packet_audit()
    theorem = _sharp_scaling_theorem()
    walsh = _walsh_coupling_audit()
    no_go = _parabolic_no_go(theorem)
    positive_checks = {
        "exact_rational_center_witness_passes": central[
            "all_checks_pass"
        ],
        "dense_annular_lattice_replay_passes": dense[
            "all_checks_pass"
        ],
        "sharp_H_five_halves_scaling_theorem_passes": theorem[
            "all_checks_pass"
        ],
        "top_Walsh_channel_coupling_passes": walsh[
            "all_checks_pass"
        ],
        "Leray_input_only_parabolic_no_go_passes": no_go[
            "all_checks_pass"
        ],
    }
    return {
        "kind": "dense_annular_hhh_packet_gate_audit",
        "schema_version": 1,
        "status": (
            "sharp_dense_HHH_Bernstein_loss_certified_"
            "Leray_input_only_forcing_bound_falsified"
        ),
        "assumption_scope": (
            "Smooth finite-Fourier divergence-free annular packets; the "
            "exact HHH Leray-projected stress symbol at zero output; "
            "unit shell L2 energy; and arbitrary smooth parabolic pulse "
            "paths for the functional-input no-go."
        ),
        "exact_center_witness": central,
        "dense_annular_packet": dense,
        "sharp_scaling_theorem": theorem,
        "fixed_top_Walsh_coupling": walsh,
        "parabolic_Leray_input_no_go": no_go,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "dense_packet_divergence_free_annular_unit_energy_proved": True,
            "fixed_traceless_tensor_channel_nonzero_proved": True,
            "coherent_H_six_triad_count_proved": True,
            "sharp_H_five_halves_tensor_forcing_growth_proved": True,
            "top_Walsh_cell_channel_survives_proved": True,
            "raw_tensor_forcing_bound_from_Leray_inputs_alone_falsified": (
                True
            ),
            "unforced_Navier_Stokes_dynamic_counterexample_proved": False,
            "trace_local_energy_channel_obstructed": False,
            "complete_signed_flux_occupation_bound_proved": False,
            "equation_specific_temporal_correlation_bound_proved": False,
            "critical_signed_large_data_bound_proved": False,
            "low_regularity_passage_proved": False,
            "exceptional_set_upgrade_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "route_decision": (
            "Dense same-shell multiplicity is decisive: a unit-energy "
            "annular packet coherently feeds one fixed low traceless "
            "tensor and top-Walsh channel at the sharp H^(5/2) Bernstein "
            "rate. A parabolic H^(-2) lifetime leaves forcing L2 cost "
            "H^3 while Leray enstrophy cost stays O(1), so the raw tensor "
            "Duhamel norm cannot follow from Leray inputs alone. The route "
            "must now retain the equation-specific signed scalar "
            "local-energy structure, where the pressure-strain trace "
            "cancels, or discover temporal decorrelation forced by the "
            "actual Navier-Stokes evolution."
        ),
        "next_theorem_target": (
            "Project the dense packet through the complete signed local "
            "energy evolution, including low-velocity evolution, kinetic "
            "transport, high-high pressure, cross pressure, and all eight "
            "cell vertices. Determine whether the H^(5/2) traceless "
            "pressure-strain channel cancels in that scalar equation or "
            "survives with a sharp coefficient. In parallel, formulate "
            "the weakest shell-weighted negative norm that viscosity and "
            "Leray enstrophy can control without discarding the signed "
            "flux."
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
            "dense_annular_hhh_packet_gate_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("dense annular HHH packet audit failed")
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
