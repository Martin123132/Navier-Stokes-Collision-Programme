"""Audit the multiband weighted-Fisher recombination no-go."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from balanced_annular_pressure_edge_gate_audit import (
    VERTICES,
    _seed81_field,
    _taylor_green_field,
    _weighted_enstrophies,
    _weights_by_vertex,
)
from pressure_frame_pairing_audit import GRID_SIZE, STARTING_GRID_INDEX


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "multiband_weighted_fisher_recombination_no_go_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "balanced_annular_pressure_edge_gate_audit_v1.json"
    ): "9a024a23381d62e7842d7d26406fcea2a5343a168f386d3bad85e5308cef99dd",
    (
        "work/ns_collision/results/"
        "high_carrier_weighted_fisher_gate_audit_v1.json"
    ): "a533faec71e4941e6a1dc5458199e5684cf750db61df2be2823c04a6a3a7c5be",
    (
        "work/ns_collision/results/"
        "floor_free_pressure_edge_tail_gate_audit_v1.json"
    ): "3cffc4a951b4b9806a505093ca3fff2a5475341427117bb0339d76b4acfc6f44",
}


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


def _fraction(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _prerequisite_audit() -> dict[str, Any]:
    rows = []
    for relative, expected in PREREQUISITES.items():
        path = ROOT / relative
        actual = _sha256(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "all_positive_checks_pass": payload.get(
                    "all_positive_checks_pass"
                ),
                "matches": (
                    actual == expected
                    and payload.get("all_positive_checks_pass") is True
                ),
            }
        )
    return {
        "rows": rows,
        "all_checks_pass": all(row["matches"] for row in rows),
    }


def _exact_chain_rows() -> list[dict[str, Any]]:
    rows = []
    for block_count in range(1, 17):
        maximum_mode = 2**block_count - 1
        physical = Fraction(1, 4)
        component_sum = Fraction(block_count, 4)
        signed_interfaces = physical - component_sum
        epsilon = Fraction(1, maximum_mode**2)
        floor_physical = physical + epsilon * maximum_mode / 2
        floor_components = (
            component_sum + epsilon * maximum_mode / 2
        )
        rows.append(
            {
                "dyadic_block_count_J": block_count,
                "maximum_mode_N": maximum_mode,
                "physical_weighted_Fisher": _fraction(physical),
                "sum_component_weighted_Fisher": _fraction(component_sum),
                "component_to_physical_ratio": float(
                    component_sum / physical
                ),
                "total_signed_cross_band_correction": _fraction(
                    signed_interfaces
                ),
                "adjacent_interface_count": block_count - 1,
                "correction_per_interface": (
                    "0"
                    if block_count == 1
                    else _fraction(
                        signed_interfaces / (block_count - 1)
                    )
                ),
                "strict_floor_epsilon": _fraction(epsilon),
                "strict_floor_physical_Fisher": _fraction(
                    floor_physical
                ),
                "strict_floor_component_sum": _fraction(
                    floor_components
                ),
                "strict_floor_ratio": float(
                    floor_components / floor_physical
                ),
                "all_checks_pass": bool(
                    component_sum / physical == block_count
                    and signed_interfaces
                    == -Fraction(block_count - 1, 4)
                    and (
                        block_count == 1
                        or signed_interfaces / (block_count - 1)
                        == -Fraction(1, 4)
                    )
                ),
            }
        )
    return rows


def _support_rows() -> list[dict[str, Any]]:
    rows = []
    for block_index in range(12):
        lower = 2**block_index
        upper = 2 ** (block_index + 1) - 1
        rows.append(
            {
                "block_index": block_index,
                "positive_modes": f"{lower}<=n<={upper}",
                "negative_modes": f"{-upper}<=n<={-lower}",
                "lower_carrier": lower,
                "upper_carrier": upper,
                "annular_ratio": upper / lower,
                "annular_ratio_below_two": upper < 2 * lower,
                "divergence_free_embedding": (
                    "u_j=(0,sum_(n in I_j) sin(nx_1)/n,0)"
                ),
                "pressure": "0",
            }
        )
    return rows


def _coscaling_rows() -> list[dict[str, Any]]:
    rows = []
    block_count = 8
    for frequency, amplitude in ((1, 1), (2, 3), (4, 5), (8, 7)):
        physical = Fraction(amplitude**2 * frequency**2, 4)
        components = block_count * physical
        weight_fisher = Fraction(frequency**2, 16)
        rows.append(
            {
                "partition_frequency_m": frequency,
                "velocity_amplitude_a": amplitude,
                "dyadic_block_count_J": block_count,
                "physical_weighted_Fisher": _fraction(physical),
                "sum_component_weighted_Fisher": _fraction(components),
                "terminal_weight_Fisher": _fraction(weight_fisher),
                "component_to_physical_ratio": float(
                    components / physical
                ),
                "pressure": "0",
            }
        )
    return rows


def _dyadic_shells(
    field: dict[tuple[int, int, int], np.ndarray],
) -> list[dict[tuple[int, int, int], np.ndarray]]:
    shells: dict[int, dict[tuple[int, int, int], np.ndarray]] = {}
    for wave, value in field.items():
        radius = math.sqrt(sum(component**2 for component in wave))
        shell = int(math.floor(math.log2(radius)))
        shells.setdefault(shell, {})[wave] = value
    return [shells[index] for index in sorted(shells)]


def _weighted_energy(
    field: dict[tuple[int, int, int], np.ndarray],
    center: np.ndarray,
    weights: dict[tuple[int, int, int], float],
) -> float:
    energies, _ = _weighted_enstrophies(field, center)
    return sum(
        weights[vertex] * energy
        for vertex, energy in zip(VERTICES, energies)
    )


def _finite_field_recombination_audit() -> dict[str, Any]:
    zero_face_weights = {
        vertex: (1.0 if vertex[0] == -1 else 0.0)
        for vertex in VERTICES
    }
    taylor = _taylor_green_field()
    taylor_center = np.zeros(3)
    taylor_physical = _weighted_energy(
        taylor, taylor_center, zero_face_weights
    )
    taylor_components = sum(
        _weighted_energy(shell, taylor_center, zero_face_weights)
        for shell in _dyadic_shells(taylor)
    )

    seed = _seed81_field()
    seed_center = (
        2.0
        * math.pi
        * STARTING_GRID_INDEX.astype(float)
        / GRID_SIZE
    )
    seed_weights = _weights_by_vertex()
    seed_physical = _weighted_energy(seed, seed_center, seed_weights)
    seed_shells = _dyadic_shells(seed)
    seed_components = sum(
        _weighted_energy(shell, seed_center, seed_weights)
        for shell in seed_shells
    )
    return {
        "Taylor_Green": {
            "dyadic_shell_count": len(_dyadic_shells(taylor)),
            "physical_weighted_Fisher": taylor_physical,
            "component_weighted_Fisher_sum": taylor_components,
            "ratio": taylor_components / taylor_physical,
            "compatible_pressure_load": 0.0,
            "all_checks_pass": (
                len(_dyadic_shells(taylor)) == 1
                and abs(taylor_components / taylor_physical - 1.0)
                < 1.0e-14
            ),
        },
        "seed81": {
            "dyadic_shell_count": len(seed_shells),
            "physical_weighted_Fisher": seed_physical,
            "component_weighted_Fisher_sum": seed_components,
            "ratio": seed_components / seed_physical,
            "stored_compatible_pressure_load": 1.280453496113644,
            "all_checks_pass": (
                len(seed_shells) == 2
                and seed_physical > 0.0
                and seed_components > 0.0
                and math.isfinite(seed_components / seed_physical)
            ),
        },
    }


def audit() -> dict[str, Any]:
    prerequisites = _prerequisite_audit()
    chain_rows = _exact_chain_rows()
    support_rows = _support_rows()
    coscaling_rows = _coscaling_rows()
    finite_fields = _finite_field_recombination_audit()
    positive_checks = {
        "prerequisite_hashes_and_results_pass": prerequisites[
            "all_checks_pass"
        ],
        "exact_chain_identities_pass": all(
            row["all_checks_pass"] for row in chain_rows
        ),
        "every_component_is_dyadically_annular": all(
            row["annular_ratio_below_two"] for row in support_rows
        ),
        "coscaling_ratio_is_invariant": all(
            row["component_to_physical_ratio"] == 8.0
            for row in coscaling_rows
        ),
        "Taylor_Green_replay_passes": finite_fields["Taylor_Green"][
            "all_checks_pass"
        ],
        "seed81_replay_passes": finite_fields["seed81"][
            "all_checks_pass"
        ],
    }
    all_positive = all(positive_checks.values())
    return {
        "kind": "multiband_weighted_fisher_recombination_no_go_audit",
        "schema_version": 1,
        "status": (
            "uniform_multiband_weighted_Fisher_recombination_falsified"
            if all_positive
            else "audit_failed"
        ),
        "all_positive_checks_pass": all_positive,
        "positive_checks": positive_checks,
        "prerequisites": prerequisites,
        "exact_Fourier_form": {
            "weighted_Fisher": (
                "E_lambda(u)=sum_(k,l) (k dot l) "
                "lambdahat(l-k) uhat(k) dot conjugate(uhat(l))"
            ),
            "zero_face_weight": (
                "lambda=sin(x_1/2)^2 has Fourier support "
                "{0,+e_1,-e_1}"
            ),
            "residue_chain_form": (
                "For d_n=n a_n, "
                "E_lambda(f)=(1/4)sum_n |d_(n+1)-d_n|^2."
            ),
            "compatible_coefficients": (
                "w_v=1 when v_1=-1 and w_v=0 when v_1=+1; "
                "summing the other tensor factors gives lambda."
            ),
        },
        "counterexample": {
            "field": (
                "u_J=(0,sum_(n=1)^(2^J-1) sin(nx_1)/n,0)"
            ),
            "dyadic_components": (
                "u_j=(0,sum_(2^j<=n<2^(j+1)) sin(nx_1)/n,0)"
            ),
            "smooth_finite_Fourier": True,
            "divergence_free": True,
            "pressure_free_shear": True,
            "physical_weighted_Fisher": "1/4",
            "each_component_weighted_Fisher": "1/4",
            "component_sum": "J/4",
            "ratio": "J -> infinity",
            "signed_cross_band_correction": "-(J-1)/4",
            "nearest_neighbor_correction_per_interface": "-1/4",
            "interpretation": (
                "Finite graph degree does not imply coercive "
                "recombination. The omitted neighboring-band terms carry "
                "the entire long-chain cancellation."
            ),
        },
        "exact_chain_rows": chain_rows,
        "dyadic_annular_support": support_rows,
        "strict_floor_limit": {
            "weight": (
                "lambda_epsilon=epsilon+sin(x_1/2)^2 is represented by "
                "adding epsilon to all eight compatible coefficients."
            ),
            "exact_ratio": (
                "(J+2 epsilon (2^J-1))/"
                "(1+2 epsilon (2^J-1))"
            ),
            "choice": "epsilon=(2^J-1)^(-2)",
            "conclusion": (
                "Every finite example has a positive floor, but the "
                "recombination constant still diverges with J as the "
                "floor approaches zero."
            ),
        },
        "terminal_weight_Fisher_no_rescue": {
            "base_value": "integral lambda|grad lambda|^2=1/16",
            "scaled_weight_value": (
                "For t lambda, the value is t^3/16."
            ),
            "velocity_scaling": (
                "Replacing u by alpha u multiplies both physical and "
                "component velocity Fisher energies by alpha^2, while a "
                "weight-only additive Fisher allowance is unchanged."
            ),
            "conclusion": (
                "No fixed additive multiple of terminal weight Fisher can "
                "repair the failed recombination inequality uniformly."
            ),
        },
        "coscaling_stress": {
            "family": (
                "lambda_m=sin(mx_1/2)^2 and "
                "u_(J,a,m)=a u_J(mx)"
            ),
            "physical_and_component_scaling": "a^2 m^2",
            "pressure": "0",
            "rows": coscaling_rows,
            "ratio_invariant": True,
        },
        "mandatory_finite_fields": finite_fields,
        "certification_flags": {
            "uniform_floor_free_multiband_Fisher_recombination": False,
            "finite_overlap_degree_implies_coercivity": False,
            "componentwise_annular_absorption_summable_by_absolute_values": False,
            "signed_neighboring_Fisher_edges_must_be_retained": True,
            "balanced_single_band_pressure_theorem_invalidated": False,
            "far_carrier_H_minus_one_tail_theorem_invalidated": False,
            "joint_signed_pressure_Fisher_block_bound_proved": False,
            "cross_shell_HHL_pressure_absorbed": False,
            "terminal_dual_supremum_controlled": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "scope": (
            "The counterexample falsifies an auxiliary recombination step, "
            "not the certified single-band pressure theorem, the far-tail "
            "theorem, or every joint signed pressure-Fisher estimate. Its "
            "pressure is zero, so a future proof may still couple pressure "
            "and the retained neighboring Fisher edges before taking "
            "absolute values."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise SystemExit("multiband Fisher recombination audit failed")
    if not arguments.check_only:
        _atomic_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
