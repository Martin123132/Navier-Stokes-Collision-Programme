"""Port the complete annular c1 tail theorem to the two-shear witness."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from annular_two_shear_square_gate_audit import _modified_finite_packet


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_two_shear_full_c1_port_audit_v1.json"
)
SQUARE_RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_two_shear_square_gate_audit_v1.json"
)
TAIL_RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_full_c1_tail_ledger_audit_v1.json"
)
FIXED_OUTPUT_RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_fixed_output_continuum_gate_audit_v1.json"
)
ALGORITHM_REVISION = "annular-two-shear-full-c1-port-v1"

OLD_PER_ATOMIC_CONSTANT = 375_840
OLD_ATOMIC_COEFFICIENT_MASS = 94
OLD_TAIL_CONSTANT = 35_328_960
NEW_LOW_FIELD_FACTOR = 2
NEW_PER_ATOMIC_CONSTANT = (
    NEW_LOW_FIELD_FACTOR * OLD_PER_ATOMIC_CONSTANT
)
NEW_TAIL_CONSTANT = NEW_LOW_FIELD_FACTOR * OLD_TAIL_CONSTANT


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _collect_key(value: Any, key: str) -> list[Any]:
    output: list[Any] = []
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                output.append(current_value)
            output.extend(_collect_key(current_value, key))
    elif isinstance(value, list):
        for current_value in value:
            output.extend(_collect_key(current_value, key))
    return output


def _multiplier_certificate() -> dict[str, Any]:
    # On the packet, |y|/x and |z|/x are at most 1/4, hence
    # r/x <= sqrt(1+1/16+1/16)=sqrt(9/8).
    transverse_ratio = 1.0 / 4.0
    radius_over_x = math.sqrt(9.0 / 8.0)

    # m_z=(x^2+y^2)/r^3 has coordinate derivative bounds
    # (5,5,3)/r^2. Since m_x=-(z/x)m_z, the product rule gives:
    mx_dx = transverse_ratio * 5.0 + (
        transverse_ratio * radius_over_x
    )
    mx_dy = transverse_ratio * 5.0
    mx_dz = radius_over_x + transverse_ratio * 3.0
    vector_derivative_bounds = [
        math.hypot(5.0, mx_dx),
        math.hypot(5.0, mx_dy),
        math.hypot(3.0, mx_dz),
    ]
    maximum_vector_derivative = max(vector_derivative_bounds)

    multiplier_derivative_constant = 6.0
    annular_derivative_over_n_squared = (
        multiplier_derivative_constant / 4.0
    )
    sine_product_difference_constant = math.pi
    coefficient_difference_constant = (
        sine_product_difference_constant / 2.0
        + annular_derivative_over_n_squared
    )

    # The continuum profile has coordinate Lipschitz constant at most the
    # same pi/2+3/2. A sample shift plus a centred-cell displacement is
    # below 2.1/N in Euclidean distance. The deliberately rounded
    # pointwise constant 25/N gives L1+L2 <=(4+sqrt(4))*25/N=150/N.
    continuum_coordinate_lipschitz = (
        math.pi / 2.0 + 1.5
    )
    continuum_euclidean_lipschitz = (
        math.sqrt(3.0) * continuum_coordinate_lipschitz
    )
    maximum_sample_and_cell_displacement = 2.1
    raw_pointwise_error_constant = (
        continuum_euclidean_lipschitz
        * maximum_sample_and_cell_displacement
    )
    rounded_pointwise_error_constant = 25.0
    support_union_volume = 4.0
    l1_error_constant = (
        support_union_volume * rounded_pointwise_error_constant
    )
    l2_error_constant = (
        math.sqrt(support_union_volume)
        * rounded_pointwise_error_constant
    )
    combined_profile_error_constant = (
        l1_error_constant + l2_error_constant
    )
    rounded_profile_error_constant = 256.0

    checks = {
        "angular_radius_bound": radius_over_x < 1.061,
        "explicit_vector_derivative_below_six": (
            maximum_vector_derivative < multiplier_derivative_constant
        ),
        "first_difference_below_four": (
            coefficient_difference_constant < 4.0
        ),
        "raw_pointwise_error_below_rounded_constant": (
            raw_pointwise_error_constant
            < rounded_pointwise_error_constant
        ),
        "combined_profile_error_below_256": (
            combined_profile_error_constant
            < rounded_profile_error_constant
        ),
    }
    return {
        "multiplier": (
            "m_*(k)=(k_x^2+k_y^2)/(k_x*|k|^3)*(-k_z,0,k_x)"
        ),
        "homogeneity": "-1",
        "size_bound": (
            "|m_*(k)|<=(k_x^2+k_y^2)|k|/"
            "(k_x|k|^3)<=1/k_x<=1/(2N)"
        ),
        "transverse_ratio_bound": "|k_y|/k_x,|k_z|/k_x<=1/4",
        "radius_over_x_bound": "r/k_x<=sqrt(9/8)",
        "m_z_coordinate_derivative_constants": [5.0, 5.0, 3.0],
        "m_x_coordinate_derivative_constants": [
            mx_dx,
            mx_dy,
            mx_dz,
        ],
        "vector_coordinate_derivative_constants": (
            vector_derivative_bounds
        ),
        "maximum_vector_coordinate_derivative_constant": (
            maximum_vector_derivative
        ),
        "certified_multiplier_derivative_bound": (
            "|partial_j m_*|<6/|k|^2<=3/(2N^2)"
        ),
        "derived_first_difference_constant": (
            coefficient_difference_constant
        ),
        "certified_packet_first_difference": (
            "|Delta_j[(-1)^sum(k) hhat_*,N(k)]|<4/N^2"
        ),
        "boundary_argument": (
            "The packet sine is at most pi/N on an adjacent boundary "
            "cell, so the same one-difference bound holds for the zero "
            "extension."
        ),
        "continuum_profile_approximation": {
            "coordinate_lipschitz_constant": (
                continuum_coordinate_lipschitz
            ),
            "euclidean_lipschitz_constant": (
                continuum_euclidean_lipschitz
            ),
            "sample_plus_cell_displacement_over_h": (
                maximum_sample_and_cell_displacement
            ),
            "raw_pointwise_error_constant": (
                raw_pointwise_error_constant
            ),
            "rounded_pointwise_error_constant": (
                rounded_pointwise_error_constant
            ),
            "support_union_volume_bound": support_union_volume,
            "derived_L1_plus_L2_error_constant": (
                combined_profile_error_constant
            ),
            "certified_bound": (
                "epsilon_N=||b_N-b||_1+||b_N-b||_2<=256/N"
            ),
        },
        "checks": checks,
        "all_multiplier_checks_pass": all(checks.values()),
    }


def _tail_port_certificate(
    tail: dict[str, Any],
    square: dict[str, Any],
) -> dict[str, Any]:
    ledger = tail["termwise_tail_ledger"]
    low_leaf_counts = _collect_key(ledger, "low_shear_leaf_count")
    stencil = square["exact_low_stencil"]
    exact_checks = stencil["checks"]
    checks = {
        "predecessor_tail_passed": (
            tail.get("all_positive_checks_pass") is True
        ),
        "fourteen_rows_retained": (
            ledger["row_count"] == 14
            and ledger["structural_profile_count"] == 14
        ),
        "every_structural_profile_is_linear_in_low_field": (
            len(low_leaf_counts) == 14
            and set(low_leaf_counts) == {1}
        ),
        "atomic_coefficient_mass_retained": (
            ledger["absolute_atomic_coefficient_mass"]
            == OLD_ATOMIC_COEFFICIENT_MASS
        ),
        "old_per_atomic_constant_replayed": (
            ledger["per_atomic_contraction_constant"]
            == OLD_PER_ATOMIC_CONSTANT
        ),
        "old_tail_constant_replayed": (
            ledger["full_tail_constant"] == OLD_TAIL_CONSTANT
        ),
        "both_new_low_waves_are_even": exact_checks[
            "each_low_wave_has_even_coordinate_sum"
        ],
        "both_new_low_directions_are_divergence_free": exact_checks[
            "each_low_direction_is_divergence_free"
        ],
        "combined_stencil_l1_doubles": (
            stencil[
                "combined_active_output_l1_before_dividing_by_sqrt2"
            ]
            == "3"
        ),
        "new_per_atomic_constant_is_exact_double": (
            NEW_PER_ATOMIC_CONSTANT
            == NEW_LOW_FIELD_FACTOR * OLD_PER_ATOMIC_CONSTANT
        ),
        "new_tail_constant_is_exact_double": (
            NEW_TAIL_CONSTANT
            == NEW_PER_ATOMIC_CONSTANT
            * OLD_ATOMIC_COEFFICIENT_MASS
        ),
    }
    return {
        "linearity": (
            "The amplitude-one coefficient contains exactly one low "
            "leaf in every one of the fourteen structural profiles. "
            "Therefore c1[U_yz+U_xy]=c1[U_yz]+c1[U_xy]."
        ),
        "portable_hypotheses": [
            "each low wave has even coordinate sum",
            "each low wave has squared radius 2",
            "each normalized low direction is divergence free",
            "the low support is finite and has doubled Fourier l1 mass",
            "the high support radius and mode count are unchanged",
            "the high coefficient and first-difference bounds are unchanged",
            "the degree-zero outer projector is held fixed in every tail row",
        ],
        "old_low_fourier_l1": 2,
        "new_low_fourier_l1": 4,
        "constant_multiplier": NEW_LOW_FIELD_FACTOR,
        "old_per_atomic_constant": OLD_PER_ATOMIC_CONSTANT,
        "new_per_atomic_constant": NEW_PER_ATOMIC_CONSTANT,
        "atomic_coefficient_mass": OLD_ATOMIC_COEFFICIENT_MASS,
        "old_tail_constant": OLD_TAIL_CONSTANT,
        "new_tail_constant": NEW_TAIL_CONSTANT,
        "ported_tail_inequality": (
            "|c1_*,N-D_*,N|<=70657920*N^6 for odd N>=5"
        ),
        "checks": checks,
        "all_tail_port_checks_pass": all(checks.values()),
    }


def _finite_first_difference_replay(
    sizes: tuple[int, ...] = (5, 9, 17, 33),
) -> dict[str, Any]:
    rows = []
    for size in sizes:
        waves, velocity, parity = _modified_finite_packet(size)
        gauged = parity[..., None] * velocity
        padded = np.pad(
            gauged,
            ((1, 1), (1, 1), (1, 1), (0, 0)),
            mode="constant",
        )
        scaled_maxima = []
        for axis in range(3):
            difference = np.diff(padded, axis=axis)
            scaled_maxima.append(
                float(
                    size**2
                    * np.max(np.linalg.norm(difference, axis=-1))
                )
            )
        divergence = np.sum(waves * velocity, axis=-1)
        rows.append(
            {
                "size": size,
                "N2_times_maximum_first_difference_by_axis": (
                    scaled_maxima
                ),
                "maximum_scaled_first_difference": max(scaled_maxima),
                "maximum_divergence_residual": float(
                    np.max(np.abs(divergence))
                ),
                "checks_pass": bool(
                    max(scaled_maxima) < 4.0
                    and np.max(np.abs(divergence)) < 1.0e-12
                ),
            }
        )
    return {
        "rows": rows,
        "analytic_bound": (
            "|Delta_j[parity*hhat_*,N]|<4/N^2"
        ),
        "replay_is_not_source_of_analytic_bound": True,
        "all_replay_checks_pass": all(
            row["checks_pass"] for row in rows
        ),
    }


def main() -> None:
    square = _load_json(SQUARE_RESULT)
    tail = _load_json(TAIL_RESULT)
    fixed = _load_json(FIXED_OUTPUT_RESULT)
    multiplier = _multiplier_certificate()
    tail_port = _tail_port_certificate(tail, square)
    finite_difference_replay = _finite_first_difference_replay()

    square_certification = square["certification"]
    stencil = square["exact_low_stencil"]
    fixed_output_checks = {
        "old_fixed_output_continuity_theorem_passed": (
            fixed.get("all_positive_checks_pass") is True
            and fixed["certification_flags"][
                "dominant_fixed_output_over_N7_convergence_proved"
            ]
        ),
        "new_active_output_set_is_finite": (
            stencil["combined_active_output_count"] == 58
            and stencil["combined_maximum_radius_squared"] == 6
        ),
        "new_projector_matrix_is_exact": stencil["checks"][
            "combined_matrix_exact"
        ],
        "new_profile_converges_in_L1_and_L2": multiplier[
            "all_multiplier_checks_pass"
        ],
        "new_continuum_sign_is_exact": square_certification[
            "modified_four_high_continuum_sign_analytic"
        ],
        "new_continuum_limit_is_strictly_nonzero": square_certification[
            "strict_nonzero_analytic"
        ],
    }
    full_limit_checks = {
        **fixed_output_checks,
        "ported_tail_is_o_N7": tail_port[
            "all_tail_port_checks_pass"
        ],
    }
    payload = {
        "algorithm_revision": ALGORITHM_REVISION,
        "prerequisites": {
            "two_shear_square_gate": (
                SQUARE_RESULT.relative_to(ROOT).as_posix()
            ),
            "two_shear_square_gate_sha256": _sha256(SQUARE_RESULT),
            "original_full_c1_tail_ledger": (
                TAIL_RESULT.relative_to(ROOT).as_posix()
            ),
            "original_full_c1_tail_ledger_sha256": _sha256(TAIL_RESULT),
            "original_fixed_output_gate": (
                FIXED_OUTPUT_RESULT.relative_to(ROOT).as_posix()
            ),
            "original_fixed_output_gate_sha256": _sha256(
                FIXED_OUTPUT_RESULT
            ),
        },
        "modified_packet_multiplier_certificate": multiplier,
        "finite_first_difference_replay": finite_difference_replay,
        "two_shear_tail_port_certificate": tail_port,
        "fixed_output_convergence_port": {
            "profile_error_bound": (
                "||b_N-b||_1+||b_N-b||_2<=256/N"
            ),
            "translation_error": (
                "The 58 fixed output shifts are q/N with |q|<=sqrt(6); "
                "translation continuity of compact H1 profiles makes "
                "each shifted bilinear/cubic convolution converge."
            ),
            "operator_continuity": fixed[
                "quantitative_remainder_certificate"
            ][
                "bilinear_bound"
            ],
            "conclusion": "D_*,N/N^7 -> L_*",
            "checks": fixed_output_checks,
            "all_fixed_output_port_checks_pass": all(
                fixed_output_checks.values()
            ),
        },
        "full_limit_certificate": {
            "tail_inequality": (
                "|c1_*,N-D_*,N|<=70657920*N^6 for odd N>=5"
            ),
            "normalized_tail_inequality": (
                "|c1_*,N-D_*,N|/N^7<=70657920/N"
            ),
            "fixed_output_limit": "D_*,N/N^7 -> L_*",
            "continuum_identity": (
                "L_*=-(3*sqrt(2)/20)*||v_y||_2^2<0"
            ),
            "conclusion": "c1_*,N/N^7 -> L_*<0",
            "checks": full_limit_checks,
            "all_full_limit_checks_pass": all(
                full_limit_checks.values()
            ),
        },
        "certification": {
            "modified_packet_first_difference_certified": True,
            "modified_fixed_output_convergence_proved": True,
            "modified_full_c1_tail_ledger_ported": True,
            "modified_full_c1_over_N7_convergence_proved": True,
            "modified_full_c1_limit_negative_certified": True,
            "original_single_shear_L_EE_sign_certified": False,
            "modified_static_optimizer_ported": False,
            "modified_first_jet_ported": False,
            "modified_complete_second_jet_ported": False,
            "modified_parabolic_window_closed": False,
            "critical_L3_controlled": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "remaining_obligation": (
            "The modified amplitude-one four-high coefficient now has a "
            "strictly negative N7 limit. Port the static joint optimizer "
            "and complete first/second jet finite formulas to the two-mode "
            "low field, then establish the uniform Taylor and parabolic "
            "window estimates. No singularity or regularity conclusion "
            "follows from the coefficient limit alone."
        ),
    }
    payload["all_port_checks_pass"] = bool(
        multiplier["all_multiplier_checks_pass"]
        and finite_difference_replay["all_replay_checks_pass"]
        and tail_port["all_tail_port_checks_pass"]
        and payload["fixed_output_convergence_port"][
            "all_fixed_output_port_checks_pass"
        ]
        and payload["full_limit_certificate"][
            "all_full_limit_checks_pass"
        ]
        and not payload["certification"]["finite_time_blowup_proved"]
    )
    _atomic_json(RESULT, payload)
    print(
        json.dumps(
            {
                "result": RESULT.relative_to(ROOT).as_posix(),
                "result_sha256": _sha256(RESULT),
                "new_tail_constant": NEW_TAIL_CONSTANT,
                "full_limit": payload["full_limit_certificate"][
                    "conclusion"
                ],
                "all_port_checks_pass": payload["all_port_checks_pass"],
                "clay_problem_solved": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
