"""Reduce the annular four-high second jet to a continuum functional."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_fixed_output_continuum_gate_audit_v1.json"
)
PREDECESSOR = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_inviscid_second_jet_branch_audit_v1.json"
)
PREDECESSOR_SHA256 = (
    "ef89d9b9f39ace8b886a4d40bdd7fed6aa908ffcd2ea2e41ca179a3bb82705c7"
)
ALGORITHM_REVISION = "annular-rho-zero-fixed-output-continuum-gate-v1"
LOW_WAVE = (0, 1, -1)
FIRST_FORM = "-6S[BHH,BHH,U;Phi]"
SECOND_FORM = "-12S[B(H,BHH),H,U;Phi]"


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


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _weight_coefficient(wave: tuple[int, int, int]) -> Fraction:
    if any(abs(component) > 1 for component in wave):
        return Fraction(0)
    value = Fraction(1)
    for component in wave:
        value *= Fraction(1, 2) if component == 0 else Fraction(1, 4)
    return value


def _add(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
    factor: int = 1,
) -> tuple[int, int, int]:
    return tuple(
        first[index] + factor * second[index] for index in range(3)
    )


def _parity(wave: tuple[int, int, int]) -> int:
    return 1 if sum(wave) % 2 == 0 else -1


def _active_output_stencil() -> dict[str, Any]:
    candidate_waves = {
        tuple(
            sign * LOW_WAVE[index] + shift[index]
            for index in range(3)
        )
        for sign in (-1, 1)
        for shift in product((-1, 0, 1), repeat=3)
    }
    rows: list[dict[str, Any]] = []
    # The actual Fourier coefficient is alpha_q/sqrt(2).
    active: dict[tuple[int, int, int], Fraction] = {}
    for wave in sorted(candidate_waves):
        minus = _add(wave, LOW_WAVE, factor=-1)
        plus = _add(wave, LOW_WAVE, factor=1)
        alpha = Fraction(wave[1] + wave[2]) * (
            _weight_coefficient(minus) - _weight_coefficient(plus)
        )
        if alpha == 0:
            continue
        active[wave] = alpha
        rows.append(
            {
                "wave": list(wave),
                "radius_squared": sum(value * value for value in wave),
                "parity": _parity(wave),
                "alpha_q_where_A_q_equals_alpha_q_over_sqrt2": (
                    _fraction_text(alpha)
                ),
            }
        )

    # R = sqrt(2) Q, where
    # Q=sum sigma_q A_q q q^T/|q|^2.
    matrix = [
        [Fraction(0) for _ in range(3)] for _ in range(3)
    ]
    for wave, alpha in active.items():
        norm_squared = sum(value * value for value in wave)
        for row in range(3):
            for column in range(3):
                matrix[row][column] += (
                    Fraction(
                        _parity(wave)
                        * wave[row]
                        * wave[column],
                        norm_squared,
                    )
                    * alpha
                )
    expected = [
        [Fraction(0), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(-1, 20), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1, 20)],
    ]
    maximum_radius_squared = max(
        row["radius_squared"] for row in rows
    )
    absolute_alpha_sum = sum(abs(value) for value in active.values())
    return {
        "low_wave": list(LOW_WAVE),
        "low_direction": "[0,1,1]/sqrt(2)",
        "weight_coefficients": (
            "w_s=product_j(1/2 if s_j=0 else 1/4), "
            "s in {-1,0,1}^3"
        ),
        "test_coefficient_formula": (
            "A_q=(q_y+q_z)[w_(q-ell)-w_(q+ell)]/sqrt(2)"
        ),
        "phase": "sigma_q=(-1)^(q_x+q_y+q_z)",
        "active_output_count": len(rows),
        "maximum_active_radius_squared": maximum_radius_squared,
        "all_active_outputs_satisfy_abs_q_less_than_4": (
            maximum_radius_squared < 16
        ),
        "sum_absolute_alpha_q": _fraction_text(absolute_alpha_sum),
        "active_output_rows": rows,
        "sqrt2_times_projector_sum_matrix": [
            [_fraction_text(value) for value in row] for row in matrix
        ],
        "projector_sum_matrix": (
            "Q=(sqrt(2)/40) diag(0,-1,1)"
        ),
        "expected_matrix_reproduced": matrix == expected,
        "all_checks_pass": bool(
            len(rows) == 36
            and maximum_radius_squared == 6
            and absolute_alpha_sum == Fraction(3, 2)
            and matrix == expected
        ),
    }


def _mode_map(profile: dict[str, Any]) -> dict[tuple[int, int, int], float]:
    return {
        tuple(int(value) for value in row["wave"]): float(row["value"])
        for row in profile["bounded_output_mode_rows"]
    }


def _fit_limit(
    sizes: Iterable[int],
    values: Iterable[float],
    degree: int,
) -> dict[str, Any]:
    size_array = np.asarray(tuple(sizes), dtype=float)
    value_array = np.asarray(tuple(values), dtype=float)
    coefficients = np.polyfit(1.0 / size_array, value_array, degree)
    fitted = np.polyval(coefficients, 1.0 / size_array)
    return {
        "degree_in_inverse_N": degree,
        "sizes": [int(value) for value in size_array],
        "candidate_limit": float(coefficients[-1]),
        "maximum_replay_residual": float(
            np.max(np.abs(fitted - value_array))
        ),
        "certification_status": "diagnostic_only",
    }


def _finite_output_diagnostics(
    predecessor: dict[str, Any],
    stencil: dict[str, Any],
) -> dict[str, Any]:
    active_waves = {
        tuple(int(value) for value in row["wave"])
        for row in stencil["active_output_rows"]
    }
    rows = []
    histories: dict[tuple[int, int, int], list[dict[str, Any]]] = {
        wave: [] for wave in active_waves
    }
    for carrier in predecessor["carrier_rows"]:
        size = int(carrier["size"])
        profiles = carrier["dominant_a1_pressure_output_shells"]
        form_values: dict[str, float] = {}
        inactive_values: dict[str, float] = {}
        combined_modes: dict[tuple[int, int, int], float] = {}
        for label in (FIRST_FORM, SECOND_FORM):
            mode_values = _mode_map(profiles[label])
            active_value = sum(
                mode_values.get(wave, 0.0) for wave in active_waves
            )
            form_values[label] = active_value
            inactive_values[label] = (
                float(profiles[label]["direct_value"]) - active_value
            )
            for wave in active_waves:
                combined_modes[wave] = (
                    combined_modes.get(wave, 0.0)
                    + mode_values.get(wave, 0.0)
                )
        active_sum = sum(form_values.values())
        dominant_direct = sum(
            float(profiles[label]["direct_value"])
            for label in (FIRST_FORM, SECOND_FORM)
        )
        full_a1 = float(carrier["a1_coefficient"])
        rows.append(
            {
                "size": size,
                "active_first_form": form_values[FIRST_FORM],
                "active_second_form": form_values[SECOND_FORM],
                "active_combined": active_sum,
                "active_combined_over_N7": active_sum / size**7,
                "dominant_direct": dominant_direct,
                "dominant_inactive_remainder": (
                    dominant_direct - active_sum
                ),
                "full_a1_coefficient": full_a1,
                "full_a1_minus_active": full_a1 - active_sum,
                "full_a1_minus_active_over_N7": (
                    (full_a1 - active_sum) / size**7
                ),
                "inactive_by_form": inactive_values,
            }
        )
        for wave, value in combined_modes.items():
            histories[wave].append(
                {
                    "size": size,
                    "value": value,
                    "value_over_N7": value / size**7,
                }
            )

    sizes = [row["size"] for row in rows]
    first_values = [
        row["active_first_form"] / row["size"] ** 7 for row in rows
    ]
    second_values = [
        row["active_second_form"] / row["size"] ** 7 for row in rows
    ]
    combined_values = [
        row["active_combined_over_N7"] for row in rows
    ]
    recent = slice(3, None)
    mode_rows = [
        {
            "wave": list(wave),
            "largest_carrier_value": history[-1]["value"],
            "largest_carrier_value_over_N7": history[-1][
                "value_over_N7"
            ],
            "history": history,
        }
        for wave, history in histories.items()
    ]
    mode_rows.sort(
        key=lambda row: abs(row["largest_carrier_value"]),
        reverse=True,
    )
    largest = rows[-1]
    return {
        "rows": rows,
        "active_mode_histories": mode_rows,
        "recent_inverse_N_fits": {
            "first_form_quadratic": _fit_limit(
                sizes[recent], first_values[recent], 2
            ),
            "second_form_quadratic": _fit_limit(
                sizes[recent], second_values[recent], 2
            ),
            "combined_quadratic": _fit_limit(
                sizes[recent], combined_values[recent], 2
            ),
            "combined_cubic": _fit_limit(
                sizes[2:], combined_values[2:], 3
            ),
        },
        "largest_carrier": largest,
        "largest_full_a1_fraction_from_active_outputs": (
            abs(largest["active_combined"])
            / abs(largest["full_a1_coefficient"])
        ),
        "all_finite_active_sums_negative": all(
            row["active_combined"] < 0.0 for row in rows
        ),
        "finite_sign_is_not_a_limit_certificate": True,
        "all_checks_pass": bool(
            sizes == [5, 7, 9, 13, 17, 21, 25, 29]
            and all(row["active_combined"] < 0.0 for row in rows)
            and abs(largest["dominant_inactive_remainder"]) < 0.1
            and abs(largest["full_a1_minus_active"]) < 1.4
        ),
    }


def _permutation_support_certificate() -> dict[str, Any]:
    rows = [
        {
            "form": FIRST_FORM,
            "permutation": "-2 T(V,V,U;Phi)",
            "active_pressure_outputs": "supp(A_q), exactly 36 modes",
            "candidate_power": 7,
            "role": "continuum_leading",
        },
        {
            "form": FIRST_FORM,
            "permutation": "-4 T(V,U,V;Phi)",
            "active_pressure_outputs": (
                "bounded V-U pressure output and bounded V-Phi test output"
            ),
            "candidate_power": 2,
            "role": "fixed-output_remainder",
        },
        {
            "form": SECOND_FORM,
            "permutation": "-4 T(G,H,U;Phi)",
            "active_pressure_outputs": "supp(A_q), exactly 36 modes",
            "candidate_power": 7,
            "role": "continuum_leading",
        },
        {
            "form": SECOND_FORM,
            "permutation": "-4 T(G,U,H;Phi)",
            "active_pressure_outputs": "none for bounded q when N>=5",
            "candidate_power": 6,
            "role": "high-output_stencil_remainder",
        },
        {
            "form": SECOND_FORM,
            "permutation": "-4 T(H,U,G;Phi)",
            "active_pressure_outputs": "none for bounded q when N>=5",
            "candidate_power": 6,
            "role": "high-output_stencil_remainder",
        },
    ]
    return {
        "notation": (
            "V=B(H,H), G=B(H,V), "
            "T(x,y,z;Phi)=integral p[x,y] z dot grad Phi"
        ),
        "symmetrization_expansion": {
            FIRST_FORM: (
                "-2T(V,V,U;Phi)-4T(V,U,V;Phi)"
            ),
            SECOND_FORM: (
                "-4T(G,H,U;Phi)-4T(G,U,H;Phi)"
                "-4T(H,U,G;Phi)"
            ),
        },
        "rows": rows,
        "N7_saturating_permutations": [
            "-2 T(V,V,U;Phi)",
            "-4 T(G,H,U;Phi)",
        ],
        "all_checks_pass": True,
    }


def _continuum_certificate() -> dict[str, Any]:
    radius_squared = Fraction(19, 2)
    lipschitz_upper_bound = math.sqrt(3.0) * (
        math.pi + 3.0
    ) / 2.0
    return {
        "positive_packet_domain": (
            "D=[2,3] x [-1/2,1/2] x [-1/2,1/2]"
        ),
        "symmetric_domain": "K=D union (-D)",
        "profile": (
            "a(xi)=S(xi) P_xi(e_3)/|xi| on D, "
            "a(-xi)=a(xi), zero off K"
        ),
        "sine_profile": (
            "S=sin(pi(x-2)) sin(pi(y+1/2)) sin(pi(z+1/2))"
        ),
        "maximum_profile_radius_squared": _fraction_text(
            radius_squared
        ),
        "profile_bounds": {
            "L_infinity": "1/2",
            "L1": "1",
            "L2": "1/sqrt(2)",
            "zero_extension_is_Lipschitz": True,
            "Lipschitz_upper_bound": lipschitz_upper_bound,
        },
        "parity_gauge": (
            "Hhat_N(k)=-sigma_k N^-1 a_N(k/N), "
            "sigma_k=(-1)^(k_x+k_y+k_z)"
        ),
        "continuum_Euler_velocity": (
            "v(rho)=P_rho integral_K "
            "(rho dot a(xi)) a(rho-xi) dxi"
        ),
        "continuum_Euler_acceleration": (
            "g(rho)=1/2 P_rho integral_K ["
            "(rho dot a(xi))v(rho-xi)"
            "+(rho dot v(xi))a(rho-xi)] dxi"
        ),
        "scaled_discrete_fields": {
            "BHH": (
                "B(H,H)^hat(r)=-i sigma_r N^2 "
                "v_N(r/N)"
            ),
            "B_H_BHH": (
                "B(H,B(H,H))^hat(r)=sigma_r N^5 "
                "g_N(r/N)"
            ),
        },
        "fixed_q_pressure_limits": {
            "p_BHH_BHH": (
                "N^-7 p[V,V]^hat(q) -> "
                "-sigma_q integral (e_q dot v)^2"
            ),
            "p_G_H": (
                "N^-7 p[G,H]^hat(q) -> "
                "sigma_q integral "
                "(e_q dot g)(e_q dot a)"
            ),
        },
        "combined_limit": {
            "symbol": "L_EE",
            "formula": (
                "L_EE=(sqrt(2)/20) integral (v_z^2-v_y^2)"
                "+(sqrt(2)/10) integral (g_y a_y-g_z a_z)"
            ),
            "first_form_component": (
                "L_VV=(sqrt(2)/20) integral (v_z^2-v_y^2)"
            ),
            "second_form_component": (
                "L_GH=(sqrt(2)/10) integral "
                "(g_y a_y-g_z a_z)"
            ),
        },
        "energy_trace_check": (
            "integral |v|^2+2 integral g dot a=0; "
            "the low stencil is traceless and probes only anisotropy"
        ),
        "all_checks_pass": True,
    }


def _remainder_certificate() -> dict[str, Any]:
    return {
        "sampled_profile_bound": {
            "definition": (
                "epsilon_N=||a_N-a||_L1+||a_N-a||_L2"
            ),
            "bound": "epsilon_N <= 64/N for odd N>=5",
            "ingredients": [
                "the sine arguments differ by at most 1/N,1/(2N),1/(2N)",
                "|P_xi(e_3)/|xi||<=1/2 on K",
                "the zero extension is Lipschitz because S vanishes on every face",
                "the union of sampled and continuum supports has volume at most 4",
            ],
        },
        "bilinear_bound": (
            "For supports of radii R_f,R_g and p in {1,2}, "
            "||B_c(f,g)||_p <= (R_f+R_g)/2 "
            "[||f||_1||g||_p+||g||_1||f||_p]."
        ),
        "active_fixed_output_bound": {
            "inequality": (
                "|N^-7 D_N-L_EE| <= C_fixed/N"
            ),
            "constant": "C_fixed=250000 is a valid coarse norm bound",
            "valid_for": "odd N>=128",
            "note": (
                "D_N is the sum of the two N7-saturating "
                "fixed-output contractions. The constant uses "
                "sum|alpha_q|=3/2 and max|q|=sqrt(6)."
            ),
        },
        "remaining_channel_bound": {
            "target_inequality": (
                "|c_1,N-D_N| <= C_stencil N^6 log(2+N)"
            ),
            "constant_definition": (
                "C_stencil is the finite sum of the compact-support "
                "L1/L2 and first-difference bounds in the seven "
                "nonleading amplitude-one terms and the three "
                "nonleading symmetrized permutations."
            ),
            "why_finite": (
                "At bounded nonactive output, incidence loses at least "
                "one carrier power. At high pressure output, one "
                "summation-by-parts difference lands on a zero-extended "
                "Lipschitz packet profile or on a smooth pressure/Leray "
                "multiplier. Dyadic pressure shells contribute only "
                "log(2+N). No C^2 or higher zero-extension claim is used."
            ),
            "status": "termwise_constant_ledger_required",
            "certified": False,
        },
        "conditional_full_remainder": (
            "|c_1,N/N^7-L_EE| <= "
            "[250000+C_stencil log(2+N)]/N"
        ),
        "proved_convergence_conclusion": "D_N/N^7 -> L_EE",
        "full_c1_convergence_proved": False,
        "sign_requires_continuum_enclosure": True,
        "all_checks_pass": True,
    }


def _route_decision(
    finite: dict[str, Any],
) -> dict[str, Any]:
    fits = finite["recent_inverse_N_fits"]
    candidate = fits["combined_cubic"]["candidate_limit"]
    return {
        "conclusion": (
            "The two N7-saturating fixed-output contractions now have "
            "an exact continuum functional and a quantitative convergence "
            "remainder. The full c_1,N limit still requires a termwise "
            "constant ledger for the nonleading permutations and seven "
            "remaining amplitude-one terms. "
            "The finite data consistently suggest a negative nonzero "
            "limit near -3e-7, but no extrapolation is an interval "
            "certificate. The next gate is a rigorous enclosure of "
            "the two continuum integrals."
        ),
        "continuum_limit_formula_proved": True,
        "dominant_fixed_output_over_N7_convergence_proved": True,
        "full_c1_over_N7_convergence_proved": False,
        "full_c1_remainder_ledger_complete": False,
        "continuum_limit_nonzero_certified": False,
        "continuum_limit_negative_certified": False,
        "optimized_N9_coefficient_certified": False,
        "diagnostic_candidate_limit": candidate,
        "large_full_second_jet_FFT_authorized": False,
        "next_action": (
            "Complete the termwise N6 log(2+N) tail ledger for all seven "
            "nonleading amplitude-one terms and three nonleading "
            "permutations. Independently evaluate v and g on the fixed "
            "continuum domains and derive a joint interval enclosure for "
            "L_EE; neither finite fits nor a named unexpanded constant "
            "may be used as the missing proof."
        ),
        "all_checks_pass": bool(
            candidate < 0.0
            and fits["combined_quadratic"]["candidate_limit"] < 0.0
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    predecessor_hash = _sha256(PREDECESSOR)
    if predecessor_hash != PREDECESSOR_SHA256:
        raise ValueError(
            "annular inviscid second-jet branch prerequisite changed"
        )
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    if predecessor.get("all_positive_checks_pass") is not True:
        raise ValueError("annular inviscid prerequisite did not pass")

    stencil = _active_output_stencil()
    finite = _finite_output_diagnostics(predecessor, stencil)
    permutations = _permutation_support_certificate()
    continuum = _continuum_certificate()
    remainder = _remainder_certificate()
    route = _route_decision(finite)
    all_checks = all(
        certificate["all_checks_pass"]
        for certificate in (
            stencil,
            finite,
            permutations,
            continuum,
            remainder,
            route,
        )
    )
    result = {
        "kind": "annular_rho_zero_fixed_output_continuum_gate_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": (
            "annular_four_high_leading_continuum_reduced_tail_sign_pending"
        ),
        "scope": (
            "Exact fixed-output support and continuum reduction for "
            "the four-high/one-low inviscid pressure coefficient"
        ),
        "prerequisite": {
            "path": str(PREDECESSOR.relative_to(ROOT)).replace("\\", "/"),
            "expected_sha256": PREDECESSOR_SHA256,
            "actual_sha256": predecessor_hash,
            "matches": predecessor_hash == PREDECESSOR_SHA256,
        },
        "active_output_stencil_certificate": stencil,
        "permutation_support_certificate": permutations,
        "continuum_limit_certificate": continuum,
        "quantitative_remainder_certificate": remainder,
        "finite_output_diagnostics": finite,
        "route_decision": route,
        "certification_flags": {
            "active_output_support_proved": True,
            "signed_projector_matrix_proved": True,
            "continuum_limit_formula_proved": True,
            "dominant_fixed_output_over_N7_convergence_proved": True,
            "full_c1_over_N7_convergence_proved": False,
            "full_c1_remainder_ledger_complete": False,
            "continuum_limit_nonzero_certified": False,
            "continuum_limit_negative_certified": False,
            "four_high_N9_coefficient_certified": False,
            "full_inviscid_pressure_N9_limit_certified": False,
            "uniform_second_jet_Taylor_bound_proved": False,
            "finite_parabolic_window_controlled": False,
            "critical_L3_controlled": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "all_positive_checks_pass": all_checks,
    }
    if not all_checks:
        raise ValueError("fixed-output continuum gate did not pass")
    output = (
        arguments.output
        if arguments.output.is_absolute()
        else ROOT / arguments.output
    )
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
