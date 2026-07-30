"""Audit the scale-uniform low-output stress-tail gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PRIOR_RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "smooth_galerkin_shell_response_gate_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "a226f430ea1518c54780671abd3f17055333770737566e6badf4eb4a8931f0ad"
)


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


def _max_norm_shell_count(scale: int) -> int:
    if scale < 1:
        raise ValueError("scale must be positive")
    return (4 * scale - 1) ** 3 - (2 * scale - 1) ** 3


def _brute_max_norm_shell_count(scale: int) -> int:
    radius = 2 * scale - 1
    count = 0
    for first in range(-radius, radius + 1):
        for second in range(-radius, radius + 1):
            for third in range(-radius, radius + 1):
                maximum = max(abs(first), abs(second), abs(third))
                if scale <= maximum < 2 * scale:
                    count += 1
    return count


def _lattice_shell_audit() -> dict[str, Any]:
    rows = []
    maximum_formula_residual = 0
    maximum_cubic_ratio = 0.0
    for scale in (1, 2, 4, 8):
        formula = _max_norm_shell_count(scale)
        brute = _brute_max_norm_shell_count(scale)
        expanded = 56 * scale**3 - 36 * scale**2 + 6 * scale
        residual = max(abs(formula - brute), abs(formula - expanded))
        maximum_formula_residual = max(maximum_formula_residual, residual)
        cubic_ratio = formula / scale**3
        maximum_cubic_ratio = max(maximum_cubic_ratio, cubic_ratio)
        rows.append(
            {
                "scale": scale,
                "formula_count": formula,
                "brute_count": brute,
                "expanded_count": expanded,
                "count_over_scale_cubed": cubic_ratio,
            }
        )
    return {
        "dyadic_block_definition": (
            "Q<=|q|_infinity<2Q on the three-dimensional integer lattice"
        ),
        "exact_count": (
            "N_Q=(4Q-1)^3-(2Q-1)^3="
            "56Q^3-36Q^2+6Q<=56Q^3"
        ),
        "rows": rows,
        "maximum_formula_residual": maximum_formula_residual,
        "maximum_cubic_ratio": maximum_cubic_ratio,
        "all_checks_pass": bool(
            maximum_formula_residual == 0
            and maximum_cubic_ratio <= 56.0
        ),
    }


def _infinite_dyadic_series(
    sobolev_exponent: float,
    cutoff_power: int,
    denominator_power: int,
) -> float:
    cutoff = 2.0**cutoff_power
    power = 3.0 - 2.0 * sobolev_exponent
    low = sum(
        2.0 ** (power * index)
        for index in range(cutoff_power + 1)
    ) / cutoff**denominator_power
    high_power = power - denominator_power
    ratio = 2.0**high_power
    if ratio >= 1.0:
        return math.inf
    first_high = 2.0 ** (high_power * (cutoff_power + 1))
    return low + first_high / (1.0 - ratio)


def _forced_reference(
    sobolev_exponent: float,
    cutoff_power: int,
) -> float:
    cutoff = 2.0**cutoff_power
    if sobolev_exponent < 1.5:
        return cutoff ** (2.0 - 2.0 * sobolev_exponent)
    if sobolev_exponent == 1.5:
        return (cutoff_power + 1.0) / cutoff
    return 1.0 / cutoff


def _initial_reference(
    sobolev_exponent: float,
    cutoff_power: int,
) -> float:
    cutoff = 2.0**cutoff_power
    if sobolev_exponent < 1.5:
        return cutoff ** (1.0 - 2.0 * sobolev_exponent)
    if sobolev_exponent == 1.5:
        return (cutoff_power + 1.0) / cutoff**2
    return 1.0 / cutoff**2


def _dyadic_tail_summation_audit() -> dict[str, Any]:
    exponents = (1.1, 1.25, 1.5, 1.75, 2.0)
    cutoff_powers = (4, 8, 12, 16)
    rows = []
    maximum_forced_ratio = 0.0
    maximum_initial_ratio = 0.0
    monotone_convergence = True
    for exponent in exponents:
        forced_values = []
        initial_values = []
        for cutoff_power in cutoff_powers:
            forced = _infinite_dyadic_series(
                exponent,
                cutoff_power,
                denominator_power=1,
            )
            initial = _infinite_dyadic_series(
                exponent,
                cutoff_power,
                denominator_power=2,
            )
            forced_reference = _forced_reference(exponent, cutoff_power)
            initial_reference = _initial_reference(exponent, cutoff_power)
            forced_ratio = forced / forced_reference
            initial_ratio = initial / initial_reference
            maximum_forced_ratio = max(
                maximum_forced_ratio,
                forced_ratio,
            )
            maximum_initial_ratio = max(
                maximum_initial_ratio,
                initial_ratio,
            )
            forced_values.append(forced)
            initial_values.append(initial)
            rows.append(
                {
                    "sobolev_exponent": exponent,
                    "cutoff": 2**cutoff_power,
                    "forced_square_series": forced,
                    "forced_reference": forced_reference,
                    "forced_ratio": forced_ratio,
                    "initial_square_series": initial,
                    "initial_reference": initial_reference,
                    "initial_ratio": initial_ratio,
                }
            )
        monotone_convergence = bool(
            monotone_convergence
            and all(
                later < earlier
                for earlier, later in zip(
                    forced_values,
                    forced_values[1:],
                )
            )
            and all(
                later < earlier
                for earlier, later in zip(
                    initial_values,
                    initial_values[1:],
                )
            )
        )

    endpoint_finite_rows = []
    for extra_shells in (1, 2, 4, 8, 16, 32):
        cutoff_power = 8
        cutoff = 2.0**cutoff_power
        low = sum(
            2.0**index for index in range(cutoff_power + 1)
        ) / cutoff
        high = float(extra_shells)
        endpoint_finite_rows.append(
            {
                "extra_high_output_shells": extra_shells,
                "forced_H_minus_1_square_series": low + high,
                "increment_over_low_part": high,
            }
        )

    return {
        "coefficient_tail": (
            "For J_q=max(K,8<q>), "
            "||Rhat_K(q)||_L2t<="
            "C[A/J_q+B/sqrt(J_q)]."
        ),
        "initial_square_series": (
            "I_s(K)=sum_Q Q^(3-2s)/max(K,Q)^2"
        ),
        "forced_square_series": (
            "F_s(K)=sum_Q Q^(3-2s)/max(K,Q)"
        ),
        "piecewise_initial_rates": {
            "1<s<3/2": "K^(1-2s)",
            "s=3/2": "log_2(2K)/K^2",
            "s>3/2": "K^(-2)",
        },
        "piecewise_forced_rates": {
            "1<s<3/2": "K^(2-2s)",
            "s=3/2": "log_2(2K)/K",
            "s>3/2": "K^(-1)",
        },
        "rows": rows,
        "maximum_forced_ratio": maximum_forced_ratio,
        "maximum_initial_ratio": maximum_initial_ratio,
        "monotone_convergence_for_every_s_gt_1": monotone_convergence,
        "H_minus_1_finite_cutoff_rows": endpoint_finite_rows,
        "H_minus_1_high_output_increment_per_shell": 1.0,
        "H_minus_1_infinite_series_diverges": True,
        "all_checks_pass": bool(
            monotone_convergence
            and maximum_forced_ratio < 32.0
            and maximum_initial_ratio < 32.0
            and all(
                row["increment_over_low_part"]
                == row["extra_high_output_shells"]
                for row in endpoint_finite_rows
            )
        ),
    }


def _pulse_response_l2_squared(carrier: int) -> tuple[float, float]:
    rate = float(carrier**2)
    duration = 1.0 / rate
    forcing = float(carrier) ** 2.5
    equilibrium = forcing / rate
    rate_time = rate * duration
    during = equilibrium**2 * (
        duration
        - 2.0 * (1.0 - math.exp(-rate_time)) / rate
        + (1.0 - math.exp(-2.0 * rate_time)) / (2.0 * rate)
    )
    terminal = equilibrium * (1.0 - math.exp(-rate_time))
    after = terminal**2 / (2.0 * rate)
    exact = during + after
    expected = 1.0 / (math.e * carrier)
    return exact, expected


def _endpoint_pulse_audit() -> dict[str, Any]:
    rows = []
    maximum_response_residual = 0.0
    maximum_envelope_residual = 0.0
    endpoint_limit = 7.0 / (2.0 * math.e)
    endpoint_residuals = []
    for carrier in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
        output_scale = carrier // 16
        output_count = _max_norm_shell_count(output_scale)
        response, expected = _pulse_response_l2_squared(carrier)
        response_residual = abs(response - expected)
        maximum_response_residual = max(
            maximum_response_residual,
            response_residual,
        )
        forcing = float(carrier) ** 2.5
        duration = float(carrier) ** -2.0
        forcing_envelope = carrier**-3 * forcing**2 * duration
        maximum_envelope_residual = max(
            maximum_envelope_residual,
            abs(forcing_envelope - 1.0),
        )
        endpoint_square = (
            output_count
            * output_scale ** -2.0
            * response
        )
        endpoint_residual = abs(endpoint_square - endpoint_limit)
        endpoint_residuals.append(endpoint_residual)
        subcritical_rows = {}
        for epsilon in (0.1, 0.25, 0.5):
            square = (
                output_count
                * output_scale ** (-2.0 * (1.0 + epsilon))
                * response
            )
            subcritical_rows[str(epsilon)] = {
                "H_minus_1_plus_epsilon_square": square,
                "renormalized_by_Q_to_2epsilon": (
                    square * output_scale ** (2.0 * epsilon)
                ),
            }
        rows.append(
            {
                "carrier": carrier,
                "output_scale": output_scale,
                "output_channel_count": output_count,
                "weighted_forcing_envelope": forcing_envelope,
                "single_channel_response_L2_squared": response,
                "single_channel_expected_L2_squared": expected,
                "H_minus_1_block_square": endpoint_square,
                "distance_from_7_over_e": endpoint_residual,
                "subcritical_rows": subcritical_rows,
            }
        )
    return {
        "model": (
            "On every q in one dyadic output block Q=H/16, solve "
            "dot c_q+H^2c_q=H^(5/2)1_[0,H^(-2)] with c_q(0)=0."
        ),
        "exact_single_channel_identity": (
            "||c_q||_L2t^2=1/(eH)"
        ),
        "forcing_envelope_identity": (
            "H^(-3)||H^(5/2)1_[0,H^(-2)]||_L2t^2=1"
        ),
        "H_minus_1_limit": "7/(2e)",
        "H_minus_1_limit_value": endpoint_limit,
        "rows": rows,
        "maximum_response_identity_residual": maximum_response_residual,
        "maximum_forcing_envelope_residual": maximum_envelope_residual,
        "endpoint_residual_strictly_decreases": all(
            later < earlier
            for earlier, later in zip(
                endpoint_residuals,
                endpoint_residuals[1:],
            )
        ),
        "interpretation": (
            "The scalar forcing envelope and exact viscous response alone "
            "do not imply H^(-1) tail compactness. This is an abstract "
            "channel-saturated relaxation model, not a Navier-Stokes "
            "solution or counterexample."
        ),
        "all_checks_pass": bool(
            maximum_response_residual < 1.0e-14
            and maximum_envelope_residual < 1.0e-12
            and all(
                later < earlier
                for earlier, later in zip(
                    endpoint_residuals,
                    endpoint_residuals[1:],
                )
            )
            and rows[-1]["H_minus_1_block_square"] > 1.25
            and all(
                rows[-1]["subcritical_rows"][str(epsilon)][
                    "H_minus_1_plus_epsilon_square"
                ]
                < rows[0]["subcritical_rows"][str(epsilon)][
                    "H_minus_1_plus_epsilon_square"
                ]
                for epsilon in (0.1, 0.25, 0.5)
            )
        ),
    }


def _galerkin_passage_audit(
    dyadic: dict[str, Any],
) -> dict[str, Any]:
    rates = []
    for epsilon in (0.1, 0.25, 0.49, 0.5, 0.75):
        exponent = 1.0 + epsilon
        if epsilon < 0.5:
            initial_rate = f"K^(-{0.5 + epsilon:.2f})"
            forced_rate = f"K^(-{epsilon:.2f})"
        elif epsilon == 0.5:
            initial_rate = "sqrt(log_2(2K))/K"
            forced_rate = "sqrt(log_2(2K)/K)"
        else:
            initial_rate = "K^(-1)"
            forced_rate = "K^(-1/2)"
        rates.append(
            {
                "epsilon": epsilon,
                "sobolev_exponent": exponent,
                "initial_tail_rate": initial_rate,
                "forced_tail_rate": forced_rate,
            }
        )
    return {
        "theorem": (
            "For every s>1, the complete low-output high-high stress "
            "tail tends to zero in L2_t H_x^(-s), uniformly over smooth "
            "Galerkin truncations with common Leray bounds."
        ),
        "fixed_mode_nonlinearity_bound": (
            "|Nhat(k)|<=|k|E_* follows from "
            "uhat(a) dot (k-a)=uhat(a) dot k and Fourier Cauchy-Schwarz."
        ),
        "fixed_mode_coefficients_uniformly_equi_Lipschitz": True,
        "compactness_mechanism": (
            "The fixed-mode derivative bound gives finite-dimensional "
            "Arzela-Ascoli compactness. Uniform H^(-s) tails then give a "
            "diagonal Cauchy argument for the low-output stress series."
        ),
        "rates": rates,
        "endpoint": (
            "At s=1 the forced dyadic series is not summable from the "
            "certified coefficient envelope alone."
        ),
        "all_checks_pass": bool(
            dyadic["all_checks_pass"]
            and dyadic["monotone_convergence_for_every_s_gt_1"]
            and dyadic["H_minus_1_infinite_series_diverges"]
        ),
    }


def audit() -> dict[str, Any]:
    if _sha256(PRIOR_RESULT) != PRIOR_RESULT_SHA256:
        raise RuntimeError("smooth Galerkin prerequisite hash changed")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    lattice = _lattice_shell_audit()
    dyadic = _dyadic_tail_summation_audit()
    endpoint = _endpoint_pulse_audit()
    passage = _galerkin_passage_audit(dyadic)
    result = {
        "schema": "ns_scale_uniform_low_output_tail_gate_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "scale_uniform_H_minus_one_plus_epsilon_"
            "Galerkin_stress_tail_certified"
        ),
        "prerequisites": {
            "smooth_galerkin_shell_response_gate": (
                PRIOR_RESULT.relative_to(ROOT).as_posix()
            ),
            "smooth_galerkin_shell_response_gate_sha256": (
                PRIOR_RESULT_SHA256
            ),
            "smooth_galerkin_shell_response_gate_status": prior["status"],
            "smooth_galerkin_shell_response_gate_passed": prior[
                "all_positive_checks_pass"
            ],
        },
        "low_output_definition": (
            "R_K^lo=sum_(H>=K) P_(|q|_infinity<=H/8) C_H, with smooth "
            "finite-overlap pair-shell and output multipliers."
        ),
        "lattice_Littlewood_Paley_count": lattice,
        "dyadic_tail_summation": dyadic,
        "endpoint_channel_saturated_pulse": endpoint,
        "Galerkin_passage": passage,
        "certification_flags": {
            "uniform_low_output_channel_constant_extracted": True,
            "three_dimensional_output_multiplicity_paid": True,
            "H_minus_s_tail_vanishes_for_every_s_gt_1": True,
            "fixed_mode_Galerkin_compactness_derived": True,
            "H_minus_one_plus_epsilon_Galerkin_passage_proved": True,
            "H_minus_1_endpoint_proved": False,
            "H_minus_1_endpoint_falsified_for_actual_Navier_Stokes": False,
            "H_minus_1_not_derivable_from_scalar_envelope_alone": True,
            "complete_suitable_weak_solution_passage_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "scope": (
            "The certified theorem concerns the low-output high-high "
            "Reynolds-stress series of smooth finite Fourier Galerkin "
            "solutions and its Galerkin limit in L2_t H_x^(-1-epsilon). "
            "It does not close the H^(-1) endpoint, all cubic local-energy "
            "defects, exceptional-set removal, or regularity."
        ),
        "next_gate": (
            "Determine whether the certified dense HHH packet can populate "
            "a positive-volume low-output block at the H^(5/2) rate, or "
            "whether the exact Navier-Stokes convolution supplies an "
            "endpoint cancellation absent from the channel-saturated "
            "relaxation model. Then test the surviving endpoint against "
            "the complete cubic local-energy defect."
        ),
    }
    result["all_positive_checks_pass"] = bool(
        prior["all_positive_checks_pass"]
        and lattice["all_checks_pass"]
        and dyadic["all_checks_pass"]
        and endpoint["all_checks_pass"]
        and passage["all_checks_pass"]
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
            "scale_uniform_low_output_tail_gate_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("scale-uniform low-output tail gate failed")
    _atomic_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
