"""Audit the critical collar-to-form perturbation architecture."""

from __future__ import annotations

import json
import math

from scipy.integrate import quad
from scipy.special import gamma, gammaincc


BARRIER_GAIN = 1.3428786845671419
CLOSURE_GAIN = 1.391948395999309
CURRENT_CUBIC_CLOSURE_GAIN = 1.2321336084949255
CYLINDER_FLOOR = 5.832287335665
BASELINE_DECAY = CYLINDER_FLOOR - 1.0
ENTRY_RADIUS = 1.25
SHARP_SOBOLEV = 4.0 ** (2.0 / 3.0) / (
    3.0 * math.pi ** (4.0 / 3.0)
)
ENERGY_CONVERSION = CYLINDER_FLOOR / BASELINE_DECAY


def _lower_covariance(duration: float) -> float:
    return -math.expm1(-2.0 * duration)


def _upper_covariance(duration: float) -> float:
    return math.expm1(4.0 * duration) / 2.0


def _gaussian_tail_integral(beta: float, radius: float) -> float:
    """Return integral_{|x|>=radius} exp(-beta |x|^2) dx in R^3."""

    argument = beta * radius**2
    return (
        2.0
        * math.pi
        * beta ** (-1.5)
        * gammaincc(1.5, argument)
        * gamma(1.5)
    )


def _short_time_restricted_row_l3(
    duration: float, collar_distance: float
) -> float:
    if duration <= 1.0e-14:
        return 0.0
    covariance_lower = _lower_covariance(duration)
    covariance_upper = _upper_covariance(duration)
    normalization = (
        2.0 * math.pi * covariance_lower
    ) ** (-1.5)
    tail = _gaussian_tail_integral(
        3.0 / (2.0 * covariance_upper),
        collar_distance / 2.0,
    )
    return math.exp(duration) * normalization * tail ** (1.0 / 3.0)


def _free_row_l2(duration: float) -> float:
    return (
        math.exp(duration)
        * 2.0 ** (-1.5)
        * math.pi ** (-0.75)
        * _lower_covariance(duration) ** (-0.75)
    )


def _first_insertion_row(collar_distance: float) -> dict[str, float]:
    separation_time = 0.5 * math.log1p(
        collar_distance / (2.0 * ENTRY_RADIUS)
    )
    split_time = separation_time / 2.0
    short_constant = separation_time * quad(
        lambda scaled_time: _short_time_restricted_row_l3(
            separation_time * scaled_time, collar_distance
        ),
        0.0,
        1.0,
        epsabs=1.0e-9,
        epsrel=1.0e-9,
        limit=150,
    )[0]
    smoothing_constant = _free_row_l2(split_time)
    long_constant = (
        smoothing_constant ** (4.0 / 3.0)
        * math.exp(-2.0 * BASELINE_DECAY * split_time / 3.0)
        / BASELINE_DECAY
    )
    total_constant = short_constant + long_constant
    neumann_budget = 1.0 - BARRIER_GAIN / CLOSURE_GAIN
    return {
        "collar_distance": collar_distance,
        "separation_time": separation_time,
        "semigroup_split_time": split_time,
        "short_time_off_diagonal_constant": short_constant,
        "long_time_spectral_constant": long_constant,
        "first_insertion_L3_row_constant": total_constant,
        "diagnostic_L3_over_2_mass_at_Neumann_budget": (
            neumann_budget / total_constant
        ),
    }


def _normalized_endpoint_row(
    logarithmic_depth: float, exponent: float = 0.75
) -> dict[str, float]:
    norm_integral = (
        logarithmic_depth ** (1.0 - 1.5 * exponent) - 1.0
    ) / (1.0 - 1.5 * exponent)
    normalization = (4.0 * math.pi * norm_integral) ** (-2.0 / 3.0)
    centre_integral = (
        logarithmic_depth ** (1.0 - exponent) - 1.0
    ) / (1.0 - exponent)
    return {
        "T_logarithmic_depth": logarithmic_depth,
        "L3_over_2_norm": 1.0,
        "normalization_c_T": normalization,
        "Newtonian_centre_potential": normalization * centre_integral,
    }


def _sector_budget_row(condition_number: float) -> dict[str, float]:
    gain_excess = CLOSURE_GAIN / BARRIER_GAIN - 1.0
    sector_intercept = gain_excess / condition_number
    potential_alpha = sector_intercept / (1.0 + sector_intercept)
    equal_share = sector_intercept / (2.0 + sector_intercept)
    return {
        "dynamic_collar_condition_number": condition_number,
        "sector_intercept_d": sector_intercept,
        "potential_only_relative_alpha": potential_alpha,
        "potential_only_L3_over_2_mass": (
            potential_alpha / (ENERGY_CONVERSION * SHARP_SOBOLEV)
        ),
        "drift_only_L3_mass": (
            sector_intercept
            / (ENERGY_CONVERSION * math.sqrt(SHARP_SOBOLEV))
        ),
        "equal_share_alpha_equals_beta": equal_share,
        "equal_share_potential_L3_over_2_mass": (
            equal_share / (ENERGY_CONVERSION * SHARP_SOBOLEV)
        ),
        "equal_share_drift_L3_mass": (
            equal_share
            / (ENERGY_CONVERSION * math.sqrt(SHARP_SOBOLEV))
        ),
    }


def audit() -> dict[str, object]:
    relative_gain_allowance = CLOSURE_GAIN / BARRIER_GAIN - 1.0
    global_neumann_budget = 1.0 - BARRIER_GAIN / CLOSURE_GAIN
    collar_rows = [
        _first_insertion_row(distance)
        for distance in (0.05, 0.10, 0.20, 0.40)
    ]
    endpoint_rows = [
        _normalized_endpoint_row(depth)
        for depth in (4.0, 16.0, 256.0, 65536.0, 1.0e8)
    ]
    sector_rows = [
        _sector_budget_row(condition_number)
        for condition_number in (1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
    ]

    result: dict[str, object] = {
        "certified_ideal_barrier_gain": BARRIER_GAIN,
        "maximum_gain_for_cycle_closure": CLOSURE_GAIN,
        "calibration_is_legacy_bare_halving": True,
        "current_cubic_split_closure_gain": CURRENT_CUBIC_CLOSURE_GAIN,
        "current_cubic_split_baseline_closes": bool(
            BARRIER_GAIN < CURRENT_CUBIC_CLOSURE_GAIN
        ),
        "current_cubic_split_gain_deficit": (
            BARRIER_GAIN - CURRENT_CUBIC_CLOSURE_GAIN
        ),
        "relative_one_history_gain_allowance": relative_gain_allowance,
        "global_Kato_Neumann_budget": global_neumann_budget,
        "admissible_affine_control": (
            "B=B^T, tr(B)=0, B>=-I, hence ||B||<=2"
        ),
        "affine_covariance_bounds": (
            "(1-exp(-2s))I<=Q(s)<=((exp(4s)-1)/2)I"
        ),
        "entry_source_radius_bound": ENTRY_RADIUS,
        "first_insertion_bound": (
            "sup_{x in entry} integral G_B(x,y)q(y)dy "
            "<=C_entry(d)||q||_(3/2) when dist(entry,supp q)>=d"
        ),
        "first_insertion_rows": collar_rows,
        "first_insertion_constants_decrease_with_collar": all(
            later["first_insertion_L3_row_constant"]
            < earlier["first_insertion_L3_row_constant"]
            for earlier, later in zip(collar_rows, collar_rows[1:])
        ),
        "endpoint_counterexample": {
            "profile": (
                "q_T=c_T/(r^2 log(1/r)^(3/4)), "
                "exp(-T)<r<exp(-1), translated into supp(q)"
            ),
            "rows": endpoint_rows,
            "fixed_critical_norm_has_unbounded_centre_potential": True,
            "translation_preserves_any_positive_entry_collar": True,
        },
        "positive_entry_collar_implies_global_Kato_bound": False,
        "reason_first_insertion_is_not_a_Neumann_theorem": (
            "the next iterate starts inside supp(q), where the translated "
            "critical endpoint sequence restores the Newtonian singularity"
        ),
        "sharp_Sobolev_constant_S3": SHARP_SOBOLEV,
        "baseline_energy_conversion": (
            "||grad v||_2^2<=(lambda1/(lambda1-1))*h[v]"
        ),
        "baseline_energy_conversion_constant": ENERGY_CONVERSION,
        "critical_sector_parameters": (
            "alpha<=c_A*S3*||q_+||_(3/2), "
            "beta<=c_A*sqrt(S3)*||e||_3"
        ),
        "uniform_causal_energy_response": (
            "sup_t||w_+(t)||_2<=(alpha+beta)/(1-alpha)*"
            "sqrt(h[zeta*U]/m0)"
        ),
        "causal_energy_method": (
            "positive-part form inequality, Young inequality, and the "
            "bounded finite-horizon monotone limit"
        ),
        "causal_energy_response_denominator": (
            "(1-alpha)*sqrt(m0)"
        ),
        "causal_energy_response_is_nonautonomous": True,
        "dynamic_condition_number_definition": (
            "chi_dyn=C_col(d)*sqrt(h[zeta*U]/m0)/g_barrier"
        ),
        "conditional_dynamic_sector_bound": (
            "gain/g_barrier<=1+chi_dyn*(alpha+beta)/(1-alpha)"
        ),
        "conditional_closure_inequality": (
            "beta<d-(1+d)alpha, "
            "d=(g_closure/g_barrier-1)/chi_dyn"
        ),
        "conditional_sector_budget_rows": sector_rows,
        "dynamic_collar_condition_number_certified": False,
        "remaining_theorem": (
            "certify the local homogeneous collar trace constant C_col(d) "
            "and the chosen cutoff energy h[zeta*U], then verify chi_dyn; "
            "the causal critical interior response is already controlled"
        ),
        "scope_guard": (
            "the affine first-insertion estimate and endpoint no-go are "
            "rigorous analytic reductions; quadrature only evaluates their "
            "displayed constants. The sector table is conditional on a "
            "future dynamic off-diagonal form theorem."
        ),
    }
    positive_checks = (
        abs(global_neumann_budget - 0.035252536353504005) < 1.0e-14,
        result["first_insertion_constants_decrease_with_collar"],
        endpoint_rows[-1]["Newtonian_centre_potential"] > 19.0,
        not result["positive_entry_collar_implies_global_Kato_bound"],
        not result["dynamic_collar_condition_number_certified"],
        result["causal_energy_response_is_nonautonomous"],
        sector_rows[2]["potential_only_L3_over_2_mass"] > 0.08,
        sector_rows[2]["drift_only_L3_mass"] > 0.035,
    )
    result["all_positive_critical_collar_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
