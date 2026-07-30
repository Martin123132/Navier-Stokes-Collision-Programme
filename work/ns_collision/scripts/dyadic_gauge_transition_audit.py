"""Audit dyadic gauge recentering against the radius-halving shrink budget."""

from __future__ import annotations

import json
import math

import sympy as sp


def audit() -> dict[str, object]:
    normalized_coordinate = sp.symbols(
        "normalized_coordinate", real=True
    )
    reynolds, dimension = sp.symbols(
        "R_star dimension", positive=True, real=True
    )
    one_coordinate_exponent_difference = sp.factor(
        (sp.Rational(1, 4) + normalized_coordinate / 2) ** 2
        - normalized_coordinate**2
    )
    critical_point = sp.solve(
        sp.diff(one_coordinate_exponent_difference, normalized_coordinate),
        normalized_coordinate,
    )[0]
    maximum_coordinate_difference = sp.simplify(
        one_coordinate_exponent_difference.subs(
            normalized_coordinate, critical_point
        )
    )
    lower_boundary_difference = sp.simplify(
        one_coordinate_exponent_difference.subs(
            normalized_coordinate, -sp.Rational(1, 2)
        )
    )
    upper_boundary_difference = sp.simplify(
        one_coordinate_exponent_difference.subs(
            normalized_coordinate, sp.Rational(1, 2)
        )
    )

    maximum_log_transition = sp.simplify(
        reynolds * dimension * maximum_coordinate_difference / 4
    )
    radius_halving_shrink_log = -sp.log(2)
    one_history_net_log = sp.simplify(
        maximum_log_transition + radius_halving_shrink_log
    )
    pair_net_log = sp.simplify(2 * one_history_net_log)
    contraction_condition = sp.solve_univariate_inequality(
        one_history_net_log < 0, reynolds
    )

    audited_reynolds = 2.0
    audited_dimension = 3.0
    audited_transition_factor = math.exp(
        audited_reynolds * audited_dimension / 48.0
    )
    audited_one_history_factor = audited_transition_factor / 2.0
    audited_pair_factor = audited_one_history_factor**2
    generation_rows = []
    for generations in (1, 5, 10, 20, 50):
        generation_rows.append(
            {
                "generations": generations,
                "unpaid_transition_product": (
                    audited_transition_factor**generations
                ),
                "shrink_paid_one_history_product": (
                    audited_one_history_factor**generations
                ),
                "shrink_paid_pair_product": (
                    audited_pair_factor**generations
                ),
            }
        )

    result: dict[str, object] = {
        "cell_gauge": "g_Q(x)=exp(-R_star*|x-c_Q|^2/(4*side(Q)^2))",
        "one_coordinate_exponent_difference": str(
            one_coordinate_exponent_difference
        ),
        "maximizing_child_coordinate": str(critical_point),
        "maximum_coordinate_exponent_difference": str(
            maximum_coordinate_difference
        ),
        "lower_boundary_exponent_difference": str(
            lower_boundary_difference
        ),
        "upper_boundary_exponent_difference": str(
            upper_boundary_difference
        ),
        "coordinate_maximum_is_one_twelfth": bool(
            maximum_coordinate_difference == sp.Rational(1, 12)
        ),
        "maximum_parent_child_log_gauge_cost": str(
            maximum_log_transition
        ),
        "maximum_parent_child_gauge_factor": "exp(R_star*dimension/48)",
        "unpaid_cost_can_accumulate_by_generation": bool(
            generation_rows[-1]["unpaid_transition_product"]
            > generation_rows[-2]["unpaid_transition_product"]
        ),
        "radius_halving_requires_reference_growth_factor": 4,
        "radius_halving_shrink_log": str(radius_halving_shrink_log),
        "one_history_net_log_per_true_split": str(one_history_net_log),
        "pair_net_log_per_true_split": str(pair_net_log),
        "one_history_contraction_condition": str(contraction_condition),
        "audited_R_star": audited_reynolds,
        "audited_dimension": int(audited_dimension),
        "audited_transition_factor": audited_transition_factor,
        "audited_one_history_net_factor": audited_one_history_factor,
        "audited_pair_net_factor": audited_pair_factor,
        "true_split_is_contractive_for_R_two_in_3d": bool(
            audited_one_history_factor < 1.0
            and audited_pair_factor < 1.0
        ),
        "generation_rows": generation_rows,
        "conditional_scope": (
            "the shrink payment applies to a history retained in a true "
            "nested core while its physical radius halves"
        ),
        "balance_refinement_warning": (
            "partition-only or balance-driven splits have cross-child "
            "transfer and renewal terms; they cannot inherit the shrink "
            "payment without a separate flux argument"
        ),
        "remaining_transition_gate": (
            "couple parent-child partition flux and replica renewal so paths "
            "crossing child interfaces do not accumulate gauge constants"
        ),
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
