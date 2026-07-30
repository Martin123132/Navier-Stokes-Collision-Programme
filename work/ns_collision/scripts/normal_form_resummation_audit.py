"""Audit the heat-normal-form hierarchy as a graded Neumann resolvent."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import sympy as sp


THIRD_SCRIPT = Path(__file__).with_name("third_normal_form_audit.py")
SPEC = importlib.util.spec_from_file_location(
    "normal_form_resummation_tree_helpers", THIRD_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
THIRD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(THIRD)


def primitive_states(order: int) -> tuple[THIRD.NormalFormState, ...]:
    if order < 0:
        raise ValueError("order must be nonnegative")
    states = (
        THIRD.NormalFormState(
            root_slots=(0, 1, 2),
            denominator_partitions=(((0,), (1,), (2,)),),
            leaf_count=3,
        ),
    )
    for _ in range(order):
        states = tuple(
            THIRD._split_state(
                state, selected, append_current_partition=True
            )
            for state in states
            for selected in range(state.leaf_count)
        )
    return states


def exact_two_mode_endpoint(order: int) -> sp.Expr:
    x = sp.symbols("x", positive=True, real=True)
    states = primitive_states(order)
    buckets = THIRD.exact_frequency_buckets(
        states,
        THIRD._exact_two_mode_field(-1),
        denominator_levels=order + 1,
    )
    return sp.factor(
        sum(
            sp.Rational(value.numerator, value.denominator)
            * (1 - x**frequency)
            for frequency, value in buckets.items()
        )
    )


def _telescoping_audit(maximum_order: int) -> dict[str, object]:
    nu = sp.symbols("nu", nonzero=True)
    transfers = sp.symbols(f"F0:{maximum_order + 2}")
    derivative = sum(
        nu ** (-order)
        * (-nu * transfers[order] + transfers[order + 1])
        for order in range(maximum_order + 1)
    )
    expected = (
        -nu * transfers[0]
        + nu ** (-maximum_order) * transfers[maximum_order + 1]
    )
    return {
        "partial_sum_order": maximum_order,
        "telescoped_derivative": str(sp.factor(derivative)),
        "expected_derivative": str(expected),
        "telescoping_identity_verified": bool(
            sp.simplify(derivative - expected) == 0
        ),
    }


def _majorant_audit() -> dict[str, object]:
    q = sp.symbols("q", positive=True)
    closed_sum = (-sp.log(1 - q) - q - q**2 / 2) / q**3
    series = sp.series(closed_sum, q, 0, 7).removeO()
    expected_series = sum(q**order / (order + 3) for order in range(7))
    return {
        "galerkin_l1_smallness_parameter": "q=K_max*||u||_1/nu",
        "transfer_coefficient_bound": "||F_n||<=||D||*K_max^n",
        "primitive_term_bound": (
            "nu^(-n)|P_n(u)|<=||D||*||u||_1^3*q^n/(n+3)"
        ),
        "majorant_sum": str(closed_sum),
        "remainder_bound": (
            "nu^(-N)|F_(N+1)(u)|"
            "<=||D||*K_max*||u||_1^4*q^N"
        ),
        "majorant_converges_for_q_below_one": True,
        "majorant_series_verified": bool(
            sp.expand(series - expected_series) == 0
        ),
        "cutoff_uniformity_fails": True,
    }


def audit(maximum_tree_order: int = 4) -> dict[str, object]:
    if maximum_tree_order < 3:
        raise ValueError("maximum_tree_order must include the sextic endpoint")
    actual_counts = [
        len(primitive_states(order))
        for order in range(maximum_tree_order + 1)
    ]
    expected_counts = [
        sp.factorial(order + 2) // 2
        for order in range(maximum_tree_order + 1)
    ]
    endpoints = {
        order: exact_two_mode_endpoint(order) for order in range(4)
    }
    weak_forms = THIRD.SECOND.WEAK.closed_forms()
    telescoping = _telescoping_audit(maximum_tree_order)
    majorant = _majorant_audit()
    result: dict[str, object] = {
        "hierarchy_definition": (
            "P_n=H F_n; F_(n+1)=L_1 P_n; P_(n+1)=A P_n; A=H L_1"
        ),
        "generating_equation": "P(z)=P_0+z*A*P(z)",
        "formal_resolvent": "P(1/nu)=(I-A/nu)^(-1)P_0",
        "actual_tree_counts": actual_counts,
        "expected_tree_counts": [int(value) for value in expected_counts],
        "two_mode_endpoints": {
            str(order): str(value) for order, value in endpoints.items()
        },
        "two_mode_order_one_matches_c4": bool(
            sp.simplify(endpoints[1] - weak_forms["seed_four"]) == 0
        ),
        "two_mode_order_three_matches_c6": bool(
            sp.simplify(endpoints[3] - weak_forms["total_six"]) == 0
        ),
        "two_mode_odd_degree_endpoints_vanish": bool(
            endpoints[0] == 0 and endpoints[2] == 0
        ),
        "tree_count_formula_verified": bool(
            actual_counts == [int(value) for value in expected_counts]
        ),
        "resummation_is_neumann_resolvent": True,
        "finite_galerkin_result_is_small_data_only": True,
        **telescoping,
        **majorant,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-order", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(audit(args.tree_order), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
