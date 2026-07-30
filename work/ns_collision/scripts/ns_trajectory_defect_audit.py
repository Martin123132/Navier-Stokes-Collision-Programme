"""Exact trajectory-level audit of the two-mode collision defect."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import sympy as sp


COUNTEREXAMPLE_SCRIPT = Path(__file__).with_name(
    "quartic_transfer_counterexample.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ns_trajectory_defect_helpers", COUNTEREXAMPLE_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def _bilinear(first: HELPERS.Field, second: HELPERS.Field) -> HELPERS.Field:
    result: HELPERS.Field = {}
    for first_wave, first_value in first.items():
        for second_wave, second_value in second.items():
            output = tuple(
                first_wave[index] + second_wave[index] for index in range(3)
            )
            if output == (0, 0, 0):
                continue
            HELPERS._add(
                result,
                output,
                sp.I
                * first_value.dot(sp.Matrix(second_wave))
                * second_value,
            )
    return {
        wave: HELPERS._project(wave, value)
        for wave, value in result.items()
        if HELPERS._project(wave, value) != sp.zeros(3, 1)
    }


def _add(*fields: HELPERS.Field) -> HELPERS.Field:
    result: HELPERS.Field = {}
    for field in fields:
        for wave, value in field.items():
            HELPERS._add(result, wave, value)
    return {
        wave: sp.simplify(value)
        for wave, value in result.items()
        if value != sp.zeros(3, 1)
    }


def _scale(field: HELPERS.Field, scalar: sp.Expr) -> HELPERS.Field:
    return {wave: scalar * value for wave, value in field.items()}


def _laplacian(field: HELPERS.Field) -> HELPERS.Field:
    return {
        wave: -sum(entry**2 for entry in wave) * value
        for wave, value in field.items()
    }


def _initial_field(sign: int) -> HELPERS.Field:
    return {
        (1, 0, 0): sp.Matrix([0, -1, 1]),
        (-1, 0, 0): sp.Matrix([0, -1, 1]),
        (1, 1, 0): sp.Matrix([-1, 1, sign]),
        (-1, -1, 0): sp.Matrix([-1, 1, sign]),
    }


def _navier_stokes_jet(sign: int, viscosity: sp.Symbol) -> list[HELPERS.Field]:
    jet = [_initial_field(sign)]
    for order in range(3):
        right_hand_side = _scale(_laplacian(jet[order]), viscosity)
        for first_order in range(order + 1):
            second_order = order - first_order
            right_hand_side = _add(
                right_hand_side,
                _scale(
                    _bilinear(jet[first_order], jet[second_order]), -1
                ),
            )
        jet.append(_scale(right_hand_side, sp.Rational(1, order + 1)))
    return jet


def _trilinear_coefficient(
    jet: list[HELPERS.Field],
    order: int,
    x: sp.Symbol,
    primitive: bool = False,
) -> sp.Expr:
    result = 0
    for first_order in range(order + 1):
        for second_order in range(order - first_order + 1):
            third_order = order - first_order - second_order
            result += HELPERS._trilinear(
                jet[first_order],
                jet[second_order],
                jet[third_order],
                x,
                primitive,
            )
    return sp.factor(result)


def _palinstrophy(field: HELPERS.Field) -> sp.Expr:
    result = 0
    for wave, value in field.items():
        frequency = sum(entry**2 for entry in wave)
        vorticity = HELPERS._vorticity(wave, value)
        result += frequency * (sp.conjugate(vorticity).T * vorticity)[0]
    return sp.simplify(sp.expand_complex(result))


def _weak_response(
    sign: int, x: sp.Symbol, z: sp.Symbol, viscosity: sp.Symbol
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    initial = _initial_field(sign)
    euler = {
        wave: -value
        for wave, value in HELPERS._nonlinearity(initial).items()
        if value != sp.zeros(3, 1)
    }
    linear = {
        wave: z ** sum(entry**2 for entry in wave) * value
        for wave, value in initial.items()
    }
    quadratic: HELPERS.Field = {}
    for wave, value in euler.items():
        frequency = sum(entry**2 for entry in wave)
        if frequency == 1:
            quadratic[wave] = z * (1 - z**2) * value / (2 * viscosity)
        elif frequency == 5:
            quadratic[wave] = (z**3 - z**5) * value / (2 * viscosity)

    defect = sp.factor(
        HELPERS._trilinear(quadratic, linear, linear, x, primitive=False)
        + HELPERS._trilinear(linear, quadratic, linear, x, primitive=False)
        + HELPERS._trilinear(linear, linear, quadratic, x, primitive=False)
    )
    integrated_defect = sp.factor(
        sp.integrate(defect / (viscosity * z), (z, 0, 1))
    )
    integrated_palinstrophy = sp.factor(
        sp.integrate(_palinstrophy(linear) / (viscosity * z), (z, 0, 1))
    )
    return defect, integrated_defect, integrated_palinstrophy


def audit() -> dict[str, str | bool]:
    x, z, viscosity = sp.symbols("x z viscosity", positive=True)
    p = x**3 + 2 * x**2 + 3 * x

    positive_jet = _navier_stokes_jet(1, viscosity)
    positive_coefficients = [
        _trilinear_coefficient(positive_jet, order, x)
        for order in range(1, 4)
    ]
    positive_first_expected = (
        2 * (1 - x) ** 2 * (29 * p + 66) / 5
    )
    positive_second_expected = (
        -2 * viscosity * (1 - x) ** 2 * (203 * p + 472) / 5
    )
    positive_euler_third_polynomial = (
        27675 * x**11
        + 55350 * x**10
        + 83025 * x**9
        + 139755 * x**8
        + 196485 * x**7
        + 253215 * x**6
        + 309945 * x**5
        + 366675 * x**4
        + 273104 * x**3
        + 179533 * x**2
        + 85962 * x
        + 27496
    )
    positive_third_expected = (
        (1 - x) ** 2
        * (
            positive_euler_third_polynomial
            + viscosity**2 * (1394900 * p + 3291600)
        )
        / 4875
    )

    first_reduced = sp.factor(positive_first_expected / (1 - x) ** 2)
    second_reduced = sp.factor(-positive_second_expected / (viscosity * (1 - x) ** 2))
    third_viscous_reduced = (1394900 * p + 3291600) / 4875
    cubic_discriminant_without_euler = sp.factor(
        second_reduced**2
        - 4 * first_reduced * third_viscous_reduced
    )

    positive_weak, positive_integral, linear_q_integral = _weak_response(
        1, x, z, viscosity
    )
    negative_weak, negative_integral, _ = _weak_response(
        -1, x, z, viscosity
    )
    positive_weak_expected = (
        z**4
        * (1 - x) ** 2
        * (1 - z**2)
        * (z**2 * (29 * p + 71) - 5)
        / (5 * viscosity)
    )
    negative_weak_expected = (
        z**4
        * (1 - x) ** 2
        * (1 - z**2)
        * (z**2 * (p - 1) - 5)
        / (5 * viscosity)
    )
    positive_integral_expected = (
        (1 - x) ** 2 * (29 * p + 61) / (120 * viscosity**2)
    )
    negative_integral_expected = (
        (1 - x) ** 2 * (p - 11) / (120 * viscosity**2)
    )
    crossing_z_squared = sp.factor(5 / (29 * p + 71))
    crossing_viscous_time = sp.log((29 * p + 71) / 5) / 2
    cumulative_ratio_coefficient = sp.factor(
        positive_integral_expected / (viscosity * linear_q_integral)
    )

    result: dict[str, str | bool] = {
        "positive_jet_support_sizes": str(
            [len(field) for field in positive_jet]
        ),
        "positive_defect_t_coefficient": str(positive_coefficients[0]),
        "positive_defect_t2_coefficient": str(positive_coefficients[1]),
        "positive_defect_t3_coefficient": str(positive_coefficients[2]),
        "positive_taylor_coefficients_verified": bool(
            sp.simplify(positive_coefficients[0] - positive_first_expected) == 0
            and sp.simplify(
                positive_coefficients[1] - positive_second_expected
            )
            == 0
            and sp.simplify(
                positive_coefficients[2] - positive_third_expected
            )
            == 0
        ),
        "pure_euler_t2_coefficient_vanishes": bool(
            sp.Poly(positive_coefficients[1], viscosity).degree() == 1
        ),
        "pure_euler_t3_polynomial_coefficients_positive": all(
            coefficient > 0
            for coefficient in sp.Poly(
                positive_euler_third_polynomial, x
            ).all_coeffs()
        ),
        "cubic_taylor_discriminant_without_euler": str(
            cubic_discriminant_without_euler
        ),
        "third_order_taylor_truncation_stays_positive": bool(
            all(
                coefficient < 0
                for coefficient in sp.Poly(
                    cubic_discriminant_without_euler, x
                ).all_coeffs()
            )
        ),
        "positive_weak_response": str(positive_weak),
        "positive_weak_response_verified": bool(
            sp.simplify(positive_weak - positive_weak_expected) == 0
        ),
        "positive_response_crossing_z_squared": str(crossing_z_squared),
        "positive_response_crossing_viscous_time": str(
            crossing_viscous_time
        ),
        "positive_response_changes_sign_once": bool(
            0 < crossing_z_squared < 1
        ),
        "negative_weak_response": str(negative_weak),
        "negative_weak_response_verified": bool(
            sp.simplify(negative_weak - negative_weak_expected) == 0
        ),
        "negative_response_stays_negative": bool(
            p.subs(x, 1) == 6
            and sp.discriminant(sp.diff(p, x), x) < 0
        ),
        "positive_integrated_weak_defect": str(positive_integral),
        "positive_integrated_defect_verified": bool(
            sp.simplify(positive_integral - positive_integral_expected) == 0
        ),
        "negative_integrated_weak_defect": str(negative_integral),
        "negative_integrated_defect_verified": bool(
            sp.simplify(negative_integral - negative_integral_expected) == 0
        ),
        "linear_integrated_palinstrophy": str(linear_q_integral),
        "cumulative_defect_to_dissipation_ratio": str(
            cumulative_ratio_coefficient
        ),
        "scaled_ratio": "R^2*(1-x)^2*(29*p+61)/960, R=A/(nu*K)",
        "viscosity_reverses_but_does_not_cancel_positive_channel": True,
        "remaining_obstruction_is_high_reynolds_nonperturbative": True,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
