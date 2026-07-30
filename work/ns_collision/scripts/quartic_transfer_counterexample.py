"""Exact two-mode counterexample to quartic-transfer positivity."""

from __future__ import annotations

import json

import sympy as sp


Wave = tuple[int, int, int]
Field = dict[Wave, sp.Matrix]


def _add(field: Field, wave: Wave, value: sp.Matrix) -> None:
    field[wave] = field.get(wave, sp.zeros(3, 1)) + value


def _project(wave: Wave, value: sp.Matrix) -> sp.Matrix:
    vector = sp.Matrix(wave)
    return sp.simplify(
        value - vector * vector.dot(value) / vector.dot(vector)
    )


def _nonlinearity(field: Field) -> Field:
    advection: Field = {}
    for first_wave, first_value in field.items():
        for second_wave, second_value in field.items():
            output = tuple(
                first_wave[index] + second_wave[index] for index in range(3)
            )
            _add(
                advection,
                output,
                sp.I * first_value.dot(sp.Matrix(second_wave)) * second_value,
            )
    return {
        wave: _project(wave, value)
        for wave, value in advection.items()
        if wave != (0, 0, 0)
    }


def _vorticity(wave: Wave, value: sp.Matrix) -> sp.Matrix:
    return sp.I * sp.Matrix(wave).cross(value)


def _strain(wave: Wave, value: sp.Matrix) -> sp.Matrix:
    vector = sp.Matrix(wave)
    return sp.I * (value * vector.T + vector * value.T) / 2


def _trilinear(
    first: Field,
    second: Field,
    third: Field,
    x: sp.Symbol,
    primitive: bool,
) -> sp.Expr:
    total = 0
    for first_wave, first_value in first.items():
        first_frequency = sum(entry**2 for entry in first_wave)
        multiplier = 1 - x**first_frequency
        for second_wave, second_value in second.items():
            third_wave = tuple(
                -first_wave[index] - second_wave[index] for index in range(3)
            )
            if third_wave not in third:
                continue
            total_frequency = (
                first_frequency
                + sum(entry**2 for entry in second_wave)
                + sum(entry**2 for entry in third_wave)
            )
            denominator = total_frequency if primitive else 1
            total += (
                multiplier
                / denominator
                * (
                    _vorticity(second_wave, second_value).T
                    * _strain(first_wave, first_value)
                    * _vorticity(third_wave, third[third_wave])
                )[0]
            )
    return sp.factor(sp.simplify(sp.expand_complex(total)))


def audit() -> dict[str, str | bool]:
    x = sp.symbols("x", positive=True)
    a, b, c, d = sp.symbols("a b c d", real=True)

    # Fourier support is +/-k and +/-m, where
    # k=(1,0,0), u_k=(0,-a,b),
    # m=(1,1,0), u_m=(-c,c,d).
    # Both coefficients are exactly transverse to their wave vectors.
    field = {
        (1, 0, 0): sp.Matrix([0, -a, b]),
        (-1, 0, 0): sp.Matrix([0, -a, b]),
        (1, 1, 0): sp.Matrix([-c, c, d]),
        (-1, -1, 0): sp.Matrix([-c, c, d]),
    }
    euler_direction = {
        wave: -value for wave, value in _nonlinearity(field).items()
    }

    def transfer_from(euler_part: Field) -> sp.Expr:
        return sp.factor(
            _trilinear(euler_part, field, field, x, primitive=True)
            + _trilinear(field, euler_part, field, x, primitive=True)
            + _trilinear(field, field, euler_part, x, primitive=True)
        )

    derived_transfer = transfer_from(euler_direction)
    difference_euler = {
        wave: value
        for wave, value in euler_direction.items()
        if sum(entry**2 for entry in wave) == 1
    }
    sum_euler = {
        wave: value
        for wave, value in euler_direction.items()
        if sum(entry**2 for entry in wave) == 5
    }
    difference_mode_transfer = transfer_from(difference_euler)
    sum_mode_transfer = transfer_from(sum_euler)
    initial_defect = _trilinear(field, field, field, x, primitive=False)
    initial_primitive = _trilinear(field, field, field, x, primitive=True)

    p = x**3 + 2 * x**2 + 3 * x
    y = a * d
    z = b * c
    first_diagonal = 5 * (p + 1)
    cross_coefficient = 14 * p + 36
    second_diagonal = 10 * (p + 2)
    expected_general_transfer = sp.factor(
        (1 - x) ** 2
        * (
            first_diagonal * y**2
            + cross_coefficient * y * z
            + second_diagonal * z**2
        )
        / 20
    )
    general_transfer = derived_transfer

    pair_matrix = sp.Matrix(
        [
            [first_diagonal, cross_coefficient / 2],
            [cross_coefficient / 2, second_diagonal],
        ]
    )
    pair_matrix_determinant = sp.factor(pair_matrix.det())
    sum_mode_matrix = sp.Matrix(
        [
            [5 * (p + 3), cross_coefficient / 2],
            [cross_coefficient / 2, second_diagonal],
        ]
    )
    sum_mode_matrix_determinant = sp.factor(sum_mode_matrix.det())
    expected_difference_mode_transfer = -(1 - x) ** 2 * y**2 / 2

    # The integer-amplitude choice a=b=c=1, d=-1 has y=-1,z=1.
    counterexample = sp.factor(
        general_transfer.subs({a: 1, b: 1, c: 1, d: -1})
    )
    expected_counterexample = sp.factor(
        (1 - x) ** 2 * (p - 11) / 20
    )
    positive_companion = sp.factor(
        general_transfer.subs({a: 1, b: 1, c: 1, d: 1})
    )

    # For x=exp(-s) and s>0, 0<x<1.  The exact derivative check below
    # gives 0<p<6, so p-11<0.  The determinant p^2-102p-224 decreases
    # from -224 on this interval and is therefore also strictly negative.
    p_derivative = sp.factor(sp.diff(p, x))
    determinant_in_p = sp.Symbol("q") ** 2 - 102 * sp.Symbol("q") - 224
    determinant_derivative_at_six = sp.diff(
        determinant_in_p, sp.Symbol("q")
    ).subs(sp.Symbol("q"), 6)
    exact_sign_gates = bool(
        p.subs(x, 0) == 0
        and p.subs(x, 1) == 6
        and sp.discriminant(p_derivative, x) < 0
        and sp.LC(sp.Poly(p_derivative, x)) > 0
        and determinant_in_p.subs(sp.Symbol("q"), 0) == -224
        and determinant_derivative_at_six < 0
    )
    divergence_free = all(
        sp.Matrix(wave).dot(value) == 0 for wave, value in field.items()
    )
    reality_condition = all(
        field[tuple(-entry for entry in wave)] == sp.conjugate(value)
        for wave, value in field.items()
    )
    result: dict[str, str | bool] = {
        "heat_variable": "x=exp(-s), 0<x<1",
        "input_wave_k": "(1, 0, 0)",
        "input_coefficient_k": "(0, -a, b)",
        "input_wave_m": "(1, 1, 0)",
        "input_coefficient_m": "(-c, c, d)",
        "general_two_mode_transfer": str(general_transfer),
        "fourier_derivation_matches_closed_form": bool(
            sp.simplify(general_transfer - expected_general_transfer) == 0
        ),
        "initial_cubic_defect": str(initial_defect),
        "initial_cubic_primitive": str(initial_primitive),
        "missing_mode_is_created_from_zero_defect": bool(
            initial_defect == 0 and initial_primitive == 0
        ),
        "pair_variables": "y=a*d, z=b*c",
        "pair_matrix": str(pair_matrix),
        "pair_matrix_determinant": str(pair_matrix_determinant),
        "pair_matrix_is_indefinite_for_all_positive_scales": bool(
            sp.simplify(
                pair_matrix_determinant - (p**2 - 102 * p - 224)
            )
            == 0
        ),
        "difference_receiving_frequency": "|m-k|^2=1",
        "difference_mode_transfer": str(difference_mode_transfer),
        "difference_mode_is_negative_semidefinite": bool(
            sp.simplify(
                difference_mode_transfer - expected_difference_mode_transfer
            )
            == 0
        ),
        "sum_receiving_frequency": "|m+k|^2=5",
        "sum_mode_transfer": str(sum_mode_transfer),
        "receiving_mode_split_verified": bool(
            sp.simplify(
                derived_transfer
                - difference_mode_transfer
                - sum_mode_transfer
            )
            == 0
        ),
        "sum_mode_pair_matrix_determinant": str(
            sum_mode_matrix_determinant
        ),
        "sum_mode_is_also_indefinite_for_positive_scales": bool(
            sp.simplify(
                sum_mode_matrix_determinant - (p - 6) * (p + 4)
            )
            == 0
            and exact_sign_gates
        ),
        "counterexample_coefficients": "a=b=c=1, d=-1",
        "counterexample_transfer": str(counterexample),
        "counterexample_formula_verified": bool(
            sp.simplify(counterexample - expected_counterexample) == 0
        ),
        "counterexample_is_negative_for_all_positive_scales": exact_sign_gates,
        "positive_companion_transfer": str(positive_companion),
        "same_family_has_both_signs": bool(
            sp.simplify(
                positive_companion
                - (1 - x) ** 2 * (29 * p + 61) / 20
            )
            == 0
            and exact_sign_gates
        ),
        "input_field_is_real_and_divergence_free": bool(
            divergence_free and reality_condition
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
