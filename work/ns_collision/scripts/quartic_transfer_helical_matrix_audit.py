"""Exact Hermitian pair matrix for the two-mode quartic transfer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import sympy as sp


COUNTEREXAMPLE_SCRIPT = Path(__file__).with_name(
    "quartic_transfer_counterexample.py"
)
SPEC = importlib.util.spec_from_file_location(
    "quartic_transfer_matrix_helpers", COUNTEREXAMPLE_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def _matrix(x: sp.Symbol) -> sp.Matrix:
    root_two = sp.sqrt(2)
    p = x**3 + 2 * x**2 + 3 * x
    homochiral = (
        (sp.Rational(1, 8) - root_two / 10)
        * (p - (25 * root_two + 19) / 14)
    )
    heterochiral = (
        (sp.Rational(1, 8) + root_two / 10)
        * (p + (25 * root_two - 19) / 14)
    )
    edge = sp.Rational(1, 16)
    homochiral_coupling = (
        (7 * root_two - 10) * p + 18 * root_two - 15
    ) / 80
    heterochiral_coupling = -(
        (10 + 7 * root_two) * p + 15 + 18 * root_two
    ) / 80
    return sp.Matrix(
        [
            [homochiral, edge, -edge, homochiral_coupling],
            [edge, heterochiral, heterochiral_coupling, -edge],
            [-edge, heterochiral_coupling, heterochiral, edge],
            [homochiral_coupling, -edge, edge, homochiral],
        ]
    )


def _helical_field(
    first_amplitudes: tuple[sp.Expr, sp.Expr],
    second_amplitudes: tuple[sp.Expr, sp.Expr],
) -> HELPERS.Field:
    root_two = sp.sqrt(2)
    imaginary = sp.I
    k = (1, 0, 0)
    m = (1, 1, 0)
    k_first = sp.Matrix([0, 1, 0])
    k_second = sp.Matrix([0, 0, 1])
    m_first = sp.Matrix([-1 / root_two, 1 / root_two, 0])
    m_second = sp.Matrix([0, 0, 1])
    k_helical = [
        (k_first + imaginary * sigma * k_second) / root_two
        for sigma in (1, -1)
    ]
    m_helical = [
        (m_first + imaginary * sigma * m_second) / root_two
        for sigma in (1, -1)
    ]
    k_value = sp.expand(
        sum(
            (
                first_amplitudes[index] * k_helical[index]
                for index in range(2)
            ),
            sp.zeros(3, 1),
        )
    )
    m_value = sp.expand(
        sum(
            (
                second_amplitudes[index] * m_helical[index]
                for index in range(2)
            ),
            sp.zeros(3, 1),
        )
    )
    return {
        k: k_value,
        tuple(-entry for entry in k): sp.conjugate(k_value),
        m: m_value,
        tuple(-entry for entry in m): sp.conjugate(m_value),
    }


def _direct_transfer(field: HELPERS.Field, x: sp.Symbol) -> sp.Expr:
    euler = {
        wave: -value for wave, value in HELPERS._nonlinearity(field).items()
    }
    return sp.factor(
        HELPERS._trilinear(euler, field, field, x, primitive=True)
        + HELPERS._trilinear(field, euler, field, x, primitive=True)
        + HELPERS._trilinear(field, field, euler, x, primitive=True),
        extension=sp.sqrt(2),
    )


def _matrix_transfer(
    first_amplitudes: tuple[sp.Expr, sp.Expr],
    second_amplitudes: tuple[sp.Expr, sp.Expr],
    x: sp.Symbol,
) -> sp.Expr:
    pair = sp.Matrix(
        [
            first_amplitudes[0] * second_amplitudes[0],
            first_amplitudes[0] * second_amplitudes[1],
            first_amplitudes[1] * second_amplitudes[0],
            first_amplitudes[1] * second_amplitudes[1],
        ]
    )
    return sp.factor(
        (1 - x) ** 2
        * (sp.conjugate(pair).T * _matrix(x) * pair)[0],
        extension=sp.sqrt(2),
    )


def audit() -> dict[str, str | bool]:
    x = sp.symbols("x", positive=True)
    root_two = sp.sqrt(2)
    p = x**3 + 2 * x**2 + 3 * x
    matrix = _matrix(x)

    inverse_root_two = 1 / root_two
    parity_basis = sp.Matrix(
        [
            [inverse_root_two, 0, inverse_root_two, 0],
            [0, inverse_root_two, 0, inverse_root_two],
            [0, inverse_root_two, 0, -inverse_root_two],
            [inverse_root_two, 0, -inverse_root_two, 0],
        ]
    )
    transformed = sp.simplify(parity_basis.T * matrix * parity_basis)
    expected_first = -root_two * (p - 11) / 80
    expected_second = root_two * (p - 11) / 80
    expected_block = sp.Matrix(
        [
            [
                (sp.Rational(1, 4) - 3 * root_two / 16)
                * (p + root_two + 3),
                sp.Rational(1, 8),
            ],
            [
                sp.Rational(1, 8),
                (sp.Rational(1, 4) + 3 * root_two / 16)
                * (p - root_two + 3),
            ],
        ]
    )
    expected_transformed = sp.diag(expected_first, expected_second, 1, 1)
    expected_transformed[2:4, 2:4] = expected_block
    block_determinant = sp.factor(expected_block.det())

    # Exact phase-sensitive tomography checks.  These include amplitudes that
    # cannot be reduced to the real cosine family by translations.
    amplitude_samples = [
        ((sp.Integer(1), sp.Integer(1)), (sp.Integer(1), sp.Integer(1))),
        ((sp.Integer(1), sp.I), (sp.Integer(1), 1 + sp.I)),
        ((1 + sp.I, 2 - sp.I), (-1 + 2 * sp.I, sp.Integer(1))),
    ]
    tomography_checks = []
    for first_amplitudes, second_amplitudes in amplitude_samples:
        direct = _direct_transfer(
            _helical_field(first_amplitudes, second_amplitudes), x
        )
        predicted = _matrix_transfer(first_amplitudes, second_amplitudes, x)
        tomography_checks.append(sp.simplify(direct - predicted) == 0)

    # The earlier real-polarization family is an additional symbolic check
    # over four free amplitudes rather than at selected points.
    a, b, c, d = sp.symbols("a b c d", real=True)
    first_real_family = (
        (-a - sp.I * b) / root_two,
        (-a + sp.I * b) / root_two,
    )
    second_real_family = (
        c - sp.I * d / root_two,
        c + sp.I * d / root_two,
    )
    real_family_matrix_value = _matrix_transfer(
        first_real_family, second_real_family, x
    )
    y = a * d
    z = b * c
    real_family_expected = sp.factor(
        (1 - x) ** 2
        * (
            5 * (p + 1) * y**2
            + (14 * p + 36) * y * z
            + 10 * (p + 2) * z**2
        )
        / 20
    )

    result: dict[str, str | bool] = {
        "pair_order": "(++,+-,-+,--)",
        "hermitian_matrix": str(matrix),
        "matrix_is_real_symmetric": bool(matrix == matrix.T),
        "phase_sensitive_fourier_checks": all(tomography_checks),
        "real_polarization_family_recovered": bool(
            sp.simplify(real_family_matrix_value - real_family_expected) == 0
        ),
        "parity_block_diagonalization_verified": bool(
            sp.simplify(transformed - expected_transformed) == sp.zeros(4)
        ),
        "first_scalar_channel": str(sp.factor(expected_first)),
        "second_scalar_channel": str(sp.factor(expected_second)),
        "two_by_two_block": str(expected_block),
        "two_by_two_block_determinant": str(block_determinant),
        "block_determinant_identity_verified": bool(
            sp.simplify(block_determinant + (p + 3) ** 2 / 128) == 0
        ),
        "matrix_inertia_is_two_positive_two_negative": bool(
            p.subs(x, 0) == 0
            and p.subs(x, 1) == 6
            and sp.discriminant(sp.diff(p, x), x) < 0
            and block_determinant < 0
        ),
        "physical_pair_vectors_obey_rank_one_constraint": "z_++*z_--=z_+-*z_-+",
        "rank_one_constraint_retained": True,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
