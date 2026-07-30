"""Exact finite-Fourier audit for the second heat normal form.

The quartic primitive K_s has two frequency denominators.  The first comes
from the cubic primitive J_s; the second integrates the four original input
modes under the heat semigroup.  Keeping the bilinear Euler inputs polarized
is essential because their squared frequencies do not add to the squared
frequency of the receiving mode.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import sympy as sp


QUARTIC_SCRIPT = Path(__file__).with_name("quartic_transfer_audit.py")
SPEC = importlib.util.spec_from_file_location(
    "second_normal_form_quartic_helpers", QUARTIC_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
QUARTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUARTIC)

WEAK_SCRIPT = Path(__file__).with_name("weak_generated_transfer_audit.py")
WEAK_SPEC = importlib.util.spec_from_file_location(
    "second_normal_form_weak_helpers", WEAK_SCRIPT
)
assert WEAK_SPEC is not None and WEAK_SPEC.loader is not None
WEAK = importlib.util.module_from_spec(WEAK_SPEC)
WEAK_SPEC.loader.exec_module(WEAK)


Wave = tuple[int, int, int]
Field = dict[Wave, np.ndarray]


def _frequency(wave: Wave) -> float:
    return float(np.dot(wave, wave))


def _euler_bilinear(first: Field, second: Field) -> Field:
    result: Field = {}
    for first_wave, first_value in first.items():
        for second_wave, second_value in second.items():
            output = tuple(
                first_wave[axis] + second_wave[axis] for axis in range(3)
            )
            if output == (0, 0, 0):
                continue
            contribution = -QUARTIC._project(
                output,
                1j * np.dot(first_value, second_wave) * second_value,
            )
            result[output] = result.get(
                output, np.zeros(3, dtype=complex)
            ) + contribution
    return {
        wave: value
        for wave, value in result.items()
        if np.linalg.norm(value) > 1.0e-13
    }


def _quartic_form(
    first: Field,
    second: Field,
    third: Field,
    fourth: Field,
    heat_scale: float,
    second_primitive: bool,
) -> float:
    total = 0j
    for first_wave, first_value in first.items():
        for second_wave, second_value in second.items():
            receiving_wave = tuple(
                first_wave[axis] + second_wave[axis]
                for axis in range(3)
            )
            if receiving_wave == (0, 0, 0):
                continue
            euler_value = -QUARTIC._project(
                receiving_wave,
                1j * np.dot(first_value, second_wave) * second_value,
            )
            if np.linalg.norm(euler_value) < 1.0e-14:
                continue
            input_pair_frequency = _frequency(first_wave) + _frequency(
                second_wave
            )
            for third_wave, third_value in third.items():
                fourth_wave = tuple(
                    -receiving_wave[axis] - third_wave[axis]
                    for axis in range(3)
                )
                fourth_value = fourth.get(fourth_wave)
                if fourth_value is None:
                    continue
                receiving_frequency = (
                    _frequency(receiving_wave)
                    + _frequency(third_wave)
                    + _frequency(fourth_wave)
                )
                original_frequency = (
                    input_pair_frequency
                    + _frequency(third_wave)
                    + _frequency(fourth_wave)
                )
                denominator = receiving_frequency
                if second_primitive:
                    denominator *= original_frequency
                fields = (
                    (receiving_wave, euler_value),
                    (third_wave, third_value),
                    (fourth_wave, fourth_value),
                )
                for ordering in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
                    strain_wave, strain_value = fields[ordering[0]]
                    second_vorticity_wave, second_vorticity_value = fields[
                        ordering[1]
                    ]
                    third_vorticity_wave, third_vorticity_value = fields[
                        ordering[2]
                    ]
                    multiplier = 1.0 - np.exp(
                        -heat_scale * _frequency(strain_wave)
                    )
                    total += multiplier / denominator * np.einsum(
                        "ij,i,j",
                        QUARTIC._strain(strain_wave, strain_value),
                        QUARTIC._vorticity(
                            second_vorticity_wave, second_vorticity_value
                        ),
                        QUARTIC._vorticity(
                            third_vorticity_wave, third_vorticity_value
                        ),
                    )
    return float(total.real)


def quartic_primitive(field: Field, heat_scale: float) -> float:
    return _quartic_form(
        field, field, field, field, heat_scale, second_primitive=True
    )


def quartic_transfer(field: Field, heat_scale: float) -> float:
    return _quartic_form(
        field, field, field, field, heat_scale, second_primitive=False
    )


def quintic_transfer(field: Field, heat_scale: float) -> float:
    euler = _euler_bilinear(field, field)
    return sum(
        (
            _quartic_form(
                euler,
                field,
                field,
                field,
                heat_scale,
                second_primitive=True,
            ),
            _quartic_form(
                field,
                euler,
                field,
                field,
                heat_scale,
                second_primitive=True,
            ),
            _quartic_form(
                field,
                field,
                euler,
                field,
                heat_scale,
                second_primitive=True,
            ),
            _quartic_form(
                field,
                field,
                field,
                euler,
                heat_scale,
                second_primitive=True,
            ),
        )
    )


def _laplacian(field: Field) -> Field:
    return {
        wave: -_frequency(wave) * value for wave, value in field.items()
    }


def _affine_field(first: Field, scale: float, second: Field) -> Field:
    waves = set(first) | set(second)
    return {
        wave: first.get(wave, np.zeros(3, dtype=complex))
        + scale * second.get(wave, np.zeros(3, dtype=complex))
        for wave in waves
    }


def heat_derivative(field: Field, heat_scale: float) -> float:
    laplacian = _laplacian(field)
    return sum(
        (
            _quartic_form(
                laplacian,
                field,
                field,
                field,
                heat_scale,
                second_primitive=True,
            ),
            _quartic_form(
                field,
                laplacian,
                field,
                field,
                heat_scale,
                second_primitive=True,
            ),
            _quartic_form(
                field,
                field,
                laplacian,
                field,
                heat_scale,
                second_primitive=True,
            ),
            _quartic_form(
                field,
                field,
                field,
                laplacian,
                heat_scale,
                second_primitive=True,
            ),
        )
    )


def _sparse_triad_field() -> Field:
    first = np.asarray([0.0, -1.0, -1.0], dtype=complex)
    second = np.asarray([-1.0, 0.0, -1.0], dtype=complex)
    third = np.asarray([1.0, -1.0, 1.0], dtype=complex)
    return {
        (1, 0, 0): first,
        (-1, 0, 0): first,
        (0, 1, 0): second,
        (0, -1, 0): second,
        (1, 1, 0): -1j * third,
        (-1, -1, 0): 1j * third,
    }


SymbolicField = dict[Wave, sp.Matrix]


def _symbolic_project(wave: Wave, value: sp.Matrix) -> sp.Matrix:
    vector = sp.Matrix(wave)
    return sp.simplify(
        value - vector * vector.dot(value) / vector.dot(vector)
    )


def _symbolic_euler_bilinear(
    first: SymbolicField, second: SymbolicField
) -> SymbolicField:
    result: SymbolicField = {}
    for first_wave, first_value in first.items():
        for second_wave, second_value in second.items():
            output = tuple(
                first_wave[axis] + second_wave[axis] for axis in range(3)
            )
            if output == (0, 0, 0):
                continue
            contribution = -_symbolic_project(
                output,
                sp.I * first_value.dot(sp.Matrix(second_wave)) * second_value,
            )
            result[output] = result.get(output, sp.zeros(3, 1)) + contribution
    return {
        wave: sp.simplify(value)
        for wave, value in result.items()
        if sp.simplify(value) != sp.zeros(3, 1)
    }


def _symbolic_strain(wave: Wave, value: sp.Matrix) -> sp.Matrix:
    vector = sp.Matrix(wave)
    return sp.I * (value * vector.T + vector * value.T) / 2


def _symbolic_vorticity(wave: Wave, value: sp.Matrix) -> sp.Matrix:
    return sp.I * sp.Matrix(wave).cross(value)


def _symbolic_quartic_form(
    first: SymbolicField,
    second: SymbolicField,
    third: SymbolicField,
    fourth: SymbolicField,
    x: sp.Symbol,
    second_primitive: bool,
) -> sp.Expr:
    total = 0
    for first_wave, first_value in first.items():
        for second_wave, second_value in second.items():
            receiving_wave = tuple(
                first_wave[axis] + second_wave[axis]
                for axis in range(3)
            )
            if receiving_wave == (0, 0, 0):
                continue
            euler_value = -_symbolic_project(
                receiving_wave,
                sp.I * first_value.dot(sp.Matrix(second_wave)) * second_value,
            )
            if euler_value == sp.zeros(3, 1):
                continue
            pair_frequency = sum(entry**2 for entry in first_wave) + sum(
                entry**2 for entry in second_wave
            )
            for third_wave, third_value in third.items():
                fourth_wave = tuple(
                    -receiving_wave[axis] - third_wave[axis]
                    for axis in range(3)
                )
                if fourth_wave not in fourth:
                    continue
                fourth_value = fourth[fourth_wave]
                receiving_frequency = (
                    sum(entry**2 for entry in receiving_wave)
                    + sum(entry**2 for entry in third_wave)
                    + sum(entry**2 for entry in fourth_wave)
                )
                original_frequency = (
                    pair_frequency
                    + sum(entry**2 for entry in third_wave)
                    + sum(entry**2 for entry in fourth_wave)
                )
                denominator = receiving_frequency
                if second_primitive:
                    denominator *= original_frequency
                fields = (
                    (receiving_wave, euler_value),
                    (third_wave, third_value),
                    (fourth_wave, fourth_value),
                )
                for ordering in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
                    strain_wave, strain_value = fields[ordering[0]]
                    second_vorticity_wave, second_vorticity_value = fields[
                        ordering[1]
                    ]
                    third_vorticity_wave, third_vorticity_value = fields[
                        ordering[2]
                    ]
                    multiplier = 1 - x ** sum(
                        entry**2 for entry in strain_wave
                    )
                    contraction = (
                        _symbolic_vorticity(
                            second_vorticity_wave, second_vorticity_value
                        ).T
                        * _symbolic_strain(strain_wave, strain_value)
                        * _symbolic_vorticity(
                            third_vorticity_wave, third_vorticity_value
                        )
                    )[0]
                    total += multiplier * contraction / denominator
    return sp.factor(sp.simplify(sp.expand_complex(total)))


def symbolic_sparse_triad_quintic() -> sp.Expr:
    x = sp.symbols("x", positive=True, real=True)
    first = sp.Matrix([0, -1, -1])
    second = sp.Matrix([-1, 0, -1])
    third = sp.Matrix([1, -1, 1])
    field = {
        (1, 0, 0): first,
        (-1, 0, 0): first,
        (0, 1, 0): second,
        (0, -1, 0): second,
        (1, 1, 0): -sp.I * third,
        (-1, -1, 0): sp.I * third,
    }
    euler = _symbolic_euler_bilinear(field, field)
    return sp.factor(
        sum(
            (
                _symbolic_quartic_form(
                    euler, field, field, field, x, True
                ),
                _symbolic_quartic_form(
                    field, euler, field, field, x, True
                ),
                _symbolic_quartic_form(
                    field, field, euler, field, x, True
                ),
                _symbolic_quartic_form(
                    field, field, field, euler, x, True
                ),
            )
        )
    )


def symbolic_two_mode_primitive() -> sp.Expr:
    x = sp.symbols("x", positive=True, real=True)
    first = sp.Matrix([0, -1, 1])
    second = sp.Matrix([-1, 1, -1])
    field = {
        (1, 0, 0): first,
        (-1, 0, 0): first,
        (1, 1, 0): second,
        (-1, -1, 0): second,
    }
    return _symbolic_quartic_form(field, field, field, field, x, True)


def two_mode_support_selection() -> dict[str, object]:
    seed = {(1, 0, 0), (-1, 0, 0), (1, 1, 0), (-1, -1, 0)}
    generated = set()
    for first in seed:
        for second in seed:
            output = tuple(first[axis] + second[axis] for axis in range(3))
            if output == (0, 0, 0) or first == second:
                continue
            generated.add(output)
    resonant_quintets = []
    for generated_wave in generated:
        for first in seed:
            for second in seed:
                for third in seed:
                    total = tuple(
                        generated_wave[axis]
                        + first[axis]
                        + second[axis]
                        + third[axis]
                        for axis in range(3)
                    )
                    if total == (0, 0, 0):
                        resonant_quintets.append(
                            (generated_wave, first, second, third)
                        )
    return {
        "generated_waves": sorted(generated),
        "resonant_quintet_count": len(resonant_quintets),
        "order_five_support_selection_rule": len(resonant_quintets) == 0,
    }


def audit(heat_scale: float = 0.5) -> dict[str, object]:
    transfer_residuals = []
    heat_residuals = []
    parity_residuals = []
    quintic_values = []
    sparse_field = _sparse_triad_field()
    two_mode = {
        (1, 0, 0): np.asarray([0.0, -1.0, 1.0], dtype=complex),
        (-1, 0, 0): np.asarray([0.0, -1.0, 1.0], dtype=complex),
        (1, 1, 0): np.asarray([-1.0, 1.0, -1.0], dtype=complex),
        (-1, -1, 0): np.asarray([-1.0, 1.0, -1.0], dtype=complex),
    }
    for field in (sparse_field, two_mode):
        direct_transfer = quartic_transfer(field, heat_scale)
        reference_transfer = QUARTIC.evaluate(field, heat_scale)[
            "quartic_transfer"
        ]
        transfer_residuals.append(direct_transfer - reference_transfer)
        heat_residuals.append(
            heat_derivative(field, heat_scale) + direct_transfer
        )
        value = quintic_transfer(field, heat_scale)
        reversed_value = quintic_transfer(
            {wave: -coefficient for wave, coefficient in field.items()},
            heat_scale,
        )
        quintic_values.append(value)
        parity_residuals.append(value + reversed_value)

    sparse_value, two_mode_value = quintic_values
    sparse_euler = _euler_bilinear(sparse_field, sparse_field)
    sparse_ns_direction = _affine_field(
        sparse_euler, 1.0, _laplacian(sparse_field)
    )
    finite_difference_step = 1.0e-6
    finite_difference_derivative = (
        quartic_primitive(
            _affine_field(
                sparse_field,
                finite_difference_step,
                sparse_ns_direction,
            ),
            heat_scale,
        )
        - quartic_primitive(
            _affine_field(
                sparse_field,
                -finite_difference_step,
                sparse_ns_direction,
            ),
            heat_scale,
        )
    ) / (2.0 * finite_difference_step)
    exact_ns_derivative = -quartic_transfer(
        sparse_field, heat_scale
    ) + sparse_value
    ns_derivative_residual = finite_difference_derivative - exact_ns_derivative
    exact_sparse_value = symbolic_sparse_triad_quintic()
    exact_two_mode_primitive = symbolic_two_mode_primitive()
    support_selection = two_mode_support_selection()
    x = sp.symbols("x", positive=True, real=True)
    weak_forms = WEAK.closed_forms()
    two_mode_order_six = weak_forms["total_six"]
    order_six_polynomial = sp.Poly(
        weak_forms["total_six_polynomial"], x
    )
    expected_two_mode_primitive = (
        (1 - x) ** 2 * (x**3 + 2 * x**2 + 3 * x - 11) / 120
    )
    result: dict[str, object] = {
        "heat_scale": heat_scale,
        "field_count": 2,
        "maximum_quartic_evaluator_residual": max(
            abs(value) for value in transfer_residuals
        ),
        "maximum_heat_identity_residual": max(
            abs(value) for value in heat_residuals
        ),
        "maximum_quintic_odd_parity_residual": max(
            abs(value) for value in parity_residuals
        ),
        "navier_stokes_directional_derivative_residual": ns_derivative_residual,
        "audited_quintic_values": quintic_values,
        "sparse_triad_quintic_value": sparse_value,
        "sparse_triad_exact_quintic": str(exact_sparse_value),
        "two_mode_quintic_value": two_mode_value,
        "two_mode_exact_quartic_primitive": str(exact_two_mode_primitive),
        "two_mode_generated_waves": support_selection["generated_waves"],
        "two_mode_resonant_quintet_count": support_selection[
            "resonant_quintet_count"
        ],
        "two_mode_first_remainder_order": 6,
        "two_mode_integrated_order_six_remainder": str(
            sp.factor(two_mode_order_six)
        ),
        "scale_half_integrated_order_six_remainder": float(
            sp.N(two_mode_order_six.subs(x, sp.exp(-sp.Rational(1, 2))), 30)
        ),
        "quartic_evaluator_matches_existing_audit": max(
            abs(value) for value in transfer_residuals
        )
        < 1.0e-9,
        "second_heat_primitive_identity_verified": max(
            abs(value) for value in heat_residuals
        )
        < 1.0e-9,
        "second_normal_form_evolution_verified": abs(ns_derivative_residual)
        < 1.0e-8,
        "quintic_transfer_is_odd": max(
            abs(value) for value in parity_residuals
        )
        < 1.0e-9,
        "quintic_does_not_vanish_generally": exact_sparse_value != 0,
        "sparse_triad_has_nonzero_quintic_transfer": abs(sparse_value)
        > 1.0e-8,
        "two_mode_quintic_cancels": bool(
            abs(two_mode_value) < 1.0e-10
            and support_selection["order_five_support_selection_rule"]
        ),
        "two_mode_primitive_matches_weak_order_four": bool(
            sp.simplify(exact_two_mode_primitive - expected_two_mode_primitive)
            == 0
        ),
        "two_mode_order_six_remainder_is_positive": all(
            coefficient > 0
            for coefficient in order_six_polynomial.all_coeffs()
        ),
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(audit(args.scale), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
