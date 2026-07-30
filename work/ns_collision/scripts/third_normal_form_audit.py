"""Tree-generated audit for the third heat normal form.

Each Euler differentiation splits one velocity leaf into an ordered bilinear
pair.  Earlier heat denominators retain the receiving frequency of that pair,
while the newest primitive sees the frequencies of all current leaves.  The
partition history below preserves those distinctions exactly.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
from itertools import product
import json
from pathlib import Path
from typing import NamedTuple, TypeAlias

import numpy as np
import sympy as sp


SECOND_SCRIPT = Path(__file__).with_name("second_normal_form_audit.py")
SPEC = importlib.util.spec_from_file_location(
    "third_normal_form_second_helpers", SECOND_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
SECOND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SECOND)


Wave = tuple[int, int, int]
Field = dict[Wave, np.ndarray]
Tree: TypeAlias = int | tuple[str, "Tree", "Tree"]
Partition: TypeAlias = tuple[tuple[int, ...], ...]
Gaussian: TypeAlias = tuple[Fraction, Fraction]
ExactVector: TypeAlias = tuple[Gaussian, Gaussian, Gaussian]
ExactField: TypeAlias = dict[Wave, ExactVector]


G_ZERO: Gaussian = (Fraction(0), Fraction(0))


class NormalFormState(NamedTuple):
    root_slots: tuple[Tree, Tree, Tree]
    denominator_partitions: tuple[Partition, ...]
    leaf_count: int


def _shift_and_split(tree: Tree, selected: int) -> Tree:
    if isinstance(tree, int):
        if tree < selected:
            return tree
        if tree == selected:
            return ("B", selected, selected + 1)
        return tree + 1
    label, left, right = tree
    return (
        label,
        _shift_and_split(left, selected),
        _shift_and_split(right, selected),
    )


def _update_partition(partition: Partition, selected: int) -> Partition:
    updated = []
    for group in partition:
        new_group = []
        for leaf in group:
            if leaf < selected:
                new_group.append(leaf)
            elif leaf == selected:
                new_group.extend((selected, selected + 1))
            else:
                new_group.append(leaf + 1)
        updated.append(tuple(new_group))
    return tuple(updated)


def _split_state(
    state: NormalFormState,
    selected: int,
    append_current_partition: bool,
) -> NormalFormState:
    leaf_count = state.leaf_count + 1
    partitions = tuple(
        _update_partition(partition, selected)
        for partition in state.denominator_partitions
    )
    if append_current_partition:
        partitions += (tuple((leaf,) for leaf in range(leaf_count)),)
    return NormalFormState(
        root_slots=tuple(
            _shift_and_split(tree, selected) for tree in state.root_slots
        ),
        denominator_partitions=partitions,
        leaf_count=leaf_count,
    )


def normal_form_states() -> dict[str, tuple[NormalFormState, ...]]:
    cubic = NormalFormState(
        root_slots=(0, 1, 2),
        denominator_partitions=(((0,), (1,), (2,)),),
        leaf_count=3,
    )
    quartic = tuple(
        _split_state(cubic, selected, append_current_partition=True)
        for selected in range(3)
    )
    quintic = tuple(
        _split_state(state, selected, append_current_partition=True)
        for state in quartic
        for selected in range(4)
    )
    sextic = tuple(
        _split_state(state, selected, append_current_partition=False)
        for state in quintic
        for selected in range(5)
    )
    return {"quartic": quartic, "quintic": quintic, "sextic": sextic}


def _frequency(wave: Wave) -> float:
    return float(np.dot(wave, wave))


def _tree_value(
    tree: Tree,
    waves: tuple[Wave, ...],
    values: tuple[np.ndarray, ...],
    cache: dict[Tree, tuple[Wave, np.ndarray]],
) -> tuple[Wave, np.ndarray]:
    if tree in cache:
        return cache[tree]
    if isinstance(tree, int):
        result = (waves[tree], values[tree])
        cache[tree] = result
        return result
    _, left, right = tree
    left_wave, left_value = _tree_value(left, waves, values, cache)
    right_wave, right_value = _tree_value(right, waves, values, cache)
    output = tuple(
        left_wave[axis] + right_wave[axis] for axis in range(3)
    )
    if output == (0, 0, 0):
        result = (output, np.zeros(3, dtype=complex))
    else:
        result = (
            output,
            -SECOND.QUARTIC._project(
                output,
                1j * np.dot(left_value, right_wave) * right_value,
            ),
        )
    cache[tree] = result
    return result


def _partition_frequency(
    partition: Partition, waves: tuple[Wave, ...]
) -> float:
    total = 0.0
    for group in partition:
        group_wave = tuple(
            sum(waves[leaf][axis] for leaf in group) for axis in range(3)
        )
        total += _frequency(group_wave)
    return total


def _partition_frequency_exact(
    partition: Partition, waves: tuple[Wave, ...]
) -> int:
    total = 0
    for group in partition:
        group_wave = tuple(
            sum(waves[leaf][axis] for leaf in group) for axis in range(3)
        )
        total += sum(entry**2 for entry in group_wave)
    return total


def evaluate_states(
    states: tuple[NormalFormState, ...],
    field: Field,
    heat_scale: float,
    denominator_levels: int,
) -> dict[str, object]:
    if not states:
        raise ValueError("at least one normal-form state is required")
    leaf_count = states[0].leaf_count
    if any(state.leaf_count != leaf_count for state in states):
        raise ValueError("all states must have the same number of leaves")
    items = tuple(field.items())
    total = 0j
    frequency_buckets: dict[int, complex] = {}
    for chosen in product(items, repeat=leaf_count - 1):
        partial_waves = tuple(entry[0] for entry in chosen)
        final_wave = tuple(
            -sum(wave[axis] for wave in partial_waves) for axis in range(3)
        )
        final_value = field.get(final_wave)
        if final_value is None:
            continue
        waves = partial_waves + (final_wave,)
        values = tuple(entry[1] for entry in chosen) + (final_value,)
        for state in states:
            cache: dict[Tree, tuple[Wave, np.ndarray]] = {}
            slots = tuple(
                _tree_value(tree, waves, values, cache)
                for tree in state.root_slots
            )
            if any(np.linalg.norm(value) < 1.0e-14 for _, value in slots):
                continue
            denominator = 1.0
            for partition in state.denominator_partitions[:denominator_levels]:
                denominator *= _partition_frequency(partition, waves)
            strain_wave, strain_value = slots[0]
            second_wave, second_value = slots[1]
            third_wave, third_value = slots[2]
            multiplier = 1.0 - np.exp(
                -heat_scale * _frequency(strain_wave)
            )
            coefficient = 1.0 / denominator * np.einsum(
                "ij,i,j",
                SECOND.QUARTIC._strain(strain_wave, strain_value),
                SECOND.QUARTIC._vorticity(second_wave, second_value),
                SECOND.QUARTIC._vorticity(third_wave, third_value),
            )
            total += multiplier * coefficient
            frequency = int(round(_frequency(strain_wave)))
            frequency_buckets[frequency] = (
                frequency_buckets.get(frequency, 0j) + coefficient
            )
    return {
        "real": float(total.real),
        "imaginary_residual": float(total.imag),
        "frequency_buckets": {
            str(frequency): float(value.real)
            for frequency, value in sorted(frequency_buckets.items())
            if abs(value) > 1.0e-13
        },
        "maximum_bucket_imaginary_residual": max(
            (abs(value.imag) for value in frequency_buckets.values()),
            default=0.0,
        ),
    }


def quintic_transfer_tree(field: Field, heat_scale: float) -> dict[str, object]:
    states = normal_form_states()["quintic"]
    return evaluate_states(states, field, heat_scale, denominator_levels=2)


def quintic_primitive(field: Field, heat_scale: float) -> dict[str, object]:
    states = normal_form_states()["quintic"]
    return evaluate_states(states, field, heat_scale, denominator_levels=3)


def sextic_transfer(field: Field, heat_scale: float) -> dict[str, object]:
    states = normal_form_states()["sextic"]
    return evaluate_states(states, field, heat_scale, denominator_levels=3)


def _g_add(first: Gaussian, second: Gaussian) -> Gaussian:
    return (first[0] + second[0], first[1] + second[1])


def _g_neg(value: Gaussian) -> Gaussian:
    return (-value[0], -value[1])


def _g_multiply(first: Gaussian, second: Gaussian) -> Gaussian:
    return (
        first[0] * second[0] - first[1] * second[1],
        first[0] * second[1] + first[1] * second[0],
    )


def _g_scale(value: Gaussian, scalar: Fraction | int) -> Gaussian:
    return (value[0] * scalar, value[1] * scalar)


def _g_i(value: Gaussian) -> Gaussian:
    return (-value[1], value[0])


def _exact_dot(first: ExactVector, second: ExactVector) -> Gaussian:
    result = G_ZERO
    for first_value, second_value in zip(first, second):
        result = _g_add(result, _g_multiply(first_value, second_value))
    return result


def _exact_dot_wave(value: ExactVector, wave: Wave) -> Gaussian:
    result = G_ZERO
    for entry, frequency in zip(value, wave):
        result = _g_add(result, _g_scale(entry, frequency))
    return result


def _exact_scale_vector(value: ExactVector, scalar: Gaussian) -> ExactVector:
    return tuple(_g_multiply(scalar, entry) for entry in value)  # type: ignore[return-value]


def _exact_project(wave: Wave, value: ExactVector) -> ExactVector:
    frequency = sum(entry**2 for entry in wave)
    correction = _g_scale(_exact_dot_wave(value, wave), Fraction(1, frequency))
    return tuple(
        _g_add(entry, _g_neg(_g_scale(correction, wave_entry)))
        for entry, wave_entry in zip(value, wave)
    )  # type: ignore[return-value]


def _exact_cross_wave(wave: Wave, value: ExactVector) -> ExactVector:
    return (
        _g_add(_g_scale(value[2], wave[1]), _g_neg(_g_scale(value[1], wave[2]))),
        _g_add(_g_scale(value[0], wave[2]), _g_neg(_g_scale(value[2], wave[0]))),
        _g_add(_g_scale(value[1], wave[0]), _g_neg(_g_scale(value[0], wave[1]))),
    )


def _exact_bilinear(
    left_wave: Wave,
    left_value: ExactVector,
    right_wave: Wave,
    right_value: ExactVector,
) -> tuple[Wave, ExactVector]:
    output = tuple(
        left_wave[axis] + right_wave[axis] for axis in range(3)
    )
    if output == (0, 0, 0):
        return output, (G_ZERO, G_ZERO, G_ZERO)
    scalar = _g_i(_exact_dot_wave(left_value, right_wave))
    raw = _exact_scale_vector(right_value, scalar)
    return output, tuple(_g_neg(entry) for entry in _exact_project(output, raw))  # type: ignore[return-value]


def _exact_tree_value(
    tree: Tree,
    waves: tuple[Wave, ...],
    values: tuple[ExactVector, ...],
    cache: dict[Tree, tuple[Wave, ExactVector]],
) -> tuple[Wave, ExactVector]:
    if tree in cache:
        return cache[tree]
    if isinstance(tree, int):
        result = (waves[tree], values[tree])
    else:
        _, left, right = tree
        left_wave, left_value = _exact_tree_value(left, waves, values, cache)
        right_wave, right_value = _exact_tree_value(right, waves, values, cache)
        result = _exact_bilinear(
            left_wave, left_value, right_wave, right_value
        )
    cache[tree] = result
    return result


def _exact_contraction(
    strain_wave: Wave,
    strain_value: ExactVector,
    second_wave: Wave,
    second_value: ExactVector,
    third_wave: Wave,
    third_value: ExactVector,
) -> Gaussian:
    second_vorticity = tuple(
        _g_i(entry) for entry in _exact_cross_wave(second_wave, second_value)
    )
    third_vorticity = tuple(
        _g_i(entry) for entry in _exact_cross_wave(third_wave, third_value)
    )
    first_product = _g_multiply(
        _exact_dot(second_vorticity, strain_value),
        _exact_dot_wave(third_vorticity, strain_wave),
    )
    second_product = _g_multiply(
        _exact_dot_wave(second_vorticity, strain_wave),
        _exact_dot(strain_value, third_vorticity),
    )
    return _g_scale(_g_i(_g_add(first_product, second_product)), Fraction(1, 2))


def exact_frequency_buckets(
    states: tuple[NormalFormState, ...],
    field: ExactField,
    denominator_levels: int,
) -> dict[int, Fraction]:
    leaf_count = states[0].leaf_count
    items = tuple(field.items())
    buckets: dict[int, Gaussian] = {}
    for chosen in product(items, repeat=leaf_count - 1):
        partial_waves = tuple(entry[0] for entry in chosen)
        final_wave = tuple(
            -sum(wave[axis] for wave in partial_waves) for axis in range(3)
        )
        final_value = field.get(final_wave)
        if final_value is None:
            continue
        waves = partial_waves + (final_wave,)
        values = tuple(entry[1] for entry in chosen) + (final_value,)
        for state in states:
            cache: dict[Tree, tuple[Wave, ExactVector]] = {}
            slots = tuple(
                _exact_tree_value(tree, waves, values, cache)
                for tree in state.root_slots
            )
            if any(all(entry == G_ZERO for entry in value) for _, value in slots):
                continue
            denominator = 1
            for partition in state.denominator_partitions[:denominator_levels]:
                denominator *= _partition_frequency_exact(partition, waves)
            coefficient = _g_scale(
                _exact_contraction(
                    slots[0][0],
                    slots[0][1],
                    slots[1][0],
                    slots[1][1],
                    slots[2][0],
                    slots[2][1],
                ),
                Fraction(1, denominator),
            )
            frequency = sum(entry**2 for entry in slots[0][0])
            buckets[frequency] = _g_add(
                buckets.get(frequency, G_ZERO), coefficient
            )
    if any(value[1] != 0 for value in buckets.values()):
        raise ArithmeticError("exact frequency bucket has an imaginary part")
    return {
        frequency: value[0]
        for frequency, value in sorted(buckets.items())
        if value[0] != 0
    }


def _exact_two_mode_field(sign: int) -> ExactField:
    def vector(*entries: int) -> ExactVector:
        return tuple((Fraction(entry), Fraction(0)) for entry in entries)  # type: ignore[return-value]

    first = vector(0, -1, 1)
    second = vector(-1, 1, sign)
    return {
        (1, 0, 0): first,
        (-1, 0, 0): first,
        (1, 1, 0): second,
        (-1, -1, 0): second,
    }


def _exact_sparse_triad_field() -> ExactField:
    def real_vector(*entries: int) -> ExactVector:
        return tuple((Fraction(entry), Fraction(0)) for entry in entries)  # type: ignore[return-value]

    first = real_vector(0, -1, -1)
    second = real_vector(-1, 0, -1)
    third = real_vector(1, -1, 1)
    negative_i_third = tuple(_g_neg(_g_i(entry)) for entry in third)
    positive_i_third = tuple(_g_i(entry) for entry in third)
    return {
        (1, 0, 0): first,
        (-1, 0, 0): first,
        (0, 1, 0): second,
        (0, -1, 0): second,
        (1, 1, 0): negative_i_third,  # type: ignore[dict-item]
        (-1, -1, 0): positive_i_third,  # type: ignore[dict-item]
    }


def exact_two_mode_sextic(sign: int) -> tuple[sp.Expr, dict[int, Fraction]]:
    x = sp.symbols("x", positive=True, real=True)
    buckets = exact_frequency_buckets(
        normal_form_states()["sextic"],
        _exact_two_mode_field(sign),
        denominator_levels=3,
    )
    expression = sp.factor(
        sum(
            sp.Rational(value.numerator, value.denominator) * (1 - x**frequency)
            for frequency, value in buckets.items()
        )
    )
    return expression, buckets


def _sign_variations(coefficients: list[sp.Expr]) -> int:
    signs = [sp.sign(coefficient) for coefficient in coefficients if coefficient != 0]
    return sum(first != second for first, second in zip(signs, signs[1:]))


def _two_mode_field(sign: int) -> Field:
    first = np.asarray([0.0, -1.0, 1.0], dtype=complex)
    second = np.asarray([-1.0, 1.0, float(sign)], dtype=complex)
    return {
        (1, 0, 0): first,
        (-1, 0, 0): first,
        (1, 1, 0): second,
        (-1, -1, 0): second,
    }


def audit(heat_scale: float = 0.5) -> dict[str, object]:
    states = normal_form_states()
    negative_two_mode = _two_mode_field(-1)
    sparse_triad = SECOND._sparse_triad_field()

    quintic_tree = quintic_transfer_tree(sparse_triad, heat_scale)
    quintic_reference = SECOND.quintic_transfer(sparse_triad, heat_scale)
    negative_expression, negative_buckets = exact_two_mode_sextic(-1)
    positive_expression, positive_buckets = exact_two_mode_sextic(1)
    x = next(iter(negative_expression.free_symbols))
    sparse_primitive_buckets = exact_frequency_buckets(
        states["quintic"],
        _exact_sparse_triad_field(),
        denominator_levels=3,
    )
    sparse_primitive_expression = sp.factor(
        sum(
            sp.Rational(value.numerator, value.denominator)
            * (1 - x**frequency)
            for frequency, value in sparse_primitive_buckets.items()
        )
    )
    sparse_primitive_polynomial = sp.Poly(
        sp.cancel(sparse_primitive_expression / (1 - x) ** 2), x
    )
    negative_polynomial = sp.Poly(
        sp.cancel(negative_expression / (1 - x) ** 2), x
    )
    positive_polynomial = sp.Poly(
        sp.cancel(positive_expression / (1 - x) ** 2), x
    )
    positive_derivative = sp.Poly(sp.diff(positive_polynomial.as_expr(), x), x)
    positive_derivative_variations = _sign_variations(
        positive_derivative.all_coeffs()
    )
    negative_value = float(
        sp.N(negative_expression.subs(x, sp.exp(-heat_scale)), 30)
    )
    positive_value = float(
        sp.N(positive_expression.subs(x, sp.exp(-heat_scale)), 30)
    )

    finite_difference_step = 1.0e-6
    sparse_laplacian = SECOND._laplacian(sparse_triad)
    primitive_heat_derivative = (
        quintic_primitive(
            SECOND._affine_field(
                sparse_triad, finite_difference_step, sparse_laplacian
            ),
            heat_scale,
        )["real"]
        - quintic_primitive(
            SECOND._affine_field(
                sparse_triad, -finite_difference_step, sparse_laplacian
            ),
            heat_scale,
        )["real"]
    ) / (2.0 * finite_difference_step)
    heat_identity_residual = primitive_heat_derivative + quintic_reference

    negative_ns_direction = SECOND._affine_field(
        SECOND._euler_bilinear(negative_two_mode, negative_two_mode),
        1.0,
        SECOND._laplacian(negative_two_mode),
    )
    primitive_ns_derivative = (
        quintic_primitive(
            SECOND._affine_field(
                negative_two_mode,
                finite_difference_step,
                negative_ns_direction,
            ),
            heat_scale,
        )["real"]
        - quintic_primitive(
            SECOND._affine_field(
                negative_two_mode,
                -finite_difference_step,
                negative_ns_direction,
            ),
            heat_scale,
        )["real"]
    ) / (2.0 * finite_difference_step)
    negative_quintic = SECOND.quintic_transfer(
        negative_two_mode, heat_scale
    )
    ns_identity_residual = primitive_ns_derivative - (
        -negative_quintic + negative_value
    )

    negative_sign_proof = all(
        coefficient > 0 for coefficient in negative_polynomial.all_coeffs()
    )
    positive_sign_proof = bool(
        positive_derivative_variations == 1
        and positive_derivative.eval(0) < 0
        and positive_derivative.LC() > 0
        and positive_polynomial.eval(0) < 0
        and positive_polynomial.eval(1) < 0
    )

    result: dict[str, object] = {
        "heat_scale": heat_scale,
        "quartic_tree_count": len(states["quartic"]),
        "quintic_tree_count": len(states["quintic"]),
        "sextic_tree_count": len(states["sextic"]),
        "quintic_tree_value": quintic_tree["real"],
        "quintic_reference_value": quintic_reference,
        "quintic_tree_residual": quintic_tree["real"] - quintic_reference,
        "maximum_quintic_imaginary_residual": abs(
            quintic_tree["imaginary_residual"]
        ),
        "negative_two_mode_exact_buckets": {
            str(frequency): str(value)
            for frequency, value in negative_buckets.items()
        },
        "positive_two_mode_exact_buckets": {
            str(frequency): str(value)
            for frequency, value in positive_buckets.items()
        },
        "negative_two_mode_exact_sextic": str(negative_expression),
        "positive_two_mode_exact_sextic": str(positive_expression),
        "sparse_triad_exact_quintic_primitive": str(
            sparse_primitive_expression
        ),
        "negative_two_mode_sextic_at_scale": negative_value,
        "positive_two_mode_sextic_at_scale": positive_value,
        "positive_companion_derivative_sign_variations": (
            positive_derivative_variations
        ),
        "third_primitive_heat_identity_residual": heat_identity_residual,
        "third_normal_form_ns_identity_residual": ns_identity_residual,
        "tree_counts_are_exact": bool(
            len(states["quartic"]) == 3
            and len(states["quintic"]) == 12
            and len(states["sextic"]) == 60
        ),
        "tree_quintic_matches_second_normal_form": abs(
            quintic_tree["real"] - quintic_reference
        )
        < 1.0e-10,
        "third_heat_primitive_identity_verified": abs(heat_identity_residual)
        < 1.0e-8,
        "third_normal_form_evolution_verified": abs(ns_identity_residual)
        < 1.0e-8,
        "sextic_transfer_is_even": all(
            state.leaf_count == 6 for state in states["sextic"]
        ),
        "quintic_primitive_endpoint_is_sign_indefinite": bool(
            all(state.leaf_count == 5 for state in states["quintic"])
            and sparse_primitive_polynomial.eval(1) < 0
            and all(
                coefficient > 0
                for coefficient in sp.diff(
                    sparse_primitive_polynomial.as_expr(), x
                ).as_poly(x).all_coeffs()
            )
        ),
        "negative_channel_sextic_is_positive_all_scales": negative_sign_proof,
        "positive_companion_sextic_is_negative_all_scales": positive_sign_proof,
        "sextic_has_both_signs_at_every_positive_scale": bool(
            negative_sign_proof and positive_sign_proof
        ),
        "imaginary_residuals_are_small": abs(
            quintic_tree["imaginary_residual"]
        )
        < 1.0e-10,
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
