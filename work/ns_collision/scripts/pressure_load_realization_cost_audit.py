"""Audit quantitative costs of realizing compatible pressure loads."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from fourier_pressure_load_surjectivity_audit import (
    STENCIL,
    SUBSET_WAVES,
    VERTICES,
    _add,
    _block_specification,
    _dot,
    _loads_from_transport,
    _low_transport_modes,
    _negate,
)
from pressure_frame_pairing_audit import _build_spectral_fields
from signed_projected_replica_generator_audit import (
    _evaluate_velocity_grid,
)


ROOT = Path(__file__).resolve().parents[3]
Wave = tuple[int, int, int]
Field = dict[Wave, np.ndarray]
TARGET_MOMENTS = {
    wave: (
        Fraction(27, 32)
        if sum(wave) == 1
        else Fraction(9, 8)
    )
    for wave in SUBSET_WAVES
}
EXPECTED_LOADS = [
    (225, -45, -27, -9)[sum(value == -1 for value in vertex)]
    / 256.0
    for vertex in VERTICES
]


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


def _fraction(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _norm_squared(vector: tuple[int, int, int]) -> int:
    return _dot(vector, vector)


def _target_transport(subset_wave: Wave) -> Fraction:
    return (
        TARGET_MOMENTS[subset_wave]
        * 2 ** (sum(subset_wave) - 1)
    )


def _scaling_audit() -> dict[str, Any]:
    load_size, partition_frequency = sp.symbols(
        "B m",
        positive=True,
    )
    amplitude = (load_size / partition_frequency) ** sp.Rational(1, 3)
    l2_cost = sp.simplify(amplitude**2)
    h1_cost = sp.simplify(
        partition_frequency**2 * amplitude**2
    )
    l3_cost = sp.simplify(amplitude**3)
    return {
        "coscaling": (
            "For integer m>=1, u_(a,m)(x)=a u(mx), "
            "p_(a,m)(x)=a^2p(mx), and Phi_(v,m)(x)=Phi_v(mx)"
        ),
        "load_scaling": "b_m[u_(a,m)]=a^3 m b_1[u]",
        "velocity_L2_squared_scaling": "a^2",
        "velocity_H1_seminorm_squared_scaling": "a^2 m^2",
        "velocity_L3_cubed_scaling": "a^3",
        "fixed_load_amplitude": str(amplitude),
        "fixed_load_L2_squared": str(l2_cost),
        "fixed_load_H1_seminorm_squared": str(h1_cost),
        "fixed_load_L3_cubed": str(l3_cost),
        "amplitude_homogeneity": {
            "L2_squared_for_tau_b": "tau^(2/3)",
            "H1_squared_for_tau_b": "tau^(2/3)",
            "L3_cubed_for_tau_b": "tau",
        },
        "interpretation": (
            "The critical dimensionless realization cost is "
            "m||u||_3^3/|b|. It is invariant under amplitude and integer "
            "spatial coscaling."
        ),
        "all_checks_pass": (
            l2_cost
            == load_size ** sp.Rational(2, 3)
            / partition_frequency ** sp.Rational(2, 3)
            and h1_cost
            == load_size ** sp.Rational(2, 3)
            * partition_frequency ** sp.Rational(4, 3)
            and l3_cost == load_size / partition_frequency
        ),
    }


def _normalized_block(
    subset_wave: Wave,
    scale: int,
) -> dict[str, Any]:
    block = _block_specification(subset_wave, scale)
    third = block["third_polarization"]
    third_norm_squared = _norm_squared(third)
    coupling = block["coupling"]
    normalized_coupling_squared = (
        coupling**2 / third_norm_squared
    )
    normalized_coupling = (
        abs(float(coupling)) / math.sqrt(third_norm_squared)
    )
    target = _target_transport(subset_wave)
    product = float(target) / normalized_coupling
    l2_amplitude = product ** (1.0 / 3.0)
    wave_norms = (
        math.sqrt(_norm_squared(block["first_wave"])),
        math.sqrt(_norm_squared(block["second_wave"])),
        math.sqrt(_norm_squared(block["third_wave"])),
    )
    h1_common = (
        product * math.prod(wave_norms)
    ) ** (1.0 / 3.0)
    h1_amplitudes = tuple(
        h1_common / norm for norm in wave_norms
    )
    l2_minimum = 6.0 * product ** (2.0 / 3.0)
    h1_minimum = (
        6.0
        * product ** (2.0 / 3.0)
        * math.prod(wave_norms) ** (2.0 / 3.0)
    )
    l2_optimal_h1 = (
        2.0
        * l2_amplitude**2
        * sum(norm**2 for norm in wave_norms)
    )
    return {
        **block,
        "third_norm_squared": third_norm_squared,
        "normalized_coupling_squared": (
            normalized_coupling_squared
        ),
        "normalized_coupling": normalized_coupling,
        "target_transport": target,
        "amplitude_product": product,
        "L2_optimal_equal_amplitude": l2_amplitude,
        "H1_optimal_amplitudes": h1_amplitudes,
        "L2_squared_minimum": l2_minimum,
        "H1_squared_minimum": h1_minimum,
        "L2_optimal_H1_squared": l2_optimal_h1,
        "wave_norms": wave_norms,
    }


def _single_block_optimization_audit() -> dict[str, Any]:
    product, first, second, third = sp.symbols(
        "P x y z",
        positive=True,
    )
    frequency_first, frequency_second, frequency_third = sp.symbols(
        "A B C",
        positive=True,
    )
    equal_value = product ** sp.Rational(1, 3)
    l2_minimum = sp.simplify(
        2
        * (first**2 + second**2 + third**2)
        .subs(
            {
                first: equal_value,
                second: equal_value,
                third: equal_value,
            }
        )
    )
    common = (
        product
        * frequency_first
        * frequency_second
        * frequency_third
    ) ** sp.Rational(1, 3)
    h1_substitution = {
        first: common / frequency_first,
        second: common / frequency_second,
        third: common / frequency_third,
    }
    h1_minimum = sp.simplify(
        2
        * (
            frequency_first**2 * first**2
            + frequency_second**2 * second**2
            + frequency_third**2 * third**2
        ).subs(h1_substitution)
    )
    product_residual = sp.simplify(
        sp.prod(h1_substitution[value] for value in (first, second, third))
        - product
    )
    return {
        "normalized_block_constraint": (
            "|kappa|xyz=gamma, so xyz=P=gamma/|kappa|"
        ),
        "real_field_L2_squared": "2(x^2+y^2+z^2)",
        "exact_L2_minimum": str(l2_minimum),
        "L2_optimizer": "x=y=z=P^(1/3)",
        "real_field_H1_squared": (
            "2(|a|^2x^2+|b|^2y^2+|k|^2z^2)"
        ),
        "exact_H1_minimum": str(h1_minimum),
        "H1_optimizer": (
            "|a|x=|b|y=|k|z="
            "(P|a||b||k|)^(1/3)"
        ),
        "H1_product_residual": str(product_residual),
        "critical_L3_upper_bound": (
            "For the L2 optimizer, ||u||_infinity<=6P^(1/3), "
            "hence ||u||_3^3<=||u||_infinity||u||_2^2<=36P."
        ),
        "all_checks_pass": (
            l2_minimum == 6 * product ** sp.Rational(2, 3)
            and h1_minimum
            == 6
            * product ** sp.Rational(2, 3)
            * (
                frequency_first
                * frequency_second
                * frequency_third
            )
            ** sp.Rational(2, 3)
            and product_residual == 0
        ),
    }


def _symbolic_coupling_limits() -> dict[str, Any]:
    carrier = sp.symbols("N", positive=True)
    rows = []
    for subset_wave in SUBSET_WAVES:
        first_wave = sp.Matrix((carrier, 0, 0))
        second_wave = sp.Matrix((0, 2 * carrier, 0))
        target = sp.Matrix(subset_wave)
        third_wave = target - first_wave - second_wave
        pair_wave = first_wave + second_wave
        first_polarization = sp.Matrix((0, 1, 0))
        second_polarization = sp.Matrix((1, 0, 0))
        third_polarization = (
            third_wave.dot(third_wave) * pair_wave
            - third_wave.dot(pair_wave) * third_wave
        )
        pair_sums = (
            first_wave + second_wave,
            first_wave + third_wave,
            second_wave + third_wave,
        )
        coupling = sp.factor(
            -2
            * sum(
                (
                    wave.dot(first_polarization)
                    * wave.dot(second_polarization)
                    * wave.dot(third_polarization)
                    / wave.dot(wave)
                )
                for wave in pair_sums
            )
        )
        normalized_squared = sp.factor(
            coupling**2 / third_polarization.dot(third_polarization)
        )
        limit_squared = sp.simplify(
            sp.limit(normalized_squared, carrier, sp.oo)
        )
        rows.append(
            {
                "subset_wave": list(subset_wave),
                "normalized_coupling_squared": str(
                    normalized_squared
                ),
                "limit_normalized_coupling_squared": str(
                    limit_squared
                ),
                "limit_normalized_coupling": math.sqrt(
                    float(limit_squared)
                ),
                "limit_is_strictly_positive": bool(limit_squared > 0),
            }
        )
    return {
        "rows": rows,
        "minimum_limit_normalized_coupling": min(
            row["limit_normalized_coupling"] for row in rows
        ),
        "interpretation": (
            "Every normalized geometric coupling tends to a nonzero "
            "constant as its carrier tends to infinity. Fixed low loads "
            "therefore require bounded coefficient amplitudes."
        ),
        "all_checks_pass": all(
            row["limit_is_strictly_positive"] for row in rows
        ),
    }


def _uniform_lacunary_support_audit() -> dict[str, Any]:
    leading_modes = []
    owner = {}
    correction = {}
    for block_index in range(7):
        factor = 4**block_index
        subset_wave = SUBSET_WAVES[block_index]
        entries = {
            "first": (factor, 0, 0),
            "second": (0, 2 * factor, 0),
            "third": (-factor, -2 * factor, 0),
        }
        for label, wave in entries.items():
            base_correction = (
                subset_wave if label == "third" else (0, 0, 0)
            )
            for sign in (-1, 1):
                signed = tuple(sign * value for value in wave)
                leading_modes.append(signed)
                owner[signed] = (block_index, label, sign)
                correction[signed] = tuple(
                    sign * value for value in base_correction
                )

    zero_rows = []
    invalid = []
    for indices in itertools.combinations_with_replacement(
        range(len(leading_modes)),
        3,
    ):
        selected = tuple(leading_modes[index] for index in indices)
        output = tuple(
            sum(wave[direction] for wave in selected)
            for direction in range(3)
        )
        if output != (0, 0, 0):
            continue
        owners = [owner[wave] for wave in selected]
        correction_output = tuple(
            sum(correction[wave][direction] for wave in selected)
            for direction in range(3)
        )
        expected_correction = tuple(
            owners[0][2] * value
            for value in SUBSET_WAVES[owners[0][0]]
        )
        valid = (
            len({value[0] for value in owners}) == 1
            and sorted(value[1] for value in owners)
            == ["first", "second", "third"]
            and len({value[2] for value in owners}) == 1
            and correction_output == expected_correction
        )
        row = {
            "modes": [list(wave) for wave in selected],
            "owners": [list(value) for value in owners],
            "correction_output": list(correction_output),
            "expected_correction_output": list(expected_correction),
            "valid_single_block_zero": valid,
        }
        zero_rows.append(row)
        if not valid:
            invalid.append(row)

    zero_pair_rows = []
    invalid_zero_pairs = []
    for indices in itertools.combinations_with_replacement(
        range(len(leading_modes)),
        2,
    ):
        selected = tuple(leading_modes[index] for index in indices)
        output = tuple(
            sum(wave[direction] for wave in selected)
            for direction in range(3)
        )
        if output != (0, 0, 0):
            continue
        correction_output = tuple(
            sum(correction[wave][direction] for wave in selected)
            for direction in range(3)
        )
        valid = (
            selected[1] == _negate(selected[0])
            and correction_output == (0, 0, 0)
        )
        row = {
            "modes": [list(wave) for wave in selected],
            "owners": [list(owner[wave]) for wave in selected],
            "correction_output": list(correction_output),
            "valid_opposite_mode_pair": valid,
        }
        zero_pair_rows.append(row)
        if not valid:
            invalid_zero_pairs.append(row)

    return {
        "leading_signed_mode_count": len(leading_modes),
        "leading_zero_triple_count": len(zero_rows),
        "invalid_leading_zero_triple_count": len(invalid),
        "leading_zero_triples": zero_rows,
        "leading_zero_pair_count": len(zero_pair_rows),
        "invalid_leading_zero_pair_count": len(invalid_zero_pairs),
        "leading_zero_pairs": zero_pair_rows,
        "cubic_finite_correction_bound_per_coordinate": 3,
        "quadratic_finite_correction_bound_per_coordinate": 2,
        "carrier_floor": 8,
        "uniform_cubic_argument": (
            "Every actual mode is M times its integer leading mode plus a "
            "correction with coordinates at most one. If a triple has "
            "nonzero leading sum, M>=8 leaves some output coordinate at "
            "least 5 in magnitude. If its leading sum is zero, the finite "
            "certificate says it is one signed block triple and its output "
            "is exactly the intended +/- subset wave."
        ),
        "uniform_quadratic_argument": (
            "A pair with nonzero leading sum has some output coordinate at "
            "least M-2>=6 in magnitude. Every zero-leading pair consists "
            "of one mode and its exact conjugate opposite, whose actual "
            "sum is zero. Thus |grad u|^2 has no nonzero partition-stencil "
            "mode for every M>=8."
        ),
        "all_checks_pass": (
            len(leading_modes) == 42
            and len(zero_rows) == 14
            and not invalid
            and len(zero_pair_rows) == 21
            and not invalid_zero_pairs
        ),
    }


def _quadratic_stencil_silence(field: Field) -> dict[str, Any]:
    occupied = tuple(field)
    low_pairs = []
    for first_index, first_wave in enumerate(occupied):
        for second_wave in occupied[first_index:]:
            output = _add(first_wave, second_wave)
            if output in STENCIL:
                low_pairs.append(
                    {
                        "first": list(first_wave),
                        "second": list(second_wave),
                        "output": list(output),
                    }
                )
    return {
        "nonzero_stencil_pair_count": len(low_pairs),
        "nonzero_stencil_pairs": low_pairs,
        "consequence": (
            "|grad u|^2 has no nonzero partition-stencil Fourier mode, "
            "so mean[Phi_v |grad u|^2]=||grad u||_2^2/8 for every vertex."
        ),
        "all_checks_pass": not low_pairs,
    }


def _build_L2_optimal_field(
    carrier_base: int,
) -> tuple[Field, list[dict[str, Any]]]:
    field: Field = {}
    rows = []
    for block_index, subset_wave in enumerate(SUBSET_WAVES):
        scale = carrier_base * 4**block_index
        block = _normalized_block(subset_wave, scale)
        amplitude = block["L2_optimal_equal_amplitude"]
        coupling = block["coupling"]
        phase_sign = 1 if coupling > 0 else -1
        third_norm = math.sqrt(block["third_norm_squared"])
        entries = (
            (
                block["first_wave"],
                np.asarray(
                    block["first_polarization"],
                    dtype=float,
                ),
                1.0 + 0.0j,
            ),
            (
                block["second_wave"],
                np.asarray(
                    block["second_polarization"],
                    dtype=float,
                ),
                1.0 + 0.0j,
            ),
            (
                block["third_wave"],
                np.asarray(
                    block["third_polarization"],
                    dtype=float,
                )
                / third_norm,
                1j * phase_sign,
            ),
        )
        for wave, unit_polarization, phase in entries:
            coefficient = (
                amplitude
                * phase
                * unit_polarization.astype(np.complex128)
            )
            field[wave] = coefficient
            field[_negate(wave)] = coefficient.conjugate()
        rows.append(block)
    return field, rows


def _field_costs(field: Field) -> dict[str, float]:
    l2_squared = sum(
        float(np.vdot(value, value).real) for value in field.values()
    )
    h1_squared = sum(
        _norm_squared(wave) * float(np.vdot(value, value).real)
        for wave, value in field.items()
    )
    coefficient_l1 = sum(
        float(np.linalg.norm(value)) for value in field.values()
    )
    return {
        "L2_squared": l2_squared,
        "H1_squared": h1_squared,
        "coefficient_l1": coefficient_l1,
        "L3_cubed_upper_bound": coefficient_l1 * l2_squared,
    }


def _carrier_family_audit() -> dict[str, Any]:
    carrier_rows = []
    direct_rows = []
    for carrier_base in (8, 16, 32, 64, 128):
        blocks = [
            _normalized_block(
                subset_wave,
                carrier_base * 4**block_index,
            )
            for block_index, subset_wave in enumerate(SUBSET_WAVES)
        ]
        l2_minimum = sum(
            block["L2_squared_minimum"] for block in blocks
        )
        h1_minimum = sum(
            block["H1_squared_minimum"] for block in blocks
        )
        linfinity_bound = sum(
            6.0
            * block["amplitude_product"] ** (1.0 / 3.0)
            for block in blocks
        )
        l3_upper = linfinity_bound * l2_minimum
        carrier_rows.append(
            {
                "carrier_base": carrier_base,
                "minimum_normalized_coupling": min(
                    block["normalized_coupling"] for block in blocks
                ),
                "L2_squared_block_minimum": l2_minimum,
                "H1_squared_block_minimum": h1_minimum,
                "H1_minimum_over_carrier_squared": (
                    h1_minimum / carrier_base**2
                ),
                "L_infinity_coefficient_bound": linfinity_bound,
                "L3_cubed_upper_bound": l3_upper,
            }
        )

    for carrier_base in (8, 128):
        field, blocks = _build_L2_optimal_field(carrier_base)
        _, transport = _low_transport_modes(field)
        loads = _loads_from_transport(transport)
        costs = _field_costs(field)
        quadratic_silence = _quadratic_stencil_silence(field)
        direct_rows.append(
            {
                "carrier_base": carrier_base,
                "velocity_mode_count": len(field),
                "maximum_load_residual": max(
                    abs(value - expected)
                    for value, expected in zip(loads, EXPECTED_LOADS)
                ),
                "maximum_relative_divergence_residual": max(
                    abs(np.dot(wave, value))
                    / (
                        np.linalg.norm(np.asarray(wave, dtype=float))
                        * np.linalg.norm(value)
                    )
                    for wave, value in field.items()
                ),
                "costs": costs,
                "block_L2_minimum_residual": (
                    costs["L2_squared"]
                    - sum(
                        block["L2_squared_minimum"]
                        for block in blocks
                    )
                ),
                "quadratic_stencil_silence": quadratic_silence,
                "vertex_weighted_H1_squared": (
                    costs["H1_squared"] / 8.0
                ),
            }
        )

    l2_values = [
        row["L2_squared_block_minimum"] for row in carrier_rows
    ]
    l3_values = [
        row["L3_cubed_upper_bound"] for row in carrier_rows
    ]
    h1_scaled = [
        row["H1_minimum_over_carrier_squared"]
        for row in carrier_rows
    ]
    return {
        "carrier_family": (
            "For base carrier M, block S uses N_S=M 4^index(S), "
            "unit polarizations, and exact L2-optimal amplitudes."
        ),
        "carrier_rows": carrier_rows,
        "direct_sparse_rows": direct_rows,
        "bounded_L2_ratio_over_sample": max(l2_values) / min(l2_values),
        "bounded_L3_upper_ratio_over_sample": (
            max(l3_values) / min(l3_values)
        ),
        "H1_over_M2_ratio_over_sample": max(h1_scaled) / min(h1_scaled),
        "conclusion": (
            "Fixed bad loads admit a high-carrier sequence with bounded L2 "
            "and bounded critical L3 cost, so those costs are not "
            "carrier-coercive. The block-optimal H1 cost grows like M^2. "
            "Because the gradient energy is stencil-silent, every vertex "
            "weight sees exactly one eighth of that H1 cost."
        ),
        "all_checks_pass": (
            max(l2_values) / min(l2_values) < 1.2
            and max(l3_values) / min(l3_values) < 1.2
            and max(h1_scaled) / min(h1_scaled) < 1.2
            and all(
                row["velocity_mode_count"] == 42
                and row["maximum_load_residual"] < 1.0e-10
                and row["maximum_relative_divergence_residual"]
                < 1.0e-14
                and abs(row["block_L2_minimum_residual"]) < 1.0e-10
                and row["quadratic_stencil_silence"]["all_checks_pass"]
                for row in direct_rows
            )
        ),
    }


def _seed81_cost_benchmark() -> dict[str, Any]:
    fields = _build_spectral_fields()
    modes, coefficients = fields["velocity"]
    velocity = _evaluate_velocity_grid(modes, coefficients)
    l2_squared = float(
        sum(np.vdot(value, value).real for value in coefficients)
    )
    h1_squared = float(
        sum(
            _norm_squared(tuple(int(component) for component in mode))
            * np.vdot(value, value).real
            for mode, value in zip(modes, coefficients)
        )
    )
    l3_cubed = float(
        np.mean(np.sum(velocity**2, axis=0) ** 1.5)
    )
    prior = json.loads(
        (
            ROOT
            / "work/ns_collision/results/"
            "fourier_pressure_load_surjectivity_audit_v1.json"
        ).read_text(encoding="utf-8")
    )
    loads = np.asarray(
        prior["seed81_sparse_benchmark"]["compatible_loads"],
        dtype=float,
    )
    target = np.asarray(EXPECTED_LOADS)
    return {
        "source": "seed-81 finite-Fourier pressure adversary",
        "velocity_L2_squared": l2_squared,
        "velocity_H1_squared": h1_squared,
        "sampled_velocity_L3_cubed": l3_cubed,
        "load_Euclidean_norm": float(np.linalg.norm(loads)),
        "load_over_L3_cubed": float(np.linalg.norm(loads) / l3_cubed),
        "cosine_with_Hamming_ray": float(
            np.dot(loads, target)
            / (np.linalg.norm(loads) * np.linalg.norm(target))
        ),
        "scope": (
            "The L3 value uses the exact stored 20^3 trigonometric grid. "
            "It is a resolved benchmark, not an interval enclosure."
        ),
        "all_checks_pass": (
            bool(
                abs(l2_squared - 100.0) < 1.0e-10
                and h1_squared > l2_squared
                and l3_cubed > l2_squared ** 1.5
                and np.linalg.norm(loads) > 0.0
            )
        ),
    }


def audit() -> dict[str, Any]:
    scaling = _scaling_audit()
    optimization = _single_block_optimization_audit()
    limits = _symbolic_coupling_limits()
    support = _uniform_lacunary_support_audit()
    family = _carrier_family_audit()
    seed81 = _seed81_cost_benchmark()
    positive_checks = {
        "amplitude_and_spatial_scaling_passes": scaling[
            "all_checks_pass"
        ],
        "single_block_cost_optimization_passes": optimization[
            "all_checks_pass"
        ],
        "normalized_coupling_limits_pass": limits["all_checks_pass"],
        "uniform_lacunary_support_passes": support["all_checks_pass"],
        "high_carrier_cost_family_passes": family["all_checks_pass"],
        "seed81_cost_benchmark_passes": seed81["all_checks_pass"],
    }
    return {
        "kind": "pressure_load_realization_cost_audit",
        "schema_version": 1,
        "status": (
            "explicit_block_critical_carrier_noncoercivity_and_"
            "quadratic_Fisher_growth_certified"
        ),
        "assumption_scope": (
            "Smooth real periodic divergence-free finite-Fourier fields, "
            "frequency-one compatible loads, and the explicit lacunary "
            "block architecture. Global least costs outside this "
            "architecture are not computed."
        ),
        "exact_scaling": scaling,
        "single_block_optimization": optimization,
        "normalized_coupling_limits": limits,
        "uniform_lacunary_support": support,
        "high_carrier_realization_family": family,
        "seed81_cost_benchmark": seed81,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "load_realization_amplitude_scaling_derived": True,
            "load_realization_spatial_scaling_derived": True,
            "single_block_L2_minimum_derived": True,
            "single_block_H1_minimum_derived": True,
            "high_carrier_fixed_load_L2_bounded": True,
            "high_carrier_fixed_load_L3_upper_bounded": True,
            "critical_L3_cost_carrier_coercive": False,
            "explicit_block_H1_cost_grows_quadratically": True,
            "explicit_family_gradient_energy_partition_stencil_silent": True,
            "explicit_family_vertex_weighted_Fisher_is_one_eighth": True,
            "global_H1_least_cost_coercivity_proved": False,
            "global_critical_L3_least_cost_computed": False,
            "pressure_L32_remainder_absorbed": False,
            "critical_signed_bound_proved": False,
            "low_regularity_passage_proved": False,
            "exceptional_set_upgrade_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "route_decision": (
            "Critical L3 size alone cannot exclude high-carrier bad loads: "
            "the explicit realization family keeps an L3 upper bound "
            "uniform. High carriers are nevertheless expensive to the "
            "velocity Fisher term within this block architecture, "
            "quadratically in carrier ratio, and the compatible vertex "
            "weights see this cost exactly. Retain velocity Fisher before "
            "optimizing pressure; the next theorem must determine whether "
            "a general high-carrier estimate can reduce the unresolved "
            "sign problem to frequencies comparable with the partition "
            "frequency."
        ),
        "next_theorem_target": (
            "Derive a rigorous high-carrier absorption threshold for the "
            "full pressure minus velocity-Fisher minus weight-Fisher "
            "functional, including arbitrary compatible coefficients. "
            "Then isolate the compact carrier band below that threshold "
            "and test whether graph cancellation, intrinsic 2:1 balance, "
            "or a finite-dimensional sharp inequality controls it without "
            "a positive weight floor."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "work/ns_collision/results/"
            "pressure_load_realization_cost_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("pressure-load realization cost audit failed")
    _atomic_json(arguments.output, result)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": _sha256(arguments.output),
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
