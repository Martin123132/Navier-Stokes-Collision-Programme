"""Certify a positive-volume low-output block for the dense HHH packet."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable, Iterator

import numpy as np
import psutil

import dense_annular_hhh_packet_gate_audit as dense
from dense_spaced_continuum_positivity_audit import (
    Interval,
    _down,
    _up,
    _interval_dynamics,
    _interval_project,
)
import nonlinear_stress_regeneration_gate_audit as regeneration


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CARRIER_MULTIPLE = 4096
DEFAULT_OUTPUT_HALF_WIDTH = Fraction(1, 2)
OFFSET_SPACING = 4
CHANNEL_NUMERATOR = -dense.CENTRAL_IMAGINARY_MATRIX
PRIOR_DENSE_RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "dense_spaced_continuum_positivity_audit_v1.json"
)
PRIOR_TAIL_RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "scale_uniform_low_output_tail_gate_audit_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "work/ns_collision/results/"
    "dense_low_output_block_gate_audit_v1.json"
)

Offset = tuple[int, int, int]


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


def _set_below_normal_priority() -> bool:
    process = psutil.Process(os.getpid())
    try:
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return process.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
        process.nice(5)
        return process.nice() >= 5
    except (psutil.AccessDenied, psutil.Error):
        return False


def _interval_vector(
    center: tuple[float, float, float],
    perturbation: list[Interval],
    scale: Interval,
) -> list[Interval]:
    return [
        Interval.point(component) + scale * offset
        for component, offset in zip(center, perturbation)
    ]


def _interval_add(
    first: list[Interval],
    second: list[Interval],
) -> list[Interval]:
    return [left + right for left, right in zip(first, second)]


def _interval_subtract(
    first: list[Interval],
    second: list[Interval],
) -> list[Interval]:
    return [left - right for left, right in zip(first, second)]


def _interval_channel_contraction(
    generated: list[Interval],
    remaining: list[Interval],
) -> Interval:
    value = Interval.point(0.0)
    for row in range(3):
        for column in range(3):
            tensor_entry = float(CHANNEL_NUMERATOR[row, column])
            value += tensor_entry * (
                generated[row] * remaining[column]
                + remaining[row] * generated[column]
            )
    return value


def _fixed_channel_symbol_interval(
    carrier_multiple: int = DEFAULT_CARRIER_MULTIPLE,
    output_half_width: float = float(DEFAULT_OUTPUT_HALF_WIDTH),
    *,
    high_offset_interval: Interval | None = None,
    output_interval: Interval | None = None,
) -> dict[str, Any]:
    if carrier_multiple < 32:
        raise ValueError("carrier multiple must be at least 32")
    if not 0.0 <= output_half_width <= 1.0:
        raise ValueError("output half-width must lie in [0,1]")

    offset = high_offset_interval or Interval(-1.0, 1.0)
    output = output_interval or Interval(
        -output_half_width,
        output_half_width,
    )
    first_offset = [offset for _ in range(3)]
    second_offset = [offset for _ in range(3)]
    output_offset = [output for _ in range(3)]
    third_offset = _interval_subtract(
        _interval_subtract(output_offset, first_offset),
        second_offset,
    )
    inverse_carrier = Interval.point(1.0 / carrier_multiple)

    first_wave = _interval_vector(
        (1.0, 0.0, 0.0),
        first_offset,
        inverse_carrier,
    )
    second_wave = _interval_vector(
        (-1.0, 1.0, 0.0),
        second_offset,
        inverse_carrier,
    )
    third_wave = _interval_vector(
        (0.0, -1.0, 0.0),
        third_offset,
        inverse_carrier,
    )

    first_value = _interval_project(
        (-4.0, -3.0, 1.0),
        first_wave,
    )
    second_value = _interval_project(
        (-3.0, -1.0, 2.0),
        second_wave,
    )
    third_value = _interval_project(
        (-3.0, 7.0, 1.0),
        third_wave,
    )

    first_second_output = _interval_vector(
        (0.0, 1.0, 0.0),
        _interval_add(first_offset, second_offset),
        inverse_carrier,
    )
    first_third_output = _interval_vector(
        (1.0, -1.0, 0.0),
        _interval_subtract(output_offset, second_offset),
        inverse_carrier,
    )
    second_third_output = _interval_vector(
        (-1.0, 0.0, 0.0),
        _interval_subtract(output_offset, first_offset),
        inverse_carrier,
    )

    first_second_generated = _interval_dynamics(
        first_wave,
        first_value,
        second_wave,
        second_value,
        first_second_output,
    )
    first_third_generated = _interval_dynamics(
        first_wave,
        first_value,
        third_wave,
        third_value,
        first_third_output,
    )
    second_third_generated = _interval_dynamics(
        second_wave,
        second_value,
        third_wave,
        third_value,
        second_third_output,
    )

    numerator = (
        _interval_channel_contraction(
            first_second_generated,
            third_value,
        )
        + _interval_channel_contraction(
            first_third_generated,
            second_value,
        )
        + _interval_channel_contraction(
            second_third_generated,
            first_value,
        )
    )
    central_norm = dense.CENTRAL_NORM
    central_norm_interval = Interval(
        _down(central_norm),
        _up(central_norm),
    )
    normalized = numerator / central_norm_interval
    return {
        "carrier_multiple_relative_to_offset_box": carrier_multiple,
        "physical_carrier": (
            f"R={OFFSET_SPACING * carrier_multiple}M"
        ),
        "normalized_output_half_width": output_half_width,
        "physical_output_block": (
            f"4q with |q_i|<=floor({output_half_width}M)"
        ),
        "relaxed_domain": (
            "x,y in [-1,1]^3, w in [-delta,delta]^3, and "
            "z=w-x-y in [-2-delta,2+delta]^3. This contains the "
            "true lattice polytope, which also requires z in [-1,1]^3."
        ),
        "channel_numerator_interval_per_carrier": [
            numerator.lower,
            numerator.upper,
        ],
        "unit_channel_interval_per_carrier": [
            normalized.lower,
            normalized.upper,
        ],
        "all_checks_pass": bool(
            numerator.lower > 0.0 and normalized.lower > 0.0
        ),
    }


def _central_interval_self_audit(
    carrier_multiple: int = DEFAULT_CARRIER_MULTIPLE,
) -> dict[str, Any]:
    center = _fixed_channel_symbol_interval(
        carrier_multiple,
        0.0,
        high_offset_interval=Interval.point(0.0),
        output_interval=Interval.point(0.0),
    )
    interval = center["unit_channel_interval_per_carrier"]
    exact = dense.CENTRAL_NORM
    return {
        "exact_unit_channel_per_carrier": "12*sqrt(43)",
        "numeric_exact_unit_channel_per_carrier": exact,
        "directed_interval": interval,
        "interval_width": interval[1] - interval[0],
        "all_checks_pass": bool(
            interval[0] <= exact <= interval[1]
            and interval[1] - interval[0] < 1.0e-9
        ),
    }


def _bounded_triple_count_1d(radius: int, output: int) -> int:
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    shifted_total = output + 3 * radius
    width = 2 * radius + 1
    count = 0
    for excluded in range(4):
        remainder = shifted_total - excluded * width
        if remainder < 0:
            continue
        count += (
            (-1) ** excluded
            * math.comb(3, excluded)
            * math.comb(remainder + 2, 2)
        )
    return count


def _bounded_triple_count_3d(radius: int, output: Offset) -> int:
    count = 1
    for component in output:
        count *= _bounded_triple_count_1d(radius, component)
    return count


def _direct_count_1d(radius: int, output: int) -> int:
    count = 0
    for first in range(-radius, radius + 1):
        for second in range(-radius, radius + 1):
            third = output - first - second
            if -radius <= third <= radius:
                count += 1
    return count


def _multiplicity_audit(
    output_half_width: Fraction = DEFAULT_OUTPUT_HALF_WIDTH,
) -> dict[str, Any]:
    maximum_formula_residual = 0
    maximum_interior_residual = 0
    for radius in range(1, 13):
        for output in range(-3 * radius, 3 * radius + 1):
            formula = _bounded_triple_count_1d(radius, output)
            direct = _direct_count_1d(radius, output)
            maximum_formula_residual = max(
                maximum_formula_residual,
                abs(formula - direct),
            )
            if abs(output) <= radius:
                interior = 3 * radius**2 + 3 * radius + 1 - output**2
                maximum_interior_residual = max(
                    maximum_interior_residual,
                    abs(formula - interior),
                )

    rows = []
    for radius in (2, 4, 8, 16, 32):
        output_radius = (
            output_half_width.numerator
            * radius
            // output_half_width.denominator
        )
        minimum_one_dimensional_count = _bounded_triple_count_1d(
            radius,
            output_radius,
        )
        rows.append(
            {
                "M": radius,
                "output_radius": output_radius,
                "output_channel_count": (2 * output_radius + 1) ** 3,
                "minimum_one_dimensional_triple_count": (
                    minimum_one_dimensional_count
                ),
                "minimum_three_dimensional_triple_count": (
                    minimum_one_dimensional_count**3
                ),
                "minimum_triple_count_over_M6": (
                    minimum_one_dimensional_count**3 / radius**6
                ),
            }
        )

    delta = output_half_width
    leading_count = Fraction(3, 1) - delta**2
    return {
        "one_dimensional_exact_formula_for_abs_q_le_M": (
            "3M^2+3M+1-q^2"
        ),
        "three_dimensional_count": (
            "product_i (3M^2+3M+1-q_i^2)"
        ),
        "output_half_width": str(output_half_width),
        "uniform_one_dimensional_leading_lower": str(leading_count),
        "uniform_three_dimensional_leading_lower": (
            f"({leading_count})^3 M^6"
        ),
        "maximum_inclusion_exclusion_residual": maximum_formula_residual,
        "maximum_interior_formula_residual": maximum_interior_residual,
        "rows": rows,
        "all_checks_pass": bool(
            maximum_formula_residual == 0
            and maximum_interior_residual == 0
            and all(
                row["minimum_three_dimensional_triple_count"] > 0
                for row in rows
            )
        ),
    }


def _representative_output_coordinates(output_radius: int) -> list[int]:
    if output_radius <= 2:
        return list(range(-output_radius, output_radius + 1))
    half = output_radius // 2
    return sorted({-output_radius, -half, 0, half, output_radius})


def _all_offsets(radius: int) -> list[Offset]:
    return list(product(range(-radius, radius + 1), repeat=3))


def _third_offset(
    output: Offset,
    first: Offset,
    second: Offset,
) -> Offset:
    return tuple(
        target - left - right
        for target, left, right in zip(output, first, second)
    )  # type: ignore[return-value]


def _in_box(offset: Offset, radius: int) -> bool:
    return all(abs(component) <= radius for component in offset)


def _exhaustive_triples(
    radius: int,
    output: Offset,
) -> Iterator[tuple[Offset, Offset, Offset]]:
    offsets = _all_offsets(radius)
    for first in offsets:
        for second in offsets:
            third = _third_offset(output, first, second)
            if _in_box(third, radius):
                yield first, second, third


def _sampled_triples(
    radius: int,
    output: Offset,
    sample_limit: int,
) -> list[tuple[Offset, Offset, Offset]]:
    selected: set[tuple[Offset, Offset]] = set()
    coarse = sorted({-radius, 0, radius})
    for values in product(coarse, repeat=6):
        first = tuple(values[:3])
        second = tuple(values[3:])
        third = _third_offset(output, first, second)
        if _in_box(third, radius):
            selected.add((first, second))
            if len(selected) >= sample_limit:
                break

    seed = (
        radius * 1_000_003
        + (output[0] + 3 * radius) * 10_007
        + (output[1] + 3 * radius) * 101
        + output[2]
    )
    generator = random.Random(seed)
    attempts = 0
    maximum_attempts = 100 * sample_limit
    while len(selected) < sample_limit and attempts < maximum_attempts:
        attempts += 1
        first = tuple(
            generator.randint(-radius, radius) for _ in range(3)
        )
        second = tuple(
            generator.randint(-radius, radius) for _ in range(3)
        )
        third = _third_offset(output, first, second)
        if _in_box(third, radius):
            selected.add((first, second))

    return [
        (first, second, _third_offset(output, first, second))
        for first, second in sorted(selected)
    ]


def _packet_modes(
    radius: int,
    carrier_multiple: int,
) -> tuple[
    int,
    list[dict[Offset, np.ndarray]],
    list[dict[Offset, np.ndarray]],
    float,
]:
    carrier = OFFSET_SPACING * carrier_multiple * radius
    offsets = _all_offsets(radius)
    waves: list[dict[Offset, np.ndarray]] = []
    values: list[dict[Offset, np.ndarray]] = []
    positive_energy = 0.0
    for center, base, phase in zip(
        dense.CENTER_DIRECTIONS,
        dense.BASE_VECTORS,
        dense.PHASES,
    ):
        cluster_waves: dict[Offset, np.ndarray] = {}
        cluster_values: dict[Offset, np.ndarray] = {}
        for offset in offsets:
            wave = (
                carrier * center
                + OFFSET_SPACING * np.asarray(offset, dtype=float)
            )
            value = phase * regeneration._project(base, wave)
            cluster_waves[offset] = wave
            cluster_values[offset] = value
            positive_energy += float(np.vdot(value, value).real)
        waves.append(cluster_waves)
        values.append(cluster_values)
    normalization = 1.0 / math.sqrt(2.0 * positive_energy)
    return carrier, waves, values, normalization


def _finite_replay_row(
    radius: int,
    carrier_multiple: int,
    output_half_width: Fraction,
    interval_lower: float,
    interval_upper: float,
    *,
    maximum_exhaustive_radius: int,
    sample_limit: int,
) -> dict[str, Any]:
    carrier, waves, values, normalization = _packet_modes(
        radius,
        carrier_multiple,
    )
    output_radius = (
        output_half_width.numerator
        * radius
        // output_half_width.denominator
    )
    coordinates = _representative_output_coordinates(output_radius)
    outputs = list(product(coordinates, repeat=3))
    minimum_symbol = math.inf
    maximum_symbol = -math.inf
    maximum_imaginary_residual = 0.0
    total_checked = 0
    minimum_exact_count = math.inf
    exhaustive = radius <= maximum_exhaustive_radius

    for output in outputs:
        minimum_exact_count = min(
            minimum_exact_count,
            _bounded_triple_count_3d(radius, output),
        )
        triples: Iterable[tuple[Offset, Offset, Offset]]
        if exhaustive:
            triples = _exhaustive_triples(radius, output)
        else:
            triples = _sampled_triples(
                radius,
                output,
                sample_limit,
            )
        for first, second, third in triples:
            contribution = regeneration._hhh_stress_forcing(
                waves[0][first],
                values[0][first],
                waves[1][second],
                values[1][second],
                waves[2][third],
                values[2][third],
            )
            pairing = (
                np.vdot(dense.CHANNEL_TENSOR, contribution) / carrier
            )
            minimum_symbol = min(minimum_symbol, float(pairing.real))
            maximum_symbol = max(maximum_symbol, float(pairing.real))
            maximum_imaginary_residual = max(
                maximum_imaginary_residual,
                abs(float(pairing.imag)),
            )
            total_checked += 1

    certified_forcing_lower = (
        2.0
        * normalization**3
        * carrier
        * minimum_exact_count
        * interval_lower
    )
    return {
        "M": radius,
        "carrier": carrier,
        "output_radius": output_radius,
        "representative_output_count": len(outputs),
        "representative_coordinates": coordinates,
        "symbol_evaluation_mode": (
            "exhaustive" if exhaustive else "deterministic_sample"
        ),
        "symbol_evaluations": total_checked,
        "minimum_exact_triple_count_over_representative_outputs": int(
            minimum_exact_count
        ),
        "minimum_sampled_unit_channel_per_carrier": minimum_symbol,
        "maximum_sampled_unit_channel_per_carrier": maximum_symbol,
        "maximum_sampled_imaginary_residual": (
            maximum_imaginary_residual
        ),
        "normalization_coefficient": normalization,
        "certified_uniform_forcing_lower": certified_forcing_lower,
        "certified_uniform_forcing_lower_over_H_5_2": (
            certified_forcing_lower / carrier**2.5
        ),
        "all_checks_pass": bool(
            total_checked > 0
            and minimum_symbol >= interval_lower - 1.0e-10
            and maximum_symbol <= interval_upper + 1.0e-10
            and maximum_imaginary_residual < 1.0e-12
            and certified_forcing_lower > 0.0
        ),
    }


def _finite_replay(
    radii: tuple[int, ...],
    carrier_multiple: int,
    output_half_width: Fraction,
    interval_lower: float,
    interval_upper: float,
    *,
    maximum_exhaustive_radius: int,
    sample_limit: int,
) -> dict[str, Any]:
    rows = [
        _finite_replay_row(
            radius,
            carrier_multiple,
            output_half_width,
            interval_lower,
            interval_upper,
            maximum_exhaustive_radius=maximum_exhaustive_radius,
            sample_limit=sample_limit,
        )
        for radius in radii
    ]
    return {
        "rows": rows,
        "role": (
            "The finite rows replay actual projected lattice symbols. "
            "The directed interval certificate, not sampling, proves the "
            "uniform continuum statement."
        ),
        "all_checks_pass": all(row["all_checks_pass"] for row in rows),
    }


def _scaling_certificate(
    interval_lower: float,
    output_half_width: Fraction,
    carrier_multiple: int,
) -> dict[str, Any]:
    base_norm_square_sum = sum(
        int(np.dot(vector, vector)) for vector in dense.BASE_VECTORS
    )
    real_packet_energy_upper_coefficient = 2 * base_norm_square_sum
    leading_count = Fraction(3, 1) - output_half_width**2
    mode_box_upper_coefficient = 3**3
    energy_upper_coefficient = (
        real_packet_energy_upper_coefficient
        * mode_box_upper_coefficient
    )
    explicit_H_5_2_constant = (
        2.0
        * interval_lower
        * float(leading_count**3)
        / (
            energy_upper_coefficient**1.5
            * (OFFSET_SPACING * carrier_multiple) ** 1.5
        )
    )
    return {
        "base_vector_norm_square_sum": base_norm_square_sum,
        "real_packet_energy_upper": (
            f"{real_packet_energy_upper_coefficient}(2M+1)^3"
        ),
        "interior_triple_count_lower": (
            f"({leading_count})^3 M^6"
        ),
        "unit_energy_mode_coefficient_lower": (
            f"[{real_packet_energy_upper_coefficient}(2M+1)^3]^(-1/2)"
        ),
        "uniform_fixed_channel_forcing_lower": (
            f"{explicit_H_5_2_constant:.17g} H^(5/2)"
        ),
        "explicit_H_5_2_constant": explicit_H_5_2_constant,
        "output_channel_count": (
            "(2 floor(M/2)+1)^3, on the physical sublattice 4Z^3"
        ),
        "spatial_endpoint_implication": (
            "At one instant, one unit-energy divergence-free packet "
            "simultaneously supplies c H^(5/2) in O(H^3) low outputs. "
            "This realizes the spatial multiplicity in the H^(-1) "
            "envelope obstruction."
        ),
        "temporal_limit": (
            "No lower bound on a parabolic-time forcing pulse follows. "
            "The unforced Navier-Stokes evolution may decorrelate the "
            "packet before time H^(-2)."
        ),
        "all_checks_pass": explicit_H_5_2_constant > 0.0,
    }


def audit(
    radii: tuple[int, ...] = (1, 2, 4, 8),
    carrier_multiple: int = DEFAULT_CARRIER_MULTIPLE,
    output_half_width: Fraction = DEFAULT_OUTPUT_HALF_WIDTH,
    *,
    maximum_exhaustive_radius: int = 2,
    sample_limit: int = 512,
) -> dict[str, Any]:
    interval = _fixed_channel_symbol_interval(
        carrier_multiple,
        float(output_half_width),
    )
    central = _central_interval_self_audit(carrier_multiple)
    multiplicity = _multiplicity_audit(output_half_width)
    interval_lower, interval_upper = interval[
        "unit_channel_interval_per_carrier"
    ]
    finite = _finite_replay(
        radii,
        carrier_multiple,
        output_half_width,
        interval_lower,
        interval_upper,
        maximum_exhaustive_radius=maximum_exhaustive_radius,
        sample_limit=sample_limit,
    )
    scaling = _scaling_certificate(
        interval_lower,
        output_half_width,
        carrier_multiple,
    )
    positive_checks = {
        "directed_interval_positive": interval["all_checks_pass"],
        "central_symbol_reproduced": central["all_checks_pass"],
        "exact_output_multiplicity": multiplicity["all_checks_pass"],
        "finite_lattice_replay": finite["all_checks_pass"],
        "uniform_H_5_2_scaling": scaling["all_checks_pass"],
    }
    return {
        "kind": "dense_low_output_block_gate_audit",
        "schema_version": 1,
        "status": (
            "positive_volume_low_output_HHH_block_certified"
            if all(positive_checks.values())
            else "failed"
        ),
        "configuration": {
            "carrier_multiple_relative_to_offset_box": carrier_multiple,
            "offset_spacing": OFFSET_SPACING,
            "physical_carrier": (
                f"R={OFFSET_SPACING * carrier_multiple}M"
            ),
            "output_half_width": str(output_half_width),
            "finite_radii": list(radii),
            "maximum_exhaustive_radius": maximum_exhaustive_radius,
            "sample_limit_per_representative_output": sample_limit,
        },
        "prerequisites": {
            "dense_continuum_positivity_result": str(
                PRIOR_DENSE_RESULT.relative_to(ROOT)
            ).replace("\\", "/"),
            "dense_continuum_positivity_sha256": _sha256(
                PRIOR_DENSE_RESULT
            ),
            "scale_uniform_tail_result": str(
                PRIOR_TAIL_RESULT.relative_to(ROOT)
            ).replace("\\", "/"),
            "scale_uniform_tail_sha256": _sha256(PRIOR_TAIL_RESULT),
        },
        "directed_interval_certificate": interval,
        "central_interval_self_audit": central,
        "exact_lattice_multiplicity": multiplicity,
        "finite_lattice_replay": finite,
        "scaling_certificate": scaling,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "one_fixed_tensor_channel_positive_on_output_block": True,
            "positive_volume_low_output_block_realized": True,
            "simultaneous_spatial_H_5_2_output_scaling_proved": True,
            "parabolic_time_persistence_proved": False,
            "H_minus_1_endpoint_for_actual_Navier_Stokes_proved": False,
            "H_minus_1_endpoint_for_actual_Navier_Stokes_falsified": False,
            "suitable_weak_solution_closure_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--radii",
        default="1,2,4,8",
        help="comma-separated packet radii",
    )
    parser.add_argument(
        "--carrier-multiple",
        type=int,
        default=DEFAULT_CARRIER_MULTIPLE,
    )
    parser.add_argument(
        "--maximum-exhaustive-radius",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    radii = tuple(
        int(value.strip())
        for value in arguments.radii.split(",")
        if value.strip()
    )
    if not radii or min(radii) < 1:
        raise ValueError("all radii must be positive")
    below_normal_priority_set = _set_below_normal_priority()
    result = audit(
        radii,
        arguments.carrier_multiple,
        maximum_exhaustive_radius=arguments.maximum_exhaustive_radius,
        sample_limit=arguments.sample_limit,
    )
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("dense low-output block gate failed")
    result["runtime"] = {
        "worker_count": 1,
        "below_normal_priority_set": below_normal_priority_set,
    }
    _atomic_json(arguments.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(arguments.output),
                "sha256": _sha256(arguments.output),
                "interval": result["directed_interval_certificate"][
                    "unit_channel_interval_per_carrier"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
