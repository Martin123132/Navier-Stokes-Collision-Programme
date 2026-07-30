"""Finite Fourier audit for the cumulative quartic transfer primitive."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np


Wave = tuple[int, int, int]
Field = dict[Wave, np.ndarray]


def _add(field: Field, wave: np.ndarray | Wave, value: np.ndarray) -> None:
    key = tuple(int(entry) for entry in wave)
    field[key] = field.get(key, np.zeros(3, dtype=complex)) + value


def _project(wave: np.ndarray | Wave, value: np.ndarray) -> np.ndarray:
    vector = np.asarray(wave, dtype=float)
    wave_number_squared = float(vector @ vector)
    if wave_number_squared == 0:
        return value
    return value - vector * (vector @ value) / wave_number_squared


def _nonlinearity(field: Field) -> Field:
    advection: Field = {}
    for first_wave, first_value in field.items():
        for second_wave, second_value in field.items():
            _add(
                advection,
                np.asarray(first_wave) + np.asarray(second_wave),
                1j * (first_value @ second_wave) * second_value,
            )

    result: Field = {}
    for wave, value in advection.items():
        if wave == (0, 0, 0):
            continue
        projected = _project(wave, value)
        if np.linalg.norm(projected) > 1.0e-12:
            result[wave] = projected
    return result


def _vorticity(wave: Wave, value: np.ndarray) -> np.ndarray:
    return 1j * np.cross(wave, value)


def _strain(wave: Wave, value: np.ndarray) -> np.ndarray:
    vector = np.asarray(wave)
    return 0.5j * (np.outer(value, vector) + np.outer(vector, value))


def _trilinear(
    first: Field,
    second: Field,
    third: Field,
    heat_scale: float,
    primitive: bool,
) -> float:
    total = 0j
    for first_wave, first_value in first.items():
        first_frequency = float(np.dot(first_wave, first_wave))
        multiplier = 1 - np.exp(-heat_scale * first_frequency)
        for second_wave, second_value in second.items():
            third_wave = tuple(
                -np.asarray(first_wave) - np.asarray(second_wave)
            )
            if third_wave not in third:
                continue
            total_frequency = (
                first_frequency
                + float(np.dot(second_wave, second_wave))
                + float(np.dot(third_wave, third_wave))
            )
            denominator = total_frequency if primitive else 1.0
            total += multiplier / denominator * np.einsum(
                "ij,i,j",
                _strain(first_wave, first_value),
                _vorticity(second_wave, second_value),
                _vorticity(third_wave, third[third_wave]),
            )
    return float(total.real)


def evaluate(field: Field, heat_scale: float) -> dict[str, float]:
    nonlinear = _nonlinearity(field)
    euler_direction = {wave: -value for wave, value in nonlinear.items()}
    heat_direction = {
        wave: -float(np.dot(wave, wave)) * value
        for wave, value in field.items()
    }

    defect = _trilinear(field, field, field, heat_scale, primitive=False)
    primitive = _trilinear(field, field, field, heat_scale, primitive=True)
    quartic_transfer = sum(
        [
            _trilinear(
                euler_direction, field, field, heat_scale, primitive=True
            ),
            _trilinear(
                field, euler_direction, field, heat_scale, primitive=True
            ),
            _trilinear(
                field, field, euler_direction, heat_scale, primitive=True
            ),
        ]
    )
    heat_derivative = sum(
        [
            _trilinear(
                heat_direction, field, field, heat_scale, primitive=True
            ),
            _trilinear(
                field, heat_direction, field, heat_scale, primitive=True
            ),
            _trilinear(
                field, field, heat_direction, heat_scale, primitive=True
            ),
        ]
    )
    return {
        "defect": defect,
        "primitive": primitive,
        "quartic_transfer": quartic_transfer,
        "heat_derivative_residual": heat_derivative + defect,
    }


def random_field(seed: int, maximum_mode: int = 1) -> Field:
    rng = np.random.default_rng(seed)
    field: Field = {}
    for wave in product(range(-maximum_mode, maximum_mode + 1), repeat=3):
        if wave == (0, 0, 0):
            continue
        first_nonzero = next(entry for entry in wave if entry != 0)
        if first_nonzero < 0:
            continue
        value = _project(
            wave,
            rng.normal(size=3) + 1j * rng.normal(size=3),
        )
        field[wave] = value
        field[tuple(-entry for entry in wave)] = np.conjugate(value)
    return field


def audit(samples: int = 8, heat_scale: float = 0.5) -> dict[str, float | int | bool]:
    values = []
    parity_residuals = []
    heat_residuals = []
    for seed in range(samples):
        field = random_field(20260717 + seed)
        result = evaluate(field, heat_scale)
        reversed_result = evaluate(
            {wave: -value for wave, value in field.items()}, heat_scale
        )
        values.append(result["quartic_transfer"])
        parity_residuals.append(
            result["quartic_transfer"]
            - reversed_result["quartic_transfer"]
        )
        heat_residuals.append(result["heat_derivative_residual"])

    output: dict[str, float | int | bool] = {
        "sample_count": samples,
        "heat_scale": heat_scale,
        "minimum_quartic_transfer": min(values),
        "maximum_quartic_transfer": max(values),
        "negative_sample_count": sum(value < -1.0e-10 for value in values),
        "maximum_parity_residual": max(abs(value) for value in parity_residuals),
        "maximum_heat_identity_residual": max(
            abs(value) for value in heat_residuals
        ),
        "quartic_transfer_is_even": max(
            abs(value) for value in parity_residuals
        )
        < 1.0e-9,
        "triad_primitive_heat_identity": max(
            abs(value) for value in heat_residuals
        )
        < 1.0e-9,
        "no_negative_samples_observed": all(value >= -1.0e-10 for value in values),
    }
    output["all_boolean_checks_pass"] = all(
        value for value in output.values() if isinstance(value, bool)
    )
    return output


def run_resumable_search(
    samples: int, heat_scale: float, start_seed: int, output_path: Path
) -> dict[str, float | int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[int] = set()
    existing_values: list[float] = []
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["heat_scale"] == heat_scale:
                completed.add(int(record["seed"]))
                existing_values.append(float(record["quartic_transfer"]))

    new_values = []
    with output_path.open("a", encoding="utf-8") as stream:
        for seed in range(start_seed, start_seed + samples):
            if seed in completed:
                continue
            value = evaluate(random_field(seed), heat_scale)["quartic_transfer"]
            record = {
                "seed": seed,
                "heat_scale": heat_scale,
                "quartic_transfer": value,
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            new_values.append(value)

    all_values = existing_values + new_values
    return {
        "records_at_scale": len(all_values),
        "new_records": len(new_values),
        "minimum_quartic_transfer": min(all_values),
        "negative_count": sum(value < -1.0e-10 for value in all_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--start-seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.output is None:
        result = audit(args.samples, args.scale)
    else:
        result = run_resumable_search(
            args.samples, args.scale, args.start_seed, args.output
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
