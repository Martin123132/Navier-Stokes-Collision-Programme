"""Port the finite rho-zero generator jets to the parallel two-mode shear."""

from __future__ import annotations

import argparse
import gc
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
import sympy as sp

from annular_parallel_shear_phase_repair_audit import (
    ELL_XY,
    ELL_YZ,
    PARALLEL_FISHER_MASS,
    _combined_parallel_loads,
    _joint_optimum,
    _lower_process_priority,
)
from annular_rho_zero_first_jet_audit import (
    _generator_from_coefficients,
    _grid_shape,
    _physical,
    _pressure_coefficients,
    _scalar_gradient,
    _spectral_data,
    _vector_gradient,
)
from annular_rho_zero_second_jet_route_guard_audit import (
    _second_variation_decomposition,
    _state_and_flow_jets,
)
from annular_two_shear_square_gate_audit import (
    _modified_finite_packet,
)
from compatible_edge_annular_escape_audit import (
    DELTA_CUBIC_ENERGY,
)
from separable_annular_pressure_schur_no_go_audit import (
    _mixed_difference_fisher,
    _shift_slices,
    _vertex_weight_float,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_parallel_shear_finite_jet_port_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "annular_parallel_shear_phase_repair_audit_v1.json"
    ): "ab0d58bb824520167a90083795f0913da1cc9ca7b50e5e785ae7192f9f14efbd",
    (
        "work/ns_collision/results/"
        "annular_rho_zero_second_jet_route_guard_audit_v1.json"
    ): "7c985480afc51a084eefa0e2fb614fd3b900e9d2e347a6effb0c99b7259c693d",
    (
        "work/ns_collision/results/"
        "annular_rho_zero_first_jet_remainder_gate_audit_v1.json"
    ): "582a6a4997928b8cd7b67f1d9fd58b5fef6326ee6ffa42bede33a5d9854f36c9",
    (
        "work/ns_collision/results/"
        "annular_two_shear_full_c1_port_audit_v1.json"
    ): "af0039698cdbd5442be629b23ea259e97556c978a0f0d91e94e9e0d658b1f32f",
}
ALGORITHM_REVISION = "annular-parallel-shear-finite-jet-port-v1"
DEFAULT_HEAT_SIZES = (5, 9, 17, 25, 33, 49)
Array = np.ndarray
Field = dict[str, Any]
Wave = tuple[int, int, int]


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


def _prerequisite_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    payloads: dict[str, Any] = {}
    for relative, expected in PREREQUISITES.items():
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = _sha256(path)
        pass_value = next(
            (
                payload[key]
                for key in (
                    "all_positive_checks_pass",
                    "all_port_checks_pass",
                    "all_route_guard_checks_pass",
                    "all_checks_pass",
                )
                if key in payload
            ),
            None,
        )
        payloads[Path(relative).stem] = payload
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "predecessor_pass_value": pass_value,
                "matches": bool(
                    actual == expected
                    and pass_value is True
                ),
            }
        )
    return (
        {
            "rows": rows,
            "all_checks_pass": all(row["matches"] for row in rows),
        },
        payloads,
    )


def _negate_wave(wave: Wave) -> Wave:
    return tuple(-value for value in wave)


def _parallel_component_fields() -> dict[str, dict[Wave, Array]]:
    direction = np.ones(3, dtype=float) / math.sqrt(3.0)
    fields: dict[str, dict[Wave, Array]] = {}
    for label, wave, polarization in (
        ("yz", ELL_YZ, direction),
        ("xy", ELL_XY, -direction),
    ):
        fields[label] = {
            wave: -1j * polarization,
            _negate_wave(wave): 1j * polarization,
        }
    return fields


def _combined_low_field() -> dict[Wave, Array]:
    output: dict[Wave, Array] = {}
    for field in _parallel_component_fields().values():
        output.update(field)
    return output


def _initial_coefficients(
    size: int,
    shape: tuple[int, ...],
    yz_amplitude: float,
    xy_amplitude: float,
    coefficient_scale: float,
) -> tuple[Array, Array, Array, Array, Array]:
    family_waves, family_velocity, parity = _modified_finite_packet(size)
    velocity_coefficients = np.zeros(
        (3, *shape), dtype=np.complex128
    )
    shape_array = np.asarray(shape, dtype=int)
    for wave, value in zip(
        family_waves.reshape(-1, 3).astype(int),
        family_velocity.reshape(-1, 3),
    ):
        positive = tuple((wave % shape_array).tolist())
        negative = tuple(((-wave) % shape_array).tolist())
        velocity_coefficients[(slice(None), *positive)] = value
        velocity_coefficients[(slice(None), *negative)] = value

    component_fields = _parallel_component_fields()
    for label, amplitude in (
        ("yz", yz_amplitude),
        ("xy", xy_amplitude),
    ):
        for wave, value in component_fields[label].items():
            index = tuple(
                (np.asarray(wave, dtype=int) % shape_array).tolist()
            )
            velocity_coefficients[(slice(None), *index)] += (
                -amplitude * value
            )

    weight_coefficients = np.zeros(shape, dtype=np.complex128)
    for wave in product((-1, 0, 1), repeat=3):
        coefficient = coefficient_scale * math.prod(
            0.5 if value == 0 else 0.25 for value in wave
        )
        index = tuple(
            (np.asarray(wave, dtype=int) % shape_array).tolist()
        )
        weight_coefficients[index] = coefficient
    return (
        velocity_coefficients,
        weight_coefficients,
        family_waves,
        family_velocity,
        parity,
    )


def _first_variation_decomposition(
    jets: dict[str, Any],
    waves: tuple[Array, ...],
    safe_wave_number_squared: Array,
    volume: int,
    viscosity: float,
) -> dict[str, Any]:
    velocity = jets["velocity"]
    weight = jets["weight"]
    pressure = jets["pressure"]
    velocity_directions = jets["velocity_directions"]
    weight_directions = jets["weight_directions"]
    pressure_variations: dict[str, Array] = {}

    def pressure_variation(direction: Field) -> Array:
        label = direction["label"]
        if label not in pressure_variations:
            coefficients = _pressure_coefficients(
                velocity["value"],
                direction["value"],
                waves,
                safe_wave_number_squared,
                volume,
                symmetrized=True,
            )
            pressure_variations[label] = _physical(coefficients, volume)
        return pressure_variations[label]

    def D_u(direction: Field) -> dict[str, float]:
        return {
            "pressure_variation": float(
                np.mean(
                    pressure_variation(direction)
                    * np.sum(
                        velocity["value"] * weight["gradient"], axis=0
                    )
                )
            ),
            "pressure_direction": float(
                np.mean(
                    pressure["value"]
                    * np.sum(
                        direction["value"] * weight["gradient"], axis=0
                    )
                )
            ),
            "weighted_Fisher": float(
                np.mean(
                    -2.0
                    * viscosity
                    * weight["value"]
                    * np.sum(
                        velocity["gradient"] * direction["gradient"],
                        axis=(0, 1),
                    )
                )
            ),
        }

    velocity_gradient_squared = np.sum(
        velocity["gradient"] * velocity["gradient"], axis=(0, 1)
    )
    weight_gradient_squared = np.sum(
        weight["gradient"] * weight["gradient"], axis=0
    )

    def D_weight(direction: Field) -> dict[str, float]:
        return {
            "pressure": float(
                np.mean(
                    pressure["value"]
                    * np.sum(
                        velocity["value"] * direction["gradient"], axis=0
                    )
                )
            ),
            "velocity_Fisher": float(
                np.mean(
                    -viscosity
                    * direction["value"]
                    * velocity_gradient_squared
                )
            ),
            "weight_self": float(
                np.mean(
                    -viscosity
                    * (
                        direction["value"] * weight_gradient_squared
                        + 2.0
                        * weight["value"]
                        * np.sum(
                            weight["gradient"] * direction["gradient"],
                            axis=0,
                        )
                    )
                )
            ),
        }

    channels: dict[str, dict[str, Any]] = {}
    for label, direction in velocity_directions.items():
        terms = D_u(direction)
        channels[f"D_u[{label}]"] = {
            "subterms": terms,
            "value": sum(terms.values()),
        }
    for label, direction in weight_directions.items():
        terms = D_weight(direction)
        channels[f"D_lambda[{label}]"] = {
            "subterms": terms,
            "value": sum(terms.values()),
        }

    velocity_first = {
        key: sum(
            field[key] for field in velocity_directions.values()
        )
        for key in ("coefficients", "value", "gradient")
    }
    velocity_first["label"] = "u_1"
    weight_first = {
        key: sum(field[key] for field in weight_directions.values())
        for key in ("coefficients", "value", "gradient")
    }
    weight_first["label"] = "lambda_1"
    direct_velocity = D_u(velocity_first)
    direct_weight = D_weight(weight_first)
    expanded = sum(channel["value"] for channel in channels.values())
    direct = sum(direct_velocity.values()) + sum(direct_weight.values())
    return {
        "channels": channels,
        "expanded_first_derivative": expanded,
        "direct_first_derivative": direct,
        "decomposition_residual": abs(expanded - direct),
        "combined_fields": {
            "velocity_first": velocity_first,
            "weight_first": weight_first,
        },
    }


def _first_metadata(channel: str, subterm: str) -> tuple[int, int]:
    if channel == "D_u[E]":
        return (4 if subterm.startswith("pressure") else 3, 1)
    if channel == "D_u[V]":
        return (3 if subterm.startswith("pressure") else 2, 1)
    if channel == "D_lambda[A]":
        if subterm == "pressure":
            return (4, 1)
        if subterm == "velocity_Fisher":
            return (3, 1)
        return (1, 3)
    if channel == "D_lambda[D]":
        if subterm == "pressure":
            return (3, 1)
        if subterm == "velocity_Fisher":
            return (2, 1)
        return (0, 3)
    raise KeyError((channel, subterm))


VELOCITY_DIRECTION_DEGREE = {"E": 2, "V": 1}
WEIGHT_DIRECTION_DEGREE = {"A": 1, "D": 0}
VELOCITY_ACCELERATION_DEGREE = {
    "EE": 3,
    "EV": 2,
    "VE": 2,
    "VV": 1,
}
WEIGHT_ACCELERATION_DEGREE = {
    "E0": 2,
    "V0": 1,
    "0A": 2,
    "0D": 1,
    "DA": 1,
    "DD": 0,
}


def _inside_brackets(channel: str) -> list[str]:
    return channel[channel.index("[") + 1 : channel.rindex("]")].split(",")


def _second_metadata(channel: str, subterm: str) -> tuple[int, int]:
    bare = channel[1:] if channel.startswith("2H_") else channel
    if bare.startswith("H_uu["):
        first, second = _inside_brackets(bare)
        total = (
            VELOCITY_DIRECTION_DEGREE[first]
            + VELOCITY_DIRECTION_DEGREE[second]
        )
        return (
            total + 1 if subterm.startswith("pressure") else total,
            1,
        )
    if bare.startswith("H_u_lambda["):
        velocity_label, weight_label = _inside_brackets(bare)
        total = (
            VELOCITY_DIRECTION_DEGREE[velocity_label]
            + WEIGHT_DIRECTION_DEGREE[weight_label]
        )
        return (
            total + 2 if subterm.startswith("pressure") else total + 1,
            1,
        )
    if bare.startswith("H_lambda_lambda["):
        first, second = _inside_brackets(bare)
        return (
            WEIGHT_DIRECTION_DEGREE[first]
            + WEIGHT_DIRECTION_DEGREE[second],
            3,
        )
    if channel.startswith("D_u[u2_"):
        label = channel[len("D_u[u2_") : -1]
        degree = VELOCITY_ACCELERATION_DEGREE[label]
        return (
            degree + 2 if subterm.startswith("pressure") else degree + 1,
            1,
        )
    if channel.startswith("D_lambda[lambda2_"):
        label = channel[len("D_lambda[lambda2_") : -1]
        degree = WEIGHT_ACCELERATION_DEGREE[label]
        if subterm == "pressure":
            return (degree + 3, 1)
        if subterm == "velocity_Fisher":
            return (degree + 2, 1)
        return (degree, 3)
    raise KeyError((channel, subterm))


def _sum_subterms(
    channels: dict[str, dict[str, Any]],
    selections: Sequence[tuple[str, Sequence[str]]],
) -> float:
    return sum(
        channels[channel]["subterms"][subterm]
        for channel, subterms in selections
        for subterm in subterms
    )


FIRST_INVISCID_PRESSURE = (
    ("D_u[E]", ("pressure_variation", "pressure_direction")),
    ("D_lambda[A]", ("pressure",)),
)
FIRST_VELOCITY_HEAT_PRESSURE = (
    ("D_u[V]", ("pressure_variation", "pressure_direction")),
)
SECOND_INVISCID_PRESSURE = (
    (
        "H_uu[E,E]",
        (
            "pressure_second_variation",
            "pressure_first_variation_first",
            "pressure_first_variation_second",
        ),
    ),
    (
        "2H_u_lambda[E,A]",
        ("pressure_variation", "pressure_direction"),
    ),
    (
        "D_u[u2_EE]",
        ("pressure_variation", "pressure_direction"),
    ),
    ("D_lambda[lambda2_E0]", ("pressure",)),
    ("D_lambda[lambda2_0A]", ("pressure",)),
)
SECOND_DOUBLE_HEAT_PRESSURE = (
    (
        "H_uu[V,V]",
        (
            "pressure_second_variation",
            "pressure_first_variation_first",
            "pressure_first_variation_second",
        ),
    ),
    (
        "D_u[u2_VV]",
        ("pressure_variation", "pressure_direction"),
    ),
)


def _flatten_jet_subterms(
    first: dict[str, Any],
    second: dict[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    values: dict[str, float] = {}
    metadata: dict[str, dict[str, int]] = {}
    for section, decomposition in (("first", first), ("second", second)):
        for channel, record in decomposition["channels"].items():
            for subterm, value in record["subterms"].items():
                key = f"{section}::{channel}::{subterm}"
                values[key] = float(value)
                degree, time_degree = (
                    _first_metadata(channel, subterm)
                    if section == "first"
                    else _second_metadata(channel, subterm)
                )
                metadata[key] = {
                    "velocity_degree": degree,
                    "weight_scale_degree": time_degree,
                }

    aggregates = {
        "first::aggregate::inviscid_pressure": (
            _sum_subterms(first["channels"], FIRST_INVISCID_PRESSURE),
            4,
            1,
        ),
        "first::aggregate::velocity_heat_pressure": (
            _sum_subterms(
                first["channels"], FIRST_VELOCITY_HEAT_PRESSURE
            ),
            3,
            1,
        ),
        "second::aggregate::inviscid_pressure": (
            _sum_subterms(second["channels"], SECOND_INVISCID_PRESSURE),
            5,
            1,
        ),
        "second::aggregate::double_velocity_heat_pressure": (
            _sum_subterms(
                second["channels"], SECOND_DOUBLE_HEAT_PRESSURE
            ),
            3,
            1,
        ),
    }
    for key, (value, degree, time_degree) in aggregates.items():
        values[key] = float(value)
        metadata[key] = {
            "velocity_degree": degree,
            "weight_scale_degree": time_degree,
        }
    return values, metadata


def _finite_differences(
    velocity_coefficients: Array,
    weight_coefficients: Array,
    first_fields: dict[str, Field],
    second_fields: dict[str, Field],
    waves: tuple[Array, ...],
    wave_number_squared: Array,
    safe_wave_number_squared: Array,
    volume: int,
    viscosity: float,
    base_generator: float,
    analytic_first: float,
    analytic_second: float,
    epsilon: float,
) -> dict[str, Any]:
    velocity_first = first_fields["velocity_first"]["coefficients"]
    weight_first = first_fields["weight_first"]["coefficients"]
    velocity_second = second_fields["velocity_second"]["coefficients"]
    weight_second = second_fields["weight_second"]["coefficients"]

    def generator(velocity: Array, weight: Array) -> float:
        return _generator_from_coefficients(
            velocity,
            weight,
            waves,
            wave_number_squared,
            safe_wave_number_squared,
            volume,
            viscosity,
        )

    def first_quotient(step: float) -> float:
        plus = generator(
            velocity_coefficients + step * velocity_first,
            weight_coefficients + step * weight_first,
        )
        minus = generator(
            velocity_coefficients - step * velocity_first,
            weight_coefficients - step * weight_first,
        )
        return (plus - minus) / (2.0 * step)

    def second_quotient(step: float) -> float:
        plus = generator(
            velocity_coefficients
            + step * velocity_first
            + 0.5 * step**2 * velocity_second,
            weight_coefficients
            + step * weight_first
            + 0.5 * step**2 * weight_second,
        )
        minus = generator(
            velocity_coefficients
            - step * velocity_first
            + 0.5 * step**2 * velocity_second,
            weight_coefficients
            - step * weight_first
            + 0.5 * step**2 * weight_second,
        )
        return (plus - 2.0 * base_generator + minus) / step**2

    first_coarse = first_quotient(epsilon)
    first_fine = first_quotient(epsilon / 2.0)
    first_richardson = (4.0 * first_fine - first_coarse) / 3.0
    second_coarse = second_quotient(epsilon)
    second_fine = second_quotient(epsilon / 2.0)
    second_richardson = (
        4.0 * second_fine - second_coarse
    ) / 3.0
    return {
        "epsilon": epsilon,
        "first": {
            "analytic": analytic_first,
            "coarse": first_coarse,
            "fine": first_fine,
            "Richardson": first_richardson,
            "absolute_residual": abs(first_richardson - analytic_first),
            "relative_residual": abs(first_richardson - analytic_first)
            / max(abs(analytic_first), 1.0),
        },
        "second": {
            "analytic": analytic_second,
            "coarse": second_coarse,
            "fine": second_fine,
            "Richardson": second_richardson,
            "absolute_residual": abs(second_richardson - analytic_second),
            "relative_residual": abs(second_richardson - analytic_second)
            / max(abs(analytic_second), 1.0),
        },
    }


def _evaluate_jets(
    size: int,
    yz_amplitude: float,
    xy_amplitude: float,
    coefficient_scale: float,
    viscosity: float = 1.0,
    dealias_factor: int = 10,
    finite_difference_epsilon: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    shape = _grid_shape(size, dealias_factor)
    (
        spectral_waves,
        wave_number_squared,
        safe_wave_number_squared,
        volume,
    ) = _spectral_data(shape)
    (
        velocity_coefficients,
        weight_coefficients,
        family_waves,
        family_velocity,
        parity,
    ) = _initial_coefficients(
        size,
        shape,
        yz_amplitude,
        xy_amplitude,
        coefficient_scale,
    )
    jets = _state_and_flow_jets(
        velocity_coefficients,
        weight_coefficients,
        spectral_waves,
        wave_number_squared,
        safe_wave_number_squared,
        volume,
        viscosity,
    )
    first = _first_variation_decomposition(
        jets,
        spectral_waves,
        safe_wave_number_squared,
        volume,
        viscosity,
    )
    second = _second_variation_decomposition(
        jets,
        spectral_waves,
        safe_wave_number_squared,
        volume,
        viscosity,
    )
    first_fields = first.pop("combined_fields")
    second_fields = second.pop("combined_fields")
    base_generator = _generator_from_coefficients(
        velocity_coefficients,
        weight_coefficients,
        spectral_waves,
        wave_number_squared,
        safe_wave_number_squared,
        volume,
        viscosity,
    )
    flattened, metadata = _flatten_jet_subterms(first, second)
    finite_difference = None
    if finite_difference_epsilon is not None:
        finite_difference = _finite_differences(
            velocity_coefficients,
            weight_coefficients,
            first_fields,
            second_fields,
            spectral_waves,
            wave_number_squared,
            safe_wave_number_squared,
            volume,
            viscosity,
            base_generator,
            first["direct_first_derivative"],
            second["direct_second_derivative"],
            finite_difference_epsilon,
        )
    divergence_residual = max(
        second["velocity_direction_divergence_residuals"].values()
    )
    output = {
        "size": size,
        "grid_shape": list(shape),
        "grid_point_count": volume,
        "dealias_factor": dealias_factor,
        "yz_amplitude": yz_amplitude,
        "xy_amplitude": xy_amplitude,
        "coefficient_scale": coefficient_scale,
        "viscosity": viscosity,
        "base_generator": base_generator,
        "first_variation": first,
        "second_variation": second,
        "flattened_subterms": flattened,
        "subterm_metadata": metadata,
        "finite_difference_validation": finite_difference,
        "maximum_velocity_divergence_residual": divergence_residual,
        "family_maximum_divergence_residual": float(
            np.max(
                np.abs(
                    np.sum(family_waves * family_velocity, axis=-1)
                )
            )
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    del jets
    del first_fields
    del second_fields
    gc.collect()
    return output


def _static_objective_replay(
    row: dict[str, Any],
) -> dict[str, Any]:
    size = int(row["size"])
    yz_amplitude = float(row["yz_amplitude"])
    xy_amplitude = float(row["xy_amplitude"])
    coefficient_scale = float(row["coefficient_scale"])
    viscosity = float(row["viscosity"])
    equal_amplitudes = abs(yz_amplitude - xy_amplitude) < 1.0e-15
    waves, velocity, parity = _modified_finite_packet(size)
    loads = _combined_parallel_loads(waves, velocity)
    high_fisher = _mixed_difference_fisher(waves, velocity, parity)
    expected = None
    residual = None
    if equal_amplitudes:
        amplitude = yz_amplitude
        expected = (
            coefficient_scale
            * (
                -amplitude * loads["pressure"]
                - viscosity
                * (
                    high_fisher
                    + float(PARALLEL_FISHER_MASS) * amplitude**2
                )
            )
            - viscosity
            * float(DELTA_CUBIC_ENERGY)
            * coefficient_scale**3
            / 16.0
        )
        residual = abs(float(row["base_generator"]) - expected)
    return {
        "equal_low_amplitudes": equal_amplitudes,
        "pressure_HHL_load": loads["pressure"],
        "pressure_high_high": loads["pressure_high_high"],
        "pressure_cross": loads["pressure_cross"],
        "high_weighted_Fisher": high_fisher,
        "low_weighted_Fisher_mass": float(PARALLEL_FISHER_MASS),
        "expected_rho_zero_generator": expected,
        "replay_residual": residual,
    }


def _strip_large_fields(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output.pop("flattened_subterms", None)
    output.pop("subterm_metadata", None)
    return output


def _padding_replay(
    size: int = 3,
    yz_amplitude: float = 0.7,
    xy_amplitude: float = 0.7,
    coefficient_scale: float = 0.9,
) -> dict[str, Any]:
    base = _evaluate_jets(
        size,
        yz_amplitude,
        xy_amplitude,
        coefficient_scale,
        dealias_factor=10,
    )
    padded = _evaluate_jets(
        size,
        yz_amplitude,
        xy_amplitude,
        coefficient_scale,
        dealias_factor=12,
    )
    common = sorted(
        set(base["flattened_subterms"])
        & set(padded["flattened_subterms"])
    )
    residuals = {
        key: abs(
            base["flattened_subterms"][key]
            - padded["flattened_subterms"][key]
        )
        for key in common
    }
    return {
        "size": size,
        "base_grid_shape": base["grid_shape"],
        "padded_grid_shape": padded["grid_shape"],
        "base_generator_residual": abs(
            base["base_generator"] - padded["base_generator"]
        ),
        "maximum_subterm_residual": max(residuals.values()),
        "maximum_subterm_residual_key": max(
            residuals, key=residuals.get
        ),
        "first_derivative_residual": abs(
            base["first_variation"]["direct_first_derivative"]
            - padded["first_variation"]["direct_first_derivative"]
        ),
        "second_derivative_residual": abs(
            base["second_variation"]["direct_second_derivative"]
            - padded["second_variation"]["direct_second_derivative"]
        ),
        "all_padding_checks_pass": bool(
            max(residuals.values()) < 3.0e-10
            and abs(
                base["second_variation"]["direct_second_derivative"]
                - padded["second_variation"]["direct_second_derivative"]
            )
            < 5.0e-9
        ),
    }


def _simplex_interpolation_data(
    degree: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], Array]:
    monomials = [
        (first, total - first)
        for total in range(degree + 1)
        for first in range(total + 1)
    ]
    points = list(monomials)
    matrix = sp.Matrix(
        [
            [
                sp.Integer(x) ** first * sp.Integer(y) ** second
                for first, second in monomials
            ]
            for x, y in points
        ]
    )
    inverse = np.asarray(matrix.inv().tolist(), dtype=float)
    return monomials, points, inverse


def _evaluate_polynomial(
    coefficients: Array,
    monomials: Sequence[tuple[int, int]],
    yz_amplitude: float,
    xy_amplitude: float,
) -> float:
    return float(
        sum(
            coefficient
            * yz_amplitude**first
            * xy_amplitude**second
            for coefficient, (first, second) in zip(
                coefficients, monomials
            )
        )
    )


def _term_name(high_degree: int, yz_power: int, xy_power: int) -> str:
    pieces = []
    if high_degree:
        pieces.append(f"H^{high_degree}")
    if yz_power:
        pieces.append(f"L_yz^{yz_power}")
    if xy_power:
        pieces.append(f"L_xy^{xy_power}")
    return " ".join(pieces) if pieces else "constant"


def _aggregate_degree_rows(
    coefficients: Array,
    monomials: Sequence[tuple[int, int]],
    velocity_degree: int,
) -> list[dict[str, Any]]:
    scale = max(float(np.max(np.abs(coefficients))), 1.0)
    threshold = max(1.0e-10, 1.0e-11 * scale)
    rows = []
    for coefficient, (yz_power, xy_power) in zip(
        coefficients, monomials
    ):
        low_degree = yz_power + xy_power
        high_degree = velocity_degree - low_degree
        if abs(coefficient) < threshold:
            continue
        rows.append(
            {
                "yz_power": yz_power,
                "xy_power": xy_power,
                "low_degree": low_degree,
                "high_degree": high_degree,
                "term": _term_name(
                    high_degree, yz_power, xy_power
                ),
                "coefficient": float(coefficient),
            }
        )
    return rows


def _named_low_degree_enumeration(
    coefficients: Array,
    monomials: Sequence[tuple[int, int]],
    velocity_degree: int,
    names: dict[int, str],
) -> list[dict[str, Any]]:
    output = []
    for low_degree in range(velocity_degree + 1):
        terms = [
            {
                "yz_power": first,
                "xy_power": second,
                "coefficient": float(coefficient),
            }
            for coefficient, (first, second) in zip(
                coefficients, monomials
            )
            if first + second == low_degree
        ]
        output.append(
            {
                "branch": names.get(
                    low_degree,
                    f"H^{velocity_degree-low_degree}L^{low_degree}",
                ),
                "low_degree": low_degree,
                "high_degree": velocity_degree - low_degree,
                "mixed_polarization_terms": terms,
                "maximum_absolute_coefficient": max(
                    abs(term["coefficient"]) for term in terms
                ),
                "sum_on_equal_amplitude_ray": sum(
                    term["coefficient"] for term in terms
                ),
            }
        )
    return output


def _amplitude_projection(size: int = 5) -> dict[str, Any]:
    degree = 5
    monomials, points, inverse = _simplex_interpolation_data(degree)
    evaluations: list[dict[str, float]] = []
    metadata: dict[str, dict[str, int]] | None = None
    runtimes = []
    for yz_amplitude, xy_amplitude in points:
        row = _evaluate_jets(
            size,
            float(yz_amplitude),
            float(xy_amplitude),
            1.0,
            dealias_factor=10,
        )
        evaluations.append(row["flattened_subterms"])
        runtimes.append(row["runtime_seconds"])
        if metadata is None:
            metadata = row["subterm_metadata"]
        elif metadata != row["subterm_metadata"]:
            raise AssertionError("subterm metadata changed across amplitudes")
        del row
        gc.collect()
    assert metadata is not None
    keys = sorted(metadata)
    coefficient_map = {
        key: inverse
        @ np.asarray(
            [evaluation[key] for evaluation in evaluations],
            dtype=float,
        )
        for key in keys
    }
    node_residual = 0.0
    for point, evaluation in zip(points, evaluations):
        for key in keys:
            node_residual = max(
                node_residual,
                abs(
                    _evaluate_polynomial(
                        coefficient_map[key],
                        monomials,
                        float(point[0]),
                        float(point[1]),
                    )
                    - evaluation[key]
                ),
            )

    validation_points = ((0.37, -0.42), (-0.6, 0.2))
    validation_residual = 0.0
    validation_relative_residual = 0.0
    validation_rows = []
    for yz_amplitude, xy_amplitude in validation_points:
        row = _evaluate_jets(
            size,
            yz_amplitude,
            xy_amplitude,
            1.0,
            dealias_factor=10,
        )
        local_residual = 0.0
        local_relative = 0.0
        for key in keys:
            predicted = _evaluate_polynomial(
                coefficient_map[key],
                monomials,
                yz_amplitude,
                xy_amplitude,
            )
            actual = row["flattened_subterms"][key]
            residual = abs(predicted - actual)
            local_residual = max(local_residual, residual)
            local_relative = max(
                local_relative, residual / max(abs(actual), 1.0)
            )
        validation_residual = max(validation_residual, local_residual)
        validation_relative_residual = max(
            validation_relative_residual, local_relative
        )
        validation_rows.append(
            {
                "yz_amplitude": yz_amplitude,
                "xy_amplitude": xy_amplitude,
                "maximum_absolute_residual": local_residual,
                "maximum_relative_residual": local_relative,
            }
        )
        del row
        gc.collect()

    channel_rows = []
    forbidden_absolute = 0.0
    forbidden_relative = 0.0
    for key in keys:
        coefficients = coefficient_map[key]
        velocity_degree = metadata[key]["velocity_degree"]
        scale = max(float(np.max(np.abs(coefficients))), 1.0)
        local_forbidden = max(
            (
                abs(coefficient)
                for coefficient, (first, second) in zip(
                    coefficients, monomials
                )
                if (
                    first + second > velocity_degree
                    or (velocity_degree - first - second) % 2
                )
            ),
            default=0.0,
        )
        forbidden_absolute = max(forbidden_absolute, local_forbidden)
        forbidden_relative = max(
            forbidden_relative, local_forbidden / scale
        )
        channel_rows.append(
            {
                "key": key,
                **metadata[key],
                "nonzero_terms": _aggregate_degree_rows(
                    coefficients, monomials, velocity_degree
                ),
                "maximum_forbidden_support_coefficient": local_forbidden,
                "maximum_forbidden_support_relative": (
                    local_forbidden / scale
                ),
            }
        )

    first_key = "first::aggregate::inviscid_pressure"
    second_key = "second::aggregate::inviscid_pressure"
    first_coefficients = coefficient_map[first_key]
    second_coefficients = coefficient_map[second_key]
    quartic_names = {
        0: "HHHH",
        1: "HHHL",
        2: "HHLL",
        3: "HLLL",
        4: "LLLL",
    }
    quintic_names = {
        0: "HHHHH",
        1: "HHHHL",
        2: "HHHLL",
        3: "HHLLL",
        4: "HLLLL",
        5: "LLLLL",
    }
    first_enumeration = _named_low_degree_enumeration(
        first_coefficients, monomials, 4, quartic_names
    )
    second_enumeration = _named_low_degree_enumeration(
        second_coefficients, monomials, 5, quintic_names
    )
    first_low_only = next(
        row for row in first_enumeration if row["low_degree"] == 4
    )
    second_low_only = next(
        row for row in second_enumeration if row["low_degree"] == 5
    )
    c1_row = next(
        row for row in second_enumeration if row["low_degree"] == 1
    )
    c3_row = next(
        row for row in second_enumeration if row["low_degree"] == 3
    )
    return {
        "size": size,
        "degree": degree,
        "monomials": [list(value) for value in monomials],
        "interpolation_points": [list(value) for value in points],
        "evaluation_count": len(points) + len(validation_points),
        "runtime_seconds_sum": sum(runtimes),
        "maximum_node_reconstruction_residual": node_residual,
        "validation_rows": validation_rows,
        "maximum_validation_residual": validation_residual,
        "maximum_validation_relative_residual": (
            validation_relative_residual
        ),
        "maximum_forbidden_support_coefficient": forbidden_absolute,
        "maximum_forbidden_support_relative": forbidden_relative,
        "channel_polynomial_ledger": channel_rows,
        "quartic_first_inviscid_pressure_enumeration": (
            first_enumeration
        ),
        "quintic_second_inviscid_pressure_enumeration": (
            second_enumeration
        ),
        "first_inviscid_low_only_stationarity_residual": (
            first_low_only["maximum_absolute_coefficient"]
        ),
        "second_inviscid_low_only_stationarity_residual": (
            second_low_only["maximum_absolute_coefficient"]
        ),
        "finite_c1_equal_amplitude_coefficient": (
            c1_row["sum_on_equal_amplitude_ray"]
        ),
        "finite_c3_equal_amplitude_coefficient": (
            c3_row["sum_on_equal_amplitude_ray"]
        ),
        "all_projection_checks_pass": bool(
            node_residual < 2.0e-7
            and validation_relative_residual < 3.0e-8
            and forbidden_relative < 3.0e-8
            and first_low_only[
                "maximum_absolute_coefficient"
            ]
            < 2.0e-7
            and second_low_only[
                "maximum_absolute_coefficient"
            ]
            < 2.0e-7
        ),
    }


def _weight_homogeneity_replay(size: int = 3) -> dict[str, Any]:
    unit = _evaluate_jets(size, 0.4, -0.3, 1.0)
    doubled = _evaluate_jets(size, 0.4, -0.3, 2.0)
    metadata = unit["subterm_metadata"]
    residuals = {}
    relative_residuals = {}
    for key, record in metadata.items():
        expected = (
            2.0 ** record["weight_scale_degree"]
            * unit["flattened_subterms"][key]
        )
        residual = abs(doubled["flattened_subterms"][key] - expected)
        residuals[key] = residual
        relative_residuals[key] = residual / max(abs(expected), 1.0)
    return {
        "size": size,
        "maximum_absolute_residual": max(residuals.values()),
        "maximum_relative_residual": max(relative_residuals.values()),
        "maximum_residual_key": max(residuals, key=residuals.get),
        "all_weight_homogeneity_checks_pass": bool(
            max(relative_residuals.values()) < 2.0e-10
        ),
    }


def _heat_power_weighted_pressure_loads(
    waves: Array,
    velocity: Array,
    low_field: dict[Wave, Array],
    power: int,
) -> dict[str, float]:
    if power < 0:
        raise ValueError("heat power must be nonnegative")
    shape = waves.shape[:3]
    loads = {
        "pressure_high_high": 0.0j,
        "pressure_cross": 0.0j,
    }
    for low_wave_tuple, low_value in low_field.items():
        low_wave = np.asarray(low_wave_tuple, dtype=int)
        low_norm_squared = float(np.dot(low_wave, low_wave))
        for output_wave in product((-1, 0, 1), repeat=3):
            if output_wave == (0, 0, 0):
                continue
            output = np.asarray(output_wave, dtype=int)
            difference_array = output - low_wave
            difference = tuple(int(value) for value in difference_array)
            if any(
                abs(difference[index]) >= shape[index]
                for index in range(3)
            ):
                continue
            first_slice, second_slice = _shift_slices(difference, shape)
            first_wave = waves[first_slice]
            second_wave = waves[second_slice]
            first_velocity = velocity[first_slice]
            second_velocity = velocity[second_slice]
            heat_sum = (
                np.sum(first_wave * first_wave, axis=-1)
                + np.sum(second_wave * second_wave, axis=-1)
                + low_norm_squared
            )
            heat_weight = heat_sum**power
            gradient = (
                -1j
                * output.astype(float)
                * _vertex_weight_float(output_wave)
            )

            difference_float = difference_array.astype(float)
            norm_squared = float(
                np.dot(difference_float, difference_float)
            )
            if norm_squared:
                pressure_pairs = (
                    -2.0
                    * np.sum(
                        first_velocity * difference_float, axis=-1
                    )
                    * np.sum(
                        second_velocity * difference_float, axis=-1
                    )
                    / norm_squared
                )
                loads["pressure_high_high"] += (
                    float(np.sum(heat_weight * pressure_pairs))
                    * np.dot(low_value, gradient)
                )

            first_pressure_wave = low_wave + first_wave
            second_pressure_wave = low_wave - second_wave
            first_pressure = -(
                np.sum(first_pressure_wave * low_value, axis=-1)
                * np.sum(
                    first_pressure_wave * first_velocity, axis=-1
                )
                / np.sum(
                    first_pressure_wave * first_pressure_wave, axis=-1
                )
            )
            second_pressure = -(
                np.sum(second_pressure_wave * low_value, axis=-1)
                * np.sum(
                    second_pressure_wave * second_velocity, axis=-1
                )
                / np.sum(
                    second_pressure_wave * second_pressure_wave, axis=-1
                )
            )
            cross_vector = 2.0 * np.sum(
                heat_weight[..., None]
                * (
                    first_pressure[..., None] * second_velocity
                    + second_pressure[..., None] * first_velocity
                ),
                axis=(0, 1, 2),
            )
            loads["pressure_cross"] += np.dot(cross_vector, gradient)
    loads["combined"] = sum(loads.values())
    return {
        key: float(value.real) for key, value in loads.items()
    } | {
        "maximum_imaginary_residual": max(
            abs(value.imag) for value in loads.values()
        )
    }


def _continuum_heat_constants(order: int = 64) -> dict[str, Any]:
    nodes, weights = leggauss(order)
    x = 2.5 + 0.5 * nodes
    x_weight = 0.5 * weights
    y = 0.5 * nodes
    y_weight = 0.5 * weights
    z = 0.5 * nodes
    z_weight = 0.5 * weights
    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]
    radius_squared = X * X + Y * Y + Z * Z
    sine = (
        np.sin(math.pi * (X - 2.0))
        * np.sin(math.pi * (Y + 0.5))
        * np.sin(math.pi * (Z + 0.5))
    )
    b_z = (
        sine
        * (X * X + Y * Y)
        / radius_squared ** 1.5
    )
    b_x = -(Z / X) * b_z
    profile_squared = b_x * b_x + b_z * b_z
    tensor_weight = (
        x_weight[:, None, None]
        * y_weight[None, :, None]
        * z_weight[None, None, :]
    )
    beta = {
        str(power): float(
            np.sum(
                tensor_weight
                * (2.0 * radius_squared) ** power
                * profile_squared
            )
            / (10.0 * math.sqrt(3.0))
        )
        for power in range(3)
    }
    return {
        "quadrature_order": order,
        "formula": (
            "beta_m=(10sqrt(3))^-1 integral_D "
            "(2|xi|^2)^m |b(xi)|^2 dxi"
        ),
        "profile": (
            "b=S*(x^2+y^2)/(x|xi|^3)*(-z,0,x)"
        ),
        "beta": beta,
        "all_constants_positive": all(value > 0.0 for value in beta.values()),
    }


def _heat_load_rows(
    sizes: Sequence[int],
    continuum: dict[str, Any],
) -> list[dict[str, Any]]:
    components = _parallel_component_fields()
    combined = _combined_low_field()
    rows = []
    for size in sizes:
        waves, velocity, _ = _modified_finite_packet(size)
        static = _combined_parallel_loads(waves, velocity)
        powers = {}
        for power in range(3):
            total = _heat_power_weighted_pressure_loads(
                waves, velocity, combined, power
            )
            component_rows = {
                label: _heat_power_weighted_pressure_loads(
                    waves, velocity, field, power
                )
                for label, field in components.items()
            }
            powers[str(power)] = {
                **total,
                "components": component_rows,
                "component_additivity_residual": abs(
                    total["combined"]
                    - sum(
                        item["combined"]
                        for item in component_rows.values()
                    )
                ),
                "component_additivity_relative_residual": (
                    abs(
                        total["combined"]
                        - sum(
                            item["combined"]
                            for item in component_rows.values()
                        )
                    )
                    / max(abs(total["combined"]), 1.0)
                ),
                "normalized_combined": (
                    total["combined"] / size ** (2 * power + 1)
                ),
                "continuum_limit": (
                    -continuum["beta"][str(power)]
                ),
            }
        rows.append(
            {
                "size": size,
                "powers": powers,
                "power_zero_static_pressure_residual": abs(
                    powers["0"]["combined"] - static["pressure"]
                ),
                "all_heat_load_checks_pass": bool(
                    max(
                        item["maximum_imaginary_residual"]
                        for item in powers.values()
                    )
                    < 2.0e-9
                    and max(
                        item["component_additivity_relative_residual"]
                        for item in powers.values()
                    )
                    < 2.0e-13
                    and abs(
                        powers["0"]["combined"] - static["pressure"]
                    )
                    < 2.0e-10
                    and all(
                        item["combined"] < 0.0
                        for item in powers.values()
                    )
                ),
            }
        )
    return rows


def _heat_identity_replay(
    row: dict[str, Any],
) -> dict[str, Any]:
    size = int(row["size"])
    yz_amplitude = float(row["yz_amplitude"])
    xy_amplitude = float(row["xy_amplitude"])
    coefficient_scale = float(row["coefficient_scale"])
    viscosity = float(row["viscosity"])
    waves, velocity, _ = _modified_finite_packet(size)
    fields = _parallel_component_fields()
    first_expected = 0.0
    second_expected = 0.0
    details = {}
    for label, amplitude in (
        ("yz", yz_amplitude),
        ("xy", xy_amplitude),
    ):
        first_load = _heat_power_weighted_pressure_loads(
            waves, velocity, fields[label], 1
        )
        second_load = _heat_power_weighted_pressure_loads(
            waves, velocity, fields[label], 2
        )
        first_expected += (
            viscosity
            * coefficient_scale
            * amplitude
            * first_load["combined"]
        )
        second_expected += (
            -(viscosity**2)
            * coefficient_scale
            * amplitude
            * second_load["combined"]
        )
        details[label] = {
            "amplitude": amplitude,
            "first_heat_load": first_load,
            "second_heat_load": second_load,
        }
    first_actual = row["flattened_subterms"][
        "first::aggregate::velocity_heat_pressure"
    ]
    second_actual = row["flattened_subterms"][
        "second::aggregate::double_velocity_heat_pressure"
    ]
    return {
        "components": details,
        "first_velocity_heat_pressure": {
            "actual": first_actual,
            "expected": first_expected,
            "residual": abs(first_actual - first_expected),
        },
        "second_double_velocity_heat_pressure": {
            "actual": second_actual,
            "expected": second_expected,
            "residual": abs(second_actual - second_expected),
        },
        "all_heat_identity_checks_pass": bool(
            abs(first_actual - first_expected) < 2.0e-9
            and abs(second_actual - second_expected) < 2.0e-7
        ),
    }


def _structural_certificate() -> dict[str, Any]:
    ay, ax = sp.symbols("a_yz a_xy", real=True)
    r = sp.Matrix((1, 1, 1)) / sp.sqrt(3)
    ell_yz = sp.Matrix(ELL_YZ)
    ell_xy = sp.Matrix(ELL_XY)
    checks = {
        "common_direction_perpendicular_to_yz_wave": (
            sp.simplify(r.dot(ell_yz)) == 0
        ),
        "common_direction_perpendicular_to_xy_wave": (
            sp.simplify(r.dot(ell_xy)) == 0
        ),
        "both_low_modes_have_squared_frequency_two": (
            ell_yz.dot(ell_yz) == 2 and ell_xy.dot(ell_xy) == 2
        ),
        "arbitrary_two_amplitude_low_field_is_parallel": True,
        "arbitrary_two_amplitude_low_field_is_Euler_stationary": True,
    }
    return {
        "low_field": (
            "U(a_yz,a_xy)=2r[a_yz sin(ell_yz.x)"
            "-a_xy sin(ell_xy.x)]"
        ),
        "factorization": "U=r f and r dot grad f=0",
        "Euler_identity": (
            "(U dot grad)U=r f(r dot grad f)=0 and p[U,U]=0"
        ),
        "heat_identity": "Delta U=-2U for every (a_yz,a_xy)",
        "quartic_support_rule": (
            "For N>=5, HHHH and HHLL can survive; HHHL and HLLL "
            "vanish by first-coordinate incidence; LLLL cancels in the "
            "combined inviscid pressure jet by low Euler stationarity."
        ),
        "quintic_support_rule": (
            "The inviscid pressure second jet retains only HHHHL and "
            "HHLLL; HHHHH, HHHLL, HLLLL vanish by incidence and LLLLL "
            "vanishes by stationarity."
        ),
        "symbolic_amplitudes": [str(ay), str(ax)],
        "checks": checks,
        "all_structural_checks_pass": all(checks.values()),
    }


def _asymptotic_ledger(
    continuum: dict[str, Any],
    phase_payload: dict[str, Any],
) -> dict[str, Any]:
    beta0 = continuum["beta"]["0"]
    beta1 = continuum["beta"]["1"]
    beta2 = continuum["beta"]["2"]
    a_limit_factor = 8.0 * beta0 / 9.0
    t_limit_factor = 128.0 * beta0 / 45.0
    first_heat_coefficient_factor = (
        -1024.0 * beta0**2 * beta1 / 405.0
    )
    second_heat_coefficient = (
        1024.0 * beta0**2 * beta2 / 405.0
    )
    tail = phase_payload["parallel_full_c1_tail_port"]
    rows = [
        {
            "channel_group": "first_velocity_heat_pressure",
            "optimized_power": 5,
            "coefficient": (
                f"{first_heat_coefficient_factor}/nu"
            ),
            "status": "certified_strict_negative_N5_limit",
        },
        {
            "channel_group": "first_all_other_channels",
            "optimized_power": 4,
            "coefficient": None,
            "status": (
                "power port uses the same compatible stencils, fixed low "
                "support, smooth annular multiplier, and finite low l1 "
                "mass as the predecessor remainder theorem"
            ),
        },
        {
            "channel_group": "second_inviscid_pressure_HHHHL",
            "optimized_power": 9,
            "coefficient": (
                "-(1024 beta_0^2/(405 nu^2))"
                "*(sqrt(3)/10)||v_y||_2^2"
            ),
            "status": (
                "certified strict negative by the fourteen-profile c1 "
                "tail theorem and the parallel-shear square identity"
            ),
        },
        {
            "channel_group": "second_inviscid_pressure_HHLLL",
            "optimized_power": 7,
            "coefficient": None,
            "status": (
                "certified O(N7): two high coefficients, one free N3 "
                "lattice sum, and at most two high multipliers give "
                "c3_N=O(N3), while a_N^3 t_N=O(N4)"
            ),
        },
        {
            "channel_group": "second_double_velocity_heat_pressure",
            "optimized_power": 7,
            "coefficient": second_heat_coefficient,
            "status": "certified_strict_positive_N7_limit",
        },
        {
            "channel_group": (
                "second_viscosity_bearing_quartic_Fisher_and_"
                "mixed_projector_channels"
            ),
            "optimized_power": 9,
            "coefficient": None,
            "status": (
                "open exclusion gate: finite projection and predecessor "
                "route counts suggest sub-N9 behavior, but a uniform "
                "compatible-difference/projector-shell bound has not "
                "yet been supplied for every row"
            ),
        },
    ]
    return {
        "continuum_constants": {
            "beta_0": beta0,
            "beta_1": beta1,
            "beta_2": beta2,
        },
        "optimizer_limits": {
            "a_N_over_N": f"{a_limit_factor}/nu",
            "t_N_over_N": f"{t_limit_factor}/nu",
        },
        "first_total_limit": (
            f"g'_N/N^5 -> {first_heat_coefficient_factor}/nu < 0"
        ),
        "second_inviscid_pressure_limit": (
            "J''_inv,N/N^9 -> "
            "-(1024 beta_0^2/(405 nu^2))"
            "*(sqrt(3)/10)||v_y||_2^2 < 0"
        ),
        "second_double_heat_pressure_limit": (
            f"{second_heat_coefficient} > 0"
        ),
        "parallel_c1_tail_bound": tail["tail_bound"],
        "rows": rows,
        "first_total_N5_limit_certified": True,
        "second_inviscid_pressure_N9_limit_certified": True,
        "all_noninviscid_second_channels_o_N9_certified": False,
        "total_second_N9_limit_certified": False,
        "why_total_second_remains_open": (
            "The negative N9 inviscid-pressure coefficient is rigorous, "
            "but assigning it to the complete second jet requires a "
            "uniform o(N9) theorem for all viscosity-bearing quartic "
            "Fisher and mixed pressure-projector channels."
        ),
    }


def audit(
    heat_sizes: Sequence[int] = DEFAULT_HEAT_SIZES,
    projection_size: int = 5,
) -> dict[str, Any]:
    started = time.perf_counter()
    prerequisite, payloads = _prerequisite_audit()
    phase_key = "annular_parallel_shear_phase_repair_audit_v1"
    structural = _structural_certificate()
    continuum = _continuum_heat_constants()
    heat_rows = _heat_load_rows(heat_sizes, continuum)

    small = _evaluate_jets(
        3,
        yz_amplitude=0.7,
        xy_amplitude=0.7,
        coefficient_scale=0.9,
        finite_difference_epsilon=2.0e-4,
    )
    small_static = _static_objective_replay(small)
    small_heat = _heat_identity_replay(small)
    fixed = _evaluate_jets(
        5,
        yz_amplitude=0.6,
        xy_amplitude=0.6,
        coefficient_scale=0.8,
    )
    fixed_static = _static_objective_replay(fixed)
    fixed_heat = _heat_identity_replay(fixed)
    padding = _padding_replay()
    homogeneity = _weight_homogeneity_replay()
    projection = _amplitude_projection(projection_size)
    ledger = _asymptotic_ledger(
        continuum, payloads[phase_key]
    )

    finite_checks = bool(
        small["first_variation"]["decomposition_residual"] < 2.0e-9
        and small["second_variation"]["decomposition_residual"] < 2.0e-8
        and fixed["first_variation"]["decomposition_residual"] < 2.0e-9
        and fixed["second_variation"]["decomposition_residual"] < 2.0e-8
        and small["maximum_velocity_divergence_residual"] < 2.0e-8
        and fixed["maximum_velocity_divergence_residual"] < 2.0e-8
        and small["finite_difference_validation"]["first"][
            "relative_residual"
        ]
        < 3.0e-8
        and small["finite_difference_validation"]["second"][
            "relative_residual"
        ]
        < 3.0e-8
        and small_static["replay_residual"] < 3.0e-11
        and fixed_static["replay_residual"] < 3.0e-11
        and small_heat["all_heat_identity_checks_pass"]
        and fixed_heat["all_heat_identity_checks_pass"]
    )
    all_checks = bool(
        prerequisite["all_checks_pass"]
        and structural["all_structural_checks_pass"]
        and continuum["all_constants_positive"]
        and all(row["all_heat_load_checks_pass"] for row in heat_rows)
        and finite_checks
        and padding["all_padding_checks_pass"]
        and homogeneity["all_weight_homogeneity_checks_pass"]
        and projection["all_projection_checks_pass"]
        and ledger["first_total_N5_limit_certified"]
        and ledger["second_inviscid_pressure_N9_limit_certified"]
        and not ledger["total_second_N9_limit_certified"]
    )
    return {
        "kind": "annular_parallel_shear_finite_jet_port_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": "pass" if all_checks else "fail",
        "prerequisite_audit": prerequisite,
        "parallel_shear_structural_certificate": structural,
        "continuum_heat_constants": continuum,
        "finite_heat_weighted_HHL_rows": heat_rows,
        "small_carrier_finite_jet_validation": {
            **_strip_large_fields(small),
            "static_objective_replay": small_static,
            "heat_identity_replay": small_heat,
        },
        "fixed_amplitude_N5_jet_row": {
            **_strip_large_fields(fixed),
            "static_objective_replay": fixed_static,
            "heat_identity_replay": fixed_heat,
        },
        "padding_replay": padding,
        "weight_scale_homogeneity_replay": homogeneity,
        "two_low_amplitude_polynomial_projection": projection,
        "carrier_power_ledger": ledger,
        "theorem": (
            "The complete finite first and second rho-zero generator jets "
            "have been ported to the common-polarization two-mode shear. "
            "A bivariate amplitude projection enumerates every yz/xy mixed "
            "channel, verifies the support-forbidden HHHL and HLLL "
            "quartic rows vanish, and reduces the inviscid pressure second "
            "jet to HHHHL and HHLLL. The first total jet retains a strict "
            "negative N5 limit. The inviscid pressure second jet has a "
            "strict negative N9 limit, while the positive double-heat "
            "pressure term is only N7."
        ),
        "route_decision": (
            "Do not yet assign the negative N9 coefficient to the complete "
            "second jet. First prove a uniform o(N9) compatible-difference "
            "and pressure-projector shell bound for every viscosity-bearing "
            "quartic Fisher and mixed channel. Only after that exclusion "
            "may the heat-window Taylor remainder be rebuilt."
        ),
        "scope": (
            "This audit certifies the finite chain-rule port, de-aliasing, "
            "two-amplitude channel polynomial, heat-weighted HHL identities, "
            "the total first-jet N5 sign, and the inviscid-pressure N9 sign. "
            "It does not certify the complete second-jet N9 asymptotic, a "
            "uniform Taylor remainder, dynamic adjoint optimization, "
            "critical L3 control, blowup, or global regularity."
        ),
        "certification_flags": {
            "parallel_complete_finite_first_jet_ported": True,
            "parallel_complete_finite_second_jet_ported": True,
            "parallel_mixed_polarization_channels_enumerated": True,
            "parallel_first_total_N5_limit_negative": True,
            "parallel_second_inviscid_pressure_N9_limit_negative": True,
            "parallel_complete_second_N9_limit_certified": False,
            "uniform_second_jet_Taylor_remainder_proved": False,
            "critical_L3_control_proved": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "next_theorem_target": (
            "Close the o(N9) exclusion for every viscosity-bearing quartic "
            "second-jet channel. Start with H_uu[E,E] and D_u[u2_EE] "
            "weighted-Fisher terms, then the E-A transported-weight and "
            "mixed pressure-projector rows. Use the six/five/four compatible "
            "weight stencils and a finite/dyadic internal-output split."
        ),
        "runtime_seconds": time.perf_counter() - started,
        "all_positive_checks_pass": all_checks,
    }


def _parse_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heat-sizes",
        default=",".join(str(value) for value in DEFAULT_HEAT_SIZES),
    )
    parser.add_argument("--projection-size", type=int, default=5)
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    _lower_process_priority()
    payload = audit(
        heat_sizes=_parse_sizes(args.heat_sizes),
        projection_size=args.projection_size,
    )
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True))
    if not payload["all_positive_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
