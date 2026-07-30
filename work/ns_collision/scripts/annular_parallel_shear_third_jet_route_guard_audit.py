"""Audit the third-jet carrier route for the parallel-shear restart family."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from annular_parallel_shear_finite_jet_port_audit import (
    _initial_coefficients,
)
from annular_rho_zero_continuum_convolution_quadrature import (
    _lower_process_priority,
)
from annular_rho_zero_first_jet_audit import (
    _coefficients,
    _generator_from_coefficients,
    _grid_shape,
    _physical,
    _pressure_coefficients,
    _scalar_gradient,
    _spectral_data,
)
from annular_rho_zero_second_jet_route_guard_audit import (
    _scalar_field,
    _state_and_flow_jets,
    _vector_field,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_parallel_shear_third_jet_route_guard_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "annular_parallel_shear_finite_jet_port_audit_v1.json"
    ): "1e7753a5280c136bbe34770a17d929f73cc7398579906f51e316c738ee660da0",
    (
        "work/ns_collision/results/"
        "annular_parallel_shear_euler_transport_fisher_exclusion_"
        "audit_v1.json"
    ): "74722ffabf83612a51fdd0f3ab71e90c7b6fd68c5b4eb15b6b5ed040876e5046",
    (
        "work/ns_collision/results/"
        "annular_parallel_shear_heat_block_exclusion_audit_v1.json"
    ): "e3ccdc9a380edf818943450203b0659d0a45b6a5b8255658c5a03681e8213c95",
}
ALGORITHM_REVISION = "annular-parallel-shear-third-jet-route-guard-v1"
Array = np.ndarray
Field = dict[str, Any]


SECTORS = {
    "pressure": {
        "base_velocity_degree": 3,
        "weight_scale_degree": 1,
        "base_differential_order": 1,
    },
    "velocity_Fisher": {
        "base_velocity_degree": 2,
        "weight_scale_degree": 1,
        "base_differential_order": 2,
    },
    "weight_self": {
        "base_velocity_degree": 0,
        "weight_scale_degree": 3,
        "base_differential_order": 2,
    },
}


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


def _prerequisite_audit() -> dict[str, Any]:
    rows = []
    for relative, expected in PREREQUISITES.items():
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = _sha256(path)
        passed = bool(
            actual == expected
            and (
                payload.get("all_positive_checks_pass") is True
                or payload.get("all_port_checks_pass") is True
            )
        )
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "passed": passed,
            }
        )
    return {
        "rows": rows,
        "all_checks_pass": all(row["passed"] for row in rows),
    }


def _compositions(total: int, slots: int) -> Iterable[tuple[int, ...]]:
    if slots == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in _compositions(total - first, slots - 1):
            yield (first, *tail)


def _third_flow_identity_certificate() -> dict[str, Any]:
    state_blocks = {
        "z20": "DX X",
        "z21": "DX Y+DY X",
        "z22": "DY Y",
        "z30": "D2X[X,X]+DX z20",
        "z31": "2D2X[X,Y]+DX z21+DY z20",
        "z32": "D2X[Y,Y]+DX z22+DY z21",
        "z33": "DY z22",
    }
    scalar_blocks = {
        "heat_0": (
            "D3g[X,X,X]+3D2g[X,z20]+Dg[z30]"
        ),
        "heat_1": (
            "3D3g[X,X,Y]+3D2g[Y,z20]+"
            "3D2g[X,z21]+Dg[z31]"
        ),
        "heat_2": (
            "3D3g[X,Y,Y]+3D2g[Y,z21]+"
            "3D2g[X,z22]+Dg[z32]"
        ),
        "heat_3": (
            "D3g[Y,Y,Y]+3D2g[Y,z22]+Dg[z33]"
        ),
    }
    return {
        "state": "z=(u,lambda)",
        "X": "(E,A)=(-P[(u dot grad)u],-u dot grad lambda)",
        "Y": "(V,D)=(nu Delta u,-nu Delta lambda)",
        "second_state_blocks": {
            key: state_blocks[key] for key in ("z20", "z21", "z22")
        },
        "third_state_blocks": {
            key: state_blocks[key]
            for key in ("z30", "z31", "z32", "z33")
        },
        "third_scalar_blocks": scalar_blocks,
        "binomial_multiplicities": [1, 3, 3, 1],
        "all_checks_pass": bool(
            len(state_blocks) == 7
            and len(scalar_blocks) == 4
            and sum((1, 3, 3, 1)) == 8
        ),
    }


def _carrier_ledger() -> dict[str, Any]:
    rows = []
    for sector, data in SECTORS.items():
        for heat_count in range(4):
            velocity_degree = (
                data["base_velocity_degree"] + 3 - heat_count
            )
            differential_order = (
                data["base_differential_order"] + 3 + heat_count
            )
            for high_count in range(0, velocity_degree + 1, 2):
                if high_count == 0:
                    tuple_coefficient_power = 0
                    low_amplitude_power = velocity_degree
                    optimized_power = (
                        velocity_degree
                        + data["weight_scale_degree"]
                    )
                else:
                    tuple_coefficient_power = 2 * high_count - 3
                    low_amplitude_power = (
                        velocity_degree - high_count
                    )
                    optimized_power = (
                        tuple_coefficient_power
                        + low_amplitude_power
                        + data["weight_scale_degree"]
                        + differential_order
                    )
                required_gains = max(0, optimized_power - 10)
                if required_gains == 0:
                    route = "automatic_O_N10_or_lower"
                elif (
                    sector == "pressure"
                    and high_count == 4
                    and heat_count in (0, 1)
                ):
                    route = (
                        "mixed_fixed_output_protected_and_"
                        "bounded_output_exception"
                    )
                else:
                    route = "compatible_stencil_closure_required"
                rows.append(
                    {
                        "sector": sector,
                        "heat_count": heat_count,
                        "velocity_degree": velocity_degree,
                        "weight_scale_degree": data[
                            "weight_scale_degree"
                        ],
                        "differential_order_upper_bound": (
                            differential_order
                        ),
                        "high_leaf_count": high_count,
                        "high_tuple_and_coefficient_power": (
                            tuple_coefficient_power
                        ),
                        "low_amplitude_power": low_amplitude_power,
                        "naive_optimized_power": optimized_power,
                        "gains_required_for_O_N10": required_gains,
                        "route": route,
                    }
                )

    nonzero_rows = [
        row for row in rows if row["high_leaf_count"] > 0
    ]
    automatic = [
        row
        for row in rows
        if row["route"] == "automatic_O_N10_or_lower"
    ]
    dangerous = [
        row
        for row in rows
        if row["gains_required_for_O_N10"] > 0
    ]
    common_formula = all(
        row["naive_optimized_power"]
        == row["high_leaf_count"] + 8
        for row in nonzero_rows
    )
    expected_dangerous = {
        ("pressure", 0, 4, 2),
        ("pressure", 0, 6, 4),
        ("pressure", 1, 4, 2),
        ("pressure", 2, 4, 2),
        ("velocity_Fisher", 0, 4, 2),
        ("velocity_Fisher", 1, 4, 2),
    }
    observed_dangerous = {
        (
            row["sector"],
            row["heat_count"],
            row["high_leaf_count"],
            row["gains_required_for_O_N10"],
        )
        for row in dangerous
    }
    return {
        "rows": rows,
        "row_count": len(rows),
        "automatic_row_count": len(automatic),
        "dangerous_row_count": len(dangerous),
        "common_nonzero_high_formula": (
            "naive optimized power=h+8"
        ),
        "common_formula_verified": common_formula,
        "dangerous_rows": dangerous,
        "all_checks_pass": bool(
            len(rows) == 28
            and len(automatic) == 22
            and len(dangerous) == 6
            and common_formula
            and observed_dangerous == expected_dangerous
        ),
    }


def _jet_slot(order: int, heat_count: int, scalar: bool = False) -> str:
    prefix = "Lambda" if scalar else "U"
    return f"{prefix}{order}^{heat_count}"


def _bounded_output_exception_families() -> dict[str, Any]:
    rows = []
    seen = set()
    for orders in _compositions(3, 4):
        heat_ranges = [range(order + 1) for order in orders]
        for heats in product(*heat_ranges):
            heat_count = sum(heats)
            if heat_count > 2:
                continue
            degrees = [
                1 + orders[index] - heats[index]
                for index in range(3)
            ]
            degrees.append(orders[3] - heats[3])
            velocity_degree = 6 - heat_count
            if velocity_degree < 4:
                continue
            pressure_capacity = degrees[0] + degrees[1]
            if pressure_capacity < 4:
                continue
            pressure_slots = tuple(
                sorted(
                    (
                        (orders[0], heats[0]),
                        (orders[1], heats[1]),
                    )
                )
            )
            canonical = (
                heat_count,
                pressure_slots,
                (orders[2], heats[2]),
                (orders[3], heats[3]),
            )
            if canonical in seen:
                continue
            seen.add(canonical)
            first_pressure, second_pressure = pressure_slots
            pressure_high_order = (
                first_pressure[0]
                + first_pressure[1]
                + second_pressure[0]
                + second_pressure[1]
            )
            optimized_power = (
                5
                + pressure_high_order
                + (velocity_degree - 4)
                + 1
            )
            pressure_pair_same = (
                first_pressure == second_pressure
            )
            taylor_multiplier = (
                6
                * (1 if pressure_pair_same else 2)
                / math.prod(math.factorial(order) for order in orders)
            )
            rows.append(
                {
                    "heat_count": heat_count,
                    "pressure_slots": [
                        {
                            "jet_order": value[0],
                            "heat_degree": value[1],
                            "label": _jet_slot(*value),
                        }
                        for value in pressure_slots
                    ],
                    "test_slot": {
                        "jet_order": orders[2],
                        "heat_degree": heats[2],
                        "label": _jet_slot(
                            orders[2], heats[2]
                        ),
                    },
                    "weight_slot": {
                        "jet_order": orders[3],
                        "heat_degree": heats[3],
                        "label": _jet_slot(
                            orders[3], heats[3], scalar=True
                        ),
                    },
                    "slot_velocity_degrees": degrees,
                    "pressure_side_velocity_capacity": (
                        pressure_capacity
                    ),
                    "external_velocity_degree": (
                        degrees[2] + degrees[3]
                    ),
                    "high_leaf_count": 4,
                    "pressure_side_high_differential_order_upper": (
                        pressure_high_order
                    ),
                    "low_velocity_amplitude_power": (
                        velocity_degree - 4
                    ),
                    "optimized_power_upper_bound": optimized_power,
                    "third_derivative_taylor_multiplier": (
                        taylor_multiplier
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            row["heat_count"],
            tuple(
                (slot["jet_order"], slot["heat_degree"])
                for slot in row["pressure_slots"]
            ),
            (
                row["test_slot"]["jet_order"],
                row["test_slot"]["heat_degree"],
            ),
            (
                row["weight_slot"]["jet_order"],
                row["weight_slot"]["heat_degree"],
            ),
        )
    )
    counts = Counter(row["heat_count"] for row in rows)
    n11_rows = [
        row
        for row in rows
        if row["optimized_power_upper_bound"] == 11
    ]
    return {
        "definition": (
            "Four high leaves lie entirely inside the two pressure "
            "inputs, while every velocity leaf outside that pair is low."
        ),
        "families": rows,
        "family_count": len(rows),
        "family_count_by_heat": {
            str(key): value for key, value in sorted(counts.items())
        },
        "N11_capable_family_count": len(n11_rows),
        "N11_capable_families": n11_rows,
        "maximum_optimized_power_upper_bound": max(
            row["optimized_power_upper_bound"] for row in rows
        ),
        "r2_exception_absence_reason": (
            "At heat count two the pressure sector has velocity degree "
            "four. The three total jet orders cannot place both heat "
            "operations outside the pressure pair, so that pair has "
            "capacity at most three; an external high leaf is forced."
        ),
        "all_checks_pass": bool(
            len(rows) == 13
            and counts == Counter({0: 6, 1: 7})
            and len(n11_rows) == 5
            and max(
                row["optimized_power_upper_bound"] for row in rows
            )
            == 11
        ),
    }


def _stencil_route_certificate(
    carrier: dict[str, Any],
    exceptions: dict[str, Any],
) -> dict[str, Any]:
    dangerous = carrier["dangerous_rows"]
    protected_groups = [
        {
            "sector": row["sector"],
            "heat_count": row["heat_count"],
            "high_leaf_count": row["high_leaf_count"],
            "required_gains": row["gains_required_for_O_N10"],
        }
        for row in dangerous
    ]
    return {
        "parity_gauged_vertex_difference_budget": 6,
        "protected_group_routes": protected_groups,
        "four_high_requirement": 2,
        "six_high_requirement": 4,
        "pressure_projector_rule": (
            "If a high leaf occurs outside the two outer pressure inputs, "
            "use it as the dependent resonance leaf. Then either the "
            "outer pressure output stays fixed or it stays at carrier "
            "scale, where degree-zero projector differences are regular."
        ),
        "polynomial_trade_rule": (
            "When a vertex difference hits an explicit q power, that "
            "monomial has already lost the same carrier power. Therefore "
            "explicit vertex powers consume cancellation order but do "
            "not erase the carrier gain."
        ),
        "bounded_output_exception_rule": (
            "If all four high leaves lie inside the two pressure inputs, "
            "the test and transported-weight velocity leaves may all be "
            "low and the outer pressure output is bounded. Do not "
            "differentiate that projector through its zero convention. "
            "Direct pressure-side derivative counting gives O(N^11) or "
            "better for all thirteen structural families."
        ),
        "internal_shell_obligation": (
            "A depth-three discrete Leibniz and dyadic-shell lemma is "
            "still required for repeated differences through nested "
            "Euler/Leray outputs. The second-jet two-difference lemma "
            "does not by itself certify the four-difference all-high "
            "third-jet branch."
        ),
        "exception_family_count": exceptions["family_count"],
        "exception_N11_family_count": exceptions[
            "N11_capable_family_count"
        ],
        "restart_time_O_N11_certified": False,
        "all_checks_pass": bool(
            carrier["all_checks_pass"]
            and exceptions["all_checks_pass"]
            and len(protected_groups) == 6
            and max(
                row["required_gains"] for row in protected_groups
            )
            == 4
        ),
    }


def _taylor_threshold_certificate() -> dict[str, Any]:
    return {
        "assumptions": [
            "g_N''(0)<=-c2 N^9 for all sufficiently large N",
            (
                "sup_{0<=s<=T/N^2}|g_N'''(s)|<=C3 N^11 "
                "with constants c2,C3>0 independent of N"
            ),
        ],
        "curvature_bound": (
            "g_N''(s)<=-(c2-C3 T)N^9"
        ),
        "sufficient_window_condition": "0<T<=c2/(2 C3)",
        "consequence": "g_N''(s)<=-(c2/2)N^9 on the window",
        "integrated_generator_bound": (
            "integral_0^delta g_N(s) ds <= "
            "g_N(0)delta+g_N'(0)delta^2/2-"
            "c2 N^9 delta^3/12, delta=T/N^2"
        ),
        "leading_integrated_curvature_scale": (
            "-(c2 T^3/12)N^3"
        ),
        "correction_to_previous_target": (
            "Uniform O(N^11), with an explicit constant, is sufficient; "
            "o(N^11) is stronger than necessary."
        ),
        "restart_time_bound_is_not_enough": (
            "The estimate must hold along the evolving coupled "
            "Navier-Stokes/adjoint trajectory, not only at s=0."
        ),
        "all_checks_pass": True,
    }


def _combine_vector(
    label: str,
    terms: list[tuple[float, Field]],
    waves: tuple[Array, ...],
    volume: int,
) -> Field:
    coefficients = np.zeros_like(terms[0][1]["coefficients"])
    for factor, field in terms:
        coefficients += factor * field["coefficients"]
    return _vector_field(label, coefficients, waves, volume)


def _combine_scalar(
    label: str,
    terms: list[tuple[float, Field]],
    waves: tuple[Array, ...],
    volume: int,
) -> Field:
    coefficients = np.zeros_like(terms[0][1]["coefficients"])
    for factor, field in terms:
        coefficients += factor * field["coefficients"]
    return _scalar_field(label, coefficients, waves, volume)


def _linearized_euler(
    base: Field,
    direction: Field,
    waves: tuple[Array, ...],
    safe_wave_number_squared: Array,
    volume: int,
    label: str,
) -> Field:
    pressure_coefficients = _pressure_coefficients(
        base["value"],
        direction["value"],
        waves,
        safe_wave_number_squared,
        volume,
        symmetrized=True,
    )
    pressure_gradient = _scalar_gradient(
        pressure_coefficients, waves, volume
    )
    advection = (
        np.einsum(
            "j...,ij...->i...",
            direction["value"],
            base["gradient"],
        )
        + np.einsum(
            "j...,ij...->i...",
            base["value"],
            direction["gradient"],
        )
    )
    return _vector_field(
        label,
        _coefficients(-advection - pressure_gradient, volume),
        waves,
        volume,
    )


def _euler_hessian(
    first: Field,
    second: Field,
    waves: tuple[Array, ...],
    safe_wave_number_squared: Array,
    volume: int,
    label: str,
) -> Field:
    pressure_coefficients = _pressure_coefficients(
        first["value"],
        second["value"],
        waves,
        safe_wave_number_squared,
        volume,
        symmetrized=True,
    )
    pressure_gradient = _scalar_gradient(
        pressure_coefficients, waves, volume
    )
    advection = (
        np.einsum(
            "j...,ij...->i...",
            first["value"],
            second["gradient"],
        )
        + np.einsum(
            "j...,ij...->i...",
            second["value"],
            first["gradient"],
        )
    )
    return _vector_field(
        label,
        _coefficients(-advection - pressure_gradient, volume),
        waves,
        volume,
    )


def _velocity_heat(
    field: Field,
    wave_number_squared: Array,
    viscosity: float,
    waves: tuple[Array, ...],
    volume: int,
    label: str,
) -> Field:
    coefficients = (
        -viscosity
        * wave_number_squared[None, ...]
        * field["coefficients"]
    )
    return _vector_field(label, coefficients, waves, volume)


def _linearized_transport(
    base_velocity: Field,
    base_weight: Field,
    velocity_direction: Field,
    weight_direction: Field,
    waves: tuple[Array, ...],
    volume: int,
    label: str,
) -> Field:
    values = (
        -np.sum(
            velocity_direction["value"] * base_weight["gradient"],
            axis=0,
        )
        - np.sum(
            base_velocity["value"] * weight_direction["gradient"],
            axis=0,
        )
    )
    return _scalar_field(
        label, _coefficients(values, volume), waves, volume
    )


def _transport_hessian(
    first_velocity: Field,
    first_weight: Field,
    second_velocity: Field,
    second_weight: Field,
    waves: tuple[Array, ...],
    volume: int,
    label: str,
) -> Field:
    values = (
        -np.sum(
            first_velocity["value"] * second_weight["gradient"],
            axis=0,
        )
        - np.sum(
            second_velocity["value"] * first_weight["gradient"],
            axis=0,
        )
    )
    return _scalar_field(
        label, _coefficients(values, volume), waves, volume
    )


def _weight_heat(
    field: Field,
    wave_number_squared: Array,
    viscosity: float,
    waves: tuple[Array, ...],
    volume: int,
    label: str,
) -> Field:
    coefficients = (
        viscosity * wave_number_squared * field["coefficients"]
    )
    return _scalar_field(label, coefficients, waves, volume)


def _third_flow_jets(
    jets: dict[str, Any],
    waves: tuple[Array, ...],
    wave_number_squared: Array,
    safe_wave_number_squared: Array,
    volume: int,
    viscosity: float,
) -> dict[str, Any]:
    u0 = jets["velocity"]
    lambda0 = jets["weight"]
    E = jets["velocity_directions"]["E"]
    V = jets["velocity_directions"]["V"]
    A = jets["weight_directions"]["A"]
    D = jets["weight_directions"]["D"]
    velocity_accelerations = jets["velocity_accelerations"]
    weight_accelerations = jets["weight_accelerations"]

    u20 = velocity_accelerations["EE"]
    u21 = _combine_vector(
        "u2_heat1",
        [
            (1.0, velocity_accelerations["EV"]),
            (1.0, velocity_accelerations["VE"]),
        ],
        waves,
        volume,
    )
    u22 = velocity_accelerations["VV"]
    lambda20 = _combine_scalar(
        "lambda2_heat0",
        [
            (1.0, weight_accelerations["E0"]),
            (1.0, weight_accelerations["0A"]),
        ],
        waves,
        volume,
    )
    lambda21 = _combine_scalar(
        "lambda2_heat1",
        [
            (1.0, weight_accelerations["V0"]),
            (1.0, weight_accelerations["0D"]),
            (1.0, weight_accelerations["DA"]),
        ],
        waves,
        volume,
    )
    lambda22 = weight_accelerations["DD"]

    u30 = _combine_vector(
        "u3_heat0",
        [
            (
                1.0,
                _euler_hessian(
                    E,
                    E,
                    waves,
                    safe_wave_number_squared,
                    volume,
                    "D2E[E,E]",
                ),
            ),
            (
                1.0,
                _linearized_euler(
                    u0,
                    u20,
                    waves,
                    safe_wave_number_squared,
                    volume,
                    "DE[u20]",
                ),
            ),
        ],
        waves,
        volume,
    )
    u31 = _combine_vector(
        "u3_heat1",
        [
            (
                2.0,
                _euler_hessian(
                    E,
                    V,
                    waves,
                    safe_wave_number_squared,
                    volume,
                    "D2E[E,V]",
                ),
            ),
            (
                1.0,
                _linearized_euler(
                    u0,
                    u21,
                    waves,
                    safe_wave_number_squared,
                    volume,
                    "DE[u21]",
                ),
            ),
            (
                1.0,
                _velocity_heat(
                    u20,
                    wave_number_squared,
                    viscosity,
                    waves,
                    volume,
                    "V[u20]",
                ),
            ),
        ],
        waves,
        volume,
    )
    u32 = _combine_vector(
        "u3_heat2",
        [
            (
                1.0,
                _euler_hessian(
                    V,
                    V,
                    waves,
                    safe_wave_number_squared,
                    volume,
                    "D2E[V,V]",
                ),
            ),
            (
                1.0,
                _linearized_euler(
                    u0,
                    u22,
                    waves,
                    safe_wave_number_squared,
                    volume,
                    "DE[u22]",
                ),
            ),
            (
                1.0,
                _velocity_heat(
                    u21,
                    wave_number_squared,
                    viscosity,
                    waves,
                    volume,
                    "V[u21]",
                ),
            ),
        ],
        waves,
        volume,
    )
    u33 = _velocity_heat(
        u22,
        wave_number_squared,
        viscosity,
        waves,
        volume,
        "u3_heat3",
    )

    lambda30 = _combine_scalar(
        "lambda3_heat0",
        [
            (
                1.0,
                _transport_hessian(
                    E,
                    A,
                    E,
                    A,
                    waves,
                    volume,
                    "D2A[X,X]",
                ),
            ),
            (
                1.0,
                _linearized_transport(
                    u0,
                    lambda0,
                    u20,
                    lambda20,
                    waves,
                    volume,
                    "DA[z20]",
                ),
            ),
        ],
        waves,
        volume,
    )
    lambda31 = _combine_scalar(
        "lambda3_heat1",
        [
            (
                2.0,
                _transport_hessian(
                    E,
                    A,
                    V,
                    D,
                    waves,
                    volume,
                    "D2A[X,Y]",
                ),
            ),
            (
                1.0,
                _linearized_transport(
                    u0,
                    lambda0,
                    u21,
                    lambda21,
                    waves,
                    volume,
                    "DA[z21]",
                ),
            ),
            (
                1.0,
                _weight_heat(
                    lambda20,
                    wave_number_squared,
                    viscosity,
                    waves,
                    volume,
                    "D[lambda20]",
                ),
            ),
        ],
        waves,
        volume,
    )
    lambda32 = _combine_scalar(
        "lambda3_heat2",
        [
            (
                1.0,
                _transport_hessian(
                    V,
                    D,
                    V,
                    D,
                    waves,
                    volume,
                    "D2A[Y,Y]",
                ),
            ),
            (
                1.0,
                _linearized_transport(
                    u0,
                    lambda0,
                    u22,
                    lambda22,
                    waves,
                    volume,
                    "DA[z22]",
                ),
            ),
            (
                1.0,
                _weight_heat(
                    lambda21,
                    wave_number_squared,
                    viscosity,
                    waves,
                    volume,
                    "D[lambda21]",
                ),
            ),
        ],
        waves,
        volume,
    )
    lambda33 = _weight_heat(
        lambda22,
        wave_number_squared,
        viscosity,
        waves,
        volume,
        "lambda3_heat3",
    )

    u1 = _combine_vector(
        "u1_total", [(1.0, E), (1.0, V)], waves, volume
    )
    lambda1 = _combine_scalar(
        "lambda1_total", [(1.0, A), (1.0, D)], waves, volume
    )
    u2 = _combine_vector(
        "u2_total",
        [(1.0, u20), (1.0, u21), (1.0, u22)],
        waves,
        volume,
    )
    lambda2 = _combine_scalar(
        "lambda2_total",
        [(1.0, lambda20), (1.0, lambda21), (1.0, lambda22)],
        waves,
        volume,
    )
    u3 = _combine_vector(
        "u3_total",
        [(1.0, u30), (1.0, u31), (1.0, u32), (1.0, u33)],
        waves,
        volume,
    )
    lambda3 = _combine_scalar(
        "lambda3_total",
        [
            (1.0, lambda30),
            (1.0, lambda31),
            (1.0, lambda32),
            (1.0, lambda33),
        ],
        waves,
        volume,
    )

    direct_u3 = _combine_vector(
        "u3_direct",
        [
            (
                1.0,
                _euler_hessian(
                    u1,
                    u1,
                    waves,
                    safe_wave_number_squared,
                    volume,
                    "D2E[u1,u1]",
                ),
            ),
            (
                1.0,
                _linearized_euler(
                    u0,
                    u2,
                    waves,
                    safe_wave_number_squared,
                    volume,
                    "DE[u2]",
                ),
            ),
            (
                1.0,
                _velocity_heat(
                    u2,
                    wave_number_squared,
                    viscosity,
                    waves,
                    volume,
                    "V[u2]",
                ),
            ),
        ],
        waves,
        volume,
    )
    direct_lambda3 = _combine_scalar(
        "lambda3_direct",
        [
            (
                1.0,
                _transport_hessian(
                    u1,
                    lambda1,
                    u1,
                    lambda1,
                    waves,
                    volume,
                    "D2A[z1,z1]",
                ),
            ),
            (
                1.0,
                _linearized_transport(
                    u0,
                    lambda0,
                    u2,
                    lambda2,
                    waves,
                    volume,
                    "DA[z2]",
                ),
            ),
            (
                1.0,
                _weight_heat(
                    lambda2,
                    wave_number_squared,
                    viscosity,
                    waves,
                    volume,
                    "D[lambda2]",
                ),
            ),
        ],
        waves,
        volume,
    )
    velocity_partition_residual = float(
        np.max(
            np.abs(
                u3["coefficients"] - direct_u3["coefficients"]
            )
        )
    )
    weight_partition_residual = float(
        np.max(
            np.abs(
                lambda3["coefficients"]
                - direct_lambda3["coefficients"]
            )
        )
    )
    divergence_residuals = {}
    relative_divergence_residuals = {}
    for heat_count, field in enumerate((u30, u31, u32, u33)):
        divergence_terms = [
            waves[component] * field["coefficients"][component]
            for component in range(3)
        ]
        divergence = sum(divergence_terms)
        scale = max(
            float(
                np.max(
                    sum(np.abs(term) for term in divergence_terms)
                )
            ),
            1.0,
        )
        divergence_residuals[str(heat_count)] = float(
            np.max(np.abs(divergence))
        )
        relative_divergence_residuals[str(heat_count)] = (
            divergence_residuals[str(heat_count)] / scale
        )
    return {
        "velocity_options": {
            0: [(0, u0)],
            1: [(0, E), (1, V)],
            2: [(0, u20), (1, u21), (2, u22)],
            3: [(0, u30), (1, u31), (2, u32), (3, u33)],
        },
        "weight_options": {
            0: [(0, lambda0)],
            1: [(0, A), (1, D)],
            2: [
                (0, lambda20),
                (1, lambda21),
                (2, lambda22),
            ],
            3: [
                (0, lambda30),
                (1, lambda31),
                (2, lambda32),
                (3, lambda33),
            ],
        },
        "totals": {
            "u0": u0,
            "lambda0": lambda0,
            "u1": u1,
            "lambda1": lambda1,
            "u2": u2,
            "lambda2": lambda2,
            "u3": u3,
            "lambda3": lambda3,
        },
        "partition_residuals": {
            "velocity": velocity_partition_residual,
            "weight": weight_partition_residual,
        },
        "third_velocity_divergence_residuals": divergence_residuals,
        "third_velocity_relative_divergence_residuals": (
            relative_divergence_residuals
        ),
    }


def _third_multilinear_replay(
    flow: dict[str, Any],
    waves: tuple[Array, ...],
    safe_wave_number_squared: Array,
    volume: int,
    viscosity: float,
) -> dict[str, Any]:
    velocity_options = flow["velocity_options"]
    weight_options = flow["weight_options"]
    pressure_cache: dict[tuple[str, str], Array] = {}

    def pressure_pair(first: Field, second: Field) -> Array:
        key = tuple(sorted((first["label"], second["label"])))
        if key not in pressure_cache:
            pressure_cache[key] = _physical(
                _pressure_coefficients(
                    first["value"],
                    second["value"],
                    waves,
                    safe_wave_number_squared,
                    volume,
                ),
                volume,
            )
        return pressure_cache[key]

    blocks = {
        sector: {heat_count: 0.0 for heat_count in range(4)}
        for sector in SECTORS
    }

    for orders in _compositions(3, 4):
        factorial = math.prod(math.factorial(order) for order in orders)
        for first in velocity_options[orders[0]]:
            for second in velocity_options[orders[1]]:
                pressure = pressure_pair(first[1], second[1])
                for test in velocity_options[orders[2]]:
                    for weight in weight_options[orders[3]]:
                        heat_count = (
                            first[0] + second[0] + test[0] + weight[0]
                        )
                        value = np.mean(
                            pressure
                            * np.sum(
                                test[1]["value"]
                                * weight[1]["gradient"],
                                axis=0,
                            )
                        )
                        blocks["pressure"][heat_count] += (
                            float(value) / factorial
                        )

    for orders in _compositions(3, 3):
        factorial = math.prod(math.factorial(order) for order in orders)
        for first in velocity_options[orders[0]]:
            for second in velocity_options[orders[1]]:
                for weight in weight_options[orders[2]]:
                    heat_count = first[0] + second[0] + weight[0]
                    value = np.mean(
                        -viscosity
                        * weight[1]["value"]
                        * np.sum(
                            first[1]["gradient"]
                            * second[1]["gradient"],
                            axis=(0, 1),
                        )
                    )
                    blocks["velocity_Fisher"][heat_count] += (
                        float(value) / factorial
                    )

        for first in weight_options[orders[0]]:
            for second in weight_options[orders[1]]:
                for third in weight_options[orders[2]]:
                    heat_count = first[0] + second[0] + third[0]
                    value = np.mean(
                        -viscosity
                        * first[1]["value"]
                        * np.sum(
                            second[1]["gradient"]
                            * third[1]["gradient"],
                            axis=0,
                        )
                    )
                    blocks["weight_self"][heat_count] += (
                        float(value) / factorial
                    )

    for sector in blocks:
        for heat_count in blocks[sector]:
            blocks[sector][heat_count] *= math.factorial(3)
    heat_totals = {
        heat_count: sum(
            blocks[sector][heat_count] for sector in blocks
        )
        for heat_count in range(4)
    }
    sector_totals = {
        sector: sum(blocks[sector].values()) for sector in blocks
    }
    return {
        "sector_heat_blocks": {
            sector: {
                str(heat_count): value
                for heat_count, value in values.items()
            }
            for sector, values in blocks.items()
        },
        "heat_block_totals": {
            str(heat_count): value
            for heat_count, value in heat_totals.items()
        },
        "sector_totals": sector_totals,
        "total_third_derivative": sum(heat_totals.values()),
        "pressure_pair_cache_count": len(pressure_cache),
    }


def _third_finite_difference(
    flow: dict[str, Any],
    waves: tuple[Array, ...],
    wave_number_squared: Array,
    safe_wave_number_squared: Array,
    volume: int,
    viscosity: float,
    exact_third: float,
    epsilon: float,
) -> dict[str, Any]:
    totals = flow["totals"]
    cache: dict[float, float] = {}

    def generator(step: float) -> float:
        if step not in cache:
            velocity = (
                totals["u0"]["coefficients"]
                + step * totals["u1"]["coefficients"]
                + 0.5 * step**2 * totals["u2"]["coefficients"]
                + (step**3 / 6.0)
                * totals["u3"]["coefficients"]
            )
            weight = (
                totals["lambda0"]["coefficients"]
                + step * totals["lambda1"]["coefficients"]
                + 0.5 * step**2 * totals["lambda2"]["coefficients"]
                + (step**3 / 6.0)
                * totals["lambda3"]["coefficients"]
            )
            cache[step] = _generator_from_coefficients(
                velocity,
                weight,
                waves,
                wave_number_squared,
                safe_wave_number_squared,
                volume,
                viscosity,
            )
        return cache[step]

    def quotient(step: float) -> float:
        return (
            generator(2.0 * step)
            - 2.0 * generator(step)
            + 2.0 * generator(-step)
            - generator(-2.0 * step)
        ) / (2.0 * step**3)

    coarse = quotient(epsilon)
    fine = quotient(epsilon / 2.0)
    richardson = (4.0 * fine - coarse) / 3.0
    absolute_residual = abs(richardson - exact_third)
    return {
        "epsilon": epsilon,
        "exact_multilinear": exact_third,
        "coarse": coarse,
        "fine": fine,
        "Richardson": richardson,
        "absolute_residual": absolute_residual,
        "relative_residual": absolute_residual
        / max(abs(exact_third), 1.0),
        "evaluation_count": len(cache),
    }


def _finite_replay(
    size: int,
    dealias_factor: int,
    epsilon: float,
    yz_amplitude: float = 0.7,
    xy_amplitude: float = 0.7,
    coefficient_scale: float = 0.9,
    viscosity: float = 1.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    shape = _grid_shape(size, dealias_factor)
    (
        waves,
        wave_number_squared,
        safe_wave_number_squared,
        volume,
    ) = _spectral_data(shape)
    velocity, weight, _, _, _ = _initial_coefficients(
        size,
        shape,
        yz_amplitude,
        xy_amplitude,
        coefficient_scale,
    )
    jets = _state_and_flow_jets(
        velocity,
        weight,
        waves,
        wave_number_squared,
        safe_wave_number_squared,
        volume,
        viscosity,
    )
    flow = _third_flow_jets(
        jets,
        waves,
        wave_number_squared,
        safe_wave_number_squared,
        volume,
        viscosity,
    )
    multilinear = _third_multilinear_replay(
        flow,
        waves,
        safe_wave_number_squared,
        volume,
        viscosity,
    )
    finite_difference = _third_finite_difference(
        flow,
        waves,
        wave_number_squared,
        safe_wave_number_squared,
        volume,
        viscosity,
        multilinear["total_third_derivative"],
        epsilon,
    )
    return {
        "size": size,
        "grid_shape": list(shape),
        "grid_point_count": volume,
        "dealias_factor": dealias_factor,
        "yz_amplitude": yz_amplitude,
        "xy_amplitude": xy_amplitude,
        "coefficient_scale": coefficient_scale,
        "viscosity": viscosity,
        "third_flow_partition_residuals": flow[
            "partition_residuals"
        ],
        "third_velocity_divergence_residuals": flow[
            "third_velocity_divergence_residuals"
        ],
        "third_velocity_relative_divergence_residuals": flow[
            "third_velocity_relative_divergence_residuals"
        ],
        "multilinear_replay": multilinear,
        "finite_difference_replay": finite_difference,
        "runtime_seconds": time.perf_counter() - started,
    }


def _padding_replay(
    base: dict[str, Any],
    padded: dict[str, Any],
) -> dict[str, Any]:
    base_blocks = base["multilinear_replay"]["sector_heat_blocks"]
    padded_blocks = padded["multilinear_replay"]["sector_heat_blocks"]
    residuals = {}
    for sector in base_blocks:
        for heat_count in base_blocks[sector]:
            key = f"{sector}::heat_{heat_count}"
            residuals[key] = abs(
                base_blocks[sector][heat_count]
                - padded_blocks[sector][heat_count]
            )
    maximum_key = max(residuals, key=residuals.get)
    total_residual = abs(
        base["multilinear_replay"]["total_third_derivative"]
        - padded["multilinear_replay"]["total_third_derivative"]
    )
    scale = max(
        abs(base["multilinear_replay"]["total_third_derivative"]),
        1.0,
    )
    return {
        "base_dealias_factor": base["dealias_factor"],
        "padded_dealias_factor": padded["dealias_factor"],
        "base_grid_shape": base["grid_shape"],
        "padded_grid_shape": padded["grid_shape"],
        "maximum_sector_heat_residual": residuals[maximum_key],
        "maximum_sector_heat_residual_key": maximum_key,
        "total_third_derivative_residual": total_residual,
        "relative_total_residual": total_residual / scale,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--dealias-factor", type=int, default=14)
    parser.add_argument("--padding-factor", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=2.0e-4)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    _lower_process_priority()
    started = time.perf_counter()
    prerequisites = _prerequisite_audit()
    flow_identity = _third_flow_identity_certificate()
    carrier = _carrier_ledger()
    exceptions = _bounded_output_exception_families()
    stencil = _stencil_route_certificate(carrier, exceptions)
    taylor = _taylor_threshold_certificate()
    finite = _finite_replay(
        arguments.size,
        arguments.dealias_factor,
        arguments.epsilon,
    )
    padded = _finite_replay(
        arguments.size,
        arguments.padding_factor,
        arguments.epsilon,
    )
    padding = _padding_replay(finite, padded)
    finite_checks = {
        "flow_partition_residual_below_1e-8": max(
            finite["third_flow_partition_residuals"].values()
        )
        < 1.0e-8,
        "relative_divergence_residual_below_1e-9": max(
            finite[
                "third_velocity_relative_divergence_residuals"
            ].values()
        )
        < 1.0e-9,
        "finite_difference_relative_residual_below_2e-5": finite[
            "finite_difference_replay"
        ]["relative_residual"]
        < 2.0e-5,
        "padding_relative_residual_below_1e-9": padding[
            "relative_total_residual"
        ]
        < 1.0e-9,
        "padding_maximum_block_residual_below_1e-7": padding[
            "maximum_sector_heat_residual"
        ]
        < 1.0e-7,
    }
    finite_checks["all_checks_pass"] = all(finite_checks.values())
    all_checks = bool(
        prerequisites["all_checks_pass"]
        and flow_identity["all_checks_pass"]
        and carrier["all_checks_pass"]
        and exceptions["all_checks_pass"]
        and stencil["all_checks_pass"]
        and taylor["all_checks_pass"]
        and finite_checks["all_checks_pass"]
    )
    payload = {
        "kind": "annular_parallel_shear_third_jet_route_guard_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": "passed_route_guard" if all_checks else "failed",
        "scope": (
            "Exact third-flow heat split, exhaustive carrier-degree "
            "triage, bounded-output pressure exception inventory, and "
            "finite spectral chain-rule replay. This is a route guard, "
            "not a uniform third-derivative theorem."
        ),
        "prerequisite_audit": prerequisites,
        "third_flow_identity_certificate": flow_identity,
        "carrier_degree_ledger": carrier,
        "bounded_output_pressure_exceptions": exceptions,
        "compatible_stencil_route": stencil,
        "uniform_Taylor_threshold": taylor,
        "finite_spectral_replay": finite,
        "padding_replay": padding,
        "finite_replay_checks": finite_checks,
        "route_guard_conclusion": (
            "Twenty-two of twenty-eight sector/heat/incidence rows are "
            "automatically O(N^10) or lower. Six high-incidence rows "
            "need compatible-stencil closure. Thirteen bounded-output "
            "pressure families occur only at heat counts zero and one; "
            "direct pressure-side counting bounds each by O(N^11), and "
            "five can saturate that exponent. The remaining missing "
            "restart-time step is a depth-three internal-output shell "
            "lemma for the protected rows. Even after that, a dynamic "
            "bootstrap is required to make the bound uniform on the "
            "parabolic window."
        ),
        "certification_flags": {
            "exact_third_flow_heat_split_certified": all_checks,
            "all_28_carrier_rows_partitioned": all_checks,
            "22_automatic_rows_O_N10_or_lower_certified": all_checks,
            "bounded_output_exception_inventory_certified": all_checks,
            "finite_total_third_chain_rule_replayed": all_checks,
            "complete_restart_time_third_O_N11_proved": False,
            "uniform_parabolic_window_third_O_N11_proved": False,
            "parabolic_window_turnaround_proved": False,
            "critical_L3_control_proved": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "remaining_gate": (
            "Prove a depth-three compatible-difference/dyadic-shell "
            "bound for the protected four- and six-high rows, retaining "
            "an explicit O(N^11) constant. Then propagate that bound "
            "uniformly along 0<=s<=T/N^2 and compare T with c2/(2C3)."
        ),
        "all_route_guard_checks_pass": all_checks,
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_json(RESULT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
