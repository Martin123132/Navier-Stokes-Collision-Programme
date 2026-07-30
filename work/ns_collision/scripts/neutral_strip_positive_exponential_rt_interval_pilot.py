#!/usr/bin/env python3
"""Pilot directed RT0/P0 moments for the weighted hypercircle pencil."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import mpmath
import numpy as np
import triangle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HYPERCIRCLE_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_weighted_hypercircle_pilot_v1.json"
)
DEFAULT_DEPENDENCY_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_continuum_ritz_dependency_audit_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_positive_exponential_rt_interval_pilot512_v1.json"
)

SPACING = 0.06
X_HALF_WIDTH = 4.2
STRIP_HALF_WIDTH = 2.1
SAMPLE_COUNT = 512
TAYLOR_DEGREE = 22
QUADRATURE_ORDER = 12
CROSS_CHECK_ORDER = 18
PRODUCTION_BETA = 0.045
PRODUCTION_BETA_DECIMAL = Decimal("0.045")

Interval = tuple[float, float]
Ball = tuple[float, float]
BallArray = tuple[np.ndarray, np.ndarray]


def _load_module(filename: str, module_name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _set_below_normal_priority() -> bool:
    try:
        import psutil

        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return process.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
    except Exception:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _positive_exponential_coefficients(
    base,
    center: float,
    degree: int,
) -> BallArray:
    center_interval = (center, center)
    exponent = base._iv_mul(center_interval, center_interval)
    exponent = base._iv_mul((0.5, 0.5), exponent)
    coefficients: list[Ball] = [base._exp_ball(exponent)]
    if degree:
        coefficients.append(
            base._ball_mul((center, 0.0), coefficients[0])
        )
    for index in range(1, degree):
        numerator = base._ball_add(
            base._ball_mul((center, 0.0), coefficients[index]),
            coefficients[index - 1],
        )
        coefficients.append(
            base._ball_divide_integer(numerator, index + 1)
        )
    return (
        np.asarray([value[0] for value in coefficients]),
        np.asarray([value[1] for value in coefficients]),
    )


def _positive_remainder_coefficient(
    base,
    derivative_order: int,
    maximum_abs_x: float,
    maximum_abs_z: float,
) -> float:
    mpmath.mp.dps = 100
    order = derivative_order
    x_value = mpmath.mpf(float(maximum_abs_x))
    coefficient = mpmath.mpf("0")
    for index in range(order // 2 + 1):
        coefficient += x_value ** (order - 2 * index) / (
            2**index
            * mpmath.factorial(index)
            * mpmath.factorial(order - 2 * index)
        )
    exponential = mpmath.exp(x_value**2 / 2)
    value = (
        exponential
        * coefficient
        * mpmath.mpf(float(maximum_abs_z)) ** order
    )
    return base._up(float(value))


def _positive_weighted_moments(
    points: np.ndarray,
    degree: int,
    base,
) -> dict[tuple[int, int, int], Interval]:
    geometry = base._triangle_geometry(points)
    determinant = geometry["determinant"]
    x_coordinates = points[:, 0]
    center = float(np.mean(x_coordinates))
    center_interval = (center, center)
    z_intervals = [
        base._iv_sub((float(value), float(value)), center_interval)
        for value in x_coordinates
    ]
    z_values = [base._interval_to_ball(value) for value in z_intervals]
    maximum_abs_x = max(abs(float(value)) for value in x_coordinates)
    maximum_abs_z = max(
        max(abs(value[0]), abs(value[1])) for value in z_intervals
    )
    remainder = _positive_remainder_coefficient(
        base,
        degree + 1,
        maximum_abs_x,
        maximum_abs_z,
    )
    coefficients = _positive_exponential_coefficients(
        base,
        center,
        degree,
    )
    factor_cache: dict[tuple[int, int, int], BallArray] = {}
    exponent_rows = {(0, 0, 0)}
    for first in range(3):
        for second in range(first, 3):
            exponents = [0, 0, 0]
            exponents[first] += 1
            exponents[second] += 1
            exponent_rows.add(tuple(exponents))
    return {
        exponents: base._weighted_moment_precomputed(
            determinant,
            z_values,
            coefficients,
            exponents,
            degree,
            remainder,
            factor_cache,
        )
        for exponents in sorted(exponent_rows)
    }


def _barycentric_moment(
    moments: dict[tuple[int, int, int], Interval],
    first: int,
    second: int,
) -> Interval:
    exponents = [0, 0, 0]
    exponents[first] += 1
    exponents[second] += 1
    return moments[tuple(exponents)]


def _local_interval_forms(
    points: np.ndarray,
    degree: int,
    base,
) -> dict[str, Any]:
    geometry = base._triangle_geometry(points)
    determinant = geometry["determinant"]
    determinant_squared = base._iv_mul(determinant, determinant)
    moments = _positive_weighted_moments(points, degree, base)
    source_mass = moments[(0, 0, 0)]
    rt_mass: list[list[Interval]] = [
        [(0.0, 0.0) for _ in range(3)] for _ in range(3)
    ]
    point_intervals = [
        [(float(value), float(value)) for value in point]
        for point in points
    ]
    for row in range(3):
        for column in range(row, 3):
            numerator = (0.0, 0.0)
            for first in range(3):
                first_difference = [
                    base._iv_sub(
                        point_intervals[first][axis],
                        point_intervals[row][axis],
                    )
                    for axis in range(2)
                ]
                for second in range(3):
                    second_difference = [
                        base._iv_sub(
                            point_intervals[second][axis],
                            point_intervals[column][axis],
                        )
                        for axis in range(2)
                    ]
                    dot = base._iv_add(
                        base._iv_mul(
                            first_difference[0],
                            second_difference[0],
                        ),
                        base._iv_mul(
                            first_difference[1],
                            second_difference[1],
                        ),
                    )
                    contribution = base._iv_mul(
                        dot,
                        _barycentric_moment(moments, first, second),
                    )
                    numerator = base._iv_add(numerator, contribution)
            value = base._iv_div(numerator, determinant_squared)
            rt_mass[row][column] = value
            rt_mass[column][row] = value
    area = base._iv_mul((0.5, 0.5), determinant)
    load = base._iv_div(area, (3.0, 3.0))
    return {
        "determinant": determinant,
        "area": area,
        "load": load,
        "source_mass": source_mass,
        "rt_mass": rt_mass,
    }


def _sqrt_interval(value: Interval, base) -> Interval:
    if value[0] < 0.0:
        raise ValueError("sqrt interval is negative")
    mpmath.iv.dps = 80
    enclosed = mpmath.iv.sqrt(
        mpmath.iv.mpf(
            [
                np.nextafter(value[0], -math.inf),
                np.nextafter(value[1], math.inf),
            ]
        )
    )
    return base._down(float(enclosed.a)), base._up(float(enclosed.b))


def _pi_interval(base) -> Interval:
    mpmath.iv.dps = 80
    return (
        base._down(float(mpmath.iv.pi.a)),
        base._up(float(mpmath.iv.pi.b)),
    )


def _exact_decimal_interval(value: Decimal, base) -> Interval:
    candidate = float(value)
    lower = candidate
    upper = candidate
    if Decimal.from_float(lower) > value:
        lower = float(np.nextafter(lower, -math.inf))
    if Decimal.from_float(upper) < value:
        upper = float(np.nextafter(upper, math.inf))
    return base._down(lower), base._up(upper)


def _triangle_geometry_constants(
    points: np.ndarray,
    base,
    pi_interval: Interval,
) -> tuple[Interval, Interval]:
    point_intervals = [
        [(float(value), float(value)) for value in point]
        for point in points
    ]
    edge_lengths = []
    for first, second in ((0, 1), (1, 2), (2, 0)):
        differences = [
            base._iv_sub(
                point_intervals[first][axis],
                point_intervals[second][axis],
            )
            for axis in range(2)
        ]
        squared = base._iv_add(
            base._iv_mul(differences[0], differences[0]),
            base._iv_mul(differences[1], differences[1]),
        )
        edge_lengths.append(_sqrt_interval(squared, base))
    diameter = (
        max(value[0] for value in edge_lengths),
        max(value[1] for value in edge_lengths),
    )

    x_values = points[:, 0]
    maximum_abs_x = max(abs(float(value)) for value in x_values)
    if float(np.min(x_values)) <= 0.0 <= float(np.max(x_values)):
        minimum_abs_x = 0.0
    else:
        minimum_abs_x = min(abs(float(value)) for value in x_values)
    maximum_square = base._iv_mul(
        (maximum_abs_x, maximum_abs_x),
        (maximum_abs_x, maximum_abs_x),
    )
    minimum_square = base._iv_mul(
        (minimum_abs_x, minimum_abs_x),
        (minimum_abs_x, minimum_abs_x),
    )
    quarter_oscillation = base._iv_mul(
        (0.25, 0.25),
        base._iv_sub(maximum_square, minimum_square),
    )
    source_factor = base._ball_interval(
        base._exp_ball(quarter_oscillation)
    )
    data_constant = base._iv_mul(
        base._iv_div(diameter, pi_interval),
        source_factor,
    )
    return source_factor, data_constant


def _distance_to_interval(value: float, interval: Interval) -> float:
    if value < interval[0]:
        return interval[0] - value
    if value > interval[1]:
        return value - interval[1]
    return 0.0


def _positive_quadrature_local_forms(
    points: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    edge_matrix = np.column_stack(
        (points[1] - points[0], points[2] - points[0])
    )
    determinant = abs(float(np.linalg.det(edge_matrix)))
    source_mass = 0.0
    rt_mass = np.zeros((3, 3))
    for first_index, first in enumerate(nodes):
        for second_index, second in enumerate(nodes):
            barycentric = np.asarray(
                [
                    1.0 - first,
                    first * (1.0 - second),
                    first * second,
                ]
            )
            jacobian_weight = (
                weights[first_index]
                * weights[second_index]
                * determinant
                * first
            )
            point = barycentric @ points
            inverse_mu = math.exp(0.5 * float(point[0]) ** 2)
            local_rt = (point[None, :] - points) / determinant
            source_mass += jacobian_weight * inverse_mu
            rt_mass += (
                jacobian_weight
                * inverse_mu
                * (local_rt @ local_rt.T)
            )
    return source_mass, rt_mass


def _mesh() -> tuple[np.ndarray, np.ndarray, str, Any]:
    boundary = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "positive_exponential_rt_boundary",
    )
    base = _load_module(
        "neutral_strip_gaussian_weighted_assembly_interval_audit.py",
        "positive_exponential_rt_base",
    )
    mesh_input, _, _ = boundary._mesh_input(
        SPACING,
        X_HALF_WIDTH,
        STRIP_HALF_WIDTH,
    )
    mesh = triangle.triangulate(
        mesh_input,
        f"pYq28a{0.45 * SPACING**2:.17g}Q",
    )
    vertices = np.asarray(mesh["vertices"], dtype=float)
    triangles = np.asarray(mesh["triangles"], dtype=int)
    return (
        vertices,
        triangles,
        base._mesh_fingerprint(vertices, triangles),
        base,
    )


def run_pilot(
    hypercircle_result_path: Path,
    dependency_result_path: Path,
    sample_count: int = SAMPLE_COUNT,
    degree: int = TAYLOR_DEGREE,
    quadrature_order: int = QUADRATURE_ORDER,
    cross_check_order: int = CROSS_CHECK_ORDER,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    hypercircle = json.loads(
        hypercircle_result_path.read_text(encoding="ascii")
    )
    dependency = json.loads(
        dependency_result_path.read_text(encoding="ascii")
    )
    vertices, triangles, mesh_fingerprint, base = _mesh()
    expected_fingerprint = hypercircle["mesh"][
        "mesh_fingerprint_sha256"
    ]
    if mesh_fingerprint != expected_fingerprint:
        raise RuntimeError("stored mesh fingerprint mismatch")
    selected_indices = np.unique(
        np.linspace(
            0,
            len(triangles) - 1,
            min(sample_count, len(triangles)),
            dtype=int,
        )
    )
    q_nodes, q_weights = base._mapped_nodes(quadrature_order)
    high_nodes, high_weights = base._mapped_nodes(cross_check_order)

    maximum_distance = {
        "q12_source_mass": 0.0,
        "q18_source_mass": 0.0,
        "q12_rt_mass": 0.0,
        "q18_rt_mass": 0.0,
        "binary_area": 0.0,
        "binary_load": 0.0,
    }
    maximum_interval_width = {
        "source_mass": 0.0,
        "rt_mass": 0.0,
        "area": 0.0,
        "load": 0.0,
    }
    maximum_quadrature_difference = {
        "source_mass": 0.0,
        "rt_mass": 0.0,
    }
    containment_failures = 0
    containment_checks = 0
    maximum_binary_geometry_guard_ratio = 0.0
    minimum_rt_diagonal_lower = math.inf
    minimum_source_mass_lower = math.inf
    worst_row: dict[str, Any] | None = None

    for triangle_index in selected_indices:
        points = vertices[triangles[int(triangle_index)]]
        exact = _local_interval_forms(points, degree, base)
        q_source, q_rt = _positive_quadrature_local_forms(
            points,
            q_nodes,
            q_weights,
        )
        high_source, high_rt = _positive_quadrature_local_forms(
            points,
            high_nodes,
            high_weights,
        )
        edge_matrix = np.column_stack(
            (points[1] - points[0], points[2] - points[0])
        )
        binary_area = 0.5 * abs(float(np.linalg.det(edge_matrix)))
        binary_load = binary_area / 3.0
        determinant_scale = (
            abs(float(edge_matrix[0, 0] * edge_matrix[1, 1]))
            + abs(float(edge_matrix[1, 0] * edge_matrix[0, 1]))
        )
        area_roundoff_guard = base._up(
            base._gamma(64)
            * max(
                abs(binary_area),
                0.5 * determinant_scale,
                np.finfo(float).tiny,
            )
        )
        load_roundoff_guard = base._up(
            area_roundoff_guard / 3.0
            + base._gamma(8) * abs(binary_load)
        )

        scalar_rows = (
            ("q12_source_mass", q_source, exact["source_mass"], 0.0),
            ("q18_source_mass", high_source, exact["source_mass"], 0.0),
            (
                "binary_area",
                binary_area,
                exact["area"],
                area_roundoff_guard,
            ),
            (
                "binary_load",
                binary_load,
                exact["load"],
                load_roundoff_guard,
            ),
        )
        for name, value, interval, diagnostic_guard in scalar_rows:
            distance = _distance_to_interval(float(value), interval)
            containment_checks += 1
            containment_failures += int(distance > diagnostic_guard)
            maximum_distance[name] = max(maximum_distance[name], distance)
            if diagnostic_guard > 0.0:
                maximum_binary_geometry_guard_ratio = max(
                    maximum_binary_geometry_guard_ratio,
                    distance / diagnostic_guard,
                )
            if distance > diagnostic_guard and (
                worst_row is None or distance > worst_row["distance"]
            ):
                worst_row = {
                    "triangle_index": int(triangle_index),
                    "form": name,
                    "value": float(value),
                    "interval": list(interval),
                    "distance": distance,
                    "diagnostic_roundoff_guard": diagnostic_guard,
                }

        maximum_interval_width["source_mass"] = max(
            maximum_interval_width["source_mass"],
            exact["source_mass"][1] - exact["source_mass"][0],
        )
        maximum_interval_width["area"] = max(
            maximum_interval_width["area"],
            exact["area"][1] - exact["area"][0],
        )
        maximum_interval_width["load"] = max(
            maximum_interval_width["load"],
            exact["load"][1] - exact["load"][0],
        )
        maximum_quadrature_difference["source_mass"] = max(
            maximum_quadrature_difference["source_mass"],
            abs(q_source - high_source),
        )
        minimum_source_mass_lower = min(
            minimum_source_mass_lower,
            exact["source_mass"][0],
        )
        for row in range(3):
            minimum_rt_diagonal_lower = min(
                minimum_rt_diagonal_lower,
                exact["rt_mass"][row][row][0],
            )
            for column in range(row, 3):
                interval = exact["rt_mass"][row][column]
                maximum_interval_width["rt_mass"] = max(
                    maximum_interval_width["rt_mass"],
                    interval[1] - interval[0],
                )
                maximum_quadrature_difference["rt_mass"] = max(
                    maximum_quadrature_difference["rt_mass"],
                    abs(q_rt[row, column] - high_rt[row, column]),
                )
                for name, value in (
                    ("q12_rt_mass", q_rt[row, column]),
                    ("q18_rt_mass", high_rt[row, column]),
                ):
                    distance = _distance_to_interval(
                        float(value),
                        interval,
                    )
                    containment_checks += 1
                    containment_failures += int(distance > 0.0)
                    maximum_distance[name] = max(
                        maximum_distance[name],
                        distance,
                    )
                    if distance > 0.0 and (
                        worst_row is None
                        or distance > worst_row["distance"]
                    ):
                        worst_row = {
                            "triangle_index": int(triangle_index),
                            "form": name,
                            "local_row": row,
                            "local_column": column,
                            "value": float(value),
                            "interval": list(interval),
                            "distance": distance,
                        }

    pi_interval = _pi_interval(base)
    alpha_lowers = []
    alpha_uppers = []
    data_lowers = []
    data_uppers = []
    for element in triangles:
        source_factor, data_constant = _triangle_geometry_constants(
            vertices[element],
            base,
            pi_interval,
        )
        alpha_lowers.append(source_factor[0])
        alpha_uppers.append(source_factor[1])
        data_lowers.append(data_constant[0])
        data_uppers.append(data_constant[1])
    alpha_interval = (max(alpha_lowers), max(alpha_uppers))
    data_interval = (max(data_lowers), max(data_uppers))
    beta_interval = _exact_decimal_interval(
        PRODUCTION_BETA_DECIMAL,
        base,
    )
    combined_interval = base._iv_add(
        data_interval,
        base._iv_mul(alpha_interval, beta_interval),
    )
    target_lower = float(
        dependency["cutoff_solution_operator_route"][
            "Ritz_projection_constant_strict_threshold_lower"
        ]
    )
    headroom_lower = base._down(target_lower - combined_interval[1])
    geometry_passes = combined_interval[1] < target_lower

    checks = {
        "mesh_fingerprint_matches": mesh_fingerprint == expected_fingerprint,
        "sample_count_exact": len(selected_indices) == sample_count,
        "all_sample_quadratures_and_geometry_values_contained": (
            containment_failures == 0
        ),
        "sample_source_mass_strictly_positive": (
            minimum_source_mass_lower > 0.0
        ),
        "sample_rt_diagonal_strictly_positive": (
            minimum_rt_diagonal_lower > 0.0
        ),
        "directed_fixed_beta_geometry_budget_passes": geometry_passes,
    }
    return {
        "kind": "neutral-strip-positive-exponential-RT-interval-pilot",
        "model": (
            "directed local P0 and RT0 moments against exp(+x^2/2), "
            "plus complete-mesh fixed-beta geometry budget"
        ),
        "spacing": SPACING,
        "mesh": {
            "vertex_count": len(vertices),
            "triangle_count": len(triangles),
            "mesh_fingerprint_sha256": mesh_fingerprint,
        },
        "sample": {
            "selection": "unique integer linspace over stored triangle order",
            "requested_count": sample_count,
            "selected_count": len(selected_indices),
            "first_triangle_index": int(selected_indices[0]),
            "last_triangle_index": int(selected_indices[-1]),
            "selected_indices_sha256": hashlib.sha256(
                np.ascontiguousarray(selected_indices).tobytes()
            ).hexdigest(),
        },
        "analytic_enclosure": {
            "Taylor_degree": degree,
            "coefficient_recurrence": (
                "(n+1)*a[n+1]=center*a[n]+a[n-1]"
            ),
            "remainder_majorant": (
                "exp(max_abs_x^2/2) times the absolute Hermite "
                "coefficient majorant times max_abs_z^(degree+1)"
            ),
            "RT0_reduction": (
                "barycentric quadratic moments divided by the directed "
                "squared determinant interval"
            ),
            "minimum_source_mass_interval_lower": (
                minimum_source_mass_lower
            ),
            "minimum_RT_diagonal_interval_lower": (
                minimum_rt_diagonal_lower
            ),
            "maximum_interval_width": maximum_interval_width,
        },
        "quadrature_crosscheck": {
            "primary_order": quadrature_order,
            "independent_order": cross_check_order,
            "containment_checks": containment_checks,
            "containment_failures": containment_failures,
            "maximum_distance_to_interval": maximum_distance,
            "maximum_q12_q18_difference": (
                maximum_quadrature_difference
            ),
            "maximum_binary_geometry_distance_to_roundoff_guard_ratio": (
                maximum_binary_geometry_guard_ratio
            ),
            "worst_failure": worst_row,
            "quadratures_are_diagnostics_not_proof_inputs": True,
        },
        "geometry_budget": {
            "triangles_checked": len(triangles),
            "pi_interval": list(pi_interval),
            "projected_source_norm_factor_interval": list(alpha_interval),
            "data_oscillation_constant_interval": list(data_interval),
            "exact_decimal_beta": str(PRODUCTION_BETA_DECIMAL),
            "exact_decimal_beta_interval": list(beta_interval),
            "combined_C_h_interval_if_kappa_below_beta": list(
                combined_interval
            ),
            "strict_target_lower": target_lower,
            "strict_headroom_lower": headroom_lower,
            "directed_budget_passes": geometry_passes,
        },
        "checks": checks,
        "all_positive_exponential_interval_pilot_checks_pass": bool(
            all(checks.values())
        ),
        "certification_flags": {
            "distributed_RT0_P0_local_interval_pilot_complete": True,
            "complete_mesh_geometry_budget_certified": geometry_passes,
            "complete_mesh_RT0_P0_matrix_entries_enclosed": False,
            "full_mesh_threshold_inertia_certified": False,
            "kappa_h_verified_upper_bound": False,
            "global_weighted_Ritz_projection_constant_certified": False,
            "continuum_spectrum_below_60_captured": False,
        },
        "premises": {
            "hypercircle_result": str(
                hypercircle_result_path
            ).replace("\\", "/"),
            "hypercircle_result_sha256": _sha256_file(
                hypercircle_result_path
            ),
            "continuum_dependency_result": str(
                dependency_result_path
            ).replace("\\", "/"),
            "continuum_dependency_result_sha256": _sha256_file(
                dependency_result_path
            ),
        },
        "next_required_step": (
            "Promote the same local analytic enclosures to a resumable "
            "complete-mesh P/W/D/B assembly audit before any full LDL."
        ),
        "below_normal_priority_set": priority_set,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypercircle-result",
        type=Path,
        default=DEFAULT_HYPERCIRCLE_RESULT,
    )
    parser.add_argument(
        "--dependency-result",
        type=Path,
        default=DEFAULT_DEPENDENCY_RESULT,
    )
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--degree", type=int, default=TAYLOR_DEGREE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_pilot(
        args.hypercircle_result,
        args.dependency_result,
        args.sample_count,
        args.degree,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
