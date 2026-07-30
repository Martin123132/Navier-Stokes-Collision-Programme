"""Enclose the exact Gaussian-weighted P1 forms around the stored q12 forms."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import mpmath
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix, csr_matrix


EXPECTED_FINGERPRINT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_eigensystem_residual_audit_v1.json"
)
EXPECTED_MATRIX_FINGERPRINTS = {
    "mass": "6962499025001d6b768ac7aaa62c38173551ed94248ba0b27a2fffe140f269c8",
    "stiffness": (
        "de55b4a7910cec423f36b8760fc164e6154c7271227f739c84b0516d0fdf268f"
    ),
    "boundary": (
        "4102a74e02d736aec2b999230b454cd33172480059e82045a487521e1b50966b"
    ),
    "boundary_mass": (
        "a2424aea92826c8408abd5188216643a260d09ebb6f9fb33c06d83e271c2c151"
    ),
}
V1_INITIAL_PRODUCTION_PROVENANCE = {
    "initial_complete_pass_elapsed_seconds": 445.4341170999687,
    "initial_complete_pass_checkpoint_count": 31,
    "initial_complete_pass_cpu_sample_minimum_percent": 29.1,
    "initial_complete_pass_cpu_sample_maximum_percent": 74.8,
    "initial_complete_pass_parked_for_cpu": False,
}


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _down(value: float) -> float:
    return float(np.nextafter(float(value), -math.inf))


def _gamma(operation_count: int) -> float:
    product = operation_count * np.finfo(float).eps
    if product >= 0.01:
        raise RuntimeError("roundoff operation count is too large")
    return _up(product / (1.0 - product))


Interval = tuple[float, float]
Ball = tuple[float, float]
BallArray = tuple[np.ndarray, np.ndarray]


def _iv_add(first: Interval, second: Interval) -> Interval:
    return (
        _down(first[0] + second[0]),
        _up(first[1] + second[1]),
    )


def _iv_sub(first: Interval, second: Interval) -> Interval:
    return (
        _down(first[0] - second[1]),
        _up(first[1] - second[0]),
    )


def _iv_mul(first: Interval, second: Interval) -> Interval:
    products = (
        first[0] * second[0],
        first[0] * second[1],
        first[1] * second[0],
        first[1] * second[1],
    )
    return _down(min(products)), _up(max(products))


def _iv_div(first: Interval, second: Interval) -> Interval:
    if second[0] <= 0.0 <= second[1]:
        raise ZeroDivisionError("interval denominator contains zero")
    reciprocal = (_down(1.0 / second[1]), _up(1.0 / second[0]))
    return _iv_mul(first, reciprocal)


def _iv_neg(value: Interval) -> Interval:
    return -value[1], -value[0]


def _iv_abs(value: Interval) -> Interval:
    if value[0] >= 0.0:
        return value
    if value[1] <= 0.0:
        return -value[1], -value[0]
    return 0.0, _up(max(-value[0], value[1]))


def _interval_to_ball(value: Interval) -> Ball:
    center = 0.5 * value[0] + 0.5 * value[1]
    radius = _up(
        max(center - value[0], value[1] - center)
        + _gamma(4) * (abs(value[0]) + abs(value[1]))
    )
    return center, radius


def _ball_interval(value: Ball) -> Interval:
    return _down(value[0] - value[1]), _up(value[0] + value[1])


def _ball_add(first: Ball, second: Ball) -> Ball:
    center = first[0] + second[0]
    radius = _up(
        first[1]
        + second[1]
        + _gamma(4)
        * (abs(first[0]) + first[1] + abs(second[0]) + second[1])
    )
    return center, radius


def _ball_neg(value: Ball) -> Ball:
    return -value[0], value[1]


def _ball_mul(first: Ball, second: Ball) -> Ball:
    center = first[0] * second[0]
    full_magnitude = (abs(first[0]) + first[1]) * (
        abs(second[0]) + second[1]
    )
    propagated = (
        abs(first[0]) * second[1]
        + abs(second[0]) * first[1]
        + first[1] * second[1]
    )
    return center, _up(propagated + _gamma(8) * full_magnitude)


def _ball_divide_integer(value: Ball, denominator: int) -> Ball:
    if denominator <= 0:
        raise ValueError("the denominator must be positive")
    center = value[0] / denominator
    magnitude = (abs(value[0]) + value[1]) / denominator
    return center, _up(value[1] / denominator + _gamma(4) * magnitude)


def _array_convolve(
    first: BallArray,
    second: BallArray,
    degree: int,
) -> BallArray:
    first_center, first_radius = first
    second_center, second_radius = second
    center = np.convolve(first_center, second_center)[: degree + 1]
    propagated = (
        np.convolve(np.abs(first_center), second_radius)
        + np.convolve(first_radius, np.abs(second_center))
        + np.convolve(first_radius, second_radius)
    )[: degree + 1]
    full_magnitude = np.convolve(
        np.abs(first_center) + first_radius,
        np.abs(second_center) + second_radius,
    )[: degree + 1]
    radius = np.nextafter(
        propagated + _gamma(2 * (degree + 1) + 16) * full_magnitude,
        math.inf,
    )
    return center, radius


def _array_dot(first: BallArray, second: BallArray) -> Ball:
    first_center, first_radius = first
    second_center, second_radius = second
    center = float(first_center @ second_center)
    propagated = float(
        np.abs(first_center) @ second_radius
        + first_radius @ np.abs(second_center)
        + first_radius @ second_radius
    )
    full_magnitude = float(
        (np.abs(first_center) + first_radius)
        @ (np.abs(second_center) + second_radius)
    )
    radius = _up(
        propagated + _gamma(2 * len(first_center) + 16) * full_magnitude
    )
    return center, radius


def _exp_ball(exponent: Interval) -> Ball:
    mpmath.iv.dps = 60
    argument = mpmath.iv.mpf(
        [np.nextafter(exponent[0], -math.inf), np.nextafter(exponent[1], math.inf)]
    )
    enclosed = mpmath.iv.exp(argument)
    lower = _down(float(enclosed.a))
    upper = _up(float(enclosed.b))
    return _interval_to_ball((lower, upper))


def _gaussian_coefficients(center: float, degree: int) -> BallArray:
    center_interval = (center, center)
    exponent = _iv_mul(center_interval, center_interval)
    exponent = _iv_mul((-0.5, -0.5), exponent)
    coefficients: list[Ball] = [_exp_ball(exponent)]
    if degree:
        coefficients.append(
            _ball_mul((-center, 0.0), coefficients[0])
        )
    for index in range(1, degree):
        numerator = _ball_add(
            _ball_mul((-center, 0.0), coefficients[index]),
            _ball_neg(coefficients[index - 1]),
        )
        coefficients.append(
            _ball_divide_integer(numerator, index + 1)
        )
    return (
        np.asarray([value[0] for value in coefficients]),
        np.asarray([value[1] for value in coefficients]),
    )


def _factor_coefficients(
    z_value: Ball,
    alpha: int,
    degree: int,
) -> BallArray:
    coefficients: list[Ball] = [(1.0, 0.0)]
    for index in range(degree):
        next_value = _ball_mul(coefficients[-1], z_value)
        next_value = _ball_mul(
            next_value,
            ((alpha + index) / (index + 1), 0.0),
        )
        coefficients.append(next_value)
    return (
        np.asarray([value[0] for value in coefficients]),
        np.asarray([value[1] for value in coefficients]),
    )


def _moment_sequence(
    determinant: Interval,
    z_values: list[Ball],
    exponents: tuple[int, int, int],
    degree: int,
    factor_cache: dict[tuple[int, int, int], BallArray] | None = None,
) -> BallArray:
    alphas = tuple(value + 1 for value in exponents)
    product: BallArray = (
        np.asarray([1.0] + [0.0] * degree),
        np.zeros(degree + 1),
    )
    for vertex_index, (z_value, alpha) in enumerate(
        zip(z_values, alphas)
    ):
        cache_key = (vertex_index, alpha, degree)
        factor = (
            factor_cache.get(cache_key)
            if factor_cache is not None
            else None
        )
        if factor is None:
            factor = _factor_coefficients(z_value, alpha, degree)
            if factor_cache is not None:
                factor_cache[cache_key] = factor
        product = _array_convolve(
            product,
            factor,
            degree,
        )

    determinant_ball = _interval_to_ball(determinant)
    base_ratio = (
        math.prod(math.factorial(value) for value in exponents)
        / math.factorial(sum(exponents) + 2)
    )
    base = _ball_mul(determinant_ball, (base_ratio, _gamma(4) * base_ratio))
    centers = np.empty(degree + 1)
    radii = np.empty(degree + 1)
    factor = 1.0
    alpha_sum = sum(alphas)
    for index in range(degree + 1):
        if index:
            factor *= index / (alpha_sum + index - 1)
        value = _ball_mul(
            (float(product[0][index]), float(product[1][index])),
            (factor, _gamma(4 * index + 4) * abs(factor)),
        )
        value = _ball_mul(base, value)
        centers[index] = value[0]
        radii[index] = value[1]
    return centers, radii


def _derivative_remainder_coefficient(
    derivative_order: int,
    maximum_abs_x: float,
    maximum_abs_z: float,
) -> float:
    mpmath.mp.dps = 80
    order = derivative_order
    x_value = mpmath.mpf(float(maximum_abs_x))
    coefficient = mpmath.mpf("0")
    for index in range(order // 2 + 1):
        coefficient += x_value ** (order - 2 * index) / (
            2**index
            * mpmath.factorial(index)
            * mpmath.factorial(order - 2 * index)
        )
    value = coefficient * mpmath.mpf(float(maximum_abs_z)) ** order
    return _up(float(value))


def _triangle_geometry(points: np.ndarray) -> dict[str, object]:
    point_intervals = [
        [(float(value), float(value)) for value in point] for point in points
    ]
    first_edge = [
        _iv_sub(point_intervals[1][axis], point_intervals[0][axis])
        for axis in range(2)
    ]
    second_edge = [
        _iv_sub(point_intervals[2][axis], point_intervals[0][axis])
        for axis in range(2)
    ]
    signed_determinant = _iv_sub(
        _iv_mul(first_edge[0], second_edge[1]),
        _iv_mul(first_edge[1], second_edge[0]),
    )
    determinant = _iv_abs(signed_determinant)
    if determinant[0] <= 0.0:
        raise RuntimeError("a triangle determinant interval contains zero")

    inverse = [
        [
            _iv_div(second_edge[1], signed_determinant),
            _iv_div(_iv_neg(second_edge[0]), signed_determinant),
        ],
        [
            _iv_div(_iv_neg(first_edge[1]), signed_determinant),
            _iv_div(first_edge[0], signed_determinant),
        ],
    ]
    gradients = [
        [
            _iv_neg(_iv_add(inverse[0][axis], inverse[1][axis]))
            for axis in range(2)
        ],
        inverse[0],
        inverse[1],
    ]
    gradient_gram: list[list[Interval]] = []
    for first in gradients:
        row = []
        for second in gradients:
            row.append(
                _iv_add(
                    _iv_mul(first[0], second[0]),
                    _iv_mul(first[1], second[1]),
                )
            )
        gradient_gram.append(row)
    return {
        "determinant": determinant,
        "signed_determinant": signed_determinant,
        "gradient_gram": gradient_gram,
    }


def _weighted_moment_precomputed(
    determinant: Interval,
    z_values: list[Ball],
    gaussian: BallArray,
    exponents: tuple[int, int, int],
    degree: int,
    remainder_coefficient: float,
    factor_cache: dict[tuple[int, int, int], BallArray],
) -> Interval:
    moments = _moment_sequence(
        determinant,
        z_values,
        exponents,
        degree,
        factor_cache,
    )
    value = _array_dot(gaussian, moments)

    base_ratio = (
        math.prod(math.factorial(exponent) for exponent in exponents)
        / math.factorial(sum(exponents) + 2)
    )
    unweighted_upper = _iv_mul(
        determinant,
        (base_ratio, base_ratio),
    )[1]
    remainder = _up(remainder_coefficient * unweighted_upper)
    return _ball_interval((value[0], _up(value[1] + remainder)))


def _exact_local_forms(
    points: np.ndarray,
    area_degree: int,
    mass_degree: int,
) -> dict[str, object]:
    geometry = _triangle_geometry(points)
    determinant = geometry["determinant"]
    x_coordinates = points[:, 0]
    center = float(np.mean(x_coordinates))
    center_interval = (center, center)
    z_intervals = [
        _iv_sub((float(value), float(value)), center_interval)
        for value in x_coordinates
    ]
    z_values = [_interval_to_ball(value) for value in z_intervals]
    maximum_abs_x = max(abs(float(value)) for value in x_coordinates)
    maximum_abs_z = max(
        max(abs(value[0]), abs(value[1])) for value in z_intervals
    )
    area_remainder = _derivative_remainder_coefficient(
        area_degree + 1,
        maximum_abs_x,
        maximum_abs_z,
    )
    mass_remainder = _derivative_remainder_coefficient(
        mass_degree + 1,
        maximum_abs_x,
        maximum_abs_z,
    )
    area_gaussian = _gaussian_coefficients(center, area_degree)
    mass_gaussian = _gaussian_coefficients(center, mass_degree)
    factor_cache: dict[tuple[int, int, int], BallArray] = {}

    weighted_area = _weighted_moment_precomputed(
        determinant,
        z_values,
        area_gaussian,
        (0, 0, 0),
        area_degree,
        area_remainder,
        factor_cache,
    )
    mass: list[list[Interval]] = [
        [(0.0, 0.0) for _ in range(3)] for _ in range(3)
    ]
    for row in range(3):
        for column in range(row, 3):
            exponents = [0, 0, 0]
            exponents[row] += 1
            exponents[column] += 1
            value = _weighted_moment_precomputed(
                determinant,
                z_values,
                mass_gaussian,
                tuple(exponents),
                mass_degree,
                mass_remainder,
                factor_cache,
            )
            mass[row][column] = value
            mass[column][row] = value
    stiffness = [
        [
            _iv_mul(weighted_area, geometry["gradient_gram"][row][column])
            for column in range(3)
        ]
        for row in range(3)
    ]
    return {
        "determinant": determinant,
        "weighted_area": weighted_area,
        "mass": mass,
        "stiffness": stiffness,
    }


def _quadrature_local_forms(
    points: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    edge_matrix = np.column_stack(
        (points[1] - points[0], points[2] - points[0])
    )
    determinant = abs(float(np.linalg.det(edge_matrix)))
    inverse = np.linalg.inv(edge_matrix)
    gradients = np.empty((3, 2))
    gradients[1] = inverse[0]
    gradients[2] = inverse[1]
    gradients[0] = -gradients[1] - gradients[2]

    weighted_area = 0.0
    weighted_mass = np.zeros((3, 3))
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
            x_coordinate = float(barycentric @ points[:, 0])
            invariant_weight = math.exp(-0.5 * x_coordinate**2)
            weighted_area += jacobian_weight * invariant_weight
            weighted_mass += (
                jacobian_weight
                * invariant_weight
                * np.outer(barycentric, barycentric)
            )
    weighted_stiffness = weighted_area * (gradients @ gradients.T)
    return weighted_area, weighted_mass, weighted_stiffness


def _mapped_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _distance_to_interval(value: float, interval: Interval) -> float:
    if value < interval[0]:
        return interval[0] - value
    if value > interval[1]:
        return value - interval[1]
    return 0.0


def _entry_error(value: float, interval: Interval) -> float:
    return _up(max(abs(value - interval[0]), abs(value - interval[1])))


def _inflate_sparse_contribution_sum(
    errors: csr_matrix,
    absolute_values: csr_matrix,
    maximum_count: int,
) -> csr_matrix:
    errors = errors.tocsr(copy=True)
    absolute_values = absolute_values.tocsr()
    if not (
        np.array_equal(errors.indptr, absolute_values.indptr)
        and np.array_equal(errors.indices, absolute_values.indices)
    ):
        raise RuntimeError("sparse error structures do not align")
    gamma = _gamma(2 * maximum_count + 12)
    errors.data = np.nextafter(
        (errors.data + gamma * absolute_values.data) / (1.0 - gamma),
        math.inf,
    )
    return errors


def _sparse_frobenius_upper(matrix: csr_matrix) -> float:
    if not len(matrix.data):
        return 0.0
    squared = matrix.data * matrix.data
    total = float(np.sum(squared))
    if total == 0.0:
        return 0.0
    total = _up(total / (1.0 - _gamma(2 * len(squared) + 8)))
    return _up(math.sqrt(max(total, 0.0)))


def _maximum_row_sum(matrix: csr_matrix) -> float:
    if not len(matrix.data):
        return 0.0
    return float(np.max(_nonnegative_row_sums_upper(matrix)))


def _nonnegative_row_sums_upper(matrix: csr_matrix) -> np.ndarray:
    if np.any(matrix.data < 0.0):
        raise ValueError("the row-sum upper helper needs nonnegative data")
    row_sums = np.asarray(matrix.sum(axis=1)).reshape(-1)
    maximum_terms = int(np.max(np.diff(matrix.indptr)))
    return np.nextafter(
        row_sums / (1.0 - _gamma(maximum_terms + 8)),
        math.inf,
    )


def _positive_row_sums_lower(matrix: csr_matrix) -> np.ndarray:
    if np.any(matrix.data <= 0.0):
        raise ValueError("the row-sum lower helper needs positive data")
    row_sums = np.asarray(matrix.sum(axis=1)).reshape(-1)
    maximum_terms = int(np.max(np.diff(matrix.indptr)))
    return np.nextafter(
        row_sums * (1.0 - _gamma(maximum_terms + 8)),
        -math.inf,
    )


def _expected_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["reference_eigensystem_cache"][
        "matrix_fingerprint_sha256"
    ]


def _sparse_matrix_fingerprint(matrix: csr_matrix) -> str:
    digest = hashlib.sha256()
    matrix = matrix.tocsr()
    for array in (matrix.indptr, matrix.indices, matrix.data):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _mesh_fingerprint(vertices: np.ndarray, triangles: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (vertices, triangles):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _write_assembly_checkpoint(
    path: Path,
    contract_json: str,
    next_selected_position: int,
    matrix_rows: dict[str, list[int]],
    matrix_columns: dict[str, list[int]],
    matrix_values: dict[str, list[float]],
    matrix_errors: dict[str, list[float]],
    metrics: dict[str, object],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "cache_version": np.asarray(1, dtype=np.int64),
        "contract_json": np.asarray(contract_json),
        "next_selected_position": np.asarray(
            next_selected_position,
            dtype=np.int64,
        ),
        "metrics_json": np.asarray(
            json.dumps(metrics, sort_keys=True, separators=(",", ":"))
        ),
    }
    for name in matrix_rows:
        arrays[f"{name}_rows"] = np.asarray(
            matrix_rows[name],
            dtype=np.int64,
        )
        arrays[f"{name}_columns"] = np.asarray(
            matrix_columns[name],
            dtype=np.int64,
        )
        arrays[f"{name}_values"] = np.asarray(
            matrix_values[name],
            dtype=float,
        )
        arrays[f"{name}_errors"] = np.asarray(
            matrix_errors[name],
            dtype=float,
        )
    _atomic_write_npz(path, arrays)


def _load_assembly_checkpoint(
    path: Path,
    contract_json: str,
    matrix_names: tuple[str, ...],
) -> dict[str, object] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cached:
        if int(cached["cache_version"].item()) != 1:
            raise RuntimeError("unsupported assembly checkpoint version")
        if str(cached["contract_json"].item()) != contract_json:
            raise RuntimeError("assembly checkpoint contract mismatch")
        result: dict[str, object] = {
            "next_selected_position": int(
                cached["next_selected_position"].item()
            ),
            "metrics": json.loads(str(cached["metrics_json"].item())),
            "matrix_rows": {},
            "matrix_columns": {},
            "matrix_values": {},
            "matrix_errors": {},
        }
        for name in matrix_names:
            rows = np.asarray(cached[f"{name}_rows"], dtype=np.int64)
            columns = np.asarray(
                cached[f"{name}_columns"],
                dtype=np.int64,
            )
            values = np.asarray(cached[f"{name}_values"], dtype=float)
            errors = np.asarray(cached[f"{name}_errors"], dtype=float)
            if not (
                len(rows) == len(columns) == len(values) == len(errors)
            ):
                raise RuntimeError("assembly checkpoint array mismatch")
            result["matrix_rows"][name] = rows.tolist()
            result["matrix_columns"][name] = columns.tolist()
            result["matrix_values"][name] = values.tolist()
            result["matrix_errors"][name] = errors.tolist()
    return result


def _audit(
    spacing: float,
    quadrature_order: int,
    area_degree: int,
    mass_degree: int,
    cross_check_order: int,
    cross_check_stride: int,
    max_elements: int,
    expected_fingerprint_path: Path,
    checkpoint_path: Path | None,
    checkpoint_interval: int,
    cpu_threshold: float,
) -> dict[str, object]:
    started = time.perf_counter()
    below_normal_priority_set = _set_below_normal_priority()
    boundary = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "gaussian_assembly_boundary",
    )
    spectral = _load_module(
        "neutral_strip_parabolic_spectral_split_audit.py",
        "gaussian_assembly_spectral",
    )
    grid = boundary._build_mesh(spacing)
    vertices = np.asarray(grid["vertices"])
    triangles = np.asarray(grid["triangles"])
    total_triangle_count = len(triangles)
    if max_elements > 0 and max_elements < total_triangle_count:
        selected_indices = np.unique(
            np.linspace(
                0,
                total_triangle_count - 1,
                max_elements,
                dtype=int,
            )
        )
    else:
        selected_indices = np.arange(total_triangle_count)
    requested_complete_mesh = (
        len(selected_indices) == total_triangle_count
    )
    if not requested_complete_mesh:
        checkpoint_path = None

    state_vertices = np.asarray(grid["state_vertices"])
    state_lookup = {
        int(vertex): index for index, vertex in enumerate(state_vertices)
    }
    inner_coordinates = np.column_stack(
        (
            np.cos(np.asarray(grid["inner_angles"])),
            np.sin(np.asarray(grid["inner_angles"])),
        )
    )
    inner_vertices = boundary._matched_vertex_indices(
        vertices,
        inner_coordinates,
    )
    inner_lookup = {
        int(vertex): index for index, vertex in enumerate(inner_vertices)
    }

    q_nodes, q_weights = _mapped_nodes(quadrature_order)
    high_nodes, high_weights = _mapped_nodes(cross_check_order)

    state_count = len(state_vertices)
    inner_count = len(inner_vertices)
    matrix_rows: dict[str, list[int]] = {
        name: [] for name in ("mass", "stiffness", "boundary", "boundary_mass")
    }
    matrix_columns = {name: [] for name in matrix_rows}
    matrix_values = {name: [] for name in matrix_rows}
    matrix_errors = {name: [] for name in matrix_rows}

    minimum_determinant_lower = math.inf
    maximum_determinant_width = 0.0
    minimum_exact_mass_lower = math.inf
    maximum_exact_interval_width = {
        "weighted_area": 0.0,
        "mass": 0.0,
        "stiffness": 0.0,
    }
    maximum_q12_distance = {
        "weighted_area": 0.0,
        "mass": 0.0,
        "stiffness": 0.0,
    }
    maximum_q24_distance = {
        "weighted_area": 0.0,
        "mass": 0.0,
        "stiffness": 0.0,
    }
    maximum_q12_q24_difference = {
        "weighted_area": 0.0,
        "mass": 0.0,
        "stiffness": 0.0,
    }
    q24_containment_checks = 0
    q24_containment_failures = 0
    worst_rows: dict[str, dict[str, object] | None] = {
        "q12_mass": None,
        "q12_stiffness": None,
        "q24_mass": None,
        "q24_stiffness": None,
    }

    contract = {
        "algorithm_version": 2,
        "spacing": spacing,
        "quadrature_order": quadrature_order,
        "area_degree": area_degree,
        "mass_degree": mass_degree,
        "cross_check_order": cross_check_order,
        "cross_check_stride": cross_check_stride,
        "selected_triangle_count": int(len(selected_indices)),
        "total_triangle_count": int(total_triangle_count),
        "state_count": int(state_count),
        "inner_count": int(inner_count),
        "mesh_fingerprint_sha256": _mesh_fingerprint(
            vertices,
            triangles,
        ),
    }
    contract_json = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    )

    def current_metrics() -> dict[str, object]:
        return {
            "minimum_determinant_lower": minimum_determinant_lower,
            "maximum_determinant_width": maximum_determinant_width,
            "minimum_exact_mass_lower": minimum_exact_mass_lower,
            "maximum_exact_interval_width": (
                maximum_exact_interval_width
            ),
            "maximum_q12_distance": maximum_q12_distance,
            "maximum_q24_distance": maximum_q24_distance,
            "maximum_q12_q24_difference": (
                maximum_q12_q24_difference
            ),
            "q24_containment_checks": q24_containment_checks,
            "q24_containment_failures": q24_containment_failures,
            "worst_rows": worst_rows,
        }

    start_position = 0
    resumed_from_checkpoint = False
    if checkpoint_path is not None:
        cached = _load_assembly_checkpoint(
            checkpoint_path,
            contract_json,
            tuple(matrix_rows),
        )
        if cached is not None:
            start_position = int(cached["next_selected_position"])
            if not 0 <= start_position <= len(selected_indices):
                raise RuntimeError("assembly checkpoint position is invalid")
            matrix_rows = cached["matrix_rows"]
            matrix_columns = cached["matrix_columns"]
            matrix_values = cached["matrix_values"]
            matrix_errors = cached["matrix_errors"]
            metrics = cached["metrics"]
            minimum_determinant_lower = float(
                metrics["minimum_determinant_lower"]
            )
            maximum_determinant_width = float(
                metrics["maximum_determinant_width"]
            )
            minimum_exact_mass_lower = float(
                metrics["minimum_exact_mass_lower"]
            )
            maximum_exact_interval_width = dict(
                metrics["maximum_exact_interval_width"]
            )
            maximum_q12_distance = dict(
                metrics["maximum_q12_distance"]
            )
            maximum_q24_distance = dict(
                metrics["maximum_q24_distance"]
            )
            maximum_q12_q24_difference = dict(
                metrics["maximum_q12_q24_difference"]
            )
            q24_containment_checks = int(
                metrics["q24_containment_checks"]
            )
            q24_containment_failures = int(
                metrics["q24_containment_failures"]
            )
            worst_rows = dict(metrics["worst_rows"])
            resumed_from_checkpoint = start_position > 0

    cpu_samples: list[float] = []
    consecutive_cpu_threshold_breaches = 0
    parked_for_cpu = False
    processed_count = start_position
    for selected_position in range(start_position, len(selected_indices)):
        triangle_index = int(selected_indices[selected_position])
        triangle = triangles[triangle_index]
        points = vertices[triangle]
        exact = _exact_local_forms(points, area_degree, mass_degree)
        q_area, q_mass, q_stiffness = _quadrature_local_forms(
            points,
            q_nodes,
            q_weights,
        )

        determinant = exact["determinant"]
        minimum_determinant_lower = min(
            minimum_determinant_lower,
            determinant[0],
        )
        maximum_determinant_width = max(
            maximum_determinant_width,
            determinant[1] - determinant[0],
        )
        exact_area = exact["weighted_area"]
        maximum_exact_interval_width["weighted_area"] = max(
            maximum_exact_interval_width["weighted_area"],
            exact_area[1] - exact_area[0],
        )
        maximum_q12_distance["weighted_area"] = max(
            maximum_q12_distance["weighted_area"],
            _distance_to_interval(q_area, exact_area),
        )

        for row in range(3):
            for column in range(3):
                exact_mass = exact["mass"][row][column]
                exact_stiffness = exact["stiffness"][row][column]
                minimum_exact_mass_lower = min(
                    minimum_exact_mass_lower,
                    exact_mass[0],
                )
                mass_width = exact_mass[1] - exact_mass[0]
                stiffness_width = (
                    exact_stiffness[1] - exact_stiffness[0]
                )
                maximum_exact_interval_width["mass"] = max(
                    maximum_exact_interval_width["mass"],
                    mass_width,
                )
                maximum_exact_interval_width["stiffness"] = max(
                    maximum_exact_interval_width["stiffness"],
                    stiffness_width,
                )
                mass_distance = _distance_to_interval(
                    float(q_mass[row, column]),
                    exact_mass,
                )
                stiffness_distance = _distance_to_interval(
                    float(q_stiffness[row, column]),
                    exact_stiffness,
                )
                if mass_distance > maximum_q12_distance["mass"]:
                    maximum_q12_distance["mass"] = mass_distance
                    worst_rows["q12_mass"] = {
                        "triangle_index": int(triangle_index),
                        "local_entry": [row, column],
                        "distance": mass_distance,
                        "interval": list(exact_mass),
                        "quadrature_value": float(q_mass[row, column]),
                    }
                if (
                    stiffness_distance
                    > maximum_q12_distance["stiffness"]
                ):
                    maximum_q12_distance["stiffness"] = stiffness_distance
                    worst_rows["q12_stiffness"] = {
                        "triangle_index": int(triangle_index),
                        "local_entry": [row, column],
                        "distance": stiffness_distance,
                        "interval": list(exact_stiffness),
                        "quadrature_value": float(
                            q_stiffness[row, column]
                        ),
                    }

                global_row = state_lookup.get(int(triangle[row]))
                if global_row is None:
                    continue
                global_column = state_lookup.get(int(triangle[column]))
                if global_column is not None:
                    for name, value, interval in (
                        ("mass", q_mass[row, column], exact_mass),
                        (
                            "stiffness",
                            q_stiffness[row, column],
                            exact_stiffness,
                        ),
                    ):
                        matrix_rows[name].append(global_row)
                        matrix_columns[name].append(global_column)
                        matrix_values[name].append(float(value))
                        matrix_errors[name].append(
                            _entry_error(float(value), interval)
                        )
                    continue
                boundary_column = inner_lookup.get(int(triangle[column]))
                if boundary_column is not None:
                    for name, value, interval in (
                        (
                            "boundary",
                            -q_stiffness[row, column],
                            _iv_neg(exact_stiffness),
                        ),
                        (
                            "boundary_mass",
                            q_mass[row, column],
                            exact_mass,
                        ),
                    ):
                        matrix_rows[name].append(global_row)
                        matrix_columns[name].append(boundary_column)
                        matrix_values[name].append(float(value))
                        matrix_errors[name].append(
                            _entry_error(float(value), interval)
                        )

        should_cross_check = (
            selected_position % cross_check_stride == 0
            or selected_position + 1 == len(selected_indices)
        )
        if should_cross_check:
            high_area, high_mass, high_stiffness = (
                _quadrature_local_forms(points, high_nodes, high_weights)
            )
            maximum_q12_q24_difference["weighted_area"] = max(
                maximum_q12_q24_difference["weighted_area"],
                abs(q_area - high_area),
            )
            maximum_q12_q24_difference["mass"] = max(
                maximum_q12_q24_difference["mass"],
                float(np.max(np.abs(q_mass - high_mass))),
            )
            maximum_q12_q24_difference["stiffness"] = max(
                maximum_q12_q24_difference["stiffness"],
                float(np.max(np.abs(q_stiffness - high_stiffness))),
            )
            checks = [
                (
                    "weighted_area",
                    high_area,
                    exact_area,
                    None,
                )
            ]
            for row in range(3):
                for column in range(3):
                    checks.extend(
                        [
                            (
                                "mass",
                                high_mass[row, column],
                                exact["mass"][row][column],
                                [row, column],
                            ),
                            (
                                "stiffness",
                                high_stiffness[row, column],
                                exact["stiffness"][row][column],
                                [row, column],
                            ),
                        ]
                    )
            for name, value, interval, local_entry in checks:
                distance = _distance_to_interval(float(value), interval)
                q24_containment_checks += 1
                q24_containment_failures += int(distance > 0.0)
                if distance > maximum_q24_distance[name]:
                    maximum_q24_distance[name] = distance
                    if name in ("mass", "stiffness"):
                        worst_rows[f"q24_{name}"] = {
                            "triangle_index": int(triangle_index),
                            "local_entry": local_entry,
                            "distance": distance,
                            "interval": list(interval),
                            "quadrature_value": float(value),
                        }

        processed_count = selected_position + 1
        checkpoint_due = (
            checkpoint_path is not None
            and (
                processed_count % checkpoint_interval == 0
                or processed_count == len(selected_indices)
            )
        )
        if checkpoint_due:
            try:
                import psutil

                cpu_sample = float(psutil.cpu_percent(interval=1.0))
                cpu_samples.append(cpu_sample)
                if cpu_sample > cpu_threshold:
                    consecutive_cpu_threshold_breaches += 1
                else:
                    consecutive_cpu_threshold_breaches = 0
            except Exception:
                pass
            _write_assembly_checkpoint(
                checkpoint_path,
                contract_json,
                processed_count,
                matrix_rows,
                matrix_columns,
                matrix_values,
                matrix_errors,
                current_metrics(),
            )
            if (
                consecutive_cpu_threshold_breaches >= 2
                and processed_count < len(selected_indices)
            ):
                parked_for_cpu = True
                break

    complete = bool(
        requested_complete_mesh
        and processed_count == len(selected_indices)
        and not parked_for_cpu
    )
    matrix_shapes = {
        "mass": (state_count, state_count),
        "stiffness": (state_count, state_count),
        "boundary": (state_count, inner_count),
        "boundary_mass": (state_count, inner_count),
    }
    central_matrices: dict[str, csr_matrix] = {}
    error_matrices: dict[str, csr_matrix] = {}
    maximum_counts: dict[str, int] = {}
    for name, shape in matrix_shapes.items():
        coordinates = (matrix_rows[name], matrix_columns[name])
        central = coo_matrix(
            (matrix_values[name], coordinates),
            shape=shape,
        ).tocsr()
        raw_errors = coo_matrix(
            (matrix_errors[name], coordinates),
            shape=shape,
        ).tocsr()
        absolute_values = coo_matrix(
            (np.abs(matrix_values[name]), coordinates),
            shape=shape,
        ).tocsr()
        counts = coo_matrix(
            (np.ones(len(matrix_values[name])), coordinates),
            shape=shape,
        ).tocsr()
        maximum_count = (
            int(np.max(counts.data)) if len(counts.data) else 0
        )
        central_matrices[name] = central
        error_matrices[name] = _inflate_sparse_contribution_sum(
            raw_errors,
            absolute_values,
            maximum_count,
        )
        maximum_counts[name] = maximum_count

    expected_fingerprint = _expected_fingerprint(
        expected_fingerprint_path
    )
    reconstructed_fingerprint = None
    fingerprint_matches = False
    matrix_fingerprints: dict[str, str] = {}
    per_matrix_fingerprints_match = False
    mass_relative_form_error = None
    stiffness_additive_mass_form_error = None
    exact_mass_stored_coercivity_lower = None
    if complete:
        matrix_fingerprints = {
            name: _sparse_matrix_fingerprint(matrix)
            for name, matrix in central_matrices.items()
        }
        per_matrix_fingerprints_match = all(
            matrix_fingerprints.get(name) == expected
            for name, expected in EXPECTED_MATRIX_FINGERPRINTS.items()
        )
        reconstructed_fingerprint = spectral._sparse_eigensystem_fingerprint(
            central_matrices["mass"],
            central_matrices["stiffness"],
        )
        fingerprint_matches = (
            expected_fingerprint is not None
            and reconstructed_fingerprint == expected_fingerprint
        )
        stored_mass_row_sums = _positive_row_sums_lower(
            central_matrices["mass"]
        )
        mass_error_row_sums = _nonnegative_row_sums_upper(
            error_matrices["mass"]
        )
        stiffness_error_row_sums = _nonnegative_row_sums_upper(
            error_matrices["stiffness"]
        )
        mass_relative_diagonal = _up(
            float(np.max(mass_error_row_sums / stored_mass_row_sums))
        )
        stiffness_additive_diagonal = _up(
            float(
                np.max(
                    stiffness_error_row_sums / stored_mass_row_sums
                )
            )
        )
        stored_coercivity_alpha = 0.14999999999999333
        mass_relative_form_error = _up(
            mass_relative_diagonal / stored_coercivity_alpha
        )
        stiffness_additive_mass_form_error = _up(
            stiffness_additive_diagonal / stored_coercivity_alpha
        )
        exact_mass_stored_coercivity_lower = _down(
            1.0 - mass_relative_form_error
        )

    matrix_rows_output = {}
    for name in matrix_shapes:
        error_matrix = error_matrices[name]
        matrix_rows_output[name] = {
            "assembled_nonzero_count": int(error_matrix.nnz),
            "maximum_entry_contribution_count": maximum_counts[name],
            "maximum_entry_error_upper": (
                float(np.max(error_matrix.data))
                if len(error_matrix.data)
                else 0.0
            ),
            "maximum_row_sum_error_upper": _maximum_row_sum(
                error_matrix
            ),
            "frobenius_error_upper": _sparse_frobenius_upper(
                error_matrix
            ),
        }

    local_enclosures_valid = bool(
        minimum_determinant_lower > 0.0
        and minimum_exact_mass_lower > 0.0
        and all(
            math.isfinite(value)
            for value in (
                *maximum_exact_interval_width.values(),
                *maximum_q12_distance.values(),
                *maximum_q24_distance.values(),
                *maximum_q12_q24_difference.values(),
            )
        )
    )
    assembly_enclosed = bool(
        complete
        and local_enclosures_valid
        and fingerprint_matches
        and per_matrix_fingerprints_match
    )
    return {
        "model": (
            "directed interval audit of Gaussian-weighted P1 assembly "
            "on the stored binary polygon"
        ),
        "spacing": spacing,
        "quadrature_order": quadrature_order,
        "taylor_area_degree": area_degree,
        "taylor_mass_degree": mass_degree,
        "cross_check_quadrature_order": cross_check_order,
        "selected_triangle_count": int(processed_count),
        "requested_triangle_count": int(len(selected_indices)),
        "total_triangle_count": int(total_triangle_count),
        "complete_mesh_audit": complete,
        "state_count": state_count,
        "inner_boundary_vertex_count": inner_count,
        "below_normal_priority_set": below_normal_priority_set,
        "v1_initial_production_provenance": (
            V1_INITIAL_PRODUCTION_PROVENANCE
        ),
        "checkpoint": {
            "enabled": checkpoint_path is not None,
            "path": (
                str(checkpoint_path)
                if checkpoint_path is not None
                else None
            ),
            "contract_sha256": hashlib.sha256(
                contract_json.encode("utf-8")
            ).hexdigest(),
            "resumed": resumed_from_checkpoint,
            "next_selected_position": processed_count,
            "checkpoint_interval": (
                checkpoint_interval
                if checkpoint_path is not None
                else None
            ),
            "cpu_threshold_percent": (
                cpu_threshold if checkpoint_path is not None else None
            ),
            "cpu_samples_percent": cpu_samples,
            "parked_for_repeated_cpu_threshold_breach": parked_for_cpu,
        },
        "method": {
            "geometry": (
                "outward IEEE-754 intervals for binary vertex determinant, "
                "inverse edge map, and P1 gradient Gram"
            ),
            "weight": (
                "Gaussian Taylor recurrence about each triangle x-centroid"
            ),
            "moments": (
                "closed barycentric Dirichlet moments with ball-arithmetic "
                "generating-function convolution"
            ),
            "remainder": (
                "absolute probabilists-Hermite derivative bound times the "
                "unweighted barycentric moment"
            ),
            "global_error": (
                "entrywise sum of local exact-minus-q12 enclosures plus "
                "duplicate-summation roundoff"
            ),
        },
        "local_interval_checks": {
            "all_local_enclosures_valid": local_enclosures_valid,
            "minimum_determinant_lower": minimum_determinant_lower,
            "maximum_determinant_interval_width": (
                maximum_determinant_width
            ),
            "minimum_exact_mass_entry_lower": minimum_exact_mass_lower,
            "maximum_exact_interval_width": (
                maximum_exact_interval_width
            ),
        },
        "quadrature_diagnostics": {
            "maximum_q12_distance_to_exact_interval": (
                maximum_q12_distance
            ),
            "q24_containment_check_count": q24_containment_checks,
            "q24_containment_failure_count": q24_containment_failures,
            "maximum_q24_distance_to_exact_interval": (
                maximum_q24_distance
            ),
            "maximum_absolute_q12_to_q24_difference": (
                maximum_q12_q24_difference
            ),
            "worst_rows": worst_rows,
            "q24_is_diagnostic_not_part_of_the_proof": True,
        },
        "assembled_error_bounds": matrix_rows_output,
        "stored_matrix_reconstruction": {
            "expected_matrix_fingerprint_sha256": expected_fingerprint,
            "reconstructed_matrix_fingerprint_sha256": (
                reconstructed_fingerprint
            ),
            "fingerprints_match": fingerprint_matches,
            "expected_per_matrix_fingerprints_sha256": (
                EXPECTED_MATRIX_FINGERPRINTS
            ),
            "reconstructed_per_matrix_fingerprints_sha256": (
                matrix_fingerprints
            ),
            "all_four_matrix_fingerprints_match": (
                per_matrix_fingerprints_match
            ),
        },
        "form_bounds": {
            "stored_mass_row_lumped_coercivity_lower_used": (
                0.14999999999999333 if complete else None
            ),
            "absolute_mass_error_relative_to_stored_mass_form": (
                mass_relative_form_error
            ),
            "absolute_stiffness_error_in_stored_mass_form_units": (
                stiffness_additive_mass_form_error
            ),
            "exact_mass_lower_relative_to_stored_mass": (
                exact_mass_stored_coercivity_lower
            ),
        },
        "finite_element_assembly_interval_enclosed": assembly_enclosed,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=0.06)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument("--area-degree", type=int, default=22)
    parser.add_argument("--mass-degree", type=int, default=20)
    parser.add_argument("--cross-check-order", type=int, default=24)
    parser.add_argument("--cross-check-stride", type=int, default=257)
    parser.add_argument(
        "--max-elements",
        type=int,
        default=0,
        help="deterministic spread sample; zero audits the complete mesh",
    )
    parser.add_argument(
        "--expected-fingerprint-result",
        type=Path,
        default=EXPECTED_FINGERPRINT_RESULT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "work/ns_collision/results/"
            "neutral_strip_h006_gaussian_assembly_interval_checkpoint_v1.npz"
        ),
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1024)
    parser.add_argument("--cpu-threshold", type=float, default=75.0)
    parser.add_argument("--disable-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint-interval must be positive")

    result = _audit(
        args.spacing,
        args.quadrature_order,
        args.area_degree,
        args.mass_degree,
        args.cross_check_order,
        args.cross_check_stride,
        args.max_elements,
        args.expected_fingerprint_result,
        None if args.disable_checkpoint else args.checkpoint,
        args.checkpoint_interval,
        args.cpu_threshold,
    )
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
