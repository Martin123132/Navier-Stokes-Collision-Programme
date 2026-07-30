from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import mpmath
import numpy as np
from scipy.linalg import cholesky, eigh, eigvalsh, solve, solve_triangular
from scipy.sparse import diags


WINDOW = 0.375
TERMINAL_TIME = 6.0
FORM_FLOOR = 4.832287335665
PATCH_HALF_HEIGHT = 0.75
FIRST_WINDOW_INTERVAL_FACTOR = 0.9523841939624662
LI_YAU_HIGH_INTERVAL_FACTOR = 0.009609366961522469
ROUNDING_GUARD_RELATIVE = 5.0e-11
TRANSFORM_DEFECT_GUARD_RELATIVE = 5.0e-13


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _set_below_normal_priority() -> bool:
    try:
        import psutil

        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return process.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
    except Exception:
        return False


def _axial_l2_global_upper(start: float) -> float:
    """Upper bound valid for every t >= start, using erf(x)<=2x/sqrt(pi)."""
    if start <= 0.0:
        raise ValueError("the later-window axial bound needs positive time")
    mpmath.mp.dps = 80
    value = mpmath.sqrt(
        mpmath.mpf(PATCH_HALF_HEIGHT)
        / (
            mpmath.pi
            * (1 - mpmath.exp(-2 * mpmath.mpf(float(start))))
        )
    )
    return _up(value)


def _spectral_norm_from_gram(gram: np.ndarray) -> float:
    symmetric = 0.5 * (gram + gram.T)
    return _up(math.sqrt(max(float(eigvalsh(symmetric)[-1]), 0.0)))


def _assemble_frozen_block(
    spacing: float,
    mode_count: int,
    quadrature_order: int,
    eigen_cache_path: Path,
) -> dict[str, object]:
    parent = _load_module(
        "neutral_strip_parabolic_spectral_split_audit.py",
        f"common_circle_parent_{spacing}",
    )
    boundary_fem = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        f"common_circle_boundary_fem_{spacing}",
    )
    consistency = _load_module(
        "neutral_strip_reversible_fem_consistency_gate.py",
        f"common_circle_consistency_{spacing}",
    )

    grid = boundary_fem._build_mesh(spacing)
    reference = consistency._reference_forms(
        grid,
        quadrature_order,
        mass_coercivity_alpha=0.15,
    )
    mass = reference["mass"]
    stiffness = reference["stiffness"]
    boundary = reference["boundary_coupling"]
    boundary_mass = reference["boundary_mass_coupling"]
    requested = min(mode_count + 1, mass.shape[0] - 2)
    eigenvalues, eigenvectors, cache_info = parent._reference_eigensystem(
        mass,
        stiffness,
        requested,
        spacing,
        quadrature_order,
        eigen_cache_path,
    )
    retained_count = requested - 1
    values = eigenvalues[:retained_count]
    vectors = eigenvectors[:, :retained_count]

    modified_mass = diags(np.asarray(grid["state_mass"]))
    modified_stiffness = (-modified_mass @ grid["generator"]).tocsr()
    modified_stiffness = 0.5 * (
        modified_stiffness + modified_stiffness.transpose()
    )
    modified_boundary = (
        modified_mass @ grid["inner_rate_matrix"]
    ).tocsr()
    restricted_mass = vectors.T @ (modified_mass @ vectors)
    restricted_stiffness = vectors.T @ (modified_stiffness @ vectors)

    reference_load_modes = np.asarray(boundary.transpose() @ vectors) + (
        np.asarray(boundary_mass.transpose() @ vectors)
        * values[None, :]
    )
    modified_load_modes = np.asarray(modified_boundary.transpose() @ vectors)
    entry_states = np.asarray(grid["entry_states"], dtype=int)
    source_modes = vectors[entry_states, :].T

    lower = cholesky(restricted_mass, lower=True)
    transformed_stiffness = solve_triangular(
        lower,
        restricted_stiffness,
        lower=True,
    )
    transformed_stiffness = solve_triangular(
        lower,
        transformed_stiffness.T,
        lower=True,
    ).T
    transformed_stiffness = 0.5 * (
        transformed_stiffness + transformed_stiffness.T
    )
    modified_values, modified_vectors = eigh(transformed_stiffness)
    reconstructed_stiffness = (
        modified_vectors * modified_values[None, :]
    ) @ modified_vectors.T
    raw_transform_defect = float(
        np.linalg.norm(
            transformed_stiffness - reconstructed_stiffness,
            2,
        )
    )
    transform_defect = _up(
        raw_transform_defect
        + TRANSFORM_DEFECT_GUARD_RELATIVE
        * max(float(np.linalg.norm(transformed_stiffness, 2)), 1.0)
    )

    modified_boundary_orthogonal = solve_triangular(
        lower,
        modified_load_modes.T,
        lower=True,
    ).T
    modified_boundary_modes = modified_boundary_orthogonal @ modified_vectors
    modified_source_orthogonal = solve_triangular(
        lower,
        source_modes,
        lower=True,
    )
    modified_source_modes = modified_vectors.T @ modified_source_orthogonal

    vertex_count = int(grid["inner_boundary_vertex_count"])
    boundary_operators = parent._regular_polygon_boundary_l2_operators(
        vertex_count
    )
    trace_mass = np.asarray(boundary_operators["trace_mass"])
    pushed_gram = np.asarray(
        boundary_operators["pushed_reference_gram"]
    )
    cross = np.asarray(
        boundary_operators["modified_to_reference_cross"]
    )
    inverse_arc = float(
        boundary_operators["modified_dual_cell_inverse_mass_scalar"]
    )
    reference_density_modes = solve(
        trace_mass,
        reference_load_modes,
        assume_a="pos",
    )

    modified_boundary_norm = _up(
        math.sqrt(inverse_arc)
        * float(np.linalg.norm(modified_boundary_orthogonal, 2))
    )
    reference_boundary_norm = _spectral_norm_from_gram(
        reference_density_modes.T
        @ (pushed_gram @ reference_density_modes)
    )
    maximum_modified_source_norm = _up(
        float(np.max(np.linalg.norm(modified_source_orthogonal, axis=0)))
    )
    return {
        "parent": parent,
        "state_count": int(mass.shape[0]),
        "entry_count": int(len(entry_states)),
        "vertex_count": vertex_count,
        "retained_count": retained_count,
        "reference_mass": mass,
        "reference_stiffness": stiffness,
        "reference_mass_coercivity": reference[
            "mass_coercivity_certificate"
        ],
        "reference_all_values": eigenvalues,
        "reference_all_vectors": eigenvectors,
        "reference_values": values,
        "reference_load_modes": reference_load_modes,
        "reference_density_modes": reference_density_modes,
        "reference_cross_density_modes": cross @ reference_density_modes,
        "reference_pushed_density_modes": (
            pushed_gram @ reference_density_modes
        ),
        "reference_source_modes": source_modes,
        "modified_values": modified_values,
        "modified_vectors": modified_vectors,
        "restricted_mass": restricted_mass,
        "restricted_stiffness": restricted_stiffness,
        "restricted_mass_cholesky": lower,
        "transformed_stiffness": transformed_stiffness,
        "modified_load_modes": modified_boundary_modes,
        "modified_source_modes": modified_source_modes,
        "modified_source_orthogonal": modified_source_orthogonal,
        "inverse_arc": inverse_arc,
        "trace_mass": trace_mass,
        "pushed_gram": pushed_gram,
        "cross_gram": cross,
        "reference_boundary_norm": reference_boundary_norm,
        "modified_boundary_norm": modified_boundary_norm,
        "maximum_modified_source_norm": maximum_modified_source_norm,
        "transform_defect": transform_defect,
        "raw_transform_defect": raw_transform_defect,
        "cache_info": cache_info,
    }


def _endpoint_column_norms(
    data: dict[str, object],
    time_value: float,
) -> dict[str, object]:
    reference_values = np.asarray(data["reference_values"])
    modified_values = np.asarray(data["modified_values"])
    reference_source = np.asarray(data["reference_source_modes"])
    modified_source = np.asarray(data["modified_source_modes"])
    reference_decay = np.exp(-reference_values * time_value)
    modified_decay = np.exp(-modified_values * time_value)

    reference_density = (
        np.asarray(data["reference_density_modes"])
        * reference_decay[None, :]
    ) @ reference_source
    reference_cross = (
        np.asarray(data["reference_cross_density_modes"])
        * reference_decay[None, :]
    ) @ reference_source
    reference_pushed = (
        np.asarray(data["reference_pushed_density_modes"])
        * reference_decay[None, :]
    ) @ reference_source
    modified_load = (
        np.asarray(data["modified_load_modes"])
        * modified_decay[None, :]
    ) @ modified_source

    modified_squared = float(data["inverse_arc"]) * np.sum(
        modified_load * modified_load,
        axis=0,
    )
    mixed = np.sum(modified_load * reference_cross, axis=0)
    reference_squared = np.sum(
        reference_density * reference_pushed,
        axis=0,
    )
    discrepancy_squared = np.maximum(
        modified_squared - 2.0 * mixed + reference_squared,
        0.0,
    )
    central = np.sqrt(discrepancy_squared)
    component_scale = np.sqrt(np.maximum(modified_squared, 0.0)) + np.sqrt(
        np.maximum(reference_squared, 0.0)
    )
    arithmetic_guard = _up(
        ROUNDING_GUARD_RELATIVE
        * max(float(np.max(component_scale)), 1.0)
    )
    return {
        "maximum_column_discrepancy": float(np.max(central)),
        "arithmetic_guard": arithmetic_guard,
        "maximum_component_scale": float(np.max(component_scale)),
        "modified_load": modified_load,
        "reference_load": (
            np.asarray(data["reference_load_modes"])
            * reference_decay[None, :]
        )
        @ reference_source,
    }


def _second_derivative_upper(
    data: dict[str, object],
    start: float,
) -> dict[str, float]:
    reference_values = np.asarray(data["reference_values"])
    reference_half = reference_values * np.exp(
        -0.5 * reference_values * start
    )
    reference_boundary_half = (
        np.asarray(data["reference_density_modes"])
        * reference_half[None, :]
    )
    reference_boundary_factor = _spectral_norm_from_gram(
        reference_boundary_half.T
        @ (np.asarray(data["pushed_gram"]) @ reference_boundary_half)
    )
    reference_source_half = reference_half[:, None] * np.asarray(
        data["reference_source_modes"]
    )
    reference_source_factor = _up(
        float(np.max(np.linalg.norm(reference_source_half, axis=0)))
    )
    reference_upper = _up(
        reference_boundary_factor * reference_source_factor
    )

    modified_values = np.asarray(data["modified_values"])
    modified_half = modified_values * np.exp(
        -0.5 * modified_values * start
    )
    modified_boundary_half = np.asarray(data["modified_load_modes"]) * (
        modified_half[None, :]
    )
    modified_boundary_factor = _up(
        math.sqrt(float(data["inverse_arc"]))
        * float(np.linalg.norm(modified_boundary_half, 2))
    )
    modified_source_half = modified_half[:, None] * np.asarray(
        data["modified_source_modes"]
    )
    modified_source_factor = _up(
        float(np.max(np.linalg.norm(modified_source_half, axis=0)))
    )
    modified_upper = _up(
        modified_boundary_factor * modified_source_factor
    )
    return {
        "reference_second_derivative_upper": reference_upper,
        "modified_second_derivative_upper": modified_upper,
        "difference_second_derivative_upper": _up(
            reference_upper + modified_upper
        ),
    }


def _modified_semigroup_defect_upper(
    data: dict[str, object],
    time_value: float,
) -> float:
    return _up(
        float(data["modified_boundary_norm"])
        * time_value
        * float(data["transform_defect"])
        * float(data["maximum_modified_source_norm"])
    )


def _tail_upper(data: dict[str, object]) -> dict[str, float]:
    reference_values = np.asarray(data["reference_values"])
    reference_state = np.exp(
        -reference_values[:, None] * TERMINAL_TIME
    ) * np.asarray(data["reference_source_modes"])
    reference_amplitude = _up(
        float(data["reference_boundary_norm"])
        * float(np.max(np.linalg.norm(reference_state, axis=0)))
    )

    modified_values = np.asarray(data["modified_values"])
    modified_state = np.exp(
        -modified_values[:, None] * TERMINAL_TIME
    ) * np.asarray(data["modified_source_modes"])
    state_defect = _up(
        TERMINAL_TIME
        * float(data["transform_defect"])
        * float(data["maximum_modified_source_norm"])
    )
    modified_amplitude = _up(
        float(data["modified_boundary_norm"])
        * (
            float(np.max(np.linalg.norm(modified_state, axis=0)))
            + state_defect
        )
    )
    reference_decay = float(reference_values[0])
    modified_decay = float(modified_values[0]) - float(
        data["transform_defect"]
    )
    if min(reference_decay, modified_decay) <= 0.0:
        raise RuntimeError("the terminal tail needs positive spectral decay")
    axial_upper = _axial_l2_global_upper(TERMINAL_TIME)
    reference_sum = reference_amplitude / (
        1.0 - math.exp(-reference_decay * WINDOW)
    )
    modified_sum = modified_amplitude / (
        1.0 - math.exp(-modified_decay * WINDOW)
    )
    return {
        "tail_first_window_index": 16,
        "reference_decay_lower": reference_decay,
        "modified_decay_lower": modified_decay,
        "reference_terminal_amplitude": reference_amplitude,
        "modified_terminal_amplitude": modified_amplitude,
        "axial_tail_upper": axial_upper,
        "post_terminal_raw_sum_upper": _up(
            axial_upper * (reference_sum + modified_sum)
        ),
    }


def _independent_interpolation_regression() -> dict[str, object]:
    rates = np.asarray([0.7, 1.3, 2.2, 4.1])
    vectors = np.asarray(
        [[0.8, -0.3, 0.2, 0.1], [-0.4, 0.5, 0.3, -0.2]]
    )
    start = 0.6
    end = 0.73
    endpoint = []
    for point in (start, end):
        endpoint.append(
            float(np.linalg.norm(vectors @ np.exp(-rates * point)))
        )
    second_upper = float(
        np.sum(
            np.linalg.norm(vectors, axis=0)
            * rates**2
            * np.exp(-rates * start)
        )
    )
    enclosure = max(endpoint) + (end - start) ** 2 * second_upper / 8.0
    dense = max(
        float(np.linalg.norm(vectors @ np.exp(-rates * point)))
        for point in np.linspace(start, end, 1001)
    )
    return {
        "endpoint_interpolation_enclosure": enclosure,
        "dense_maximum": dense,
        "enclosure_margin": enclosure - dense,
        "passes": bool(enclosure >= dense),
    }


def certificate(
    spacing: float,
    mode_count: int,
    quadrature_order: int,
    eigen_cache_path: Path,
    substep: float,
) -> dict[str, object]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    if abs(round(WINDOW / substep) * substep - WINDOW) > 1.0e-12:
        raise ValueError("substep must divide the 3/8 window")
    data = _assemble_frozen_block(
        spacing,
        mode_count,
        quadrature_order,
        eigen_cache_path,
    )
    parent = data["parent"]
    endpoint_cache: dict[float, dict[str, object]] = {}

    def endpoint(point: float) -> dict[str, object]:
        key = round(point, 14)
        if key not in endpoint_cache:
            endpoint_cache[key] = _endpoint_column_norms(data, key)
        return endpoint_cache[key]

    subslabs_per_window = int(round(WINDOW / substep))
    window_rows = []
    maximum_interpolation_charge = 0.0
    maximum_arithmetic_guard = 0.0
    maximum_semigroup_defect = 0.0
    for window_index in range(1, 16):
        window_start = window_index * WINDOW
        derivative = _second_derivative_upper(data, window_start)
        second_upper = derivative["difference_second_derivative_upper"]
        raw_upper = 0.0
        discrepancy_upper = 0.0
        for subslab in range(subslabs_per_window):
            start = window_start + subslab * substep
            end = start + substep
            left = endpoint(start)
            right = endpoint(end)
            arithmetic_guard = max(
                float(left["arithmetic_guard"]),
                float(right["arithmetic_guard"]),
            )
            interpolation_charge = substep**2 * second_upper / 8.0
            semigroup_defect = _modified_semigroup_defect_upper(data, end)
            slab_discrepancy = _up(
                max(
                    float(left["maximum_column_discrepancy"]),
                    float(right["maximum_column_discrepancy"]),
                )
                + arithmetic_guard
                + interpolation_charge
                + semigroup_defect
            )
            slab_raw = _up(
                _axial_l2_global_upper(start) * slab_discrepancy
            )
            raw_upper = max(raw_upper, slab_raw)
            discrepancy_upper = max(discrepancy_upper, slab_discrepancy)
            maximum_interpolation_charge = max(
                maximum_interpolation_charge,
                interpolation_charge,
            )
            maximum_arithmetic_guard = max(
                maximum_arithmetic_guard,
                arithmetic_guard,
            )
            maximum_semigroup_defect = max(
                maximum_semigroup_defect,
                semigroup_defect,
            )
        window_rows.append(
            {
                "window_index": window_index,
                "start": window_start,
                "end": (window_index + 1) * WINDOW,
                "subslab_count": subslabs_per_window,
                "maximum_column_discrepancy_upper": discrepancy_upper,
                "raw_source_discrepancy_upper": raw_upper,
                **derivative,
            }
        )

    tail = _tail_upper(data)
    finite_raw_sum = math.fsum(
        float(row["raw_source_discrepancy_upper"]) for row in window_rows
    )
    total_raw_sum = _up(
        finite_raw_sum + tail["post_terminal_raw_sum_upper"]
    )
    interval_factor = _up((WINDOW + 1.0 / FORM_FLOOR) * total_raw_sum)
    combined = _up(
        FIRST_WINDOW_INTERVAL_FACTOR
        + LI_YAU_HIGH_INTERVAL_FACTOR
        + interval_factor
    )

    cross_check_rows = []
    for point in (WINDOW, 3.0, TERMINAL_TIME):
        row = endpoint(point)
        comparison = parent._common_circle_boundary_l2_comparison(
            np.asarray(row["reference_load"]),
            np.asarray(row["modified_load"]),
            int(data["vertex_count"]),
        )
        independent = float(comparison["maximum_column_discrepancy"])
        direct = float(row["maximum_column_discrepancy"])
        cross_check_rows.append(
            {
                "time": point,
                "direct": direct,
                "independent": independent,
                "absolute_difference": abs(direct - independent),
            }
        )
    interpolation_regression = _independent_interpolation_regression()
    checks = [
        bool(data["cache_info"]["loaded"]),
        len(window_rows) == 15,
        int(tail["tail_first_window_index"]) == 16,
        all(
            row["absolute_difference"] < 2.0e-11
            for row in cross_check_rows
        ),
        interpolation_regression["passes"],
        float(data["transform_defect"])
        < 1.0e-8 * max(float(np.max(data["modified_values"])), 1.0),
        combined < 1.0,
    ]
    return {
        "model": "frozen K-mode common-circle source discrepancy",
        "spacing": spacing,
        "state_count": data["state_count"],
        "entry_count": data["entry_count"],
        "inner_boundary_vertex_count": data["vertex_count"],
        "retained_mode_count": data["retained_count"],
        "quadrature_order": quadrature_order,
        "window_length": WINDOW,
        "terminal_time": TERMINAL_TIME,
        "finite_window_indices": [1, 15],
        "tail_first_window_index": 16,
        "substep": substep,
        "subslabs_per_window": subslabs_per_window,
        "reference_eigensystem_cache": data["cache_info"],
        "below_normal_priority_set": priority_set,
        "modified_orthogonal_transform_raw_defect": data[
            "raw_transform_defect"
        ],
        "modified_orthogonal_transform_defect_upper": data[
            "transform_defect"
        ],
        "maximum_interpolation_charge": maximum_interpolation_charge,
        "maximum_arithmetic_guard": maximum_arithmetic_guard,
        "maximum_modified_semigroup_defect_upper": (
            maximum_semigroup_defect
        ),
        "window_rows": window_rows,
        "finite_raw_sum_upper": finite_raw_sum,
        "post_terminal_tail": tail,
        "total_later_low_block_raw_sum_upper": total_raw_sum,
        "later_low_block_interval_factor_upper": interval_factor,
        "combined_screen_total": combined,
        "combined_screen_headroom": 1.0 - combined,
        "combined_screen_below_one": combined < 1.0,
        "cross_check_rows": cross_check_rows,
        "independent_interpolation_regression": interpolation_regression,
        "frozen_finite_block_time_slab_enclosure_proved": True,
        "frozen_finite_block_post_terminal_tail_enclosure_proved": True,
        "coefficient_matrices_interval_enclosed": False,
        "reference_generalized_eigenpairs_interval_enclosed": False,
        "modified_projected_matrices_interval_enclosed": False,
        "uses_guarded_floating_endpoint_arithmetic": True,
        "off_block_leakage_interval_certified": False,
        "continuum_Ritz_projector_error_certified": False,
        "polygon_domain_perturbation_certified": False,
        "continuum_return_response_certified": False,
        "scope": (
            "The endpoint-interpolation and geometric-tail inequalities are "
            "analytic for the frozen finite block. Endpoint arithmetic uses "
            "an explicit conservative floating guard; the assembled forms, "
            "reference eigensystem, and projected matrices are not yet "
            "directed-interval enclosed."
        ),
        "all_frozen_time_slab_certificate_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing", type=float, default=0.06)
    parser.add_argument("--mode-count", type=int, default=240)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument("--substep", type=float, default=0.025)
    parser.add_argument("--eigen-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = certificate(
        args.spacing,
        args.mode_count,
        args.quadrature_order,
        args.eigen_cache,
        args.substep,
    )
    if args.output is not None:
        _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
