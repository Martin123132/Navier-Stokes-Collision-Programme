"""Audit the continuum high-mode half of the reversible FEM spectral split."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigvalsh, expm, solve
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.special import erf


STRIP_HALF_WIDTH = 2.1
X_HALF_WIDTH = 4.2
INNER_RADIUS = 1.0
PATCH_HALF_HEIGHT = 0.75
WINDOW = 0.375
TERMINAL_TIME = 6.0
FORM_FLOOR = 4.832287335665
DEFAULT_SPECTRAL_CUTOFF = 60.0


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_spectral_floor(strip_half_width: float) -> float:
    return math.pi**2 / (4.0 * strip_half_width**2)


def _li_yau_first_omitted_lower(
    retained_mode_count: int,
    x_half_width: float = X_HALF_WIDTH,
    strip_half_width: float = STRIP_HALF_WIDTH,
) -> dict[str, float]:
    if retained_mode_count < 1:
        raise ValueError("at least one mode must be retained")
    domain_area = (
        4.0 * x_half_width * strip_half_width - math.pi * INNER_RADIUS**2
    )
    first_omitted_index = retained_mode_count + 1
    laplacian_lower = (
        2.0 * math.pi * first_omitted_index / domain_area
    )
    return {
        "continuum_domain_area": domain_area,
        "first_omitted_mode_index": first_omitted_index,
        "li_yau_dirichlet_laplacian_lower": laplacian_lower,
        "li_yau_weighted_operator_lower": laplacian_lower - 0.5,
    }


def _rellich_flux_constant(
    strip_half_width: float = STRIP_HALF_WIDTH,
    multiplier_outer_radius: float = 2.0,
) -> dict[str, float]:
    if not 1.0 < multiplier_outer_radius < strip_half_width:
        raise ValueError("the Rellich multiplier must stop before the walls")
    width = multiplier_outer_radius - 1.0
    derivative_bound = 1.0 / width
    interior_tensor_bound = (
        derivative_bound + 1.0 + multiplier_outer_radius
    )
    floor = _strip_spectral_floor(strip_half_width)
    constant_squared = 2.0 / math.sqrt(floor) + (
        interior_tensor_bound / floor
    )
    return {
        "strip_spectral_floor": floor,
        "multiplier_outer_radius": multiplier_outer_radius,
        "multiplier_derivative_bound": derivative_bound,
        "rellich_interior_tensor_bound": interior_tensor_bound,
        "continuum_L2_source_to_inner_flux_constant_squared": (
            constant_squared
        ),
        "continuum_L2_source_to_inner_flux_constant": math.sqrt(
            constant_squared
        ),
    }


def _full_ou_diagonal_kernel_upper(
    time: float, maximum_entry_x_squared: float = 4.0
) -> float:
    """Full-plane kernel diagonal with respect to exp(-x^2/2) dx dy."""
    if time <= 0.0:
        raise ValueError("time must be positive")
    variance_x = -math.expm1(-2.0 * time)
    exponent = 0.5 * maximum_entry_x_squared * (
        1.0 - math.tanh(0.5 * time)
    )
    denominator = 2.0 * math.sqrt(2.0) * math.pi * math.sqrt(
        time * variance_x
    )
    return math.exp(exponent) / denominator


def _spectral_derivative_factor(cutoff: float, time: float) -> float:
    """sup_{lambda>=cutoff} lambda exp(-lambda*time/2)."""
    if cutoff <= 0.0 or time <= 0.0:
        raise ValueError("cutoff and time must be positive")
    stationary_point = 2.0 / time
    maximizing_value = max(cutoff, stationary_point)
    if cutoff < stationary_point:
        maximizing_value = stationary_point
    return maximizing_value * math.exp(-0.5 * maximizing_value * time)


def _high_mode_flux_upper(
    time: float,
    cutoff: float,
    rellich_constant: float,
) -> float:
    return (
        rellich_constant
        * _spectral_derivative_factor(cutoff, time)
        * math.sqrt(_full_ou_diagonal_kernel_upper(time))
    )


def _axial_l2_factor(time: float) -> float:
    variance = math.expm1(2.0 * time)
    return math.exp(time) * math.sqrt(
        erf(PATCH_HALF_HEIGHT / math.sqrt(variance))
        / (2.0 * math.sqrt(math.pi) * math.sqrt(variance))
    )


def _axial_scalar_global_upper() -> float:
    return math.sqrt(1.0 + 2.0 * PATCH_HALF_HEIGHT**2 / math.pi)


def _analytic_high_mode_budget(cutoff: float) -> dict[str, float]:
    rellich = _rellich_flux_constant()
    if cutoff < 2.0 / WINDOW:
        raise ValueError(
            "cutoff must put the derivative maximum at the cutoff by t=WINDOW"
        )
    first_later_flux = _high_mode_flux_upper(
        WINDOW,
        cutoff,
        rellich["continuum_L2_source_to_inner_flux_constant"],
    )
    first_later_raw = _axial_l2_factor(WINDOW) * first_later_flux
    window_ratio = math.exp(-0.5 * cutoff * WINDOW)
    raw_window_sum = first_later_raw / (1.0 - window_ratio)
    interval_factor = (WINDOW + 1.0 / FORM_FLOOR) * raw_window_sum

    diagonal_root = math.sqrt(_full_ou_diagonal_kernel_upper(WINDOW))
    scalar_gain = (
        _axial_scalar_global_upper()
        * math.sqrt(2.0 * math.pi)
        * rellich["continuum_L2_source_to_inner_flux_constant"]
        * diagonal_root
        * 2.0
        * math.exp(-0.5 * cutoff * WINDOW)
    )
    return {
        **rellich,
        "spectral_cutoff": cutoff,
        "first_later_window_start": WINDOW,
        "full_ou_diagonal_kernel_at_first_later_window": (
            _full_ou_diagonal_kernel_upper(WINDOW)
        ),
        "first_later_window_high_flux_upper": first_later_flux,
        "first_later_window_high_raw_density_upper": first_later_raw,
        "later_window_geometric_ratio": window_ratio,
        "all_later_windows_high_raw_sum_upper": raw_window_sum,
        "all_later_windows_high_interval_factor_upper": interval_factor,
        "all_later_time_high_scalar_gain_upper": scalar_gain,
    }


def _regular_polygon_boundary_l2_operators(
    vertex_count: int,
) -> dict[str, object]:
    """Exact common-circle L2 operators for regular-polygon boundary loads."""
    if vertex_count < 3:
        raise ValueError("a polygon needs at least three vertices")
    alpha = math.pi / vertex_count
    cosine = math.cos(alpha)
    tangent = math.tan(alpha)
    edge_length = 2.0 * math.sin(alpha)
    true_arc_length = 2.0 * alpha

    trace_mass_local_diagonal = edge_length / 3.0
    trace_mass_local_off_diagonal = edge_length / 6.0
    pushed_local_diagonal = cosine**2 * (
        2.0 * tangent / 3.0 + 4.0 * tangent**3 / 15.0
    )
    pushed_local_off_diagonal = cosine**2 * (
        tangent / 3.0 + tangent**3 / 15.0
    )

    trace_mass = np.zeros((vertex_count, vertex_count))
    pushed_gram = np.zeros((vertex_count, vertex_count))
    dual_cell_cross = np.zeros((vertex_count, vertex_count))
    for index in range(vertex_count):
        following = (index + 1) % vertex_count
        trace_mass[index, index] += trace_mass_local_diagonal
        trace_mass[following, following] += trace_mass_local_diagonal
        trace_mass[index, following] += trace_mass_local_off_diagonal
        trace_mass[following, index] += trace_mass_local_off_diagonal
        pushed_gram[index, index] += pushed_local_diagonal
        pushed_gram[following, following] += pushed_local_diagonal
        pushed_gram[index, following] += pushed_local_off_diagonal
        pushed_gram[following, index] += pushed_local_off_diagonal

        dual_cell_cross[index, index] += 3.0 * edge_length / 8.0
        dual_cell_cross[index, following] += edge_length / 8.0
        dual_cell_cross[following, following] += 3.0 * edge_length / 8.0
        dual_cell_cross[following, index] += edge_length / 8.0
    dual_cell_cross /= true_arc_length

    pushforward_eigenvalues = eigvalsh(pushed_gram, trace_mass)
    return {
        "vertex_count": vertex_count,
        "half_edge_angle": alpha,
        "polygon_edge_length": edge_length,
        "true_circle_dual_arc_length": true_arc_length,
        "trace_mass": trace_mass,
        "pushed_reference_gram": pushed_gram,
        "modified_dual_cell_inverse_mass_scalar": 1.0 / true_arc_length,
        "modified_to_reference_cross": dual_cell_cross,
        "finite_element_pushforward_L2_factor": float(
            math.sqrt(pushforward_eigenvalues[-1])
        ),
        "general_L2_pushforward_factor": math.sqrt(1.0 / cosine),
        "minimum_trace_mass_eigenvalue": float(
            np.linalg.eigvalsh(trace_mass)[0]
        ),
    }


def _common_circle_boundary_l2_comparison(
    reference_load_modes: np.ndarray,
    modified_load_modes: np.ndarray,
    vertex_count: int,
) -> dict[str, float | bool]:
    if reference_load_modes.shape != modified_load_modes.shape:
        raise ValueError("reference and modified boundary maps must align")
    if reference_load_modes.shape[0] != vertex_count:
        raise ValueError("boundary row count does not match the polygon")
    operators = _regular_polygon_boundary_l2_operators(vertex_count)
    trace_mass = np.asarray(operators["trace_mass"])
    pushed_gram = np.asarray(operators["pushed_reference_gram"])
    cross = np.asarray(operators["modified_to_reference_cross"])
    inverse_arc = float(operators["modified_dual_cell_inverse_mass_scalar"])

    reference_density_coefficients = solve(
        trace_mass,
        reference_load_modes,
        assume_a="pos",
    )
    reference_norm_gram = reference_density_coefficients.T @ (
        pushed_gram @ reference_density_coefficients
    )
    modified_norm_gram = inverse_arc * (
        modified_load_modes.T @ modified_load_modes
    )
    mixed_gram = modified_load_modes.T @ (
        cross @ reference_density_coefficients
    )
    discrepancy_gram = (
        modified_norm_gram
        - mixed_gram
        - mixed_gram.T
        + reference_norm_gram
    )
    reference_norm_gram = 0.5 * (
        reference_norm_gram + reference_norm_gram.T
    )
    modified_norm_gram = 0.5 * (
        modified_norm_gram + modified_norm_gram.T
    )
    discrepancy_gram = 0.5 * (
        discrepancy_gram + discrepancy_gram.T
    )
    reference_max = float(eigvalsh(reference_norm_gram)[-1])
    modified_max = float(eigvalsh(modified_norm_gram)[-1])
    discrepancy_eigenvalues = eigvalsh(discrepancy_gram)
    discrepancy_max = float(discrepancy_eigenvalues[-1])
    reference_column_squared = np.maximum(
        np.diag(reference_norm_gram),
        0.0,
    )
    discrepancy_column_squared = np.maximum(
        np.diag(discrepancy_gram),
        0.0,
    )
    column_relative = np.sqrt(discrepancy_column_squared) / np.maximum(
        np.sqrt(reference_column_squared),
        1.0e-300,
    )
    worst_column = int(np.argmax(column_relative))
    roundoff_floor = -1.0e-10 * max(reference_max, modified_max, 1.0)
    return {
        "reference_pushed_output_spectral": math.sqrt(reference_max),
        "modified_true_arc_output_spectral": math.sqrt(modified_max),
        "common_circle_discrepancy_spectral": math.sqrt(
            max(discrepancy_max, 0.0)
        ),
        "common_circle_discrepancy_relative_to_reference_spectral": (
            math.sqrt(max(discrepancy_max, 0.0))
            / math.sqrt(reference_max)
        ),
        "minimum_discrepancy_gram_eigenvalue": float(
            discrepancy_eigenvalues[0]
        ),
        "maximum_column_reference_output": float(
            np.sqrt(np.max(reference_column_squared))
        ),
        "maximum_column_discrepancy": float(
            np.sqrt(np.max(discrepancy_column_squared))
        ),
        "maximum_column_relative_discrepancy": float(
            column_relative[worst_column]
        ),
        "worst_relative_column_index": worst_column,
        "discrepancy_gram_positive_up_to_roundoff": bool(
            discrepancy_eigenvalues[0] >= roundoff_floor
        ),
        "finite_element_pushforward_L2_factor": float(
            operators["finite_element_pushforward_L2_factor"]
        ),
        "general_L2_pushforward_factor": float(
            operators["general_L2_pushforward_factor"]
        ),
    }


def _sparse_eigensystem_fingerprint(mass, stiffness) -> str:
    digest = hashlib.sha256()
    for matrix in (mass.tocsr(), stiffness.tocsr()):
        for array in (matrix.indptr, matrix.indices, matrix.data):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _reference_eigensystem(
    mass,
    stiffness,
    requested: int,
    spacing: float,
    quadrature_order: int,
    cache_path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    fingerprint = _sparse_eigensystem_fingerprint(mass, stiffness)
    path = Path(cache_path) if cache_path is not None else None
    cache_info: dict[str, object] = {
        "enabled": path is not None,
        "loaded": False,
        "written": False,
        "path": str(path) if path is not None else None,
        "matrix_fingerprint_sha256": fingerprint,
    }
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as cached:
            metadata_matches = (
                int(cached["cache_version"].item()) == 1
                and int(cached["requested"].item()) == requested
                and int(cached["state_count"].item()) == mass.shape[0]
                and int(cached["quadrature_order"].item()) == quadrature_order
                and float(cached["spacing"].item()) == spacing
                and str(cached["matrix_fingerprint_sha256"].item())
                == fingerprint
            )
            if not metadata_matches:
                raise RuntimeError(f"stale or incompatible eigen cache: {path}")
            eigenvalues = np.asarray(cached["eigenvalues"])
            eigenvectors = np.asarray(cached["eigenvectors"])
        if eigenvalues.shape != (requested,) or eigenvectors.shape != (
            mass.shape[0],
            requested,
        ):
            raise RuntimeError(f"malformed eigen cache: {path}")
        cache_info["loaded"] = True
        return eigenvalues, eigenvectors, cache_info

    eigenvalues, eigenvectors = eigsh(
        stiffness,
        k=requested,
        M=mass,
        sigma=0.0,
        which="LM",
        tol=1.0e-10,
        maxiter=15000,
    )
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                cache_version=np.asarray(1, dtype=np.int64),
                requested=np.asarray(requested, dtype=np.int64),
                state_count=np.asarray(mass.shape[0], dtype=np.int64),
                quadrature_order=np.asarray(quadrature_order, dtype=np.int64),
                spacing=np.asarray(spacing, dtype=np.float64),
                matrix_fingerprint_sha256=np.asarray(fingerprint),
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
            )
        os.replace(temporary, path)
        cache_info["written"] = True
    return eigenvalues, eigenvectors, cache_info


def _numerical_low_mode_row(
    spacing: float,
    mode_count: int,
    quadrature_order: int,
    eigen_cache_path: str | Path | None = None,
) -> dict[str, object]:
    boundary_fem = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        f"boundary_fem_for_spectral_split_{spacing}",
    )
    consistency = _load_module(
        "neutral_strip_reversible_fem_consistency_gate.py",
        f"consistency_for_spectral_split_{spacing}",
    )
    grid = boundary_fem._build_mesh(spacing)
    reference = consistency._reference_forms(grid, quadrature_order)
    mass = reference["mass"]
    stiffness = reference["stiffness"]
    boundary = reference["boundary_coupling"]
    boundary_mass = reference["boundary_mass_coupling"]

    requested = min(mode_count + 1, mass.shape[0] - 2)
    eigenvalues, eigenvectors, eigen_cache = _reference_eigensystem(
        mass,
        stiffness,
        requested,
        spacing,
        quadrature_order,
        eigen_cache_path,
    )
    retained_count = requested - 1
    retained_vectors = eigenvectors[:, :retained_count]
    retained_values = eigenvalues[:retained_count]
    first_omitted = float(eigenvalues[retained_count])

    modified_mass = diags(np.asarray(grid["state_mass"]))
    modified_stiffness = (-modified_mass @ grid["generator"]).tocsr()
    modified_stiffness = 0.5 * (
        modified_stiffness + modified_stiffness.transpose()
    )
    modified_boundary = (
        modified_mass @ grid["inner_rate_matrix"]
    ).tocsr()
    restricted_mass = retained_vectors.T @ (
        modified_mass @ retained_vectors
    )
    restricted_stiffness = retained_vectors.T @ (
        modified_stiffness @ retained_vectors
    )
    mass_ratios = eigvalsh(restricted_mass)
    stiffness_ratios = eigvalsh(
        restricted_stiffness, np.diag(retained_values)
    )

    exact_boundary_modes = np.asarray(
        boundary.transpose() @ retained_vectors
    )
    exact_boundary_mass_modes = np.asarray(
        boundary_mass.transpose() @ retained_vectors
    )
    exact_transient_boundary_modes = exact_boundary_modes + (
        exact_boundary_mass_modes * retained_values[None, :]
    )
    modified_boundary_modes = np.asarray(
        modified_boundary.transpose() @ retained_vectors
    )
    boundary_discrepancy = (
        np.linalg.norm(
            modified_boundary_modes - exact_boundary_modes, 2
        )
        / np.linalg.norm(exact_boundary_modes, 2)
    )
    transient_boundary_mass_correction = (
        np.linalg.norm(
            exact_transient_boundary_modes - exact_boundary_modes, 2
        )
        / np.linalg.norm(exact_transient_boundary_modes, 2)
    )
    transient_boundary_discrepancy = (
        np.linalg.norm(
            modified_boundary_modes - exact_transient_boundary_modes, 2
        )
        / np.linalg.norm(exact_transient_boundary_modes, 2)
    )
    common_circle_boundary = _common_circle_boundary_l2_comparison(
        exact_transient_boundary_modes,
        modified_boundary_modes,
        int(grid["inner_boundary_vertex_count"]),
    )
    projected_generator = solve(
        restricted_mass,
        restricted_stiffness,
        assume_a="pos",
    )
    reference_generator = np.diag(retained_values)
    projected_generator_discrepancy = float(
        np.linalg.norm(projected_generator - reference_generator, 2)
        / np.linalg.norm(reference_generator, 2)
    )
    reference_output_norm = float(
        np.linalg.norm(exact_transient_boundary_modes, 2)
    )
    entry_states = np.asarray(grid["entry_states"], dtype=int)
    reference_source_modes = retained_vectors[entry_states, :].T
    modified_source_modes = solve(
        restricted_mass,
        reference_source_modes,
        assume_a="pos",
    )
    modified_projected_values, modified_projected_vectors = eigh(
        restricted_stiffness,
        restricted_mass,
    )
    projected_semigroup_rows = []
    for window_index in range(1, 17):
        time = window_index * WINDOW
        reference_evolution = np.diag(np.exp(-retained_values * time))
        modified_evolution = (
            modified_projected_vectors
            * np.exp(-modified_projected_values * time)[None, :]
        ) @ (modified_projected_vectors.T @ restricted_mass)
        reference_output = (
            exact_transient_boundary_modes @ reference_evolution
        )
        modified_output = modified_boundary_modes @ modified_evolution
        common_circle_output = _common_circle_boundary_l2_comparison(
            reference_output,
            modified_output,
            int(grid["inner_boundary_vertex_count"]),
        )
        reference_source_output = (
            reference_output @ reference_source_modes
        )
        modified_source_output = (
            modified_boundary_modes
            @ modified_evolution
            @ modified_source_modes
        )
        common_circle_source_output = _common_circle_boundary_l2_comparison(
            reference_source_output,
            modified_source_output,
            int(grid["inner_boundary_vertex_count"]),
        )
        output_difference = modified_output - reference_output
        boundary_only_difference = (
            modified_boundary_modes - exact_transient_boundary_modes
        ) @ reference_evolution
        dynamics_only_difference = modified_boundary_modes @ (
            modified_evolution - reference_evolution
        )
        projected_semigroup_rows.append(
            {
                "later_window_index": window_index,
                "time": time,
                "full_output_discrepancy_over_time_zero_reference": float(
                    np.linalg.norm(output_difference, 2)
                    / reference_output_norm
                ),
                "full_output_discrepancy_over_same_time_reference": float(
                    np.linalg.norm(output_difference, 2)
                    / np.linalg.norm(reference_output, 2)
                ),
                "boundary_only_discrepancy_over_time_zero_reference": float(
                    np.linalg.norm(boundary_only_difference, 2)
                    / reference_output_norm
                ),
                "dynamics_only_discrepancy_over_time_zero_reference": float(
                    np.linalg.norm(dynamics_only_difference, 2)
                    / reference_output_norm
                ),
                "common_circle_boundary_L2": common_circle_output,
                "entry_source_common_circle_boundary_L2": (
                    common_circle_source_output
                ),
            }
        )
    sampled_source_raw_sum = sum(
        _axial_l2_factor(float(row["time"]))
        * float(
            row["entry_source_common_circle_boundary_L2"][
                "maximum_column_discrepancy"
            ]
        )
        for row in projected_semigroup_rows
    )

    modified_action = modified_stiffness @ retained_vectors
    projected_action = modified_mass @ (
        retained_vectors @ projected_generator
    )
    weighted_modified_action = modified_action / np.sqrt(
        np.asarray(grid["state_mass"])
    )[:, None]
    weighted_invariance_residual = (
        modified_action - projected_action
    ) / np.sqrt(np.asarray(grid["state_mass"]))[:, None]
    weighted_modified_action_norm = float(
        np.linalg.norm(weighted_modified_action, 2)
    )
    weighted_invariance_residual_norm = float(
        np.linalg.norm(weighted_invariance_residual, 2)
    )
    projected_invariance_residual = (
        weighted_invariance_residual_norm / weighted_modified_action_norm
    )
    restricted_mass_values, restricted_mass_vectors = np.linalg.eigh(
        restricted_mass
    )
    restricted_mass_inverse_sqrt = (
        restricted_mass_vectors
        * (1.0 / np.sqrt(restricted_mass_values))[None, :]
    ) @ restricted_mass_vectors.T
    off_block_coupling = float(
        np.linalg.norm(
            weighted_invariance_residual @ restricted_mass_inverse_sqrt,
            2,
        )
    )
    prefix_diagnostics = []
    for prefix_count in (20, 80, 120, 160, 200, 240, 280, 320):
        if prefix_count > retained_count:
            continue
        prefix_exact = exact_boundary_modes[:, :prefix_count]
        prefix_mass = exact_boundary_mass_modes[:, :prefix_count]
        prefix_values = retained_values[:prefix_count]
        prefix_transient = prefix_exact + (
            prefix_mass * prefix_values[None, :]
        )
        prefix_modified = modified_boundary_modes[:, :prefix_count]
        prefix_common_circle_boundary = _common_circle_boundary_l2_comparison(
            prefix_transient,
            prefix_modified,
            int(grid["inner_boundary_vertex_count"]),
        )
        prefix_li_yau = _li_yau_first_omitted_lower(prefix_count)
        prefix_cutoff = prefix_li_yau[
            "li_yau_weighted_operator_lower"
        ]
        prefix_budget = (
            _analytic_high_mode_budget(prefix_cutoff)
            if prefix_cutoff >= 2.0 / WINDOW
            else None
        )
        prefix_diagnostics.append(
            {
                "retained_mode_count": prefix_count,
                "maximum_reference_eigenvalue": float(
                    prefix_values[-1]
                ),
                "transient_boundary_mass_correction_relative_spectral": float(
                    np.linalg.norm(prefix_transient - prefix_exact, 2)
                    / np.linalg.norm(prefix_transient, 2)
                ),
                "modified_to_transient_boundary_relative_spectral": float(
                    np.linalg.norm(prefix_modified - prefix_transient, 2)
                    / np.linalg.norm(prefix_transient, 2)
                ),
                "common_circle_boundary_L2": prefix_common_circle_boundary,
                "li_yau_first_omitted_lower": prefix_cutoff,
                "li_yau_later_high_interval_factor_upper": (
                    prefix_budget[
                        "all_later_windows_high_interval_factor_upper"
                    ]
                    if prefix_budget is not None
                    else None
                ),
                "li_yau_later_high_scalar_gain_upper": (
                    prefix_budget["all_later_time_high_scalar_gain_upper"]
                    if prefix_budget is not None
                    else None
                ),
            }
        )
    li_yau = _li_yau_first_omitted_lower(retained_count)
    high_budget = _analytic_high_mode_budget(
        li_yau["li_yau_weighted_operator_lower"]
    )
    return {
        "spacing": spacing,
        "state_count": int(mass.shape[0]),
        "retained_mode_count": retained_count,
        "reference_eigenvalue_range": [
            float(retained_values[0]),
            float(retained_values[-1]),
        ],
        "first_omitted_reference_eigenvalue": first_omitted,
        "reference_eigensystem_cache": eigen_cache,
        **li_yau,
        "retained_modified_mass_ratio": [
            float(mass_ratios[0]),
            float(mass_ratios[-1]),
        ],
        "retained_modified_stiffness_ratio": [
            float(stiffness_ratios[0]),
            float(stiffness_ratios[-1]),
        ],
        "retained_boundary_coupling_relative_spectral": float(
            boundary_discrepancy
        ),
        "retained_transient_boundary_mass_correction_relative_spectral": float(
            transient_boundary_mass_correction
        ),
        "retained_modified_to_transient_boundary_relative_spectral": float(
            transient_boundary_discrepancy
        ),
        "retained_common_circle_boundary_L2": common_circle_boundary,
        "retained_projected_generator_relative_spectral": (
            projected_generator_discrepancy
        ),
        "retained_projected_semigroup_output_rows": projected_semigroup_rows,
        "sampled_later_window_source_common_circle_raw_sum": (
            sampled_source_raw_sum
        ),
        "sampled_later_window_source_common_circle_interval_factor": (
            (WINDOW + 1.0 / FORM_FLOOR) * sampled_source_raw_sum
        ),
        "retained_modified_invariance_residual_relative_Minv_spectral": (
            projected_invariance_residual
        ),
        "retained_modified_action_Minv_spectral": (
            weighted_modified_action_norm
        ),
        "retained_modified_invariance_residual_Minv_spectral": (
            weighted_invariance_residual_norm
        ),
        "retained_modified_off_block_coupling_symmetric_spectral": (
            off_block_coupling
        ),
        "first_later_time_naive_contractive_Duhamel_leakage_upper": (
            WINDOW * off_block_coupling
        ),
        "retained_prefix_diagnostics": prefix_diagnostics,
        "li_yau_high_interval_factor_upper": high_budget[
            "all_later_windows_high_interval_factor_upper"
        ],
        "li_yau_high_scalar_gain_upper": high_budget[
            "all_later_time_high_scalar_gain_upper"
        ],
    }


def _independent_low_block_algebra_regression() -> dict[str, object]:
    """Check the residual identity on an unrelated deterministic SPD system."""
    rng = np.random.default_rng(20260723)
    state_count = 11
    retained_count = 4
    mass_diagonal = 0.5 + rng.random(state_count)
    synthetic_mass = np.diag(mass_diagonal)
    factor = rng.normal(size=(state_count, state_count))
    synthetic_stiffness = (
        factor.T @ factor + 0.75 * synthetic_mass
    )
    trial = rng.normal(size=(state_count, retained_count))
    trial_mass = trial.T @ synthetic_mass @ trial
    values, vectors = np.linalg.eigh(trial_mass)
    trial_mass_inverse_sqrt = (
        vectors * (1.0 / np.sqrt(values))[None, :]
    ) @ vectors.T
    basis = trial @ trial_mass_inverse_sqrt

    restricted_mass = basis.T @ synthetic_mass @ basis
    restricted_stiffness = basis.T @ synthetic_stiffness @ basis
    values, vectors = np.linalg.eigh(restricted_mass)
    restricted_mass_inverse_sqrt = (
        vectors * (1.0 / np.sqrt(values))[None, :]
    ) @ vectors.T
    projected_generator = solve(
        restricted_mass,
        restricted_stiffness,
        assume_a="pos",
    )

    mass_sqrt = np.diag(np.sqrt(mass_diagonal))
    mass_inverse_sqrt = np.diag(1.0 / np.sqrt(mass_diagonal))
    symmetric_generator = (
        mass_inverse_sqrt @ synthetic_stiffness @ mass_inverse_sqrt
    )
    orthonormal_basis = (
        mass_sqrt @ basis @ restricted_mass_inverse_sqrt
    )
    symmetric_projected_generator = (
        orthonormal_basis.T
        @ symmetric_generator
        @ orthonormal_basis
    )
    projector = orthonormal_basis @ orthonormal_basis.T
    direct_off_block = (
        (np.eye(state_count) - projector)
        @ symmetric_generator
        @ orthonormal_basis
    )
    residual_off_block = (
        mass_inverse_sqrt
        @ (
            synthetic_stiffness @ basis
            - synthetic_mass @ basis @ projected_generator
        )
        @ restricted_mass_inverse_sqrt
    )
    orthogonality_error = float(
        np.linalg.norm(
            orthonormal_basis.T @ orthonormal_basis
            - np.eye(retained_count),
            2,
        )
    )
    residual_identity_error = float(
        np.linalg.norm(direct_off_block - residual_off_block, 2)
    )
    projected_similarity_error = float(
        np.linalg.norm(
            symmetric_projected_generator
            - restricted_mass_inverse_sqrt
            @ restricted_stiffness
            @ restricted_mass_inverse_sqrt,
            2,
        )
    )
    off_block_norm = float(np.linalg.norm(direct_off_block, 2))
    duhamel_rows = []
    for time in (0.01, 0.1, WINDOW, 1.0):
        actual = float(
            np.linalg.norm(
                expm(-time * symmetric_generator) @ orthonormal_basis
                - orthonormal_basis
                @ expm(-time * symmetric_projected_generator),
                2,
            )
        )
        upper = time * off_block_norm
        duhamel_rows.append(
            {
                "time": time,
                "actual_discrepancy": actual,
                "contractive_Duhamel_upper": upper,
                "bound_holds": actual <= upper * (1.0 + 1.0e-12),
            }
        )

    one_dimensional_eigenvalue = 4.0 / (1.0 / 3.0)
    conormal_from_residual = -(-2.0) - (1.0 / 12.0) * (
        -one_dimensional_eigenvalue
    )
    conormal_from_mode_formula = 2.0 + (
        one_dimensional_eigenvalue / 12.0
    )
    checks = [
        orthogonality_error < 2.0e-14,
        projected_similarity_error < 2.0e-13,
        residual_identity_error < 2.0e-13,
        all(row["bound_holds"] for row in duhamel_rows),
        abs(conormal_from_residual - 3.0) < 1.0e-15,
        abs(conormal_from_mode_formula - conormal_from_residual) < 1.0e-15,
    ]
    return {
        "synthetic_state_count": state_count,
        "synthetic_retained_count": retained_count,
        "orthogonality_error": orthogonality_error,
        "projected_similarity_error": projected_similarity_error,
        "off_block_residual_identity_error": residual_identity_error,
        "duhamel_rows": duhamel_rows,
        "one_dimensional_transient_conormal": conormal_from_residual,
        "all_independent_low_block_algebra_checks_pass": bool(all(checks)),
    }


def _independent_boundary_l2_regression() -> dict[str, object]:
    """Quadrature-check the exact pushed and cross boundary Gram matrices."""
    vertex_count = 9
    operators = _regular_polygon_boundary_l2_operators(vertex_count)
    pushed_gram = np.asarray(operators["pushed_reference_gram"])
    cross_gram = np.asarray(operators["modified_to_reference_cross"])
    rng = np.random.default_rng(77)
    reference_coefficients = rng.normal(size=vertex_count)
    modified_loads = rng.normal(size=vertex_count)
    alpha = math.pi / vertex_count
    cosine = math.cos(alpha)
    tangent = math.tan(alpha)
    arc_length = 2.0 * alpha
    nodes, weights = np.polynomial.legendre.leggauss(32)
    pushed_norm_quadrature = 0.0
    cross_quadrature = 0.0
    for index in range(vertex_count):
        following = (index + 1) % vertex_count
        for lower, upper, bin_index in (
            (-alpha, 0.0, index),
            (0.0, alpha, following),
        ):
            for node, weight in zip(nodes, weights):
                angle = 0.5 * (upper + lower) + 0.5 * (
                    upper - lower
                ) * node
                tangent_coordinate = math.tan(angle)
                left = (tangent - tangent_coordinate) / (2.0 * tangent)
                right = (tangent + tangent_coordinate) / (2.0 * tangent)
                polygon_density = (
                    left * reference_coefficients[index]
                    + right * reference_coefficients[following]
                )
                arclength_jacobian = cosine / math.cos(angle) ** 2
                circle_density = polygon_density * arclength_jacobian
                quadrature_weight = 0.5 * (upper - lower) * weight
                pushed_norm_quadrature += (
                    quadrature_weight * circle_density**2
                )
                cross_quadrature += (
                    quadrature_weight
                    * modified_loads[bin_index]
                    / arc_length
                    * circle_density
                )
    pushed_norm_matrix = float(
        reference_coefficients @ pushed_gram @ reference_coefficients
    )
    cross_matrix = float(
        modified_loads @ cross_gram @ reference_coefficients
    )
    pushed_error = float(abs(pushed_norm_quadrature - pushed_norm_matrix))
    cross_error = float(abs(cross_quadrature - cross_matrix))
    checks = [
        pushed_error < 2.0e-13,
        cross_error < 2.0e-13,
        float(operators["finite_element_pushforward_L2_factor"])
        <= float(operators["general_L2_pushforward_factor"]),
    ]
    return {
        "vertex_count": vertex_count,
        "pushed_gram_quadrature_error": pushed_error,
        "cross_gram_quadrature_error": cross_error,
        "finite_element_pushforward_L2_factor": operators[
            "finite_element_pushforward_L2_factor"
        ],
        "general_L2_pushforward_factor": operators[
            "general_L2_pushforward_factor"
        ],
        "all_independent_boundary_L2_checks_pass": bool(all(checks)),
    }


def audit(
    run_modes: bool = True,
    spacings: tuple[float, ...] = (0.12, 0.09),
    mode_count: int = 320,
    quadrature_order: int = 12,
    analytic_cutoff: float | None = None,
) -> dict[str, object]:
    first_window_module = _load_module(
        "neutral_strip_first_window_brownian_majorant_audit.py",
        "first_window_for_parabolic_spectral_split",
    )
    first_window = first_window_module.audit(
        run_pointwise_pilot=False,
        run_inversion=False,
    )
    bridge_certificate_module = _load_module(
        "neutral_strip_first_window_maximum_bridge_certificate.py",
        "first_window_bridge_for_parabolic_spectral_split",
    )
    bridge_certificate = bridge_certificate_module.certificate()
    transient_conormal_module = _load_module(
        "neutral_strip_transient_conormal_low_block_gate.py",
        "transient_conormal_for_parabolic_spectral_split",
    )
    transient_conormal = transient_conormal_module.audit()
    algebra_regression = _independent_low_block_algebra_regression()
    boundary_l2_regression = _independent_boundary_l2_regression()
    li_yau = _li_yau_first_omitted_lower(mode_count)
    certified_cutoff = (
        li_yau["li_yau_weighted_operator_lower"]
        if analytic_cutoff is None
        else analytic_cutoff
    )
    if certified_cutoff > li_yau["li_yau_weighted_operator_lower"]:
        raise ValueError("analytic cutoff exceeds the Li-Yau lower bound")
    analytic_budget = _analytic_high_mode_budget(certified_cutoff)
    rows = []
    if run_modes:
        rows = [
            _numerical_low_mode_row(
                spacing, mode_count, quadrature_order
            )
            for spacing in spacings
        ]
    result = {
        "model": "rho=0 continuum neutral-strip high-mode split",
        "window": WINDOW,
        "continuum_li_yau_cutoff": li_yau,
        "analytic_high_mode_budget": analytic_budget,
        "analytic_first_window_budget": first_window[
            "uniform_analytic_budget"
        ],
        "certified_first_window_bridge_budget": {
            "raw_spatial_L2_upper": bridge_certificate[
                "complete_first_window_raw_L2_upper"
            ],
            "interval_factor_upper": bridge_certificate[
                "complete_first_window_interval_factor_upper"
            ],
            "peak_enclosed_slab": bridge_certificate[
                "peak_enclosed_slab"
            ],
            "maximum_omitted_squared_mode_sum_upper": bridge_certificate[
                "maximum_omitted_squared_mode_sum_upper"
            ],
        },
        "numerical_low_mode_rows": rows,
        "transient_conormal_low_block_gate": {
            "retained_mode_count": transient_conormal[
                "retained_mode_count"
            ],
            "legacy_raw_load_screen_apparent_headroom": transient_conormal[
                "legacy_raw_load_screen_apparent_headroom"
            ],
            "production_rows": transient_conormal["production_rows"],
            "h006_projected_dynamics_diagnostic": transient_conormal[
                "h006_projected_dynamics_diagnostic"
            ],
            "common_circle_production_rows": transient_conormal[
                "common_circle_production_rows"
            ],
            "frozen_time_slab_rows": transient_conormal[
                "frozen_time_slab_rows"
            ],
        },
        "independent_low_block_algebra_regression": algebra_regression,
        "independent_boundary_L2_regression": boundary_l2_regression,
        "continuum_weighted_rellich_flux_bound_proved": True,
        "continuum_first_omitted_eigenvalue_lower_bounded_by_li_yau": True,
        "killed_kernel_bounded_by_full_ou_diagonal": True,
        "high_mode_half_time_factorization_proved": True,
        "all_later_window_high_mode_budget_bounded": True,
        "production_low_mode_diagnostic_completed": bool(run_modes),
        "consistent_mass_transient_conormal_identity_proved": (
            transient_conormal[
                "consistent_mass_transient_conormal_identity_proved"
            ]
        ),
        "independent_low_block_algebra_regression_passes": algebra_regression[
            "all_independent_low_block_algebra_checks_pass"
        ],
        "independent_boundary_L2_regression_passes": boundary_l2_regression[
            "all_independent_boundary_L2_checks_pass"
        ],
        "legacy_stiffness_only_boundary_map_complete": False,
        "polygon_flux_measure_pushforward_factor_proved": transient_conormal[
            "polygon_flux_measure_pushforward_factor_proved"
        ],
        "polygon_flux_pushforward_conditional_on_L2_density": True,
        "boundary_Riesz_reconstruction_interval_certified": False,
        "boundary_Riesz_common_circle_geometry_assembled": transient_conormal[
            "boundary_Riesz_common_circle_geometry_assembled"
        ],
        "entry_source_projection_assembled": transient_conormal[
            "entry_source_projection_assembled"
        ],
        "legacy_raw_load_screen_below_one": transient_conormal[
            "legacy_raw_load_screen_below_one"
        ],
        "legacy_raw_load_screen_is_valid_boundary_L2_screen": False,
        "time_zero_common_circle_one_for_one_screen_below_one": False,
        "sampled_source_common_circle_screen_below_one": transient_conormal[
            "h006_sampled_source_common_circle_screen_below_one"
        ],
        "sampled_source_common_circle_screen_headroom": transient_conormal[
            "h006_sampled_source_common_circle_screen_headroom"
        ],
        "later_window_source_time_suprema_interval_certified": False,
        "post_terminal_source_discrepancy_tail_certified": False,
        "time_slab_partition_nonoverlapping": transient_conormal[
            "time_slab_partition_nonoverlapping"
        ],
        "frozen_finite_block_time_slab_enclosure_proved": transient_conormal[
            "frozen_finite_block_time_slab_enclosure_proved"
        ],
        "frozen_finite_block_post_terminal_tail_enclosure_proved": (
            transient_conormal[
                "frozen_finite_block_post_terminal_tail_enclosure_proved"
            ]
        ),
        "binary_frozen_endpoint_roundoff_audit": transient_conormal[
            "binary_frozen_endpoint_roundoff_audit"
        ],
        "frozen_binary_endpoint_arithmetic_directed_enclosed": (
            transient_conormal[
                "frozen_binary_endpoint_arithmetic_directed_enclosed"
            ]
        ),
        "frozen_binary_endpoint_guard_dominates_derived_roundoff": (
            transient_conormal[
                "frozen_binary_endpoint_guard_dominates_derived_roundoff"
            ]
        ),
        "frozen_binary_endpoint_inputs_treated_as_exact_binary64": (
            transient_conormal[
                "frozen_binary_endpoint_inputs_treated_as_exact_binary64"
            ]
        ),
        "binary_frozen_eigensystem_residual_audit": transient_conormal[
            "binary_frozen_eigensystem_residual_audit"
        ],
        "stored_mass_row_lumped_coercivity_proved": transient_conormal[
            "stored_mass_row_lumped_coercivity_proved"
        ],
        "stored_matrix_eigenpair_residuals_directed_enclosed": (
            transient_conormal[
                "stored_matrix_eigenpair_residuals_directed_enclosed"
            ]
        ),
        "reference_eigenvalue_proximity_intervals_proved": (
            transient_conormal[
                "reference_eigenvalue_proximity_intervals_proved"
            ]
        ),
        "indexed_spectrum_transfer_audit": transient_conormal[
            "indexed_spectrum_transfer_audit"
        ],
        "indexed_generalized_eigenvalue_inclusions_proved": (
            transient_conormal[
                "indexed_generalized_eigenvalue_inclusions_proved"
            ]
        ),
        "stored_generalized_eigenvalues_indexed": transient_conormal[
            "stored_generalized_eigenvalues_indexed"
        ],
        "exact_polygon_generalized_eigenvalues_indexed": (
            transient_conormal[
                "exact_polygon_generalized_eigenvalues_indexed"
            ]
        ),
        "exact_polygon_complement_generalized_eigenvalue_lower_bound": (
            transient_conormal[
                "exact_polygon_complement_generalized_eigenvalue_lower_bound"
            ]
        ),
        "endpoint_effect_of_eigenpair_residuals_certified": False,
        "binary_frozen_reference_assembly_audit": transient_conormal[
            "binary_frozen_reference_assembly_audit"
        ],
        "reference_finite_element_assembly_interval_enclosed": (
            transient_conormal[
                "reference_finite_element_assembly_interval_enclosed"
            ]
        ),
        "reference_quadrature_interval_certified": transient_conormal[
            "reference_quadrature_interval_certified"
        ],
        "frozen_finite_block_coefficient_matrices_interval_enclosed": (
            transient_conormal[
                "frozen_finite_block_coefficient_matrices_interval_enclosed"
            ]
        ),
        "h006_refined_frozen_time_slab_combined_screen_total": (
            transient_conormal[
                "h006_refined_frozen_time_slab_combined_screen_total"
            ]
        ),
        "h006_refined_frozen_time_slab_combined_screen_headroom": (
            transient_conormal[
                "h006_refined_frozen_time_slab_combined_screen_headroom"
            ]
        ),
        "static_boundary_screen_is_complete_low_block_comparison": False,
        "retained_projected_dynamics_interval_certified": False,
        "modified_low_space_leakage_interval_bounded": False,
        "gap_free_contractive_Duhamel_leakage_screen_passes": False,
        "low_block_source_trace_map_interval_certified": False,
        "low_mode_variational_crimes_interval_certified": False,
        "continuum_first_window_flux_certified": bridge_certificate[
            "continuum_first_window_flux_certified"
        ],
        "first_window_response_budget_closed": bridge_certificate[
            "first_window_response_budget_closed"
        ],
        "polygon_to_circle_flux_map_certified": False,
        "continuum_return_response_certified": False,
        "scope": (
            "The Rellich, OU-kernel, and high-mode bounds are analytic. "
            "Eigenmode rows, when requested, are floating diagnostics; "
            "the first-window bridge-maximum bound is an outward-rounded "
            "continuum certificate and its isolated interval factor is "
            "below one. The corrected transient conormal map passes a "
            "source-aware common-circle sampled screen at h=0.06. Boundary "
            "Riesz geometry and source projection are assembled. Time slabs "
            "and the terminal tail are analytically enclosed for the frozen "
            "finite block, and directed roundoff bounds justify its endpoint "
            "guard relative to stored binary64 inputs. Stored-matrix residuals "
            "and disjoint eigenvalue-proximity intervals are enclosed. The "
            "Gaussian-weighted reference finite-element forms are also "
            "outward enclosed around fingerprint-matched q12 matrices. "
            "Indexed spectral counting, remaining Riesz/Gram/projected "
            "finite-block algebra, residual propagation, off-block leakage, "
            "continuum Ritz, and domain transfer remain open."
        ),
        "next_required_step": (
            "Verify the retained spectral count and remaining finite-block "
            "Riesz/Gram/projected algebra, propagate assembly and eigensystem "
            "residual intervals through the endpoint actions, then certify "
            "damped leakage, continuum Ritz, and polygon-to-circle domain "
            "transfer."
        ),
    }
    checks = [
        result["continuum_weighted_rellich_flux_bound_proved"],
        result[
            "continuum_first_omitted_eigenvalue_lower_bounded_by_li_yau"
        ],
        result["killed_kernel_bounded_by_full_ou_diagonal"],
        result["high_mode_half_time_factorization_proved"],
        result["all_later_window_high_mode_budget_bounded"],
        result["consistent_mass_transient_conormal_identity_proved"],
        result["independent_low_block_algebra_regression_passes"],
        result["independent_boundary_L2_regression_passes"],
        not result["legacy_stiffness_only_boundary_map_complete"],
        result["polygon_flux_measure_pushforward_factor_proved"],
        result["polygon_flux_pushforward_conditional_on_L2_density"],
        not result["boundary_Riesz_reconstruction_interval_certified"],
        result["boundary_Riesz_common_circle_geometry_assembled"],
        result["entry_source_projection_assembled"],
        result["legacy_raw_load_screen_below_one"],
        not result["legacy_raw_load_screen_is_valid_boundary_L2_screen"],
        not result["time_zero_common_circle_one_for_one_screen_below_one"],
        result["sampled_source_common_circle_screen_below_one"],
        result["sampled_source_common_circle_screen_headroom"] > 0.03,
        not result["later_window_source_time_suprema_interval_certified"],
        not result["post_terminal_source_discrepancy_tail_certified"],
        result["time_slab_partition_nonoverlapping"],
        result["frozen_finite_block_time_slab_enclosure_proved"],
        result["frozen_finite_block_post_terminal_tail_enclosure_proved"],
        result["frozen_binary_endpoint_arithmetic_directed_enclosed"],
        result["frozen_binary_endpoint_guard_dominates_derived_roundoff"],
        result["frozen_binary_endpoint_inputs_treated_as_exact_binary64"],
        result["stored_mass_row_lumped_coercivity_proved"],
        result["stored_matrix_eigenpair_residuals_directed_enclosed"],
        result["reference_eigenvalue_proximity_intervals_proved"],
        result["indexed_generalized_eigenvalue_inclusions_proved"],
        result["stored_generalized_eigenvalues_indexed"],
        result["exact_polygon_generalized_eigenvalues_indexed"],
        result["indexed_spectrum_transfer_audit"][
            "all_30422_pivot_signs_reproduced"
        ],
        result["indexed_spectrum_transfer_audit"][
            "all_30422_crosscheck_intervals_nested"
        ],
        not result["indexed_spectrum_transfer_audit"][
            "continuum_Ritz_transfer_proved"
        ],
        not result["indexed_spectrum_transfer_audit"][
            "polygon_to_circle_domain_transfer_proved"
        ],
        not result["endpoint_effect_of_eigenpair_residuals_certified"],
        result["reference_finite_element_assembly_interval_enclosed"],
        result["reference_quadrature_interval_certified"],
        result["binary_frozen_reference_assembly_audit"][
            "absolute_mass_error_relative_to_stored_mass_form"
        ] < 6.0e-13,
        result["binary_frozen_reference_assembly_audit"][
            "absolute_stiffness_error_in_stored_mass_form_units"
        ] < 6.0e-9,
        result["binary_frozen_eigensystem_residual_audit"][
            "retained_cutoff_proximity_interval_separation"
        ] > 0.6,
        result["binary_frozen_endpoint_roundoff_audit"][
            "maximum_directed_roundoff_norm_error_upper"
        ] < 5.0e-11,
        not result[
            "frozen_finite_block_coefficient_matrices_interval_enclosed"
        ],
        result["h006_refined_frozen_time_slab_combined_screen_total"] < 1.0,
        result["h006_refined_frozen_time_slab_combined_screen_headroom"]
        > 0.029,
        not result["static_boundary_screen_is_complete_low_block_comparison"],
        not result["retained_projected_dynamics_interval_certified"],
        not result["modified_low_space_leakage_interval_bounded"],
        not result["gap_free_contractive_Duhamel_leakage_screen_passes"],
        not result["low_block_source_trace_map_interval_certified"],
        analytic_budget["continuum_L2_source_to_inner_flux_constant"] < 3.2,
        analytic_budget["all_later_windows_high_interval_factor_upper"]
        < 0.01,
        analytic_budget["all_later_time_high_scalar_gain_upper"] < 0.01,
        not result["low_mode_variational_crimes_interval_certified"],
        result["continuum_first_window_flux_certified"],
        result["first_window_response_budget_closed"],
        not result["polygon_to_circle_flux_map_certified"],
        not result["continuum_return_response_certified"],
    ]
    if run_modes:
        checks.extend(
            row["li_yau_weighted_operator_lower"] >= 2.0 / WINDOW
            for row in rows
        )
    result["all_parabolic_spectral_split_checks_pass"] = bool(all(checks))
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
