#!/usr/bin/env python3
"""Floating weighted P1/RT0 hypercircle pilot for the neutral-strip mesh."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import bmat, coo_matrix, csc_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu
import triangle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONTINUUM_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_continuum_ritz_dependency_audit_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h012_weighted_hypercircle_pilot_v1.json"
)

X_HALF_WIDTH = 4.2
STRIP_HALF_WIDTH = 2.1
DEFAULT_SPACING = 0.12
DEFAULT_QUADRATURE_ORDER = 12
CPU_SAMPLE_PERIOD_SECONDS = 5.0
DAYTIME_CPU_PARK_THRESHOLD_PERCENT = 75.0


def _load_module(filename: str, module_name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
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


def _mapped_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _edge_sign(
    vertices: np.ndarray,
    edge: tuple[int, int],
    opposite_vertex: int,
) -> int:
    """Map a global right-normal edge flux to a local outward flux."""
    first, second = edge
    side = _cross(
        vertices[second] - vertices[first],
        vertices[opposite_vertex] - vertices[first],
    )
    if side == 0.0:
        raise RuntimeError("degenerate edge-triangle incidence")
    return 1 if side > 0.0 else -1


def _triangle_quadrature(
    points: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return integral(mu), P1 massless stiffness, and RT0 mu^-1 mass."""
    edge_matrix = np.column_stack(
        (points[1] - points[0], points[2] - points[0])
    )
    determinant = abs(float(np.linalg.det(edge_matrix)))
    if determinant <= 0.0:
        raise RuntimeError("degenerate triangle")
    inverse = np.linalg.inv(edge_matrix)
    gradients = np.empty((3, 2))
    gradients[1] = inverse[0]
    gradients[2] = inverse[1]
    gradients[0] = -gradients[1] - gradients[2]

    weighted_area = 0.0
    inverse_weighted_area = 0.0
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
            exponent = 0.5 * float(point[0]) ** 2
            mu = math.exp(-exponent)
            inverse_mu = math.exp(exponent)
            weighted_area += jacobian_weight * mu
            inverse_weighted_area += jacobian_weight * inverse_mu
            local_rt = (point[None, :] - points) / determinant
            rt_mass += (
                jacobian_weight
                * inverse_mu
                * (local_rt @ local_rt.T)
            )
    stiffness = weighted_area * (gradients @ gradients.T)
    return inverse_weighted_area, stiffness, rt_mass


@dataclass
class Assembly:
    vertices: np.ndarray
    triangles: np.ndarray
    state_vertices: np.ndarray
    areas: np.ndarray
    source_mass: np.ndarray
    p1_stiffness: csc_matrix
    p1_load: csc_matrix
    rt_mass: csc_matrix
    divergence: csc_matrix
    data_oscillation_constant_upper: float
    projected_source_norm_factor_upper: float
    mesh_fingerprint_sha256: str
    diagnostics: dict[str, Any]


def _assemble(
    spacing: float,
    quadrature_order: int,
) -> Assembly:
    boundary = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "weighted_hypercircle_boundary",
    )
    assembly_module = _load_module(
        "neutral_strip_gaussian_weighted_assembly_interval_audit.py",
        "weighted_hypercircle_assembly",
    )
    mesh_input, _, _ = boundary._mesh_input(
        spacing,
        X_HALF_WIDTH,
        STRIP_HALF_WIDTH,
    )
    maximum_area = 0.45 * spacing**2
    mesh = triangle.triangulate(
        mesh_input,
        f"pYq28a{maximum_area:.17g}Q",
    )
    vertices = np.asarray(mesh["vertices"], dtype=float)
    triangles = np.asarray(mesh["triangles"], dtype=int)
    boundary_segments = np.asarray(mesh["segments"], dtype=int)
    boundary_vertices = set(int(value) for value in boundary_segments.ravel())
    state_vertices = np.asarray(
        [
            index
            for index in range(len(vertices))
            if index not in boundary_vertices
        ],
        dtype=int,
    )
    state_lookup = {
        int(vertex): state for state, vertex in enumerate(state_vertices)
    }

    edge_indices: dict[tuple[int, int], int] = {}
    local_edges: list[list[int]] = []
    local_signs: list[list[int]] = []
    edge_incidence_signs: dict[int, list[int]] = {}
    for element in triangles:
        element_edges = []
        element_signs = []
        for opposite in range(3):
            edge = tuple(
                sorted(
                    (
                        int(element[(opposite + 1) % 3]),
                        int(element[(opposite + 2) % 3]),
                    )
                )
            )
            edge_index = edge_indices.setdefault(edge, len(edge_indices))
            sign = _edge_sign(vertices, edge, int(element[opposite]))
            element_edges.append(edge_index)
            element_signs.append(sign)
            edge_incidence_signs.setdefault(edge_index, []).append(sign)
        local_edges.append(element_edges)
        local_signs.append(element_signs)

    p1_rows: list[int] = []
    p1_columns: list[int] = []
    p1_values: list[float] = []
    load_rows: list[int] = []
    load_columns: list[int] = []
    load_values: list[float] = []
    rt_rows: list[int] = []
    rt_columns: list[int] = []
    rt_values: list[float] = []
    div_rows: list[int] = []
    div_columns: list[int] = []
    div_values: list[float] = []
    areas = np.empty(len(triangles))
    source_mass = np.empty(len(triangles))
    data_oscillation_rows = np.empty(len(triangles))
    source_norm_factor_rows = np.empty(len(triangles))
    nodes, weights = _mapped_nodes(quadrature_order)

    for triangle_index, element in enumerate(triangles):
        points = vertices[element]
        determinant = abs(
            _cross(points[1] - points[0], points[2] - points[0])
        )
        area = 0.5 * determinant
        inverse_weighted_area, local_stiffness, local_rt_mass = (
            _triangle_quadrature(points, nodes, weights)
        )
        areas[triangle_index] = area
        source_mass[triangle_index] = inverse_weighted_area

        edge_lengths = [
            float(np.linalg.norm(points[(index + 1) % 3] - points[index]))
            for index in range(3)
        ]
        absolute_x = np.abs(points[:, 0])
        maximum_absolute_x = float(np.max(absolute_x))
        if float(np.min(points[:, 0])) <= 0.0 <= float(
            np.max(points[:, 0])
        ):
            minimum_absolute_x = 0.0
        else:
            minimum_absolute_x = float(np.min(absolute_x))
        square_oscillation = (
            maximum_absolute_x**2 - minimum_absolute_x**2
        )
        source_norm_factor_rows[triangle_index] = math.exp(
            0.25 * square_oscillation
        )
        data_oscillation_rows[triangle_index] = (
            max(edge_lengths)
            / math.pi
            * source_norm_factor_rows[triangle_index]
        )

        for local_row, vertex_row in enumerate(element):
            state_row = state_lookup.get(int(vertex_row))
            if state_row is not None:
                load_rows.append(state_row)
                load_columns.append(triangle_index)
                load_values.append(area / 3.0)
            for local_column, vertex_column in enumerate(element):
                state_column = state_lookup.get(int(vertex_column))
                if state_row is not None and state_column is not None:
                    p1_rows.append(state_row)
                    p1_columns.append(state_column)
                    p1_values.append(
                        float(local_stiffness[local_row, local_column])
                    )

        edges = local_edges[triangle_index]
        signs = local_signs[triangle_index]
        for local_row, edge_row in enumerate(edges):
            div_rows.append(triangle_index)
            div_columns.append(edge_row)
            div_values.append(float(signs[local_row]))
            for local_column, edge_column in enumerate(edges):
                rt_rows.append(edge_row)
                rt_columns.append(edge_column)
                rt_values.append(
                    float(
                        signs[local_row]
                        * signs[local_column]
                        * local_rt_mass[local_row, local_column]
                    )
                )

    state_count = len(state_vertices)
    triangle_count = len(triangles)
    edge_count = len(edge_indices)
    p1_stiffness = coo_matrix(
        (p1_values, (p1_rows, p1_columns)),
        shape=(state_count, state_count),
    ).tocsc()
    p1_load = coo_matrix(
        (load_values, (load_rows, load_columns)),
        shape=(state_count, triangle_count),
    ).tocsc()
    rt_mass = coo_matrix(
        (rt_values, (rt_rows, rt_columns)),
        shape=(edge_count, edge_count),
    ).tocsc()
    divergence = coo_matrix(
        (div_values, (div_rows, div_columns)),
        shape=(triangle_count, edge_count),
    ).tocsc()

    incidence_counts = np.asarray(
        [len(edge_incidence_signs[index]) for index in range(edge_count)]
    )
    interior_sign_sums = [
        sum(edge_incidence_signs[index])
        for index in range(edge_count)
        if len(edge_incidence_signs[index]) == 2
    ]
    diagnostics = {
        "boundary_edge_count": int(np.count_nonzero(incidence_counts == 1)),
        "edge_count": edge_count,
        "every_edge_has_one_or_two_incident_triangles": bool(
            np.all((incidence_counts == 1) | (incidence_counts == 2))
        ),
        "interior_edge_outward_signs_cancel": bool(
            all(value == 0 for value in interior_sign_sums)
        ),
        "maximum_triangle_diameter": float(
            max(
                np.linalg.norm(
                    vertices[element[(index + 1) % 3]]
                    - vertices[element[index]]
                )
                for element in triangles
                for index in range(3)
            )
        ),
        "minimum_source_mass": float(np.min(source_mass)),
        "p1_stiffness_symmetry_error": float(
            np.max(np.abs((p1_stiffness - p1_stiffness.T).data))
            if (p1_stiffness - p1_stiffness.T).nnz
            else 0.0
        ),
        "rt_mass_symmetry_error": float(
            np.max(np.abs((rt_mass - rt_mass.T).data))
            if (rt_mass - rt_mass.T).nnz
            else 0.0
        ),
    }
    return Assembly(
        vertices=vertices,
        triangles=triangles,
        state_vertices=state_vertices,
        areas=areas,
        source_mass=source_mass,
        p1_stiffness=p1_stiffness,
        p1_load=p1_load,
        rt_mass=rt_mass,
        divergence=divergence,
        data_oscillation_constant_upper=float(
            np.max(data_oscillation_rows)
        ),
        projected_source_norm_factor_upper=float(
            np.max(source_norm_factor_rows)
        ),
        mesh_fingerprint_sha256=assembly_module._mesh_fingerprint(
            vertices,
            triangles,
        ),
        diagnostics=diagnostics,
    )


class _CpuMonitor:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.samples: list[float] = []
        self.last_sample = time.monotonic()
        self.consecutive_high = 0
        if enabled:
            try:
                import psutil

                psutil.cpu_percent(interval=None)
            except Exception:
                self.enabled = False

    def sample_if_due(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last_sample < CPU_SAMPLE_PERIOD_SECONDS:
            return
        import psutil

        value = float(psutil.cpu_percent(interval=None))
        self.samples.append(value)
        self.last_sample = now
        if value > DAYTIME_CPU_PARK_THRESHOLD_PERCENT:
            self.consecutive_high += 1
        else:
            self.consecutive_high = 0
        if self.consecutive_high >= 2:
            raise RuntimeError(
                "daytime CPU park threshold reached during eigensolve"
            )


def _solve_pilot(
    assembly: Assembly,
    eig_tolerance: float,
    maximum_iterations: int,
    monitor_cpu: bool,
) -> dict[str, Any]:
    state_count = len(assembly.state_vertices)
    triangle_count = len(assembly.triangles)
    edge_count = assembly.rt_mass.shape[0]
    mixed = bmat(
        [
            [assembly.rt_mass, assembly.divergence.T],
            [assembly.divergence, None],
        ],
        format="csc",
    )

    factor_started = time.perf_counter()
    p1_factor = splu(assembly.p1_stiffness)
    mixed_factor = splu(mixed)
    factor_seconds = time.perf_counter() - factor_started
    inverse_source_sqrt = 1.0 / np.sqrt(assembly.source_mass)
    monitor = _CpuMonitor(monitor_cpu)
    matvec_count = 0

    def apply_normalized(source: np.ndarray) -> np.ndarray:
        nonlocal matvec_count
        monitor.sample_if_due()
        matvec_count += 1
        coefficient = inverse_source_sqrt * np.asarray(source)
        mixed_rhs = np.zeros(edge_count + triangle_count)
        mixed_rhs[edge_count:] = -assembly.areas * coefficient
        mixed_solution = mixed_factor.solve(mixed_rhs)
        multiplier = mixed_solution[edge_count:]
        p1_solution = p1_factor.solve(assembly.p1_load @ coefficient)
        quadratic_action = (
            assembly.areas * multiplier
            - assembly.p1_load.T @ p1_solution
        )
        return inverse_source_sqrt * np.asarray(quadratic_action)

    operator = LinearOperator(
        (triangle_count, triangle_count),
        matvec=apply_normalized,
        rmatvec=apply_normalized,
        dtype=float,
    )
    initial = np.sin(
        np.arange(1, triangle_count + 1, dtype=float) * math.sqrt(2.0)
    )
    eig_started = time.perf_counter()
    values, vectors = eigsh(
        operator,
        k=1,
        which="LA",
        v0=initial,
        tol=eig_tolerance,
        maxiter=maximum_iterations,
    )
    eig_seconds = time.perf_counter() - eig_started
    eigenvalue = float(values[0])
    eigenvector = np.asarray(vectors[:, 0])
    action = apply_normalized(eigenvector)
    eigen_residual = float(
        np.linalg.norm(action - eigenvalue * eigenvector)
    )

    rng = np.random.default_rng(20260724)
    first = rng.standard_normal(triangle_count)
    second = rng.standard_normal(triangle_count)
    action_first = apply_normalized(first)
    action_second = apply_normalized(second)
    symmetry_defect = float(
        abs(first @ action_second - second @ action_first)
        / max(
            1.0,
            abs(first @ action_second),
            abs(second @ action_first),
        )
    )

    coefficient = inverse_source_sqrt * first
    mixed_rhs = np.zeros(edge_count + triangle_count)
    mixed_rhs[edge_count:] = -assembly.areas * coefficient
    mixed_solution = mixed_factor.solve(mixed_rhs)
    flux = mixed_solution[:edge_count]
    multiplier = mixed_solution[edge_count:]
    p1_solution = p1_factor.solve(assembly.p1_load @ coefficient)
    mixed_residual = mixed @ mixed_solution - mixed_rhs
    p1_residual = (
        assembly.p1_stiffness @ p1_solution
        - assembly.p1_load @ coefficient
    )
    direct_objective = float(
        flux @ (assembly.rt_mass @ flux)
        - p1_solution @ (assembly.p1_stiffness @ p1_solution)
    )
    action_objective = float(
        coefficient
        @ (
            assembly.areas * multiplier
            - assembly.p1_load.T @ p1_solution
        )
    )
    objective_relative_defect = float(
        abs(direct_objective - action_objective)
        / max(1.0, abs(direct_objective), abs(action_objective))
    )

    checks = {
        "eigenvalue_nonnegative": eigenvalue >= 0.0,
        "eigen_residual_below_1e_7": eigen_residual <= 1.0e-7,
        "matrix_action_symmetric_to_1e_9": symmetry_defect <= 1.0e-9,
        "mixed_relative_residual_below_1e_9": (
            float(np.linalg.norm(mixed_residual))
            / max(1.0, float(np.linalg.norm(mixed_rhs)))
            <= 1.0e-9
        ),
        "objective_identity_relative_defect_below_1e_9": (
            objective_relative_defect <= 1.0e-9
        ),
        "p1_relative_residual_below_1e_9": (
            float(np.linalg.norm(p1_residual))
            / max(
                1.0,
                float(np.linalg.norm(assembly.p1_load @ coefficient)),
            )
            <= 1.0e-9
        ),
    }
    return {
        "all_floating_linear_algebra_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "cpu_samples_percent": monitor.samples,
        "eigen_residual_norm": eigen_residual,
        "factor_seconds": factor_seconds,
        "kappa_h_floating": math.sqrt(max(0.0, eigenvalue)),
        "largest_normalized_quadratic_eigenvalue_floating": eigenvalue,
        "matvec_count": matvec_count,
        "mixed_matrix_dimension": edge_count + triangle_count,
        "mixed_matrix_nnz": int(mixed.nnz),
        "mixed_relative_residual": float(
            np.linalg.norm(mixed_residual)
            / max(1.0, float(np.linalg.norm(mixed_rhs)))
        ),
        "normalized_action_symmetry_defect": symmetry_defect,
        "objective_identity_relative_defect": objective_relative_defect,
        "p1_matrix_dimension": state_count,
        "p1_matrix_nnz": int(assembly.p1_stiffness.nnz),
        "p1_relative_residual": float(
            np.linalg.norm(p1_residual)
            / max(
                1.0,
                float(np.linalg.norm(assembly.p1_load @ coefficient)),
            )
        ),
        "eigensolve_seconds": eig_seconds,
    }


def run_pilot(
    spacing: float,
    quadrature_order: int,
    eig_tolerance: float,
    maximum_iterations: int,
    continuum_result_path: Path,
    monitor_cpu: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    continuum = json.loads(
        continuum_result_path.read_text(encoding="ascii")
    )
    threshold = float(
        continuum["cutoff_solution_operator_route"][
            "Ritz_projection_constant_strict_threshold_lower"
        ]
    )
    assembly_started = time.perf_counter()
    assembled = _assemble(spacing, quadrature_order)
    assembly_seconds = time.perf_counter() - assembly_started
    solve = _solve_pilot(
        assembled,
        eig_tolerance,
        maximum_iterations,
        monitor_cpu,
    )
    kappa = float(solve["kappa_h_floating"])
    combined = (
        assembled.data_oscillation_constant_upper
        + assembled.projected_source_norm_factor_upper * kappa
    )

    structural_checks = {
        "edge_incidence_valid": assembled.diagnostics[
            "every_edge_has_one_or_two_incident_triangles"
        ],
        "interior_flux_orientation_valid": assembled.diagnostics[
            "interior_edge_outward_signs_cancel"
        ],
        "positive_source_mass": (
            assembled.diagnostics["minimum_source_mass"] > 0.0
        ),
        "p1_stiffness_symmetric": (
            assembled.diagnostics["p1_stiffness_symmetry_error"]
            <= 1.0e-12
        ),
        "rt_mass_symmetric": (
            assembled.diagnostics["rt_mass_symmetry_error"] <= 1.0e-12
        ),
    }
    diagnostic_passes_threshold = combined < threshold
    return {
        "kind": "neutral-strip-weighted-hypercircle-pilot",
        "model": (
            "Gaussian-weighted Dirichlet operator with conforming P1 "
            "potentials, RT0 equilibrated physical fluxes, and weighted "
            "physical-source norm"
        ),
        "spacing": spacing,
        "quadrature_order": quadrature_order,
        "source_space": {
            "continuum_physical_source": "g=mu*f",
            "projected_physical_source": (
                "g_h=unweighted cell average of g"
            ),
            "corresponding_operator_source": "f_h=g_h/mu",
            "weighted_source_mass": "integral_T mu^-1",
            "projection": (
                "unweighted L2 projection of physical source g onto P0"
            ),
        },
        "hypercircle_bound": {
            "formula": (
                "C_h <= C_data + alpha_source*kappa_h"
            ),
            "data_oscillation_bound": (
                "C_data,T <= diameter(T)/pi * "
                "sqrt(mu_max,T/mu_min,T)"
            ),
            "data_oscillation_constant_upper_floating_geometry": (
                assembled.data_oscillation_constant_upper
            ),
            "projected_source_norm_factor_upper_floating_geometry": (
                assembled.projected_source_norm_factor_upper
            ),
            "kappa_h_floating": kappa,
            "combined_C_h_floating": combined,
            "strict_target_lower": threshold,
            "floating_diagnostic_passes_strict_target": (
                diagnostic_passes_threshold
            ),
            "remaining_kappa_budget_if_combined_to_pass": (
                max(
                    0.0,
                    threshold
                    - assembled.data_oscillation_constant_upper,
                )
                / assembled.projected_source_norm_factor_upper
            ),
        },
        "mesh": {
            "vertex_count": len(assembled.vertices),
            "triangle_count": len(assembled.triangles),
            "state_count": len(assembled.state_vertices),
            "edge_count": assembled.rt_mass.shape[0],
            "mesh_fingerprint_sha256": (
                assembled.mesh_fingerprint_sha256
            ),
            **assembled.diagnostics,
        },
        "linear_algebra": solve,
        "structural_checks": structural_checks,
        "all_floating_pilot_checks_pass": bool(
            all(structural_checks.values())
            and solve["all_floating_linear_algebra_checks_pass"]
        ),
        "certification_flags": {
            "weighted_hypercircle_decomposition_encoded": True,
            "physical_source_P0_projection_decomposition_encoded": True,
            "projected_source_norm_inflation_encoded": True,
            "directed_roundoff_enclosure_complete": False,
            "quadrature_enclosure_complete": False,
            "kappa_h_verified_upper_bound": False,
            "global_weighted_Ritz_constant_certified": False,
            "continuum_spectrum_below_60_captured": False,
        },
        "interpretation": (
            "This is a floating feasibility and implementation pilot. "
            "Passing the target does not certify C_h; failing it rules out "
            "this mesh/formulation unless a sharper valid bound is found."
        ),
        "premises": {
            "continuum_dependency_result": str(
                continuum_result_path
            ).replace("\\", "/"),
            "continuum_dependency_result_sha256": _sha256_file(
                continuum_result_path
            ),
            "literature_method": (
                "Liu and Oishi, SIAM J. Numer. Anal. 51 (2013), "
                "Theorems 3.2-3.3 and Section 3.3, adapted to mu"
            ),
        },
        "below_normal_priority_set": priority_set,
        "assembly_seconds": assembly_seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING)
    parser.add_argument(
        "--quadrature-order",
        type=int,
        default=DEFAULT_QUADRATURE_ORDER,
    )
    parser.add_argument("--eig-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--maximum-iterations", type=int, default=300)
    parser.add_argument(
        "--continuum-result",
        type=Path,
        default=DEFAULT_CONTINUUM_RESULT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_pilot(
        args.spacing,
        args.quadrature_order,
        args.eig_tolerance,
        args.maximum_iterations,
        args.continuum_result,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
