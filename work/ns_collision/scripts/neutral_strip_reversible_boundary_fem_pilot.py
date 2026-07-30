"""Build a reversible body-fitted strip generator and its return-flux pilot."""

from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import triangle
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply, splu


INNER_MARKER = 1
WALL_MARKER = 2
TRUNCATION_MARKER = 3
STRIP_HALF_WIDTH = 2.1
X_HALF_WIDTH = 4.2
ENTRY_RADIUS = 2.0


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _multiple_of_eight_ceiling(value: float) -> int:
    return 8 * int(math.ceil(value / 8.0))


def _outer_boundary(
    spacing: float,
    x_half_width: float,
    strip_half_width: float,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]], list[int]]:
    corners = (
        (-x_half_width, -strip_half_width),
        (x_half_width, -strip_half_width),
        (x_half_width, strip_half_width),
        (-x_half_width, strip_half_width),
    )
    vertices: list[tuple[float, float]] = []
    for start, end in zip(corners, corners[1:] + corners[:1]):
        interval_count = int(math.ceil(math.dist(start, end) / spacing))
        for index in range(interval_count):
            fraction = index / interval_count
            vertices.append(
                (
                    start[0] * (1.0 - fraction) + end[0] * fraction,
                    start[1] * (1.0 - fraction) + end[1] * fraction,
                )
            )

    segments = []
    markers = []
    for index in range(len(vertices)):
        next_index = (index + 1) % len(vertices)
        segments.append((index, next_index))
        midpoint_y = 0.5 * (
            vertices[index][1] + vertices[next_index][1]
        )
        if abs(abs(midpoint_y) - strip_half_width) < 1.0e-10:
            markers.append(WALL_MARKER)
        else:
            markers.append(TRUNCATION_MARKER)
    return vertices, segments, markers


def _mesh_input(
    spacing: float,
    x_half_width: float,
    strip_half_width: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    vertices, segments, segment_markers = _outer_boundary(
        spacing, x_half_width, strip_half_width
    )

    circle_count = _multiple_of_eight_ceiling(2.0 * math.pi / spacing)
    inner_coordinates = []
    inner_offset = len(vertices)
    for index in range(circle_count):
        angle = 2.0 * math.pi * index / circle_count
        coordinate = (math.cos(angle), math.sin(angle))
        vertices.append(coordinate)
        inner_coordinates.append(coordinate)
    for index in range(circle_count):
        segments.append(
            (
                inner_offset + index,
                inner_offset + (index + 1) % circle_count,
            )
        )
        segment_markers.append(INNER_MARKER)

    entry_coordinates = []
    for index in range(circle_count):
        angle = 2.0 * math.pi * index / circle_count
        coordinate = (
            ENTRY_RADIUS * math.cos(angle),
            ENTRY_RADIUS * math.sin(angle),
        )
        vertices.append(coordinate)
        entry_coordinates.append(coordinate)

    mesh_input = {
        "vertices": np.asarray(vertices, dtype=float),
        "segments": np.asarray(segments, dtype=int),
        "segment_markers": np.asarray(segment_markers, dtype=int)[:, None],
        "holes": np.asarray([[0.0, 0.0]], dtype=float),
    }
    return (
        mesh_input,
        np.asarray(inner_coordinates),
        np.asarray(entry_coordinates),
    )


def _triangle_area(points: np.ndarray) -> float:
    edge_one = points[1] - points[0]
    edge_two = points[2] - points[0]
    return 0.5 * abs(
        edge_one[0] * edge_two[1] - edge_one[1] * edge_two[0]
    )


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _matched_vertex_indices(
    vertices: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    matches = []
    for coordinate in coordinates:
        distances = np.linalg.norm(vertices - coordinate, axis=1)
        index = int(np.argmin(distances))
        if distances[index] > 1.0e-12:
            raise RuntimeError("Triangle did not preserve an input vertex")
        matches.append(index)
    return np.asarray(matches, dtype=int)


def _build_mesh(
    spacing: float,
    rho: float = 0.0,
    x_half_width: float = X_HALF_WIDTH,
    strip_half_width: float = STRIP_HALF_WIDTH,
) -> dict[str, object]:
    mesh_input, inner_coordinates, entry_coordinates = _mesh_input(
        spacing, x_half_width, strip_half_width
    )
    maximum_area = 0.45 * spacing**2
    mesh = triangle.triangulate(
        mesh_input,
        f"pYq28a{maximum_area:.17g}Q",
    )
    vertices = np.asarray(mesh["vertices"], dtype=float)
    triangles = np.asarray(mesh["triangles"], dtype=int)
    boundary_segments = np.asarray(mesh["segments"], dtype=int)
    boundary_segment_markers = np.asarray(
        mesh["segment_markers"], dtype=int
    ).reshape(-1)

    marker_sets: list[set[int]] = [set() for _ in range(len(vertices))]
    for segment, marker in zip(
        boundary_segments, boundary_segment_markers
    ):
        marker_sets[int(segment[0])].add(int(marker))
        marker_sets[int(segment[1])].add(int(marker))

    inner_vertices = _matched_vertex_indices(vertices, inner_coordinates)
    entry_vertices = _matched_vertex_indices(vertices, entry_coordinates)
    inner_set = set(int(index) for index in inner_vertices)
    wall_set = {
        index
        for index, markers in enumerate(marker_sets)
        if WALL_MARKER in markers and index not in inner_set
    }
    truncation_set = {
        index
        for index, markers in enumerate(marker_sets)
        if TRUNCATION_MARKER in markers
        and index not in inner_set
        and index not in wall_set
    }
    killed_set = inner_set | wall_set | truncation_set
    state_vertices = np.asarray(
        [index for index in range(len(vertices)) if index not in killed_set],
        dtype=int,
    )
    state_lookup = {
        int(vertex): state for state, vertex in enumerate(state_vertices)
    }
    if not all(int(vertex) in state_lookup for vertex in entry_vertices):
        raise RuntimeError("an exact r=2 entry vertex was classified as killed")

    geometric_edge_weights: defaultdict[tuple[int, int], float] = (
        defaultdict(float)
    )
    lumped_mass = np.zeros(len(vertices))
    triangle_areas = []
    for element in triangles:
        points = vertices[element]
        area = _triangle_area(points)
        if area <= 0.0:
            raise RuntimeError("degenerate triangle")
        triangle_areas.append(area)
        centroid = np.mean(points, axis=0)
        invariant_weight = math.exp(
            -0.5 * (centroid[0] ** 2 + rho * centroid[1] ** 2)
        )
        lumped_mass[element] += invariant_weight * area / 3.0
        for opposite in range(3):
            first = int(element[(opposite + 1) % 3])
            second = int(element[(opposite + 2) % 3])
            origin = int(element[opposite])
            first_vector = vertices[first] - vertices[origin]
            second_vector = vertices[second] - vertices[origin]
            cotangent = float(
                np.dot(first_vector, second_vector)
                / abs(_cross(first_vector, second_vector))
            )
            edge = tuple(sorted((first, second)))
            geometric_edge_weights[edge] += 0.5 * cotangent

    state_mass = lumped_mass[state_vertices]
    if float(np.min(state_mass)) <= 0.0:
        raise RuntimeError("nonpositive lumped state mass")

    inner_order = sorted(
        inner_set,
        key=lambda index: math.atan2(
            vertices[index, 1], vertices[index, 0]
        )
        % (2.0 * math.pi),
    )
    inner_column = {
        vertex: column for column, vertex in enumerate(inner_order)
    }
    inner_angles = np.asarray(
        [
            math.atan2(vertices[index, 1], vertices[index, 0])
            % (2.0 * math.pi)
            for index in inner_order
        ]
    )
    angle_gaps = np.diff(
        np.concatenate((inner_angles, inner_angles[:1] + 2.0 * math.pi))
    )
    inner_dual_arcs = 0.5 * (angle_gaps + np.roll(angle_gaps, 1))

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    diagonal_rates = np.zeros(len(state_vertices))
    inner_rows: list[int] = []
    inner_columns: list[int] = []
    inner_values: list[float] = []
    wall_rates = np.zeros(len(state_vertices))
    truncation_rates = np.zeros(len(state_vertices))
    retained_conductances = []
    clipped_roundoff_edges = 0

    for (first, second), geometric_weight in geometric_edge_weights.items():
        first_is_state = first in state_lookup
        second_is_state = second in state_lookup
        if not first_is_state and not second_is_state:
            continue
        midpoint_x = 0.5 * (vertices[first, 0] + vertices[second, 0])
        midpoint_y = 0.5 * (vertices[first, 1] + vertices[second, 1])
        conductance = geometric_weight * math.exp(
            -0.5 * (midpoint_x**2 + rho * midpoint_y**2)
        )
        if conductance < -1.0e-12:
            raise RuntimeError(
                "negative state conductance; mesh is not Markov-positive"
            )
        if conductance <= 1.0e-12:
            clipped_roundoff_edges += 1
            continue
        retained_conductances.append(conductance)

        if first_is_state and second_is_state:
            first_state = state_lookup[first]
            second_state = state_lookup[second]
            first_rate = conductance / state_mass[first_state]
            second_rate = conductance / state_mass[second_state]
            row_indices.extend((first_state, second_state))
            column_indices.extend((second_state, first_state))
            values.extend((first_rate, second_rate))
            diagonal_rates[first_state] += first_rate
            diagonal_rates[second_state] += second_rate
            continue

        state_vertex = first if first_is_state else second
        boundary_vertex = second if first_is_state else first
        state_index = state_lookup[state_vertex]
        rate = conductance / state_mass[state_index]
        diagonal_rates[state_index] += rate
        if boundary_vertex in inner_set:
            inner_rows.append(state_index)
            inner_columns.append(inner_column[boundary_vertex])
            inner_values.append(rate)
        elif boundary_vertex in wall_set:
            wall_rates[state_index] += rate
        elif boundary_vertex in truncation_set:
            truncation_rates[state_index] += rate
        else:
            raise RuntimeError("unclassified killed boundary vertex")

    row_indices.extend(range(len(state_vertices)))
    column_indices.extend(range(len(state_vertices)))
    values.extend((-diagonal_rates).tolist())
    generator = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(state_vertices), len(state_vertices)),
    ).tocsc()
    inner_rate_matrix = coo_matrix(
        (inner_values, (inner_rows, inner_columns)),
        shape=(len(state_vertices), len(inner_order)),
    ).tocsc()
    inner_rates = np.asarray(inner_rate_matrix.sum(axis=1)).reshape(-1)

    entry_states = np.asarray(
        [state_lookup[int(vertex)] for vertex in entry_vertices], dtype=int
    )
    entry_angles = np.asarray(
        [
            math.atan2(vertices[vertex, 1], vertices[vertex, 0])
            % (2.0 * math.pi)
            for vertex in entry_vertices
        ]
    )
    entry_sort = np.argsort(entry_angles)
    entry_states = entry_states[entry_sort]
    entry_angles = entry_angles[entry_sort]

    return {
        "spacing": spacing,
        "rho": rho,
        "x_half_width": x_half_width,
        "strip_half_width": strip_half_width,
        "vertices": vertices,
        "triangles": triangles,
        "state_vertices": state_vertices,
        "state_mass": state_mass,
        "generator": generator,
        "inner_rate_matrix": inner_rate_matrix,
        "inner_rates": inner_rates,
        "wall_rates": wall_rates,
        "truncation_rates": truncation_rates,
        "inner_dual_arcs": inner_dual_arcs,
        "inner_angles": inner_angles,
        "entry_states": entry_states,
        "entry_angles": entry_angles,
        "maximum_triangle_area": float(max(triangle_areas)),
        "minimum_retained_conductance": float(min(retained_conductances)),
        "clipped_roundoff_edge_count": clipped_roundoff_edges,
        "inner_boundary_vertex_count": len(inner_order),
        "wall_boundary_vertex_count": len(wall_set),
        "truncation_boundary_vertex_count": len(truncation_set),
    }


def _structural_row(grid: dict[str, object]) -> dict[str, object]:
    generator = grid["generator"]
    state_mass = np.asarray(grid["state_mass"])
    total_boundary_rates = (
        np.asarray(grid["inner_rates"])
        + np.asarray(grid["wall_rates"])
        + np.asarray(grid["truncation_rates"])
    )
    probabilities = splu(-generator).solve(
        np.column_stack(
            (
                grid["inner_rates"],
                grid["wall_rates"],
                grid["truncation_rates"],
            )
        )
    )
    generator_rows = generator.tocsr()
    maximum_balance_error = 0.0
    for state in range(generator_rows.shape[0]):
        for pointer in range(
            generator_rows.indptr[state], generator_rows.indptr[state + 1]
        ):
            target = int(generator_rows.indices[pointer])
            if state == target:
                continue
            forward = state_mass[state] * generator_rows.data[pointer]
            reverse = state_mass[target] * float(generator_rows[target, state])
            scale = max(abs(forward), abs(reverse), 1.0e-300)
            maximum_balance_error = max(
                maximum_balance_error, abs(forward - reverse) / scale
            )

    algebraic_partition_error = float(
        np.max(np.abs(generator @ np.ones(generator.shape[0]) + total_boundary_rates))
    )
    probability_partition_error = float(
        np.max(np.abs(np.sum(probabilities, axis=1) - 1.0))
    )
    entry_probabilities = probabilities[grid["entry_states"], :]
    off_diagonal = generator_rows.copy()
    off_diagonal.setdiag(0.0)
    off_diagonal.eliminate_zeros()
    return {
        "spacing": grid["spacing"],
        "vertex_count": len(grid["vertices"]),
        "triangle_count": len(grid["triangles"]),
        "state_count": generator.shape[0],
        "inner_boundary_vertex_count": grid["inner_boundary_vertex_count"],
        "entry_vertex_count": len(grid["entry_states"]),
        "maximum_triangle_area": grid["maximum_triangle_area"],
        "minimum_off_diagonal_rate": float(np.min(off_diagonal.data)),
        "minimum_retained_conductance": grid[
            "minimum_retained_conductance"
        ],
        "clipped_roundoff_edge_count": grid["clipped_roundoff_edge_count"],
        "maximum_relative_detailed_balance_error": maximum_balance_error,
        "algebraic_probability_partition_error": algebraic_partition_error,
        "resolvent_probability_partition_error": probability_partition_error,
        "inner_dual_arc_sum": float(np.sum(grid["inner_dual_arcs"])),
        "inner_dual_arc_sum_error": abs(
            float(np.sum(grid["inner_dual_arcs"])) - 2.0 * math.pi
        ),
        "minimum_inner_dual_arc": float(
            np.min(grid["inner_dual_arcs"])
        ),
        "maximum_inner_dual_arc": float(
            np.max(grid["inner_dual_arcs"])
        ),
        "minimum_entry_probability_partition": float(
            np.min(np.sum(entry_probabilities, axis=1))
        ),
        "maximum_entry_return_probability": float(
            np.max(entry_probabilities[:, 0])
        ),
        "maximum_entry_wall_probability": float(
            np.max(entry_probabilities[:, 1])
        ),
        "maximum_entry_truncation_probability": float(
            np.max(entry_probabilities[:, 2])
        ),
    }


def _density_row(
    grid: dict[str, object], return_density
) -> dict[str, object]:
    state_count = grid["generator"].shape[0]
    entry_states = np.asarray(grid["entry_states"])
    state = np.zeros((state_count, len(entry_states)))
    state[entry_states, np.arange(len(entry_states))] = 1.0
    generator_transpose = grid["generator"].transpose().tocsc()
    times = []
    raw_l2_rows = []
    scalar_density_rows = []
    current_time = 0.0
    segments = (
        (0.05, 26),
        (0.2, 31),
        (1.0, 41),
        (4.0, 31),
        (12.0, 33),
    )
    for segment_end, point_count in segments:
        duration = segment_end - current_time
        trajectory = expm_multiply(
            generator_transpose,
            state,
            start=0.0,
            stop=duration,
            num=point_count,
            endpoint=True,
        )
        for local_index in range(1, point_count):
            time = current_time + duration * local_index / (point_count - 1)
            snapshot = trajectory[local_index]
            boundary_flux = np.asarray(
                grid["inner_rate_matrix"].transpose() @ snapshot
            ).T
            transverse_l2 = np.sqrt(
                np.sum(
                    boundary_flux**2 / grid["inner_dual_arcs"][None, :],
                    axis=1,
                )
            )
            transverse_mass = np.sum(boundary_flux, axis=1)
            patch_mass, axial_l2 = return_density._axial_factors(
                time, grid["rho"]
            )
            deformation = math.exp(time)
            times.append(time)
            raw_l2_rows.append(deformation * axial_l2 * transverse_l2)
            scalar_density_rows.append(
                deformation * patch_mass * transverse_mass
            )
        state = trajectory[-1]
        current_time = segment_end

    time_array = np.asarray(times)
    raw_l2 = np.asarray(raw_l2_rows)
    scalar_density = np.asarray(scalar_density_rows)
    branch_mass = np.trapezoid(scalar_density, time_array, axis=0)
    angle_rows = []
    for angle_index, angle in enumerate(grid["entry_angles"]):
        factor = return_density._sampled_interval_factor(
            time_array, raw_l2[:, angle_index]
        )
        response = math.sqrt(
            float(branch_mass[angle_index])
            * return_density.TRACE_L4_FORM_CONSTANT
            * factor["sampled_stressed_factor"]
        )
        angle_rows.append(
            {
                "angle": float(angle),
                "weighted_return_scalar_gain": float(
                    branch_mass[angle_index]
                ),
                "raw_interval_factor": factor["sampled_stressed_factor"],
                "raw_trace_response_at_alpha_zero": response,
                "peak_raw_spatial_L2_density": float(
                    np.max(raw_l2[:, angle_index])
                ),
                "peak_time": float(
                    time_array[int(np.argmax(raw_l2[:, angle_index]))]
                ),
            }
        )
    worst = max(
        angle_rows,
        key=lambda row: row["raw_trace_response_at_alpha_zero"],
    )
    return {
        "spacing": grid["spacing"],
        "entry_angle_count": len(grid["entry_angles"]),
        "time_sample_count": len(time_array),
        "maximum_time": float(time_array[-1]),
        "maximum_terminal_state": float(np.max(state)),
        "maximum_raw_interval_factor": max(
            row["raw_interval_factor"] for row in angle_rows
        ),
        "maximum_raw_trace_response_at_alpha_zero": worst[
            "raw_trace_response_at_alpha_zero"
        ],
        "worst_entry_angle": worst["angle"],
        "worst_weighted_return_scalar_gain": worst[
            "weighted_return_scalar_gain"
        ],
        "angle_rows": angle_rows,
    }


def audit(
    spacings: tuple[float, ...] = (0.16, 0.12, 0.09),
    run_density: bool = True,
) -> dict[str, object]:
    return_density = _load_module(
        "neutral_strip_return_density_pilot.py",
        "return_density_for_reversible_fem",
    )
    structural_rows = []
    density_rows = []
    for spacing in spacings:
        grid = _build_mesh(spacing)
        structural_rows.append(_structural_row(grid))
        if run_density:
            density_rows.append(_density_row(grid, return_density))

    response_change = None
    if len(density_rows) >= 2:
        response_change = abs(
            density_rows[-1]["maximum_raw_trace_response_at_alpha_zero"]
            - density_rows[-2][
                "maximum_raw_trace_response_at_alpha_zero"
            ]
        )
    result = {
        "model": (
            "rho=0 weighted lumped P1 FEM on a constrained-Delaunay "
            "polygonal neutral strip"
        ),
        "triangle_version": getattr(triangle, "__version__", "unknown"),
        "generator_form": (
            "q_ij=c_ij/m_i with symmetric positive c_ij and lumped "
            "invariant masses m_i"
        ),
        "outer_x_boundary_policy": (
            "absorbing third truncation branch; no untracked reflection"
        ),
        "inner_boundary_density_policy": (
            "flux into each Dirichlet circle vertex divided by its disjoint "
            "dual true-arclength face"
        ),
        "structural_rows": structural_rows,
        "density_rows": density_rows,
        "finest_response_change": response_change,
        "all_retained_off_diagonal_rates_positive": all(
            row["minimum_off_diagonal_rate"] > 0.0
            for row in structural_rows
        ),
        "exact_discrete_reversibility_verified": all(
            row["maximum_relative_detailed_balance_error"] < 1.0e-12
            for row in structural_rows
        ),
        "probability_partition_verified": all(
            row["resolvent_probability_partition_error"] < 1.0e-10
            for row in structural_rows
        ),
        "inner_boundary_faces_partition_true_circle": all(
            row["inner_dual_arc_sum_error"] < 1.0e-12
            for row in structural_rows
        ),
        "exact_r2_entry_nodes_inserted": all(
            row["entry_vertex_count"] == row["inner_boundary_vertex_count"]
            for row in structural_rows
        ),
        "physical_boundary_L2_flux_computed": run_density,
        "coupled_mesh_response_pilot_stabilized": (
            response_change is not None and response_change < 0.002
        ),
        "coupled_mesh_response_converged": False,
        "time_window_and_tail_enclosed": False,
        "x_truncation_removed": False,
        "continuum_return_density_certified": False,
        "scope_guard": (
            "This replaces the atomic fitted-edge histogram by a positive, "
            "reversible, conservative polygonal FEM law with physical "
            "boundary faces. It remains a floating-point convergence pilot: "
            "polygonal-domain error, mass/stiffness consistency, time-window "
            "maxima, tail error, and x-truncation are not enclosed."
        ),
        "next_gate": (
            "stress coupled mesh refinement and x-width, then enclose the "
            "finite-dimensional time/tail response before attempting a "
            "continuum boundary-flux theorem"
        ),
    }
    checks = (
        result["all_retained_off_diagonal_rates_positive"],
        result["exact_discrete_reversibility_verified"],
        result["probability_partition_verified"],
        result["inner_boundary_faces_partition_true_circle"],
        result["exact_r2_entry_nodes_inserted"],
        not result["coupled_mesh_response_converged"],
        not result["time_window_and_tail_enclosed"],
        not result["x_truncation_removed"],
        not result["continuum_return_density_certified"],
    )
    if run_density:
        checks += (
            result["coupled_mesh_response_pilot_stabilized"],
            all(
                row["maximum_raw_trace_response_at_alpha_zero"] > 0.0
                for row in density_rows
            ),
            max(row["maximum_terminal_state"] for row in density_rows)
            < 1.0e-8,
        )
    result["all_positive_reversible_boundary_fem_checks_pass"] = all(checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
