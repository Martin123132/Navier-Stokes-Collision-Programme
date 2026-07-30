#!/usr/bin/env python3
"""Audit the continuum-to-P1 Ritz dependency without promoting missing data."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from decimal import Context, Decimal, localcontext
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
DEFAULT_ASSEMBLY_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
)
DEFAULT_ASSEMBLY_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_checkpoint_v1.npz"
)
DEFAULT_SPECTRUM_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_exact_polygon_indexed_spectrum_transfer_v1.json"
)
DEFAULT_PROJECTED_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_projected_interval_two_block_transfer_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_continuum_ritz_dependency_audit_v1.json"
)

SPACING = 0.06
RETAINED_MODE_COUNT = 240
CONTINUUM_CUTOFF = 60.0
X_HALF_WIDTH = 4.2
STRIP_HALF_WIDTH = 2.1
DECIMAL_CONTEXT = Context(prec=100)


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


def _decimal_down(value: Decimal) -> float:
    candidate = float(value)
    if Decimal.from_float(candidate) > value:
        candidate = float(np.nextafter(candidate, -math.inf))
    return float(np.nextafter(candidate, -math.inf))


def _decimal_up(value: Decimal) -> float:
    candidate = float(value)
    if Decimal.from_float(candidate) < value:
        candidate = float(np.nextafter(candidate, math.inf))
    return float(np.nextafter(candidate, math.inf))


def _cutoff_threshold(
    omitted_eigenvalue_lower: float,
    cutoff: float,
) -> dict[str, float]:
    if omitted_eigenvalue_lower <= cutoff:
        raise ValueError("the discrete omitted-mode lower bound must exceed cutoff")
    with localcontext(DECIMAL_CONTEXT):
        inverse_cutoff = Decimal(1) / Decimal.from_float(cutoff)
        inverse_omitted = (
            Decimal(1) / Decimal.from_float(omitted_eigenvalue_lower)
        )
        resolvent_gap = inverse_cutoff - inverse_omitted
        projection_constant = resolvent_gap.sqrt()
    return {
        "inverse_cutoff_lower": _decimal_down(inverse_cutoff),
        "inverse_cutoff_upper": _decimal_up(inverse_cutoff),
        "inverse_omitted_mode_upper": _decimal_up(inverse_omitted),
        "solution_operator_error_strict_threshold_lower": _decimal_down(
            resolvent_gap
        ),
        "Ritz_projection_constant_strict_threshold_lower": _decimal_down(
            projection_constant
        ),
    }


def _angle_target_rows(resolvent_gap_lower: float) -> list[dict[str, float]]:
    rows = []
    for target in (0.5, 0.25, 0.1, 0.05, 0.01):
        with localcontext(DECIMAL_CONTEXT):
            delta = Decimal.from_float(resolvent_gap_lower) * Decimal.from_float(
                target
            )
            projection_constant = delta.sqrt()
        rows.append(
            {
                "projector_angle_target": target,
                "sufficient_solution_operator_error_strict_upper": (
                    _decimal_down(delta)
                ),
                "sufficient_Ritz_projection_constant_strict_upper": (
                    _decimal_down(projection_constant)
                ),
            }
        )
    return rows


def _li_yau_index_bound(index_one_based: int) -> dict[str, float]:
    mpmath.iv.dps = 80
    pi = mpmath.iv.pi
    area = (
        4
        * mpmath.iv.mpf(str(X_HALF_WIDTH))
        * mpmath.iv.mpf(str(STRIP_HALF_WIDTH))
        - pi
    )
    bound = 2 * pi * index_one_based / area - mpmath.iv.mpf("0.5")
    area_lower = float(np.nextafter(float(area.a), -math.inf))
    area_upper = float(np.nextafter(float(area.b), math.inf))
    bound_lower = float(np.nextafter(float(bound.a), -math.inf))
    bound_upper = float(np.nextafter(float(bound.b), math.inf))
    return {
        "domain_area_lower": area_lower,
        "domain_area_upper": area_upper,
        "continuum_eigenvalue_lower": bound_lower,
        "continuum_eigenvalue_formula_upper": bound_upper,
    }


def _component_count(adjacency: dict[int, set[int]], vertices: set[int]) -> int:
    unseen = set(vertices)
    components = 0
    while unseen:
        components += 1
        start = unseen.pop()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, set()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return components


def _mesh_conformity(checkpoint_path: Path) -> dict[str, Any]:
    boundary = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "continuum_ritz_boundary_mesh",
    )
    assembly = _load_module(
        "neutral_strip_gaussian_weighted_assembly_interval_audit.py",
        "continuum_ritz_assembly_mesh",
    )
    mesh_input, _, _ = boundary._mesh_input(
        SPACING,
        X_HALF_WIDTH,
        STRIP_HALF_WIDTH,
    )
    maximum_area = 0.45 * SPACING**2
    mesh = triangle.triangulate(
        mesh_input,
        f"pYq28a{maximum_area:.17g}Q",
    )
    vertices = np.asarray(mesh["vertices"], dtype=float)
    triangles = np.asarray(mesh["triangles"], dtype=int)
    segments = np.asarray(mesh["segments"], dtype=int)
    markers = np.asarray(mesh["segment_markers"], dtype=int).reshape(-1)

    edge_triangles: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    minimum_determinant_magnitude = math.inf
    for triangle_index, element in enumerate(triangles):
        points = vertices[element]
        first = points[1] - points[0]
        second = points[2] - points[0]
        determinant = float(
            first[0] * second[1] - first[1] * second[0]
        )
        minimum_determinant_magnitude = min(
            minimum_determinant_magnitude,
            abs(determinant),
        )
        for local_index in range(3):
            edge = tuple(
                sorted(
                    (
                        int(element[local_index]),
                        int(element[(local_index + 1) % 3]),
                    )
                )
            )
            edge_triangles[edge].append(triangle_index)

    all_edge_incidences_valid = all(
        len(incident) in (1, 2) for incident in edge_triangles.values()
    )
    boundary_edges = {
        edge for edge, incident in edge_triangles.items() if len(incident) == 1
    }
    segment_edges = {
        tuple(sorted((int(segment[0]), int(segment[1]))))
        for segment in segments
    }
    boundary_segments_complete = boundary_edges == segment_edges

    triangle_adjacency: dict[int, set[int]] = defaultdict(set)
    for incident in edge_triangles.values():
        if len(incident) == 2:
            first, second = incident
            triangle_adjacency[first].add(second)
            triangle_adjacency[second].add(first)
    triangle_component_count = _component_count(
        triangle_adjacency,
        set(range(len(triangles))),
    )

    boundary_adjacency: dict[int, set[int]] = defaultdict(set)
    boundary_vertices: set[int] = set()
    for first, second in segment_edges:
        boundary_vertices.update((first, second))
        boundary_adjacency[first].add(second)
        boundary_adjacency[second].add(first)
    boundary_component_count = _component_count(
        boundary_adjacency,
        boundary_vertices,
    )
    boundary_degrees_are_two = all(
        len(boundary_adjacency[vertex]) == 2 for vertex in boundary_vertices
    )

    inner_marker = int(boundary.INNER_MARKER)
    allowed_markers = {
        int(boundary.INNER_MARKER),
        int(boundary.WALL_MARKER),
        int(boundary.TRUNCATION_MARKER),
    }
    marker_set_valid = set(int(value) for value in markers) == allowed_markers
    inner_segment_count = int(np.count_nonzero(markers == inner_marker))
    state_count = len(vertices) - len(boundary_vertices)
    euler_characteristic = (
        len(vertices) - len(edge_triangles) + len(triangles)
    )

    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        contract = json.loads(str(checkpoint["contract_json"].item()))
    mesh_fingerprint = assembly._mesh_fingerprint(vertices, triangles)
    fingerprint_matches = (
        mesh_fingerprint == contract["mesh_fingerprint_sha256"]
    )

    checks = [
        minimum_determinant_magnitude > 0.0,
        all_edge_incidences_valid,
        boundary_segments_complete,
        triangle_component_count == 1,
        boundary_component_count == 2,
        boundary_degrees_are_two,
        marker_set_valid,
        inner_segment_count == 112,
        state_count == int(contract["state_count"]) == 15211,
        len(triangles) == int(contract["total_triangle_count"]) == 30954,
        euler_characteristic == 0,
        fingerprint_matches,
    ]
    return {
        "all_integer_topology_and_mesh_identity_checks_pass": bool(all(checks)),
        "boundary_component_count": boundary_component_count,
        "boundary_degrees_are_two": boundary_degrees_are_two,
        "boundary_edge_count": len(boundary_edges),
        "boundary_segments_equal_incidence_one_edges": (
            boundary_segments_complete
        ),
        "continuous_P1_state_basis_has_zero_boundary_trace": bool(
            boundary_segments_complete
            and boundary_degrees_are_two
            and state_count == 15211
        ),
        "euler_characteristic": euler_characteristic,
        "inner_boundary_segment_count": inner_segment_count,
        "interior_triangle_component_count": triangle_component_count,
        "marker_set_exact": marker_set_valid,
        "mesh_fingerprint_matches_directed_assembly_checkpoint": (
            fingerprint_matches
        ),
        "mesh_fingerprint_sha256": mesh_fingerprint,
        "minimum_binary_triangle_determinant_magnitude": (
            minimum_determinant_magnitude
        ),
        "state_count": state_count,
        "total_triangle_count": len(triangles),
        "vertex_count": len(vertices),
    }


def audit(
    assembly_result_path: Path,
    assembly_checkpoint_path: Path,
    spectrum_result_path: Path,
    projected_result_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    assembly = json.loads(assembly_result_path.read_text(encoding="ascii"))
    spectrum = json.loads(spectrum_result_path.read_text(encoding="ascii"))
    projected = json.loads(projected_result_path.read_text(encoding="ascii"))

    if not assembly["finite_element_assembly_interval_enclosed"]:
        raise RuntimeError("exact-form assembly premise is not certified")
    if not spectrum[
        "all_exact_polygon_indexed_spectrum_transfer_checks_pass"
    ]:
        raise RuntimeError("exact-polygon spectrum premise is not certified")
    if not projected["exact_polygon_low_projector_certified"]:
        raise RuntimeError("exact-polygon low-projector premise is not certified")

    mesh = _mesh_conformity(assembly_checkpoint_path)
    omitted_lower = float(
        spectrum[
            "exact_polygon_complement_generalized_eigenvalue_lower_bound"
        ]
    )
    retained_upper = float(spectrum["exact_retained_index_239_interval_upper"])
    cutoff = _cutoff_threshold(omitted_lower, CONTINUUM_CUTOFF)
    li_yau = _li_yau_index_bound(RETAINED_MODE_COUNT + 1)
    analytic_index_shortfall = float(
        np.nextafter(
            retained_upper - li_yau["continuum_eigenvalue_lower"],
            math.inf,
        )
    )
    resolvent_gap_lower = cutoff[
        "solution_operator_error_strict_threshold_lower"
    ]

    certification_flags = {
        "stored_polygon_P1_space_conforming_certified": bool(
            mesh["all_integer_topology_and_mesh_identity_checks_pass"]
            and mesh["continuous_P1_state_basis_has_zero_boundary_trace"]
        ),
        "conforming_Rayleigh_eigenvalue_upper_direction_certified": True,
        "stored_FE_complement_lower_substitutable_as_continuum_lower": False,
        "available_analytic_index_gap_closes": False,
        "cutoff_solution_operator_transfer_theorem_encoded": True,
        "weighted_global_Ritz_projection_constant_certified": False,
        "weighted_solution_operator_norm_error_certified": False,
        "continuum_spectrum_below_60_captured_by_240_FE_modes": False,
        "continuum_low_to_FE_projector_angle_certified": False,
        "positive_time_point_source_semigroup_transfer_certified": False,
        "continuum_polygon_conormal_response_certified": False,
        "polygon_to_circle_domain_transfer_certified": False,
        "continuum_return_response_certified": False,
    }

    premises = {
        "assembly_checkpoint": str(assembly_checkpoint_path).replace("\\", "/"),
        "assembly_checkpoint_sha256": _sha256_file(
            assembly_checkpoint_path
        ),
        "assembly_result": str(assembly_result_path).replace("\\", "/"),
        "assembly_result_sha256": _sha256_file(assembly_result_path),
        "projected_result": str(projected_result_path).replace("\\", "/"),
        "projected_result_sha256": _sha256_file(projected_result_path),
        "spectrum_result": str(spectrum_result_path).replace("\\", "/"),
        "spectrum_result_sha256": _sha256_file(spectrum_result_path),
    }

    result: dict[str, Any] = {
        "kind": "neutral-strip-continuum-Ritz-dependency-audit",
        "model": (
            "weighted Dirichlet solution operator on the stored binary "
            "polygon and its exact conforming Gaussian-weighted P1 Galerkin "
            "operator"
        ),
        "spacing": SPACING,
        "retained_mode_count": RETAINED_MODE_COUNT,
        "continuum_spectral_cutoff": CONTINUUM_CUTOFF,
        "below_normal_priority_set": priority_set,
        "elapsed_seconds": time.perf_counter() - started,
        "premise_artifacts": premises,
        "stored_polygon_P1_conformity": mesh,
        "variational_direction": {
            "continuum_to_conforming_FE_relation": (
                "lambda_j(continuum polygon) <= lambda_j(exact P1)"
            ),
            "exact_P1_index_239_upper": retained_upper,
            "exact_P1_index_240_lower": omitted_lower,
            "conforming_upper_bound_direction_only": True,
            "exact_P1_index_240_lower_is_not_a_continuum_lower_bound": True,
            "naive_complement_substitution_rejected": True,
        },
        "available_analytic_index_bound": {
            "index_one_based": RETAINED_MODE_COUNT + 1,
            **li_yau,
            "exact_P1_retained_upper": retained_upper,
            "shortfall_below_exact_P1_retained_upper": (
                analytic_index_shortfall
            ),
            "proves_continuum_index_separation_from_retained_FE_block": False,
            "interpretation": (
                "The bound is a valid continuum lower bound, but it is too "
                "small to separate the continuum index-241 mode from the "
                "retained exact-P1 block."
            ),
        },
        "cutoff_solution_operator_route": {
            **cutoff,
            "angle_target_rows": _angle_target_rows(resolvent_gap_lower),
            "solution_operator_definition": (
                "T solves a(Tf,v)=m(f,v); T_h is its conforming Galerkin "
                "solution operator extended by zero off V_h"
            ),
            "conditional_rank_statement": (
                "If ||T-T_h||_m < d, where d is the reported inverse "
                "cutoff-to-omitted-mode gap, then lambda_241(continuum)>60 "
                "and at most 240 continuum modes lie at or below cutoff 60."
            ),
            "conditional_projector_statement": (
                "For P=1_[1/60,infinity)(T) and Q equal to the first 240 "
                "P1 modes, the separated-spectrum residual theorem gives "
                "||(I-Q)P|| <= ||T-T_h||_m/d."
            ),
            "Ritz_to_solution_operator_statement": (
                "A certified weighted projection inequality "
                "||u-R_hu||_m <= C_h||u-R_hu||_a implies "
                "||T-T_h||_(m->m) <= C_h^2."
            ),
            "first_missing_quantitative_inequality": (
                "Certify the global weighted Ritz projection constant C_h "
                "on the stored nonconvex polygon, preferably by a "
                "hypercircle/equilibrated-flux source-problem bound."
            ),
        },
        "positive_time_point_source_gate": {
            "continuum_point_mass_belongs_to_weighted_L2": False,
            "continuum_point_evaluation_bounded_on_H0_1_in_two_dimensions": (
                False
            ),
            "raw_time_zero_Ritz_source_projection_is_valid": False,
            "heat_smoothing_makes_positive_time_source_state_L2": True,
            "existing_killed_kernel_diagonal_majorant_available": True,
            "remaining_requirement": (
                "Certify a positive-time semigroup or elliptic-reconstruction "
                "error for the continuum point source and the nodal P1 "
                "source; the L2 solution-operator bound alone does not act "
                "on delta_z at time zero."
            ),
        },
        "continuum_conormal_gate": {
            "boundary_conormal_map_bounded_from_weighted_L2_state": False,
            "existing_Rellich_flux_smoothing_constant_available": True,
            "existing_continuum_flux_constant_upper": 3.134170665703,
            "remaining_requirement": (
                "Propagate an equilibrated parabolic residual through the "
                "positive-time Rellich/half-time factorization to bound the "
                "continuum polygon conormal output in L2 of the boundary."
            ),
        },
        "polygon_to_circle_gate": {
            "common_circle_output_measure_geometry_certified": True,
            "common_circle_geometry_compares_domain_semigroups": False,
            "remaining_requirement": (
                "After the continuum polygon response is enclosed, certify "
                "the circle-versus-polygon domain perturbation separately."
            ),
        },
        "certification_flags": certification_flags,
        "strongest_valid_subcertificate": (
            "The exact stored-polygon P1 space is conforming, so its indexed "
            "eigenvalues are continuum upper bounds. The cutoff-resolvent "
            "transfer theorem and its exact numerical thresholds are now "
            "encoded. No continuum spectral, source, conormal, or domain "
            "transfer is promoted without the missing computable constants."
        ),
        "next_required_step": (
            "Construct a weighted equilibrated-flux/hypercircle estimator "
            "for the Galerkin source problem and certify C_h. Start with a "
            "pilot on the existing mesh to test the strict 0.0855711575 "
            "rank threshold before attempting the stronger projector-angle "
            "and positive-time point-source output bounds."
        ),
        "scope": (
            "This is a theorem-dependency and mesh-conformity certificate. "
            "It does not certify continuum Ritz transfer, the positive-time "
            "point-source semigroup error, polygon conormal response, "
            "polygon-to-circle domain transfer, a Navier-Stokes estimate, "
            "or a regularity proof."
        ),
    }

    checks = [
        mesh["all_integer_topology_and_mesh_identity_checks_pass"],
        mesh["continuous_P1_state_basis_has_zero_boundary_trace"],
        omitted_lower > CONTINUUM_CUTOFF,
        resolvent_gap_lower > 0.0073,
        cutoff["Ritz_projection_constant_strict_threshold_lower"] > 0.085,
        li_yau["continuum_eigenvalue_lower"] < retained_upper,
        analytic_index_shortfall > 59.0,
        certification_flags[
            "stored_polygon_P1_space_conforming_certified"
        ],
        certification_flags[
            "conforming_Rayleigh_eigenvalue_upper_direction_certified"
        ],
        not certification_flags[
            "stored_FE_complement_lower_substitutable_as_continuum_lower"
        ],
        not certification_flags["available_analytic_index_gap_closes"],
        certification_flags[
            "cutoff_solution_operator_transfer_theorem_encoded"
        ],
        not certification_flags[
            "weighted_global_Ritz_projection_constant_certified"
        ],
        not certification_flags[
            "positive_time_point_source_semigroup_transfer_certified"
        ],
        not certification_flags[
            "continuum_polygon_conormal_response_certified"
        ],
        not certification_flags["polygon_to_circle_domain_transfer_certified"],
        not certification_flags["continuum_return_response_certified"],
    ]
    result["all_continuum_Ritz_dependency_audit_checks_pass"] = bool(
        all(checks)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assembly-result",
        type=Path,
        default=DEFAULT_ASSEMBLY_RESULT,
    )
    parser.add_argument(
        "--assembly-checkpoint",
        type=Path,
        default=DEFAULT_ASSEMBLY_CHECKPOINT,
    )
    parser.add_argument(
        "--spectrum-result",
        type=Path,
        default=DEFAULT_SPECTRUM_RESULT,
    )
    parser.add_argument(
        "--projected-result",
        type=Path,
        default=DEFAULT_PROJECTED_RESULT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = audit(
        args.assembly_result,
        args.assembly_checkpoint,
        args.spectrum_result,
        args.projected_result,
    )
    if not result["all_continuum_Ritz_dependency_audit_checks_pass"]:
        raise SystemExit("continuum Ritz dependency audit failed")
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
