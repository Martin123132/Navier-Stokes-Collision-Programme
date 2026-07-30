#!/usr/bin/env python3
"""Validate the weighted hypercircle threshold-pencil inertia route."""

from __future__ import annotations

import argparse
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
from scipy.linalg import eigvalsh
from scipy.sparse import bmat, csc_matrix, diags
from scipy.sparse.linalg import splu


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HYPERCIRCLE_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_weighted_hypercircle_pilot_v1.json"
)
DEFAULT_ASSEMBLY_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_weighted_hypercircle_sparse_inertia_pilot_v1.json"
)

IDENTITY_SPACING = 0.6
RESOURCE_SPACING = 0.12
QUADRATURE_ORDER = 12
PRODUCTION_BETA = 0.045
SIGN_RELATIVE_TOLERANCE = 2.0e-11


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


def _threshold_pencil(assembly, beta: float) -> csc_matrix:
    edge_count = assembly.rt_mass.shape[0]
    triangle_count = len(assembly.triangles)
    state_count = len(assembly.state_vertices)
    zero_pe = csc_matrix((edge_count, state_count))
    zero_pc = csc_matrix((edge_count, triangle_count))
    zero_zu = csc_matrix((triangle_count, state_count))
    area = diags(assembly.areas, format="csc")
    source_mass = diags(assembly.source_mass, format="csc")
    return bmat(
        [
            [assembly.rt_mass, assembly.divergence.T, zero_pe, zero_pc],
            [
                assembly.divergence,
                None,
                zero_zu,
                area,
            ],
            [
                zero_pe.T,
                zero_zu.T,
                assembly.p1_stiffness,
                -assembly.p1_load,
            ],
            [
                zero_pc.T,
                area,
                -assembly.p1_load.T,
                -(beta**2) * source_mass,
            ],
        ],
        format="csc",
    )


def _direct_hypercircle_form(assembly) -> tuple[np.ndarray, np.ndarray]:
    p_matrix = assembly.rt_mass.toarray()
    divergence = assembly.divergence.toarray()
    stiffness = assembly.p1_stiffness.toarray()
    load = assembly.p1_load.toarray()
    area = np.diag(assembly.areas)
    source_mass = np.diag(assembly.source_mass)
    flux_schur = divergence @ np.linalg.solve(p_matrix, divergence.T)
    flux_energy = area @ np.linalg.solve(flux_schur, area)
    p1_energy = load.T @ np.linalg.solve(stiffness, load)
    quadratic_form = 0.5 * (
        flux_energy - p1_energy + (flux_energy - p1_energy).T
    )
    return quadratic_form, source_mass


def _inertia(values: np.ndarray) -> dict[str, Any]:
    scale = max(1.0, float(np.max(np.abs(values))))
    tolerance = SIGN_RELATIVE_TOLERANCE * scale
    return {
        "negative": int(np.count_nonzero(values < -tolerance)),
        "positive": int(np.count_nonzero(values > tolerance)),
        "zero_or_unresolved": int(
            np.count_nonzero(np.abs(values) <= tolerance)
        ),
        "minimum_absolute_eigenvalue": float(np.min(np.abs(values))),
        "sign_tolerance": tolerance,
    }


def _coarse_identity_rows(assembly) -> tuple[list[dict[str, Any]], float]:
    quadratic_form, source_mass = _direct_hypercircle_form(assembly)
    source_sqrt_inverse = np.diag(
        1.0 / np.sqrt(np.diag(source_mass))
    )
    normalized = (
        source_sqrt_inverse
        @ quadratic_form
        @ source_sqrt_inverse
    )
    normalized = 0.5 * (normalized + normalized.T)
    normalized_values = eigvalsh(normalized, check_finite=True)
    kappa = math.sqrt(max(0.0, float(normalized_values[-1])))
    thresholds = [
        ("production_candidate", PRODUCTION_BETA),
        ("deliberate_failure", 0.8 * kappa),
        ("deliberate_pass", 1.2 * kappa),
    ]
    edge_count = assembly.rt_mass.shape[0]
    state_count = len(assembly.state_vertices)
    triangle_count = len(assembly.triangles)
    rows = []
    for label, beta in thresholds:
        shifted_q = quadratic_form - beta**2 * source_mass
        shifted_q_inertia = _inertia(eigvalsh(shifted_q))
        pencil = _threshold_pencil(assembly, beta).toarray()
        pencil_inertia = _inertia(eigvalsh(pencil))
        predicted = {
            "positive": (
                edge_count
                + state_count
                + shifted_q_inertia["positive"]
            ),
            "negative": (
                triangle_count + shifted_q_inertia["negative"]
            ),
            "zero_or_unresolved": shifted_q_inertia[
                "zero_or_unresolved"
            ],
        }
        counts_match = all(
            pencil_inertia[key] == predicted[key]
            for key in ("positive", "negative", "zero_or_unresolved")
        )
        rows.append(
            {
                "label": label,
                "beta": beta,
                "beta_relative_to_coarse_kappa": beta / kappa,
                "Q_minus_beta_squared_W_inertia": shifted_q_inertia,
                "direct_threshold_pencil_inertia": pencil_inertia,
                "predicted_threshold_pencil_inertia": predicted,
                "Schur_complement_inertia_counts_match": counts_match,
            }
        )
    return rows, kappa


def _sparse_resource_probe(assembly) -> dict[str, Any]:
    pencil = _threshold_pencil(assembly, PRODUCTION_BETA)
    factor_started = time.perf_counter()
    factor = splu(
        pencil,
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True},
    )
    factor_seconds = time.perf_counter() - factor_started
    factor_nnz = int(factor.L.nnz + factor.U.nnz)
    return {
        "spacing": RESOURCE_SPACING,
        "dimension": pencil.shape[0],
        "pencil_nnz": int(pencil.nnz),
        "central_L_plus_U_nnz": factor_nnz,
        "central_factor_fill_ratio": factor_nnz / pencil.nnz,
        "factor_seconds": factor_seconds,
        "row_and_column_permutations_equal": bool(
            np.array_equal(factor.perm_r, factor.perm_c)
        ),
        "minimum_absolute_central_U_diagonal": float(
            np.min(np.abs(factor.U.diagonal()))
        ),
    }


def _full_mesh_inventory(
    hypercircle_result: dict[str, Any],
    hypercircle_module,
) -> dict[str, Any]:
    triangle_count = int(hypercircle_result["mesh"]["triangle_count"])
    edge_count = int(hypercircle_result["mesh"]["edge_count"])
    state_count = int(hypercircle_result["mesh"]["state_count"])
    mixed_nnz = int(
        hypercircle_result["linear_algebra"]["mixed_matrix_nnz"]
    )
    divergence_nnz = 3 * triangle_count
    rt_mass_nnz = mixed_nnz - 2 * divergence_nnz
    p1_stiffness_nnz = int(
        hypercircle_result["linear_algebra"]["p1_matrix_nnz"]
    )
    boundary = hypercircle_module._load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "weighted_hypercircle_inertia_full_topology",
    )
    spacing = float(hypercircle_result["spacing"])
    mesh_input, _, _ = boundary._mesh_input(
        spacing,
        hypercircle_module.X_HALF_WIDTH,
        hypercircle_module.STRIP_HALF_WIDTH,
    )
    mesh = hypercircle_module.triangle.triangulate(
        mesh_input,
        f"pYq28a{0.45 * spacing**2:.17g}Q",
    )
    triangles = np.asarray(mesh["triangles"], dtype=int)
    boundary_vertices = set(
        int(value)
        for value in np.asarray(mesh["segments"], dtype=int).ravel()
    )
    p1_load_nnz = sum(
        int(int(vertex) not in boundary_vertices)
        for element in triangles
        for vertex in element
    )
    topology_counts_match = bool(
        len(triangles) == triangle_count
        and len(mesh["vertices"]) - len(boundary_vertices) == state_count
    )
    pencil_nnz = (
        rt_mass_nnz
        + 2 * divergence_nnz
        + 2 * triangle_count
        + p1_stiffness_nnz
        + 2 * p1_load_nnz
        + triangle_count
    )
    return {
        "dimension": edge_count + state_count + 2 * triangle_count,
        "edge_count": edge_count,
        "state_count": state_count,
        "triangle_count": triangle_count,
        "rt_mass_nnz": rt_mass_nnz,
        "divergence_nnz": divergence_nnz,
        "p1_stiffness_nnz": p1_stiffness_nnz,
        "p1_load_nnz": p1_load_nnz,
        "threshold_pencil_nnz": pencil_nnz,
        "reconstructed_topology_counts_match": topology_counts_match,
        "expected_inertia_if_kappa_below_beta": {
            "positive": edge_count + state_count,
            "negative": 2 * triangle_count,
            "zero": 0,
        },
    }


def run_pilot(
    hypercircle_result_path: Path,
    assembly_result_path: Path,
    identity_spacing: float = IDENTITY_SPACING,
    resource_spacing: float = RESOURCE_SPACING,
    quadrature_order: int = QUADRATURE_ORDER,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    hypercircle = json.loads(
        hypercircle_result_path.read_text(encoding="ascii")
    )
    assembly_result = json.loads(
        assembly_result_path.read_text(encoding="ascii")
    )
    module = _load_module(
        "neutral_strip_weighted_hypercircle_pilot.py",
        "weighted_hypercircle_inertia_dependency",
    )

    identity_assembly = module._assemble(
        identity_spacing,
        quadrature_order,
    )
    coarse_rows, coarse_kappa = _coarse_identity_rows(identity_assembly)
    resource_assembly = module._assemble(
        resource_spacing,
        quadrature_order,
    )
    resource_probe = _sparse_resource_probe(resource_assembly)

    data_constant = float(
        hypercircle["hypercircle_bound"][
            "data_oscillation_constant_upper_floating_geometry"
        ]
    )
    source_factor = float(
        hypercircle["hypercircle_bound"][
            "projected_source_norm_factor_upper_floating_geometry"
        ]
    )
    strict_target = float(
        hypercircle["hypercircle_bound"]["strict_target_lower"]
    )
    production_combined = data_constant + source_factor * PRODUCTION_BETA
    production_headroom = strict_target - production_combined
    full_inventory = _full_mesh_inventory(hypercircle, module)

    identity_checks = [
        row["Schur_complement_inertia_counts_match"]
        for row in coarse_rows
    ]
    candidate_row = next(
        row for row in coarse_rows if row["label"] == "production_candidate"
    )
    checks = {
        "all_coarse_Schur_inertia_rows_match": all(identity_checks),
        "coarse_candidate_correctly_fails": (
            candidate_row["Q_minus_beta_squared_W_inertia"]["positive"] > 0
        ),
        "fixed_beta_preserves_floating_continuum_headroom": (
            production_headroom > 0.0
        ),
        "full_dimension_matches_123816": (
            full_inventory["dimension"] == 123816
        ),
        "full_topology_counts_reconstruct": full_inventory[
            "reconstructed_topology_counts_match"
        ],
        "existing_weighted_P1_stiffness_form_enclosed": bool(
            assembly_result[
                "finite_element_assembly_interval_enclosed"
            ]
        ),
    }
    return {
        "kind": "neutral-strip-weighted-hypercircle-sparse-inertia-pilot",
        "production_candidate": {
            "beta": PRODUCTION_BETA,
            "beta_squared": PRODUCTION_BETA**2,
            "floating_combined_C_h_if_kappa_below_beta": (
                production_combined
            ),
            "floating_continuum_target": strict_target,
            "floating_geometry_headroom": production_headroom,
            "observed_full_mesh_kappa_h": hypercircle[
                "hypercircle_bound"
            ]["kappa_h_floating"],
            "beta_over_observed_full_mesh_kappa": (
                PRODUCTION_BETA
                / float(
                    hypercircle["hypercircle_bound"]["kappa_h_floating"]
                )
            ),
        },
        "coarse_dense_identity": {
            "spacing": identity_spacing,
            "quadrature_order": quadrature_order,
            "edge_count": identity_assembly.rt_mass.shape[0],
            "state_count": len(identity_assembly.state_vertices),
            "triangle_count": len(identity_assembly.triangles),
            "pencil_dimension": (
                identity_assembly.rt_mass.shape[0]
                + len(identity_assembly.state_vertices)
                + 2 * len(identity_assembly.triangles)
            ),
            "coarse_kappa_h": coarse_kappa,
            "rows": coarse_rows,
        },
        "medium_sparse_resource_probe": resource_probe,
        "full_mesh_structural_inventory": full_inventory,
        "interval_entry_inventory": {
            "P1_stiffness_A": (
                "Reuse the completed directed Gaussian-weighted P1 "
                "stiffness contribution intervals."
            ),
            "divergence_N": (
                "Exact integer entries {-1,+1}; topology and orientation "
                "are already checked."
            ),
            "area_D_and_load_B": (
                "New directed determinant enclosures from binary mesh "
                "vertices; B entries are D/3."
            ),
            "RT_mass_P_and_source_mass_W": (
                "New local analytic enclosures for polynomial moments "
                "against exp(+x^2/2)."
            ),
            "beta_squared_W": (
                "Use the exact decimal beta=0.045 and directed products "
                "with the W intervals."
            ),
            "geometry_budget": (
                "Outward-enclose C_data and alpha before confirming that "
                "the fixed-beta headroom remains positive."
            ),
            "factorization": (
                "A verified symmetric-indefinite sparse LDL or equivalent "
                "inertia method is still required; SuperLU is used here "
                "only for a central fill probe."
            ),
        },
        "checks": checks,
        "all_sparse_inertia_pilot_checks_pass": bool(all(checks.values())),
        "certification_flags": {
            "coarse_Schur_complement_inertia_identity_validated": bool(
                all(identity_checks)
            ),
            "full_mesh_threshold_pencil_structure_inventoried": True,
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
            "assembly_result": str(assembly_result_path).replace("\\", "/"),
            "assembly_result_sha256": _sha256_file(assembly_result_path),
        },
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
        "--assembly-result",
        type=Path,
        default=DEFAULT_ASSEMBLY_RESULT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_pilot(
        args.hypercircle_result,
        args.assembly_result,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
