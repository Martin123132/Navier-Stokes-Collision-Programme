from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "work"
    / "ns_collision"
    / "scripts"
    / "neutral_strip_continuum_ritz_dependency_audit.py"
)
RESULT = (
    ROOT
    / "work"
    / "ns_collision"
    / "results"
    / "neutral_strip_h006_continuum_ritz_dependency_audit_v1.json"
)
SUMMARY = (
    ROOT
    / "work"
    / "ns_collision"
    / "results"
    / "neutral_strip_common_circle_source_summary.json"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "continuum_ritz_dependency_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cutoff_threshold_is_the_inverse_spectral_gap():
    module = _module()
    row = module._cutoff_threshold(107.01775717228844, 60.0)
    expected = 1.0 / 60.0 - 1.0 / 107.01775717228844
    assert row["solution_operator_error_strict_threshold_lower"] <= expected
    assert expected - row[
        "solution_operator_error_strict_threshold_lower"
    ] < 1.0e-15
    assert row["Ritz_projection_constant_strict_threshold_lower"] ** 2 < (
        expected
    )


def test_production_dependency_result_is_fail_closed():
    payload = json.loads(RESULT.read_text(encoding="ascii"))
    assert payload["all_continuum_Ritz_dependency_audit_checks_pass"]
    mesh = payload["stored_polygon_P1_conformity"]
    assert mesh["all_integer_topology_and_mesh_identity_checks_pass"]
    assert mesh["continuous_P1_state_basis_has_zero_boundary_trace"]
    assert mesh["boundary_component_count"] == 2
    assert mesh["euler_characteristic"] == 0
    assert mesh["state_count"] == 15211
    assert mesh["total_triangle_count"] == 30954

    direction = payload["variational_direction"]
    assert direction["conforming_upper_bound_direction_only"]
    assert direction[
        "exact_P1_index_240_lower_is_not_a_continuum_lower_bound"
    ]
    assert direction["naive_complement_substitution_rejected"]

    route = payload["cutoff_solution_operator_route"]
    assert route["solution_operator_error_strict_threshold_lower"] > 0.0073
    assert (
        route["Ritz_projection_constant_strict_threshold_lower"] > 0.085
    )
    assert len(route["angle_target_rows"]) == 5

    flags = payload["certification_flags"]
    assert flags["stored_polygon_P1_space_conforming_certified"]
    assert flags["cutoff_solution_operator_transfer_theorem_encoded"]
    assert not flags[
        "stored_FE_complement_lower_substitutable_as_continuum_lower"
    ]
    assert not flags["weighted_global_Ritz_projection_constant_certified"]
    assert not flags[
        "positive_time_point_source_semigroup_transfer_certified"
    ]
    assert not flags["continuum_polygon_conormal_response_certified"]
    assert not flags["polygon_to_circle_domain_transfer_certified"]
    assert not flags["continuum_return_response_certified"]

    for key, expected_hash in payload["premise_artifacts"].items():
        if not key.endswith("_sha256"):
            continue
        path_key = key.removesuffix("_sha256")
        assert _sha256(ROOT / payload["premise_artifacts"][path_key]) == (
            expected_hash
        )

    summary = json.loads(SUMMARY.read_text(encoding="ascii"))
    summary_row = summary["h006_continuum_Ritz_dependency_audit"]
    assert summary_row["result_sha256"] == _sha256(RESULT)
    assert summary_row["stored_polygon_P1_space_conforming_certified"]
    assert not summary_row[
        "weighted_global_Ritz_projection_constant_certified"
    ]
    assert not summary["certification_flags"][
        "continuum_Ritz_projector_error_certified"
    ]
