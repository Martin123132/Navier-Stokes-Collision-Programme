import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def test_stored_state_region64064_audit_validates_expected_obstruction():
    result = json.loads(
        (
            RESULTS_DIR
            / "neutral_strip_h006_hypercircle_"
            "state_region64064_audit_v1.json"
        ).read_text(encoding="ascii")
    )
    assert result["all_current_stage_checks_pass"]
    assert result["status"] == "pass_with_certification_obstruction"
    assert result["certificate_summary"]["maximum_pivots"] == 64064
    assert result["checks"][
        "precision_crosscheck_has_only_expected_closure_failure"
    ]
    assert result["checks"]["leading_factor_is_bitwise_unchanged"]
    assert result["transition_cluster"]["observed_pivots"] == [
        63733,
        63735,
        63900,
        64043,
        64044,
        64049,
        64056,
    ]
    assert not result["route_obstruction"][
        "standalone_64064_inertia_certified"
    ]
    assert result["route_obstruction"]["bound_exceeds_minimum_diagonal"]
    assert not result["next_boundary"]["larger_prefix_run_admitted"]
    assert not result["next_boundary"]["full_run_admitted"]
    assert result["certification_flags"][
        "standalone_63680_inertia_certified"
    ]
    assert not result["certification_flags"][
        "standalone_64064_inertia_certified"
    ]
    assert not result["certification_flags"][
        "full_123816_pivot_inertia_certified"
    ]
    assert not result["certification_flags"][
        "continuum_spectrum_below_60_captured"
    ]
    assert not result["certification_flags"][
        "navier_stokes_regularity_certified"
    ]
