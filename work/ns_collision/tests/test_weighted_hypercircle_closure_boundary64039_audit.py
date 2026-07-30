import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def test_stored_closure_boundary64039_audit_is_minimal_and_fail_closed():
    result = json.loads(
        (
            RESULTS_DIR
            / "neutral_strip_h006_hypercircle_"
            "closure_boundary64039_audit_v1.json"
        ).read_text(encoding="ascii")
    )
    assert result["all_current_stage_checks_pass"]
    assert result["status"] == "pass_with_minimal_certificate_obstruction"
    assert result["boundary"]["last_certified_prefix_pivots"] == 64039
    assert result["boundary"]["last_certified_pivot"] == 64038
    assert result["boundary"]["first_nonclosing_prefix_pivots"] == 64040
    assert result["boundary"]["first_nonclosing_added_pivot"] == 64039
    assert result["checks"]["bisection_ratios_are_nondecreasing"]
    assert result["checks"][
        "passing_factor_is_bitwise_unchanged_leading_block"
    ]
    assert result["checks"][
        "failing_crosscheck_has_only_expected_closure_failure"
    ]
    assert result["added_pivot_profile"]["original_index"] == 21
    assert result["added_pivot_profile"]["block"] == "edge_metric"
    assert result["added_pivot_profile"]["strict_lower_row_nnz"] == 9
    assert result["certification_flags"][
        "standalone_64039_inertia_certified"
    ]
    assert not result["certification_flags"][
        "standalone_64040_inertia_certified"
    ]
    assert not result["repair_gate"]["larger_prefix_run_admitted"]
    assert not result["certification_flags"][
        "full_123816_pivot_inertia_certified"
    ]
    assert not result["certification_flags"][
        "continuum_spectrum_below_60_captured"
    ]
    assert not result["certification_flags"][
        "navier_stokes_regularity_certified"
    ]
