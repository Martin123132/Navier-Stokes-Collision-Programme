import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def test_stored_state_entry63680_audit_is_certified_and_fail_closed():
    result = json.loads(
        (
            RESULTS_DIR
            / "neutral_strip_h006_hypercircle_"
            "state_entry63680_audit_v1.json"
        ).read_text(encoding="ascii")
    )
    assert result["all_current_stage_checks_pass"]
    assert result["certificate_summary"]["maximum_pivots"] == 63680
    assert (
        result["certificate_summary"][
            "minimum_absolute_reference_diagonal_index"
        ]
        == 63629
    )
    assert (
        result["certificate_summary"][
            "minimum_absolute_reference_diagonal_block"
        ]
        == "edge_metric"
    )
    assert result["state_entry_profile"]["first_state_pivot"] == 63644
    assert result["state_entry_profile"]["state_pivot_count"] == 11
    assert result["state_entry_profile"]["state_signs"] == {
        "negative": 0,
        "positive": 11,
        "zero": 0,
    }
    assert result["certification_flags"][
        "standalone_63680_inertia_certified"
    ]
    assert result["certification_flags"]["first_state_entry_certified"]
    assert not result["next_boundary"]["full_run_admitted"]
    assert not result["certification_flags"][
        "full_123816_pivot_inertia_certified"
    ]
    assert not result["certification_flags"][
        "continuum_spectrum_below_60_captured"
    ]
    assert not result["certification_flags"][
        "navier_stokes_regularity_certified"
    ]
