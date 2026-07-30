import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def test_stored_transition33280_audit_is_two_route_and_fail_closed():
    result = json.loads(
        (
            RESULTS_DIR
            / "neutral_strip_h006_hypercircle_transition33280_audit_v1.json"
        ).read_text(encoding="ascii")
    )
    assert result["all_current_stage_checks_pass"]
    assert result["current_certified_prefix"]["completed_pivots"] == 33280
    assert result["current_certified_prefix"]["negative"] == 31971
    assert result["current_certified_prefix"]["positive"] == 1309
    assert result["new_segment_32064_33279"]["pivot_count"] == 1216
    assert result["delicate_pivot"]["index"] == 32849
    assert result["delicate_pivot"]["input_diagonal_is_exact_zero"]
    assert result["fill_transition_33224_33279"]["transition_pivot"] == 33224
    assert result["next_boundary"]["first_state_pivot"] == 63644
    assert result["next_boundary"][
        "residual_only_mode_currently_requires_implementation"
    ]
    assert result["certification_flags"][
        "bounded_33280_inertia_certified_by_two_routes"
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
