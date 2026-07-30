import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def test_stored_componentwise_state_region64064_audit_is_certified():
    result = json.loads(
        (
            RESULTS_DIR
            / "neutral_strip_h006_hypercircle_"
            "componentwise_state_region64064_audit_v1.json"
        ).read_text(encoding="ascii")
    )
    assert result["all_current_stage_checks_pass"]
    assert result["status"] == "componentwise_state_region_certified"
    assert result["boundary_recovery"]["maximum_pivots"] == 64040
    assert result["state_region_recovery"]["maximum_pivots"] == 64064
    assert result["state_region_recovery"]["last_certified_pivot"] == 64063
    assert result["state_region_recovery"]["reference_signs"] == {
        "negative": 32500,
        "positive": 31564,
        "zero": 0,
    }
    assert result["next_boundary"][
        "recommended_bounded_pivot_count"
    ] == 64128
    assert not result["next_boundary"]["crosses_new_symbolic_transition"]
    assert not result["next_boundary"]["full_run_admitted"]
    assert result["certification_flags"][
        "standalone_componentwise_64040_inertia_certified"
    ]
    assert result["certification_flags"][
        "standalone_componentwise_64064_inertia_certified"
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
