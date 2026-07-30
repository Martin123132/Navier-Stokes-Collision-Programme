import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def test_stored_componentwise_growth64128_audit_is_certified():
    result = json.loads(
        (
            RESULTS_DIR
            / "neutral_strip_h006_hypercircle_"
            "componentwise_growth64128_audit_v1.json"
        ).read_text(encoding="ascii")
    )
    assert result["all_current_stage_checks_pass"]
    assert result["status"] == "componentwise_prefix64128_certified"
    assert result["extension"]["prior_maximum_pivots"] == 64064
    assert result["extension"]["target_maximum_pivots"] == 64128
    assert result["extension"]["last_certified_pivot"] == 64127
    assert result["extension"]["added_reference_signs"] == {
        "negative": 64,
        "positive": 0,
        "zero": 0,
    }
    assert result["extension"]["reference_signs"] == {
        "negative": 32564,
        "positive": 31564,
        "zero": 0,
    }
    assert (
        result["certificate"][
            "componentwise_bound_growth_from_64064"
        ]
        == "1"
    )
    assert result["next_boundary"][
        "recommended_bounded_pivot_count"
    ] == 64256
    assert not result["next_boundary"]["crosses_new_symbolic_transition"]
    assert not result["next_boundary"]["full_run_admitted"]
    assert result["certification_flags"][
        "standalone_componentwise_64128_inertia_certified"
    ]
    assert not result["certification_flags"][
        "standalone_componentwise_64256_inertia_certified"
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
