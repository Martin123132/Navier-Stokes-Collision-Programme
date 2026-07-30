import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def test_stored_standalone_residual_regression_passes():
    result = json.loads(
        (
            RESULTS_DIR
            / "neutral_strip_h006_hypercircle_"
            "standalone_residual_regression_v1.json"
        ).read_text(encoding="ascii")
    )
    assert result["all_checks_pass"]
    assert result["state_entry_pilot_admitted"]
    assert len(result["prefix_comparisons"]) == 3
    assert [
        row["maximum_pivots"]
        for row in result["prefix_comparisons"]
    ] == [2304, 32064, 33280]
    assert all(
        all(row["checks"].values())
        for row in result["prefix_comparisons"]
    )
    assert result["certification_flags"][
        "standalone_residual_implementation_validated"
    ]
    assert not result["certification_flags"][
        "full_123816_pivot_inertia_certified"
    ]
