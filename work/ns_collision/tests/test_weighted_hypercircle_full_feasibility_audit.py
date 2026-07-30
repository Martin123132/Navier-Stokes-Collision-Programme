import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def test_stored_full_feasibility_audit_is_fail_closed():
    result = json.loads(
        (
            RESULTS_DIR
            / "neutral_strip_h006_hypercircle_full_feasibility_audit_v1.json"
        ).read_text(encoding="ascii")
    )
    assert result["all_current_stage_checks_pass"]
    assert result["current_certified_boundary"]["completed_pivots"] == 32064
    assert result["full_symbolic_workload"]["dimension"] == 123816
    assert result["full_symbolic_workload"]["first_state_pivot"] == 63644
    assert result["full_symbolic_workload"]["workload_is_strongly_backloaded"]
    assert not result["launch_decision"]["full_directed_LDL_launch_ready"]
    assert not result["launch_decision"][
        "full_congruence_residual_launch_ready"
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
