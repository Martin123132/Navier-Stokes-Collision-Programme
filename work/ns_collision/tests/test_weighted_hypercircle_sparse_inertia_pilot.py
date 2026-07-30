import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_weighted_hypercircle_sparse_inertia_pilot.py"
)
SPEC = importlib.util.spec_from_file_location(
    "weighted_hypercircle_sparse_inertia_pilot",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_threshold_pencil_identity_on_small_mesh():
    result = MODULE.run_pilot(
        MODULE.DEFAULT_HYPERCIRCLE_RESULT,
        MODULE.DEFAULT_ASSEMBLY_RESULT,
        identity_spacing=1.2,
        resource_spacing=0.6,
        quadrature_order=6,
    )
    assert result["all_sparse_inertia_pilot_checks_pass"]
    assert result["certification_flags"][
        "coarse_Schur_complement_inertia_identity_validated"
    ]
    assert not result["certification_flags"][
        "full_mesh_threshold_inertia_certified"
    ]
    labels = {
        row["label"]: row
        for row in result["coarse_dense_identity"]["rows"]
    }
    assert labels["deliberate_failure"][
        "Q_minus_beta_squared_W_inertia"
    ]["positive"] > 0
    assert labels["deliberate_pass"][
        "Q_minus_beta_squared_W_inertia"
    ]["positive"] == 0


def test_stored_sparse_inertia_pilot_is_fail_closed():
    result_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "neutral_strip_weighted_hypercircle_sparse_inertia_pilot_v1.json"
    )
    result = MODULE.json.loads(result_path.read_text(encoding="ascii"))
    assert result["all_sparse_inertia_pilot_checks_pass"]
    assert result["production_candidate"]["beta"] == 0.045
    assert result["full_mesh_structural_inventory"]["dimension"] == 123816
    assert (
        result["full_mesh_structural_inventory"][
            "threshold_pencil_nnz"
        ]
        == 798384
    )
    assert result["full_mesh_structural_inventory"][
        "reconstructed_topology_counts_match"
    ]
    assert not result["certification_flags"][
        "full_mesh_threshold_inertia_certified"
    ]
    assert not result["certification_flags"][
        "continuum_spectrum_below_60_captured"
    ]
