import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_positive_exponential_complete_assembly.py"
)
SPEC = importlib.util.spec_from_file_location(
    "positive_exponential_complete_assembly",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def test_atomic_checkpoint_round_trip_and_resume(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    metadata = tmp_path / "checkpoint.json"
    matrices = tmp_path / "matrices.npz"
    first = MODULE.run_assembly(
        MODULE.DEFAULT_PILOT_RESULT,
        checkpoint,
        metadata,
        matrices,
        checkpoint_interval=2,
        maximum_chunks=1,
    )
    assert first["status"] == "parked_at_requested_chunk_limit"
    assert first["next_triangle"] == 2
    assert first["all_current_stage_checks_pass"]
    assert not first["certification_flags"][
        "complete_mesh_RT0_P0_matrix_entries_enclosed"
    ]

    second = MODULE.run_assembly(
        MODULE.DEFAULT_PILOT_RESULT,
        checkpoint,
        metadata,
        matrices,
        checkpoint_interval=3,
        maximum_chunks=1,
    )
    assert second["resumed_from_checkpoint"]
    assert second["next_triangle"] == 5
    assert second["artifacts"]["checkpoint_sha256"]
    assert second["all_current_stage_checks_pass"]

    with MODULE.np.load(checkpoint, allow_pickle=False) as saved:
        assert len(saved["b_rows"]) == len(saved["b_columns"])
        assert len(saved["b_rows"]) == len(saved["b_values"])
        assert len(saved["b_rows"]) == len(saved["b_errors"])
        assert MODULE.np.all(saved["b_columns"] < 5)


def test_stored_complete_assembly_archive_is_hash_bound_and_fail_closed():
    result_path = (
        RESULTS_DIR
        / "neutral_strip_h006_positive_exponential_complete_assembly_v1.json"
    )
    checkpoint_path = (
        RESULTS_DIR
        / "neutral_strip_h006_positive_exponential_assembly_checkpoint_v1.npz"
    )
    checkpoint_metadata_path = (
        RESULTS_DIR
        / "neutral_strip_h006_positive_exponential_assembly_checkpoint_v1.json"
    )
    matrices_path = (
        RESULTS_DIR
        / "neutral_strip_h006_positive_exponential_assembly_matrices_v1.npz"
    )
    result = MODULE.json.loads(result_path.read_text(encoding="ascii"))
    metadata = MODULE.json.loads(
        checkpoint_metadata_path.read_text(encoding="ascii")
    )

    assert result["status"] == "complete"
    assert result["next_triangle"] == 30954
    assert result["all_current_stage_checks_pass"]
    assert result["artifacts"]["checkpoint_sha256"] == MODULE._sha256_file(
        checkpoint_path
    )
    assert result["artifacts"][
        "checkpoint_metadata_sha256"
    ] == MODULE._sha256_file(checkpoint_metadata_path)
    assert result["artifacts"]["matrices_sha256"] == MODULE._sha256_file(
        matrices_path
    )
    assert metadata["checkpoint_npz_sha256"] == MODULE._sha256_file(
        checkpoint_path
    )

    with np.load(matrices_path, allow_pickle=False) as stored:
        array_hash = MODULE._sha256_arrays(
            stored["p_indptr"],
            stored["p_indices"],
            stored["p_values"],
            stored["p_errors"],
            stored["w_values"],
            stored["w_errors"],
            stored["d_values"],
            stored["d_errors"],
            stored["b_rows"],
            stored["b_columns"],
            stored["b_values"],
            stored["b_errors"],
            stored["n_rows"],
            stored["n_columns"],
            stored["n_values"],
        )
        assert (
            array_hash
            == "6deb9b9c41f842320cfeee2abf64a81047a45e234fe98b9cd66bbb55127b0274"
        )
        assert (
            array_hash
            == result["sparse_matrix_diagnostics"]["matrix_arrays_sha256"]
        )

        p_shape = (46697, 46697)
        p_center = csr_matrix(
            (
                stored["p_values"],
                stored["p_indices"],
                stored["p_indptr"],
            ),
            shape=p_shape,
        )
        p_errors = csr_matrix(
            (
                stored["p_errors"],
                stored["p_indices"],
                stored["p_indptr"],
            ),
            shape=p_shape,
        )
        assert p_center.nnz == 232421
        assert (p_center != p_center.transpose()).nnz == 0
        assert (p_errors != p_errors.transpose()).nnz == 0
        assert np.all(np.isfinite(p_center.data))
        assert np.all(np.isfinite(p_errors.data))
        assert np.all(p_errors.data >= 0.0)
        assert np.all(stored["w_values"] - stored["w_errors"] > 0.0)
        assert np.all(stored["d_values"] - stored["d_errors"] > 0.0)
        assert np.all(stored["b_values"] - stored["b_errors"] > 0.0)
        assert len(stored["b_values"]) == 91124
        assert len(stored["n_values"]) == 92862
        assert np.all(np.abs(stored["n_values"].astype(int)) == 1)
        assert np.all(
            np.bincount(stored["n_rows"], minlength=30954) == 3
        )

    flags = result["certification_flags"]
    assert flags["complete_mesh_RT0_P0_matrix_entries_enclosed"]
    assert flags["complete_mesh_P_W_D_B_checkpoint_hash_bound"]
    assert not flags["full_mesh_threshold_inertia_certified"]
    assert not flags["kappa_h_verified_upper_bound"]
    assert not flags["continuum_spectrum_below_60_captured"]
