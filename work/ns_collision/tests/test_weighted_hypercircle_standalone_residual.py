import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csc_matrix


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREFIX = _load(
    "standalone_residual_contract_prefix_test",
    "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
)
MODULE = _load(
    "weighted_hypercircle_standalone_residual_test",
    "neutral_strip_weighted_hypercircle_standalone_residual.py",
)


def _problem():
    center = csc_matrix(
        np.asarray(
            [
                [4.0, 0.25, 0.0],
                [0.25, -3.0, 0.5],
                [0.0, 0.5, 2.0],
            ]
        )
    ).tocsr()
    radius = csc_matrix(np.full((3, 3), 1.0e-12)).tocsr()
    order = np.asarray([2, 0, 1], dtype=np.int64)
    positions = np.empty(3, dtype=np.int64)
    positions[order] = np.arange(3, dtype=np.int64)
    return PREFIX.PrefixProblem(
        center=center,
        radius=radius,
        scale=np.asarray([1.0, 2.0, 3.0]),
        order=order,
        positions=positions,
        lower=csc_matrix(np.eye(3)),
        central_pivots=np.diag(center.toarray()),
    )


def test_sparse_fingerprint_is_value_and_structure_sensitive():
    first = csc_matrix(np.asarray([[1.0, 0.0], [0.0, 2.0]]))
    changed_value = csc_matrix(np.asarray([[1.0, 0.0], [0.0, 3.0]]))
    changed_structure = csc_matrix(np.asarray([[1.0, 0.5], [0.0, 2.0]]))
    first_hash = MODULE._sparse_fingerprint(first, PREFIX)
    assert len(first_hash) == 64
    assert first_hash != MODULE._sparse_fingerprint(changed_value, PREFIX)
    assert first_hash != MODULE._sparse_fingerprint(changed_structure, PREFIX)


def test_standalone_contract_binds_prefix_and_source_files(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"source-{index}.bin"
        path.write_bytes(f"source-{index}".encode("ascii"))
        paths.append(path)
    preparation = {
        "hashes": {
            "scale_sha256": "a" * 64,
            "raw_permutation_sha256": "b" * 64,
            "order_sha256": "c" * 64,
            "factor_pattern_sha256": "d" * 64,
            "U_diagonal_sha256": "e" * 64,
        }
    }
    contract = MODULE._build_standalone_contract(
        _problem(),
        preparation,
        complete_result_path=paths[0],
        matrices_path=paths[1],
        gaussian_result_path=paths[2],
        gaussian_checkpoint_path=paths[3],
        maximum_pivots=2,
        prefix_module=PREFIX,
    )
    assert len(contract["contract_sha256"]) == 64
    payload = contract["contract"]
    assert payload["validation_mode"] == "standalone_hash_bound"
    assert payload["maximum_pivots"] == 2
    assert payload["ordered_center_prefix_nnz"] > 0
    assert len(payload["ordered_original_indices_sha256"]) == 64
    assert len(payload["ordered_positive_scale_sha256"]) == 64
    assert all(
        len(record["sha256"]) == 64
        for record in payload["source_artifacts"].values()
    )
