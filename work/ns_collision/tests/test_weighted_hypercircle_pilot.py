import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_weighted_hypercircle_pilot.py"
)
SPEC = importlib.util.spec_from_file_location(
    "weighted_hypercircle_pilot",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_edge_signs_cancel_on_shared_edge():
    vertices = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    edge = (1, 2)
    first = MODULE._edge_sign(vertices, edge, 0)
    second = MODULE._edge_sign(vertices, edge, 3)
    assert first == -second


def test_small_weighted_hypercircle_pipeline():
    result = MODULE.run_pilot(
        spacing=0.6,
        quadrature_order=6,
        eig_tolerance=1.0e-8,
        maximum_iterations=200,
        continuum_result_path=MODULE.DEFAULT_CONTINUUM_RESULT,
        monitor_cpu=False,
    )
    assert result["all_floating_pilot_checks_pass"]
    assert result["mesh"]["interior_edge_outward_signs_cancel"]
    assert result["linear_algebra"][
        "objective_identity_relative_defect"
    ] <= 1.0e-9
    assert result["hypercircle_bound"]["combined_C_h_floating"] > 0.0
    assert not result["certification_flags"][
        "global_weighted_Ritz_constant_certified"
    ]


def test_stored_mesh_quadrature_crosscheck_and_fail_closed_flags():
    result_root = Path(__file__).resolve().parents[1] / "results"
    q12 = MODULE.json.loads(
        (
            result_root
            / "neutral_strip_h006_weighted_hypercircle_pilot_v1.json"
        ).read_text(encoding="ascii")
    )
    q18 = MODULE.json.loads(
        (
            result_root
            / "neutral_strip_h006_weighted_hypercircle_q18_crosscheck_v1.json"
        ).read_text(encoding="ascii")
    )
    coarse = MODULE.json.loads(
        (
            result_root
            / "neutral_strip_h012_weighted_hypercircle_pilot_v1.json"
        ).read_text(encoding="ascii")
    )

    expected_mesh_hash = (
        "174d325adf2b1a7f6c70a023982060bc"
        "492dbb279d267e4cdc2a2a85e9270835"
    )
    assert q12["mesh"]["mesh_fingerprint_sha256"] == expected_mesh_hash
    assert q18["mesh"]["mesh_fingerprint_sha256"] == expected_mesh_hash
    assert abs(
        q12["hypercircle_bound"]["kappa_h_floating"]
        - q18["hypercircle_bound"]["kappa_h_floating"]
    ) <= 2.0e-14
    assert q12["hypercircle_bound"][
        "floating_diagnostic_passes_strict_target"
    ]
    assert not coarse["hypercircle_bound"][
        "floating_diagnostic_passes_strict_target"
    ]
    assert not q12["certification_flags"]["kappa_h_verified_upper_bound"]
    assert not q12["certification_flags"][
        "continuum_spectrum_below_60_captured"
    ]
