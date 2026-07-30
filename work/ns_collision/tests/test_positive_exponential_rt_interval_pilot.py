import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_positive_exponential_rt_interval_pilot.py"
)
SPEC = importlib.util.spec_from_file_location(
    "positive_exponential_rt_interval_pilot",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _base_module():
    return MODULE._load_module(
        "neutral_strip_gaussian_weighted_assembly_interval_audit.py",
        "positive_exponential_rt_test_base",
    )


def test_positive_exponential_coefficient_recurrence_at_zero():
    base = _base_module()
    centers, radii = MODULE._positive_exponential_coefficients(
        base,
        0.0,
        4,
    )
    expected = np.asarray([1.0, 0.0, 0.5, 0.0, 0.125])
    assert np.all(np.abs(centers - expected) <= radii + 1.0e-15)


def test_exact_decimal_beta_is_enclosed():
    base = _base_module()
    interval = MODULE._exact_decimal_interval(
        MODULE.PRODUCTION_BETA_DECIMAL,
        base,
    )
    from decimal import Decimal

    assert Decimal.from_float(interval[0]) < Decimal("0.045")
    assert Decimal.from_float(interval[1]) > Decimal("0.045")


def test_standard_triangle_rt_and_source_quadrature_contained():
    base = _base_module()
    points = np.asarray([[0.0, 0.0], [0.4, 0.0], [0.0, 0.3]])
    exact = MODULE._local_interval_forms(points, 18, base)
    nodes, weights = base._mapped_nodes(18)
    source, rt_mass = MODULE._positive_quadrature_local_forms(
        points,
        nodes,
        weights,
    )
    assert MODULE._distance_to_interval(source, exact["source_mass"]) == 0.0
    for row in range(3):
        assert exact["rt_mass"][row][row][0] > 0.0
        for column in range(row, 3):
            assert (
                MODULE._distance_to_interval(
                    rt_mass[row, column],
                    exact["rt_mass"][row][column],
                )
                == 0.0
            )


def test_small_distributed_pilot_and_complete_geometry_budget():
    result = MODULE.run_pilot(
        MODULE.DEFAULT_HYPERCIRCLE_RESULT,
        MODULE.DEFAULT_DEPENDENCY_RESULT,
        sample_count=8,
        degree=18,
        quadrature_order=8,
        cross_check_order=12,
    )
    assert result["all_positive_exponential_interval_pilot_checks_pass"]
    assert result["geometry_budget"]["directed_budget_passes"]
    assert result["quadrature_crosscheck"]["containment_failures"] == 0
    assert not result["certification_flags"][
        "complete_mesh_RT0_P0_matrix_entries_enclosed"
    ]
    assert not result["certification_flags"][
        "continuum_spectrum_below_60_captured"
    ]


def test_stored_512_triangle_pilot_is_fail_closed():
    result_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "neutral_strip_h006_positive_exponential_rt_interval_pilot512_v1.json"
    )
    result = MODULE.json.loads(result_path.read_text(encoding="ascii"))
    assert result["all_positive_exponential_interval_pilot_checks_pass"]
    assert result["sample"]["selected_count"] == 512
    assert result["quadrature_crosscheck"]["containment_checks"] == 8192
    assert result["quadrature_crosscheck"]["containment_failures"] == 0
    assert result["geometry_budget"]["directed_budget_passes"]
    assert result["geometry_budget"]["strict_headroom_lower"] > 0.0025
    assert result["certification_flags"][
        "complete_mesh_geometry_budget_certified"
    ]
    assert not result["certification_flags"][
        "complete_mesh_RT0_P0_matrix_entries_enclosed"
    ]
    assert not result["certification_flags"][
        "full_mesh_threshold_inertia_certified"
    ]
