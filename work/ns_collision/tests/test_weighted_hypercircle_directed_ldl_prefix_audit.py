from decimal import Context, Decimal
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "weighted_hypercircle_directed_ldl_prefix_audit",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _problem(
    center_values: np.ndarray,
    radius_values: np.ndarray,
    order: np.ndarray,
    scale: np.ndarray | None = None,
):
    center = csc_matrix(center_values)
    radius = csc_matrix(radius_values)
    permuted = center[order, :][:, order]
    factor = splu(
        permuted,
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    lower = factor.L.tocsc()
    lower.sort_indices()
    positions = np.empty(len(order), dtype=np.int64)
    positions[order] = np.arange(len(order), dtype=np.int64)
    center_csr = center.tocsr()
    radius_csr = radius.tocsr()
    center_csr.sort_indices()
    radius_csr.sort_indices()
    return MODULE.PrefixProblem(
        center=center_csr,
        radius=radius_csr,
        scale=(
            np.ones(len(order), dtype=float)
            if scale is None
            else np.asarray(scale, dtype=float)
        ),
        order=np.asarray(order, dtype=np.int64),
        positions=positions,
        lower=lower,
        central_pivots=np.asarray(factor.U.diagonal(), dtype=float),
    )


def _dense_decimal_pivots(
    values: np.ndarray,
    order: np.ndarray,
    scale: np.ndarray,
) -> list[Decimal]:
    context = Context(prec=100)
    dimension = len(order)
    matrix = [
        [
            context.multiply(
                context.multiply(
                    Decimal.from_float(
                        float(values[int(order[row]), int(order[column])])
                    ),
                    Decimal.from_float(float(scale[int(order[row])])),
                ),
                Decimal.from_float(float(scale[int(order[column])])),
            )
            for column in range(dimension)
        ]
        for row in range(dimension)
    ]
    lower = [
        [Decimal(0) for _ in range(dimension)]
        for _ in range(dimension)
    ]
    pivots: list[Decimal] = []
    for pivot in range(dimension):
        lower[pivot][pivot] = Decimal(1)
        diagonal_sum = Decimal(0)
        for prior in range(pivot):
            term = context.multiply(lower[pivot][prior], lower[pivot][prior])
            diagonal_sum = context.add(
                diagonal_sum,
                context.multiply(term, pivots[prior]),
            )
        pivots.append(
            context.subtract(matrix[pivot][pivot], diagonal_sum)
        )
        for row in range(pivot + 1, dimension):
            update = Decimal(0)
            for prior in range(pivot):
                term = context.multiply(
                    lower[row][prior],
                    lower[pivot][prior],
                )
                update = context.add(
                    update,
                    context.multiply(term, pivots[prior]),
                )
            lower[row][pivot] = context.divide(
                context.subtract(matrix[row][pivot], update),
                pivots[pivot],
            )
    return pivots


def test_superlu_permutation_semantics_use_inverse_permutation():
    rng = np.random.default_rng(20260724)
    raw = rng.normal(size=(9, 9))
    values = 0.5 * (raw + raw.T)
    values += np.diag(np.linspace(-4.0, 5.0, 9))
    values[np.abs(values) < 0.45] = 0.0
    values = 0.5 * (values + values.T)
    factor = splu(
        csc_matrix(values),
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    assert np.array_equal(factor.perm_r, factor.perm_c)
    raw_permutation = np.asarray(factor.perm_r)
    order = np.argsort(raw_permutation)
    reconstruction = factor.L.toarray() @ factor.U.toarray()
    inverse_residual = np.linalg.norm(
        values[np.ix_(order, order)] - reconstruction,
        ord=np.inf,
    )
    raw_residual = np.linalg.norm(
        values[np.ix_(raw_permutation, raw_permutation)]
        - reconstruction,
        ord=np.inf,
    )
    assert inverse_residual < 1.0e-12
    assert raw_residual > 1.0


def test_symbolic_scan_finds_dense_clique_transitions():
    center = np.asarray(
        [
            [5.0, 0.2, 0.3, 0.4],
            [0.2, 4.0, 0.5, 0.6],
            [0.3, 0.5, 3.0, 0.7],
            [0.4, 0.6, 0.7, 2.0],
        ]
    )
    problem = _problem(
        center,
        np.zeros_like(center),
        np.arange(4, dtype=np.int64),
    )
    scan = MODULE._scan_symbolic_prefix(
        problem,
        maximum_pivots=4,
        checkpoints=(2, 4),
    )
    assert scan["first_pivot_by_diagonal_term_count"] == {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
    }
    assert scan["first_pivot_by_descendant_count"] == {
        "0": 3,
        "1": 2,
        "2": 1,
        "3": 0,
    }
    assert scan["first_pivot_by_off_diagonal_common_term_count"] == {
        "0": 0,
        "1": 1,
        "2": 2,
    }
    assert scan["symbolic_lower_entry_count"] == 6
    assert scan["total_diagonal_term_count"] == 6
    assert scan["total_off_diagonal_common_term_count"] == 4
    assert scan["reference_product_pair_term_count"] == 20
    assert not scan["arithmetic_signs_certified"]


def test_directed_prefix_contains_dense_high_precision_samples(tmp_path):
    center = np.asarray(
        [
            [4.0, 0.75, -0.2, 0.35],
            [0.75, -3.25, 0.4, 0.15],
            [-0.2, 0.4, 2.5, -0.6],
            [0.35, 0.15, -0.6, -1.75],
        ],
        dtype=float,
    )
    radius = np.full((4, 4), 2.0e-12)
    radius = 0.5 * (radius + radius.T)
    order = np.asarray([2, 0, 3, 1], dtype=np.int64)
    scale = np.asarray([0.75, 1.25, 1.5, 0.625])
    problem = _problem(center, radius, order, scale)
    checkpoint = tmp_path / "dense-prefix.json"
    contract = {
        "test": "dense-high-precision-containment",
        "maximum_pivots": 4,
    }
    result = MODULE._run_adaptive_prefix(
        problem,
        maximum_pivots=4,
        precision_schedule=(50,),
        checkpoint_batch=2,
        checkpoint_path=checkpoint,
        contract=contract,
    )
    assert result["overall_status"] == "completed"
    attempt = result["current_attempt"]
    summary = MODULE._summarize_attempt(attempt)
    interaction = summary["interaction_profile"]
    assert interaction["pivot_count"] == 3
    assert interaction["first_pivot"] == 1
    assert interaction["last_pivot"] == 3
    assert interaction["minimum_pivot_margin_index"] in (1, 2, 3)
    assert interaction["maximum_cancellation_charge_index"] in (1, 2, 3)
    lower_bounds = [
        Decimal(value) for value in attempt["pivot_lower_decimal"]
    ]
    upper_bounds = [
        Decimal(value) for value in attempt["pivot_upper_decimal"]
    ]

    sample_levels = (-0.5, 0.0, 0.5)
    for sample_index in range(9):
        perturbation = np.empty_like(center)
        for row in range(4):
            for column in range(row + 1):
                level = sample_levels[
                    (sample_index + 2 * row + 3 * column) % 3
                ]
                perturbation[row, column] = level
                perturbation[column, row] = level
        sampled = center + radius * perturbation
        reference = _dense_decimal_pivots(sampled, order, scale)
        assert all(
            lower <= pivot <= upper
            for lower, pivot, upper in zip(
                lower_bounds,
                reference,
                upper_bounds,
                strict=True,
            )
        )


def test_adaptive_precision_escalates_after_fail_closed_pivot(tmp_path):
    center = np.asarray(
        [
            [1.0, 1.0],
            [1.0, np.nextafter(1.0, np.inf)],
        ]
    )
    problem = _problem(
        center,
        np.zeros_like(center),
        np.arange(2, dtype=np.int64),
    )
    checkpoint = tmp_path / "adaptive-prefix.json"
    contract = {
        "test": "adaptive-near-cancellation",
        "maximum_pivots": 2,
    }
    result = MODULE._run_adaptive_prefix(
        problem,
        maximum_pivots=2,
        precision_schedule=(8, 40),
        checkpoint_batch=1,
        checkpoint_path=checkpoint,
        contract=contract,
    )
    assert result["overall_status"] == "completed"
    assert result["current_attempt"]["precision"] == 40
    assert result["current_attempt"]["next_pivot"] == 2
    assert len(result["attempt_summaries"]) == 1
    failure = result["attempt_summaries"][0]["failed_pivot"]
    assert failure["index"] == 1
    assert failure["failure_kind"] == "zero_containing_pivot"


def test_checkpoint_resumes_and_rejects_corruption(tmp_path):
    center = np.asarray(
        [
            [3.0, 0.5, 0.25],
            [0.5, -2.0, 0.75],
            [0.25, 0.75, 4.0],
        ]
    )
    problem = _problem(
        center,
        np.zeros_like(center),
        np.arange(3, dtype=np.int64),
    )
    checkpoint = tmp_path / "resume-prefix.json"
    contract = {"test": "resume-and-corruption", "maximum_pivots": 3}
    parked = MODULE._run_adaptive_prefix(
        problem,
        maximum_pivots=3,
        precision_schedule=(40,),
        checkpoint_batch=1,
        checkpoint_path=checkpoint,
        contract=contract,
        pivot_budget=1,
    )
    assert parked["overall_status"] == "parked"
    assert parked["current_attempt"]["next_pivot"] == 1
    checkpoint_text = checkpoint.read_text(encoding="ascii")
    assert "\n  " not in checkpoint_text
    assert json.loads(checkpoint_text) == parked
    completed = MODULE._run_adaptive_prefix(
        problem,
        maximum_pivots=3,
        precision_schedule=(40,),
        checkpoint_batch=1,
        checkpoint_path=checkpoint,
        contract=contract,
    )
    assert completed["overall_status"] == "completed"
    assert completed["current_attempt"]["next_pivot"] == 3

    corrupted = json.loads(checkpoint.read_text(encoding="ascii"))
    corrupted["current_attempt"]["pivot_lower_decimal"][0] = "999"
    checkpoint.write_text(
        json.dumps(corrupted, sort_keys=True),
        encoding="ascii",
    )
    with pytest.raises(RuntimeError, match="state hash mismatch"):
        MODULE._load_checkpoint(
            checkpoint,
            contract,
            (40,),
        )


def test_stored_prefix_certificate_is_bounded_and_fail_closed():
    results = Path(__file__).resolve().parents[1] / "results"
    audit = json.loads(
        (
            results
            / "neutral_strip_h006_hypercircle_directed_ldl_prefix_audit_v1.json"
        ).read_text(encoding="ascii")
    )
    crosscheck = json.loads(
        (
            results
            / "neutral_strip_h006_hypercircle_directed_ldl_prefix_precision_crosscheck_v1.json"
        ).read_text(encoding="ascii")
    )
    prefix = audit["directed_LDL_prefix"]
    flags = audit["certification_flags"]
    assert audit["all_current_stage_checks_pass"]
    assert flags["bounded_prefix_directed_LDL_certified"]
    assert prefix["completed_pivot_count"] == 1024
    assert prefix["negative_pivot_count"] == 1024
    assert prefix["positive_pivot_count"] == 0
    assert prefix["pivot_block_counts"] == {
        "edge_metric": 0,
        "source_triangle": 1024,
        "state": 0,
        "triangle_constraint": 0,
    }
    assert prefix["pivots_with_nonzero_diagonal_term_count"] == 0
    assert audit["preparation"]["first_input_graph_dependent_pivot"] == 1738
    assert not flags["full_123816_pivot_inertia_certified"]
    assert not flags[
        "reversible_weighted_hypercircle_full_inertia_certified"
    ]
    assert not flags["continuum_spectrum_below_60_captured"]
    assert crosscheck["status"] == "pass"
    assert all(crosscheck["checks"].values())


def test_stored_interaction_prefix_and_next_boundary_are_fail_closed():
    results = Path(__file__).resolve().parents[1] / "results"
    audit = json.loads(
        (
            results
            / "neutral_strip_h006_hypercircle_directed_ldl_interaction2048_audit_v1.json"
        ).read_text(encoding="ascii")
    )
    replay = json.loads(
        (
            results
            / "neutral_strip_h006_hypercircle_directed_ldl_interaction2048_p80_audit_v1.json"
        ).read_text(encoding="ascii")
    )
    crosscheck = json.loads(
        (
            results
            / "neutral_strip_h006_hypercircle_directed_ldl_interaction2048_precision_crosscheck_v1.json"
        ).read_text(encoding="ascii")
    )
    scan = json.loads(
        (
            results
            / "neutral_strip_h006_hypercircle_directed_ldl_interaction2048_structural_scan_v1.json"
        ).read_text(encoding="ascii")
    )
    prefix = audit["directed_LDL_prefix"]
    interaction = prefix["interaction_profile"]
    assert audit["status"] == "certified_bounded_prefix"
    assert replay["status"] == "certified_bounded_prefix"
    assert prefix["completed_pivot_count"] == 2048
    assert prefix["negative_pivot_count"] == 1516
    assert prefix["positive_pivot_count"] == 532
    assert interaction["pivot_count"] == 310
    assert interaction["first_pivot"] == 1738
    assert interaction["last_pivot"] == 2047
    assert interaction["negative_pivot_count"] == 310
    assert interaction["positive_pivot_count"] == 0
    assert interaction["block_counts"] == {
        "edge_metric": 0,
        "source_triangle": 0,
        "state": 0,
        "triangle_constraint": 310,
    }
    assert crosscheck["status"] == "pass"
    assert all(crosscheck["checks"].values())
    transition = scan["next_complexity_transition"]
    assert transition["first_diagonal_term_count_4_pivot"] == 2270
    assert (
        transition["first_off_diagonal_common_term_count_2_pivot"]
        == 2270
    )
    assert transition["first_descendant_count_4_pivot"] == 2274
    assert transition["recommended_next_bounded_pivot_count"] == 2304
    assert not transition["arithmetic_extension_launched"]
    assert not audit["certification_flags"][
        "full_123816_pivot_inertia_certified"
    ]
    assert not replay["certification_flags"][
        "full_123816_pivot_inertia_certified"
    ]
    assert not scan["certification_flags"][
        "full_123816_pivot_inertia_certified"
    ]
