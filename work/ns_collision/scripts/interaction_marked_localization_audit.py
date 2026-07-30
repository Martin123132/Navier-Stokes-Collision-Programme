"""Audit exact interaction marking and its skew/localization obstruction."""

from __future__ import annotations

import itertools
import json

import numpy as np
from scipy.linalg import block_diag


def _operator_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def _two_coordinate_counterexample(magnitude: float = 10.0) -> dict[str, object]:
    skew = np.array([[0.0, magnitude], [-magnitude, 0.0]])
    weights = (
        np.diag([1.0, 0.0]),
        np.diag([0.0, 1.0]),
    )
    pieces = [weight @ skew for weight in weights]
    symmetric_parts = [0.5 * (piece + piece.T) for piece in pieces]
    return {
        "skew_magnitude": magnitude,
        "global_symmetric_part_norm": _operator_norm(
            0.5 * (skew + skew.T)
        ),
        "piece_symmetric_part_norms": [
            _operator_norm(part) for part in symmetric_parts
        ],
        "piece_maximum_real_eigenvalues": [
            float(np.linalg.eigvalsh(part)[-1])
            for part in symmetric_parts
        ],
        "piece_sum_residual": _operator_norm(sum(pieces) - skew),
        "global_skew_cancellation_exact": bool(
            _operator_norm(0.5 * (skew + skew.T)) == 0.0
        ),
        "each_exact_label_piece_has_adverse_real_part": all(
            np.linalg.eigvalsh(part)[-1] > 0.0
            for part in symmetric_parts
        ),
    }


def _partition_lift(seed: int = 20260719) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    dimension = 7
    label_count = 4
    raw_weights = rng.uniform(0.05, 1.0, size=(label_count, dimension))
    weights = raw_weights / np.sum(raw_weights, axis=0, keepdims=True)
    square_roots = [np.diag(np.sqrt(row)) for row in weights]
    linear_weights = [np.diag(row) for row in weights]
    injection = np.vstack(square_roots)

    raw_skew = rng.normal(size=(dimension, dimension))
    skew = raw_skew - raw_skew.T
    potential_diagonal = rng.uniform(0.2, 1.4, size=dimension)
    potential = np.diag(potential_diagonal)

    lifted_skew = injection @ skew @ injection.T
    lifted_potential = injection @ potential @ injection.T
    diagonal_skew = block_diag(
        *[root @ skew @ root for root in square_roots]
    )
    diagonal_potential = block_diag(
        *[root @ potential @ root for root in square_roots]
    )
    compressed_diagonal_skew = injection.T @ diagonal_skew @ injection
    compressed_diagonal_potential = (
        injection.T @ diagonal_potential @ injection
    )

    one_sided_pieces = [weight @ skew for weight in linear_weights]
    commutator_residuals = []
    adverse_real_parts = []
    for weight, piece in zip(linear_weights, one_sided_pieces):
        symmetric_part = 0.5 * (piece + piece.T)
        commutator = 0.5 * (weight @ skew - skew @ weight)
        commutator_residuals.append(
            _operator_norm(symmetric_part - commutator)
        )
        adverse_real_parts.append(
            float(np.linalg.eigvalsh(symmetric_part)[-1])
        )

    return {
        "dimension": dimension,
        "label_count": label_count,
        "quadratic_partition_residual": _operator_norm(
            injection.T @ injection - np.eye(dimension)
        ),
        "full_lifted_skew_residual": _operator_norm(
            lifted_skew + lifted_skew.T
        ),
        "full_lift_compresses_to_physical_skew_residual": _operator_norm(
            injection.T @ lifted_skew @ injection - skew
        ),
        "full_lift_compresses_to_physical_potential_residual": _operator_norm(
            injection.T @ lifted_potential @ injection - potential
        ),
        "decoupled_skew_compression_error": _operator_norm(
            compressed_diagonal_skew - skew
        ),
        "decoupled_potential_compression_error": _operator_norm(
            compressed_diagonal_potential - potential
        ),
        "one_sided_piece_sum_residual": _operator_norm(
            sum(one_sided_pieces) - skew
        ),
        "one_sided_piece_commutator_residuals": commutator_residuals,
        "one_sided_piece_adverse_real_parts": adverse_real_parts,
        "sum_of_one_sided_symmetric_parts_residual": _operator_norm(
            sum(0.5 * (piece + piece.T) for piece in one_sided_pieces)
        ),
        "cross_label_blocks_required_for_exact_compression": bool(
            _operator_norm(compressed_diagonal_skew - skew) > 1.0e-3
            and _operator_norm(compressed_diagonal_potential - potential)
            > 1.0e-3
        ),
        "global_one_sided_skew_cancellation_exact": bool(
            _operator_norm(
                sum(0.5 * (piece + piece.T) for piece in one_sided_pieces)
            )
            < 1.0e-13
        ),
        "individual_one_sided_pieces_are_not_skew": all(
            value > 1.0e-4 for value in adverse_real_parts
        ),
    }


def _dyson_marking(seed: int = 20260720) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    dimension = 5
    label_count = 3
    raw = rng.normal(size=(dimension, dimension))
    baseline = raw.T @ raw + 4.0 * np.eye(dimension)
    baseline_inverse = np.linalg.inv(baseline)

    pieces = []
    for _ in range(label_count):
        piece = rng.normal(size=(dimension, dimension))
        pieces.append(piece)
    perturbation = sum(pieces)
    relative_radius = max(
        abs(np.linalg.eigvals(baseline_inverse @ perturbation))
    )
    scale = 0.35 / float(relative_radius)
    pieces = [scale * piece for piece in pieces]
    perturbation = sum(pieces)
    exact_resolvent = np.linalg.inv(baseline - perturbation)

    labelled_partial = baseline_inverse.copy()
    unlabelled_partial = baseline_inverse.copy()
    labelled_term_residuals = []
    labelled_triangle_norms = []
    for order in range(1, 6):
        unlabelled_term = np.linalg.matrix_power(
            baseline_inverse @ perturbation, order
        ) @ baseline_inverse
        unlabelled_partial += unlabelled_term
        labelled_term = np.zeros_like(baseline)
        triangle_norm = 0.0
        for word in itertools.product(range(label_count), repeat=order):
            term = baseline_inverse.copy()
            for label in word:
                term = term @ pieces[label] @ baseline_inverse
            labelled_term += term
            triangle_norm += _operator_norm(term)
        labelled_partial += labelled_term
        labelled_term_residuals.append(
            _operator_norm(labelled_term - unlabelled_term)
        )
        labelled_triangle_norms.append(triangle_norm)

    return {
        "relative_spectral_radius": float(
            max(
                abs(
                    np.linalg.eigvals(
                        baseline_inverse @ perturbation
                    )
                )
            )
        ),
        "labelled_word_term_residuals": labelled_term_residuals,
        "maximum_labelled_word_term_residual": max(
            labelled_term_residuals
        ),
        "labelled_partial_to_exact_resolvent_error": _operator_norm(
            labelled_partial - exact_resolvent
        ),
        "unlabelled_partial_to_exact_resolvent_error": _operator_norm(
            unlabelled_partial - exact_resolvent
        ),
        "final_labelled_triangle_to_cancelled_term_ratio": (
            labelled_triangle_norms[-1]
            / _operator_norm(
                np.linalg.matrix_power(
                    baseline_inverse @ perturbation, 5
                )
                @ baseline_inverse
            )
        ),
        "marking_identity_exact_to_roundoff": bool(
            max(labelled_term_residuals) < 1.0e-12
        ),
        "taking_label_norms_separately_loses_cancellation": bool(
            labelled_triangle_norms[-1]
            > 2.0
            * _operator_norm(
                np.linalg.matrix_power(
                    baseline_inverse @ perturbation, 5
                )
                @ baseline_inverse
            )
        ),
    }


def audit() -> dict[str, object]:
    counterexample = _two_coordinate_counterexample()
    lift = _partition_lift()
    dyson = _dyson_marking()
    result: dict[str, object] = {
        "exact_linear_interaction_split": (
            "P=sum_j Phi_j P when Phi_j>=0 and sum_j Phi_j=I"
        ),
        "localized_drift_real_part": (
            "Sym(Phi_j K)=0.5*(Phi_j K-K Phi_j); these commutators "
            "cancel only after summing j"
        ),
        "two_coordinate_counterexample": counterexample,
        "quadratic_direct_sum_lift": lift,
        "Dyson_word_marking": dyson,
        "interaction_marking_reconstructs_physical_resolvent": (
            dyson["marking_identity_exact_to_roundoff"]
        ),
        "interaction_marking_creates_a_collar_by_itself": False,
        "independent_labelwise_energy_estimates_preserve_global_skew": False,
        "decoupled_quadratic_label_blocks_reconstruct_generator": False,
        "coupled_lift_dichotomy": (
            "keeping every cross-label block preserves the physical "
            "generator and skewness but does not yield independent collar "
            "problems; dropping cross-label blocks yields independent "
            "problems but changes both drift and potential"
        ),
        "PDE_analogue": (
            "the commutator of multiplication by Phi_j with e.grad is "
            "multiplication by e.grad(Phi_j); a square-root Hilbert lift "
            "moves the same obstruction into partition-gradient/IMS terms"
        ),
        "requirements_for_a_surviving_marked_theorem": [
            "retain the global sum before taking the real energy part",
            "provide a contractive reconstruction of physical observations",
            "control rather than discard all cross-label collar transfers",
            "charge label changes only in a physical Markov norm or at paid stopping times",
        ],
        "scope_guard": (
            "this rules out naive independent interaction labels as a free "
            "collar mechanism. It does not rule out a coupled domain "
            "decomposition with proved cross-label cancellation, a paid "
            "stopping construction, or an averaged physical entry norm."
        ),
    }
    positive_checks = (
        counterexample["global_skew_cancellation_exact"],
        counterexample["each_exact_label_piece_has_adverse_real_part"],
        lift["cross_label_blocks_required_for_exact_compression"],
        lift["global_one_sided_skew_cancellation_exact"],
        lift["individual_one_sided_pieces_are_not_skew"],
        dyson["marking_identity_exact_to_roundoff"],
        dyson["taking_label_norms_separately_loses_cancellation"],
        not result["interaction_marking_creates_a_collar_by_itself"],
        not result[
            "independent_labelwise_energy_estimates_preserve_global_skew"
        ],
    )
    result["all_positive_interaction_marking_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
