"""Audit the exact dynamic L2 norm of a positive weighted kernel."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.linalg import svdvals


def _dynamic_l2_data(
    kernel: np.ndarray, source_law: np.ndarray
) -> tuple[np.ndarray, float, float]:
    mass_function = kernel @ np.ones(kernel.shape[1])
    squared_gain = float(np.sum(source_law * mass_function**2))
    target_law = (
        (source_law * mass_function) @ kernel / squared_gain
    )
    conjugated = (
        np.sqrt(source_law)[:, None]
        * kernel
        / np.sqrt(target_law)[None, :]
    )
    operator_norm = float(svdvals(conjugated)[0])
    return target_law, math.sqrt(squared_gain), operator_norm


def _random_kernel_stress(seed: int = 270719) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows = []
    for source_size, target_size in ((3, 5), (5, 4), (7, 6)):
        for _ in range(8):
            kernel = rng.lognormal(
                mean=-1.0, sigma=0.8, size=(source_size, target_size)
            )
            source_law = rng.random(source_size)
            source_law /= np.sum(source_law)
            target_law, exact_gain, operator_norm = _dynamic_l2_data(
                kernel, source_law
            )
            rows.append(
                {
                    "source_size": source_size,
                    "target_size": target_size,
                    "target_law_mass_error": abs(
                        float(np.sum(target_law)) - 1.0
                    ),
                    "operator_norm_error_from_exact_gain": abs(
                        operator_norm - exact_gain
                    ),
                }
            )
    return {
        "trial_count": len(rows),
        "maximum_target_law_mass_error": max(
            row["target_law_mass_error"] for row in rows
        ),
        "maximum_operator_norm_error_from_exact_gain": max(
            row["operator_norm_error_from_exact_gain"] for row in rows
        ),
    }


def _markov_and_pair_test(seed: int = 290719) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    kernel = rng.random((4, 5))
    kernel /= np.sum(kernel, axis=1, keepdims=True)
    source_law = rng.random(4)
    source_law /= np.sum(source_law)
    target_law, gain, operator_norm = _dynamic_l2_data(
        kernel, source_law
    )

    weighted_kernel = rng.lognormal(mean=-1.0, sigma=0.5, size=(4, 5))
    weighted_target, weighted_gain, _ = _dynamic_l2_data(
        weighted_kernel, source_law
    )
    pair_kernel = np.kron(weighted_kernel, weighted_kernel)
    pair_source = np.kron(source_law, source_law)
    pair_target = np.kron(weighted_target, weighted_target)
    pair_conjugated = (
        np.sqrt(pair_source)[:, None]
        * pair_kernel
        / np.sqrt(pair_target)[None, :]
    )
    pair_norm = float(svdvals(pair_conjugated)[0])
    return {
        "Markov_target_law_pushforward_error": float(
            np.max(np.abs(target_law - source_law @ kernel))
        ),
        "Markov_exact_gain": gain,
        "Markov_operator_norm": operator_norm,
        "weighted_one_history_gain": weighted_gain,
        "weighted_pair_operator_norm": pair_norm,
        "pair_norm_equals_one_history_gain_squared": bool(
            abs(pair_norm - weighted_gain**2) < 1.0e-12
        ),
    }


def audit() -> dict[str, object]:
    random_stress = _random_kernel_stress()
    markov_pair = _markov_and_pair_test()
    compact_static_visit_norm = 0.55681307217
    compact_static_generation = 0.160019377035
    legacy_cycle_coefficient = (
        compact_static_generation / compact_static_visit_norm**2
    )
    cycle_coefficient = 0.6586950386676936
    maximum_dynamic_gain = 1.0 / math.sqrt(cycle_coefficient)
    rotating_point_diagnostic = 0.59090525998

    result: dict[str, object] = {
        "positive_kernel": "Kf(x)=int K(x,dy)f(y), m(x)=K1(x)",
        "source_law": "mu is any probability law on the entry boundary",
        "squared_gain": "A=int m(x)^2 mu(dx)",
        "square_tilted_exit_law": (
            "nu(dy)=A^(-1) int mu(dx)m(x)K(x,dy)"
        ),
        "exact_dynamic_L2_norm": (
            "||K||_(L2(nu)->L2(mu))=sqrt(A)=||K1||_(L2(mu))"
        ),
        "proof": (
            "|Kf|^2<=(K1)K(f^2); integration gives the upper bound, "
            "and f=1 attains equality"
        ),
        "independent_replica_pair_norm": "A=(sqrt(A))^2",
        "ordinary_Markov_special_case": (
            "m=1, A=1, and nu is the usual pushed-forward law"
        ),
        "random_kernel_stress_test": random_stress,
        "Markov_and_pair_test": markov_pair,
        "legacy_bare_halving_cycle_coefficient": legacy_cycle_coefficient,
        "current_cubic_split_cycle_coefficient": cycle_coefficient,
        "compact_dynamic_generation_formula": (
            "C_dynamic=0.658695038668*g^2, "
            "g=||K1||_(L2(entry law))"
        ),
        "maximum_dynamic_one_history_gain_for_closure": (
            maximum_dynamic_gain
        ),
        "generation_criterion_if_gain_is_at_most_one": cycle_coefficient,
        "rotating_point_payoff_diagnostic": rotating_point_diagnostic,
        "generation_criterion_at_rotating_point_diagnostic": (
            cycle_coefficient * rotating_point_diagnostic**2
        ),
        "fixed_Gaussian_boundary_conversion_required": False,
        "dynamic_weighted_boundary_theorem_closed": False,
        "remaining_scalar_gate": (
            "prove a uniform bound g<1.232133609 for the constant-payoff "
            "nonautonomous full-affine visit, then extend the estimate to "
            "the spatial Campanato and strain perturbations"
        ),
        "scope_guard": (
            "the kernel norm identity is exact; the displayed compact "
            "cycle coefficient inherits the current finite-element "
            "calibration and the scalar nonautonomous gain is not yet bounded"
        ),
        "next_gate": (
            "construct a scalar supersolution or dynamic energy/trace bound "
            "for K1; a gain bound by one would leave criterion 0.6587"
        ),
    }
    positive_checks = (
        random_stress["maximum_target_law_mass_error"] < 1.0e-13,
        random_stress[
            "maximum_operator_norm_error_from_exact_gain"
        ]
        < 1.0e-12,
        abs(markov_pair["Markov_exact_gain"] - 1.0) < 1.0e-13,
        abs(markov_pair["Markov_operator_norm"] - 1.0) < 1.0e-12,
        markov_pair["pair_norm_equals_one_history_gain_squared"],
        maximum_dynamic_gain > 1.23,
        0.658 < cycle_coefficient < 0.659,
        result["generation_criterion_at_rotating_point_diagnostic"] < 0.24,
    )
    result["all_positive_weighted_kernel_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
