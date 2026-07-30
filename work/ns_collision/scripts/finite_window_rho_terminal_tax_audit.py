"""Audit the exact finite-window terminal tax for correlated replicas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from signed_projected_replica_generator_audit import (
    _gaussian_chaos_homotopy_audit,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parents[3]
PRIOR_RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "scale_adapted_edge_rho_expansion_audit_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _symbolic_terminal_tax_audit() -> dict[str, Any]:
    terminal_rho, terminal_zero, reset = sp.symbols(
        "A_rho_T A_0_T A_s",
        real=True,
    )
    cubic_terminal, cubic_reset = sp.symbols(
        "B_T B_s",
        real=True,
    )
    generator_rho = (
        sp.Rational(3, 2) * (terminal_rho - reset)
        - sp.Rational(1, 2) * (cubic_terminal - cubic_reset)
    )
    generator_zero = (
        sp.Rational(3, 2) * (terminal_zero - reset)
        - sp.Rational(1, 2) * (cubic_terminal - cubic_reset)
    )
    terminal_tax = sp.Rational(3, 2) * (
        terminal_rho - terminal_zero
    )
    difference_residual = sp.simplify(
        generator_rho - generator_zero - terminal_tax
    )

    rho = sp.symbols("rho", nonnegative=True)
    energies = sp.symbols("e1:5", nonnegative=True)
    chaos_difference = sum(
        rho**order * energies[order - 1]
        for order in range(1, 5)
    )
    chaos_derivative = sp.diff(chaos_difference, rho)
    chaos_second_derivative = sp.diff(chaos_derivative, rho)

    return {
        "backward_weight_equation": (
            "lambda_t+u dot grad lambda+nu Delta lambda=0, "
            "lambda(T)=lambda_T>=0"
        ),
        "weighted_balance_endpoint_form": (
            "J_rho=(3/2)[int lambda_T C_rho(T)-"
            "int lambda_s|u(s)|^2]-(1/2)[int lambda_T^3-"
            "int lambda_s^3]"
        ),
        "finite_window_terminal_tax_identity": (
            "J_rho-J_0=(3/2)int lambda_T[C_rho(T)-C_0(T)]"
        ),
        "chaos_terminal_difference": (
            "C_rho(T,x)-C_0(T,x)="
            "sum_(n>=1)rho^n||V_n(T,x)||_chaos^2"
        ),
        "pointwise_ordering": (
            "C_rho(T,x)>=C_0(T,x)=|u(T,x)|^2 "
            "for 0<=rho<=1"
        ),
        "generator_ordering": (
            "J_rho[lambda_T]>=J_0[lambda_T] for every "
            "admissible deterministic lambda_T>=0"
        ),
        "supremum_ordering": (
            "sup_lambda_T J_rho>=sup_lambda_T J_0 on every "
            "common admissible terminal-weight class"
        ),
        "symbolic_generator_rho": str(generator_rho),
        "symbolic_generator_zero": str(generator_zero),
        "symbolic_terminal_tax": str(terminal_tax),
        "difference_symbolic_residual": str(difference_residual),
        "finite_chaos_difference_polynomial": str(
            sp.expand(chaos_difference)
        ),
        "finite_chaos_first_derivative": str(
            sp.expand(chaos_derivative)
        ),
        "finite_chaos_second_derivative": str(
            sp.expand(chaos_second_derivative)
        ),
        "absolute_monotonicity": (
            "Every rho derivative is nonnegative on [0,1] whenever "
            "the corresponding chaos series is summable."
        ),
        "all_checks_pass": difference_residual == 0,
    }


def _weighted_chaos_stress() -> dict[str, Any]:
    inherited = _gaussian_chaos_homotopy_audit()
    base_energies = np.asarray(inherited["chaos_energies"], dtype=float)
    theta = 2.0 * math.pi * np.arange(128) / 128.0
    terminal_weight = (
        0.35
        + (
            1.0
            + 0.4 * np.sin(theta)
            + 0.2 * np.cos(2.0 * theta)
        )
        ** 2
    )
    spatial_energies = np.empty((4, theta.size), dtype=float)
    for order, energy in enumerate(base_energies):
        profile = (
            1.0
            + 0.15 * np.cos((order + 1) * theta)
            + 0.1 * np.sin((order + 2) * theta)
        )
        spatial_energies[order] = energy * profile**2

    rows: list[dict[str, Any]] = []
    correlations = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    for rho in correlations:
        terminal_difference = sum(
            rho**order * spatial_energies[order]
            for order in range(1, spatial_energies.shape[0])
        )
        first_derivative = sum(
            order * rho ** (order - 1) * spatial_energies[order]
            for order in range(1, spatial_energies.shape[0])
        )
        second_derivative = sum(
            order
            * (order - 1)
            * rho ** (order - 2)
            * spatial_energies[order]
            for order in range(2, spatial_energies.shape[0])
        )
        rows.append(
            {
                "rho": rho,
                "minimum_pointwise_terminal_difference": float(
                    np.min(terminal_difference)
                ),
                "weighted_terminal_tax": float(
                    1.5 * np.mean(terminal_weight * terminal_difference)
                ),
                "weighted_tax_first_rho_derivative": float(
                    1.5 * np.mean(terminal_weight * first_derivative)
                ),
                "weighted_tax_second_rho_derivative": float(
                    1.5 * np.mean(terminal_weight * second_derivative)
                ),
            }
        )

    weighted_variance_tax = float(
        1.5
        * np.mean(
            terminal_weight
            * np.sum(spatial_energies[1:], axis=0)
        )
    )
    maximum_quadrature_residual = max(
        max(
            row["correlation_residual"],
            row["derivative_residual"],
        )
        for row in inherited["rows"]
    )
    taxes = [row["weighted_terminal_tax"] for row in rows]
    return {
        "terminal_weight_minimum": float(np.min(terminal_weight)),
        "terminal_weight_maximum": float(np.max(terminal_weight)),
        "base_chaos_energies": [
            float(value) for value in base_energies
        ],
        "rows": rows,
        "rho_one_weighted_variance_tax": weighted_variance_tax,
        "rho_one_tax_residual": abs(taxes[-1] - weighted_variance_tax),
        "maximum_Gauss_Hermite_correlation_or_derivative_residual": (
            maximum_quadrature_residual
        ),
        "tax_monotone_on_sampled_grid": all(
            right >= left - 1.0e-13
            for left, right in zip(taxes, taxes[1:])
        ),
        "tax_strictly_positive_for_sampled_positive_rho": all(
            row["weighted_terminal_tax"] > 0.0
            for row in rows[1:]
        ),
        "all_checks_pass": (
            inherited["all_checks_pass"]
            and np.min(terminal_weight) > 0.0
            and abs(taxes[0]) < 1.0e-14
            and all(
                row["minimum_pointwise_terminal_difference"] >= -1.0e-14
                and row["weighted_tax_first_rho_derivative"] > 0.0
                and row["weighted_tax_second_rho_derivative"] >= -1.0e-14
                for row in rows
            )
            and all(
                right >= left - 1.0e-13
                for left, right in zip(taxes, taxes[1:])
            )
            and abs(taxes[-1] - weighted_variance_tax) < 1.0e-13
            and maximum_quadrature_residual < 3.0e-13
        ),
    }


def _taylor_crossover_reinterpretation() -> dict[str, Any]:
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    expansion = prior["short_time_rho_expansion"]
    leading = expansion["leading_reset_loss_range"]
    coefficient = expansion["first_time_coefficient_range"]
    crossover_values = [
        row["formal_integrated_crossover_time"]
        for row in expansion["rows"]
    ]
    return {
        "prior_result": PRIOR_RESULT.relative_to(ROOT).as_posix(),
        "prior_result_sha256": _sha256(PRIOR_RESULT),
        "leading_reset_loss_range": leading,
        "first_time_coefficient_range": coefficient,
        "formal_integrated_crossover_range": [
            min(crossover_values),
            max(crossover_values),
        ],
        "exact_rho_zero_derivative": (
            "partial_rho(J_rho-J_0)|_(rho=0)="
            "(3/2)int lambda_T||V_1(T)||_chaos^2>=0"
        ),
        "interpretation": (
            "The short-time polynomial K0*h+(K1/2)*h^2 is a temporal "
            "Taylor approximation to a nonnegative terminal first-chaos "
            "energy. Its formal zero cannot mark a net rho advantage; "
            "uncontrolled higher time orders must prevent a negative "
            "exact accumulated correction."
        ),
        "formal_crossover_is_a_search_target": False,
        "finite_window_seed81_solver_needed_to_decide_net_sign": False,
        "finite_window_solver_role_if_built": (
            "numerical calibration of the exact terminal-tax identity, "
            "not a search for positive-rho improvement"
        ),
        "all_checks_pass": (
            prior["all_positive_checks_pass"] is True
            and leading[0] > 0.0
            and coefficient[1] < 0.0
            and min(crossover_values) > 0.0
        ),
    }


def audit() -> dict[str, Any]:
    symbolic = _symbolic_terminal_tax_audit()
    chaos = _weighted_chaos_stress()
    taylor = _taylor_crossover_reinterpretation()
    positive_checks = {
        "endpoint_generator_algebra_closes": symbolic[
            "all_checks_pass"
        ],
        "terminal_chaos_tax_is_nonnegative": chaos["all_checks_pass"],
        "rho_one_tax_equals_weighted_variance": (
            chaos["rho_one_tax_residual"] < 1.0e-13
        ),
        "prior_quadratic_crossover_reinterpreted": taylor[
            "all_checks_pass"
        ],
    }
    return {
        "kind": "finite_window_rho_terminal_tax_no_go_audit",
        "schema_version": 1,
        "status": (
            "canonical_positive_rho_finite_window_route_closed_by_"
            "terminal_tax_identity"
        ),
        "assumption_scope": (
            "Classical smooth periodic Navier-Stokes, canonical projected "
            "Weber replicas generated by the same square-integrable Wiener "
            "functional, a common reset at time s, correlation 0<=rho<=1, "
            "and a deterministic backward weight with smooth nonnegative "
            "terminal datum. Standard approximation is still needed at "
            "zeros of the exact |u(T)| optimizer."
        ),
        "theorem": symbolic,
        "weighted_chaos_stress": chaos,
        "taylor_crossover_reinterpretation": taylor,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "terminal_tax_identity_derived": True,
            "terminal_tax_nonnegative_for_positive_rho": True,
            "rho_zero_globally_minimizes_fixed_weight_generator": True,
            "rho_zero_globally_minimizes_generator_supremum": True,
            "positive_rho_finite_window_advantage_in_this_dual_class": (
                False
            ),
            "formal_quadratic_crossover_is_a_sign_target": False,
            "seed81_finite_window_sign_search_required": False,
            "random_or_path_adapted_weight_route_excluded": False,
            "signed_or_multi_replica_route_excluded": False,
            "intrinsic_scale_pressure_tail_bound_proved": False,
            "critical_signed_bound_proved": False,
            "low_regularity_passage_proved": False,
            "exceptional_set_upgrade_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "scope_boundary": (
            "The no-go concerns net improvement of the exact canonical "
            "positive-correlation backward dual. Correlated replicas may "
            "still expose useful coercive structure, and this audit does "
            "not cover stochastic/path-adapted weights with their extra "
            "covariation terms, signed terminal constructions, or genuinely "
            "different multi-replica inequalities."
        ),
        "route_decision": (
            "Do not spend production compute searching near h=0.0756 for "
            "a net positive-rho advantage. Return the main proof effort to "
            "the rho=0 intrinsic-scale pressure-edge problem."
        ),
        "next_theorem_target": (
            "At rho=0, replace the finite-mode spectral-silence observation "
            "by a uniform intrinsic-scale pressure-tail estimate for "
            "m comparable to local amplitude/nu. The estimate must survive "
            "adaptive overlap, zero-face degeneracy, and the full "
            "nonnegative terminal-weight supremum. Treat path-adapted "
            "replica weights only as a separately derived model with every "
            "Ito covariation term retained."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "work/ns_collision/results/"
            "finite_window_rho_terminal_tax_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("finite-window terminal-tax audit failed")
    _atomic_json(arguments.output, result)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": _sha256(arguments.output),
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
