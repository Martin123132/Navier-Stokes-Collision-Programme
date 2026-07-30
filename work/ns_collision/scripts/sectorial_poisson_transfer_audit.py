"""Audit the sectorial form-to-boundary perturbation theorem."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.linalg import eigh, svdvals


def _symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    eigenvalues, eigenvectors = eigh(matrix)
    return (eigenvectors * eigenvalues**power) @ eigenvectors.T


def _random_sector_trial(
    seed: int, skew_strength: float, alpha: float, beta: float = 0.11
) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    dimension = 9
    trace_dimension = 4
    raw = rng.normal(size=(dimension, dimension))
    symmetric = raw.T @ raw + np.eye(dimension)
    skew_raw = rng.normal(size=(dimension, dimension))
    skew = skew_strength * (skew_raw - skew_raw.T)
    operator = symmetric + skew

    contraction_raw = rng.normal(size=(dimension, dimension))
    contraction = contraction_raw.T @ contraction_raw
    contraction /= float(eigh(contraction, eigvals_only=True)[-1])
    symmetric_half = _symmetric_power(symmetric, 0.5)
    perturbation = (
        alpha * symmetric_half @ contraction @ symmetric_half
    )
    trace = rng.normal(size=(trace_dimension, dimension))
    symmetric_inverse_half = _symmetric_power(symmetric, -0.5)
    trace_energy_norm = float(
        svdvals(trace @ symmetric_inverse_half)[0]
    )
    difference_map = (
        trace
        @ np.linalg.solve(operator - perturbation, perturbation)
        @ symmetric_inverse_half
    )
    actual_norm = float(svdvals(difference_map)[0])
    theorem_bound = alpha / (1.0 - alpha) * trace_energy_norm

    error_skew_raw = rng.normal(size=(dimension, dimension))
    error_skew_energy = error_skew_raw - error_skew_raw.T
    error_skew_energy *= beta / float(svdvals(error_skew_energy)[0])
    error_skew = (
        symmetric_half @ error_skew_energy @ symmetric_half
    )
    combined_difference_map = (
        trace
        @ np.linalg.solve(
            operator + error_skew - perturbation,
            perturbation - error_skew,
        )
        @ symmetric_inverse_half
    )
    combined_actual_norm = float(svdvals(combined_difference_map)[0])
    combined_theorem_bound = (
        (alpha + beta) / (1.0 - alpha) * trace_energy_norm
    )
    return {
        "seed": seed,
        "skew_strength": skew_strength,
        "alpha": alpha,
        "actual_energy_to_trace_difference_norm": actual_norm,
        "theorem_upper_bound": theorem_bound,
        "actual_to_bound_ratio": actual_norm / theorem_bound,
        "bound_holds": bool(actual_norm <= theorem_bound + 1.0e-11),
        "drift_error_sector_beta": beta,
        "combined_potential_and_drift_actual_norm": combined_actual_norm,
        "combined_potential_and_drift_theorem_bound": (
            combined_theorem_bound
        ),
        "combined_bound_holds": bool(
            combined_actual_norm <= combined_theorem_bound + 1.0e-11
        ),
    }


def audit() -> dict[str, object]:
    trials = [
        _random_sector_trial(seed, skew_strength, alpha=0.23)
        for seed, skew_strength in enumerate(
            (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0)
        )
    ]

    working_generation_criterion = 0.25663566049129444
    working_condition_number = 4.767866264013159
    working_alpha = 0.16962754460975144
    working_mass_budget = 0.9292034203149158
    sharp_sobolev_constant = 4.0 ** (2.0 / 3.0) / (
        3.0 * math.pi ** (4.0 / 3.0)
    )
    amplification = 1.0 + (
        working_condition_number
        * working_alpha
        / (1.0 - working_alpha)
    )
    closure_value = amplification**2 * working_generation_criterion
    closure_excess = 1.0 / math.sqrt(working_generation_criterion) - 1.0
    combined_budget_intercept = closure_excess / working_condition_number
    maximum_drift_sector_beta_if_alpha_zero = combined_budget_intercept
    maximum_drift_L3_over_nu_if_alpha_zero = (
        maximum_drift_sector_beta_if_alpha_zero
        / math.sqrt(sharp_sobolev_constant)
    )
    equal_alpha_beta = combined_budget_intercept / (
        2.0 + combined_budget_intercept
    )

    result: dict[str, object] = {
        "sector_decomposition": (
            "a_0=h+k with h symmetric coercive and Re k(v,v)=0"
        ),
        "relative_form_hypothesis": (
            "0<=q[v]<=alpha*h[v] for every zero-boundary v, alpha<1"
        ),
        "difference_equation": (
            "(A_0-Q)w=Q*u_0 with w having zero boundary data"
        ),
        "energy_difference_bound": (
            "sqrt(h[w])<=alpha/(1-alpha)*sqrt(h[zeta*u_0])"
        ),
        "trace_difference_bound": (
            "||T*w||<=alpha/(1-alpha)*C_T*sqrt(E_zeta)"
        ),
        "combined_potential_and_drift_difference_bound": (
            "||T*w||<=(alpha+beta)/(1-alpha)*C_T*sqrt(E_zeta)"
        ),
        "drift_sector_hypothesis": (
            "|k_e(u,v)|<=beta*sqrt(h[u]h[v]); a divergence-free "
            "e in L3 has beta<=sqrt(S3)*||e||_3/nu"
        ),
        "sector_condition_number": (
            "chi_sec=C_T*sqrt(E_zeta)/||B_0||"
        ),
        "renewal_closure_condition": (
            "C_0*(1+chi_sec*alpha/(1-alpha))^2<1"
        ),
        "skew_strength_does_not_enter_energy_bound": True,
        "cutoff_identity_survives_nonsymmetry": (
            "a_0(zeta*u,zeta*u)=integral |grad zeta|^2*u^2 "
            "when A_0*u=0 and zeta vanishes at the boundary"
        ),
        "random_sector_trials": trials,
        "all_random_sector_bounds_hold": all(
            row["bound_holds"] and row["combined_bound_holds"]
            for row in trials
        ),
        "maximum_random_actual_to_bound_ratio": max(
            row["actual_to_bound_ratio"] for row in trials
        ),
        "working_taper_geometry": (
            "taper radius=2.65, outer radius=2.75, H/L=1.2, t=1"
        ),
        "working_generation_criterion": working_generation_criterion,
        "working_sector_condition_number": working_condition_number,
        "working_allowable_relative_form_alpha": working_alpha,
        "working_sharp_Sobolev_mass_budget_over_nu": working_mass_budget,
        "working_mass_budget_matches_alpha_over_S3": bool(
            abs(
                working_mass_budget
                - working_alpha / sharp_sobolev_constant
            )
            < 1.0e-12
        ),
        "working_threshold_reproduces_closure_equality": bool(
            abs(closure_value - 1.0) < 1.0e-12
        ),
        "working_combined_budget_inequality": (
            "beta < d-(1+d)alpha, d=(C0^(-1/2)-1)/chi_sec"
        ),
        "working_combined_budget_intercept_d": combined_budget_intercept,
        "maximum_drift_sector_beta_if_alpha_zero": (
            maximum_drift_sector_beta_if_alpha_zero
        ),
        "maximum_drift_L3_over_nu_if_alpha_zero": (
            maximum_drift_L3_over_nu_if_alpha_zero
        ),
        "equal_share_alpha_and_beta": equal_alpha_beta,
        "equal_share_potential_L3_over_2_mass_over_nu": (
            equal_alpha_beta / sharp_sobolev_constant
        ),
        "equal_share_drift_L3_over_nu": (
            equal_alpha_beta / math.sqrt(sharp_sobolev_constant)
        ),
        "theorem_scope": (
            "rigorous conditional operator theorem; the displayed geometry "
            "constants are converged finite-element values, not enclosures"
        ),
        "remaining_drift_gate": (
            "an uncontrolled divergence-free first-order error is skew in "
            "energy but can still alter the boundary map; control it by a "
            "sector norm, physical hitting law, or include it in the fitted "
            "reference drift"
        ),
    }
    positive_checks = (
        result["skew_strength_does_not_enter_energy_bound"],
        result["all_random_sector_bounds_hold"],
        result["working_mass_budget_matches_alpha_over_S3"],
        result["working_threshold_reproduces_closure_equality"],
    )
    result["all_positive_sector_checks_pass"] = all(positive_checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
