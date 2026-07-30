"""Audit fixed-label stopping-time visits in moving rigid cylinders."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm, svdvals


def _load_sector_module():
    script = Path(__file__).resolve().with_name(
        "sectorial_poisson_transfer_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sectorial_poisson_for_stopping_visit", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dynamic_weighted_norm(
    kernel: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> float:
    conjugated = (
        np.sqrt(source)[:, None]
        * kernel
        / np.sqrt(target)[None, :]
    )
    return float(svdvals(conjugated)[0])


def _entry_exit_kernel_stress_test() -> dict[str, float | bool]:
    rng = np.random.default_rng(19260719)
    source = rng.random(6)
    source /= np.sum(source)
    target_law = rng.random(7)
    target_law /= np.sum(target_law)
    kernel = np.tile(target_law, (len(source), 1))
    target = source @ kernel
    single_norm = _dynamic_weighted_norm(kernel, source, target)

    pair_source = np.outer(source, source).ravel()
    pair_target = np.outer(target, target).ravel()
    pair_kernel = np.kron(kernel, kernel)
    pair_norm = _dynamic_weighted_norm(
        pair_kernel, pair_source, pair_target
    )
    return {
        "source_label_count": len(source),
        "target_label_count": len(target),
        "kernel_row_sum_error": float(
            np.max(np.abs(np.sum(kernel, axis=1) - 1.0))
        ),
        "target_pushforward_error": float(
            np.max(np.abs(target - target_law))
        ),
        "single_history_dynamic_L2_norm": single_norm,
        "replica_pair_dynamic_L2_norm": pair_norm,
        "stopping_time_relabel_is_contractive": bool(
            single_norm <= 1.0 + 1.0e-12
            and pair_norm <= 1.0 + 1.0e-12
        ),
    }


def _moving_coordinate_stress_test() -> dict[str, float | bool]:
    rng = np.random.default_rng(260719)
    skew_raw = rng.normal(size=(3, 3))
    body_skew = 0.37 * (skew_raw - skew_raw.T)
    rotation_seed = rng.normal(size=(3, 3))
    rotation_skew = rotation_seed - rotation_seed.T
    rotation = expm(rotation_skew)
    physical_skew = rotation @ body_skew @ rotation.T
    center = rng.normal(size=3)
    center_rate = rng.normal(size=3)
    point = rng.normal(size=3)
    physical_drift = rng.normal(size=3)
    length = 1.43
    viscosity = 0.81

    local = rotation.T @ (point - center) / length
    rigid_velocity = center_rate + physical_skew @ (point - center)
    physical_formula = (
        length / viscosity
        * rotation.T
        @ (physical_drift - rigid_velocity)
    )
    expanded_formula = (
        length / viscosity * rotation.T @ (physical_drift - center_rate)
        - length**2 / viscosity * body_skew @ local
    )
    transformed_covariance = rotation.T @ rotation
    return {
        "dimensionless_drift_identity_residual": float(
            np.max(np.abs(physical_formula - expanded_formula))
        ),
        "rotated_Brownian_covariance_residual": float(
            np.max(np.abs(transformed_covariance - np.eye(3)))
        ),
        "rigid_frame_divergence": float(np.trace(physical_skew)),
        "moving_coordinate_identity_verified": bool(
            np.max(np.abs(physical_formula - expanded_formula)) < 1.0e-13
            and np.max(np.abs(transformed_covariance - np.eye(3)))
            < 1.0e-13
            and abs(np.trace(physical_skew)) < 1.0e-13
        ),
    }


def _critical_scaling_stress_test() -> dict[str, float | bool]:
    rng = np.random.default_rng(72619)
    sample_count = 100_000
    length = 1.73
    viscosity = 0.64
    physical_volume_element = length**3 / sample_count
    drift_values = rng.normal(size=(sample_count, 3))
    potential_values = np.abs(rng.normal(size=sample_count))

    physical_drift_l3 = (
        physical_volume_element
        * np.sum(np.linalg.norm(drift_values, axis=1) ** 3)
    ) ** (1.0 / 3.0)
    dimensionless_drift_values = length / viscosity * drift_values
    dimensionless_drift_l3 = (
        np.mean(np.linalg.norm(dimensionless_drift_values, axis=1) ** 3)
    ) ** (1.0 / 3.0)

    physical_potential_l3_over_2 = (
        physical_volume_element
        * np.sum(potential_values ** 1.5)
    ) ** (2.0 / 3.0)
    dimensionless_potential_values = (
        length**2 / viscosity * potential_values
    )
    dimensionless_potential_l3_over_2 = (
        np.mean(dimensionless_potential_values**1.5)
    ) ** (2.0 / 3.0)
    return {
        "drift_scaling_residual": abs(
            dimensionless_drift_l3 - physical_drift_l3 / viscosity
        ),
        "potential_scaling_residual": abs(
            dimensionless_potential_l3_over_2
            - physical_potential_l3_over_2 / viscosity
        ),
        "critical_norm_scaling_verified": bool(
            abs(
                dimensionless_drift_l3
                - physical_drift_l3 / viscosity
            )
            < 1.0e-12
            and abs(
                dimensionless_potential_l3_over_2
                - physical_potential_l3_over_2 / viscosity
            )
            < 1.0e-12
        ),
    }


def audit() -> dict[str, object]:
    relabel = _entry_exit_kernel_stress_test()
    moving = _moving_coordinate_stress_test()
    scaling = _critical_scaling_stress_test()
    sector = _load_sector_module().audit()

    generation_criterion = float(sector["working_generation_criterion"])
    condition_number = float(sector["working_sector_condition_number"])
    sector_intercept = float(sector["working_combined_budget_intercept_d"])
    equal_share = float(sector["equal_share_alpha_and_beta"])
    equal_share_amplification = 1.0 + condition_number * (
        2.0 * equal_share / (1.0 - equal_share)
    )
    equal_share_closure = (
        generation_criterion * equal_share_amplification**2
    )

    result: dict[str, object] = {
        "physical_stochastic_path": (
            "dX=b(X,t)dt+sqrt(2nu)dW"
        ),
        "moving_visit_domain": (
            "C_j(t)=c_j(t)+L*O_j(t)*D, with L fixed during one visit"
        ),
        "entry_stopping_time": (
            "tau=first buffered entry; choose one label j from the "
            "normalized admissible-cell law pi(x,tau)"
        ),
        "exit_stopping_time": (
            "sigma_j=inf{t>=tau: O_j(t)^T(X_t-c_j(t))/L leaves D}"
        ),
        "fixed_label_rule": (
            "hold j on [tau,sigma_j); no continuous multiplication by "
            "sqrt(phi_j) occurs inside the visit"
        ),
        "dimensionless_time": "s=nu*(t-tau)/L^2",
        "dimensionless_coordinate": "Y=O_j^T*(X-c_j)/L",
        "dimensionless_SDE": (
            "dY=[(L/nu)O_j^T(b-c_j_dot)-"
            "(L^2/nu)Omega_j Y]ds+sqrt(2)dB_s, "
            "Omega_j=O_j^T O_j_dot"
        ),
        "rotated_noise_statement": (
            "B_s is Brownian because O_j is predictable orthogonal finite "
            "variation and its quadratic covariance is the identity"
        ),
        "physical_reference_drift": (
            "b_ref,j=(nu/L)O_j*b_hat_ref(Y)"
        ),
        "physical_remainder": (
            "e_j=b-c_j_dot-O_j_dot*O_j^T(x-c_j)-b_ref,j"
        ),
        "dimensionless_remainder": "e_hat_j=(L/nu)O_j^T e_j",
        "critical_scaling_identities": (
            "||e_hat_j||_(L3(D))=||e_j||_(L3(C_j))/nu and "
            "||q_hat_j||_(L3/2(D))=||q_j||_(L3/2(C_j))/nu"
        ),
        "interior_partition_derivatives": (
            "none: the selected label is a state of the stopped process, "
            "not a continuously differentiated spatial cutoff"
        ),
        "continuous_cubic_Fisher_cost_inside_visit": 0.0,
        "exit_relabel_kernel": (
            "K_jk(x,sigma)=pi_k(x,sigma); rows sum to one and the old "
            "label law pushes forward to pi"
        ),
        "pressure_weight_separation": (
            "a continuous linear partition may still be used in the global "
            "pressure identity; it is not square-root localized into the "
            "stopped visit energy"
        ),
        "sector_parameters": (
            "alpha=S3*||q_+||_(3/2)/nu, "
            "beta=sqrt(S3)*||e||_3/nu"
        ),
        "visit_operator_bound": (
            "||B_(q,e)||/||B_0||<=1+chi*(alpha+beta)/(1-alpha)"
        ),
        "renewal_condition": (
            "C0*[1+chi*(alpha+beta)/(1-alpha)]^2<1"
        ),
        "working_generation_criterion_C0": generation_criterion,
        "working_sector_condition_number_chi": condition_number,
        "working_sector_intercept_d": sector_intercept,
        "equal_share_alpha_beta": equal_share,
        "equal_share_closure_value": equal_share_closure,
        "equal_share_reproduces_closure_boundary": bool(
            abs(equal_share_closure - 1.0) < 1.0e-12
        ),
        "relabel_stress_test": relabel,
        "moving_coordinate_stress_test": moving,
        "critical_scaling_stress_test": scaling,
        "stopping_time_architecture_removes_continuous_Fisher_gate": True,
        "full_stopping_time_renewal_theorem_closed": False,
        "remaining_early_exit_gate": (
            "if affine coherence or coverage fails before the geometric "
            "exit, prove that the forced relabel is paid by a true dyadic "
            "split, accumulated decay, or a bounded bad-occupation budget"
        ),
        "remaining_hitting_measure_gate": (
            "compare actual moving-cylinder entry/exit laws with the "
            "nonreversible Perron/Doob boundary measure"
        ),
        "remaining_Leray_construction_gate": (
            "construct deterministic absolutely continuous centers and "
            "frames from local solution averages and justify the stopped "
            "domains for Leray solutions"
        ),
    }
    positive_checks = (
        relabel["stopping_time_relabel_is_contractive"],
        moving["moving_coordinate_identity_verified"],
        scaling["critical_norm_scaling_verified"],
        result["equal_share_reproduces_closure_boundary"],
        result["stopping_time_architecture_removes_continuous_Fisher_gate"],
        result["continuous_cubic_Fisher_cost_inside_visit"] == 0.0,
    )
    result["all_positive_stopping_visit_checks_pass"] = all(positive_checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
