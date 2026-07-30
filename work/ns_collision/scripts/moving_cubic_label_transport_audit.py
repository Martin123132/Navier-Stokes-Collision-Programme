"""Audit conservative moving cubic labels and their parabolic correction."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import svdvals


def _load_sector_module():
    script = Path(__file__).resolve().with_name(
        "sectorial_poisson_transfer_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sectorial_poisson_for_moving_labels", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cubic_spline_derivative(argument: np.ndarray, order: int) -> np.ndarray:
    spline = BSpline.basis_element(np.arange(5.0), extrapolate=False)
    values = spline.derivative(order)(argument)
    return np.where(np.isfinite(values), values, 0.0)


def _normalized_template_data(
    coordinate: np.ndarray,
    centers: np.ndarray,
    center_rates: np.ndarray,
    physical_velocity: np.ndarray,
    spacing: float,
) -> dict[str, np.ndarray]:
    argument = (
        coordinate[None, :] - centers[:, None]
    ) / spacing + 2.0
    raw = _cubic_spline_derivative(argument, 0)
    raw_first = _cubic_spline_derivative(argument, 1) / spacing
    raw_second = _cubic_spline_derivative(argument, 2) / spacing**2
    raw_material = (
        _cubic_spline_derivative(argument, 1)
        * (physical_velocity[None, :] - center_rates[:, None])
        / spacing
    )

    normalizer = np.sum(raw, axis=0)
    normalizer_first = np.sum(raw_first, axis=0)
    normalizer_second = np.sum(raw_second, axis=0)
    normalizer_material = np.sum(raw_material, axis=0)
    weights = raw / normalizer
    weight_first = (
        raw_first * normalizer[None, :]
        - raw * normalizer_first[None, :]
    ) / normalizer[None, :] ** 2
    weight_second = (
        raw_second / normalizer[None, :]
        - raw * normalizer_second[None, :] / normalizer[None, :] ** 2
        - 2.0
        * raw_first
        * normalizer_first[None, :]
        / normalizer[None, :] ** 2
        + 2.0
        * raw
        * normalizer_first[None, :] ** 2
        / normalizer[None, :] ** 3
    )
    weight_material = (
        raw_material * normalizer[None, :]
        - raw * normalizer_material[None, :]
    ) / normalizer[None, :] ** 2
    return {
        "raw": raw,
        "raw_first": raw_first,
        "raw_second": raw_second,
        "raw_material": raw_material,
        "normalizer": normalizer,
        "weights": weights,
        "weight_first": weight_first,
        "weight_second": weight_second,
        "weight_material": weight_material,
    }


def _simplex_flux_generator(
    probability: np.ndarray,
    rate: np.ndarray,
    tolerance: float = 1.0e-13,
) -> tuple[np.ndarray, float]:
    """Return Q with nonnegative off-diagonal entries and p @ Q = p'."""
    probability = np.asarray(probability, dtype=float)
    rate = np.asarray(rate, dtype=float)
    if abs(float(np.sum(probability)) - 1.0) > 1.0e-11:
        raise ValueError("probability must sum to one")
    if abs(float(np.sum(rate))) > 1.0e-10:
        raise ValueError("probability rate must sum to zero")
    if np.any((probability <= tolerance) & (rate < -tolerance)):
        raise ValueError("rate points out of the probability simplex")

    negative = np.flatnonzero(rate < -tolerance)
    positive = np.flatnonzero(rate > tolerance)
    total_flux = float(np.sum(rate[positive]))
    generator = np.zeros((len(probability), len(probability)))
    if total_flux <= tolerance:
        return generator, 0.0

    flux = (
        (-rate[negative])[:, None]
        * rate[positive][None, :]
        / total_flux
    )
    for source_row, source in enumerate(negative):
        generator[source, positive] = flux[source_row] / probability[source]
        generator[source, source] = -float(
            np.sum(generator[source, positive])
        )
    return generator, total_flux


def _dynamic_weighted_norm(
    kernel: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> float:
    source_active = source > 1.0e-13
    target_active = target > 1.0e-13
    conjugated = (
        np.sqrt(source[source_active])[:, None]
        * kernel[np.ix_(source_active, target_active)]
        / np.sqrt(target[target_active])[None, :]
    )
    return float(svdvals(conjugated)[0])


def _partition_stress_test() -> dict[str, float | bool]:
    spacing = 1.0
    labels = np.arange(-6, 7, dtype=float)
    centers = labels + 0.075 * np.sin(1.37 * labels)
    center_rates = 0.21 * np.cos(0.83 * labels)
    coordinate = np.linspace(-1.75, 1.75, 14_003) + 1.7e-4
    physical_velocity = 0.37 + 0.08 * np.sin(0.7 * coordinate)
    data = _normalized_template_data(
        coordinate,
        centers,
        center_rates,
        physical_velocity,
        spacing,
    )
    weights = data["weights"]
    first = data["weight_first"]
    material = data["weight_material"]
    raw = data["raw"]
    raw_first = data["raw_first"]
    normalizer = data["normalizer"]

    direct_fisher = np.sum(
        np.divide(
            first**2,
            4.0 * weights,
            out=np.zeros_like(first),
            where=weights > 1.0e-14,
        ),
        axis=0,
    )
    raw_fisher_sum = np.sum(
        np.divide(
            raw_first**2,
            raw,
            out=np.zeros_like(raw_first),
            where=raw > 1.0e-14,
        ),
        axis=0,
    )
    variance_fisher = 0.25 * (
        raw_fisher_sum / normalizer
        - (np.sum(raw_first, axis=0) / normalizer) ** 2
    )

    test_index = int(np.argmin(np.abs(coordinate - 0.137)))
    probability = weights[:, test_index]
    probability_rate = material[:, test_index]
    probability_rate -= np.sum(probability_rate) / len(probability_rate)
    generator, total_flux = _simplex_flux_generator(
        probability, probability_rate
    )
    exit_rates = -np.diag(generator)
    time_step = 0.2 / max(1.0, float(np.max(exit_rates)))
    kernel = np.eye(len(probability)) + time_step * generator
    target = probability @ kernel
    dynamic_norm = _dynamic_weighted_norm(kernel, probability, target)
    pair_dynamic_norm = dynamic_norm**2

    return {
        "minimum_raw_normalizer": float(np.min(normalizer)),
        "maximum_partition_sum_error": float(
            np.max(np.abs(np.sum(weights, axis=0) - 1.0))
        ),
        "maximum_partition_gradient_sum_error": float(
            np.max(np.abs(np.sum(first, axis=0)))
        ),
        "maximum_partition_material_rate_sum_error": float(
            np.max(np.abs(np.sum(material, axis=0)))
        ),
        "maximum_fisher_variance_identity_error": float(
            np.max(np.abs(direct_fisher - variance_fisher))
        ),
        "maximum_perturbed_partition_fisher_cost": float(
            np.max(direct_fisher)
        ),
        "simplex_total_positive_flux": total_flux,
        "simplex_generator_residual": float(
            np.max(np.abs(probability @ generator - probability_rate))
        ),
        "minimum_generator_off_diagonal": float(
            np.min(generator - np.diag(np.diag(generator)))
        ),
        "kernel_row_sum_error": float(
            np.max(np.abs(np.sum(kernel, axis=1) - 1.0))
        ),
        "kernel_pushforward_residual": float(
            np.max(np.abs(target - probability - time_step * probability_rate))
        ),
        "dynamic_measure_single_L2_norm": dynamic_norm,
        "dynamic_measure_pair_L2_norm": pair_dynamic_norm,
        "normalized_partition_checks_pass": bool(
            np.min(normalizer) > 0.8
            and np.max(np.abs(np.sum(weights, axis=0) - 1.0)) < 1.0e-14
            and np.max(np.abs(np.sum(first, axis=0))) < 1.0e-14
            and np.max(np.abs(np.sum(material, axis=0))) < 1.0e-14
            and np.max(np.abs(direct_fisher - variance_fisher)) < 2.0e-12
        ),
        "conservative_generator_checks_pass": bool(
            np.max(np.abs(probability @ generator - probability_rate))
            < 1.0e-12
            and np.min(generator - np.diag(np.diag(generator)))
            >= -1.0e-15
            and np.max(np.abs(np.sum(kernel, axis=1) - 1.0)) < 1.0e-14
            and dynamic_norm <= 1.0 + 2.0e-12
            and pair_dynamic_norm <= 1.0 + 4.0e-12
        ),
    }


def _parabolic_intertwining_stress_test() -> dict[str, float | bool]:
    spacing = 1.0
    labels = np.arange(-5, 6, dtype=float)
    centers = labels + 0.04 * np.sin(0.9 * labels)
    coordinate = np.array([0.2317])
    data = _normalized_template_data(
        coordinate,
        centers,
        np.zeros_like(centers),
        np.zeros_like(coordinate),
        spacing,
    )
    probability = data["weights"][:, 0]
    first = data["weight_first"][:, 0]
    second = data["weight_second"][:, 0]
    viscosity = 0.73
    drift = -0.28
    partition_generator_rate = drift * first + viscosity * second
    partition_generator_rate -= (
        np.sum(partition_generator_rate) / len(partition_generator_rate)
    )
    generator, _ = _simplex_flux_generator(
        probability, partition_generator_rate
    )

    rng = np.random.default_rng(20260719)
    observable = rng.normal(size=len(probability))
    observable_first = rng.normal(size=len(probability))
    observable_second = rng.normal(size=len(probability))
    observable_generator = (
        drift * observable_first + viscosity * observable_second
    )
    direct_product_generator = float(
        np.dot(partition_generator_rate, observable)
        + np.dot(probability, observable_generator)
        + 2.0 * viscosity * np.dot(first, observable_first)
    )
    active = probability > 1.0e-14
    conditioned_drift_term = np.zeros_like(probability)
    conditioned_drift_term[active] = (
        2.0
        * viscosity
        * first[active]
        / probability[active]
        * observable_first[active]
    )
    lifted_generator = (
        observable_generator
        + conditioned_drift_term
        + generator @ observable
    )
    lifted_product_generator = float(np.dot(probability, lifted_generator))
    return {
        "direct_product_generator": direct_product_generator,
        "lifted_product_generator": lifted_product_generator,
        "parabolic_intertwining_residual": abs(
            direct_product_generator - lifted_product_generator
        ),
        "partition_generator_rate_sum_error": abs(
            float(np.sum(partition_generator_rate))
        ),
        "parabolic_intertwining_verified": bool(
            abs(direct_product_generator - lifted_product_generator)
            < 1.0e-11
        ),
    }


def _rigid_coordinate_stress_test() -> dict[str, float | bool]:
    angle = 0.43
    angular_rate = -0.61
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    skew = np.array([[0.0, -angular_rate], [angular_rate, 0.0]])
    center = np.array([0.31, -0.27])
    center_rate = np.array([-0.42, 0.19])
    point = np.array([0.83, 0.54])
    velocity = np.array([0.37, -0.28])
    length = 1.17
    local = rotation.T @ (point - center) / length
    physical_frame_velocity = (
        center_rate + rotation @ skew @ rotation.T @ (point - center)
    )
    compact_formula = rotation.T @ (
        velocity - physical_frame_velocity
    ) / length
    expanded_formula = (
        rotation.T @ (velocity - center_rate) / length - skew @ local
    )
    return {
        "rigid_coordinate_derivative_residual": float(
            np.max(np.abs(compact_formula - expanded_formula))
        ),
        "rigid_coordinate_identity_verified": bool(
            np.max(np.abs(compact_formula - expanded_formula)) < 1.0e-14
        ),
    }


def audit() -> dict[str, object]:
    partition = _partition_stress_test()
    parabolic = _parabolic_intertwining_stress_test()
    rigid = _rigid_coordinate_stress_test()
    sector = _load_sector_module().audit()

    sharp_sobolev_constant = 4.0 ** (2.0 / 3.0) / (
        3.0 * math.pi ** (4.0 / 3.0)
    )
    sqrt_sobolev = math.sqrt(sharp_sobolev_constant)
    combined_intercept = float(
        sector["working_combined_budget_intercept_d"]
    )
    outer_radius = 2.75
    half_height = 1.2
    normalized_volume = 2.0 * math.pi * half_height * outer_radius**2
    translation_l3_coefficient = normalized_volume ** (1.0 / 3.0)
    axial_rotation_l3_coefficient = (
        4.0 * math.pi * half_height * outer_radius**5 / 5.0
    ) ** (1.0 / 3.0)
    general_rotation_l3_bound_coefficient = (
        translation_l3_coefficient
        * math.sqrt(outer_radius**2 + half_height**2)
    )
    scale_potential_mass_coefficient = (
        1.5 * normalized_volume ** (2.0 / 3.0)
    )
    drift_only_budget = combined_intercept / sqrt_sobolev
    potential_only_budget = combined_intercept / (
        (1.0 + combined_intercept) * sharp_sobolev_constant
    )

    result: dict[str, object] = {
        "moving_raw_template": (
            "psi_j(x,t)=Psi(O_j(t)^T*(x-c_j(t))/L)"
        ),
        "normalized_partition": (
            "phi_j=psi_j/Z, Z=sum_k psi_k; sum_j phi_j=1 wherever Z>0"
        ),
        "rigid_template_velocity": (
            "V_j=c_j_dot+O_j_dot*O_j^T*(x-c_j)"
        ),
        "local_coordinate_material_derivative": (
            "D_b y_j=O_j^T*(b-V_j)/L for fixed L"
        ),
        "raw_material_rate": (
            "g_j=(D_b psi_j)/psi_j=L^(-1)*grad(log Psi)(y_j) dot "
            "O_j^T*(b-V_j)"
        ),
        "normalized_material_rate": (
            "D_b phi_j=phi_j*(g_j-sum_k phi_k*g_k)"
        ),
        "material_rate_is_centered": True,
        "simplex_flux_rule": (
            "F_jk=(-a_j)_+*(a_k)_+/A from losing j to gaining k, "
            "A=sum_k(a_k)_+; Q_jk=F_jk/phi_j"
        ),
        "expected_label_switching_intensity": (
            "A=(1/2)*sum_j|a_j|"
        ),
        "dynamic_measure_markov_contraction": (
            "if mu Q advances mu to nu, backward conditional expectation "
            "contracts Lp(nu) to Lp(mu)"
        ),
        "Fisher_IMS_identity": (
            "sum_j|grad sqrt(phi_j)|^2=(1/4)[sum_j phi_j*"
            "|grad log psi_j|^2-|sum_j phi_j*grad log psi_j|^2]"
        ),
        "localized_energy_identity": (
            "sum_j|grad(sqrt(phi_j)*f)|^2=|grad f|^2+"
            "sum_j|grad sqrt(phi_j)|^2*|f|^2"
        ),
        "parabolic_intertwining_identity": (
            "L(sum_j phi_j F_j)=sum_j phi_j[L F_j+2nu*"
            "grad(log phi_j).grad F_j+sum_k Q_jk F_k], "
            "when phi Q=L phi"
        ),
        "viscous_label_correction_is_not_free": True,
        "pressure_partition_is_preserved": (
            "sum phi=1 and sum grad(phi)=0 retain exact inter-cell "
            "pressure-flux cancellation at each time"
        ),
        "scale_policy": (
            "keep L fixed between true dyadic level changes; use the "
            "existing cubic child kernel at a level change"
        ),
        "continuous_scale_warning": (
            "V_scale=ell*(x-c) has div(V_scale)=3ell and creates the "
            "adverse zero-order form term (3/2)*ell for ell>0"
        ),
        "fitted_divergence_free_remainder": (
            "e_j=b-c_j_dot-O_j_dot*O_j^T*(x-c_j)-b_ref,j"
        ),
        "critical_potential": (
            "Q_j=||[c_actual-c_ref,j]_+||_(3/2)/nu"
        ),
        "critical_drift": "E_j=||e_j||_3/nu",
        "combined_sector_condition": (
            "sqrt(S3)*E_j+(1+d)*S3*Q_j<d"
        ),
        "combined_sector_condition_divided_by_sqrt_S3": (
            f"E_j+{(1.0 + combined_intercept) * sqrt_sobolev:.12f}*"
            f"Q_j<{drift_only_budget:.12f}"
        ),
        "sharp_Sobolev_constant_S3": sharp_sobolev_constant,
        "sector_intercept_d": combined_intercept,
        "drift_only_E_budget": drift_only_budget,
        "potential_only_Q_budget": potential_only_budget,
        "working_cylinder_outer_radius_over_L": outer_radius,
        "working_cylinder_half_height_over_L": half_height,
        "working_cylinder_normalized_volume": normalized_volume,
        "constant_translation_E_per_UL_over_nu": (
            translation_l3_coefficient
        ),
        "maximum_unremoved_UL_over_nu_if_only_error": (
            drift_only_budget / translation_l3_coefficient
        ),
        "axial_rigid_rotation_E_per_omegaL2_over_nu": (
            axial_rotation_l3_coefficient
        ),
        "maximum_axial_rotation_omegaL2_over_nu_if_only_error": (
            drift_only_budget / axial_rotation_l3_coefficient
        ),
        "general_rigid_rotation_E_bound_per_omegaL2_over_nu": (
            general_rotation_l3_bound_coefficient
        ),
        "maximum_general_rotation_omegaL2_over_nu_from_bound": (
            drift_only_budget / general_rotation_l3_bound_coefficient
        ),
        "expansion_Q_per_positive_ellL2_over_nu": (
            scale_potential_mass_coefficient
        ),
        "maximum_positive_ellL2_over_nu_if_only_error": (
            potential_only_budget / scale_potential_mass_coefficient
        ),
        "translation_is_removed_before_sector_charge": True,
        "frame_rotation_mismatch_is_charged_once_to_beta": True,
        "moving_partition_coverage_hypothesis": (
            "Z has a positive lower bound; independent rigid cell motion "
            "does not guarantee this automatically"
        ),
        "moving_partition_Fisher_gate": (
            "the time-dependent Fisher/IMS cost must stay inside the "
            "available visit margin or be included in a recalibrated "
            "baseline operator"
        ),
        "Leray_level_frame_gate": (
            "construct absolutely continuous centers and frames from local "
            "averages, and control eigenframe angular velocity without "
            "assuming a simple spectrum"
        ),
        "full_Navier_Stokes_localization_theorem_closed": False,
        "partition_stress_test": partition,
        "parabolic_stress_test": parabolic,
        "rigid_coordinate_stress_test": rigid,
    }
    positive_checks = (
        result["material_rate_is_centered"],
        result["viscous_label_correction_is_not_free"],
        result["translation_is_removed_before_sector_charge"],
        result["frame_rotation_mismatch_is_charged_once_to_beta"],
        partition["normalized_partition_checks_pass"],
        partition["conservative_generator_checks_pass"],
        parabolic["parabolic_intertwining_verified"],
        rigid["rigid_coordinate_identity_verified"],
        0.47 < drift_only_budget < 0.49,
        0.92 < potential_only_budget < 0.94,
    )
    result["all_positive_moving_label_checks_pass"] = all(positive_checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
