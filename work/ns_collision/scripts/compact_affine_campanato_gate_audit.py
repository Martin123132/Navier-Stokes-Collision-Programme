"""Audit the compact full-affine Campanato remainder and failure classes."""

from __future__ import annotations

import json
import math

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.linalg import expm


def _mollifier_constants() -> dict[str, object]:
    radius = sp.symbols("radius", nonnegative=True)
    normalization = sp.Rational(3465, 512) / sp.pi
    profile = normalization * (1 - radius**2) ** 4
    normalization_integral = sp.simplify(
        4 * sp.pi * sp.integrate(radius**2 * profile, (radius, 0, 1))
    )
    coordinate_second_moment = sp.simplify(
        4
        * sp.pi
        / 3
        * sp.integrate(radius**4 * profile, (radius, 0, 1))
    )
    l2_squared = sp.simplify(
        4 * sp.pi * sp.integrate(radius**2 * profile**2, (radius, 0, 1))
    )
    first = sp.diff(profile, radius)
    gradient_l2_squared = sp.simplify(
        4 * sp.pi * sp.integrate(radius**2 * first**2, (radius, 0, 1))
    )
    hessian_l2_squared = sp.simplify(
        4
        * sp.pi
        * sp.integrate(
            radius**2
            * (
                sp.diff(profile, radius, 2) ** 2
                + 2 * (first / radius) ** 2
            ),
            (radius, 0, 1),
        )
    )
    support_ratio = sp.Rational(3, 4)
    return {
        "base_normalization": str(normalization),
        "normalization_integral": str(normalization_integral),
        "base_coordinate_second_moment": str(coordinate_second_moment),
        "base_L2_norm_squared": str(sp.factor(l2_squared)),
        "base_gradient_L2_norm_squared": str(
            sp.factor(gradient_l2_squared)
        ),
        "base_Hessian_L2_norm_squared": str(
            sp.factor(hessian_l2_squared)
        ),
        "support_radius_over_L": float(support_ratio),
        "scaled_coordinate_second_moment_over_L2": float(
            support_ratio**2 * coordinate_second_moment
        ),
        "scaled_L2_norm_times_L3_over_2": float(
            sp.sqrt(l2_squared) / support_ratio ** sp.Rational(3, 2)
        ),
        "scaled_gradient_L2_norm_times_L5_over_2": float(
            sp.sqrt(gradient_l2_squared)
            / support_ratio ** sp.Rational(5, 2)
        ),
        "scaled_Hessian_L2_norm_times_L7_over_2": float(
            sp.sqrt(hessian_l2_squared)
            / support_ratio ** sp.Rational(7, 2)
        ),
        "normalization_verified": bool(normalization_integral == 1),
        "second_moment_verified": bool(
            coordinate_second_moment == sp.Rational(1, 13)
        ),
    }


def _cylinder_geometry() -> dict[str, float]:
    outer_radius = 2.0
    half_height = 0.75
    normalized_volume = 2.0 * math.pi * half_height * outer_radius**2
    radial_third_moment = (
        4.0
        * math.pi
        / 5.0
        * quad(
            lambda axial: (
                (outer_radius**2 + axial**2) ** 2.5 - axial**5
            ),
            0.0,
            half_height,
            epsabs=1.0e-13,
        )[0]
    )
    return {
        "outer_radius_over_L": outer_radius,
        "half_height_over_L": half_height,
        "normalized_volume": normalized_volume,
        "constant_L3_over_2_coefficient": normalized_volume ** (2.0 / 3.0),
        "linear_matrix_L3_coefficient": radial_third_moment ** (1.0 / 3.0),
        "L2_to_L3_over_2_volume_coefficient": normalized_volume ** (1.0 / 6.0),
    }


def _decomposition_stress_test() -> dict[str, float | bool]:
    rng = np.random.default_rng(19072026)
    points = rng.normal(size=(20_000, 3))
    mean_velocity = rng.normal(size=3)
    raw = rng.normal(size=(3, 3))
    mean_gradient = raw - np.trace(raw) * np.eye(3) / 3.0
    spin = 0.5 * (mean_gradient - mean_gradient.T)
    strain = 0.5 * (mean_gradient + mean_gradient.T)
    reference_raw = rng.normal(size=(3, 3))
    reference_strain = 0.5 * (reference_raw + reference_raw.T)
    reference_strain -= (
        np.trace(reference_strain) * np.eye(3) / 3.0
    )
    nonlinear = np.column_stack(
        [
            0.03 * points[:, 1] * points[:, 2],
            -0.03 * points[:, 0] * points[:, 2],
            np.zeros(len(points)),
        ]
    )
    velocity = (
        mean_velocity[None, :]
        + points @ mean_gradient.T
        + nonlinear
    )
    campanato = velocity - mean_velocity[None, :] - points @ mean_gradient.T
    direct_remainder = (
        velocity
        - mean_velocity[None, :]
        - points @ spin.T
        - points @ reference_strain.T
    )
    decomposed_remainder = (
        campanato + points @ (strain - reference_strain).T
    )
    return {
        "maximum_remainder_decomposition_error": float(
            np.max(np.abs(direct_remainder - decomposed_remainder))
        ),
        "remainder_decomposition_verified": bool(
            np.max(np.abs(direct_remainder - decomposed_remainder))
            < 1.0e-13
        ),
    }


def _constant_spectrum_rotation_test(
    no_restart_temporal_threshold: float,
    split_paid_temporal_threshold: float,
) -> dict[str, float | bool]:
    diagonal = np.diag([-1.0, 0.0, 1.0])

    def rotated(angle: float) -> np.ndarray:
        generator = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        rotation = expm(angle * generator)
        return rotation @ diagonal @ rotation.T

    no_restart_angle = math.asin(no_restart_temporal_threshold / 2.0)
    split_paid_angle = math.asin(split_paid_temporal_threshold / 2.0)
    stress_angle = 0.05
    stressed = rotated(stress_angle)
    eigenvalue_residual = np.max(
        np.abs(np.linalg.eigvalsh(stressed) - np.linalg.eigvalsh(diagonal))
    )
    matrix_difference = float(np.linalg.norm(stressed - diagonal, ord=2))
    return {
        "constant_eigenvalue_rotation_angle_radians": stress_angle,
        "constant_eigenvalue_rotation_angle_degrees": math.degrees(stress_angle),
        "maximum_eigenvalue_change": float(
            np.linalg.eigvalsh(stressed)[-1]
            - np.linalg.eigvalsh(diagonal)[-1]
        ),
        "full_eigenvalue_residual": float(eigenvalue_residual),
        "dimensionless_matrix_drift": matrix_difference,
        "no_restart_drift_only_abort_angle_radians": no_restart_angle,
        "no_restart_drift_only_abort_angle_degrees": math.degrees(
            no_restart_angle
        ),
        "split_paid_drift_only_abort_angle_radians": split_paid_angle,
        "split_paid_drift_only_abort_angle_degrees": math.degrees(
            split_paid_angle
        ),
        "constant_spectrum_can_fail_without_envelope_growth": bool(
            matrix_difference > no_restart_temporal_threshold
            and abs(np.linalg.eigvalsh(stressed)[-1] - 1.0) < 1.0e-13
        ),
    }


def audit() -> dict[str, object]:
    mollifier = _mollifier_constants()
    geometry = _cylinder_geometry()
    decomposition = _decomposition_stress_test()
    sharp_sobolev = 4.0 ** (2.0 / 3.0) / (
        3.0 * math.pi ** (4.0 / 3.0)
    )
    sqrt_sobolev = math.sqrt(sharp_sobolev)
    no_restart_intercept = 0.13048394692524337
    split_paid_intercept = 0.041600122674968626
    matrix_coefficient = geometry["linear_matrix_L3_coefficient"]
    constant_coefficient = geometry["constant_L3_over_2_coefficient"]
    no_restart_temporal_drift_threshold = (
        no_restart_intercept / (sqrt_sobolev * matrix_coefficient)
    )
    split_paid_temporal_drift_threshold = (
        split_paid_intercept / (sqrt_sobolev * matrix_coefficient)
    )
    no_restart_combined_temporal_coefficient = (
        sqrt_sobolev * matrix_coefficient
        + (1.0 + no_restart_intercept)
        * sharp_sobolev
        * constant_coefficient
    )
    split_paid_combined_temporal_coefficient = (
        sqrt_sobolev * matrix_coefficient
        + (1.0 + split_paid_intercept)
        * sharp_sobolev
        * constant_coefficient
    )
    rotation = _constant_spectrum_rotation_test(
        no_restart_temporal_drift_threshold,
        split_paid_temporal_drift_threshold,
    )

    result: dict[str, object] = {
        "compact_mollifier": (
            "rho_(kL)(x)=(kL)^(-3)*[3465/(512pi)]*"
            "(1-|x|^2/(kL)^2)^4_+, k=3/4"
        ),
        "mollifier_regular_class": (
            "compact C3 kernel; sufficient for the displayed convolution "
            "and center/frame ODE, with C-infinity approximation available"
        ),
        "mollifier_constants": mollifier,
        "compact_cylinder_geometry": geometry,
        "instantaneous_affine_fit": (
            "U_L=int rho*b, A_L=int rho*grad(b)=S_L+W_L"
        ),
        "entry_reference_strain": (
            "S_ref(t)=O(t)O(tau)^T*S_L(tau)*O(tau)O(t)^T"
        ),
        "Campanato_remainder": (
            "R_L=b-U_L-A_L(x-c)"
        ),
        "exact_drift_remainder_decomposition": (
            "e=R_L+[S_L-S_ref](x-c)"
        ),
        "stretching_excess_decomposition": (
            "q_+<=[lambda_max(S(x))-lambda_max(S_L)]_+ + "
            "[lambda_max(S_L)-lambda_ref]_+"
        ),
        "critical_functionals": (
            "C3=||R_L||_3/nu, P=||[lambda_max(S)-"
            "lambda_max(S_L)]_+||_(3/2)/nu, "
            "T=L^2||S_L-S_ref||_op/nu, "
            "G=L^2[lambda_max(S_L)-lambda_ref]_+/nu"
        ),
        "drift_bound": (
            f"||e||_3/nu<=C3+{matrix_coefficient:.12f}*T"
        ),
        "potential_bound": (
            f"||q_+||_(3/2)/nu<=P+{constant_coefficient:.12f}*G"
        ),
        "no_restart_compact_sector_condition": (
            "sqrt(S3)*C3+(1+d)*S3*P+sqrt(S3)*C_r*T+"
            "(1+d)*S3*C_V*G<d, d=0.130483946925"
        ),
        "split_paid_compact_sector_condition": (
            "same inequality with d=0.041600122675"
        ),
        "no_restart_temporal_drift_only_T_threshold": (
            no_restart_temporal_drift_threshold
        ),
        "split_paid_temporal_drift_only_T_threshold": (
            split_paid_temporal_drift_threshold
        ),
        "no_restart_combined_T_equals_G_coefficient": (
            no_restart_combined_temporal_coefficient
        ),
        "no_restart_combined_T_equals_G_threshold": (
            no_restart_intercept / no_restart_combined_temporal_coefficient
        ),
        "split_paid_combined_T_equals_G_coefficient": (
            split_paid_combined_temporal_coefficient
        ),
        "split_paid_combined_T_equals_G_threshold": (
            split_paid_intercept / split_paid_combined_temporal_coefficient
        ),
        "Leray_finiteness": (
            "u in L6_loc makes C3 finite; grad(u) in L2 and finite cylinder "
            "volume make P finite by L2-to-L3/2 Holder; finiteness does not "
            "imply any displayed smallness"
        ),
        "decomposition_stress_test": decomposition,
        "constant_spectrum_rotation_test": rotation,
        "amplitude_envelope_does_not_classify_all_failures": bool(
            rotation["constant_spectrum_can_fail_without_envelope_growth"]
        ),
        "full_bad_occupation_bound_closed": False,
        "consequence": (
            "true level splits can pay only failures tied to the dyadic "
            "amplitude threshold; orientation, spectral-shape, and spatial "
            "Campanato failures require an occupation estimate or a "
            "time-dependent reference theorem"
        ),
        "next_gate": (
            "derive a dynamic L2 occupation estimate for the C3/P/T bad "
            "set under the compact full-affine stopped diffusion, or extend "
            "the visit theorem to a slowly varying affine reference"
        ),
    }
    positive_checks = (
        mollifier["normalization_verified"],
        mollifier["second_moment_verified"],
        decomposition["remainder_decomposition_verified"],
        rotation["constant_spectrum_can_fail_without_envelope_growth"],
        geometry["linear_matrix_L3_coefficient"] < 4.1,
        no_restart_temporal_drift_threshold > 0.07,
        split_paid_temporal_drift_threshold > 0.02,
    )
    result["all_positive_compact_campanato_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
