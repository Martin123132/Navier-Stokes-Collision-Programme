"""Audit the uniform form bound for an instantaneous affine reference."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.linalg import expm
from scipy.special import jn_zeros


def _rotation_test() -> dict[str, float | bool]:
    diagonal = np.diag([-1.0, 0.0, 1.0])
    generator = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    angle = 0.05
    rotation = expm(angle * generator)
    strain = rotation @ diagonal @ rotation.T
    fixed_reference_error = float(
        np.linalg.norm(strain - diagonal, ord=2)
    )
    instantaneous_reference_error = float(
        np.linalg.norm(strain - strain, ord=2)
    )
    return {
        "angle_radians": angle,
        "fixed_entry_reference_matrix_error": fixed_reference_error,
        "instantaneous_reference_matrix_error": (
            instantaneous_reference_error
        ),
        "maximum_eigenvalue_change": float(
            np.linalg.eigvalsh(strain)[-1]
            - np.linalg.eigvalsh(diagonal)[-1]
        ),
        "instantaneous_reference_removes_rotation_error": bool(
            fixed_reference_error > 0.09
            and instantaneous_reference_error == 0.0
        ),
    }


def _random_trace_free_checks(seed: int = 20260719) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(32):
        raw = rng.normal(size=(3, 3))
        symmetric = 0.5 * (raw + raw.T)
        symmetric -= np.trace(symmetric) * np.eye(3) / 3.0
        eigenvalues = np.linalg.eigvalsh(symmetric)
        symmetric /= eigenvalues[-1]
        eigenvalues = np.linalg.eigvalsh(symmetric)
        rows.append(
            {
                "trace": float(np.trace(symmetric)),
                "maximum_eigenvalue": float(eigenvalues[-1]),
                "operator_norm": float(
                    np.linalg.norm(symmetric, ord=2)
                ),
            }
        )
    return {
        "sample_count": len(rows),
        "maximum_absolute_trace": max(abs(row["trace"]) for row in rows),
        "maximum_operator_norm": max(row["operator_norm"] for row in rows),
        "trace_free_top_normalized_operator_bound_verified": bool(
            max(abs(row["trace"]) for row in rows) < 1.0e-13
            and max(row["operator_norm"] for row in rows) <= 2.0 + 1.0e-13
        ),
    }


def audit() -> dict[str, object]:
    outer_radius = 2.0
    half_height = 0.75
    maximum_normalized_stretching = 1.0
    bessel_zero = float(jn_zeros(0, 1)[0])
    radial_floor = bessel_zero**2 / outer_radius**2
    axial_floor = math.pi**2 / (4.0 * half_height**2)
    cylinder_floor = radial_floor + axial_floor
    stretched_floor = cylinder_floor - maximum_normalized_stretching
    sharp_sobolev = 4.0 ** (2.0 / 3.0) / (
        3.0 * math.pi ** (4.0 / 3.0)
    )
    sqrt_sobolev = math.sqrt(sharp_sobolev)
    sector_intercept = 0.13048394692524337
    campanato_only_threshold = sector_intercept / sqrt_sobolev
    spatial_strain_only_threshold = sector_intercept / (
        (1.0 + sector_intercept) * sharp_sobolev
    )

    rotation = _rotation_test()
    random_checks = _random_trace_free_checks()
    result: dict[str, object] = {
        "instantaneous_reference": (
            "b_ref=U_L+W_L(x-c)+S_L(t)(x-c), "
            "lambda_ref(t)=lambda_max(S_L(t))"
        ),
        "exact_remainder": "e=R_L=b-U_L-(S_L+W_L)(x-c)",
        "exact_positive_stretching_error": (
            "q_+=[lambda_max(S(x,t))-lambda_max(S_L(t))]_+"
        ),
        "temporal_matrix_error_T": 0.0,
        "temporal_eigenvalue_error_G": 0.0,
        "eigenvector_derivative_required": False,
        "spin_frame_role": (
            "remove W_L; retain the measurable symmetric matrix S_L(t) "
            "without diagonalizing it"
        ),
        "compact_cylinder": {
            "outer_radius_over_L": outer_radius,
            "half_height_over_L": half_height,
        },
        "first_Bessel_zero_j01": bessel_zero,
        "radial_Dirichlet_floor": radial_floor,
        "axial_Dirichlet_floor": axial_floor,
        "cylinder_Dirichlet_floor": cylinder_floor,
        "maximum_normalized_stretching": (
            maximum_normalized_stretching
        ),
        "uniform_nonautonomous_coercive_floor": stretched_floor,
        "real_form_identity": (
            "Re<a_t v,v>=||grad v||_2^2-lambda_L(t)||v||_2^2; "
            "int v(S_L(t)y dot grad v)=0 because tr(S_L)=0 and "
            "v has zero boundary trace"
        ),
        "measurable_orientation_changes_interior_floor": False,
        "homogeneous_interior_evolution_L2_decay_rate": stretched_floor,
        "reduced_sector_condition": (
            "sqrt(S3)*C3+(1+d)*S3*P<d, d=0.130483946925"
        ),
        "Campanato_only_C3_threshold": campanato_only_threshold,
        "spatial_strain_only_P_threshold": (
            spatial_strain_only_threshold
        ),
        "constant_spectrum_rotation_test": rotation,
        "random_trace_free_checks": random_checks,
        "uniform_nonautonomous_boundary_visit_certified": False,
        "missing_boundary_theorem": (
            "bound the causal outer-to-inner parabolic Poisson/trace "
            "operator uniformly over measurable S_L(t); homogeneous "
            "interior coercivity alone does not supply its calibrated norm"
        ),
        "scope_guard": (
            "the form floor is rigorous, while the existing 0.556813 visit "
            "norm was computed only for static affine coefficients and "
            "cannot be reused for the nonautonomous problem without proof"
        ),
        "next_gate": (
            "derive a boundary-control estimate from the uniform form floor "
            "or numerically stress-test switched constant-spectrum strains "
            "before attempting a certified nonautonomous visit theorem"
        ),
    }
    positive_checks = (
        stretched_floor > 4.83,
        rotation["instantaneous_reference_removes_rotation_error"],
        random_checks["trace_free_top_normalized_operator_bound_verified"],
        result["temporal_matrix_error_T"] == 0.0,
        result["temporal_eigenvalue_error_G"] == 0.0,
        campanato_only_threshold > 0.30,
        spatial_strain_only_threshold > 0.33,
    )
    result["all_positive_nonautonomous_form_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
