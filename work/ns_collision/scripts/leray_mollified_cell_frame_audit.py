"""Audit mollified translating and spin-following cell frames."""

from __future__ import annotations

import json

import numpy as np
from scipy.linalg import expm


def _periodic_displacement(coordinate: np.ndarray, center: float) -> np.ndarray:
    return (coordinate - center + np.pi) % (2.0 * np.pi) - np.pi


def audit(grid_size: int = 48) -> dict[str, object]:
    grid = 2.0 * np.pi * np.arange(grid_size) / grid_size
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    center = np.array([0.37, 1.11, 2.03])
    dx = _periodic_displacement(x, center[0])
    dy = _periodic_displacement(y, center[1])
    dz = _periodic_displacement(z, center[2])
    concentration = 2.4
    unnormalized_weight = np.exp(
        concentration * (np.cos(dx) + np.cos(dy) + np.cos(dz))
    )
    weight = unnormalized_weight / np.sum(unnormalized_weight)

    velocity = np.array(
        [
            np.sin(z) + np.cos(y),
            np.sin(x) + np.cos(z),
            np.sin(y) + np.cos(x),
        ]
    )
    gradient = np.zeros((3, 3, grid_size, grid_size, grid_size))
    gradient[0, 1] = -np.sin(y)
    gradient[0, 2] = np.cos(z)
    gradient[1, 0] = np.cos(x)
    gradient[1, 2] = -np.sin(z)
    gradient[2, 0] = -np.sin(x)
    gradient[2, 1] = np.cos(y)

    mean_velocity = np.einsum("xyz,ixyz->i", weight, velocity)
    mean_gradient = np.einsum("xyz,ijxyz->ij", weight, gradient)
    weight_gradient = np.array(
        [
            -concentration * np.sin(dx) * weight,
            -concentration * np.sin(dy) * weight,
            -concentration * np.sin(dz) * weight,
        ]
    )
    integrated_by_parts_gradient = -np.einsum(
        "ixyz,jxyz->ij", velocity, weight_gradient
    )
    symmetric = 0.5 * (mean_gradient + mean_gradient.T)
    skew = 0.5 * (mean_gradient - mean_gradient.T)

    displacement = np.array([dx, dy, dz])
    weighted_displacement_mean = np.einsum(
        "xyz,ixyz->i", weight, displacement
    )
    centered_displacement = (
        displacement
        - weighted_displacement_mean[:, None, None, None]
    )
    linear_spin = np.einsum(
        "ij,jxyz->ixyz", skew, centered_displacement
    )
    residual = velocity - mean_velocity[:, None, None, None] - linear_spin
    weighted_residual_mean = np.einsum(
        "xyz,ixyz->i", weight, residual
    )
    residual_mean_gradient = mean_gradient - skew
    residual_mean_gradient_skew = 0.5 * (
        residual_mean_gradient - residual_mean_gradient.T
    )

    galilean_velocity = np.array([3.7, -1.9, 0.83])
    shifted_velocity = velocity + galilean_velocity[:, None, None, None]
    shifted_mean_velocity = np.einsum(
        "xyz,ixyz->i", weight, shifted_velocity
    )
    shifted_residual = (
        shifted_velocity
        - shifted_mean_velocity[:, None, None, None]
        - linear_spin
    )

    time_step = 0.37
    frame = expm(time_step * skew)
    frame_orthogonality_residual = np.max(
        np.abs(frame.T @ frame - np.eye(3))
    )
    determinant_residual = abs(float(np.linalg.det(frame)) - 1.0)

    eigenvalues = np.linalg.eigvalsh(symmetric)
    maximum_eigenvalue = float(eigenvalues[-1])
    spectrum_parameter = float(eigenvalues[1] / maximum_eigenvalue)
    normalized_spectrum_residual = float(
        max(
            abs(eigenvalues[0] / maximum_eigenvalue + 1.0 + spectrum_parameter),
            abs(eigenvalues[2] / maximum_eigenvalue - 1.0),
        )
    )

    result: dict[str, object] = {
        "mollified_center_velocity": (
            "U_L(c,t)=integral rho_L(x-c)u(x,t)dx"
        ),
        "mollified_velocity_gradient": (
            "A_L(c,t)=integral rho_L(x-c)grad(u)(x,t)dx="
            "-integral u tensor grad(rho_L)(x-c)dx"
        ),
        "mollified_spin": "W_L=(A_L-A_L^T)/2",
        "mollified_strain": "S_L=(A_L+A_L^T)/2",
        "center_frame_ODE": "c_dot=U_L(c,t), O_dot=W_L(c,t)O",
        "orthogonality_identity": (
            "d(O^T O)/dt=O^T(W_L^T+W_L)O=0"
        ),
        "Leray_well_posedness": (
            "for smooth rho, U_L and W_L are measurable in time and "
            "Lipschitz in c with bounds from ||u(t)||_2 and derivatives "
            "of rho_L; Caratheodory gives an absolutely continuous frame"
        ),
        "entry_affine_fit": (
            "at a visit entry time diagonalize the symmetric trace-free "
            "S_L once, initialize the tapered affine reference, and then "
            "let the frame follow W_L rather than differentiating strain "
            "eigenvectors"
        ),
        "frame_remainder": (
            "e=u-U_L-W_L(x-c)-b_ref; its weighted mean translation and "
            "weighted mean skew gradient vanish"
        ),
        "mean_velocity": mean_velocity.tolist(),
        "mean_gradient": mean_gradient.tolist(),
        "mean_gradient_trace": float(np.trace(mean_gradient)),
        "distributional_gradient_identity_residual": float(
            np.max(np.abs(mean_gradient - integrated_by_parts_gradient))
        ),
        "mollified_spin_trace": float(np.trace(skew)),
        "mollified_strain_trace": float(np.trace(symmetric)),
        "wrapped_periodic_displacement_mean_before_centering": (
            weighted_displacement_mean.tolist()
        ),
        "weighted_centered_displacement_mean": np.einsum(
            "xyz,ixyz->i", weight, centered_displacement
        ).tolist(),
        "weighted_remainder_mean": weighted_residual_mean.tolist(),
        "weighted_remainder_mean_skew_gradient_norm": float(
            np.linalg.norm(residual_mean_gradient_skew)
        ),
        "Galilean_mean_velocity_shift_residual": float(
            np.max(
                np.abs(
                    shifted_mean_velocity
                    - mean_velocity
                    - galilean_velocity
                )
            )
        ),
        "Galilean_remainder_invariance_residual": float(
            np.max(np.abs(shifted_residual - residual))
        ),
        "frame_orthogonality_residual": float(
            frame_orthogonality_residual
        ),
        "frame_determinant_residual": determinant_residual,
        "mollified_strain_eigenvalues": eigenvalues.tolist(),
        "normalized_affine_spectrum_parameter_t": spectrum_parameter,
        "normalized_affine_spectrum_residual": normalized_spectrum_residual,
        "normalized_spectrum_parameter_is_admissible": bool(
            -0.5 - 1.0e-12 <= spectrum_parameter <= 1.0 + 1.0e-12
        ),
        "mollified_gradient_is_trace_free": bool(
            abs(np.trace(mean_gradient)) < 1.0e-13
        ),
        "distributional_gradient_formula_verified": bool(
            np.max(np.abs(mean_gradient - integrated_by_parts_gradient))
            < 1.0e-12
        ),
        "translation_fit_is_Galilean_covariant": bool(
            np.max(np.abs(shifted_residual - residual)) < 1.0e-13
        ),
        "spin_following_frame_stays_in_SO3": bool(
            frame_orthogonality_residual < 1.0e-13
            and determinant_residual < 1.0e-13
        ),
        "weighted_translation_and_spin_are_removed": bool(
            np.max(np.abs(weighted_residual_mean)) < 1.0e-13
            and np.linalg.norm(residual_mean_gradient_skew) < 1.0e-13
        ),
        "pointwise_velocity_or_eigenframe_derivative_required": False,
        "full_Leray_stopping_visit_theorem_closed": False,
        "remaining_reference_drift_gate": (
            "relate the entry tensor S_L and chosen level L to the "
            "divergence-free tapered reference while keeping the residual "
            "inside the sector and abort-renewal budgets"
        ),
        "remaining_time_regularization_gate": (
            "write the weak-solution construction with a fixed smooth "
            "compact mollifier, stopping-time measurability, and stable "
            "limits from smooth approximants"
        ),
    }
    positive_checks = (
        result["normalized_spectrum_parameter_is_admissible"],
        result["mollified_gradient_is_trace_free"],
        result["distributional_gradient_formula_verified"],
        result["translation_fit_is_Galilean_covariant"],
        result["spin_following_frame_stays_in_SO3"],
        result["weighted_translation_and_spin_are_removed"],
        not result["pointwise_velocity_or_eigenframe_derivative_required"],
    )
    result["all_positive_mollified_frame_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
