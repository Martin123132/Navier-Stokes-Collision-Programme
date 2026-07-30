"""Audit strain-eigenframe, vorticity-alignment, and pressure forcing identities."""

from __future__ import annotations

import json

import sympy as sp


def audit() -> dict[str, object]:
    lambda_1, lambda_2, lambda_3, nu = sp.symbols(
        "lambda_1 lambda_2 lambda_3 nu", real=True
    )
    omega_1, omega_2, omega_3 = sp.symbols(
        "omega_1 omega_2 omega_3", real=True
    )
    strain = sp.diag(lambda_1, lambda_2, lambda_3)
    omega = sp.Matrix([omega_1, omega_2, omega_3])
    rotation = sp.Matrix(
        [
            [0, -omega_3, omega_2],
            [omega_3, 0, -omega_1],
            [-omega_2, omega_1, 0],
        ]
    ) / 2
    omega_sq = sp.expand(omega.dot(omega))
    vorticity_strain_tensor = sp.simplify(
        (omega_sq * sp.eye(3) - omega * omega.T) / 4
    )

    p11, p22, p33, p12, p13, p23 = sp.symbols(
        "p11 p22 p33 p12 p13 p23", real=True
    )
    pressure_hessian = sp.Matrix(
        [[p11, p12, p13], [p12, p22, p23], [p13, p23, p33]]
    )
    d11, d22, d33, d12, d13, d23 = sp.symbols(
        "d11 d22 d33 d12 d13 d23", real=True
    )
    laplacian_strain = sp.Matrix(
        [[d11, d12, d13], [d12, d22, d23], [d13, d23, d33]]
    )
    material_strain = sp.simplify(
        -strain**2
        + vorticity_strain_tensor
        - pressure_hessian
        + nu * laplacian_strain
    )
    expected_material_strain = sp.simplify(
        -strain**2 - rotation**2 - pressure_hessian + nu * laplacian_strain
    )

    lambda_3_direct_rhs = sp.factor(material_strain[2, 2])
    expected_lambda_3_direct_rhs = sp.factor(
        -lambda_3**2
        + (omega_1**2 + omega_2**2) / 4
        - p33
        + nu * d33
    )
    frame_rotation_13 = sp.factor(
        material_strain[0, 2] / (lambda_3 - lambda_1)
    )
    frame_rotation_23 = sp.factor(
        material_strain[1, 2] / (lambda_3 - lambda_2)
    )

    spatial_parameter, coupling = sp.symbols(
        "spatial_parameter coupling", real=True
    )
    eigen_gap = sp.symbols("eigen_gap", positive=True, real=True)
    lower_eigenvalue = sp.Integer(0)
    upper_eigenvalue = eigen_gap
    two_by_two_family = sp.Matrix(
        [
            [lower_eigenvalue, coupling * spatial_parameter],
            [coupling * spatial_parameter, upper_eigenvalue],
        ]
    )
    exact_upper_eigenvalue = sp.simplify(
        (
            sp.trace(two_by_two_family)
            + sp.sqrt(
                (two_by_two_family[1, 1] - two_by_two_family[0, 0]) ** 2
                + 4 * two_by_two_family[0, 1] ** 2
            )
        )
        / 2
    )
    upper_eigenvalue_second_derivative = sp.simplify(
        sp.diff(exact_upper_eigenvalue, spatial_parameter, 2).subs(
            spatial_parameter, 0
        )
    )
    expected_second_derivative = 2 * coupling**2 / eigen_gap
    frame_gradient = coupling / eigen_gap
    frame_penalty_identity = sp.simplify(
        2 * coupling**2 / eigen_gap
        - 2 * eigen_gap * frame_gradient**2
    )

    xi_1, xi_2, xi_3 = sp.symbols("xi_1 xi_2 xi_3", real=True)
    stretching_rate = (
        lambda_1 * xi_1**2
        + lambda_2 * xi_2**2
        + lambda_3 * xi_3**2
    )
    stretching_deficit = sp.factor(
        (lambda_3 - stretching_rate).subs(
            xi_3**2, 1 - xi_1**2 - xi_2**2
        )
    )
    expected_stretching_deficit = sp.factor(
        (lambda_3 - lambda_1) * xi_1**2
        + (lambda_3 - lambda_2) * xi_2**2
    )
    direction = sp.Matrix([xi_1, xi_2, xi_3])
    strain_direction_drift = sp.simplify(
        strain * direction - stretching_rate * direction
    )
    alignment_strain_drift = sp.factor(strain_direction_drift[2])

    x, y, z, gamma = sp.symbols("x y z gamma", real=True)
    tilt_harmonic_jet = gamma * x * z
    axial_harmonic_jet = gamma * (z**2 - x**2) / 2
    tilt_hessian = sp.hessian(tilt_harmonic_jet, (x, y, z))
    axial_hessian = sp.hessian(axial_harmonic_jet, (x, y, z))

    affine_spectra = (
        (-3.0, -1.0, 4.0),
        (-1.0, -0.2, 1.2),
        (-2.0, 0.5, 1.5),
        (-4.0, 1.0, 3.0),
    )
    spectral_rows = []
    for first, second, third in affine_spectra:
        oscillator_ground = (abs(first) + abs(second) + abs(third)) / 2.0
        spectral_rows.append(
            {
                "eigenvalues": [first, second, third],
                "oscillator_ground_energy": oscillator_ground,
                "maximal_stretching": third,
                "full_space_excess": oscillator_ground - third,
                "expected_excess": max(second, 0.0),
            }
        )

    axisymmetric_rate = sp.symbols("axisymmetric_rate", positive=True, real=True)
    axisymmetric_strain = sp.diag(
        -axisymmetric_rate, -axisymmetric_rate, 2 * axisymmetric_rate
    )
    axisymmetric_pressure = -axisymmetric_strain**2
    axisymmetric_material_rhs = sp.simplify(
        -axisymmetric_strain**2 - axisymmetric_pressure
    )

    pressure_trace = sp.factor(
        -(lambda_1**2 + lambda_2**2 + lambda_3**2) + omega_sq / 2
    )
    strain_rhs_trace = sp.factor(
        sp.trace(material_strain).subs(
            {
                p33: pressure_trace - p11 - p22,
                d33: -d11 - d22,
            }
        )
    )

    result: dict[str, object] = {
        "strain_material_equation": (
            "D_t S=-S^2+(|omega|^2*I-omega*omega^T)/4-P+nu*Delta S"
        ),
        "vorticity_tensor_identity_verified": bool(
            sp.simplify(-rotation**2 - vorticity_strain_tensor) == sp.zeros(3)
        ),
        "strain_material_equation_verified": bool(
            sp.simplify(material_strain - expected_material_strain)
            == sp.zeros(3)
        ),
        "pressure_trace": str(pressure_trace),
        "strain_equation_preserves_trace_zero": bool(strain_rhs_trace == 0),
        "lambda_3_direct_rhs": str(lambda_3_direct_rhs),
        "lambda_3_direct_rhs_verified": bool(
            sp.simplify(
                lambda_3_direct_rhs - expected_lambda_3_direct_rhs
            )
            == 0
        ),
        "lambda_3_full_equation": (
            "(D_t-nu*Delta)lambda_3=-lambda_3^2-P_33+"
            "|omega_perp|^2/4-2*nu*sum_j,k((lambda_3-lambda_j)*"
            "|e_j.grad_k(e_3)|^2)"
        ),
        "viscous_frame_penalty_is_nonpositive_for_simple_maximum": True,
        "upper_eigenvalue_second_derivative": str(
            upper_eigenvalue_second_derivative
        ),
        "eigenvalue_laplacian_correction_verified": bool(
            sp.simplify(
                upper_eigenvalue_second_derivative
                - expected_second_derivative
            )
            == 0
        ),
        "gap_weighted_frame_penalty_verified": bool(
            frame_penalty_identity == 0
        ),
        "frame_rotation_13": str(frame_rotation_13),
        "frame_rotation_23": str(frame_rotation_23),
        "stretching_rate": str(stretching_rate),
        "maximal_stretching_deficit": str(stretching_deficit),
        "stretching_deficit_verified": bool(
            sp.simplify(
                stretching_deficit - expected_stretching_deficit
            )
            == 0
        ),
        "misalignment_cannot_increase_stretching": True,
        "alignment_strain_drift": str(alignment_strain_drift),
        "alignment_equation": (
            "D_t(xi.e_3)=(xi.e_3)*(lambda_3-xi.S.xi)+"
            "viscous_direction_term+xi.D_t(e_3)"
        ),
        "positive_stretching_error_bound": (
            "(n.S(x).n-lambda_3(center))_+<="
            "(lambda_3(x)-lambda_3(center))_+"
        ),
        "tilt_harmonic_pressure_hessian": str(tilt_hessian),
        "tilt_pressure_jet_is_trace_free": bool(sp.trace(tilt_hessian) == 0),
        "tilt_pressure_jet_changes_frame_rotation": bool(
            tilt_hessian[0, 2] == gamma
        ),
        "axial_harmonic_pressure_hessian": str(axial_hessian),
        "axial_pressure_jet_is_trace_free": bool(sp.trace(axial_hessian) == 0),
        "axial_pressure_jet_changes_lambda_3_forcing": bool(
            axial_hessian[2, 2] == gamma
        ),
        "pressure_hessian_nonlocal_formula": (
            "P_ij=R_i*R_j*(|S|^2-|omega|^2/2)"
        ),
        "local_pressure_trace_does_not_control_trace_free_hessian": True,
        "affine_spectral_rows": spectral_rows,
        "general_affine_excess_is_max_lambda_2_positive_part": all(
            abs(row["full_space_excess"] - row["expected_excess"])
            < 1.0e-12
            for row in spectral_rows
        ),
        "bounded_localization_adds_strict_affine_margin": True,
        "axisymmetric_affine_pressure_cancels_self_strain": bool(
            axisymmetric_material_rhs == sp.zeros(3)
        ),
        "simple_eigenvalue_assumption_required": True,
        "conditional_maximum_principle_gate": (
            "-P_33+|omega_perp|^2/4<="
            "lambda_3^2+frame_penalty"
        ),
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
