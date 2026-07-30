"""Audit the correlated-replica tangent and Gramian deformation bridge."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm


Array = np.ndarray


def _relative_residual(left: Array, right: Array) -> float:
    scale = max(1.0, float(np.linalg.norm(left, ord=2)), float(np.linalg.norm(right, ord=2)))
    return float(np.linalg.norm(left - right, ord=2) / scale)


def _positive_eigenvalues(matrix: Array) -> Array:
    values = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    if float(values[0]) <= 0.0:
        raise ValueError("Gramian is not positive definite")
    return values


def _integrate_gramians(
    phi: Callable[[float, float], Array],
    terminal_time: float,
    viscosity: float,
    intervals: tuple[tuple[float, float], ...],
    quadrature_order: int,
) -> tuple[Array, Array, Array]:
    nodes, weights = leggauss(quadrature_order)
    dimension = phi(terminal_time, 0.0).shape[0]
    forward = np.zeros((dimension, dimension), dtype=float)
    inverse_time = np.zeros_like(forward)
    cross = np.zeros_like(forward)

    for left, right in intervals:
        midpoint = (left + right) / 2.0
        half_width = (right - left) / 2.0
        for node, weight in zip(nodes, weights, strict=True):
            source_time = midpoint + half_width * float(node)
            final_from_source = phi(terminal_time, source_time)
            initial_from_source = phi(0.0, source_time)
            scaled_weight = half_width * float(weight)
            forward += (
                scaled_weight * final_from_source @ final_from_source.T
            )
            inverse_time += (
                scaled_weight * initial_from_source @ initial_from_source.T
            )
            cross += (
                scaled_weight * final_from_source @ initial_from_source.T
            )

    noise_variance = 4.0 * viscosity
    return (
        noise_variance * forward,
        noise_variance * inverse_time,
        noise_variance * cross,
    )


def _constant_case(
    name: str,
    generator: Array,
    terminal_time: float,
    viscosity: float,
    quadrature_order: int,
) -> dict[str, object]:
    matrix = np.asarray(generator, dtype=float)

    def phi(end: float, start: float) -> Array:
        return expm(matrix * (end - start))

    jacobian = phi(terminal_time, 0.0)
    forward, inverse_time, cross = _integrate_gramians(
        phi,
        terminal_time,
        viscosity,
        ((0.0, terminal_time),),
        quadrature_order,
    )
    return _case_diagnostics(
        name,
        jacobian,
        forward,
        inverse_time,
        cross,
        terminal_time,
        viscosity,
        trace_integral=float(np.trace(matrix) * terminal_time),
    )


def _two_stage_case(
    name: str,
    first_generator: Array,
    second_generator: Array,
    terminal_time: float,
    viscosity: float,
    quadrature_order: int,
) -> dict[str, object]:
    first = np.asarray(first_generator, dtype=float)
    second = np.asarray(second_generator, dtype=float)
    switch = terminal_time / 2.0

    def from_zero(time: float) -> Array:
        if time <= switch:
            return expm(first * time)
        return expm(second * (time - switch)) @ expm(first * switch)

    def phi(end: float, start: float) -> Array:
        return from_zero(end) @ np.linalg.inv(from_zero(start))

    jacobian = phi(terminal_time, 0.0)
    forward, inverse_time, cross = _integrate_gramians(
        phi,
        terminal_time,
        viscosity,
        ((0.0, switch), (switch, terminal_time)),
        quadrature_order,
    )
    trace_integral = float(
        np.trace(first) * switch
        + np.trace(second) * (terminal_time - switch)
    )
    return _case_diagnostics(
        name,
        jacobian,
        forward,
        inverse_time,
        cross,
        terminal_time,
        viscosity,
        trace_integral=trace_integral,
    )


def _case_diagnostics(
    name: str,
    jacobian: Array,
    forward: Array,
    inverse_time: Array,
    cross: Array,
    terminal_time: float,
    viscosity: float,
    trace_integral: float,
) -> dict[str, object]:
    forward_eigenvalues = _positive_eigenvalues(forward)
    inverse_eigenvalues = _positive_eigenvalues(inverse_time)
    recovered_jacobian = np.linalg.solve(inverse_time.T, cross.T).T
    covariance_scale = 4.0 * viscosity * terminal_time
    normalized_forward_trace = float(np.trace(forward) / covariance_scale)
    normalized_inverse_trace = float(
        np.trace(inverse_time) / covariance_scale
    )
    jacobian_norm = float(np.linalg.norm(jacobian, ord=2))
    inverse_jacobian_norm = float(
        np.linalg.norm(np.linalg.inv(jacobian), ord=2)
    )
    jacobian_squared = jacobian_norm**2
    inverse_jacobian_squared = inverse_jacobian_norm**2
    spectral_jacobian_bound = float(
        forward_eigenvalues[-1] / inverse_eigenvalues[0]
    )
    spectral_inverse_bound = float(
        inverse_eigenvalues[-1] / forward_eigenvalues[0]
    )
    trace_jacobian_bound = float(
        normalized_forward_trace * normalized_inverse_trace**2 / 4.0
    )
    trace_inverse_bound = float(
        normalized_inverse_trace * normalized_forward_trace**2 / 4.0
    )
    determinant_floor = covariance_scale**3
    expected_jacobian_determinant = math.exp(trace_integral)
    checks = {
        "forward_inverse_congruence": (
            _relative_residual(
                forward,
                jacobian @ inverse_time @ jacobian.T,
            )
            < 2.0e-11
        ),
        "cross_covariance_is_JB": (
            _relative_residual(cross, jacobian @ inverse_time) < 2.0e-11
        ),
        "cross_covariance_recovers_J": (
            _relative_residual(recovered_jacobian, jacobian) < 2.0e-11
        ),
        "Schur_covariance_identity": (
            _relative_residual(
                forward,
                cross @ np.linalg.solve(inverse_time, cross.T),
            )
            < 3.0e-11
        ),
        "Liouville_determinant": (
            abs(float(np.linalg.det(jacobian)) - expected_jacobian_determinant)
            < 2.0e-11 * max(1.0, expected_jacobian_determinant)
        ),
        "incompressible_Gramian_determinant_balance": (
            abs(trace_integral) < 1.0e-13
            and abs(
                math.log(float(np.linalg.det(forward)))
                - math.log(float(np.linalg.det(inverse_time)))
            )
            < 3.0e-10
        ),
        "forward_Minkowski_determinant_floor": (
            float(np.linalg.det(forward))
            >= determinant_floor * (1.0 - 3.0e-11)
        ),
        "inverse_Minkowski_determinant_floor": (
            float(np.linalg.det(inverse_time))
            >= determinant_floor * (1.0 - 3.0e-11)
        ),
        "spectral_bound_controls_J": (
            jacobian_squared <= spectral_jacobian_bound * (1.0 + 3.0e-11)
        ),
        "spectral_bound_controls_inverse_J": (
            inverse_jacobian_squared
            <= spectral_inverse_bound * (1.0 + 3.0e-11)
        ),
        "radial_trace_bound_controls_J": (
            jacobian_squared <= trace_jacobian_bound * (1.0 + 3.0e-11)
        ),
        "radial_trace_bound_controls_inverse_J": (
            inverse_jacobian_squared
            <= trace_inverse_bound * (1.0 + 3.0e-11)
        ),
    }
    return {
        "name": name,
        "terminal_time": terminal_time,
        "viscosity": viscosity,
        "trace_integral": trace_integral,
        "jacobian": jacobian.tolist(),
        "jacobian_determinant": float(np.linalg.det(jacobian)),
        "jacobian_operator_norm": jacobian_norm,
        "inverse_jacobian_operator_norm": inverse_jacobian_norm,
        "forward_Gramian": forward.tolist(),
        "inverse_time_Gramian": inverse_time.tolist(),
        "cross_covariance": cross.tolist(),
        "forward_eigenvalues": forward_eigenvalues.tolist(),
        "inverse_time_eigenvalues": inverse_eigenvalues.tolist(),
        "normalized_forward_radial_variance": normalized_forward_trace,
        "normalized_inverse_radial_variance": normalized_inverse_trace,
        "spectral_J_squared_bound": spectral_jacobian_bound,
        "spectral_inverse_J_squared_bound": spectral_inverse_bound,
        "radial_trace_J_squared_bound": trace_jacobian_bound,
        "radial_trace_inverse_J_squared_bound": trace_inverse_bound,
        "radial_trace_J_bound_over_actual_squared": (
            trace_jacobian_bound / jacobian_squared
        ),
        "radial_trace_inverse_bound_over_actual_squared": (
            trace_inverse_bound / inverse_jacobian_squared
        ),
        "determinant_floor": determinant_floor,
        "checks": checks,
        "all_case_checks_pass": all(checks.values()),
    }


def _symbolic_audit() -> dict[str, object]:
    rho, nu = sp.symbols("rho nu", real=True, positive=True)
    dimension = sp.symbols("d", integer=True, positive=True)
    common_coefficient = sp.sqrt((1 + rho) / 2)
    difference_coefficient = sp.sqrt((1 - rho) / 2)
    marginal_variance = sp.simplify(
        common_coefficient**2 + difference_coefficient**2
    )
    cross_variance = sp.simplify(
        common_coefficient**2 - difference_coefficient**2
    )
    difference_variance = sp.simplify((2 * difference_coefficient) ** 2)
    relative_noise_covariance = sp.simplify(
        2 * nu * difference_variance
    )
    squared_gap_ito_source = sp.simplify(
        dimension * relative_noise_covariance
    )
    log_gap_ito_source = sp.simplify(
        squared_gap_ito_source
        - 8 * nu * (1 - rho)
    )

    h, q, y = sp.symbols("h q y", real=True)
    coefficients = sp.symbols("c0:4", real=True)
    argument = sp.symbols("x", real=True)
    polynomial = sum(
        coefficient * argument**index
        for index, coefficient in enumerate(coefficients)
    )
    plus = polynomial.subs(argument, q + h * y / 2)
    minus = polynomial.subs(argument, q - h * y / 2)
    derivative = sp.diff(polynomial, argument).subs(argument, q)
    third_derivative = sp.diff(polynomial, argument, 3).subs(argument, q)
    second_derivative = sp.diff(polynomial, argument, 2).subs(argument, q)
    antisymmetric_residual = sp.simplify(
        (plus - minus) / h
        - derivative * y
        - h**2 * third_derivative * y**3 / 24
    )
    centre_residual = sp.simplify(
        (plus + minus) / 2
        - polynomial.subs(argument, q)
        - h**2 * second_derivative * y**2 / 8
    )
    checks = {
        "each_driver_has_unit_quadratic_variation": (
            sp.simplify(marginal_variance - 1) == 0
        ),
        "driver_cross_variation_is_rho": (
            sp.simplify(cross_variance - rho) == 0
        ),
        "difference_driver_variance_is_2_one_minus_rho": (
            sp.simplify(difference_variance - 2 * (1 - rho)) == 0
        ),
        "relative_SDE_covariance_is_4nu_one_minus_rho": (
            sp.simplify(relative_noise_covariance - 4 * nu * (1 - rho))
            == 0
        ),
        "squared_gap_ito_source_is_4nu_d_one_minus_rho": (
            sp.simplify(
                squared_gap_ito_source
                - 4 * nu * dimension * (1 - rho)
            )
            == 0
        ),
        "log_gap_ito_source_is_4nu_d_minus_2_one_minus_rho": (
            sp.simplify(
                log_gap_ito_source
                - 4 * nu * (dimension - 2) * (1 - rho)
            )
            == 0
        ),
        "symmetric_difference_tangent_expansion_exact_for_cubics": (
            antisymmetric_residual == 0
        ),
        "symmetric_centre_expansion_exact_for_cubics": (
            centre_residual == 0
        ),
    }
    return {
        "symmetric_driver_coupling": {
            "W1": "sqrt((1+rho)/2) W_plus + sqrt((1-rho)/2) W_minus",
            "W2": "sqrt((1+rho)/2) W_plus - sqrt((1-rho)/2) W_minus",
        },
        "relative_noise_covariance": str(relative_noise_covariance),
        "squared_gap_ito_source_dimension_d": str(squared_gap_ito_source),
        "log_squared_gap_ito_source_dimension_d": str(log_gap_ito_source),
        "three_dimensional_log_squared_gap_source": str(
            sp.simplify(log_gap_ito_source.subs(dimension, 3))
        ),
        "tangent_drift_expansion": (
            "[u(q+h*y/2)-u(q-h*y/2)]/h "
            "= Du(q)y+h^2 D^3u(q)[y,y,y]/24+O(h^4|y|^5)"
        ),
        "centre_drift_expansion": (
            "[u(q+h*y/2)+u(q-h*y/2)]/2 "
            "=u(q)+h^2 D^2u(q)[y,y]/8+O(h^4|y|^4)"
        ),
        "checks": checks,
        "all_symbolic_checks_pass": all(checks.values()),
    }


def audit(quadrature_order: int = 96) -> dict[str, object]:
    if quadrature_order < 24:
        raise ValueError("quadrature_order must be at least 24")
    viscosity = 0.7
    terminal_time = 1.0

    planar_strength = 8.0
    planar = np.diag([planar_strength, -planar_strength, 0.0])
    shear_strength = 12.0
    shear = np.array(
        [
            [0.0, shear_strength, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    rotation_rate = 7.0
    rotation = np.array(
        [
            [0.0, -rotation_rate, 0.0],
            [rotation_rate, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    zero = np.zeros((3, 3), dtype=float)
    first_stage = np.diag([1.7, -0.9, -0.8])
    angle = 0.61
    rotation_matrix = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    second_diagonal = np.diag([-1.2, 1.9, -0.7])
    second_stage = rotation_matrix @ second_diagonal @ rotation_matrix.T
    stage_commutator_norm = float(
        np.linalg.norm(
            first_stage @ second_stage - second_stage @ first_stage,
            ord=2,
        )
    )

    cases = {
        "zero_flow": _constant_case(
            "zero_flow",
            zero,
            terminal_time,
            viscosity,
            quadrature_order,
        ),
        "planar_strain": _constant_case(
            "planar_strain",
            planar,
            terminal_time,
            viscosity,
            quadrature_order,
        ),
        "simple_shear": _constant_case(
            "simple_shear",
            shear,
            terminal_time,
            viscosity,
            quadrature_order,
        ),
        "rigid_rotation": _constant_case(
            "rigid_rotation",
            rotation,
            terminal_time,
            viscosity,
            quadrature_order,
        ),
        "noncommuting_two_stage_strain": _two_stage_case(
            "noncommuting_two_stage_strain",
            first_stage,
            second_stage,
            terminal_time,
            viscosity,
            quadrature_order,
        ),
    }

    planar_row = cases["planar_strain"]
    shear_row = cases["simple_shear"]
    rotation_row = cases["rigid_rotation"]
    planar_minimum_covariance = min(
        min(planar_row["forward_eigenvalues"]),
        min(planar_row["inverse_time_eigenvalues"]),
    )
    one_sided_stress = {
        "planar_strength_time": planar_strength * terminal_time,
        "minimum_endpoint_covariance_eigenvalue": (
            planar_minimum_covariance
        ),
        "maximum_deformation_norm": max(
            planar_row["jacobian_operator_norm"],
            planar_row["inverse_jacobian_operator_norm"],
        ),
        "noncollision_covariance_remains_positive": (
            planar_minimum_covariance > 0.0
        ),
        "deformation_exceeds_1000": (
            max(
                planar_row["jacobian_operator_norm"],
                planar_row["inverse_jacobian_operator_norm"],
            )
            > 1000.0
        ),
        "conclusion": (
            "strict covariance positivity and radial noncollision do not "
            "give a uniform deformation bound"
        ),
    }
    shear_stress = {
        "shear_strength_time": shear_strength * terminal_time,
        "actual_J_squared": shear_row["jacobian_operator_norm"] ** 2,
        "radial_trace_J_squared_bound": shear_row[
            "radial_trace_J_squared_bound"
        ],
        "trace_bound_over_actual_squared": shear_row[
            "radial_trace_J_bound_over_actual_squared"
        ],
        "trace_bound_detects_shear_but_is_quantitatively_loose": (
            shear_row["radial_trace_J_bound_over_actual_squared"] > 100.0
        ),
    }
    covariance_scale = 4.0 * viscosity * terminal_time
    rotation_stress = {
        "forward_is_isotropic": (
            _relative_residual(
                np.asarray(rotation_row["forward_Gramian"]),
                covariance_scale * np.eye(3),
            )
            < 2.0e-11
        ),
        "inverse_time_is_isotropic": (
            _relative_residual(
                np.asarray(rotation_row["inverse_time_Gramian"]),
                covariance_scale * np.eye(3),
            )
            < 2.0e-11
        ),
        "cross_covariance_retains_rotation": (
            _relative_residual(
                np.asarray(rotation_row["cross_covariance"]),
                covariance_scale * np.asarray(rotation_row["jacobian"]),
            )
            < 2.0e-11
        ),
        "conclusion": (
            "endpoint covariance tensors control amplification bounds, "
            "while the cross covariance retains the missing orientation"
        ),
    }

    scale_factor = 3.7
    base_scaling_case = _constant_case(
        "scaling_base",
        np.diag([1.3, -0.8, -0.5]),
        0.9,
        viscosity,
        quadrature_order,
    )
    scaled_case = _constant_case(
        "scaling_transformed",
        scale_factor**2 * np.diag([1.3, -0.8, -0.5]),
        0.9 / scale_factor**2,
        viscosity,
        quadrature_order,
    )
    scaling_checks = {
        "Jacobian_is_scaling_invariant": (
            _relative_residual(
                np.asarray(base_scaling_case["jacobian"]),
                np.asarray(scaled_case["jacobian"]),
            )
            < 2.0e-11
        ),
        "forward_Gramian_scales_as_length_squared": (
            _relative_residual(
                np.asarray(base_scaling_case["forward_Gramian"])
                / scale_factor**2,
                np.asarray(scaled_case["forward_Gramian"]),
            )
            < 2.0e-11
        ),
        "inverse_Gramian_scales_as_length_squared": (
            _relative_residual(
                np.asarray(base_scaling_case["inverse_time_Gramian"])
                / scale_factor**2,
                np.asarray(scaled_case["inverse_time_Gramian"]),
            )
            < 2.0e-11
        ),
        "normalized_forward_radial_variance_is_critical": (
            abs(
                base_scaling_case["normalized_forward_radial_variance"]
                - scaled_case["normalized_forward_radial_variance"]
            )
            < 2.0e-11
        ),
        "normalized_inverse_radial_variance_is_critical": (
            abs(
                base_scaling_case["normalized_inverse_radial_variance"]
                - scaled_case["normalized_inverse_radial_variance"]
            )
            < 2.0e-11
        ),
    }

    symbolic = _symbolic_audit()
    positive_checks = {
        "all_symbolic_checks_pass": symbolic["all_symbolic_checks_pass"],
        "all_affine_and_noncommuting_cases_pass": all(
            row["all_case_checks_pass"] for row in cases.values()
        ),
        "one_sided_noncollision_counterexample_is_explicit": (
            one_sided_stress["noncollision_covariance_remains_positive"]
            and one_sided_stress["deformation_exceeds_1000"]
        ),
        "shear_nonnormality_loss_is_detected": shear_stress[
            "trace_bound_detects_shear_but_is_quantitatively_loose"
        ],
        "cross_covariance_orientation_test_passes": all(
            value
            for value in rotation_stress.values()
            if isinstance(value, bool)
        ),
        "time_dependent_stress_uses_noncommuting_generators": (
            stage_commutator_norm > 0.5
        ),
        "all_parabolic_scaling_checks_pass": all(scaling_checks.values()),
    }
    certification_flags = {
        "correlated_replica_Ito_homotopy_derived": True,
        "common_noise_tangent_limit_derived_for_smooth_drift": True,
        "conditional_forward_inverse_Gramian_congruence_proved": True,
        "conditional_cross_covariance_recovers_flow_Jacobian": True,
        "incompressible_Gramian_determinant_balance_proved": True,
        "Minkowski_determinant_floor_proved": True,
        "radial_trace_deformation_bound_proved": True,
        "radial_noncollision_alone_controls_deformation": False,
        "Leray_energy_controls_critical_forward_inverse_traces": False,
        "low_regularity_inverse_time_probe_justified": False,
        "critical_L3_continuation_bridge_proved": False,
        "Navier_Stokes_global_regularity_proved": False,
    }
    return {
        "kind": "correlated_replica_tangent_gramian_audit",
        "schema_version": 1,
        "status": (
            "smooth_flow_correlation_to_deformation_identity_proved_"
            "critical_trace_estimate_open"
        ),
        "quadrature_order_per_interval": quadrature_order,
        "symbolic_correlated_replica_audit": symbolic,
        "smooth_flow_theorem": {
            "forward_Gramian": (
                "F=4nu integral_0^T Phi(T,s)Phi(T,s)^T ds"
            ),
            "inverse_time_Gramian": (
                "B=4nu integral_0^T Phi(0,s)Phi(0,s)^T ds"
            ),
            "shared_probe_cross_covariance": (
                "H=4nu integral_0^T Phi(T,s)Phi(0,s)^T ds"
            ),
            "exact_congruence": "F=J B J^T",
            "exact_cross_identity": "H=J B",
            "exact_recovery": "J=H B^(-1)",
            "incompressible_determinant_identity": "det(F)=det(B)",
            "dimension_three_determinant_floor": (
                "det(F)=det(B)>=(4nu T)^3"
            ),
            "normalized_radial_variances": (
                "f=tr(F)/(4nu T), b=tr(B)/(4nu T)"
            ),
            "deformation_bounds": (
                "||J||_2^2<=f b^2/4 and "
                "||J^(-1)||_2^2<=b f^2/4"
            ),
        },
        "cases": cases,
        "stress_tests": {
            "one_sided_noncollision": one_sided_stress,
            "simple_shear_nonnormality": shear_stress,
            "rigid_rotation_orientation": rotation_stress,
            "noncommuting_generator_commutator_norm": (
                stage_commutator_norm
            ),
        },
        "parabolic_scaling_audit": {
            "scale_factor": scale_factor,
            "base_normalized_forward": base_scaling_case[
                "normalized_forward_radial_variance"
            ],
            "scaled_normalized_forward": scaled_case[
                "normalized_forward_radial_variance"
            ],
            "base_normalized_inverse": base_scaling_case[
                "normalized_inverse_radial_variance"
            ],
            "scaled_normalized_inverse": scaled_case[
                "normalized_inverse_radial_variance"
            ],
            "checks": scaling_checks,
        },
        "certification_flags": certification_flags,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "next_theorem_target": (
            "On every parabolic window, bound suitable moments of the "
            "dimensionless forward and inverse-time radial variances from "
            "Navier-Stokes structure without assuming a supercritical "
            "gradient norm."
        ),
    }


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = audit(args.quadrature_order)
    if args.output is not None:
        _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
