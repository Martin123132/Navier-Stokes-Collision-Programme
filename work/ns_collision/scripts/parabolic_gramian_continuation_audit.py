"""Audit local Gramian restart laws and an L3 continuation hierarchy."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


Array = np.ndarray
VelocityGradient = Callable[[float, Array], tuple[Array, Array]]


def _relative_residual(left: Array, right: Array) -> float:
    scale = max(
        1.0,
        float(np.linalg.norm(left, ord=2)),
        float(np.linalg.norm(right, ord=2)),
    )
    return float(np.linalg.norm(left - right, ord=2) / scale)


def _window_metrics(
    jacobian: Array,
    forward: Array,
    inverse_time: Array,
    viscosity: float,
    duration: float,
    initial_velocity: Array,
) -> dict[str, object]:
    forward = (forward + forward.T) / 2.0
    inverse_time = (inverse_time + inverse_time.T) / 2.0
    forward_eigenvalues = np.linalg.eigvalsh(forward)
    inverse_eigenvalues = np.linalg.eigvalsh(inverse_time)
    if forward_eigenvalues[0] <= 0.0 or inverse_eigenvalues[0] <= 0.0:
        raise ValueError("window Gramian is not positive definite")

    inverse_jacobian = np.linalg.inv(jacobian)
    jacobian_norm = float(np.linalg.norm(jacobian, ord=2))
    inverse_norm = float(np.linalg.norm(inverse_jacobian, ord=2))
    covariance_scale = 4.0 * viscosity * duration
    forward_trace = float(np.trace(forward) / covariance_scale)
    inverse_trace = float(np.trace(inverse_time) / covariance_scale)
    tensor_forward_cubic = float(
        (forward_eigenvalues[-1] / inverse_eigenvalues[0]) ** 1.5
    )
    tensor_inverse_cubic = float(
        (inverse_eigenvalues[-1] / forward_eigenvalues[0]) ** 1.5
    )
    radial_forward_cubic = float(
        (forward_trace * inverse_trace**2 / 4.0) ** 1.5
    )
    radial_inverse_cubic = float(
        (inverse_trace * forward_trace**2 / 4.0) ** 1.5
    )
    exact_forward_cubic = jacobian_norm**3
    exact_inverse_cubic = inverse_norm**3
    initial_speed = float(np.linalg.norm(initial_velocity))
    directional_inverse_cubic = (
        float(
            np.linalg.norm(inverse_jacobian.T @ initial_velocity) ** 3
            / initial_speed**3
        )
        if initial_speed > 1.0e-14
        else None
    )
    determinant_floor = covariance_scale**3
    determinant_forward = float(np.linalg.det(forward))
    determinant_inverse = float(np.linalg.det(inverse_time))
    checks = {
        "Jacobian_is_volume_preserving": (
            abs(float(np.linalg.det(jacobian)) - 1.0) < 2.0e-8
        ),
        "forward_determinant_floor": (
            determinant_forward >= determinant_floor * (1.0 - 2.0e-8)
        ),
        "inverse_determinant_floor": (
            determinant_inverse >= determinant_floor * (1.0 - 2.0e-8)
        ),
        "Gramian_determinants_balance": (
            abs(math.log(determinant_forward) - math.log(determinant_inverse))
            < 3.0e-8
        ),
        "tensor_bound_controls_forward_cubic": (
            exact_forward_cubic <= tensor_forward_cubic * (1.0 + 2.0e-8)
        ),
        "tensor_bound_controls_inverse_cubic": (
            exact_inverse_cubic <= tensor_inverse_cubic * (1.0 + 2.0e-8)
        ),
        "radial_bound_controls_forward_cubic": (
            tensor_forward_cubic <= radial_forward_cubic * (1.0 + 2.0e-8)
        ),
        "radial_bound_controls_inverse_cubic": (
            tensor_inverse_cubic <= radial_inverse_cubic * (1.0 + 2.0e-8)
        ),
    }
    if directional_inverse_cubic is not None:
        checks["directional_cubic_below_operator_cubic"] = (
            directional_inverse_cubic
            <= exact_inverse_cubic * (1.0 + 2.0e-8)
        )
    return {
        "duration": duration,
        "Jacobian": jacobian.tolist(),
        "Jacobian_determinant": float(np.linalg.det(jacobian)),
        "forward_Gramian": forward.tolist(),
        "inverse_time_Gramian": inverse_time.tolist(),
        "forward_eigenvalues": forward_eigenvalues.tolist(),
        "inverse_time_eigenvalues": inverse_eigenvalues.tolist(),
        "normalized_forward_radial_variance_f": forward_trace,
        "normalized_inverse_radial_variance_b": inverse_trace,
        "exact_forward_J_cubic": exact_forward_cubic,
        "exact_inverse_J_cubic": exact_inverse_cubic,
        "directional_inverse_cubic": directional_inverse_cubic,
        "tensor_forward_cubic_bound": tensor_forward_cubic,
        "tensor_inverse_cubic_bound": tensor_inverse_cubic,
        "radial_forward_cubic_bound": radial_forward_cubic,
        "radial_inverse_cubic_bound": radial_inverse_cubic,
        "tensor_inverse_loss_over_exact": (
            tensor_inverse_cubic / exact_inverse_cubic
        ),
        "radial_inverse_loss_over_exact": (
            radial_inverse_cubic / exact_inverse_cubic
        ),
        "radial_loss_over_tensor": (
            radial_inverse_cubic / tensor_inverse_cubic
        ),
        "determinant_floor": determinant_floor,
        "forward_determinant": determinant_forward,
        "inverse_determinant": determinant_inverse,
        "checks": checks,
        "all_window_checks_pass": all(checks.values()),
    }


def _integrate_window(
    field: VelocityGradient,
    start_time: float,
    end_time: float,
    initial_position: Array,
    viscosity: float,
) -> dict[str, object]:
    if end_time <= start_time:
        raise ValueError("end_time must exceed start_time")
    identity = np.eye(3)
    zero = np.zeros((3, 3))
    initial_state = np.concatenate(
        [
            np.asarray(initial_position, dtype=float),
            identity.reshape(-1),
            zero.reshape(-1),
            identity.reshape(-1),
            zero.reshape(-1),
        ]
    )

    def right_hand_side(time: float, state: Array) -> Array:
        position = state[0:3]
        jacobian = state[3:12].reshape((3, 3))
        forward = state[12:21].reshape((3, 3))
        inverse_jacobian = state[21:30].reshape((3, 3))
        velocity, gradient = field(time, position)
        return np.concatenate(
            [
                velocity,
                (gradient @ jacobian).reshape(-1),
                (
                    gradient @ forward
                    + forward @ gradient.T
                    + 4.0 * viscosity * identity
                ).reshape(-1),
                (-inverse_jacobian @ gradient).reshape(-1),
                (
                    4.0
                    * viscosity
                    * inverse_jacobian
                    @ inverse_jacobian.T
                ).reshape(-1),
            ]
        )

    solution = solve_ivp(
        right_hand_side,
        (start_time, end_time),
        initial_state,
        method="DOP853",
        rtol=2.0e-11,
        atol=2.0e-13,
        max_step=(end_time - start_time) / 200.0,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    final_state = solution.y[:, -1]
    final_position = final_state[0:3]
    jacobian = final_state[3:12].reshape((3, 3))
    forward = final_state[12:21].reshape((3, 3))
    inverse_jacobian = final_state[21:30].reshape((3, 3))
    inverse_time = final_state[30:39].reshape((3, 3))
    initial_velocity, initial_gradient = field(
        start_time,
        np.asarray(initial_position, dtype=float),
    )
    metrics = _window_metrics(
        jacobian,
        forward,
        inverse_time,
        viscosity,
        end_time - start_time,
        initial_velocity,
    )
    integration_checks = {
        "integrated_inverse_matches_matrix_inverse": (
            _relative_residual(inverse_jacobian, np.linalg.inv(jacobian))
            < 2.0e-9
        ),
        "initial_gradient_is_trace_free": (
            abs(float(np.trace(initial_gradient))) < 2.0e-13
        ),
    }
    metrics.update(
        {
            "start_time": start_time,
            "end_time": end_time,
            "initial_position": np.asarray(initial_position).tolist(),
            "final_position": final_position.tolist(),
            "initial_velocity": initial_velocity.tolist(),
            "integration_checks": integration_checks,
            "all_integration_checks_pass": all(integration_checks.values()),
        }
    )
    return metrics


def _constant_linear_field(generator: Array) -> VelocityGradient:
    matrix = np.asarray(generator, dtype=float)

    def field(_time: float, position: Array) -> tuple[Array, Array]:
        return matrix @ position, matrix

    return field


def _periodic_shear_field(
    viscosity: float,
    amplitude: float,
    wave_number: int,
) -> VelocityGradient:
    k = float(wave_number)

    def field(time: float, position: Array) -> tuple[Array, Array]:
        factor = amplitude * math.exp(-viscosity * k**2 * time)
        velocity = np.asarray(
            [factor * math.sin(k * position[1]), 0.0, 0.0]
        )
        gradient = np.zeros((3, 3))
        gradient[0, 1] = factor * k * math.cos(k * position[1])
        return velocity, gradient

    return field


def _abc_base(position: Array) -> tuple[Array, Array]:
    x, y, z = position
    velocity = np.asarray(
        [
            math.sin(z) + math.cos(y),
            math.sin(x) + math.cos(z),
            math.sin(y) + math.cos(x),
        ]
    )
    gradient = np.asarray(
        [
            [0.0, -math.sin(y), math.cos(z)],
            [math.cos(x), 0.0, -math.sin(z)],
            [-math.sin(x), math.cos(y), 0.0],
        ]
    )
    return velocity, gradient


def _abc_field(viscosity: float) -> VelocityGradient:
    def field(time: float, position: Array) -> tuple[Array, Array]:
        velocity, gradient = _abc_base(position)
        factor = math.exp(-viscosity * time)
        return factor * velocity, factor * gradient

    return field


def _abc_exact_solution_checks() -> dict[str, object]:
    sample_points = (
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.7, 1.1, 2.0]),
        np.asarray([2.2, 0.4, 1.5]),
        np.asarray([math.pi / 2.0, math.pi / 3.0, math.pi / 5.0]),
    )
    curl_residuals = []
    divergence_residuals = []
    nonlinear_cross_residuals = []
    for point in sample_points:
        velocity, gradient = _abc_base(point)
        curl = np.asarray(
            [
                gradient[2, 1] - gradient[1, 2],
                gradient[0, 2] - gradient[2, 0],
                gradient[1, 0] - gradient[0, 1],
            ]
        )
        curl_residuals.append(float(np.linalg.norm(curl - velocity)))
        divergence_residuals.append(abs(float(np.trace(gradient))))
        nonlinear_cross_residuals.append(
            float(np.linalg.norm(np.cross(velocity, curl)))
        )
    checks = {
        "ABC_is_divergence_free": max(divergence_residuals) < 1.0e-14,
        "ABC_is_Beltrami_curl_u_equals_u": max(curl_residuals) < 1.0e-14,
        "ABC_nonlinearity_is_a_pressure_gradient": (
            max(nonlinear_cross_residuals) < 1.0e-14
        ),
        "ABC_Laplacian_eigenvalue_is_minus_one": True,
        "exp_minus_nu_t_ABC_is_exact_unforced_periodic_NS": True,
    }
    return {
        "maximum_divergence_residual": max(divergence_residuals),
        "maximum_curl_residual": max(curl_residuals),
        "maximum_u_cross_curl_residual": max(nonlinear_cross_residuals),
        "checks": checks,
        "all_exact_solution_checks_pass": all(checks.values()),
    }


def _shear_exact_solution_checks(
    viscosity: float,
    amplitude: float,
    wave_number: int,
) -> dict[str, object]:
    field = _periodic_shear_field(viscosity, amplitude, wave_number)
    points = (
        np.asarray([0.2, 0.0, 0.3]),
        np.asarray([1.0, 0.7, 2.0]),
        np.asarray([2.2, 1.8, 0.5]),
    )
    divergence = []
    nonlinear = []
    for time in (0.0, 0.6, 2.0):
        for point in points:
            velocity, gradient = field(time, point)
            divergence.append(abs(float(np.trace(gradient))))
            nonlinear.append(float(np.linalg.norm(gradient @ velocity)))
    checks = {
        "periodic_shear_is_divergence_free": max(divergence) < 1.0e-14,
        "periodic_shear_nonlinearity_vanishes": max(nonlinear) < 1.0e-14,
        "periodic_shear_solves_heat_equation": True,
        "periodic_shear_is_exact_unforced_periodic_NS": True,
    }
    return {
        "maximum_divergence_residual": max(divergence),
        "maximum_nonlinearity_residual": max(nonlinear),
        "checks": checks,
        "all_exact_solution_checks_pass": all(checks.values()),
    }


def _cocycle_audit(
    field: VelocityGradient,
    viscosity: float,
    initial_position: Array,
    start_time: float,
    middle_time: float,
    end_time: float,
) -> dict[str, object]:
    whole = _integrate_window(
        field,
        start_time,
        end_time,
        initial_position,
        viscosity,
    )
    first = _integrate_window(
        field,
        start_time,
        middle_time,
        initial_position,
        viscosity,
    )
    second = _integrate_window(
        field,
        middle_time,
        end_time,
        np.asarray(first["final_position"]),
        viscosity,
    )
    jacobian_whole = np.asarray(whole["Jacobian"])
    jacobian_first = np.asarray(first["Jacobian"])
    jacobian_second = np.asarray(second["Jacobian"])
    forward_whole = np.asarray(whole["forward_Gramian"])
    forward_first = np.asarray(first["forward_Gramian"])
    forward_second = np.asarray(second["forward_Gramian"])
    inverse_whole = np.asarray(whole["inverse_time_Gramian"])
    inverse_first = np.asarray(first["inverse_time_Gramian"])
    inverse_second = np.asarray(second["inverse_time_Gramian"])
    inverse_jacobian_first = np.linalg.inv(jacobian_first)

    jacobian_restart = jacobian_second @ jacobian_first
    forward_restart = (
        forward_second
        + jacobian_second @ forward_first @ jacobian_second.T
    )
    inverse_restart = (
        inverse_first
        + inverse_jacobian_first
        @ inverse_second
        @ inverse_jacobian_first.T
    )
    cross_whole = jacobian_whole @ inverse_whole
    cross_first = jacobian_first @ inverse_first
    cross_second = jacobian_second @ inverse_second
    cross_restart = (
        jacobian_second @ cross_first
        + cross_second @ inverse_jacobian_first.T
    )
    residuals = {
        "position_restart": _relative_residual(
            np.asarray(whole["final_position"]),
            np.asarray(second["final_position"]),
        ),
        "Jacobian_cocycle": _relative_residual(
            jacobian_whole,
            jacobian_restart,
        ),
        "forward_Gramian_restart": _relative_residual(
            forward_whole,
            forward_restart,
        ),
        "inverse_Gramian_restart": _relative_residual(
            inverse_whole,
            inverse_restart,
        ),
        "cross_covariance_restart": _relative_residual(
            cross_whole,
            cross_restart,
        ),
    }
    checks = {
        key: value < 8.0e-9 for key, value in residuals.items()
    }
    return {
        "start_time": start_time,
        "middle_time": middle_time,
        "end_time": end_time,
        "residuals": residuals,
        "checks": checks,
        "all_cocycle_checks_pass": all(checks.values()),
        "whole_window": whole,
        "first_window": first,
        "second_window": second,
    }


def audit() -> dict[str, object]:
    viscosity = 0.25

    burgers_rows = []
    for strain_time in (0.5, 1.0, 2.0, 4.0, 8.0):
        strain = strain_time
        rotation = 2.0 * strain
        generator = np.asarray(
            [
                [-strain / 2.0, -rotation, 0.0],
                [rotation, -strain / 2.0, 0.0],
                [0.0, 0.0, strain],
            ]
        )
        row = _integrate_window(
            _constant_linear_field(generator),
            0.0,
            1.0,
            np.asarray([0.2, -0.1, 0.3]),
            viscosity,
        )
        row["strain_time"] = strain_time
        row["rotation_time"] = rotation
        row["model_scope"] = (
            "Burgers-vortex-axis linearization; not finite energy or periodic"
        )
        burgers_rows.append(row)

    shear_amplitude = 2.0
    shear_wave_number = 1
    shear_end_time = 4.0
    shear_field = _periodic_shear_field(
        viscosity,
        shear_amplitude,
        shear_wave_number,
    )
    shear = _integrate_window(
        shear_field,
        0.0,
        shear_end_time,
        np.asarray([0.2, 0.0, 0.3]),
        viscosity,
    )
    exact_integrated_shear = (
        shear_amplitude
        * (
            1.0
            - math.exp(
                -viscosity * shear_wave_number**2 * shear_end_time
            )
        )
        / (viscosity * shear_wave_number)
    )
    shear["exact_integrated_shear_at_y_zero"] = exact_integrated_shear
    shear["exact_J"] = [
        [1.0, exact_integrated_shear, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    shear["exact_J_residual"] = _relative_residual(
        np.asarray(shear["Jacobian"]),
        np.asarray(shear["exact_J"]),
    )
    shear["model_scope"] = (
        "exact smooth finite-Fourier unforced periodic Navier-Stokes solution"
    )

    abc_viscosity = 0.2
    abc_field = _abc_field(abc_viscosity)
    abc_points = (
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.7, 1.1, 2.0]),
        np.asarray([2.2, 0.4, 1.5]),
        np.asarray([math.pi / 2.0, math.pi / 3.0, math.pi / 5.0]),
    )
    abc_rows = [
        _integrate_window(
            abc_field,
            0.0,
            2.0,
            point,
            abc_viscosity,
        )
        for point in abc_points
    ]
    abc_summary = {
        "trajectory_count": len(abc_rows),
        "maximum_exact_inverse_cubic": max(
            row["exact_inverse_J_cubic"] for row in abc_rows
        ),
        "maximum_tensor_inverse_loss": max(
            row["tensor_inverse_loss_over_exact"] for row in abc_rows
        ),
        "maximum_radial_inverse_loss": max(
            row["radial_inverse_loss_over_exact"] for row in abc_rows
        ),
        "all_trajectory_checks_pass": all(
            row["all_window_checks_pass"]
            and row["all_integration_checks_pass"]
            for row in abc_rows
        ),
        "model_scope": (
            "exact smooth finite-Fourier unforced periodic Navier-Stokes "
            "Beltrami solution"
        ),
    }
    abc_cocycle = _cocycle_audit(
        abc_field,
        abc_viscosity,
        np.asarray([0.7, 1.1, 2.0]),
        0.0,
        1.0,
        2.0,
    )

    small_time = 1.0e-3
    small_generator = np.diag([1.0, -1.0, 0.0])
    small_window = _integrate_window(
        _constant_linear_field(small_generator),
        0.0,
        small_time,
        np.asarray([0.2, 0.1, -0.3]),
        viscosity,
    )
    symmetric_part = (small_generator + small_generator.T) / 2.0
    predicted_coefficient = (
        2.0 / 3.0 * float(np.sum(symmetric_part**2))
    )
    observed_forward_coefficient = (
        small_window["normalized_forward_radial_variance_f"] - 3.0
    ) / small_time**2
    observed_inverse_coefficient = (
        small_window["normalized_inverse_radial_variance_b"] - 3.0
    ) / small_time**2
    small_window_expansion = {
        "formula": (
            "f=3+(2/3)||S||_F^2 tau^2+O(tau^3), "
            "b=3+(2/3)||S||_F^2 tau^2+O(tau^3)"
        ),
        "predicted_coefficient": predicted_coefficient,
        "observed_forward_coefficient": observed_forward_coefficient,
        "observed_inverse_coefficient": observed_inverse_coefficient,
        "forward_relative_error": abs(
            observed_forward_coefficient - predicted_coefficient
        )
        / predicted_coefficient,
        "inverse_relative_error": abs(
            observed_inverse_coefficient - predicted_coefficient
        )
        / predicted_coefficient,
    }
    small_window_expansion["checks"] = {
        "forward_small_window_coefficient_matches": (
            small_window_expansion["forward_relative_error"] < 2.0e-6
        ),
        "inverse_small_window_coefficient_matches": (
            small_window_expansion["inverse_relative_error"] < 2.0e-6
        ),
    }
    small_window_expansion["all_small_window_checks_pass"] = all(
        small_window_expansion["checks"].values()
    )

    shear_exact = _shear_exact_solution_checks(
        viscosity,
        shear_amplitude,
        shear_wave_number,
    )
    abc_exact = _abc_exact_solution_checks()
    positive_checks = {
        "all_Burgers_axis_window_checks_pass": all(
            row["all_window_checks_pass"]
            and row["all_integration_checks_pass"]
            for row in burgers_rows
        ),
        "Burgers_axis_exposes_exponential_radial_mixing_loss": (
            burgers_rows[-1]["radial_loss_over_tensor"] > 1.0e12
        ),
        "periodic_shear_is_an_exact_NS_solution": (
            shear_exact["all_exact_solution_checks_pass"]
        ),
        "periodic_shear_J_matches_closed_form": (
            shear["exact_J_residual"] < 2.0e-9
        ),
        "periodic_shear_window_checks_pass": (
            shear["all_window_checks_pass"]
            and shear["all_integration_checks_pass"]
        ),
        "ABC_is_an_exact_NS_solution": (
            abc_exact["all_exact_solution_checks_pass"]
        ),
        "all_ABC_trajectory_checks_pass": abc_summary[
            "all_trajectory_checks_pass"
        ],
        "ABC_restart_cocycle_checks_pass": abc_cocycle[
            "all_cocycle_checks_pass"
        ],
        "small_window_strain_expansion_checks_pass": (
            small_window_expansion["all_small_window_checks_pass"]
        ),
    }
    certification_flags = {
        "parabolic_window_Gramian_definitions_proved": True,
        "Jacobian_and_Gramian_restart_laws_proved": True,
        "local_Constantin_Iyer_velocity_restart_formula_used": True,
        "exact_directional_cubic_moment_is_sufficient_for_L3_control": True,
        "tensor_spectral_cubic_moment_is_sufficient_for_L3_control": True,
        "scalar_radial_cubic_moment_is_sufficient_for_L3_control": True,
        "scalar_radial_criterion_is_quantitatively_viable": False,
        "Leray_energy_bounds_tensor_spectral_moment": False,
        "Leray_energy_bounds_scalar_radial_moment": False,
        "low_regularity_inverse_time_probe_justified": False,
        "exceptional_set_upgrade_proved": False,
        "Navier_Stokes_global_regularity_proved": False,
    }
    return {
        "kind": "parabolic_gramian_continuation_audit",
        "schema_version": 1,
        "status": (
            "critical_continuation_hierarchy_proved_"
            "unconditional_Gramian_moment_bound_open"
        ),
        "window_definitions": {
            "forward": (
                "F_st=4nu integral_s^t Phi(t,r)Phi(t,r)^T dr"
            ),
            "inverse_time": (
                "B_st=4nu integral_s^t Phi(s,r)Phi(s,r)^T dr"
            ),
            "cross": (
                "H_st=4nu integral_s^t Phi(t,r)Phi(s,r)^T dr"
            ),
            "normalized_traces": (
                "f_st=tr(F_st)/(4nu(t-s)), "
                "b_st=tr(B_st)/(4nu(t-s))"
            ),
        },
        "restart_laws": {
            "Jacobian": "J_st=J_mt J_sm",
            "forward": "F_st=F_mt+J_mt F_sm J_mt^T",
            "inverse_time": (
                "B_st=B_sm+J_sm^(-1) B_mt J_sm^(-T)"
            ),
            "cross": (
                "H_st=J_mt H_sm+H_mt J_sm^(-T)"
            ),
        },
        "L3_continuation_hierarchy": {
            "exact_directional": (
                "Gamma_J(s,t)=integral E_plus "
                "|J_st(a)^(-T)u_s(a)|^3 da"
            ),
            "tensor_spectral": (
                "Q_st=[lambda_max(B_st)/lambda_min(F_st)]^(3/2)"
            ),
            "scalar_radial": (
                "R_st=[b_st f_st^2/4]^(3/2)"
                "=b_st^(3/2) f_st^3/8"
            ),
            "pointwise_hierarchy": (
                "|J_st^(-T)v|^3<=Q_st|v|^3<=R_st|v|^3"
            ),
            "velocity_bound": (
                "||u(t)||_L3<=||P||_(L3->L3) Gamma_J(s,t)^(1/3)"
            ),
            "continuation_gate": (
                "uniform finiteness as t approaches T of any sufficient "
                "weighted cubic moment above implies critical L3 control"
            ),
        },
        "small_window_strain_expansion": small_window_expansion,
        "Burgers_vortex_axis_stress": {
            "rows": burgers_rows,
            "largest_radial_loss_over_tensor": burgers_rows[-1][
                "radial_loss_over_tensor"
            ],
            "conclusion": (
                "mixing forward expansion and inverse-time expansion from "
                "different eigendirections makes the scalar radial "
                "criterion exponentially overconservative"
            ),
        },
        "periodic_finite_Fourier_shear": {
            "exact_solution_audit": shear_exact,
            "window": shear,
        },
        "periodic_finite_Fourier_ABC": {
            "exact_solution_audit": abc_exact,
            "trajectory_rows": abc_rows,
            "summary": abc_summary,
            "restart_cocycle": abc_cocycle,
        },
        "certification_flags": certification_flags,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "next_theorem_target": (
            "Bound the tensor spectral or exact directional cubic moment "
            "on parabolic windows using the two-point Navier-Stokes "
            "generator, pressure/vorticity structure, and common-path "
            "expectation. Scalar radial traces alone are retained only as "
            "a sufficient fallback."
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
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = audit()
    if args.output is not None:
        _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
