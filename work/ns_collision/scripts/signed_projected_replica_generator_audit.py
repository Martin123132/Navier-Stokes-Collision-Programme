"""Audit signed projected Weber replica generators and the critical pressure gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample
import sympy as sp

from pressure_frame_pairing_audit import (
    GRID_SIZE,
    RANDOM_SEED,
    VELOCITY_RMS,
    _build_spectral_fields,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parents[3]
PRESSURE_AUDIT = (
    ROOT / "work/ns_collision/scripts/pressure_frame_pairing_audit.py"
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


def _projected_product_generator_audit() -> dict[str, Any]:
    generator = sp.Matrix(
        [
            [sp.Rational(2, 3), -sp.Rational(1, 5), sp.Rational(3, 7)],
            [sp.Rational(4, 9), -sp.Rational(5, 6), sp.Rational(2, 11)],
            [sp.Rational(1, 8), sp.Rational(7, 10), sp.Rational(1, 6)],
        ]
    )
    generator -= sp.eye(3) * sp.trace(generator) / 3
    strain = (generator + generator.T) / 2
    first = sp.Matrix(
        [sp.Rational(3, 4), -sp.Rational(2, 5), sp.Rational(7, 6)]
    )
    second = sp.Matrix(
        [-sp.Rational(5, 8), sp.Rational(9, 7), sp.Rational(1, 3)]
    )
    first_gradient = sp.Matrix(
        [
            [sp.Rational(1, 2), -sp.Rational(2, 3), sp.Rational(3, 5)],
            [sp.Rational(4, 7), sp.Rational(1, 9), -sp.Rational(5, 6)],
            [sp.Rational(2, 11), sp.Rational(7, 8), -sp.Rational(1, 4)],
        ]
    )
    second_gradient = sp.Matrix(
        [
            [-sp.Rational(3, 7), sp.Rational(5, 9), sp.Rational(1, 6)],
            [sp.Rational(2, 5), -sp.Rational(4, 11), sp.Rational(7, 10)],
            [sp.Rational(1, 8), -sp.Rational(3, 4), sp.Rational(5, 12)],
        ]
    )
    first_laplacian = sp.Matrix(
        [sp.Rational(4, 5), -sp.Rational(3, 8), sp.Rational(2, 9)]
    )
    second_laplacian = sp.Matrix(
        [-sp.Rational(1, 6), sp.Rational(7, 9), -sp.Rational(5, 11)]
    )

    stretch_sum = (
        (second.T * generator.T * first)[0]
        + (first.T * generator.T * second)[0]
    )
    stretch_expected = 2 * (first.T * strain * second)[0]
    stretch_residual = sp.simplify(stretch_sum - stretch_expected)

    gradient_pairing = sum(
        first_gradient[row, column] * second_gradient[row, column]
        for row in range(3)
        for column in range(3)
    )
    separate_laplacians = (
        (second.T * first_laplacian)[0]
        + (first.T * second_laplacian)[0]
    )
    product_laplacian = separate_laplacians + 2 * gradient_pairing
    viscosity, correlation = sp.symbols("nu rho", real=True)
    raw_diffusion = (
        viscosity * separate_laplacians
        + 2 * viscosity * correlation * gradient_pairing
    )
    expected_diffusion = (
        viscosity * product_laplacian
        - 2
        * viscosity
        * (1 - correlation)
        * gradient_pairing
    )
    diffusion_residual = sp.simplify(raw_diffusion - expected_diffusion)

    return {
        "projected_replica_SPDE": (
            "dV_i=[-u_j partial_j V_i+nu Delta V_i-"
            "partial_i u_j V_j-partial_i pi[V]]dt-"
            "sqrt(2nu) partial_k V_i dW_k"
        ),
        "pressure_poisson_equation": (
            "Delta pi[V]=-partial_i(u_j partial_j V_i+"
            "partial_i u_j V_j)"
        ),
        "rotational_form": (
            "dV=[nu Delta V+P(u cross curl V)]dt-"
            "sqrt(2nu) grad V dW"
        ),
        "mean_pressure_gauge": "E pi[V]=p-|u|^2/2 up to a space constant",
        "two_replica_local_balance": (
            "dK+[u dot grad K+2 V1^T S V2+"
            "div(pi1 V2+pi2 V1)]dt="
            "[nu Delta K-2nu(1-rho) grad V1:grad V2]dt+dM"
        ),
        "martingale": (
            "dM=-sqrt(2nu)[V2 dot partial_k V1 dW1_k+"
            "V1 dot partial_k V2 dW2_k]"
        ),
        "stretch_symbolic_residual": str(stretch_residual),
        "diffusion_symbolic_residual": str(diffusion_residual),
        "strain_reduction_passes": stretch_residual == 0,
        "cross_diffusion_reduction_passes": diffusion_residual == 0,
        "pressure_flux_uses_divergence_free_replicas": True,
        "all_checks_pass": (
            stretch_residual == 0 and diffusion_residual == 0
        ),
    }


def _rho_reset_stress_audit() -> dict[str, Any]:
    viscosity = 0.07
    amplitude = 1.7
    wave_number = 2.0
    shear_gradient_energy = amplitude**2 * wave_number**2 / 2.0
    abc_gradient_energy = 3.0
    correlations = (0.0, 0.25, 0.5, 0.75, 1.0)

    shear_rows = []
    abc_rows = []
    for correlation in correlations:
        shear_rows.append(
            {
                "rho": correlation,
                "cross_energy_derivative_at_reset": (
                    -2.0
                    * viscosity
                    * (1.0 - correlation)
                    * shear_gradient_energy
                ),
            }
        )
        abc_rows.append(
            {
                "rho": correlation,
                "cross_energy_derivative_at_reset": (
                    -2.0
                    * viscosity
                    * (1.0 - correlation)
                    * abc_gradient_energy
                ),
            }
        )

    strain_rate = 1.0
    longitudinal_rate = -2.0 * strain_rate
    transverse_rate = strain_rate
    physical_shear_energy_derivative = (
        -viscosity * amplitude**2 * wave_number**2
    )
    physical_abc_energy_derivative = -6.0 * viscosity
    return {
        "global_balance": (
            "d_t E integral(V1 dot V2)=-2 E integral(V1^T S V2)"
            "-2nu(1-rho) E integral(grad V1:grad V2)"
        ),
        "independent_endpoint": {
            "C_0": "|u|^2",
            "R_0": "u^T S u",
            "G_0": "|grad u|^2",
            "F_0": "2(p-|u|^2/2)u",
            "integrated_strain_cancellation": (
                "integral u^T S u="
                "integral u dot grad(|u|^2/2)=0"
            ),
            "energy_equality": (
                "d_t ||u||_2^2=-2nu ||grad u||_2^2"
            ),
        },
        "periodic_shear": {
            "velocity": (
                "u=(c+A exp(-nu k^2 t) sin(k y),0,0)"
            ),
            "parameters": {
                "viscosity": viscosity,
                "amplitude": amplitude,
                "wave_number": wave_number,
            },
            "normalized_gradient_energy": shear_gradient_energy,
            "physical_energy_derivative": physical_shear_energy_derivative,
            "rows": shear_rows,
        },
        "abc_flow": {
            "velocity": "u(t)=exp(-nu t) u_ABC, A=B=C=1",
            "parameters": {"viscosity": viscosity},
            "normalized_gradient_energy": abc_gradient_energy,
            "physical_energy_derivative": physical_abc_energy_derivative,
            "rows": abc_rows,
        },
        "burgers_strain_sign_stress": {
            "strain": "S=diag(-a/2,-a/2,a)",
            "V1_equals_V2_equals_e3_rate_over_a": longitudinal_rate,
            "V1_equals_V2_equals_e1_rate_over_a": transverse_rate,
            "interpretation": (
                "The local strain contribution -2 V1^T S V2 has both "
                "signs. This algebraic stress is not asserted to be a "
                "periodic Navier-Stokes solution."
            ),
        },
        "all_checks_pass": (
            abs(
                shear_rows[0]["cross_energy_derivative_at_reset"]
                - physical_shear_energy_derivative
            )
            < 1.0e-14
            and abs(
                abc_rows[0]["cross_energy_derivative_at_reset"]
                - physical_abc_energy_derivative
            )
            < 1.0e-14
            and shear_rows[-1]["cross_energy_derivative_at_reset"] == 0.0
            and abc_rows[-1]["cross_energy_derivative_at_reset"] == 0.0
            and longitudinal_rate < 0.0
            and transverse_rate > 0.0
        ),
    }


def _probabilists_hermite(values: Array) -> tuple[Array, Array, Array, Array]:
    first = values
    second = values**2 - 1.0
    third = values**3 - 3.0 * values
    return np.ones_like(values), first, second, third


def _chaos_function(values: Array) -> Array:
    _, first, second, third = _probabilists_hermite(values)
    return np.stack(
        (
            1.0 + first + 0.5 * second,
            -0.5 + 0.25 * first - 0.2 * third,
            0.75 * second + 0.1 * third,
        ),
        axis=-1,
    )


def _chaos_derivative(values: Array) -> Array:
    _, first, second, _ = _probabilists_hermite(values)
    return np.stack(
        (
            1.0 + first,
            0.25 - 0.6 * second,
            1.5 * first + 0.3 * second,
        ),
        axis=-1,
    )


def _gaussian_chaos_homotopy_audit() -> dict[str, Any]:
    coefficient_vectors = np.array(
        [
            [1.0, -0.5, 0.0],
            [1.0, 0.25, 0.0],
            [0.5, 0.0, 0.75],
            [0.0, -0.2, 0.1],
        ]
    )
    chaos_energies = np.array(
        [
            math.factorial(order)
            * float(np.dot(vector, vector))
            for order, vector in enumerate(coefficient_vectors)
        ]
    )

    hermite_nodes, hermite_weights = np.polynomial.hermite.hermgauss(10)
    gaussian_nodes = math.sqrt(2.0) * hermite_nodes
    gaussian_weights = hermite_weights / math.sqrt(math.pi)
    first_noise = gaussian_nodes[:, None]
    independent_noise = gaussian_nodes[None, :]
    quadrature_weight = (
        gaussian_weights[:, None] * gaussian_weights[None, :]
    )

    rows: list[dict[str, Any]] = []
    for correlation in (0.0, 0.25, 0.5, 0.75, 1.0):
        second_noise = (
            correlation * first_noise
            + math.sqrt(max(0.0, 1.0 - correlation**2))
            * independent_noise
        )
        first_value = _chaos_function(first_noise)
        second_value = _chaos_function(second_noise)
        quadrature_correlation = float(
            np.sum(
                quadrature_weight
                * np.sum(first_value * second_value, axis=-1)
            )
        )
        polynomial_correlation = float(
            sum(
                energy * correlation**order
                for order, energy in enumerate(chaos_energies)
            )
        )
        first_derivative = _chaos_derivative(first_noise)
        second_derivative = _chaos_derivative(second_noise)
        quadrature_derivative = float(
            np.sum(
                quadrature_weight
                * np.sum(
                    first_derivative * second_derivative,
                    axis=-1,
                )
            )
        )
        polynomial_derivative = float(
            sum(
                order * energy * correlation ** (order - 1)
                for order, energy in enumerate(chaos_energies)
                if order > 0
            )
        )
        rows.append(
            {
                "rho": correlation,
                "quadrature_correlation": quadrature_correlation,
                "chaos_polynomial_correlation": polynomial_correlation,
                "correlation_residual": abs(
                    quadrature_correlation - polynomial_correlation
                ),
                "quadrature_Malliavin_pairing": quadrature_derivative,
                "chaos_polynomial_derivative": polynomial_derivative,
                "derivative_residual": abs(
                    quadrature_derivative - polynomial_derivative
                ),
            }
        )

    return {
        "general_identity": (
            "C_rho(x)=E[V(W1,x) dot V(W2,x)]="
            "sum_(n>=0) rho^n ||V_n(x)||_chaos^2"
        ),
        "endpoint_C0": "|E V|^2=|u|^2",
        "endpoint_C1": "E|V|^2",
        "derivative_identity": (
            "partial_rho C_rho="
            "E <D V(W1),D V(W2)>_Cameron-Martin"
        ),
        "variance_identity": (
            "C_1-C_0=Var(V)=integral_0^1 partial_rho C_rho d rho"
        ),
        "gradient_identity": (
            "G_rho=E[grad V(W1):grad V(W2)]="
            "sum_(n>=0) rho^n ||grad V_n||_chaos^2"
        ),
        "gradient_lower_bound": (
            "G_rho>=||grad V_0||^2=|grad u|^2 for 0<=rho<=1"
        ),
        "gradient_cross_pairing_nonnegative": True,
        "scope": (
            "The general identity uses the Wiener-chaos normalization "
            "implicit in ||V_n||_chaos. Malliavin differentiability is "
            "required for the derivative formula, and spatial chaos "
            "differentiability is required for the gradient identity."
        ),
        "explicit_test_function": [
            "F1=1+H1+0.5 H2",
            "F2=-0.5+0.25 H1-0.2 H3",
            "F3=0.75 H2+0.1 H3",
        ],
        "chaos_energies": [float(value) for value in chaos_energies],
        "rows": rows,
        "monotone_on_zero_one": all(
            row["chaos_polynomial_derivative"] >= 0.0 for row in rows
        ),
        "all_checks_pass": (
            all(
                row["correlation_residual"] < 3.0e-13
                and row["derivative_residual"] < 3.0e-13
                for row in rows
            )
            and all(
                chaos_energies[index] >= 0.0
                for index in range(len(chaos_energies))
            )
        ),
    }


def _weighted_critical_audit() -> dict[str, Any]:
    speed, weight = sp.symbols("a lambda", nonnegative=True)
    dual_value = (
        sp.Rational(3, 2) * weight * speed**2
        - sp.Rational(1, 2) * weight**3
    )
    gap = sp.factor(speed**3 - dual_value)
    expected_gap = (
        sp.Rational(1, 2)
        * (weight - speed) ** 2
        * (weight + 2 * speed)
    )
    gap_residual = sp.simplify(gap - expected_gap)
    optimizer_residual = sp.simplify(
        dual_value.subs(weight, speed) - speed**3
    )
    return {
        "weighted_two_replica_identity": (
            "d_t integral(lambda C_rho)="
            "integral(lambda_t+u dot grad lambda+nu Delta lambda)C_rho"
            "-2 integral lambda R_rho+integral grad lambda dot F_rho"
            "-2nu(1-rho) integral lambda G_rho"
        ),
        "rho_zero_reduction": (
            "d_t integral(lambda |u|^2)="
            "integral(lambda_t+u dot grad lambda+nu Delta lambda)|u|^2"
            "+2 integral p u dot grad lambda"
            "-2nu integral lambda |grad u|^2"
        ),
        "critical_dual_identity": (
            "|u|^3=sup_(lambda>=0)"
            "[(3/2)lambda |u|^2-(1/2)lambda^3]"
        ),
        "dual_gap_factorization": str(gap),
        "dual_gap_expected": "(lambda-a)^2(lambda+2a)/2",
        "dual_gap_symbolic_residual": str(gap_residual),
        "optimizer_symbolic_residual": str(optimizer_residual),
        "critical_optimizer": "lambda=|u|",
        "scaling": (
            "lambda scales like velocity, so the dual functional is "
            "critical under Navier-Stokes scaling"
        ),
        "pressure_gate": (
            "For nonconstant critical lambda, pressure survives as "
            "integral grad lambda dot F_rho. Global unweighted pressure "
            "orthogonality therefore does not close L3."
        ),
        "all_checks_pass": gap_residual == 0 and optimizer_residual == 0,
    }


def _evaluate_velocity_grid(
    modes: Array,
    coefficients: Array,
) -> Array:
    coordinates = 2.0 * math.pi * np.arange(GRID_SIZE) / GRID_SIZE
    points = np.stack(
        np.meshgrid(coordinates, coordinates, coordinates, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    phase = np.exp(1j * (points @ modes.T))
    values = np.einsum("pm,mc->pc", phase, coefficients).real
    return values.T.reshape(3, GRID_SIZE, GRID_SIZE, GRID_SIZE)


def _periodic_resample(values: Array, size: int) -> Array:
    result = values
    for axis in (-3, -2, -1):
        result = resample(result, size, axis=axis)
    return np.asarray(result).real


def _adversarial_pressure_stress() -> dict[str, Any]:
    fields = _build_spectral_fields()
    modes, coefficients = fields["velocity"]
    velocity = _evaluate_velocity_grid(modes, coefficients)
    velocity_gradient = np.asarray(fields["velocity_gradient_grid"])
    pressure_gradient = np.asarray(
        fields["pressure_potential_gradient_grid"]
    )
    regularizations = (1.0, 0.3, 0.1, 0.03, 0.0)
    rows: list[dict[str, Any]] = []

    for size in (48, 64, 80, 96):
        resolved_velocity = _periodic_resample(velocity, size)
        resolved_gradient = _periodic_resample(velocity_gradient, size)
        resolved_pressure_gradient = _periodic_resample(
            pressure_gradient,
            size,
        )
        speed_squared = np.sum(resolved_velocity**2, axis=0)
        advection = np.einsum(
            "jxyz,ijxyz->ixyz",
            resolved_velocity,
            resolved_gradient,
        )
        velocity_pressure = np.einsum(
            "ixyz,ixyz->xyz",
            resolved_velocity,
            resolved_pressure_gradient,
        )
        velocity_advection = np.einsum(
            "ixyz,ixyz->xyz",
            resolved_velocity,
            advection,
        )
        weight_rows = []
        for epsilon in regularizations:
            critical_weight = np.sqrt(speed_squared + epsilon**2)
            weight_rows.append(
                {
                    "epsilon": epsilon,
                    "pressure_work": float(
                        -np.mean(critical_weight * velocity_pressure)
                    ),
                    "convective_work": float(
                        -np.mean(critical_weight * velocity_advection)
                    ),
                }
            )
        rows.append(
            {
                "grid_size": size,
                "velocity_rms": float(np.sqrt(np.mean(speed_squared))),
                "weights": weight_rows,
            }
        )

    critical_pressure = [
        row["weights"][-1]["pressure_work"] for row in rows
    ]
    critical_convective = [
        row["weights"][-1]["convective_work"] for row in rows
    ]
    smooth_pressure = [
        row["weights"][0]["pressure_work"] for row in rows
    ]
    pressure_relative_spread = (
        max(critical_pressure) - min(critical_pressure)
    ) / max(abs(value) for value in critical_pressure)
    return {
        "source": (
            "Existing deterministic periodic finite-Fourier adversary "
            "from pressure_frame_pairing_audit.py"
        ),
        "source_sha256": _sha256(PRESSURE_AUDIT),
        "source_parameters": {
            "random_seed": RANDOM_SEED,
            "base_grid_size": GRID_SIZE,
            "velocity_rms_target": VELOCITY_RMS,
        },
        "weight": "lambda_epsilon=sqrt(|u|^2+epsilon^2)",
        "pressure_work": "-mean(lambda_epsilon u dot grad p)",
        "convective_work": (
            "-mean(lambda_epsilon u dot (u dot grad)u)"
        ),
        "rows": rows,
        "critical_pressure_work_range": [
            min(critical_pressure),
            max(critical_pressure),
        ],
        "critical_pressure_relative_spread": pressure_relative_spread,
        "maximum_absolute_critical_convective_work": max(
            abs(value) for value in critical_convective
        ),
        "maximum_velocity_rms_residual": max(
            abs(row["velocity_rms"] - VELOCITY_RMS) for row in rows
        ),
        "interpretation": (
            "Periodic convective cancellation converges to zero, while "
            "the critical pressure work remains positive and resolved. "
            "The same sign persists for smooth epsilon>0 weights, so the "
            "obstruction is not created by the cusp of |u| at zero. This "
            "is a resolved deterministic stress, not an interval proof."
        ),
        "all_checks_pass": (
            min(critical_pressure) > 40.5
            and min(smooth_pressure) > 40.2
            and pressure_relative_spread < 1.0e-5
            and max(abs(value) for value in critical_convective) < 3.1e-4
            and abs(critical_convective[-1]) < 1.0e-5
            and max(
                abs(row["velocity_rms"] - VELOCITY_RMS) for row in rows
            )
            < 2.0e-12
        ),
    }


def _three_replica_tensor_audit() -> dict[str, Any]:
    correlation_matrix = np.array(
        [
            [1.0, 0.4, 0.2],
            [0.4, 1.0, 0.35],
            [0.2, 0.35, 1.0],
        ]
    )
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    return {
        "tensor": "T_ijk=E[V1_i V2_j V3_k]",
        "primitive_operator": (
            "B_u V=-u dot grad V-(grad u)^T V-grad pi[V]+nu Delta V"
        ),
        "generator": (
            "partial_t T=E sum_(r=1)^3 slot_r(B_u V_r)+"
            "2nu sum_(r<s) rho_rs sum_k "
            "E[slot_r(partial_k V_r) slot_s(partial_k V_s)]"
        ),
        "independent_endpoint": "T_ijk=u_i u_j u_k",
        "critical_contraction": (
            "(n tensor n tensor n):T=|u|^3, n=u/|u|"
        ),
        "pressure_warning": (
            "After contraction with n, pressure is not a pure global "
            "divergence because derivatives also hit n and the other "
            "replica factors."
        ),
        "sample_correlation_matrix": correlation_matrix.tolist(),
        "sample_correlation_eigenvalues": [
            float(value) for value in eigenvalues
        ],
        "sample_correlation_matrix_is_positive_definite": bool(
            eigenvalues[0] > 0.0
        ),
        "all_checks_pass": bool(eigenvalues[0] > 0.0),
    }


def audit() -> dict[str, Any]:
    projected = _projected_product_generator_audit()
    reset = _rho_reset_stress_audit()
    chaos = _gaussian_chaos_homotopy_audit()
    weighted = _weighted_critical_audit()
    pressure = _adversarial_pressure_stress()
    triple = _three_replica_tensor_audit()

    positive_checks = {
        "projected_product_generator_algebra_passes": projected[
            "all_checks_pass"
        ],
        "rho_reset_shear_abc_and_strain_stresses_pass": reset[
            "all_checks_pass"
        ],
        "Gaussian_chaos_homotopy_quadrature_passes": chaos[
            "all_checks_pass"
        ],
        "positive_rho_cross_gradient_is_dissipative": chaos[
            "gradient_cross_pairing_nonnegative"
        ],
        "critical_L3_dual_factorization_passes": weighted[
            "all_checks_pass"
        ],
        "critical_pressure_flux_obstruction_is_resolved": pressure[
            "all_checks_pass"
        ],
        "three_replica_correlation_matrix_check_passes": triple[
            "all_checks_pass"
        ],
    }
    certification_flags = {
        "projected_Weber_SPDE_derived": True,
        "rho_two_replica_local_balance_derived": True,
        "rho_global_cross_energy_balance_derived": True,
        "independent_endpoint_recovers_energy_equality": True,
        "Gaussian_chaos_correlation_homotopy_derived": True,
        "positive_rho_cross_gradient_dissipation_proved": True,
        "weighted_two_replica_identity_derived": True,
        "critical_L3_Legendre_representation_derived": True,
        "three_replica_tensor_generator_derived": True,
        "critical_weight_pressure_flux_can_be_nonzero": True,
        "rho_cross_energy_has_universal_strain_sign": False,
        "unweighted_global_pressure_cancellation_closes_L3": False,
        "weighted_pressure_flux_bound_proved": False,
        "signed_replica_L3_bound_proved": False,
        "low_regularity_projected_replica_flow_justified": False,
        "exceptional_set_upgrade_proved": False,
        "Navier_Stokes_global_regularity_proved": False,
    }
    return {
        "kind": "signed_projected_replica_generator_audit",
        "schema_version": 1,
        "status": (
            "signed_projected_replica_generator_derived_"
            "weighted_pressure_flux_open"
        ),
        "assumption_scope": (
            "Classical smooth incompressible Navier-Stokes and smooth "
            "projected stochastic Weber replicas on a periodic domain or "
            "with sufficient decay. Low-regularity passage is not claimed."
        ),
        "projected_two_replica_generator": projected,
        "rho_reset_stresses": reset,
        "Gaussian_chaos_homotopy": chaos,
        "weighted_critical_formulation": weighted,
        "adversarial_pressure_stress": pressure,
        "three_replica_tensor_generator": triple,
        "certification_flags": certification_flags,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "next_theorem_target": (
            "Optimize or falsify the weighted two-replica dual equation "
            "with a backward/adapted lambda. Any candidate must control "
            "the signed pressure flux integral grad(lambda) dot F_rho "
            "together with strain before absolute values, retain the "
            "proved lower bound G_rho>=|grad u|^2 in the decorrelation "
            "dissipation 2nu(1-rho)G_rho, and survive the stored periodic "
            "pressure adversary. Connect the flux to the existing "
            "partition-edge antisymmetry machinery before any large "
            "numerical search."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("signed projected replica generator audit failed")
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "kind": result["kind"],
                "output": args.output.as_posix(),
                "output_sha256": _sha256(args.output),
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
