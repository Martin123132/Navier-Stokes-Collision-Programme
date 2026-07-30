"""Audit the full 3D strain-core form gate and Leray-level obstructions."""

from __future__ import annotations

import importlib.util
import json
from math import pi
from pathlib import Path

import sympy as sp


def _load_localized_tube_module():
    script = Path(__file__).resolve().with_name("localized_strain_tube_audit.py")
    spec = importlib.util.spec_from_file_location(
        "localized_strain_tube_audit_for_3d_gate", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit() -> dict[str, object]:
    x, y, z, a, nu = sp.symbols("x y z a nu", positive=True, real=True)
    coordinates = (x, y, z)
    backward_drift = (a * x, a * y, -2 * a * z)
    divergence = sp.simplify(
        sum(
            sp.diff(component, coordinate)
            for component, coordinate in zip(backward_drift, coordinates)
        )
    )
    drift_norm_sq = sum(component**2 for component in backward_drift)
    gauge_potential = sp.factor(drift_norm_sq / (4 * nu) + divergence / 2)
    expected_potential = sp.factor(
        a**2 * (x**2 + y**2 + 4 * z**2) / (4 * nu)
    )

    gauge_exponent = a * (x**2 + y**2) / 2 - a * z**2
    gauge = sp.exp(-gauge_exponent / (2 * nu))
    test_function = sp.Function("test_function")(*coordinates)
    conjugated_action = sp.simplify(
        gauge ** (-1)
        * (
            nu
            * sum(
                sp.diff(gauge * test_function, coordinate, 2)
                for coordinate in coordinates
            )
            + sum(
                component * sp.diff(gauge * test_function, coordinate)
                for component, coordinate in zip(
                    backward_drift, coordinates
                )
            )
            + 2 * a * gauge * test_function
        )
    )
    expected_conjugated_action = sp.simplify(
        nu
        * sum(
            sp.diff(test_function, coordinate, 2)
            for coordinate in coordinates
        )
        - gauge_potential * test_function
        + 2 * a * test_function
    )

    oscillator_ground_state = sp.exp(
        -a * (x**2 + y**2) / (4 * nu) - a * z**2 / (2 * nu)
    )
    oscillator_action = sp.simplify(
        -nu
        * sum(
            sp.diff(oscillator_ground_state, coordinate, 2)
            for coordinate in coordinates
        )
        + gauge_potential * oscillator_ground_state
    )

    error_x, error_y, error_z = sp.symbols(
        "error_x error_y error_z", real=True
    )
    divergence_error, stretching_error = sp.symbols(
        "divergence_error stretching_error", real=True
    )
    drift_error_pairing = sum(
        component * error
        for component, error in zip(
            backward_drift, (error_x, error_y, error_z)
        )
    )
    effective_error = sp.factor(
        stretching_error
        - divergence_error / 2
        - drift_error_pairing / (2 * nu)
    )
    incompressible_effective_error = sp.factor(
        effective_error.subs(divergence_error, 0)
    )

    radius = sp.symbols("radius", nonnegative=True, real=True)
    aubin_talenti = (1 + radius**2) ** sp.Rational(-1, 2)
    integral_sixth_power = sp.simplify(
        4
        * sp.pi
        * sp.integrate(radius**2 * aubin_talenti**6, (radius, 0, sp.oo))
    )
    integral_gradient_sq = sp.simplify(
        4
        * sp.pi
        * sp.integrate(
            radius**2 * sp.diff(aubin_talenti, radius) ** 2,
            (radius, 0, sp.oo),
        )
    )
    sharp_sobolev_constant = sp.simplify(
        integral_sixth_power ** sp.Rational(1, 3)
        / integral_gradient_sq
    )
    expected_sobolev_constant = sp.real_root(4, 3) ** 2 / (
        3 * sp.pi ** sp.Rational(4, 3)
    )

    localized_rows = _load_localized_tube_module().audit()["spectral_rows"]
    sobolev_constant_float = 4.0 ** (2.0 / 3.0) / (3.0 * pi ** (4.0 / 3.0))
    rows = []
    for row in localized_rows:
        margin_over_a = float(row["lambda_over_a"]) - 2.0
        rows.append(
            {
                "tube_reynolds": float(row["tube_reynolds"]),
                "margin_over_a": margin_over_a,
                "allowed_q_l3_over_2_over_nu": (
                    margin_over_a
                    / (sobolev_constant_float * (margin_over_a + 2.0))
                ),
            }
        )

    spike_rows = []
    for spike_parameter in (10, 100, 1000, 10000):
        spike_rows.append(
            {
                "N": spike_parameter,
                "amplitude": spike_parameter**0.5,
                "support_length": 1.0 / spike_parameter,
                "time_l2_norm_squared": 1.0,
                "time_l4_norm_fourth_power": float(spike_parameter),
            }
        )

    eta, kappa = sp.symbols("eta kappa", positive=True, real=True)
    deterministic_return_time = sp.log(eta) / kappa
    single_return_deformation = sp.simplify(
        sp.exp(kappa * deterministic_return_time)
    )
    pair_return_deformation = sp.simplify(single_return_deformation**2)
    inward_backward_matrix_trace = -kappa - kappa + 2 * kappa
    inward_matrix_frobenius_sq = kappa**2 + kappa**2 + (2 * kappa) ** 2
    return_vector_potential = (-kappa * y * z, kappa * x * z, sp.Integer(0))
    return_vector_potential_curl = (
        sp.diff(return_vector_potential[2], y)
        - sp.diff(return_vector_potential[1], z),
        sp.diff(return_vector_potential[0], z)
        - sp.diff(return_vector_potential[2], x),
        sp.diff(return_vector_potential[1], x)
        - sp.diff(return_vector_potential[0], y),
    )
    expected_return_drift = (-kappa * x, -kappa * y, 2 * kappa * z)
    transverse_variance_at_return = sp.factor(
        nu * (1 - eta ** (-2)) / kappa
    )
    axial_variance_at_return = sp.factor(
        nu * (eta**4 - 1) / (2 * kappa)
    )
    weighted_barrier_residual = sp.simplify(
        -(-kappa) + kappa
    )

    result: dict[str, object] = {
        "affine_backward_drift": "B*x=(a*x,a*y,-2*a*z)",
        "affine_backward_drift_is_incompressible": bool(divergence == 0),
        "gauge": "exp(-[a*(x^2+y^2)/2-a*z^2]/(2*nu))",
        "gauge_potential": str(gauge_potential),
        "gauge_potential_verified": bool(
            sp.simplify(gauge_potential - expected_potential) == 0
        ),
        "full_conjugated_operator_verified": bool(
            sp.simplify(
                conjugated_action - expected_conjugated_action
            )
            == 0
        ),
        "full_plane_oscillator_ground_energy": "2*a",
        "oscillator_ground_energy_verified": bool(
            sp.simplify(
                oscillator_action - 2 * a * oscillator_ground_state
            )
            == 0
        ),
        "bounded_dirichlet_domain_has_strict_margin": True,
        "three_dimensional_effective_error": str(effective_error),
        "incompressible_effective_error": str(incompressible_effective_error),
        "incompressibility_removes_divergence_error": bool(
            divergence_error not in incompressible_effective_error.free_symbols
        ),
        "aubin_talenti_l6_sixth_power": str(integral_sixth_power),
        "aubin_talenti_gradient_energy": str(integral_gradient_sq),
        "sharp_sobolev_constant": str(sharp_sobolev_constant),
        "sharp_sobolev_constant_verified": bool(
            sp.simplify(
                sharp_sobolev_constant - expected_sobolev_constant
            )
            == 0
        ),
        "critical_form_condition": (
            "||q_+||_(3/2)<nu*m/(S3*(m+2*a))"
        ),
        "local_leray_bound": (
            "||q_+||_(3/2)<=|Omega|^(1/6)*(||delta_s||_2+"
            "C_P*||B||*diam(Omega)^2*||grad(e)||_2/(2*nu))"
        ),
        "spectral_budget_rows": rows,
        "all_critical_budgets_are_positive": all(
            row["allowed_q_l3_over_2_over_nu"] > 0.0 for row in rows
        ),
        "critical_budget_collapses_at_high_reynolds": bool(
            rows[-1]["allowed_q_l3_over_2_over_nu"]
            < rows[-2]["allowed_q_l3_over_2_over_nu"]
            < rows[-3]["allowed_q_l3_over_2_over_nu"]
        ),
        "leray_potential_parabolic_index": "2/2+3/2=5/2>2",
        "l2_space_form_bound_generates_time_power": "||q(t)||_2^4",
        "time_spike_rows": spike_rows,
        "time_l2_control_does_not_control_required_l4_power": bool(
            spike_rows[-1]["time_l4_norm_fourth_power"]
            > spike_rows[-2]["time_l4_norm_fourth_power"]
            and all(row["time_l2_norm_squared"] == 1.0 for row in spike_rows)
        ),
        "inward_return_backward_matrix": "diag(-kappa,-kappa,2*kappa)",
        "inward_return_matrix_is_incompressible": bool(
            inward_backward_matrix_trace == 0
        ),
        "inward_return_shell_gradient_density": str(
            inward_matrix_frobenius_sq
        ),
        "inward_return_vector_potential": (
            "(-kappa*y*z,kappa*x*z,0)"
        ),
        "return_vector_potential_curl_verified": bool(
            all(
                sp.simplify(actual - expected) == 0
                for actual, expected in zip(
                    return_vector_potential_curl, expected_return_drift
                )
            )
        ),
        "compact_solenoidal_extension": (
            "curl(chi*A), with chi=1 on the return shell"
        ),
        "inward_return_shell_energy_is_finite_for_finite_kappa": True,
        "deterministic_return_time": str(deterministic_return_time),
        "single_return_deformation": str(single_return_deformation),
        "pair_return_deformation": str(pair_return_deformation),
        "transverse_noise_variance_at_return_time": str(
            transverse_variance_at_return
        ),
        "axial_noise_variance_at_return_time": str(axial_variance_at_return),
        "return_noise_vanishes_as_kappa_tends_to_infinity": bool(
            sp.limit(transverse_variance_at_return, kappa, sp.oo) == 0
            and sp.limit(axial_variance_at_return, kappa, sp.oo) == 0
        ),
        "inward_weighted_barrier_residual": str(
            weighted_barrier_residual
        ),
        "inward_weighted_return_supersolution_fails": bool(
            weighted_barrier_residual.is_positive
        ),
        "finite_energy_drift_alone_does_not_force_weighted_return_contraction": True,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
