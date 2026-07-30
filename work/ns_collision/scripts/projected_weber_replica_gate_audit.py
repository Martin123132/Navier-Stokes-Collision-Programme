"""Audit projection loss in the parabolic Gramian continuation route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad
import sympy as sp


Array = np.ndarray


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


def _radial_weight_symbolic_audit() -> dict[str, Any]:
    radius, epsilon, exponent, viscosity = sp.symbols(
        "r epsilon q nu",
        positive=True,
    )
    weight = (epsilon / (radius**2 + epsilon)) ** (exponent / 2)
    radial_laplacian = sp.diff(weight, radius, 2) + (
        2 * sp.diff(weight, radius) / radius
    )
    expected_ratio = (
        exponent
        * ((exponent - 1) * radius**2 - 3 * epsilon)
        / (radius**2 + epsilon) ** 2
    )
    residual = sp.simplify(radial_laplacian / weight - expected_ratio)
    damping = (
        2
        * viscosity
        * exponent
        * ((1 - exponent) * radius**2 + 3 * epsilon)
        / (radius**2 + epsilon) ** 2
    )
    return {
        "weight": "w=(epsilon/(|Y|^2+epsilon))^(q/2)",
        "laplacian_ratio": (
            "Delta_Y w/w=q*((q-1)|Y|^2-3epsilon)"
            "/(|Y|^2+epsilon)^2"
        ),
        "symbolic_residual": str(residual),
        "symbolic_identity_passes": residual == 0,
        "superharmonic_range": "0<q<=1",
        "positive_damping_in_superharmonic_range": str(damping),
    }


def _tensor_proxy_audit() -> dict[str, Any]:
    generator = np.array(
        [
            [-0.8, 1.1, -0.3],
            [0.2, 0.15, 0.7],
            [0.4, -0.5, 0.65],
        ],
        dtype=float,
    )
    generator -= np.trace(generator) * np.eye(3) / 3.0
    strain = (generator + generator.T) / 2.0
    seed = np.array(
        [
            [1.7, 0.3, -0.2],
            [0.3, 0.9, 0.15],
            [-0.2, 0.15, 1.2],
        ],
        dtype=float,
    )
    covariance = seed @ seed.T
    covariance_derivative = (
        -generator.T @ covariance - covariance @ generator
    )
    rows: list[dict[str, Any]] = []
    for power in (1, 2, 4, 8):
        covariance_power = np.linalg.matrix_power(covariance, power)
        trace_power = float(np.trace(covariance_power))
        direct_trace_derivative = 0.0
        for left_power in range(power):
            direct_trace_derivative += float(
                np.trace(
                    np.linalg.matrix_power(covariance, left_power)
                    @ covariance_derivative
                    @ np.linalg.matrix_power(
                        covariance,
                        power - 1 - left_power,
                    )
                )
            )
        predicted_trace_derivative = float(
            -2.0 * power * np.trace(strain @ covariance_power)
        )
        proxy = trace_power ** (3.0 / (2.0 * power))
        mu_power = float(
            np.trace(strain @ covariance_power) / trace_power
        )
        direct_proxy_derivative = float(
            proxy
            * (3.0 / (2.0 * power))
            * direct_trace_derivative
            / trace_power
        )
        predicted_proxy_derivative = -3.0 * mu_power * proxy
        scale = max(
            1.0,
            abs(direct_proxy_derivative),
            abs(predicted_proxy_derivative),
        )
        rows.append(
            {
                "power": power,
                "proxy": proxy,
                "mu_power": mu_power,
                "trace_derivative_residual": abs(
                    direct_trace_derivative
                    - predicted_trace_derivative
                ),
                "proxy_derivative_relative_residual": abs(
                    direct_proxy_derivative
                    - predicted_proxy_derivative
                )
                / scale,
            }
        )
    return {
        "matrix_equation": "C_dot=-A^T C-C A",
        "smooth_proxy": "Psi_p(C)=(tr(C^p))^(3/(2p))",
        "proxy_generator": (
            "L Psi_p/Psi_p=-3 tr(S C^p)/tr(C^p)"
        ),
        "spectral_limit": (
            "Psi_p decreases to lambda_max(C)^(3/2)"
            "=||J^(-1)||_2^3 as p tends to infinity"
        ),
        "rows": rows,
        "all_checks_pass": all(
            row["trace_derivative_residual"] < 2.0e-10
            and row["proxy_derivative_relative_residual"] < 2.0e-12
            for row in rows
        ),
    }


def _affine_generator_stress() -> dict[str, Any]:
    strain_rate = 1.0
    exponent = 1.0
    inverse_covector_rate = -strain_rate / 2.0
    expanding_tangent_rate = strain_rate
    contracting_tangent_rate = -strain_rate / 2.0
    expanding_far_rate = (
        -3.0 * inverse_covector_rate
        - exponent * expanding_tangent_rate
    )
    contracting_far_rate = (
        -3.0 * inverse_covector_rate
        - exponent * contracting_tangent_rate
    )
    minimum_superharmonic_expanding_rate = min(
        -3.0 * inverse_covector_rate - q * expanding_tangent_rate
        for q in np.linspace(0.0, 1.0, 1001)
    )
    return {
        "strain": "S=diag(-a/2,-a/2,a)",
        "inverse_covector_direction": "z=e_1",
        "inverse_covector_strain_rate_mu": inverse_covector_rate,
        "far_field_formula": "-3*mu-q*sigma",
        "q": exponent,
        "expanding_tangent_direction": {
            "direction": "Y parallel e_3",
            "sigma": expanding_tangent_rate,
            "far_field_generator_rate_over_a": expanding_far_rate,
        },
        "contracting_tangent_direction": {
            "direction": "Y parallel e_1",
            "sigma": contracting_tangent_rate,
            "far_field_generator_rate_over_a": contracting_far_rate,
        },
        "minimum_rate_over_0_le_q_le_1_with_expanding_tangent": (
            minimum_superharmonic_expanding_rate
        ),
        "single_superharmonic_collision_weight_has_universal_sign": False,
        "interpretation": (
            "One positive superharmonic collision factor cannot give a "
            "pointwise nonpositive generator for the cubic inverse "
            "deformation. The claim is a sign obstruction, not a "
            "regularity counterexample."
        ),
    }


def _exact_shear_symbolic_audit() -> dict[str, Any]:
    time, y = sp.symbols("tau y", real=True)
    viscosity, wave_number = sp.symbols("nu k", positive=True)
    mean_speed, amplitude = sp.symbols("c A", real=True)
    decay = sp.exp(-viscosity * wave_number**2 * time)
    velocity = mean_speed + amplitude * decay * sp.sin(wave_number * y)
    velocity_gradient = sp.diff(velocity, y)
    magnetization_x = velocity
    magnetization_y = (
        -mean_speed
        * amplitude
        * wave_number
        * time
        * decay
        * sp.cos(wave_number * y)
        - amplitude**2
        / (4 * viscosity * wave_number)
        * decay**2
        * (1 - decay**2)
        * sp.sin(2 * wave_number * y)
    )
    first_residual = sp.simplify(
        sp.diff(magnetization_x, time)
        - viscosity * sp.diff(magnetization_x, y, 2)
    )
    second_residual = sp.simplify(
        sp.diff(magnetization_y, time)
        - viscosity * sp.diff(magnetization_y, y, 2)
        + velocity_gradient * magnetization_x
    )
    initial_second_component = sp.simplify(
        magnetization_y.subs(time, 0)
    )
    return {
        "velocity": (
            "u=(c+A exp(-nu k^2 tau) sin(k y),0,0)"
        ),
        "magnetization_x": str(magnetization_x),
        "magnetization_y": str(magnetization_y),
        "magnetization_PDE": (
            "partial_t m+u dot grad m-nu Delta m+(grad u)^T m=0"
        ),
        "first_component_residual": str(first_residual),
        "second_component_residual": str(second_residual),
        "initial_second_component": str(initial_second_component),
        "transverse_component_is_pure_gradient": True,
        "Leray_projection": (
            "P m=(c+A exp(-nu k^2 tau) sin(k y),0,0)=u"
        ),
        "all_checks_pass": (
            first_residual == 0
            and second_residual == 0
            and initial_second_component == 0
        ),
    }


def _shear_magnetization_inflation() -> dict[str, Any]:
    mean_speed = 2.0
    amplitude = 1.0
    wave_number = 1.0
    y = np.linspace(0.0, 2.0 * math.pi, 4096, endpoint=False)
    initial_velocity = mean_speed + amplitude * np.sin(y)
    initial_cubic = float(np.mean(initial_velocity**3))
    rows: list[dict[str, Any]] = []
    for viscosity in (0.25, 0.1, 0.03):
        best_time = 0.0
        best_cubic = initial_cubic
        physical_maximum = initial_cubic
        for duration in np.linspace(0.0, 20.0, 401):
            decay = math.exp(-viscosity * duration)
            magnetization_x = mean_speed + amplitude * decay * np.sin(y)
            magnetization_y = (
                -mean_speed
                * amplitude
                * duration
                * decay
                * np.cos(y)
                - amplitude**2
                / (4.0 * viscosity)
                * decay**2
                * (1.0 - decay**2)
                * np.sin(2.0 * y)
            )
            cubic = float(
                np.mean(
                    (
                        magnetization_x**2 + magnetization_y**2
                    )
                    ** 1.5
                )
            )
            physical_cubic = float(np.mean(magnetization_x**3))
            physical_maximum = max(physical_maximum, physical_cubic)
            if cubic > best_cubic:
                best_time = float(duration)
                best_cubic = cubic
        rows.append(
            {
                "viscosity": viscosity,
                "maximum_time": best_time,
                "maximum_unprojected_magnetization_cubic": best_cubic,
                "unprojected_inflation_ratio": (
                    best_cubic / initial_cubic
                ),
                "maximum_physical_velocity_cubic_ratio": (
                    physical_maximum / initial_cubic
                ),
            }
        )
    normalized_reset_gauge_variation = -(
        amplitude**2
        * wave_number**2
        * (
            mean_speed**3 / 2.0
            + 3.0 * mean_speed * amplitude**2 / 8.0
        )
    )
    return {
        "parameters": {
            "mean_speed": mean_speed,
            "amplitude": amplitude,
            "wave_number": wave_number,
            "duration_search_interval": [0.0, 20.0],
            "time_grid_count": 401,
            "space_grid_count": int(y.size),
        },
        "initial_spatial_mean_L3_cubed": initial_cubic,
        "rows": rows,
        "reset_strain_integral": 0.0,
        "reset_gauge_first_variation": (
            normalized_reset_gauge_variation
        ),
        "reset_gauge_first_variation_formula": (
            "-mean_y[(c+A sin y)^3 (A cos y)^2]"
        ),
        "all_checks_pass": (
            normalized_reset_gauge_variation < 0.0
            and rows[0]["unprojected_inflation_ratio"] > 2.0
            and rows[1]["unprojected_inflation_ratio"] > 15.0
            and rows[2]["unprojected_inflation_ratio"] > 100.0
            and all(
                row["maximum_physical_velocity_cubic_ratio"]
                <= 1.0 + 2.0e-12
                for row in rows
            )
        ),
    }


def _projected_harmonic_variance_audit() -> dict[str, Any]:
    angle, phase = sp.symbols("theta delta", real=True)
    phase_pairing = sp.simplify(
        sp.integrate(
            sp.sin(angle) * sp.cos(angle + phase),
            (angle, 0, 2 * sp.pi),
        )
        / (2 * sp.pi)
    )
    expected_phase_pairing = -sp.sin(phase) / 2
    phase_pairing_residual = sp.simplify(
        phase_pairing - expected_phase_pairing
    )
    viscosity = 0.2
    wave_number = 1.3
    amplitude = 1.4
    duration = 3.0
    rate = viscosity * wave_number**2
    scaled_time = rate * duration
    numerator = (
        4.0
        - (3.0 + 12.0 * scaled_time)
        * math.exp(-2.0 * scaled_time)
        - math.exp(-6.0 * scaled_time)
    )
    closed_variance = (
        amplitude**4
        / (96.0 * viscosity**2 * wave_number**2)
        * numerator
    )
    first_integral = quad(
        lambda value: value * math.exp(-2.0 * rate * value),
        0.0,
        duration,
        epsabs=1.0e-13,
        epsrel=1.0e-13,
    )[0]
    second_integral = quad(
        lambda value: (
            math.exp(-2.0 * rate * value)
            * (1.0 - math.exp(-4.0 * rate * value))
            / (4.0 * rate)
        ),
        0.0,
        duration,
        epsabs=1.0e-13,
        epsrel=1.0e-13,
    )[0]
    quadrature_variance = (
        amplitude**4
        * wave_number**2
        / 4.0
        * (first_integral - second_integral)
    )
    relative_residual = abs(closed_variance - quadrature_variance) / max(
        1.0,
        abs(closed_variance),
        abs(quadrature_variance),
    )
    asymptotic_variance = (
        amplitude**4 / (24.0 * viscosity**2 * wave_number**2)
    )
    return {
        "shear_phase_harmonic": (
            "c_W=(A_s^2 k/2) integral_0^tau "
            "exp(-nu k^2 r) sin(k sqrt(2nu) W_r) dr"
        ),
        "Fourier_phase_pairing": (
            "mean_theta[sin(theta)cos(theta+delta)]"
            "=-sin(delta)/2"
        ),
        "Fourier_phase_pairing_symbolic_residual": str(
            phase_pairing_residual
        ),
        "expectation": 0.0,
        "variance_formula": (
            "A_s^4/[96 nu^2 k^2] * "
            "[4-(3+12x)exp(-2x)-exp(-6x)], "
            "x=nu k^2 tau"
        ),
        "parameters": {
            "viscosity": viscosity,
            "wave_number": wave_number,
            "amplitude": amplitude,
            "duration": duration,
            "scaled_time": scaled_time,
        },
        "closed_form_variance": closed_variance,
        "quadrature_variance": quadrature_variance,
        "relative_residual": relative_residual,
        "infinite_time_variance": asymptotic_variance,
        "variance_numerator_small_time": "16x^3+O(x^4)",
        "variance_numerator_is_monotone": True,
        "positive_projected_moment_lower_bound": (
            "E|c_W|^3 >= Var(c_W)^(3/2)"
        ),
        "signed_expectation_cancels_harmonic": True,
        "all_checks_pass": (
            phase_pairing_residual == 0
            and closed_variance > 0.0
            and quadrature_variance > 0.0
            and relative_residual < 2.0e-12
            and asymptotic_variance > closed_variance
        ),
    }


def audit() -> dict[str, Any]:
    radial = _radial_weight_symbolic_audit()
    tensor = _tensor_proxy_audit()
    affine = _affine_generator_stress()
    shear_symbolic = _exact_shear_symbolic_audit()
    shear_inflation = _shear_magnetization_inflation()
    harmonic = _projected_harmonic_variance_audit()

    positive_checks = {
        "radial_weight_symbolic_identity_passes": radial[
            "symbolic_identity_passes"
        ],
        "tensor_proxy_generator_checks_pass": tensor["all_checks_pass"],
        "single_weight_affine_sign_obstruction_exposed": (
            affine[
                "minimum_rate_over_0_le_q_le_1_with_expanding_tangent"
            ]
            >= 0.5 - 2.0e-12
            and affine[
                "contracting_tangent_direction"
            ]["far_field_generator_rate_over_a"]
            > 0.0
        ),
        "exact_shear_magnetization_formula_passes": shear_symbolic[
            "all_checks_pass"
        ],
        "exact_shear_projection_loss_exposed": shear_inflation[
            "all_checks_pass"
        ],
        "projected_harmonic_variance_formula_passes": harmonic[
            "all_checks_pass"
        ],
    }
    certification_flags = {
        "joint_common_path_tangent_covector_generator_derived": True,
        "bare_directional_cubic_has_direct_collision_diffusion": False,
        "smooth_tensor_spectral_proxy_generator_derived": True,
        "single_superharmonic_collision_weight_closes_pointwise": False,
        "mean_Weber_magnetization_equation_derived": True,
        "mean_magnetization_reset_strain_integral_cancels": True,
        "unprojected_positive_moment_retains_Leray_gradient_loss": True,
        "projected_positive_Jensen_moment_retains_noise_variance": True,
        "two_replica_signed_L3_identity_derived": True,
        "three_replica_signed_cubic_identity_derived": True,
        "Leray_energy_bounds_unprojected_directional_moment": False,
        "projected_positive_Jensen_moment_bound_proved": False,
        "signed_projected_replica_closure_bound_proved": False,
        "low_regularity_projected_replica_flow_justified": False,
        "exceptional_set_upgrade_proved": False,
        "Navier_Stokes_global_regularity_proved": False,
    }
    return {
        "kind": "projected_weber_replica_gate_audit",
        "schema_version": 1,
        "status": (
            "projection_loss_identified_"
            "signed_projected_replica_closure_open"
        ),
        "joint_generator": {
            "state_equations": [
                "dX=u(X,t)dt+sqrt(2nu)dW_plus",
                "dY=A(X,t)Ydt+2sqrt(nu)dW_minus",
                "dZ=-A(X,t)^T Zdt",
                "dC=(-A^T C-C A)dt",
            ],
            "generator": (
                "L=partial_t+u dot grad_X+nu Delta_X"
                "+(A Y) dot grad_Y+2nu Delta_Y"
                "-(A^T Z) dot grad_Z"
                "+(-A^T C-C A):grad_C"
            ),
            "directional_cubic": (
                "L|Z|^3/|Z|^3=-3 n_Z^T S n_Z"
            ),
            "coupled_weighted_directional_cubic": (
                "L(w|Z|^3)/(w|Z|^3)="
                "-3mu-q sigma |Y|^2/(|Y|^2+epsilon)"
                "-2nu q[(1-q)|Y|^2+3epsilon]"
                "/(|Y|^2+epsilon)^2"
            ),
        },
        "radial_weight": radial,
        "tensor_proxy": tensor,
        "affine_generator_stress": affine,
        "mean_Weber_magnetization": {
            "random_integrand": (
                "W=(grad A_st)^T(u(s) composed with A_st)"
            ),
            "mean": "m=E W",
            "equation": (
                "partial_t m+u dot grad m-nu Delta m"
                "+(grad u)^T m=0, m(s)=u(s)"
            ),
            "projection_identity": "u(t)=P m(t)",
            "strict_target_hierarchy": (
                "||u(t)||_3<=||P|| ||m||_3"
                "<=||P|| (integral E|W|^3)^(1/3)"
            ),
            "L3_balance": (
                "(1/3)d_t||m||_3^3+nu D_3(m)"
                "=-integral |m| m^T S m"
            ),
            "reset_cancellation": (
                "integral |u| u^T S u"
                "=(1/3)integral u dot grad(|u|^3)=0"
            ),
            "gauge": (
                "m=u+grad q; "
                "q_t+u dot grad q-nu Delta q=p-|u|^2/2"
            ),
            "naive_absolute_value_closure": (
                "|integral |m|m^T S m|"
                "<=C||S||_2 ||m|^(3/2)||_4^2"
                "<=epsilon D_3+C epsilon^(-3)||S||_2^4||m||_3^3"
            ),
            "Leray_obstruction": (
                "energy controls integral ||S||_2^2 dt, "
                "not integral ||S||_2^4 dt"
            ),
        },
        "exact_periodic_shear": {
            "symbolic": shear_symbolic,
            "unprojected_magnetization_inflation": shear_inflation,
            "projected_common_path_harmonic_variance": harmonic,
        },
        "signed_replica_identities": {
            "projected_random_velocity": "v_W=P W_W",
            "mean": "E v_W=u",
            "two_replica": (
                "|u|^3=|u| E[v_1 dot v_2] "
                "for independent projected replicas"
            ),
            "three_replica": (
                "|u|^3=E product_(j=1)^3(v_j dot n), "
                "n=u/|u|"
            ),
            "shear_harmonic_cancellation": (
                "E c_W=0 and E[c_1 c_2]=0, while "
                "E|c_W|^3 is strictly positive"
            ),
            "interpretation": (
                "The sign across common-noise realizations and the Leray "
                "projection must be retained before taking a critical "
                "moment. Positive Jensen bounds discard both cancellations."
            ),
        },
        "certification_flags": certification_flags,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "next_theorem_target": (
            "Derive the signed projected two- and three-replica generator, "
            "including Leray pressure transfer, and seek a scale-critical "
            "integrated bound before any absolute value or operator norm. "
            "The unprojected Gramian hierarchy remains sufficient but is "
            "no longer the primary closure target."
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
        raise RuntimeError("projected Weber replica audit failed")
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
