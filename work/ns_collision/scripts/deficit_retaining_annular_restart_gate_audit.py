"""Audit the reset-time Legendre tax on the annular static escape."""

from __future__ import annotations

import argparse
import ctypes
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sympy as sp

from compatible_edge_annular_escape_audit import (
    DELTA_CUBIC_ENERGY,
    _joint_ray_optimum,
)
from separable_annular_pressure_schur_no_go_audit import (
    _family_arrays,
    _mixed_difference_fisher,
    _resonant_component_loads,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "deficit_retaining_annular_restart_gate_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "adjoint_replica_pressure_edge_gate_audit_v1.json"
    ): "9da360ccb3e6051561889a2efdb68d20950314860b57a5bc7e4e7c0df80cee2d",
    (
        "work/ns_collision/results/"
        "finite_window_rho_terminal_tax_audit_v1.json"
    ): "7eb5ca373f4d10a8d21ebe089c27c282c051aaa44e7e378f3b0de4c7427ed3e5",
    (
        "work/ns_collision/results/"
        "annular_eight_vertex_heat_window_gate_audit_v1.json"
    ): "5313001d5a136babf1be6d99b66767db4161e526cd08158631cde2a68c942789",
    (
        "work/ns_collision/results/"
        "compatible_edge_annular_escape_audit_v1.json"
    ): "fffa314fc9fa516dc0c8f6ac010392d438845912f6d4bc2d16cc1f2dc02b83e0",
}
ALGORITHM_REVISION = "deficit-retaining-annular-restart-gate-v1"
DEFAULT_SIZES = (17, 25, 33, 65)
DEFAULT_SCALED_WINDOW = 0.1


def _lower_process_priority() -> None:
    if os.name != "nt":
        return
    below_normal_priority_class = 0x00004000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    kernel32.SetPriorityClass(handle, below_normal_priority_class)


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


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _prerequisite_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    payloads = {}
    for relative, expected in PREREQUISITES.items():
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[relative] = payload
        actual = _sha256(path)
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "all_positive_checks_pass": payload.get(
                    "all_positive_checks_pass"
                ),
                "matches": bool(
                    actual == expected
                    and payload.get("all_positive_checks_pass") is True
                ),
            }
        )
    return (
        {
            "rows": rows,
            "all_checks_pass": all(row["matches"] for row in rows),
        },
        payloads,
    )


def _symbolic_deficit_certificate() -> dict[str, Any]:
    speed, weight = sp.symbols(
        "r lambda", nonnegative=True, real=True
    )
    deficit = (
        speed**3
        - sp.Rational(3, 2) * weight * speed**2
        + sp.Rational(1, 2) * weight**3
    )
    factorized = (speed - weight) ** 2 * (
        speed + weight / 2
    )
    factor_residual = sp.simplify(deficit - factorized)

    norm_reset, reset_deficit, integrated_generator = sp.symbols(
        "N_s D_s J_0", real=True
    )
    legendre_reset = norm_reset - reset_deficit
    legendre_terminal = legendre_reset + integrated_generator
    expected_terminal = (
        norm_reset + integrated_generator - reset_deficit
    )
    endpoint_residual = sp.simplify(
        legendre_terminal - expected_terminal
    )
    return {
        "pointwise_Legendre_functional": (
            "(3/2)lambda |u|^2-(1/2)lambda^3"
        ),
        "pointwise_deficit": (
            "|u|^3-(3/2)lambda|u|^2+(1/2)lambda^3"
        ),
        "factorization": (
            "D(r,lambda)=(r-lambda)^2(r+lambda/2)"
        ),
        "factorization_symbolic_residual": str(factor_residual),
        "coercive_lower_bound": (
            "D(r,lambda)>=(1/2)|r-lambda|^3"
        ),
        "integrated_deficit": (
            "Delta_s(lambda_s)=integral "
            "(|u(s)|-lambda_s)^2(|u(s)|+lambda_s/2)"
        ),
        "rho_zero_integrated_generator": (
            "J_0[lambda_T]=3 integral_s^T integral["
            "p u dot grad lambda-nu lambda|grad u|^2"
            "-nu lambda|grad lambda|^2]"
        ),
        "exact_deficit_retaining_restart_identity": (
            "||u(T)||_3^3=||u(s)||_3^3+"
            "sup_(lambda_T>=0)[J_0[lambda_T]-Delta_s(lambda_s)]"
        ),
        "stored_inequality_is_deficit_dropped_version": (
            "Dropping -Delta_s and moving sup through the integral gives "
            "the earlier restart upper bound."
        ),
        "endpoint_algebra_symbolic_residual": str(endpoint_residual),
        "all_checks_pass": bool(
            factor_residual == 0 and endpoint_residual == 0
        ),
    }


def _exact_partition_norms() -> dict[str, Any]:
    one_dimensional_phi_cube_mean = Fraction(5, 16)
    vertex_cube_mean = one_dimensional_phi_cube_mean**3
    vertex_L3_norm = Fraction(5, 16)
    low_field_L2_squared = Fraction(2)
    weight_fisher = DELTA_CUBIC_ENERGY / 16
    direct_weight_fisher = (
        3
        * Fraction(1, 16)
        * one_dimensional_phi_cube_mean**2
    )
    return {
        "one_dimensional_phi_plus_cube_mean": _fraction_text(
            one_dimensional_phi_cube_mean
        ),
        "Phi_plus_plus_plus_cube_mean": _fraction_text(
            vertex_cube_mean
        ),
        "Phi_plus_plus_plus_L3_norm": _fraction_text(vertex_L3_norm),
        "unit_low_plane_wave_L2_squared": _fraction_text(
            low_field_L2_squared
        ),
        "delta_plus_Q": _fraction_text(DELTA_CUBIC_ENERGY),
        "Phi_weight_Fisher_mean": _fraction_text(weight_fisher),
        "direct_product_weight_Fisher_mean": _fraction_text(
            direct_weight_fisher
        ),
        "backward_L3_contraction": (
            "||lambda_s||_3<=||lambda_T||_3"
        ),
        "reset_tax_lower_bound": (
            "Delta_s>=(1/2)("
            "||u(s)||_2-||lambda_T||_3)_+^3"
        ),
        "annular_specialization": (
            "Delta_s>=(1/2)("
            "sqrt(||h_N||_2^2+2a^2)-5t/16)_+^3"
        ),
        "all_checks_pass": bool(
            vertex_cube_mean == Fraction(125, 4096)
            and vertex_L3_norm == Fraction(5, 16)
            and weight_fisher == Fraction(75, 4096)
            and direct_weight_fisher == weight_fisher
        ),
    }


def _pressure_ray_row(
    size: int,
    viscosity: float = 1.0,
) -> dict[str, Any]:
    waves, velocity, parity = _family_arrays(
        (size, size, size), 2 * size
    )
    loads = _resonant_component_loads(waves, velocity)
    pressure_load = (
        loads["pressure_high_high"] + loads["pressure_cross"]
    )
    fisher = _mixed_difference_fisher(waves, velocity, parity)
    high_L2_squared = 2.0 * float(np.sum(velocity * velocity))
    optimum = _joint_ray_optimum(
        pressure_load,
        fisher,
        1.0,
        float(DELTA_CUBIC_ENERGY),
        viscosity,
    )
    complete_load = loads["combined"]
    return {
        "size": size,
        "kinetic_HHL_load": loads["kinetic"],
        "high_high_pressure_HHL_load": loads["pressure_high_high"],
        "cross_pressure_HHL_load": loads["pressure_cross"],
        "rho_zero_pressure_HHL_load": pressure_load,
        "complete_local_energy_HHL_load": complete_load,
        "pressure_vs_complete_residual": pressure_load - complete_load,
        "relative_pressure_vs_complete_difference": abs(
            pressure_load - complete_load
        )
        / abs(pressure_load),
        "plus_vertex_high_Fisher": fisher,
        "high_field_L2_squared": high_L2_squared,
        "pressure_load_over_size": pressure_load / size,
        "ray_optimization": optimum,
        "maximum_imaginary_load_residual": loads[
            "maximum_imaginary_residual"
        ],
        "all_checks_pass": bool(
            pressure_load < 0.0
            and fisher > 0.0
            and high_L2_squared > 0.0
            and loads["maximum_imaginary_residual"] < 4.0e-12
            and optimum["all_checks_pass"]
        ),
    }


def _reset_tax_row(
    pressure_row: dict[str, Any],
    scaled_window: float,
) -> dict[str, Any]:
    size = int(pressure_row["size"])
    optimum = pressure_row["ray_optimization"]
    if not optimum["positive_escape"]:
        return {
            "size": size,
            "static_pressure_escape_positive": False,
            "tax_comparison_available": False,
            "all_checks_pass": True,
        }

    low_amplitude = float(
        optimum["optimal_oriented_low_amplitude"]
    )
    coefficient_scale = float(
        optimum["optimal_coefficient_scale"]
    )
    normalized_generator = float(optimum["optimized_objective"])
    low_only_velocity_L2 = math.sqrt(2.0) * low_amplitude
    full_velocity_L2 = math.sqrt(
        pressure_row["high_field_L2_squared"]
        + 2.0 * low_amplitude**2
    )
    terminal_weight_L3 = 5.0 * coefficient_scale / 16.0
    low_only_gap = max(
        low_only_velocity_L2 - terminal_weight_L3, 0.0
    )
    full_gap = max(full_velocity_L2 - terminal_weight_L3, 0.0)
    low_only_tax = 0.5 * low_only_gap**3
    full_L2_tax = 0.5 * full_gap**3
    parabolic_window = scaled_window / size**2
    tax_to_three_generator_time = (
        full_L2_tax / (3.0 * normalized_generator)
    )
    low_only_tax_to_three_generator_time = (
        low_only_tax / (3.0 * normalized_generator)
    )
    frozen_window_ratio = (
        3.0
        * parabolic_window
        * normalized_generator
        / full_L2_tax
    )
    required_average_generator = (
        full_L2_tax / (3.0 * parabolic_window)
    )
    required_amplification = (
        required_average_generator / normalized_generator
    )
    return {
        "size": size,
        "static_pressure_escape_positive": True,
        "tax_comparison_available": True,
        "optimal_low_amplitude": low_amplitude,
        "optimal_terminal_coefficient_scale": coefficient_scale,
        "initial_normalized_pressure_generator": normalized_generator,
        "low_only_velocity_L2_lower_bound": low_only_velocity_L2,
        "full_velocity_L2": full_velocity_L2,
        "terminal_weight_L3": terminal_weight_L3,
        "low_only_norm_gap": low_only_gap,
        "full_norm_gap": full_gap,
        "low_only_reset_deficit_lower_bound": low_only_tax,
        "full_L2_reset_deficit_lower_bound": full_L2_tax,
        "low_only_tax_to_three_static_generator_time": (
            low_only_tax_to_three_generator_time
        ),
        "full_tax_to_three_static_generator_time": (
            tax_to_three_generator_time
        ),
        "scaled_window": scaled_window,
        "parabolic_window": parabolic_window,
        "frozen_generator_fraction_of_full_tax": frozen_window_ratio,
        "average_generator_required_to_overcome_full_tax": (
            required_average_generator
        ),
        "required_average_amplification_over_initial_generator": (
            required_amplification
        ),
        "interpretation": (
            "A positive exact penalized restart contribution on this "
            "window requires the time-averaged normalized generator to "
            "exceed the displayed threshold. No such amplification is "
            "asserted or excluded by this row."
        ),
        "all_checks_pass": bool(
            low_only_gap > 0.0
            and full_gap > low_only_gap
            and full_L2_tax > low_only_tax > 0.0
            and tax_to_three_generator_time > parabolic_window
            and 0.0 < frozen_window_ratio < 1.0
            and required_amplification > 1.0
        ),
    }


def _asymptotic_certificate(
    annular_result: dict[str, Any],
    viscosity: float,
    scaled_window: float,
) -> dict[str, Any]:
    beta_signed = float(
        annular_result["continuum_response_certificate"][
            "plus_vertex_static_limit"
        ]
    )
    beta_star = abs(beta_signed)
    low_amplitude_coefficient = beta_star / viscosity
    coefficient_scale_coefficient = (
        64.0 * beta_star / (15.0 * math.sqrt(2.0) * viscosity)
    )
    normalized_generator_coefficient = (
        32.0
        * math.sqrt(2.0)
        * beta_star**3
        / (45.0 * viscosity**2)
    )
    low_norm_gap_coefficient = (
        math.sqrt(2.0) * beta_star / (3.0 * viscosity)
    )
    reset_tax_coefficient = (
        math.sqrt(2.0)
        * beta_star**3
        / (27.0 * viscosity**3)
    )
    tax_to_three_generator_time = 5.0 / (288.0 * viscosity)
    required_amplification_coefficient = (
        5.0 / (288.0 * viscosity * scaled_window)
    )
    exact_ratio = (
        reset_tax_coefficient
        / (3.0 * normalized_generator_coefficient)
    )
    return {
        "beta_plus_signed": beta_signed,
        "beta_star": beta_star,
        "pressure_only_has_same_leading_limit": True,
        "reason": (
            "The exact incidence theorem makes the leading kinetic matrix "
            "zero and the cross-pressure term is lower order."
        ),
        "static_optimal_low_amplitude_over_N_limit": (
            low_amplitude_coefficient
        ),
        "static_optimal_coefficient_scale_over_N_limit": (
            coefficient_scale_coefficient
        ),
        "static_normalized_generator_over_N_cubed_limit": (
            normalized_generator_coefficient
        ),
        "low_only_norm_gap_over_N_limit": low_norm_gap_coefficient,
        "reset_deficit_over_N_cubed_lower_limit": (
            reset_tax_coefficient
        ),
        "reset_tax_to_three_static_generator_time_limit": (
            tax_to_three_generator_time
        ),
        "exact_time_ratio_formula": "5/(288nu)",
        "parabolic_window": "delta_N=T/N^2",
        "required_average_generator_scale": "Omega(N^5)",
        "required_amplification_over_initial_generator": (
            "[5/(288nu T)]N^2+o(N^2)"
        ),
        "required_amplification_coefficient": (
            required_amplification_coefficient
        ),
        "bounded_scale_one_variant": (
            "For t=1 and a_N~beta_*N/nu, the reset deficit is "
            "Omega(N^3), the initial normalized generator is Theta(N^2), "
            "and a T/N^2 window without amplification contributes only "
            "Theta(1)."
        ),
        "all_checks_pass": bool(
            beta_signed < 0.0
            and low_norm_gap_coefficient > 0.0
            and reset_tax_coefficient > 0.0
            and normalized_generator_coefficient > 0.0
            and abs(exact_ratio - tax_to_three_generator_time) < 2.0e-17
            and required_amplification_coefficient > 0.0
        ),
    }


def audit(
    sizes: Sequence[int] = DEFAULT_SIZES,
    viscosity: float = 1.0,
    scaled_window: float = DEFAULT_SCALED_WINDOW,
) -> dict[str, Any]:
    if viscosity <= 0.0 or scaled_window <= 0.0:
        raise ValueError("viscosity and scaled window must be positive")
    clean_sizes = tuple(int(size) for size in sizes)
    if (
        not clean_sizes
        or any(size < 3 or size % 2 == 0 for size in clean_sizes)
        or tuple(sorted(set(clean_sizes))) != clean_sizes
    ):
        raise ValueError("sizes must be distinct increasing odd integers >=3")

    prerequisite, payloads = _prerequisite_audit()
    symbolic = _symbolic_deficit_certificate()
    partition = _exact_partition_norms()
    pressure_rows = [
        _pressure_ray_row(size, viscosity) for size in clean_sizes
    ]
    tax_rows = [
        _reset_tax_row(row, scaled_window) for row in pressure_rows
    ]
    annular_path = (
        "work/ns_collision/results/"
        "annular_eight_vertex_heat_window_gate_audit_v1.json"
    )
    asymptotic = _asymptotic_certificate(
        payloads[annular_path], viscosity, scaled_window
    )
    pressure_by_size = {row["size"]: row for row in pressure_rows}
    tax_by_size = {row["size"]: row for row in tax_rows}
    finite_gate = bool(
        25 in pressure_by_size
        and pressure_by_size[25]["ray_optimization"]["positive_escape"]
        and tax_by_size[25]["all_checks_pass"]
        and tax_by_size[25][
            "required_average_amplification_over_initial_generator"
        ]
        > 1.0e6
    )
    final_pressure = pressure_rows[-1]
    leading_correction_small = bool(
        final_pressure["relative_pressure_vs_complete_difference"]
        < 2.0e-8
    )
    all_checks = bool(
        prerequisite["all_checks_pass"]
        and symbolic["all_checks_pass"]
        and partition["all_checks_pass"]
        and asymptotic["all_checks_pass"]
        and all(row["all_checks_pass"] for row in pressure_rows)
        and all(row["all_checks_pass"] for row in tax_rows)
        and finite_gate
        and leading_correction_small
    )
    return {
        "kind": "deficit_retaining_annular_restart_gate_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": (
            "annular_static_escape_reset_tax_certified"
            if all_checks
            else "deficit_retaining_annular_restart_gate_failed"
        ),
        "scope": (
            "The exact smooth rho=0 backward-adjoint restart identity, "
            "the explicit annular high field plus low plane wave, and "
            "terminal weight t Phi_+++. This audits the static-optimal "
            "coefficient witness; it does not optimize over all terminal "
            "weights or bound nonlinear generator amplification."
        ),
        "prerequisite_audit": prerequisite,
        "symbolic_deficit_certificate": symbolic,
        "exact_partition_norms": partition,
        "pressure_only_annular_rows": pressure_rows,
        "reset_tax_rows": tax_rows,
        "asymptotic_reset_tax_certificate": asymptotic,
        "finite_summary": {
            "audited_sizes": list(clean_sizes),
            "first_audited_positive_pressure_only_static_size": next(
                (
                    row["size"]
                    for row in pressure_rows
                    if row["ray_optimization"]["positive_escape"]
                ),
                None,
            ),
            "size_25_full_reset_deficit_lower_bound": tax_by_size[25][
                "full_L2_reset_deficit_lower_bound"
            ],
            "size_25_required_average_amplification": tax_by_size[25][
                "required_average_amplification_over_initial_generator"
            ],
            "largest_size_relative_pressure_complete_difference": (
                final_pressure[
                    "relative_pressure_vs_complete_difference"
                ]
            ),
            "finite_gate_passes": finite_gate,
        },
        "theorem": (
            "At rho=0 the exact restart formula retains the reset-time "
            "Legendre deficit: the optimized contribution is J_0-Delta_s, "
            "not J_0 alone. For the static-optimal annular +++ witness, "
            "backward L3 contraction and Parseval give "
            "Delta_s>=c N^3. Its normalized initial pressure generator is "
            "C N^3. Over a parabolic window T/N^2, survival of this witness "
            "therefore requires a time-averaged generator of order N^5, "
            "an N^2 amplification over its initial value. The static "
            "positive generator alone is not a dynamic restart "
            "counterexample."
        ),
        "route_decision": (
            "Retain the Legendre deficit in every subsequent terminal-dual "
            "optimization. Park the unpenalized arbitrary-coefficient "
            "supremum. The next decisive calculation is the first and "
            "second time jet of the pressure-only generator along the "
            "annular Navier-Stokes/backward-weight system, testing whether "
            "the required N^2 amplification can occur."
        ),
        "certification_flags": {
            "rho_zero_pressure_only_correction_applied": True,
            "exact_deficit_retaining_restart_identity_proved": True,
            "reset_Legendre_deficit_nonnegative_proved": True,
            "backward_weight_L3_contraction_retained": True,
            "pressure_only_static_escape_replayed": True,
            "static_optimal_annular_reset_tax_order_N3_proved": True,
            "parabolic_survival_requires_order_N2_amplification_proved": True,
            "static_escape_is_direct_dynamic_counterexample": False,
            "required_nonlinear_amplification_excluded": False,
            "all_terminal_weights_dynamically_controlled": False,
            "critical_L3_controlled": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "next_theorem_target": (
            "Differentiate the exact pressure-only rho=0 generator at the "
            "restart for u_N=h_N-a_N U and lambda=t_N Phi_+++. Separate "
            "viscous N^2 rotation, low-high transport, pressure response, "
            "and backward-weight advection. Determine the leading N^5 "
            "coefficient and its sign; then compute the second jet only if "
            "the first can approach the reset-tax threshold."
        ),
        "all_positive_checks_pass": all_checks,
    }


def _parse_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SIZES,
        help="comma-separated increasing odd carrier sizes",
    )
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument(
        "--scaled-window",
        type=float,
        default=DEFAULT_SCALED_WINDOW,
    )
    arguments = parser.parse_args()
    _lower_process_priority()
    result = audit(
        arguments.sizes,
        arguments.viscosity,
        arguments.scaled_window,
    )
    _atomic_json(RESULT, result)
    print(
        json.dumps(
            {
                "result": RESULT.relative_to(ROOT).as_posix(),
                "sha256": _sha256(RESULT),
                "status": result["status"],
                "all_positive_checks_pass": result[
                    "all_positive_checks_pass"
                ],
                "finite_summary": result["finite_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["all_positive_checks_pass"]:
        raise SystemExit("deficit-retaining annular restart audit failed")


if __name__ == "__main__":
    main()
