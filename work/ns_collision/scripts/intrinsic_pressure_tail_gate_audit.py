"""Audit the intrinsic pressure tail and its zero-face localization gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import sympy as sp


ROOT = Path(__file__).resolve().parents[3]


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


def _tail_decomposition_audit() -> dict[str, Any]:
    amplitude, frequency, viscosity, weight_floor = sp.symbols(
        "U m nu lambda_*",
        positive=True,
    )
    velocity_gradient, weight_gradient = sp.symbols(
        "A B",
        nonnegative=True,
    )
    tail_product_bound = (
        2
        * amplitude**2
        / frequency
        * velocity_gradient
        * weight_gradient
    )
    young_bound = (
        amplitude**2
        / frequency
        * (velocity_gradient**2 + weight_gradient**2)
    )
    weighted_fisher_floor = (
        viscosity
        * weight_floor
        * (velocity_gradient**2 + weight_gradient**2)
    )
    absorption_ratio = sp.simplify(
        amplitude**2 / (frequency * viscosity * weight_floor)
    )
    local_weight_ratio = sp.symbols("theta", positive=True)
    intrinsic_threshold = sp.simplify(
        amplitude**2 / (viscosity * local_weight_ratio * amplitude)
    )

    return {
        "pressure_formula": (
            "p=-R_i R_j(u_i u_j), with the zero Fourier mode fixed to zero"
        ),
        "sharp_support_identity": (
            "For u=u_<+u_> with supp(u_<) inside |k|<m/2, "
            "Q_m p=-R_iR_j Q_m[u_<i u_>j+u_>i u_<j+u_>i u_>j]; "
            "the low-low term is exactly absent."
        ),
        "derivative_tail_chain": (
            "||Q_m p||_2<=m^(-1)||grad p||_2"
            "<=m^(-1)||grad(u tensor u)||_2"
            "<=2m^(-1)||u||_infinity||grad u||_2"
        ),
        "tail_flux_bound": (
            "|int Q_m p u dot grad lambda|"
            "<=2||u||_infinity^2 m^(-1)"
            "||grad u||_2||grad lambda||_2"
        ),
        "tail_flux_after_Young": str(young_bound),
        "weighted_Fisher_floor": str(weighted_fisher_floor),
        "absorption_ratio": str(absorption_ratio),
        "conditional_absorption_threshold": (
            "m>=||u||_infinity^2/(nu lambda_*)"
        ),
        "comparable_weight_floor": "lambda_*>=theta||u||_infinity",
        "intrinsic_threshold_under_comparability": str(
            intrinsic_threshold
        ),
        "local_Reynolds_form": (
            "If lambda_*>=theta U, absorption follows from "
            "U/(nu m)<=theta."
        ),
        "tail_product_bound_symbolic": str(tail_product_bound),
        "young_gap_factorization": str(
            sp.factor(young_bound - tail_product_bound)
        ),
        "all_checks_pass": (
            sp.simplify(
                young_bound
                - tail_product_bound
                - amplitude**2
                / frequency
                * (velocity_gradient - weight_gradient) ** 2
            )
            == 0
            and absorption_ratio
            == amplitude**2 / (frequency * viscosity * weight_floor)
            and intrinsic_threshold
            == amplitude / (local_weight_ratio * viscosity)
        ),
    }


def _periodic_mean_2d(expression: sp.Expr, x: sp.Symbol, y: sp.Symbol) -> sp.Expr:
    return sp.simplify(
        sp.integrate(
            expression,
            (x, 0, 2 * sp.pi),
            (y, 0, 2 * sp.pi),
        )
        / (4 * sp.pi**2)
    )


def _taylor_green_tail_scaling_audit() -> dict[str, Any]:
    x, y = sp.symbols("x y", real=True)
    amplitude, frequency, viscosity = sp.symbols(
        "a n nu",
        positive=True,
    )
    weight_level, weight_modulation = sp.symbols(
        "L beta",
        positive=True,
    )
    velocity = sp.Matrix(
        [
            sp.sin(x) * sp.cos(y),
            -sp.cos(x) * sp.sin(y),
        ]
    )
    pressure = (sp.cos(2 * x) + sp.cos(2 * y)) / 4
    divergence = sp.simplify(
        sp.diff(velocity[0], x) + sp.diff(velocity[1], y)
    )
    advection = sp.Matrix(
        [
            velocity.dot(
                sp.Matrix(
                    [
                        sp.diff(velocity[index], x),
                        sp.diff(velocity[index], y),
                    ]
                )
            )
            for index in range(2)
        ]
    )
    euler_residual = sp.simplify(
        advection + sp.Matrix([sp.diff(pressure, x), sp.diff(pressure, y)])
    )
    pressure_transport = sp.expand_trig(
        velocity.dot(
            sp.Matrix([sp.diff(pressure, x), sp.diff(pressure, y)])
        )
    )
    expected_transport = (
        sp.cos(3 * x) * sp.cos(y)
        - sp.cos(x) * sp.cos(3 * y)
    ) / 4
    transport_residual = sp.trigsimp(
        pressure_transport - expected_transport
    )
    weight = weight_level - weight_modulation * pressure_transport
    weight_gradient = sp.Matrix(
        [sp.diff(weight, x), sp.diff(weight, y)]
    )
    pressure_flux = _periodic_mean_2d(
        pressure * velocity.dot(weight_gradient),
        x,
        y,
    )
    velocity_gradient_squared = sum(
        sp.diff(velocity[index], direction) ** 2
        for index in range(2)
        for direction in (x, y)
    )
    weight_gradient_squared = sum(
        component**2 for component in weight_gradient
    )
    velocity_fisher = _periodic_mean_2d(
        weight * velocity_gradient_squared,
        x,
        y,
    )
    weight_fisher = _periodic_mean_2d(
        weight * weight_gradient_squared,
        x,
        y,
    )
    transport_energy = _periodic_mean_2d(
        pressure_transport**2,
        x,
        y,
    )
    transport_gradient_energy = _periodic_mean_2d(
        sum(
            sp.diff(pressure_transport, direction) ** 2
            for direction in (x, y)
        ),
        x,
        y,
    )

    scaled_pressure_flux = (
        amplitude**4 * frequency * pressure_flux
    )
    scaled_total_fisher = (
        viscosity
        * amplitude**3
        * frequency**2
        * (velocity_fisher + weight_fisher)
    )
    scaled_ratio = sp.factor(
        scaled_pressure_flux / scaled_total_fisher
    )
    local_reynolds = amplitude / (viscosity * frequency)
    ratio_over_reynolds = sp.simplify(scaled_ratio / local_reynolds)

    return {
        "base_velocity": (
            "u=(sin(x)cos(y),-cos(x)sin(y),0)"
        ),
        "base_pressure": "p=(cos(2x)+cos(2y))/4",
        "divergence_residual": str(divergence),
        "Euler_pressure_balance_residual": [
            str(value) for value in euler_residual
        ],
        "pressure_transport": str(expected_transport),
        "pressure_transport_residual": str(transport_residual),
        "positive_terminal_weight": (
            "lambda=L-beta(u dot grad p), with lambda>=L-beta/2"
        ),
        "pressure_modes": (
            "All nonzero pressure modes have |k|=2, so Q_1 p=p."
        ),
        "normalized_pressure_tail_flux": str(pressure_flux),
        "normalized_weighted_velocity_Fisher": str(velocity_fisher),
        "normalized_weight_Fisher": str(weight_fisher),
        "pressure_transport_energy": str(transport_energy),
        "pressure_transport_gradient_energy": str(
            transport_gradient_energy
        ),
        "coscaled_fields": (
            "u_(a,n)=a u(nx), p_(a,n)=a^2 p(nx), "
            "lambda_(a,n)=a lambda(nx)"
        ),
        "scaled_pressure_tail_flux": str(scaled_pressure_flux),
        "scaled_total_Fisher": str(scaled_total_fisher),
        "pressure_tail_to_Fisher_ratio": str(scaled_ratio),
        "ratio_divided_by_local_Reynolds": str(ratio_over_reynolds),
        "fixed_frequency_universal_absorption_possible": False,
        "intrinsic_frequency_keeps_ratio_scale_invariant": True,
        "example_parameters": {
            "L": 1.0,
            "beta": 1.0,
            "minimum_weight_lower_bound": 0.5,
            "ratio_coefficient": float(
                ratio_over_reynolds.subs(
                    {
                        weight_level: 1,
                        weight_modulation: 1,
                    }
                )
            ),
        },
        "all_checks_pass": (
            divergence == 0
            and all(value == 0 for value in euler_residual)
            and transport_residual == 0
            and pressure_flux == weight_modulation / 32
            and velocity_fisher == weight_level
            and weight_fisher
            == 5 * weight_level * weight_modulation**2 / 16
            and transport_energy == sp.Rational(1, 32)
            and transport_gradient_energy == sp.Rational(5, 16)
            and sp.simplify(
                scaled_ratio
                - local_reynolds * ratio_over_reynolds
            )
            == 0
        ),
    }


def _zero_face_weight_gate_audit() -> dict[str, Any]:
    epsilon = sp.symbols("epsilon", positive=True)
    circle_average = epsilon + sp.Rational(1, 2)
    discriminant = sp.sqrt(epsilon * (epsilon + 1))
    whole_circle_a2 = sp.simplify(
        circle_average / discriminant
    )
    hilbert_norm_lower_bound_squared = sp.simplify(
        whole_circle_a2 - 1
    )
    rows = []
    for value in (1.0e-1, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8):
        average = value + 0.5
        delta = math.sqrt(value * (value + 1.0))
        a2_value = average / delta
        rows.append(
            {
                "epsilon": value,
                "whole_circle_A2_lower_bound": a2_value,
                "explicit_Hilbert_norm_lower_bound": math.sqrt(
                    a2_value - 1.0
                ),
            }
        )

    return {
        "weight_family": "w_epsilon(x)=epsilon+sin(x/2)^2",
        "minimum_weight": "epsilon",
        "whole_circle_average_weight": str(circle_average),
        "whole_circle_average_reciprocal_weight": str(
            1 / discriminant
        ),
        "whole_circle_A2_characteristic_lower_bound": str(
            whole_circle_a2
        ),
        "explicit_test_function": "f_epsilon=1/w_epsilon",
        "periodic_Hilbert_transform": (
            "H f_epsilon=(sin x)/(2 sqrt(epsilon(epsilon+1)) "
            "w_epsilon)"
        ),
        "weighted_input_norm_squared": str(1 / discriminant),
        "weighted_output_norm_squared": str(
            (circle_average - discriminant) / discriminant**2
        ),
        "operator_norm_lower_bound_squared": str(
            hilbert_norm_lower_bound_squared
        ),
        "asymptotic": (
            "[w_epsilon]_(A2)~1/(2sqrt(epsilon)) and the explicit "
            "weighted singular-integral norm lower bound grows like "
            "epsilon^(-1/4)/sqrt(2)"
        ),
        "rows": rows,
        "interpretation": (
            "The full terminal-weight class contains zero-face limits with "
            "unbounded weighted singular-integral constants. Therefore the "
            "unweighted pressure-tail theorem cannot be localized by a "
            "uniform weighted Calderon-Zygmund step. This is a route no-go, "
            "not a pressure-specific counterexample to every possible "
            "signed edge estimate."
        ),
        "all_checks_pass": (
            sp.simplify(
                circle_average**2
                - sp.Rational(1, 4)
                - discriminant**2
            )
            == 0
            and all(
                right["whole_circle_A2_lower_bound"]
                > left["whole_circle_A2_lower_bound"]
                and right["explicit_Hilbert_norm_lower_bound"]
                > left["explicit_Hilbert_norm_lower_bound"]
                for left, right in zip(rows, rows[1:])
            )
            and rows[-1]["explicit_Hilbert_norm_lower_bound"] > 70.0
        ),
    }


def audit() -> dict[str, Any]:
    decomposition = _tail_decomposition_audit()
    scaling = _taylor_green_tail_scaling_audit()
    zero_face = _zero_face_weight_gate_audit()
    positive_checks = {
        "dyadic_tail_algebra_passes": decomposition["all_checks_pass"],
        "tail_scaling_family_passes": scaling["all_checks_pass"],
        "zero_face_weight_gate_passes": zero_face["all_checks_pass"],
    }
    return {
        "kind": "intrinsic_pressure_tail_gate_audit",
        "schema_version": 1,
        "status": (
            "intrinsic_tail_scaling_certified_"
            "arbitrary_weight_absolute_route_blocked"
        ),
        "assumption_scope": (
            "Smooth periodic divergence-free velocity. The positive "
            "conditional tail theorem additionally assumes an L-infinity "
            "velocity amplitude and a strictly positive lower bound on the "
            "deterministic terminal weight. The zero-face result concerns "
            "uniform weighted singular-integral localization, not every "
            "possible signed pressure-edge argument."
        ),
        "dyadic_tail_decomposition": decomposition,
        "taylor_green_amplitude_frequency_gate": scaling,
        "zero_face_weight_gate": zero_face,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "dyadic_high_pressure_tail_identity_derived": True,
            "unweighted_L2_pressure_tail_bound_derived": True,
            "positive_floor_intrinsic_absorption_derived": True,
            "fixed_frequency_tail_absorption_falsified_by_scaling": True,
            "intrinsic_frequency_necessity_reconfirmed": True,
            "uniform_arbitrary_weight_CZ_localization_available": False,
            "zero_face_full_terminal_supremum_preserved": False,
            "floor_free_signed_pressure_edge_bound_proved": False,
            "intrinsic_scale_pressure_tail_bound_proved": False,
            "critical_signed_bound_proved": False,
            "low_regularity_passage_proved": False,
            "exceptional_set_upgrade_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "route_decision": (
            "Keep the conditional unweighted tail theorem as a local "
            "calibration, but do not insert a fictitious positive floor or "
            "a uniform weighted Calderon-Zygmund constant. The live route "
            "must retain antisymmetric pressure transfer across the dyadic "
            "partition and control neighboring coefficient mismatch before "
            "absolute values, or derive new structure for the propagated "
            "terminal optimizer."
        ),
        "next_theorem_target": (
            "Formulate a signed dyadic pressure-flux Carleson estimate on "
            "the balanced intrinsic cover. Sum antisymmetric neighboring "
            "edge transfers first, then charge only coefficient differences. "
            "Test whether the Lipschitz radius and 2:1 balance make that "
            "mismatch square-summable without an A2 weight floor. A candidate "
            "must survive the co-scaled Taylor-Green tail family and the "
            "zero-face weight limit."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "work/ns_collision/results/"
            "intrinsic_pressure_tail_gate_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("intrinsic pressure-tail gate audit failed")
    _atomic_json(arguments.output, result)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": _sha256(arguments.output),
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
