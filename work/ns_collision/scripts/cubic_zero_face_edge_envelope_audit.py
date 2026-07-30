"""Audit the sharp cubic zero-face envelope for pressure edges."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
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


def _sharp_edge_envelope_audit() -> dict[str, Any]:
    edge_size, flux_size, penalty = sp.symbols(
        "x y c",
        positive=True,
    )
    optimum_edge_size = sp.sqrt(flux_size / (3 * penalty))
    objective = edge_size * flux_size - penalty * edge_size**3
    optimum_value = sp.simplify(
        objective.subs(edge_size, optimum_edge_size)
    )
    expected_value = (
        2
        * flux_size ** sp.Rational(3, 2)
        / (3 * sp.sqrt(3 * penalty))
    )
    optimum_residual = sp.simplify(optimum_value - expected_value)

    viscosity, frequency = sp.symbols("nu m", positive=True)
    fisher_coefficient = viscosity * frequency**2 / 16
    pressure_constant = sp.simplify(
        expected_value.subs(penalty, fisher_coefficient)
        / flux_size ** sp.Rational(3, 2)
    )
    expected_pressure_constant = (
        8 / (3 * sp.sqrt(3) * frequency * sp.sqrt(viscosity))
    )

    random_rows = []
    rng = np.random.default_rng(20260726)
    for _ in range(16):
        edge_flux = float(10 ** rng.uniform(-3.0, 3.0))
        coefficient = float(10 ** rng.uniform(-2.0, 2.0))
        optimum = math.sqrt(edge_flux / (3.0 * coefficient))
        objective_value = (
            optimum * edge_flux - coefficient * optimum**3
        )
        envelope = (
            2.0
            * edge_flux**1.5
            / (3.0 * math.sqrt(3.0 * coefficient))
        )
        random_rows.append(
            {
                "edge_flux_size": edge_flux,
                "penalty_coefficient": coefficient,
                "optimal_face_difference": optimum,
                "objective": objective_value,
                "envelope": envelope,
                "residual": abs(objective_value - envelope),
            }
        )

    return {
        "face_variables": "H=A+B, D=A-B, with A,B>=0",
        "feasible_cone": "H>=|D|>=0",
        "cubic_domination": "H D^2>=|D|^3",
        "edge_objective": "D e-c H D^2",
        "sharp_scalar_reduction": (
            "D e-c H D^2<=x|e|-c x^3, x=|D|"
        ),
        "sharp_general_supremum": str(expected_value),
        "optimal_difference": str(optimum_edge_size),
        "optimizer_geometry": (
            "H=|D| and sign(D)=sign(e), so one of A,B is exactly zero"
        ),
        "Navier_Stokes_edge_penalty": "c=nu m^2/16",
        "sharp_pressure_edge_envelope": (
            "sup_(A,B>=0)[(A-B)e-(nu m^2/16)"
            "(A+B)(A-B)^2]="
            "8|e|^(3/2)/(3sqrt(3) m sqrt(nu))"
        ),
        "sharp_pressure_constant": str(pressure_constant),
        "symbolic_optimum_residual": str(optimum_residual),
        "random_rows": random_rows,
        "maximum_random_residual": max(
            row["residual"] for row in random_rows
        ),
        "all_checks_pass": (
            optimum_residual == 0
            and sp.simplify(
                pressure_constant - expected_pressure_constant
            )
            == 0
            and max(row["residual"] for row in random_rows) < 1.0e-10
        ),
    }


def _conditional_pressure_L32_audit() -> dict[str, Any]:
    viscosity, frequency, amplitude = sp.symbols(
        "nu m U",
        positive=True,
    )
    sine_cube_mean = sp.Rational(4, 1) / (3 * sp.pi)
    partition_derivative_L3_cubed = sp.simplify(
        frequency**3 * sine_cube_mean / 8
    )
    conditional_holder_factor = sp.simplify(
        sp.sqrt(partition_derivative_L3_cubed)
        * amplitude ** sp.Rational(3, 2)
    )
    sharp_edge_constant = (
        8 / (3 * sp.sqrt(3) * frequency * sp.sqrt(viscosity))
    )
    per_direction_constant = sp.simplify(
        sharp_edge_constant * conditional_holder_factor
    )
    expected_per_direction = (
        8
        / (9 * sp.sqrt(2 * sp.pi))
        * amplitude ** sp.Rational(3, 2)
        * sp.sqrt(frequency / viscosity)
    )
    three_direction_constant = sp.simplify(3 * per_direction_constant)
    intrinsic_value = sp.simplify(
        three_direction_constant.subs(
            frequency,
            amplitude / viscosity,
        )
    )

    return {
        "conditional_edge_density": (
            "e_j(x_hat_j)=mean_(x_j)[p u_j "
            "partial_j phi_+(m x_j)]"
        ),
        "partition_derivative_cube_mean": str(
            partition_derivative_L3_cubed
        ),
        "conditional_Holder": (
            "|e_j|^(3/2)<=U^(3/2)m^(3/2)/sqrt(6pi) "
            "mean_(x_j)|p|^(3/2)"
        ),
        "per_direction_remainder": (
            "8/(9sqrt(2pi)) U^(3/2)sqrt(m/nu) "
            "integral|p|^(3/2)"
        ),
        "per_direction_symbolic_constant": str(
            per_direction_constant
        ),
        "three_direction_remainder": (
            "8/(3sqrt(2pi)) U^(3/2)sqrt(m/nu) "
            "integral|p|^(3/2)"
        ),
        "three_direction_symbolic_constant": str(
            three_direction_constant
        ),
        "intrinsic_m_equals_U_over_nu": str(intrinsic_value),
        "pressure_L32_reduction": (
            "Using ||p||_(3/2)<=C_R||u||_3^2 gives a remainder "
            "<=8 C_R^(3/2)/(3sqrt(2pi)) "
            "(U^2/nu)||u||_3^3 at m=U/nu."
        ),
        "interpretation": (
            "The zero-face supremum is finite and critical, but the "
            "remaining U^2||u||_3^3/nu coefficient is not controlled by "
            "the Leray energy inequality. Intrinsic localization or signed "
            "inter-edge cancellation is still required."
        ),
        "all_checks_pass": (
            partition_derivative_L3_cubed
            == frequency**3 / (6 * sp.pi)
            and sp.simplify(
                per_direction_constant - expected_per_direction
            )
            == 0
            and intrinsic_value
            == 8 * amplitude**2 / (
                3 * sp.sqrt(2 * sp.pi) * viscosity
            )
        ),
    }


def _scale_homogeneity_audit() -> dict[str, Any]:
    amplitude, frequency, viscosity = sp.symbols(
        "a m nu",
        positive=True,
    )
    edge_density_scale = amplitude**3 * frequency
    cubic_envelope_scale = (
        edge_density_scale ** sp.Rational(3, 2)
        / (sp.sqrt(viscosity) * frequency)
    )
    fisher_scale = viscosity * amplitude**3 * frequency**2
    ratio = sp.simplify(cubic_envelope_scale / fisher_scale)
    local_reynolds = amplitude / (viscosity * frequency)
    residual = sp.simplify(
        ratio - local_reynolds ** sp.Rational(3, 2)
    )
    return {
        "coscaling": (
            "u_(a,m)=a u(mx), p_(a,m)=a^2p(mx), "
            "lambda_(a,m)=a lambda(mx)"
        ),
        "conditional_edge_density_scale": "a^3 m",
        "cubic_zero_face_envelope_scale": str(cubic_envelope_scale),
        "velocity_and_weight_Fisher_scale": "nu a^3 m^2",
        "envelope_to_Fisher_ratio": str(ratio),
        "local_Reynolds_power": "(a/(nu m))^(3/2)",
        "symbolic_residual": str(residual),
        "fixed_frequency_uniform_absorption_possible": False,
        "intrinsic_frequency_makes_ratio_scale_invariant": True,
        "all_checks_pass": residual == 0,
    }


def _taylor_green_edge_stress(size: int = 8192) -> dict[str, Any]:
    coordinate = 2.0 * math.pi * np.arange(size) / size
    edge_x = np.cos(coordinate) * (
        1.0 / 32.0 - np.cos(2.0 * coordinate) / 16.0
    )
    edge_y = -edge_x
    viscosity = 1.0
    frequency = 1.0
    coefficient = viscosity * frequency**2 / 16.0

    def optimized_row(edge: np.ndarray) -> dict[str, float]:
        difference = np.sign(edge) * np.sqrt(
            np.abs(edge) / (3.0 * coefficient)
        )
        face_sum = np.abs(difference)
        objective = (
            difference * edge
            - coefficient * face_sum * difference**2
        )
        envelope = (
            8.0
            / (3.0 * math.sqrt(3.0))
            * np.abs(edge) ** 1.5
        )
        return {
            "mean_objective": float(np.mean(objective)),
            "mean_envelope": float(np.mean(envelope)),
            "maximum_pointwise_residual": float(
                np.max(np.abs(objective - envelope))
            ),
            "maximum_face_difference": float(
                np.max(np.abs(difference))
            ),
        }

    row_x = optimized_row(edge_x)
    row_y = optimized_row(edge_y)
    return {
        "field": (
            "u=(sin x cos y,-cos x sin y,0), "
            "p=(cos 2x+cos 2y)/4"
        ),
        "partition_frequency": 1,
        "edge_x_formula": "cos(y)[1/32-cos(2y)/16]",
        "edge_y_formula": "-cos(x)[1/32-cos(2x)/16]",
        "edge_z_formula": "0",
        "grid_size": size,
        "x_direction": row_x,
        "y_direction": row_y,
        "summed_sharp_envelope": (
            row_x["mean_envelope"] + row_y["mean_envelope"]
        ),
        "scope": (
            "The directionwise optimizer permits independent conditional "
            "face values and therefore upper-bounds, but need not equal, "
            "the globally compatible eight-cell coefficient supremum."
        ),
        "all_checks_pass": (
            row_x["maximum_pointwise_residual"] < 2.0e-16
            and row_y["maximum_pointwise_residual"] < 2.0e-16
            and abs(
                row_x["mean_envelope"] - row_y["mean_envelope"]
            )
            < 1.0e-15
            and row_x["mean_envelope"] > 0.0
        ),
    }


def audit() -> dict[str, Any]:
    sharp = _sharp_edge_envelope_audit()
    pressure = _conditional_pressure_L32_audit()
    scaling = _scale_homogeneity_audit()
    stress = _taylor_green_edge_stress()
    positive_checks = {
        "sharp_edge_optimization_passes": sharp["all_checks_pass"],
        "conditional_pressure_reduction_passes": pressure[
            "all_checks_pass"
        ],
        "scale_homogeneity_passes": scaling["all_checks_pass"],
        "Taylor_Green_edge_stress_passes": stress["all_checks_pass"],
    }
    return {
        "kind": "cubic_zero_face_edge_envelope_audit",
        "schema_version": 1,
        "status": (
            "zero_face_singularity_removed_"
            "critical_L32_pressure_remainder_open"
        ),
        "assumption_scope": (
            "Smooth periodic velocity and pressure, nonnegative conditional "
            "face coefficients, and the cosine partition at frequency m. "
            "The global L3/2 pressure reduction uses an L-infinity velocity "
            "envelope. Directionwise optimization is a valid upper bound "
            "but relaxes compatibility among all edges of one global "
            "partition coefficient vector."
        ),
        "sharp_cubic_edge_envelope": sharp,
        "conditional_pressure_L32_reduction": pressure,
        "scale_homogeneity": scaling,
        "taylor_green_edge_stress": stress,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "sharp_zero_face_edge_supremum_derived": True,
            "zero_face_reciprocal_singularity_removed": True,
            "edge_remainder_reduced_to_pressure_L32": True,
            "cubic_edge_ratio_is_local_Reynolds_power_3_over_2": True,
            "full_nonnegative_directionwise_edge_supremum_preserved": True,
            "globally_compatible_partition_supremum_evaluated": False,
            "pressure_L32_remainder_absorbed": False,
            "signed_inter_edge_Carleson_cancellation_proved": False,
            "critical_signed_bound_proved": False,
            "low_regularity_passage_proved": False,
            "exceptional_set_upgrade_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "route_decision": (
            "Replace the reciprocal e^2/H Young remainder by the sharp "
            "cubic |e|^(3/2) envelope whenever taking the full nonnegative "
            "edge supremum. The zero-face singularity is not fundamental. "
            "The live obstruction is now quantitative control of the "
            "pressure L3/2 edge measure and compatibility/cancellation "
            "across the balanced dyadic graph."
        ),
        "next_theorem_target": (
            "Evaluate the globally compatible nonnegative coefficient "
            "supremum on the eight-cell partition rather than optimizing "
            "each edge independently. Derive its homogeneous cubic graph "
            "reduction, retain antisymmetric edge cancellation, and test "
            "whether 2:1 balance plus the intrinsic pressure-tail estimate "
            "lowers the directionwise L3/2 envelope enough to fit the "
            "available velocity Fisher budget."
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
            "cubic_zero_face_edge_envelope_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("cubic zero-face edge envelope audit failed")
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
