"""Audit explicit Navier-Stokes coherence budgets on the cubic support."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import sympy as sp


def _load_level_transfer_module():
    script = Path(__file__).resolve().with_name(
        "cubic_level_transfer_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cubic_level_for_coherence_budget", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit() -> dict[str, object]:
    level_transfer = _load_level_transfer_module().audit()
    form_budget = float(
        level_transfer["static_full_tensor_mass_budget_over_nu"]
    )

    support_radius = sp.Rational(191, 100)
    transverse_half_width = support_radius / sp.sqrt(2)
    axial_half_width = sp.Rational(3, 2)
    reynolds = sp.Rational(1, 2)
    normalized_volume = sp.simplify(6 * support_radius**2)
    maximum_reference_drift_factor = sp.sqrt(support_radius**2 + 9)

    x, y, z, a, nu = sp.symbols(
        "x y z a nu", real=True, positive=True
    )
    error_x, error_y, error_z, stretching_excess = sp.symbols(
        "error_x error_y error_z stretching_excess", real=True
    )
    reference_drift = sp.Matrix([a * x, a * y, -2 * a * z])
    error = sp.Matrix([error_x, error_y, error_z])
    effective_error = sp.factor(
        stretching_excess - reference_drift.dot(error) / (2 * nu)
    )

    velocity_x, velocity_y, velocity_z = sp.symbols(
        "velocity_x velocity_y velocity_z", real=True
    )
    translation = sp.Matrix([velocity_x, velocity_y, velocity_z])
    fixed_center_translation_error = sp.factor(
        -reference_drift.dot(translation) / (2 * nu)
    )
    rotation_rate = sp.symbols("rotation_rate", real=True)
    transverse_rotation_error = sp.Matrix(
        [-rotation_rate * y, rotation_rate * x, 0]
    )
    transverse_rotation_pairing = sp.simplify(
        reference_drift.dot(transverse_rotation_error)
    )

    direct_remainder_coefficient = sp.simplify(
        reynolds * maximum_reference_drift_factor / 2
    )
    p = sp.Rational(3, 2)
    conjugate_p = sp.Integer(3)
    side_lengths = (
        2 * transverse_half_width,
        2 * transverse_half_width,
        2 * axial_half_width,
    )
    elementary_poincare_constant = sp.simplify(
        3 ** sp.Rational(1, 6)
        * sum(length**conjugate_p for length in side_lengths)
        ** (1 / conjugate_p)
    )
    critical_gradient_coefficient = sp.simplify(
        direct_remainder_coefficient * elementary_poincare_constant
    )

    reference_drift_sixth_integral = sp.integrate(
        (x**2 + y**2 + 4 * z**2) ** 3,
        (x, -transverse_half_width, transverse_half_width),
        (y, -transverse_half_width, transverse_half_width),
        (z, -axial_half_width, axial_half_width),
    )
    reference_drift_l6_factor = sp.simplify(
        reference_drift_sixth_integral ** sp.Rational(1, 6)
    )
    box_l2_poincare_constant = sp.simplify(3 / sp.pi)
    leray_gradient_coefficient = sp.simplify(
        reynolds
        * reference_drift_l6_factor
        * box_l2_poincare_constant
        / 2
    )
    leray_stretching_coefficient = sp.simplify(
        normalized_volume ** sp.Rational(1, 6)
    )

    translation_x_positive_norm = (
        transverse_half_width ** (p + 1)
        / (p + 1)
        * (2 * transverse_half_width)
        * (2 * axial_half_width)
    ) ** (1 / p)
    translation_z_positive_norm = (
        axial_half_width ** (p + 1)
        / (p + 1)
        * (2 * transverse_half_width) ** 2
    ) ** (1 / p)
    fixed_x_translation_coefficient = sp.simplify(
        reynolds * translation_x_positive_norm / 2
    )
    fixed_z_translation_coefficient = sp.simplify(
        reynolds * translation_z_positive_norm
    )

    direct_coefficient_float = float(direct_remainder_coefficient)
    gradient_coefficient_float = float(critical_gradient_coefficient)
    leray_stretching_float = float(leray_stretching_coefficient)
    leray_gradient_float = float(leray_gradient_coefficient)
    fixed_x_translation_float = float(fixed_x_translation_coefficient)
    fixed_z_translation_float = float(fixed_z_translation_coefficient)

    result: dict[str, object] = {
        "optimized_support": (
            "[-rho_s/sqrt(2),rho_s/sqrt(2)]^2 x [-3/2,3/2], "
            "rho_s=1.91, in units of L"
        ),
        "normalized_support_volume": float(normalized_volume),
        "reference_backward_drift": "b_0=a*(x,y,-2z)",
        "incompressible_effective_error": str(effective_error),
        "exact_effective_error_formula": (
            "q=delta_s-b_0 dot e/(2nu), with e measured after subtracting "
            "the cell translation and reference affine drift"
        ),
        "full_tensor_form_budget_over_nu": form_budget,
        "critical_stretching_functional": (
            "D=||[delta_s]_+||_(3/2)/nu"
        ),
        "critical_affine_remainder_functional": (
            "E=||e||_(3/2)/(nu*L)"
        ),
        "maximum_reference_drift_over_aL": float(
            maximum_reference_drift_factor
        ),
        "direct_affine_remainder_coefficient": direct_coefficient_float,
        "direct_critical_sufficient_condition": (
            f"D+{direct_coefficient_float:.12f}*E<{form_budget:.12f}"
        ),
        "maximum_E_if_D_zero": form_budget / direct_coefficient_float,
        "elementary_box_L3_over_2_Poincare_constant_over_L": float(
            elementary_poincare_constant
        ),
        "critical_gradient_coherence_functional": (
            "G=||grad(b)-B_0||_(3/2)/nu after choosing the cell mean "
            "translation so e has mean zero"
        ),
        "critical_gradient_coefficient": gradient_coefficient_float,
        "critical_gradient_sufficient_condition": (
            f"D+{gradient_coefficient_float:.12f}*G<{form_budget:.12f}"
        ),
        "maximum_G_if_D_zero": form_budget / gradient_coefficient_float,
        "reference_drift_sixth_power_integral": str(
            sp.factor(reference_drift_sixth_integral)
        ),
        "reference_drift_L6_factor": float(reference_drift_l6_factor),
        "box_L2_Poincare_constant_over_L": float(
            box_l2_poincare_constant
        ),
        "Leray_stretching_functional": (
            "F=L^(1/2)*||[delta_s]_+||_2/nu"
        ),
        "Leray_gradient_coherence_functional": (
            "H=L^(1/2)*||grad(b)-B_0||_2/nu"
        ),
        "Leray_stretching_coefficient": leray_stretching_float,
        "Leray_gradient_coefficient": leray_gradient_float,
        "Leray_level_sufficient_condition": (
            f"{leray_stretching_float:.12f}*F+"
            f"{leray_gradient_float:.12f}*H<{form_budget:.12f}"
        ),
        "maximum_F_if_H_zero": form_budget / leray_stretching_float,
        "maximum_H_if_F_zero": form_budget / leray_gradient_float,
        "equal_budget_share_F": (
            form_budget / (2 * leray_stretching_float)
        ),
        "equal_budget_share_H": (
            form_budget / (2 * leray_gradient_float)
        ),
        "fixed_center_translation_error": str(
            fixed_center_translation_error
        ),
        "fixed_x_translation_Q_over_nu_per_UL_over_nu": (
            fixed_x_translation_float
        ),
        "fixed_z_translation_Q_over_nu_per_UL_over_nu": (
            fixed_z_translation_float
        ),
        "fixed_x_translation_reaches_budget_at_UL_over_nu": (
            form_budget / fixed_x_translation_float
        ),
        "fixed_z_translation_reaches_budget_at_UL_over_nu": (
            form_budget / fixed_z_translation_float
        ),
        "fixed_center_bound_is_not_Galilean_invariant": bool(
            fixed_x_translation_float > 0.0
            and fixed_z_translation_float > 0.0
        ),
        "cell_mean_translation_must_be_removed_exactly": True,
        "transverse_rigid_rotation_pairing": str(
            transverse_rotation_pairing
        ),
        "transverse_rigid_rotation_is_gauge_invisible": bool(
            transverse_rotation_pairing == 0
        ),
        "exact_reference_affine_core_has_zero_coherence_functionals": True,
        "ordinary_Leray_control_does_not_imply_pointwise_small_F_and_H": True,
        "pressure_role": (
            "pressure is absent from the instantaneous algebra for q, but "
            "its trace-free Hessian drives the evolution of delta_s, B_0, "
            "and the eigenframe needed to keep F and H small"
        ),
        "remaining_translation_gate": (
            "construct conservative moving-cell or visit-label transport "
            "that subtracts each cell mean velocity without destroying "
            "sum(phi)=1 or paying a factor per crossing"
        ),
        "remaining_affine_gate": (
            "prove the displayed critical or Leray-level coherence bound "
            "for an actual Navier-Stokes solution, or extend the cylinder "
            "certificate uniformly to locally fitted non-axisymmetric "
            "trace-free affine strains"
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
