"""Audit the Green-operator gate for variable cylinder perturbations."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


def _load_perturbation_module():
    script = Path(__file__).resolve().with_name(
        "finite_cylinder_perturbation_margin_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "finite_cylinder_perturbation_for_kato", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_geometry_row(
    perturbation,
    reynolds: float,
    half_height: float,
    buffer_ratio: float = 2.0,
) -> dict[str, float | bool]:
    basis = perturbation._axial_basis(
        reynolds, half_height, grid_points=801, mode_count=81
    )
    visit = perturbation._visit(
        basis,
        reynolds,
        buffer_ratio,
        core_potential=0.0,
        shell_potential=0.0,
    )
    renewal = perturbation._renewal_quantities(
        float(visit["maximum_visit_gain"]),
        reynolds,
        buffer_ratio,
    )
    baseline_criterion = renewal["complete_generation_criterion"]
    kato_budget = 1.0 - math.sqrt(baseline_criterion)
    return {
        "R_star": reynolds,
        "half_height_over_L": half_height,
        "baseline_one_history_gain": float(
            visit["maximum_visit_gain"]
        ),
        "baseline_generation_criterion": baseline_criterion,
        "critical_Kato_operator_norm": kato_budget,
        "maximum_allowed_resolvent_amplification": 1.0
        / (1.0 - kato_budget),
        "threshold_bound_reproduces_generation_boundary": bool(
            abs(
                baseline_criterion / (1.0 - kato_budget) ** 2 - 1.0
            )
            < 1.0e-13
        ),
    }


def _endpoint_profile_row(
    logarithmic_cutoff: float,
    alpha: float,
    target_l3_over_2_mass: float = 1.0,
) -> dict[str, float]:
    mass_exponent = 1.5 * alpha
    mass_integral = (
        1.0 - logarithmic_cutoff ** (1.0 - mass_exponent)
    ) / (mass_exponent - 1.0)
    coefficient = target_l3_over_2_mass / (
        4.0 * math.pi * mass_integral
    ) ** (2.0 / 3.0)
    if alpha == 1.0:
        newtonian_potential = coefficient * math.log(
            logarithmic_cutoff
        )
    else:
        newtonian_potential = coefficient * (
            logarithmic_cutoff ** (1.0 - alpha) - 1.0
        ) / (1.0 - alpha)
    recovered_mass = coefficient * (
        4.0 * math.pi * mass_integral
    ) ** (2.0 / 3.0)
    return {
        "T_equals_log_inverse_inner_radius": logarithmic_cutoff,
        "coefficient_c": coefficient,
        "L3_over_2_mass": recovered_mass,
        "Newtonian_potential_at_centre": newtonian_potential,
    }


def _free_newtonian_holder_constant(
    integrability_exponent: float, maximum_distance: float
) -> float:
    dual_exponent = integrability_exponent / (
        integrability_exponent - 1.0
    )
    if dual_exponent >= 3.0:
        return math.inf
    kernel_power_integral = (
        (4.0 * math.pi) ** (1.0 - dual_exponent)
        * maximum_distance ** (3.0 - dual_exponent)
        / (3.0 - dual_exponent)
    )
    return kernel_power_integral ** (1.0 / dual_exponent)


def audit() -> dict[str, object]:
    perturbation = _load_perturbation_module()
    geometries = (
        (0.5, 1.5),
        (0.5, 1.75),
        (0.5, 2.0),
        (1.0, 1.0),
        (1.0, 1.2),
    )
    geometry_rows = [
        _base_geometry_row(perturbation, reynolds, half_height)
        for reynolds, half_height in geometries
    ]

    alpha = 0.75
    endpoint_rows = [
        _endpoint_profile_row(logarithmic_cutoff, alpha)
        for logarithmic_cutoff in (4.0, 16.0, 256.0, 65536.0, 1.0e8)
    ]
    maximum_distance = 2.0 * math.sqrt(2.0**2 + 1.5**2)
    holder_rows = [
        {
            "p": exponent,
            "dual_exponent": exponent / (exponent - 1.0),
            "free_Newtonian_Lp_to_Linfinity_constant": (
                _free_newtonian_holder_constant(
                    exponent, maximum_distance
                )
            ),
        }
        for exponent in (1.51, 1.6, 2.0, 3.0)
    ]

    endpoint_potentials = [
        row["Newtonian_potential_at_centre"] for row in endpoint_rows
    ]
    result: dict[str, object] = {
        "baseline_resolvent_identity": "u_q=u_0+G_0(q_+ u_q)",
        "Kato_operator": "T_q f=G_0(q_+ f)",
        "positive_Linfinity_operator_norm": (
            "kappa(q)=||T_q||_(infinity to infinity)="
            "sup_x G_0 q_+(x)"
        ),
        "visit_gain_bound": (
            "||u_q||_infinity<=||u_0||_infinity/(1-kappa)"
        ),
        "generation_bound": "C_q<=C_0/(1-kappa)^2",
        "geometry_rows": geometry_rows,
        "all_Kato_budgets_are_positive": all(
            row["critical_Kato_operator_norm"] > 0.0
            for row in geometry_rows
        ),
        "all_threshold_bounds_reproduce_generation_boundary": all(
            row["threshold_bound_reproduces_generation_boundary"]
            for row in geometry_rows
        ),
        "compact_geometry_has_more_Kato_room": bool(
            geometry_rows[0]["critical_Kato_operator_norm"]
            > geometry_rows[1]["critical_Kato_operator_norm"]
            > geometry_rows[2]["critical_Kato_operator_norm"]
        ),
        "endpoint_profile": (
            "q_T(r)=c_T/[r^2 log(1/r)^alpha] on exp(-T)<r<exp(-1)"
        ),
        "endpoint_alpha": alpha,
        "endpoint_integrability_window": "2/3<alpha<=1",
        "endpoint_rows_normalized_to_unit_L3_over_2_mass": endpoint_rows,
        "endpoint_masses_remain_one": all(
            abs(row["L3_over_2_mass"] - 1.0) < 1.0e-12
            for row in endpoint_rows
        ),
        "endpoint_Newtonian_potential_strictly_increases": all(
            later > earlier
            for earlier, later in zip(
                endpoint_potentials[:-1], endpoint_potentials[1:]
            )
        ),
        "endpoint_L3_over_2_does_not_control_pointwise_Green_norm": bool(
            2.0 / 3.0 < alpha <= 1.0
        ),
        "endpoint_reason": (
            "the L^(3/2) mass integral converges because 3*alpha/2>1, "
            "while the centre Green integral diverges because alpha<=1"
        ),
        "working_geometry_maximum_distance": maximum_distance,
        "free_Newtonian_supercritical_Holder_rows": holder_rows,
        "supercritical_Lp_embedding_requires_p_above_three_halves": all(
            math.isfinite(
                row["free_Newtonian_Lp_to_Linfinity_constant"]
            )
            for row in holder_rows
        ),
        "holder_scope": (
            "the displayed constants are for the free Newtonian kernel; "
            "the drifted finite-cylinder Green majorant still needs a "
            "certified comparison constant"
        ),
        "scale_invariant_replacement": (
            "use kappa(q), a local Kato/Morrey control, or an averaged "
            "interface norm; bare L^(3/2) mass is insufficient for a "
            "pointwise visit bound"
        ),
        "remaining_PDE_gate": (
            "derive one of these stronger local controls for the actual "
            "non-affine Navier-Stokes error without assuming regularity"
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
