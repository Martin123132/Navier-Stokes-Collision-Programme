"""Audit the unavoidable affine-continuation error of the tapered shell."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def _load_script(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_mismatch_l3(
    taper_polynomial,
    taper_radius: float,
    outer_radius: float,
    half_height: float,
    strength: float,
    radial_points: int = 2_001,
    angular_points: int = 2_048,
) -> float:
    radius = np.linspace(1.0, outer_radius, radial_points)
    angle = 2.0 * math.pi * np.arange(angular_points) / angular_points
    cosine = np.cos(angle)[None, :]
    sine = np.sin(angle)[None, :]
    radial = radius[:, None]
    normalized = (radius - 1.0) / (taper_radius - 1.0)
    active = radius < taper_radius
    taper = np.zeros_like(radius)
    first = np.zeros_like(radius)
    taper[active] = taper_polynomial(normalized[active])
    first[active] = (
        taper_polynomial.deriv(1)(normalized[active])
        / (taper_radius - 1.0)
    )
    x = radial * cosine
    y = radial * sine
    taper_2d = taper[:, None]
    first_2d = first[:, None]
    tapered_x = x * taper_2d + x * y**2 * first_2d / radial
    tapered_y = -y * taper_2d - x**2 * y * first_2d / radial
    mismatch_x = strength * (x - tapered_x)
    mismatch_y = strength * (-y - tapered_y)
    magnitude_cubed = (mismatch_x**2 + mismatch_y**2) ** 1.5
    angular_integral = 2.0 * math.pi * np.mean(magnitude_cubed, axis=1)
    volume_integral = 2.0 * half_height * np.trapezoid(
        angular_integral * radius, radius
    )
    return float(volume_integral ** (1.0 / 3.0))


def audit() -> dict[str, object]:
    taper_module = _load_script(
        "divergence_free_shell_taper_audit.py",
        "divergence_free_taper_for_affine_residual",
    )
    sector_module = _load_script(
        "sectorial_poisson_transfer_audit.py",
        "sectorial_poisson_for_affine_residual",
    )
    affine_module = _load_script(
        "anisotropic_poisson_transfer_pilot.py",
        "compact_affine_for_affine_residual",
    )
    taper_radius = 2.65
    tapered_outer_radius = 2.75
    tapered_half_height = 1.2
    taper_polynomial, _ = taper_module.optimize_taper(
        taper_radius,
        maximum_degree=16,
        constraint_points=4_001,
        validation_points=50_001,
    )

    collar_l3_per_strength = (
        4.0
        * math.pi
        * tapered_half_height
        * (tapered_outer_radius**5 - taper_radius**5)
        / 5.0
    ) ** (1.0 / 3.0)
    spectrum_rows = []
    for t_parameter in np.linspace(-0.5, 1.0, 7):
        strength = float(t_parameter + 0.5)
        spectrum_rows.append(
            {
                "t_parameter": float(t_parameter),
                "anisotropic_strength": strength,
                "outer_collar_L3_lower_bound_over_nu": (
                    strength * collar_l3_per_strength
                ),
            }
        )
    worst_strength = 1.5
    full_worst_mismatch = _full_mismatch_l3(
        taper_polynomial,
        taper_radius,
        tapered_outer_radius,
        tapered_half_height,
        worst_strength,
    )

    sector = sector_module.audit()
    tapered_drift_budget = float(
        sector["maximum_drift_L3_over_nu_if_alpha_zero"]
    )
    maximum_strength_allowed_by_collar = (
        tapered_drift_budget / collar_l3_per_strength
    )
    maximum_t_allowed_by_collar = maximum_strength_allowed_by_collar - 0.5

    compact_half_height = 0.75
    compact_axial_eigenvalue = affine_module._axial_principal_eigenvalue(
        compact_half_height
    )
    compact_radii = affine_module._radial_mesh(24, 30, 30)
    compact_row = affine_module._transfer_row(
        1.0,
        compact_axial_eigenvalue,
        compact_radii,
        48,
        continuation="full_affine",
    )
    compact_generation = float(
        compact_row["baseline_generation_criterion"]
    )
    compact_good_norm = math.sqrt(compact_generation)
    compact_condition = float(
        compact_row["Poisson_cutoff_condition_number"]
    )
    compact_intercept = (
        1.0 / compact_good_norm - 1.0
    ) / compact_condition
    sharp_sobolev = 4.0 ** (2.0 / 3.0) / (
        3.0 * math.pi ** (4.0 / 3.0)
    )
    compact_equal_share = compact_intercept / (2.0 + compact_intercept)
    pair_true_split = 0.40869503866769363
    compact_split_intercept = (
        (1.0 - pair_true_split) / compact_good_norm - 1.0
    ) / compact_condition
    compact_split_equal_share = compact_split_intercept / (
        2.0 + compact_split_intercept
    )

    result: dict[str, object] = {
        "exact_affine_transverse_drift": (
            "b_aniso=s*(x,-y), s=t+1/2"
        ),
        "tapered_reference_transverse_drift": (
            "b_taper=s*(x*f+x*y^2*f'/r,-y*f-x^2*y*f'/r)"
        ),
        "collar_identity": (
            "for r_t<=r<=eta, f=f'=0 and "
            "|b_aniso-b_taper|=s*r"
        ),
        "taper_radius_over_L": taper_radius,
        "tapered_outer_radius_over_L": tapered_outer_radius,
        "tapered_half_height_over_L": tapered_half_height,
        "collar_L3_lower_bound_per_unit_strength": collar_l3_per_strength,
        "spectrum_collar_rows": spectrum_rows,
        "worst_t1_collar_L3_lower_bound_over_nu": (
            worst_strength * collar_l3_per_strength
        ),
        "worst_t1_dense_full_taper_mismatch_L3_over_nu": (
            full_worst_mismatch
        ),
        "tapered_sector_drift_only_budget_over_nu": tapered_drift_budget,
        "worst_collar_to_entire_drift_budget_ratio": (
            worst_strength * collar_l3_per_strength / tapered_drift_budget
        ),
        "maximum_anisotropic_strength_allowed_by_collar_budget": (
            maximum_strength_allowed_by_collar
        ),
        "maximum_t_parameter_allowed_by_collar_budget": (
            maximum_t_allowed_by_collar
        ),
        "tapered_reference_fails_exact_affine_consistency": bool(
            worst_strength * collar_l3_per_strength > tapered_drift_budget
        ),
        "compact_full_affine_reference": (
            "hold the complete fitted affine drift through r<2 and stop "
            "at the moving cylinder boundary"
        ),
        "compact_full_affine_exact_affine_remainder": 0.0,
        "compact_full_affine_half_height_over_L": compact_half_height,
        "compact_full_affine_t1_axial_eigenvalue": compact_axial_eigenvalue,
        "compact_full_affine_t1_visit_norm": compact_row[
            "visit_operator_norm"
        ],
        "compact_full_affine_t1_generation_criterion": compact_generation,
        "compact_full_affine_t1_condition_number": compact_condition,
        "compact_full_affine_sector_intercept_d": compact_intercept,
        "compact_full_affine_equal_share_alpha_beta": compact_equal_share,
        "compact_full_affine_equal_share_potential_L3_over_2_over_nu": (
            compact_equal_share / sharp_sobolev
        ),
        "compact_full_affine_equal_share_drift_L3_over_nu": (
            compact_equal_share / math.sqrt(sharp_sobolev)
        ),
        "compact_full_affine_restart_allowance": 1.0 - compact_good_norm,
        "compact_full_affine_split_paid_sector_intercept_d": (
            compact_split_intercept
        ),
        "compact_full_affine_split_paid_equal_alpha_beta": (
            compact_split_equal_share
        ),
        "compact_full_affine_split_paid_equal_potential_L3_over_2_over_nu": (
            compact_split_equal_share / sharp_sobolev
        ),
        "compact_full_affine_split_paid_equal_drift_L3_over_nu": (
            compact_split_equal_share / math.sqrt(sharp_sobolev)
        ),
        "compact_full_affine_sampled_transfer_closes": bool(
            compact_generation < 1.0
        ),
        "rigorous_compact_full_affine_transfer_certified": False,
        "architecture_consequence": (
            "use the compact full-affine baseline for fixed-label stopped "
            "visits; retain the divergence-free taper only as a separate "
            "model or when physical taper coherence is independently proved"
        ),
        "next_gate": (
            "fit the compact full-affine baseline from mollified Leray data "
            "and express its exact remainder as instantaneous Campanato "
            "oscillation plus temporal strain drift"
        ),
    }
    positive_checks = (
        result["tapered_reference_fails_exact_affine_consistency"],
        result["compact_full_affine_sampled_transfer_closes"],
        full_worst_mismatch
        >= worst_strength * collar_l3_per_strength - 1.0e-10,
        worst_strength * collar_l3_per_strength > 6.4,
        compact_generation < 0.17,
        compact_split_intercept > 0.04,
    )
    result["all_positive_affine_taper_no_go_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
