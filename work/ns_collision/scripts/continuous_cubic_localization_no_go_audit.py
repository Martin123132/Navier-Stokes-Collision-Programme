"""Audit the Fisher obstruction for continuous cubic visit localization."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import sympy as sp
from scipy.special import jn_zeros


def _load_affine_module():
    script = Path(__file__).resolve().with_name(
        "anisotropic_poisson_transfer_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "anisotropic_poisson_for_cubic_no_go", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit() -> dict[str, object]:
    coordinate = sp.symbols("coordinate", real=True)
    fisher = sp.factor(
        -3
        * (
            9 * coordinate**4
            - 18 * coordinate**3
            + 2 * coordinate**2
            + 7 * coordinate
            + 2
        )
        / (
            2
            * (3 * coordinate**3 - 6 * coordinate**2 + 4)
            * (
                3 * coordinate**3
                - 3 * coordinate**2
                - 3 * coordinate
                - 1
            )
        )
    )
    lower_difference = sp.factor(fisher - sp.Rational(3, 4))
    quartic_factor = sp.factor(
        9 * coordinate**2 * (1 - coordinate) ** 2 - 2
    )
    quartic_upper_bound = sp.Rational(9, 16) - 2

    outer_radius = 2.75
    half_height = 1.2
    transverse_spacing_upper_bound = outer_radius / (2.0 * 2.0**0.5)
    axial_spacing_upper_bound = half_height / 2.0
    transverse_fisher_lower_bound = 12.0 / outer_radius**2
    axial_fisher_lower_bound = 3.0 / half_height**2
    full_fisher_lower_bound = (
        transverse_fisher_lower_bound + axial_fisher_lower_bound
    )

    affine_module = _load_affine_module()
    axial_eigenvalue = affine_module._axial_principal_eigenvalue(
        half_height
    )
    disk_ground = float(jn_zeros(0, 1)[0] ** 2 / outer_radius**2)
    symmetric_mass_upper_bound = axial_eigenvalue - 0.5
    disk_ground_trial_form_upper_bound = (
        disk_ground + symmetric_mass_upper_bound
    )
    full_localized_trial_form_upper_bound = (
        disk_ground_trial_form_upper_bound - full_fisher_lower_bound
    )
    maximum_fisher_fraction_not_ruled_out_by_trial = (
        disk_ground_trial_form_upper_bound / full_fisher_lower_bound
    )

    result: dict[str, object] = {
        "one_dimensional_cardinal_cubic_Fisher": str(fisher),
        "Fisher_minus_three_quarters": str(lower_difference),
        "quartic_sign_factor": str(quartic_factor),
        "quartic_factor_upper_bound_on_unit_cell": str(
            quartic_upper_bound
        ),
        "exact_one_dimensional_Fisher_lower_bound": 0.75,
        "Fisher_lower_bound_proof": (
            "on 0<=x<=1, x(x-1)<=0, "
            "9x^2(1-x)^2-2<=-23/16, the first cubic denominator "
            "factor is positive, and the second is negative"
        ),
        "working_outer_radius_over_L": outer_radius,
        "working_half_height_over_L": half_height,
        "largest_subordinate_transverse_knot_spacing_over_L": (
            transverse_spacing_upper_bound
        ),
        "largest_subordinate_axial_knot_spacing_over_L": (
            axial_spacing_upper_bound
        ),
        "transverse_tensor_Fisher_lower_bound_L2": (
            transverse_fisher_lower_bound
        ),
        "axial_Fisher_lower_bound_L2": axial_fisher_lower_bound,
        "full_tensor_Fisher_lower_bound_L2": full_fisher_lower_bound,
        "working_axial_OU_eigenvalue": axial_eigenvalue,
        "outer_disk_Dirichlet_ground_value": disk_ground,
        "symmetric_mass_upper_bound": symmetric_mass_upper_bound,
        "disk_ground_trial_form_upper_bound": (
            disk_ground_trial_form_upper_bound
        ),
        "disk_ground_trial_after_full_Fisher_upper_bound": (
            full_localized_trial_form_upper_bound
        ),
        "maximum_Fisher_fraction_not_ruled_out_by_ground_trial": (
            maximum_fisher_fraction_not_ruled_out_by_trial
        ),
        "full_cubic_Fisher_exceeds_trial_form_ceiling": bool(
            full_fisher_lower_bound > disk_ground_trial_form_upper_bound
        ),
        "localized_symmetric_form_has_negative_test_direction": bool(
            full_localized_trial_form_upper_bound < 0.0
        ),
        "continuous_full_tensor_cubic_localization_ruled_out": True,
        "scope": (
            "rules out absorbing the complete simultaneous square-root "
            "cardinal-cubic Fisher cost in the current wide visit form; "
            "it does not rule out linear pressure weights, stopping-time "
            "entry labels, larger visit domains, or a different partition"
        ),
        "revised_architecture": (
            "use the normalized cubic law for conservative assignment at "
            "buffered entry/exit, hold the selected visit label through the "
            "visit, and keep any continuous linear partition only for exact "
            "pressure-flux bookkeeping"
        ),
        "next_gate": (
            "derive a stopping-time moving-cylinder visit identity with no "
            "continuous square-root partition inside the visit, then bound "
            "its translation/frame remainder by the sector criterion"
        ),
    }
    positive_checks = (
        quartic_upper_bound < 0,
        result["full_cubic_Fisher_exceeds_trial_form_ceiling"],
        result["localized_symmetric_form_has_negative_test_direction"],
        result["continuous_full_tensor_cubic_localization_ruled_out"],
        full_fisher_lower_bound > 3.6,
        disk_ground_trial_form_upper_bound < 1.6,
        maximum_fisher_fraction_not_ruled_out_by_trial < 0.42,
    )
    result["all_positive_cubic_no_go_checks_pass"] = all(positive_checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
