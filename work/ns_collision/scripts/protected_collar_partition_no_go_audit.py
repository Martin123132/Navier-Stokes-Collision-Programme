"""Audit the cubic IMS cost of the protected collar supports."""

from __future__ import annotations

import json


BASELINE_FORM_FLOOR = 4.832287335665
CARDINAL_CUBIC_FISHER_UPPER = 157.0 / 200.0


def _row(collar_distance: float) -> dict[str, float | bool]:
    support_radius = 1.0 - collar_distance
    support_half_height = 0.75 - collar_distance
    transverse_knot_spacing = support_radius / (2.0 * 2.0**0.5)
    axial_knot_spacing = support_half_height / 2.0
    transverse_ims = (
        2.0
        * CARDINAL_CUBIC_FISHER_UPPER
        / transverse_knot_spacing**2
    )
    axial_ims = (
        CARDINAL_CUBIC_FISHER_UPPER / axial_knot_spacing**2
    )
    full_ims = transverse_ims + axial_ims
    return {
        "collar_distance": collar_distance,
        "support_radius": support_radius,
        "support_half_height": support_half_height,
        "transverse_knot_spacing": transverse_knot_spacing,
        "axial_knot_spacing": axial_knot_spacing,
        "transverse_cubic_IMS_cost": transverse_ims,
        "axial_cubic_IMS_cost": axial_ims,
        "full_tensor_cubic_IMS_cost": full_ims,
        "transverse_cost_exceeds_complete_form_floor": bool(
            transverse_ims > BASELINE_FORM_FLOOR
        ),
        "full_cost_exceeds_complete_form_floor": bool(
            full_ims > BASELINE_FORM_FLOOR
        ),
        "full_cost_to_form_floor_ratio": (
            full_ims / BASELINE_FORM_FLOOR
        ),
    }


def audit() -> dict[str, object]:
    rows = [_row(distance) for distance in (0.0, 0.10, 0.20, 0.30, 0.40)]
    result: dict[str, object] = {
        "protected_support": "E_d={r<=1-d, |z|<=0.75-d}",
        "cardinal_cubic_Fisher_bound": "I_3<157/200",
        "transverse_spacing_constraint": (
            "2*sqrt(2)*h_perp=1-d"
        ),
        "axial_spacing_constraint": "2*h_z=0.75-d",
        "transverse_IMS_formula": "314/[25*(1-d)^2]",
        "axial_IMS_formula": "157/[50*(0.75-d)^2]",
        "baseline_compact_form_floor": BASELINE_FORM_FLOOR,
        "rows": rows,
        "continuous_transverse_cubic_localization_viable": False,
        "continuous_full_tensor_cubic_localization_viable": False,
        "every_transverse_row_exceeds_form_floor": all(
            row["transverse_cost_exceeds_complete_form_floor"]
            for row in rows
        ),
        "every_full_tensor_row_exceeds_form_floor": all(
            row["full_cost_exceeds_complete_form_floor"] for row in rows
        ),
        "interpretation": (
            "the protected collar cannot be manufactured by keeping the "
            "current cardinal-cubic square-root partition active during "
            "the visit; localization must occur at stopping or perturbation "
            "interaction times, or use a genuinely different mechanism"
        ),
        "scope_guard": (
            "this is a no-go for the audited cardinal-cubic continuous "
            "partition, not for every conceivable localization scheme"
        ),
    }
    positive_checks = (
        not result["continuous_transverse_cubic_localization_viable"],
        not result["continuous_full_tensor_cubic_localization_viable"],
        result["every_transverse_row_exceeds_form_floor"],
        result["every_full_tensor_row_exceeds_form_floor"],
        rows[2]["full_cost_to_form_floor_ratio"] > 6.2,
    )
    result["all_positive_protected_collar_no_go_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
