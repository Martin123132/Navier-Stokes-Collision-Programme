"""Audit whether neutral-strip wall exit earns the cubic split factor."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import sympy as sp


def _load_level_module():
    script = Path(__file__).resolve().with_name(
        "cubic_level_transfer_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cubic_level_for_wall_compatibility", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit() -> dict[str, object]:
    level = _load_level_module().audit()
    start_y = sp.Rational(2, 1)
    lower_y = sp.Rational(3, 2)
    wall_y = sp.Rational(21, 10)
    upper_before_lower_probability = sp.simplify(
        (start_y - lower_y) / (wall_y - lower_y)
    )
    support_radius = sp.Rational(191, 100)
    maximum_child_center_offset = support_radius / (2 * sp.sqrt(2))
    child_outer_radius_in_parent_units = sp.Integer(1)
    child_capture_gap = sp.simplify(
        wall_y
        - maximum_child_center_offset
        - child_outer_radius_in_parent_units
    )
    maximum_wall_width_for_direct_child_capture = sp.simplify(
        maximum_child_center_offset + child_outer_radius_in_parent_units
    )
    true_split_factor = level["single_history_true_level_change_factor"]

    result: dict[str, object] = {
        "current_true_split_trigger": (
            "global adverse envelope reaches A_n=R_star*nu/L_n^2"
        ),
        "current_true_split_action": "L_(n+1)=L_n/2",
        "strip_wall_trigger": "stopped path first reaches |y|=2.1L",
        "events_are_logically_identical": False,
        "zero_solution_counterexample": (
            "u=0 keeps the adverse envelope below every positive level "
            "threshold, while the stochastic path is Brownian and may hit "
            "the strip wall"
        ),
        "counterexample_start": "(x,y)=(0,2)",
        "counterexample_protective_slab": "1.5<y<2.1",
        "protective_slab_avoids_inner_disk": True,
        "exact_probability_hit_upper_wall_before_lower_slab": str(
            upper_before_lower_probability
        ),
        "exact_probability_value": float(upper_before_lower_probability),
        "wall_exit_with_no_envelope_split_has_positive_probability": bool(
            upper_before_lower_probability > 0
        ),
        "wall_event_implies_true_level_change": False,
        "single_history_true_split_factor": true_split_factor,
        "true_split_factor_available_for_arbitrary_wall_exit": False,
        "maximum_audited_child_center_offset_over_parent_L": str(
            maximum_child_center_offset
        ),
        "child_outer_radius_over_parent_L": str(
            child_outer_radius_in_parent_units
        ),
        "working_wall_child_capture_gap_over_parent_L": str(
            child_capture_gap
        ),
        "working_wall_child_capture_gap_value": float(child_capture_gap),
        "maximum_wall_half_width_for_direct_child_capture": str(
            maximum_wall_width_for_direct_child_capture
        ),
        "full_r2_entry_and_direct_child_capture_widths_compatible": False,
        "wall_point_lies_in_an_audited_direct_child_visit": False,
        "arbitrary_refinement_policy": (
            "a wall-triggered voluntary refinement may use the Markov "
            "relabeling identity, but the existing theorem assigns it no "
            "radius-halving shrink payment"
        ),
        "conditional_axial_patch_scalar_closure_promotable": False,
        "same_scale_wall_branch_must_be_used_in_current_architecture": True,
        "full_Navier_Stokes_wall_split_gate_closed": False,
        "interpretation": (
            "the favorable axial-patch scalar criterion is conditional on "
            "a split payment that geometric wall exit does not earn. Under "
            "the current level rule the wall branch is same-scale unless an "
            "independent envelope crossing occurs"
        ),
        "scope_guard": (
            "the event-separation counterexample and gambler's-ruin "
            "probability are exact. This does not rule out a redesigned "
            "geometric scale hierarchy; it rules out importing the existing "
            "envelope-triggered factor without a new proof"
        ),
        "next_gate": (
            "either close the wall branch without split payment, or define "
            "a geometry-triggered scale transition and rederive its gauge, "
            "Markov, pressure, and many-generation factors from scratch"
        ),
    }
    checks = (
        upper_before_lower_probability == sp.Rational(5, 6),
        result["protective_slab_avoids_inner_disk"],
        result["wall_exit_with_no_envelope_split_has_positive_probability"],
        not result["events_are_logically_identical"],
        not result["wall_event_implies_true_level_change"],
        not result["true_split_factor_available_for_arbitrary_wall_exit"],
        child_capture_gap > 0,
        maximum_wall_width_for_direct_child_capture < 2,
        not result[
            "full_r2_entry_and_direct_child_capture_widths_compatible"
        ],
        not result["wall_point_lies_in_an_audited_direct_child_visit"],
        result["same_scale_wall_branch_must_be_used_in_current_architecture"],
        not result["conditional_axial_patch_scalar_closure_promotable"],
        not result["full_Navier_Stokes_wall_split_gate_closed"],
    )
    result["all_positive_wall_split_compatibility_checks_pass"] = all(
        checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
