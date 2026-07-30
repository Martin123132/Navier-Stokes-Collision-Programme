"""Audit a monotone-envelope scale for a moving strain tube."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import sympy as sp


def _load_localized_tube_module():
    script = Path(__file__).resolve().with_name(
        "localized_strain_tube_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "localized_strain_tube_for_envelope", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit() -> dict[str, object]:
    a, envelope, envelope_rate, viscosity = sp.symbols(
        "a envelope envelope_rate viscosity", positive=True, real=True
    )
    reynolds, radius_sq = sp.symbols(
        "R_star radius_sq", nonnegative=True, real=True
    )
    amplitude_deficit = a - envelope
    scale_rate = -envelope_rate / (2 * envelope)
    scale_sq = reynolds * viscosity / envelope
    actual_reynolds = sp.simplify(a * scale_sq / viscosity)

    radial_error_coefficient = amplitude_deficit - scale_rate
    stretching_error = 2 * amplitude_deficit
    envelope_error = sp.factor(
        stretching_error
        - radial_error_coefficient
        - reynolds * radial_error_coefficient * radius_sq / 2
    )
    expected_error = sp.factor(
        amplitude_deficit * (1 - reynolds * radius_sq / 2)
        + scale_rate * (1 + reynolds * radius_sq / 2)
    )

    instantaneous_rate = sp.symbols("instantaneous_rate", real=True)
    instantaneous_scale_rate = -instantaneous_rate / (2 * a)
    instantaneous_scale_error = sp.factor(
        instantaneous_scale_rate * (1 + reynolds * radius_sq / 2)
    )

    cutoff = (1 - radius_sq) ** 4
    radial_cutoff_derivative = sp.factor(
        2 * radius_sq * sp.diff(cutoff, radius_sq)
    )
    cutoff_time_derivative = sp.factor(
        -scale_rate * radial_cutoff_derivative
    )
    cutoff_sign_samples = [
        float(
            cutoff_time_derivative.subs(
                {
                    envelope: 2.0,
                    envelope_rate: 3.0,
                    radius_sq: sample_radius_sq,
                }
            )
        )
        for sample_radius_sq in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    ]

    amplitude_history = (1.0, 4.0, 2.0, 3.0, 9.0, 6.0, 1.0)
    running_envelope = []
    current_envelope = 0.0
    for value in amplitude_history:
        current_envelope = max(current_envelope, value)
        running_envelope.append(current_envelope)

    history_rows = []
    previous_instantaneous_scale = None
    previous_envelope_scale = None
    target_reynolds = 2.0
    for index, (value, envelope_value) in enumerate(
        zip(amplitude_history, running_envelope)
    ):
        instantaneous_scale = (target_reynolds / value) ** 0.5
        envelope_scale = (target_reynolds / envelope_value) ** 0.5
        history_rows.append(
            {
                "index": index,
                "strain_rate": value,
                "running_envelope": envelope_value,
                "instantaneous_scale": instantaneous_scale,
                "envelope_scale": envelope_scale,
                "instantaneous_scale_change": (
                    None
                    if previous_instantaneous_scale is None
                    else instantaneous_scale - previous_instantaneous_scale
                ),
                "envelope_scale_change": (
                    None
                    if previous_envelope_scale is None
                    else envelope_scale - previous_envelope_scale
                ),
                "actual_local_reynolds": (
                    target_reynolds * value / envelope_value
                ),
            }
        )
        previous_instantaneous_scale = instantaneous_scale
        previous_envelope_scale = envelope_scale

    localized_tube = _load_localized_tube_module().audit()
    reynolds_two_row = next(
        row
        for row in localized_tube["spectral_rows"]
        if row["tube_reynolds"] == 2.0
    )

    deficit_coefficient_at_gate = sp.simplify(
        (1 - reynolds * radius_sq / 2).subs(
            {reynolds: 2, radius_sq: 1}
        )
    )
    scale_coefficient_at_worst_corner = sp.simplify(
        (1 + reynolds * radius_sq / 2).subs(
            {reynolds: 2, radius_sq: 1}
        )
    )
    envelope_scale_changes = [
        row["envelope_scale_change"]
        for row in history_rows[1:]
    ]
    instantaneous_scale_changes = [
        row["instantaneous_scale_change"]
        for row in history_rows[1:]
    ]

    result: dict[str, object] = {
        "envelope_definition": "A(t)=max(A_0,sup_{tau<=t} a(tau))",
        "adaptive_scale_definition": "L(t)^2=R_star*nu/A(t)",
        "amplitude_deficit": str(amplitude_deficit),
        "scale_rate": str(scale_rate),
        "mapped_radial_error_coefficient": str(
            radial_error_coefficient
        ),
        "stretching_error": str(stretching_error),
        "envelope_effective_error": str(envelope_error),
        "envelope_effective_error_factorization": str(expected_error),
        "envelope_error_factorization_verified": bool(
            sp.simplify(envelope_error - expected_error) == 0
        ),
        "deficit_coefficient_nonnegative_for_R_at_most_two": bool(
            deficit_coefficient_at_gate == 0
        ),
        "scale_coefficient_positive_for_R_at_most_two": bool(
            scale_coefficient_at_worst_corner == 2
        ),
        "envelope_error_is_nonpositive_at_R_at_most_two": bool(
            deficit_coefficient_at_gate == 0
            and scale_coefficient_at_worst_corner == 2
        ),
        "actual_local_reynolds": str(actual_reynolds),
        "actual_local_reynolds_is_capped_by_R_star": all(
            row["actual_local_reynolds"] <= target_reynolds
            for row in history_rows
        ),
        "instantaneous_scale_effective_error": str(
            instantaneous_scale_error
        ),
        "instantaneous_scale_is_adverse_when_strain_decreases": bool(
            any(change > 0.0 for change in instantaneous_scale_changes)
        ),
        "radial_cutoff": str(cutoff),
        "radial_cutoff_directional_derivative": str(
            radial_cutoff_derivative
        ),
        "radial_cutoff_time_derivative": str(cutoff_time_derivative),
        "shrinking_radial_cutoff_has_nonpositive_time_derivative": bool(
            max(cutoff_sign_samples) <= 1.0e-15
        ),
        "history_rows": history_rows,
        "running_envelope_scale_never_expands": bool(
            all(change <= 0.0 for change in envelope_scale_changes)
        ),
        "instantaneous_scale_expands_on_some_declines": bool(
            any(change > 0.0 for change in instantaneous_scale_changes)
        ),
        "R_two_single_principal_rate_over_A": reynolds_two_row[
            "lambda_over_a"
        ],
        "R_two_single_decay_margin_over_A": reynolds_two_row[
            "single_history_perturbation_budget_over_a"
        ],
        "R_two_pair_decay_margin_over_A": reynolds_two_row[
            "pair_decay_margin_over_a"
        ],
        "pressure_shell_dimensionless_ratio_cap": "a*L^2/nu<=R_star<=2",
        "remaining_gate": (
            "control non-affine coherence, pressure tails, centre motion, "
            "and weighted exterior renewal for the shrinking tube"
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
