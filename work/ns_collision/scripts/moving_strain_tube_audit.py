"""Audit moving-tube gauge terms and an L2 robustness budget."""

from __future__ import annotations

import importlib.util
import json
from math import sqrt
from pathlib import Path

import sympy as sp


def _load_localized_tube_module():
    script = Path(__file__).resolve().with_name("localized_strain_tube_audit.py")
    spec = importlib.util.spec_from_file_location(
        "localized_strain_tube_audit_for_moving_tube", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit() -> dict[str, object]:
    x, y = sp.symbols("x y", real=True)
    tube_reynolds, reynolds_rate = sp.symbols(
        "R R_rate", real=True
    )
    error_x, error_y = sp.symbols("error_x error_y", real=True)
    divergence_error, stretching_error = sp.symbols(
        "divergence_error stretching_error", real=True
    )
    radius_sq = x**2 + y**2

    moving_effective_error = sp.factor(
        stretching_error
        - divergence_error / 2
        - tube_reynolds * (x * error_x + y * error_y) / 2
        + reynolds_rate * radius_sq / 4
    )

    omega, scale_rate = sp.symbols("omega scale_rate", real=True)
    rotation_error = sp.simplify(
        moving_effective_error.subs(
            {
                error_x: omega * y,
                error_y: -omega * x,
                divergence_error: 0,
                stretching_error: 0,
                reynolds_rate: 0,
            }
        )
    )
    scale_error = sp.factor(
        moving_effective_error.subs(
            {
                error_x: -scale_rate * x,
                error_y: -scale_rate * y,
                divergence_error: -2 * scale_rate,
                stretching_error: 0,
            }
        )
    )
    expected_scale_error = sp.factor(
        scale_rate
        + tube_reynolds * scale_rate * radius_sq / 2
        + reynolds_rate * radius_sq / 4
    )

    kappa, reference_rate = sp.symbols(
        "kappa reference_rate", positive=True, real=True
    )
    form_ratio = sp.sqrt(kappa + reference_rate) / kappa
    form_ratio_derivative = sp.factor(sp.diff(form_ratio, kappa))
    expected_derivative = sp.factor(
        -(kappa + 2 * reference_rate)
        / (2 * kappa**2 * sp.sqrt(kappa + reference_rate))
    )

    localized_module = _load_localized_tube_module()
    localized_rows = localized_module.audit()["spectral_rows"]
    rows = []
    for row in localized_rows:
        reynolds = float(row["tube_reynolds"])
        margin_over_a = float(row["lambda_over_a"]) - 2.0
        l2_budget_over_a = margin_over_a / sqrt(
            2.0 * reynolds * (margin_over_a + 1.0)
        )
        rows.append(
            {
                "tube_reynolds": reynolds,
                "spectral_margin_over_a": margin_over_a,
                "pointwise_budget_over_a": margin_over_a,
                "l2_budget_over_a_on_unit_disk": l2_budget_over_a,
            }
        )

    result: dict[str, object] = {
        "moving_coordinate_map": "x=center+L*O*y",
        "mapped_diffusivity": "k=nu/L^2",
        "mapped_drift_error": (
            "e=L^(-1)*O^T*(b-center_rate)-(L_rate/L)*y-"
            "Omega*y-a*y"
        ),
        "tube_reynolds_definition": "R=a/k=a*L^2/nu",
        "gauge": "exp(-R*|y|^2/4)",
        "moving_effective_error": str(moving_effective_error),
        "moving_effective_error_formula": (
            "q=delta_s-div(e)/2-(R/2)*y.e+(R_rate/4)*|y|^2"
        ),
        "pure_rotation_energy_error": str(rotation_error),
        "pure_rotation_is_invisible_to_radial_gauge": bool(
            rotation_error == 0
        ),
        "scale_error": str(scale_error),
        "scale_error_verified": bool(
            sp.simplify(scale_error - expected_scale_error) == 0
        ),
        "time_dependent_gauge_term": "R_rate*|y|^2/4",
        "ladyzhenskaya_inequality": (
            "||psi||_4^2<=sqrt(2)*||psi||_2*||grad psi||_2"
        ),
        "l2_relative_form_factor": (
            "alpha=sqrt(2)*Q*sqrt(m+a)/(sqrt(k)*m)"
        ),
        "l2_robustness_condition": (
            "Q=||q_+||_2<m*sqrt(k)/(sqrt(2)*sqrt(m+a))"
        ),
        "form_ratio_derivative": str(form_ratio_derivative),
        "form_ratio_is_decreasing": bool(
            sp.simplify(form_ratio_derivative - expected_derivative) == 0
        ),
        "spectral_budget_rows": rows,
        "all_l2_budgets_are_positive": all(
            row["l2_budget_over_a_on_unit_disk"] > 0.0 for row in rows
        ),
        "high_reynolds_l2_budget_collapses": bool(
            rows[-1]["l2_budget_over_a_on_unit_disk"]
            < rows[-2]["l2_budget_over_a_on_unit_disk"]
            < rows[-3]["l2_budget_over_a_on_unit_disk"]
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
