"""Audit finite-height axial OU killing in the buffered visit model."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import hyp1f1, iv, kv


def _constant_killing_visit_gain(
    reynolds: float, buffer_ratio: float, axial_killing: float
) -> float:
    if axial_killing == 0.0:
        denominator = 1.0 - reynolds * math.log(buffer_ratio)
        return math.inf if denominator <= 0.0 else 1.0 / denominator

    kummer_parameter = 1.0 - axial_killing / (2.0 * reynolds)
    core_value = hyp1f1(kummer_parameter, 1.0, -reynolds / 2.0)
    core_slope = (
        -reynolds
        * kummer_parameter
        * hyp1f1(kummer_parameter + 1.0, 2.0, -reynolds / 2.0)
        / core_value
    )
    square_root_killing = math.sqrt(axial_killing)
    interface_matrix = np.array(
        [
            [
                iv(0, square_root_killing),
                kv(0, square_root_killing),
            ],
            [
                square_root_killing * iv(1, square_root_killing),
                -square_root_killing * kv(1, square_root_killing),
            ],
        ]
    )
    coefficient_i, coefficient_k = np.linalg.solve(
        interface_matrix, np.array([1.0, core_slope])
    )
    outer_transfer = (
        coefficient_i * iv(0, square_root_killing * buffer_ratio)
        + coefficient_k * kv(0, square_root_killing * buffer_ratio)
    )
    return math.inf if outer_transfer <= 0.0 else 1.0 / outer_transfer


def _axial_ou_principal_killing(
    reynolds: float, half_height_ratio: float
) -> float:
    boundary_argument = reynolds * half_height_ratio**2

    def boundary_value(axial_killing: float) -> float:
        return float(
            hyp1f1(
                -axial_killing / (4.0 * reynolds),
                0.5,
                boundary_argument,
            )
        )

    brownian_upper_scale = math.pi**2 / (4.0 * half_height_ratio**2)
    upper = max(1.0, brownian_upper_scale)
    while boundary_value(upper) > 0.0:
        upper *= 1.5
        if upper > 1.0e5:
            raise RuntimeError("failed to bracket axial OU eigenvalue")
    return float(brentq(boundary_value, 0.0, upper, xtol=1.0e-13))


def _generation_criterion(
    reynolds: float,
    buffer_ratio: float,
    beta: float,
    axial_killing: float,
) -> float:
    one_history_gain = _constant_killing_visit_gain(
        reynolds, buffer_ratio, axial_killing
    )
    pair_return = buffer_ratio ** (-2.0 * beta)
    true_split = math.exp(reynolds * 3.0 / 24.0) / 4.0
    return one_history_gain**2 * (true_split + pair_return)


def audit() -> dict[str, object]:
    buffer_ratio = 2.0
    beta = 1.0
    reynolds_values = (0.25, 0.5, 1.0, 2.0)
    threshold_rows = []
    for reynolds in reynolds_values:
        zero_killing_criterion = _generation_criterion(
            reynolds, buffer_ratio, beta, 0.0
        )
        if zero_killing_criterion < 1.0:
            required_killing = 0.0
            maximum_half_height = math.inf
            pure_brownian_half_height = math.inf
            ou_boundary_residual = 0.0
        else:

            def killing_equation(axial_killing: float) -> float:
                return (
                    _generation_criterion(
                        reynolds, buffer_ratio, beta, axial_killing
                    )
                    - 1.0
                )

            upper_killing = 1.0
            while killing_equation(upper_killing) > 0.0:
                upper_killing *= 2.0
            required_killing = float(
                brentq(killing_equation, 0.0, upper_killing)
            )
            pure_brownian_half_height = math.pi / (
                2.0 * math.sqrt(required_killing)
            )

            def height_equation(half_height: float) -> float:
                return (
                    _axial_ou_principal_killing(reynolds, half_height)
                    - required_killing
                )

            maximum_half_height = float(
                brentq(height_equation, 0.2, 5.0, xtol=1.0e-11)
            )
            ou_killing = _axial_ou_principal_killing(
                reynolds, maximum_half_height
            )
            ou_boundary_residual = float(
                hyp1f1(
                    -ou_killing / (4.0 * reynolds),
                    0.5,
                    reynolds * maximum_half_height**2,
                )
            )

        threshold_rows.append(
            {
                "R_star": reynolds,
                "zero_axial_killing_generation_criterion": (
                    zero_killing_criterion
                ),
                "required_dimensionless_axial_killing": required_killing,
                "optimistic_brownian_maximum_half_height_over_L": (
                    pure_brownian_half_height
                ),
                "axial_OU_maximum_half_height_over_L": maximum_half_height,
                "axial_OU_maximum_full_height_over_L": (
                    2.0 * maximum_half_height
                ),
                "axial_OU_boundary_residual": ou_boundary_residual,
                "threshold_generation_criterion": (
                    _generation_criterion(
                        reynolds,
                        buffer_ratio,
                        beta,
                        required_killing,
                    )
                ),
            }
        )

    monotonic_rows = []
    for reynolds in (0.5, 1.0):
        gains = [
            _constant_killing_visit_gain(reynolds, 2.0, killing)
            for killing in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
        ]
        monotonic_rows.append(
            {
                "R_star": reynolds,
                "axial_killing_values": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
                "one_history_visit_gains": gains,
                "gain_strictly_decreases_with_axial_killing": all(
                    later < earlier
                    for earlier, later in zip(gains[:-1], gains[1:])
                ),
            }
        )

    ou_comparison_rows = []
    for reynolds in (0.25, 0.5, 1.0, 2.0):
        for half_height in (1.0, 2.0, 3.0):
            ou_killing = _axial_ou_principal_killing(
                reynolds, half_height
            )
            brownian_killing = math.pi**2 / (4.0 * half_height**2)
            ou_comparison_rows.append(
                {
                    "R_star": reynolds,
                    "half_height_over_L": half_height,
                    "axial_OU_killing": ou_killing,
                    "brownian_axial_killing": brownian_killing,
                    "inward_OU_drift_reduces_axial_escape": bool(
                        ou_killing < brownian_killing
                    ),
                }
            )

    R_one_row = next(
        row for row in threshold_rows if row["R_star"] == 1.0
    )
    result: dict[str, object] = {
        "constant_killing_core_parameter": (
            "M(1-zeta/(2R_star),1,-R_star*rho^2/2)"
        ),
        "constant_killing_shell_basis": (
            "I_0(sqrt(zeta)*rho), K_0(sqrt(zeta)*rho)"
        ),
        "axial_OU_generator": (
            "partial_yy-2*R_star*y*partial_y on (-h,h)"
        ),
        "axial_OU_boundary_equation": (
            "M(-zeta/(4R_star),1/2,R_star*h^2)=0"
        ),
        "threshold_rows": threshold_rows,
        "all_thresholds_reproduce_generation_boundary": all(
            abs(row["threshold_generation_criterion"] - 1.0) < 1.0e-9
            for row in threshold_rows
            if row["required_dimensionless_axial_killing"] > 0.0
        ),
        "all_axial_eigenvalue_residuals_small": all(
            abs(row["axial_OU_boundary_residual"]) < 1.0e-9
            for row in threshold_rows
        ),
        "monotonic_gain_rows": monotonic_rows,
        "all_visit_gains_decrease_with_axial_killing": all(
            row["gain_strictly_decreases_with_axial_killing"]
            for row in monotonic_rows
        ),
        "OU_vs_brownian_rows": ou_comparison_rows,
        "inward_axial_drift_always_reduces_tested_escape": all(
            row["inward_OU_drift_reduces_axial_escape"]
            for row in ou_comparison_rows
        ),
        "R_one_required_axial_killing": R_one_row[
            "required_dimensionless_axial_killing"
        ],
        "R_one_maximum_half_height_over_L": R_one_row[
            "axial_OU_maximum_half_height_over_L"
        ],
        "R_one_requires_genuinely_finite_axial_core": bool(
            R_one_row["axial_OU_maximum_half_height_over_L"] < 2.0
        ),
        "surrogate_scope": (
            "constant killing is exact for one separated axial mode; a full "
            "finite-cylinder boundary operator requires the complete axial "
            "mode expansion"
        ),
        "next_cylinder_gate": (
            "sum the Dirichlet axial modes with inward OU drift and verify "
            "an operator bound, not only the principal-mode surrogate"
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
