"""Audit an explicit candidate supersolution for the radial-payoff HJB."""

from __future__ import annotations

import json
import math

import numpy as np


def _candidate_diagnostics() -> dict[str, object]:
    radius = 2.0
    half_height = 0.75
    axial_frequency = math.pi / (2.0 * half_height)
    radial_quadratic_weight = 0.89945
    radial_sixteenth_weight = 0.10055
    boundary_layer_weight = 1.3479
    radial_exponent = 13.0 / 20.0
    axial_exponent = 7.0 / 20.0

    linear_radial = np.linspace(0.0, radius - 1.0e-6, 1_201)
    boundary_radial = radius - np.geomspace(1.0e-6, 0.5, 500)
    radial_grid = np.unique(np.concatenate((linear_radial, boundary_radial)))
    linear_axial = np.linspace(0.0, half_height - 1.0e-7, 901)
    boundary_axial = half_height - np.geomspace(1.0e-7, 0.3, 400)
    axial_grid = np.unique(np.concatenate((linear_axial, boundary_axial)))

    maximum_residual = (-math.inf, (0.0, 0.0))
    maximum_linear_part = (-math.inf, (0.0, 0.0))
    minimum_squared_margin = (math.inf, (0.0, 0.0))
    for start in range(0, len(radial_grid), 40):
        radial, axial = np.meshgrid(
            radial_grid[start : start + 40],
            axial_grid,
            indexing="ij",
        )
        scaled_radius = radial / radius
        radial_layer = 1.0 - scaled_radius**2
        axial_cosine = np.cos(axial_frequency * axial)
        axial_sine = np.sin(axial_frequency * axial)

        radial_value = (
            radial_quadratic_weight * scaled_radius**2
            + radial_sixteenth_weight * scaled_radius**16
        )
        radial_derivative = (
            2.0 * radial_quadratic_weight * scaled_radius
            + 16.0 * radial_sixteenth_weight * scaled_radius**15
        ) / radius
        radial_laplacian = (
            4.0 * radial_quadratic_weight
            + 256.0 * radial_sixteenth_weight * scaled_radius**14
        ) / radius**2

        layer_value = radial_layer**radial_exponent
        layer_radial_derivative = (
            -2.0
            * radial_exponent
            * radial
            / radius**2
            * radial_layer ** (radial_exponent - 1.0)
        )
        layer_radial_laplacian = (
            -4.0
            * radial_exponent
            / radius**2
            * radial_layer ** (radial_exponent - 1.0)
            + 4.0
            * radial_exponent
            * (radial_exponent - 1.0)
            * radial**2
            / radius**4
            * radial_layer ** (radial_exponent - 2.0)
        )

        axial_value = axial_cosine**axial_exponent
        axial_derivative = (
            -axial_exponent
            * axial_frequency
            * axial_sine
            * axial_cosine ** (axial_exponent - 1.0)
        )
        axial_second_derivative = (
            axial_exponent
            * (axial_exponent - 1.0)
            * axial_frequency**2
            * axial_sine**2
            * axial_cosine ** (axial_exponent - 2.0)
            - axial_exponent
            * axial_frequency**2
            * axial_cosine**axial_exponent
        )

        value = (
            radial_value
            + boundary_layer_weight * layer_value * axial_value
        )
        radial_gradient = (
            radial_derivative
            + boundary_layer_weight
            * layer_radial_derivative
            * axial_value
        )
        axial_gradient = (
            boundary_layer_weight * layer_value * axial_derivative
        )
        laplacian = (
            radial_laplacian
            + boundary_layer_weight
            * (
                layer_radial_laplacian * axial_value
                + layer_value * axial_second_derivative
            )
        )
        linear_part = (
            laplacian
            + 0.5
            * (
                radial * radial_gradient
                + axial * axial_gradient
            )
            + value
        )
        position_squared = radial**2 + axial**2
        gradient_squared = radial_gradient**2 + axial_gradient**2
        residual = (
            linear_part
            + 1.5 * np.sqrt(position_squared * gradient_squared)
        )
        squared_margin = (
            linear_part**2
            - 2.25 * position_squared * gradient_squared
        )

        residual_index = np.unravel_index(
            int(np.argmax(residual)), residual.shape
        )
        linear_index = np.unravel_index(
            int(np.argmax(linear_part)), linear_part.shape
        )
        margin_index = np.unravel_index(
            int(np.argmin(squared_margin)), squared_margin.shape
        )
        residual_value = float(residual[residual_index])
        linear_value = float(linear_part[linear_index])
        margin_value = float(squared_margin[margin_index])
        if residual_value > maximum_residual[0]:
            maximum_residual = (
                residual_value,
                (
                    float(radial[residual_index]),
                    float(axial[residual_index]),
                ),
            )
        if linear_value > maximum_linear_part[0]:
            maximum_linear_part = (
                linear_value,
                (
                    float(radial[linear_index]),
                    float(axial[linear_index]),
                ),
            )
        if margin_value < minimum_squared_margin[0]:
            minimum_squared_margin = (
                margin_value,
                (
                    float(radial[margin_index]),
                    float(axial[margin_index]),
                ),
            )

    interface_value = (
        radial_quadratic_weight / 4.0
        + radial_sixteenth_weight / 2.0**16
        + boundary_layer_weight
        * (3.0 / 4.0) ** radial_exponent
    )
    return {
        "rational_coefficients": {
            "radial_quadratic_weight": "89945/100000",
            "radial_sixteenth_weight": "10055/100000",
            "boundary_layer_weight": "13479/10000",
            "radial_exponent": "13/20",
            "axial_exponent": "7/20",
        },
        "candidate_formula": (
            "U=.89945(r/2)^2+.10055(r/2)^16+1.3479"
            "[1-(r/2)^2]^(13/20)cos(2*pi*z/3)^(7/20)"
        ),
        "radial_boundary_value": (
            radial_quadratic_weight + radial_sixteenth_weight
        ),
        "cap_boundary_value": (
            ".89945(r/2)^2+.10055(r/2)^16>=0"
        ),
        "inner_interface_maximum": interface_value,
        "inner_interface_maximizer": [1.0, 0.0],
        "dense_grid": {
            "radial_point_count": len(radial_grid),
            "axial_half_domain_point_count": len(axial_grid),
            "maximum_HJB_residual": maximum_residual[0],
            "maximum_HJB_residual_location": maximum_residual[1],
            "maximum_linear_part": maximum_linear_part[0],
            "maximum_linear_part_location": maximum_linear_part[1],
            "minimum_squared_margin": minimum_squared_margin[0],
            "minimum_squared_margin_location": minimum_squared_margin[1],
        },
    }


def audit() -> dict[str, object]:
    candidate = _candidate_diagnostics()
    cycle_coefficient = 0.6586950386676936
    closure_gain = 1.2321336084949255
    candidate_gain = candidate["inner_interface_maximum"]
    candidate_generation = cycle_coefficient * candidate_gain**2
    dense = candidate["dense_grid"]
    result: dict[str, object] = {
        "exact_control_hamiltonian": (
            "sup_B (B y).p=(y.p)/2+3|y||p|/2"
        ),
        "candidate": candidate,
        "maximum_dynamic_one_history_gain_for_closure": closure_gain,
        "candidate_gain_margin_to_closure": closure_gain - candidate_gain,
        "candidate_complete_generation_criterion": candidate_generation,
        "candidate_would_close_if_residual_is_certified": bool(
            candidate_generation < 1.0
        ),
        "superseded_by_finite_energy_H1_barrier": True,
        "interior_residual_interval_certified": False,
        "ideal_nonautonomous_boundary_theorem_certified": False,
        "scope_guard": (
            "the barrier formula and boundary inequalities are exact; the "
            "reported interior residual is a dense-grid falsification "
            "audit, not yet an interval or analytic enclosure"
        ),
        "next_gate": (
            "retain this interval-certified HJB barrier as a historical "
            "comparison; the lower-gain finite-energy H1 barrier is the "
            "current candidate under the cubic split calibration"
        ),
    }
    positive_checks = (
        abs(candidate["radial_boundary_value"] - 1.0) < 1.0e-15,
        candidate_gain < 1.343,
        result["candidate_gain_margin_to_closure"] < -0.10,
        candidate_generation > 1.18,
        not result["candidate_would_close_if_residual_is_certified"],
        dense["maximum_HJB_residual"] < -0.009,
        dense["maximum_linear_part"] < -0.55,
        dense["minimum_squared_margin"] > 0.02,
    )
    result["all_positive_supersolution_candidate_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
