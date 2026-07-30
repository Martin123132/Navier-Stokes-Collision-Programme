"""Pilot a finite-energy supersolution for the radial-payoff HJB."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import comb


RADIUS = 2.0
HALF_HEIGHT = 0.75
AXIAL_FREQUENCY = math.pi / (2.0 * HALF_HEIGHT)
RADIAL_QUADRATIC_WEIGHT = 197.0 / 200.0
RADIAL_POWER_WEIGHT = 3.0 / 200.0
RADIAL_POWER = 32
LAYER_WEIGHT = 11.0 / 10.0
RADIAL_EXPONENT = 7.0 / 10.0
AXIAL_EXPONENT = 69.0 / 100.0
AXIAL_SHAPE = 3.0 / 10.0
POTENTIAL = 1.005


def _candidate_fields(
    radial: np.ndarray, axial: np.ndarray
) -> tuple[np.ndarray, ...]:
    scaled = radial / RADIUS
    radial_layer = 1.0 - scaled**2
    cosine = np.cos(AXIAL_FREQUENCY * axial)
    sine = np.sin(AXIAL_FREQUENCY * axial)

    radial_value = (
        RADIAL_QUADRATIC_WEIGHT * scaled**2
        + RADIAL_POWER_WEIGHT * scaled**RADIAL_POWER
    )
    radial_first = (
        2.0 * RADIAL_QUADRATIC_WEIGHT * scaled
        + RADIAL_POWER
        * RADIAL_POWER_WEIGHT
        * scaled ** (RADIAL_POWER - 1)
    ) / RADIUS
    radial_laplacian = (
        4.0 * RADIAL_QUADRATIC_WEIGHT
        + RADIAL_POWER**2
        * RADIAL_POWER_WEIGHT
        * scaled ** (RADIAL_POWER - 2)
    ) / RADIUS**2

    layer_value = radial_layer**RADIAL_EXPONENT
    layer_first = (
        -2.0
        * RADIAL_EXPONENT
        * radial
        / RADIUS**2
        * radial_layer ** (RADIAL_EXPONENT - 1.0)
    )
    layer_laplacian = (
        -4.0
        * RADIAL_EXPONENT
        / RADIUS**2
        * radial_layer ** (RADIAL_EXPONENT - 1.0)
        + 4.0
        * RADIAL_EXPONENT
        * (RADIAL_EXPONENT - 1.0)
        * radial**2
        / RADIUS**4
        * radial_layer ** (RADIAL_EXPONENT - 2.0)
    )

    axial_value = (
        (1.0 + AXIAL_SHAPE) * cosine**AXIAL_EXPONENT
        - AXIAL_SHAPE * cosine ** (AXIAL_EXPONENT + 1.0)
    )
    axial_first = (
        -(1.0 + AXIAL_SHAPE)
        * AXIAL_EXPONENT
        * AXIAL_FREQUENCY
        * sine
        * cosine ** (AXIAL_EXPONENT - 1.0)
        + AXIAL_SHAPE
        * (AXIAL_EXPONENT + 1.0)
        * AXIAL_FREQUENCY
        * sine
        * cosine**AXIAL_EXPONENT
    )
    axial_second = (
        (1.0 + AXIAL_SHAPE)
        * (
            AXIAL_EXPONENT
            * (AXIAL_EXPONENT - 1.0)
            * AXIAL_FREQUENCY**2
            * sine**2
            * cosine ** (AXIAL_EXPONENT - 2.0)
            - AXIAL_EXPONENT
            * AXIAL_FREQUENCY**2
            * cosine**AXIAL_EXPONENT
        )
        - AXIAL_SHAPE
        * (
            (AXIAL_EXPONENT + 1.0)
            * AXIAL_EXPONENT
            * AXIAL_FREQUENCY**2
            * sine**2
            * cosine ** (AXIAL_EXPONENT - 1.0)
            - (AXIAL_EXPONENT + 1.0)
            * AXIAL_FREQUENCY**2
            * cosine ** (AXIAL_EXPONENT + 1.0)
        )
    )

    value = radial_value + LAYER_WEIGHT * layer_value * axial_value
    radial_gradient = (
        radial_first + LAYER_WEIGHT * layer_first * axial_value
    )
    axial_gradient = LAYER_WEIGHT * layer_value * axial_first
    laplacian = radial_laplacian + LAYER_WEIGHT * (
        layer_laplacian * axial_value
        + layer_value * axial_second
    )
    linear_part = (
        laplacian
        + 0.5
        * (radial * radial_gradient + axial * axial_gradient)
        + POTENTIAL * value
    )
    gradient_squared = radial_gradient**2 + axial_gradient**2
    residual = linear_part + 1.5 * np.sqrt(
        (radial**2 + axial**2) * gradient_squared
    )
    squared_margin = (
        linear_part**2
        - 2.25 * (radial**2 + axial**2) * gradient_squared
    )
    return (
        value,
        radial_gradient,
        axial_gradient,
        linear_part,
        residual,
        squared_margin,
    )


def _dense_grid_diagnostics() -> dict[str, object]:
    radial_grid = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, RADIUS - 1.0e-6, 1_301),
                RADIUS - np.geomspace(1.0e-8, 0.5, 700),
            )
        )
    )
    axial_grid = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, HALF_HEIGHT - 1.0e-7, 1_001),
                HALF_HEIGHT - np.geomspace(1.0e-9, 0.3, 550),
            )
        )
    )
    maximum_residual = (-math.inf, (0.0, 0.0))
    maximum_linear = (-math.inf, (0.0, 0.0))
    minimum_margin = (math.inf, (0.0, 0.0))
    for start in range(0, len(radial_grid), 24):
        radial, axial = np.meshgrid(
            radial_grid[start : start + 24],
            axial_grid,
            indexing="ij",
        )
        fields = _candidate_fields(radial, axial)
        linear_part, residual, squared_margin = fields[3:]
        residual_index = np.unravel_index(
            int(np.argmax(residual)), residual.shape
        )
        linear_index = np.unravel_index(
            int(np.argmax(linear_part)), linear_part.shape
        )
        margin_index = np.unravel_index(
            int(np.argmin(squared_margin)), squared_margin.shape
        )
        rows = (
            (float(residual[residual_index]), residual_index),
            (float(linear_part[linear_index]), linear_index),
            (float(squared_margin[margin_index]), margin_index),
        )
        if rows[0][0] > maximum_residual[0]:
            maximum_residual = (
                rows[0][0],
                (
                    float(radial[rows[0][1]]),
                    float(axial[rows[0][1]]),
                ),
            )
        if rows[1][0] > maximum_linear[0]:
            maximum_linear = (
                rows[1][0],
                (
                    float(radial[rows[1][1]]),
                    float(axial[rows[1][1]]),
                ),
            )
        if rows[2][0] < minimum_margin[0]:
            minimum_margin = (
                rows[2][0],
                (
                    float(radial[rows[2][1]]),
                    float(axial[rows[2][1]]),
                ),
            )
    return {
        "radial_point_count": len(radial_grid),
        "axial_half_domain_point_count": len(axial_grid),
        "maximum_HJB_residual": maximum_residual[0],
        "maximum_HJB_residual_location": maximum_residual[1],
        "maximum_linear_part": maximum_linear[0],
        "maximum_linear_part_location": maximum_linear[1],
        "minimum_squared_margin": minimum_margin[0],
        "minimum_squared_margin_location": minimum_margin[1],
    }


def _quad(function) -> float:
    return float(
        quad(
            function,
            0.0,
            1.0,
            epsabs=2.0e-10,
            epsrel=2.0e-10,
            limit=500,
        )[0]
    )


def _finite_energy_norms() -> dict[str, float]:
    def radial_value(radial: float) -> float:
        scaled = radial / RADIUS
        return (
            RADIAL_QUADRATIC_WEIGHT * scaled**2
            + RADIAL_POWER_WEIGHT * scaled**RADIAL_POWER
        )

    def radial_first(radial: float) -> float:
        scaled = radial / RADIUS
        return (
            2.0 * RADIAL_QUADRATIC_WEIGHT * scaled
            + RADIAL_POWER
            * RADIAL_POWER_WEIGHT
            * scaled ** (RADIAL_POWER - 1)
        ) / RADIUS

    def layer(radial: float) -> float:
        return (1.0 - (radial / RADIUS) ** 2) ** RADIAL_EXPONENT

    def layer_first(radial: float) -> float:
        return (
            -2.0
            * RADIAL_EXPONENT
            * radial
            / RADIUS**2
            * (1.0 - (radial / RADIUS) ** 2)
            ** (RADIAL_EXPONENT - 1.0)
        )

    def axial_value(axial: float) -> float:
        cosine = math.cos(AXIAL_FREQUENCY * axial)
        return (
            (1.0 + AXIAL_SHAPE) * cosine**AXIAL_EXPONENT
            - AXIAL_SHAPE * cosine ** (AXIAL_EXPONENT + 1.0)
        )

    def axial_first(axial: float) -> float:
        cosine = math.cos(AXIAL_FREQUENCY * axial)
        sine = math.sin(AXIAL_FREQUENCY * axial)
        return (
            -(1.0 + AXIAL_SHAPE)
            * AXIAL_EXPONENT
            * AXIAL_FREQUENCY
            * sine
            * cosine ** (AXIAL_EXPONENT - 1.0)
            + AXIAL_SHAPE
            * (AXIAL_EXPONENT + 1.0)
            * AXIAL_FREQUENCY
            * sine
            * cosine**AXIAL_EXPONENT
        )

    radial_integrals = {
        "radial_gradient_squared": quad(
            lambda radial: radial * radial_first(radial) ** 2,
            0.0,
            RADIUS,
            limit=500,
        )[0],
        "radial_gradient_layer_cross": quad(
            lambda radial: (
                radial * radial_first(radial) * layer_first(radial)
            ),
            0.0,
            RADIUS,
            limit=500,
        )[0],
        "layer_gradient_squared": quad(
            lambda radial: radial * layer_first(radial) ** 2,
            0.0,
            RADIUS,
            limit=500,
        )[0],
        "layer_squared": quad(
            lambda radial: radial * layer(radial) ** 2,
            0.0,
            RADIUS,
            limit=500,
        )[0],
    }
    axial_integrals = {
        "axial_value": 2.0
        * quad(axial_value, 0.0, HALF_HEIGHT, limit=500)[0],
        "axial_value_squared": 2.0
        * quad(
            lambda axial: axial_value(axial) ** 2,
            0.0,
            HALF_HEIGHT,
            limit=500,
        )[0],
        "axial_gradient_squared": 2.0
        * quad(
            lambda axial: axial_first(axial) ** 2,
            0.0,
            HALF_HEIGHT,
            limit=500,
        )[0],
    }
    gradient_squared = 2.0 * math.pi * (
        2.0
        * HALF_HEIGHT
        * radial_integrals["radial_gradient_squared"]
        + 2.0
        * LAYER_WEIGHT
        * radial_integrals["radial_gradient_layer_cross"]
        * axial_integrals["axial_value"]
        + LAYER_WEIGHT**2
        * radial_integrals["layer_gradient_squared"]
        * axial_integrals["axial_value_squared"]
        + LAYER_WEIGHT**2
        * radial_integrals["layer_squared"]
        * axial_integrals["axial_gradient_squared"]
    )

    sixth_power = 0.0
    for exponent in range(7):
        radial_integral = quad(
            lambda radial, exponent=exponent: (
                radial
                * radial_value(radial) ** (6 - exponent)
                * (LAYER_WEIGHT * layer(radial)) ** exponent
            ),
            0.0,
            RADIUS,
            limit=500,
        )[0]
        if exponent == 0:
            axial_integral = 2.0 * HALF_HEIGHT
        else:
            axial_integral = 2.0 * quad(
                lambda axial, exponent=exponent: (
                    axial_value(axial) ** exponent
                ),
                0.0,
                HALF_HEIGHT,
                limit=500,
            )[0]
        sixth_power += (
            comb(6, exponent, exact=True)
            * radial_integral
            * axial_integral
        )
    sixth_power *= 2.0 * math.pi
    return {
        "gradient_L2_squared": gradient_squared,
        "gradient_L2_norm": math.sqrt(gradient_squared),
        "value_L6_sixth_power": sixth_power,
        "value_L6_norm": sixth_power ** (1.0 / 6.0),
    }


def audit() -> dict[str, object]:
    dense = _dense_grid_diagnostics()
    norms = _finite_energy_norms()
    gain = (
        RADIAL_QUADRATIC_WEIGHT / 4.0
        + RADIAL_POWER_WEIGHT / 2.0**RADIAL_POWER
        + LAYER_WEIGHT * (3.0 / 4.0) ** RADIAL_EXPONENT
    )
    reynolds_level = 0.5
    cubic_support_radius = 1.91
    split_log_gauge_cost = (
        reynolds_level
        * (cubic_support_radius**2 / 3.0 + 0.75)
        / 4.0
    )
    split_one_history_factor = math.exp(split_log_gauge_cost) / 2.0
    split_pair_factor = split_one_history_factor**2
    legacy_return_pair_factor = 0.25
    cycle_coefficient = split_pair_factor + legacy_return_pair_factor
    closure_gain = 1.0 / math.sqrt(cycle_coefficient)
    bare_halving_cycle_coefficient = 0.5161236147249065
    generation = cycle_coefficient * gain**2
    first_eigenvalue = 5.832287335665
    form_floor = first_eigenvalue - 1.0
    poincare_factor = first_eigenvalue / form_floor
    sobolev_constant = 4.0 ** (2.0 / 3.0) / (
        3.0 * math.pi ** (4.0 / 3.0)
    )
    energy_dual_factor = math.sqrt(
        sobolev_constant * poincare_factor
    )
    result: dict[str, object] = {
        "candidate_formula": (
            "U=(197/200)s^2+(3/200)s^32+(11/10)"
            "(1-s^2)^(7/10)[(13/10)c^(69/100)-"
            "(3/10)c^(169/100)]"
        ),
        "coordinates": "s=r/2, c=cos(2*pi*z/3)",
        "radial_boundary_value": 1.0,
        "cap_boundary_lower_bound": 0.0,
        "radial_corner_exponent": RADIAL_EXPONENT,
        "axial_corner_exponent": AXIAL_EXPONENT,
        "candidate_is_in_H1": bool(
            RADIAL_EXPONENT > 0.5 and AXIAL_EXPONENT > 0.5
        ),
        "entry_gain": gain,
        "current_cycle_factors": {
            "cubic_support_radius_over_L": cubic_support_radius,
            "split_one_history_factor": split_one_history_factor,
            "split_pair_factor": split_pair_factor,
            "legacy_return_pair_factor": legacy_return_pair_factor,
            "cycle_coefficient": cycle_coefficient,
        },
        "legacy_bare_halving_cycle_coefficient": (
            bare_halving_cycle_coefficient
        ),
        "maximum_dynamic_one_history_gain_for_closure": closure_gain,
        "additive_gain_allowance": closure_gain - gain,
        "relative_gain_allowance": closure_gain / gain - 1.0,
        "candidate_complete_generation_criterion": generation,
        "remaining_generation_margin": 1.0 - generation,
        "dense_grid": dense,
        "finite_energy_norms": norms,
        "global_energy_forcing_coefficients": {
            "potential_L3_over_2": (
                energy_dual_factor * norms["value_L6_norm"]
            ),
            "drift_L3": (
                energy_dual_factor * norms["gradient_L2_norm"]
            ),
            "potential_relative_form": (
                poincare_factor * sobolev_constant
            ),
        },
        "HJB_residual_interval_certified": False,
        "finite_energy_supersolution_certified": False,
        "averaged_dynamic_entry_trace_closed": False,
        "scope_guard": (
            "the formula, boundary values, H1 threshold, norm quadrature, "
            "and renewal arithmetic are analytic or reproducible; the HJB "
            "sign is currently a dense-grid falsification audit, not an "
            "interval enclosure"
        ),
        "next_gate": (
            "interval-certify the HJB residual and then combine the global "
            "energy response with a space-time L2 trace estimate for the "
            "actual unnormalized exterior-return law"
        ),
    }
    positive_checks = (
        result["candidate_is_in_H1"],
        abs(result["radial_boundary_value"] - 1.0) < 1.0e-15,
        gain < 1.15,
        result["additive_gain_allowance"] > 0.08,
        generation < 0.87,
        dense["maximum_HJB_residual"] < -0.012,
        dense["maximum_linear_part"] < -0.19,
        dense["minimum_squared_margin"] > 0.01,
        1.70 < norms["value_L6_norm"] < 1.71,
        6.54 < norms["gradient_L2_norm"] < 6.56,
    )
    result["all_positive_H1_supersolution_pilot_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
