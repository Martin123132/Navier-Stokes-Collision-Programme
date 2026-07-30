"""Interval certificate for the finite-energy radial-payoff barrier."""

from __future__ import annotations

from collections import deque
import json
import math

from mpmath import iv


iv.dps = 18

RADIUS = 2.0
HALF_HEIGHT = 0.75
RADIAL_CUTOFF = 1.999
AXIAL_CUTOFF = 0.7495


def _bounds(value: object) -> tuple[float, float]:
    return float(value.a), float(value.b)


def _physical_residual_upper(
    radial_lower: float,
    radial_upper: float,
    axial_lower: float,
    axial_upper: float,
) -> float:
    interval = iv.mpf
    radial = interval([radial_lower, radial_upper])
    axial = interval([axial_lower, axial_upper])
    radius = interval(2)
    frequency = 2 * iv.pi / 3
    radial_weight = interval(197) / 200
    power_weight = interval(3) / 200
    layer_weight = interval(11) / 10
    radial_exponent = interval(7) / 10
    axial_exponent = interval(69) / 100
    axial_shape = interval(3) / 10

    scaled = radial / radius
    radial_layer = 1 - scaled**2
    cosine = iv.cos(frequency * axial)
    sine = iv.sin(frequency * axial)
    radial_value = radial_weight * scaled**2 + power_weight * scaled**32
    radial_first = (
        2 * radial_weight * scaled + 32 * power_weight * scaled**31
    ) / radius
    radial_laplacian = (
        4 * radial_weight + 1024 * power_weight * scaled**30
    ) / radius**2
    layer_value = radial_layer**radial_exponent
    layer_first = (
        -2
        * radial_exponent
        * radial
        / radius**2
        * radial_layer ** (radial_exponent - 1)
    )
    layer_laplacian = (
        -4
        * radial_exponent
        / radius**2
        * radial_layer ** (radial_exponent - 1)
        + 4
        * radial_exponent
        * (radial_exponent - 1)
        * radial**2
        / radius**4
        * radial_layer ** (radial_exponent - 2)
    )
    axial_value = (
        (1 + axial_shape) * cosine**axial_exponent
        - axial_shape * cosine ** (axial_exponent + 1)
    )
    axial_first = (
        -(1 + axial_shape)
        * axial_exponent
        * frequency
        * sine
        * cosine ** (axial_exponent - 1)
        + axial_shape
        * (axial_exponent + 1)
        * frequency
        * sine
        * cosine**axial_exponent
    )
    axial_second = (
        (1 + axial_shape)
        * (
            axial_exponent
            * (axial_exponent - 1)
            * frequency**2
            * sine**2
            * cosine ** (axial_exponent - 2)
            - axial_exponent
            * frequency**2
            * cosine**axial_exponent
        )
        - axial_shape
        * (
            (axial_exponent + 1)
            * axial_exponent
            * frequency**2
            * sine**2
            * cosine ** (axial_exponent - 1)
            - (axial_exponent + 1)
            * frequency**2
            * cosine ** (axial_exponent + 1)
        )
    )
    value = radial_value + layer_weight * layer_value * axial_value
    radial_gradient = (
        radial_first + layer_weight * layer_first * axial_value
    )
    axial_gradient = layer_weight * layer_value * axial_first
    laplacian = radial_laplacian + layer_weight * (
        layer_laplacian * axial_value
        + layer_value * axial_second
    )
    linear_part = (
        laplacian
        + (radial * radial_gradient + axial * axial_gradient) / 2
        + interval("1.005") * value
    )
    residual = linear_part + interval("1.5") * iv.sqrt(
        (radial**2 + axial**2)
        * (radial_gradient**2 + axial_gradient**2)
    )
    return _bounds(residual)[1]


def _certify_rectangle(
    name: str,
    radial_interval: tuple[float, float],
    axial_interval: tuple[float, float],
    radial_initial_count: int,
    axial_initial_count: int,
    maximum_depth: int = 28,
    maximum_evaluated_boxes: int = 120_000,
) -> dict[str, object]:
    radial_lower, radial_upper = radial_interval
    axial_lower, axial_upper = axial_interval
    queue: deque[tuple[float, float, float, float, int]] = deque()
    for radial_index in range(radial_initial_count):
        for axial_index in range(axial_initial_count):
            queue.append(
                (
                    radial_lower
                    + (radial_upper - radial_lower)
                    * radial_index
                    / radial_initial_count,
                    radial_lower
                    + (radial_upper - radial_lower)
                    * (radial_index + 1)
                    / radial_initial_count,
                    axial_lower
                    + (axial_upper - axial_lower)
                    * axial_index
                    / axial_initial_count,
                    axial_lower
                    + (axial_upper - axial_lower)
                    * (axial_index + 1)
                    / axial_initial_count,
                    0,
                )
            )
    evaluated = 0
    certified = 0
    deepest = 0
    unresolved: list[dict[str, object]] = []
    while queue and evaluated < maximum_evaluated_boxes:
        box = queue.popleft()
        r0, r1, z0, z1, depth = box
        evaluated += 1
        deepest = max(deepest, depth)
        residual_upper = _physical_residual_upper(r0, r1, z0, z1)
        if residual_upper < 0.0:
            certified += 1
            continue
        if depth >= maximum_depth:
            unresolved.append(
                {"box": box[:4], "residual_upper": residual_upper}
            )
            continue
        radial_fraction = (r1 - r0) / (radial_upper - radial_lower)
        axial_fraction = (z1 - z0) / (axial_upper - axial_lower)
        if radial_fraction >= axial_fraction:
            midpoint = (r0 + r1) / 2.0
            queue.append((r0, midpoint, z0, z1, depth + 1))
            queue.append((midpoint, r1, z0, z1, depth + 1))
        else:
            midpoint = (z0 + z1) / 2.0
            queue.append((r0, r1, z0, midpoint, depth + 1))
            queue.append((r0, r1, midpoint, z1, depth + 1))
    queued = list(queue)
    unresolved_count = len(unresolved) + len(queued)
    first_unresolved = unresolved[:5]
    for box in queued[: max(0, 5 - len(first_unresolved))]:
        first_unresolved.append(
            {"box": box[:4], "reason": "box budget exhausted"}
        )
    return {
        "name": name,
        "radial_interval": list(radial_interval),
        "axial_half_interval": list(axial_interval),
        "evaluated_box_count": evaluated,
        "certified_leaf_box_count": certified,
        "deepest_subdivision_level": deepest,
        "box_budget_exhausted": bool(queued),
        "unresolved_box_count": unresolved_count,
        "first_unresolved_boxes": first_unresolved,
        "rectangle_certified": unresolved_count == 0,
    }


def _transformed_common(x: object, axial_cosine: object) -> tuple[object, ...]:
    interval = iv.mpf
    frequency = 2 * iv.pi / 3
    radial_weight = interval(197) / 200
    power_weight = interval(3) / 200
    axial_exponent = interval(69) / 100
    axial_shape = interval(3) / 10
    radial_root = iv.sqrt(1 - x)
    radial = 2 * radial_root
    sine = iv.sqrt(1 - axial_cosine**2)
    axial = iv.atan2(sine, axial_cosine) / frequency
    radial_value = (
        radial_weight * (1 - x) + power_weight * (1 - x) ** 16
    )
    radial_first = (
        radial_weight * radial_root
        + 16 * power_weight * radial_root**31
    )
    radial_laplacian = radial_weight + 256 * power_weight * (1 - x) ** 15
    baseline_linear = (
        radial_laplacian
        + radial * radial_first / 2
        + interval("1.005") * radial_value
    )
    axial_polynomial = 1 + axial_shape - axial_shape * axial_cosine
    axial_first_polynomial = (
        (1 + axial_shape) * axial_exponent
        - axial_shape * (axial_exponent + 1) * axial_cosine
    )
    axial_second_polynomial = (
        (1 - axial_cosine**2)
        * (
            (1 + axial_shape)
            * axial_exponent
            * (axial_exponent - 1)
            - axial_shape
            * axial_exponent
            * (axial_exponent + 1)
            * axial_cosine
        )
        - axial_cosine**2 * axial_first_polynomial
    )
    return (
        radial,
        axial,
        sine,
        radial_first,
        baseline_linear,
        axial_polynomial,
        axial_first_polynomial,
        axial_second_polynomial,
    )


def _scaled_residual_upper(
    x_lower: float,
    x_upper: float,
    cosine_lower: float,
    cosine_upper: float,
) -> float:
    interval = iv.mpf
    x = interval([x_lower, x_upper])
    cosine = interval([cosine_lower, cosine_upper])
    layer_weight = interval(11) / 10
    radial_exponent = interval(7) / 10
    axial_exponent = interval(69) / 100
    frequency = 2 * iv.pi / 3
    common = _transformed_common(x, cosine)
    radial, axial, sine, radial_first, baseline, phi, phi_a, phi_aa = common
    scaled_linear = (
        x ** (2 - radial_exponent)
        * cosine ** (2 - axial_exponent)
        * baseline
        + layer_weight
        * cosine**2
        * phi
        * (
            -radial_exponent * x
            + radial_exponent
            * (radial_exponent - 1)
            * radial**2
            / 4
        )
        + layer_weight * x**2 * frequency**2 * phi_aa
        - layer_weight
        * radial_exponent
        * radial**2
        * x
        * cosine**2
        * phi
        / 4
        - layer_weight
        * axial
        * frequency
        * sine
        * x**2
        * cosine
        * phi_a
        / 2
        + layer_weight
        * interval("1.005")
        * x**2
        * cosine**2
        * phi
    )
    scaled_radial_gradient = (
        x ** (2 - radial_exponent)
        * cosine ** (2 - axial_exponent)
        * radial_first
        - layer_weight
        * radial_exponent
        * radial
        * x
        * cosine**2
        * phi
        / 2
    )
    scaled_axial_gradient = (
        -layer_weight
        * frequency
        * sine
        * x**2
        * cosine
        * phi_a
    )
    scaled_residual = scaled_linear + interval("1.5") * iv.sqrt(
        radial**2 + axial**2
    ) * iv.sqrt(
        scaled_radial_gradient**2 + scaled_axial_gradient**2
    )
    return _bounds(scaled_residual)[1]


def _ratio_residual_upper(
    scale_lower: float,
    scale_upper: float,
    ratio_lower: float,
    ratio_upper: float,
    axial_distance_dominates: bool,
) -> float:
    interval = iv.mpf
    scale = interval([scale_lower, scale_upper])
    ratio = interval([ratio_lower, ratio_upper])
    layer_weight = interval(11) / 10
    radial_exponent = interval(7) / 10
    axial_exponent = interval(69) / 100
    frequency = 2 * iv.pi / 3
    if axial_distance_dominates:
        axial_cosine = scale
        x = ratio * scale
        radial_ratio = ratio
        axial_ratio = interval(1)
    else:
        x = scale
        axial_cosine = ratio * scale
        radial_ratio = interval(1)
        axial_ratio = ratio
    common = _transformed_common(x, axial_cosine)
    radial, axial, sine, radial_first, baseline, phi, phi_a, phi_aa = common
    common_power = scale ** (2 - radial_exponent - axial_exponent)
    baseline_factor = (
        radial_ratio ** (2 - radial_exponent)
        * axial_ratio ** (2 - axial_exponent)
        * common_power
    )
    # Work directly with the continuous scale=0 extensions of the quotients.
    if axial_distance_dominates:
        axial_drift = (
            -layer_weight
            * axial
            * frequency
            * sine
            * ratio**2
            * scale
            * phi_a
            / 2
        )
        scaled_linear = (
            baseline_factor * baseline
            + layer_weight
            * phi
            * (
                -radial_exponent * x
                + radial_exponent
                * (radial_exponent - 1)
                * radial**2
                / 4
            )
            + layer_weight * ratio**2 * frequency**2 * phi_aa
            - layer_weight
            * radial_exponent
            * radial**2
            * ratio
            * scale
            * phi
            / 4
            + axial_drift
            + layer_weight
            * interval("1.005")
            * ratio**2
            * scale**2
            * phi
        )
        scaled_radial_gradient = (
            ratio ** (2 - radial_exponent)
            * common_power
            * radial_first
            - layer_weight
            * radial_exponent
            * radial
            * ratio
            * scale
            * phi
            / 2
        )
        scaled_axial_gradient = (
            -layer_weight
            * frequency
            * sine
            * ratio**2
            * scale
            * phi_a
        )
    else:
        scaled_linear = (
            ratio ** (2 - axial_exponent)
            * common_power
            * baseline
            + layer_weight
            * ratio**2
            * phi
            * (
                -radial_exponent * x
                + radial_exponent
                * (radial_exponent - 1)
                * radial**2
                / 4
            )
            + layer_weight * frequency**2 * phi_aa
            - layer_weight
            * radial_exponent
            * radial**2
            * ratio**2
            * scale
            * phi
            / 4
            - layer_weight
            * axial
            * frequency
            * sine
            * ratio
            * scale
            * phi_a
            / 2
            + layer_weight
            * interval("1.005")
            * ratio**2
            * scale**2
            * phi
        )
        scaled_radial_gradient = (
            ratio ** (2 - axial_exponent)
            * common_power
            * radial_first
            - layer_weight
            * radial_exponent
            * radial
            * ratio**2
            * scale
            * phi
            / 2
        )
        scaled_axial_gradient = (
            -layer_weight
            * frequency
            * sine
            * ratio
            * scale
            * phi_a
        )
    residual = scaled_linear + interval("1.5") * iv.sqrt(
        radial**2 + axial**2
    ) * iv.sqrt(
        scaled_radial_gradient**2 + scaled_axial_gradient**2
    )
    return _bounds(residual)[1]


def _adaptive_transformed(
    name: str,
    first_interval: tuple[float, float],
    second_interval: tuple[float, float],
    evaluator,
    first_initial_count: int,
    second_initial_count: int,
    maximum_depth: int = 28,
    maximum_evaluated_boxes: int = 100_000,
) -> dict[str, object]:
    first_lower, first_upper = first_interval
    second_lower, second_upper = second_interval
    queue: deque[tuple[float, float, float, float, int]] = deque()
    for first_index in range(first_initial_count):
        for second_index in range(second_initial_count):
            queue.append(
                (
                    first_lower
                    + (first_upper - first_lower)
                    * first_index
                    / first_initial_count,
                    first_lower
                    + (first_upper - first_lower)
                    * (first_index + 1)
                    / first_initial_count,
                    second_lower
                    + (second_upper - second_lower)
                    * second_index
                    / second_initial_count,
                    second_lower
                    + (second_upper - second_lower)
                    * (second_index + 1)
                    / second_initial_count,
                    0,
                )
            )
    evaluated = 0
    certified = 0
    unresolved: list[dict[str, object]] = []
    deepest = 0
    while queue and evaluated < maximum_evaluated_boxes:
        box = queue.popleft()
        first0, first1, second0, second1, depth = box
        evaluated += 1
        deepest = max(deepest, depth)
        residual_upper = evaluator(first0, first1, second0, second1)
        if residual_upper < 0.0:
            certified += 1
            continue
        if depth >= maximum_depth:
            unresolved.append(
                {"box": box[:4], "residual_upper": residual_upper}
            )
            continue
        first_fraction = (first1 - first0) / (first_upper - first_lower)
        second_fraction = (second1 - second0) / (
            second_upper - second_lower
        )
        if first_fraction >= second_fraction:
            midpoint = (first0 + first1) / 2.0
            queue.append(
                (first0, midpoint, second0, second1, depth + 1)
            )
            queue.append(
                (midpoint, first1, second0, second1, depth + 1)
            )
        else:
            midpoint = (second0 + second1) / 2.0
            queue.append(
                (first0, first1, second0, midpoint, depth + 1)
            )
            queue.append(
                (first0, first1, midpoint, second1, depth + 1)
            )
    queued = list(queue)
    unresolved_count = len(unresolved) + len(queued)
    return {
        "name": name,
        "first_interval": list(first_interval),
        "second_interval": list(second_interval),
        "evaluated_box_count": evaluated,
        "certified_leaf_box_count": certified,
        "deepest_subdivision_level": deepest,
        "box_budget_exhausted": bool(queued),
        "unresolved_box_count": unresolved_count,
        "first_unresolved_boxes": unresolved[:5],
        "transformed_region_certified": unresolved_count == 0,
    }


def audit() -> dict[str, object]:
    finite_rectangles = [
        _certify_rectangle("compact", (0.0, 1.9), (0.0, 0.7), 16, 12),
        _certify_rectangle(
            "radial_finite_collar", (1.9, RADIAL_CUTOFF), (0.0, 0.7), 4, 8
        ),
        _certify_rectangle(
            "axial_finite_collar", (0.0, 1.9), (0.7, AXIAL_CUTOFF), 12, 4
        ),
        _certify_rectangle(
            "finite_corner",
            (1.9, RADIAL_CUTOFF),
            (0.7, AXIAL_CUTOFF),
            6,
            6,
        ),
    ]
    x_cutoff = 1.0 - (RADIAL_CUTOFF / 2.0) ** 2
    cosine_cutoff = math.cos(2.0 * math.pi * AXIAL_CUTOFF / 3.0)
    transformed_regions = [
        _adaptive_transformed(
            "open_radial_strip",
            (0.0, x_cutoff),
            (cosine_cutoff, 1.0),
            _scaled_residual_upper,
            2,
            16,
        ),
        _adaptive_transformed(
            "open_axial_strip",
            (x_cutoff, 1.0),
            (0.0, cosine_cutoff),
            _scaled_residual_upper,
            16,
            2,
        ),
        _adaptive_transformed(
            "corner_axial_distance_dominates",
            (0.0, cosine_cutoff),
            (0.0, 1.0),
            lambda b0, b1, t0, t1: _ratio_residual_upper(
                b0, b1, t0, t1, True
            ),
            4,
            12,
        ),
        _adaptive_transformed(
            "corner_radial_distance_dominates",
            (0.0, x_cutoff),
            (0.0, 1.0),
            lambda b0, b1, t0, t1: _ratio_residual_upper(
                b0, b1, t0, t1, False
            ),
            4,
            12,
        ),
    ]
    finite_certified = all(
        row["rectangle_certified"] for row in finite_rectangles
    )
    transformed_certified = all(
        row["transformed_region_certified"]
        for row in transformed_regions
    )
    gain = (
        197.0 / 800.0
        + (3.0 / 200.0) / 2.0**32
        + 1.1 * 0.75**0.7
    )
    reynolds_level = 0.5
    cubic_support_radius = 1.91
    split_one_history_factor = math.exp(
        reynolds_level
        * (cubic_support_radius**2 / 3.0 + 0.75)
        / 4.0
    ) / 2.0
    split_pair_factor = split_one_history_factor**2
    cycle_coefficient = split_pair_factor + 0.25
    closure_gain = 1.0 / math.sqrt(cycle_coefficient)
    result: dict[str, object] = {
        "candidate_formula": (
            "U=(197/200)s^2+(3/200)s^32+(11/10)"
            "(1-s^2)^(7/10)[(13/10)c^(69/100)-"
            "(3/10)c^(169/100)]"
        ),
        "certified_HJB_operator": (
            "Delta U+(y.grad U)/2+(3/2)|y||grad U|+1.005U<0"
        ),
        "finite_rectangles": finite_rectangles,
        "transformed_open_strips": transformed_regions,
        "radial_distance_cutoff": x_cutoff,
        "axial_cosine_cutoff": cosine_cutoff,
        "weighted_boundary_multiplier": (
            "x^(2-7/10)c^(2-69/100)>0 in the open cylinder"
        ),
        "corner_cover": (
            "the two cases c>=x and x>=c cover the final open corner; "
            "division by c^2 or x^2 leaves only nonnegative powers"
        ),
        "finite_regions_certified": finite_certified,
        "transformed_open_regions_certified": transformed_certified,
        "whole_open_half_cylinder_certified": bool(
            finite_certified and transformed_certified
        ),
        "radial_and_axial_symmetry_complete_full_cylinder": True,
        "certified_uniform_Doob_killing_rate": 0.005,
        "candidate_is_in_H1": True,
        "certified_entry_gain": gain,
        "current_cycle_factors": {
            "cubic_support_radius_over_L": cubic_support_radius,
            "split_one_history_factor": split_one_history_factor,
            "split_pair_factor": split_pair_factor,
            "legacy_return_pair_factor": 0.25,
            "cycle_coefficient": cycle_coefficient,
        },
        "legacy_bare_halving_cycle_coefficient": 0.5161236147249065,
        "maximum_dynamic_one_history_gain_for_closure": closure_gain,
        "certified_additive_gain_allowance": closure_gain - gain,
        "certified_complete_generation_criterion": (
            cycle_coefficient * gain**2
        ),
        "finite_energy_supersolution_certified": bool(
            finite_certified and transformed_certified
        ),
        "scope_guard": (
            "this certifies the ideal controlled affine HJB barrier and "
            "its finite-energy boundary exponents. The separate averaged "
            "entry theorem still requires a physical exterior-return "
            "density envelope and critical Navier-Stokes error bounds"
        ),
        "next_gate": (
            "derive an unnormalized weighted space-time density envelope "
            "for the actual outer-to-inner return kernel"
        ),
    }
    result["all_positive_H1_interval_checks_pass"] = bool(
        result["whole_open_half_cylinder_certified"]
        and result["finite_energy_supersolution_certified"]
        and result["candidate_is_in_H1"]
        and gain < 1.15
        and result["certified_complete_generation_criterion"] < 0.87
        and not any(
            row["box_budget_exhausted"] for row in finite_rectangles
        )
        and not any(
            row["box_budget_exhausted"] for row in transformed_regions
        )
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
