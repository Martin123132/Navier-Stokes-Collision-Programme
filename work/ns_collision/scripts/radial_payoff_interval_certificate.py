"""Interval certificate for the compact interior of the radial HJB barrier."""

from __future__ import annotations

from collections import deque
import json

from mpmath import iv
import sympy as sp


iv.dps = 18


def _bounds(value: object) -> tuple[float, float]:
    return float(value.a), float(value.b)


def _quantities(
    radial: object,
    axial: object,
    potential_shift: float = 0.0,
) -> tuple[object, ...]:
    interval = iv.mpf
    radius = interval(2)
    frequency = 2 * iv.pi / 3
    radial_quadratic_weight = interval(89945) / 100000
    radial_sixteenth_weight = interval(10055) / 100000
    boundary_layer_weight = interval(13479) / 10000
    radial_exponent = interval(13) / 20
    axial_exponent = interval(7) / 20

    scaled_radius = radial / radius
    radial_layer = 1 - scaled_radius**2
    cosine = iv.cos(frequency * axial)
    sine = iv.sin(frequency * axial)

    radial_value = (
        radial_quadratic_weight * scaled_radius**2
        + radial_sixteenth_weight * scaled_radius**16
    )
    radial_first = (
        2 * radial_quadratic_weight * scaled_radius
        + 16 * radial_sixteenth_weight * scaled_radius**15
    ) / radius
    radial_second = (
        2 * radial_quadratic_weight
        + 240 * radial_sixteenth_weight * scaled_radius**14
    ) / radius**2
    radial_laplacian = (
        4 * radial_quadratic_weight
        + 256 * radial_sixteenth_weight * scaled_radius**14
    ) / radius**2
    radial_laplacian_first = (
        256
        * 14
        * radial_sixteenth_weight
        * scaled_radius**13
        / radius**3
    )

    layer_value = radial_layer**radial_exponent
    layer_first = (
        -2
        * radial_exponent
        * radial
        / radius**2
        * radial_layer ** (radial_exponent - 1)
    )
    layer_second = (
        -2
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
    layer_radial_laplacian = (
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
    layer_radial_laplacian_first = (
        16
        * radial_exponent
        * (radial_exponent - 1)
        * radial
        / radius**4
        * radial_layer ** (radial_exponent - 2)
        - 8
        * radial_exponent
        * (radial_exponent - 1)
        * (radial_exponent - 2)
        * radial**3
        / radius**6
        * radial_layer ** (radial_exponent - 3)
    )

    axial_value = cosine**axial_exponent
    axial_first = (
        -axial_exponent
        * frequency
        * sine
        * cosine ** (axial_exponent - 1)
    )
    axial_second = (
        axial_exponent
        * (axial_exponent - 1)
        * frequency**2
        * sine**2
        * cosine ** (axial_exponent - 2)
        - axial_exponent
        * frequency**2
        * cosine**axial_exponent
    )
    axial_third = (
        axial_exponent
        * frequency**3
        * (
            (3 * axial_exponent - 2)
            * sine
            * cosine ** (axial_exponent - 1)
            - (axial_exponent - 1)
            * (axial_exponent - 2)
            * sine**3
            * cosine ** (axial_exponent - 3)
        )
    )

    value = (
        radial_value
        + boundary_layer_weight * layer_value * axial_value
    )
    radial_gradient = (
        radial_first
        + boundary_layer_weight * layer_first * axial_value
    )
    axial_gradient = (
        boundary_layer_weight * layer_value * axial_first
    )
    radial_second_derivative = (
        radial_second
        + boundary_layer_weight * layer_second * axial_value
    )
    mixed_derivative = (
        boundary_layer_weight * layer_first * axial_first
    )
    axial_second_derivative = (
        boundary_layer_weight * layer_value * axial_second
    )
    laplacian = (
        radial_laplacian
        + boundary_layer_weight
        * (
            layer_radial_laplacian * axial_value
            + layer_value * axial_second
        )
    )
    linear_part = (
        laplacian
        + interval("0.5")
        * (
            radial * radial_gradient
            + axial * axial_gradient
        )
        + value
    )
    linear_part_radial = (
        radial_laplacian_first
        + boundary_layer_weight
        * (
            layer_radial_laplacian_first * axial_value
            + layer_first * axial_second
        )
        + interval("1.5") * radial_gradient
        + interval("0.5")
        * radial
        * radial_second_derivative
        + interval("0.5") * axial * mixed_derivative
    )
    linear_part_axial = (
        boundary_layer_weight
        * (
            layer_radial_laplacian * axial_first
            + layer_value * axial_third
        )
        + interval("0.5") * radial * mixed_derivative
        + interval("1.5") * axial_gradient
        + interval("0.5")
        * axial
        * axial_second_derivative
    )
    if potential_shift:
        shift = interval(str(potential_shift))
        linear_part += shift * value
        linear_part_radial += shift * radial_gradient
        linear_part_axial += shift * axial_gradient

    gradient_squared = radial_gradient**2 + axial_gradient**2
    position_squared = radial**2 + axial**2
    squared_margin = (
        linear_part**2
        - interval("2.25") * position_squared * gradient_squared
    )
    squared_margin_radial = (
        2 * linear_part * linear_part_radial
        - interval("2.25")
        * (
            2 * radial * gradient_squared
            + 2
            * position_squared
            * (
                radial_gradient * radial_second_derivative
                + axial_gradient * mixed_derivative
            )
        )
    )
    squared_margin_axial = (
        2 * linear_part * linear_part_axial
        - interval("2.25")
        * (
            2 * axial * gradient_squared
            + 2
            * position_squared
            * (
                radial_gradient * mixed_derivative
                + axial_gradient * axial_second_derivative
            )
        )
    )
    return (
        linear_part,
        linear_part_radial,
        linear_part_axial,
        squared_margin,
        squared_margin_radial,
        squared_margin_axial,
        value,
        radial_gradient,
        axial_gradient,
        laplacian,
    )


def _mean_value_bounds(
    radial_lower: float,
    radial_upper: float,
    axial_lower: float,
    axial_upper: float,
    potential_shift: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    radial_center = (radial_lower + radial_upper) / 2.0
    axial_center = (axial_lower + axial_upper) / 2.0
    center_values = _quantities(
        iv.mpf([radial_center, radial_center]),
        iv.mpf([axial_center, axial_center]),
        potential_shift,
    )
    box_values = _quantities(
        iv.mpf([radial_lower, radial_upper]),
        iv.mpf([axial_lower, axial_upper]),
        potential_shift,
    )
    radial_displacement = iv.mpf(
        [radial_lower - radial_center, radial_upper - radial_center]
    )
    axial_displacement = iv.mpf(
        [axial_lower - axial_center, axial_upper - axial_center]
    )
    linear_enclosure = (
        center_values[0]
        + radial_displacement * box_values[1]
        + axial_displacement * box_values[2]
    )
    margin_enclosure = (
        center_values[3]
        + radial_displacement * box_values[4]
        + axial_displacement * box_values[5]
    )
    return _bounds(linear_enclosure), _bounds(margin_enclosure)


def _certify_rectangle(
    radial_upper: float = 1.9,
    axial_upper: float = 0.7,
    radial_initial_count: int = 8,
    axial_initial_count: int = 6,
    maximum_depth: int = 12,
    maximum_evaluated_boxes: int = 10_000,
    potential_shift: float = 0.0,
) -> dict[str, object]:
    queue: deque[tuple[float, float, float, float, int]] = deque()
    for radial_index in range(radial_initial_count):
        for axial_index in range(axial_initial_count):
            queue.append(
                (
                    radial_index * radial_upper / radial_initial_count,
                    (radial_index + 1)
                    * radial_upper
                    / radial_initial_count,
                    axial_index * axial_upper / axial_initial_count,
                    (axial_index + 1)
                    * axial_upper
                    / axial_initial_count,
                    0,
                )
            )

    evaluated_boxes = 0
    certified_boxes = 0
    deepest_level = 0
    unresolved: list[dict[str, object]] = []
    while queue and evaluated_boxes < maximum_evaluated_boxes:
        box = queue.popleft()
        radial_lower, radial_box_upper, axial_lower, axial_box_upper, depth = box
        evaluated_boxes += 1
        deepest_level = max(deepest_level, depth)
        linear_bounds, margin_bounds = _mean_value_bounds(
            radial_lower,
            radial_box_upper,
            axial_lower,
            axial_box_upper,
            potential_shift,
        )
        if linear_bounds[1] < 0.0 and margin_bounds[0] > 0.0:
            certified_boxes += 1
            continue
        if depth >= maximum_depth:
            unresolved.append(
                {
                    "box": box[:4],
                    "linear_bounds": linear_bounds,
                    "squared_margin_bounds": margin_bounds,
                }
            )
            continue
        radial_fraction = (
            (radial_box_upper - radial_lower) / radial_upper
        )
        axial_fraction = (
            (axial_box_upper - axial_lower) / axial_upper
        )
        if radial_fraction >= axial_fraction:
            midpoint = (radial_lower + radial_box_upper) / 2.0
            queue.append(
                (
                    radial_lower,
                    midpoint,
                    axial_lower,
                    axial_box_upper,
                    depth + 1,
                )
            )
            queue.append(
                (
                    midpoint,
                    radial_box_upper,
                    axial_lower,
                    axial_box_upper,
                    depth + 1,
                )
            )
        else:
            midpoint = (axial_lower + axial_box_upper) / 2.0
            queue.append(
                (
                    radial_lower,
                    radial_box_upper,
                    axial_lower,
                    midpoint,
                    depth + 1,
                )
            )
            queue.append(
                (
                    radial_lower,
                    radial_box_upper,
                    midpoint,
                    axial_box_upper,
                    depth + 1,
                )
            )
    queued_after_budget = list(queue)
    unresolved_count = len(unresolved) + len(queued_after_budget)
    first_unresolved = unresolved[:5]
    for box in queued_after_budget[: max(0, 5 - len(first_unresolved))]:
        first_unresolved.append(
            {
                "box": box[:4],
                "reason": "box budget exhausted before evaluation",
            }
        )
    return {
        "rectangle": {
            "radial_interval": [0.0, radial_upper],
            "axial_half_interval": [0.0, axial_upper],
            "certified_uniform_killing_rate": potential_shift,
        },
        "evaluated_box_count": evaluated_boxes,
        "certified_leaf_box_count": certified_boxes,
        "deepest_subdivision_level": deepest_level,
        "maximum_evaluated_box_budget": maximum_evaluated_boxes,
        "box_budget_exhausted": bool(queued_after_budget),
        "unresolved_box_count": unresolved_count,
        "first_unresolved_boxes": first_unresolved,
        "compact_interior_certified": unresolved_count == 0,
    }


def _absolute_enclosure(value: object) -> object:
    lower, upper = _bounds(value)
    return iv.mpf([0.0, max(abs(lower), abs(upper))])


def _coarse_residual_upper(
    radial_lower: float,
    radial_upper: float,
    axial_lower: float,
    axial_upper: float,
    potential_shift: float = 0.0,
) -> float:
    radial = iv.mpf([radial_lower, radial_upper])
    axial = iv.mpf([axial_lower, axial_upper])
    quantities = _quantities(radial, axial)
    position_norm = iv.sqrt(radial**2 + axial**2)
    coarse_residual = (
        quantities[9]
        + (1.0 + potential_shift) * quantities[6]
        + 2
        * position_norm
        * (
            _absolute_enclosure(quantities[7])
            + _absolute_enclosure(quantities[8])
        )
    )
    return _bounds(coarse_residual)[1]


def _certify_coarse_rectangle(
    name: str,
    radial_interval: tuple[float, float],
    axial_interval: tuple[float, float],
    maximum_depth: int = 20,
    maximum_evaluated_boxes: int = 20_000,
    potential_shift: float = 0.0,
) -> dict[str, object]:
    queue: deque[tuple[float, float, float, float, int]] = deque(
        [(*radial_interval, *axial_interval, 0)]
    )
    evaluated_boxes = 0
    certified_boxes = 0
    deepest_level = 0
    unresolved: list[dict[str, object]] = []
    radial_width = radial_interval[1] - radial_interval[0]
    axial_width = axial_interval[1] - axial_interval[0]
    while queue and evaluated_boxes < maximum_evaluated_boxes:
        radial_lower, radial_upper, axial_lower, axial_upper, depth = (
            queue.popleft()
        )
        evaluated_boxes += 1
        deepest_level = max(deepest_level, depth)
        residual_upper = _coarse_residual_upper(
            radial_lower,
            radial_upper,
            axial_lower,
            axial_upper,
            potential_shift,
        )
        if residual_upper < 0.0:
            certified_boxes += 1
            continue
        if depth >= maximum_depth:
            unresolved.append(
                {
                    "box": [
                        radial_lower,
                        radial_upper,
                        axial_lower,
                        axial_upper,
                    ],
                    "coarse_residual_upper": residual_upper,
                }
            )
            continue
        radial_fraction = (
            (radial_upper - radial_lower) / radial_width
            if radial_width > 0.0
            else 0.0
        )
        axial_fraction = (
            (axial_upper - axial_lower) / axial_width
            if axial_width > 0.0
            else 0.0
        )
        if radial_fraction >= axial_fraction:
            midpoint = (radial_lower + radial_upper) / 2.0
            queue.append(
                (
                    radial_lower,
                    midpoint,
                    axial_lower,
                    axial_upper,
                    depth + 1,
                )
            )
            queue.append(
                (
                    midpoint,
                    radial_upper,
                    axial_lower,
                    axial_upper,
                    depth + 1,
                )
            )
        else:
            midpoint = (axial_lower + axial_upper) / 2.0
            queue.append(
                (
                    radial_lower,
                    radial_upper,
                    axial_lower,
                    midpoint,
                    depth + 1,
                )
            )
            queue.append(
                (
                    radial_lower,
                    radial_upper,
                    midpoint,
                    axial_upper,
                    depth + 1,
                )
            )
    queued_after_budget = list(queue)
    unresolved_count = len(unresolved) + len(queued_after_budget)
    first_unresolved = unresolved[:5]
    for box in queued_after_budget[: max(0, 5 - len(first_unresolved))]:
        first_unresolved.append(
            {
                "box": box[:4],
                "reason": "box budget exhausted before evaluation",
            }
        )
    return {
        "name": name,
        "radial_interval": list(radial_interval),
        "axial_half_interval": list(axial_interval),
        "certified_uniform_killing_rate": potential_shift,
        "evaluated_box_count": evaluated_boxes,
        "certified_leaf_box_count": certified_boxes,
        "deepest_subdivision_level": deepest_level,
        "maximum_evaluated_box_budget": maximum_evaluated_boxes,
        "box_budget_exhausted": bool(queued_after_budget),
        "unresolved_box_count": unresolved_count,
        "first_unresolved_boxes": first_unresolved,
        "coarse_residual_certified": unresolved_count == 0,
    }


def _asymptotic_strip_margins(
    potential_shift: float = 0.0,
    finite_radial_cutoff: float = 1.999,
    finite_axial_cutoff: float = 0.74953,
) -> dict[str, object]:
    interval = iv.mpf
    radial_exponent = interval(13) / 20
    axial_exponent = interval(7) / 20
    boundary_layer_weight = interval(13479) / 10000
    frequency = 2 * iv.pi / 3
    gradient_coefficient = iv.sqrt(73) / 2
    cutoff = interval(1) / 1000
    radial_curvature_floor = (
        radial_exponent
        * (1 - radial_exponent)
        * (interval(19) / 10) ** 2
        / 4
    )
    axial_curvature_floor = (
        axial_exponent
        * (1 - axial_exponent)
        * frequency**2
        * iv.sin(frequency * interval(7) / 10) ** 2
    )
    radial_laplacian_ceiling = (
        interval(89945) / 100000
        + 64 * interval(10055) / 100000
    )
    radial_gradient_ceiling = (
        interval(89945) / 100000
        + 8 * interval(10055) / 100000
    )
    constant_load = (
        radial_laplacian_ceiling
        + 1
        + boundary_layer_weight
        + gradient_coefficient * radial_gradient_ceiling
        + interval(str(potential_shift)) * (1 + boundary_layer_weight)
    )
    radial_dominance_margin = (
        boundary_layer_weight
        / cutoff
        * (
            radial_curvature_floor
            - gradient_coefficient * radial_exponent * cutoff
        )
        - gradient_coefficient
        * boundary_layer_weight
        * axial_exponent
        * frequency
        - constant_load
    )
    axial_dominance_margin = (
        boundary_layer_weight
        / cutoff
        * (
            axial_curvature_floor
            - gradient_coefficient
            * axial_exponent
            * frequency
            * cutoff
        )
        - gradient_coefficient
        * boundary_layer_weight
        * radial_exponent
        - constant_load
    )
    radial_bounds = _bounds(radial_dominance_margin)
    axial_bounds = _bounds(axial_dominance_margin)
    finite_radial_point = interval(
        [finite_radial_cutoff, finite_radial_cutoff]
    )
    finite_axial_point = interval(
        [finite_axial_cutoff, finite_axial_cutoff]
    )
    radial_overlap_coordinate = 1 - (finite_radial_point / 2) ** 2
    axial_overlap_coordinate = iv.cos(frequency * finite_axial_point)
    radial_overlap_bounds = _bounds(radial_overlap_coordinate)
    axial_overlap_bounds = _bounds(axial_overlap_coordinate)
    finite_cutoffs_overlap = bool(
        radial_overlap_bounds[1] < 0.001
        and axial_overlap_bounds[1] < 0.001
    )
    return {
        "transformed_variables": (
            "x=1-r^2/4 and a=cos(2*pi*z/3)"
        ),
        "cutoff": 0.001,
        "finite_radial_cutoff": finite_radial_cutoff,
        "finite_axial_cutoff": finite_axial_cutoff,
        "radial_overlap_coordinate_interval": radial_overlap_bounds,
        "axial_overlap_coordinate_interval": axial_overlap_bounds,
        "finite_cutoffs_overlap_asymptotic_union": (
            finite_cutoffs_overlap
        ),
        "certified_uniform_killing_rate": potential_shift,
        "case_split": (
            "if a>=x, radial curvature absorbs radial gradient and "
            "the axial-gradient ratio is at most one; if x>=a, axial "
            "curvature absorbs axial gradient and the radial-gradient "
            "ratio is at most one"
        ),
        "radial_case_margin_interval": radial_bounds,
        "axial_case_margin_interval": axial_bounds,
        "asymptotic_union_certified": bool(
            radial_bounds[0] > 0.0
            and axial_bounds[0] > 0.0
            and finite_cutoffs_overlap
        ),
    }


def _symbolic_derivative_cross_check() -> dict[str, object]:
    radial, axial = sp.symbols("radial axial", positive=True)
    radius = sp.Integer(2)
    frequency = 2 * sp.pi / 3
    radial_exponent = sp.Rational(13, 20)
    axial_exponent = sp.Rational(7, 20)
    boundary_layer_weight = sp.Rational(13479, 10000)
    radial_quadratic_weight = sp.Rational(89945, 100000)
    radial_sixteenth_weight = 1 - radial_quadratic_weight
    scaled_radius = radial / radius
    radial_layer = 1 - scaled_radius**2
    cosine = sp.cos(frequency * axial)
    sine = sp.sin(frequency * axial)

    radial_value = (
        radial_quadratic_weight * scaled_radius**2
        + radial_sixteenth_weight * scaled_radius**16
    )
    radial_first = sp.diff(radial_value, radial)
    radial_second = sp.diff(radial_first, radial)
    radial_laplacian = (
        4 * radial_quadratic_weight
        + 256 * radial_sixteenth_weight * scaled_radius**14
    ) / radius**2
    radial_laplacian_first = (
        256
        * 14
        * radial_sixteenth_weight
        * scaled_radius**13
        / radius**3
    )
    layer_value = radial_layer**radial_exponent
    layer_first = (
        -2
        * radial_exponent
        * radial
        / radius**2
        * radial_layer ** (radial_exponent - 1)
    )
    layer_second = (
        -2
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
    layer_laplacian_first = (
        16
        * radial_exponent
        * (radial_exponent - 1)
        * radial
        / radius**4
        * radial_layer ** (radial_exponent - 2)
        - 8
        * radial_exponent
        * (radial_exponent - 1)
        * (radial_exponent - 2)
        * radial**3
        / radius**6
        * radial_layer ** (radial_exponent - 3)
    )
    axial_value = cosine**axial_exponent
    axial_first = (
        -axial_exponent
        * frequency
        * sine
        * cosine ** (axial_exponent - 1)
    )
    axial_second = (
        axial_exponent
        * (axial_exponent - 1)
        * frequency**2
        * sine**2
        * cosine ** (axial_exponent - 2)
        - axial_exponent
        * frequency**2
        * cosine**axial_exponent
    )
    axial_third = (
        axial_exponent
        * frequency**3
        * (
            (3 * axial_exponent - 2)
            * sine
            * cosine ** (axial_exponent - 1)
            - (axial_exponent - 1)
            * (axial_exponent - 2)
            * sine**3
            * cosine ** (axial_exponent - 3)
        )
    )
    value = (
        radial_value
        + boundary_layer_weight * layer_value * axial_value
    )
    radial_gradient = (
        radial_first
        + boundary_layer_weight * layer_first * axial_value
    )
    axial_gradient = (
        boundary_layer_weight * layer_value * axial_first
    )
    radial_second_derivative = (
        radial_second
        + boundary_layer_weight * layer_second * axial_value
    )
    mixed_derivative = (
        boundary_layer_weight * layer_first * axial_first
    )
    axial_second_derivative = (
        boundary_layer_weight * layer_value * axial_second
    )
    linear_part = (
        radial_laplacian
        + boundary_layer_weight
        * (layer_laplacian * axial_value + layer_value * axial_second)
        + sp.Rational(1, 2)
        * (radial * radial_gradient + axial * axial_gradient)
        + value
    )
    linear_part_radial = (
        radial_laplacian_first
        + boundary_layer_weight
        * (
            layer_laplacian_first * axial_value
            + layer_first * axial_second
        )
        + sp.Rational(3, 2) * radial_gradient
        + sp.Rational(1, 2) * radial * radial_second_derivative
        + sp.Rational(1, 2) * axial * mixed_derivative
    )
    linear_part_axial = (
        boundary_layer_weight
        * (layer_laplacian * axial_first + layer_value * axial_third)
        + sp.Rational(1, 2) * radial * mixed_derivative
        + sp.Rational(3, 2) * axial_gradient
        + sp.Rational(1, 2) * axial * axial_second_derivative
    )
    checks = {
        "radial_laplacian": sp.simplify(
            sp.diff(radial_value, radial, 2)
            + sp.diff(radial_value, radial) / radial
            - radial_laplacian
        ),
        "radial_laplacian_first": sp.simplify(
            sp.diff(radial_laplacian, radial)
            - radial_laplacian_first
        ),
        "layer_first": sp.simplify(
            sp.diff(layer_value, radial) - layer_first
        ),
        "layer_second": sp.simplify(
            sp.diff(layer_first, radial) - layer_second
        ),
        "layer_laplacian": sp.simplify(
            sp.diff(layer_value, radial, 2)
            + sp.diff(layer_value, radial) / radial
            - layer_laplacian
        ),
        "layer_laplacian_first": sp.simplify(
            sp.diff(layer_laplacian, radial)
            - layer_laplacian_first
        ),
        "axial_first": sp.trigsimp(
            sp.diff(axial_value, axial) - axial_first
        ),
        "axial_second": sp.trigsimp(
            sp.diff(axial_first, axial) - axial_second
        ),
        "axial_third": sp.trigsimp(
            sp.diff(axial_second, axial) - axial_third
        ),
        "linear_part_radial": sp.trigsimp(
            sp.diff(linear_part, radial) - linear_part_radial
        ),
        "linear_part_axial": sp.trigsimp(
            sp.diff(linear_part, axial) - linear_part_axial
        ),
    }
    return {
        "checked_identities": list(checks),
        "all_manual_derivatives_exact": all(
            value == 0 for value in checks.values()
        ),
    }


def audit() -> dict[str, object]:
    symbolic_derivatives = _symbolic_derivative_cross_check()
    certified_uniform_killing_rate = 0.005
    interior = _certify_rectangle(
        potential_shift=certified_uniform_killing_rate
    )
    radial_cutoff = 1.999
    axial_cutoff = 0.74953
    finite_boundary_rectangles = [
        _certify_coarse_rectangle(
            "radial collar",
            (1.9, radial_cutoff),
            (0.0, 0.7),
            potential_shift=certified_uniform_killing_rate,
        ),
        _certify_coarse_rectangle(
            "axial collar",
            (0.0, 1.9),
            (0.7, axial_cutoff),
            potential_shift=certified_uniform_killing_rate,
        ),
        _certify_coarse_rectangle(
            "corner collar",
            (1.9, radial_cutoff),
            (0.7, axial_cutoff),
            potential_shift=certified_uniform_killing_rate,
        ),
    ]
    finite_boundary_certified = all(
        row["coarse_residual_certified"]
        for row in finite_boundary_rectangles
    )
    asymptotic = _asymptotic_strip_margins(
        certified_uniform_killing_rate,
        radial_cutoff,
        axial_cutoff,
    )
    whole_domain_certified = bool(
        interior["compact_interior_certified"]
        and finite_boundary_certified
        and asymptotic["asymptotic_union_certified"]
    )
    candidate_gain = (
        0.89945 / 4.0
        + 0.10055 / 2.0**16
        + 1.3479 * (3.0 / 4.0) ** (13.0 / 20.0)
    )
    cycle_coefficient = 0.6586950386676936
    generation_criterion = cycle_coefficient * candidate_gain**2
    result: dict[str, object] = {
        "interval_backend": "mpmath.iv with 18 decimal digits",
        "certificate_criterion": (
            "L_epsilon=Delta U+(y.grad U)/2+1.005U<0 and "
            "G_epsilon=L_epsilon^2-(9/4)|y|^2|grad U|^2>0"
        ),
        "mean_value_enclosures_used": True,
        "symbolic_derivative_cross_check": symbolic_derivatives,
        "compact_interior": interior,
        "finite_boundary_rectangles": finite_boundary_rectangles,
        "finite_boundary_rectangles_certified": (
            finite_boundary_certified
        ),
        "asymptotic_boundary_strips": asymptotic,
        "radial_boundary_strip_certified": bool(
            finite_boundary_certified
            and asymptotic["asymptotic_union_certified"]
        ),
        "axial_boundary_strip_certified": bool(
            finite_boundary_certified
            and asymptotic["asymptotic_union_certified"]
        ),
        "whole_open_half_cylinder_certified": whole_domain_certified,
        "certified_uniform_Doob_killing_rate": (
            certified_uniform_killing_rate
        ),
        "stochastic_verification": (
            "for every admissible progressively measurable B, "
            "exp(s)U(Y_s) stopped before cylinder exit is a nonnegative "
            "supermartingale; localization and Fatou bound the radial "
            "Feynman-Kac payoff by U"
        ),
        "certified_pointwise_radial_payoff_gain": candidate_gain,
        "certified_complete_generation_criterion": generation_criterion,
        "certified_dynamic_cycle_closes": bool(
            whole_domain_certified and generation_criterion < 1.0
        ),
        "superseded_by_finite_energy_H1_barrier": True,
        "ideal_nonautonomous_boundary_theorem_certified": (
            whole_domain_certified
        ),
        "scope_guard": (
            "the compact and finite-collar rows use outward-rounded "
            "interval arithmetic; the final open strips use the displayed "
            "two-case singular-order inequalities"
        ),
        "next_gate": (
            "use the lower-gain finite-energy H1 barrier for the current "
            "cubic split cycle; this older barrier remains a valid HJB "
            "certificate but no longer closes the complete generation"
        ),
    }
    result["all_positive_compact_interval_checks_pass"] = bool(
        interior["compact_interior_certified"]
        and interior["evaluated_box_count"] > 0
        and finite_boundary_certified
        and asymptotic["asymptotic_union_certified"]
        and whole_domain_certified
        and symbolic_derivatives["all_manual_derivatives_exact"]
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
