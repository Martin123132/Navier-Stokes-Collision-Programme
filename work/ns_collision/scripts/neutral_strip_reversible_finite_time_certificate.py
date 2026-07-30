"""Certify the finite-time response of the reversible strip FEM models."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import mpmath
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


TERMINAL_TIME = 6.0
CERTIFIED_WINDOW = 0.375
EARLY_TIME = 0.005
EARLY_SUBSTEP = 0.0001
POISSON_TAIL_TARGET = 1.0e-18
TAYLOR_ORDER = 3


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _up(value) -> float:
    return np.nextafter(float(value), math.inf)


def _gamma(operation_count: int) -> float:
    unit = np.finfo(float).eps
    product = operation_count * unit
    if product >= 0.01:
        raise RuntimeError("roundoff operation count is too large")
    return _up(product / (1.0 - product))


def _maximum_row_nnz(matrix) -> int:
    csr = matrix.tocsr()
    return int(np.max(np.diff(csr.indptr)))


def _matmul_roundoff(matrix_abs, values: np.ndarray) -> np.ndarray:
    degree = _maximum_row_nnz(matrix_abs)
    local_gamma = _gamma(2 * degree + 4)
    absolute_product = matrix_abs @ np.abs(values)
    return np.nextafter(
        local_gamma * np.linalg.norm(absolute_product, axis=0),
        math.inf,
    )


def _poisson_tail_upper(mu: float, last_term: int) -> float:
    if mu == 0.0:
        return 0.0
    if last_term + 2 <= mu:
        return math.inf
    mpmath.mp.dps = 80
    mp_mu = mpmath.mpf(float(mu))
    first = (
        mpmath.exp(-mp_mu)
        * mp_mu ** (last_term + 1)
        / mpmath.factorial(last_term + 1)
    )
    ratio = mp_mu / (last_term + 2)
    return _up(first / (1 - ratio))


def _poisson_data(mu: float) -> tuple[np.ndarray, float]:
    last_term = max(0, int(math.ceil(mu)))
    while _poisson_tail_upper(mu, last_term) > POISSON_TAIL_TARGET:
        last_term += 1
    weights = np.empty(last_term + 1)
    weights[0] = math.exp(-mu)
    for index in range(1, last_term + 1):
        weights[index] = weights[index - 1] * mu / index
    return weights, _poisson_tail_upper(mu, last_term)


def _uniformization_step(
    poisson_matrix,
    poisson_matrix_abs,
    state: np.ndarray,
    state_error: np.ndarray,
    duration: float,
    rate: float,
    decay_lower: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    weights, tail = _poisson_data(rate * duration)
    power = state.copy()
    power_error = np.zeros(state.shape[1])
    result = weights[0] * power
    result_error = _gamma(8) * weights[0] * np.linalg.norm(power, axis=0)
    absolute_term_norm_sum = weights[0] * np.linalg.norm(power, axis=0)

    for index in range(1, len(weights)):
        multiplication_error = _matmul_roundoff(
            poisson_matrix_abs, power
        )
        power = poisson_matrix @ power
        power_error = np.nextafter(
            power_error + multiplication_error, math.inf
        )
        coefficient_error = _gamma(4 * index + 8) * weights[index]
        power_norm = np.linalg.norm(power, axis=0)
        result += weights[index] * power
        result_error += (
            weights[index] * power_error
            + coefficient_error * power_norm
        )
        absolute_term_norm_sum += weights[index] * power_norm

    result_error += _gamma(len(weights) + 2) * absolute_term_norm_sum
    input_norm = np.linalg.norm(state, axis=0)
    next_error = (
        math.exp(-decay_lower * duration) * state_error
        + tail * input_norm
        + result_error
    )
    return (
        result,
        np.nextafter(next_error, math.inf),
        {
            "poisson_mean": rate * duration,
            "poisson_terms": len(weights),
            "poisson_tail_upper": tail,
            "maximum_arithmetic_error": float(np.max(result_error)),
        },
    )


def _axial_l2_upper(start: float, end: float, half_height: float) -> float:
    if start <= 0.0:
        return math.inf
    mpmath.mp.dps = 80
    a = mpmath.mpf(float(start))
    b = mpmath.mpf(float(end))
    variance = mpmath.expm1(2 * a)
    upper = mpmath.exp(b) * mpmath.sqrt(
        mpmath.erf(half_height / mpmath.sqrt(variance))
        / (2 * mpmath.sqrt(mpmath.pi) * mpmath.sqrt(variance))
    )
    return _up(upper)


def _axial_scalar_upper(
    start: float, end: float, half_height: float
) -> float:
    mpmath.mp.dps = 80
    b = mpmath.mpf(float(end))
    if start <= 0.0:
        return _up(mpmath.exp(b))
    a = mpmath.mpf(float(start))
    variance = mpmath.expm1(2 * a)
    upper = mpmath.exp(b) * mpmath.erf(
        half_height / mpmath.sqrt(2 * variance)
    )
    return _up(upper)


def _operator_data(grid: dict[str, object], spectral: dict[str, object]):
    mass = np.asarray(grid["state_mass"])
    root_mass = np.sqrt(mass)
    raw_symmetric = spectral["symmetric_generator"].tocsr()
    symmetric_generator = (
        0.5 * (raw_symmetric + raw_symmetric.transpose())
    ).tocsr()
    positive_operator = -symmetric_generator
    absolute_positive = abs(positive_operator).tocsr()
    gershgorin_rows = np.asarray(absolute_positive.sum(axis=1)).reshape(-1)
    largest_decay_upper = _up(float(np.max(gershgorin_rows)))
    uniformization_rate = _up(
        max(
            float(np.max(-symmetric_generator.diagonal())),
            0.5 * largest_decay_upper,
        )
        * (1.0 + 32.0 * np.finfo(float).eps)
    )
    poisson_matrix = (
        eye(symmetric_generator.shape[0], format="csr")
        + symmetric_generator / uniformization_rate
    ).tocsr()
    poisson_matrix_abs = abs(poisson_matrix).tocsr()

    inner_arcs = np.asarray(grid["inner_dual_arcs"])
    boundary_operator = (
        diags(1.0 / np.sqrt(inner_arcs))
        @ grid["inner_rate_matrix"].transpose()
        @ diags(root_mass)
    ).tocsr()
    boundary_operator_abs = abs(boundary_operator).tocsr()
    return {
        "root_mass": root_mass,
        "symmetric_generator": symmetric_generator,
        "symmetric_generator_abs": abs(symmetric_generator).tocsr(),
        "largest_decay_gershgorin_upper": largest_decay_upper,
        "uniformization_rate": uniformization_rate,
        "poisson_matrix": poisson_matrix,
        "poisson_matrix_abs": poisson_matrix_abs,
        "boundary_operator": boundary_operator,
        "boundary_operator_abs": boundary_operator_abs,
        "minimum_poisson_entry": float(np.min(poisson_matrix.data)),
        "maximum_poisson_contraction_endpoint": max(
            abs(
                1.0
                - spectral["principal_decay_barta_lower"]
                / uniformization_rate
            ),
            abs(1.0 - largest_decay_upper / uniformization_rate),
        ),
    }


def _output_norm_upper(
    operator_data: dict[str, object],
    state: np.ndarray,
    state_error: np.ndarray,
    duration: float,
    boundary_norm_upper: float,
) -> np.ndarray:
    generator = operator_data["symmetric_generator"]
    generator_abs = operator_data["symmetric_generator_abs"]
    boundary = operator_data["boundary_operator"]
    boundary_abs = operator_data["boundary_operator_abs"]
    largest_decay = operator_data["largest_decay_gershgorin_upper"]

    derivative = state
    derivative_error = np.zeros(state.shape[1])
    powers = []
    for order in range(TAYLOR_ORDER):
        output = boundary @ derivative
        output_error = (
            boundary_norm_upper * derivative_error
            + _matmul_roundoff(boundary_abs, derivative)
        )
        output_norm = np.linalg.norm(output, axis=0) + output_error
        powers.append(duration**order / math.factorial(order) * output_norm)
        multiplication_error = _matmul_roundoff(
            generator_abs, derivative
        )
        derivative = generator @ derivative
        derivative_error = (
            largest_decay * derivative_error + multiplication_error
        )

    remainder = (
        duration**TAYLOR_ORDER
        / math.factorial(TAYLOR_ORDER)
        * boundary_norm_upper
        * (np.linalg.norm(derivative, axis=0) + derivative_error)
    )
    initial_error = boundary_norm_upper * state_error
    return np.nextafter(
        np.sum(np.asarray(powers), axis=0) + remainder + initial_error,
        math.inf,
    )


def _early_output_upper(
    operator_data: dict[str, object],
    initial_state: np.ndarray,
    boundary_norm_upper: float,
    half_height: float,
) -> tuple[np.ndarray, dict[str, object]]:
    rate = operator_data["uniformization_rate"]
    poisson_matrix = operator_data["poisson_matrix"]
    poisson_matrix_abs = operator_data["poisson_matrix_abs"]
    boundary = operator_data["boundary_operator"]
    boundary_abs = operator_data["boundary_operator_abs"]
    weights_at_end, tail_at_end = _poisson_data(rate * EARLY_TIME)

    powers = [initial_state.copy()]
    power_errors = [np.zeros(initial_state.shape[1])]
    boundary_powers = []
    boundary_power_errors = []
    for index in range(len(weights_at_end)):
        power = powers[index]
        power_error = power_errors[index]
        boundary_power = boundary @ power
        boundary_powers.append(boundary_power)
        boundary_power_errors.append(
            boundary_norm_upper * power_error
            + _matmul_roundoff(boundary_abs, power)
        )
        if index + 1 == len(weights_at_end):
            break
        multiplication_error = _matmul_roundoff(poisson_matrix_abs, power)
        powers.append(poisson_matrix @ power)
        power_errors.append(power_error + multiplication_error)

    graph_distances = []
    for column in range(initial_state.shape[1]):
        nonzero_orders = [
            order
            for order, output in enumerate(boundary_powers)
            if np.any(output[:, column] != 0.0)
        ]
        if not nonzero_orders:
            raise RuntimeError("early Poisson expansion did not reach boundary")
        graph_distances.append(min(nonzero_orders))
    minimum_distance = min(graph_distances)
    if rate * EARLY_TIME >= minimum_distance:
        raise RuntimeError("early interval exceeds graph-distance monotonicity")

    column_norms = np.linalg.norm(initial_state, axis=0)
    raw_upper = np.zeros(initial_state.shape[1])
    first_end = EARLY_SUBSTEP
    mpmath.mp.dps = 80
    prefactor = mpmath.exp(first_end) / mpmath.sqrt(
        2 * mpmath.sqrt(mpmath.pi) * mpmath.sqrt(2)
    )
    first_terms = np.zeros(initial_state.shape[1])
    for order in range(minimum_distance, len(boundary_powers)):
        coefficient = (
            mpmath.mpf(rate) ** order
            * mpmath.mpf(first_end) ** (order - mpmath.mpf("0.25"))
            / mpmath.factorial(order)
        )
        first_terms += _up(prefactor * coefficient) * (
            np.linalg.norm(boundary_powers[order], axis=0)
            + boundary_power_errors[order]
        )
    tail_without_exponential = (
        mpmath.mpf(rate * first_end) ** len(boundary_powers)
        / mpmath.factorial(len(boundary_powers))
        / (1 - mpmath.mpf(rate * first_end) / (len(boundary_powers) + 1))
    )
    first_terms += _up(
        prefactor
        * mpmath.mpf(first_end) ** (-mpmath.mpf("0.25"))
        * tail_without_exponential
    ) * boundary_norm_upper * column_norms
    raw_upper = np.maximum(raw_upper, first_terms)

    maximum_direct_roundoff = 0.0
    subinterval_count = int(round(EARLY_TIME / EARLY_SUBSTEP))
    for subinterval in range(1, subinterval_count):
        start = subinterval * EARLY_SUBSTEP
        end = (subinterval + 1) * EARLY_SUBSTEP
        weights, tail = _poisson_data(rate * end)
        output = np.zeros_like(boundary_powers[0])
        output_error = np.zeros(initial_state.shape[1])
        absolute_norm_sum = np.zeros(initial_state.shape[1])
        for order, weight in enumerate(weights):
            output += weight * boundary_powers[order]
            power_norm = np.linalg.norm(boundary_powers[order], axis=0)
            output_error += weight * boundary_power_errors[order]
            output_error += _gamma(4 * order + 8) * weight * power_norm
            absolute_norm_sum += weight * power_norm
        output_error += _gamma(len(weights) + 2) * absolute_norm_sum
        maximum_direct_roundoff = max(
            maximum_direct_roundoff, float(np.max(output_error))
        )
        output_upper = (
            np.linalg.norm(output, axis=0)
            + output_error
            + boundary_norm_upper * tail * column_norms
        )
        raw_upper = np.maximum(
            raw_upper,
            _axial_l2_upper(start, end, half_height) * output_upper,
        )

    return raw_upper, {
        "minimum_graph_distance": minimum_distance,
        "maximum_graph_distance": max(graph_distances),
        "poisson_mean_at_early_end": rate * EARLY_TIME,
        "poisson_terms_at_early_end": len(weights_at_end),
        "poisson_tail_at_early_end": tail_at_end,
        "maximum_direct_roundoff": maximum_direct_roundoff,
        "subinterval_count": subinterval_count,
    }


def _time_slabs() -> list[tuple[float, float]]:
    slabs = [(0.0, EARLY_TIME)]
    current = EARLY_TIME
    for end, step in (
        (0.05, 0.0005),
        (0.2, 0.0025),
        (1.0, 0.005),
        (3.0, 0.0125),
        (TERMINAL_TIME, 0.025),
    ):
        count = int(round((end - current) / step))
        if abs(current + count * step - end) > 1.0e-12:
            raise RuntimeError("time schedule is not integral")
        for _ in range(count):
            next_time = current + step
            slabs.append((current, next_time))
            current = next_time
    if abs(current - TERMINAL_TIME) > 1.0e-12:
        raise RuntimeError("time schedule does not end at T")
    return slabs


def _potential_data(
    operator_data: dict[str, object],
    grid: dict[str, object],
    decay_lower: float,
) -> dict[str, object]:
    generator = operator_data["symmetric_generator"]
    generator_abs = operator_data["symmetric_generator_abs"]
    root_mass = operator_data["root_mass"]
    inner_rates = np.asarray(grid["inner_rates"])
    scalar_output = root_mass * inner_rates
    potential = spsolve((-generator).tocsc(), scalar_output)
    residual = scalar_output + generator @ potential
    residual_roundoff = float(
        _matmul_roundoff(generator_abs, potential[:, None])[0]
    )
    residual_upper = _up(
        float(np.linalg.norm(residual)) + residual_roundoff
    )
    potential_error = _up(residual_upper / decay_lower)
    return {
        "scalar_output": scalar_output,
        "potential": potential,
        "potential_residual_upper": residual_upper,
        "potential_error": potential_error,
        "potential_norm_upper": _up(
            float(np.linalg.norm(potential)) + potential_error
        ),
        "minimum_potential_entry": float(np.min(potential)),
    }


def _absorption_mass_upper(
    potential_data: dict[str, object],
    start_state: np.ndarray,
    end_state: np.ndarray,
    start_error: np.ndarray,
    end_error: np.ndarray,
) -> np.ndarray:
    potential = potential_data["potential"]
    potential_error = potential_data["potential_error"]
    potential_norm = potential_data["potential_norm_upper"]
    difference = start_state - end_state
    central = potential @ difference
    dot_roundoff = _gamma(2 * len(potential) + 4) * (
        np.abs(potential) @ (np.abs(start_state) + np.abs(end_state))
    )
    uncertainty = (
        potential_norm * (start_error + end_error)
        + potential_error
        * (
            np.linalg.norm(start_state, axis=0)
            + np.linalg.norm(end_state, axis=0)
            + start_error
            + end_error
        )
        + dot_roundoff
    )
    return np.nextafter(np.maximum(0.0, central + uncertainty), math.inf)


def _finite_time_row(
    grid: dict[str, object],
    spectral: dict[str, object],
    return_density,
) -> dict[str, object]:
    operator_data = _operator_data(grid, spectral)
    boundary_norm = spectral["boundary_operator_norm_gershgorin_upper"]
    decay_lower = spectral["principal_decay_barta_lower"]
    entry_states = np.asarray(grid["entry_states"])
    root_mass = operator_data["root_mass"]
    state = np.zeros((len(root_mass), len(entry_states)))
    state[entry_states, np.arange(len(entry_states))] = 1.0 / root_mass[
        entry_states
    ]
    state_error = np.zeros(len(entry_states))
    initial_state = state.copy()
    initial_norms = np.linalg.norm(initial_state, axis=0)

    window_count = int(round(TERMINAL_TIME / CERTIFIED_WINDOW))
    if abs(window_count * CERTIFIED_WINDOW - TERMINAL_TIME) > 1.0e-12:
        raise RuntimeError("certified window must divide terminal time")
    window_upper = np.zeros((window_count, len(entry_states)))
    sampled_window_maximum = np.zeros_like(window_upper)
    early_upper, early_data = _early_output_upper(
        operator_data,
        initial_state,
        boundary_norm,
        return_density.PATCH_HALF_HEIGHT,
    )
    window_upper[0] = early_upper

    potential_data = _potential_data(operator_data, grid, decay_lower)
    scalar_finite_upper = np.zeros(len(entry_states))
    total_absorption_upper = np.zeros(len(entry_states))
    maximum_state_error = 0.0
    maximum_poisson_tail = 0.0
    maximum_step_arithmetic_error = 0.0
    maximum_taylor_to_sample_ratio = 1.0
    slab_count = 0

    for start, end in _time_slabs():
        duration = end - start
        start_state = state
        start_error = state_error
        state, state_error, step_data = _uniformization_step(
            operator_data["poisson_matrix"],
            operator_data["poisson_matrix_abs"],
            start_state,
            start_error,
            duration,
            operator_data["uniformization_rate"],
            decay_lower,
        )
        slab_count += 1
        maximum_state_error = max(
            maximum_state_error, float(np.max(state_error))
        )
        maximum_poisson_tail = max(
            maximum_poisson_tail, step_data["poisson_tail_upper"]
        )
        maximum_step_arithmetic_error = max(
            maximum_step_arithmetic_error,
            step_data["maximum_arithmetic_error"],
        )

        absorption_upper = _absorption_mass_upper(
            potential_data,
            start_state,
            state,
            start_error,
            state_error,
        )
        scalar_weight = _axial_scalar_upper(
            start, end, return_density.PATCH_HALF_HEIGHT
        )
        scalar_finite_upper += scalar_weight * absorption_upper
        total_absorption_upper += absorption_upper

        if start < EARLY_TIME - 1.0e-14:
            continue
        output_upper = _output_norm_upper(
            operator_data,
            start_state,
            start_error,
            duration,
            boundary_norm,
        )
        raw_upper = (
            _axial_l2_upper(
                start, end, return_density.PATCH_HALF_HEIGHT
            )
            * output_upper
        )
        window_index = min(
            int(math.floor((start + 1.0e-12) / CERTIFIED_WINDOW)),
            window_count - 1,
        )
        if end > (window_index + 1) * CERTIFIED_WINDOW + 1.0e-11:
            raise RuntimeError("a time slab crosses a certified window")
        window_upper[window_index] = np.maximum(
            window_upper[window_index], raw_upper
        )

        boundary_value = operator_data["boundary_operator"] @ start_state
        boundary_value_error = _matmul_roundoff(
            operator_data["boundary_operator_abs"], start_state
        )
        _, axial_l2 = return_density._axial_factors(start, grid["rho"])
        sample = math.exp(start) * axial_l2 * (
            np.linalg.norm(boundary_value, axis=0)
            + boundary_value_error
            + boundary_norm * start_error
        )
        sampled_window_maximum[window_index] = np.maximum(
            sampled_window_maximum[window_index], sample
        )
        positive_sample = np.maximum(sample, 1.0e-300)
        maximum_taylor_to_sample_ratio = max(
            maximum_taylor_to_sample_ratio,
            float(np.max(raw_upper / positive_sample)),
        )

    terminal_norm_upper = np.linalg.norm(state, axis=0) + state_error
    axial_tail_norm = (
        math.sqrt(return_density.PATCH_HALF_HEIGHT / math.pi)
        / math.sqrt(1.0 - math.exp(-2.0 * TERMINAL_TIME))
    )
    terminal_amplitude = axial_tail_norm * boundary_norm * terminal_norm_upper
    interval_tail_sum = terminal_amplitude / (
        1.0 - math.exp(-decay_lower * CERTIFIED_WINDOW)
    )
    finite_interval_sum = np.sum(window_upper, axis=0)
    interval_factor = (
        CERTIFIED_WINDOW + 1.0 / return_density.FORM_FLOOR
    ) * (finite_interval_sum + interval_tail_sum)

    scalar_tail = (
        math.sqrt(2.0 / math.pi)
        * return_density.PATCH_HALF_HEIGHT
        / math.sqrt(1.0 - math.exp(-2.0 * TERMINAL_TIME))
        * math.sqrt(2.0 * math.pi)
        * boundary_norm
        / decay_lower
        * terminal_norm_upper
    )
    scalar_gain = scalar_finite_upper + scalar_tail
    response = np.sqrt(
        scalar_gain
        * return_density.TRACE_L4_FORM_CONSTANT
        * interval_factor
    )
    sampled_finite_sum = np.sum(sampled_window_maximum, axis=0)

    angle_rows = []
    for index, angle in enumerate(grid["entry_angles"]):
        angle_rows.append(
            {
                "angle": float(angle),
                "certified_finite_interval_sum": float(
                    finite_interval_sum[index]
                ),
                "sampled_finite_interval_sum": float(
                    sampled_finite_sum[index]
                ),
                "certified_interval_factor_with_tail": float(
                    interval_factor[index]
                ),
                "interval_spectral_tail_contribution": float(
                    (
                        CERTIFIED_WINDOW
                        + 1.0 / return_density.FORM_FLOOR
                    )
                    * interval_tail_sum[index]
                ),
                "certified_scalar_gain_with_tail": float(scalar_gain[index]),
                "scalar_darboux_finite_upper": float(
                    scalar_finite_upper[index]
                ),
                "scalar_spectral_tail_bound": float(scalar_tail[index]),
                "certified_response": float(response[index]),
            }
        )
    worst = max(angle_rows, key=lambda item: item["certified_response"])
    return {
        "certified_window": CERTIFIED_WINDOW,
        "terminal_time": TERMINAL_TIME,
        "time_slab_count": slab_count,
        "uniformization_rate": operator_data["uniformization_rate"],
        "largest_decay_gershgorin_upper": operator_data[
            "largest_decay_gershgorin_upper"
        ],
        "minimum_poisson_entry": operator_data["minimum_poisson_entry"],
        "maximum_poisson_contraction_endpoint": operator_data[
            "maximum_poisson_contraction_endpoint"
        ],
        "maximum_state_l2_error_bound": maximum_state_error,
        "maximum_local_poisson_tail": maximum_poisson_tail,
        "maximum_step_arithmetic_error": maximum_step_arithmetic_error,
        "maximum_taylor_to_left_sample_ratio": (
            maximum_taylor_to_sample_ratio
        ),
        "potential_residual_upper": potential_data[
            "potential_residual_upper"
        ],
        "potential_error_bound": potential_data["potential_error"],
        "minimum_potential_entry": potential_data[
            "minimum_potential_entry"
        ],
        "maximum_total_absorption_slab_upper": float(
            np.max(total_absorption_upper)
        ),
        "maximum_terminal_norm_upper": float(
            np.max(terminal_norm_upper)
        ),
        "maximum_finite_interval_enclosure_ratio": float(
            np.max(
                finite_interval_sum
                / np.maximum(sampled_finite_sum, 1.0e-300)
            )
        ),
        "maximum_interval_tail_contribution": max(
            item["interval_spectral_tail_contribution"]
            for item in angle_rows
        ),
        "maximum_scalar_tail_bound": float(np.max(scalar_tail)),
        "maximum_certified_response": worst["certified_response"],
        "worst_entry_angle": worst["angle"],
        "early_interval": early_data,
        "angle_rows": angle_rows,
        "initial_state_norm_range": [
            float(np.min(initial_norms)),
            float(np.max(initial_norms)),
        ],
    }


def _width_row(
    fem,
    spectral_tail,
    return_density,
    spacing: float,
    x_half_width: float,
    run_density: bool,
) -> dict[str, object]:
    grid = fem._build_mesh(spacing, x_half_width=x_half_width)
    structural = fem._structural_row(grid)
    spectral = spectral_tail._spectral_row(grid)
    row: dict[str, object] = {
        "spacing": spacing,
        "x_half_width": x_half_width,
        "structural": structural,
        "spectral": {
            key: value
            for key, value in spectral.items()
            if key != "symmetric_generator"
        },
    }
    if run_density:
        row["finite_time_certificate"] = _finite_time_row(
            grid, spectral, return_density
        )
    return row


def audit(
    spacing: float = 0.12,
    x_half_widths: tuple[float, ...] = (4.2, 5.25, 6.3),
    run_density: bool = True,
) -> dict[str, object]:
    fem = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "reversible_fem_for_finite_time_certificate",
    )
    spectral_tail = _load_module(
        "neutral_strip_reversible_spectral_tail_width_audit.py",
        "spectral_tail_for_finite_time_certificate",
    )
    return_density = _load_module(
        "neutral_strip_return_density_pilot.py",
        "return_density_for_finite_time_certificate",
    )
    width_rows = [
        _width_row(
            fem,
            spectral_tail,
            return_density,
            spacing,
            x_half_width,
            run_density,
        )
        for x_half_width in x_half_widths
    ]
    positive_uniformizations = all(
        row.get("finite_time_certificate", {}).get(
            "minimum_poisson_entry", 0.0
        )
        >= -1.0e-15
        and row.get("finite_time_certificate", {}).get(
            "maximum_poisson_contraction_endpoint", 1.0
        )
        <= 1.0 + 1.0e-12
        for row in width_rows
    )
    finite_time_certified = bool(
        run_density
        and positive_uniformizations
        and all(
            row["finite_time_certificate"][
                "maximum_local_poisson_tail"
            ]
            < 1.0e-16
            for row in width_rows
        )
    )
    scalar_certified = bool(
        finite_time_certified
        and all(
            row["finite_time_certificate"]["minimum_potential_entry"]
            > -1.0e-10
            for row in width_rows
        )
    )
    responses = (
        [
            row["finite_time_certificate"]["maximum_certified_response"]
            for row in width_rows
        ]
        if run_density
        else []
    )
    result = {
        "model": (
            "rho=0 symmetrized reversible boundary-FEM finite-time "
            "semigroup certificate"
        ),
        "spacing": spacing,
        "x_half_widths": list(x_half_widths),
        "width_rows": width_rows,
        "positive_contractive_uniformization_verified": (
            positive_uniformizations if run_density else False
        ),
        "finite_time_window_maxima_certified": finite_time_certified,
        "scalar_time_quadrature_certified": scalar_certified,
        "spectral_tail_reused": bool(run_density),
        "maximum_certified_response": (
            max(responses) if responses else None
        ),
        "maximum_width_response_spread": (
            max(responses) - min(responses) if responses else None
        ),
        "x_truncation_analytically_removed": False,
        "continuum_return_response_certified": False,
        "rho_uniformity_certified": False,
        "scope_guard": (
            "The maxima, scalar Darboux sum, propagation remainder, and "
            "post-T tail are upper-enclosed for each symmetrized stored "
            "floating FEM matrix. This does not remove the artificial x "
            "boundary, enclose FEM continuum consistency error, or extend "
            "the result away from rho=0."
        ),
        "next_gate": (
            "derive an analytic x-exit correction before attempting "
            "polygonal-circle and weighted FEM continuum consistency"
        ),
    }
    checks = (
        not result["x_truncation_analytically_removed"],
        not result["continuum_return_response_certified"],
        not result["rho_uniformity_certified"],
    )
    if run_density:
        checks += (
            result["positive_contractive_uniformization_verified"],
            result["finite_time_window_maxima_certified"],
            result["scalar_time_quadrature_certified"],
            result["spectral_tail_reused"],
        )
    result["all_finite_time_certificate_checks_pass"] = bool(all(checks))
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
