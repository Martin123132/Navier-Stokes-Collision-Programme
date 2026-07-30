"""Mode pilot for the Brownian exterior-cylinder return density."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln, k0e, kve


INNER_RADIUS = 1.0
START_RADIUS = 2.0
AXIAL_HALF_HEIGHT = 0.75
FORM_FLOOR = 4.832287335665
SOBOLEV_CONSTANT = 4.0 ** (2.0 / 3.0) / (
    3.0 * math.pi ** (4.0 / 3.0)
)
POINCARE_FACTOR = (FORM_FLOOR + 1.0) / FORM_FLOOR
TRACE_L4_FORM_CONSTANT = (
    2.0 * SOBOLEV_CONSTANT**0.75 * POINCARE_FACTOR
)


def _load_barrier_module():
    script = Path(__file__).with_name(
        "radial_h1_payoff_supersolution_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "radial_h1_for_cylinder_return", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stehfest_weights(order: int) -> np.ndarray:
    if order % 2:
        raise ValueError("Stehfest order must be even")
    half_order = order // 2
    weights = []
    for index in range(1, order + 1):
        total = 0.0
        for summation_index in range(
            (index + 1) // 2,
            min(index, half_order) + 1,
        ):
            log_term = (
                half_order * math.log(summation_index)
                + gammaln(2 * summation_index + 1)
                - gammaln(half_order - summation_index + 1)
                - gammaln(summation_index + 1)
                - gammaln(summation_index)
                - gammaln(index - summation_index + 1)
                - gammaln(2 * summation_index - index + 1)
            )
            total += math.exp(log_term)
        weights.append((-1) ** (index + half_order) * total)
    return np.asarray(weights)


def _hitting_modes(
    time: float,
    inversion_order: int,
    maximum_mode: int,
) -> np.ndarray:
    weights = _stehfest_weights(inversion_order)
    laplace_points = (
        np.arange(1, inversion_order + 1, dtype=float)
        * math.log(2.0)
        / time
    )
    roots = np.sqrt(laplace_points)
    modes = []
    for mode in range(maximum_mode + 1):
        # kve removes the common large-argument exponential from K_n.
        transform = (
            kve(mode, START_RADIUS * roots)
            / kve(mode, INNER_RADIUS * roots)
            * np.exp(-(START_RADIUS - INNER_RADIUS) * roots)
        )
        modes.append(
            math.log(2.0) / time * float(np.dot(weights, transform))
        )
    return np.asarray(modes)


def _cylinder_spatial_l2(
    time: float,
    inversion_order: int,
    maximum_mode: int = 40,
) -> float:
    modes = _hitting_modes(time, inversion_order, maximum_mode)
    angular_l2 = math.sqrt(
        (modes[0] ** 2 + 2.0 * float(np.sum(modes[1:] ** 2)))
        / (2.0 * math.pi)
    )
    axial_gaussian_l2 = (
        2.0 ** (-0.75) * (math.pi * time) ** (-0.25)
    )
    return angular_l2 * axial_gaussian_l2


def _half_space_spatial_l2(time: float) -> float:
    if time <= 0.0:
        return 0.0
    distance = START_RADIUS - INNER_RADIUS
    return (
        distance
        / (4.0 * math.sqrt(2.0) * math.pi)
        * time ** (-2.0)
        * math.exp(-(distance**2) / (4.0 * time))
    )


def _axial_patch_return_probability(
    cutoff: float = 40.0,
) -> dict[str, float]:
    """Pilot the probability of first hitting r=1 inside |z|<3/4."""

    def bessel_ratio(frequency: float) -> float:
        if frequency == 0.0:
            return 1.0
        return float(
            k0e(START_RADIUS * frequency)
            / k0e(INNER_RADIUS * frequency)
            * math.exp(
                -(START_RADIUS - INNER_RADIUS) * frequency
            )
        )

    def integrand(frequency: float) -> float:
        if frequency == 0.0:
            return AXIAL_HALF_HEIGHT
        return (
            math.sin(AXIAL_HALF_HEIGHT * frequency)
            / frequency
            * bessel_ratio(frequency)
        )

    def truncated_integral(upper: float) -> tuple[float, float]:
        breakpoints = (0.0, 1.0, 5.0, 10.0, 20.0, upper)
        value = 0.0
        error = 0.0
        for lower, interval_upper in zip(
            breakpoints[:-1], breakpoints[1:]
        ):
            if interval_upper <= lower:
                continue
            piece, piece_error = quad(
                integrand,
                lower,
                interval_upper,
                epsabs=1.0e-13,
                epsrel=1.0e-13,
                limit=500,
            )
            value += piece
            error += piece_error
        scale = 2.0 / math.pi
        return scale * value, scale * error

    probability, quadrature_error = truncated_integral(cutoff)
    shorter_probability, _ = truncated_integral(cutoff / 2.0)
    return {
        "frequency_cutoff": cutoff,
        "probability": probability,
        "reported_quadrature_error": quadrature_error,
        "cutoff_halving_change": abs(
            probability - shorter_probability
        ),
    }


def _pilot_envelope() -> dict[str, object]:
    times = np.geomspace(0.02, 10_000.0, 241)
    orders = (12, 14, 16)
    rows = {
        order: np.asarray(
            [_cylinder_spatial_l2(time, order) for time in times]
        )
        for order in orders
    }
    raw_pilot = np.maximum(rows[14], rows[16])
    inflated_mode_pilot = 1.03 * raw_pilot
    half_space_stress = np.asarray(
        [1.5 * _half_space_spatial_l2(time) for time in times]
    )
    envelope = np.maximum(inflated_mode_pilot, half_space_stress)
    peak_index = int(np.argmax(envelope))
    raw_peak_index = int(np.argmax(raw_pilot))
    relative_spread = np.abs(rows[16] - rows[14]) / np.maximum(
        np.maximum(np.abs(rows[16]), np.abs(rows[14])), 1.0e-15
    )
    stable_mask = times >= 0.03
    tail_mask = times >= 100.0
    tail_coefficients = (
        envelope[tail_mask]
        * times[tail_mask] ** 1.25
        * np.log(times[tail_mask]) ** 2
    )
    tail_coefficient = 1.5 * float(np.max(tail_coefficients))
    return {
        "times": times,
        "order_rows": rows,
        "envelope": envelope,
        "raw_peak_time": float(times[raw_peak_index]),
        "raw_peak": float(raw_pilot[raw_peak_index]),
        "stress_envelope_peak_time": float(times[peak_index]),
        "stress_envelope_peak": float(envelope[peak_index]),
        "maximum_order_14_16_relative_spread_t_ge_0p03": float(
            np.max(relative_spread[stable_mask])
        ),
        "tail_coefficient_range_t_ge_100": [
            float(np.min(tail_coefficients)),
            float(np.max(tail_coefficients)),
        ],
        "stress_tail_coefficient": tail_coefficient,
    }


def _envelope_function(pilot: dict[str, object]):
    times = pilot["times"]
    envelope = pilot["envelope"]
    log_times = np.log(times)
    log_envelope = np.log(np.maximum(envelope, 1.0e-300))
    peak_time = float(pilot["stress_envelope_peak_time"])
    tail_coefficient = float(pilot["stress_tail_coefficient"])

    def value(time: float) -> float:
        if time <= 0.0:
            return 0.0
        if time < float(times[0]):
            return 2.0 * _half_space_spatial_l2(time)
        if time <= float(times[-1]):
            return math.exp(
                float(np.interp(math.log(time), log_times, log_envelope))
            )
        return (
            tail_coefficient
            * time ** (-1.25)
            / math.log(time) ** 2
        )

    return value, peak_time


def _interval_sum(window: float, pilot: dict[str, object]) -> float:
    envelope, peak_time = _envelope_function(pilot)
    maximum_tabulated_time = float(pilot["times"][-1])
    term_count = int(math.ceil(maximum_tabulated_time / window)) + 1
    indices = np.arange(term_count, dtype=float)
    lower = indices * window
    upper = (indices + 1.0) * window
    sample = np.where(
        upper < peak_time,
        upper,
        np.where(lower <= peak_time, peak_time, lower),
    )
    explicit_sum = float(
        np.sum([envelope(float(time)) for time in sample])
    )
    tail_start = term_count * window
    tail_upper = envelope(tail_start) + quad(
        envelope,
        tail_start,
        math.inf,
        epsabs=1.0e-10,
        epsrel=1.0e-8,
        limit=500,
    )[0] / window
    return explicit_sum + tail_upper


def _time_factor(alpha: float, pilot: dict[str, object]) -> dict[str, float]:
    coercivity = 1.0 - alpha

    def objective(log_window: float) -> float:
        window = math.exp(log_window)
        energy = (
            window / coercivity**2
            + 1.0 / (coercivity**3 * FORM_FLOOR)
        )
        return energy * _interval_sum(window, pilot)

    optimum = minimize_scalar(
        objective,
        bounds=(math.log(0.03), math.log(3.0)),
        method="bounded",
        options={"xatol": 1.0e-5},
    )
    return {
        "optimal_window": math.exp(float(optimum.x)),
        "factor": float(optimum.fun),
    }


def _thresholds(
    pilot: dict[str, object], constants: dict[str, float]
) -> dict[str, float]:
    allowance = constants["additive_gain_allowance"]

    def potential_overshoot(mass: float) -> float:
        alpha = constants["potential_relative_form"] * mass
        forcing = constants["potential_forcing"] * mass
        return forcing * math.sqrt(
            TRACE_L4_FORM_CONSTANT
            * _time_factor(alpha, pilot)["factor"]
        )

    potential_threshold = brentq(
        lambda mass: potential_overshoot(mass) - allowance,
        0.0,
        0.999 / constants["potential_relative_form"],
    )
    zero_alpha_factor = _time_factor(0.0, pilot)["factor"]
    drift_threshold = allowance / (
        constants["drift_forcing"]
        * math.sqrt(TRACE_L4_FORM_CONSTANT * zero_alpha_factor)
    )
    return {
        "potential_L3_over_2_threshold": potential_threshold,
        "potential_alpha_at_threshold": (
            constants["potential_relative_form"] * potential_threshold
        ),
        "drift_L3_threshold": drift_threshold,
    }


def audit() -> dict[str, object]:
    barrier_module = _load_barrier_module()
    barrier = barrier_module.audit()
    forcing = barrier["global_energy_forcing_coefficients"]
    constants = {
        "additive_gain_allowance": barrier["additive_gain_allowance"],
        "potential_forcing": forcing["potential_L3_over_2"],
        "drift_forcing": forcing["drift_L3"],
        "potential_relative_form": forcing[
            "potential_relative_form"
        ],
    }
    pilot = _pilot_envelope()
    patch_return = _axial_patch_return_probability()
    time_factor = _time_factor(0.0, pilot)
    thresholds = _thresholds(pilot, constants)
    result: dict[str, object] = {
        "exact_mode_transform": (
            "Laplace h_n(lambda)=K_n(2sqrt(lambda))/"
            "K_n(sqrt(lambda))"
        ),
        "exact_kernel_factorization": (
            "k(t,theta,z)=h_disk(t,theta)*(4pi t)^(-1/2)"
            "exp(-z^2/(4t))"
        ),
        "angular_L2_Parseval_identity": (
            "||h||_2^2=(h_0^2+2 sum_(n>=1)h_n^2)/(2pi)"
        ),
        "maximum_angular_mode": 40,
        "Stehfest_orders": [12, 14, 16],
        "time_grid": {
            "minimum": 0.02,
            "maximum": 10_000.0,
            "point_count": 241,
        },
        "raw_pilot_peak_time": pilot["raw_peak_time"],
        "raw_pilot_peak_spatial_L2_density": pilot["raw_peak"],
        "maximum_order_14_16_relative_spread_t_ge_0p03": pilot[
            "maximum_order_14_16_relative_spread_t_ge_0p03"
        ],
        "stress_envelope_definition": (
            "max(1.03*max(order14,order16),1.5*half_space_L2); "
            "after t=10000 use 1.5 times the maximum sampled "
            "t^(5/4)log(t)^2 coefficient"
        ),
        "stress_envelope_peak_time": pilot["stress_envelope_peak_time"],
        "stress_envelope_peak": pilot["stress_envelope_peak"],
        "tail_coefficient_range_t_ge_100": pilot[
            "tail_coefficient_range_t_ge_100"
        ],
        "stress_tail_coefficient": pilot["stress_tail_coefficient"],
        "stress_time_energy_factor": time_factor,
        "stress_conditional_thresholds": thresholds,
        "finite_axial_patch_return": {
            "patch_half_height": AXIAL_HALF_HEIGHT,
            "start_point": "(r,z)=(2,0)",
            "exact_Fourier_formula": (
                "p_H=(2/pi)int_0^infinity sin(kH)/k "
                "K_0(2k)/K_0(k) dk"
            ),
            "center_is_worst_axial_start": True,
            **patch_return,
            "numerically_certified": False,
        },
        "Brownian_cylinder_L2_envelope_certified": False,
        "weighted_Navier_Stokes_cylinder_envelope_certified": False,
        "full_return_gate_closed": False,
        "scope_guard": (
            "the Bessel mode transform and Parseval factorization are exact. "
            "Stehfest inversion, mode truncation, interpolation, and the "
            "inflated tail are a convergence pilot, not an enclosure"
        ),
        "next_gate": (
            "replace the stress envelope by rigorous inverse-Laplace mode "
            "and tail bounds, then perturb it by the physical drift and "
            "deformation in a norm that preserves summability"
        ),
    }
    positive_checks = (
        result["maximum_order_14_16_relative_spread_t_ge_0p03"] < 0.017,
        0.11 < result["raw_pilot_peak_time"] < 0.15,
        0.42 < result["raw_pilot_peak_spatial_L2_density"] < 0.45,
        0.78 < time_factor["factor"] < 0.82,
        0.14 < thresholds["potential_L3_over_2_threshold"] < 0.16,
        0.03 < thresholds["drift_L3_threshold"] < 0.05,
        0.30 < patch_return["probability"] < 0.32,
        patch_return["cutoff_halving_change"] < 1.0e-10,
        not result["Brownian_cylinder_L2_envelope_certified"],
        not result["full_return_gate_closed"],
    )
    result["all_positive_cylindrical_return_pilot_checks_pass"] = all(
        positive_checks
    )
    del pilot["times"]
    del pilot["order_rows"]
    del pilot["envelope"]
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
