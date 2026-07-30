"""Audit polynomial exterior tails for the averaged entry theorem."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar


FORM_FLOOR = 4.832287335665
TRACE_FORM_CONSTANT = 1.4138826731678131
BUFFER_DISTANCE = 1.0
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
        "radial_h1_for_exterior_tail", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _half_space_supremum(time: float, distance: float = 1.0) -> float:
    if time <= 0.0:
        return 0.0
    return (
        distance
        / (4.0 * math.pi) ** 1.5
        * time ** (-2.5)
        * math.exp(-(distance**2) / (4.0 * time))
    )


def _half_space_spatial_l2(time: float, distance: float = 1.0) -> float:
    if time <= 0.0:
        return 0.0
    return (
        distance
        / (4.0 * math.sqrt(2.0) * math.pi)
        * time ** (-2.0)
        * math.exp(-(distance**2) / (4.0 * time))
    )


def _half_space_interval_sum(
    window: float,
    distance: float = 1.0,
) -> dict[str, float | int]:
    peak_time = distance**2 / 10.0
    term_count = max(2_000, int(math.ceil(200.0 / window)))
    indices = np.arange(term_count, dtype=float)
    lower = indices * window
    upper = (indices + 1.0) * window
    sample = np.where(
        upper < peak_time,
        upper,
        np.where(lower <= peak_time, peak_time, lower),
    )
    values = np.zeros_like(sample)
    positive = sample > 0.0
    values[positive] = (
        distance
        / (4.0 * math.pi) ** 1.5
        * sample[positive] ** (-2.5)
        * np.exp(-(distance**2) / (4.0 * sample[positive]))
    )
    tail_start = term_count * window
    tail_integral = quad(
        lambda time: _half_space_supremum(time, distance),
        tail_start,
        math.inf,
        epsabs=1.0e-13,
        epsrel=1.0e-11,
        limit=300,
    )[0]
    tail_upper = (
        _half_space_supremum(tail_start, distance)
        + tail_integral / window
    )
    return {
        "window": window,
        "explicit_interval_count": term_count,
        "explicit_interval_sum": float(np.sum(values)),
        "tail_upper_bound": tail_upper,
        "certified_series_upper_bound": float(np.sum(values)) + tail_upper,
    }


def _half_space_time_factor(alpha: float) -> dict[str, float | int]:
    coercivity = 1.0 - alpha
    if not 0.0 < coercivity <= 1.0:
        raise ValueError("alpha must lie in [0,1)")

    def objective(log_window: float) -> float:
        window = math.exp(log_window)
        interval_sum = _half_space_interval_sum(window)[
            "certified_series_upper_bound"
        ]
        energy_per_interval = (
            window / coercivity**2
            + 1.0 / (coercivity**3 * FORM_FLOOR)
        )
        return energy_per_interval * float(interval_sum)

    optimum = minimize_scalar(
        objective,
        bounds=(math.log(0.01), math.log(3.0)),
        method="bounded",
        options={"xatol": 2.0e-8},
    )
    window = math.exp(float(optimum.x))
    interval_data = _half_space_interval_sum(window)
    return {
        "optimal_window": window,
        "factor": float(optimum.fun),
        "explicit_interval_count": interval_data[
            "explicit_interval_count"
        ],
        "tail_upper_bound": interval_data["tail_upper_bound"],
    }


def _half_space_l2_interval_sum(
    window: float,
    distance: float = 1.0,
) -> dict[str, float | int]:
    peak_time = distance**2 / 8.0
    term_count = max(2_000, int(math.ceil(200.0 / window)))
    indices = np.arange(term_count, dtype=float)
    lower = indices * window
    upper = (indices + 1.0) * window
    sample = np.where(
        upper < peak_time,
        upper,
        np.where(lower <= peak_time, peak_time, lower),
    )
    values = np.zeros_like(sample)
    positive = sample > 0.0
    values[positive] = (
        distance
        / (4.0 * math.sqrt(2.0) * math.pi)
        * sample[positive] ** (-2.0)
        * np.exp(-(distance**2) / (4.0 * sample[positive]))
    )
    tail_start = term_count * window
    tail_integral = quad(
        lambda time: _half_space_spatial_l2(time, distance),
        tail_start,
        math.inf,
        epsabs=1.0e-13,
        epsrel=1.0e-11,
        limit=300,
    )[0]
    tail_upper = (
        _half_space_spatial_l2(tail_start, distance)
        + tail_integral / window
    )
    return {
        "window": window,
        "explicit_interval_count": term_count,
        "explicit_interval_sum": float(np.sum(values)),
        "tail_upper_bound": tail_upper,
        "certified_series_upper_bound": float(np.sum(values)) + tail_upper,
    }


def _half_space_l2_time_factor(alpha: float) -> dict[str, float | int]:
    coercivity = 1.0 - alpha
    if not 0.0 < coercivity <= 1.0:
        raise ValueError("alpha must lie in [0,1)")

    def objective(log_window: float) -> float:
        window = math.exp(log_window)
        interval_sum = _half_space_l2_interval_sum(window)[
            "certified_series_upper_bound"
        ]
        energy_per_interval = (
            window / coercivity**2
            + 1.0 / (coercivity**3 * FORM_FLOOR)
        )
        return energy_per_interval * float(interval_sum)

    optimum = minimize_scalar(
        objective,
        bounds=(math.log(0.01), math.log(3.0)),
        method="bounded",
        options={"xatol": 2.0e-8},
    )
    window = math.exp(float(optimum.x))
    interval_data = _half_space_l2_interval_sum(window)
    return {
        "optimal_window": window,
        "factor": float(optimum.fun),
        "explicit_interval_count": interval_data[
            "explicit_interval_count"
        ],
        "tail_upper_bound": interval_data["tail_upper_bound"],
    }


def _thresholds(constants: dict[str, float]) -> dict[str, float]:
    allowance = constants["additive_gain_allowance"]

    def overshoot(potential: float, drift: float) -> float:
        alpha = constants["potential_relative_form"] * potential
        if alpha >= 1.0:
            return math.inf
        time_factor = _half_space_time_factor(alpha)["factor"]
        forcing = (
            constants["potential_forcing"] * potential
            + constants["drift_forcing"] * drift
        )
        return forcing * math.sqrt(
            TRACE_FORM_CONSTANT * float(time_factor)
        )

    potential_upper = 0.999 / constants["potential_relative_form"]
    potential_threshold = brentq(
        lambda mass: overshoot(mass, 0.0) - allowance,
        0.0,
        potential_upper,
    )
    zero_alpha_factor = float(_half_space_time_factor(0.0)["factor"])
    drift_threshold = allowance / (
        constants["drift_forcing"]
        * math.sqrt(TRACE_FORM_CONSTANT * zero_alpha_factor)
    )
    return {
        "potential_L3_over_2_threshold": potential_threshold,
        "potential_alpha_at_threshold": (
            constants["potential_relative_form"] * potential_threshold
        ),
        "drift_L3_threshold": drift_threshold,
    }


def _l2_thresholds(constants: dict[str, float]) -> dict[str, float]:
    allowance = constants["additive_gain_allowance"]

    def overshoot(potential: float, drift: float) -> float:
        alpha = constants["potential_relative_form"] * potential
        if alpha >= 1.0:
            return math.inf
        time_factor = _half_space_l2_time_factor(alpha)["factor"]
        forcing = (
            constants["potential_forcing"] * potential
            + constants["drift_forcing"] * drift
        )
        return forcing * math.sqrt(
            TRACE_L4_FORM_CONSTANT * float(time_factor)
        )

    potential_upper = 0.999 / constants["potential_relative_form"]
    potential_threshold = brentq(
        lambda mass: overshoot(mass, 0.0) - allowance,
        0.0,
        potential_upper,
    )
    zero_alpha_factor = float(_half_space_l2_time_factor(0.0)["factor"])
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
    half_space_mass = quad(
        lambda time: (
            4.0
            * math.pi
            * time
            * _half_space_supremum(time, BUFFER_DISTANCE)
        ),
        0.0,
        math.inf,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=500,
    )[0]
    unit_factor = _half_space_time_factor(0.0)
    thresholds = _thresholds(constants)
    unit_l2_factor = _half_space_l2_time_factor(0.0)
    l2_thresholds = _l2_thresholds(constants)
    sphere_inner_radius = 1.0
    sphere_start_radius = 2.0
    sphere_distance = sphere_start_radius - sphere_inner_radius
    sphere_tail_coefficient = (
        sphere_inner_radius
        / sphere_start_radius
        * sphere_distance
        / (2.0 * math.sqrt(math.pi))
    )
    result: dict[str, object] = {
        "general_summable_envelope_theorem": (
            "if dnu/(ds dsigma)<=rho(s), then split time into windows "
            "I_n and use J_rho=inf_ell "
            "[ell/a^2+1/(a^3*m0)] sum_n sup_(I_n)rho"
        ),
        "exponential_envelope_required": False,
        "unbounded_exterior_exponential_envelope_viable": False,
        "sphere_hitting_time_density": (
            "f(t)=(a/R)(R-a)/(2sqrt(pi)) t^(-3/2) "
            "exp(-(R-a)^2/(4t))"
        ),
        "sphere_large_time_coefficient": sphere_tail_coefficient,
        "sphere_tail_disproves_every_positive_exponential_rate": True,
        "half_space_kernel": (
            "K(t,y)=d(4pi)^(-3/2)t^(-5/2)"
            "exp(-(d^2+|y|^2)/(4t))"
        ),
        "half_space_buffer_distance": BUFFER_DISTANCE,
        "half_space_kernel_peak_time": BUFFER_DISTANCE**2 / 10.0,
        "half_space_kernel_peak": _half_space_supremum(
            BUFFER_DISTANCE**2 / 10.0, BUFFER_DISTANCE
        ),
        "half_space_total_hitting_mass": half_space_mass,
        "half_space_time_energy_factor": unit_factor,
        "half_space_conditional_thresholds": thresholds,
        "surface_L4_trace_inequality": (
            "||T v||_4^2<=2*S3^(3/4)*c_A*h[v]"
        ),
        "surface_L4_trace_form_constant": TRACE_L4_FORM_CONSTANT,
        "spatial_L2_envelope_theorem": (
            "if ||dnu/(ds dsigma)||_L2(Sigma)<=rho_2(s), then "
            "int |Tw|^2 dnu<=C_4 F^2 J_(rho_2)"
        ),
        "half_space_spatial_L2_kernel": (
            "||K(t,.)||_2=d/(4sqrt(2)pi)t^(-2)"
            "exp(-d^2/(4t))"
        ),
        "half_space_spatial_L2_peak_time": BUFFER_DISTANCE**2 / 8.0,
        "half_space_spatial_L2_peak": _half_space_spatial_l2(
            BUFFER_DISTANCE**2 / 8.0, BUFFER_DISTANCE
        ),
        "half_space_L2_time_energy_factor": unit_l2_factor,
        "half_space_L2_conditional_thresholds": l2_thresholds,
        "constant_positive_exterior_deformation_has_finite_return_moment": False,
        "polynomial_barrier_identity": (
            "for h=(1+s)^gamma(L/r)^beta, "
            "(partial_s+nu Delta+b.grad+c)h/h="
            "gamma/(1+s)+beta[nu(beta-1)-b.x]/r^2+c"
        ),
        "actual_weighted_exterior_envelope_certified": False,
        "full_Navier_Stokes_return_gate_closed": False,
        "scope_guard": (
            "the half-space and sphere formulas are exact Brownian "
            "benchmarks. They correct the time-tail class but do not bound "
            "the cylindrical return kernel with Navier-Stokes drift and "
            "deformation"
        ),
        "next_gate": (
            "upgrade the exact axisymmetric affine axial-compensation model "
            "to a certified global envelope, then control all affine "
            "histories and the nonaffine Navier-Stokes exterior error; total "
            "return mass alone is insufficient"
        ),
    }
    positive_checks = (
        abs(half_space_mass - 1.0) < 1.0e-10,
        0.44 < unit_factor["factor"] < 0.47,
        0.31 < unit_factor["optimal_window"] < 0.34,
        0.12 < thresholds["potential_L3_over_2_threshold"] < 0.14,
        0.03 < thresholds["drift_L3_threshold"] < 0.04,
        0.66 < TRACE_L4_FORM_CONSTANT < 0.69,
        0.51 < unit_l2_factor["factor"] < 0.54,
        0.17 < l2_thresholds["potential_L3_over_2_threshold"] < 0.18,
        0.04 < l2_thresholds["drift_L3_threshold"] < 0.05,
        result["sphere_tail_disproves_every_positive_exponential_rate"],
        not result["exponential_envelope_required"],
        not result[
            "constant_positive_exterior_deformation_has_finite_return_moment"
        ],
    )
    result["all_positive_exterior_tail_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
