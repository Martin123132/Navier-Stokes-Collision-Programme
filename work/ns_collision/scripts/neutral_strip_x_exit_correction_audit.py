"""Bound the artificial x-side branch and correct the stored FEM response."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import mpmath
import numpy as np


STRIP_HALF_WIDTH = 2.1
ENTRY_RADIUS = 2.0
PATCH_HALF_HEIGHT = 0.75
MODE_COUNT = 12
TRACE_L4_FORM_CONSTANT = 0.6741481379606137
PILOT_TRUNCATION_PROBABILITIES = {
    4.2: 0.0016642440807681795,
    5.25: 1.320720817809944e-05,
    6.3: 3.3747522647980825e-08,
}


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lower(interval) -> float:
    return float(interval.a)


def _upper(interval) -> float:
    return np.nextafter(float(interval.b), math.inf)


def _positive_hyp1f1_bounds(
    parameter,
    argument: float,
    relative_tolerance: float = 1.0e-40,
) -> tuple[float, float, int]:
    """Enclose M(a,1/2,z) by its positive defining series."""

    iv = mpmath.iv
    iv.dps = 80
    a = parameter
    b = iv.mpf("0.5")
    z = iv.mpf(str(argument))
    term = iv.mpf(1)
    total = iv.mpf(1)
    tail_upper = math.inf
    for index in range(1, 20000):
        term *= (a + index - 1) * z / ((b + index - 1) * index)
        total += term
        next_ratio = (a + index) * z / ((b + index) * (index + 1))
        ratio_upper = _upper(next_ratio)
        monotonicity_numerator = (
            (b - a) * (index + 1) - (a + index) * (b + index)
        )
        ratios_decrease_hereafter = _upper(monotonicity_numerator) <= 0.0
        if (
            index >= 12
            and ratio_upper < 1.0
            and ratios_decrease_hereafter
        ):
            tail_upper = _upper(term) * ratio_upper / (1.0 - ratio_upper)
            if tail_upper <= relative_tolerance * max(_lower(total), 1.0):
                lower = np.nextafter(_lower(total), -math.inf)
                upper = np.nextafter(_upper(total) + tail_upper, math.inf)
                return lower, upper, index
    raise RuntimeError("positive hypergeometric series did not converge")


def _mode_laplace_upper(
    mode: int,
    x_half_width: float,
    entry_radius: float = ENTRY_RADIUS,
    strip_half_width: float = STRIP_HALF_WIDTH,
) -> tuple[float, dict[str, float | int]]:
    iv = mpmath.iv
    iv.dps = 80
    wave_number = (
        (2 * mode + 1) * iv.pi / (2 * iv.mpf(str(strip_half_width)))
    )
    parameter = wave_number**2 / 2
    numerator = _positive_hyp1f1_bounds(
        parameter, entry_radius**2 / 2.0
    )
    denominator = _positive_hyp1f1_bounds(
        parameter, x_half_width**2 / 2.0
    )
    ratio_upper = np.nextafter(
        numerator[1] / denominator[0], math.inf
    )
    return ratio_upper, {
        "mode": mode,
        "wave_number_lower": _lower(wave_number),
        "wave_number_upper": _upper(wave_number),
        "laplace_transform_upper": ratio_upper,
        "numerator_terms": numerator[2],
        "denominator_terms": denominator[2],
    }


def _mode_tail_upper(
    x_half_width: float,
    first_omitted_mode: int,
    entry_radius: float = ENTRY_RADIUS,
    strip_half_width: float = STRIP_HALF_WIDTH,
) -> float:
    mpmath.mp.dps = 80
    distance = mpmath.mpf(str(x_half_width - entry_radius))
    width = mpmath.mpf(str(strip_half_width))
    mode = first_omitted_mode
    wave_number = (2 * mode + 1) * mpmath.pi / (2 * width)
    ratio = mpmath.exp(-mpmath.pi * distance / width)
    first = (
        8
        / (mpmath.pi * (2 * mode + 1))
        * mpmath.exp(-wave_number * distance)
    )
    return np.nextafter(float(first / (1 - ratio)), math.inf)


def _side_exit_probability_upper(
    x_half_width: float,
    mode_count: int = MODE_COUNT,
) -> dict[str, object]:
    """Bound side-before-wall probability after removing the inner disk."""

    if x_half_width <= ENTRY_RADIUS:
        raise ValueError("x half-width must exceed the entry radius")
    mpmath.mp.dps = 80
    mode_rows = []
    finite_sum = 0.0
    for mode in range(mode_count):
        transform, row = _mode_laplace_upper(mode, x_half_width)
        coefficient = np.nextafter(
            float(4 / (mpmath.pi * (2 * mode + 1))), math.inf
        )
        contribution = np.nextafter(coefficient * transform, math.inf)
        finite_sum = np.nextafter(finite_sum + contribution, math.inf)
        row["absolute_fourier_coefficient_upper"] = coefficient
        row["probability_contribution_upper"] = contribution
        mode_rows.append(row)
    tail = _mode_tail_upper(x_half_width, mode_count)
    probability_upper = np.nextafter(finite_sum + tail, math.inf)
    pilot = PILOT_TRUNCATION_PROBABILITIES.get(x_half_width)
    return {
        "x_half_width": x_half_width,
        "entry_radius": ENTRY_RADIUS,
        "strip_half_width": STRIP_HALF_WIDTH,
        "retained_mode_count": mode_count,
        "finite_mode_sum_upper": finite_sum,
        "mode_tail_upper": tail,
        "continuum_side_exit_probability_upper": probability_upper,
        "twice_probability_upper": 2.0 * probability_upper,
        "interval_renewal_denominator_lower": np.nextafter(
            1.0 - 2.0 * probability_upper, -math.inf
        ),
        "stored_fem_truncation_probability_pilot": pilot,
        "analytic_upper_exceeds_stored_fem_pilot": bool(
            pilot is None or probability_upper >= pilot
        ),
        "mode_rows": mode_rows,
    }


def _scalar_axial_weight_upper(
    patch_half_height: float = PATCH_HALF_HEIGHT,
) -> float:
    # erf(x)<=min(1,2x/sqrt(pi)); optimizing the two bounds gives this.
    return np.nextafter(
        math.sqrt(1.0 + 2.0 * patch_half_height**2 / math.pi),
        math.inf,
    )


def _correct_direct_row(
    direct_row: dict[str, object],
    side_row: dict[str, object],
) -> dict[str, object]:
    probability = side_row["continuum_side_exit_probability_upper"]
    denominator = side_row["interval_renewal_denominator_lower"]
    scalar_weight = _scalar_axial_weight_upper()
    corrected_angles = []
    for angle_row in direct_row["finite_time_certificate"]["angle_rows"]:
        direct_factor = angle_row["certified_interval_factor_with_tail"]
        direct_scalar = angle_row["certified_scalar_gain_with_tail"]
        corrected_factor = np.nextafter(
            direct_factor / denominator, math.inf
        )
        corrected_scalar = np.nextafter(
            direct_scalar + scalar_weight * probability, math.inf
        )
        corrected_response = np.nextafter(
            math.sqrt(
                corrected_scalar
                * TRACE_L4_FORM_CONSTANT
                * corrected_factor
            ),
            math.inf,
        )
        corrected_angles.append(
            {
                "angle": angle_row["angle"],
                "direct_certified_interval_factor": direct_factor,
                "x_corrected_interval_factor_upper": corrected_factor,
                "direct_certified_scalar_gain": direct_scalar,
                "x_corrected_scalar_gain_upper": corrected_scalar,
                "direct_certified_response": angle_row[
                    "certified_response"
                ],
                "x_corrected_stored_matrix_response": corrected_response,
            }
        )
    worst = max(
        corrected_angles,
        key=lambda row: row["x_corrected_stored_matrix_response"],
    )
    return {
        "x_half_width": direct_row["x_half_width"],
        "direct_maximum_certified_response": direct_row[
            "finite_time_certificate"
        ]["maximum_certified_response"],
        "maximum_x_corrected_stored_matrix_response": worst[
            "x_corrected_stored_matrix_response"
        ],
        "worst_entry_angle": worst["angle"],
        "scalar_axial_weight_global_upper": scalar_weight,
        "interval_envelope_renewal_multiplier_upper": np.nextafter(
            1.0 / denominator, math.inf
        ),
        "angle_rows": corrected_angles,
    }


def audit(
    spacing: float = 0.12,
    x_half_widths: tuple[float, ...] = (4.2, 5.25, 6.3),
    run_direct_certificate: bool = True,
) -> dict[str, object]:
    side_rows = [
        _side_exit_probability_upper(width) for width in x_half_widths
    ]
    side_gate = bool(
        all(
            row["interval_renewal_denominator_lower"] > 0.0
            and row["analytic_upper_exceeds_stored_fem_pilot"]
            for row in side_rows
        )
    )
    corrected_rows = []
    direct_result = None
    if run_direct_certificate:
        finite_time = _load_module(
            "neutral_strip_reversible_finite_time_certificate.py",
            "finite_time_for_x_exit_correction",
        )
        direct_result = finite_time.audit(
            spacing=spacing,
            x_half_widths=x_half_widths,
            run_density=True,
        )
        side_by_width = {row["x_half_width"]: row for row in side_rows}
        corrected_rows = [
            _correct_direct_row(row, side_by_width[row["x_half_width"]])
            for row in direct_result["width_rows"]
        ]

    result = {
        "model": (
            "analytic continuum x-side excursion correction composed with "
            "the symmetrized stored rho=0 FEM certificate"
        ),
        "spacing": spacing,
        "x_half_widths": list(x_half_widths),
        "side_exit_rows": side_rows,
        "corrected_response_rows": corrected_rows,
        "rectangle_side_probability_formula": (
            "sum_n 4(-1)^n/((2n+1)pi) cos(k_n y) "
            "M(k_n^2/2,1/2,x^2/2)/M(k_n^2/2,1/2,X^2/2)"
        ),
        "inner_disk_removal_is_probability_upper_bound": True,
        "positive_hypergeometric_series_interval_enclosed": True,
        "omitted_return_must_recross_r2": True,
        "axial_L2_factor_nonincreasing": True,
        "shifted_interval_sum_cost_upper": 2.0,
        "scalar_axial_weight_global_upper": _scalar_axial_weight_upper(),
        "continuum_side_exit_probability_analytically_bounded": side_gate,
        "x_exit_interval_renewal_correction_proved": side_gate,
        "x_exit_scalar_correction_proved": side_gate,
        "x_exit_correction_composed_with_stored_matrix_certificate": bool(
            run_direct_certificate
            and direct_result is not None
            and direct_result["all_finite_time_certificate_checks_pass"]
        ),
        "maximum_x_corrected_stored_matrix_response": (
            max(
                row["maximum_x_corrected_stored_matrix_response"]
                for row in corrected_rows
            )
            if corrected_rows
            else None
        ),
        "x_truncation_removed_from_continuum_return_theorem": False,
        "continuum_return_response_certified": False,
        "scope_guard": (
            "The continuum side-exit probability and the renewal/scalar "
            "correction are analytic. The corrected numbers still use the "
            "stored finite-matrix direct response; polygonal-domain and "
            "weighted FEM consistency errors remain unenclosed."
        ),
        "next_gate": (
            "enclose polygonal-circle geometry and weighted lumped-FEM "
            "mass/stiffness consistency for the direct truncated response"
        ),
    }
    checks = (
        result["continuum_side_exit_probability_analytically_bounded"],
        result["x_exit_interval_renewal_correction_proved"],
        result["x_exit_scalar_correction_proved"],
        not result["x_truncation_removed_from_continuum_return_theorem"],
        not result["continuum_return_response_certified"],
    )
    if run_direct_certificate:
        checks += (
            result[
                "x_exit_correction_composed_with_stored_matrix_certificate"
            ],
            result["maximum_x_corrected_stored_matrix_response"] < 0.7,
        )
    result["all_x_exit_correction_checks_pass"] = bool(all(checks))
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
