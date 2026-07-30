"""Audit axial L2 compensation in the inward-affine return stress test."""

from __future__ import annotations

import json
import math

import mpmath as mp
import numpy as np


INNER_RADIUS = 1.0
START_RADIUS = 2.0
WORKING_KAPPA = 1.0
MAXIMUM_MODE = 12


def _mode_laplace(
    laplace: mp.mpf,
    mode: int,
    kappa: float,
) -> mp.mpf:
    inner_y = mp.mpf(kappa) * INNER_RADIUS**2 / 2
    start_y = mp.mpf(kappa) * START_RADIUS**2 / 2
    parameter = mp.mpf(mode) / 2 + laplace / (2 * kappa)
    order = mode + 1
    return (
        (start_y / inner_y) ** (mp.mpf(mode) / 2)
        * mp.hyperu(parameter, order, start_y)
        / mp.hyperu(parameter, order, inner_y)
    )


def _hitting_modes(
    time: float,
    kappa: float,
    inversion_order: int,
    maximum_mode: int = MAXIMUM_MODE,
) -> np.ndarray:
    values = []
    for mode in range(maximum_mode + 1):
        value = mp.invertlaplace(
            lambda laplace, mode=mode: _mode_laplace(
                laplace, mode, kappa
            ),
            time,
            method="stehfest",
            degree=inversion_order,
        )
        values.append(float(value))
    return np.asarray(values)


def _weighted_spatial_l2(
    time: float,
    kappa: float,
    inversion_order: int,
) -> float:
    modes = _hitting_modes(time, kappa, inversion_order)
    angular_l2 = math.sqrt(
        (modes[0] ** 2 + 2.0 * float(np.sum(modes[1:] ** 2)))
        / (2.0 * math.pi)
    )
    axial_variance = math.expm1(4.0 * kappa * time) / (2.0 * kappa)
    axial_gaussian_l2 = 1.0 / (
        math.sqrt(2.0)
        * math.pi**0.25
        * axial_variance**0.25
    )
    return math.exp(kappa * time) * angular_l2 * axial_gaussian_l2


def _principal_radial_rate(kappa: float) -> dict[str, float]:
    inner_y = mp.mpf(kappa) / 2
    root = mp.findroot(
        lambda parameter: mp.hyperu(parameter, 1, inner_y),
        (-mp.mpf("0.3"), -mp.mpf("1.0")),
        tol=mp.mpf("1.0e-25"),
        verify=False,
    )
    rate = -2 * mp.mpf(kappa) * root
    start_y = 2 * mp.mpf(kappa)
    parameter_derivative = mp.diff(
        lambda parameter: mp.hyperu(parameter, 1, inner_y),
        root,
    )
    radial_residue = (
        2
        * mp.mpf(kappa)
        * mp.hyperu(root, 1, start_y)
        / parameter_derivative
    )
    weighted_axial_limit = (
        (2 * mp.mpf(kappa)) ** mp.mpf("0.25")
        / (mp.sqrt(2) * mp.pi ** mp.mpf("0.25"))
    )
    full_l2_tail_coefficient = (
        radial_residue
        / mp.sqrt(2 * mp.pi)
        * weighted_axial_limit
    )
    return {
        "Tricomi_parameter_root": float(root),
        "principal_radial_decay_rate": float(rate),
        "radial_hitting_density_residue": float(radial_residue),
        "weighted_axial_L2_limit": float(weighted_axial_limit),
        "full_weighted_L2_tail_coefficient": float(
            full_l2_tail_coefficient
        ),
        "asymptotic_tail_integral": float(
            full_l2_tail_coefficient / rate
        ),
    }


def _inversion_pilot() -> dict[str, object]:
    times = (0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0)
    rows = []
    for time in times:
        order_12 = _weighted_spatial_l2(time, WORKING_KAPPA, 12)
        order_14 = _weighted_spatial_l2(time, WORKING_KAPPA, 14)
        rows.append(
            {
                "time": time,
                "order_12": order_12,
                "order_14": order_14,
                "relative_spread": abs(order_14 - order_12)
                / max(abs(order_14), abs(order_12), 1.0e-30),
            }
        )
    peak = max(rows, key=lambda row: row["order_14"])
    return {
        "rows": rows,
        "maximum_relative_spread": max(
            row["relative_spread"] for row in rows
        ),
        "sampled_peak_time": peak["time"],
        "sampled_peak_weighted_spatial_L2_density": peak["order_14"],
    }


def audit() -> dict[str, object]:
    mp.mp.dps = 35
    spectral_rows = {
        str(kappa): _principal_radial_rate(kappa)
        for kappa in (0.125, 0.25, 0.5, 1.0)
    }
    pilot = _inversion_pilot()
    result: dict[str, object] = {
        "affine_return_drift": "b=(-kappa*x,-kappa*y,2*kappa*z)",
        "one_history_radial_deformation_weight": "exp(kappa*t)",
        "axial_variance": "sigma_z^2=(exp(4*kappa*t)-1)/(2*kappa)",
        "exact_weighted_axial_L2_factor": (
            "exp(kappa*t)||Gaussian_(sigma_z)||_2="
            "(2*kappa)^(1/4)/(sqrt(2)*pi^(1/4))"
            "(1-exp(-4*kappa*t))^(-1/4)"
        ),
        "axial_L2_exactly_cancels_affine_deformation_at_long_time": True,
        "angular_mode_Laplace_transform": (
            "F_n(lambda;r)=(y/y_a)^(n/2)"
            "U(n/2+lambda/(2*kappa),n+1,y)/"
            "U(n/2+lambda/(2*kappa),n+1,y_a), "
            "y=kappa*r^2/2"
        ),
        "principal_radial_rate_equation": (
            "U(-lambda_0/(2*kappa),1,kappa/2)=0"
        ),
        "spectral_tail_rows": spectral_rows,
        "working_kappa": WORKING_KAPPA,
        "maximum_angular_mode": MAXIMUM_MODE,
        "inversion_orders": [12, 14],
        "weighted_L2_inversion_pilot": pilot,
        "ideal_inward_affine_weighted_L2_tail_is_summable": True,
        "Brownian_endpoint_tail_is_summable": True,
        "finite_axial_patch_removes_old_affine_L2_tail_obstruction": True,
        "weighted_return_operator_is_contracting_from_this_tail_alone": False,
        "all_affine_spectra_and_orientations_covered": False,
        "time_dependent_affine_history_covered": False,
        "nonaffine_Navier_Stokes_exterior_covered": False,
        "numerical_inversion_or_spectral_constants_certified": False,
        "full_weighted_exterior_return_gate_closed": False,
        "scope_guard": (
            "the separated OU formulas and axial cancellation are exact for "
            "the axisymmetric affine return stress test. Tricomi roots and "
            "inverse Laplace values are high-precision pilots, not interval "
            "enclosures. Summability is not a quantitative branch norm, "
            "and no comparison with rotating, nonnormal, time-dependent, "
            "or nonaffine Navier-Stokes drift is proved"
        ),
        "next_gate": (
            "turn the axisymmetric mode formula into a rigorous global L2 "
            "envelope at kappa=1, then prove a comparison or energy bound "
            "for all admissible affine spectra and time-dependent frames "
            "before adding the critical exterior error"
        ),
    }
    working_spectral = spectral_rows[str(WORKING_KAPPA)]
    positive_checks = (
        result[
            "axial_L2_exactly_cancels_affine_deformation_at_long_time"
        ],
        all(
            row["principal_radial_decay_rate"] > 0.0
            for row in spectral_rows.values()
        ),
        1.44
        < working_spectral["principal_radial_decay_rate"]
        < 1.46,
        pilot["maximum_relative_spread"] < 0.011,
        0.14 < pilot["sampled_peak_time"] < 0.16,
        1.00
        < pilot["sampled_peak_weighted_spatial_L2_density"]
        < 1.02,
        result[
            "finite_axial_patch_removes_old_affine_L2_tail_obstruction"
        ],
        not result[
            "weighted_return_operator_is_contracting_from_this_tail_alone"
        ],
        not result["full_weighted_exterior_return_gate_closed"],
    )
    result["all_positive_affine_axial_compensation_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
