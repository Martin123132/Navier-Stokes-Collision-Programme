"""Audit scalar gain bounds for the nonautonomous compact affine visit."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.special import iv, jn_zeros, kv


def _stationary_sector_constants(
    radial_modes: int = 500, axial_modes: int = 500
) -> dict[str, float]:
    radius = 2.0
    half_height = 0.75
    interface_radius = 1.0
    bessel_zeros = jn_zeros(0, radial_modes)[:, None]
    odd_axial_modes = np.arange(1, 2 * axial_modes, 2, dtype=float)[
        None, :
    ]
    eigenvalues = (
        (bessel_zeros / radius) ** 2
        + (odd_axial_modes * math.pi / (2.0 * half_height)) ** 2
        - 1.0
    )
    constant_coefficients_squared = (
        64.0
        * radius**2
        * half_height
        / (
            math.pi
            * bessel_zeros**2
            * odd_axial_modes**2
        )
    )
    dual_norm_squared = float(
        np.sum(constant_coefficients_squared / eigenvalues)
    )

    axial_floor = (math.pi / (2.0 * half_height)) ** 2
    radial_parameter = math.sqrt(axial_floor - 1.0)
    trace_norm_squared = float(
        iv(0, radial_parameter * interface_radius)
        * (
            kv(0, radial_parameter * interface_radius)
            - kv(0, radial_parameter * radius)
            / iv(0, radial_parameter * radius)
            * iv(0, radial_parameter * interface_radius)
        )
    )
    interface_area = (
        4.0 * math.pi * interface_radius * half_height
    )
    normalized_trace_increment = math.sqrt(
        trace_norm_squared * dual_norm_squared / interface_area
    )
    return {
        "constant_H_inverse_dual_norm_squared": dual_norm_squared,
        "constant_H_inverse_dual_norm": math.sqrt(dual_norm_squared),
        "inner_trace_norm_squared": trace_norm_squared,
        "inner_trace_norm": math.sqrt(trace_norm_squared),
        "normalized_uniform_surface_increment": (
            normalized_trace_increment
        ),
        "normalized_uniform_surface_gain_bound": (
            1.0 + normalized_trace_increment
        ),
    }


def audit() -> dict[str, object]:
    radius = 2.0
    half_height = 0.75
    volume = 2.0 * half_height * math.pi * radius**2
    bessel_zero = float(jn_zeros(0, 1)[0])
    dirichlet_floor = (
        bessel_zero**2 / radius**2
        + math.pi**2 / (4.0 * half_height**2)
    )
    maximum_stretching = 1.0
    weighted_decay_rate = dirichlet_floor - maximum_stretching
    volume_gain_bound = 1.0 + 1.0 / weighted_decay_rate

    compact_visit_norm = 0.55681307217
    compact_generation = 0.160019377035
    legacy_cycle_coefficient = compact_generation / compact_visit_norm**2
    cycle_coefficient = 0.6586950386676936
    closure_gain = 1.0 / math.sqrt(cycle_coefficient)
    volume_density_allowance = (closure_gain / volume_gain_bound) ** 2

    stationary = _stationary_sector_constants()
    surface_density_allowance = (
        closure_gain
        / stationary["normalized_uniform_surface_gain_bound"]
    ) ** 2

    static_pilot = 1.3252767028921395
    static_standard_error = 0.00042033603448914754
    switched_pilot = 1.3275372021259952
    switched_standard_error = 0.00043479226494866656
    switch_z_score = (switched_pilot - static_pilot) / math.sqrt(
        static_standard_error**2 + switched_standard_error**2
    )
    artificial_inward_pilot = 1.3912394160956878
    artificial_inward_standard_error = 0.0007197960627532246
    optimistic_covariance_nash_bound = 1.482946657861099

    result: dict[str, object] = {
        "compact_geometry": {
            "outer_radius_over_L": radius,
            "half_height_over_L": half_height,
            "normalized_volume": volume,
        },
        "unweighted_Dirichlet_floor": dirichlet_floor,
        "maximum_normalized_stretching": maximum_stretching,
        "uniform_weighted_L2_decay_rate": weighted_decay_rate,
        "nonautonomous_volume_identity": (
            "m=1+w, w(t)=int_t^infinity U(t,s)lambda(s) ds, "
            "||U(t,s)||_(2->2)<=exp[-4.832287335665(s-t)]"
        ),
        "normalized_uniform_volume_gain_bound": volume_gain_bound,
        "uniform_volume_gain_closes_ideal_cycle": bool(
            volume_gain_bound < closure_gain
        ),
        "maximum_volume_density_ratio_for_same_bound": (
            volume_density_allowance
        ),
        "stationary_sector_theorem": (
            "for time-independent divergence-free drift, "
            "(H+K)w=1 with H=-Delta-1 and K skew; "
            "sqrt(h[w])<=||1||_(H^-1)"
        ),
        "stationary_sector_constants": stationary,
        "stationary_uniform_surface_gain_closes_ideal_cycle": bool(
            stationary["normalized_uniform_surface_gain_bound"]
            < closure_gain
        ),
        "maximum_surface_density_ratio_for_same_bound": (
            surface_density_allowance
        ),
        "compact_cycle_coefficient": cycle_coefficient,
        "legacy_bare_halving_cycle_coefficient": legacy_cycle_coefficient,
        "maximum_dynamic_one_history_gain_for_closure": closure_gain,
        "all_exit_dominates_radial_exit_kernel": True,
        "static_worst_spectrum_contracting_direction_pilot": {
            "path_count": 500_000,
            "time_step": 0.00125,
            "all_exit_gain": static_pilot,
            "standard_error": static_standard_error,
        },
        "half_time_xy_switch_pilot": {
            "path_count": 500_000,
            "time_step": 0.00125,
            "switch_block_duration": 0.5,
            "all_exit_gain": switched_pilot,
            "standard_error": switched_standard_error,
            "increase_over_static_combined_standard_errors": (
                switch_z_score
            ),
        },
        "sampled_static_worst_comparison_supported": False,
        "sampled_switching_gain_remains_below_closure": bool(
            switched_pilot < closure_gain
        ),
        "position_dependent_inward_control_stress_test": {
            "interpretation": (
                "the inadmissible feedback choice can imitate drift -y "
                "and nearly exhaust the renewal allowance"
            ),
            "all_exit_gain": artificial_inward_pilot,
            "standard_error": artificial_inward_standard_error,
        },
        "pointwise_orientation_Bellman_bound_recommended": False,
        "affine_transition_covariance_constraints": (
            "Q(t)>=(1-exp(-2t))I and det Q(t)>=(2t)^3; "
            "the determinant follows from det F(t,s)=1 and Minkowski"
        ),
        "optimistic_covariance_plus_Nash_gain_bound": (
            optimistic_covariance_nash_bound
        ),
        "optimistic_covariance_plus_Nash_closes": bool(
            optimistic_covariance_nash_bound < closure_gain
        ),
        "full_nonautonomous_boundary_gain_closed": False,
        "architecture_consequence": (
            "the arbitrary-history estimate only narrowly closes in "
            "normalized volume L2 under the current cubic split, while the "
            "stationary surface and switching pilots no longer close; use an unnormalized "
            "smoothing/occupation kernel or a storage-augmented boundary "
            "renewal instead of a static-worst or feedback comparison"
        ),
        "next_gate": (
            "construct a buffered sub-Markov map from the boundary law to "
            "a smoothed volume law without conditioning, and include its "
            "lost mass as contraction; then compose the 1.20694134 volume "
            "gain with the square-tilted dynamic kernel"
        ),
        "scope_guard": (
            "the volume and stationary sector estimates are analytic; "
            "the switched/static rows are Monte Carlo diagnostics, and "
            "the covariance-Nash number uses optimistic sharp smoothing "
            "constants yet still fails"
        ),
    }
    positive_checks = (
        weighted_decay_rate > 4.83,
        volume_gain_bound < 1.207,
        result["uniform_volume_gain_closes_ideal_cycle"],
        stationary["normalized_uniform_surface_gain_bound"] < 1.27,
        not result["stationary_uniform_surface_gain_closes_ideal_cycle"],
        switch_z_score > 3.7,
        not result["sampled_switching_gain_remains_below_closure"],
        not result["optimistic_covariance_plus_Nash_closes"],
        artificial_inward_pilot > closure_gain,
    )
    result["all_positive_scalar_gain_gate_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
