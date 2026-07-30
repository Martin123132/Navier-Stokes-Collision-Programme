"""Audit the scaling obstruction to an energy-only occupation bound."""

from __future__ import annotations

import json
import math


def _interval_survival(
    normalized_time: float, term_count: int = 100
) -> float:
    """Survival at the origin for generator Delta on (-1, 1)."""
    return sum(
        4.0
        / math.pi
        * (-1.0) ** mode
        / (2 * mode + 1)
        * math.exp(
            -((2 * mode + 1) ** 2)
            * math.pi**2
            * normalized_time
            / 4.0
        )
        for mode in range(term_count)
    )


def audit() -> dict[str, object]:
    half_width = 0.5
    normalized_duration = 0.06
    burst_amplitude = 0.7
    no_restart_potential_only_budget = 0.63227660014
    tapered_probability_allowance = 0.243451253799
    compact_visit_norm = 0.55681307217
    compact_probability_allowance = (1.0 - compact_visit_norm) ** 2

    interval_time = normalized_duration / half_width**2
    one_dimensional_survival = _interval_survival(interval_time)
    cube_survival = one_dimensional_survival**3
    critical_spatial_mass = (
        burst_amplitude * (2.0 * half_width) ** 2
    )
    dissipation_coefficient = (
        burst_amplitude**2
        * normalized_duration
        * (2.0 * half_width) ** 3
    )

    epsilon_rows = []
    for epsilon in (1.0, 0.1, 0.01, 0.0001, 0.00001):
        epsilon_rows.append(
            {
                "epsilon": epsilon,
                "burst_height": burst_amplitude / epsilon**2,
                "burst_duration": normalized_duration * epsilon**2,
                "bad_cube_half_width": half_width * epsilon,
                "spatial_L3_over_2_norm": critical_spatial_mass,
                "spacetime_L2_norm_squared": (
                    dissipation_coefficient * epsilon
                ),
                "conditional_cube_survival_probability": cube_survival,
            }
        )

    result: dict[str, object] = {
        "normalized_diffusion_generator": "Delta",
        "critical_burst": (
            "Q_epsilon=a*epsilon^(-2) on "
            "0<t<tau*epsilon^2 and |x_i|<h*epsilon"
        ),
        "burst_parameters": {
            "a": burst_amplitude,
            "h": half_width,
            "tau": normalized_duration,
        },
        "critical_spatial_mass_identity": (
            "||Q_epsilon(t)||_(3/2)=4*a*h^2"
        ),
        "critical_spatial_L3_over_2_norm": critical_spatial_mass,
        "compact_no_restart_potential_only_budget": (
            no_restart_potential_only_budget
        ),
        "burst_exceeds_compact_potential_only_budget": bool(
            critical_spatial_mass > no_restart_potential_only_budget
        ),
        "Leray_dissipation_proxy_identity": (
            "||Q_epsilon||_(L2_t,x)^2=8*a^2*tau*h^3*epsilon"
        ),
        "Leray_dissipation_proxy_coefficient_times_epsilon": (
            dissipation_coefficient
        ),
        "one_dimensional_survival_series": (
            "(4/pi) sum_n (-1)^n/(2n+1) "
            "exp(-(2n+1)^2*pi^2*tau/(4*h^2))"
        ),
        "one_dimensional_survival_probability": (
            one_dimensional_survival
        ),
        "conditional_cube_survival_probability": cube_survival,
        "tapered_probability_paid_allowance": (
            tapered_probability_allowance
        ),
        "compact_full_affine_probability_paid_allowance_candidate": (
            compact_probability_allowance
        ),
        "survival_exceeds_both_probability_allowances": bool(
            cube_survival > tapered_probability_allowance
            and cube_survival > compact_probability_allowance
        ),
        "epsilon_rows": epsilon_rows,
        "Navier_Stokes_scaling": (
            "u_epsilon(x,t)=epsilon^(-1)u(x/epsilon,t/epsilon^2)"
        ),
        "Navier_Stokes_scaling_consequences": {
            "kinetic_energy_squared_norm_factor": "epsilon",
            "integrated_enstrophy_factor": "epsilon",
            "critical_Campanato_and_strain_functional_factor": "1",
            "conditional_diffusion_occupation_factor": "1",
        },
        "energy_only_conditional_occupation_bound_closed": False,
        "scope_guard": (
            "this is a sharp scaling obstruction for deductions from the "
            "Leray inequality alone, not a construction of a singular "
            "Navier-Stokes solution or proof that every scalar burst is "
            "realized by the equation"
        ),
        "consequence": (
            "a conditional bad-branch estimate must use equation-specific "
            "geometry, hitting-law smoothing, cancellation, collision "
            "damping, or another scale-critical hypothesis"
        ),
        "next_gate": (
            "test whether a time-dependent full-affine reference converts "
            "constant-spectrum orientation drift from an abort into a "
            "controlled nonautonomous form perturbation"
        ),
    }
    positive_checks = (
        result["burst_exceeds_compact_potential_only_budget"],
        result["survival_exceeds_both_probability_allowances"],
        abs(critical_spatial_mass - 0.7) < 1.0e-14,
        abs(dissipation_coefficient - 0.0294) < 1.0e-14,
        0.34 < cube_survival < 0.35,
        epsilon_rows[-1]["spacetime_L2_norm_squared"] < 1.0e-6,
        all(
            abs(
                row["conditional_cube_survival_probability"]
                - cube_survival
            )
            < 1.0e-15
            for row in epsilon_rows
        ),
    )
    result["all_positive_Leray_occupation_no_go_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
