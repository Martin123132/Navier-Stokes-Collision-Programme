"""Pilot the complete static-affine scalar branches in neutral storage."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import erf
from scipy.sparse import eye
from scipy.sparse.linalg import eigs, splu


PATCH_HALF_HEIGHT = 0.75


def _load_resolvent_module():
    script = Path(__file__).resolve().with_name(
        "neutral_strip_branch_resolvent_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "neutral_strip_resolvent_for_axial_patch", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _axial_patch_weight(time: float, rho: float) -> float:
    if time <= 0.0:
        return 1.0
    axial_rate = 1.0 + rho
    variance = math.expm1(2.0 * axial_rate * time) / axial_rate
    patch_probability = erf(
        PATCH_HALF_HEIGHT / math.sqrt(2.0 * variance)
    )
    return math.exp(time) * float(patch_probability)


def _integrate_patched_return(
    grid: dict[str, object],
    rho: float,
    time_step_scale: float,
) -> tuple[np.ndarray, float, float]:
    generator = grid["generator"]
    state_count = generator.shape[0]
    identity = eye(state_count, format="csc")
    flux = np.asarray(grid["inner_rates"], dtype=float).copy()
    patched_return = np.zeros(state_count)
    unweighted_integral = np.zeros(state_count)
    time = 0.0
    schedule = (
        (0.02, 0.0002),
        (0.1, 0.0005),
        (0.5, 0.002),
        (2.0, 0.01),
        (10.0, 0.05),
        (30.0, 0.1),
    )
    for segment_end, base_step in schedule:
        desired_step = base_step * time_step_scale
        step_count = int(round((segment_end - time) / desired_step))
        step = (segment_end - time) / step_count
        factorization = splu(identity - step * generator)
        for _ in range(step_count):
            flux = factorization.solve(flux)
            time += step
            patched_return += (
                step * _axial_patch_weight(time, rho) * flux
            )
            unweighted_integral += step * flux

    exact_unweighted = splu(-generator).solve(grid["inner_rates"])
    recovery_error = float(
        np.max(np.abs(unweighted_integral - exact_unweighted))
    )
    return patched_return, recovery_error, float(np.max(flux))


def _row(
    module,
    y_intervals: int,
    rho: float,
    x_half_width: float = 4.2,
    time_step_scale: float = 1.0,
    fit_inner_boundary: bool = True,
    compute_spectral_rate: bool = False,
) -> dict[str, object]:
    grid = module._build_generator(
        y_intervals,
        rho,
        x_half_width,
        fit_inner_boundary,
    )
    generator = grid["generator"]
    patched_return, recovery_error, terminal_flux = (
        _integrate_patched_return(grid, rho, time_step_scale)
    )
    identity = eye(generator.shape[0], format="csc")
    wall_moment = splu(-generator - identity).solve(grid["wall_rates"])
    branch_values = np.column_stack([patched_return, wall_moment])

    angle_rows = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False):
        return_gain, wall_gain = module._interpolate_entry(
            branch_values, angle, grid
        )
        criterion = module.H1_ENTRY_GAIN**2 * (
            return_gain**2
            + (module.CUBIC_SPLIT_FACTOR * wall_gain) ** 2
        )
        unpaid_criterion = module.H1_ENTRY_GAIN**2 * (
            return_gain**2 + wall_gain**2
        )
        angle_rows.append(
            {
                "angle": float(angle),
                "axial_patch_return_gain": float(return_gain),
                "deformation_weighted_wall_gain": float(wall_gain),
                "complete_scalar_branch_criterion": float(criterion),
                "criterion_without_true_split_payment": float(
                    unpaid_criterion
                ),
            }
        )
    worst = max(
        angle_rows,
        key=lambda item: item["complete_scalar_branch_criterion"],
    )
    largest_return = max(
        angle_rows,
        key=lambda item: item["axial_patch_return_gain"],
    )
    largest_wall = max(
        angle_rows,
        key=lambda item: item["deformation_weighted_wall_gain"],
    )
    worst_unpaid = max(
        angle_rows,
        key=lambda item: item["criterion_without_true_split_payment"],
    )
    spectral_rate = None
    if compute_spectral_rate:
        eigenvalues = eigs(
            generator,
            k=2,
            which="LR",
            return_eigenvectors=False,
            tol=1.0e-9,
            maxiter=100000,
        )
        spectral_rate = float(np.min(-np.real(eigenvalues)))
    return {
        "y_intervals": y_intervals,
        "spacing": grid["spacing"],
        "x_half_width": grid["x_half_width"],
        "interior_state_count": generator.shape[0],
        "rho": rho,
        "time_step_scale": time_step_scale,
        "inner_boundary_scheme": grid["inner_boundary_scheme"],
        "unweighted_return_resolvent_recovery_error": recovery_error,
        "terminal_inner_flux_maximum": terminal_flux,
        "principal_killed_rate_pilot": spectral_rate,
        "maximum_axial_patch_return_gain": largest_return[
            "axial_patch_return_gain"
        ],
        "maximum_axial_patch_return_angle": largest_return["angle"],
        "maximum_deformation_weighted_wall_gain": largest_wall[
            "deformation_weighted_wall_gain"
        ],
        "maximum_deformation_weighted_wall_angle": largest_wall["angle"],
        "maximum_complete_scalar_branch_criterion": worst[
            "complete_scalar_branch_criterion"
        ],
        "worst_complete_scalar_branch_angle": worst["angle"],
        "worst_angle_row": worst,
        "maximum_criterion_without_true_split_payment": worst_unpaid[
            "criterion_without_true_split_payment"
        ],
        "worst_unpaid_angle": worst_unpaid["angle"],
    }


def audit() -> dict[str, object]:
    module = _load_resolvent_module()
    rho_values = (0.0, 0.25, 0.5, 0.75, 1.0)
    mesh_intervals = (30, 40, 50)
    convergence_rows = [
        _row(module, intervals, rho)
        for intervals in mesh_intervals
        for rho in rho_values
    ]
    finest_rows = {
        row["rho"]: row
        for row in convergence_rows
        if row["y_intervals"] == mesh_intervals[-1]
    }
    middle_rows = {
        row["rho"]: row
        for row in convergence_rows
        if row["y_intervals"] == mesh_intervals[-2]
    }
    maximum_mesh_change = max(
        abs(
            finest_rows[rho]["maximum_complete_scalar_branch_criterion"]
            - middle_rows[rho][
                "maximum_complete_scalar_branch_criterion"
            ]
        )
        for rho in rho_values
    )

    boundary_refinement_rows = [
        _row(module, intervals, rho)
        for intervals in (60, 80)
        for rho in (0.0, 1.0)
    ]
    time_refinement_rows = [
        _row(module, 50, rho, time_step_scale=scale)
        for rho in (0.0, 1.0)
        for scale in (1.0, 0.5)
    ]
    time_groups = {
        rho: [row for row in time_refinement_rows if row["rho"] == rho]
        for rho in (0.0, 1.0)
    }
    maximum_time_refinement_change = max(
        abs(
            rows[0]["maximum_complete_scalar_branch_criterion"]
            - rows[1]["maximum_complete_scalar_branch_criterion"]
        )
        for rows in time_groups.values()
    )

    x_truncation_rows = [
        _row(module, 40, 0.0, x_half_width=width)
        for width in (3.15, 4.2, 5.25)
    ]
    maximum_x_truncation_spread = max(
        row["maximum_complete_scalar_branch_criterion"]
        for row in x_truncation_rows
    ) - min(
        row["maximum_complete_scalar_branch_criterion"]
        for row in x_truncation_rows
    )

    scheme_rows = [
        _row(module, intervals, 0.0, fit_inner_boundary=fit_boundary)
        for intervals in (40, 60, 80)
        for fit_boundary in (False, True)
    ]
    spectral_rows = [
        _row(module, 50, rho, compute_spectral_rate=True)
        for rho in rho_values
    ]

    finest_summary = [
        {
            "rho": rho,
            "maximum_axial_patch_return_gain": finest_rows[rho][
                "maximum_axial_patch_return_gain"
            ],
            "maximum_deformation_weighted_wall_gain": finest_rows[rho][
                "maximum_deformation_weighted_wall_gain"
            ],
            "maximum_complete_scalar_branch_criterion": finest_rows[rho][
                "maximum_complete_scalar_branch_criterion"
            ],
            "maximum_criterion_without_true_split_payment": finest_rows[rho][
                "maximum_criterion_without_true_split_payment"
            ],
            "worst_complete_scalar_branch_angle": finest_rows[rho][
                "worst_complete_scalar_branch_angle"
            ],
        }
        for rho in rho_values
    ]
    maximum_criterion = max(
        row["maximum_complete_scalar_branch_criterion"]
        for row in convergence_rows + boundary_refinement_rows
    )
    minimum_spectral_rate = min(
        row["principal_killed_rate_pilot"] for row in spectral_rows
    )
    maximum_unpaid_criterion = max(
        row["maximum_criterion_without_true_split_payment"]
        for row in convergence_rows + boundary_refinement_rows
    )

    result: dict[str, object] = {
        "stopped_domain": "Omega_Y={r>1, |y|<2.1}",
        "affine_subfamily": "b_rho=(-x,-rho*y,(1+rho)*z)",
        "axial_patch_half_height": PATCH_HALF_HEIGHT,
        "axial_variance": "V_rho(t)=(exp(2(1+rho)t)-1)/(1+rho)",
        "return_branch_gain": (
            "k_R=E[exp(tau)1_R erf(H/sqrt(2V_rho(tau)))]"
        ),
        "wall_branch_gain": "k_S=E[exp(tau)1_S]",
        "complete_scalar_criterion": (
            "g_H^2[k_R^2+(s_cubic*k_S)^2]"
        ),
        "convergence_rows": convergence_rows,
        "finest_summary": finest_summary,
        "boundary_refinement_rows": boundary_refinement_rows,
        "time_refinement_rows": time_refinement_rows,
        "x_truncation_rows": x_truncation_rows,
        "inner_boundary_scheme_comparison_rows": scheme_rows,
        "spectral_rows": spectral_rows,
        "maximum_middle_to_fine_criterion_change": maximum_mesh_change,
        "maximum_time_refinement_criterion_change": (
            maximum_time_refinement_change
        ),
        "maximum_x_truncation_criterion_spread": (
            maximum_x_truncation_spread
        ),
        "maximum_sampled_complete_scalar_criterion": maximum_criterion,
        "maximum_sampled_criterion_without_true_split_payment": (
            maximum_unpaid_criterion
        ),
        "minimum_sampled_principal_killed_rate": minimum_spectral_rate,
        "wall_deformation_moment_finite_on_all_sampled_grids": bool(
            minimum_spectral_rate > 1.0
        ),
        "all_sampled_complete_scalar_branch_stresses_close": all(
            row["maximum_complete_scalar_branch_criterion"] < 1.0
            for row in convergence_rows + boundary_refinement_rows
        ),
        "all_sampled_scalar_stresses_close_without_split_payment": all(
            row["maximum_criterion_without_true_split_payment"] < 1.0
            for row in convergence_rows + boundary_refinement_rows
        ),
        "physical_wall_exit_to_true_split_identification_proved": False,
        "boundary_flux_space_time_L2_error_gain_certified": False,
        "time_dependent_eigenframes_covered": False,
        "negative_affine_parameter_half_covered": False,
        "full_Navier_Stokes_generation_gate_closed": False,
        "interpretation": (
            "the exact axial patch and the deformation-weighted wall moment "
            "complement each other by entry orientation. Every sampled "
            "static 0<=rho<=1 scalar branch criterion closes with substantial "
            "pilot margin, unlike the raw residual-moment stress"
        ),
        "scope_guard": (
            "this is a boundary-fitted finite-state and implicit-time pilot, "
            "not a certified PDE enclosure. It assumes each strip-wall hit "
            "is followed by the audited cubic true-split factor, and it "
            "controls constant branch payoff only. Perturbative H1 errors "
            "still require separate physical space-time boundary-density "
            "bounds"
        ),
        "next_gate": (
            "prove or disprove that a strip-wall hit is a genuine cubic "
            "level split, then certify the two boundary kernels and their "
            "space-time L2 density response under static affine drift before "
            "adding nonaffine Navier-Stokes errors"
        ),
    }
    positive_checks = (
        maximum_mesh_change < 0.01,
        maximum_time_refinement_change < 0.003,
        maximum_x_truncation_spread < 0.01,
        max(
            row["unweighted_return_resolvent_recovery_error"]
            for row in convergence_rows + boundary_refinement_rows
        )
        < 1.0e-10,
        max(
            row["terminal_inner_flux_maximum"]
            for row in convergence_rows + boundary_refinement_rows
        )
        < 1.0e-18,
        result["wall_deformation_moment_finite_on_all_sampled_grids"],
        result["all_sampled_complete_scalar_branch_stresses_close"],
        not result[
            "all_sampled_scalar_stresses_close_without_split_payment"
        ],
        maximum_criterion < 0.7,
        not result["physical_wall_exit_to_true_split_identification_proved"],
        not result["boundary_flux_space_time_L2_error_gain_certified"],
        not result["full_Navier_Stokes_generation_gate_closed"],
    )
    result["all_positive_axial_patch_branch_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
