"""Sweep neutral-strip width without an unavailable split payment."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import eigs, splu


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    resolvent,
    axial,
    half_width: float,
    target_spacing: float = 0.07,
    time_step_scale: float = 1.0,
    x_half_width: float | None = None,
) -> dict[str, object]:
    y_intervals = int(round(2.0 * half_width / target_spacing))
    if x_half_width is None:
        x_half_width = max(4.2, half_width + 2.1)
    grid = resolvent._build_generator(
        y_intervals,
        0.0,
        x_half_width,
        True,
        half_width,
    )
    generator = grid["generator"]
    eigenvalues = eigs(
        generator,
        k=2,
        which="LR",
        return_eigenvectors=False,
        tol=1.0e-8,
        maxiter=100000,
    )
    killed_rate = float(np.min(-np.real(eigenvalues)))
    if killed_rate <= 1.0:
        return {
            "strip_half_width": half_width,
            "spacing": grid["spacing"],
            "x_half_width": grid["x_half_width"],
            "interior_state_count": generator.shape[0],
            "time_step_scale": time_step_scale,
            "principal_killed_rate_pilot": killed_rate,
            "wall_deformation_moment_finite": False,
            "maximum_same_scale_criterion": math.inf,
        }

    patched_return, recovery_error, terminal_flux = (
        axial._integrate_patched_return(grid, 0.0, time_step_scale)
    )
    wall_moment = splu(
        -generator - eye(generator.shape[0], format="csc")
    ).solve(grid["wall_rates"])
    branch_values = np.column_stack([patched_return, wall_moment])

    angle_rows = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False):
        return_gain, wall_gain = resolvent._interpolate_entry(
            branch_values, angle, grid
        )
        same_scale_criterion = resolvent.H1_ENTRY_GAIN**2 * (
            return_gain**2 + wall_gain**2
        )
        conditionally_paid_criterion = resolvent.H1_ENTRY_GAIN**2 * (
            return_gain**2
            + (resolvent.CUBIC_SPLIT_FACTOR * wall_gain) ** 2
        )
        angle_rows.append(
            {
                "angle": float(angle),
                "axial_patch_return_gain": float(return_gain),
                "wall_deformation_gain": float(wall_gain),
                "same_scale_criterion": float(same_scale_criterion),
                "conditionally_split_paid_criterion": float(
                    conditionally_paid_criterion
                ),
            }
        )
    worst = max(angle_rows, key=lambda row: row["same_scale_criterion"])
    best = min(angle_rows, key=lambda row: row["same_scale_criterion"])
    return {
        "strip_half_width": half_width,
        "spacing": grid["spacing"],
        "x_half_width": grid["x_half_width"],
        "interior_state_count": generator.shape[0],
        "time_step_scale": time_step_scale,
        "principal_killed_rate_pilot": killed_rate,
        "wall_deformation_moment_finite": True,
        "maximum_same_scale_criterion": worst["same_scale_criterion"],
        "worst_entry_angle": worst["angle"],
        "worst_angle_row": worst,
        "angle_rows": angle_rows,
        "minimum_same_scale_criterion_over_angles": best[
            "same_scale_criterion"
        ],
        "maximum_conditionally_split_paid_criterion": max(
            row["conditionally_split_paid_criterion"] for row in angle_rows
        ),
        "unweighted_return_resolvent_recovery_error": recovery_error,
        "terminal_inner_flux_maximum": terminal_flux,
    }


def audit() -> dict[str, object]:
    resolvent = _load_module(
        "neutral_strip_branch_resolvent_pilot.py",
        "neutral_strip_resolvent_for_width_sweep",
    )
    axial = _load_module(
        "neutral_strip_axial_patch_branch_pilot.py",
        "neutral_strip_axial_for_width_sweep",
    )
    broad_widths = (
        2.02,
        2.05,
        2.1,
        2.15,
        2.2,
        2.25,
        2.3,
        2.4,
        2.5,
        2.6,
        2.8,
        3.0,
        3.25,
        3.5,
    )
    broad_rows = [
        _row(resolvent, axial, half_width) for half_width in broad_widths
    ]
    local_widths = (2.24, 2.26, 2.28, 2.30, 2.32, 2.34, 2.36)
    local_rows = [
        _row(resolvent, axial, half_width) for half_width in local_widths
    ]
    all_width_rows = broad_rows + local_rows
    optimum = min(
        all_width_rows, key=lambda row: row["maximum_same_scale_criterion"]
    )

    mesh_rows = [
        _row(resolvent, axial, 2.3, target_spacing=spacing)
        for spacing in (0.1, 0.075, 0.06)
    ]
    mesh_spread = max(
        row["maximum_same_scale_criterion"] for row in mesh_rows
    ) - min(row["maximum_same_scale_criterion"] for row in mesh_rows)
    time_rows = [
        _row(resolvent, axial, 2.3, time_step_scale=scale)
        for scale in (1.0, 0.5)
    ]
    time_change = abs(
        time_rows[0]["maximum_same_scale_criterion"]
        - time_rows[1]["maximum_same_scale_criterion"]
    )
    x_rows = [
        _row(resolvent, axial, 2.3, x_half_width=width)
        for width in (4.2, 4.4, 5.0)
    ]
    x_spread = max(
        row["maximum_same_scale_criterion"] for row in x_rows
    ) - min(row["maximum_same_scale_criterion"] for row in x_rows)

    result: dict[str, object] = {
        "affine_stress_parameter": "rho=0",
        "wall_branch_policy": "same scale; no cubic shrink factor",
        "same_scale_criterion": "g_H^2[k_R(Y)^2+k_S(Y)^2]",
        "broad_width_rows": broad_rows,
        "local_width_rows": local_rows,
        "sampled_optimum": optimum,
        "mesh_refinement_rows": mesh_rows,
        "time_refinement_rows": time_rows,
        "x_truncation_rows": x_rows,
        "maximum_mesh_refinement_spread": mesh_spread,
        "time_refinement_change": time_change,
        "maximum_x_truncation_spread": x_spread,
        "same_scale_width_sweep_finds_closure": bool(
            optimum["maximum_same_scale_criterion"] < 1.0
        ),
        "conditional_split_paid_rows_are_not_used_for_conclusion": True,
        "width_tuning_repairs_current_architecture": False,
        "full_Navier_Stokes_wall_gate_closed": False,
        "interpretation": (
            "narrow strips are dominated by rapid wall exit, while wider "
            "strips accumulate the exp(tau) wall moment. The fitted static "
            "rho=0 sweep has an interior optimum but remains above one"
        ),
        "scope_guard": (
            "this is a boundary-fitted finite-state pilot, not a rigorous "
            "continuum minimization over Y. Failure at rho=0 is enough to "
            "falsify the sampled uniform affine route, but a proof-level "
            "no-go requires certified discretization and continuous-width "
            "bounds"
        ),
        "next_gate": (
            "do not tune the same strip further. Either rederive a legitimate "
            "geometry-triggered scale transition or seek a different storage "
            "boundary whose unpaid wall operator is contractive"
        ),
    }
    checks = (
        all(row["wall_deformation_moment_finite"] for row in all_width_rows),
        not result["same_scale_width_sweep_finds_closure"],
        optimum["maximum_same_scale_criterion"] > 1.12,
        optimum["maximum_same_scale_criterion"] < 1.2,
        2.2 <= optimum["strip_half_width"] <= 2.4,
        mesh_spread < 0.03,
        time_change < 0.003,
        x_spread < 0.01,
        not result["width_tuning_repairs_current_architecture"],
        not result["full_Navier_Stokes_wall_gate_closed"],
    )
    result["all_positive_same_scale_width_checks_pass"] = all(checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
