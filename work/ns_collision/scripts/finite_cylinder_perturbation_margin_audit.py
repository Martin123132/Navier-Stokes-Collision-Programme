"""Calibrate adverse-potential margins in the finite-cylinder visit model."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
from scipy.special import hyp1f1, iv, jv, kv, yv


def _load_finite_cylinder_module():
    script = Path(__file__).resolve().with_name(
        "finite_cylinder_mode_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "finite_cylinder_for_perturbation", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _axial_basis(
    reynolds: float,
    half_height: float,
    grid_points: int,
    mode_count: int,
) -> dict[str, np.ndarray | int]:
    spacing = 2.0 * half_height / (grid_points + 1)
    axial_grid = -half_height + spacing * np.arange(1, grid_points + 1)
    diagonal = (
        2.0 / spacing**2
        + reynolds**2 * axial_grid**2
        - reynolds
    )
    off_diagonal = np.full(grid_points - 1, -1.0 / spacing**2)
    eigenvalues, eigenvectors = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, mode_count - 1),
    )
    if np.min(eigenvalues) <= 0.0:
        raise RuntimeError("axial discretization produced a nonpositive mode")

    physical_factor = np.exp(reynolds * axial_grid**2 / 2.0)
    coefficients = (
        np.exp(-reynolds * axial_grid**2 / 2.0) @ eigenvectors
    )
    centre_index = int(np.argmin(np.abs(axial_grid)))
    return {
        "axial_grid": axial_grid,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "physical_factor": physical_factor,
        "coefficients": coefficients,
        "centre_index": centre_index,
    }


def _radial_mode_gain(
    reynolds: float,
    buffer_ratio: float,
    axial_killing: float,
    core_potential: float,
    shell_potential: float,
) -> float:
    """Return the outer-to-inner transfer for one axial mode."""
    kummer_parameter = 1.0 + (
        core_potential - axial_killing
    ) / (2.0 * reynolds)
    core_value = hyp1f1(
        kummer_parameter, 1.0, -reynolds / 2.0
    )
    if not math.isfinite(float(core_value)) or abs(core_value) < 1.0e-14:
        return math.inf
    core_slope = (
        -reynolds
        * kummer_parameter
        * hyp1f1(
            kummer_parameter + 1.0,
            2.0,
            -reynolds / 2.0,
        )
        / core_value
    )

    effective_killing = axial_killing - shell_potential
    if abs(effective_killing) < 1.0e-12:
        outer_transfer = 1.0 + core_slope * math.log(buffer_ratio)
    elif effective_killing > 0.0:
        root = math.sqrt(effective_killing)
        i0_at_one = iv(0, root)
        i1_at_one = iv(1, root)
        k0_at_one = kv(0, root)
        k1_at_one = kv(1, root)
        coefficient_i = root * k1_at_one + core_slope * k0_at_one
        coefficient_k = root * i1_at_one - core_slope * i0_at_one
        outer_transfer = (
            coefficient_i * iv(0, root * buffer_ratio)
            + coefficient_k * kv(0, root * buffer_ratio)
        )
    else:
        root = math.sqrt(-effective_killing)
        interface_matrix = np.array(
            [
                [jv(0, root), yv(0, root)],
                [-root * jv(1, root), -root * yv(1, root)],
            ]
        )
        coefficient_j, coefficient_y = np.linalg.solve(
            interface_matrix, np.array([1.0, core_slope])
        )
        outer_transfer = (
            coefficient_j * jv(0, root * buffer_ratio)
            + coefficient_y * yv(0, root * buffer_ratio)
        )

    if not math.isfinite(float(outer_transfer)) or outer_transfer <= 0.0:
        return math.inf
    return float(1.0 / outer_transfer)


def _visit(
    basis: dict[str, np.ndarray | int],
    reynolds: float,
    buffer_ratio: float,
    core_potential: float,
    shell_potential: float,
) -> dict[str, float | bool]:
    eigenvalues = np.asarray(basis["eigenvalues"])
    radial_gains = np.array(
        [
            _radial_mode_gain(
                reynolds,
                buffer_ratio,
                float(eigenvalue),
                core_potential,
                shell_potential,
            )
            for eigenvalue in eigenvalues
        ]
    )
    if not np.all(np.isfinite(radial_gains)):
        return {
            "centre_visit_gain": math.inf,
            "maximum_visit_gain": math.inf,
            "maximum_gain_axial_location": math.nan,
            "maximum_occurs_at_centre": False,
        }

    eigenvectors = np.asarray(basis["eigenvectors"])
    coefficients = np.asarray(basis["coefficients"])
    physical_factor = np.asarray(basis["physical_factor"])
    axial_grid = np.asarray(basis["axial_grid"])
    centre_index = int(basis["centre_index"])
    profile = physical_factor * (
        eigenvectors @ (coefficients * radial_gains)
    )
    maximum_index = int(np.argmax(profile))
    return {
        "centre_visit_gain": float(profile[centre_index]),
        "maximum_visit_gain": float(profile[maximum_index]),
        "maximum_gain_axial_location": float(axial_grid[maximum_index]),
        "maximum_occurs_at_centre": bool(maximum_index == centre_index),
    }


def _renewal_quantities(
    visit_gain: float, reynolds: float, buffer_ratio: float
) -> dict[str, float]:
    pair_visit = visit_gain**2
    pair_return = buffer_ratio ** (-2.0)
    true_split = math.exp(reynolds * 3.0 / 24.0) / 4.0
    criterion = pair_visit * (true_split + pair_return)
    denominator = 1.0 - pair_return * pair_visit
    generation_factor = (
        math.inf
        if denominator <= 0.0
        else true_split * pair_visit / denominator
    )
    return {
        "pair_visit_gain": pair_visit,
        "pair_return_factor": pair_return,
        "true_split_factor": true_split,
        "complete_generation_criterion": criterion,
        "renewed_generation_factor": generation_factor,
    }


def _criterion(
    basis: dict[str, np.ndarray | int],
    reynolds: float,
    buffer_ratio: float,
    core_potential: float,
    shell_potential: float,
) -> float:
    visit = _visit(
        basis,
        reynolds,
        buffer_ratio,
        core_potential,
        shell_potential,
    )
    gain = float(visit["maximum_visit_gain"])
    return float(
        _renewal_quantities(gain, reynolds, buffer_ratio)[
            "complete_generation_criterion"
        ]
    )


def _critical_potential(
    basis: dict[str, np.ndarray | int],
    reynolds: float,
    buffer_ratio: float,
    location: str,
) -> float:
    if location not in {"core", "shell", "uniform"}:
        raise ValueError(f"unknown perturbation location: {location}")

    def equation(potential: float) -> float:
        core_potential = potential if location in {"core", "uniform"} else 0.0
        shell_potential = (
            potential if location in {"shell", "uniform"} else 0.0
        )
        value = _criterion(
            basis,
            reynolds,
            buffer_ratio,
            core_potential,
            shell_potential,
        )
        return 1.0e12 if not math.isfinite(value) else value - 1.0

    if equation(0.0) >= 0.0:
        return 0.0
    upper = 0.125
    while equation(upper) < 0.0:
        upper *= 2.0
        if upper > 1.0e4:
            raise RuntimeError("failed to bracket perturbation threshold")
    return float(brentq(equation, 0.0, upper, xtol=1.0e-10))


def _calibrated_l3_over_2_mass(
    potential: float,
    half_height: float,
    buffer_ratio: float,
    location: str,
) -> float:
    if location == "core":
        dimensionless_volume = 2.0 * math.pi * half_height
    elif location == "shell":
        dimensionless_volume = (
            2.0
            * math.pi
            * half_height
            * (buffer_ratio**2 - 1.0)
        )
    elif location == "uniform":
        dimensionless_volume = (
            2.0 * math.pi * half_height * buffer_ratio**2
        )
    else:
        raise ValueError(f"unknown perturbation location: {location}")
    return potential * dimensionless_volume ** (2.0 / 3.0)


def _geometry_row(
    reynolds: float,
    half_height: float,
    buffer_ratio: float,
    coarse_grid_points: int = 401,
    coarse_mode_count: int = 61,
    fine_grid_points: int = 801,
    fine_mode_count: int = 81,
) -> dict[str, object]:
    coarse_basis = _axial_basis(
        reynolds,
        half_height,
        coarse_grid_points,
        coarse_mode_count,
    )
    fine_basis = _axial_basis(
        reynolds,
        half_height,
        fine_grid_points,
        fine_mode_count,
    )
    base_visit = _visit(
        fine_basis, reynolds, buffer_ratio, 0.0, 0.0
    )
    base_renewal = _renewal_quantities(
        float(base_visit["maximum_visit_gain"]),
        reynolds,
        buffer_ratio,
    )

    thresholds: dict[str, float] = {}
    coarse_thresholds: dict[str, float] = {}
    threshold_residuals: dict[str, float] = {}
    calibrated_masses: dict[str, float] = {}
    maximum_at_threshold: dict[str, bool] = {}
    for location in ("core", "shell", "uniform"):
        coarse_threshold = _critical_potential(
            coarse_basis, reynolds, buffer_ratio, location
        )
        fine_threshold = _critical_potential(
            fine_basis, reynolds, buffer_ratio, location
        )
        core_potential = (
            fine_threshold if location in {"core", "uniform"} else 0.0
        )
        shell_potential = (
            fine_threshold if location in {"shell", "uniform"} else 0.0
        )
        threshold_visit = _visit(
            fine_basis,
            reynolds,
            buffer_ratio,
            core_potential,
            shell_potential,
        )
        threshold_criterion = _criterion(
            fine_basis,
            reynolds,
            buffer_ratio,
            core_potential,
            shell_potential,
        )
        thresholds[location] = fine_threshold
        coarse_thresholds[location] = coarse_threshold
        threshold_residuals[location] = abs(threshold_criterion - 1.0)
        calibrated_masses[location] = _calibrated_l3_over_2_mass(
            fine_threshold, half_height, buffer_ratio, location
        )
        maximum_at_threshold[location] = bool(
            threshold_visit["maximum_occurs_at_centre"]
        )

    return {
        "R_star": reynolds,
        "half_height_over_L": half_height,
        "full_height_over_L": 2.0 * half_height,
        "buffer_ratio": buffer_ratio,
        "base_one_history_visit_gain": base_visit["maximum_visit_gain"],
        "base_maximum_gain_axial_location": base_visit[
            "maximum_gain_axial_location"
        ],
        "base_maximum_occurs_at_centre": base_visit[
            "maximum_occurs_at_centre"
        ],
        **base_renewal,
        "base_log_criterion_margin": -math.log(
            base_renewal["complete_generation_criterion"]
        ),
        "critical_dimensionless_potentials": thresholds,
        "coarse_critical_dimensionless_potentials": coarse_thresholds,
        "critical_potential_refinement_changes": {
            location: abs(thresholds[location] - coarse_thresholds[location])
            for location in thresholds
        },
        "threshold_criterion_residuals": threshold_residuals,
        "calibrated_critical_L3_over_2_mass_over_nu": calibrated_masses,
        "maximum_occurs_at_centre_at_threshold": maximum_at_threshold,
    }


def audit() -> dict[str, object]:
    buffer_ratio = 2.0
    geometries = (
        (0.5, 1.5),
        (0.5, 1.75),
        (0.5, 2.0),
        (1.0, 1.0),
        (1.0, 1.2),
    )
    rows = [
        _geometry_row(reynolds, half_height, buffer_ratio)
        for reynolds, half_height in geometries
    ]

    finite_cylinder = _load_finite_cylinder_module()
    axial = finite_cylinder._load_axial_module()
    legacy_visit = finite_cylinder._full_mode_visit(
        axial,
        reynolds=1.0,
        half_height=1.2,
        buffer_ratio=buffer_ratio,
        grid_points=401,
        mode_count=61,
    )
    comparison_basis = _axial_basis(1.0, 1.2, 401, 61)
    generalized_visit = _visit(
        comparison_basis, 1.0, buffer_ratio, 0.0, 0.0
    )
    zero_potential_residual = abs(
        float(legacy_visit["maximum_visit_gain"])
        - float(generalized_visit["maximum_visit_gain"])
    )

    all_refinement_changes = [
        change
        for row in rows
        for change in row["critical_potential_refinement_changes"].values()
    ]
    all_threshold_residuals = [
        residual
        for row in rows
        for residual in row["threshold_criterion_residuals"].values()
    ]
    result: dict[str, object] = {
        "core_mode_equation": (
            "u_rr+u_r/r+R_star*r*u_r+"
            "(2*R_star+delta_core-zeta_n)*u=0"
        ),
        "core_kummer_parameter": (
            "1+(delta_core-zeta_n)/(2*R_star)"
        ),
        "shell_mode_equation": (
            "u_rr+u_r/r+(delta_shell-zeta_n)*u=0"
        ),
        "constant_potential_scaling": "delta=q*L^2/nu",
        "calibrated_mass_scaling": (
            "||q||_(3/2)/nu=delta*|Omega/L^3|^(2/3)"
        ),
        "geometry_rows": rows,
        "zero_potential_legacy_gain_residual": zero_potential_residual,
        "generalized_transfer_recovers_legacy_model": bool(
            zero_potential_residual < 1.0e-11
        ),
        "all_base_geometries_close": all(
            row["complete_generation_criterion"] < 1.0 for row in rows
        ),
        "all_critical_potentials_are_positive": all(
            threshold > 0.0
            for row in rows
            for threshold in row[
                "critical_dimensionless_potentials"
            ].values()
        ),
        "all_thresholds_reproduce_generation_boundary": bool(
            max(all_threshold_residuals) < 1.0e-8
        ),
        "all_threshold_maxima_occur_at_centre": all(
            all(row["maximum_occurs_at_centre_at_threshold"].values())
            for row in rows
        ),
        "maximum_threshold_refinement_change": max(
            all_refinement_changes
        ),
        "thresholds_converge_under_refinement": bool(
            max(all_refinement_changes) < 2.0e-4
        ),
        "constant_potential_scope": (
            "these are exact margins for nonnegative constants on the core "
            "or shell, not rearrangement bounds for arbitrary potentials"
        ),
        "remaining_PDE_gate": (
            "bound the positive non-affine Navier-Stokes error by a valid "
            "finite-cylinder Feynman-Kac or critical-form operator estimate"
        ),
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
