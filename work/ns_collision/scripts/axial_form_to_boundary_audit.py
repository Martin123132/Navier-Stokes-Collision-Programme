"""Audit critical-form control of the Gaussian L2 visit multiplier."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def _load_axial_module():
    script = Path(__file__).resolve().with_name(
        "axial_killing_buffered_visit_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "axial_killing_for_form_to_boundary", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _axial_matrix(
    reynolds: float, half_height: float, grid_points: int
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    spacing = 2.0 * half_height / (grid_points + 1)
    grid = -half_height + spacing * np.arange(1, grid_points + 1)
    diagonal = (
        2.0 / spacing**2 + reynolds**2 * grid**2 - reynolds
    )
    off_diagonal = np.full(grid_points - 1, -1.0 / spacing**2)
    return grid, spacing, diagonal, off_diagonal


def _principal_eigenvalue(
    diagonal: np.ndarray,
    off_diagonal: np.ndarray,
    potential: np.ndarray,
) -> float:
    return float(
        eigh_tridiagonal(
            diagonal - potential,
            off_diagonal,
            select="i",
            select_range=(0, 0),
            eigvals_only=True,
        )[0]
    )


def _relative_form_bound(
    diagonal: np.ndarray,
    off_diagonal: np.ndarray,
    potential: np.ndarray,
) -> float:
    hamiltonian = diags(
        (off_diagonal, diagonal, off_diagonal),
        offsets=(-1, 0, 1),
        format="csc",
    )
    multiplication = diags(potential, offsets=0, format="csc")
    eigenvalue = eigsh(
        multiplication,
        k=1,
        M=hamiltonian,
        which="LA",
        return_eigenvectors=False,
        tol=1.0e-11,
    )[0]
    return float(eigenvalue)


def _generation_criterion(
    axial,
    reynolds: float,
    axial_killing: float,
    buffer_ratio: float = 2.0,
) -> float:
    visit_gain = axial._constant_killing_visit_gain(
        reynolds, buffer_ratio, axial_killing
    )
    pair_return = buffer_ratio ** (-2.0)
    true_split = math.exp(reynolds * 3.0 / 24.0) / 4.0
    return visit_gain**2 * (true_split + pair_return)


def _required_axial_killing(
    axial, reynolds: float, buffer_ratio: float = 2.0
) -> float:
    def equation(axial_killing: float) -> float:
        return (
            _generation_criterion(
                axial, reynolds, axial_killing, buffer_ratio
            )
            - 1.0
        )

    if equation(0.0) < 0.0:
        return 0.0
    upper = 1.0
    while equation(upper) > 0.0:
        upper *= 2.0
    return float(brentq(equation, 0.0, upper, xtol=1.0e-13))


def _critical_mass(
    potential: np.ndarray,
    spacing: float,
    buffer_ratio: float,
) -> float:
    axial_integral = spacing * float(np.sum(potential**1.5))
    return (
        math.pi * buffer_ratio**2 * axial_integral
    ) ** (2.0 / 3.0)


def _profile_rows(
    axial,
    reynolds: float,
    half_height: float,
    buffer_ratio: float,
    grid_points: int = 401,
) -> list[dict[str, object]]:
    grid, spacing, diagonal, off_diagonal = _axial_matrix(
        reynolds, half_height, grid_points
    )
    unperturbed_principal = _principal_eigenvalue(
        diagonal, off_diagonal, np.zeros(grid_points)
    )
    required_killing = _required_axial_killing(
        axial, reynolds, buffer_ratio
    )
    universal_critical_alpha = (
        1.0 - required_killing / unperturbed_principal
    )
    target_alpha = 0.5 * universal_critical_alpha

    profiles = {
        "constant": np.ones(grid_points),
        "central_Gaussian_width_0.15h": np.exp(
            -(grid / (0.15 * half_height)) ** 2
        ),
        "central_Gaussian_width_0.35h": np.exp(
            -(grid / (0.35 * half_height)) ** 2
        ),
        "twin_cap_Gaussians": (
            np.exp(
                -((grid - 0.75 * half_height) / (0.1 * half_height))
                ** 2
            )
            + np.exp(
                -((grid + 0.75 * half_height) / (0.1 * half_height))
                ** 2
            )
        ),
    }

    rows = []
    for name, shape in profiles.items():
        unit_form_bound = _relative_form_bound(
            diagonal, off_diagonal, shape
        )

        def threshold_equation(scale: float) -> float:
            return (
                _principal_eigenvalue(
                    diagonal, off_diagonal, scale * shape
                )
                - required_killing
            )

        upper = 1.0
        while threshold_equation(upper) > 0.0:
            upper *= 2.0
        threshold_scale = float(
            brentq(threshold_equation, 0.0, upper, xtol=1.0e-11)
        )
        threshold_form_bound = threshold_scale * unit_form_bound
        threshold_potential = threshold_scale * shape

        half_budget_scale = target_alpha / unit_form_bound
        half_budget_potential = half_budget_scale * shape
        half_budget_principal = _principal_eigenvalue(
            diagonal, off_diagonal, half_budget_potential
        )
        form_lower_bound = (
            1.0 - target_alpha
        ) * unperturbed_principal
        actual_criterion = _generation_criterion(
            axial,
            reynolds,
            half_budget_principal,
            buffer_ratio,
        )
        form_bound_criterion = _generation_criterion(
            axial,
            reynolds,
            form_lower_bound,
            buffer_ratio,
        )
        rows.append(
            {
                "profile": name,
                "unit_profile_relative_form_bound": unit_form_bound,
                "critical_amplitude": threshold_scale,
                "critical_relative_form_bound": threshold_form_bound,
                "universal_sufficient_relative_form_bound": (
                    universal_critical_alpha
                ),
                "critical_L3_over_2_mass_over_nu": _critical_mass(
                    threshold_potential, spacing, buffer_ratio
                ),
                "half_budget_target_alpha": target_alpha,
                "half_budget_actual_principal_eigenvalue": (
                    half_budget_principal
                ),
                "half_budget_form_lower_bound": form_lower_bound,
                "half_budget_actual_generation_criterion": (
                    actual_criterion
                ),
                "half_budget_form_generation_bound": (
                    form_bound_criterion
                ),
                "form_lower_bound_holds": bool(
                    half_budget_principal
                    >= form_lower_bound - 1.0e-9
                ),
                "generation_bound_holds": bool(
                    actual_criterion <= form_bound_criterion + 1.0e-9
                ),
                "actual_critical_alpha_exceeds_universal_budget": bool(
                    threshold_form_bound
                    >= universal_critical_alpha - 1.0e-8
                ),
            }
        )
    return rows


def audit() -> dict[str, object]:
    axial = _load_axial_module()
    buffer_ratio = 2.0
    geometries = (
        (0.5, 1.5),
        (0.5, 1.75),
        (0.5, 2.0),
        (1.0, 1.0),
        (1.0, 1.2),
    )
    geometry_rows = []
    for reynolds, half_height in geometries:
        _, _, diagonal, off_diagonal = _axial_matrix(
            reynolds, half_height, grid_points=801
        )
        principal = _principal_eigenvalue(
            diagonal, off_diagonal, np.zeros(801)
        )
        required_killing = _required_axial_killing(
            axial, reynolds, buffer_ratio
        )
        critical_alpha = 1.0 - required_killing / principal
        criterion_at_bound = _generation_criterion(
            axial,
            reynolds,
            (1.0 - critical_alpha) * principal,
            buffer_ratio,
        )
        geometry_rows.append(
            {
                "R_star": reynolds,
                "half_height_over_L": half_height,
                "unperturbed_principal_axial_eigenvalue": principal,
                "required_principal_axial_eigenvalue": required_killing,
                "critical_relative_form_bound": critical_alpha,
                "threshold_generation_criterion": criterion_at_bound,
                "threshold_is_reproduced": bool(
                    abs(criterion_at_bound - 1.0) < 1.0e-10
                ),
            }
        )

    profile_rows = _profile_rows(
        axial,
        reynolds=0.5,
        half_height=1.5,
        buffer_ratio=buffer_ratio,
    )
    constant_row = profile_rows[0]
    result: dict[str, object] = {
        "perturbed_axial_operator": (
            "H_q=-partial_yy+R_star^2*y^2-R_star-q_+(y)"
        ),
        "relative_form_hypothesis": (
            "<v,q_+v><=alpha<v,Hv>, equivalently "
            "||H^(-1/2)q_+H^(-1/2)||<=alpha"
        ),
        "principal_eigenvalue_bound": (
            "zeta_0(q)>= (1-alpha) zeta_0(0)"
        ),
        "boundary_operator_bound": (
            "||B_q||_2=U(zeta_0(q))<="
            "U((1-alpha)zeta_0(0))"
        ),
        "geometry_rows": geometry_rows,
        "all_geometry_thresholds_are_reproduced": all(
            row["threshold_is_reproduced"] for row in geometry_rows
        ),
        "working_geometry_profile_rows": profile_rows,
        "all_profile_form_and_generation_bounds_hold": all(
            row["form_lower_bound_holds"]
            and row["generation_bound_holds"]
            and row[
                "actual_critical_alpha_exceeds_universal_budget"
            ]
            for row in profile_rows
        ),
        "constant_profile_saturates_form_bound": bool(
            abs(
                constant_row["critical_relative_form_bound"]
                - constant_row[
                    "universal_sufficient_relative_form_bound"
                ]
            )
            < 1.0e-7
        ),
        "restricted_scope": (
            "q_+(y) is axial and acts throughout the radial cylinder, so "
            "separation survives; a general three-dimensional perturbation "
            "requires a Poisson/trace form estimate"
        ),
        "remaining_general_gate": (
            "extend the form-to-boundary estimate from this exactly "
            "separable class to non-axisymmetric q_+(rho,theta,y) and "
            "moving cell geometry"
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
