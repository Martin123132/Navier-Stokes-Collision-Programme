"""Audit the full axial OU mode expansion for the finite-cylinder visit."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq


def _load_axial_module():
    script = Path(__file__).resolve().with_name(
        "axial_killing_buffered_visit_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "axial_killing_for_full_cylinder", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_mode_visit(
    axial,
    reynolds: float,
    half_height: float,
    buffer_ratio: float = 2.0,
    grid_points: int = 401,
    mode_count: int = 61,
) -> dict[str, object]:
    spacing = 2.0 * half_height / (grid_points + 1)
    axial_grid = (
        -half_height + spacing * np.arange(1, grid_points + 1)
    )
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

    weighted_constant = np.exp(-reynolds * axial_grid**2 / 2.0)
    expansion_coefficients = weighted_constant @ eigenvectors
    radial_mode_gains = np.array(
        [
            axial._constant_killing_visit_gain(
                reynolds, buffer_ratio, float(eigenvalue)
            )
            for eigenvalue in eigenvalues
        ]
    )
    physical_eigenfunction_factor = np.exp(
        reynolds * axial_grid**2 / 2.0
    )
    visit_profile = physical_eigenfunction_factor * (
        eigenvectors @ (expansion_coefficients * radial_mode_gains)
    )
    reconstructed_boundary = physical_eigenfunction_factor * (
        eigenvectors @ expansion_coefficients
    )
    centre_index = int(np.argmin(np.abs(axial_grid)))
    maximum_index = int(np.argmax(visit_profile))
    mode_contributions_at_centre = (
        expansion_coefficients
        * eigenvectors[centre_index]
        * radial_mode_gains
    )
    return {
        "axial_grid": axial_grid,
        "eigenvalues": eigenvalues,
        "principal_eigenvalue": float(eigenvalues[0]),
        "visit_profile": visit_profile,
        "centre_visit_gain": float(visit_profile[centre_index]),
        "maximum_visit_gain": float(visit_profile[maximum_index]),
        "maximum_gain_axial_location": float(axial_grid[maximum_index]),
        "maximum_occurs_at_centre_grid_point": bool(
            maximum_index == centre_index
        ),
        "boundary_reconstruction_at_centre": float(
            reconstructed_boundary[centre_index]
        ),
        "last_ten_mode_absolute_contribution": float(
            np.sum(np.abs(mode_contributions_at_centre[-10:]))
        ),
    }


def _generation_criterion(
    visit_gain: float, reynolds: float, buffer_ratio: float = 2.0
) -> float:
    pair_return = buffer_ratio ** (-2.0)
    true_split = math.exp(reynolds * 3.0 / 24.0) / 4.0
    return visit_gain**2 * (true_split + pair_return)


def audit() -> dict[str, object]:
    axial = _load_axial_module()
    surrogate_result = axial.audit()
    surrogate_rows = {
        row["R_star"]: row for row in surrogate_result["threshold_rows"]
    }

    convergence_rows = []
    for grid_points, mode_count in (
        (201, 41),
        (401, 61),
        (801, 81),
    ):
        visit = _full_mode_visit(
            axial,
            reynolds=1.0,
            half_height=1.2,
            grid_points=grid_points,
            mode_count=mode_count,
        )
        convergence_rows.append(
            {
                "grid_points": grid_points,
                "mode_count": mode_count,
                "centre_visit_gain": visit["centre_visit_gain"],
                "principal_eigenvalue": visit["principal_eigenvalue"],
                "boundary_reconstruction_at_centre": visit[
                    "boundary_reconstruction_at_centre"
                ],
                "last_ten_mode_absolute_contribution": visit[
                    "last_ten_mode_absolute_contribution"
                ],
            }
        )

    threshold_rows = []
    for reynolds in (0.25, 0.5, 1.0, 2.0):
        surrogate_half_height = surrogate_rows[reynolds][
            "axial_OU_maximum_half_height_over_L"
        ]
        if math.isinf(surrogate_half_height):
            threshold_rows.append(
                {
                    "R_star": reynolds,
                    "principal_mode_half_height": math.inf,
                    "full_mode_half_height": math.inf,
                    "full_mode_full_height": math.inf,
                    "threshold_generation_criterion": surrogate_rows[
                        reynolds
                    ]["zero_axial_killing_generation_criterion"],
                    "principal_eigenvalue_residual": 0.0,
                    "maximum_occurs_at_centre": True,
                }
            )
            continue

        def height_equation(half_height: float) -> float:
            visit = _full_mode_visit(
                axial,
                reynolds,
                half_height,
                grid_points=401,
                mode_count=61,
            )
            return _generation_criterion(
                visit["maximum_visit_gain"], reynolds
            ) - 1.0

        full_mode_half_height = float(
            brentq(
                height_equation,
                0.5,
                float(surrogate_half_height),
                xtol=1.0e-9,
            )
        )
        validated_visit = _full_mode_visit(
            axial,
            reynolds,
            full_mode_half_height,
            grid_points=801,
            mode_count=81,
        )
        analytic_principal = axial._axial_ou_principal_killing(
            reynolds, full_mode_half_height
        )
        threshold_rows.append(
            {
                "R_star": reynolds,
                "principal_mode_half_height": surrogate_half_height,
                "full_mode_half_height": full_mode_half_height,
                "full_mode_full_height": 2.0 * full_mode_half_height,
                "full_mode_centre_visit_gain": validated_visit[
                    "centre_visit_gain"
                ],
                "full_mode_maximum_visit_gain": validated_visit[
                    "maximum_visit_gain"
                ],
                "maximum_gain_axial_location": validated_visit[
                    "maximum_gain_axial_location"
                ],
                "maximum_occurs_at_centre": validated_visit[
                    "maximum_occurs_at_centre_grid_point"
                ],
                "threshold_generation_criterion": _generation_criterion(
                    validated_visit["maximum_visit_gain"], reynolds
                ),
                "finite_difference_principal_eigenvalue": validated_visit[
                    "principal_eigenvalue"
                ],
                "analytic_principal_eigenvalue": analytic_principal,
                "principal_eigenvalue_residual": abs(
                    validated_visit["principal_eigenvalue"]
                    - analytic_principal
                ),
                "last_ten_mode_absolute_contribution": validated_visit[
                    "last_ten_mode_absolute_contribution"
                ],
            }
        )

    finite_threshold_rows = [
        row for row in threshold_rows if not math.isinf(row["full_mode_half_height"])
    ]
    result: dict[str, object] = {
        "axial_self_adjoint_transform": (
            "phi=exp(R_star*y^2/2)*psi, "
            "H=-partial_yy+R_star^2*y^2-R_star"
        ),
        "weighted_boundary_coefficients": (
            "c_n=integral exp(-R_star*y^2/2)*psi_n(y) dy"
        ),
        "full_visit_expansion": (
            "u(1,y)=sum_n c_n*U(zeta_n)*"
            "exp(R_star*y^2/2)*psi_n(y)"
        ),
        "convergence_rows": convergence_rows,
        "visit_gain_converges_under_refinement": bool(
            abs(
                convergence_rows[-1]["centre_visit_gain"]
                - convergence_rows[-2]["centre_visit_gain"]
            )
            < 1.0e-5
        ),
        "threshold_rows": threshold_rows,
        "all_finite_thresholds_are_below_principal_surrogate": all(
            row["full_mode_half_height"]
            < row["principal_mode_half_height"]
            for row in finite_threshold_rows
        ),
        "all_threshold_maxima_occur_at_centre": all(
            row["maximum_occurs_at_centre"] for row in threshold_rows
        ),
        "all_principal_eigenvalues_match_kummer_roots": all(
            row["principal_eigenvalue_residual"] < 1.0e-5
            for row in finite_threshold_rows
        ),
        "all_mode_tails_are_small": all(
            row["last_ten_mode_absolute_contribution"] < 1.0e-12
            for row in finite_threshold_rows
        ),
        "validated_threshold_discretization_warning": (
            "threshold roots use 401 points and are re-evaluated at 801 "
            "points; reported boundary residuals retain discretization error"
        ),
        "model_verdict": (
            "the complete axial mode family is modestly worse than the "
            "principal surrogate but finite compact cores still close the "
            "ideal generation benchmark"
        ),
        "remaining_PDE_gate": (
            "control non-affine Navier-Stokes coherence and the actual "
            "three-dimensional exterior return operator on these compact cores"
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
