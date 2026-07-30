"""Audit a Lipschitz minorant of the adaptive diffusion radius."""

from __future__ import annotations

import json

import numpy as np
import sympy as sp


def _lipschitz_minorant(
    coordinates: np.ndarray, raw_radius: np.ndarray, kappa: float
) -> np.ndarray:
    distance = np.abs(coordinates[:, None] - coordinates[None, :])
    return np.min(raw_radius[None, :] + kappa * distance, axis=1)


def audit() -> dict[str, object]:
    target_reynolds = 2.0
    viscosity = 1.0
    kappa = 0.125
    enlargement = 2.0
    coordinates = np.linspace(0.0, 4.0, 401)

    early_envelope = (
        1.0
        + 8.0 * np.exp(-((coordinates - 1.0) / 0.28) ** 2)
        + 3.0 * np.exp(-((coordinates - 2.6) / 0.45) ** 2)
    )
    later_candidate = (
        1.0
        + 20.0 * np.exp(-((coordinates - 3.1) / 0.22) ** 2)
    )
    later_envelope = np.maximum(early_envelope, later_candidate)
    current_strain = later_envelope * (
        0.55 + 0.4 * np.cos(1.7 * coordinates) ** 2
    )

    early_raw_radius = np.sqrt(
        target_reynolds * viscosity / early_envelope
    )
    later_raw_radius = np.sqrt(
        target_reynolds * viscosity / later_envelope
    )
    early_radius = _lipschitz_minorant(
        coordinates, early_raw_radius, kappa
    )
    later_radius = _lipschitz_minorant(
        coordinates, later_raw_radius, kappa
    )

    early_reference_envelope = (
        target_reynolds * viscosity / early_radius**2
    )
    later_reference_envelope = (
        target_reynolds * viscosity / later_radius**2
    )
    actual_local_reynolds = (
        current_strain * later_radius**2 / viscosity
    )

    grid_spacing = coordinates[1] - coordinates[0]
    maximum_adjacent_slope = float(
        np.max(np.abs(np.diff(later_radius))) / grid_spacing
    )
    distance = np.abs(coordinates[:, None] - coordinates[None, :])
    overlap = distance <= enlargement * (
        later_radius[:, None] + later_radius[None, :]
    )
    radius_ratio = np.maximum(
        later_radius[:, None] / later_radius[None, :],
        later_radius[None, :] / later_radius[:, None],
    )
    maximum_overlapping_radius_ratio = float(np.max(radius_ratio[overlap]))
    overlapping_reference_ratio = radius_ratio**2
    maximum_overlapping_reference_ratio = float(
        np.max(overlapping_reference_ratio[overlap])
    )
    theta = kappa * enlargement
    theoretical_radius_ratio = (1.0 + theta) / (1.0 - theta)
    theoretical_reference_ratio = theoretical_radius_ratio**2

    kappa_rows = []
    for trial_kappa in (0.05, 0.125, 0.25, 0.4):
        trial_theta = enlargement * trial_kappa
        trial_radius = _lipschitz_minorant(
            coordinates, later_raw_radius, trial_kappa
        )
        kappa_rows.append(
            {
                "kappa": trial_kappa,
                "minimum_regularized_radius": float(np.min(trial_radius)),
                "mean_regularized_radius": float(np.mean(trial_radius)),
                "theoretical_neighbor_radius_ratio": (
                    (1 + trial_theta) / (1 - trial_theta)
                ),
                "theoretical_neighbor_reference_ratio": (
                    ((1 + trial_theta) / (1 - trial_theta)) ** 2
                ),
            }
        )

    kappa_symbol, enlargement_symbol = sp.symbols(
        "kappa enlargement", positive=True, real=True
    )
    symbolic_radius_ratio = sp.factor(
        (1 + kappa_symbol * enlargement_symbol)
        / (1 - kappa_symbol * enlargement_symbol)
    )
    symbolic_reference_ratio = sp.factor(symbolic_radius_ratio**2)

    result: dict[str, object] = {
        "raw_radius": "ell(x,t)=sqrt(R_star*nu/A(x,t))",
        "regularized_radius": (
            "rho_kappa(x,t)=inf_y(ell(y,t)+kappa*distance(x,y))"
        ),
        "regularized_radius_is_a_minorant": bool(
            np.all(early_radius <= early_raw_radius + 1.0e-14)
            and np.all(later_radius <= later_raw_radius + 1.0e-14)
        ),
        "maximum_adjacent_lipschitz_slope": maximum_adjacent_slope,
        "kappa_lipschitz_bound_verified": bool(
            maximum_adjacent_slope <= kappa + 1.0e-12
        ),
        "raw_radius_is_nonincreasing_in_time": bool(
            np.all(later_raw_radius <= early_raw_radius + 1.0e-14)
        ),
        "regularized_radius_is_nonincreasing_in_time": bool(
            np.all(later_radius <= early_radius + 1.0e-14)
        ),
        "inflated_reference_envelope": (
            "A_tilde=R_star*nu/rho_kappa^2>=A>=a"
        ),
        "inflated_reference_dominates_raw_envelope": bool(
            np.all(early_reference_envelope >= early_envelope - 1.0e-12)
            and np.all(later_reference_envelope >= later_envelope - 1.0e-12)
        ),
        "inflated_reference_is_nondecreasing_in_time": bool(
            np.all(
                later_reference_envelope
                >= early_reference_envelope - 1.0e-12
            )
        ),
        "maximum_actual_local_reynolds": float(
            np.max(actual_local_reynolds)
        ),
        "actual_local_reynolds_cap_verified": bool(
            np.max(actual_local_reynolds) <= target_reynolds + 1.0e-12
        ),
        "enlargement_factor": enlargement,
        "kappa": kappa,
        "neighbor_radius_ratio_bound": str(symbolic_radius_ratio),
        "neighbor_reference_envelope_ratio_bound": str(
            symbolic_reference_ratio
        ),
        "theoretical_neighbor_radius_ratio": theoretical_radius_ratio,
        "maximum_observed_overlapping_radius_ratio": (
            maximum_overlapping_radius_ratio
        ),
        "overlapping_radius_comparability_verified": bool(
            maximum_overlapping_radius_ratio
            <= theoretical_radius_ratio + 1.0e-12
        ),
        "theoretical_neighbor_reference_ratio": (
            theoretical_reference_ratio
        ),
        "maximum_observed_overlapping_reference_ratio": (
            maximum_overlapping_reference_ratio
        ),
        "overlapping_reference_comparability_verified": bool(
            maximum_overlapping_reference_ratio
            <= theoretical_reference_ratio + 1.0e-12
        ),
        "kappa_tradeoff_rows": kappa_rows,
        "cover_consequence": (
            "a disjoint Vitali subfamily has bounded overlap after fixed "
            "enlargement because intersecting radii are comparable"
        ),
        "remaining_cover_gate": (
            "construct time-coherent centers and partition weights while "
            "controlling reselection, pressure edge flux, and renewal cost"
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
