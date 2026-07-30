"""Audit density inheritance and the deterministic split-time obstruction."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import brentq


FORM_FLOOR = 4.832287335665
CYLINDER_VOLUME = 6.0 * math.pi
POTENTIAL_FORCING = 0.7989685513198063
DRIFT_FORCING = 3.072840583265365
POTENTIAL_RELATIVE_FORM = 0.2203290376862308
CURRENT_SPLIT_ONLY_ALLOWANCE_WITH_LEGACY_RETURN = 0.0873075660287922


def _markov_density_stress(seed: int = 190726) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows = []
    for point_count, parent_count, child_count in (
        (17, 4, 7),
        (31, 9, 5),
        (53, 6, 13),
    ):
        parent_density = rng.lognormal(
            mean=-0.5, sigma=1.0, size=(point_count, parent_count)
        )
        transition = rng.random(
            (point_count, parent_count, child_count)
        )
        transition /= np.sum(transition, axis=2, keepdims=True)
        child_density = np.einsum(
            "xi,xij->xj", parent_density, transition
        )
        parent_marginal = np.sum(parent_density, axis=1)
        child_marginal = np.sum(child_density, axis=1)
        rows.append(
            {
                "point_count": point_count,
                "parent_label_count": parent_count,
                "child_label_count": child_count,
                "maximum_marginal_error": float(
                    np.max(np.abs(parent_marginal - child_marginal))
                ),
                "child_ell2_L2_to_physical_marginal_L2_ratio": float(
                    np.linalg.norm(child_density)
                    / np.linalg.norm(parent_marginal)
                ),
            }
        )
    return {
        "rows": rows,
        "maximum_marginal_error": max(
            row["maximum_marginal_error"] for row in rows
        ),
        "maximum_child_ell2_ratio": max(
            row["child_ell2_L2_to_physical_marginal_L2_ratio"]
            for row in rows
        ),
    }


def _time_atom_counterexample() -> list[dict[str, float]]:
    rows = []
    for concentration in (4, 16, 64, 256, 1024):
        rows.append(
            {
                "concentration": concentration,
                "integrated_energy": 1.0,
                "point_time_energy": float(concentration),
                "point_to_integrated_ratio": float(concentration),
            }
        )
    return rows


def _volume_density_thresholds() -> list[dict[str, float]]:
    allowance = CURRENT_SPLIT_ONLY_ALLOWANCE_WITH_LEGACY_RETURN
    rows = []
    for normalized_density_ratio in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        response_scale = math.sqrt(
            normalized_density_ratio
            / (CYLINDER_VOLUME * FORM_FLOOR)
        )

        def potential_error(mass: float) -> float:
            alpha = POTENTIAL_RELATIVE_FORM * mass
            return (
                POTENTIAL_FORCING
                * mass
                * response_scale
                / (1.0 - alpha)
            )

        potential_threshold = brentq(
            lambda mass: potential_error(mass) - allowance,
            0.0,
            0.999 / POTENTIAL_RELATIVE_FORM,
        )
        drift_threshold = allowance / (
            DRIFT_FORCING * response_scale
        )
        rows.append(
            {
                "normalized_volume_density_ratio": (
                    normalized_density_ratio
                ),
                "potential_L3_over_2_threshold": potential_threshold,
                "potential_alpha_at_threshold": (
                    POTENTIAL_RELATIVE_FORM * potential_threshold
                ),
                "drift_L3_threshold": drift_threshold,
            }
        )
    return rows


def audit() -> dict[str, object]:
    stress = _markov_density_stress()
    atom_rows = _time_atom_counterexample()
    volume_rows = _volume_density_thresholds()
    result: dict[str, object] = {
        "pointwise_Markov_split": (
            "g_j(s,x)=sum_i f_i(s,x)P_ij(s,x), P_ij>=0, "
            "sum_j P_ij=1"
        ),
        "physical_density_identity": (
            "sum_j g_j(s,x)=sum_i f_i(s,x) pointwise"
        ),
        "label_ell2_bound": (
            "sum_j g_j(s,x)^2<=(sum_i f_i(s,x))^2"
        ),
        "random_Markov_density_stress": stress,
        "pointwise_split_preserves_existing_physical_density": True,
        "pointwise_split_creates_spatial_or_temporal_smoothing": False,
        "coordinate_change_rule": (
            "under a child surface map with Jacobian J>=J_min>0, "
            "the spatial L2 density norm grows by at most J_min^(-1/2)"
        ),
        "temporal_atom_counterexample": {
            "family": (
                "w_N(t,x)=sqrt(N) 1_[t0,t0+1/N](t)v(x), "
                "with h[v]=1"
            ),
            "rows": atom_rows,
            "conclusion": (
                "bounded integrated energy does not control evaluation at "
                "a deterministic split time"
            ),
        },
        "deterministic_split_time_covered_by_averaged_surface_trace": False,
        "fixed_time_volume_entry_bound": (
            "if dnu/dx<=M_V/|D|, then int |w(t,x)|^2 dnu "
            "<=M_V F^2/(|D|(1-alpha)^2*m0)"
        ),
        "cylinder_volume": CYLINDER_VOLUME,
        "current_split_only_error_allowance_with_legacy_return": (
            CURRENT_SPLIT_ONLY_ALLOWANCE_WITH_LEGACY_RETURN
        ),
        "conditional_volume_density_threshold_rows": volume_rows,
        "inherited_space_time_route": (
            "if the pre-split unnormalized law already has a summable "
            "space-time density and the split time/coordinate map does not "
            "collapse it, the child-label law inherits that envelope"
        ),
        "split_density_dichotomy": (
            "use inherited absolute space-time density, or prove a bounded "
            "child-volume density at a deterministic split time; a bare "
            "pointwise relabel supplies neither"
        ),
        "physical_child_volume_density_certified": False,
        "zero_lag_or_surface_supported_split_handled": False,
        "full_true_split_entry_gate_closed": False,
        "scope_guard": (
            "the Markov inheritance identities, atom counterexample, and "
            "conditional volume estimate are exact. The table assumes the "
            "entire current split-only error allowance and does not prove "
            "that a Navier-Stokes level-change law has bounded volume "
            "density or a nondegenerate coordinate Jacobian"
        ),
        "next_gate": (
            "locate the true level-change time inside the stopped-process "
            "construction and prove either that it inherits a prior random "
            "space-time density or that its child-volume law has a uniform "
            "L-infinity density; isolate and handle zero-lag cascades"
        ),
    }
    positive_checks = (
        stress["maximum_marginal_error"] < 1.0e-12,
        stress["maximum_child_ell2_ratio"] <= 1.0 + 1.0e-12,
        atom_rows[-1]["point_to_integrated_ratio"] == 1024.0,
        volume_rows[0]["potential_L3_over_2_threshold"] > 0.84,
        volume_rows[0]["drift_L3_threshold"] > 0.27,
        volume_rows[-1]["potential_L3_over_2_threshold"] > 0.17,
        volume_rows[-1]["drift_L3_threshold"] > 0.047,
        result["pointwise_split_preserves_existing_physical_density"],
        not result["pointwise_split_creates_spatial_or_temporal_smoothing"],
        not result[
            "deterministic_split_time_covered_by_averaged_surface_trace"
        ],
        not result["full_true_split_entry_gate_closed"],
    )
    result["all_positive_split_density_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
