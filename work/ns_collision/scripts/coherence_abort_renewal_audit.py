"""Audit the restart budget for coherence-aborted moving visits."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import svdvals


def _load_sector_module():
    script = Path(__file__).resolve().with_name(
        "sectorial_poisson_transfer_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sectorial_poisson_for_abort_renewal", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix_restart_trials(
    good_norm: float, restart_norms: tuple[float, ...]
) -> list[dict[str, float | bool]]:
    rng = np.random.default_rng(12060719)
    dimension = 8
    raw_good = rng.normal(size=(dimension, dimension))
    good = good_norm * raw_good / float(svdvals(raw_good)[0])
    rows = []
    for restart_norm in restart_norms:
        raw_restart = rng.normal(size=(dimension, dimension))
        restart = (
            restart_norm
            * raw_restart
            / float(svdvals(raw_restart)[0])
        )
        complete = good @ np.linalg.inv(np.eye(dimension) - restart)
        actual_norm = float(svdvals(complete)[0])
        theorem_bound = good_norm / (1.0 - restart_norm)
        rows.append(
            {
                "restart_norm": restart_norm,
                "actual_complete_norm": actual_norm,
                "Neumann_upper_bound": theorem_bound,
                "bound_holds": bool(
                    actual_norm <= theorem_bound + 1.0e-12
                ),
            }
        )
    return rows


def audit() -> dict[str, object]:
    sector = _load_sector_module().audit()
    generation_criterion = float(sector["working_generation_criterion"])
    condition_number = float(sector["working_sector_condition_number"])
    good_norm = math.sqrt(generation_criterion)
    restart_allowance_without_perturbation = 1.0 - good_norm

    pair_true_split_factor = 0.40869503866769363
    maximum_split_mismatch = (
        restart_allowance_without_perturbation / pair_true_split_factor
    )
    split_paid_sector_intercept = (
        (1.0 - pair_true_split_factor) / good_norm - 1.0
    ) / condition_number
    split_paid_equal_share = split_paid_sector_intercept / (
        2.0 + split_paid_sector_intercept
    )
    sharp_sobolev_constant = 4.0 ** (2.0 / 3.0) / (
        3.0 * math.pi ** (4.0 / 3.0)
    )
    probability_paid_abort_budget = (
        restart_allowance_without_perturbation**2
    )
    trials = _matrix_restart_trials(
        good_norm,
        (0.0, 0.2, pair_true_split_factor, 0.49),
    )

    result: dict[str, object] = {
        "restart_equation": "T=G+T*R, hence T=G*(I-R)^(-1)",
        "restart_norm_bound": (
            "||T||<=g/(1-r), g=||G|| and r=||R||<1"
        ),
        "strict_closure_condition": "g+r<1",
        "working_complete_generation_criterion": generation_criterion,
        "working_good_generation_norm_g": good_norm,
        "maximum_unperturbed_restart_norm": (
            restart_allowance_without_perturbation
        ),
        "conservative_relabel_norm": 1.0,
        "free_conservative_restarts_do_not_close": bool(
            1.0 >= restart_allowance_without_perturbation
        ),
        "pair_true_dyadic_split_restart_factor": pair_true_split_factor,
        "split_paid_restart_closes_without_sector_error": bool(
            good_norm + pair_true_split_factor < 1.0
        ),
        "maximum_extra_split_restart_mismatch": maximum_split_mismatch,
        "sector_with_restart_condition": (
            "g0*[1+chi*(alpha+beta)/(1-alpha)]+r<1"
        ),
        "split_paid_sector_condition": (
            "beta<d_split-(1+d_split)*alpha"
        ),
        "split_paid_sector_intercept_d": split_paid_sector_intercept,
        "split_paid_equal_alpha_beta": split_paid_equal_share,
        "split_paid_equal_share_potential_L3_over_2_over_nu": (
            split_paid_equal_share / sharp_sobolev_constant
        ),
        "split_paid_equal_share_drift_L3_over_nu": (
            split_paid_equal_share / math.sqrt(sharp_sobolev_constant)
        ),
        "probability_paid_restart_rule": (
            "an abort branch of probability p and conditional norm M has "
            "restart norm at most sqrt(p)*M"
        ),
        "maximum_unit_norm_abort_probability_without_sector_error": (
            probability_paid_abort_budget
        ),
        "matrix_restart_trials": trials,
        "all_matrix_restart_bounds_hold": all(
            row["bound_holds"] for row in trials
        ),
        "interpretation": (
            "stopping-time relabeling is lossless but not contractive; an "
            "early coherence abort must carry a true-split factor, a small "
            "probability/occupation factor, or another strict decay payment"
        ),
        "full_coherence_abort_bound_from_Navier_Stokes_closed": False,
        "next_gate": (
            "show that every pre-exit failure of the fitted sector budget "
            "either triggers a genuine envelope level halving or belongs to "
            "a branch with conditional L2 mass below the displayed "
            "probability threshold"
        ),
    }
    positive_checks = (
        result["free_conservative_restarts_do_not_close"],
        result["split_paid_restart_closes_without_sector_error"],
        result["all_matrix_restart_bounds_hold"],
        0.49 < restart_allowance_without_perturbation < 0.50,
        1.20 < maximum_split_mismatch < 1.21,
        0.035 < split_paid_sector_intercept < 0.036,
        0.24 < probability_paid_abort_budget < 0.25,
    )
    result["all_positive_abort_renewal_checks_pass"] = all(positive_checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
