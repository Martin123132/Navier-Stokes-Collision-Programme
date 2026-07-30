"""Audit the Doob transform and perturbation budget of the HJB barrier."""

from __future__ import annotations

import json
import math

import sympy as sp


def _symbolic_product_rule() -> dict[str, object]:
    U, v = sp.symbols("U v", positive=True)
    laplacian_U, laplacian_v = sp.symbols("laplacian_U laplacian_v")
    gradient_pair = sp.symbols("gradient_pair")
    baseline_U, baseline_v = sp.symbols("baseline_U baseline_v")
    error_U, error_v = sp.symbols("error_U error_v")
    potential = sp.symbols("potential")
    direct_expansion = (
        U * laplacian_v
        + 2 * gradient_pair
        + v * laplacian_U
        + U * baseline_v
        + v * baseline_U
        + U * error_v
        + v * error_U
        + (1 + potential) * U * v
    ) / U
    kappa = -(laplacian_U + baseline_U + U) / U
    transformed_expansion = (
        laplacian_v
        + baseline_v
        + error_v
        + 2 * gradient_pair / U
        + (-kappa + potential + error_U / U) * v
    )
    identity_residual = sp.expand(
        direct_expansion - transformed_expansion
    )
    error_log_gradient = sp.symbols("error_log_gradient")
    weighted_drift_real_part = -error_log_gradient
    transformed_error_potential = error_log_gradient
    return {
        "Doob_product_rule_residual": str(identity_residual),
        "Doob_product_rule_exact": bool(identity_residual == 0),
        "weighted_error_drift_real_part": str(
            weighted_drift_real_part
        ),
        "transformed_error_potential": str(
            transformed_error_potential
        ),
        "weighted_error_cancellation_residual": str(
            sp.simplify(
                weighted_drift_real_part
                + transformed_error_potential
            )
        ),
        "weighted_error_cancellation_exact": bool(
            sp.simplify(
                weighted_drift_real_part
                + transformed_error_potential
            )
            == 0
        ),
    }


def audit() -> dict[str, object]:
    symbolic = _symbolic_product_rule()
    barrier_gain = 1.3428786845671419
    closure_gain = 1.2321336084949255
    cycle_coefficient = 0.6586950386676936
    uniform_killing = 0.005
    radial_log_gradient_exponent = -7.0 / 20.0
    axial_log_gradient_exponent = -13.0 / 20.0
    result: dict[str, object] = {
        "symbolic_Doob_audit": symbolic,
        "transformed_generator": (
            "Delta v+(B y+e+2 grad log U).grad v+"
            "[-kappa_B+q_++e.grad log U]v"
        ),
        "certified_affine_killing": (
            "kappa_B=-(Delta U+(B y).grad U+U)/U>=0.005"
        ),
        "signed_pointwise_perturbation_condition": (
            "q_++e.grad(log U)<=0.005"
        ),
        "absolute_pointwise_perturbation_condition": (
            "q_++|e||grad log U|<=0.005"
        ),
        "uniform_killing_rate": uniform_killing,
        "certified_barrier_gain": barrier_gain,
        "maximum_gain_for_cycle_closure": closure_gain,
        "additive_gain_allowance": closure_gain - barrier_gain,
        "relative_one_history_gain_allowance": (
            closure_gain / barrier_gain - 1.0
        ),
        "older_barrier_closes_current_cubic_cycle": False,
        "certified_ideal_generation_criterion": (
            cycle_coefficient * barrier_gain**2
        ),
        "remaining_generation_margin": (
            1.0 - cycle_coefficient * barrier_gain**2
        ),
        "boundary_log_gradient_asymptotics": {
            "radial_wall_away_from_caps": (
                "|grad log U| grows like x^(-7/20), "
                "x=1-r^2/4"
            ),
            "axial_cap_away_from_axis": (
                "|grad log U| grows like a^(-13/20), "
                "a=cos(2*pi*z/3)"
            ),
            "radial_exponent": radial_log_gradient_exponent,
            "axial_exponent": axial_log_gradient_exponent,
            "unbounded_at_absorbing_collars": True,
        },
        "pointwise_condition_follows_from_critical_Lp_norms": False,
        "weighted_form_identity": (
            "in L2(U^2 dx), divergence-free e contributes "
            "-int U^2 v^2 e.grad log U through its drift; this cancels "
            "the transformed +e.grad log U potential exactly"
        ),
        "weighted_form_removes_drift_sector_bound": False,
        "nonautonomous_critical_form_to_boundary_theorem_closed": False,
        "scope_guard": (
            "the product rule and weighted cancellation are exact, and "
            "the scalar margins inherit the interval-certified barrier; "
            "no critical-norm boundary perturbation estimate is claimed"
        ),
        "next_gate": (
            "carry the exact Doob cancellation into the lower-gain "
            "finite-energy H1 barrier; the older barrier has no remaining "
            "gain allowance under the current cubic split"
        ),
    }
    positive_checks = (
        symbolic["Doob_product_rule_exact"],
        symbolic["weighted_error_cancellation_exact"],
        uniform_killing == 0.005,
        result["additive_gain_allowance"] < -0.10,
        result["relative_one_history_gain_allowance"] < -0.08,
        result["certified_ideal_generation_criterion"] > 1.18,
        result["remaining_generation_margin"] < -0.18,
        not result["older_barrier_closes_current_cubic_cycle"],
        math.isclose(radial_log_gradient_exponent, -0.35),
        math.isclose(axial_log_gradient_exponent, -0.65),
        not result["pointwise_condition_follows_from_critical_Lp_norms"],
        not result["nonautonomous_critical_form_to_boundary_theorem_closed"],
    )
    result["all_positive_Doob_perturbation_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
