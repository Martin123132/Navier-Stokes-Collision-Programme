"""Audit an averaged space-time entry trace bound for the H1 barrier."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

from scipy.optimize import brentq, minimize_scalar


def _load_barrier_module():
    script = Path(__file__).with_name(
        "radial_h1_payoff_supersolution_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "radial_h1_for_averaged_entry", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _time_energy_factor(alpha: float, decay: float) -> dict[str, float]:
    coercivity = 1.0 - alpha
    if not 0.0 < coercivity <= 1.0:
        raise ValueError("alpha must lie in [0,1)")
    if decay <= 0.0:
        raise ValueError("decay must be positive")
    form_floor = 4.832287335665

    def factor(window: float) -> float:
        return (
            window / coercivity**2
            + 1.0 / (coercivity**3 * form_floor)
        ) / (1.0 - math.exp(-decay * window))

    optimum = minimize_scalar(
        factor,
        bounds=(1.0e-5 / decay, 40.0 / decay),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    return {
        "optimal_window": float(optimum.x),
        "factor": float(optimum.fun),
    }


def _overshoot(
    potential_mass: float,
    drift_mass: float,
    density_ratio: float,
    decay: float,
    constants: dict[str, float],
) -> float:
    alpha = constants["potential_relative_form"] * potential_mass
    if alpha >= 1.0:
        return math.inf
    time_factor = _time_energy_factor(alpha, decay)["factor"]
    density_envelope = (
        density_ratio * decay / constants["interface_area"]
    )
    forcing = (
        constants["potential_forcing"] * potential_mass
        + constants["drift_forcing"] * drift_mass
    )
    return forcing * math.sqrt(
        density_envelope * constants["trace_form_constant"] * time_factor
    )


def _one_error_thresholds(
    density_ratio: float,
    decay: float,
    constants: dict[str, float],
) -> dict[str, float]:
    allowance = constants["additive_gain_allowance"]
    potential_limit = 0.999 / constants["potential_relative_form"]
    potential_threshold = brentq(
        lambda mass: (
            _overshoot(
                mass, 0.0, density_ratio, decay, constants
            )
            - allowance
        ),
        0.0,
        potential_limit,
    )
    drift_threshold = brentq(
        lambda mass: (
            _overshoot(
                0.0, mass, density_ratio, decay, constants
            )
            - allowance
        ),
        0.0,
        10.0,
    )
    return {
        "density_ratio": density_ratio,
        "decay": decay,
        "potential_L3_over_2_threshold": potential_threshold,
        "drift_L3_threshold": drift_threshold,
        "potential_alpha_at_threshold": (
            constants["potential_relative_form"] * potential_threshold
        ),
    }


def audit() -> dict[str, object]:
    barrier_module = _load_barrier_module()
    barrier = barrier_module.audit()
    form_floor = 4.832287335665
    first_eigenvalue = form_floor + 1.0
    poincare_factor = first_eigenvalue / form_floor
    trace_form_constant = poincare_factor + 1.0 / form_floor
    interface_area = 3.0 * math.pi
    forcing = barrier["global_energy_forcing_coefficients"]
    constants = {
        "entry_gain": barrier["entry_gain"],
        "additive_gain_allowance": barrier["additive_gain_allowance"],
        "potential_forcing": forcing["potential_L3_over_2"],
        "drift_forcing": forcing["drift_L3"],
        "potential_relative_form": forcing[
            "potential_relative_form"
        ],
        "interface_area": interface_area,
        "trace_form_constant": trace_form_constant,
    }
    decay = 1.0
    rows = [
        _one_error_thresholds(ratio, decay, constants)
        for ratio in (1.0, 2.0, 4.0, 8.0, 16.0)
    ]
    unit_energy = _time_energy_factor(0.0, decay)
    result: dict[str, object] = {
        "return_density_hypothesis": (
            "d nu/(ds d sigma)<=C_R exp(-kappa s), with nu "
            "the unnormalized exterior-return law"
        ),
        "normalized_density_ratio": (
            "M_R=C_R*|Sigma|/kappa; M_R=1 is the normalized "
            "exponential-uniform reference density"
        ),
        "exponential_envelope_is_only_a_special_case": True,
        "raw_unbounded_exterior_exponential_envelope_viable": False,
        "general_summable_envelope_followup": (
            "exterior_return_tail_gate.py replaces exp(-kappa s) by "
            "sum_n sup_(I_n)rho<infinity"
        ),
        "interface_trace_inequality": (
            "||T v||_L2(Sigma)^2<=C_Sigma h[v]"
        ),
        "trace_form_constant": trace_form_constant,
        "interface_area": interface_area,
        "interval_energy_inequality": (
            "int_I h[w]<=F^2(|I|/(1-alpha)^2+"
            "1/((1-alpha)^3*m0))"
        ),
        "exponential_time_energy_factor_at_alpha_zero": unit_energy,
        "composite_gain_bound": (
            "||R u||_L2(mu)<=g_H+F*sqrt(C_R*C_Sigma*J)"
        ),
        "candidate_constants": constants,
        "one_error_threshold_rows": rows,
        "return_law_remains_unnormalized": True,
        "early_nonreturns_keep_sub_Markov_contraction": True,
        "geometric_protected_support_required_by_this_route": False,
        "continuous_partition_IMS_cost_incurred": False,
        "actual_exterior_return_density_envelope_certified": False,
        "true_split_entry_density_envelope_certified": False,
        "pointwise_split_density_inheritance_proved_separately": True,
        "deterministic_split_time_atom_covered": False,
        "fixed_time_volume_density_alternative_available": True,
        "all_generation_entry_transitions_covered": False,
        "H1_barrier_interval_certified": True,
        "full_Navier_Stokes_entry_theorem_closed": False,
        "scope_guard": (
            "the trace and interval-energy implications are conditional "
            "functional inequalities. The table does not assert that the "
            "physical Navier-Stokes return law satisfies the stated "
            "space-time density envelope. The exponential rows apply to a "
            "bounded or additionally killed storage phase, not a raw "
            "unbounded exterior. A true split may inherit an existing "
            "space-time density, but a deterministic split-time atom is "
            "not covered by this averaged surface theorem. The finite-energy HJB "
            "barrier is certified separately by "
            "radial_h1_payoff_interval_certificate.py"
        ),
        "next_gate": (
            "use the summable-envelope theorem for the unnormalized return "
            "kernel, and use split_entry_density_inheritance_audit.py to "
            "decide whether each true split inherits that density or needs "
            "the fixed-time child-volume alternative"
        ),
    }
    potential_thresholds = [
        row["potential_L3_over_2_threshold"] for row in rows
    ]
    drift_thresholds = [row["drift_L3_threshold"] for row in rows]
    positive_checks = (
        barrier["all_positive_H1_supersolution_pilot_checks_pass"],
        1.41 < trace_form_constant < 1.42,
        1.7 < unit_energy["factor"] < 1.9,
        all(
            potential_thresholds[index]
            > potential_thresholds[index + 1]
            for index in range(len(potential_thresholds) - 1)
        ),
        all(
            drift_thresholds[index] > drift_thresholds[index + 1]
            for index in range(len(drift_thresholds) - 1)
        ),
        rows[-1]["potential_L3_over_2_threshold"] > 0.05,
        rows[-1]["drift_L3_threshold"] > 0.013,
        result["return_law_remains_unnormalized"],
        result["H1_barrier_interval_certified"],
        not result["raw_unbounded_exterior_exponential_envelope_viable"],
        not result["all_generation_entry_transitions_covered"],
        result["pointwise_split_density_inheritance_proved_separately"],
        not result["deterministic_split_time_atom_covered"],
        not result["geometric_protected_support_required_by_this_route"],
    )
    result["all_positive_averaged_entry_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
