"""Policy-iteration pilot for the radial-payoff affine Bellman problem."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def _bellman_drift(
    radial_grid: np.ndarray,
    axial_grid: np.ndarray,
    radial_gradient: np.ndarray,
    axial_gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    radial_position = radial_grid[:, None]
    axial_position = axial_grid[None, :]
    position_norm = np.sqrt(radial_position**2 + axial_position**2)
    gradient_norm = np.sqrt(radial_gradient**2 + axial_gradient**2)
    direction_radial = np.divide(
        radial_gradient,
        gradient_norm,
        out=np.zeros_like(radial_gradient),
        where=gradient_norm > 0.0,
    )
    direction_axial = np.divide(
        axial_gradient,
        gradient_norm,
        out=np.zeros_like(axial_gradient),
        where=gradient_norm > 0.0,
    )
    return (
        0.5 * radial_position
        + 1.5 * position_norm * direction_radial,
        0.5 * axial_position
        + 1.5 * position_norm * direction_axial,
    )


def _gradients(
    value: np.ndarray,
    radial_step: float,
    axial_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    radial_gradient = np.empty_like(value)
    axial_gradient = np.empty_like(value)
    radial_gradient[0, :] = 0.0
    radial_gradient[1:-1, :] = (
        value[2:, :] - value[:-2, :]
    ) / (2.0 * radial_step)
    radial_gradient[-1, :] = (
        1.0 - value[-2, :]
    ) / (2.0 * radial_step)
    axial_gradient[:, 1:-1] = (
        value[:, 2:] - value[:, :-2]
    ) / (2.0 * axial_step)
    axial_gradient[:, 0] = value[:, 1] / (2.0 * axial_step)
    axial_gradient[:, -1] = -value[:, -2] / (2.0 * axial_step)
    return radial_gradient, axial_gradient


def _linear_policy_solve(
    radial_drift: np.ndarray,
    axial_drift: np.ndarray,
    radius: float,
    half_height: float,
) -> np.ndarray:
    radial_count, axial_unknown_count = radial_drift.shape
    axial_interval_count = axial_unknown_count + 1
    radial_step = radius / radial_count
    axial_step = 2.0 * half_height / axial_interval_count
    radial_grid = np.arange(radial_count, dtype=float) * radial_step
    unknown_count = radial_count * axial_unknown_count
    matrix = lil_matrix((unknown_count, unknown_count))
    right_hand_side = np.zeros(unknown_count)

    def index(radial_index: int, axial_index: int) -> int:
        return radial_index * axial_unknown_count + axial_index

    for radial_index, radial_position in enumerate(radial_grid):
        for axial_index in range(axial_unknown_count):
            row = index(radial_index, axial_index)
            matrix[row, row] = 1.0 - 2.0 / axial_step**2
            lower_axial = (
                1.0 / axial_step**2
                - axial_drift[radial_index, axial_index]
                / (2.0 * axial_step)
            )
            upper_axial = (
                1.0 / axial_step**2
                + axial_drift[radial_index, axial_index]
                / (2.0 * axial_step)
            )
            if axial_index > 0:
                matrix[row, index(radial_index, axial_index - 1)] += (
                    lower_axial
                )
            if axial_index < axial_unknown_count - 1:
                matrix[row, index(radial_index, axial_index + 1)] += (
                    upper_axial
                )

            if radial_index == 0:
                matrix[row, row] += -4.0 / radial_step**2
                matrix[row, index(1, axial_index)] += (
                    4.0 / radial_step**2
                )
                continue

            lower_radial = (
                1.0 / radial_step**2
                - 1.0 / (2.0 * radial_position * radial_step)
                - radial_drift[radial_index, axial_index]
                / (2.0 * radial_step)
            )
            upper_radial = (
                1.0 / radial_step**2
                + 1.0 / (2.0 * radial_position * radial_step)
                + radial_drift[radial_index, axial_index]
                / (2.0 * radial_step)
            )
            matrix[row, row] += -2.0 / radial_step**2
            matrix[row, index(radial_index - 1, axial_index)] += (
                lower_radial
            )
            if radial_index < radial_count - 1:
                matrix[row, index(radial_index + 1, axial_index)] += (
                    upper_radial
                )
            else:
                right_hand_side[row] -= upper_radial

    return spsolve(matrix.tocsr(), right_hand_side).reshape(
        radial_count, axial_unknown_count
    )


def _policy_row(
    radial_count: int,
    axial_interval_count: int,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 40,
) -> dict[str, float | int | bool]:
    radius = 2.0
    half_height = 0.75
    radial_step = radius / radial_count
    axial_step = 2.0 * half_height / axial_interval_count
    radial_grid = np.arange(radial_count, dtype=float) * radial_step
    axial_grid = (
        -half_height
        + np.arange(1, axial_interval_count, dtype=float) * axial_step
    )
    radial_drift = 2.0 * radial_grid[:, None] * np.ones(
        (1, axial_interval_count - 1)
    )
    axial_drift = -np.ones((radial_count, 1)) * axial_grid[None, :]
    value = np.zeros_like(radial_drift)
    value_change = math.inf
    policy_change = math.inf

    for iteration in range(1, maximum_iterations + 1):
        new_value = _linear_policy_solve(
            radial_drift,
            axial_drift,
            radius,
            half_height,
        )
        radial_gradient, axial_gradient = _gradients(
            new_value,
            radial_step,
            axial_step,
        )
        new_radial_drift, new_axial_drift = _bellman_drift(
            radial_grid,
            axial_grid,
            radial_gradient,
            axial_gradient,
        )
        value_change = float(np.max(np.abs(new_value - value)))
        policy_change = float(
            max(
                np.max(np.abs(new_radial_drift - radial_drift)),
                np.max(np.abs(new_axial_drift - axial_drift)),
            )
        )
        value = new_value
        radial_drift = new_radial_drift
        axial_drift = new_axial_drift
        if value_change < tolerance and policy_change < tolerance:
            break

    interface_index = int(round(1.0 / radial_step))
    interface_values = value[interface_index, :]
    maximum_index = int(np.argmax(interface_values))
    return {
        "radial_intervals": radial_count,
        "axial_intervals": axial_interval_count,
        "radial_step": radial_step,
        "axial_step": axial_step,
        "policy_iterations": iteration,
        "policy_converged": bool(
            value_change < tolerance and policy_change < tolerance
        ),
        "final_value_change": value_change,
        "final_policy_change": policy_change,
        "minimum_grid_value": float(np.min(value)),
        "maximum_grid_value": float(np.max(value)),
        "inner_interface_maximum": float(interface_values[maximum_index]),
        "inner_interface_maximizing_axial_coordinate": float(
            axial_grid[maximum_index]
        ),
    }


def audit() -> dict[str, object]:
    resolutions = ((40, 30), (60, 46), (80, 60), (120, 90))
    rows = [_policy_row(*resolution) for resolution in resolutions]
    interface_values = np.array(
        [row["inner_interface_maximum"] for row in rows], dtype=float
    )
    closure_gain = 1.2321336084949255
    finest_value = float(interface_values[-1])
    tail_spread = float(np.max(interface_values[-3:]) - np.min(interface_values[-3:]))
    feedback_monte_carlo = 0.6386859024600177
    feedback_standard_error = 0.0008869662969647623
    result: dict[str, object] = {
        "admissible_backward_spectra": (
            "B has eigenvalues (1+t,-t,-1), -1/2<=t<=1"
        ),
        "exact_control_hamiltonian": (
            "sup_B (B y).p = (y.p)/2 + 3|y||p|/2"
        ),
        "endpoint_dominance_identity": (
            "the t=1 value exceeds the t=-1/2 value by "
            "3(|y||p|+y.p)/4>=0"
        ),
        "bellman_boundary_problem": (
            "Delta u+(y.grad u)/2+3|y||grad u|/2+u=0; "
            "u=1 on r=2 and u=0 on |z|=0.75"
        ),
        "refinement_rows": rows,
        "finest_inner_interface_maximum": finest_value,
        "last_three_grid_spread": tail_spread,
        "maximum_dynamic_one_history_gain_for_closure": closure_gain,
        "sampled_Bellman_margin_to_closure": closure_gain - finest_value,
        "sampled_Bellman_value_below_closure": bool(
            finest_value < closure_gain
        ),
        "outward_radial_feedback_monte_carlo": {
            "payoff": feedback_monte_carlo,
            "standard_error": feedback_standard_error,
            "path_count": 500_000,
            "time_step": 0.001,
        },
        "ideal_nonautonomous_boundary_theorem_certified": False,
        "scope_guard": (
            "the Hamiltonian reduction and domination of every open-loop "
            "affine history by the feedback problem are exact; the HJB "
            "values use centered finite differences and are a converged "
            "pilot, not a comparison-certified enclosure"
        ),
        "next_gate": (
            "construct a monotone or interval-residual supersolution for "
            "the axisymmetric HJB with inner-interface value below "
            "1.23213361"
        ),
    }
    positive_checks = (
        all(row["policy_converged"] for row in rows),
        finest_value > 0.65,
        finest_value < 0.72,
        tail_spread < 0.01,
        result["sampled_Bellman_value_below_closure"],
        result["sampled_Bellman_margin_to_closure"] > 0.54,
        feedback_monte_carlo < finest_value,
    )
    result["all_positive_radial_Bellman_pilot_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
