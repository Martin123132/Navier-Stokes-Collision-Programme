"""Audit the reversible return tail and artificial x-width dependence."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import mpmath
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh, expm_multiply


TERMINAL_TIME = 6.0
FINITE_TIME_STRESS = 1.05
SCALAR_QUADRATURE_STRESS = 1.01


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _high_precision_barta_bounds(
    matrix, positive_vector: np.ndarray
) -> tuple[float, float]:
    matrix = matrix.tocsr()
    mpmath.mp.dps = 80
    ratios = []
    for row in range(matrix.shape[0]):
        terms = []
        for pointer in range(matrix.indptr[row], matrix.indptr[row + 1]):
            column = int(matrix.indices[pointer])
            terms.append(
                mpmath.mpf(float(matrix.data[pointer]))
                * mpmath.mpf(float(positive_vector[column]))
            )
        numerator = mpmath.fsum(terms)
        denominator = mpmath.mpf(float(positive_vector[row]))
        ratios.append(numerator / denominator)
    lower = np.nextafter(float(min(ratios)), -math.inf)
    upper = np.nextafter(float(max(ratios)), math.inf)
    return lower, upper


def _spectral_row(grid: dict[str, object]) -> dict[str, object]:
    generator = grid["generator"]
    mass = np.asarray(grid["state_mass"])
    root_mass = np.sqrt(mass)
    symmetric_generator = (
        diags(root_mass) @ generator @ diags(1.0 / root_mass)
    ).tocsc()
    symmetry_defect = symmetric_generator - symmetric_generator.transpose()
    maximum_symmetry_defect = (
        float(np.max(np.abs(symmetry_defect.data)))
        if symmetry_defect.nnz
        else 0.0
    )
    symmetric_part = 0.5 * (
        symmetric_generator + symmetric_generator.transpose()
    )
    positive_operator = -symmetric_part
    eigenvalues, eigenvectors = eigsh(
        positive_operator,
        k=2,
        which="SA",
        tol=1.0e-12,
        maxiter=200000,
    )
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    principal_vector = np.abs(eigenvectors[:, order[0]])
    barta_lower, barta_upper = _high_precision_barta_bounds(
        positive_operator, principal_vector
    )

    inner_arcs = np.asarray(grid["inner_dual_arcs"])
    boundary_operator = (
        diags(1.0 / np.sqrt(inner_arcs))
        @ grid["inner_rate_matrix"].transpose()
        @ diags(root_mass)
    ).tocsr()
    boundary_gram = (boundary_operator @ boundary_operator.transpose()).tocsr()
    row_sums = []
    for row in range(boundary_gram.shape[0]):
        row_sums.append(
            math.fsum(
                abs(float(value))
                for value in boundary_gram.data[
                    boundary_gram.indptr[row] : boundary_gram.indptr[row + 1]
                ]
            )
        )
    gram_gershgorin_upper = np.nextafter(
        max(row_sums) * (1.0 + 1.0e-12), math.inf
    )
    boundary_operator_norm_upper = math.sqrt(gram_gershgorin_upper)
    boundary_operator_norm_pilot = math.sqrt(
        float(np.linalg.eigvalsh(boundary_gram.toarray())[-1])
    )
    return {
        "symmetric_generator": symmetric_generator,
        "principal_decay_barta_lower": barta_lower,
        "principal_decay_barta_upper": barta_upper,
        "principal_decay_barta_width": barta_upper - barta_lower,
        "principal_decay_eigsh_pilot": float(eigenvalues[0]),
        "second_decay_eigsh_pilot": float(eigenvalues[1]),
        "spectral_gap_eigsh_pilot": float(eigenvalues[1] - eigenvalues[0]),
        "maximum_mass_symmetry_defect": maximum_symmetry_defect,
        "minimum_principal_vector_entry": float(np.min(principal_vector)),
        "boundary_operator_norm_gershgorin_upper": (
            boundary_operator_norm_upper
        ),
        "boundary_operator_norm_pilot": boundary_operator_norm_pilot,
        "finite_matrix_barta_bound_high_precision": True,
        "boundary_operator_norm_uses_row_sum_upper_bound": True,
    }


def _axial_tail_norm_factor(time: float, patch_half_height: float) -> float:
    return (
        math.sqrt(patch_half_height / math.pi)
        / math.sqrt(1.0 - math.exp(-2.0 * time))
    )


def _axial_tail_scalar_factor(
    time: float, patch_half_height: float
) -> float:
    return (
        math.sqrt(2.0 / math.pi)
        * patch_half_height
        / math.sqrt(1.0 - math.exp(-2.0 * time))
    )


def _propagate(
    grid: dict[str, object], return_density
) -> dict[str, object]:
    generator = grid["generator"]
    entry_states = np.asarray(grid["entry_states"])
    state = np.zeros((generator.shape[0], len(entry_states)))
    state[entry_states, np.arange(len(entry_states))] = 1.0
    generator_transpose = generator.transpose().tocsc()
    times = []
    raw_l2_rows = []
    scalar_density_rows = []
    current_time = 0.0
    segments = (
        (0.05, 26),
        (0.2, 31),
        (1.0, 41),
        (4.0, 31),
        (TERMINAL_TIME, 17),
    )
    for segment_end, point_count in segments:
        duration = segment_end - current_time
        trajectory = expm_multiply(
            generator_transpose,
            state,
            start=0.0,
            stop=duration,
            num=point_count,
            endpoint=True,
        )
        for local_index in range(1, point_count):
            time = current_time + duration * local_index / (point_count - 1)
            snapshot = trajectory[local_index]
            boundary_flux = np.asarray(
                grid["inner_rate_matrix"].transpose() @ snapshot
            ).T
            transverse_l2 = np.sqrt(
                np.sum(
                    boundary_flux**2 / grid["inner_dual_arcs"][None, :],
                    axis=1,
                )
            )
            transverse_mass = np.sum(boundary_flux, axis=1)
            patch_mass, axial_l2 = return_density._axial_factors(
                time, grid["rho"]
            )
            deformation = math.exp(time)
            times.append(time)
            raw_l2_rows.append(deformation * axial_l2 * transverse_l2)
            scalar_density_rows.append(
                deformation * patch_mass * transverse_mass
            )
        state = trajectory[-1]
        current_time = segment_end
    return {
        "times": np.asarray(times),
        "raw_l2": np.asarray(raw_l2_rows),
        "scalar_density": np.asarray(scalar_density_rows),
        "terminal_state": state,
    }


def _spectral_interval_factor(
    times: np.ndarray,
    envelope: np.ndarray,
    terminal_weighted_norm: float,
    decay_lower: float,
    boundary_operator_norm_upper: float,
    return_density,
) -> dict[str, float]:
    axial_upper = _axial_tail_norm_factor(
        float(times[-1]), return_density.PATCH_HALF_HEIGHT
    )
    terminal_amplitude = (
        axial_upper
        * boundary_operator_norm_upper
        * terminal_weighted_norm
    )
    rows = []
    for window in np.geomspace(0.04, 1.0, 48):
        interval_indices = np.floor(times / window).astype(int)
        finite_sum = math.fsum(
            float(np.max(envelope[interval_indices == interval_index]))
            for interval_index in np.unique(interval_indices)
        )
        spectral_tail_sum = terminal_amplitude / (
            1.0 - math.exp(-decay_lower * window)
        )
        stressed_sum = FINITE_TIME_STRESS * finite_sum + spectral_tail_sum
        energy = window + 1.0 / return_density.FORM_FLOOR
        rows.append(
            (
                energy * stressed_sum,
                window,
                energy * spectral_tail_sum,
                finite_sum,
            )
        )
    factor, window, tail_factor, finite_sum = min(rows)
    return {
        "spectral_interval_factor": float(factor),
        "optimal_window": float(window),
        "spectral_tail_factor_contribution": float(tail_factor),
        "unstressed_finite_sample_sum": float(finite_sum),
        "terminal_tail_amplitude": float(terminal_amplitude),
    }


def _width_row(
    fem,
    return_density,
    spacing: float,
    x_half_width: float,
    run_density: bool,
) -> dict[str, object]:
    grid = fem._build_mesh(spacing, x_half_width=x_half_width)
    structural = fem._structural_row(grid)
    spectral = _spectral_row(grid)
    row: dict[str, object] = {
        "spacing": spacing,
        "x_half_width": x_half_width,
        "structural": structural,
        "spectral": {
            key: value
            for key, value in spectral.items()
            if key != "symmetric_generator"
        },
    }
    if not run_density:
        return row

    propagated = _propagate(grid, return_density)
    times = propagated["times"]
    terminal_state = propagated["terminal_state"]
    terminal_weighted = terminal_state / np.sqrt(grid["state_mass"])[:, None]
    terminal_norms = np.linalg.norm(terminal_weighted, axis=0)
    scalar_finite = np.trapezoid(
        propagated["scalar_density"], times, axis=0
    )
    scalar_tail_coefficient = (
        _axial_tail_scalar_factor(
            float(times[-1]), return_density.PATCH_HALF_HEIGHT
        )
        * math.sqrt(2.0 * math.pi)
        * spectral["boundary_operator_norm_gershgorin_upper"]
        / spectral["principal_decay_barta_lower"]
    )
    scalar_tail_bounds = scalar_tail_coefficient * terminal_norms

    angle_rows = []
    for angle_index, angle in enumerate(grid["entry_angles"]):
        factor = _spectral_interval_factor(
            times,
            propagated["raw_l2"][:, angle_index],
            float(terminal_norms[angle_index]),
            spectral["principal_decay_barta_lower"],
            spectral["boundary_operator_norm_gershgorin_upper"],
            return_density,
        )
        scalar_gain_upper = (
            SCALAR_QUADRATURE_STRESS * float(scalar_finite[angle_index])
            + float(scalar_tail_bounds[angle_index])
        )
        response = math.sqrt(
            scalar_gain_upper
            * return_density.TRACE_L4_FORM_CONSTANT
            * factor["spectral_interval_factor"]
        )
        angle_rows.append(
            {
                "angle": float(angle),
                "stressed_scalar_gain_with_spectral_tail": scalar_gain_upper,
                "scalar_spectral_tail_bound": float(
                    scalar_tail_bounds[angle_index]
                ),
                "spectral_interval_factor": factor[
                    "spectral_interval_factor"
                ],
                "spectral_tail_factor_contribution": factor[
                    "spectral_tail_factor_contribution"
                ],
                "response_with_spectral_tail": response,
            }
        )
    worst = max(
        angle_rows, key=lambda item: item["response_with_spectral_tail"]
    )
    row["density"] = {
        "terminal_time": float(times[-1]),
        "time_sample_count": len(times),
        "maximum_terminal_state": float(np.max(terminal_state)),
        "maximum_terminal_mass_weighted_norm": float(np.max(terminal_norms)),
        "maximum_scalar_spectral_tail_bound": float(
            np.max(scalar_tail_bounds)
        ),
        "maximum_spectral_tail_factor_contribution": max(
            item["spectral_tail_factor_contribution"] for item in angle_rows
        ),
        "maximum_response_with_spectral_tail": worst[
            "response_with_spectral_tail"
        ],
        "worst_entry_angle": worst["angle"],
        "fitted_tail_used": False,
        "finite_time_sample_stress": FINITE_TIME_STRESS,
        "scalar_quadrature_stress": SCALAR_QUADRATURE_STRESS,
        "angle_rows": angle_rows,
    }
    return row


def audit(
    spacing: float = 0.12,
    x_half_widths: tuple[float, ...] = (4.2, 5.25, 6.3),
    run_density: bool = True,
) -> dict[str, object]:
    fem = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "reversible_fem_for_spectral_width",
    )
    return_density = _load_module(
        "neutral_strip_return_density_pilot.py",
        "return_density_for_spectral_width",
    )
    width_rows = [
        _width_row(
            fem,
            return_density,
            spacing,
            x_half_width,
            run_density,
        )
        for x_half_width in x_half_widths
    ]
    response_spread = None
    if run_density:
        responses = [
            row["density"]["maximum_response_with_spectral_tail"]
            for row in width_rows
        ]
        response_spread = float(max(responses) - min(responses))

    result = {
        "model": "rho=0 reversible boundary-FEM spectral tail and x-width audit",
        "spacing": spacing,
        "terminal_time": TERMINAL_TIME,
        "x_half_widths": list(x_half_widths),
        "width_rows": width_rows,
        "maximum_width_response_spread": response_spread,
        "finite_matrix_decay_enclosed_by_high_precision_barta": bool(all(
            row["spectral"]["principal_decay_barta_lower"] > 0.0
            and row["spectral"]["principal_decay_barta_width"] < 1.0e-6
            for row in width_rows
        )),
        "boundary_operator_norm_has_analytic_upper_bound": bool(all(
            row["spectral"]["boundary_operator_norm_gershgorin_upper"]
            >= row["spectral"]["boundary_operator_norm_pilot"]
            for row in width_rows
        )),
        "fitted_tail_eliminated": (
            bool(run_density)
            and all(not row["density"]["fitted_tail_used"] for row in width_rows)
        ),
        "spectral_tail_factor_below_1e_4": bool(
            run_density
            and max(
                row["density"][
                    "maximum_spectral_tail_factor_contribution"
                ]
                for row in width_rows
            )
            < 1.0e-4
        ),
        "x_width_response_pilot_controlled": bool(
            run_density and response_spread is not None and response_spread < 0.003
        ),
        "finite_time_window_maxima_certified": False,
        "scalar_time_quadrature_certified": False,
        "x_truncation_analytically_removed": False,
        "continuum_return_response_certified": False,
        "scope_guard": (
            "The post-T tail is bounded for each stored floating FEM matrix "
            "using a high-precision Barta decay lower bound and a Gershgorin "
            "boundary-operator upper bound; no fitted tail remains. The "
            "finite-time maxima and scalar quadrature retain explicit "
            "empirical stresses, and width stability is numerical rather "
            "than an analytic removal of the artificial x boundary."
        ),
        "next_gate": (
            "certify finite-time window maxima and scalar quadrature, derive "
            "an analytic bound for paths lost at the x truncation, then "
            "address polygonal and weighted FEM continuum error"
        ),
    }
    checks = (
        result["finite_matrix_decay_enclosed_by_high_precision_barta"],
        result["boundary_operator_norm_has_analytic_upper_bound"],
        not result["finite_time_window_maxima_certified"],
        not result["scalar_time_quadrature_certified"],
        not result["x_truncation_analytically_removed"],
        not result["continuum_return_response_certified"],
    )
    if run_density:
        checks += (
            result["fitted_tail_eliminated"],
            result["spectral_tail_factor_below_1e_4"],
            result["x_width_response_pilot_controlled"],
        )
    result["all_positive_spectral_tail_width_checks_pass"] = bool(all(checks))
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
