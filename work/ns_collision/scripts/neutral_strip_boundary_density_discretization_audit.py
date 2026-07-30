"""Audit whether fitted-edge strip flux defines a continuum boundary L2 density."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import expm_multiply


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cycle_defect_row(grid: dict[str, object]) -> dict[str, object]:
    generator = grid["generator"].tocsr()
    lookup = grid["node_lookup"]
    defects: list[tuple[float, tuple[int, int]]] = []
    for (x_index, y_index), lower_left in lookup.items():
        keys = (
            (x_index, y_index),
            (x_index + 1, y_index),
            (x_index + 1, y_index + 1),
            (x_index, y_index + 1),
        )
        if not all(key in lookup for key in keys):
            continue
        nodes = [lookup[key] for key in keys]
        forward = 0.0
        reverse = 0.0
        valid = True
        for index in range(4):
            source = nodes[index]
            target = nodes[(index + 1) % 4]
            forward_rate = float(generator[source, target])
            reverse_rate = float(generator[target, source])
            if forward_rate <= 0.0 or reverse_rate <= 0.0:
                valid = False
                break
            forward += math.log(forward_rate)
            reverse += math.log(reverse_rate)
        if valid:
            defects.append((abs(forward - reverse), (x_index, y_index)))

    maximum, square = max(defects)
    x_index, y_index = square
    return {
        "y_intervals": int(round(2.0 * grid["strip_half_width"] / grid["spacing"])),
        "spacing": grid["spacing"],
        "state_count": generator.shape[0],
        "inner_boundary_edge_count": len(grid["inner_boundary_edges"]),
        "elementary_square_count": len(defects),
        "maximum_log_detailed_balance_cycle_defect": maximum,
        "maximum_defect_square_lower_left": [
            float(grid["xs"][x_index]),
            float(grid["ys"][y_index]),
        ],
        "squares_with_defect_above_1e_10": sum(
            defect > 1.0e-10 for defect, _ in defects
        ),
        "discrete_generator_reversible": maximum < 1.0e-10,
    }


def _boundary_spacing_row(grid: dict[str, object]) -> dict[str, float]:
    angles = np.sort(
        np.asarray(
            [
                math.atan2(hit_y, hit_x) % (2.0 * math.pi)
                for _, _, hit_x, hit_y in grid["inner_boundary_edges"]
            ]
        )
    )
    gaps = np.diff(np.concatenate((angles, angles[:1] + 2.0 * math.pi)))
    return {
        "minimum_edge_angle_gap": float(np.min(gaps)),
        "maximum_edge_angle_gap": float(np.max(gaps)),
        "maximum_to_minimum_edge_gap_ratio": float(
            np.max(gaps) / np.min(gaps)
        ),
    }


def _bin_sensitivity_row(
    resolvent,
    return_density,
    axial,
    y_intervals: int = 40,
    rho: float = 0.0,
    entry_angle_count: int = 32,
    bin_counts: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024),
) -> dict[str, object]:
    grid = resolvent._build_generator(y_intervals, rho)
    entry_angles = np.linspace(
        0.0, 2.0 * math.pi, entry_angle_count, endpoint=False
    )
    state = return_density._entry_matrix(grid, entry_angles)
    generator_transpose = grid["generator"].transpose().tocsc()
    inner_matrices = {
        boundary_bins: return_density._inner_rate_matrix(
            grid, boundary_bins
        )
        for boundary_bins in bin_counts
    }
    raw_l2_rows: dict[int, list[np.ndarray]] = {
        boundary_bins: [] for boundary_bins in bin_counts
    }
    scalar_density_rows: list[np.ndarray] = []
    times: list[float] = []
    current_time = 0.0
    segments = (
        (0.05, 26),
        (0.2, 31),
        (1.0, 41),
        (4.0, 31),
        (12.0, 33),
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
            patch_mass, axial_l2 = return_density._axial_factors(time, rho)
            deformation = math.exp(time)
            aggregate_flux = np.asarray(grid["inner_rates"] @ snapshot)
            times.append(time)
            scalar_density_rows.append(
                deformation * patch_mass * aggregate_flux
            )
            for boundary_bins, inner_matrix in inner_matrices.items():
                binned_flux = np.asarray(
                    inner_matrix.transpose() @ snapshot
                ).T
                bin_width = 2.0 * math.pi / boundary_bins
                transverse_l2 = np.sqrt(
                    np.sum(binned_flux**2, axis=1) / bin_width
                )
                raw_l2_rows[boundary_bins].append(
                    deformation * axial_l2 * transverse_l2
                )
        state = trajectory[-1]
        current_time = segment_end

    time_array = np.asarray(times)
    patched_return, _, _ = axial._integrate_patched_return(grid, rho, 1.0)
    exact_scalar = return_density._entry_matrix(
        grid, entry_angles
    ).T @ patched_return
    integrated_scalar = np.trapezoid(
        np.asarray(scalar_density_rows), time_array, axis=0
    )

    bin_rows = []
    for boundary_bins in bin_counts:
        raw_l2 = np.asarray(raw_l2_rows[boundary_bins])
        maximum_response = -math.inf
        maximum_factor = -math.inf
        worst_angle = 0.0
        for angle_index, angle in enumerate(entry_angles):
            factor = return_density._sampled_interval_factor(
                time_array, raw_l2[:, angle_index]
            )["sampled_stressed_factor"]
            response = math.sqrt(
                float(exact_scalar[angle_index])
                * return_density.TRACE_L4_FORM_CONSTANT
                * factor
            )
            if response > maximum_response:
                maximum_response = response
                maximum_factor = factor
                worst_angle = float(angle)
        bin_rows.append(
            {
                "boundary_bin_count": boundary_bins,
                "bin_width": 2.0 * math.pi / boundary_bins,
                "maximum_raw_interval_factor": maximum_factor,
                "maximum_raw_trace_response_at_alpha_zero": (
                    maximum_response
                ),
                "worst_entry_angle": worst_angle,
            }
        )

    baseline = next(
        row for row in bin_rows if row["boundary_bin_count"] == 32
    )
    finest = bin_rows[-1]
    return {
        "y_intervals": y_intervals,
        "spacing": grid["spacing"],
        "rho": rho,
        "entry_angle_count": entry_angle_count,
        "inner_boundary_edge_count": len(grid["inner_boundary_edges"]),
        "time_sample_count": len(time_array),
        "maximum_time": float(time_array[-1]),
        "scalar_resolvent_recovery_error": float(
            np.max(np.abs(integrated_scalar - exact_scalar))
        ),
        "maximum_terminal_state": float(np.max(state)),
        "bin_rows": bin_rows,
        "finest_boundary_bin_count": finest["boundary_bin_count"],
        "response_ratio_finest_bins_to_32_bins": (
            finest["maximum_raw_trace_response_at_alpha_zero"]
            / baseline["maximum_raw_trace_response_at_alpha_zero"]
        ),
    }


def audit(
    structural_meshes: tuple[int, ...] = (30, 40, 50, 60),
    density_mesh: int = 40,
    bin_counts: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024),
) -> dict[str, object]:
    resolvent = _load_module(
        "neutral_strip_branch_resolvent_pilot.py",
        "resolvent_for_boundary_discretization",
    )
    return_density = _load_module(
        "neutral_strip_return_density_pilot.py",
        "return_density_for_boundary_discretization",
    )
    axial = _load_module(
        "neutral_strip_axial_patch_branch_pilot.py",
        "axial_for_boundary_discretization",
    )

    structural_rows = []
    for y_intervals in structural_meshes:
        grid = resolvent._build_generator(y_intervals, 0.0)
        row = _cycle_defect_row(grid)
        row.update(_boundary_spacing_row(grid))
        structural_rows.append(row)
    density_row = _bin_sensitivity_row(
        resolvent,
        return_density,
        axial,
        y_intervals=density_mesh,
        bin_counts=bin_counts,
    )

    result = {
        "model": "static rho=0 neutral-strip fitted-edge boundary flux",
        "discrete_boundary_measure": (
            "mu_h(t)=sum_e rate_e*state_node(e,t)*delta_angle(e)"
        ),
        "histogram_identity_after_atom_separation": (
            "||P_B mu_h||_2^2=(B/(2*pi))*sum_e mass_e(t)^2"
        ),
        "fixed_mesh_histogram_L2_limit_finite": False,
        "fixed_mesh_histogram_response_scaling_after_atom_separation": (
            "K_R is proportional to B^(1/4)"
        ),
        "structural_rows": structural_rows,
        "density_row": density_row,
        "shortley_weller_generator_reversible_on_all_meshes": all(
            row["discrete_generator_reversible"] for row in structural_rows
        ),
        "previous_32_bin_response_is_a_continuum_upper_bound": False,
        "continuum_boundary_density_certified": False,
        "required_replacement": (
            "a conservative boundary-fitted finite-volume or lumped-FEM "
            "generator with disjoint boundary faces, physical face lengths, "
            "and coupled interior/boundary refinement"
        ),
        "scope_guard": (
            "This audit does not show that the continuum return density is "
            "large or divergent. It shows that the current edge-atom "
            "histogram cannot certify its L2 norm and that the monotone "
            "Shortley-Weller generator has no exact reversible volume "
            "measure near the fitted circle. The earlier K_R remains a "
            "coarse finite-state diagnostic only."
        ),
        "next_gate": (
            "replace the curved-boundary discretization by a conservative "
            "reversible mesh with genuine boundary-face quadrature, then "
            "repeat coupled mesh and time-tail convergence"
        ),
    }
    checks = (
        not result["fixed_mesh_histogram_L2_limit_finite"],
        not result["shortley_weller_generator_reversible_on_all_meshes"],
        max(
            row["maximum_log_detailed_balance_cycle_defect"]
            for row in structural_rows
        )
        > 1.0e-3,
        density_row["response_ratio_finest_bins_to_32_bins"] > 1.1,
        density_row["scalar_resolvent_recovery_error"] < 0.03,
        density_row["maximum_terminal_state"] < 1.0e-8,
        not result["previous_32_bin_response_is_a_continuum_upper_bound"],
        not result["continuum_boundary_density_certified"],
    )
    result["all_positive_boundary_discretization_checks_pass"] = all(checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
