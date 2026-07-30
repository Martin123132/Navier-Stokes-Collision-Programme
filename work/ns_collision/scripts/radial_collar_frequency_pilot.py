"""Time-harmonic axisymmetric stress test for the radial collar trace."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import splu


OUTER_RADIUS = 2.0
HALF_HEIGHT = 0.75


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frequency_rows(
    trace_module,
    radial_cells: int,
    axial_cells: int,
    collar_distance: float,
    transverse_strain: float,
    frequencies: tuple[float, ...],
    cutoff_factor: float,
) -> list[dict[str, float | int]]:
    operator, mass, radial_grid, axial_grid = trace_module._assemble_collar(
        radial_cells,
        axial_cells,
        collar_distance,
        transverse_strain,
    )
    radial_mesh, axial_mesh = np.meshgrid(
        radial_grid, axial_grid, indexing="ij"
    )
    radial_support = 1.0 - collar_distance
    axial_support = HALF_HEIGHT - collar_distance
    tolerance = 1.0e-12
    strict_core = (
        (radial_mesh < radial_support - tolerance)
        & (axial_mesh < axial_support - tolerance)
    )
    interface = (
        (
            np.isclose(radial_mesh, radial_support, atol=tolerance)
            & (axial_mesh <= axial_support + tolerance)
        )
        | (
            np.isclose(axial_mesh, axial_support, atol=tolerance)
            & (radial_mesh <= radial_support + tolerance)
        )
    )
    absorbing_boundary = np.zeros_like(strict_core)
    absorbing_boundary[-1, :] = True
    absorbing_boundary[:, -1] = True
    collar_unknown = ~(strict_core | absorbing_boundary)
    free = np.flatnonzero((collar_unknown & ~interface).ravel())
    boundary = np.flatnonzero(interface.ravel())
    unknown = np.concatenate([free, boundary])
    mass_free = mass[free][:, free]
    mass_interface = mass[free][:, boundary]
    collar_mass = mass[unknown][:, unknown]
    operator_free = operator[free][:, free]
    operator_interface = operator[free][:, boundary]
    radial_point = int(round(1.0 / (OUTER_RADIUS / radial_cells)))
    point_global = radial_point * (axial_cells + 1)
    point_position = int(np.flatnonzero(free == point_global)[0])
    rows = []
    for frequency in frequencies:
        free_system = (
            operator_free + 1j * frequency * mass_free
        ).tocsc()
        interface_system = (
            operator_interface + 1j * frequency * mass_interface
        )
        factorization = splu(free_system)
        free_poisson = factorization.solve(
            -interface_system.toarray()
        )
        poisson = np.vstack(
            [
                free_poisson,
                np.eye(len(boundary), dtype=np.complex128),
            ]
        )
        boundary_gram = poisson.conj().T @ (collar_mass @ poisson)
        evaluation = free_poisson[point_position]
        riesz = solve(
            boundary_gram,
            evaluation.conj(),
            assume_a="her",
            check_finite=False,
        )
        norm_squared = float(np.real(evaluation @ riesz))
        trace_norm = math.sqrt(max(norm_squared, 0.0))
        rows.append(
            {
                "radial_cells": radial_cells,
                "axial_cells": axial_cells,
                "collar_distance": collar_distance,
                "transverse_strain": transverse_strain,
                "temporal_frequency": frequency,
                "time_harmonic_L2_to_point_trace_norm": trace_norm,
                "time_harmonic_chi_pilot": cutoff_factor * trace_norm,
            }
        )
    return rows


def audit() -> dict[str, object]:
    trace_module = _load_module(
        "radial_collar_trace_pilot.py", "radial_trace_for_frequency"
    )
    cutoff_module = _load_module(
        "radial_barrier_cutoff_energy_pilot.py",
        "radial_cutoff_for_frequency",
    )
    cutoff = cutoff_module.audit()
    cutoff_factors = {
        row["collar_distance"]: row[
            "sqrt_energy_over_m0_over_barrier_gain"
        ]
        for row in cutoff["fine_rows"]
    }
    radial_cells, axial_cells = 100, 75
    distances = (0.20, 0.30, 0.40)
    transverse_strains = (-1.0, 0.0, 0.5)
    frequencies = (
        0.0,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        10.0,
        12.0,
        16.0,
        32.0,
    )
    rows = []
    for distance in distances:
        for transverse_strain in transverse_strains:
            rows.extend(
                _frequency_rows(
                    trace_module,
                    radial_cells,
                    axial_cells,
                    distance,
                    transverse_strain,
                    frequencies,
                    cutoff_factors[distance],
                )
            )

    maximum_rows = []
    monotonic_rows = []
    for distance in distances:
        for transverse_strain in transverse_strains:
            family = [
                row
                for row in rows
                if row["collar_distance"] == distance
                and row["transverse_strain"] == transverse_strain
            ]
            maximum_rows.append(
                max(
                    family,
                    key=lambda row: row[
                        "time_harmonic_L2_to_point_trace_norm"
                    ],
                )
            )
            monotonic_rows.append(
                {
                    "collar_distance": distance,
                    "transverse_strain": transverse_strain,
                    "maximum_occurs_at_zero_frequency": bool(
                        max(
                            family,
                            key=lambda row: row[
                                "time_harmonic_L2_to_point_trace_norm"
                            ],
                        )["temporal_frequency"]
                        == 0.0
                    ),
                    "sampled_norms_nonincreasing": all(
                        later[
                            "time_harmonic_L2_to_point_trace_norm"
                        ]
                        <= earlier[
                            "time_harmonic_L2_to_point_trace_norm"
                        ]
                        + 1.0e-10
                        for earlier, later in zip(family, family[1:])
                    ),
                }
            )

    result: dict[str, object] = {
        "status": (
            "time-harmonic axisymmetric finite-element stress test; "
            "not a nonautonomous enclosure"
        ),
        "frequency_equation": (
            "(-Delta-B_a y.grad-1+i*omega)w_omega=0 on D\\E_d"
        ),
        "sampled_frequencies": frequencies,
        "rows": rows,
        "maximum_rows": maximum_rows,
        "monotonicity_rows": monotonic_rows,
        "all_sampled_maxima_at_zero_frequency": all(
            row["maximum_occurs_at_zero_frequency"]
            for row in monotonic_rows
        ),
        "all_sampled_frequency_norms_nonincreasing": all(
            row["sampled_norms_nonincreasing"] for row in monotonic_rows
        ),
        "static_worst_frequency_supported": all(
            row["maximum_occurs_at_zero_frequency"]
            for row in monotonic_rows
        ),
        "nonzero_frequency_resonance_detected": any(
            not row["maximum_occurs_at_zero_frequency"]
            for row in monotonic_rows
        ),
        "all_sampled_d0p2_or_wider_chi_below_two": all(
            row["time_harmonic_chi_pilot"] < 2.0
            for row in rows
        ),
        "scope_guard": (
            "frequency decay for each fixed axisymmetric affine matrix "
            "does not prove a common bound for switching, rotating, or "
            "non-axisymmetric measurable affine histories"
        ),
    }
    positive_checks = (
        result["nonzero_frequency_resonance_detected"],
        not result["static_worst_frequency_supported"],
        result["all_sampled_d0p2_or_wider_chi_below_two"],
        len(maximum_rows) == len(distances) * len(transverse_strains),
    )
    result["all_positive_frequency_pilot_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
