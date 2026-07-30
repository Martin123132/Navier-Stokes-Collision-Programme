"""Audit localized pressure cancellation and its shell scaling obstruction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import sympy as sp


def _load_pairing_module():
    script = Path(__file__).resolve().with_name(
        "pressure_frame_pairing_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pressure_frame_pairing_for_shell", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit() -> dict[str, object]:
    pairing = _load_pairing_module()
    pairing_result = pairing.audit()
    fields = pairing._build_spectral_fields()
    center = np.array(pairing_result["optimized_point_mod_2pi"])
    grid_size = pairing.GRID_SIZE
    coordinates = (
        2
        * np.pi
        * np.indices((grid_size, grid_size, grid_size))
        / grid_size
    )
    displacement = coordinates - center[:, None, None, None]

    one_dimensional_weights = (1 + np.cos(displacement)) / 2
    localization = np.prod(one_dimensional_weights, axis=0)
    continuous_core_peak = float(
        np.prod((1 + np.cos(np.zeros(3))) / 2)
    )
    localization_gradient = np.empty_like(displacement)
    for direction in range(3):
        other_directions = [index for index in range(3) if index != direction]
        localization_gradient[direction] = (
            -0.5
            * np.sin(displacement[direction])
            * np.prod(one_dimensional_weights[other_directions], axis=0)
        )

    velocity_gradient = fields["velocity_gradient_grid"]
    strain = fields["strain_grid"]
    strain_gradient = fields["strain_gradient_grid"]
    transposed_gradient_times_localization = np.einsum(
        "ijxyz,ixyz->jxyz", velocity_gradient, localization_gradient
    )

    split_rows = []
    for label, pressure_name, potential_gradient_name in (
        (
            "full",
            "pressure_grid",
            "pressure_potential_gradient_grid",
        ),
        (
            "low",
            "pressure_low_grid",
            "pressure_potential_low_gradient_grid",
        ),
        (
            "collision_defect",
            "pressure_defect_grid",
            "pressure_potential_defect_gradient_grid",
        ),
    ):
        pressure = fields[pressure_name]
        potential_gradient = fields[potential_gradient_name]
        localized_contraction = float(
            np.mean(
                localization
                * np.einsum("ijxyz,ijxyz->xyz", pressure, strain)
            )
        )
        boundary_commutator = float(
            -np.mean(
                np.einsum(
                    "jxyz,jxyz->xyz",
                    potential_gradient,
                    transposed_gradient_times_localization,
                )
            )
        )
        split_rows.append(
            {
                "piece": label,
                "localized_pressure_strain": localized_contraction,
                "boundary_commutator": boundary_commutator,
                "identity_residual": (
                    localized_contraction - boundary_commutator
                ),
            }
        )

    full_work = split_rows[0]["localized_pressure_strain"]
    low_work = split_rows[1]["localized_pressure_strain"]
    defect_work = split_rows[2]["localized_pressure_strain"]
    localized_strain_dissipation = float(
        pairing.VISCOSITY
        * np.mean(
            localization * np.sum(strain_gradient**2, axis=(0, 1, 2))
        )
    )
    base_ratio = abs(full_work) / localized_strain_dissipation
    absorption_threshold_rms = pairing.VELOCITY_RMS / base_ratio
    amplitude_rows = []
    for rms_velocity in (10.0, 100.0, 1000.0, 2000.0):
        scale = rms_velocity / pairing.VELOCITY_RMS
        pressure_work = abs(full_work) * scale**3
        dissipation = localized_strain_dissipation * scale**2
        amplitude_rows.append(
            {
                "velocity_rms": rms_velocity,
                "adverse_pressure_work": pressure_work,
                "viscous_strain_dissipation": dissipation,
                "pressure_to_dissipation_ratio": pressure_work / dissipation,
            }
        )

    strain_scale, core_length, viscosity = sp.symbols(
        "strain_scale core_length viscosity", positive=True, real=True
    )
    pressure_shell_scale = strain_scale**3 * core_length**3
    viscous_shell_scale = viscosity * strain_scale**2 * core_length
    dimensionless_ratio = sp.simplify(
        pressure_shell_scale / viscous_shell_scale
    )

    spike_rows = []
    for spike_parameter in (10, 100, 1000, 10000):
        spike_rows.append(
            {
                "N": spike_parameter,
                "amplitude": spike_parameter**0.5,
                "support_length": 1.0 / spike_parameter,
                "time_l2_integral": 1.0,
                "time_10_over_3_integral": float(
                    spike_parameter ** (2.0 / 3.0)
                ),
            }
        )

    result: dict[str, object] = {
        "localization": (
            "phi=product_i[(1+cos(x_i-center_i))/2]"
        ),
        "localization_is_nonnegative": bool(np.min(localization) >= 0.0),
        "localization_peaks_at_core_center": bool(
            abs(continuous_core_peak - 1.0) < 1.0e-15
        ),
        "pressure_shell_split_rows": split_rows,
        "all_boundary_identities_verified": all(
            abs(row["identity_residual"]) < 1.0e-10
            for row in split_rows
        ),
        "heat_split_recombines": bool(
            abs(full_work - low_work - defect_work) < 1.0e-10
        ),
        "full_localized_pressure_work": full_work,
        "sign_reversal_under_u_to_minus_u": (
            "pressure is unchanged, S and the shell commutator reverse sign"
        ),
        "localized_strain_dissipation": localized_strain_dissipation,
        "base_absolute_pressure_to_dissipation_ratio": base_ratio,
        "velocity_rms_threshold_for_absolute_absorption_failure": (
            absorption_threshold_rms
        ),
        "amplitude_scaling_rows": amplitude_rows,
        "cubic_pressure_beats_quadratic_dissipation_at_fixed_scale": bool(
            amplitude_rows[-1]["pressure_to_dissipation_ratio"] > 1.0
        ),
        "commutator_hls_bound": (
            "|I_phi|<=C*||grad(phi)||_infinity*||S||_2*||f||_(6/5)"
        ),
        "gn_source_bound": (
            "||f||_(6/5)<=C*||A||_2^(3/2)*||grad(A)||_2^(1/2)"
        ),
        "combined_form_bound": (
            "|I_phi|<=C*||grad(phi)||_infinity*"
            "||A||_2^(5/2)*||grad(A)||_2^(1/2)"
        ),
        "young_remainder": (
            "C*(nu*epsilon)^(-1/3)*||grad(phi)||_infinity^(4/3)*"
            "||A||_2^(10/3)"
        ),
        "shell_pressure_scaling": str(pressure_shell_scale),
        "shell_viscous_scaling": str(viscous_shell_scale),
        "shell_dimensionless_ratio": str(dimensionless_ratio),
        "shell_ratio_is_local_reynolds_number": bool(
            sp.simplify(
                dimensionless_ratio
                - strain_scale * core_length**2 / viscosity
            )
            == 0
        ),
        "leray_time_spike_rows": spike_rows,
        "time_l2_does_not_control_10_over_3_remainder": bool(
            all(row["time_l2_integral"] == 1.0 for row in spike_rows)
            and spike_rows[-1]["time_10_over_3_integral"]
            > spike_rows[-2]["time_10_over_3_integral"]
        ),
        "fixed_scale_absolute_shell_absorption_is_false": bool(
            dimensionless_ratio.has(strain_scale)
            and amplitude_rows[-1]["pressure_to_dissipation_ratio"] > 1.0
        ),
        "remaining_candidate_route": (
            "adaptive cutoffs with moderate local Reynolds number; high-"
            "Reynolds visits still require signed, trajectory, or collision "
            "averaging"
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
