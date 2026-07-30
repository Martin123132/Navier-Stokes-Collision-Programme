"""Audit an explicit escape from the compatible twelve-edge objective."""

from __future__ import annotations

import argparse
import ctypes
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from compatible_eight_cell_cubic_graph_audit import (
    VERTICES,
    _cubic_energy,
)
from cross_shell_modulated_wave_gate_audit import (
    _energy_flux,
    _field_sum,
)
from primitive_hhl_chain_hardy_envelope_audit import (
    _translated_vertex_fisher,
    _translated_vertex_load,
)
from separable_annular_pressure_schur_no_go_audit import (
    TRANSLATION,
    _family_arrays,
    _high_field,
    _low_field,
    _mixed_difference_fisher,
    _resonant_component_loads,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "compatible_edge_annular_escape_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "compatible_eight_cell_cubic_graph_audit_v1.json"
    ): "067625c4b44aa6085ff2b59cbbcef351253dcbd5b1fb9f0eac641fdd7a48682c",
    (
        "work/ns_collision/results/"
        "separable_annular_pressure_schur_no_go_audit_v1.json"
    ): "16579e713c5bacb7b19bb9e3d63f059b9f0915588013e40aa49fdb8bf0bfea0b",
    (
        "work/ns_collision/results/"
        "annular_eight_vertex_heat_window_gate_audit_v1.json"
    ): "5313001d5a136babf1be6d99b66767db4161e526cd08158631cde2a68c942789",
}
ALGORITHM_REVISION = "compatible-edge-annular-escape-v1"
DEFAULT_SIZES = (3, 5, 7, 9, 13, 17, 25, 33, 49, 65, 137)
PLUS_VERTEX = (1, 1, 1)
MINUS_VERTEX = (-1, -1, -1)
DELTA_CUBIC_ENERGY = Fraction(75, 256)
Wave = tuple[int, int, int]


def _lower_process_priority() -> None:
    if os.name != "nt":
        return
    below_normal_priority_class = 0x00004000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    kernel32.SetPriorityClass(handle, below_normal_priority_class)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _vertex_label(vertex: Wave) -> str:
    return "".join("+" if value == 1 else "-" for value in vertex)


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _prerequisite_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    payloads = {}
    for relative, expected in PREREQUISITES.items():
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[relative] = payload
        actual = _sha256(path)
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "all_positive_checks_pass": payload.get(
                    "all_positive_checks_pass"
                ),
                "matches": bool(
                    actual == expected
                    and payload.get("all_positive_checks_pass") is True
                ),
            }
        )
    return (
        {
            "rows": rows,
            "all_checks_pass": all(row["matches"] for row in rows),
        },
        payloads,
    )


def _delta_weights(vertex: Wave) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(1) if candidate == vertex else Fraction(0)
        for candidate in VERTICES
    )


def _joint_ray_optimum(
    load: float,
    high_fisher: float,
    mass: float,
    cubic_energy: float,
    viscosity: float,
) -> dict[str, Any]:
    """Optimize first over the signed low amplitude, then ray scale."""

    if mass <= 0.0:
        raise ValueError("coefficient mass must be positive")
    if cubic_energy <= 0.0:
        raise ValueError("cubic energy must be positive")
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive")

    absolute_load = abs(load)
    low_amplitude = absolute_load / (viscosity * mass)
    linear_margin = (
        absolute_load**2 / (2.0 * viscosity * mass)
        - viscosity * high_fisher
    )
    cubic_coefficient = viscosity * cubic_energy / 16.0
    if linear_margin > 0.0:
        coefficient_scale = math.sqrt(
            linear_margin / (3.0 * cubic_coefficient)
        )
        maximum = (
            2.0 * linear_margin * coefficient_scale / 3.0
        )
    else:
        coefficient_scale = 0.0
        maximum = 0.0

    bounded_scale_objective = linear_margin - cubic_coefficient
    low_stationarity_residual = abs(
        absolute_load
        - viscosity * mass * low_amplitude
    )
    coefficient_stationarity_residual = (
        abs(
            linear_margin
            - 3.0
            * cubic_coefficient
            * coefficient_scale**2
        )
        if linear_margin > 0.0
        else 0.0
    )
    replayed_maximum = (
        coefficient_scale * linear_margin
        - cubic_coefficient * coefficient_scale**3
    )
    return {
        "absolute_unit_low_load": absolute_load,
        "high_field_weighted_Fisher": high_fisher,
        "coefficient_mass": mass,
        "coefficient_cubic_energy": cubic_energy,
        "viscosity": viscosity,
        "optimal_oriented_low_amplitude": low_amplitude,
        "optimized_linear_margin": linear_margin,
        "optimal_coefficient_scale": coefficient_scale,
        "optimized_objective": maximum,
        "replayed_optimized_objective": replayed_maximum,
        "bounded_coefficient_scale_one_objective": (
            bounded_scale_objective
        ),
        "low_amplitude_stationarity_residual": (
            low_stationarity_residual
        ),
        "coefficient_scale_stationarity_residual": (
            coefficient_stationarity_residual
        ),
        "positive_escape": linear_margin > 0.0,
        "bounded_scale_one_escape": bounded_scale_objective > 0.0,
        "all_checks_pass": bool(
            low_stationarity_residual < 2.0e-15
            and coefficient_stationarity_residual < 2.0e-14
            and abs(maximum - replayed_maximum) < 2.0e-14
        ),
    }


def _finite_row(size: int, viscosity: float = 1.0) -> dict[str, Any]:
    waves, velocity, parity = _family_arrays(
        (size, size, size), 2 * size
    )
    loads = _resonant_component_loads(waves, velocity)
    fisher = _mixed_difference_fisher(waves, velocity, parity)
    signed_load = loads["combined"]
    optimum = _joint_ray_optimum(
        signed_load,
        fisher,
        1.0,
        float(DELTA_CUBIC_ENERGY),
        viscosity,
    )
    return {
        "size": size,
        "signed_complete_HHL_load": signed_load,
        "absolute_complete_HHL_load": abs(signed_load),
        "plus_vertex_high_Fisher": fisher,
        "load_over_size": signed_load / size,
        "Fisher_times_size_cubed": fisher * size**3,
        "maximum_imaginary_load_residual": loads[
            "maximum_imaginary_residual"
        ],
        "ray_optimization": optimum,
        "all_checks_pass": bool(
            signed_load < 0.0
            and fisher > 0.0
            and loads["maximum_imaginary_residual"] < 4.0e-12
            and optimum["all_checks_pass"]
        ),
    }


def _full_field_support_replay() -> dict[str, Any]:
    low = _low_field()
    low_flux = _energy_flux(low)
    low_loads = {
        _vertex_label(vertex): _translated_vertex_load(
            low_flux, 1, vertex, TRANSLATION
        )
        for vertex in VERTICES
    }
    low_fishers = {
        _vertex_label(vertex): _translated_vertex_fisher(
            low, 1, vertex, TRANSLATION
        )
        for vertex in VERTICES
    }

    waves, velocity, _ = _family_arrays((3, 3, 3), 6)
    high = _high_field(waves, velocity)
    signed_hhl_load = _resonant_component_loads(
        waves, velocity
    )["combined"]
    high_fisher = _mixed_difference_fisher(
        waves,
        velocity,
        np.where(
            (
                np.arange(1, 4)[:, None, None]
                + np.arange(1, 4)[None, :, None]
                + np.arange(1, 4)[None, None, :]
            )
            % 2
            == 0,
            1.0,
            -1.0,
        ),
    )

    amplitudes = (-2.0, -1.0, 0.0, 1.0, 2.0)
    full_loads = {}
    full_fishers = {}
    for amplitude in amplitudes:
        field = _field_sum(high, low, amplitude)
        full_loads[str(amplitude)] = float(
            _translated_vertex_load(
                _energy_flux(field),
                1,
                PLUS_VERTEX,
                TRANSLATION,
            ).real
        )
        full_fishers[str(amplitude)] = float(
            _translated_vertex_fisher(
                field,
                1,
                PLUS_VERTEX,
                TRANSLATION,
            ).real
        )

    load_residual = max(
        abs(full_loads[str(amplitude)] - amplitude * signed_hhl_load)
        for amplitude in amplitudes
    )
    fisher_residual = max(
        abs(
            full_fishers[str(amplitude)]
            - high_fisher
            - amplitude**2 / 2.0
        )
        for amplitude in amplitudes
    )
    low_load_residual = max(abs(value) for value in low_loads.values())
    low_fisher_residual = max(
        abs(value - 0.5) for value in low_fishers.values()
    )
    return {
        "replay_size": 3,
        "partition_stencil": "[-1,1]^3 minus the origin",
        "positive_high_first_coordinate_interval": "[2N,3N-1]",
        "HHH_first_coordinate_gap": (
            "Every mixed-sign HHH output has |k_1|>=N+1>1; "
            "same-sign outputs are farther away."
        ),
        "HLL_first_coordinate_gap": "|k_1|>=2N>1",
        "high_low_Fisher_cross_gap": "|k_1|>=2N>1",
        "low_wave_is_plane_shear": True,
        "low_only_flux_loads": {
            label: float(value.real)
            for label, value in low_loads.items()
        },
        "low_only_Fisher_by_vertex": {
            label: float(value.real)
            for label, value in low_fishers.items()
        },
        "low_only_Fisher_exact_value": "1/2",
        "signed_unit_low_HHL_load": signed_hhl_load,
        "high_only_plus_Fisher": high_fisher,
        "full_field_plus_load_by_low_amplitude": full_loads,
        "full_field_plus_Fisher_by_low_amplitude": full_fishers,
        "maximum_full_load_vs_linear_HHL_residual": load_residual,
        "maximum_full_Fisher_vs_additive_formula_residual": (
            fisher_residual
        ),
        "maximum_low_only_load_residual": low_load_residual,
        "maximum_low_Fisher_half_residual": float(
            low_fisher_residual
        ),
        "all_checks_pass": bool(
            load_residual < 4.0e-12
            and fisher_residual < 4.0e-12
            and low_load_residual < 4.0e-15
            and low_fisher_residual < 4.0e-15
            and full_loads["0.0"] == 0.0
        ),
    }


def _edge_penalty_certificate() -> dict[str, Any]:
    energies = {
        _vertex_label(vertex): _cubic_energy(
            _delta_weights(vertex)
        )
        for vertex in VERTICES
    }
    expected = DELTA_CUBIC_ENERGY
    return {
        "coefficient_partition_frequency": 1,
        "objective_cubic_coefficient": "nu/16",
        "formula": "Q(w)=sum_j mean[H_j D_j^2]",
        "delta_vertex_cubic_energies": {
            label: _fraction_text(value)
            for label, value in energies.items()
        },
        "common_exact_value": _fraction_text(expected),
        "homogeneity": "Q(t z)=t^3 Q(z)",
        "all_checks_pass": all(
            value == expected for value in energies.values()
        ),
    }


def _asymptotic_certificate(
    annular_result: dict[str, Any],
    viscosity: float,
) -> dict[str, Any]:
    continuum = annular_result["continuum_response_certificate"]
    theorem = annular_result["analytic_theorem_certificate"]
    beta = continuum["static_pressure_limit_by_vertex"]
    beta_plus_signed = float(beta["+++"])
    beta_star = abs(beta_plus_signed)
    q = float(DELTA_CUBIC_ENERGY)
    low_amplitude_limit = beta_star / viscosity
    margin_limit = beta_star**2 / (2.0 * viscosity)
    coefficient_scale_limit = (
        64.0 * beta_star / (15.0 * math.sqrt(2.0) * viscosity)
    )
    objective_limit = (
        32.0
        * math.sqrt(2.0)
        * beta_star**3
        / (45.0 * viscosity**2)
    )
    signs = continuum["sign_certificates"]
    fisher_theorem = theorem["vertex_Fisher_scaling_theorem"]
    return {
        "limiting_vertex_load_vector_beta": beta,
        "beta_plus_signed": beta_plus_signed,
        "beta_star": beta_star,
        "beta_plus_integral_formula": (
            "beta_+++=(sqrt(2)/20) integral_D "
            "S^2(V_y^2-V_z^2)<0"
        ),
        "exact_joint_ray_formula": {
            "definitions": (
                "M=sum_v z_v, B_N=b_N.z, "
                "D_N=sum_v z_v E_v(h_N)"
            ),
            "objective": (
                "J_N(a,t;z)=t[a|B_N|-nu(D_N+a^2 M/2)]"
                "-(nu/16)t^3 Q(z)"
            ),
            "optimal_low_amplitude": "a_N=|B_N|/(nu M)",
            "optimized_linear_margin": (
                "A_N=|B_N|^2/(2nu M)-nu D_N"
            ),
            "positive_ray_scale": (
                "t_N=sqrt(16A_N/(3nu Q(z)))"
            ),
            "positive_maximum": (
                "sup J_N=2A_N^(3/2)/(3sqrt(3nu Q(z)/16))"
            ),
            "nonpositive_case": "sup_(a,t>=0) J_N=0 when A_N<=0",
        },
        "fixed_ray_dichotomy": {
            "suppressed_class": (
                "For fixed z>=0 with z_--->0, D_N(z)=Theta(N^3), "
                "while |B_N(z)|^2=O(N^2); hence A_N<0 eventually."
            ),
            "escaping_class": (
                "For fixed z>=0 with z_---=0 and beta.z!=0, "
                "D_N(z)=O(N), while the positive part of A_N is "
                "Theta(N^2); hence A_N>0 eventually."
            ),
            "pressure_null_boundary": (
                "Rays with z_---=0 and beta.z=0 require subleading "
                "analysis and are not classified here."
            ),
            "projective_leading_functional": (
                "|beta.z|^3/[M(z)^(3/2) sqrt(Q(z))]"
            ),
        },
        "delta_plus_asymptotics": {
            "oriented_low_amplitude_over_N_limit": low_amplitude_limit,
            "linear_margin_over_N_squared_limit": margin_limit,
            "coefficient_scale_over_N_limit": coefficient_scale_limit,
            "optimized_objective_over_N_cubed_limit": objective_limit,
            "bounded_scale_one_objective_over_N_squared_limit": (
                margin_limit
            ),
            "coefficient_scale_formula": (
                "t_N/N -> 64 beta_*/(15sqrt(2)nu)"
            ),
            "objective_formula": (
                "max J_N/N^3 -> 32sqrt(2)beta_*^3/(45nu^2)"
            ),
        },
        "all_checks_pass": bool(
            beta_plus_signed < 0.0
            and beta_star > 0.0
            and signs["y_z_xyz_negative"] is True
            and fisher_theorem["formula"]
            == (
                "If r(v) is the number of minus signs, then "
                "E_v(h_N)=Theta(N^(2r(v)-3))."
            )
            and low_amplitude_limit > 0.0
            and margin_limit > 0.0
            and coefficient_scale_limit > 0.0
            and objective_limit > 0.0
        ),
    }


def audit(
    sizes: Sequence[int] = DEFAULT_SIZES,
    viscosity: float = 1.0,
) -> dict[str, Any]:
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive")
    clean_sizes = tuple(int(size) for size in sizes)
    if (
        not clean_sizes
        or any(size < 3 or size % 2 == 0 for size in clean_sizes)
        or tuple(sorted(set(clean_sizes))) != clean_sizes
    ):
        raise ValueError("sizes must be distinct increasing odd integers >=3")

    prerequisite, payloads = _prerequisite_audit()
    graph = _edge_penalty_certificate()
    support = _full_field_support_replay()
    rows = [_finite_row(size, viscosity) for size in clean_sizes]
    annular_path = (
        "work/ns_collision/results/"
        "annular_eight_vertex_heat_window_gate_audit_v1.json"
    )
    asymptotic = _asymptotic_certificate(
        payloads[annular_path], viscosity
    )
    row_by_size = {row["size"]: row for row in rows}
    positive_optimized_sizes = [
        row["size"]
        for row in rows
        if row["ray_optimization"]["positive_escape"]
    ]
    positive_bounded_sizes = [
        row["size"]
        for row in rows
        if row["ray_optimization"]["bounded_scale_one_escape"]
    ]
    required_sizes_present = 25 in row_by_size and 137 in row_by_size
    finite_escape = bool(
        required_sizes_present
        and row_by_size[25]["ray_optimization"]["positive_escape"]
        and row_by_size[137]["ray_optimization"][
            "bounded_scale_one_escape"
        ]
    )
    final_row = rows[-1]
    beta_star = asymptotic["beta_star"]
    final_load_limit_relative_error = abs(
        abs(final_row["load_over_size"]) - beta_star
    ) / beta_star
    all_checks = bool(
        prerequisite["all_checks_pass"]
        and graph["all_checks_pass"]
        and support["all_checks_pass"]
        and asymptotic["all_checks_pass"]
        and all(row["all_checks_pass"] for row in rows)
        and finite_escape
        and final_load_limit_relative_error < 0.04
    )
    return {
        "kind": "compatible_edge_annular_escape_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": (
            "complete_compatible_edge_escape_certified"
            if all_checks
            else "compatible_edge_annular_escape_audit_failed"
        ),
        "scope": (
            "The normalized instantaneous complete-flux bracket at "
            "partition frequency one, for arbitrary nonnegative compatible "
            "coefficients and the explicit annular high field plus one "
            "low plane wave. The harmless overall factor three is omitted."
        ),
        "prerequisite_audit": prerequisite,
        "exact_edge_penalty_certificate": graph,
        "full_field_support_replay": support,
        "asymptotic_ray_certificate": asymptotic,
        "finite_annular_rows": rows,
        "finite_escape_summary": {
            "audited_sizes": list(clean_sizes),
            "positive_optimized_sizes": positive_optimized_sizes,
            "positive_bounded_scale_one_sizes": positive_bounded_sizes,
            "first_audited_positive_optimized_size": (
                positive_optimized_sizes[0]
                if positive_optimized_sizes
                else None
            ),
            "first_audited_positive_bounded_scale_one_size": (
                positive_bounded_sizes[0]
                if positive_bounded_sizes
                else None
            ),
            "largest_size": final_row["size"],
            "largest_size_load_limit_relative_error": (
                final_load_limit_relative_error
            ),
            "finite_escape_checks_pass": finite_escape,
        },
        "theorem": (
            "For the explicit smooth divergence-free fields "
            "u_N=h_N-a_N U and weights lambda_N=t_N Phi_+++, the complete "
            "normalized compatible objective is positive from an audited "
            "finite carrier and its optimized value is asymptotic to a "
            "strictly positive constant times N^3. Even the bounded choice "
            "t=1 is eventually positive, with order-N^2 growth. Therefore "
            "the exact twelve-edge cubic penalty and full weighted velocity "
            "Fisher term do not give a universal nonpositive or uniformly "
            "coercive instantaneous bracket over arbitrary compatible "
            "coefficients and smooth divergence-free fields."
        ),
        "route_decision": (
            "Close the static arbitrary-coefficient twelve-edge coercivity "
            "route. Any repair must add information absent from this "
            "optimization: a dynamic coefficient/state relation, a "
            "low-frequency or amplitude tax, a nonhomogeneous controlled "
            "remainder, or delayed trajectory structure."
        ),
        "certification_flags": {
            "complete_full_field_flux_included": True,
            "full_weighted_velocity_Fisher_included": True,
            "low_field_Fisher_cost_included": True,
            "exact_twelve_edge_cubic_penalty_included": True,
            "exact_joint_low_amplitude_and_ray_scale_optimization_proved": True,
            "fixed_ray_asymptotic_dichotomy_proved": True,
            "delta_plus_optimized_escape_proved": True,
            "bounded_compatible_coefficient_escape_proved": True,
            "static_arbitrary_coefficient_coercivity_proved": False,
            "dynamic_adjoint_coefficient_escape_proved": False,
            "critical_L3_growth_proved": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "next_theorem_target": (
            "Impose the actual backward-adjoint coefficient evolution or "
            "an explicit state-coupled admissibility law on the escaping "
            "annular family. Determine whether the required low amplitude "
            "and coefficient scale force a controlled critical endpoint "
            "tax, or whether the escape persists over a restart interval."
        ),
        "all_positive_checks_pass": all_checks,
    }


def _parse_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SIZES,
        help="comma-separated increasing odd carrier sizes",
    )
    parser.add_argument("--viscosity", type=float, default=1.0)
    arguments = parser.parse_args()
    _lower_process_priority()
    result = audit(arguments.sizes, arguments.viscosity)
    _atomic_json(RESULT, result)
    print(
        json.dumps(
            {
                "result": RESULT.relative_to(ROOT).as_posix(),
                "sha256": _sha256(RESULT),
                "status": result["status"],
                "all_positive_checks_pass": result[
                    "all_positive_checks_pass"
                ],
                "finite_escape_summary": result[
                    "finite_escape_summary"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["all_positive_checks_pass"]:
        raise SystemExit("compatible edge annular escape audit failed")


if __name__ == "__main__":
    main()
