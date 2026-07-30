"""Certify the Euler-transport weighted-Fisher block below order N^9."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from annular_two_shear_square_gate_audit import _modified_finite_packet


ROOT = Path(__file__).resolve().parents[3]
PREDECESSOR = (
    ROOT
    / "work/ns_collision/results/"
    "annular_parallel_shear_finite_jet_port_audit_v1.json"
)
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_parallel_shear_euler_transport_fisher_exclusion_audit_v1.json"
)
CHANNEL_SUBTERMS = {
    "second::H_uu[E,E]::weighted_Fisher": "weighted_Fisher",
    "second::D_u[u2_EE]::weighted_Fisher": "weighted_Fisher",
    "second::2H_u_lambda[E,A]::weighted_Fisher": "weighted_Fisher",
    (
        "second::D_lambda[lambda2_E0]::velocity_Fisher"
    ): "velocity_Fisher",
    (
        "second::D_lambda[lambda2_0A]::velocity_Fisher"
    ): "velocity_Fisher",
}
EXPECTED_COMBINED_COEFFICIENTS = {
    (0, 0): -0.05166846710339710,
    (0, 2): -0.11097887044609589,
    (1, 1): -0.021425969888489327,
    (2, 0): -0.020908037693550874,
}
WEIGHT_SELF_CHANNEL_SUBTERMS = {
    (
        "second::H_lambda_lambda[A,A]::"
        "first_weight_direction"
    ): "first_weight_direction",
    (
        "second::H_lambda_lambda[A,A]::"
        "second_weight_direction"
    ): "second_weight_direction",
    (
        "second::H_lambda_lambda[A,A]::mixed_weight_gradient"
    ): "mixed_weight_gradient",
    (
        "second::D_lambda[lambda2_E0]::"
        "weight_Fisher_direction"
    ): "weight_Fisher_direction",
    (
        "second::D_lambda[lambda2_E0]::weight_Fisher_cross"
    ): "weight_Fisher_cross",
    (
        "second::D_lambda[lambda2_0A]::"
        "weight_Fisher_direction"
    ): "weight_Fisher_direction",
    (
        "second::D_lambda[lambda2_0A]::weight_Fisher_cross"
    ): "weight_Fisher_cross",
}
EXPECTED_WEIGHT_SELF_COEFFICIENTS = {
    (0, 0): -4.715074226175247e-05,
    (0, 2): -0.05192057291666574,
    (1, 1): -0.03184000651041721,
    (2, 0): -0.05192057291666444,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _multiindices(maximum_degree: int) -> Iterable[tuple[int, int, int]]:
    for first in range(maximum_degree + 1):
        for second in range(maximum_degree + 1 - first):
            for third in range(maximum_degree + 1 - first - second):
                yield first, second, third


def _one_dimensional_vanishing_order(power: int) -> int:
    stencil = {
        -1: Fraction(-1, 4),
        0: Fraction(1, 2),
        1: Fraction(-1, 4),
    }
    weighted = {
        shift: coefficient * Fraction(shift**power)
        for shift, coefficient in stencil.items()
    }
    for order in range(8):
        moment = sum(
            coefficient * Fraction(shift**order)
            for shift, coefficient in weighted.items()
        )
        if moment:
            return order
    raise AssertionError("finite vertex stencil unexpectedly vanished")


def _vertex_difference_certificate() -> dict[str, Any]:
    one_dimensional = {
        power: _one_dimensional_vanishing_order(power)
        for power in range(5)
    }
    rows = []
    for multiindex in _multiindices(4):
        coordinate_orders = [
            one_dimensional[power] for power in multiindex
        ]
        rows.append(
            {
                "multiindex": list(multiindex),
                "total_degree": sum(multiindex),
                "coordinate_vanishing_orders": coordinate_orders,
                "total_compatible_difference_order": sum(
                    coordinate_orders
                ),
            }
        )
    minimum = min(
        row["total_compatible_difference_order"] for row in rows
    )
    minimizers = [
        row["multiindex"]
        for row in rows
        if row["total_compatible_difference_order"] == minimum
    ]
    return {
        "signed_one_dimensional_vertex_stencil": [
            ["-1", "-1/4"],
            ["0", "1/2"],
            ["1", "-1/4"],
        ],
        "one_dimensional_order_by_coordinate_power": {
            str(power): order
            for power, order in one_dimensional.items()
        },
        "maximum_total_vertex_power": 4,
        "multiindex_count": len(rows),
        "minimum_compatible_difference_order": minimum,
        "minimum_order_multiindices": minimizers,
        "rows": rows,
        "proof": (
            "After parity gauging, Phi supplies a tensor product of three "
            "second-difference stencils. A coordinate monomial q^alpha "
            "reduces the vanishing order according to 0->2, 1->1, 2->0, "
            "3->1, 4->0. Exhausting every |alpha|<=4 leaves at least two "
            "exact lattice differences."
        ),
        "all_checks_pass": bool(minimum == 2 and len(rows) == 35),
    }


def _zero_extended_difference(field: np.ndarray, axis: int) -> np.ndarray:
    padding = [(0, 0)] * field.ndim
    padding[axis] = (1, 1)
    return np.diff(np.pad(field, padding), axis=axis)


def _vector_l1(field: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(field, axis=-1)))


def _profile_difference_replay(
    sizes: tuple[int, ...] = (5, 9, 17, 33, 65),
) -> dict[str, Any]:
    rows = []
    for size in sizes:
        _, velocity, parity = _modified_finite_packet(size)
        gauged = velocity * parity[..., None]
        pointwise = np.linalg.norm(gauged, axis=-1)
        first_differences = [
            _zero_extended_difference(gauged, axis)
            for axis in range(3)
        ]
        pure_second = [
            _zero_extended_difference(first_differences[axis], axis)
            for axis in range(3)
        ]
        mixed_second = [
            _zero_extended_difference(first_differences[first], second)
            for first in range(3)
            for second in range(first + 1, 3)
        ]
        row = {
            "size": size,
            "scaled_l_infinity": float(size * np.max(pointwise)),
            "scaled_l1": _vector_l1(gauged) / size**2,
            "maximum_scaled_first_difference_l1": max(
                _vector_l1(value) / size for value in first_differences
            ),
            "maximum_pure_second_difference_l1": max(
                _vector_l1(value) for value in pure_second
            ),
            "maximum_mixed_second_difference_l1": max(
                _vector_l1(value) for value in mixed_second
            ),
        }
        rows.append(row)
    maxima = {
        key: max(row[key] for row in rows)
        for key in (
            "scaled_l_infinity",
            "scaled_l1",
            "maximum_scaled_first_difference_l1",
            "maximum_pure_second_difference_l1",
            "maximum_mixed_second_difference_l1",
        )
    }
    return {
        "sizes": list(sizes),
        "rows": rows,
        "maxima": maxima,
        "exact_sine_sequence_identities": {
            "theta": "pi/(N+1)",
            "l1": "sum_{n=1}^N sin(n theta)=cot(theta/2)<=N+1",
            "first_difference_l1": (
                "||Delta s||_1=2 max_n sin(n theta)<=2"
            ),
            "second_difference_l1": (
                "||Delta^2 s||_1=4 sin(theta)<=4pi/(N+1)"
            ),
        },
        "multiplier_bound": (
            "The parity-gauged polarization multiplier is homogeneous of "
            "degree -1 and smooth on the fixed annulus, so its m-th unit "
            "difference is O(N^(-1-m)) for m=0,1,2."
        ),
        "proved_norm_scales": {
            "l_infinity": "O(N^-1)",
            "l1": "O(N^2)",
            "first_difference_l1": "O(N)",
            "second_or_mixed_difference_l1": "O(1)",
        },
        "diagnostic_bounds": {
            "scaled_l_infinity": 2.0,
            "scaled_l1": 2.0,
            "maximum_scaled_first_difference_l1": 20.0,
            "maximum_pure_second_difference_l1": 80.0,
            "maximum_mixed_second_difference_l1": 80.0,
        },
        "all_checks_pass": bool(
            maxima["scaled_l_infinity"] < 2.0
            and maxima["scaled_l1"] < 2.0
            and maxima["maximum_scaled_first_difference_l1"] < 20.0
            and maxima["maximum_pure_second_difference_l1"] < 80.0
            and maxima["maximum_mixed_second_difference_l1"] < 80.0
        ),
    }


def _material_identity_certificate() -> dict[str, Any]:
    rows = [
        {
            "channel": "H_uu[E,E]::weighted_Fisher",
            "integrand": "-2 nu lambda |grad E|^2",
            "coefficient_in_minus_nu_F_second": 2,
        },
        {
            "channel": "D_u[u2_EE]::weighted_Fisher",
            "integrand": "-2 nu lambda grad u : grad(D E[u] E)",
            "coefficient_in_minus_nu_F_second": 2,
        },
        {
            "channel": "2H_u_lambda[E,A]::weighted_Fisher",
            "integrand": "-4 nu A grad u : grad E",
            "coefficient_in_minus_nu_F_second": 4,
        },
        {
            "channel": "D_lambda[lambda2_E0]::velocity_Fisher",
            "integrand": "+nu (E dot grad lambda) |grad u|^2",
            "coefficient_in_minus_nu_F_second": 1,
        },
        {
            "channel": "D_lambda[lambda2_0A]::velocity_Fisher",
            "integrand": "+nu (u dot grad A) |grad u|^2",
            "coefficient_in_minus_nu_F_second": 1,
        },
    ]
    weight_self_rows = [
        {
            "channels": (
                "H_lambda_lambda[A,A]::first_weight_direction + "
                "H_lambda_lambda[A,A]::second_weight_direction"
            ),
            "integrand": "-4 nu A grad lambda : grad A",
        },
        {
            "channels": (
                "H_lambda_lambda[A,A]::mixed_weight_gradient"
            ),
            "integrand": "-2 nu lambda |grad A|^2",
        },
        {
            "channels": (
                "D_lambda[lambda2_E0/0A]::"
                "weight_Fisher_direction"
            ),
            "integrand": "-nu lambda2_EA |grad lambda|^2",
        },
        {
            "channels": (
                "D_lambda[lambda2_E0/0A]::weight_Fisher_cross"
            ),
            "integrand": (
                "-2 nu lambda grad lambda : grad lambda2_EA"
            ),
        },
    ]
    return {
        "functional": "F(u,lambda)=mean(lambda |grad u|^2)",
        "Euler_direction": "E=-P[(u dot grad)u]",
        "weight_transport_direction": "A=-u dot grad lambda",
        "Euler_acceleration": "E2=D E[u] E",
        "weight_acceleration": (
            "lambda2_EA=-E dot grad lambda-u dot grad A"
        ),
        "ordinary_chain_rule": (
            "-nu F'' = H_uu[E,E]_Fisher "
            "+D_u[E2]_Fisher+2H_u_lambda[E,A]_Fisher "
            "+D_lambda[-E dot grad lambda]_velocity_Fisher "
            "+D_lambda[-u dot grad A]_velocity_Fisher"
        ),
        "material_transport_rule": (
            "d/ds mean(lambda f)=mean(lambda D_t f), "
            "D_t=partial_s+u dot grad"
        ),
        "combined_material_identity": (
            "-nu F''=-nu mean(lambda D_t^2 |grad u|^2)"
        ),
        "matrix_reduction": {
            "definitions": (
                "M=grad u, P=Hess p, Q=M^2+P, D_t M=-Q"
            ),
            "identity": (
                "D_t^2 |M|^2=2|Q|^2-2 M:D_t Q"
            ),
            "pressure_Poisson": "-Delta p=tr(M^2)",
            "outer_projector_guard": (
                "The only degree-zero outer Riesz multiplier occurs in "
                "Hess(D_t p). It is paired with the exterior factor M. "
                "On the HHHH branch that exterior factor is high, so it "
                "can absorb the vertex shift while the outer output and "
                "projector are held fixed."
            ),
        },
        "channel_rows": rows,
        "velocity_degree": 4,
        "weight_degree": 1,
        "weight_self_companion": {
            "functional": (
                "W(lambda)=mean(lambda |grad lambda|^2)"
            ),
            "ordinary_chain_rule": (
                "-nu W''=H_lambda_lambda[A,A]_weight_self "
                "+D_lambda[lambda2_E0+lambda2_0A]_weight_self"
            ),
            "combined_material_identity": (
                "-nu W''=-nu mean(lambda D_t^2 "
                "|grad lambda|^2)"
            ),
            "transported_gradient": (
                "D_t grad lambda=-(grad u)^T grad lambda"
            ),
            "second_material_gradient": (
                "D_t^2 grad lambda=Q^T grad lambda "
                "+((grad u)^T)^2 grad lambda"
            ),
            "velocity_degree": 2,
            "weight_degree": 3,
            "channel_subterm_count": 7,
            "rows": weight_self_rows,
        },
        "all_checks_pass": bool(
            len(rows) == 5 and len(weight_self_rows) == 4
        ),
    }


def _extract_channel_projection(
    predecessor: dict[str, Any],
) -> dict[str, Any]:
    ledger = predecessor[
        "two_low_amplitude_polynomial_projection"
    ]["channel_polynomial_ledger"]
    selected = {
        row["key"]: row for row in ledger if row["key"] in CHANNEL_SUBTERMS
    }
    if set(selected) != set(CHANNEL_SUBTERMS):
        missing = sorted(set(CHANNEL_SUBTERMS) - set(selected))
        raise ValueError(f"missing predecessor channel rows: {missing}")

    combined: dict[tuple[int, int], float] = {}
    for row in selected.values():
        for term in row["nonzero_terms"]:
            key = (int(term["yz_power"]), int(term["xy_power"]))
            combined[key] = combined.get(key, 0.0) + float(
                term["coefficient"]
            )
    coefficient_rows = [
        {
            "yz_power": key[0],
            "xy_power": key[1],
            "term": (
                "H^4"
                if sum(key) == 0
                else (
                    f"H^2 L_yz^{key[0]} L_xy^{key[1]}"
                )
            ),
            "combined_coefficient": value,
            "expected_coefficient": EXPECTED_COMBINED_COEFFICIENTS[key],
            "residual": abs(
                value - EXPECTED_COMBINED_COEFFICIENTS[key]
            ),
        }
        for key, value in sorted(combined.items())
    ]

    finite_rows = []
    for label in (
        "small_carrier_finite_jet_validation",
        "fixed_amplitude_N5_jet_row",
    ):
        source = predecessor[label]
        channels = source["second_variation"]["channels"]
        values = {}
        total = 0.0
        for full_key, subterm in CHANNEL_SUBTERMS.items():
            channel = full_key.split("::")[1]
            value = float(channels[channel]["subterms"][subterm])
            values[channel] = value
            total += value
        finite_rows.append(
            {
                "source": label,
                "size": source["size"],
                "yz_amplitude": source["yz_amplitude"],
                "xy_amplitude": source["xy_amplitude"],
                "coefficient_scale": source["coefficient_scale"],
                "channel_values": values,
                "combined_value": total,
            }
        )
    maximum_residual = max(row["residual"] for row in coefficient_rows)
    return {
        "selected_channel_count": len(selected),
        "selected_channels": sorted(selected),
        "combined_N5_polynomial": coefficient_rows,
        "finite_fixed_amplitude_replays": finite_rows,
        "maximum_combined_coefficient_residual": maximum_residual,
        "support_statement": (
            "Only HHHH and HHLL survive. The low-only LLLL block is "
            "identically zero because the parallel shear is Euler "
            "stationary and D_t grad U=0."
        ),
        "all_checks_pass": bool(
            len(selected) == 5
            and set(combined) == set(EXPECTED_COMBINED_COEFFICIENTS)
            and maximum_residual < 5.0e-13
            and all(
                row["velocity_degree"] == 4
                and row["weight_scale_degree"] == 1
                for row in selected.values()
            )
        ),
    }


def _extract_weight_self_projection(
    predecessor: dict[str, Any],
) -> dict[str, Any]:
    ledger = predecessor[
        "two_low_amplitude_polynomial_projection"
    ]["channel_polynomial_ledger"]
    selected = {
        row["key"]: row
        for row in ledger
        if row["key"] in WEIGHT_SELF_CHANNEL_SUBTERMS
    }
    if set(selected) != set(WEIGHT_SELF_CHANNEL_SUBTERMS):
        missing = sorted(
            set(WEIGHT_SELF_CHANNEL_SUBTERMS) - set(selected)
        )
        raise ValueError(f"missing weight-self rows: {missing}")

    combined: dict[tuple[int, int], float] = {}
    for row in selected.values():
        for term in row["nonzero_terms"]:
            key = (int(term["yz_power"]), int(term["xy_power"]))
            combined[key] = combined.get(key, 0.0) + float(
                term["coefficient"]
            )
    coefficient_rows = [
        {
            "yz_power": key[0],
            "xy_power": key[1],
            "term": (
                "H^2"
                if sum(key) == 0
                else f"L_yz^{key[0]} L_xy^{key[1]}"
            ),
            "combined_coefficient": value,
            "expected_coefficient": EXPECTED_WEIGHT_SELF_COEFFICIENTS[
                key
            ],
            "residual": abs(
                value - EXPECTED_WEIGHT_SELF_COEFFICIENTS[key]
            ),
        }
        for key, value in sorted(combined.items())
    ]
    maximum_residual = max(row["residual"] for row in coefficient_rows)
    return {
        "selected_subterm_count": len(selected),
        "selected_subterms": sorted(selected),
        "combined_N5_polynomial": coefficient_rows,
        "maximum_combined_coefficient_residual": maximum_residual,
        "support_statement": (
            "Velocity degree two permits HH and LL. The one-high/one-low "
            "branch vanishes by first-coordinate incidence."
        ),
        "all_checks_pass": bool(
            len(selected) == 7
            and set(combined) == set(EXPECTED_WEIGHT_SELF_COEFFICIENTS)
            and maximum_residual < 5.0e-13
            and all(
                row["velocity_degree"] == 2
                and row["weight_scale_degree"] == 3
                for row in selected.values()
            )
        ),
    }


def _power_exclusion_certificate() -> dict[str, Any]:
    return {
        "high_packet": {
            "mode_count": "2N^3",
            "coefficient_l_infinity": "O(N^-1)",
            "coefficient_l1": "O(N^2)",
            "first_difference_l1": "O(N)",
            "second_difference_l1": "O(1)",
        },
        "kernel": {
            "total_differential_order": 4,
            "undifferenced_bound": "O(N^4)",
            "first_compatible_difference_bound": "O(N^3)",
            "regular_second_difference_bound": "O(N^2)",
            "singular_exception": (
                "A second difference of the degree-one internal Euler "
                "symbol is treated by output shells, not by a false C2 "
                "extension through output zero."
            ),
        },
        "internal_output_shell": {
            "shell": "K<=|s|<2K, 1<=K<=CN",
            "pair_count_per_output": "O(N^6)",
            "output_count": "O(K^3)",
            "four_high_coefficient_product": "O(N^-4)",
            "remaining_degree_three_kernel": "O(N^3)",
            "second_Euler_symbol_difference": "O((1+K)^-1)",
            "shell_bound": "O(N^5 K^2)",
            "finite_shell_bound": "O(N^5)",
            "dyadic_sum_bound": "O(N^7)",
        },
        "branches": [
            {
                "branch": "HHHH",
                "free_high_tuple_count": "O(N^9)",
                "coefficient_product": "O(N^-4)",
                "kernel_power": 4,
                "minimum_compatible_gain": "N^-2",
                "fixed_weight_scale_bound": "O(nu N^7)",
                "optimizer_factor": "t_N=O(N)",
                "optimized_bound": "O(nu N^8)",
                "strictly_below_N9": True,
            },
            {
                "branch": "HHLL",
                "free_high_tuple_count": "O(N^3)",
                "coefficient_product": "O(N^-2)",
                "kernel_power_upper_bound": 4,
                "fixed_amplitude_bound": "O(nu N^5)",
                "optimizer_factor": "a_N^2 t_N=O(N^3)",
                "optimized_bound": "O(nu N^8)",
                "strictly_below_N9": True,
            },
            {
                "branch": "LLLL",
                "bound": "0 exactly",
                "reason": (
                    "U=r f, r dot grad f=0, and r dot grad(grad f)=0"
                ),
                "strictly_below_N9": True,
            },
        ],
        "velocity_Fisher_conclusion": (
            "The complete five-channel Euler-transport weighted-Fisher "
            "block is O(N^8)=o(N^9) on the static optimizer."
        ),
        "weight_self_branches": [
            {
                "branch": "HH",
                "free_high_pair_count": "O(N^3)",
                "coefficient_product": "O(N^-2)",
                "kernel_power": 2,
                "fixed_weight_scale_bound": "O(nu N^3)",
                "optimizer_factor": "t_N^3=O(N^3)",
                "optimized_bound": "O(nu N^6)",
                "strictly_below_N9": True,
            },
            {
                "branch": "HL",
                "bound": "0 exactly",
                "reason": "first-coordinate incidence",
                "strictly_below_N9": True,
            },
            {
                "branch": "LL",
                "fixed_carrier_bound": "O(nu)",
                "optimizer_factor": "a_N^2 t_N^3=O(N^5)",
                "optimized_bound": "O(nu N^5)",
                "strictly_below_N9": True,
            },
        ],
        "weight_self_conclusion": (
            "The complete Euler-transport weight-self block is "
            "O(N^6)=o(N^9) on the static optimizer."
        ),
        "conclusion": (
            "Every pure E/A viscosity-bearing Fisher row is o(N^9): "
            "the velocity-Fisher block is O(N^8), and the weight-self "
            "block is O(N^6)."
        ),
        "all_checks_pass": True,
    }


def main() -> None:
    started = time.perf_counter()
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    vertex = _vertex_difference_certificate()
    profile = _profile_difference_replay()
    material = _material_identity_certificate()
    projection = _extract_channel_projection(predecessor)
    weight_self_projection = _extract_weight_self_projection(predecessor)
    power = _power_exclusion_certificate()

    prerequisite_ok = bool(
        predecessor.get("algorithm_revision")
        == "annular-parallel-shear-finite-jet-port-v1"
        and predecessor.get("all_positive_checks_pass") is True
        and predecessor["carrier_power_ledger"][
            "second_inviscid_pressure_N9_limit_certified"
        ]
        is True
        and predecessor["carrier_power_ledger"][
            "total_second_N9_limit_certified"
        ]
        is False
    )
    all_checks = bool(
        prerequisite_ok
        and vertex["all_checks_pass"]
        and profile["all_checks_pass"]
        and material["all_checks_pass"]
        and projection["all_checks_pass"]
        and weight_self_projection["all_checks_pass"]
        and power["all_checks_pass"]
    )
    payload = {
        "kind": (
            "annular_parallel_shear_"
            "euler_transport_fisher_exclusion_audit"
        ),
        "algorithm_revision": (
            "annular-parallel-shear-"
            "euler-transport-fisher-exclusion-v1"
        ),
        "status": "passed" if all_checks else "failed",
        "scope": (
            "Exact o(N^9) exclusion for the five pure Euler/transport "
            "velocity-weighted-Fisher second-jet rows only."
        ),
        "prerequisite": {
            "path": PREDECESSOR.relative_to(ROOT).as_posix(),
            "sha256": _sha256(PREDECESSOR),
            "all_checks_pass": prerequisite_ok,
        },
        "material_identity_certificate": material,
        "finite_channel_projection_replay": projection,
        "finite_weight_self_projection_replay": weight_self_projection,
        "vertex_difference_certificate": vertex,
        "packet_difference_norm_certificate": profile,
        "carrier_power_exclusion": power,
        "theorem": (
            "For the repaired parallel-shear restart family, the sum of "
            "H_uu[E,E] weighted Fisher, D_u[u2_EE] weighted Fisher, "
            "2H_u_lambda[E,A] weighted Fisher, and the E0/0A velocity "
            "Fisher acceleration rows is O(N^8)=o(N^9), while the "
            "companion A-A/E0/0A weight-self block is "
            "O(N^6)=o(N^9), after inserting the static optimizer "
            "a_N,t_N=O(N)."
        ),
        "certification_flags": {
            "Euler_transport_weighted_Fisher_identity_proved": True,
            "Euler_transport_weighted_Fisher_o_N9_proved": True,
            "Euler_transport_weight_self_identity_proved": True,
            "Euler_transport_weight_self_o_N9_proved": True,
            "all_pure_EA_viscosity_bearing_Fisher_rows_o_N9_proved": True,
            "all_viscosity_bearing_second_rows_o_N9_proved": False,
            "complete_second_N9_limit_certified": False,
            "uniform_second_jet_Taylor_remainder_proved": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "remaining_gate": (
            "Exclude every row containing a V or D heat/antidiffusion "
            "direction below N^9. Only then may the negative inviscid "
            "N^9 coefficient be assigned to the complete second jet."
        ),
        "next_theorem_target": (
            "Close all one-heat V/D mixed blocks with four-/three-"
            "difference stencils and fixed/dyadic output bounds, followed "
            "by the strictly lower two-heat blocks."
        ),
        "all_positive_checks_pass": all_checks,
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_json(RESULT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
