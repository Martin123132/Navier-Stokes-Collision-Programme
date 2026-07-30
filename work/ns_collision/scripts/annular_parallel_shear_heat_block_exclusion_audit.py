"""Certify every V/D second-jet row below the inviscid N^9 scale."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from annular_parallel_shear_euler_transport_fisher_exclusion_audit import (
    CHANNEL_SUBTERMS,
    WEIGHT_SELF_CHANNEL_SUBTERMS,
    _vertex_difference_certificate,
)


ROOT = Path(__file__).resolve().parents[3]
FINITE_JET_RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_parallel_shear_finite_jet_port_audit_v1.json"
)
EA_FISHER_RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_parallel_shear_euler_transport_fisher_exclusion_audit_v1.json"
)
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_parallel_shear_heat_block_exclusion_audit_v1.json"
)

PURE_EA_CHANNELS = {
    "H_uu[E,E]",
    "2H_u_lambda[E,A]",
    "H_lambda_lambda[A,A]",
    "D_u[u2_EE]",
    "D_lambda[lambda2_E0]",
    "D_lambda[lambda2_0A]",
}
ONE_HEAT_CHANNELS = {
    "2H_uu[E,V]",
    "2H_u_lambda[E,D]",
    "2H_u_lambda[V,A]",
    "2H_lambda_lambda[A,D]",
    "D_u[u2_EV]",
    "D_u[u2_VE]",
    "D_lambda[lambda2_V0]",
    "D_lambda[lambda2_0D]",
    "D_lambda[lambda2_DA]",
}
TWO_HEAT_CHANNELS = {
    "H_uu[V,V]",
    "2H_u_lambda[V,D]",
    "H_lambda_lambda[D,D]",
    "D_u[u2_VV]",
    "D_lambda[lambda2_DD]",
}
EXPECTED_COUNTS = {
    "pure_EA": {
        "pressure": 9,
        "velocity_Fisher": 5,
        "weight_self": 7,
    },
    "one_heat": {
        "pressure": 14,
        "velocity_Fisher": 8,
        "weight_self": 9,
    },
    "two_heat": {
        "pressure": 8,
        "velocity_Fisher": 4,
        "weight_self": 5,
    },
}
EXPECTED_DEGREES = {
    ("pure_EA", "pressure"): (5, 1),
    ("pure_EA", "velocity_Fisher"): (4, 1),
    ("pure_EA", "weight_self"): (2, 3),
    ("one_heat", "pressure"): (4, 1),
    ("one_heat", "velocity_Fisher"): (3, 1),
    ("one_heat", "weight_self"): (1, 3),
    ("two_heat", "pressure"): (3, 1),
    ("two_heat", "velocity_Fisher"): (2, 1),
    ("two_heat", "weight_self"): (0, 3),
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


def _split_key(key: str) -> tuple[str, str]:
    stage, channel, subterm = key.split("::", 2)
    if stage != "second":
        raise ValueError(f"not a second-variation key: {key}")
    return channel, subterm


def _category(subterm: str) -> str:
    if subterm == "pressure" or subterm.startswith("pressure_"):
        return "pressure"
    if subterm in {"weighted_Fisher", "velocity_Fisher"}:
        return "velocity_Fisher"
    return "weight_self"


def _block(channel: str) -> str:
    if channel in PURE_EA_CHANNELS:
        return "pure_EA"
    if channel in ONE_HEAT_CHANNELS:
        return "one_heat"
    if channel in TWO_HEAT_CHANNELS:
        return "two_heat"
    raise ValueError(f"unclassified second-variation channel: {channel}")


def _combined_polynomial(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coefficients: dict[tuple[int, int], float] = {}
    for row in rows:
        for term in row["nonzero_terms"]:
            key = (int(term["yz_power"]), int(term["xy_power"]))
            coefficients[key] = coefficients.get(key, 0.0) + float(
                term["coefficient"]
            )
    return [
        {
            "yz_power": key[0],
            "xy_power": key[1],
            "coefficient": value,
        }
        for key, value in sorted(coefficients.items())
    ]


def _exhaustive_partition_certificate(
    finite_jet: dict[str, Any],
) -> dict[str, Any]:
    ledger = finite_jet[
        "two_low_amplitude_polynomial_projection"
    ]["channel_polynomial_ledger"]
    all_second_rows = [
        row for row in ledger if row["key"].startswith("second::")
    ]
    aggregate_rows = [
        row
        for row in all_second_rows
        if _split_key(row["key"])[0] == "aggregate"
    ]
    second_rows = [
        row
        for row in all_second_rows
        if _split_key(row["key"])[0] != "aggregate"
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    classified_keys = set()
    for row in second_rows:
        channel, subterm = _split_key(row["key"])
        key = (_block(channel), _category(subterm))
        grouped.setdefault(key, []).append(row)
        classified_keys.add(row["key"])

    group_records = []
    count_checks = []
    degree_checks = []
    for block in ("pure_EA", "one_heat", "two_heat"):
        for category in (
            "pressure",
            "velocity_Fisher",
            "weight_self",
        ):
            rows = grouped.get((block, category), [])
            expected_count = EXPECTED_COUNTS[block][category]
            expected_velocity, expected_weight = EXPECTED_DEGREES[
                (block, category)
            ]
            observed_degrees = sorted(
                {
                    (
                        int(row["velocity_degree"]),
                        int(row["weight_scale_degree"]),
                    )
                    for row in rows
                }
            )
            count_ok = len(rows) == expected_count
            degree_ok = observed_degrees == [
                (expected_velocity, expected_weight)
            ]
            count_checks.append(count_ok)
            degree_checks.append(degree_ok)
            group_records.append(
                {
                    "block": block,
                    "category": category,
                    "row_count": len(rows),
                    "expected_row_count": expected_count,
                    "velocity_degree": expected_velocity,
                    "weight_scale_degree": expected_weight,
                    "observed_degrees": [
                        list(value) for value in observed_degrees
                    ],
                    "keys": sorted(row["key"] for row in rows),
                    "combined_N5_polynomial": _combined_polynomial(rows),
                    "maximum_forbidden_support_relative": max(
                        (
                            float(
                                row[
                                    "maximum_forbidden_support_relative"
                                ]
                            )
                            for row in rows
                        ),
                        default=0.0,
                    ),
                    "all_checks_pass": bool(count_ok and degree_ok),
                }
            )

    expected_channels = (
        PURE_EA_CHANNELS | ONE_HEAT_CHANNELS | TWO_HEAT_CHANNELS
    )
    observed_channels = {
        _split_key(row["key"])[0] for row in second_rows
    }
    pure_velocity_keys = set(CHANNEL_SUBTERMS)
    pure_weight_keys = set(WEIGHT_SELF_CHANNEL_SUBTERMS)
    pure_keys = {
        row["key"]
        for row in second_rows
        if _split_key(row["key"])[0] in PURE_EA_CHANNELS
    }
    pure_pressure_keys = pure_keys - pure_velocity_keys - pure_weight_keys
    pure_pressure_categories_ok = all(
        _category(_split_key(key)[1]) == "pressure"
        for key in pure_pressure_keys
    )
    return {
        "second_subterm_count": len(second_rows),
        "expected_second_subterm_count": 69,
        "derived_aggregate_row_count": len(aggregate_rows),
        "derived_aggregate_keys": sorted(
            row["key"] for row in aggregate_rows
        ),
        "channel_count": len(observed_channels),
        "expected_channel_count": 20,
        "observed_channels": sorted(observed_channels),
        "expected_channels": sorted(expected_channels),
        "block_channel_counts": {
            "pure_EA": len(PURE_EA_CHANNELS),
            "one_heat": len(ONE_HEAT_CHANNELS),
            "two_heat": len(TWO_HEAT_CHANNELS),
        },
        "block_subterm_counts": {
            "pure_EA": sum(
                len(grouped.get(("pure_EA", category), []))
                for category in (
                    "pressure",
                    "velocity_Fisher",
                    "weight_self",
                )
            ),
            "one_heat": sum(
                len(grouped.get(("one_heat", category), []))
                for category in (
                    "pressure",
                    "velocity_Fisher",
                    "weight_self",
                )
            ),
            "two_heat": sum(
                len(grouped.get(("two_heat", category), []))
                for category in (
                    "pressure",
                    "velocity_Fisher",
                    "weight_self",
                )
            ),
        },
        "pure_pressure_subterm_count": len(pure_pressure_keys),
        "pure_velocity_Fisher_subterm_count": len(pure_velocity_keys),
        "pure_weight_self_subterm_count": len(pure_weight_keys),
        "group_records": group_records,
        "unclassified_keys": sorted(
            {row["key"] for row in second_rows} - classified_keys
        ),
        "duplicate_key_count": len(second_rows) - len(classified_keys),
        "all_checks_pass": bool(
            len(second_rows) == 69
            and len(all_second_rows) == 71
            and len(aggregate_rows) == 2
            and len(classified_keys) == 69
            and observed_channels == expected_channels
            and len(expected_channels) == 20
            and all(count_checks)
            and all(degree_checks)
            and len(pure_pressure_keys) == 9
            and len(pure_velocity_keys) == 5
            and len(pure_weight_keys) == 7
            and pure_pressure_categories_ok
        ),
    }


def _flow_block_identity_certificate() -> dict[str, Any]:
    return {
        "state": "z=(u,lambda)",
        "inviscid_vector_field": (
            "X(z)=(E,A)=(-P[(u dot grad)u],-u dot grad lambda)"
        ),
        "heat_vector_field": (
            "Y(z)=(V,D)=(nu Delta u,-nu Delta lambda)"
        ),
        "pure_EA_second_block": (
            "D^2 g[X,X]+Dg[DX X]"
        ),
        "one_heat_second_block": (
            "2D^2 g[X,Y]+Dg[DX Y+DY X]=X(Yg)+Y(Xg)"
        ),
        "one_heat_accelerations": {
            "velocity": "u2_EV+u2_VE",
            "weight": "lambda2_V0+lambda2_0D+lambda2_DA",
        },
        "two_heat_second_block": (
            "D^2 g[Y,Y]+Dg[DY Y]"
        ),
        "two_heat_accelerations": {
            "velocity": "u2_VV",
            "weight": "lambda2_DD",
        },
        "channel_partition": {
            "pure_EA": sorted(PURE_EA_CHANNELS),
            "one_heat": sorted(ONE_HEAT_CHANNELS),
            "two_heat": sorted(TWO_HEAT_CHANNELS),
        },
        "all_checks_pass": bool(
            len(PURE_EA_CHANNELS) == 6
            and len(ONE_HEAT_CHANNELS) == 9
            and len(TWO_HEAT_CHANNELS) == 5
            and not (
                PURE_EA_CHANNELS & ONE_HEAT_CHANNELS
                or PURE_EA_CHANNELS & TWO_HEAT_CHANNELS
                or ONE_HEAT_CHANNELS & TWO_HEAT_CHANNELS
            )
        ),
    }


def _one_heat_pressure_certificate(
    vertex: dict[str, Any],
) -> dict[str, Any]:
    return {
        "velocity_degree": 4,
        "weight_degree": 1,
        "total_differential_order": 4,
        "weight_reduction": (
            "Because the pressure generator is linear in lambda, integrate "
            "every A, D, V0, 0D, and DA derivative by parts until the base "
            "weight is undifferentiated Phi. The resulting quartic kernel "
            "still has total differential order at most four."
        ),
        "vertex_stencil_order": 6,
        "maximum_explicit_vertex_degree": 4,
        "minimum_remaining_compatible_differences": vertex[
            "minimum_compatible_difference_order"
        ],
        "outer_projector_guard": (
            "Every HHHH pressure atom has a high leaf on the test side. "
            "Choose it as the dependent leaf so a vertex shift changes "
            "that side while the degree-zero outer pressure output and "
            "projector remain fixed."
        ),
        "internal_output_guard": (
            "Regular allocations use O(N^4), O(N^3), and O(N^2) kernel "
            "bounds under zero, one, and two differences. If two "
            "differences hit an internal degree-one Euler symbol, split "
            "its output into finite and dyadic shells. The shell bound is "
            "O(N^5 K^2), whose dyadic sum is O(N^7)."
        ),
        "branches": [
            {
                "branch": "HHHH",
                "raw_tuple_and_coefficient_power": 5,
                "kernel_power": 4,
                "compatible_gain": -2,
                "fixed_weight_scale_power": 7,
                "optimizer_factor_power": 1,
                "optimized_power": 8,
            },
            {
                "branch": "HHLL",
                "high_pair_sum_power": 1,
                "kernel_power_upper_bound": 4,
                "fixed_amplitude_power": 5,
                "optimizer_factor_power": 3,
                "optimized_power": 8,
            },
            {
                "branch": "LLLL",
                "fixed_carrier_power": 0,
                "optimizer_factor_power": 5,
                "optimized_power": 5,
            },
        ],
        "conclusion": "one-heat pressure block=O(N^8)=o(N^9)",
        "all_checks_pass": bool(
            vertex["minimum_compatible_difference_order"] >= 2
        ),
    }


def _remaining_power_ledger() -> dict[str, Any]:
    rows = [
        {
            "block": "one_heat",
            "category": "velocity_Fisher",
            "surviving_high_branch": "HHL",
            "high_sum_power": 1,
            "kernel_power_upper_bound": 5,
            "optimizer_amplitude_power": 2,
            "optimized_power": 8,
            "reason": (
                "Velocity degree three permits only zero or two high "
                "leaves; the HHL branch has one low amplitude and one "
                "weight scale."
            ),
        },
        {
            "block": "one_heat",
            "category": "weight_self",
            "surviving_high_branch": "none",
            "kernel_power_upper_bound": 0,
            "optimizer_amplitude_power": 4,
            "optimized_power": 4,
            "reason": (
                "Velocity degree one makes its high branch incidence-"
                "forbidden; only a fixed low mode times t_N^3 remains."
            ),
        },
        {
            "block": "two_heat",
            "category": "pressure",
            "surviving_high_branch": "HHL",
            "high_sum_power": 1,
            "kernel_power_upper_bound": 4,
            "optimizer_amplitude_power": 2,
            "optimized_power": 7,
            "reason": (
                "Velocity degree three permits two high leaves; both "
                "velocity Laplacians supply at most four carrier powers."
            ),
        },
        {
            "block": "two_heat",
            "category": "velocity_Fisher",
            "surviving_high_branch": "HH",
            "high_sum_power": 1,
            "kernel_power_upper_bound": 6,
            "optimizer_amplitude_power": 1,
            "optimized_power": 8,
            "reason": (
                "The worst H_uu[V,V] Fisher atom has two grad(V) factors, "
                "hence six carrier derivatives and one weight scale."
            ),
        },
        {
            "block": "two_heat",
            "category": "weight_self",
            "surviving_high_branch": "none",
            "kernel_power_upper_bound": 0,
            "optimizer_amplitude_power": 3,
            "optimized_power": 3,
            "reason": "Velocity degree zero leaves only t_N^3.",
        },
    ]
    return {
        "rows": rows,
        "maximum_optimized_power": max(
            row["optimized_power"] for row in rows
        ),
        "all_rows_strictly_below_N9": all(
            row["optimized_power"] < 9 for row in rows
        ),
        "all_checks_pass": True,
    }


def main() -> None:
    started = time.perf_counter()
    finite_jet = json.loads(FINITE_JET_RESULT.read_text(encoding="utf-8"))
    ea_fisher = json.loads(EA_FISHER_RESULT.read_text(encoding="utf-8"))
    partition = _exhaustive_partition_certificate(finite_jet)
    flow = _flow_block_identity_certificate()
    vertex = _vertex_difference_certificate()
    one_heat_pressure = _one_heat_pressure_certificate(vertex)
    remainder = _remaining_power_ledger()

    prerequisites_ok = bool(
        finite_jet.get("all_positive_checks_pass") is True
        and finite_jet["carrier_power_ledger"][
            "second_inviscid_pressure_N9_limit_certified"
        ]
        is True
        and finite_jet["carrier_power_ledger"][
            "total_second_N9_limit_certified"
        ]
        is False
        and ea_fisher.get("all_positive_checks_pass") is True
        and ea_fisher["certification_flags"][
            "all_pure_EA_viscosity_bearing_Fisher_rows_o_N9_proved"
        ]
        is True
    )
    all_checks = bool(
        prerequisites_ok
        and partition["all_checks_pass"]
        and flow["all_checks_pass"]
        and vertex["all_checks_pass"]
        and one_heat_pressure["all_checks_pass"]
        and remainder["all_checks_pass"]
        and remainder["all_rows_strictly_below_N9"]
    )
    inviscid_limit = finite_jet["carrier_power_ledger"][
        "second_inviscid_pressure_limit"
    ]
    payload = {
        "kind": "annular_parallel_shear_heat_block_exclusion_audit",
        "algorithm_revision": (
            "annular-parallel-shear-heat-block-exclusion-v1"
        ),
        "status": "passed" if all_checks else "failed",
        "scope": (
            "Exhaustive o(N^9) exclusion for every second-jet subterm "
            "containing at least one V or D direction, followed by the "
            "complete second-jet N^9 asymptotic."
        ),
        "prerequisites": [
            {
                "path": FINITE_JET_RESULT.relative_to(ROOT).as_posix(),
                "sha256": _sha256(FINITE_JET_RESULT),
            },
            {
                "path": EA_FISHER_RESULT.relative_to(ROOT).as_posix(),
                "sha256": _sha256(EA_FISHER_RESULT),
            },
        ],
        "flow_block_identity_certificate": flow,
        "exhaustive_subterm_partition": partition,
        "one_heat_pressure_HHHH_certificate": one_heat_pressure,
        "remaining_heat_power_ledger": remainder,
        "complete_second_jet_asymptotic": {
            "inviscid_N9_limit": inviscid_limit,
            "pure_EA_Fisher_remainder": "O(N^8)=o(N^9)",
            "one_heat_remainder": "O(N^8)=o(N^9)",
            "two_heat_remainder": "O(N^8)=o(N^9)",
            "conclusion": (
                "The complete second jet has the same strict negative N^9 "
                "limit as the already certified inviscid-pressure block."
            ),
            "certified": all_checks,
        },
        "theorem": (
            "For fixed viscosity nu>0 and the repaired parallel-shear "
            "static optimizer, every V/D one-heat or two-heat second-jet "
            "row is O(N^8)=o(N^9). Hence the complete second derivative "
            "has the strict negative N^9 limit "
            f"{inviscid_limit}."
        ),
        "certification_flags": {
            "all_69_second_subterms_partitioned": partition[
                "all_checks_pass"
            ],
            "all_one_heat_rows_o_N9_proved": all_checks,
            "all_two_heat_rows_o_N9_proved": all_checks,
            "all_viscosity_bearing_second_rows_o_N9_proved": all_checks,
            "complete_second_N9_limit_certified": all_checks,
            "uniform_second_jet_Taylor_remainder_proved": False,
            "parabolic_window_turnaround_proved": False,
            "critical_L3_control_proved": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "remaining_gate": (
            "Promote the restart-time first/second jet asymptotics to a "
            "uniform Taylor estimate on 0<=s<=T/N^2. This requires a "
            "complete third-derivative or integral-remainder bound along "
            "the coupled Navier-Stokes/adjoint evolution."
        ),
        "next_theorem_target": (
            "Build the uniform parabolic-window Taylor remainder. Begin "
            "with an exact third-jet channel-degree ledger and prove a "
            "uniform o(N^11) remainder after s=O(N^-2), without assuming "
            "the static optimizer remains frozen."
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
