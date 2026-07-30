"""Install the validated transition-32064 checkpoint in the NS bookmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BOOKMARK = ROOT / "work/ns_collision/results/session_bookmark.json"

RESULTS = {
    "directed_2304": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_directed_ldl_transition2304_audit_v1.json"
    ),
    "directed_2304_p80": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_directed_ldl_transition2304_p80_audit_v1.json"
    ),
    "directed_cross_2304": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_directed_ldl_transition2304_"
        "precision_crosscheck_v1.json"
    ),
    "residual_2304": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_congruence_residual_pilot2304_v1.json"
    ),
    "residual_2304_p100": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_congruence_residual_pilot2304_p100_v1.json"
    ),
    "residual_cross_2304": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_congruence_residual_"
        "precision_crosscheck2304_v1.json"
    ),
    "symbolic_32768": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_symbolic_transition_map32768_v1.json"
    ),
    "directed_32064": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_directed_ldl_transition32064_audit_v1.json"
    ),
    "directed_32064_p80": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_directed_ldl_transition32064_p80_audit_v1.json"
    ),
    "directed_cross_32064": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_directed_ldl_transition32064_"
        "precision_crosscheck_v1.json"
    ),
    "residual_32064": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_congruence_residual_pilot32064_v1.json"
    ),
    "residual_32064_p100": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_congruence_residual_pilot32064_p100_v1.json"
    ),
    "residual_cross_32064": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_congruence_residual_"
        "precision_crosscheck32064_v1.json"
    ),
    "symbolic_full": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_symbolic_transition_map123816_v1.json"
    ),
    "feasibility": (
        "work/ns_collision/results/"
        "neutral_strip_h006_hypercircle_full_feasibility_audit_v1.json"
    ),
}

CHECKPOINTS = (
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_directed_ldl_transition2304_checkpoint_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_directed_ldl_transition2304_p80_checkpoint_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_directed_ldl_transition32064_checkpoint_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_directed_ldl_transition32064_p80_checkpoint_v1.json",
)

IMPLEMENTATION_ARTIFACTS = (
    "work/ns_collision/README.md",
    "work/ns_collision/notes/"
    "neutral_strip_weighted_hypercircle_transition32064_feasibility.md",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_directed_ldl_precision_crosscheck.py",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_congruence_residual_pilot.py",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_congruence_residual_precision_crosscheck.py",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_symbolic_transition_map.py",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_full_feasibility_audit.py",
    "work/ns_collision/scripts/update_transition32064_bookmark.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_directed_ldl_prefix_audit.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_directed_ldl_precision_crosscheck.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_congruence_residual_pilot.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_congruence_residual_precision_crosscheck.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_symbolic_transition_map.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_full_feasibility_audit.py",
)


def _load_json(relative_path: str | Path) -> dict[str, Any]:
    path = ROOT / relative_path
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{relative_path} must contain a JSON object")
    return value


def _sha256(relative_path: str | Path) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _append_once(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def _validate_inputs(results: dict[str, dict[str, Any]]) -> None:
    for name in ("directed_2304", "directed_2304_p80", "directed_32064",
                 "directed_32064_p80"):
        result = results[name]
        _require(
            result.get("all_current_stage_checks_pass") is True,
            f"{name} does not pass all current-stage checks",
        )
        _require(
            result.get("certification_flags", {}).get(
                "bounded_prefix_directed_LDL_certified"
            )
            is True,
            f"{name} is not a bounded-prefix certificate",
        )
        _require(
            result.get("certification_flags", {}).get(
                "full_123816_pivot_inertia_certified"
            )
            is False,
            f"{name} unexpectedly claims full inertia",
        )

    for name in ("directed_cross_2304", "directed_cross_32064"):
        _require(
            results[name].get("all_checks_pass") is True,
            f"{name} crosscheck does not pass",
        )

    for name in ("residual_2304", "residual_2304_p100", "residual_32064",
                 "residual_32064_p100"):
        result = results[name]
        _require(
            result.get("all_current_stage_checks_pass") is True,
            f"{name} does not pass all current-stage checks",
        )
        _require(
            result.get("certificate", {}).get(
                "interval_family_inertia_certified"
            )
            is True,
            f"{name} does not close its residual certificate",
        )

    for name in ("residual_cross_2304", "residual_cross_32064"):
        _require(
            results[name].get("all_checks_pass") is True,
            f"{name} crosscheck does not pass",
        )

    symbolic = results["symbolic_full"]
    _require(symbolic.get("status") == "pass", "full symbolic map does not pass")
    _require(
        symbolic.get("certification_flags", {}).get(
            "any_new_pivot_sign_certified"
        )
        is False,
        "symbolic map unexpectedly claims arithmetic signs",
    )

    feasibility = results["feasibility"]
    _require(
        feasibility.get("all_current_stage_checks_pass") is True,
        "feasibility audit does not pass",
    )
    flags = feasibility.get("certification_flags", {})
    _require(
        flags.get("bounded_32064_inertia_certified_by_two_routes") is True,
        "feasibility audit does not recognize both bounded certificates",
    )
    _require(
        flags.get("full_123816_pivot_inertia_certified") is False,
        "feasibility audit unexpectedly claims full inertia",
    )
    _require(
        flags.get("navier_stokes_regularity_certified") is False,
        "feasibility audit unexpectedly claims regularity",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-test-count", type=int, required=True)
    parser.add_argument("--full-test-seconds", type=float, required=True)
    parser.add_argument("--targeted-test-count", type=int, required=True)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _require(args.full_test_count > 0, "full test count must be positive")
    _require(args.targeted_test_count > 0, "targeted test count must be positive")
    _require(args.full_test_seconds >= 0.0, "full test runtime must be nonnegative")
    _require(
        args.targeted_test_seconds >= 0.0,
        "targeted test runtime must be nonnegative",
    )

    results = {name: _load_json(path) for name, path in RESULTS.items()}
    _validate_inputs(results)
    for path in (*RESULTS.values(), *CHECKPOINTS, *IMPLEMENTATION_ARTIFACTS):
        _require((ROOT / path).is_file(), f"missing required artifact: {path}")

    bookmark = _load_json(BOOKMARK.relative_to(ROOT))
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark",
        "refusing to update a non-NS bookmark",
    )
    _require(
        bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "bookmark workspace boundary is not standalone",
    )

    directed = results["directed_32064"]["directed_LDL_prefix"]
    residual = results["residual_32064"]["certificate"]
    symbolic = results["symbolic_full"]["profile"]
    feasibility = results["feasibility"]
    decision = feasibility["launch_decision"]
    workload = feasibility["full_symbolic_workload"]
    storage = feasibility["checkpoint_storage_projection"]

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "checkpointed"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The stored finite-chain screen, h=0.06 weighted-hypercircle geometry "
        "and complete P/W/D/B/N interval assembly, and frozen ten-step Ruiz "
        "plus MMD_AT_PLUS_A order remain validated. Directed interval LDL at "
        "precisions 50 and 80 now certifies pivots 0..32063 for every matrix "
        "in the stored interval family: 31486 negative and 578 positive, "
        "minimum margin 0.1464628437608216996667076129, 125492 lower "
        "intervals, and complete higher-precision nesting. An independent "
        "congruence-residual certificate at precisions 60 and 100 obtains "
        "the same signs with transformed-bound/minimum-diagonal ratio below "
        "8.438e-11. The exact full symbolic map records 2699822 lower "
        "entries and 159047977 off-diagonal common terms, with the first "
        "state pivot at 63644 and a strongly backloaded tail. Full "
        "123816-pivot inertia, the weighted global Ritz constant, continuum "
        "spectral capture, and every broader Navier-Stokes claim remain "
        "false. This remains far short of a regularity or Clay-prize proof."
    )

    principal = bookmark.setdefault("principal_results", {})
    principal.update(
        {
            "reversible_weighted_hypercircle_transition2304_certified": True,
            "reversible_weighted_hypercircle_transition2304_signs": {
                "negative": 1768,
                "positive": 536,
                "zero": 0,
            },
            "reversible_weighted_hypercircle_transition2304_p80_nesting_crosscheck": True,
            "reversible_weighted_hypercircle_congruence_residual_method_implemented": True,
            "reversible_weighted_hypercircle_transition32064_certified": True,
            "reversible_weighted_hypercircle_transition32064_pivot_count": directed[
                "completed_pivot_count"
            ],
            "reversible_weighted_hypercircle_transition32064_signs": {
                "negative": directed["negative_pivot_count"],
                "positive": directed["positive_pivot_count"],
                "zero": 0,
            },
            "reversible_weighted_hypercircle_transition32064_minimum_margin": directed[
                "minimum_pivot_margin_decimal"
            ],
            "reversible_weighted_hypercircle_transition32064_symbolic_lower_entries": directed[
                "symbolic_lower_entry_count"
            ],
            "reversible_weighted_hypercircle_transition32064_maximum_relative_radius": directed[
                "maximum_pivot_radius_to_margin_upper_decimal"
            ],
            "reversible_weighted_hypercircle_transition32064_p80_nesting_crosscheck": True,
            "reversible_weighted_hypercircle_residual32064_certified": residual[
                "interval_family_inertia_certified"
            ],
            "reversible_weighted_hypercircle_residual32064_bound_to_diagonal_ratio": residual[
                "transformed_bound_to_minimum_diagonal_upper_decimal"
            ],
            "reversible_weighted_hypercircle_residual32064_p100_nesting_crosscheck": True,
            "reversible_weighted_hypercircle_full_symbolic_profile_validated": True,
            "reversible_weighted_hypercircle_full_symbolic_lower_entries": symbolic[
                "symbolic_lower_entry_count"
            ],
            "reversible_weighted_hypercircle_full_off_diagonal_common_terms": symbolic[
                "total_off_diagonal_common_term_count"
            ],
            "reversible_weighted_hypercircle_full_reference_product_pair_terms": symbolic[
                "reference_product_pair_term_count"
            ],
            "reversible_weighted_hypercircle_first_state_pivot": workload[
                "first_state_pivot"
            ],
            "reversible_weighted_hypercircle_next_symbolic_transition": decision[
                "next_directed_transition_pivot"
            ],
            "reversible_weighted_hypercircle_next_bounded_pivot_target": decision[
                "next_directed_bounded_pivot_count"
            ],
            "reversible_weighted_hypercircle_last9128_common_term_fraction": workload[
                "last_9128_pivots_common_term_fraction"
            ],
            "reversible_weighted_hypercircle_last9128_reference_pair_fraction": workload[
                "last_9128_pivots_reference_pair_fraction"
            ],
            "reversible_weighted_hypercircle_projected_full_checkpoint_MiB_p50": storage[
                "precision_50"
            ]["projected_full_checkpoint_MiB"],
            "reversible_weighted_hypercircle_projected_full_checkpoint_MiB_p80": storage[
                "precision_80"
            ]["projected_full_checkpoint_MiB"],
            "reversible_weighted_hypercircle_full_directed_launch_ready": decision[
                "full_directed_LDL_launch_ready"
            ],
            "reversible_weighted_hypercircle_full_residual_launch_ready": decision[
                "full_congruence_residual_launch_ready"
            ],
            "reversible_weighted_hypercircle_full_inertia_certified": False,
            "validated_ns_collision_test_count": args.full_test_count,
            "validated_ns_collision_test_runtime_seconds": args.full_test_seconds,
            "validated_new_targeted_test_count": args.targeted_test_count,
            "validated_new_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "validated_directed_LDL_test_count": args.targeted_test_count,
        }
    )
    for name, relative_path in RESULTS.items():
        principal[
            f"reversible_weighted_hypercircle_{name}_result_sha256"
        ] = _sha256(relative_path)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Crossed the 2270/2274 recurrence transition and certified all "
            "2304 directed-LDL pivots at precisions 50 and 80, with 1768 "
            "negative and 536 positive signs and complete pivot/lower "
            "interval nesting."
        ),
    )
    _append_once(
        completed,
        (
            "Implemented and adversarially tested an independent directed "
            "congruence-residual certificate with no interval pivot "
            "divisions; together with componentwise directed LDL it "
            "certifies the bounded 32064-pivot problem as 31486 negative "
            "and 578 positive at two precision levels for each route."
        ),
    )
    _append_once(
        completed,
        (
            "Mapped the exact potential fill across all 123816 pivots, "
            "identified pivot 33224 as the next transition and 63644 as the "
            "first state pivot, quantified the strongly backloaded tail, "
            "and rejected a full launch until checkpoint writes are "
            "hash-chained and bounded state-entry behavior is tested."
        ),
    )

    bookmark["unfinished_obligation"] = (
        "The finite weighted-hypercircle threshold pencil is now certified "
        "only through pivot 32063 by two interval-propagation routes. The "
        "next bounded gate is 33280, crossing the fill transition at 33224. "
        "After that, run a residual-only 63680 pilot across first state entry "
        "at pivot 63644 and replace monolithic checkpoints with chunked "
        "hash-chained records before reconsidering all 123816 pivots. Full "
        "inertia 61908/61908/0 remains unproved. Even full finite inertia "
        "would leave the strict solution-operator threshold "
        "0.007322422996991409, weighted Ritz threshold "
        "0.08557115750643675, positive-time source transfer, conormal "
        "transfer, polygon-circle perturbation, and Navier-Stokes closure "
        "open. Every broader claim remains fail-closed."
    )
    bookmark["resume_command"] = "not_applicable_no_parked_compute"
    bookmark["next_action"] = (
        "After a fresh five-second CPU gate, create separate precision-50 "
        "and precision-80 directed-LDL checkpoints through exactly 33280 "
        "pivots, run the independent precision-60/100 congruence-residual "
        "certificate on the same bounded prefix, and require all nesting and "
        "sign checks to pass. Do not launch the full pencil. The following "
        "stage is a residual-only 63680 state-entry pilot, followed by a "
        "chunked hash-chain checkpoint redesign."
    )

    artifacts = bookmark.setdefault("primary_artifacts", [])
    for path in (*RESULTS.values(), *CHECKPOINTS, *IMPLEMENTATION_ARTIFACTS):
        _append_once(artifacts, path)

    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK.relative_to(ROOT)),
                "primary_artifact_count": len(artifacts),
                "completed_obligation_count": len(completed),
                "status": bookmark["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
