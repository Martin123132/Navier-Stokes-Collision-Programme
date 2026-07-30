#!/usr/bin/env python3
"""Fail-closed validation for the standalone NS workspace split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_MANIFEST = ROOT / "MIGRATION_MANIFEST.json"
HASH_MANIFEST = ROOT / "migration" / "SOURCE_TREE_SHA256.json"
PATH_MANIFEST = ROOT / "PROJECT_PATHS.json"
SPLIT_REPORT = ROOT / "migration" / "VALIDATION_REPORT.json"
LIVE_REPORT = ROOT / "migration" / "LIVE_BOUNDARY_VALIDATION_REPORT.json"
FORBIDDEN_RUNTIME_PATTERNS = (
    re.compile(r"work[/\\]rh_compute", re.IGNORECASE),
    re.compile(r"\.\.[/\\]l[/\\]work[/\\]ns_collision", re.IGNORECASE),
)
RH_NS_REFERENCE_PATTERNS = (
    re.compile(r"ns_collision", re.IGNORECASE),
    re.compile(r"navier[-_ ]stokes", re.IGNORECASE),
    re.compile(r"neutral_strip", re.IGNORECASE),
    re.compile(r"collision-test", re.IGNORECASE),
)
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".tex", ".txt"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="write validation JSON")
    args = parser.parse_args()

    errors: list[str] = []
    checks: dict[str, Any] = {}
    try:
        migration = load_json(MIGRATION_MANIFEST)
        hashes = load_json(HASH_MANIFEST)
        paths = load_json(PATH_MANIFEST)
    except Exception as exc:
        print(f"ERROR: failed to load control manifests: {exc}")
        return 1

    source_root = Path(migration["source_research_root"])
    destination_root = Path(migration["destination_research_root"])
    records = hashes["files"]
    tree_digest = hashlib.sha256()
    source_mismatches: list[str] = []
    destination_mismatches: list[str] = []
    for record in records:
        relative = Path(record["path"])
        expected_size = record["bytes"]
        expected_hash = record["sha256"]
        for root, mismatches in (
            (source_root, source_mismatches),
            (destination_root, destination_mismatches),
        ):
            candidate = root / relative
            if not candidate.is_file():
                mismatches.append(f"missing:{record['path']}")
                continue
            if candidate.stat().st_size != expected_size:
                mismatches.append(f"size:{record['path']}")
                continue
            if sha256_file(candidate) != expected_hash:
                mismatches.append(f"sha256:{record['path']}")
        tree_digest.update(
            f"{expected_hash} {expected_size} {record['path']}\n".encode("utf-8")
        )

    expected_tree_hash = hashes["tree_sha256"]
    actual_manifest_tree_hash = tree_digest.hexdigest()
    if source_mismatches:
        errors.append(f"source snapshot mismatches: {source_mismatches[:10]}")
    destination_evolution_allowed = bool(
        migration["source_snapshot"].get(
            "destination_evolution_allowed_after_split", False
        )
    )
    if destination_mismatches and not destination_evolution_allowed:
        errors.append(f"destination copy mismatches: {destination_mismatches[:10]}")
    if actual_manifest_tree_hash != expected_tree_hash:
        errors.append("hash-manifest tree digest mismatch")
    checks["source_snapshot"] = {
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "tree_sha256": expected_tree_hash,
        "source_mismatches": len(source_mismatches),
        "destination_mismatches": len(destination_mismatches),
        "destination_evolution_allowed_after_split": destination_evolution_allowed,
        "destination_divergence_sample": destination_mismatches[:10],
    }

    resolved_paths: dict[str, str] = {}
    for key in (
        "research_root",
        "research_readme",
        "session_bookmark",
        "runtime_policy",
        "migration_manifest",
        "source_tree_hash_manifest",
        "initial_checkpoint_archive",
        "split_validation_report",
    ):
        candidate = (ROOT / paths[key]).resolve()
        resolved_paths[key] = str(candidate)
        if not candidate.exists():
            errors.append(f"PROJECT_PATHS missing target: {key} -> {candidate}")
    checks["project_paths"] = resolved_paths

    bookmark_path = ROOT / paths["session_bookmark"]
    bookmark = load_json(bookmark_path)
    checkpoint_info = migration["checkpoint_migration"]
    source_fields = checkpoint_info["source_checkpoint_fields"]
    initial_checkpoint_path = ROOT / checkpoint_info["initial_checkpoint_archive"]
    initial_checkpoint = load_json(initial_checkpoint_path)
    initial_projection = {key: initial_checkpoint[key] for key in source_fields}
    initial_checkpoint_hash = canonical_sha256(initial_projection)
    expected_checkpoint_hash = checkpoint_info["source_checkpoint_canonical_sha256"]
    if initial_checkpoint_hash != expected_checkpoint_hash:
        errors.append("initial checkpoint archive canonical hash mismatch")
    expected_archive_file_hash = checkpoint_info.get(
        "initial_checkpoint_archive_file_sha256"
    )
    initial_archive_file_hash = sha256_file(initial_checkpoint_path)
    if (
        expected_archive_file_hash
        and initial_archive_file_hash != expected_archive_file_hash
    ):
        errors.append("initial checkpoint archive file hash mismatch")
    live_projection = {key: bookmark[key] for key in source_fields}
    live_checkpoint_hash = canonical_sha256(live_projection)

    rh_bookmark_path = Path(checkpoint_info["source_bookmark"])
    rh_checkpoint_state = "source_bookmark_unavailable"
    rh_bookmark_hash_state = "source_bookmark_unavailable"
    rh_active_ns_references: list[str] = []
    if rh_bookmark_path.is_file():
        rh_bookmark = load_json(rh_bookmark_path)
        rh_node = (
            rh_bookmark.get("parallel_research_checkpoints", {})
            .get("ns_collision_fork")
        )
        if rh_node is None:
            rh_checkpoint_state = "removed_by_rh_owner"
        elif canonical_sha256(rh_node) == expected_checkpoint_hash:
            rh_checkpoint_state = "present_and_hash_matched"
        else:
            rh_checkpoint_state = "present_but_hash_mismatched"
            errors.append("RH source checkpoint changed before handoff")
        acknowledged_rh_hash = checkpoint_info.get("rh_bookmark_sha256_after_removal")
        current_rh_hash = sha256_file(rh_bookmark_path)
        if not acknowledged_rh_hash:
            rh_bookmark_hash_state = "not_declared"
        elif current_rh_hash == acknowledged_rh_hash:
            rh_bookmark_hash_state = "acknowledged_hash_matched"
        elif rh_checkpoint_state == "removed_by_rh_owner":
            serialized_rh_bookmark = json.dumps(
                rh_bookmark,
                sort_keys=True,
                ensure_ascii=True,
            )
            rh_active_ns_references = [
                pattern.pattern
                for pattern in RH_NS_REFERENCE_PATTERNS
                if pattern.search(serialized_rh_bookmark)
            ]
            if rh_active_ns_references:
                rh_bookmark_hash_state = (
                    "evolved_after_acknowledgement_with_ns_references"
                )
                errors.append(
                    "post-removal RH bookmark evolution reintroduced NS references"
                )
            else:
                rh_bookmark_hash_state = (
                    "evolved_after_acknowledgement_without_ns_references"
                )
        else:
            rh_bookmark_hash_state = (
                "hash_mismatched_before_source_removal"
            )
            errors.append("acknowledged post-removal RH bookmark hash mismatch")
    removal_pending = checkpoint_info.get("source_removal_pending")
    bookmark_migration = bookmark.get("migration", {})
    if removal_pending is False and rh_checkpoint_state != "removed_by_rh_owner":
        errors.append("source removal is marked complete but RH checkpoint is still present")
    for key in (
        "source_removal_pending",
        "source_removal_status",
        "source_removal_acknowledged_at",
        "rh_bookmark_sha256_after_removal",
    ):
        if checkpoint_info.get(key) != bookmark_migration.get(key):
            errors.append(f"bookmark/manifest migration acknowledgement mismatch: {key}")
    checks["checkpoint"] = {
        "initial_archive": str(initial_checkpoint_path.resolve()),
        "initial_archive_file_sha256": initial_archive_file_hash,
        "initial_archive_canonical_sha256": initial_checkpoint_hash,
        "live_bookmark_canonical_sha256": live_checkpoint_hash,
        "live_bookmark_differs_from_initial": (
            live_checkpoint_hash != initial_checkpoint_hash
        ),
        "expected_canonical_sha256": expected_checkpoint_hash,
        "rh_source_state": rh_checkpoint_state,
        "rh_bookmark_hash_state": rh_bookmark_hash_state,
        "rh_active_ns_references": rh_active_ns_references,
        "source_removal_pending": removal_pending,
    }

    missing_artifacts: list[str] = []
    for artifact in bookmark.get("primary_artifacts", []):
        candidate = ROOT / Path(artifact)
        if not candidate.exists():
            missing_artifacts.append(artifact)
    if missing_artifacts:
        errors.append(f"missing primary artifacts: {missing_artifacts[:10]}")
    checks["primary_artifacts"] = {
        "declared": len(bookmark.get("primary_artifacts", [])),
        "missing": missing_artifacts,
    }

    invalid_json: list[str] = []
    for json_path in sorted(destination_root.rglob("*.json")):
        try:
            load_json(json_path)
        except Exception as exc:
            invalid_json.append(f"{json_path.relative_to(ROOT).as_posix()}: {exc}")
    if invalid_json:
        errors.append(f"invalid JSON: {invalid_json[:10]}")
    checks["json_parse"] = {
        "files_checked": len(list(destination_root.rglob("*.json"))),
        "invalid": invalid_json,
    }

    forbidden_references: list[str] = []
    baseline_paths = {record["path"] for record in records}
    for relative_text_path in sorted(baseline_paths):
        path = destination_root / Path(relative_text_path)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN_RUNTIME_PATTERNS:
            if pattern.search(content):
                forbidden_references.append(relative_text_path)
                break
    if forbidden_references:
        errors.append(f"forbidden cross-workspace runtime references: {forbidden_references}")
    checks["cross_workspace_runtime_references"] = forbidden_references

    reparse_points: list[str] = []
    for path in destination_root.rglob("*"):
        is_junction = getattr(path, "is_junction", lambda: False)()
        if path.is_symlink() or is_junction:
            reparse_points.append(path.relative_to(ROOT).as_posix())
    if reparse_points:
        errors.append(f"reparse points found: {reparse_points}")
    checks["reparse_points"] = reparse_points

    expected_report_hash = migration.get("validation", {}).get("report_sha256")
    report_hash_state = "not_declared"
    if expected_report_hash:
        if not SPLIT_REPORT.is_file():
            report_hash_state = "missing"
            errors.append("declared validation report is missing")
        elif sha256_file(SPLIT_REPORT) != expected_report_hash:
            report_hash_state = "mismatch"
            errors.append("validation report hash mismatch")
        else:
            report_hash_state = "matched"
    checks["validation_report_hash"] = report_hash_state

    report = {
        "kind": "navier_stokes_workspace_split_validation",
        "schema_version": 1,
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "workspace_root": str(ROOT),
        "status": "pass" if not errors else "fail",
        "checks": checks,
        "errors": errors,
    }
    if args.report:
        atomic_json(LIVE_REPORT, report)
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
