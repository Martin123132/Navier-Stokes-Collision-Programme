#!/usr/bin/env python3
"""Create the additive NS workspace migration records from frozen sources."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKSPACE = ROOT.parent / "l"
SOURCE_TREE = SOURCE_WORKSPACE / "work" / "ns_collision"
DESTINATION_TREE = ROOT / "work" / "ns_collision"
RH_BOOKMARK = SOURCE_WORKSPACE / "work" / "rh_compute" / "results" / "session_bookmark.json"
HASH_MANIFEST = ROOT / "migration" / "SOURCE_TREE_SHA256.json"
MIGRATION_MANIFEST = ROOT / "MIGRATION_MANIFEST.json"
NS_BOOKMARK = DESTINATION_TREE / "results" / "session_bookmark.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


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


def source_records() -> tuple[list[dict[str, Any]], int, str]:
    records: list[dict[str, Any]] = []
    total_bytes = 0
    tree_digest = hashlib.sha256()
    for source_path in sorted(
        (path for path in SOURCE_TREE.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(SOURCE_TREE).as_posix(),
    ):
        relative = source_path.relative_to(SOURCE_TREE).as_posix()
        destination_path = DESTINATION_TREE / Path(relative)
        if not destination_path.is_file():
            raise RuntimeError(f"missing copied file: {destination_path}")
        size = source_path.stat().st_size
        source_hash = sha256_file(source_path)
        destination_hash = sha256_file(destination_path)
        if source_hash != destination_hash or size != destination_path.stat().st_size:
            raise RuntimeError(f"copy mismatch: {relative}")
        records.append({"path": relative, "bytes": size, "sha256": source_hash})
        total_bytes += size
        tree_digest.update(f"{source_hash} {size} {relative}\n".encode("utf-8"))
    return records, total_bytes, tree_digest.hexdigest()


def main() -> None:
    generated_paths = (HASH_MANIFEST, MIGRATION_MANIFEST, NS_BOOKMARK)
    existing = [str(path) for path in generated_paths if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to replace existing migration records: {existing}")

    rh_raw = RH_BOOKMARK.read_bytes()
    rh_data = json.loads(rh_raw)
    ns_checkpoint = rh_data["parallel_research_checkpoints"]["ns_collision_fork"]
    checkpoint_hash = sha256_bytes(canonical_json(ns_checkpoint))
    records, total_bytes, tree_hash = source_records()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    hash_manifest = {
        "kind": "navier_stokes_source_tree_sha256_manifest",
        "schema_version": 1,
        "generated_at": timestamp,
        "algorithm": "sha256",
        "tree_digest_encoding": "sha256 + space + byte_count + space + relative_posix_path + newline",
        "source_root": str(SOURCE_TREE),
        "destination_root": str(DESTINATION_TREE),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "tree_sha256": tree_hash,
        "files": records,
    }

    standalone_bookmark = {
        "kind": "navier_stokes_collision_session_bookmark",
        "schema_version": 1,
        "project": "navier_stokes_collision_programme",
        "workspace_root": ".",
        "research_root": "work/ns_collision",
        "migration": {
            "copied_at": timestamp,
            "source_bookmark": str(RH_BOOKMARK),
            "source_json_path": "parallel_research_checkpoints.ns_collision_fork",
            "source_bookmark_sha256_at_copy": sha256_bytes(rh_raw),
            "source_checkpoint_canonical_sha256": checkpoint_hash,
            "source_checkpoint_fields": list(ns_checkpoint),
            "copy_verified": True,
            "source_removal_owner": "RH task",
            "source_removal_pending": True,
        },
        **ns_checkpoint,
    }

    migration_manifest = {
        "kind": "navier_stokes_workspace_migration_manifest",
        "schema_version": 1,
        "created_at": timestamp,
        "strategy": "additive_copy_then_owner_verified_source_checkpoint_removal",
        "source_workspace": str(SOURCE_WORKSPACE),
        "source_research_root": str(SOURCE_TREE),
        "destination_workspace": str(ROOT),
        "destination_research_root": str(DESTINATION_TREE),
        "source_was_moved_or_deleted": False,
        "rh_bookmark_was_edited_by_ns_task": False,
        "source_snapshot": {
            "file_count": len(records),
            "total_bytes": total_bytes,
            "tree_sha256": tree_hash,
            "file_hash_manifest": "migration/SOURCE_TREE_SHA256.json",
        },
        "checkpoint_migration": {
            "source_bookmark": str(RH_BOOKMARK),
            "source_json_path": "parallel_research_checkpoints.ns_collision_fork",
            "source_bookmark_sha256_at_copy": sha256_bytes(rh_raw),
            "source_checkpoint_canonical_sha256": checkpoint_hash,
            "standalone_bookmark": "work/ns_collision/results/session_bookmark.json",
            "source_checkpoint_fields": list(ns_checkpoint),
            "source_removal_owner": "RH task",
            "source_removal_pending": True,
        },
        "path_compatibility": {
            "preserved_prefix": "work/ns_collision",
            "mass_path_rewrite_performed": False,
            "runtime_dependencies_on_rh": [],
        },
        "shared_foundations": [
            {
                "kind": "conceptual_provenance",
                "path": "work/ns_collision/notes/replica_correlation_bridge.md",
                "description": "Comparison with the zeta mechanism; copied into NS and not linked to RH files.",
            }
        ],
        "validation": {
            "status": "pending",
            "report": "migration/VALIDATION_REPORT.json",
        },
    }

    atomic_json(HASH_MANIFEST, hash_manifest)
    atomic_json(NS_BOOKMARK, standalone_bookmark)
    atomic_json(MIGRATION_MANIFEST, migration_manifest)
    print(
        json.dumps(
            {
                "files": len(records),
                "bytes": total_bytes,
                "tree_sha256": tree_hash,
                "checkpoint_sha256": checkpoint_hash,
                "standalone_bookmark": str(NS_BOOKMARK),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
