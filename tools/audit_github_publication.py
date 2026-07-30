#!/usr/bin/env python3
"""Fail closed when the staged GitHub publication is incomplete or unwieldy."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "release-manifests/research-data-v1.json"
DEFAULT_ARCHIVE_DIRECTORY = ROOT / "release-assets"
DEFAULT_OUTPUT = ROOT / "release-manifests/PUBLICATION_AUDIT.json"
MAX_TRACKED_FILE_BYTES = 10 * 1024 * 1024
MAX_DIRECT_FILES_PER_DIRECTORY = 500
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "release-assets",
    "venv",
}
IGNORED_SUFFIXES = {".log", ".pyc", ".pyo", ".tmp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _git_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(
        value.decode("utf-8")
        for value in completed.stdout.split(b"\0")
        if value
    )


def _candidate_files(archived: set[str]) -> set[str]:
    output = set()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts):
            continue
        relative_text = relative.as_posix()
        if relative_text in archived or path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        output.add(relative_text)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIRECTORY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    manifest = _load_json(arguments.manifest)
    archived = set(manifest["excluded_from_git"])
    errors = []
    archive_rows = []

    selected_now = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "work/ns_collision/results").iterdir()
        if path.is_file()
        and (
            path.suffix.lower() == ".npz"
            or path.stat().st_size > manifest["large_threshold_bytes"]
        )
    }
    if selected_now != archived:
        errors.append("release manifest does not cover the current large data set")

    for row in manifest["archives"]:
        archive_path = arguments.archive_dir / row["asset"]
        state = {
            "asset": row["asset"],
            "exists": archive_path.is_file(),
            "sha256_matches": False,
            "members_match": False,
        }
        if archive_path.is_file():
            state["sha256_matches"] = (
                _sha256(archive_path) == row["asset_sha256"]
            )
            with zipfile.ZipFile(archive_path, "r") as archive:
                state["members_match"] = sorted(archive.namelist()) == sorted(
                    file_row["path"] for file_row in row["files"]
                )
        if not all(
            state[key]
            for key in ("exists", "sha256_matches", "members_match")
        ):
            errors.append(f"release asset failed validation: {row['asset']}")
        archive_rows.append(state)

    tracked = _git_files()
    tracked_set = set(tracked)
    candidates = _candidate_files(archived)
    missing_from_index = sorted(candidates - tracked_set)
    unexpected_tracked = sorted(tracked_set - candidates)
    if missing_from_index:
        errors.append(
            f"publication files missing from Git index: {missing_from_index[:20]}"
        )
    if unexpected_tracked:
        errors.append(
            f"ignored/archive files unexpectedly tracked: {unexpected_tracked[:20]}"
        )

    oversized = []
    directory_counts: Counter[str] = Counter()
    for relative_text in tracked:
        path = ROOT / Path(relative_text)
        if path.stat().st_size > MAX_TRACKED_FILE_BYTES:
            oversized.append(relative_text)
        directory_counts[Path(relative_text).parent.as_posix()] += 1
    crowded = {
        directory: count
        for directory, count in directory_counts.items()
        if count > MAX_DIRECT_FILES_PER_DIRECTORY
    }
    if oversized:
        errors.append(f"oversized tracked files: {oversized}")
    if crowded:
        errors.append(f"crowded tracked directories: {crowded}")

    report = {
        "kind": "navier_stokes_github_publication_audit",
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "repository": manifest["repository"],
        "release_tag": manifest["release_tag"],
        "tracked_file_count": len(tracked),
        "tracked_bytes": sum(
            (ROOT / Path(relative)).stat().st_size for relative in tracked
        ),
        "maximum_tracked_file_bytes": max(
            (ROOT / Path(relative)).stat().st_size for relative in tracked
        ),
        "maximum_direct_directory_file_count": max(directory_counts.values()),
        "maximum_direct_directory": max(
            directory_counts, key=directory_counts.get
        ),
        "archived_file_count": len(archived),
        "archived_source_bytes": manifest["archived_source_bytes"],
        "archive_asset_bytes": manifest["archive_asset_bytes"],
        "archives": archive_rows,
        "missing_from_index": missing_from_index,
        "unexpected_tracked": unexpected_tracked,
        "oversized_tracked_files": oversized,
        "crowded_tracked_directories": crowded,
        "errors": errors,
    }
    _atomic_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
