#!/usr/bin/env python3
"""Build deterministic release archives for large research checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile

import psutil


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "work/ns_collision/results"
DEFAULT_OUTPUT = ROOT / "release-assets"
DEFAULT_MANIFEST = ROOT / "release-manifests/research-data-v1.json"
REPOSITORY = "Martin123132/Navier-Stokes-Collision-Programme"
RELEASE_TAG = "research-data-2026-07-30"
LARGE_THRESHOLD = 10 * 1024 * 1024
EXPECTED_GROUP_COUNTS = {
    "hypercircle-transition-32064": 4,
    "hypercircle-transition-33280": 4,
    "sparse-inertia": 9,
    "reference-eigensystems": 4,
    "neutral-strip-support": 8,
}
ZIP_TIMESTAMP = (2026, 7, 30, 0, 0, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lower_process_priority() -> bool:
    process = psutil.Process(os.getpid())
    try:
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return process.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
        process.nice(5)
        return process.nice() >= 5
    except (psutil.AccessDenied, psutil.Error):
        return False


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


def _selected_files() -> list[Path]:
    selected = [
        path
        for path in RESULTS.iterdir()
        if path.is_file()
        and (path.suffix.lower() == ".npz" or path.stat().st_size > LARGE_THRESHOLD)
    ]
    return sorted(selected, key=lambda path: path.name)


def _group(path: Path) -> str:
    name = path.name
    if "transition32064" in name:
        return "hypercircle-transition-32064"
    if "transition33280" in name:
        return "hypercircle-transition-33280"
    if "q12_k240" in name:
        return "sparse-inertia"
    if "reference_eigensystem" in name:
        return "reference-eigensystems"
    return "neutral-strip-support"


def _archive_name(group: str) -> str:
    return f"{RELEASE_TAG}-{group}.zip"


def _build_archive(path: Path, sources: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for source in sources:
                relative = source.relative_to(ROOT).as_posix()
                info = zipfile.ZipInfo(relative, date_time=ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with source.open("rb") as input_handle:
                    with archive.open(
                        info, mode="w", force_zip64=True
                    ) as output_handle:
                        shutil.copyfileobj(
                            input_handle, output_handle, length=1024 * 1024
                        )
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing assets instead of rebuilding them",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    below_normal_priority_set = _lower_process_priority()
    selected = _selected_files()
    grouped: dict[str, list[Path]] = {
        name: [] for name in EXPECTED_GROUP_COUNTS
    }
    for source in selected:
        grouped[_group(source)].append(source)
    observed_counts = {
        name: len(paths) for name, paths in grouped.items()
    }
    if observed_counts != EXPECTED_GROUP_COUNTS:
        raise ValueError(
            "large-file grouping changed: "
            f"expected {EXPECTED_GROUP_COUNTS}, observed {observed_counts}"
        )

    archives = []
    archived_paths = []
    for group, sources in grouped.items():
        archive_path = arguments.output_directory / _archive_name(group)
        if not arguments.verify_only:
            _build_archive(archive_path, sources)
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        file_rows = []
        for source in sources:
            row = {
                "path": source.relative_to(ROOT).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
            file_rows.append(row)
            archived_paths.append(row["path"])
        with zipfile.ZipFile(archive_path, "r") as archive:
            names = sorted(archive.namelist())
        expected_names = sorted(row["path"] for row in file_rows)
        if names != expected_names:
            raise ValueError(f"archive member mismatch: {archive_path}")
        archives.append(
            {
                "group": group,
                "asset": archive_path.name,
                "asset_bytes": archive_path.stat().st_size,
                "asset_sha256": _sha256(archive_path),
                "file_count": len(file_rows),
                "source_bytes": sum(row["bytes"] for row in file_rows),
                "files": file_rows,
            }
        )

    manifest = {
        "kind": "navier_stokes_research_data_release_manifest",
        "schema_version": 1,
        "checkpoint_date": "2026-07-30",
        "repository": REPOSITORY,
        "release_tag": RELEASE_TAG,
        "selection_rule": (
            "all work/ns_collision/results/*.npz files and every result "
            "file larger than 10 MiB"
        ),
        "large_threshold_bytes": LARGE_THRESHOLD,
        "archive_count": len(archives),
        "archived_file_count": len(archived_paths),
        "archived_source_bytes": sum(
            archive["source_bytes"] for archive in archives
        ),
        "archive_asset_bytes": sum(
            archive["asset_bytes"] for archive in archives
        ),
        "build_runtime": {
            "active_worker_count": 1,
            "below_normal_priority_set": below_normal_priority_set,
        },
        "excluded_from_git": sorted(archived_paths),
        "archives": archives,
    }
    if len(archived_paths) != 29:
        raise ValueError(f"expected 29 archived files, found {len(archived_paths)}")
    _atomic_json(arguments.manifest, manifest)
    print(
        json.dumps(
            {
                "manifest": arguments.manifest.relative_to(ROOT).as_posix(),
                "archive_count": len(archives),
                "archived_file_count": len(archived_paths),
                "archived_source_bytes": manifest["archived_source_bytes"],
                "archive_asset_bytes": manifest["archive_asset_bytes"],
                "assets": [
                    {
                        "name": row["asset"],
                        "bytes": row["asset_bytes"],
                        "sha256": row["asset_sha256"],
                    }
                    for row in archives
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
