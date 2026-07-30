#!/usr/bin/env python3
"""Restore large research checkpoints from the hashed GitHub release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "release-manifests/research-data-v1.json"
DEFAULT_ARCHIVE_DIRECTORY = ROOT / "release-assets"


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


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe archive path: {value}")
    return Path(*pure.parts)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Navier-Stokes-Collision-Programme-restorer"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIRECTORY
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="fail when an archive is absent instead of downloading it",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the restored tree without extracting files",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    manifest = _load_json(arguments.manifest)
    repository = manifest["repository"]
    release_tag = manifest["release_tag"]
    restored = 0
    already_present = 0

    for archive_row in manifest["archives"]:
        archive_path = arguments.archive_dir / archive_row["asset"]
        if not archive_path.is_file():
            if arguments.no_download:
                raise FileNotFoundError(archive_path)
            url = (
                f"https://github.com/{repository}/releases/download/"
                f"{release_tag}/{archive_row['asset']}"
            )
            _download(url, archive_path)
        if _sha256(archive_path) != archive_row["asset_sha256"]:
            raise ValueError(f"archive SHA-256 mismatch: {archive_path}")

        expected = {
            row["path"]: row for row in archive_row["files"]
        }
        with zipfile.ZipFile(archive_path, "r") as archive:
            observed = set(archive.namelist())
            if observed != set(expected):
                raise ValueError(
                    f"unexpected archive members in {archive_path.name}"
                )
            for relative_text, file_row in expected.items():
                relative = _safe_relative(relative_text)
                destination = ROOT / relative
                if destination.is_file():
                    if (
                        destination.stat().st_size != file_row["bytes"]
                        or _sha256(destination) != file_row["sha256"]
                    ):
                        raise ValueError(
                            "existing file conflicts with release data: "
                            f"{relative_text}"
                        )
                    already_present += 1
                    continue
                if arguments.verify_only:
                    raise FileNotFoundError(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".restore",
                    dir=destination.parent,
                )
                try:
                    with os.fdopen(fd, "wb") as output:
                        with archive.open(relative_text, "r") as source:
                            shutil.copyfileobj(
                                source, output, length=1024 * 1024
                            )
                        output.flush()
                        os.fsync(output.fileno())
                    temporary = Path(temporary_name)
                    if (
                        temporary.stat().st_size != file_row["bytes"]
                        or _sha256(temporary) != file_row["sha256"]
                    ):
                        raise ValueError(
                            f"restored file mismatch: {relative_text}"
                        )
                    os.replace(temporary, destination)
                    restored += 1
                except BaseException:
                    try:
                        os.unlink(temporary_name)
                    except FileNotFoundError:
                        pass
                    raise

    print(
        json.dumps(
            {
                "status": "pass",
                "archive_count": manifest["archive_count"],
                "verified_file_count": manifest["archived_file_count"],
                "restored_file_count": restored,
                "already_present_file_count": already_present,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
