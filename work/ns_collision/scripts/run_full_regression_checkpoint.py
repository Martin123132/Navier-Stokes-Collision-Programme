"""Run the full NS regression suite and persist its exit status atomically."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

import psutil
import pytest


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    ROOT
    / "work/ns_collision/results/"
    "full_regression_checkpoint_v1.json"
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _set_below_normal_priority() -> bool:
    process = psutil.Process(os.getpid())
    try:
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return process.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
        process.nice(5)
        return process.nice() >= 5
    except (psutil.AccessDenied, psutil.Error):
        return False


class _PytestRecorder:
    """Collect stable aggregate data without parsing terminal output."""

    def __init__(self) -> None:
        self.collected_count = 0
        self.records: dict[str, dict[str, str]] = {}
        self.unexpected_successes: list[str] = []

    def pytest_collection_finish(self, session: Any) -> None:
        self.collected_count = len(session.items)

    def pytest_runtest_logreport(self, report: Any) -> None:
        node_id = str(report.nodeid)
        if report.failed:
            status = "failure" if report.when == "call" else "error"
            self.records[node_id] = {
                "status": status,
                "traceback_tail": str(report.longrepr)[-8000:],
            }
            return
        if report.skipped:
            self.records.setdefault(
                node_id,
                {"status": "skipped", "traceback_tail": ""},
            )
            return
        if report.when == "call" and report.passed:
            if getattr(report, "wasxfail", None):
                self.unexpected_successes.append(node_id)
                self.records[node_id] = {
                    "status": "unexpected_success",
                    "traceback_tail": "",
                }
            else:
                self.records[node_id] = {
                    "status": "passed",
                    "traceback_tail": "",
                }

    def rows(self, status: str) -> list[dict[str, str]]:
        return [
            {
                "test": node_id,
                "traceback_tail": record["traceback_tail"],
            }
            for node_id, record in sorted(self.records.items())
            if record["status"] == status
        ]

    def count(self, status: str) -> int:
        return sum(
            record["status"] == status
            for record in self.records.values()
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-directory",
        default="work/ns_collision/tests",
    )
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=1)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    below_normal = _set_below_normal_priority()
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    recorder = _PytestRecorder()
    verbosity_arguments = {
        0: ["-qq"],
        1: ["-q"],
        2: ["-vv"],
    }[arguments.verbosity]
    exit_code = int(
        pytest.main(
            [
                *verbosity_arguments,
                "-p",
                "no:cacheprovider",
                "-o",
                f"python_files={arguments.pattern}",
                str(ROOT / arguments.start_directory),
            ],
            plugins=[recorder],
        )
    )
    duration = time.perf_counter() - started
    discovered_count = recorder.collected_count
    expected_count_matched = (
        arguments.expected_count is None
        or discovered_count == arguments.expected_count
    )
    successful = exit_code == 0 and expected_count_matched
    failures = recorder.rows("failure")
    errors = recorder.rows("error")
    tests_run = len(recorder.records)
    payload = {
        "kind": "navier_stokes_full_regression_checkpoint",
        "schema_version": 2,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "configuration": {
            "test_engine": "pytest",
            "start_directory": arguments.start_directory,
            "pattern": arguments.pattern,
            "expected_count": arguments.expected_count,
        },
        "runtime": {
            "active_worker_count": 1,
            "below_normal_priority_set": below_normal,
        },
        "discovered_test_count": discovered_count,
        "tests_run": tests_run,
        "passed_count": recorder.count("passed"),
        "skipped_count": recorder.count("skipped"),
        "expected_count_matched": expected_count_matched,
        "failures": failures,
        "errors": errors,
        "unexpected_successes": recorder.unexpected_successes,
        "pytest_exit_code": exit_code,
        "successful": successful,
        "exit_code": 0 if successful else 1,
    }
    _atomic_json(arguments.output, payload)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "discovered_test_count": discovered_count,
                "tests_run": tests_run,
                "successful": successful,
                "duration_seconds": duration,
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(payload["exit_code"])


if __name__ == "__main__":
    main()
