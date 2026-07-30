import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "work"
    / "ns_collision"
    / "scripts"
    / "neutral_strip_post_terminal_boundary_tail_certificate.py"
)
RESULT = (
    ROOT
    / "work"
    / "ns_collision"
    / "results"
    / "neutral_strip_h006_post_terminal_boundary_tail_v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "post_terminal_boundary_tail_test_module",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_directed_exponential_upper():
    module = _module()
    observed = module._exp_upper(-1.9 * 0.375)
    expected = math.exp(-1.9 * 0.375)
    assert observed >= expected


def test_production_tail_result_when_present():
    if not RESULT.is_file():
        return
    payload = json.loads(RESULT.read_text(encoding="ascii"))
    assert payload["all_post_terminal_tail_checks_pass"]
    assert payload["post_terminal_time_tail_certified"]
    assert payload["stored_finite_chain_boundary_leakage_screen_complete"]
    assert payload["screen_updated"]
    assert payload["tail_first_window_index"] == 16
    assert payload["geometric_tail"]["full_window_ratio_upper"] < 1.0
    assert payload["geometric_tail"]["reduced_window_ratio_upper"] < 1.0
    screen = payload["screen_composition"]
    assert screen["complete_stored_chain_screen_upper"] < 1.0
    assert screen["complete_stored_chain_screen_headroom_lower"] > 0.0
    assert (
        screen["complete_stored_chain_screen_upper"]
        >= screen["existing_certified_screen_upper"]
        + screen["finite_window_boundary_charge_upper"]
        + screen["post_terminal_boundary_charge_upper"]
    )
    for key, expected_hash in payload["premise_artifacts"].items():
        if not key.endswith("_sha256"):
            continue
        path_key = key.removesuffix("_sha256")
        assert _sha256(ROOT / payload["premise_artifacts"][path_key]) == (
            expected_hash
        )
