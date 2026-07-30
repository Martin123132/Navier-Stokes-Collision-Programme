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
    / "neutral_strip_within_window_second_derivative_certificate.py"
)
RESULT = (
    ROOT
    / "work"
    / "ns_collision"
    / "results"
    / "neutral_strip_h006_within_window_second_derivative_v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "within_window_second_derivative_test_module",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vector_interpolation_charge_uses_h_squared_over_eight():
    module = _module()
    observed = module._vector_interpolation_upper(1.0, 2.0, 3.0)
    expected = 2.0 + module.SUBSTEP**2 * 3.0 / 8.0
    assert observed >= expected
    assert observed <= math.nextafter(expected, math.inf)


def test_functional_maximum_uses_the_interior_critical_point():
    module = _module()
    observed = module._functional_maximum(0.375, 2.36)
    critical = 2.0 / 0.375
    expected = critical**2 * math.exp(-2.0)
    assert observed >= expected
    assert observed <= math.nextafter(expected, math.inf)


def test_production_second_derivative_result_when_present():
    if not RESULT.is_file():
        return
    payload = json.loads(RESULT.read_text(encoding="ascii"))
    assert payload["all_second_derivative_checks_pass"]
    assert payload["all_fifteen_within_window_suprema_certified"]
    assert not payload["post_terminal_time_tail_certified"]
    assert not payload["screen_updated"]
    assert len(payload["window_rows"]) == 15
    assert len(payload["full_window_derivative_rows"]) == 15
    assert len(payload["reduced_window_derivative_rows"]) == 15
    assert all(len(row["subslabs"]) == 10 for row in payload["window_rows"])
    for index, window in enumerate(payload["window_rows"]):
        full = payload["full_window_derivative_rows"][index]
        reduced = payload["reduced_window_derivative_rows"][index]
        assert window["window_index"] == index + 1
        assert full["window_index"] == index + 1
        assert reduced["window_index"] == index + 1
        assert window["full_second_derivative_upper"] == (
            full["exact_column_maximum_upper"]
        )
        assert window["reduced_second_derivative_upper"] == (
            reduced["exact_column_maximum_upper"]
        )
        assert window["difference_second_derivative_upper"] >= (
            full["exact_column_maximum_upper"]
            + reduced["exact_column_maximum_upper"]
        )
    derivatives = [
        row["difference_second_derivative_upper"]
        for row in payload["window_rows"]
    ]
    assert all(
        right < left
        for left, right in zip(derivatives, derivatives[1:])
    )
    assert payload["finite_window_combined_screen_upper"] < 1.0
