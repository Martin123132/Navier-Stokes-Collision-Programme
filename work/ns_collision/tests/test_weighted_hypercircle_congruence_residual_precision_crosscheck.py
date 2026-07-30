import importlib.util
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name,
        SCRIPT_DIR / filename,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load(
    "weighted_hypercircle_congruence_residual_precision_crosscheck",
    "neutral_strip_weighted_hypercircle_"
    "congruence_residual_precision_crosscheck.py",
)


def test_stored_residual_precision_pair_nests():
    result = MODULE.run_crosscheck(
        lower_path=RESULTS_DIR
        / "neutral_strip_h006_hypercircle_"
        "congruence_residual_pilot2304_v1.json",
        higher_path=RESULTS_DIR
        / "neutral_strip_h006_hypercircle_"
        "congruence_residual_pilot2304_p100_v1.json",
    )
    assert result["status"] == "pass"
    assert result["all_checks_pass"]
    assert all(result["checks"].values())
    assert all(result["upper_bound_nesting_checks"].values())
