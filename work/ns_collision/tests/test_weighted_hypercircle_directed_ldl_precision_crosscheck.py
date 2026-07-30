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
    "weighted_hypercircle_directed_ldl_precision_crosscheck",
    "neutral_strip_weighted_hypercircle_directed_ldl_precision_crosscheck.py",
)


def test_existing_interaction2048_precision_pair_replays():
    result = MODULE.run_crosscheck(
        lower_audit_path=RESULTS_DIR
        / "neutral_strip_h006_hypercircle_directed_ldl_"
        "interaction2048_audit_v1.json",
        lower_checkpoint_path=RESULTS_DIR
        / "neutral_strip_h006_hypercircle_directed_ldl_"
        "interaction2048_checkpoint_v1.json",
        higher_audit_path=RESULTS_DIR
        / "neutral_strip_h006_hypercircle_directed_ldl_"
        "interaction2048_p80_audit_v1.json",
        higher_checkpoint_path=RESULTS_DIR
        / "neutral_strip_h006_hypercircle_directed_ldl_"
        "interaction2048_p80_checkpoint_v1.json",
        label="interaction2048",
    )
    assert result["status"] == "pass"
    assert result["all_checks_pass"]
    assert all(result["checks"].values())
    assert result["comparison"]["pivot_count"] == 2048
    assert result["comparison"]["symbolic_lower_entry_count"] == 5612
