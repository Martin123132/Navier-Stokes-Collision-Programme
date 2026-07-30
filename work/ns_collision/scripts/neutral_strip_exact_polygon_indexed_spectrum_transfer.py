"""Transfer indexed stored-pencil eigenvalue intervals to exact polygon forms."""

from __future__ import annotations

import argparse
from decimal import Context, Decimal
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


ASSEMBLY_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
)
EIGENSYSTEM_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_eigensystem_residual_audit_v1.json"
)
INERTIA_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_sparse_inertia_audit_v1.json"
)
PRECISION_CROSSCHECK_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_sparse_inertia_precision_crosscheck_v1.json"
)


def _down(value: float) -> float:
    return float(np.nextafter(float(value), -math.inf))


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _exact_form_interval(
    stored_lower: float,
    stored_upper: float,
    stiffness_error: float,
    exact_mass_lower: float,
    exact_mass_upper: float,
) -> tuple[float, float]:
    numerator_lower = _down(stored_lower - stiffness_error)
    numerator_upper = _up(stored_upper + stiffness_error)
    if numerator_lower <= 0.0 or exact_mass_lower <= 0.0:
        raise ValueError("positive-form transfer assumptions failed")
    return (
        _down(numerator_lower / exact_mass_upper),
        _up(numerator_upper / exact_mass_lower),
    )


def transfer(
    assembly_result_path: Path,
    eigensystem_result_path: Path,
    inertia_result_path: Path,
    precision_crosscheck_result_path: Path,
) -> dict[str, object]:
    with assembly_result_path.open("r", encoding="utf-8") as handle:
        assembly = json.load(handle)
    with eigensystem_result_path.open("r", encoding="utf-8") as handle:
        eigensystem = json.load(handle)
    with inertia_result_path.open("r", encoding="utf-8") as handle:
        inertia = json.load(handle)
    with precision_crosscheck_result_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        precision_crosscheck = json.load(handle)

    form_bounds = assembly["form_bounds"]
    mass_error = float(
        form_bounds["absolute_mass_error_relative_to_stored_mass_form"]
    )
    stiffness_error = float(
        form_bounds[
            "absolute_stiffness_error_in_stored_mass_form_units"
        ]
    )
    exact_mass_lower = float(
        form_bounds["exact_mass_lower_relative_to_stored_mass"]
    )
    exact_mass_upper = _up(1.0 + mass_error)
    pair_rows = eigensystem["reference_generalized_eigensystem"][
        "pair_rows"
    ]

    decimal_context = Context(prec=100)
    mass_upper_decimal = decimal_context.add(
        Decimal(1),
        Decimal.from_float(mass_error),
    )
    mass_lower_decimal = Decimal.from_float(exact_mass_lower)
    stiffness_error_decimal = Decimal.from_float(stiffness_error)
    rows = []
    all_decimal_formula_values_contained = True
    for expected_index, pair_row in enumerate(pair_rows):
        if int(pair_row["index"]) != expected_index:
            raise RuntimeError("eigenvalue rows are not in index order")
        stored_lower = float(pair_row["proximity_interval_lower"])
        stored_upper = float(pair_row["proximity_interval_upper"])
        exact_lower, exact_upper = _exact_form_interval(
            stored_lower,
            stored_upper,
            stiffness_error,
            exact_mass_lower,
            exact_mass_upper,
        )
        decimal_lower = decimal_context.divide(
            decimal_context.subtract(
                Decimal.from_float(stored_lower),
                stiffness_error_decimal,
            ),
            mass_upper_decimal,
        )
        decimal_upper = decimal_context.divide(
            decimal_context.add(
                Decimal.from_float(stored_upper),
                stiffness_error_decimal,
            ),
            mass_lower_decimal,
        )
        decimal_contained = bool(
            Decimal.from_float(exact_lower)
            <= decimal_lower
            <= decimal_upper
            <= Decimal.from_float(exact_upper)
        )
        all_decimal_formula_values_contained &= decimal_contained
        rows.append(
            {
                "index": expected_index,
                "stored_proximity_interval_lower": stored_lower,
                "stored_proximity_interval_upper": stored_upper,
                "exact_polygon_indexed_interval_lower": exact_lower,
                "exact_polygon_indexed_interval_upper": exact_upper,
                "decimal_formula_values_contained": decimal_contained,
            }
        )

    adjacent_separations = [
        _down(
            rows[index + 1]["exact_polygon_indexed_interval_lower"]
            - rows[index]["exact_polygon_indexed_interval_upper"]
        )
        for index in range(len(rows) - 1)
    ]
    minimum_adjacent_separation = min(adjacent_separations)
    retained_upper = rows[239][
        "exact_polygon_indexed_interval_upper"
    ]
    complement_lower = rows[240][
        "exact_polygon_indexed_interval_lower"
    ]
    retained_complement_separation = _down(
        complement_lower - retained_upper
    )

    premises = {
        "complete_exact_assembly_audit": bool(
            assembly["complete_mesh_audit"]
            and assembly["finite_element_assembly_interval_enclosed"]
        ),
        "all_four_stored_matrix_fingerprints_match": bool(
            assembly["stored_matrix_reconstruction"][
                "all_four_matrix_fingerprints_match"
            ]
        ),
        "stored_mass_coercivity_proved": bool(
            inertia["mass_coercivity_proved"]
        ),
        "all_241_stored_intervals_indexed": bool(
            inertia[
                "all_241_stored_generalized_eigenvalue_intervals_indexed"
            ]
        ),
        "stored_inertia_audit_passed": bool(
            inertia["all_sparse_inertia_audit_checks_pass"]
        ),
        "independent_precision_crosscheck_passed": bool(
            precision_crosscheck[
                "all_sparse_inertia_precision_crosscheck_checks_pass"
            ]
        ),
        "exact_mass_lower_positive": exact_mass_lower > 0.0,
        "all_stored_lower_endpoints_exceed_stiffness_error": all(
            float(row["proximity_interval_lower"]) > stiffness_error
            for row in pair_rows
        ),
    }
    all_exact_intervals_disjoint = minimum_adjacent_separation > 0.0
    exact_first_240_indexed = bool(
        all(premises.values())
        and all_decimal_formula_values_contained
        and all_exact_intervals_disjoint
        and retained_complement_separation > 0.0
    )
    exact_all_241_indexed = bool(
        exact_first_240_indexed and len(rows) == 241
    )
    checks = [
        len(rows) == 241,
        all(premises.values()),
        all_decimal_formula_values_contained,
        all_exact_intervals_disjoint,
        retained_complement_separation > 0.0,
        exact_first_240_indexed,
        exact_all_241_indexed,
    ]
    return {
        "model": (
            "min-max transfer of indexed stored generalized eigenvalues "
            "to exact Gaussian-weighted P1 forms on the stored polygon"
        ),
        "theorem": (
            "If |a_exact-a_stored| <= eta_A m_stored and "
            "mu_- m_stored <= m_exact <= mu_+ m_stored, then "
            "(lambda_stored_j-eta_A)/mu_+ <= lambda_exact_j <= "
            "(lambda_stored_j+eta_A)/mu_- for every indexed j."
        ),
        "assembly_result": str(assembly_result_path),
        "assembly_result_sha256": _sha256_file(assembly_result_path),
        "eigensystem_result": str(eigensystem_result_path),
        "eigensystem_result_sha256": _sha256_file(
            eigensystem_result_path
        ),
        "inertia_result": str(inertia_result_path),
        "inertia_result_sha256": _sha256_file(inertia_result_path),
        "precision_crosscheck_result": str(
            precision_crosscheck_result_path
        ),
        "precision_crosscheck_result_sha256": _sha256_file(
            precision_crosscheck_result_path
        ),
        "state_count": int(inertia["state_count"]),
        "indexed_interval_count": len(rows),
        "mass_form_relative_error_upper": mass_error,
        "stiffness_form_error_in_stored_mass_units_upper": (
            stiffness_error
        ),
        "exact_mass_relative_lower": exact_mass_lower,
        "exact_mass_relative_upper": exact_mass_upper,
        "premises": premises,
        "rows": rows,
        "all_decimal_formula_values_contained": (
            all_decimal_formula_values_contained
        ),
        "all_exact_polygon_indexed_intervals_disjoint": (
            all_exact_intervals_disjoint
        ),
        "minimum_adjacent_exact_interval_separation": (
            minimum_adjacent_separation
        ),
        "exact_retained_index_239_interval_upper": retained_upper,
        "exact_complement_index_240_interval_lower": complement_lower,
        "exact_retained_complement_separation": (
            retained_complement_separation
        ),
        "first_240_exact_polygon_generalized_eigenvalues_indexed": (
            exact_first_240_indexed
        ),
        "all_241_exact_polygon_generalized_eigenvalues_indexed": (
            exact_all_241_indexed
        ),
        "exact_polygon_complement_generalized_eigenvalue_lower_bound": (
            complement_lower if exact_first_240_indexed else None
        ),
        "exact_polygon_generalized_eigenvectors_interval_enclosed": False,
        "endpoint_effect_of_indexed_spectrum_transfer_certified": False,
        "continuum_Ritz_transfer_proved": False,
        "polygon_to_circle_domain_transfer_proved": False,
        "all_exact_polygon_indexed_spectrum_transfer_checks_pass": bool(
            all(checks)
        ),
        "scope": (
            "This transfers only from the stored binary matrices to the "
            "exact Gaussian-weighted finite-element forms on the same "
            "stored polygon. It does not prove continuum Ritz or "
            "polygon-to-circle domain transfer."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assembly-result",
        type=Path,
        default=ASSEMBLY_RESULT,
    )
    parser.add_argument(
        "--eigensystem-result",
        type=Path,
        default=EIGENSYSTEM_RESULT,
    )
    parser.add_argument(
        "--inertia-result",
        type=Path,
        default=INERTIA_RESULT,
    )
    parser.add_argument(
        "--precision-crosscheck-result",
        type=Path,
        default=PRECISION_CROSSCHECK_RESULT,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = transfer(
        args.assembly_result,
        args.eigensystem_result,
        args.inertia_result,
        args.precision_crosscheck_result,
    )
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
