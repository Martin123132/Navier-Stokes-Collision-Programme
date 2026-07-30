"""Audit a pressure-active HHL chain against the full signed Fisher graph."""

from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from cross_shell_modulated_wave_gate_audit import (
    _add_real_mode,
    _component_fluxes,
    _direct_linear_flux,
    _load,
    _maximum_vector_difference,
    _pressure_bilinear,
    audit as cross_shell_audit,
)
from multiband_weighted_fisher_recombination_no_go_audit import (
    _finite_field_recombination_audit,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "pressure_active_fisher_null_compatibility_gate_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "cross_shell_modulated_wave_gate_audit_v1.json"
    ): "d6c330cba935e2bc8bcac55e462adfb97d91f04b42ed92faf209cea598d35597",
    (
        "work/ns_collision/results/"
        "balanced_annular_pressure_edge_gate_audit_v1.json"
    ): "9a024a23381d62e7842d7d26406fcea2a5343a168f386d3bad85e5308cef99dd",
    (
        "work/ns_collision/results/"
        "multiband_weighted_fisher_recombination_no_go_audit_v1.json"
    ): "47b8704985671f0dac66ae38ff87a186acd6b938928828d3299a571337a7f087",
}
VERTICES = ((-1, 1, -1), (-1, 1, 1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _prerequisite_audit() -> dict[str, Any]:
    rows = []
    for relative, expected in PREREQUISITES.items():
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = _sha256(path)
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "all_positive_checks_pass": payload.get(
                    "all_positive_checks_pass"
                ),
                "matches": (
                    actual == expected
                    and payload.get("all_positive_checks_pass") is True
                ),
            }
        )
    return {
        "rows": rows,
        "all_checks_pass": all(row["matches"] for row in rows),
    }


def _geometry(edge: int) -> dict[str, float]:
    n = float(edge)
    left = n * n + 1.0
    right = (n + 1.0) ** 2 + 1.0
    dot = n * (n + 1.0) + 1.0
    cosine = dot / math.sqrt(left * right)
    sine_squared = 1.0 / (left * right)
    shear_complete = -1.0 / (4.0 * math.sqrt(left * right))
    inplane_kinetic_scalar = -dot / (4.0 * left * right)
    inplane_kinetic_anisotropic = -0.5 * sine_squared
    pressure_high_high = 0.5 * sine_squared
    numerator = (
        edge**6
        + 3 * edge**5
        + 14 * edge**4
        + 23 * edge**3
        + 35 * edge**2
        + 24 * edge
        + 4
    )
    denominator = (
        4
        * (edge**2 + 1)
        * (edge**2 + 4)
        * (edge**2 + 2 * edge + 2)
        * (edge**2 + 2 * edge + 5)
    )
    inplane_complete = -numerator / denominator
    pressure_cross = inplane_complete - inplane_kinetic_scalar
    return {
        "cosine": cosine,
        "sine_squared": sine_squared,
        "shear_Fisher_Gram": cosine,
        "inplane_Fisher_Gram": cosine * cosine,
        "shear_complete": shear_complete,
        "inplane_kinetic_scalar": inplane_kinetic_scalar,
        "inplane_kinetic_anisotropic": (
            inplane_kinetic_anisotropic
        ),
        "pressure_high_high": pressure_high_high,
        "pressure_cross": pressure_cross,
        "inplane_complete": inplane_complete,
    }


def _chain_field(
    start: int,
    shear: np.ndarray,
    inplane: np.ndarray,
) -> dict[tuple[int, int, int], np.ndarray]:
    if shear.shape != inplane.shape:
        raise ValueError("polarization sequences must have equal shapes")
    field: dict[tuple[int, int, int], np.ndarray] = {}
    for offset, (alpha, beta) in enumerate(zip(shear, inplane)):
        n = start + offset
        norm_squared = float(n * n + 1)
        coefficient = (
            alpha
            * np.asarray((0.0, 0.0, 1.0))
            / math.sqrt(norm_squared)
            + beta
            * np.asarray((-1.0, float(n), 0.0))
            / norm_squared
        )
        _add_real_mode(field, (n, 1, 0), coefficient)
    return field


def _low_field() -> dict[tuple[int, int, int], np.ndarray]:
    field: dict[tuple[int, int, int], np.ndarray] = {}
    _add_real_mode(
        field,
        (0, -1, 0),
        np.asarray((1.0, 0.0, 0.0)),
    )
    return field


def _compatible_load(
    flux: dict[tuple[int, int, int], np.ndarray],
) -> complex:
    return sum((_load(flux, vertex) for vertex in VERTICES), 0.0j)


def _weight_hat(wave: tuple[int, int, int]) -> complex:
    if wave[2] != 0 or abs(wave[0]) > 1 or abs(wave[1]) > 1:
        return 0.0j
    first = 0.5 if wave[0] == 0 else -0.25
    second = 0.5 if wave[1] == 0 else 0.25
    return complex(first * second)


def _direct_weighted_fisher(
    field: dict[tuple[int, int, int], np.ndarray],
) -> complex:
    value = 0.0j
    for first_wave, first_value in field.items():
        for second_wave, second_value in field.items():
            difference = tuple(
                second_wave[index] - first_wave[index]
                for index in range(3)
            )
            weight = _weight_hat(difference)
            if weight == 0.0:
                continue
            wave_dot = sum(
                first_wave[index] * second_wave[index]
                for index in range(3)
            )
            value += (
                wave_dot
                * weight
                * np.dot(first_value, np.conjugate(second_value))
            )
    return value


def _edge_fisher(
    left: complex,
    right: complex,
    gram: float,
) -> float:
    return float(
        abs(left) ** 2
        + abs(right) ** 2
        - 2.0 * gram * (right * np.conjugate(left)).real
    )


def _analytic_weighted_fisher(
    start: int,
    shear: np.ndarray,
    inplane: np.ndarray,
) -> float:
    shear_edges = abs(shear[0]) ** 2 + abs(shear[-1]) ** 2
    inplane_edges = (
        abs(inplane[0]) ** 2 + abs(inplane[-1]) ** 2
    )
    for offset in range(len(shear) - 1):
        geometry = _geometry(start + offset)
        shear_edges += _edge_fisher(
            shear[offset],
            shear[offset + 1],
            geometry["shear_Fisher_Gram"],
        )
        inplane_edges += _edge_fisher(
            inplane[offset],
            inplane[offset + 1],
            geometry["inplane_Fisher_Gram"],
        )
    return float(0.25 * (shear_edges + inplane_edges))


def _analytic_loads(
    start: int,
    shear: np.ndarray,
    inplane: np.ndarray,
) -> dict[str, float]:
    loads = {
        "kinetic": 0.0,
        "pressure_high_high": 0.0,
        "pressure_cross": 0.0,
        "combined": 0.0,
    }
    for offset in range(len(shear) - 1):
        geometry = _geometry(start + offset)
        shear_skew = (
            shear[offset + 1] * np.conjugate(shear[offset])
        ).imag
        inplane_skew = (
            inplane[offset + 1] * np.conjugate(inplane[offset])
        ).imag
        loads["kinetic"] += (
            geometry["shear_complete"] * shear_skew
            + (
                geometry["inplane_kinetic_scalar"]
                + geometry["inplane_kinetic_anisotropic"]
            )
            * inplane_skew
        )
        loads["pressure_high_high"] += (
            geometry["pressure_high_high"] * inplane_skew
        )
        loads["pressure_cross"] += (
            geometry["pressure_cross"] * inplane_skew
        )
        loads["combined"] += (
            geometry["shear_complete"] * shear_skew
            + geometry["inplane_complete"] * inplane_skew
        )
    return loads


def _maximum_divergence_residual(
    field: dict[tuple[int, int, int], np.ndarray],
) -> float:
    return max(
        (
            abs(np.dot(np.asarray(wave, dtype=float), coefficient))
            for wave, coefficient in field.items()
        ),
        default=0.0,
    )


def _symbolic_certificate() -> dict[str, Any]:
    n = sp.symbols("n", integer=True, positive=True)
    left = n**2 + 1
    right = n**2 + 2 * n + 2
    dot = n**2 + n + 1
    shear_gram = dot / sp.sqrt(left * right)
    shear_coefficient = -1 / (4 * sp.sqrt(left * right))

    sign = sp.symbols("sign", integer=True, nonzero=True)
    left_vector = sp.Matrix((-1, n))
    right_vector = sp.Matrix((-1, n + 1))
    first_denominator = (n + 1) ** 2 + (1 + sign) ** 2
    second_denominator = n**2 + (1 - sign) ** 2
    cross_vector = 2 * sign * (
        -((n + 1) ** 2)
        / (right * first_denominator)
        * (left_vector / left)
        + n**2
        / (left * second_denominator)
        * (right_vector / right)
    )
    cross_contraction = sp.Matrix((1, sign)).dot(cross_vector)
    cross_sum = sp.simplify(
        cross_contraction.subs(sign, 1)
        + cross_contraction.subs(sign, -1)
    )
    pressure_cross = sp.factor(-cross_sum / 8)
    kinetic_scalar = -dot / (4 * left * right)
    derived_complete = sp.factor(kinetic_scalar + pressure_cross)

    polynomial = (
        n**6
        + 3 * n**5
        + 14 * n**4
        + 23 * n**3
        + 35 * n**2
        + 24 * n
        + 4
    )
    claimed_complete = -polynomial / (
        4
        * left
        * (n**2 + 4)
        * right
        * (n**2 + 2 * n + 5)
    )
    inplane_gram = dot**2 / (left * right)
    residual = sp.factor(
        (1 - inplane_gram**2) / 16 - claimed_complete**2
    )
    numerator, denominator = sp.fraction(residual)
    coefficients = [
        int(value) for value in sp.Poly(numerator, n).all_coeffs()
    ]
    shear_identity = sp.simplify(
        shear_coefficient**2 - (1 - shear_gram**2) / 16
    )
    return {
        "inplane_complete_coefficient": str(
            sp.factor(claimed_complete)
        ),
        "derived_cross_pressure_coefficient": str(pressure_cross),
        "derived_complete_matches_claimed": (
            sp.simplify(derived_complete - claimed_complete) == 0
        ),
        "high_high_pressure_plus_anisotropic_kinetic": "0",
        "shear_coefficient_identity": (
            "C_s(n)^2=(1-gamma_s(n)^2)/16"
        ),
        "shear_identity_residual": str(shear_identity),
        "inplane_bound_residual": str(residual),
        "positive_numerator_coefficients_descending": coefficients,
        "positive_denominator": str(sp.factor(denominator)),
        "all_numerator_coefficients_positive": all(
            value > 0 for value in coefficients
        ),
        "all_checks_pass": bool(
            sp.simplify(derived_complete - claimed_complete) == 0
            and shear_identity == 0
            and all(value > 0 for value in coefficients)
        ),
    }


def _sequence(
    kind: str,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(length, dtype=float)
    if "dirichlet" in kind:
        envelope = np.sin(math.pi * (index + 1.0) / (length + 1.0))
    else:
        envelope = np.ones(length)
    if "tilted" in kind:
        shear = (
            0.5
            * envelope
            * np.exp(-0.5j * math.pi * index / 7.0)
        )
        inplane = (
            envelope * np.exp(1j * math.pi * index / 7.0)
        )
    else:
        shear = 0.5 * envelope.astype(np.complex128)
        inplane = envelope.astype(np.complex128)
    return shear, inplane


def _sparse_replay_rows() -> list[dict[str, Any]]:
    rows = []
    low = _low_field()
    for length in (4, 8, 16, 32):
        start = max(4, length)
        for kind in (
            "constant",
            "first_dirichlet",
            "tilted_constant",
            "tilted_first_dirichlet",
        ):
            shear, inplane = _sequence(kind, length)
            high = _chain_field(start, shear, inplane)
            components = _component_fluxes(high, low)
            direct = _direct_linear_flux(high, low)
            sparse_loads = {
                key: _compatible_load(value)
                for key, value in components.items()
            }
            analytic_loads = _analytic_loads(
                start, shear, inplane
            )
            direct_fisher = _direct_weighted_fisher(high)
            analytic_fisher = _analytic_weighted_fisher(
                start, shear, inplane
            )
            pressure = _pressure_bilinear(high, high)
            pressure_coefficient = pressure.get((1, 0, 0), 0.0j)
            maximum_load_residual = max(
                abs(sparse_loads[key].real - analytic_loads[key])
                for key in analytic_loads
            )
            maximum_imaginary_residual = max(
                abs(value.imag) for value in sparse_loads.values()
            )
            combined = sparse_loads["combined"].real
            ratio = abs(combined) / analytic_fisher
            pressure_active = bool(
                abs(sparse_loads["pressure_high_high"].real) > 1.0e-14
            )
            expected_active = kind.startswith("tilted")
            rows.append(
                {
                    "kind": kind,
                    "start_mode": start,
                    "chain_length": length,
                    "weighted_Fisher": analytic_fisher,
                    "direct_weighted_Fisher": direct_fisher.real,
                    "direct_weighted_Fisher_imaginary_residual": abs(
                        direct_fisher.imag
                    ),
                    "analytic_loads": analytic_loads,
                    "sparse_loads": {
                        key: value.real
                        for key, value in sparse_loads.items()
                    },
                    "pressure_coefficient_at_e1": [
                        pressure_coefficient.real,
                        pressure_coefficient.imag,
                    ],
                    "pressure_load_active": pressure_active,
                    "expected_pressure_load_active": expected_active,
                    "complete_load_over_weighted_Fisher": ratio,
                    "maximum_component_load_residual": (
                        maximum_load_residual
                    ),
                    "direct_flux_coefficient_residual": (
                        _maximum_vector_difference(
                            components["combined"], direct
                        )
                    ),
                    "maximum_imaginary_load_residual": (
                        maximum_imaginary_residual
                    ),
                    "maximum_divergence_residual": (
                        _maximum_divergence_residual(high)
                    ),
                    "all_checks_pass": bool(
                        maximum_load_residual < 2.0e-12
                        and abs(direct_fisher.real - analytic_fisher)
                        < 2.0e-12
                        and abs(direct_fisher.imag) < 2.0e-12
                        and _maximum_vector_difference(
                            components["combined"], direct
                        )
                        < 2.0e-12
                        and maximum_imaginary_residual < 2.0e-12
                        and _maximum_divergence_residual(high) < 2.0e-12
                        and ratio <= 0.5 + 2.0e-12
                        and pressure_active == expected_active
                    ),
                }
            )
    return rows


def _form_matrices(
    start: int,
    length: int,
    polarization: str,
) -> tuple[np.ndarray, np.ndarray]:
    fisher = np.eye(length, dtype=float) * 0.5
    load = np.zeros((length, length), dtype=np.complex128)
    for offset in range(length - 1):
        geometry = _geometry(start + offset)
        if polarization == "shear":
            gram = geometry["shear_Fisher_Gram"]
            coefficient = geometry["shear_complete"]
        elif polarization == "inplane":
            gram = geometry["inplane_Fisher_Gram"]
            coefficient = geometry["inplane_complete"]
        else:
            raise ValueError(f"unknown polarization {polarization}")
        fisher[offset, offset + 1] = -gram / 4.0
        fisher[offset + 1, offset] = -gram / 4.0
        load[offset, offset + 1] = -0.5j * coefficient
        load[offset + 1, offset] = 0.5j * coefficient
    return fisher, load


def _generalized_spectral_rows() -> list[dict[str, Any]]:
    rows = []
    for start in (1, 4, 16):
        for length in (8, 16, 32, 64, 128):
            row: dict[str, Any] = {
                "start_mode": start,
                "chain_length": length,
            }
            maxima = []
            for polarization in ("shear", "inplane"):
                fisher, load = _form_matrices(
                    start, length, polarization
                )
                inverse_cholesky = np.linalg.inv(
                    np.linalg.cholesky(fisher)
                )
                normalized = (
                    inverse_cholesky
                    @ load
                    @ np.conjugate(inverse_cholesky.T)
                )
                eigenvalues = np.linalg.eigvalsh(normalized)
                maximum = float(np.max(np.abs(eigenvalues)))
                row[f"{polarization}_maximum_absolute_eigenvalue"] = (
                    maximum
                )
                maxima.append(maximum)
            row["two_polarization_maximum"] = max(maxima)
            row["certified_upper_bound"] = 0.5
            row["all_checks_pass"] = max(maxima) < 0.5
            rows.append(row)
    return rows


def _mandatory_replays() -> dict[str, Any]:
    finite_fields = _finite_field_recombination_audit()
    cross_shell = cross_shell_audit()
    cross_replay = cross_shell["finite_mode_asymptotic_replay"]
    return {
        "Taylor_Green": finite_fields["Taylor_Green"],
        "seed81": finite_fields["seed81"],
        "modulated_wave_HHL": {
            "status": cross_shell["status"],
            "all_positive_checks_pass": cross_shell[
                "all_positive_checks_pass"
            ],
            "analytic_limit": cross_replay[
                "analytic_combined_HHL_load_limit"
            ],
            "last_carrier": cross_replay["carriers"][-1],
            "last_complete_over_limit": cross_replay[
                "combined_last_over_limit"
            ],
            "component_reconstruction_passes": all(
                row["component_vs_direct_flux_residual"] < 3.0e-15
                for row in cross_replay["rows"]
            ),
        },
    }


def audit() -> dict[str, Any]:
    prerequisites = _prerequisite_audit()
    symbolic = _symbolic_certificate()
    sparse_rows = _sparse_replay_rows()
    spectral_rows = _generalized_spectral_rows()
    adversaries = _mandatory_replays()
    pressure_active_rows = [
        row for row in sparse_rows if row["expected_pressure_load_active"]
    ]
    null_rows = [
        row for row in sparse_rows if not row["expected_pressure_load_active"]
    ]
    positive_checks = {
        "prerequisite_hashes_and_results_pass": prerequisites[
            "all_checks_pass"
        ],
        "symbolic_coefficient_and_positive_polynomial_pass": symbolic[
            "all_checks_pass"
        ],
        "sparse_complete_symbol_reconstruction_passes": all(
            row["all_checks_pass"] for row in sparse_rows
        ),
        "constant_and_Dirichlet_null_rows_vanish": all(
            abs(row["sparse_loads"]["combined"]) < 2.0e-12
            and abs(row["sparse_loads"]["pressure_high_high"])
            < 2.0e-12
            for row in null_rows
        ),
        "phase_tilt_activates_pressure_load": all(
            row["pressure_load_active"] for row in pressure_active_rows
        ),
        "every_sparse_row_obeys_half_Fisher_bound": all(
            row["complete_load_over_weighted_Fisher"]
            <= 0.5 + 2.0e-12
            for row in sparse_rows
        ),
        "generalized_chain_spectra_obey_half_Fisher_bound": all(
            row["all_checks_pass"] for row in spectral_rows
        ),
        "Taylor_Green_replay_passes": adversaries["Taylor_Green"][
            "all_checks_pass"
        ],
        "seed81_replay_passes": adversaries["seed81"][
            "all_checks_pass"
        ],
        "modulated_wave_HHL_no_go_remains_live": adversaries[
            "modulated_wave_HHL"
        ]["all_positive_checks_pass"],
    }
    all_positive = all(positive_checks.values())
    return {
        "kind": "pressure_active_fisher_null_compatibility_gate_audit",
        "schema_version": 1,
        "status": (
            "canonical_pressure_active_chain_Fisher_compatibility_proved"
            if all_positive
            else "audit_failed"
        ),
        "all_positive_checks_pass": all_positive,
        "positive_checks": positive_checks,
        "prerequisites": prerequisites,
        "canonical_geometry": {
            "high_frequencies": "k_n=(n,1,0), n=N_0,...,N_1",
            "shear_polarization": (
                "uhat_s(k_n)=alpha_n e_3/sqrt(n^2+1)"
            ),
            "inplane_polarization": (
                "uhat_t(k_n)=beta_n(-1,n,0)/(n^2+1)"
            ),
            "reality": "uhat(-k_n)=conjugate(uhat(k_n))",
            "low_wave": (
                "Uhat(0,-1,0)=Uhat(0,1,0)=e_1"
            ),
            "compatible_weight": (
                "lambda=phi_-(x_1)phi_+(x_2), obtained by summing "
                "the two vertices (-,+,-) and (-,+,+)"
            ),
            "weighted_Fisher": (
                "E_lambda(h)=(1/4)sum_edges "
                "[F_gamma_s(alpha)+F_gamma_t(beta)], including both "
                "zero endpoints"
            ),
            "edge_form": (
                "F_gamma(a,b)=|a|^2+|b|^2"
                "-2 gamma Re(b conjugate(a))"
            ),
        },
        "exact_edge_theorem": {
            "skew_lemma": (
                "2 sqrt(1-gamma^2)|Im(b conjugate(a))|"
                "<=F_gamma(a,b)"
            ),
            "complete_HHL_form": (
                "B_HHL=sum_n C_s(n) Im(alpha_(n+1) conjugate(alpha_n))"
                "+C_t(n) Im(beta_(n+1) conjugate(beta_n))"
            ),
            "shear_coefficient": (
                "C_s(n)=-1/[4 sqrt((n^2+1)((n+1)^2+1))]"
            ),
            "inplane_coefficient": symbolic[
                "inplane_complete_coefficient"
            ],
            "coefficient_bounds": (
                "|C_sigma(n)|<=(1/4)sqrt(1-gamma_sigma(n)^2), "
                "sigma in {s,t}"
            ),
            "edge_bound": (
                "|C_sigma Im(b conjugate(a))|<=F_gamma(a,b)/8"
            ),
            "global_bound": "|B_HHL|<=E_lambda(h)/2",
            "high_high_pressure_cancellation": (
                "The +sin(theta_n)^2/2 high-high pressure coefficient "
                "cancels the -sin(theta_n)^2/2 anisotropic kinetic "
                "coefficient exactly before the cross-pressure term is "
                "added."
            ),
            "null_compatibility": (
                "Real constant and first Dirichlet chain data have zero "
                "skew product and hence zero complete HHL load. Complex "
                "phase tilts activate pressure but are paid by the same "
                "full signed Fisher edges."
            ),
        },
        "symbolic_certificate": symbolic,
        "sparse_symbol_replays": sparse_rows,
        "generalized_spectral_replays": spectral_rows,
        "mandatory_adversary_replays": adversaries,
        "certification_flags": {
            "canonical_two_polarization_complete_HHL_Fisher_bound_proved": (
                True
            ),
            "canonical_bound_constant": 0.5,
            "constant_chain_null_compatible": True,
            "first_Dirichlet_chain_null_compatible": True,
            "pressure_active_phase_tilts_controlled": True,
            "full_signed_Fisher_interfaces_retained": True,
            "multiband_absolute_Fisher_recombination_restored": False,
            "arbitrary_residue_chain_bound_proved": False,
            "arbitrary_low_field_and_partition_bound_proved": False,
            "cross_shell_modulated_wave_no_go_invalidated": False,
            "all_cross_shell_HHL_absorbed": False,
            "terminal_dual_supremum_controlled": False,
            "critical_L3_controlled": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "scope": (
            "This is an exact compatibility theorem for one canonical "
            "pressure-active affine residue chain, both of its "
            "divergence-free polarizations, one fixed low wave, and one "
            "compatible two-vertex weight. It proves that the Fisher "
            "near-null does not obstruct this geometry. It does not yet "
            "cover arbitrary transverse residues, low fields, partition "
            "phases, cross-residue couplings, or the full multiband HHL "
            "operator."
        ),
        "next_theorem_target": (
            "Lift the edge calculation to arbitrary primitive partition "
            "steps and transverse residues, then assemble the finite "
            "low-wave/vertex matrix. The required statement is a uniform "
            "block Schur bound against the unsplit physical Fisher graph; "
            "the modulated-wave HHL adversary must remain an explicit "
            "block in that matrix."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise SystemExit("pressure-active Fisher compatibility audit failed")
    if not arguments.check_only:
        _atomic_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
