"""Audit surjectivity of the finite-Fourier pressure-load map."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from pressure_frame_pairing_audit import (
    GRID_SIZE,
    STARTING_GRID_INDEX,
    _build_spectral_fields,
)
from scale_adapted_edge_rho_expansion_audit import COEFFICIENTS


ROOT = Path(__file__).resolve().parents[3]
Wave = tuple[int, int, int]
Field = dict[Wave, np.ndarray]
SIGNS = (-1, 1)
VERTICES = tuple(itertools.product(SIGNS, repeat=3))
STENCIL = tuple(
    wave
    for wave in itertools.product((-1, 0, 1), repeat=3)
    if wave != (0, 0, 0)
)
SUBSET_WAVES = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
BLOCK_SCALES = tuple(8 * 4**index for index in range(7))


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


def _add(first: Wave, second: Wave) -> Wave:
    return tuple(
        first[index] + second[index] for index in range(3)
    )


def _negate(wave: Wave) -> Wave:
    return tuple(-value for value in wave)


def _dot(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    return sum(
        first[index] * second[index] for index in range(len(first))
    )


def _fraction(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _pressure_modes(field: Field) -> dict[Wave, complex]:
    pressure: dict[Wave, complex] = {}
    for first_wave, first_value in field.items():
        for second_wave, second_value in field.items():
            output = _add(first_wave, second_wave)
            output_squared = _dot(output, output)
            if output_squared == 0:
                continue
            contribution = -(
                np.dot(output, first_value)
                * np.dot(output, second_value)
                / output_squared
            )
            pressure[output] = pressure.get(output, 0j) + contribution
    return pressure


def _low_transport_modes(
    field: Field,
) -> tuple[dict[Wave, complex], dict[Wave, complex]]:
    pressure = _pressure_modes(field)
    transport = {wave: 0j for wave in STENCIL}
    for output in STENCIL:
        for velocity_wave, velocity_value in field.items():
            pressure_wave = tuple(
                output[index] - velocity_wave[index]
                for index in range(3)
            )
            pressure_value = pressure.get(pressure_wave)
            if pressure_value is None:
                continue
            transport[output] += (
                -1j
                * np.dot(pressure_wave, velocity_value)
                * pressure_value
            )
    return pressure, transport


def _partition_coefficient(
    vertex: Wave,
    wave: Wave,
    center: np.ndarray,
) -> complex:
    value = 1.0 + 0.0j
    for direction, frequency in enumerate(wave):
        if frequency == 0:
            value *= 0.5
        elif frequency == 1:
            value *= (
                vertex[direction]
                * np.exp(-1j * center[direction])
                / 4.0
            )
        elif frequency == -1:
            value *= (
                vertex[direction]
                * np.exp(1j * center[direction])
                / 4.0
            )
        else:
            return 0j
    return value


def _loads_from_transport(
    transport: dict[Wave, complex],
    center: np.ndarray | None = None,
) -> list[float]:
    if center is None:
        center = np.zeros(3)
    loads = []
    for vertex in VERTICES:
        value = 0j
        for wave in STENCIL:
            value += (
                _partition_coefficient(vertex, wave, center)
                * transport[_negate(wave)]
            )
        loads.append(float(value.real))
    return loads


def _walsh_loads(
    moments: dict[Wave, Fraction],
) -> list[Fraction]:
    loads = []
    for vertex in VERTICES:
        value = Fraction(0)
        for subset_wave, moment in moments.items():
            character = 1
            for direction, occupied in enumerate(subset_wave):
                if occupied:
                    character *= vertex[direction]
            value += Fraction(character, 8) * moment
        loads.append(value)
    return loads


def _map_identity_audit() -> dict[str, Any]:
    target_moments = {
        wave: (
            Fraction(27, 32)
            if sum(wave) == 1
            else Fraction(9, 8)
        )
        for wave in SUBSET_WAVES
    }
    loads = _walsh_loads(target_moments)
    expected = [
        Fraction((225, -45, -27, -9)[sum(value == -1 for value in vertex)], 256)
        for vertex in VERTICES
    ]
    return {
        "Fourier_convention": (
            "u(x)=sum_k u_hat(k)e^(ik.x), "
            "u_hat(-k)=conj(u_hat(k)), k.u_hat(k)=0"
        ),
        "pressure_map": (
            "p_hat(q)=-|q|^(-2) sum_(a+b=q)"
            "(q.u_hat(a))(q.u_hat(b)), q!=0"
        ),
        "transport_scalar": "g=-u.grad p",
        "transport_map": (
            "g_hat(n)=-i sum_(q+k=n)"
            "(q.u_hat(k))p_hat(q)"
        ),
        "partition_identity": (
            "Phi_v=1/8 sum_(S subset {1,2,3}) "
            "chi_S(v) product_(j in S)cos(x_j)"
        ),
        "Walsh_moments": (
            "G_S=mean[g product_(j in S)cos(x_j)]"
        ),
        "load_map": (
            "b_v=mean[Phi_v g]=1/8 sum_(S nonempty)chi_S(v)G_S"
        ),
        "load_conservation": "sum_v b_v=0",
        "target_Walsh_moments": {
            str(wave): _fraction(value)
            for wave, value in target_moments.items()
        },
        "target_loads": [
            _fraction(value) for value in loads
        ],
        "expected_Hamming_loads": [
            _fraction(value) for value in expected
        ],
        "all_checks_pass": (
            loads == expected
            and sum(loads, Fraction(0)) == 0
        ),
    }


def _block_specification(
    subset_wave: Wave,
    scale: int,
) -> dict[str, Any]:
    first_wave = (scale, 0, 0)
    second_wave = (0, 2 * scale, 0)
    pair_wave = _add(first_wave, second_wave)
    third_wave = tuple(
        subset_wave[index] - pair_wave[index]
        for index in range(3)
    )
    first_polarization = (0, 1, 0)
    second_polarization = (1, 0, 0)
    third_squared = _dot(third_wave, third_wave)
    third_pairing = _dot(third_wave, pair_wave)
    third_polarization = tuple(
        third_squared * pair_wave[index]
        - third_pairing * third_wave[index]
        for index in range(3)
    )
    pair_sums = (
        _add(first_wave, second_wave),
        _add(first_wave, third_wave),
        _add(second_wave, third_wave),
    )
    coupling_sum = Fraction(0)
    for pair_sum in pair_sums:
        coupling_sum += Fraction(
            _dot(pair_sum, first_polarization)
            * _dot(pair_sum, second_polarization)
            * _dot(pair_sum, third_polarization),
            _dot(pair_sum, pair_sum),
        )
    coupling = -2 * coupling_sum
    return {
        "subset_wave": subset_wave,
        "scale": scale,
        "first_wave": first_wave,
        "second_wave": second_wave,
        "third_wave": third_wave,
        "first_polarization": first_polarization,
        "second_polarization": second_polarization,
        "third_polarization": third_polarization,
        "coupling": coupling,
        "divergence_residuals": (
            _dot(first_wave, first_polarization),
            _dot(second_wave, second_polarization),
            _dot(third_wave, third_polarization),
        ),
        "pair_output": tuple(
            first_wave[index]
            + second_wave[index]
            + third_wave[index]
            for index in range(3)
        ),
    }


def _support_certificate(
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    modes: list[Wave] = []
    owner: dict[Wave, tuple[int, str, int]] = {}
    for block_index, block in enumerate(blocks):
        for label in ("first_wave", "second_wave", "third_wave"):
            wave = block[label]
            for sign in SIGNS:
                signed_wave = tuple(sign * value for value in wave)
                modes.append(signed_wave)
                owner[signed_wave] = (block_index, label, sign)

    low_triples = []
    invalid_triples = []
    stencil = set(STENCIL)
    for indices in itertools.combinations_with_replacement(
        range(len(modes)),
        3,
    ):
        selected = tuple(modes[index] for index in indices)
        output = tuple(
            sum(wave[direction] for wave in selected)
            for direction in range(3)
        )
        if output not in stencil:
            continue
        owners = [owner[wave] for wave in selected]
        same_block = len({value[0] for value in owners}) == 1
        labels = sorted(value[1] for value in owners)
        same_sign = len({value[2] for value in owners}) == 1
        valid = (
            same_block
            and labels
            == ["first_wave", "second_wave", "third_wave"]
            and same_sign
        )
        row = {
            "modes": [list(wave) for wave in selected],
            "output": list(output),
            "owners": [list(value) for value in owners],
            "valid_isolated_block_triple": valid,
        }
        low_triples.append(row)
        if not valid:
            invalid_triples.append(row)

    expected_outputs = {
        wave for block in blocks for wave in (
            block["subset_wave"],
            _negate(block["subset_wave"]),
        )
    }
    actual_outputs = {
        tuple(row["output"]) for row in low_triples
    }
    return {
        "signed_mode_count": len(modes),
        "signed_modes_are_distinct": len(set(modes)) == len(modes),
        "unordered_low_triple_count": len(low_triples),
        "expected_low_triple_count": 14,
        "invalid_low_triple_count": len(invalid_triples),
        "low_outputs": [
            list(wave) for wave in sorted(actual_outputs)
        ],
        "expected_low_outputs": [
            list(wave) for wave in sorted(expected_outputs)
        ],
        "low_triples": low_triples,
        "interpretation": (
            "Every triple reaching the partition stencil is a permutation "
            "of the three positive modes of one block or all three "
            "negative modes. Cross-block cubic terms are spectrally silent "
            "on all seven Walsh load coordinates."
        ),
        "all_checks_pass": (
            len(set(modes)) == len(modes)
            and len(low_triples) == 14
            and not invalid_triples
            and actual_outputs == expected_outputs
        ),
    }


def _surjectivity_construction_audit() -> dict[str, Any]:
    target_moments = {
        wave: (
            Fraction(27, 32)
            if sum(wave) == 1
            else Fraction(9, 8)
        )
        for wave in SUBSET_WAVES
    }
    blocks = [
        _block_specification(wave, scale)
        for wave, scale in zip(SUBSET_WAVES, BLOCK_SCALES)
    ]
    support = _support_certificate(blocks)

    field: Field = {}
    block_rows = []
    for block in blocks:
        subset_wave = block["subset_wave"]
        subset_size = sum(subset_wave)
        target_transport_mode = (
            target_moments[subset_wave] * 2 ** (subset_size - 1)
        )
        coupling = block["coupling"]
        phase_sign = 1 if coupling > 0 else -1
        amplitude_cubed = target_transport_mode / abs(coupling)
        amplitude = float(amplitude_cubed) ** (1.0 / 3.0)
        entries = (
            (
                block["first_wave"],
                block["first_polarization"],
                1.0 + 0.0j,
            ),
            (
                block["second_wave"],
                block["second_polarization"],
                1.0 + 0.0j,
            ),
            (
                block["third_wave"],
                block["third_polarization"],
                1j * phase_sign,
            ),
        )
        for wave, polarization, phase in entries:
            value = (
                amplitude
                * phase
                * np.asarray(polarization, dtype=np.complex128)
            )
            field[wave] = value
            field[_negate(wave)] = value.conjugate()
        block_rows.append(
            {
                "subset_wave": list(subset_wave),
                "scale": block["scale"],
                "waves": [
                    list(block["first_wave"]),
                    list(block["second_wave"]),
                    list(block["third_wave"]),
                ],
                "polarizations": [
                    list(block["first_polarization"]),
                    list(block["second_polarization"]),
                    list(block["third_polarization"]),
                ],
                "divergence_residuals": list(
                    block["divergence_residuals"]
                ),
                "triple_output": list(block["pair_output"]),
                "exact_unit_coupling": _fraction(coupling),
                "phase_sign": phase_sign,
                "target_transport_mode": _fraction(
                    target_transport_mode
                ),
                "exact_amplitude_cubed": _fraction(amplitude_cubed),
                "amplitude": amplitude,
            }
        )

    pressure, transport = _low_transport_modes(field)
    target_transport = {}
    for subset_wave, moment in target_moments.items():
        coefficient = float(
            moment * 2 ** (sum(subset_wave) - 1)
        )
        target_transport[subset_wave] = coefficient
        target_transport[_negate(subset_wave)] = coefficient
    target_mode_residual = max(
        abs(transport[wave] - target_transport.get(wave, 0.0))
        for wave in STENCIL
    )
    undesired_modes = [
        wave
        for wave in STENCIL
        if wave not in target_transport
    ]
    maximum_undesired_mode = max(
        abs(transport[wave]) for wave in undesired_modes
    )

    loads = _loads_from_transport(transport)
    expected_loads = [
        (225, -45, -27, -9)[
            sum(value == -1 for value in vertex)
        ]
        / 256.0
        for vertex in VERTICES
    ]
    load_residual = max(
        abs(value - expected)
        for value, expected in zip(loads, expected_loads)
    )
    divergence_residual = max(
        abs(np.dot(wave, value)) for wave, value in field.items()
    )
    relative_divergence_residual = max(
        abs(np.dot(wave, value))
        / (
            np.linalg.norm(np.asarray(wave, dtype=float))
            * np.linalg.norm(value)
        )
        for wave, value in field.items()
    )
    reality_residual = max(
        float(np.max(np.abs(field[_negate(wave)] - value.conjugate())))
        for wave, value in field.items()
    )
    energy = sum(float(np.vdot(value, value).real) for value in field.values())
    enstrophy = sum(
        _dot(wave, wave) * float(np.vdot(value, value).real)
        for wave, value in field.items()
    )

    return {
        "construction": (
            "Seven lacunary real divergence-free three-mode blocks, one "
            "for each nonempty coordinate subset. The third polarization "
            "is |k|^2 q-(k.q)k and therefore exactly perpendicular to k."
        ),
        "block_scales": list(BLOCK_SCALES),
        "block_rows": block_rows,
        "support_certificate": support,
        "combined_velocity_mode_count": len(field),
        "combined_pressure_mode_count": len(pressure),
        "maximum_divergence_residual": float(divergence_residual),
        "maximum_relative_divergence_residual": float(
            relative_divergence_residual
        ),
        "maximum_reality_residual": reality_residual,
        "maximum_target_transport_mode_residual": float(
            target_mode_residual
        ),
        "maximum_undesired_stencil_mode": float(
            maximum_undesired_mode
        ),
        "realized_loads": loads,
        "expected_saturating_loads": expected_loads,
        "maximum_load_residual": load_residual,
        "velocity_L2_energy": energy,
        "velocity_H1_seminorm_squared": enstrophy,
        "surjectivity_argument": (
            "For arbitrary real Walsh data G_S, set the S-block amplitude "
            "cube to 2^(|S|-1)|G_S|/|K_S| and choose the sign of its "
            "imaginary phase to match sign(G_S/K_S). Every K_S is nonzero "
            "and the support certificate removes all cross-block low "
            "interactions. Thus every seven-vector G, equivalently every "
            "zero-sum eight-load vector b, is realized."
        ),
        "scope": (
            "This realizes the compatible load vector of the abstract "
            "vertex saturator using an actual smooth divergence-free "
            "trigonometric polynomial and its Poisson pressure. It does "
            "not realize the abstract pointwise conditional edge functions "
            "or make their L^(3/2) envelope sharp for this velocity."
        ),
        "all_checks_pass": bool(
            support["all_checks_pass"]
            and all(block["coupling"] != 0 for block in blocks)
            and all(
                residual == 0
                for block in blocks
                for residual in block["divergence_residuals"]
            )
            and len(field) == 42
            and relative_divergence_residual < 1.0e-14
            and reality_residual < 1.0e-15
            and target_mode_residual < 1.0e-11
            and maximum_undesired_mode < 1.0e-12
            and load_residual < 1.0e-11
        ),
    }


def _taylor_green_sparse_audit() -> dict[str, Any]:
    field: Field = {}
    for first_sign, second_sign in itertools.product(SIGNS, repeat=2):
        wave = (first_sign, second_sign, 0)
        field[wave] = np.asarray(
            (
                -1j * first_sign / 4.0,
                1j * second_sign / 4.0,
                0.0,
            ),
            dtype=np.complex128,
        )
    _, transport = _low_transport_modes(field)
    loads = _loads_from_transport(transport)
    return {
        "velocity_modes": {
            str(wave): [
                str(value) for value in coefficient
            ]
            for wave, coefficient in field.items()
        },
        "maximum_stencil_transport_mode": max(
            abs(value) for value in transport.values()
        ),
        "compatible_loads": loads,
        "maximum_compatible_load": max(abs(value) for value in loads),
        "interpretation": (
            "The sparse Fourier map independently recovers the prior "
            "Taylor-Green annihilation: all partition-stencil modes of "
            "g=-u.grad p vanish."
        ),
        "all_checks_pass": bool(
            max(abs(value) for value in transport.values()) < 1.0e-15
            and max(abs(value) for value in loads) < 1.0e-15
        ),
    }


def _seed81_sparse_benchmark() -> dict[str, Any]:
    fields = _build_spectral_fields()
    modes, coefficients = fields["velocity"]
    field = {
        tuple(int(value) for value in mode): np.asarray(
            coefficient,
            dtype=np.complex128,
        )
        for mode, coefficient in zip(modes, coefficients)
    }
    pressure, transport = _low_transport_modes(field)
    center = (
        2.0 * math.pi * STARTING_GRID_INDEX.astype(float) / GRID_SIZE
    )
    loads = _loads_from_transport(transport, center)

    weighted_pressure = 0.0
    for coefficient, bits in zip(
        COEFFICIENTS,
        itertools.product((0, 1), repeat=3),
    ):
        vertex = tuple(1 if bit == 0 else -1 for bit in bits)
        weighted_pressure += (
            float(coefficient) * loads[VERTICES.index(vertex)]
        )
    stored = json.loads(
        (
            ROOT
            / "work/ns_collision/results/"
            "scale_adapted_edge_rho_expansion_audit_v1.json"
        ).read_text(encoding="utf-8")
    )
    stored_pressure = stored["partition_frequency_sweep"]["rows"][0][
        "direct_pressure"
    ]
    target = np.asarray(
        [
            (225, -45, -27, -9)[
                sum(value == -1 for value in vertex)
            ]
            / 256.0
            for vertex in VERTICES
        ]
    )
    load_array = np.asarray(loads)
    cosine = float(
        np.dot(load_array, target)
        / (np.linalg.norm(load_array) * np.linalg.norm(target))
    )
    weighted_pressure = float(weighted_pressure)
    stored_pressure = float(stored_pressure)
    return {
        "source": "seed-81 finite-Fourier pressure adversary",
        "velocity_mode_count": len(field),
        "pressure_mode_count": len(pressure),
        "partition_center": center.tolist(),
        "compatible_loads": loads,
        "load_sum": float(sum(loads)),
        "weighted_pressure_from_sparse_loads": weighted_pressure,
        "stored_direct_pressure": stored_pressure,
        "pressure_residual": weighted_pressure - stored_pressure,
        "cosine_with_positive_vertex_saturating_ray": cosine,
        "interpretation": (
            "The independent sparse convolution matches the existing "
            "grid pressure work. Seed-81 is a generic nonzero load "
            "benchmark, not the constructed saturating ray."
        ),
        "all_checks_pass": bool(
            len(field) == 116
            and abs(sum(loads)) < 1.0e-12
            and abs(weighted_pressure - stored_pressure) < 1.0e-11
        ),
    }


def audit() -> dict[str, Any]:
    identity = _map_identity_audit()
    construction = _surjectivity_construction_audit()
    taylor_green = _taylor_green_sparse_audit()
    seed81 = _seed81_sparse_benchmark()
    positive_checks = {
        "Fourier_pressure_load_identity_passes": identity[
            "all_checks_pass"
        ],
        "seven_block_surjectivity_certificate_passes": construction[
            "all_checks_pass"
        ],
        "Taylor_Green_sparse_annihilation_passes": taylor_green[
            "all_checks_pass"
        ],
        "seed81_sparse_pressure_benchmark_passes": seed81[
            "all_checks_pass"
        ],
    }
    return {
        "kind": "fourier_pressure_load_surjectivity_audit",
        "schema_version": 1,
        "status": (
            "instantaneous_zero_sum_load_surjectivity_certified_"
            "quantitative_multiscale_gate_open"
        ),
        "assumption_scope": (
            "Smooth real periodic divergence-free trigonometric "
            "polynomials at one instant, with pressure fixed by the "
            "Navier-Stokes Poisson equation and partition frequency one. "
            "Spatial rescaling gives the corresponding construction at "
            "integer partition frequencies."
        ),
        "exact_Fourier_and_Walsh_map": identity,
        "lacunary_surjectivity_construction": construction,
        "taylor_green_sparse_check": taylor_green,
        "seed81_sparse_benchmark": seed81,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "Fourier_velocity_to_pressure_to_load_map_derived": True,
            "finite_support_cross_block_isolation_certified": True,
            "instantaneous_zero_sum_load_space_surjectivity_proved": True,
            "vertex_saturating_Hamming_load_ray_PDE_realized": True,
            "Taylor_Green_zero_load_reconfirmed": True,
            "seed81_sparse_map_cross_validated": True,
            "abstract_pointwise_edge_saturator_PDE_realized": False,
            "uniform_quantitative_load_bound_proved": False,
            "pressure_L32_remainder_absorbed": False,
            "critical_signed_bound_proved": False,
            "low_regularity_passage_proved": False,
            "exceptional_set_upgrade_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "route_decision": (
            "Abandon any route that seeks an instantaneous algebraic "
            "exclusion of bad compatible load directions from "
            "incompressibility and the pressure Poisson law alone: the "
            "finite-Fourier map is onto. Retain the exact graph projection, "
            "but seek a quantitative inequality that charges the velocity "
            "norm, frequency separation, intrinsic scale, time evolution, "
            "or cross-level Carleson coherence required to realize a load."
        ),
        "next_theorem_target": (
            "Optimize the scale-invariant cost of realizing a prescribed "
            "load vector. Define the least velocity energy/enstrophy or "
            "critical L3 cost among divergence-free fields with b fixed, "
            "derive its scaling law, and compare the saturating Hamming ray "
            "with Taylor-Green and seed-81. A useful theorem must remain "
            "coercive under frequency translation and spatial rescaling; "
            "mere finite-dimensional surjectivity is now settled."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "work/ns_collision/results/"
            "fourier_pressure_load_surjectivity_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("Fourier pressure-load surjectivity audit failed")
    _atomic_json(arguments.output, result)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": _sha256(arguments.output),
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
