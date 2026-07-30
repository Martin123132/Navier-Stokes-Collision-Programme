"""Compute the fixed-domain annular second-jet continuum functional."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from scipy.fft import fftn, ifftn, next_fast_len

from separable_annular_pressure_schur_no_go_audit import _family_arrays


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_continuum_convolution_quadrature_v1.json"
)
PREREQUISITE = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_fixed_output_continuum_gate_audit_v1.json"
)
PREREQUISITE_SHA256 = (
    "6b29ef28146f86d87ba4eeb22de596083d8b18fa451394b5f3ade69b1353d072"
)
ALGORITHM_REVISION = "annular-rho-zero-continuum-convolution-quadrature-v1"
DEFAULT_SIZES = (9, 13, 17, 21, 25, 29, 33, 37, 41)
Array = np.ndarray


def _lower_process_priority() -> None:
    if os.name != "nt":
        return
    below_normal_priority_class = 0x00004000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    kernel32.SetPriorityClass(handle, below_normal_priority_class)


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


def _grid_shape(size: int) -> tuple[int, int, int]:
    maxima = (
        3 * size - 1,
        max((size - 1) // 2, 1),
        max((size - 1) // 2, 1),
    )
    return tuple(
        next_fast_len(4 * maximum + 1) for maximum in maxima
    )


def _frequency_axes(
    shape: tuple[int, int, int],
) -> tuple[Array, Array, Array]:
    axes = []
    for axis, length in enumerate(shape):
        reshape = [1, 1, 1]
        reshape[axis] = length
        axes.append(
            (np.fft.fftfreq(length) * length).reshape(reshape)
        )
    return tuple(axes)  # type: ignore[return-value]


def _physical(coefficients: Array, volume: int) -> Array:
    return ifftn(coefficients, workers=1).real * volume


def _coefficients(values: Array, volume: int) -> Array:
    return fftn(values, workers=1) / volume


def _high_coefficients(
    size: int,
    shape: tuple[int, int, int],
) -> Array:
    waves, velocity, _ = _family_arrays(
        (size, size, size),
        2 * size,
    )
    coefficients = np.zeros((3, *shape), dtype=np.complex128)
    shape_array = np.asarray(shape, dtype=int)
    for wave, value in zip(
        waves.reshape(-1, 3).astype(int),
        velocity.reshape(-1, 3),
    ):
        positive = tuple((wave % shape_array).tolist())
        negative = tuple(((-wave) % shape_array).tolist())
        coefficients[(slice(None), *positive)] = value
        coefficients[(slice(None), *negative)] = value
    return coefficients


def _project_negative(
    coefficients: Array,
    frequencies: tuple[Array, Array, Array],
    safe_wave_number_squared: Array,
) -> Array:
    divergence = sum(
        frequencies[axis] * coefficients[axis] for axis in range(3)
    )
    for component in range(3):
        coefficients[component] = (
            -coefficients[component]
            + frequencies[component]
            * divergence
            / safe_wave_number_squared
        )
    coefficients[(slice(None), 0, 0, 0)] = 0.0
    return coefficients


def _divergence_residual(
    coefficients: Array,
    frequencies: tuple[Array, Array, Array],
) -> float:
    divergence = sum(
        frequencies[axis] * coefficients[axis] for axis in range(3)
    )
    scale = max(float(np.max(np.abs(coefficients))), 1.0)
    return float(np.max(np.abs(divergence)) / scale)


def _euler_quadratic(
    high_coefficients: Array,
    high_values: Array,
    frequencies: tuple[Array, Array, Array],
    safe_wave_number_squared: Array,
    volume: int,
) -> Array:
    advection = np.zeros_like(high_values)
    for component in range(3):
        for direction in range(3):
            derivative = _physical(
                1j
                * frequencies[direction]
                * high_coefficients[component],
                volume,
            )
            advection[component] += (
                high_values[direction] * derivative
            )
            del derivative
    coefficients = np.stack(
        [
            _coefficients(advection[component], volume)
            for component in range(3)
        ],
        axis=0,
    )
    del advection
    return _project_negative(
        coefficients,
        frequencies,
        safe_wave_number_squared,
    )


def _euler_cross(
    high_coefficients: Array,
    high_values: Array,
    velocity_coefficients: Array,
    velocity_values: Array,
    frequencies: tuple[Array, Array, Array],
    safe_wave_number_squared: Array,
    volume: int,
) -> Array:
    advection = np.zeros_like(high_values)
    for component in range(3):
        for direction in range(3):
            derivative_velocity = _physical(
                1j
                * frequencies[direction]
                * velocity_coefficients[component],
                volume,
            )
            derivative_high = _physical(
                1j
                * frequencies[direction]
                * high_coefficients[component],
                volume,
            )
            advection[component] += 0.5 * (
                high_values[direction] * derivative_velocity
                + velocity_values[direction] * derivative_high
            )
            del derivative_velocity
            del derivative_high
    coefficients = np.stack(
        [
            _coefficients(advection[component], volume)
            for component in range(3)
        ],
        axis=0,
    )
    del advection
    return _project_negative(
        coefficients,
        frequencies,
        safe_wave_number_squared,
    )


def _quadrature_row(size: int) -> dict[str, Any]:
    started = time.perf_counter()
    shape = _grid_shape(size)
    volume = int(np.prod(shape))
    frequencies = _frequency_axes(shape)
    wave_number_squared = sum(
        frequency * frequency for frequency in frequencies
    )
    wave_number_squared[0, 0, 0] = 1.0

    high_coefficients = _high_coefficients(size, shape)
    high_values = np.stack(
        [
            _physical(high_coefficients[component], volume)
            for component in range(3)
        ],
        axis=0,
    )
    velocity_coefficients = _euler_quadratic(
        high_coefficients,
        high_values,
        frequencies,
        wave_number_squared,
        volume,
    )
    velocity_values = np.stack(
        [
            _physical(velocity_coefficients[component], volume)
            for component in range(3)
        ],
        axis=0,
    )
    acceleration_coefficients = _euler_cross(
        high_coefficients,
        high_values,
        velocity_coefficients,
        velocity_values,
        frequencies,
        wave_number_squared,
        volume,
    )

    velocity_energy_components = [
        float(
            np.sum(
                np.abs(velocity_coefficients[component]) ** 2
            )
        )
        for component in range(3)
    ]
    acceleration_pair_components = [
        float(
            np.sum(
                (
                    acceleration_coefficients[component]
                    * np.conjugate(high_coefficients[component])
                ).real
            )
        )
        for component in range(3)
    ]
    velocity_energy = sum(velocity_energy_components)
    acceleration_pair = sum(acceleration_pair_components)
    energy_trace_residual = velocity_energy + 2.0 * acceleration_pair
    energy_trace_scale = max(
        velocity_energy,
        2.0 * abs(acceleration_pair),
        1.0,
    )
    normalization = float(size**7)
    first_component = (
        math.sqrt(2.0)
        / 20.0
        * (
            velocity_energy_components[2]
            - velocity_energy_components[1]
        )
        / normalization
    )
    second_component = (
        math.sqrt(2.0)
        / 10.0
        * (
            acceleration_pair_components[2]
            - acceleration_pair_components[1]
        )
        / normalization
    )
    divergence_residuals = {
        "H": _divergence_residual(
            high_coefficients,
            frequencies,
        ),
        "V": _divergence_residual(
            velocity_coefficients,
            frequencies,
        ),
        "G": _divergence_residual(
            acceleration_coefficients,
            frequencies,
        ),
    }
    row = {
        "size": size,
        "grid_shape": list(shape),
        "grid_point_count": volume,
        "first_form_continuum_quadrature": first_component,
        "second_form_continuum_quadrature": second_component,
        "combined_continuum_quadrature": (
            first_component + second_component
        ),
        "velocity_energy_components_over_N7": [
            value / normalization for value in velocity_energy_components
        ],
        "acceleration_pair_components_over_N7": [
            value / normalization
            for value in acceleration_pair_components
        ],
        "energy_trace_relative_residual": (
            abs(energy_trace_residual) / energy_trace_scale
        ),
        "maximum_divergence_residual": max(
            divergence_residuals.values()
        ),
        "divergence_residuals": divergence_residuals,
        "runtime_seconds": time.perf_counter() - started,
    }
    row["all_checks_pass"] = bool(
        row["energy_trace_relative_residual"] < 1.0e-10
        and row["maximum_divergence_residual"] < 1.0e-9
        and first_component > 0.0
        and second_component < 0.0
        and row["combined_continuum_quadrature"] < 0.0
    )
    del acceleration_coefficients
    del velocity_values
    del velocity_coefficients
    del high_values
    del high_coefficients
    del wave_number_squared
    gc.collect()
    return row


def _fit_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sizes = np.asarray([row["size"] for row in rows], dtype=float)
    output: dict[str, Any] = {}
    for field in (
        "first_form_continuum_quadrature",
        "second_form_continuum_quadrature",
        "combined_continuum_quadrature",
    ):
        values = np.asarray(
            [row[field] for row in rows],
            dtype=float,
        )
        fits = []
        for degree in (1, 2, 3, 4):
            if len(rows) <= degree:
                continue
            coefficients = np.polyfit(1.0 / sizes, values, degree)
            replay = np.polyval(coefficients, 1.0 / sizes)
            fits.append(
                {
                    "degree_in_inverse_N": degree,
                    "candidate_limit": float(coefficients[-1]),
                    "maximum_replay_residual": float(
                        np.max(np.abs(replay - values))
                    ),
                    "certification_status": "diagnostic_only",
                }
            )
        output[field] = fits
    return output


def _fixed_output_cross_replay(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    prerequisite = json.loads(PREREQUISITE.read_text(encoding="utf-8"))
    fixed_rows = {
        int(row["size"]): row
        for row in prerequisite["finite_output_diagnostics"]["rows"]
    }
    replay_rows = []
    for row in rows:
        size = int(row["size"])
        if size not in fixed_rows:
            continue
        active = float(fixed_rows[size]["active_combined_over_N7"])
        quadrature = float(row["combined_continuum_quadrature"])
        replay_rows.append(
            {
                "size": size,
                "fixed_output_active_over_N7": active,
                "zero_shift_continuum_quadrature": quadrature,
                "difference": active - quadrature,
                "absolute_difference": abs(active - quadrature),
            }
        )
    return {
        "rows": replay_rows,
        "largest_common_size": (
            replay_rows[-1]["size"] if replay_rows else None
        ),
        "largest_common_absolute_difference": (
            replay_rows[-1]["absolute_difference"]
            if replay_rows
            else None
        ),
        "interpretation": (
            "The difference is the finite q/N shift and sampled-profile "
            "error. Agreement is a cross-replay, not an interval bound."
        ),
        "all_checks_pass": bool(
            replay_rows
            and replay_rows[-1]["size"] == 29
            and replay_rows[-1]["absolute_difference"] < 4.0e-9
        ),
    }


def _payload(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fits = _fit_rows(rows)
    tail_rows = rows[-9:] if len(rows) >= 9 else rows
    tail_fits = _fit_rows(tail_rows)
    cross_replay = _fixed_output_cross_replay(rows)
    combined_candidates = [
        fit["candidate_limit"]
        for fit in fits.get("combined_continuum_quadrature", [])
    ]
    tail_combined_candidates = [
        fit["candidate_limit"]
        for fit in tail_fits.get(
            "combined_continuum_quadrature",
            [],
        )
    ]
    row_integrity = bool(
        rows
        and all(row["all_checks_pass"] for row in rows)
    )
    numerical_sign_stable = bool(
        len(rows) >= 4
        and combined_candidates
        and tail_combined_candidates
        and all(value < 0.0 for value in combined_candidates)
        and all(value < 0.0 for value in tail_combined_candidates)
    )
    return {
        "kind": "annular_rho_zero_continuum_convolution_quadrature",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": (
            "continuum_convolution_quadrature_complete_sign_not_interval"
            if row_integrity
            else "continuum_convolution_quadrature_in_progress"
        ),
        "scope": (
            "Fourfold dealiased lattice quadrature for the two "
            "fixed-domain continuum components"
        ),
        "prerequisite": {
            "path": str(PREREQUISITE.relative_to(ROOT)).replace("\\", "/"),
            "expected_sha256": PREREQUISITE_SHA256,
            "actual_sha256": _sha256(PREREQUISITE),
        },
        "dealias_certificate": {
            "factor": 4,
            "reason": (
                "The two diagnostics are quartic means: "
                "||B(H,H)||_2^2 and <B(H,B(H,H)),H>. "
                "A grid length greater than four times each one-field "
                "maximum prevents wraparound into the zero mode."
            ),
        },
        "rows": list(rows),
        "inverse_N_fits": fits,
        "tail_inverse_N_fits": {
            "sizes": [int(row["size"]) for row in tail_rows],
            "fits": tail_fits,
        },
        "fixed_output_cross_replay": cross_replay,
        "route_decision": {
            "continuum_sign_numerically_stable": numerical_sign_stable,
            "continuum_sign_interval_certified": False,
            "nonzero_N9_coefficient_certified": False,
            "next_action": (
                "Convert the convolution quadrature into an interval "
                "enclosure by bounding profile sampling, Fourier "
                "projection, and inverse-N extrapolation errors. "
                "Do not use fit agreement as the sign proof."
            ),
        },
        "certification_flags": {
            "fourfold_quartic_dealiasing_proved": True,
            "Euler_energy_trace_replayed": all(
                row["energy_trace_relative_residual"] < 1.0e-10
                for row in rows
            ),
            "continuum_sign_numerically_stable": numerical_sign_stable,
            "continuum_sign_interval_certified": False,
            "four_high_N9_coefficient_certified": False,
            "uniform_second_jet_Taylor_bound_proved": False,
            "finite_parabolic_window_controlled": False,
            "critical_L3_controlled": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "all_positive_checks_pass": row_integrity,
    }


def _parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item)
    if not sizes or any(size < 3 or size % 2 == 0 for size in sizes):
        raise argparse.ArgumentTypeError(
            "sizes must be comma-separated odd integers at least three"
        )
    if tuple(sorted(set(sizes))) != sizes:
        raise argparse.ArgumentTypeError(
            "sizes must be strictly increasing"
        )
    return sizes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SIZES,
    )
    parser.add_argument("--output", type=Path, default=RESULT)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    _lower_process_priority()
    prerequisite_hash = _sha256(PREREQUISITE)
    if prerequisite_hash != PREREQUISITE_SHA256:
        raise ValueError("fixed-output continuum prerequisite changed")
    prerequisite = json.loads(PREREQUISITE.read_text(encoding="utf-8"))
    if prerequisite.get("all_positive_checks_pass") is not True:
        raise ValueError("fixed-output continuum prerequisite did not pass")

    output = (
        arguments.output
        if arguments.output.is_absolute()
        else ROOT / arguments.output
    )
    existing_rows: dict[int, dict[str, Any]] = {}
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (
            existing.get("algorithm_revision") == ALGORITHM_REVISION
            and existing.get("prerequisite", {}).get("actual_sha256")
            == prerequisite_hash
        ):
            existing_rows = {
                int(row["size"]): row for row in existing.get("rows", [])
            }

    rows = []
    for size in arguments.sizes:
        if size in existing_rows:
            row = existing_rows[size]
        else:
            row = _quadrature_row(size)
        if not row["all_checks_pass"]:
            raise ValueError(f"quadrature row N={size} did not pass")
        rows.append(row)
        _atomic_json(output, _payload(rows))
        print(
            f"N={size} L={row['combined_continuum_quadrature']:.12e} "
            f"trace={row['energy_trace_relative_residual']:.3e} "
            f"seconds={row['runtime_seconds']:.2f}",
            flush=True,
        )

    result = _payload(rows)
    if not result["all_positive_checks_pass"]:
        raise ValueError("continuum quadrature did not pass")
    _atomic_json(output, result)


if __name__ == "__main__":
    main()
