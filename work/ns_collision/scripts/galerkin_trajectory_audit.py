"""Validated Fourier-Galerkin trajectories for the collision defect.

The nonlinear convolution is projected exactly onto a finite Fourier cube.
No pseudospectral aliasing is used.  The JSONL sweep is append-only and skips
completed parameter rows, making bounded searches safely resumable.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
from scipy.integrate import simpson, solve_ivp


Wave = tuple[int, int, int]


class Case(NamedTuple):
    maximum_mode: int
    reynolds: float
    viscosity: float
    heat_scale: float
    sign: int
    final_time: float
    samples: int

    def key(self) -> tuple[int, float, float, float, int, float, int]:
        return (
            self.maximum_mode,
            self.reynolds,
            self.viscosity,
            self.heat_scale,
            self.sign,
            self.final_time,
            self.samples,
        )


class GalerkinSystem:
    """Exact projected convolution on a symmetric Fourier cube."""

    def __init__(self, maximum_mode: int, viscosity: float, heat_scale: float):
        if maximum_mode < 1:
            raise ValueError("maximum_mode must be at least one")
        self.maximum_mode = int(maximum_mode)
        self.viscosity = float(viscosity)
        self.heat_scale = float(heat_scale)
        self.waves: list[Wave] = sorted(
            wave
            for wave in product(
                range(-maximum_mode, maximum_mode + 1), repeat=3
            )
            if wave != (0, 0, 0)
        )
        self.wave_index = {
            wave: index for index, wave in enumerate(self.waves)
        }
        self.wave_array = np.asarray(self.waves, dtype=float)
        self.frequency = np.einsum(
            "ij,ij->i", self.wave_array, self.wave_array
        )
        self.state_size = 3 * len(self.waves)
        self.negative_index = np.asarray(
            [
                self.wave_index[tuple(-entry for entry in wave)]
                for wave in self.waves
            ],
            dtype=int,
        )
        self.boundary_mask = np.asarray(
            [max(abs(entry) for entry in wave) == maximum_mode for wave in self.waves]
        )

        projection = np.eye(3)[None, :, :] - (
            self.wave_array[:, :, None] * self.wave_array[:, None, :]
        ) / self.frequency[:, None, None]
        self.projection = projection
        self._build_convolution_indices()
        self._build_triad_indices()

    def _build_convolution_indices(self) -> None:
        first_indices: list[int] = []
        second_indices: list[int] = []
        output_indices: list[int] = []
        second_waves: list[Wave] = []
        for first_index, first_wave in enumerate(self.waves):
            for second_index, second_wave in enumerate(self.waves):
                output = tuple(
                    first_wave[axis] + second_wave[axis] for axis in range(3)
                )
                if output not in self.wave_index:
                    continue
                first_indices.append(first_index)
                second_indices.append(second_index)
                output_indices.append(self.wave_index[output])
                second_waves.append(second_wave)
        self.pair_first = np.asarray(first_indices, dtype=int)
        self.pair_second = np.asarray(second_indices, dtype=int)
        self.pair_output = np.asarray(output_indices, dtype=int)
        self.pair_second_wave = np.asarray(second_waves, dtype=float)

    def _build_triad_indices(self) -> None:
        first_indices: list[int] = []
        second_indices: list[int] = []
        third_indices: list[int] = []
        defect_weights: list[float] = []
        primitive_weights: list[float] = []
        for first_index, first_wave in enumerate(self.waves):
            first_frequency = self.frequency[first_index]
            multiplier = 1.0 - np.exp(-self.heat_scale * first_frequency)
            for second_index, second_wave in enumerate(self.waves):
                third_wave = tuple(
                    -first_wave[axis] - second_wave[axis] for axis in range(3)
                )
                if third_wave not in self.wave_index:
                    continue
                third_index = self.wave_index[third_wave]
                total_frequency = (
                    first_frequency
                    + self.frequency[second_index]
                    + self.frequency[third_index]
                )
                first_indices.append(first_index)
                second_indices.append(second_index)
                third_indices.append(third_index)
                defect_weights.append(multiplier)
                primitive_weights.append(multiplier / total_frequency)
        self.triad_first = np.asarray(first_indices, dtype=int)
        self.triad_second = np.asarray(second_indices, dtype=int)
        self.triad_third = np.asarray(third_indices, dtype=int)
        self.defect_weight = np.asarray(defect_weights)
        self.primitive_weight = np.asarray(primitive_weights)

    def initial_velocity(self, amplitude: float, sign: int) -> np.ndarray:
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 or 1")
        values = np.zeros((len(self.waves), 3), dtype=complex)
        coefficients = {
            (1, 0, 0): np.asarray([0.0, -1.0, 1.0]),
            (-1, 0, 0): np.asarray([0.0, -1.0, 1.0]),
            (1, 1, 0): np.asarray([-1.0, 1.0, float(sign)]),
            (-1, -1, 0): np.asarray([-1.0, 1.0, float(sign)]),
        }
        for wave, coefficient in coefficients.items():
            if wave not in self.wave_index:
                raise ValueError("truncation does not contain the initial field")
            values[self.wave_index[wave]] = amplitude * coefficient
        return values

    def euler_direction(self, velocity: np.ndarray) -> np.ndarray:
        first = velocity[self.pair_first]
        second = velocity[self.pair_second]
        scalar = np.einsum("ij,ij->i", first, self.pair_second_wave)
        contributions = 1j * scalar[:, None] * second
        advection = np.zeros_like(velocity)
        np.add.at(advection, self.pair_output, contributions)
        projected = np.einsum("mij,mj->mi", self.projection, advection)
        return -projected

    def right_hand_side(self, _time: float, state: np.ndarray) -> np.ndarray:
        velocity = state[: self.state_size].reshape((-1, 3))
        derivative = self.euler_direction(velocity)
        derivative -= self.viscosity * self.frequency[:, None] * velocity
        kinetic_dissipation = self.viscosity * float(
            np.sum(self.frequency[:, None] * np.abs(velocity) ** 2)
        )
        return np.concatenate(
            [derivative.reshape(-1), np.asarray([kinetic_dissipation])]
        )

    @staticmethod
    def _vorticity(waves: np.ndarray, values: np.ndarray) -> np.ndarray:
        return 1j * np.cross(waves, values)

    def trilinear(
        self,
        first: np.ndarray,
        second: np.ndarray,
        third: np.ndarray,
        primitive: bool,
    ) -> complex:
        first_value = first[self.triad_first]
        first_wave = self.wave_array[self.triad_first]
        second_vorticity = self._vorticity(
            self.wave_array[self.triad_second], second[self.triad_second]
        )
        third_vorticity = self._vorticity(
            self.wave_array[self.triad_third], third[self.triad_third]
        )
        contraction = 0.5j * (
            np.einsum("ij,ij->i", first_value, second_vorticity)
            * np.einsum("ij,ij->i", first_wave, third_vorticity)
            + np.einsum("ij,ij->i", first_wave, second_vorticity)
            * np.einsum("ij,ij->i", first_value, third_vorticity)
        )
        weights = self.primitive_weight if primitive else self.defect_weight
        return np.sum(weights * contraction)

    def diagnostics(self, velocity: np.ndarray) -> dict[str, float]:
        euler = self.euler_direction(velocity)
        heat = -self.frequency[:, None] * velocity
        defect_value = self.trilinear(
            velocity, velocity, velocity, primitive=False
        )
        primitive_value = self.trilinear(
            velocity, velocity, velocity, primitive=True
        )
        transfer_value = sum(
            (
                self.trilinear(euler, velocity, velocity, primitive=True),
                self.trilinear(velocity, euler, velocity, primitive=True),
                self.trilinear(velocity, velocity, euler, primitive=True),
            )
        )
        heat_derivative = sum(
            (
                self.trilinear(heat, velocity, velocity, primitive=True),
                self.trilinear(velocity, heat, velocity, primitive=True),
                self.trilinear(velocity, velocity, heat, primitive=True),
            )
        )
        vorticity = self._vorticity(self.wave_array, velocity)
        mode_energy = np.sum(np.abs(velocity) ** 2, axis=1)
        kinetic_energy = float(np.sum(mode_energy))
        enstrophy = float(np.sum(np.abs(vorticity) ** 2))
        palinstrophy = float(
            np.sum(self.frequency[:, None] * np.abs(vorticity) ** 2)
        )
        divergence = np.einsum("ij,ij->i", self.wave_array, velocity)
        reality = velocity[self.negative_index] - np.conjugate(velocity)
        boundary_fraction = (
            float(np.sum(mode_energy[self.boundary_mask])) / kinetic_energy
            if kinetic_energy > 0
            else 0.0
        )
        identity_residual = heat_derivative + defect_value
        return {
            "kinetic_energy": kinetic_energy,
            "enstrophy": enstrophy,
            "palinstrophy": palinstrophy,
            "defect": float(defect_value.real),
            "primitive": float(primitive_value.real),
            "quartic_transfer": float(transfer_value.real),
            "identity_residual": float(abs(identity_residual)),
            "maximum_imaginary_functional_residual": float(
                max(
                    abs(defect_value.imag),
                    abs(primitive_value.imag),
                    abs(transfer_value.imag),
                )
            ),
            "maximum_divergence": float(np.max(np.abs(divergence))),
            "maximum_reality_residual": float(np.max(np.abs(reality))),
            "boundary_energy_fraction": boundary_fraction,
        }


def _default_final_time(reynolds: float) -> float:
    return min(2.0, 2.0 / max(reynolds, 1.0e-12))


def solve_case(case: Case) -> dict[str, object]:
    system = GalerkinSystem(
        case.maximum_mode, case.viscosity, case.heat_scale
    )
    amplitude = case.reynolds * case.viscosity
    initial_velocity = system.initial_velocity(amplitude, case.sign)
    initial_state = np.concatenate(
        [initial_velocity.reshape(-1), np.asarray([0.0])]
    )
    times = np.linspace(0.0, case.final_time, case.samples)
    solution = solve_ivp(
        system.right_hand_side,
        (0.0, case.final_time),
        initial_state,
        method="DOP853",
        t_eval=times,
        rtol=1.0e-9,
        atol=1.0e-11,
        max_step=case.final_time / max(case.samples - 1, 1),
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    rows = []
    for column in range(solution.y.shape[1]):
        velocity = solution.y[: system.state_size, column].reshape((-1, 3))
        row = system.diagnostics(velocity)
        row["time"] = float(solution.t[column])
        row["kinetic_dissipation_integral"] = float(
            solution.y[system.state_size, column].real
        )
        rows.append(row)

    defect = np.asarray([row["defect"] for row in rows])
    transfer = np.asarray([row["quartic_transfer"] for row in rows])
    palinstrophy = np.asarray([row["palinstrophy"] for row in rows])
    integrated_defect = float(simpson(defect, x=solution.t))
    integrated_transfer = float(simpson(transfer, x=solution.t))
    integrated_palinstrophy = float(simpson(palinstrophy, x=solution.t))
    initial_energy = rows[0]["kinetic_energy"]
    energy_residuals = [
        abs(
            0.5 * row["kinetic_energy"]
            + row["kinetic_dissipation_integral"]
            - 0.5 * initial_energy
        )
        / max(0.5 * initial_energy, 1.0e-30)
        for row in rows
    ]
    primitive_balance = (
        rows[-1]["primitive"]
        - rows[0]["primitive"]
        + case.viscosity * integrated_defect
        - integrated_transfer
    )
    dissipation_budget = case.viscosity * integrated_palinstrophy
    ratio = (
        integrated_defect / dissipation_budget
        if dissipation_budget > 0
        else float("nan")
    )

    return {
        "maximum_mode": case.maximum_mode,
        "mode_count": len(system.waves),
        "reynolds": case.reynolds,
        "amplitude": amplitude,
        "viscosity": case.viscosity,
        "heat_scale": case.heat_scale,
        "sign": case.sign,
        "final_time": case.final_time,
        "samples": case.samples,
        "solver_success": bool(solution.success),
        "solver_evaluations": int(solution.nfev),
        "integrated_defect": integrated_defect,
        "integrated_quartic_transfer": integrated_transfer,
        "integrated_palinstrophy": integrated_palinstrophy,
        "defect_to_viscous_palinstrophy_ratio": ratio,
        "initial_primitive": rows[0]["primitive"],
        "final_primitive": rows[-1]["primitive"],
        "primitive_balance_residual": primitive_balance,
        "maximum_relative_energy_balance_residual": max(energy_residuals),
        "maximum_identity_residual": max(
            row["identity_residual"] for row in rows
        ),
        "maximum_divergence": max(row["maximum_divergence"] for row in rows),
        "maximum_reality_residual": max(
            row["maximum_reality_residual"] for row in rows
        ),
        "maximum_imaginary_functional_residual": max(
            row["maximum_imaginary_functional_residual"] for row in rows
        ),
        "maximum_boundary_energy_fraction": max(
            row["boundary_energy_fraction"] for row in rows
        ),
        "final_boundary_energy_fraction": rows[-1]["boundary_energy_fraction"],
        "underresolved_warning": max(
            row["boundary_energy_fraction"] for row in rows
        )
        > 1.0e-3,
        "initial_defect": rows[0]["defect"],
        "maximum_defect": max(defect),
        "minimum_defect": min(defect),
        "final_defect": rows[-1]["defect"],
        "defect_sign_changes": int(np.sum(defect[:-1] * defect[1:] < 0)),
    }


def audit() -> dict[str, object]:
    case = Case(
        maximum_mode=2,
        reynolds=0.5,
        viscosity=1.0,
        heat_scale=0.5,
        sign=1,
        final_time=0.1,
        samples=9,
    )
    result = solve_case(case)
    return {
        **result,
        "constraint_checks_pass": bool(
            result["maximum_relative_energy_balance_residual"] < 1.0e-9
            and result["maximum_identity_residual"] < 1.0e-9
            and result["maximum_divergence"] < 1.0e-10
            and result["maximum_reality_residual"] < 1.0e-10
            and result["maximum_imaginary_functional_residual"] < 1.0e-10
        ),
    }


def _parse_floats(value: str) -> list[float]:
    return [float(entry) for entry in value.split(",") if entry.strip()]


def _parse_ints(value: str) -> list[int]:
    return [int(entry) for entry in value.split(",") if entry.strip()]


def run_sweep(cases: Iterable[Case], output: Path | None) -> list[dict[str, object]]:
    lock_descriptor: int | None = None
    lock_path: Path | None = None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(str(output) + ".lock")
        try:
            lock_descriptor = os.open(
                lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError as error:
            raise RuntimeError(
                f"another sweep owns the output lock {lock_path}"
            ) from error
        os.write(lock_descriptor, str(os.getpid()).encode("ascii"))

    completed: set[tuple[int, float, float, float, int, float, int]] = set()
    try:
        if output is not None and output.exists():
            for line in output.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                completed.add(
                    Case(
                        maximum_mode=int(record["maximum_mode"]),
                        reynolds=float(record["reynolds"]),
                        viscosity=float(record["viscosity"]),
                        heat_scale=float(record["heat_scale"]),
                        sign=int(record["sign"]),
                        final_time=float(record["final_time"]),
                        samples=int(record["samples"]),
                    ).key()
                )

        results = []
        stream = (
            output.open("a", encoding="utf-8") if output is not None else None
        )
        try:
            for case in cases:
                if case.key() in completed:
                    continue
                result = solve_case(case)
                results.append(result)
                if stream is not None:
                    stream.write(json.dumps(result, sort_keys=True) + "\n")
                    stream.flush()
        finally:
            if stream is not None:
                stream.close()
        return results
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if lock_path is not None:
            lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="2,3")
    parser.add_argument("--reynolds", default="0.25,0.5,1,2,4")
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = [
        Case(
            maximum_mode=maximum_mode,
            reynolds=reynolds,
            viscosity=args.viscosity,
            heat_scale=args.scale,
            sign=args.sign,
            final_time=_default_final_time(reynolds),
            samples=args.samples,
        )
        for maximum_mode in _parse_ints(args.modes)
        for reynolds in _parse_floats(args.reynolds)
    ]
    results = run_sweep(cases, args.output)
    summary = {
        "new_records": len(results),
        "negative_integrated_defect_count": sum(
            result["integrated_defect"] < 0 for result in results
        ),
        "underresolved_count": sum(
            result["underresolved_warning"] for result in results
        ),
        "maximum_energy_residual": max(
            (
                result["maximum_relative_energy_balance_residual"]
                for result in results
            ),
            default=0.0,
        ),
        "maximum_primitive_balance_residual": max(
            (abs(result["primitive_balance_residual"]) for result in results),
            default=0.0,
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
