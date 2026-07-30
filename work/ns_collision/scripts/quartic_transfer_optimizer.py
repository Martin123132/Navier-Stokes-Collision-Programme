"""Adversarial finite-Galerkin search for the quartic transfer sign.

The field is represented by independent real coordinates on one member of
each Fourier reality pair.  All quadratic Euler interactions are retained,
including output modes outside the input support, so the calculation is a
Galerkin restriction of the exact quartic polynomial rather than an aliased
pseudospectral approximation.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import minimize

from quartic_transfer_audit import evaluate


Wave = tuple[int, int, int]
Field = dict[Wave, np.ndarray]


def _positive_representative(wave: Wave) -> bool:
    return next(entry for entry in wave if entry != 0) > 0


def cube_half_modes(maximum_mode: int) -> list[Wave]:
    return [
        wave
        for wave in product(
            range(-maximum_mode, maximum_mode + 1), repeat=3
        )
        if wave != (0, 0, 0) and _positive_representative(wave)
    ]


def named_half_modes(name: str) -> list[Wave]:
    families: dict[str, list[Wave]] = {
        "triad": [(1, 0, 0), (0, 1, 0), (1, 1, 0)],
        "coupled": [
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
        ],
        "tetrahedral": [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ],
    }
    if name == "cube1":
        return cube_half_modes(1)
    try:
        return families[name]
    except KeyError as error:
        choices = ", ".join([*families, "cube1"])
        raise ValueError(f"unknown family {name!r}; choose from {choices}") from error


def _transverse_basis(wave: Wave) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(wave, dtype=float)
    vector /= np.linalg.norm(vector)
    axis = np.eye(3)[int(np.argmin(np.abs(vector)))]
    first = np.cross(vector, axis)
    first /= np.linalg.norm(first)
    second = np.cross(vector, first)
    return first, second


class GalerkinQuartic:
    """Exact finite-support quartic evaluator with analytic derivatives."""

    def __init__(self, half_modes: Iterable[Wave], heat_scale: float):
        self.half_modes = sorted(set(half_modes))
        if not self.half_modes:
            raise ValueError("at least one half mode is required")
        if any(not _positive_representative(wave) for wave in self.half_modes):
            raise ValueError("half modes must be positive reality representatives")

        self.heat_scale = float(heat_scale)
        self.dimension = 4 * len(self.half_modes)
        full_modes = {
            wave
            for positive in self.half_modes
            for wave in (positive, tuple(-entry for entry in positive))
        }
        self.input_waves = sorted(full_modes)
        self.input_index = {
            wave: index for index, wave in enumerate(self.input_waves)
        }

        basis = np.zeros(
            (len(self.input_waves), 3, self.dimension), dtype=complex
        )
        for mode_index, wave in enumerate(self.half_modes):
            first, second = _transverse_basis(wave)
            positive_index = self.input_index[wave]
            negative = tuple(-entry for entry in wave)
            negative_index = self.input_index[negative]
            offset = 4 * mode_index
            basis[positive_index, :, offset] = first
            basis[positive_index, :, offset + 1] = 1j * first
            basis[positive_index, :, offset + 2] = second
            basis[positive_index, :, offset + 3] = 1j * second
            basis[negative_index, :, offset] = first
            basis[negative_index, :, offset + 1] = -1j * first
            basis[negative_index, :, offset + 2] = second
            basis[negative_index, :, offset + 3] = -1j * second

        self.basis = torch.as_tensor(basis, dtype=torch.complex128)
        self.input_wave_tensor = torch.as_tensor(
            self.input_waves, dtype=torch.float64
        )

        output_waves = sorted(
            {
                tuple(np.asarray(first) + np.asarray(second))
                for first in self.input_waves
                for second in self.input_waves
                if tuple(np.asarray(first) + np.asarray(second)) != (0, 0, 0)
            }
        )
        self.output_waves = output_waves
        self.output_index = {
            wave: index for index, wave in enumerate(output_waves)
        }
        self.output_wave_tensor = torch.as_tensor(
            output_waves, dtype=torch.float64
        )

        pair_first: list[int] = []
        pair_second: list[int] = []
        pair_output: list[int] = []
        pair_second_wave: list[Wave] = []
        for first_index, first in enumerate(self.input_waves):
            for second_index, second in enumerate(self.input_waves):
                output = tuple(np.asarray(first) + np.asarray(second))
                if output == (0, 0, 0):
                    continue
                pair_first.append(first_index)
                pair_second.append(second_index)
                pair_output.append(self.output_index[output])
                pair_second_wave.append(second)
        self.pair_first = torch.as_tensor(pair_first, dtype=torch.long)
        self.pair_second = torch.as_tensor(pair_second, dtype=torch.long)
        self.pair_output = torch.as_tensor(pair_output, dtype=torch.long)
        self.pair_second_wave = torch.as_tensor(
            pair_second_wave, dtype=torch.float64
        )

        projection = []
        for wave in output_waves:
            vector = np.asarray(wave, dtype=float)
            projection.append(
                np.eye(3) - np.outer(vector, vector) / float(vector @ vector)
            )
        self.output_projection = torch.as_tensor(
            np.asarray(projection), dtype=torch.complex128
        )

        self.term_euu = self._make_triad_indices("output", "input", "input")
        self.term_ueu = self._make_triad_indices("input", "output", "input")
        self.term_uue = self._make_triad_indices("input", "input", "output")

    def _make_triad_indices(
        self, first_kind: str, second_kind: str, third_kind: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        waves = {
            "input": self.input_waves,
            "output": self.output_waves,
        }
        indices = {
            "input": self.input_index,
            "output": self.output_index,
        }
        first_indices: list[int] = []
        second_indices: list[int] = []
        third_indices: list[int] = []
        coefficients: list[float] = []
        for first_index, first in enumerate(waves[first_kind]):
            first_frequency = float(np.dot(first, first))
            multiplier = 1.0 - np.exp(-self.heat_scale * first_frequency)
            for second_index, second in enumerate(waves[second_kind]):
                third = tuple(-np.asarray(first) - np.asarray(second))
                if third not in indices[third_kind]:
                    continue
                denominator = (
                    first_frequency
                    + float(np.dot(second, second))
                    + float(np.dot(third, third))
                )
                first_indices.append(first_index)
                second_indices.append(second_index)
                third_indices.append(indices[third_kind][third])
                coefficients.append(multiplier / denominator)
        return (
            torch.as_tensor(first_indices, dtype=torch.long),
            torch.as_tensor(second_indices, dtype=torch.long),
            torch.as_tensor(third_indices, dtype=torch.long),
            torch.as_tensor(coefficients, dtype=torch.float64),
        )

    def field_tensor(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.einsum("mcd,d->mc", self.basis, coordinates.to(torch.complex128))

    def field(self, coordinates: np.ndarray) -> Field:
        values = np.einsum("mcd,d->mc", self.basis.numpy(), coordinates)
        return {
            wave: values[index]
            for index, wave in enumerate(self.input_waves)
            if np.linalg.norm(values[index]) > 1.0e-14
        }

    def euler_direction(self, velocity: torch.Tensor) -> torch.Tensor:
        first = velocity[self.pair_first]
        second = velocity[self.pair_second]
        scalar = torch.sum(first * self.pair_second_wave, dim=1)
        contributions = -1j * scalar[:, None] * second
        advection = torch.zeros(
            (len(self.output_waves), 3), dtype=torch.complex128
        )
        advection.index_add_(0, self.pair_output, contributions)
        return torch.einsum("mij,mj->mi", self.output_projection, advection)

    @staticmethod
    def _vorticity(waves: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return 1j * torch.linalg.cross(waves.to(values.dtype), values)

    @staticmethod
    def _triad_sum(
        first_values: torch.Tensor,
        first_waves: torch.Tensor,
        second_values: torch.Tensor,
        second_waves: torch.Tensor,
        third_values: torch.Tensor,
        third_waves: torch.Tensor,
        data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        first_index, second_index, third_index, coefficient = data
        first_value = first_values[first_index]
        first_wave = first_waves[first_index]
        second_vorticity = GalerkinQuartic._vorticity(
            second_waves[second_index], second_values[second_index]
        )
        third_vorticity = GalerkinQuartic._vorticity(
            third_waves[third_index], third_values[third_index]
        )
        contraction = 0.5j * (
            torch.sum(first_value * second_vorticity, dim=1)
            * torch.sum(first_wave * third_vorticity, dim=1)
            + torch.sum(first_wave * second_vorticity, dim=1)
            * torch.sum(first_value * third_vorticity, dim=1)
        )
        return torch.real(torch.sum(coefficient * contraction))

    def quartic_transfer(self, coordinates: torch.Tensor) -> torch.Tensor:
        velocity = self.field_tensor(coordinates)
        euler = self.euler_direction(velocity)
        return sum(
            (
                self._triad_sum(
                    euler,
                    self.output_wave_tensor,
                    velocity,
                    self.input_wave_tensor,
                    velocity,
                    self.input_wave_tensor,
                    self.term_euu,
                ),
                self._triad_sum(
                    velocity,
                    self.input_wave_tensor,
                    euler,
                    self.output_wave_tensor,
                    velocity,
                    self.input_wave_tensor,
                    self.term_ueu,
                ),
                self._triad_sum(
                    velocity,
                    self.input_wave_tensor,
                    velocity,
                    self.input_wave_tensor,
                    euler,
                    self.output_wave_tensor,
                    self.term_uue,
                ),
            )
        )

    def normalized_value_and_gradient(
        self, coordinates: np.ndarray
    ) -> tuple[float, np.ndarray]:
        variable = torch.tensor(
            coordinates, dtype=torch.float64, requires_grad=True
        )
        norm_squared = torch.dot(variable, variable)
        value = self.quartic_transfer(variable) / norm_squared**2
        value.backward()
        return float(value.detach()), variable.grad.detach().numpy()

    def cross_check(self, coordinates: np.ndarray) -> tuple[float, float]:
        direct = float(
            self.quartic_transfer(torch.as_tensor(coordinates)).detach()
        )
        reference = evaluate(self.field(coordinates), self.heat_scale)[
            "quartic_transfer"
        ]
        return direct, reference


def search(
    model: GalerkinQuartic,
    restarts: int,
    seed: int,
    maximum_iterations: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    best_value = np.inf
    best_coordinates: np.ndarray | None = None
    runs = []
    for restart in range(restarts):
        initial = rng.normal(size=model.dimension)
        initial /= np.linalg.norm(initial)
        result = minimize(
            model.normalized_value_and_gradient,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maximum_iterations, "ftol": 1.0e-14},
        )
        coordinates = np.asarray(result.x)
        coordinates /= np.linalg.norm(coordinates)
        value, gradient = model.normalized_value_and_gradient(coordinates)
        runs.append(
            {
                "restart": restart,
                "value": value,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "iterations": int(result.nit),
                "success": bool(result.success),
            }
        )
        if value < best_value:
            best_value = value
            best_coordinates = coordinates.copy()

    assert best_coordinates is not None
    direct, reference = model.cross_check(best_coordinates)
    return {
        "dimension": model.dimension,
        "half_modes": model.half_modes,
        "heat_scale": model.heat_scale,
        "best_normalized_transfer": best_value,
        "cross_check_direct": direct,
        "cross_check_reference": reference,
        "cross_check_residual": direct - reference,
        "negative_candidate": best_value < -1.0e-10,
        "best_coordinates": best_coordinates.tolist(),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=["triad", "coupled", "tetrahedral", "cube1"],
        default="coupled",
    )
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = GalerkinQuartic(named_half_modes(args.family), args.scale)
    result = search(model, args.restarts, args.seed, args.iterations)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
