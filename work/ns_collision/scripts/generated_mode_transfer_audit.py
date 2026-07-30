"""Generation-by-generation quartic transfer along a Galerkin trajectory."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.integrate import simpson, solve_ivp


GALERKIN_SCRIPT = Path(__file__).with_name("galerkin_trajectory_audit.py")
SPEC = importlib.util.spec_from_file_location(
    "generated_mode_galerkin_helpers", GALERKIN_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
GALERKIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GALERKIN)


SEED_WAVES = ((1, 0, 0), (-1, 0, 0), (1, 1, 0), (-1, -1, 0))
DIFFERENCE_WAVES = ((0, 1, 0), (0, -1, 0))
SUM_WAVES = ((2, 1, 0), (-2, -1, 0))
COMPONENTS = (
    "seed",
    "difference_mode_increment",
    "sum_mode_increment",
    "difference_sum_interaction",
    "higher_generation_remainder",
)


def _projection(
    system: GALERKIN.GalerkinSystem,
    velocity: np.ndarray,
    waves: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    result = np.zeros_like(velocity)
    for wave in waves:
        result[system.wave_index[wave]] = velocity[system.wave_index[wave]]
    return result


def _transfer(system: GALERKIN.GalerkinSystem, velocity: np.ndarray) -> float:
    field = {
        wave: velocity[index]
        for index, wave in enumerate(system.waves)
        if np.linalg.norm(velocity[index]) > 1.0e-14
    }
    advection: dict[tuple[int, int, int], np.ndarray] = {}
    for first_wave, first_value in field.items():
        for second_wave, second_value in field.items():
            output = tuple(
                first_wave[axis] + second_wave[axis] for axis in range(3)
            )
            if output not in system.wave_index:
                continue
            contribution = (
                1j * np.dot(first_value, second_wave) * second_value
            )
            advection[output] = advection.get(
                output, np.zeros(3, dtype=complex)
            ) + contribution
    euler = {}
    for wave, value in advection.items():
        index = system.wave_index[wave]
        projected = -(system.projection[index] @ value)
        if np.linalg.norm(projected) > 1.0e-14:
            euler[wave] = projected

    def trilinear(
        first: dict[tuple[int, int, int], np.ndarray],
        second: dict[tuple[int, int, int], np.ndarray],
        third: dict[tuple[int, int, int], np.ndarray],
    ) -> complex:
        total = 0j
        for first_wave, first_value in first.items():
            first_frequency = float(np.dot(first_wave, first_wave))
            multiplier = 1.0 - np.exp(-system.heat_scale * first_frequency)
            for second_wave, second_value in second.items():
                third_wave = tuple(
                    -first_wave[axis] - second_wave[axis] for axis in range(3)
                )
                if third_wave not in third:
                    continue
                total_frequency = (
                    first_frequency
                    + float(np.dot(second_wave, second_wave))
                    + float(np.dot(third_wave, third_wave))
                )
                second_vorticity = 1j * np.cross(
                    second_wave, second_value
                )
                third_vorticity = 1j * np.cross(
                    third_wave, third[third_wave]
                )
                contraction = 0.5j * (
                    np.dot(first_value, second_vorticity)
                    * np.dot(first_wave, third_vorticity)
                    + np.dot(first_wave, second_vorticity)
                    * np.dot(first_value, third_vorticity)
                )
                total += multiplier * contraction / total_frequency
        return total

    value = sum(
        (
            trilinear(euler, field, field),
            trilinear(field, euler, field),
            trilinear(field, field, euler),
        )
    )
    return float(value.real)


def generation_channels(
    system: GALERKIN.GalerkinSystem, velocity: np.ndarray
) -> dict[str, float]:
    seed = _projection(system, velocity, SEED_WAVES)
    difference = _projection(system, velocity, DIFFERENCE_WAVES)
    summation = _projection(system, velocity, SUM_WAVES)
    seed_difference = seed + difference
    seed_sum = seed + summation
    first_generation = seed + difference + summation

    seed_value = _transfer(system, seed)
    difference_value = _transfer(system, seed_difference)
    sum_value = _transfer(system, seed_sum)
    first_generation_value = _transfer(system, first_generation)
    full_value = _transfer(system, velocity)
    result = {
        "seed": seed_value,
        "difference_mode_increment": difference_value - seed_value,
        "sum_mode_increment": sum_value - seed_value,
        "difference_sum_interaction": (
            first_generation_value
            - difference_value
            - sum_value
            + seed_value
        ),
        "higher_generation_remainder": full_value - first_generation_value,
        "first_generation_total": first_generation_value,
        "full_total": full_value,
    }
    result["decomposition_residual"] = full_value - sum(
        result[name] for name in COMPONENTS
    )
    return result


def solve_case(case: GALERKIN.Case) -> dict[str, object]:
    if case.maximum_mode < 2:
        raise ValueError("maximum_mode must contain the first sum mode")
    system = GALERKIN.GalerkinSystem(
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

    series = {name: [] for name in (*COMPONENTS, "full_total", "defect", "primitive")}
    maximum_decomposition_residual = 0.0
    maximum_boundary_fraction = 0.0
    for column in range(solution.y.shape[1]):
        velocity = solution.y[: system.state_size, column].reshape((-1, 3))
        channels = generation_channels(system, velocity)
        diagnostic = system.diagnostics(velocity)
        for name in COMPONENTS:
            series[name].append(channels[name])
        series["full_total"].append(channels["full_total"])
        series["defect"].append(diagnostic["defect"])
        series["primitive"].append(diagnostic["primitive"])
        maximum_decomposition_residual = max(
            maximum_decomposition_residual,
            abs(channels["decomposition_residual"]),
        )
        maximum_boundary_fraction = max(
            maximum_boundary_fraction, diagnostic["boundary_energy_fraction"]
        )

    integrals = {
        name: float(simpson(np.asarray(values), x=times))
        for name, values in series.items()
        if name != "primitive"
    }
    endpoint_primitive = series["primitive"][-1] - series["primitive"][0]
    integral_sum_residual = integrals["full_total"] - sum(
        integrals[name] for name in COMPONENTS
    )
    primitive_balance_residual = (
        endpoint_primitive
        + case.viscosity * integrals["defect"]
        - integrals["full_total"]
    )
    return {
        "maximum_mode": case.maximum_mode,
        "reynolds": case.reynolds,
        "viscosity": case.viscosity,
        "heat_scale": case.heat_scale,
        "sign": case.sign,
        "final_time": case.final_time,
        "samples": case.samples,
        "integrated_components": {
            name: integrals[name] for name in COMPONENTS
        },
        "integrated_total_transfer": integrals["full_total"],
        "integrated_defect": integrals["defect"],
        "endpoint_primitive": endpoint_primitive,
        "maximum_decomposition_residual": maximum_decomposition_residual,
        "integral_sum_residual": integral_sum_residual,
        "primitive_balance_residual": primitive_balance_residual,
        "maximum_boundary_energy_fraction": maximum_boundary_fraction,
        "underresolved_warning": maximum_boundary_fraction > 1.0e-3,
        "decomposition_checks_pass": bool(
            maximum_decomposition_residual < 1.0e-10
            and abs(integral_sum_residual) < 1.0e-10
        ),
    }


def audit() -> dict[str, object]:
    return solve_case(
        GALERKIN.Case(
            maximum_mode=2,
            reynolds=0.5,
            viscosity=1.0,
            heat_scale=0.5,
            sign=-1,
            final_time=0.02,
            samples=5,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=3)
    parser.add_argument("--reynolds", type=float, default=0.922)
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    final_time = min(2.0, 2.0 / max(args.reynolds, 1.0e-12))
    result = solve_case(
        GALERKIN.Case(
            maximum_mode=args.mode,
            reynolds=args.reynolds,
            viscosity=args.viscosity,
            heat_scale=args.scale,
            sign=args.sign,
            final_time=final_time,
            samples=args.samples,
        )
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
