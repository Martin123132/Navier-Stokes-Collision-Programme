"""Parity/helical decomposition of quartic transfer along Galerkin flow."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import simpson, solve_ivp


GALERKIN_SCRIPT = Path(__file__).with_name("galerkin_trajectory_audit.py")
SPEC = importlib.util.spec_from_file_location(
    "helical_trajectory_galerkin_helpers", GALERKIN_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
GALERKIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GALERKIN)


CHANNEL_NAMES = (
    "symmetric_homochiral",
    "symmetric_heterochiral",
    "antisymmetric_homochiral_diagonal",
    "antisymmetric_heterochiral_diagonal",
    "antisymmetric_interference",
)


def _helical_amplitudes(
    system: GALERKIN.GalerkinSystem, velocity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    root_two = np.sqrt(2.0)
    k_plus = np.asarray([0.0, 1.0, 1j]) / root_two
    k_minus = np.asarray([0.0, 1.0, -1j]) / root_two
    m_first = np.asarray([-1.0 / root_two, 1.0 / root_two, 0.0])
    m_plus = (m_first + np.asarray([0.0, 0.0, 1j])) / root_two
    m_minus = (m_first + np.asarray([0.0, 0.0, -1j])) / root_two
    k_value = velocity[system.wave_index[(1, 0, 0)]]
    m_value = velocity[system.wave_index[(1, 1, 0)]]
    first = np.asarray(
        [np.vdot(k_plus, k_value), np.vdot(k_minus, k_value)]
    )
    second = np.asarray(
        [np.vdot(m_plus, m_value), np.vdot(m_minus, m_value)]
    )
    return first, second


def _seed_projection(
    system: GALERKIN.GalerkinSystem, velocity: np.ndarray
) -> np.ndarray:
    seed = np.zeros_like(velocity)
    for wave in ((1, 0, 0), (-1, 0, 0), (1, 1, 0), (-1, -1, 0)):
        seed[system.wave_index[wave]] = velocity[system.wave_index[wave]]
    return seed


def seed_channels(
    system: GALERKIN.GalerkinSystem, velocity: np.ndarray
) -> dict[str, float]:
    first, second = _helical_amplitudes(system, velocity)
    pair = np.asarray(
        [
            first[0] * second[0],
            first[0] * second[1],
            first[1] * second[0],
            first[1] * second[1],
        ]
    )
    inverse_root_two = 1.0 / np.sqrt(2.0)
    parity_basis = np.asarray(
        [
            [inverse_root_two, 0.0, inverse_root_two, 0.0],
            [0.0, inverse_root_two, 0.0, inverse_root_two],
            [0.0, inverse_root_two, 0.0, -inverse_root_two],
            [inverse_root_two, 0.0, -inverse_root_two, 0.0],
        ]
    )
    parity = parity_basis.T @ pair

    x = np.exp(-system.heat_scale)
    p = x**3 + 2.0 * x**2 + 3.0 * x
    root_two = np.sqrt(2.0)
    scalar_first = -root_two * (p - 11.0) / 80.0
    scalar_second = root_two * (p - 11.0) / 80.0
    block_first = (
        (0.25 - 3.0 * root_two / 16.0) * (p + root_two + 3.0)
    )
    block_second = (
        (0.25 + 3.0 * root_two / 16.0) * (p - root_two + 3.0)
    )
    scale = (1.0 - x) ** 2
    values = {
        "symmetric_homochiral": scale
        * scalar_first
        * float(abs(parity[0]) ** 2),
        "symmetric_heterochiral": scale
        * scalar_second
        * float(abs(parity[1]) ** 2),
        "antisymmetric_homochiral_diagonal": scale
        * block_first
        * float(abs(parity[2]) ** 2),
        "antisymmetric_heterochiral_diagonal": scale
        * block_second
        * float(abs(parity[3]) ** 2),
        "antisymmetric_interference": scale
        * 0.25
        * float(np.real(np.conjugate(parity[2]) * parity[3])),
    }
    values["seed_matrix_total"] = sum(values.values())

    seed = _seed_projection(system, velocity)
    euler = system.euler_direction(seed)
    direct = sum(
        (
            system.trilinear(euler, seed, seed, primitive=True),
            system.trilinear(seed, euler, seed, primitive=True),
            system.trilinear(seed, seed, euler, primitive=True),
        )
    )
    values["seed_direct_total"] = float(direct.real)
    values["seed_matrix_residual"] = float(
        abs(values["seed_matrix_total"] - direct)
    )
    values["rank_one_residual"] = float(abs(pair[0] * pair[3] - pair[1] * pair[2]))
    return values


def solve_channel_case(case: GALERKIN.Case) -> dict[str, object]:
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

    series: dict[str, list[float]] = {
        name: []
        for name in (
            *CHANNEL_NAMES,
            "seed_matrix_total",
            "seed_direct_total",
            "generated_mode_remainder",
            "total_transfer",
            "defect",
            "primitive",
            "palinstrophy",
        )
    }
    maximum_seed_matrix_residual = 0.0
    maximum_rank_one_residual = 0.0
    maximum_boundary_fraction = 0.0
    maximum_identity_residual = 0.0
    for column in range(solution.y.shape[1]):
        velocity = solution.y[: system.state_size, column].reshape((-1, 3))
        diagnostic = system.diagnostics(velocity)
        channels = seed_channels(system, velocity)
        for name in CHANNEL_NAMES:
            series[name].append(channels[name])
        series["seed_matrix_total"].append(channels["seed_matrix_total"])
        series["seed_direct_total"].append(channels["seed_direct_total"])
        series["total_transfer"].append(diagnostic["quartic_transfer"])
        series["generated_mode_remainder"].append(
            diagnostic["quartic_transfer"] - channels["seed_direct_total"]
        )
        series["defect"].append(diagnostic["defect"])
        series["primitive"].append(diagnostic["primitive"])
        series["palinstrophy"].append(diagnostic["palinstrophy"])
        maximum_seed_matrix_residual = max(
            maximum_seed_matrix_residual, channels["seed_matrix_residual"]
        )
        maximum_rank_one_residual = max(
            maximum_rank_one_residual, channels["rank_one_residual"]
        )
        maximum_boundary_fraction = max(
            maximum_boundary_fraction, diagnostic["boundary_energy_fraction"]
        )
        maximum_identity_residual = max(
            maximum_identity_residual, diagnostic["identity_residual"]
        )

    integrals = {
        name: float(simpson(np.asarray(values), x=times))
        for name, values in series.items()
        if name not in ("primitive",)
    }
    total_decomposition_residual = integrals["total_transfer"] - (
        integrals["seed_matrix_total"]
        + integrals["generated_mode_remainder"]
    )
    seed_channel_residual = integrals["seed_matrix_total"] - sum(
        integrals[name] for name in CHANNEL_NAMES
    )
    endpoint_primitive = series["primitive"][-1] - series["primitive"][0]
    primitive_balance_residual = (
        endpoint_primitive
        + case.viscosity * integrals["defect"]
        - integrals["total_transfer"]
    )

    def _sign_changes(values: list[float]) -> int:
        array = np.asarray(values)
        return int(np.sum(array[:-1] * array[1:] < 0))

    return {
        "maximum_mode": case.maximum_mode,
        "reynolds": case.reynolds,
        "viscosity": case.viscosity,
        "heat_scale": case.heat_scale,
        "sign": case.sign,
        "final_time": case.final_time,
        "samples": case.samples,
        "integrated_channels": {
            name: integrals[name] for name in CHANNEL_NAMES
        },
        "integrated_seed_transfer": integrals["seed_matrix_total"],
        "integrated_generated_mode_remainder": integrals[
            "generated_mode_remainder"
        ],
        "integrated_total_transfer": integrals["total_transfer"],
        "integrated_defect": integrals["defect"],
        "integrated_palinstrophy": integrals["palinstrophy"],
        "endpoint_primitive": endpoint_primitive,
        "seed_fraction_of_total_transfer": (
            integrals["seed_matrix_total"] / integrals["total_transfer"]
            if abs(integrals["total_transfer"]) > 1.0e-30
            else float("nan")
        ),
        "generated_fraction_of_total_transfer": (
            integrals["generated_mode_remainder"]
            / integrals["total_transfer"]
            if abs(integrals["total_transfer"]) > 1.0e-30
            else float("nan")
        ),
        "initial_total_transfer": series["total_transfer"][0],
        "final_total_transfer": series["total_transfer"][-1],
        "initial_seed_transfer": series["seed_matrix_total"][0],
        "final_seed_transfer": series["seed_matrix_total"][-1],
        "total_transfer_sign_changes": _sign_changes(series["total_transfer"]),
        "seed_transfer_sign_changes": _sign_changes(series["seed_matrix_total"]),
        "generated_remainder_sign_changes": _sign_changes(
            series["generated_mode_remainder"]
        ),
        "maximum_seed_matrix_residual": maximum_seed_matrix_residual,
        "maximum_rank_one_residual": maximum_rank_one_residual,
        "maximum_total_decomposition_residual": abs(
            total_decomposition_residual
        ),
        "maximum_seed_channel_residual": abs(seed_channel_residual),
        "maximum_identity_residual": maximum_identity_residual,
        "primitive_balance_residual": primitive_balance_residual,
        "maximum_boundary_energy_fraction": maximum_boundary_fraction,
        "underresolved_warning": maximum_boundary_fraction > 1.0e-3,
        "channel_decomposition_checks_pass": bool(
            maximum_seed_matrix_residual < 1.0e-10
            and maximum_rank_one_residual < 1.0e-10
            and abs(total_decomposition_residual) < 1.0e-10
            and abs(seed_channel_residual) < 1.0e-10
            and maximum_identity_residual < 1.0e-10
        ),
    }


def audit() -> dict[str, object]:
    case = GALERKIN.Case(
        maximum_mode=2,
        reynolds=0.5,
        viscosity=1.0,
        heat_scale=0.5,
        sign=-1,
        final_time=0.05,
        samples=7,
    )
    return solve_channel_case(case)


def _parse_floats(value: str) -> list[float]:
    return [float(entry) for entry in value.split(",") if entry.strip()]


def run_sweep(cases: list[GALERKIN.Case], output: Path) -> list[dict[str, object]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(output) + ".lock")
    try:
        lock_descriptor = os.open(
            lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )
    except FileExistsError as error:
        raise RuntimeError(f"another sweep owns {lock_path}") from error
    os.write(lock_descriptor, str(os.getpid()).encode("ascii"))
    try:
        completed = set()
        if output.exists():
            for line in output.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                completed.add(
                    (
                        record["maximum_mode"],
                        record["reynolds"],
                        record["viscosity"],
                        record["heat_scale"],
                        record["sign"],
                        record["final_time"],
                        record["samples"],
                    )
                )
        results = []
        with output.open("a", encoding="utf-8") as stream:
            for case in cases:
                if case.key() in completed:
                    continue
                result = solve_channel_case(case)
                results.append(result)
                stream.write(json.dumps(result, sort_keys=True) + "\n")
                stream.flush()
        return results
    finally:
        os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=3)
    parser.add_argument("--reynolds", default="0.9,0.922,0.94,1")
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        GALERKIN.Case(
            maximum_mode=args.mode,
            reynolds=reynolds,
            viscosity=args.viscosity,
            heat_scale=args.scale,
            sign=args.sign,
            final_time=min(2.0, 2.0 / max(reynolds, 1.0e-12)),
            samples=args.samples,
        )
        for reynolds in _parse_floats(args.reynolds)
    ]
    results = run_sweep(cases, args.output)
    print(
        json.dumps(
            {
                "new_records": len(results),
                "decomposition_failures": sum(
                    not result["channel_decomposition_checks_pass"]
                    for result in results
                ),
                "underresolved_count": sum(
                    result["underresolved_warning"] for result in results
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
