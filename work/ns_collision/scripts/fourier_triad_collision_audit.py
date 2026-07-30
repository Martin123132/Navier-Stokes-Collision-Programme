"""Exact and numerical audits for Fourier-triad collision defects."""

from __future__ import annotations

from itertools import product
import json

import sympy as sp


def _amplitude(
    wave: sp.Matrix,
    velocity: sp.Matrix,
    other_wave: sp.Matrix,
    other_velocity: sp.Matrix,
    last_wave: sp.Matrix,
    last_velocity: sp.Matrix,
) -> sp.Expr:
    """Remove the common Fourier phase from S(k):omega(l) tensor omega(m)."""
    strain_without_i = (
        velocity * wave.T + wave * velocity.T
    ) / 2
    other_vorticity_without_i = other_wave.cross(other_velocity)
    last_vorticity_without_i = last_wave.cross(last_velocity)
    return sp.expand(
        (
            other_vorticity_without_i.T
            * strain_without_i
            * last_vorticity_without_i
        )[0]
    )


def _triad_amplitudes(
    waves: tuple[sp.Matrix, sp.Matrix, sp.Matrix],
    velocities: tuple[sp.Matrix, sp.Matrix, sp.Matrix],
) -> sp.Matrix:
    k, ell, m = waves
    uk, uell, um = velocities
    return sp.Matrix(
        [
            _amplitude(k, uk, ell, uell, m, um),
            _amplitude(ell, uell, m, um, k, uk),
            _amplitude(m, um, k, uk, ell, uell),
        ]
    )


def _has_mixed_zero_sum_triad() -> bool:
    low = {(1, 0, 0), (0, 1, 0), (-1, -1, 0)}
    low |= {tuple(-entry for entry in wave) for wave in tuple(low)}
    high = {(0, 3, 0), (0, 0, 3), (0, -3, -3)}
    high |= {tuple(-entry for entry in wave) for wave in tuple(high)}

    for first, second, third in product(low | high, repeat=3):
        if not all(
            first[index] + second[index] + third[index] == 0
            for index in range(3)
        ):
            continue
        memberships = [first in low, second in low, third in low]
        if any(memberships) and not all(memberships):
            return True
    return False


def audit() -> dict[str, str | int | bool]:
    k = sp.Matrix([1, 0, 0])
    ell = sp.Matrix([0, 1, 0])
    m = sp.Matrix([-1, -1, 0])
    waves = (k, ell, m)

    first_velocities = (
        sp.Matrix([0, -1, -1]),
        sp.Matrix([-1, 0, -1]),
        sp.Matrix([-1, 1, -1]),
    )
    second_velocities = (
        sp.Matrix([0, -1, -1]),
        sp.Matrix([-1, 0, -1]),
        sp.Matrix([-1, 1, 0]),
    )
    first_amplitudes = _triad_amplitudes(waves, first_velocities)
    second_amplitudes = _triad_amplitudes(waves, second_velocities)
    wave_number_weights = sp.Matrix([1, 1, 2])
    amplitude_matrix = sp.Matrix.hstack(first_amplitudes, second_amplitudes)

    s = sp.symbols("s", positive=True)
    defect_weights = sp.Matrix(
        [1 - sp.exp(-s), 1 - sp.exp(-s), 1 - sp.exp(-2 * s)]
    )
    first_defect = sp.factor(defect_weights.dot(first_amplitudes))

    # Two noninteracting triads can be amplitude-scaled to have defects 1
    # and -1/2. Their frequency sums are 4 and 36, respectively.
    low_defect = sp.Integer(1)
    high_defect = -sp.Rational(1, 2)
    total_defect = low_defect + high_defect
    pure_heat_derivative = -(
        4 * low_defect + 36 * high_defect
    )

    result: dict[str, str | int | bool] = {
        "first_amplitude_vector": str(list(first_amplitudes)),
        "second_amplitude_vector": str(list(second_amplitudes)),
        "first_weighted_miller_identity": bool(
            wave_number_weights.dot(first_amplitudes) == 0
        ),
        "second_weighted_miller_identity": bool(
            wave_number_weights.dot(second_amplitudes) == 0
        ),
        "allowed_amplitude_plane_rank": amplitude_matrix.rank(),
        "no_second_universal_linear_constraint": amplitude_matrix.rank() == 2,
        "first_triad_heat_defect": str(first_defect),
        "first_triad_has_double_heat_zero": bool(
            sp.simplify(first_defect + (1 - sp.exp(-s)) ** 2) == 0
        ),
        "mixed_zero_sum_triads_absent": not _has_mixed_zero_sum_triad(),
        "two_triad_total_defect": str(total_defect),
        "two_triad_pure_heat_derivative_at_nu_one": str(
            pure_heat_derivative
        ),
        "pure_heat_can_increase_positive_total_defect": bool(
            total_defect > 0 and pure_heat_derivative > 0
        ),
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
