"""Certify continuum positivity for the spaced dense HHHL packet."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CARRIER_MULTIPLE = 4096
OFFSET_SPACING = 4


def _down(value: float) -> float:
    return math.nextafter(value, -math.inf)


def _up(value: float) -> float:
    return math.nextafter(value, math.inf)


@dataclass(frozen=True)
class Interval:
    """Closed binary64 interval with one-ulp directed rounding."""

    lower: float
    upper: float

    @classmethod
    def point(cls, value: float) -> "Interval":
        return cls(float(value), float(value))

    def __add__(self, other: object) -> "Interval":
        right = _as_interval(other)
        return Interval(
            _down(self.lower + right.lower),
            _up(self.upper + right.upper),
        )

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval(-self.upper, -self.lower)

    def __sub__(self, other: object) -> "Interval":
        return self + (-_as_interval(other))

    def __rsub__(self, other: object) -> "Interval":
        return _as_interval(other) - self

    def __mul__(self, other: object) -> "Interval":
        right = _as_interval(other)
        products = (
            self.lower * right.lower,
            self.lower * right.upper,
            self.upper * right.lower,
            self.upper * right.upper,
        )
        return Interval(_down(min(products)), _up(max(products)))

    __rmul__ = __mul__

    def reciprocal(self) -> "Interval":
        if self.lower <= 0.0 <= self.upper:
            raise ZeroDivisionError("interval denominator contains zero")
        values = (1.0 / self.lower, 1.0 / self.upper)
        return Interval(_down(min(values)), _up(max(values)))

    def __truediv__(self, other: object) -> "Interval":
        return self * _as_interval(other).reciprocal()

    def __rtruediv__(self, other: object) -> "Interval":
        return _as_interval(other) / self

    def square(self) -> "Interval":
        if self.lower <= 0.0 <= self.upper:
            return Interval(
                0.0,
                _up(max(self.lower**2, self.upper**2)),
            )
        values = (self.lower**2, self.upper**2)
        return Interval(_down(min(values)), _up(max(values)))

    def maximum_absolute_value(self) -> float:
        return max(abs(self.lower), abs(self.upper))


def _as_interval(value: object) -> Interval:
    if isinstance(value, Interval):
        return value
    return Interval.point(float(value))


@dataclass(frozen=True)
class ComplexInterval:
    real: Interval
    imaginary: Interval

    @classmethod
    def point(cls, value: complex | float) -> "ComplexInterval":
        number = complex(value)
        return cls(
            Interval.point(number.real),
            Interval.point(number.imag),
        )

    def __add__(self, other: object) -> "ComplexInterval":
        right = _as_complex_interval(other)
        return ComplexInterval(
            self.real + right.real,
            self.imaginary + right.imaginary,
        )

    __radd__ = __add__

    def __neg__(self) -> "ComplexInterval":
        return ComplexInterval(-self.real, -self.imaginary)

    def __sub__(self, other: object) -> "ComplexInterval":
        return self + (-_as_complex_interval(other))

    def __rsub__(self, other: object) -> "ComplexInterval":
        return _as_complex_interval(other) - self

    def __mul__(self, other: object) -> "ComplexInterval":
        right = _as_complex_interval(other)
        return ComplexInterval(
            self.real * right.real
            - self.imaginary * right.imaginary,
            self.real * right.imaginary
            + self.imaginary * right.real,
        )

    __rmul__ = __mul__

    def divide_by_real(self, denominator: Interval) -> "ComplexInterval":
        return ComplexInterval(
            self.real / denominator,
            self.imaginary / denominator,
        )


def _as_complex_interval(value: object) -> ComplexInterval:
    if isinstance(value, ComplexInterval):
        return value
    if isinstance(value, Interval):
        return ComplexInterval(value, Interval.point(0.0))
    return ComplexInterval.point(complex(value))


@dataclass(frozen=True)
class Jet:
    """Complex interval value and derivative with respect to tau=1/R."""

    value: ComplexInterval
    derivative: ComplexInterval

    @classmethod
    def constant(cls, value: object) -> "Jet":
        return cls(
            _as_complex_interval(value),
            ComplexInterval.point(0.0),
        )

    def __add__(self, other: object) -> "Jet":
        right = _as_jet(other)
        return Jet(
            self.value + right.value,
            self.derivative + right.derivative,
        )

    __radd__ = __add__

    def __neg__(self) -> "Jet":
        return Jet(-self.value, -self.derivative)

    def __sub__(self, other: object) -> "Jet":
        return self + (-_as_jet(other))

    def __rsub__(self, other: object) -> "Jet":
        return _as_jet(other) - self

    def __mul__(self, other: object) -> "Jet":
        right = _as_jet(other)
        return Jet(
            self.value * right.value,
            self.derivative * right.value
            + self.value * right.derivative,
        )

    __rmul__ = __mul__

    def divide_by_real_jet(self, denominator: "Jet") -> "Jet":
        if not (
            denominator.value.imaginary.lower
            <= 0.0
            <= denominator.value.imaginary.upper
        ):
            raise ValueError("denominator must be real")
        real_denominator = denominator.value.real
        value = self.value.divide_by_real(real_denominator)
        numerator_derivative = (
            self.derivative * denominator.value
            - self.value * denominator.derivative
        )
        denominator_square = (
            real_denominator * real_denominator
        )
        return Jet(
            value,
            numerator_derivative.divide_by_real(denominator_square),
        )

    def __truediv__(self, other: object) -> "Jet":
        return self.divide_by_real_jet(_as_jet(other))

    def __rtruediv__(self, other: object) -> "Jet":
        return _as_jet(other).divide_by_real_jet(self)


def _as_jet(value: object) -> Jet:
    return value if isinstance(value, Jet) else Jet.constant(value)


def _jet_variable(value: Interval) -> Jet:
    return Jet(
        ComplexInterval(value, Interval.point(0.0)),
        ComplexInterval.point(1.0),
    )


def _interval_dot(
    first: list[Interval] | tuple[float, ...],
    second: list[Interval] | tuple[float, ...],
) -> Interval:
    value = Interval.point(0.0)
    for left, right in zip(first, second):
        value += _as_interval(left) * _as_interval(right)
    return value


def _interval_norm_squared(vector: list[Interval]) -> Interval:
    value = Interval.point(0.0)
    for component in vector:
        value += component.square()
    return value


def _interval_project(
    vector: tuple[float, float, float] | list[Interval],
    wave: list[Interval],
) -> list[Interval]:
    coefficient = _interval_dot(wave, vector) / _interval_norm_squared(
        wave
    )
    return [
        _as_interval(value) - component * coefficient
        for value, component in zip(vector, wave)
    ]


def _interval_add_vectors(
    first: list[Interval],
    second: list[Interval],
) -> list[Interval]:
    return [left + right for left, right in zip(first, second)]


def _interval_scale_vector(
    scalar: Interval,
    vector: list[Interval],
) -> list[Interval]:
    return [scalar * value for value in vector]


def _interval_dynamics(
    first_wave: list[Interval],
    first_value: list[Interval],
    second_wave: list[Interval],
    second_value: list[Interval],
    output_wave: list[Interval],
) -> list[Interval]:
    raw = _interval_add_vectors(
        _interval_scale_vector(
            _interval_dot(first_value, second_wave),
            second_value,
        ),
        _interval_scale_vector(
            _interval_dot(second_value, first_wave),
            first_value,
        ),
    )
    return _interval_project(raw, output_wave)


def _interval_symmetrized_contraction(
    generated: list[Interval],
    remaining: list[Interval],
) -> Interval:
    low_vector = (1.0, -1.0, 0.0)
    partition_wave = (1.0, 1.0, 1.0)
    return (
        _interval_dot(remaining, low_vector)
        * _interval_dot(generated, partition_wave)
        + _interval_dot(generated, low_vector)
        * _interval_dot(remaining, partition_wave)
    )


def _relaxed_high_waves(
    carrier_multiple: int,
    offset_interval: Interval | None = None,
) -> tuple[list[Interval], list[Interval], list[Interval]]:
    delta = 1.0 / carrier_multiple
    offset = offset_interval or Interval(-1.0, 1.0)
    first_offset = [offset for _ in range(3)]
    second_offset = [offset for _ in range(3)]
    third_offset = [
        -left - right
        for left, right in zip(first_offset, second_offset)
    ]
    first_wave = [
        Interval.point(1.0) + delta * first_offset[0],
        delta * first_offset[1],
        delta * first_offset[2],
    ]
    second_wave = [
        Interval.point(-1.0) + delta * second_offset[0],
        Interval.point(1.0) + delta * second_offset[1],
        delta * second_offset[2],
    ]
    third_wave = [
        delta * third_offset[0],
        Interval.point(-1.0) + delta * third_offset[1],
        delta * third_offset[2],
    ]
    return first_wave, second_wave, third_wave


def _leading_stress_interval(
    carrier_multiple: int,
    offset_interval: Interval | None = None,
) -> dict[str, Any]:
    first_wave, second_wave, third_wave = _relaxed_high_waves(
        carrier_multiple,
        offset_interval,
    )
    first_value = _interval_project(
        (-4.0, -3.0, 1.0),
        first_wave,
    )
    second_value = _interval_project(
        (-3.0, -1.0, 2.0),
        second_wave,
    )
    third_value = _interval_project(
        (-3.0, 7.0, 1.0),
        third_wave,
    )
    first_generated = _interval_dynamics(
        first_wave,
        first_value,
        second_wave,
        second_value,
        [-value for value in third_wave],
    )
    second_generated = _interval_dynamics(
        first_wave,
        first_value,
        third_wave,
        third_value,
        [-value for value in second_wave],
    )
    third_generated = _interval_dynamics(
        second_wave,
        second_value,
        third_wave,
        third_value,
        [-value for value in first_wave],
    )
    scaled_coefficient = -(
        _interval_symmetrized_contraction(
            first_generated,
            third_value,
        )
        + _interval_symmetrized_contraction(
            second_generated,
            second_value,
        )
        + _interval_symmetrized_contraction(
            third_generated,
            first_value,
        )
    ) / 64.0
    square_root_two = math.sqrt(2.0)
    square_root_two_interval = Interval(
        _down(square_root_two),
        _up(square_root_two),
    )
    actual_coefficient = (
        scaled_coefficient / square_root_two_interval
    )
    return {
        "scaled_coefficient_interval": [
            scaled_coefficient.lower,
            scaled_coefficient.upper,
        ],
        "actual_coefficient_interval_after_low_normalization": [
            actual_coefficient.lower,
            actual_coefficient.upper,
        ],
        "relaxed_domain": (
            "x,y in [-1,1]^3 and z=-x-y in [-2,2]^3. This contains "
            "the true polytope, which also requires z in [-1,1]^3."
        ),
        "all_checks_pass": scaled_coefficient.lower > 0.0,
    }


def _central_symbol_self_audit(
    carrier_multiple: int,
) -> dict[str, Any]:
    center = _leading_stress_interval(
        carrier_multiple,
        Interval.point(0.0),
    )
    interval = center["scaled_coefficient_interval"]
    exact_scaled_coefficient = 3.0 / 16.0
    return {
        "exact_scaled_center_coefficient": "3/16",
        "numeric_exact_scaled_center_coefficient": (
            exact_scaled_coefficient
        ),
        "directed_interval": interval,
        "interval_width": interval[1] - interval[0],
        "all_checks_pass": bool(
            interval[0]
            <= exact_scaled_coefficient
            <= interval[1]
            and interval[1] - interval[0] < 1.0e-10
        ),
    }


def _tau_zero_identity_self_audit(
    carrier_multiple: int,
) -> dict[str, Any]:
    """Check the exact affine identities used to identify S(0)."""

    inverse_carrier_multiple = Fraction(1, carrier_multiple)
    centers = (
        (1, 0, 0),
        (-1, 1, 0),
        (0, -1, 0),
    )
    affine_waves: list[list[list[Fraction]]] = []
    for wave_index, center in enumerate(centers):
        wave: list[list[Fraction]] = []
        for component in range(3):
            coefficients = [Fraction(0) for _ in range(7)]
            coefficients[0] = Fraction(center[component])
            if wave_index == 0:
                coefficients[1 + component] = (
                    inverse_carrier_multiple
                )
            elif wave_index == 1:
                coefficients[4 + component] = (
                    inverse_carrier_multiple
                )
            else:
                coefficients[1 + component] = (
                    -inverse_carrier_multiple
                )
                coefficients[4 + component] = (
                    -inverse_carrier_multiple
                )
            wave.append(coefficients)
        affine_waves.append(wave)
    affine_sum = [
        [
            sum(
                affine_waves[wave][component][coefficient]
                for wave in range(3)
            )
            for coefficient in range(7)
        ]
        for component in range(3)
    ]
    center_sum = tuple(int(row[0]) for row in affine_sum)
    offset_coefficients_cancel = all(
        coefficient == 0
        for row in affine_sum
        for coefficient in row[1:]
    )
    partition_dot_low_polarization = sum(
        left * right
        for left, right in zip((1, 1, 1), (1, -1, 0))
    )
    return {
        "normalized_high_wave_sum": list(center_sum),
        "exact_affine_sum_coefficients": [
            [str(coefficient) for coefficient in row]
            for row in affine_sum
        ],
        "all_six_affine_offset_coefficients_cancel_exactly": (
            offset_coefficients_cancel
        ),
        "partition_dot_unnormalized_low_polarization": (
            partition_dot_low_polarization
        ),
        "low_evolution_identity": (
            "At tau=0, N[U_j,Z_0]+N[Z_0,U_j]="
            "-i(Z_0 dot a_j)U_j. Symmetry and trilinearity of E make "
            "the sum of the three low-evolution rows proportional to "
            "Z_0 dot (a+b+c), hence exactly zero."
        ),
        "high_evolution_identity": (
            "At tau=0, cross-pressure pairs vanish by transversality "
            "and the zero-output pressure gauge. The kinetic trace term "
            "vanishes after contraction because r dot Z_0=0, leaving "
            "exactly -r^T G(a,b,c)Z_0/64."
        ),
        "all_checks_pass": bool(
            center_sum == (0, 0, 0)
            and offset_coefficients_cancel
            and partition_dot_low_polarization == 0
        ),
    }


def _jet_dot(first: list[Jet], second: list[Jet]) -> Jet:
    value = Jet.constant(0.0)
    for left, right in zip(first, second):
        value += left * right
    return value


def _jet_norm_squared(vector: list[Jet]) -> Jet:
    value = Jet.constant(0.0)
    for component in vector:
        value += component * component
    return value


def _jet_project(vector: list[Jet], wave: list[Jet]) -> list[Jet]:
    coefficient = _jet_dot(wave, vector).divide_by_real_jet(
        _jet_norm_squared(wave)
    )
    return [
        value - component * coefficient
        for value, component in zip(vector, wave)
    ]


def _jet_bilinear_ns(
    first_wave: list[Jet],
    first_value: list[Jet],
    second_wave: list[Jet],
    second_value: list[Jet],
    *,
    exact_output_wave: list[Jet] | None = None,
) -> tuple[list[Jet], list[Jet]]:
    output_wave = exact_output_wave or [
        left + right
        for left, right in zip(first_wave, second_wave)
    ]
    raw = [
        _jet_dot(first_value, second_wave) * value
        + _jet_dot(second_value, first_wave) * other
        for value, other in zip(second_value, first_value)
    ]
    projected = _jet_project(raw, output_wave)
    return (
        [(-1j) * value for value in projected],
        output_wave,
    )


def _jet_pressure_pair(
    first_wave: list[Jet],
    first_value: list[Jet],
    second_wave: list[Jet],
    second_value: list[Jet],
    *,
    identically_zero_output: bool = False,
) -> Jet:
    if identically_zero_output:
        return Jet.constant(0.0)
    output_wave = [
        left + right
        for left, right in zip(first_wave, second_wave)
    ]
    numerator = -(
        _jet_dot(output_wave, first_value)
        * _jet_dot(output_wave, second_value)
    )
    return numerator.divide_by_real_jet(
        _jet_norm_squared(output_wave)
    )


def _jet_energy_trilinear(
    first_wave: list[Jet],
    first_value: list[Jet],
    second_wave: list[Jet],
    second_value: list[Jet],
    third_wave: list[Jet],
    third_value: list[Jet],
    *,
    first_third_zero_pressure: bool = False,
) -> list[Jet]:
    kinetic = [
        (
            _jet_dot(first_value, second_value) * third
            + _jet_dot(first_value, third_value) * second
            + _jet_dot(second_value, third_value) * first
        )
        / 6.0
        for first, second, third in zip(
            first_value,
            second_value,
            third_value,
        )
    ]
    first_second_pressure = _jet_pressure_pair(
        first_wave,
        first_value,
        second_wave,
        second_value,
    )
    first_third_pressure = _jet_pressure_pair(
        first_wave,
        first_value,
        third_wave,
        third_value,
        identically_zero_output=first_third_zero_pressure,
    )
    second_third_pressure = _jet_pressure_pair(
        second_wave,
        second_value,
        third_wave,
        third_value,
    )
    pressure = [
        (
            first_second_pressure * third
            + first_third_pressure * second
            + second_third_pressure * first
        )
        / 3.0
        for first, second, third in zip(
            first_value,
            second_value,
            third_value,
        )
    ]
    return [
        left + right for left, right in zip(kinetic, pressure)
    ]


def _as_jet_vector(
    values: list[Interval] | tuple[float, float, float],
    phase: complex = 1.0,
) -> list[Jet]:
    return [
        Jet.constant(ComplexInterval(value, Interval.point(0.0)))
        * phase
        if isinstance(value, Interval)
        else Jet.constant(value * phase)
        for value in values
    ]


def _complete_tau_derivative_interval(
    carrier_multiple: int,
) -> dict[str, Any]:
    first_interval, second_interval, third_interval = (
        _relaxed_high_waves(carrier_multiple)
    )
    first_wave = _as_jet_vector(first_interval)
    second_wave = _as_jet_vector(second_interval)
    third_wave = _as_jet_vector(third_interval)
    first_value = _jet_project(
        _as_jet_vector(
            [
                Interval.point(-4.0),
                Interval.point(-3.0),
                Interval.point(1.0),
            ]
        ),
        first_wave,
    )
    second_value = _jet_project(
        _as_jet_vector(
            [
                Interval.point(-3.0),
                Interval.point(-1.0),
                Interval.point(2.0),
            ]
        ),
        second_wave,
    )
    third_value = [
        1j * value
        for value in _jet_project(
            _as_jet_vector(
                [
                    Interval.point(-3.0),
                    Interval.point(7.0),
                    Interval.point(1.0),
                ]
            ),
            third_wave,
        )
    ]
    tau_maximum = 1.0 / (
        OFFSET_SPACING * carrier_multiple
    )
    tau = _jet_variable(Interval(0.0, tau_maximum))
    low_wave = [-tau, -tau, -tau]
    low_value = _as_jet_vector((1.0, -1.0, 0.0), phase=1j)
    high_legs = (
        (first_wave, first_value),
        (second_wave, second_value),
        (third_wave, third_value),
    )
    output = [Jet.constant(0.0) for _ in range(3)]

    high_rows = (
        (high_legs[0], high_legs[1], high_legs[2]),
        (high_legs[0], high_legs[2], high_legs[1]),
        (high_legs[1], high_legs[2], high_legs[0]),
    )
    for first, second, remaining in high_rows:
        exact_output = [-value for value in remaining[0]]
        generated, generated_wave = _jet_bilinear_ns(
            first[0],
            first[1],
            second[0],
            second[1],
            exact_output_wave=exact_output,
        )
        flux = _jet_energy_trilinear(
            remaining[0],
            remaining[1],
            low_wave,
            low_value,
            generated_wave,
            generated,
            first_third_zero_pressure=True,
        )
        output = [
            value + 6.0 * contribution
            for value, contribution in zip(output, flux)
        ]

    low_rows = (
        (high_legs[0], high_legs[1], high_legs[2]),
        (high_legs[1], high_legs[0], high_legs[2]),
        (high_legs[2], high_legs[0], high_legs[1]),
    )
    for interacting, remaining_first, remaining_second in low_rows:
        generated, generated_wave = _jet_bilinear_ns(
            interacting[0],
            interacting[1],
            low_wave,
            low_value,
        )
        flux = _jet_energy_trilinear(
            remaining_first[0],
            remaining_first[1],
            remaining_second[0],
            remaining_second[1],
            generated_wave,
            generated,
        )
        output = [
            value + 6.0 * contribution
            for value, contribution in zip(output, flux)
        ]

    scaled_derivative = Interval.point(0.0)
    for component in output:
        scaled_derivative += -component.derivative.imaginary / 64.0
    maximum_derivative = scaled_derivative.maximum_absolute_value()
    return {
        "tau_interval": [0.0, tau_maximum],
        "scaled_coefficient_tau_derivative_interval": [
            scaled_derivative.lower,
            scaled_derivative.upper,
        ],
        "maximum_absolute_tau_derivative": maximum_derivative,
        "mean_value_correction_upper": _up(
            tau_maximum * maximum_derivative
        ),
        "all_checks_pass": math.isfinite(maximum_derivative),
    }


def _rounding_self_audit() -> dict[str, Any]:
    cases = (
        Interval(-1.0, 2.0) + Interval(3.0, 4.0),
        Interval(-1.0, 2.0) * Interval(3.0, 4.0),
        Interval(2.0, 3.0) / Interval(4.0, 5.0),
        Interval(-2.0, 3.0).square(),
    )
    expected_points = (
        (-1.0 + 3.0, 2.0 + 4.0),
        (-1.0 * 4.0, 2.0 * 4.0),
        (2.0 / 5.0, 3.0 / 4.0),
        (0.0, 9.0),
    )
    containment = [
        interval.lower <= lower
        and interval.upper >= upper
        for interval, (lower, upper) in zip(cases, expected_points)
    ]
    return {
        "binary64_rounding": (
            "Every elementary endpoint is expanded by math.nextafter "
            "toward the corresponding infinity."
        ),
        "containment_checks": containment,
        "all_checks_pass": all(containment),
    }


def _certificate(carrier_multiple: int) -> dict[str, Any]:
    leading = _leading_stress_interval(carrier_multiple)
    derivative = _complete_tau_derivative_interval(carrier_multiple)
    center = _central_symbol_self_audit(carrier_multiple)
    tau_zero = _tau_zero_identity_self_audit(carrier_multiple)
    leading_scaled_lower = leading["scaled_coefficient_interval"][0]
    correction = derivative["mean_value_correction_upper"]
    complete_scaled_lower = _down(leading_scaled_lower - correction)
    square_root_two = math.sqrt(2.0)
    square_root_two_upper = _up(square_root_two)
    actual_lower = _down(
        complete_scaled_lower / square_root_two_upper
    )
    rounding = _rounding_self_audit()
    return {
        "carrier_multiple_relative_to_offset_box": carrier_multiple,
        "offset_spacing": OFFSET_SPACING,
        "physical_carrier": (
            f"R={OFFSET_SPACING * carrier_multiple}M"
        ),
        "relative_high_offset_half_width": 1.0 / carrier_multiple,
        "maximum_tau_equals_inverse_carrier": (
            1.0 / (OFFSET_SPACING * carrier_multiple)
        ),
        "leading_stress_certificate": leading,
        "complete_low_frequency_correction": derivative,
        "central_symbol_self_audit": center,
        "tau_zero_identity_self_audit": tau_zero,
        "complete_scaled_coefficient_lower": complete_scaled_lower,
        "complete_actual_positive_quartet_coefficient_lower": actual_lower,
        "rounding_self_audit": rounding,
        "proof": (
            "The leading interval encloses -r^T G(a,b,c)Z_0/64 over "
            "a relaxed six-dimensional box containing every continuum "
            "offset. The complete quartic coefficient S(tau) equals this "
            "leading value at tau=0. Directed interval automatic "
            "differentiation encloses dS/dtau for "
            "0<=tau<=1/R. The mean-value theorem therefore gives the "
            "reported complete lower bound. Division by sqrt(2) restores "
            "the normalized low polarization."
        ),
        "all_checks_pass": bool(
            leading["all_checks_pass"]
            and derivative["all_checks_pass"]
            and center["all_checks_pass"]
            and tau_zero["all_checks_pass"]
            and rounding["all_checks_pass"]
            and actual_lower > 0.0
        ),
    }


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--carrier-multiple",
        type=int,
        default=DEFAULT_CARRIER_MULTIPLE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "work/ns_collision/results/"
            "dense_spaced_continuum_positivity_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.carrier_multiple < 32:
        raise ValueError("carrier multiple must be at least 32")
    certificate = _certificate(args.carrier_multiple)
    result = {
        "kind": "dense_spaced_continuum_positivity_audit",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "complete_continuum_positive_quartic_coefficient_certified"
        ),
        "certificate": certificate,
        "prior_scalar_gate": (
            "work/ns_collision/results/"
            "scalar_local_energy_regeneration_gate_audit_v1.json"
        ),
        "prior_scalar_gate_sha256": _sha256(
            ROOT
            / "work/ns_collision/results/"
            "scalar_local_energy_regeneration_gate_audit_v1.json"
        ),
        "all_positive_checks_pass": certificate["all_checks_pass"],
    }
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("continuum positivity certificate failed")
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
