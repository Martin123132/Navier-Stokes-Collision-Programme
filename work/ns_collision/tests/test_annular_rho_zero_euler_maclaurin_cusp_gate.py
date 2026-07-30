"""Tests for the exact-box Euler-Maclaurin and Leray-cusp gate."""

from __future__ import annotations

from collections import defaultdict
import json
import math
import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "work/ns_collision/scripts"
DIRECT_RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_direct_continuum_quadrature_v1.json"
)
BOUNDARY_RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_euler_maclaurin_boundary_pilot_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_rho_zero_direct_continuum_quadrature import (  # noqa: E402
    _grid_shape,
    _profile_samples,
    _quadrature_row,
)
from annular_rho_zero_euler_maclaurin_boundary_pilot import (  # noqa: E402
    _sixth_order_corrected_samples,
)


class AnnularEulerMaclaurinCuspGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.direct = json.loads(DIRECT_RESULT.read_text(encoding="utf-8"))
        cls.boundary = json.loads(
            BOUNDARY_RESULT.read_text(encoding="utf-8")
        )

    def test_exact_box_grid_and_profile_faces(self) -> None:
        self.assertEqual(_grid_shape(8), (98, 18, 18))
        _, _, _, profile = _profile_samples(8)
        for axis in range(1, 4):
            lower = np.take(profile, 0, axis=axis)
            upper = np.take(profile, -1, axis=axis)
            self.assertEqual(float(np.max(np.abs(lower))), 0.0)
            self.assertEqual(float(np.max(np.abs(upper))), 0.0)

    def test_direct_rows_replay_second_order_sequence(self) -> None:
        rows = self.direct["rows"]
        self.assertEqual([row["size"] for row in rows], [8, 16, 32, 64])
        expected = [
            -2.658690240168762e-7,
            -2.906753381816596e-7,
            -2.971856920621700e-7,
            -2.9883445926209503e-7,
        ]
        for row, value in zip(rows, expected):
            self.assertAlmostEqual(
                row["combined_continuum_quadrature"],
                value,
                delta=2.0e-20,
            )
            self.assertTrue(row["all_numerical_checks_pass"])
            self.assertLess(
                row["energy_trace_relative_residual"],
                1.0e-14,
            )

        differences = [
            expected[index] - expected[index + 1]
            for index in range(3)
        ]
        self.assertGreater(differences[0] / differences[1], 3.7)
        self.assertGreater(differences[1] / differences[2], 3.9)

    def test_direct_fft_matches_independent_mode_dictionary(self) -> None:
        size = 4
        modes: dict[tuple[int, int, int], np.ndarray] = {}
        x, y, z, profile = _profile_samples(size)
        for i, x_value in enumerate(x):
            for j, y_value in enumerate(y):
                for k, z_value in enumerate(z):
                    value = profile[:, i, j, k]
                    if not np.any(value):
                        continue
                    wave = (
                        int(round(size * x_value)),
                        int(round(size * y_value)),
                        int(round(size * z_value)),
                    )
                    modes[wave] = value.copy()
                    modes[tuple(-entry for entry in wave)] = value.copy()

        def project(
            wave: tuple[int, int, int],
            value: np.ndarray,
        ) -> np.ndarray:
            frequency = np.asarray(wave, dtype=float)
            frequency_squared = float(frequency @ frequency)
            if frequency_squared == 0.0:
                return np.zeros(3)
            return (
                value
                - frequency
                * float(frequency @ value)
                / frequency_squared
            )

        raw_velocity: defaultdict[
            tuple[int, int, int], np.ndarray
        ] = defaultdict(lambda: np.zeros(3))
        for first_wave, first_value in modes.items():
            for second_wave, second_value in modes.items():
                output = tuple(
                    first_wave[axis] + second_wave[axis]
                    for axis in range(3)
                )
                raw_velocity[output] += (
                    float(first_value @ np.asarray(second_wave, dtype=float))
                    * second_value
                )
        velocity = {
            wave: project(wave, value)
            for wave, value in raw_velocity.items()
            if wave != (0, 0, 0)
        }

        raw_acceleration: defaultdict[
            tuple[int, int, int], np.ndarray
        ] = defaultdict(lambda: np.zeros(3))
        for first_wave, first_value in modes.items():
            for second_wave, second_value in velocity.items():
                output = tuple(
                    first_wave[axis] + second_wave[axis]
                    for axis in range(3)
                )
                raw_acceleration[output] += (
                    0.5
                    * float(
                        first_value
                        @ np.asarray(second_wave, dtype=float)
                    )
                    * second_value
                )
        for first_wave, first_value in velocity.items():
            for second_wave, second_value in modes.items():
                output = tuple(
                    first_wave[axis] + second_wave[axis]
                    for axis in range(3)
                )
                raw_acceleration[output] += (
                    0.5
                    * float(
                        first_value
                        @ np.asarray(second_wave, dtype=float)
                    )
                    * second_value
                )
        acceleration_code = {
            wave: -project(wave, value)
            for wave, value in raw_acceleration.items()
            if wave != (0, 0, 0)
        }

        velocity_energy = np.asarray(
            [
                sum(value[axis] ** 2 for value in velocity.values())
                for axis in range(3)
            ]
        )
        acceleration_pair = np.asarray(
            [
                sum(
                    acceleration_code.get(wave, np.zeros(3))[axis]
                    * value[axis]
                    for wave, value in modes.items()
                )
                for axis in range(3)
            ]
        )
        alpha = math.sqrt(2.0) / 20.0
        normalization = size**-11
        first = (
            alpha
            * normalization
            * (velocity_energy[2] - velocity_energy[1])
        )
        second = (
            2.0
            * alpha
            * normalization
            * (acceleration_pair[2] - acceleration_pair[1])
        )
        fft = _quadrature_row(size)
        self.assertAlmostEqual(
            first,
            fft["first_form_continuum_quadrature"],
            delta=2.0e-20,
        )
        self.assertAlmostEqual(
            second,
            fft["second_form_continuum_quadrature"],
            delta=2.0e-20,
        )
        self.assertLess(
            abs(
                float(np.sum(velocity_energy))
                + 2.0 * float(np.sum(acceleration_pair))
            ),
            1.0e-10,
        )

    def test_face_coefficient_removes_leading_h2_error(self) -> None:
        rows = self.boundary["rows"]
        self.assertEqual([row["size"] for row in rows], [8, 16, 32, 64])
        coefficients = [row["face_correction_c2"] for row in rows]
        self.assertTrue(
            all(
                coefficients[index] < coefficients[index + 1]
                for index in range(3)
            )
        )
        corrected = [row["face_corrected_value"] for row in rows]
        differences = [
            corrected[index] - corrected[index + 1]
            for index in range(3)
        ]
        self.assertGreater(differences[0] / differences[1], 15.0)
        self.assertGreater(differences[1] / differences[2], 15.9)
        self.assertTrue(all(value < 0.0 for value in corrected))

    def test_sixth_order_packet_rule_replays_on_linear_integral(self) -> None:
        values = []
        for size in (8, 16, 32):
            corrected = _sixth_order_corrected_samples(size)
            values.append(
                2.0 * float(np.sum(corrected[2])) / size**3
            )
        first_difference = values[0] - values[1]
        second_difference = values[1] - values[2]
        self.assertGreater(first_difference / second_difference, 60.0)
        self.assertLess(first_difference / second_difference, 68.0)

    def test_origin_cusp_is_diagonal_and_cubic_after_subtraction(self) -> None:
        rows = self.direct["rows"]
        for row in rows:
            cusp = row["origin_leray_cusp_replay"]
            self.assertLess(cusp["maximum_covariance_off_diagonal"], 1.0e-18)
            self.assertLess(
                cusp["maximum_residual_over_rho_cubed"],
                0.11,
            )
        final_cusp = rows[-1]["origin_leray_cusp_replay"]
        diagonal = [
            final_cusp["covariance_matrix_trapezoid"][j][j]
            for j in range(3)
        ]
        self.assertGreater(diagonal[2], 0.039)
        self.assertLess(diagonal[0], 0.00021)
        self.assertLess(diagonal[1], 0.000002)

    def test_no_numerical_row_impersonates_interval_certificate(self) -> None:
        self.assertFalse(
            self.direct["certification"][
                "continuum_sign_interval_certified"
            ]
        )
        self.assertFalse(
            self.boundary["certification"][
                "h4_remainder_interval_certified"
            ]
        )
        self.assertFalse(
            self.boundary["certification"][
                "continuum_sign_interval_certified"
            ]
        )


if __name__ == "__main__":
    unittest.main()
