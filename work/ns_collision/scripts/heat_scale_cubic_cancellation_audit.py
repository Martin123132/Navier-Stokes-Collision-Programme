"""Spectral audit of the heat-scale vortex-stretching cancellation."""

from __future__ import annotations

import json

import numpy as np


def _spectral_field(
    grid_size: int = 24, maximum_wave_number: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260717)
    field = rng.normal(size=(3, grid_size, grid_size, grid_size))
    field_hat = np.fft.fftn(field, axes=(1, 2, 3))

    frequencies = np.fft.fftfreq(grid_size) * grid_size
    kx, ky, kz = np.meshgrid(
        frequencies, frequencies, frequencies, indexing="ij"
    )
    wave_vector = np.array([kx, ky, kz])
    wave_number_squared = np.sum(wave_vector**2, axis=0)
    nonzero_wave_number = np.where(wave_number_squared == 0, 1, wave_number_squared)
    retained = (wave_number_squared > 0) & (
        wave_number_squared <= maximum_wave_number**2
    )

    longitudinal_part = np.sum(wave_vector * field_hat, axis=0)
    for component in range(3):
        field_hat[component] = np.where(
            retained,
            field_hat[component]
            - wave_vector[component] * longitudinal_part / nonzero_wave_number,
            0,
        )

    # Re-transforming a real field removes roundoff-level violations of
    # conjugate symmetry before the differential identities are tested.
    velocity = np.fft.ifftn(field_hat, axes=(1, 2, 3)).real
    velocity_hat = np.fft.fftn(velocity, axes=(1, 2, 3))
    return velocity_hat, wave_vector, wave_number_squared


def _vorticity_hat(
    velocity_hat: np.ndarray, wave_vector: np.ndarray
) -> np.ndarray:
    kx, ky, kz = wave_vector
    result = np.empty_like(velocity_hat)
    result[0] = 1j * (ky * velocity_hat[2] - kz * velocity_hat[1])
    result[1] = 1j * (kz * velocity_hat[0] - kx * velocity_hat[2])
    result[2] = 1j * (kx * velocity_hat[1] - ky * velocity_hat[0])
    return result


def _strain_hat(
    velocity_hat: np.ndarray, wave_vector: np.ndarray
) -> np.ndarray:
    grid_shape = velocity_hat.shape[1:]
    result = np.empty((3, 3, *grid_shape), dtype=complex)
    for row in range(3):
        for column in range(3):
            result[row, column] = 0.5j * (
                wave_vector[column] * velocity_hat[row]
                + wave_vector[row] * velocity_hat[column]
            )
    return result


def _physical(field_hat: np.ndarray) -> np.ndarray:
    spatial_axes = tuple(range(field_hat.ndim - 3, field_hat.ndim))
    return np.fft.ifftn(field_hat, axes=spatial_axes).real


def _mean_contraction(matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.mean(
            np.einsum("ijxyz,ixyz,jxyz->xyz", matrix, left, right),
            dtype=np.float64,
        )
    )


def _explicit_triad_defect(heat_time: float = 0.37) -> tuple[float, float]:
    """Return the measured and exact defect for a nonzero Fourier triad."""
    grid_size = 8
    coordinates = 2 * np.pi * np.indices((grid_size,) * 3) / grid_size
    x, y = coordinates[0], coordinates[1]
    velocity = (
        np.array([0, -1, -1])[:, None, None, None] * np.cos(x)
        + np.array([-1, 0, -1])[:, None, None, None] * np.cos(y)
        + np.array([1, -1, 1])[:, None, None, None] * np.sin(x + y)
    )
    velocity_hat = np.fft.fftn(velocity, axes=(1, 2, 3))
    frequencies = np.fft.fftfreq(grid_size) * grid_size
    wave_vector = np.array(
        np.meshgrid(frequencies, frequencies, frequencies, indexing="ij")
    )
    wave_number_squared = np.sum(wave_vector**2, axis=0)
    vorticity = _physical(_vorticity_hat(velocity_hat, wave_vector))
    strain_hat = _strain_hat(velocity_hat, wave_vector)
    strain = _physical(strain_hat)
    smoothed_strain = _physical(
        np.exp(-heat_time * wave_number_squared)[None, None] * strain_hat
    )
    measured = _mean_contraction(
        strain - smoothed_strain, vorticity, vorticity
    )
    exact = 0.5 * (1 - np.exp(-heat_time)) ** 2
    return measured, exact


def audit() -> dict[str, float | bool]:
    velocity_hat, wave_vector, wave_number_squared = _spectral_field()
    vorticity_hat = _vorticity_hat(velocity_hat, wave_vector)
    strain_hat = _strain_hat(velocity_hat, wave_vector)
    vorticity = _physical(vorticity_hat)
    strain = _physical(strain_hat)
    minus_laplacian_strain = _physical(
        wave_number_squared[None, None] * strain_hat
    )
    laplacian_squared_strain = _physical(
        (wave_number_squared**2)[None, None] * strain_hat
    )

    orthogonality_residual = _mean_contraction(
        minus_laplacian_strain, vorticity, vorticity
    )
    strain_norm_squared = float(np.mean(np.sum(strain**2, axis=(0, 1))))
    half_vorticity_norm_squared = 0.5 * float(
        np.mean(np.sum(vorticity**2, axis=0))
    )

    heat_time = 1.0e-4
    smoothed_strain = _physical(
        np.exp(-heat_time * wave_number_squared)[None, None] * strain_hat
    )
    defect_stretching = _mean_contraction(
        strain - smoothed_strain, vorticity, vorticity
    )
    second_derivative = -_mean_contraction(
        laplacian_squared_strain, vorticity, vorticity
    )
    second_order_prediction = 0.5 * second_derivative

    polarization_time = 0.17
    heat_factor = np.exp(-polarization_time * wave_number_squared)
    smoothed_vorticity = _physical(heat_factor[None] * vorticity_hat)
    smoothed_minus_laplacian_strain = _physical(
        heat_factor[None, None]
        * wave_number_squared[None, None]
        * strain_hat
    )
    polarized_residual = _mean_contraction(
        smoothed_minus_laplacian_strain, vorticity, vorticity
    ) + 2 * _mean_contraction(
        minus_laplacian_strain, smoothed_vorticity, vorticity
    )

    second_order_ratio = defect_stretching / (heat_time**2)
    second_order_relative_error = abs(
        (second_order_ratio - second_order_prediction) / second_order_prediction
    )
    triad_defect, exact_triad_defect = _explicit_triad_defect()

    result: dict[str, float | bool] = {
        "strain_vorticity_l2_isometry": abs(
            strain_norm_squared - half_vorticity_norm_squared
        )
        < 1.0e-12,
        "cubic_orthogonality_residual": orthogonality_residual,
        "cubic_orthogonality_holds": abs(orthogonality_residual) < 1.0e-12,
        "polarized_orthogonality_residual": polarized_residual,
        "polarized_orthogonality_holds": abs(polarized_residual) < 1.0e-12,
        "defect_divided_by_heat_time": defect_stretching / heat_time,
        "defect_divided_by_heat_time_squared": second_order_ratio,
        "predicted_second_order_coefficient": second_order_prediction,
        "second_order_relative_error": second_order_relative_error,
        "heat_defect_begins_at_second_order": second_order_relative_error < 1.0e-3,
        "explicit_triad_defect": triad_defect,
        "explicit_triad_formula": exact_triad_defect,
        "explicit_triad_formula_holds": bool(
            abs(triad_defect - exact_triad_defect) < 1.0e-12
        ),
        "defect_has_no_universal_sign": bool(triad_defect > 0),
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
