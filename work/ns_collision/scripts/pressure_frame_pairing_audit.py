"""Adversarial audit of pointwise pressure/frame pairing at a strain maximum."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize


GRID_SIZE = 20
RANDOM_SEED = 81
MAXIMUM_WAVE_NUMBER = 3
VELOCITY_RMS = 10.0
HEAT_SCALE = 0.08
VISCOSITY = 1.0
STARTING_GRID_INDEX = np.array([11, 9, 16])


def _prepare_coefficients(
    field_hat: np.ndarray, wave_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tensor_axes = tuple(range(field_hat.ndim - 3))
    retained = np.any(np.abs(field_hat) > 1.0e-11, axis=tensor_axes)
    modes = wave_vector[:, retained].T
    coefficients = np.moveaxis(
        field_hat[..., retained] / GRID_SIZE**3, -1, 0
    )
    return modes, coefficients


def _evaluate(
    modes: np.ndarray,
    coefficients: np.ndarray,
    point: np.ndarray,
    derivatives: tuple[int, ...] = (),
) -> np.ndarray:
    phase = np.exp(1j * (modes @ point))
    for direction in derivatives:
        phase *= 1j * modes[:, direction]
    return np.einsum("m,m...->...", phase, coefficients).real


def _build_spectral_fields() -> dict[str, object]:
    rng = np.random.default_rng(RANDOM_SEED)
    velocity = rng.normal(size=(3, GRID_SIZE, GRID_SIZE, GRID_SIZE))
    velocity_hat = np.fft.fftn(velocity, axes=(1, 2, 3))

    frequencies = np.fft.fftfreq(GRID_SIZE) * GRID_SIZE
    wave_vector = np.array(
        np.meshgrid(frequencies, frequencies, frequencies, indexing="ij")
    )
    wave_number_squared = np.sum(wave_vector**2, axis=0)
    nonzero_wave_number = np.where(
        wave_number_squared == 0, 1, wave_number_squared
    )
    retained = (wave_number_squared > 0) & (
        wave_number_squared <= MAXIMUM_WAVE_NUMBER**2
    )
    longitudinal = np.sum(wave_vector * velocity_hat, axis=0)
    velocity_hat = (
        velocity_hat
        - wave_vector * longitudinal / nonzero_wave_number
    ) * retained

    velocity = np.fft.ifftn(velocity_hat, axes=(1, 2, 3)).real
    velocity *= VELOCITY_RMS / np.sqrt(
        np.mean(np.sum(velocity**2, axis=0))
    )
    velocity_hat = np.fft.fftn(velocity, axes=(1, 2, 3))
    divergence_hat = 1j * np.sum(wave_vector * velocity_hat, axis=0)

    gradient_hat = np.empty(
        (3, 3, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=complex
    )
    for row in range(3):
        for column in range(3):
            gradient_hat[row, column] = (
                1j * wave_vector[column] * velocity_hat[row]
            )
    strain_hat = (gradient_hat + gradient_hat.swapaxes(0, 1)) / 2

    gradient = np.fft.ifftn(gradient_hat, axes=(2, 3, 4)).real
    advection = np.einsum("jxyz,ijxyz->ixyz", velocity, gradient)
    advection_hat = np.fft.fftn(advection, axes=(1, 2, 3))
    advection_longitudinal = np.sum(wave_vector * advection_hat, axis=0)
    velocity_time_derivative_hat = (
        -advection_hat
        + wave_vector
        * advection_longitudinal
        / nonzero_wave_number
        - VISCOSITY * wave_number_squared * velocity_hat
    )
    velocity_time_derivative_hat *= wave_number_squared > 0
    strain_time_derivative_hat = np.empty_like(strain_hat)
    for row in range(3):
        for column in range(3):
            strain_time_derivative_hat[row, column] = 0.5j * (
                wave_vector[column] * velocity_time_derivative_hat[row]
                + wave_vector[row] * velocity_time_derivative_hat[column]
            )

    vorticity_hat = np.empty_like(velocity_hat)
    vorticity_hat[0] = 1j * (
        wave_vector[1] * velocity_hat[2]
        - wave_vector[2] * velocity_hat[1]
    )
    vorticity_hat[1] = 1j * (
        wave_vector[2] * velocity_hat[0]
        - wave_vector[0] * velocity_hat[2]
    )
    vorticity_hat[2] = 1j * (
        wave_vector[0] * velocity_hat[1]
        - wave_vector[1] * velocity_hat[0]
    )

    strain = np.fft.ifftn(strain_hat, axes=(2, 3, 4)).real
    vorticity = np.fft.ifftn(vorticity_hat, axes=(1, 2, 3)).real
    pressure_source = np.sum(strain**2, axis=(0, 1)) - 0.5 * np.sum(
        vorticity**2, axis=0
    )
    pressure_source_hat = np.fft.fftn(pressure_source)

    pressure_hat = np.empty_like(strain_hat)
    for row in range(3):
        for column in range(3):
            multiplier = -(
                wave_vector[row]
                * wave_vector[column]
                / nonzero_wave_number
            )
            if row == column:
                multiplier += 1.0 / 3.0
            pressure_hat[row, column] = (
                multiplier
                * pressure_source_hat
                * (wave_number_squared > 0)
            )
    pressure_low_hat = (
        np.exp(-HEAT_SCALE * wave_number_squared)[None, None]
        * pressure_hat
    )
    pressure_defect_hat = pressure_hat - pressure_low_hat
    pressure = np.fft.ifftn(pressure_hat, axes=(2, 3, 4)).real
    pressure_low = np.fft.ifftn(
        pressure_low_hat, axes=(2, 3, 4)
    ).real
    pressure_defect = pressure - pressure_low
    pressure_potential_hat = (
        pressure_source_hat
        / nonzero_wave_number
        * (wave_number_squared > 0)
    )
    pressure_potential_low_hat = (
        np.exp(-HEAT_SCALE * wave_number_squared)
        * pressure_potential_hat
    )
    pressure_potential_defect_hat = (
        pressure_potential_hat - pressure_potential_low_hat
    )

    def potential_gradient(potential_hat: np.ndarray) -> np.ndarray:
        result = np.empty(
            (3, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=float
        )
        for direction in range(3):
            result[direction] = np.fft.ifftn(
                1j * wave_vector[direction] * potential_hat
            ).real
        return result

    strain_gradient = np.empty(
        (3, 3, 3, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=float
    )
    for direction in range(3):
        strain_gradient[direction] = np.fft.ifftn(
            1j * wave_vector[direction] * strain_hat,
            axes=(2, 3, 4),
        ).real
    pressure_strain_contraction = float(
        np.mean(np.einsum("ijxyz,ijxyz->xyz", pressure, strain))
    )
    low_pressure_strain_contraction = float(
        np.mean(np.einsum("ijxyz,ijxyz->xyz", pressure_low, strain))
    )
    defect_pressure_strain_contraction = float(
        np.mean(
            np.einsum("ijxyz,ijxyz->xyz", pressure_defect, strain)
        )
    )

    return {
        "strain": _prepare_coefficients(strain_hat, wave_vector),
        "velocity": _prepare_coefficients(velocity_hat, wave_vector),
        "strain_time_derivative": _prepare_coefficients(
            strain_time_derivative_hat, wave_vector
        ),
        "vorticity": _prepare_coefficients(vorticity_hat, wave_vector),
        "pressure": _prepare_coefficients(pressure_hat, wave_vector),
        "pressure_low": _prepare_coefficients(
            pressure_low_hat, wave_vector
        ),
        "pressure_defect": _prepare_coefficients(
            pressure_defect_hat, wave_vector
        ),
        "maximum_divergence_coefficient": float(
            np.max(np.abs(divergence_hat)) / GRID_SIZE**3
        ),
        "global_pressure_strain_contraction": pressure_strain_contraction,
        "global_low_pressure_strain_contraction": (
            low_pressure_strain_contraction
        ),
        "global_defect_pressure_strain_contraction": (
            defect_pressure_strain_contraction
        ),
        "velocity_gradient_grid": gradient,
        "strain_grid": strain,
        "strain_gradient_grid": strain_gradient,
        "pressure_grid": pressure,
        "pressure_low_grid": pressure_low,
        "pressure_defect_grid": pressure_defect,
        "pressure_potential_gradient_grid": potential_gradient(
            pressure_potential_hat
        ),
        "pressure_potential_low_gradient_grid": potential_gradient(
            pressure_potential_low_hat
        ),
        "pressure_potential_defect_gradient_grid": potential_gradient(
            pressure_potential_defect_hat
        ),
    }


def audit() -> dict[str, object]:
    fields = _build_spectral_fields()
    strain_modes, strain_coefficients = fields["strain"]

    def strain(point: np.ndarray, derivatives: tuple[int, ...] = ()) -> np.ndarray:
        return _evaluate(
            strain_modes, strain_coefficients, point, derivatives
        )

    def eigen_data(point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(strain(point))

    def objective(point: np.ndarray) -> float:
        return -float(eigen_data(point)[0][2])

    def objective_gradient(point: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = eigen_data(point)
        maximum_vector = eigenvectors[:, 2]
        return -np.array(
            [
                maximum_vector
                @ strain(point, (direction,))
                @ maximum_vector
                for direction in range(3)
            ]
        )

    starting_point = 2 * np.pi * STARTING_GRID_INDEX / GRID_SIZE
    optimization = minimize(
        objective,
        starting_point,
        jac=objective_gradient,
        method="BFGS",
        options={"gtol": 1.0e-11, "maxiter": 500},
    )
    point = optimization.x
    eigenvalues, eigenvectors = eigen_data(point)
    maximum_vector = eigenvectors[:, 2]
    gaps = eigenvalues[2] - eigenvalues[:2]

    maximum_gradient = -objective_gradient(point)
    maximum_hessian = np.zeros((3, 3))
    for first_direction in range(3):
        for second_direction in range(3):
            value = (
                maximum_vector
                @ strain(point, (first_direction, second_direction))
                @ maximum_vector
            )
            for lower_index in range(2):
                first_coupling = (
                    eigenvectors[:, lower_index]
                    @ strain(point, (first_direction,))
                    @ maximum_vector
                )
                second_coupling = (
                    eigenvectors[:, lower_index]
                    @ strain(point, (second_direction,))
                    @ maximum_vector
                )
                value += (
                    2
                    * first_coupling
                    * second_coupling
                    / gaps[lower_index]
                )
            maximum_hessian[first_direction, second_direction] = value
    maximum_hessian_eigenvalues = np.linalg.eigvalsh(maximum_hessian)

    frame_penalty = 0.0
    for direction in range(3):
        strain_derivative = strain(point, (direction,))
        for lower_index in range(2):
            coupling = (
                eigenvectors[:, lower_index]
                @ strain_derivative
                @ maximum_vector
            )
            frame_penalty += (
                2 * VISCOSITY * coupling**2 / gaps[lower_index]
            )

    def field_value(name: str) -> np.ndarray:
        modes, coefficients = fields[name]
        return _evaluate(modes, coefficients, point)

    vorticity = field_value("vorticity")
    pressure = field_value("pressure")
    pressure_low = field_value("pressure_low")
    pressure_defect = field_value("pressure_defect")
    parallel_vorticity = float(maximum_vector @ vorticity)
    perpendicular_vorticity_squared = float(
        vorticity @ vorticity - parallel_vorticity**2
    )
    local_reaction = float(
        np.sum(eigenvalues**2) / 3
        - eigenvalues[2] ** 2
        + perpendicular_vorticity_squared / 12
        - parallel_vorticity**2 / 6
    )
    projected_pressure = float(maximum_vector @ pressure @ maximum_vector)
    projected_pressure_low = float(
        maximum_vector @ pressure_low @ maximum_vector
    )
    projected_pressure_defect = float(
        maximum_vector @ pressure_defect @ maximum_vector
    )
    reaction_without_scalar_diffusion = (
        local_reaction - projected_pressure - frame_penalty
    )
    scalar_diffusion = VISCOSITY * float(np.trace(maximum_hessian))
    material_growth = scalar_diffusion + reaction_without_scalar_diffusion
    velocity = field_value("velocity")
    strain_time_derivative = field_value("strain_time_derivative")
    direct_material_strain = strain_time_derivative.copy()
    for direction in range(3):
        direct_material_strain += velocity[direction] * strain(
            point, (direction,)
        )
    direct_material_growth = float(
        maximum_vector @ direct_material_strain @ maximum_vector
    )
    material_growth_residual = abs(
        direct_material_growth - material_growth
    )
    split_residual = abs(
        projected_pressure
        - projected_pressure_low
        - projected_pressure_defect
    )

    algebraic_reaction = local_reaction - projected_pressure
    amplitude_threshold = (
        VELOCITY_RMS * frame_penalty / algebraic_reaction
    )
    material_growth_amplitude_threshold = (
        VELOCITY_RMS
        * (frame_penalty - scalar_diffusion)
        / algebraic_reaction
    )

    result: dict[str, object] = {
        "field": (
            "deterministic divergence-free trigonometric polynomial on T^3"
        ),
        "random_seed": RANDOM_SEED,
        "grid_size_used_to_generate_coefficients": GRID_SIZE,
        "maximum_wave_number": MAXIMUM_WAVE_NUMBER,
        "velocity_rms": VELOCITY_RMS,
        "heat_scale": HEAT_SCALE,
        "viscosity": VISCOSITY,
        "maximum_divergence_coefficient": fields[
            "maximum_divergence_coefficient"
        ],
        "divergence_free_check": bool(
            fields["maximum_divergence_coefficient"] < 1.0e-12
        ),
        "optimized_point_mod_2pi": list(np.mod(point, 2 * np.pi)),
        "strain_eigenvalues": list(eigenvalues),
        "simple_maximum_eigenvalue_gaps": list(gaps),
        "maximum_eigenvalue_gradient_norm": float(
            np.linalg.norm(maximum_gradient)
        ),
        "maximum_eigenvalue_hessian_eigenvalues": list(
            maximum_hessian_eigenvalues
        ),
        "is_strict_continuous_local_maximum": bool(
            np.linalg.norm(maximum_gradient) < 1.0e-7
            and maximum_hessian_eigenvalues[-1] < -1.0e-3
        ),
        "local_reaction": local_reaction,
        "negative_projected_pressure": -projected_pressure,
        "negative_low_pressure": -projected_pressure_low,
        "negative_pressure_collision_defect": -projected_pressure_defect,
        "viscous_frame_penalty": frame_penalty,
        "pressure_split_residual": split_residual,
        "pressure_split_verified": bool(split_residual < 1.0e-10),
        "reaction_without_scalar_diffusion": reaction_without_scalar_diffusion,
        "pointwise_pressure_frame_gate_fails": bool(
            reaction_without_scalar_diffusion > 1.0
        ),
        "scalar_maximum_diffusion": scalar_diffusion,
        "instantaneous_material_growth": material_growth,
        "direct_fourier_navier_stokes_material_growth": (
            direct_material_growth
        ),
        "material_growth_identity_residual": material_growth_residual,
        "material_growth_identity_verified": bool(
            material_growth_residual < 1.0e-9
        ),
        "scalar_diffusion_does_not_restore_pointwise_decay": bool(
            material_growth > 1.0
        ),
        "amplitude_scaling": (
            "local/pressure terms scale as A^2; viscous frame and scalar "
            "diffusion scale as nu*A"
        ),
        "velocity_rms_threshold_for_positive_reaction_without_scalar_diffusion": (
            amplitude_threshold
        ),
        "velocity_rms_threshold_for_positive_material_growth": (
            material_growth_amplitude_threshold
        ),
        "universal_pointwise_signed_pairing_is_false": True,
        "global_pressure_strain_orthogonality": bool(
            abs(fields["global_pressure_strain_contraction"]) < 1.0e-10
        ),
        "global_heat_split_pressure_orthogonality": bool(
            abs(fields["global_low_pressure_strain_contraction"])
            < 1.0e-10
            and abs(fields["global_defect_pressure_strain_contraction"])
            < 1.0e-10
        ),
        "localized_pressure_identity": (
            "integral phi*B_s^P:S=-integral grad(p-p_s)."
            "(grad(u)^T*grad(phi))"
        ),
        "trajectory_or_replica_averaging_is_required": True,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
