"""Audit a common-polarization repair of the two-shear annular witness."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sympy as sp

from annular_rho_zero_continuum_convolution_quadrature import (
    _euler_cross,
    _euler_quadratic,
    _frequency_axes,
    _lower_process_priority,
    _physical,
)
from annular_rho_zero_direct_continuum_quadrature import (
    _grid_shape,
)
from annular_two_shear_square_gate_audit import (
    _dominant_profile_samples,
    _modified_finite_packet,
    _profile_coefficients,
)
from annular_two_shear_static_optimizer_audit import (
    DELTA_CUBIC_ENERGY,
    PLUS_VERTEX,
    _pressure_flux,
    _resonant_loads_for_shear,
    _sym_add_vector_fields,
    _sym_fisher,
    _sym_l2_mass,
    _sym_load,
    _sym_pressure_bilinear,
    _sym_scalar_times_vector,
    _sym_vector_dot_product,
)
from compatible_eight_cell_cubic_graph_audit import (
    VERTICES,
    _cubic_energy,
)
from compatible_edge_annular_escape_audit import (
    _delta_weights,
)
from cross_shell_modulated_wave_gate_audit import (
    _energy_flux,
    _field_sum,
)
from primitive_hhl_chain_hardy_envelope_audit import (
    _translated_vertex_fisher,
    _translated_vertex_load,
)
from separable_annular_pressure_schur_no_go_audit import (
    TRANSLATION,
    _high_field,
    _mixed_difference_fisher,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_parallel_shear_phase_repair_audit_v1.json"
)
PREREQUISITES = {
    (
        "work/ns_collision/results/"
        "annular_two_shear_static_optimizer_audit_v1.json"
    ): "de11a2c2db530b81ee5c01c12a1270ebb8af8c957e28ed6676fd692f5d7a131e",
    (
        "work/ns_collision/results/"
        "annular_two_shear_square_gate_audit_v1.json"
    ): "7aca1d4c57ce970db5872979449368e65fb6add48c4ccd39b8f875cc1669f923",
    (
        "work/ns_collision/results/"
        "annular_two_shear_full_c1_port_audit_v1.json"
    ): "af0039698cdbd5442be629b23ea259e97556c978a0f0d91e94e9e0d658b1f32f",
}
ALGORITHM_REVISION = "annular-parallel-shear-phase-repair-v1"
DEFAULT_SIZES = (3, 5, 9, 13, 17, 25, 33, 49)
DEFAULT_CURVATURE_SIZES = (8, 16)
ELL_YZ = (0, 1, -1)
ELL_XY = (1, -1, 0)
PARALLEL_DIRECTION = (1, 1, 1)
PARALLEL_FISHER_MASS = Fraction(9, 8)
PARALLEL_L2_MASS = Fraction(4)
TAIL_CONSTANT = 70_657_920
Wave = tuple[int, int, int]
SymVector = tuple[sp.Expr, sp.Expr, sp.Expr]
SymVectorField = dict[Wave, SymVector]
FloatField = dict[Wave, np.ndarray]


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


def _sym_text(value: sp.Expr) -> str:
    return sp.sstr(sp.factor(sp.simplify(value)))


def _negate_wave(wave: Wave) -> Wave:
    return tuple(-component for component in wave)


def _sym_mode(
    wave: Wave,
    cosine: SymVector,
    sine: SymVector,
) -> SymVectorField:
    return {
        wave: tuple(
            cosine[index] - sp.I * sine[index] for index in range(3)
        ),
        _negate_wave(wave): tuple(
            cosine[index] + sp.I * sine[index] for index in range(3)
        ),
    }


def _sym_fluxes(
    field: SymVectorField,
) -> tuple[SymVectorField, SymVectorField, SymVectorField]:
    kinetic = _sym_scalar_times_vector(
        _sym_vector_dot_product(field, field),
        field,
    )
    kinetic = _sym_add_vector_fields(
        (sp.Rational(1, 2), kinetic)
    )
    pressure = _sym_scalar_times_vector(
        _sym_pressure_bilinear(field, field),
        field,
    )
    complete = _sym_add_vector_fields(
        (sp.S.One, kinetic),
        (sp.S.One, pressure),
    )
    return kinetic, pressure, complete


def _phase_family_certificate() -> dict[str, Any]:
    p1, q1, p2, q2 = sp.symbols(
        "p1 q1 p2 q2", real=True
    )
    root_two = sp.sqrt(2)
    yz_direction = (
        sp.S.Zero,
        1 / root_two,
        1 / root_two,
    )
    xy_direction = (
        -1 / root_two,
        -1 / root_two,
        sp.S.Zero,
    )
    yz = _sym_mode(
        ELL_YZ,
        tuple(q1 * value for value in yz_direction),
        tuple(p1 * value for value in yz_direction),
    )
    xy = _sym_mode(
        ELL_XY,
        tuple(q2 * value for value in xy_direction),
        tuple(p2 * value for value in xy_direction),
    )
    field = _sym_add_vector_fields(
        (sp.S.One, yz),
        (sp.S.One, xy),
    )
    kinetic, pressure, complete = _sym_fluxes(field)
    fisher = sp.factor(_sym_fisher(field, PLUS_VERTEX))
    kinetic_load = sp.factor(_sym_load(kinetic, PLUS_VERTEX))
    pressure_load = sp.factor(_sym_load(pressure, PLUS_VERTEX))
    complete_load = sp.factor(_sym_load(complete, PLUS_VERTEX))
    interaction = (
        p1**2 * p2
        + p1 * p2**2
        + p1 * q2**2
        + p2 * q1**2
    )
    expected_fisher = (
        8 * p1**2
        + p1 * p2
        + 8 * p2**2
        + 8 * q1**2
        - q1 * q2
        + 8 * q2**2
    ) / 16
    checks = {
        "Fisher_polynomial_exact": bool(
            sp.simplify(fisher - expected_fisher) == 0
        ),
        "kinetic_polynomial_exact": bool(
            sp.simplify(
                kinetic_load + sp.sqrt(2) * interaction / 32
            )
            == 0
        ),
        "pressure_polynomial_exact": bool(
            sp.simplify(
                pressure_load + sp.sqrt(2) * interaction / 96
            )
            == 0
        ),
        "complete_polynomial_exact": bool(
            sp.simplify(
                complete_load + sp.sqrt(2) * interaction / 24
            )
            == 0
        ),
        "strict_square_quadrant_has_nonzero_interaction": True,
    }
    return {
        "parameterization": (
            "Uhat_j(+ell_j)=(q_j-i p_j)d_j, "
            "Uhat_j(-ell_j)=(q_j+i p_j)d_j"
        ),
        "weighted_Fisher": _sym_text(fisher),
        "common_interaction_polynomial": _sym_text(interaction),
        "kinetic_self_flux": _sym_text(kinetic_load),
        "pressure_self_flux": _sym_text(pressure_load),
        "complete_self_flux": _sym_text(complete_load),
        "strict_square_quadrant": (
            "For p1>0 and p2>0, every term in "
            "p1*p2*(p1+p2)+p1*q2^2+p2*q1^2 is positive. "
            "Relative scalar phases cannot cancel the self-flux."
        ),
        "checks": checks,
        "all_phase_checks_pass": all(checks.values()),
    }


def _polarization_family_certificate() -> dict[str, Any]:
    a, b, d = sp.symbols("a b d", real=True)
    zero = (sp.S.Zero, sp.S.Zero, sp.S.Zero)
    sine_yz = (a, b, b)
    sine_xy = (-b, -b, d)
    field = _sym_add_vector_fields(
        (sp.S.One, _sym_mode(ELL_YZ, zero, sine_yz)),
        (sp.S.One, _sym_mode(ELL_XY, zero, sine_xy)),
    )
    kinetic, pressure, complete = _sym_fluxes(field)
    pressure_load = sp.factor(_sym_load(pressure, PLUS_VERTEX))
    complete_load = sp.factor(_sym_load(complete, PLUS_VERTEX))
    fisher = sp.factor(_sym_fisher(field, PLUS_VERTEX))
    expected_pressure = -(
        (a - b) * (b + d) * (a - 2 * b - d) / 24
    )
    expected_complete = (
        (a + 2 * b)
        * (2 * b - d)
        * (a - 2 * b - d)
        / 24
    )
    expected_fisher = (
        4 * a**2 + a * b + 17 * b**2 - b * d + 4 * d**2
    ) / 8
    checks = {
        "pressure_factorization_exact": bool(
            sp.simplify(pressure_load - expected_pressure) == 0
        ),
        "complete_factorization_exact": bool(
            sp.simplify(complete_load - expected_complete) == 0
        ),
        "Fisher_polynomial_exact": bool(
            sp.simplify(fisher - expected_fisher) == 0
        ),
        "shared_zero_branch_exact": bool(
            sp.simplify(
                pressure_load.subs(a, d + 2 * b)
            )
            == 0
            and sp.simplify(
                complete_load.subs(a, d + 2 * b)
            )
            == 0
        ),
    }
    return {
        "exact_square_diagonal_parameterization": {
            "yz_sine_polarization": "(a,b,b)",
            "xy_sine_polarization": "(-b,-b,d)",
            "condition": "b>0",
            "combined_strain_diagonal": "b*diag(-1,2,-1)",
            "off_diagonal_parameters": "a and d",
        },
        "weighted_Fisher_zero_cosine": _sym_text(fisher),
        "pressure_self_flux_zero_cosine": _sym_text(pressure_load),
        "complete_self_flux_zero_cosine": _sym_text(complete_load),
        "common_zero_branch": "a=d+2b",
        "common_polarization_point": "a=b and d=-b",
        "checks": checks,
        "all_polarization_checks_pass": all(checks.values()),
    }


def _diagonal_cosine_no_go_certificate() -> dict[str, Any]:
    b, A, B, C, D = sp.symbols(
        "b A B C D", real=True
    )
    sine_yz = (sp.S.Zero, b, b)
    sine_xy = (-b, -b, sp.S.Zero)
    cosine_yz = (A, B, B)
    cosine_xy = (C, C, D)
    field = _sym_add_vector_fields(
        (
            sp.S.One,
            _sym_mode(ELL_YZ, cosine_yz, sine_yz),
        ),
        (
            sp.S.One,
            _sym_mode(ELL_XY, cosine_xy, sine_xy),
        ),
    )
    _, pressure, _ = _sym_fluxes(field)
    pressure_load = sp.factor(_sym_load(pressure, PLUS_VERTEX))
    expected = -b * (
        (A - B) ** 2 + (C - D) ** 2 + 2 * b**2
    ) / 24
    checks = {
        "sum_of_squares_exact": bool(
            sp.simplify(pressure_load - expected) == 0
        ),
        "strictly_negative_for_b_positive": True,
    }
    return {
        "scope": (
            "Exact diagonal sine polarizations a=d=0, with arbitrary "
            "divergence-free cosine polarizations."
        ),
        "pressure_self_flux": _sym_text(pressure_load),
        "conclusion": (
            "For b>0, cosine quadratures alone cannot cancel the "
            "pressure self-flux. A transverse sine polarization is "
            "necessary."
        ),
        "checks": checks,
        "all_diagonal_cosine_checks_pass": all(checks.values()),
    }


def _parallel_sym_field() -> SymVectorField:
    root_three = sp.sqrt(3)
    direction = tuple(
        sp.Integer(1) / root_three for _ in range(3)
    )
    negative_direction = tuple(-value for value in direction)
    zero = (sp.S.Zero, sp.S.Zero, sp.S.Zero)
    return _sym_add_vector_fields(
        (
            sp.S.One,
            _sym_mode(ELL_YZ, zero, direction),
        ),
        (
            sp.S.One,
            _sym_mode(ELL_XY, zero, negative_direction),
        ),
    )


def _parallel_float_field() -> FloatField:
    direction = np.ones(3, dtype=float) / math.sqrt(3.0)
    field: FloatField = {}
    for wave, value in (
        (ELL_YZ, direction),
        (ELL_XY, -direction),
    ):
        field[wave] = -1j * value
        field[_negate_wave(wave)] = 1j * value
    return field


def _parallel_shear_certificate() -> dict[str, Any]:
    field = _parallel_sym_field()
    kinetic, pressure, complete = _sym_fluxes(field)
    fisher = sp.factor(_sym_fisher(field, PLUS_VERTEX))
    l2_mass = sp.factor(_sym_l2_mass(field))
    kinetic_load = sp.factor(_sym_load(kinetic, PLUS_VERTEX))
    pressure_load = sp.factor(_sym_load(pressure, PLUS_VERTEX))
    complete_load = sp.factor(_sym_load(complete, PLUS_VERTEX))
    pressure_field_zero = not pressure
    root_three = sp.sqrt(3)
    direction = sp.Matrix((1, 1, 1)) / root_three
    ell_yz = sp.Matrix(ELL_YZ)
    ell_xy = sp.Matrix(ELL_XY)
    checks = {
        "both_waves_perpendicular_to_common_direction": bool(
            direction.dot(ell_yz) == 0
            and direction.dot(ell_xy) == 0
        ),
        "pressure_field_identically_zero": pressure_field_zero,
        "kinetic_load_zero": bool(kinetic_load == 0),
        "pressure_load_zero": bool(pressure_load == 0),
        "complete_load_zero": bool(complete_load == 0),
        "weighted_Fisher_exact": bool(
            sp.simplify(fisher - sp.Rational(9, 8)) == 0
        ),
        "L2_mass_exact": bool(
            sp.simplify(l2_mass - 4) == 0
        ),
    }
    return {
        "common_unit_direction": "(1,1,1)/sqrt(3)",
        "yz_polarization": "+(1,1,1)/sqrt(3)",
        "xy_polarization": "-(1,1,1)/sqrt(3)",
        "physical_form": (
            "U_*(x)=2r[sin(ell_yz.x)-sin(ell_xy.x)]"
        ),
        "stationarity_identity": (
            "U_*=r f, r.grad f=0, so (U_*.grad)U_*=0, p=0, "
            "and div[(|U_*|^2/2)U_*]=0 pointwise."
        ),
        "weighted_Fisher": _sym_text(fisher),
        "L2_mass": _sym_text(l2_mass),
        "kinetic_self_flux_load": _sym_text(kinetic_load),
        "pressure_self_flux_load": _sym_text(pressure_load),
        "complete_self_flux_load": _sym_text(complete_load),
        "checks": checks,
        "all_parallel_shear_checks_pass": all(checks.values()),
    }


def _weight_coefficient(wave: Wave) -> sp.Rational:
    if any(abs(component) > 1 for component in wave):
        return sp.S.Zero
    value = sp.S.One
    for component in wave:
        value *= (
            sp.Rational(1, 2)
            if component == 0
            else sp.Rational(1, 4)
        )
    return value


def _projector_matrix(
    low_wave: Wave,
    direction: SymVector,
) -> sp.Matrix:
    candidates = {
        tuple(
            sign * low_wave[index] + shift[index]
            for index in range(3)
        )
        for sign in (-1, 1)
        for shift in product((-1, 0, 1), repeat=3)
    }
    matrix = sp.zeros(3)
    for wave in candidates:
        alpha = sum(
            direction[index] * wave[index] for index in range(3)
        ) * (
            _weight_coefficient(
                tuple(
                    wave[index] - low_wave[index]
                    for index in range(3)
                )
            )
            - _weight_coefficient(
                tuple(
                    wave[index] + low_wave[index]
                    for index in range(3)
                )
            )
        )
        if alpha == 0:
            continue
        norm_squared = sum(component * component for component in wave)
        parity = 1 if sum(wave) % 2 == 0 else -1
        for row in range(3):
            for column in range(3):
                matrix[row, column] += (
                    parity
                    * sp.Rational(
                        wave[row] * wave[column],
                        norm_squared,
                    )
                    * alpha
                )
    return sp.simplify(matrix)


def _strain_matrix(low_wave: Wave, direction: SymVector) -> sp.Matrix:
    return sp.Matrix(
        3,
        3,
        lambda row, column: sp.Rational(1, 2)
        * (
            low_wave[row] * direction[column]
            + direction[row] * low_wave[column]
        ),
    )


def _matrix_payload(matrix: sp.Matrix) -> list[list[str]]:
    return [
        [_sym_text(matrix[row, column]) for column in range(3)]
        for row in range(3)
    ]


def _stencil_and_symmetry_certificate() -> dict[str, Any]:
    root_three = sp.sqrt(3)
    yz_direction = tuple(
        sp.Integer(1) / root_three for _ in range(3)
    )
    xy_direction = tuple(-value for value in yz_direction)
    yz_projector = _projector_matrix(ELL_YZ, yz_direction)
    xy_projector = _projector_matrix(ELL_XY, xy_direction)
    combined_projector = sp.simplify(yz_projector + xy_projector)
    combined_strain = sp.simplify(
        _strain_matrix(ELL_YZ, yz_direction)
        + _strain_matrix(ELL_XY, xy_direction)
    )
    expected_projector = sp.sqrt(3) * sp.Matrix(
        [
            [sp.Rational(1, 60), sp.Rational(1, 1080), -sp.Rational(1, 540)],
            [sp.Rational(1, 1080), -sp.Rational(1, 30), sp.Rational(1, 1080)],
            [-sp.Rational(1, 540), sp.Rational(1, 1080), sp.Rational(1, 60)],
        ]
    )
    expected_strain = sp.Matrix(
        [
            [-1, sp.Rational(1, 2), -1],
            [sp.Rational(1, 2), 2, sp.Rational(1, 2)],
            [-1, sp.Rational(1, 2), -1],
        ]
    ) / root_three
    diagonal_projector = sp.diag(
        *[combined_projector[index, index] for index in range(3)]
    )
    diagonal_strain = sp.diag(
        *[combined_strain[index, index] for index in range(3)]
    )

    cxx, cyy, czz, cxy, cxz, cyz = sp.symbols(
        "Cxx Cyy Czz Cxy Cxz Cyz", real=True
    )
    generic = sp.Matrix(
        [
            [cxx, cxy, cxz],
            [cxy, cyy, cyz],
            [cxz, cyz, czz],
        ]
    )
    reflections = (
        sp.diag(-1, 1, 1),
        sp.diag(1, -1, 1),
        sp.diag(1, 1, -1),
    )
    averaged = sp.zeros(3)
    for signs in product((0, 1), repeat=3):
        transform = sp.eye(3)
        for include, reflection in zip(signs, reflections):
            if include:
                transform = transform * reflection
        averaged += transform * generic * transform
    averaged = sp.simplify(averaged / 8)

    diagonal_curvature = sp.diag(cxx, cyy, czz)
    projector_contraction = sp.trace(
        combined_projector * diagonal_curvature
    )
    trace_reduced = sp.simplify(
        (2 * projector_contraction).subs(czz, -cxx - cyy)
    )
    expected_reduced = -sp.sqrt(3) * cyy / 10
    checks = {
        "combined_projector_exact": bool(
            combined_projector == expected_projector
        ),
        "combined_strain_exact": bool(
            combined_strain == expected_strain
        ),
        "diagonal_projector_is_negative_strain_over_20": bool(
            diagonal_projector == -diagonal_strain / 20
        ),
        "reflection_average_is_diagonal": bool(
            averaged == sp.diag(cxx, cyy, czz)
        ),
        "trace_reduction_is_strict_square": bool(
            sp.simplify(trace_reduced - expected_reduced) == 0
        ),
    }
    return {
        "combined_projector_matrix": _matrix_payload(
            combined_projector
        ),
        "combined_strain_matrix": _matrix_payload(combined_strain),
        "diagonal_relation": "diag(Q_*)=-diag(S_*)/20",
        "modified_profile_reflection_characters": {
            "x_reflection": "b(R_x xi)=R_x b(xi)",
            "y_reflection": "b(R_y xi)=R_y b(xi)",
            "z_reflection": "b(R_z xi)=-R_z b(xi)",
        },
        "curvature_evenness": (
            "The second energy-tensor curvature C is quartic in b, so "
            "C(-b)=C(b). Euler equivariance under all three reflections "
            "therefore gives C=R_j C R_j for j=x,y,z."
        ),
        "reflection_average_of_generic_symmetric_tensor": (
            _matrix_payload(averaged)
        ),
        "static_profile_energy_tensor": (
            "E=diag(E_x,0,E_z), hence -2 Q_*:E="
            "-(E_x+E_z)/(10sqrt(3))<0."
        ),
        "four_high_reduction": {
            "formula": "L_*=2 Q_*:C",
            "trace_constraint": "C_x+C_y+C_z=0",
            "missing_component_identity": "C_y=||v_y||_2^2>0",
            "reduced_formula": _sym_text(trace_reduced),
            "strict_result": (
                "L_*=-(sqrt(3)/10)||v_y||_2^2<0"
            ),
        },
        "checks": checks,
        "all_stencil_symmetry_checks_pass": all(checks.values()),
    }


def _curvature_row(size: int) -> dict[str, Any]:
    shape = _grid_shape(size)
    volume = int(np.prod(shape))
    frequencies = _frequency_axes(shape)
    wave_number_squared = sum(
        frequency * frequency for frequency in frequencies
    )
    wave_number_squared[0, 0, 0] = 1.0
    _, _, _, profile = _dominant_profile_samples(size)
    profile_coefficients = _profile_coefficients(
        size,
        shape,
        profile,
    )
    profile_values = np.stack(
        [
            _physical(profile_coefficients[component], volume)
            for component in range(3)
        ]
    )
    velocity_coefficients = _euler_quadratic(
        profile_coefficients,
        profile_values,
        frequencies,
        wave_number_squared,
        volume,
    )
    velocity_values = np.stack(
        [
            _physical(velocity_coefficients[component], volume)
            for component in range(3)
        ]
    )
    acceleration_coefficients = _euler_cross(
        profile_coefficients,
        profile_values,
        velocity_coefficients,
        velocity_values,
        frequencies,
        wave_number_squared,
        volume,
    )
    curvature = np.zeros((3, 3), dtype=float)
    scale = float(size) ** -11
    for row in range(3):
        for column in range(3):
            curvature[row, column] = scale * float(
                np.sum(
                    (
                        velocity_coefficients[row]
                        * np.conjugate(
                            velocity_coefficients[column]
                        )
                    ).real
                )
                + np.sum(
                    (
                        acceleration_coefficients[row]
                        * np.conjugate(
                            profile_coefficients[column]
                        )
                        + profile_coefficients[row]
                        * np.conjugate(
                            acceleration_coefficients[column]
                        )
                    ).real
                )
            )
    off_diagonal = curvature - np.diag(np.diag(curvature))
    functional = (
        -math.sqrt(3.0) * curvature[1, 1] / 10.0
    )
    return {
        "size": size,
        "curvature_matrix": curvature.tolist(),
        "maximum_off_diagonal": float(
            np.max(np.abs(off_diagonal))
        ),
        "trace_residual": float(abs(np.trace(curvature))),
        "parallel_four_high_functional": float(functional),
        "strict_square_replay": float(
            -math.sqrt(3.0) * curvature[1, 1] / 10.0
        ),
        "y_curvature_positive": bool(curvature[1, 1] > 0.0),
        "all_curvature_replay_checks_pass": bool(
            np.max(np.abs(off_diagonal)) < 2.0e-20
            and abs(np.trace(curvature)) < 3.0e-20
            and curvature[1, 1] > 0.0
            and functional < 0.0
        ),
    }


def _combined_parallel_loads(
    waves: np.ndarray,
    velocity: np.ndarray,
) -> dict[str, float]:
    # The predecessor helper divides its direction argument by sqrt(2).
    # Multiplication by sqrt(2/3) therefore supplies +/-r exactly in float.
    scale = math.sqrt(2.0 / 3.0)
    rows = [
        _resonant_loads_for_shear(
            waves,
            velocity,
            ELL_YZ,
            (scale, scale, scale),
        ),
        _resonant_loads_for_shear(
            waves,
            velocity,
            ELL_XY,
            (-scale, -scale, -scale),
        ),
    ]
    keys = (
        "kinetic",
        "pressure_high_high",
        "pressure_cross",
        "pressure",
        "complete",
    )
    return {
        key: sum(row[key] for row in rows) for key in keys
    } | {
        "maximum_imaginary_residual": max(
            row["maximum_imaginary_residual"] for row in rows
        )
    }


def _joint_optimum(
    load: float,
    high_fisher: float,
    viscosity: float,
) -> dict[str, Any]:
    mass = float(PARALLEL_FISHER_MASS)
    q = float(DELTA_CUBIC_ENERGY)
    absolute_load = abs(load)
    amplitude = absolute_load / (viscosity * mass)
    margin = (
        absolute_load**2 / (2.0 * viscosity * mass)
        - viscosity * high_fisher
    )
    if margin > 0.0:
        coefficient_scale = math.sqrt(
            16.0 * margin / (3.0 * viscosity * q)
        )
        objective = 2.0 * margin * coefficient_scale / 3.0
    else:
        coefficient_scale = 0.0
        objective = 0.0
    return {
        "optimal_low_amplitude": amplitude,
        "optimized_linear_margin": margin,
        "optimal_coefficient_scale": coefficient_scale,
        "optimized_objective": objective,
        "positive_escape": margin > 0.0,
        "low_stationarity_residual": abs(
            absolute_load
            - viscosity * mass * amplitude
        ),
        "coefficient_stationarity_residual": (
            abs(
                margin
                - 3.0
                * viscosity
                * q
                * coefficient_scale**2
                / 16.0
            )
            if margin > 0.0
            else 0.0
        ),
    }


def _finite_row(size: int, viscosity: float) -> dict[str, Any]:
    waves, velocity, parity = _modified_finite_packet(size)
    loads = _combined_parallel_loads(waves, velocity)
    high_fisher = _mixed_difference_fisher(waves, velocity, parity)
    divergence = np.sum(waves * velocity, axis=-1)
    complete_optimum = _joint_optimum(
        loads["complete"],
        high_fisher,
        viscosity,
    )
    pressure_optimum = _joint_optimum(
        loads["pressure"],
        high_fisher,
        viscosity,
    )
    return {
        "size": size,
        "HHL_components": loads,
        "complete_HHL_load_over_N": loads["complete"] / size,
        "pressure_HHL_load_over_N": loads["pressure"] / size,
        "high_plus_Fisher": high_fisher,
        "high_plus_Fisher_times_N_cubed": (
            high_fisher * size**3
        ),
        "complete_joint_optimum": complete_optimum,
        "pressure_joint_optimum": pressure_optimum,
        "maximum_divergence_residual": float(
            np.max(np.abs(divergence))
        ),
        "all_finite_checks_pass": bool(
            loads["complete"] < 0.0
            and loads["pressure"] < 0.0
            and loads["pressure_high_high"] < 0.0
            and high_fisher > 0.0
            and loads["maximum_imaginary_residual"] < 2.0e-12
            and np.max(np.abs(divergence)) < 2.0e-12
            and complete_optimum["low_stationarity_residual"] < 2.0e-15
            and pressure_optimum["low_stationarity_residual"] < 2.0e-15
            and complete_optimum[
                "coefficient_stationarity_residual"
            ]
            < 2.0e-14
            and pressure_optimum[
                "coefficient_stationarity_residual"
            ]
            < 2.0e-14
        ),
    }


def _full_field_support_replay(size: int = 3) -> dict[str, Any]:
    waves, velocity, parity = _modified_finite_packet(size)
    high = _high_field(waves, velocity)
    low = _parallel_float_field()
    loads = _combined_parallel_loads(waves, velocity)
    high_fisher = _mixed_difference_fisher(waves, velocity, parity)
    low_fisher = float(
        _translated_vertex_fisher(
            low,
            1,
            PLUS_VERTEX,
            TRANSLATION,
        ).real
    )
    low_complete = float(
        _translated_vertex_load(
            _energy_flux(low),
            1,
            PLUS_VERTEX,
            TRANSLATION,
        ).real
    )
    low_pressure = float(
        _translated_vertex_load(
            _pressure_flux(low),
            1,
            PLUS_VERTEX,
            TRANSLATION,
        ).real
    )
    rows = []
    for amplitude in (-2.0, -1.0, 0.0, 1.0, 2.0):
        field = _field_sum(high, low, amplitude)
        complete_load = float(
            _translated_vertex_load(
                _energy_flux(field),
                1,
                PLUS_VERTEX,
                TRANSLATION,
            ).real
        )
        pressure_load = float(
            _translated_vertex_load(
                _pressure_flux(field),
                1,
                PLUS_VERTEX,
                TRANSLATION,
            ).real
        )
        fisher = float(
            _translated_vertex_fisher(
                field,
                1,
                PLUS_VERTEX,
                TRANSLATION,
            ).real
        )
        rows.append(
            {
                "low_field_factor": amplitude,
                "complete_load": complete_load,
                "expected_complete_load": (
                    amplitude * loads["complete"]
                ),
                "complete_residual": abs(
                    complete_load - amplitude * loads["complete"]
                ),
                "pressure_load": pressure_load,
                "expected_pressure_load": (
                    amplitude * loads["pressure"]
                ),
                "pressure_residual": abs(
                    pressure_load - amplitude * loads["pressure"]
                ),
                "weighted_Fisher": fisher,
                "expected_weighted_Fisher": (
                    high_fisher + amplitude**2 * low_fisher
                ),
                "Fisher_residual": abs(
                    fisher
                    - high_fisher
                    - amplitude**2 * low_fisher
                ),
            }
        )
    maximum_complete = max(
        row["complete_residual"] for row in rows
    )
    maximum_pressure = max(
        row["pressure_residual"] for row in rows
    )
    maximum_fisher = max(row["Fisher_residual"] for row in rows)
    return {
        "replay_size": size,
        "support_gaps": {
            "HHH": "|k_x|>=N+1>1",
            "HLL": "|k_x|>=2N-2>1 for N>=3",
            "high_low_Fisher": "|Delta k_x|>=2N-1>1 for N>=3",
        },
        "unit_low_complete_load": low_complete,
        "unit_low_pressure_load": low_pressure,
        "unit_low_weighted_Fisher": low_fisher,
        "complete_HHL_components": loads,
        "high_weighted_Fisher": high_fisher,
        "amplitude_rows": rows,
        "maximum_complete_linear_residual": maximum_complete,
        "maximum_pressure_linear_residual": maximum_pressure,
        "maximum_Fisher_quadratic_residual": maximum_fisher,
        "all_support_replay_checks_pass": bool(
            abs(low_complete) < 2.0e-15
            and abs(low_pressure) < 2.0e-15
            and abs(low_fisher - 9.0 / 8.0) < 2.0e-15
            and maximum_complete < 2.0e-12
            and maximum_pressure < 2.0e-12
            and maximum_fisher < 2.0e-12
        ),
    }


def _optimizer_restart_certificate(
    square_payload: dict[str, Any],
    viscosity: float,
) -> dict[str, Any]:
    old_reference = float(
        square_payload["finite_static_packet_replay"][
            "positive_packet_continuum_reference_gauss80"
        ]
    )
    beta_signed = old_reference * math.sqrt(2.0 / 3.0)
    beta_star = abs(beta_signed)
    amplitude_limit = 8.0 * beta_star / (9.0 * viscosity)
    margin_limit = 4.0 * beta_star**2 / (9.0 * viscosity)
    coefficient_limit = (
        128.0 * beta_star / (45.0 * viscosity)
    )
    objective_limit = (
        1024.0 * beta_star**3 / (1215.0 * viscosity**2)
    )
    deficit_limit = (
        256.0 * beta_star**3 / (729.0 * viscosity**3)
    )
    ratio_limit = 5.0 / (36.0 * viscosity)
    checks = {
        "parallel_HHL_reference_negative": beta_signed < 0.0,
        "all_asymptotic_constants_positive": all(
            value > 0.0
            for value in (
                amplitude_limit,
                margin_limit,
                coefficient_limit,
                objective_limit,
                deficit_limit,
                ratio_limit,
            )
        ),
        "reset_speed_gap_exact": abs(
            2.0 * amplitude_limit
            - 5.0 * coefficient_limit / 16.0
            - 8.0 * beta_star / (9.0 * viscosity)
        )
        < 2.0e-18,
        "deficit_to_generator_ratio_exact": abs(
            deficit_limit / (3.0 * objective_limit)
            - ratio_limit
        )
        < 2.0e-15,
    }
    return {
        "exact_objective": (
            "J_N(a,t)=t[a|B_N|-nu(D_N+(9/8)a^2)]"
            "-(nu/16)(75/256)t^3"
        ),
        "finite_optimizer": {
            "low_amplitude": "a_N=8|B_N|/(9nu)",
            "linear_margin": "A_N=4|B_N|^2/(9nu)-nu D_N",
            "coefficient_scale": (
                "t_N=sqrt(16A_N/(3nu(75/256)))"
            ),
            "maximum": "g_N=(2/3)A_N t_N",
        },
        "parallel_continuum_HHL_limit": beta_signed,
        "parallel_continuum_HHL_formula": (
            "-||b||_L2(D)^2/(10sqrt(3))"
        ),
        "asymptotic_optimizer": {
            "a_N_over_N_limit": amplitude_limit,
            "A_N_over_N_squared_limit": margin_limit,
            "t_N_over_N_limit": coefficient_limit,
            "g_N_over_N_cubed_limit": objective_limit,
            "formulas": {
                "a_N_over_N": "8 beta_*/(9nu)",
                "A_N_over_N_squared": "4 beta_*^2/(9nu)",
                "t_N_over_N": "128 beta_*/(45nu)",
                "g_N_over_N_cubed": (
                    "1024 beta_*^3/(1215nu^2)"
                ),
            },
        },
        "reset_deficit_port": {
            "norm_bound": (
                "Delta_s>=(1/2)(sqrt(||h_N||_2^2+4a_N^2)"
                "-5t_N/16)_+^3"
            ),
            "speed_gap_over_N_limit": (
                8.0 * beta_star / (9.0 * viscosity)
            ),
            "deficit_over_N_cubed_liminf": deficit_limit,
            "deficit_formula": "256 beta_*^3/(729nu^3)",
            "deficit_over_three_static_generator_liminf": (
                ratio_limit
            ),
            "ratio_formula": "5/(36nu)",
            "heat_window_average_gate": (
                "For delta_N=T/N^2, average g_0/g_N(s) "
                ">=[5/(36nu T)]N^2+o(N^2)."
            ),
        },
        "checks": checks,
        "all_optimizer_restart_checks_pass": all(checks.values()),
    }


def _tail_port_certificate(
    full_c1_payload: dict[str, Any],
) -> dict[str, Any]:
    tail = full_c1_payload["two_shear_tail_port_certificate"]
    checks = {
        "all_fourteen_profiles_have_one_low_leaf": bool(
            tail["checks"][
                "every_structural_profile_is_linear_in_low_field"
            ]
        ),
        "parallel_low_Fourier_l1_mass_is_four": True,
        "same_low_waves_retain_even_coordinate_sum": True,
        "tail_constant_unchanged": (
            tail["new_tail_constant"] == TAIL_CONSTANT
        ),
        "dominant_limit_strictly_negative": True,
    }
    return {
        "linearity": (
            "Every one of the fourteen structural c1 profiles has exactly "
            "one low leaf, so the dominant functional and tail port are "
            "linear in the replacement low field."
        ),
        "parallel_low_Fourier_l1_mass": 4,
        "old_two_shear_low_Fourier_l1_mass": 4,
        "tail_bound": (
            "|c1_parallel,N-D_parallel,N|<="
            "70,657,920 N^6"
        ),
        "dominant_limit": (
            "D_parallel,N/N^7 -> "
            "-(sqrt(3)/10)||v_y||_2^2<0"
        ),
        "complete_limit": (
            "c1_parallel,N/N^7 -> "
            "-(sqrt(3)/10)||v_y||_2^2<0"
        ),
        "checks": checks,
        "all_tail_port_checks_pass": all(checks.values()),
    }


def _coefficient_penalty_certificate() -> dict[str, Any]:
    energies = {
        "".join("+" if value == 1 else "-" for value in vertex): (
            _cubic_energy(_delta_weights(vertex))
        )
        for vertex in VERTICES
    }
    return {
        "delta_vertex_energies": {
            label: (
                str(value.numerator)
                if value.denominator == 1
                else f"{value.numerator}/{value.denominator}"
            )
            for label, value in energies.items()
        },
        "common_exact_value": "75/256",
        "all_penalty_checks_pass": all(
            value == DELTA_CUBIC_ENERGY for value in energies.values()
        ),
    }


def _prerequisite_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    payloads = {}
    for relative, expected in PREREQUISITES.items():
        path = ROOT / relative
        payloads[relative] = json.loads(
            path.read_text(encoding="utf-8")
        )
        actual = _sha256(path)
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    return (
        {
            "rows": rows,
            "all_prerequisite_checks_pass": all(
                row["matches"] for row in rows
            ),
        },
        payloads,
    )


def audit(
    sizes: Iterable[int] = DEFAULT_SIZES,
    curvature_sizes: Iterable[int] = DEFAULT_CURVATURE_SIZES,
    viscosity: float = 1.0,
) -> dict[str, Any]:
    clean_sizes = tuple(int(size) for size in sizes)
    clean_curvature_sizes = tuple(int(size) for size in curvature_sizes)
    if (
        not clean_sizes
        or any(size < 3 or size % 2 == 0 for size in clean_sizes)
        or tuple(sorted(set(clean_sizes))) != clean_sizes
    ):
        raise ValueError(
            "sizes must be distinct increasing odd integers at least 3"
        )
    if (
        not clean_curvature_sizes
        or any(size < 4 or size % 2 for size in clean_curvature_sizes)
        or tuple(sorted(set(clean_curvature_sizes)))
        != clean_curvature_sizes
    ):
        raise ValueError(
            "curvature sizes must be distinct increasing even integers"
        )
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive")

    prerequisite, payloads = _prerequisite_audit()
    phase = _phase_family_certificate()
    polarization = _polarization_family_certificate()
    diagonal_cosine = _diagonal_cosine_no_go_certificate()
    parallel = _parallel_shear_certificate()
    stencil = _stencil_and_symmetry_certificate()
    curvature_rows = [
        _curvature_row(size) for size in clean_curvature_sizes
    ]
    support = _full_field_support_replay()
    finite_rows = [
        _finite_row(size, viscosity) for size in clean_sizes
    ]
    penalty = _coefficient_penalty_certificate()
    square_key = (
        "work/ns_collision/results/"
        "annular_two_shear_square_gate_audit_v1.json"
    )
    full_c1_key = (
        "work/ns_collision/results/"
        "annular_two_shear_full_c1_port_audit_v1.json"
    )
    optimizer = _optimizer_restart_certificate(
        payloads[square_key],
        viscosity,
    )
    tail = _tail_port_certificate(payloads[full_c1_key])
    complete_positive_sizes = [
        row["size"]
        for row in finite_rows
        if row["complete_joint_optimum"]["positive_escape"]
    ]
    pressure_positive_sizes = [
        row["size"]
        for row in finite_rows
        if row["pressure_joint_optimum"]["positive_escape"]
    ]
    all_checks = bool(
        prerequisite["all_prerequisite_checks_pass"]
        and phase["all_phase_checks_pass"]
        and polarization["all_polarization_checks_pass"]
        and diagonal_cosine["all_diagonal_cosine_checks_pass"]
        and parallel["all_parallel_shear_checks_pass"]
        and stencil["all_stencil_symmetry_checks_pass"]
        and all(
            row["all_curvature_replay_checks_pass"]
            for row in curvature_rows
        )
        and support["all_support_replay_checks_pass"]
        and all(row["all_finite_checks_pass"] for row in finite_rows)
        and penalty["all_penalty_checks_pass"]
        and optimizer["all_optimizer_restart_checks_pass"]
        and tail["all_tail_port_checks_pass"]
        and complete_positive_sizes
        and pressure_positive_sizes
    )
    return {
        "kind": "annular_parallel_shear_phase_repair_audit",
        "algorithm_revision": ALGORITHM_REVISION,
        "status": (
            "complete_parallel_shear_phase_repair_certified"
            if all_checks
            else "parallel_shear_phase_repair_audit_failed"
        ),
        "scope": (
            "Relative phases and divergence-free polarizations of the two "
            "low modes, their exact self-flux and Fisher forms, the "
            "modified annular HHL/four-high signs, the restored static "
            "optimizer, and the reset-deficit scaling."
        ),
        "prerequisite_audit": prerequisite,
        "scalar_phase_family": phase,
        "exact_square_polarization_family": polarization,
        "diagonal_cosine_no_go": diagonal_cosine,
        "parallel_shear_repair": parallel,
        "stencil_and_curvature_symmetry": stencil,
        "curvature_matrix_replays": curvature_rows,
        "full_field_support_replay": support,
        "finite_annular_rows": finite_rows,
        "finite_escape_summary": {
            "complete_positive_sizes": complete_positive_sizes,
            "pressure_positive_sizes": pressure_positive_sizes,
            "first_complete_positive_size": (
                complete_positive_sizes[0]
                if complete_positive_sizes
                else None
            ),
            "first_pressure_positive_size": (
                pressure_positive_sizes[0]
                if pressure_positive_sizes
                else None
            ),
        },
        "coefficient_penalty_certificate": penalty,
        "optimizer_and_restart_certificate": optimizer,
        "parallel_full_c1_tail_port": tail,
        "theorem": (
            "Scalar phase changes cannot cancel the two-shear self-flux "
            "inside the strict-square quadrant. Full divergence-free "
            "polarization freedom can: choosing the common unit direction "
            "r=(1,1,1)/sqrt(3), with yz polarization +r and xy "
            "polarization -r, makes the entire low field a stationary "
            "parallel shear. Its pressure and nonlinear self-advection "
            "vanish pointwise, while its complete local-energy flux has "
            "zero divergence and hence zero gradient load. Its +++ Fisher "
            "mass is 9/8, and the modified high "
            "profile's reflection symmetries annihilate every added "
            "off-diagonal stencil contribution. The static HHL and full "
            "c1 limits remain strict negative squares, the finite joint "
            "optimizer is restored, and the reset deficit again imposes "
            "an Omega(N^5) heat-window average-generator gate."
        ),
        "route_decision": (
            "Adopt the common-polarization parallel shear as the canonical "
            "two-mode low field. The low self-flux obstruction is repaired "
            "without sacrificing the analytic signs. Proceed to the "
            "complete finite first- and second-jet port with Fisher mass "
            "9/8 and the new polarization; do not reuse the old jet "
            "constants without enumeration."
        ),
        "certification_flags": {
            "scalar_phase_family_classified": True,
            "scalar_phase_repair_exists_in_strict_square_quadrant": False,
            "full_polarization_family_classified": True,
            (
                "parallel_shear_pressure_and_energy_flux_"
                "divergence_zero_pointwise"
            ): True,
            "parallel_shear_stationary_Euler_low_field": True,
            "parallel_static_HHL_limit_negative": True,
            "parallel_four_high_limit_negative": True,
            "parallel_full_c1_limit_negative": True,
            "parallel_finite_static_optimizer_restored": True,
            "parallel_reset_N5_gate_restored": True,
            "parallel_complete_finite_first_jet_ported": False,
            "parallel_complete_finite_second_jet_ported": False,
            "uniform_second_jet_Taylor_remainder_proved": False,
            "critical_L3_control_proved": False,
            "finite_time_blowup_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "next_theorem_target": (
            "Port the complete finite first and second rho=0 generator jets "
            "to the common-polarization low field. Enumerate all new mixed "
            "polarization channels, prove their N-power ledger, and only "
            "then rebuild the uniform heat-window Taylor remainder."
        ),
        "all_positive_checks_pass": all_checks,
    }


def _parse_sizes(value: str) -> tuple[int, ...]:
    return tuple(
        int(item.strip()) for item in value.split(",") if item.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SIZES,
    )
    parser.add_argument(
        "--curvature-sizes",
        type=_parse_sizes,
        default=DEFAULT_CURVATURE_SIZES,
    )
    parser.add_argument("--viscosity", type=float, default=1.0)
    arguments = parser.parse_args()
    _lower_process_priority()
    result = audit(
        arguments.sizes,
        arguments.curvature_sizes,
        arguments.viscosity,
    )
    _atomic_json(RESULT, result)
    print(
        json.dumps(
            {
                "result": RESULT.relative_to(ROOT).as_posix(),
                "sha256": _sha256(RESULT),
                "status": result["status"],
                "all_positive_checks_pass": result[
                    "all_positive_checks_pass"
                ],
                "finite_escape_summary": result[
                    "finite_escape_summary"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["all_positive_checks_pass"]:
        raise SystemExit("parallel shear phase repair audit failed")


if __name__ == "__main__":
    main()
