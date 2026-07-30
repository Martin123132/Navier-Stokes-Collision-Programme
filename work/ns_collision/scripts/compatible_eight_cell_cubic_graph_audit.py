"""Audit the globally compatible eight-cell cubic pressure graph."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[3]
SIGNS = (-1, 1)
VERTICES = tuple(itertools.product(SIGNS, repeat=3))
FACE_VERTICES = tuple(itertools.product(SIGNS, repeat=2))


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


def _moment2(first: int, second: int) -> Fraction:
    return Fraction(3 if first == second else 1, 8)


def _moment3(first: int, second: int, third: int) -> Fraction:
    return Fraction(5 if first == second == third else 1, 16)


def _vertex_index(
    direction: int,
    direction_sign: int,
    other_signs: tuple[int, int],
) -> int:
    other_directions = [index for index in range(3) if index != direction]
    vertex = [0, 0, 0]
    vertex[direction] = direction_sign
    for index, sign in zip(other_directions, other_signs):
        vertex[index] = sign
    return VERTICES.index(tuple(vertex))


def _face_value(
    weights: Sequence[Fraction],
    direction: int,
    direction_sign: int,
    other_signs: tuple[int, int],
) -> Fraction:
    return weights[
        _vertex_index(direction, direction_sign, other_signs)
    ]


def _cubic_energy(weights: Sequence[Fraction]) -> Fraction:
    total = Fraction(0)
    for direction in range(3):
        for first in FACE_VERTICES:
            face_sum = (
                _face_value(weights, direction, 1, first)
                + _face_value(weights, direction, -1, first)
            )
            for second in FACE_VERTICES:
                first_difference = (
                    _face_value(weights, direction, 1, second)
                    - _face_value(weights, direction, -1, second)
                )
                for third in FACE_VERTICES:
                    second_difference = (
                        _face_value(weights, direction, 1, third)
                        - _face_value(weights, direction, -1, third)
                    )
                    moment = _moment3(
                        first[0],
                        second[0],
                        third[0],
                    ) * _moment3(
                        first[1],
                        second[1],
                        third[1],
                    )
                    total += (
                        face_sum
                        * first_difference
                        * second_difference
                        * moment
                    )
    return total


def _quadratic_difference_energy(
    weights: Sequence[Fraction],
) -> Fraction:
    total = Fraction(0)
    for direction in range(3):
        for first in FACE_VERTICES:
            first_difference = (
                _face_value(weights, direction, 1, first)
                - _face_value(weights, direction, -1, first)
            )
            for second in FACE_VERTICES:
                second_difference = (
                    _face_value(weights, direction, 1, second)
                    - _face_value(weights, direction, -1, second)
                )
                moment = _moment2(
                    first[0],
                    second[0],
                ) * _moment2(
                    first[1],
                    second[1],
                )
                total += (
                    first_difference * second_difference * moment
                )
    return total


def _fraction(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _projection_and_energy_audit() -> dict[str, Any]:
    sample_weights = tuple(
        Fraction(value) for value in (1, 2, 3, 4, 2, 3, 4, 1)
    )
    common_shift = Fraction(3, 7)
    shifted = tuple(value + common_shift for value in sample_weights)
    energy = _cubic_energy(sample_weights)
    difference_energy = _quadratic_difference_energy(sample_weights)
    shifted_energy = _cubic_energy(shifted)
    translation_residual = (
        shifted_energy
        - energy
        - 2 * common_shift * difference_energy
    )
    constant_energy = _cubic_energy((Fraction(2),) * 8)

    load_size, cubic_size, coefficient = sp.symbols(
        "B Q c",
        positive=True,
    )
    scale = sp.sqrt(load_size / (3 * coefficient * cubic_size))
    ray_objective = (
        scale * load_size
        - coefficient * scale**3 * cubic_size
    )
    ray_value = sp.simplify(ray_objective)
    expected_ray_value = (
        2
        * load_size ** sp.Rational(3, 2)
        / (3 * sp.sqrt(3 * coefficient * cubic_size))
    )
    ray_residual = sp.simplify(ray_value - expected_ray_value)

    return {
        "partition": (
            "phi_s(x)=(1+s cos(mx))/2; "
            "Phi_v=product_j phi_(v_j)"
        ),
        "global_weight": "lambda_w=sum_(v in {+,-}^3) w_v Phi_v",
        "conditional_faces": (
            "A_j=sum_(v_hat)w_(+,v_hat)Phi_(v_hat), "
            "B_j=sum_(v_hat)w_(-,v_hat)Phi_(v_hat)"
        ),
        "pressure_load": (
            "L_e(w)=b(e).w, b_v=sum_j v_j "
            "mean_(x_hat_j)[Phi_(v_hat_j)e_j]"
        ),
        "load_conservation": "sum_v b_v=0",
        "projection_fact": (
            "Only the tensor span of 1 and cos(mx_k) in the two "
            "conditional coordinates contributes to b(e)."
        ),
        "cubic_energy": "Q(w)=sum_j mean[H_j D_j^2]",
        "Fisher_coefficient": "c=nu m^2/16",
        "objective": "S_e(w)=b.w-cQ(w), w_v>=0",
        "triple_partition_moments": {
            "all_three_signs_equal": "5/16",
            "otherwise": "1/16",
        },
        "sample_energy": _fraction(energy),
        "sample_difference_energy": _fraction(difference_energy),
        "common_shift": _fraction(common_shift),
        "shifted_energy": _fraction(shifted_energy),
        "translation_identity": "Q(w+a1)=Q(w)+2a R(w)",
        "translation_residual": _fraction(translation_residual),
        "constant_vector_energy": _fraction(constant_energy),
        "projective_reduction": (
            "sup_(t>=0)[tB-ct^3Q]="
            "2 B_+^(3/2)/(3sqrt(3cQ))"
        ),
        "projective_symbolic_value": str(ray_value),
        "projective_symbolic_residual": str(ray_residual),
        "normalization": (
            "Because b.1=0 and a positive common shift increases Q, "
            "the projective search may be restricted to min_v w_v=0."
        ),
        "all_checks_pass": (
            translation_residual == 0
            and constant_energy == 0
            and energy > 0
            and difference_energy > 0
            and ray_residual == 0
        ),
    }


def _nonconvexity_audit() -> dict[str, Any]:
    midpoint = tuple(
        Fraction(value) for value in (1, 2, 3, 4, 2, 3, 4, 1)
    )
    direction = tuple(
        Fraction(value)
        for value in (
            Fraction(1, 2),
            Fraction(1, 4),
            Fraction(1, 4),
            Fraction(1, 4),
            Fraction(1, 4),
            Fraction(1, 4),
            Fraction(1, 4),
            Fraction(1, 2),
        )
    )
    lower = tuple(
        value - step for value, step in zip(midpoint, direction)
    )
    upper = tuple(
        value + step for value, step in zip(midpoint, direction)
    )
    lower_energy = _cubic_energy(lower)
    midpoint_energy = _cubic_energy(midpoint)
    upper_energy = _cubic_energy(upper)
    convex_average = (lower_energy + upper_energy) / 2
    violation = midpoint_energy - convex_average
    return {
        "lower_weights": [_fraction(value) for value in lower],
        "midpoint_weights": [_fraction(value) for value in midpoint],
        "upper_weights": [_fraction(value) for value in upper],
        "lower_energy": _fraction(lower_energy),
        "midpoint_energy": _fraction(midpoint_energy),
        "upper_energy": _fraction(upper_energy),
        "endpoint_average": _fraction(convex_average),
        "exact_convexity_violation": _fraction(violation),
        "interpretation": (
            "Q(midpoint) exceeds the endpoint average by 39/128. "
            "The compatible cubic graph problem is nonconvex in the "
            "eight nonnegative vertex coefficients."
        ),
        "all_checks_pass": (
            all(value >= 0 for value in lower)
            and all(value >= 0 for value in upper)
            and violation == Fraction(39, 128)
        ),
    }


def _vertex_saturator_audit() -> dict[str, Any]:
    partition_cube_mean = Fraction(5, 16)
    conditional_face_cube_mean = partition_cube_mean**2
    cubic_energy = 3 * conditional_face_cube_mean
    pressure_load = 9 * conditional_face_cube_mean
    objective = pressure_load - cubic_energy
    edgewise_envelope = 6 * conditional_face_cube_mean
    ray_derivative_at_one = pressure_load - 3 * cubic_energy

    vertex_rows = []
    load_profiles = {}
    for vertex in VERTICES:
        load_vector = []
        for load_vertex in VERTICES:
            load = Fraction(0)
            for direction in range(3):
                conditional_moment = Fraction(1)
                for other_direction in range(3):
                    if other_direction == direction:
                        continue
                    conditional_moment *= _moment3(
                        load_vertex[other_direction],
                        vertex[other_direction],
                        vertex[other_direction],
                    )
                load += (
                    3
                    * load_vertex[direction]
                    * vertex[direction]
                    * conditional_moment
                )
            load_vector.append(load)
        load_profiles[str(vertex)] = [
            _fraction(value) for value in load_vector
        ]
        vertex_rows.append(
            {
                "vertex": list(vertex),
                "weight": "w=t delta_vertex",
                "conditional_difference": (
                    "D_j=v_j t Phi_(v_hat_j)"
                ),
                "conditional_sum": "H_j=t Phi_(v_hat_j)",
                "edge_flux": (
                    "e_j=3c v_j t^2 Phi_(v_hat_j)^2"
                ),
                "normalized_load_vector": [
                    _fraction(value) for value in load_vector
                ],
                "load_sum": _fraction(sum(load_vector, Fraction(0))),
            }
        )

    hamming_profile = {
        "0": "225/256",
        "1": "-45/256",
        "2": "-27/256",
        "3": "-9/256",
    }
    load_checks = all(
        row["load_sum"] == "0"
        and row["normalized_load_vector"][
            VERTICES.index(tuple(row["vertex"]))
        ]
        == "225/256"
        for row in vertex_rows
    )
    return {
        "construction_scope": (
            "Smooth abstract conditional edge fluxes. No claim is made "
            "that this family is generated by a divergence-free velocity "
            "and its Navier-Stokes pressure."
        ),
        "partition_cube_mean": _fraction(partition_cube_mean),
        "conditional_face_cube_mean": _fraction(
            conditional_face_cube_mean
        ),
        "normalized_cubic_energy_c_equals_t_equals_one": _fraction(
            cubic_energy
        ),
        "normalized_pressure_load": _fraction(pressure_load),
        "normalized_objective": _fraction(objective),
        "normalized_directionwise_envelope": _fraction(
            edgewise_envelope
        ),
        "normalized_load_by_Hamming_distance": hamming_profile,
        "all_normalized_load_vectors": load_profiles,
        "ray_derivative_at_t_equals_one": _fraction(
            ray_derivative_at_one
        ),
        "vertex_rows": vertex_rows,
        "conclusion": (
            "Every conditional edge attains the sharp scalar cubic "
            "envelope simultaneously. Hence no universal factor kappa<1 "
            "can improve the directionwise envelope using eight-cell "
            "coefficient compatibility alone."
        ),
        "all_checks_pass": (
            cubic_energy == Fraction(75, 256)
            and pressure_load == Fraction(225, 256)
            and objective == Fraction(75, 128)
            and objective == edgewise_envelope
            and ray_derivative_at_one == 0
            and load_checks
        ),
    }


def _taylor_green_projection_audit(size: int = 8192) -> dict[str, Any]:
    coordinate = 2.0 * math.pi * np.arange(size) / size
    edge_x = -np.cos(3.0 * coordinate) / 32.0
    edge_y = np.cos(3.0 * coordinate) / 32.0

    def phi(sign: int) -> np.ndarray:
        return (1.0 + sign * np.cos(coordinate)) / 2.0

    phi_means = {sign: float(np.mean(phi(sign))) for sign in SIGNS}
    projected_x = {
        sign: float(np.mean(phi(sign) * edge_x)) for sign in SIGNS
    }
    projected_y = {
        sign: float(np.mean(phi(sign) * edge_y)) for sign in SIGNS
    }
    loads = []
    for vertex in VERTICES:
        load_x = (
            vertex[0]
            * projected_x[vertex[1]]
            * phi_means[vertex[2]]
        )
        load_y = (
            vertex[1]
            * projected_y[vertex[0]]
            * phi_means[vertex[2]]
        )
        loads.append(load_x + load_y)

    scalar_constant = 2.0 / (3.0 * math.sqrt(3.0))
    directionwise_envelope = scalar_constant * (
        float(np.mean(np.abs(edge_x) ** 1.5))
        + float(np.mean(np.abs(edge_y) ** 1.5))
    )
    maximum_load = max(abs(value) for value in loads)
    return {
        "field": (
            "u=(sin x cos y,-cos x sin y,0), "
            "p=(cos 2x+cos 2y)/4"
        ),
        "partition_frequency": 1,
        "simplified_edges": {
            "e_x": "-cos(3y)/32",
            "e_y": "cos(3x)/32",
            "e_z": "0",
        },
        "analytic_projected_load": "b_v=0 for every vertex",
        "numerical_vertex_loads": loads,
        "maximum_numerical_load": maximum_load,
        "normalized_global_graph_supremum": 0.0,
        "normalized_directionwise_envelope": directionwise_envelope,
        "interpretation": (
            "The globally compatible coefficient space sees only "
            "conditional modes 0 and 1, while these pressure edges are "
            "pure mode 3. Compatibility annihilates this actual "
            "Navier-Stokes example even though its relaxed edgewise "
            "envelope is strictly positive."
        ),
        "all_checks_pass": (
            maximum_load < 1.0e-17
            and directionwise_envelope > 0.0
        ),
    }


def audit() -> dict[str, Any]:
    projection = _projection_and_energy_audit()
    nonconvexity = _nonconvexity_audit()
    saturator = _vertex_saturator_audit()
    taylor_green = _taylor_green_projection_audit()
    positive_checks = {
        "exact_graph_projection_and_ray_reduction_pass": projection[
            "all_checks_pass"
        ],
        "exact_nonconvexity_witness_passes": nonconvexity[
            "all_checks_pass"
        ],
        "abstract_vertex_saturator_passes": saturator[
            "all_checks_pass"
        ],
        "Taylor_Green_graph_annihilation_passes": taylor_green[
            "all_checks_pass"
        ],
    }
    return {
        "kind": "compatible_eight_cell_cubic_graph_audit",
        "schema_version": 1,
        "status": (
            "graph_projection_derived_uniform_compatibility_gain_"
            "falsified_PDE_load_cone_open"
        ),
        "assumption_scope": (
            "Smooth periodic conditional pressure-edge functions and the "
            "frequency-m tensor cosine partition with eight nonnegative "
            "vertex coefficients. The strict-gain counterexample is an "
            "abstract edge-flux family; its realization by a "
            "Navier-Stokes pressure is not asserted."
        ),
        "exact_projection_and_cubic_energy": projection,
        "nonconvexity_witness": nonconvexity,
        "abstract_vertex_saturator": saturator,
        "taylor_green_projection": taylor_green,
        "positive_checks": positive_checks,
        "all_positive_checks_pass": all(positive_checks.values()),
        "certification_flags": {
            "eight_cell_pressure_load_projection_derived": True,
            "eight_cell_projective_cubic_reduction_derived": True,
            "eight_cell_cubic_energy_nonconvexity_proved": True,
            "compatibility_only_uniform_strict_gain_falsified": True,
            "Taylor_Green_compatible_pressure_load_annihilated": True,
            "abstract_vertex_saturator_PDE_realized": False,
            "Navier_Stokes_pressure_load_cone_characterized": False,
            "pressure_L32_remainder_absorbed": False,
            "critical_signed_bound_proved": False,
            "low_regularity_passage_proved": False,
            "exceptional_set_upgrade_proved": False,
            "Navier_Stokes_global_regularity_proved": False,
        },
        "route_decision": (
            "Do not expect coefficient compatibility by itself to provide "
            "a universal constant gain: a smooth abstract vertex flux "
            "saturates every direction simultaneously. Do exploit the "
            "exact four-mode projection before taking absolute values: "
            "Taylor-Green is annihilated completely. The next question is "
            "therefore the algebraic range of projected loads generated by "
            "divergence-free velocity-pressure pairs, not an unconstrained "
            "eight-variable numerical maximization."
        ),
        "next_theorem_target": (
            "Derive the Fourier-triad map from a finite divergence-free "
            "velocity field to the seven-dimensional zero-sum load vector "
            "b. Determine whether the eight abstract vertex-saturating "
            "load rays lie in its closure. If they are excluded, quantify "
            "the separation in a scale-invariant norm; if one is realized, "
            "record a PDE-realizable no-go for graph compatibility alone."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "work/ns_collision/results/"
            "compatible_eight_cell_cubic_graph_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = audit()
    if not result["all_positive_checks_pass"]:
        raise RuntimeError("compatible eight-cell cubic graph audit failed")
    _atomic_json(arguments.output, result)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": _sha256(arguments.output),
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
