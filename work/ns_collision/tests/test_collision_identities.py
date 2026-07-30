from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.sparse.linalg import expm_multiply
from scipy.special import hyp1f1


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_collision_identities.py"
)
SPEC = importlib.util.spec_from_file_location("verify_collision_identities", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

GRAM_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "gram_boundary_audit.py"
)
GRAM_SPEC = importlib.util.spec_from_file_location("gram_boundary_audit", GRAM_SCRIPT)
assert GRAM_SPEC is not None and GRAM_SPEC.loader is not None
GRAM_MODULE = importlib.util.module_from_spec(GRAM_SPEC)
GRAM_SPEC.loader.exec_module(GRAM_MODULE)

BACKWARD_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "backward_replica_audit.py"
)
BACKWARD_SPEC = importlib.util.spec_from_file_location(
    "backward_replica_audit", BACKWARD_SCRIPT
)
assert BACKWARD_SPEC is not None and BACKWARD_SPEC.loader is not None
BACKWARD_MODULE = importlib.util.module_from_spec(BACKWARD_SPEC)
BACKWARD_SPEC.loader.exec_module(BACKWARD_MODULE)

NEWTONIAN_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "newtonian_boundary_audit.py"
)
NEWTONIAN_SPEC = importlib.util.spec_from_file_location(
    "newtonian_boundary_audit", NEWTONIAN_SCRIPT
)
assert NEWTONIAN_SPEC is not None and NEWTONIAN_SPEC.loader is not None
NEWTONIAN_MODULE = importlib.util.module_from_spec(NEWTONIAN_SPEC)
NEWTONIAN_SPEC.loader.exec_module(NEWTONIAN_MODULE)

TWO_POINT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "two_point_vorticity_audit.py"
)
TWO_POINT_SPEC = importlib.util.spec_from_file_location(
    "two_point_vorticity_audit", TWO_POINT_SCRIPT
)
assert TWO_POINT_SPEC is not None and TWO_POINT_SPEC.loader is not None
TWO_POINT_MODULE = importlib.util.module_from_spec(TWO_POINT_SPEC)
TWO_POINT_SPEC.loader.exec_module(TWO_POINT_MODULE)

MULTIPLIER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "strain_boundary_multiplier_audit.py"
)
MULTIPLIER_SPEC = importlib.util.spec_from_file_location(
    "strain_boundary_multiplier_audit", MULTIPLIER_SCRIPT
)
assert MULTIPLIER_SPEC is not None and MULTIPLIER_SPEC.loader is not None
MULTIPLIER_MODULE = importlib.util.module_from_spec(MULTIPLIER_SPEC)
MULTIPLIER_SPEC.loader.exec_module(MULTIPLIER_MODULE)

HEAT_CUBIC_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "heat_scale_cubic_cancellation_audit.py"
)
HEAT_CUBIC_SPEC = importlib.util.spec_from_file_location(
    "heat_scale_cubic_cancellation_audit", HEAT_CUBIC_SCRIPT
)
assert HEAT_CUBIC_SPEC is not None and HEAT_CUBIC_SPEC.loader is not None
HEAT_CUBIC_MODULE = importlib.util.module_from_spec(HEAT_CUBIC_SPEC)
HEAT_CUBIC_SPEC.loader.exec_module(HEAT_CUBIC_MODULE)

TRIAD_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "fourier_triad_collision_audit.py"
)
TRIAD_SPEC = importlib.util.spec_from_file_location(
    "fourier_triad_collision_audit", TRIAD_SCRIPT
)
assert TRIAD_SPEC is not None and TRIAD_SPEC.loader is not None
TRIAD_MODULE = importlib.util.module_from_spec(TRIAD_SPEC)
TRIAD_SPEC.loader.exec_module(TRIAD_MODULE)

BARRIER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "first_crossing_barrier_audit.py"
)
BARRIER_SPEC = importlib.util.spec_from_file_location(
    "first_crossing_barrier_audit", BARRIER_SCRIPT
)
assert BARRIER_SPEC is not None and BARRIER_SPEC.loader is not None
BARRIER_MODULE = importlib.util.module_from_spec(BARRIER_SPEC)
BARRIER_SPEC.loader.exec_module(BARRIER_MODULE)

ADAPTIVE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "adaptive_scale_barrier_audit.py"
)
ADAPTIVE_SPEC = importlib.util.spec_from_file_location(
    "adaptive_scale_barrier_audit", ADAPTIVE_SCRIPT
)
assert ADAPTIVE_SPEC is not None and ADAPTIVE_SPEC.loader is not None
ADAPTIVE_MODULE = importlib.util.module_from_spec(ADAPTIVE_SPEC)
ADAPTIVE_SPEC.loader.exec_module(ADAPTIVE_MODULE)

CUMULATIVE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "cumulative_defect_audit.py"
)
CUMULATIVE_SPEC = importlib.util.spec_from_file_location(
    "cumulative_defect_audit", CUMULATIVE_SCRIPT
)
assert CUMULATIVE_SPEC is not None and CUMULATIVE_SPEC.loader is not None
CUMULATIVE_MODULE = importlib.util.module_from_spec(CUMULATIVE_SPEC)
CUMULATIVE_SPEC.loader.exec_module(CUMULATIVE_MODULE)

QUARTIC_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "quartic_transfer_audit.py"
)
QUARTIC_SPEC = importlib.util.spec_from_file_location(
    "quartic_transfer_audit", QUARTIC_SCRIPT
)
assert QUARTIC_SPEC is not None and QUARTIC_SPEC.loader is not None
QUARTIC_MODULE = importlib.util.module_from_spec(QUARTIC_SPEC)
QUARTIC_SPEC.loader.exec_module(QUARTIC_MODULE)

QUARTIC_COUNTEREXAMPLE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "quartic_transfer_counterexample.py"
)
QUARTIC_COUNTEREXAMPLE_SPEC = importlib.util.spec_from_file_location(
    "quartic_transfer_counterexample", QUARTIC_COUNTEREXAMPLE_SCRIPT
)
assert (
    QUARTIC_COUNTEREXAMPLE_SPEC is not None
    and QUARTIC_COUNTEREXAMPLE_SPEC.loader is not None
)
QUARTIC_COUNTEREXAMPLE_MODULE = importlib.util.module_from_spec(
    QUARTIC_COUNTEREXAMPLE_SPEC
)
QUARTIC_COUNTEREXAMPLE_SPEC.loader.exec_module(
    QUARTIC_COUNTEREXAMPLE_MODULE
)

QUARTIC_HELICAL_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "quartic_transfer_helical_audit.py"
)
QUARTIC_HELICAL_SPEC = importlib.util.spec_from_file_location(
    "quartic_transfer_helical_audit", QUARTIC_HELICAL_SCRIPT
)
assert QUARTIC_HELICAL_SPEC is not None and QUARTIC_HELICAL_SPEC.loader is not None
QUARTIC_HELICAL_MODULE = importlib.util.module_from_spec(QUARTIC_HELICAL_SPEC)
QUARTIC_HELICAL_SPEC.loader.exec_module(QUARTIC_HELICAL_MODULE)

QUARTIC_HELICAL_MATRIX_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "quartic_transfer_helical_matrix_audit.py"
)
QUARTIC_HELICAL_MATRIX_SPEC = importlib.util.spec_from_file_location(
    "quartic_transfer_helical_matrix_audit", QUARTIC_HELICAL_MATRIX_SCRIPT
)
assert (
    QUARTIC_HELICAL_MATRIX_SPEC is not None
    and QUARTIC_HELICAL_MATRIX_SPEC.loader is not None
)
QUARTIC_HELICAL_MATRIX_MODULE = importlib.util.module_from_spec(
    QUARTIC_HELICAL_MATRIX_SPEC
)
QUARTIC_HELICAL_MATRIX_SPEC.loader.exec_module(QUARTIC_HELICAL_MATRIX_MODULE)

TRAJECTORY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ns_trajectory_defect_audit.py"
)
TRAJECTORY_SPEC = importlib.util.spec_from_file_location(
    "ns_trajectory_defect_audit", TRAJECTORY_SCRIPT
)
assert TRAJECTORY_SPEC is not None and TRAJECTORY_SPEC.loader is not None
TRAJECTORY_MODULE = importlib.util.module_from_spec(TRAJECTORY_SPEC)
TRAJECTORY_SPEC.loader.exec_module(TRAJECTORY_MODULE)

GALERKIN_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "galerkin_trajectory_audit.py"
)
GALERKIN_SPEC = importlib.util.spec_from_file_location(
    "galerkin_trajectory_audit", GALERKIN_SCRIPT
)
assert GALERKIN_SPEC is not None and GALERKIN_SPEC.loader is not None
GALERKIN_MODULE = importlib.util.module_from_spec(GALERKIN_SPEC)
GALERKIN_SPEC.loader.exec_module(GALERKIN_MODULE)

GALERKIN_ANALYSIS_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "galerkin_sweep_analysis.py"
)
GALERKIN_ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "galerkin_sweep_analysis", GALERKIN_ANALYSIS_SCRIPT
)
assert (
    GALERKIN_ANALYSIS_SPEC is not None
    and GALERKIN_ANALYSIS_SPEC.loader is not None
)
GALERKIN_ANALYSIS_MODULE = importlib.util.module_from_spec(
    GALERKIN_ANALYSIS_SPEC
)
GALERKIN_ANALYSIS_SPEC.loader.exec_module(GALERKIN_ANALYSIS_MODULE)

HELICAL_TRAJECTORY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "helical_trajectory_channel_audit.py"
)
HELICAL_TRAJECTORY_SPEC = importlib.util.spec_from_file_location(
    "helical_trajectory_channel_audit", HELICAL_TRAJECTORY_SCRIPT
)
assert (
    HELICAL_TRAJECTORY_SPEC is not None
    and HELICAL_TRAJECTORY_SPEC.loader is not None
)
HELICAL_TRAJECTORY_MODULE = importlib.util.module_from_spec(
    HELICAL_TRAJECTORY_SPEC
)
HELICAL_TRAJECTORY_SPEC.loader.exec_module(HELICAL_TRAJECTORY_MODULE)

GENERATED_TRANSFER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generated_mode_transfer_audit.py"
)
GENERATED_TRANSFER_SPEC = importlib.util.spec_from_file_location(
    "generated_mode_transfer_audit", GENERATED_TRANSFER_SCRIPT
)
assert (
    GENERATED_TRANSFER_SPEC is not None
    and GENERATED_TRANSFER_SPEC.loader is not None
)
GENERATED_TRANSFER_MODULE = importlib.util.module_from_spec(
    GENERATED_TRANSFER_SPEC
)
GENERATED_TRANSFER_SPEC.loader.exec_module(GENERATED_TRANSFER_MODULE)

WEAK_TRANSFER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "weak_generated_transfer_audit.py"
)
WEAK_TRANSFER_SPEC = importlib.util.spec_from_file_location(
    "weak_generated_transfer_audit", WEAK_TRANSFER_SCRIPT
)
assert WEAK_TRANSFER_SPEC is not None and WEAK_TRANSFER_SPEC.loader is not None
WEAK_TRANSFER_MODULE = importlib.util.module_from_spec(WEAK_TRANSFER_SPEC)
WEAK_TRANSFER_SPEC.loader.exec_module(WEAK_TRANSFER_MODULE)

SECOND_NORMAL_FORM_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "second_normal_form_audit.py"
)
SECOND_NORMAL_FORM_SPEC = importlib.util.spec_from_file_location(
    "second_normal_form_audit", SECOND_NORMAL_FORM_SCRIPT
)
assert (
    SECOND_NORMAL_FORM_SPEC is not None
    and SECOND_NORMAL_FORM_SPEC.loader is not None
)
SECOND_NORMAL_FORM_MODULE = importlib.util.module_from_spec(
    SECOND_NORMAL_FORM_SPEC
)
SECOND_NORMAL_FORM_SPEC.loader.exec_module(SECOND_NORMAL_FORM_MODULE)

THIRD_NORMAL_FORM_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "third_normal_form_audit.py"
)
THIRD_NORMAL_FORM_SPEC = importlib.util.spec_from_file_location(
    "third_normal_form_audit", THIRD_NORMAL_FORM_SCRIPT
)
assert (
    THIRD_NORMAL_FORM_SPEC is not None
    and THIRD_NORMAL_FORM_SPEC.loader is not None
)
THIRD_NORMAL_FORM_MODULE = importlib.util.module_from_spec(
    THIRD_NORMAL_FORM_SPEC
)
THIRD_NORMAL_FORM_SPEC.loader.exec_module(THIRD_NORMAL_FORM_MODULE)

RESUMMATION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "normal_form_resummation_audit.py"
)
RESUMMATION_SPEC = importlib.util.spec_from_file_location(
    "normal_form_resummation_audit", RESUMMATION_SCRIPT
)
assert RESUMMATION_SPEC is not None and RESUMMATION_SPEC.loader is not None
RESUMMATION_MODULE = importlib.util.module_from_spec(RESUMMATION_SPEC)
RESUMMATION_SPEC.loader.exec_module(RESUMMATION_MODULE)

COLLISION_COHERENCE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "collision_coherence_weight_audit.py"
)
COLLISION_COHERENCE_SPEC = importlib.util.spec_from_file_location(
    "collision_coherence_weight_audit", COLLISION_COHERENCE_SCRIPT
)
assert (
    COLLISION_COHERENCE_SPEC is not None
    and COLLISION_COHERENCE_SPEC.loader is not None
)
COLLISION_COHERENCE_MODULE = importlib.util.module_from_spec(
    COLLISION_COHERENCE_SPEC
)
COLLISION_COHERENCE_SPEC.loader.exec_module(COLLISION_COHERENCE_MODULE)

LOCALIZED_TUBE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "localized_strain_tube_audit.py"
)
LOCALIZED_TUBE_SPEC = importlib.util.spec_from_file_location(
    "localized_strain_tube_audit", LOCALIZED_TUBE_SCRIPT
)
assert LOCALIZED_TUBE_SPEC is not None and LOCALIZED_TUBE_SPEC.loader is not None
LOCALIZED_TUBE_MODULE = importlib.util.module_from_spec(LOCALIZED_TUBE_SPEC)
LOCALIZED_TUBE_SPEC.loader.exec_module(LOCALIZED_TUBE_MODULE)

MOVING_TUBE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "moving_strain_tube_audit.py"
)
MOVING_TUBE_SPEC = importlib.util.spec_from_file_location(
    "moving_strain_tube_audit", MOVING_TUBE_SCRIPT
)
assert MOVING_TUBE_SPEC is not None and MOVING_TUBE_SPEC.loader is not None
MOVING_TUBE_MODULE = importlib.util.module_from_spec(MOVING_TUBE_SPEC)
MOVING_TUBE_SPEC.loader.exec_module(MOVING_TUBE_MODULE)

REENTRY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "strain_tube_reentry_audit.py"
)
REENTRY_SPEC = importlib.util.spec_from_file_location(
    "strain_tube_reentry_audit", REENTRY_SCRIPT
)
assert REENTRY_SPEC is not None and REENTRY_SPEC.loader is not None
REENTRY_MODULE = importlib.util.module_from_spec(REENTRY_SPEC)
REENTRY_SPEC.loader.exec_module(REENTRY_MODULE)

THREE_DIMENSIONAL_GATE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "three_dimensional_leray_gate_audit.py"
)
THREE_DIMENSIONAL_GATE_SPEC = importlib.util.spec_from_file_location(
    "three_dimensional_leray_gate_audit", THREE_DIMENSIONAL_GATE_SCRIPT
)
assert (
    THREE_DIMENSIONAL_GATE_SPEC is not None
    and THREE_DIMENSIONAL_GATE_SPEC.loader is not None
)
THREE_DIMENSIONAL_GATE_MODULE = importlib.util.module_from_spec(
    THREE_DIMENSIONAL_GATE_SPEC
)
THREE_DIMENSIONAL_GATE_SPEC.loader.exec_module(THREE_DIMENSIONAL_GATE_MODULE)

EIGENFRAME_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "strain_eigenframe_geometry_audit.py"
)
EIGENFRAME_SPEC = importlib.util.spec_from_file_location(
    "strain_eigenframe_geometry_audit", EIGENFRAME_SCRIPT
)
assert EIGENFRAME_SPEC is not None and EIGENFRAME_SPEC.loader is not None
EIGENFRAME_MODULE = importlib.util.module_from_spec(EIGENFRAME_SPEC)
EIGENFRAME_SPEC.loader.exec_module(EIGENFRAME_MODULE)

PRESSURE_COLLISION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "pressure_collision_kernel_audit.py"
)
PRESSURE_COLLISION_SPEC = importlib.util.spec_from_file_location(
    "pressure_collision_kernel_audit", PRESSURE_COLLISION_SCRIPT
)
assert (
    PRESSURE_COLLISION_SPEC is not None
    and PRESSURE_COLLISION_SPEC.loader is not None
)
PRESSURE_COLLISION_MODULE = importlib.util.module_from_spec(
    PRESSURE_COLLISION_SPEC
)
PRESSURE_COLLISION_SPEC.loader.exec_module(PRESSURE_COLLISION_MODULE)

PRESSURE_PAIRING_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "pressure_frame_pairing_audit.py"
)
PRESSURE_PAIRING_SPEC = importlib.util.spec_from_file_location(
    "pressure_frame_pairing_audit", PRESSURE_PAIRING_SCRIPT
)
assert (
    PRESSURE_PAIRING_SPEC is not None
    and PRESSURE_PAIRING_SPEC.loader is not None
)
PRESSURE_PAIRING_MODULE = importlib.util.module_from_spec(
    PRESSURE_PAIRING_SPEC
)
PRESSURE_PAIRING_SPEC.loader.exec_module(PRESSURE_PAIRING_MODULE)

PRESSURE_SHELL_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "pressure_shell_commutator_audit.py"
)
PRESSURE_SHELL_SPEC = importlib.util.spec_from_file_location(
    "pressure_shell_commutator_audit", PRESSURE_SHELL_SCRIPT
)
assert (
    PRESSURE_SHELL_SPEC is not None
    and PRESSURE_SHELL_SPEC.loader is not None
)
PRESSURE_SHELL_MODULE = importlib.util.module_from_spec(PRESSURE_SHELL_SPEC)
PRESSURE_SHELL_SPEC.loader.exec_module(PRESSURE_SHELL_MODULE)

REYNOLDS_ENVELOPE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "adaptive_reynolds_envelope_audit.py"
)
REYNOLDS_ENVELOPE_SPEC = importlib.util.spec_from_file_location(
    "adaptive_reynolds_envelope_audit", REYNOLDS_ENVELOPE_SCRIPT
)
assert (
    REYNOLDS_ENVELOPE_SPEC is not None
    and REYNOLDS_ENVELOPE_SPEC.loader is not None
)
REYNOLDS_ENVELOPE_MODULE = importlib.util.module_from_spec(
    REYNOLDS_ENVELOPE_SPEC
)
REYNOLDS_ENVELOPE_SPEC.loader.exec_module(REYNOLDS_ENVELOPE_MODULE)

SHRINKING_RENEWAL_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "shrinking_tube_renewal_audit.py"
)
SHRINKING_RENEWAL_SPEC = importlib.util.spec_from_file_location(
    "shrinking_tube_renewal_audit", SHRINKING_RENEWAL_SCRIPT
)
assert (
    SHRINKING_RENEWAL_SPEC is not None
    and SHRINKING_RENEWAL_SPEC.loader is not None
)
SHRINKING_RENEWAL_MODULE = importlib.util.module_from_spec(
    SHRINKING_RENEWAL_SPEC
)
SHRINKING_RENEWAL_SPEC.loader.exec_module(SHRINKING_RENEWAL_MODULE)

PRESSURE_PARTITION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "pressure_partition_flux_audit.py"
)
PRESSURE_PARTITION_SPEC = importlib.util.spec_from_file_location(
    "pressure_partition_flux_audit", PRESSURE_PARTITION_SCRIPT
)
assert (
    PRESSURE_PARTITION_SPEC is not None
    and PRESSURE_PARTITION_SPEC.loader is not None
)
PRESSURE_PARTITION_MODULE = importlib.util.module_from_spec(
    PRESSURE_PARTITION_SPEC
)
PRESSURE_PARTITION_SPEC.loader.exec_module(PRESSURE_PARTITION_MODULE)

INTRINSIC_COVER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "intrinsic_radius_cover_audit.py"
)
INTRINSIC_COVER_SPEC = importlib.util.spec_from_file_location(
    "intrinsic_radius_cover_audit", INTRINSIC_COVER_SCRIPT
)
assert (
    INTRINSIC_COVER_SPEC is not None
    and INTRINSIC_COVER_SPEC.loader is not None
)
INTRINSIC_COVER_MODULE = importlib.util.module_from_spec(
    INTRINSIC_COVER_SPEC
)
INTRINSIC_COVER_SPEC.loader.exec_module(INTRINSIC_COVER_MODULE)

DYADIC_COVER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "monotone_dyadic_cover_audit.py"
)
DYADIC_COVER_SPEC = importlib.util.spec_from_file_location(
    "monotone_dyadic_cover_audit", DYADIC_COVER_SCRIPT
)
assert (
    DYADIC_COVER_SPEC is not None
    and DYADIC_COVER_SPEC.loader is not None
)
DYADIC_COVER_MODULE = importlib.util.module_from_spec(DYADIC_COVER_SPEC)
DYADIC_COVER_SPEC.loader.exec_module(DYADIC_COVER_MODULE)

DYADIC_TRANSITION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "dyadic_gauge_transition_audit.py"
)
DYADIC_TRANSITION_SPEC = importlib.util.spec_from_file_location(
    "dyadic_gauge_transition_audit", DYADIC_TRANSITION_SCRIPT
)
assert (
    DYADIC_TRANSITION_SPEC is not None
    and DYADIC_TRANSITION_SPEC.loader is not None
)
DYADIC_TRANSITION_MODULE = importlib.util.module_from_spec(
    DYADIC_TRANSITION_SPEC
)
DYADIC_TRANSITION_SPEC.loader.exec_module(DYADIC_TRANSITION_MODULE)

BRANCHING_TRANSFER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "branching_transfer_operator_audit.py"
)
BRANCHING_TRANSFER_SPEC = importlib.util.spec_from_file_location(
    "branching_transfer_operator_audit", BRANCHING_TRANSFER_SCRIPT
)
assert (
    BRANCHING_TRANSFER_SPEC is not None
    and BRANCHING_TRANSFER_SPEC.loader is not None
)
BRANCHING_TRANSFER_MODULE = importlib.util.module_from_spec(
    BRANCHING_TRANSFER_SPEC
)
BRANCHING_TRANSFER_SPEC.loader.exec_module(BRANCHING_TRANSFER_MODULE)

INTERFACE_WEIGHT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "interface_weight_no_go_audit.py"
)
INTERFACE_WEIGHT_SPEC = importlib.util.spec_from_file_location(
    "interface_weight_no_go_audit", INTERFACE_WEIGHT_SCRIPT
)
assert (
    INTERFACE_WEIGHT_SPEC is not None
    and INTERFACE_WEIGHT_SPEC.loader is not None
)
INTERFACE_WEIGHT_MODULE = importlib.util.module_from_spec(
    INTERFACE_WEIGHT_SPEC
)
INTERFACE_WEIGHT_SPEC.loader.exec_module(INTERFACE_WEIGHT_MODULE)

TWO_NORM_CYCLE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "two_norm_generation_cycle_audit.py"
)
TWO_NORM_CYCLE_SPEC = importlib.util.spec_from_file_location(
    "two_norm_generation_cycle_audit", TWO_NORM_CYCLE_SCRIPT
)
assert (
    TWO_NORM_CYCLE_SPEC is not None
    and TWO_NORM_CYCLE_SPEC.loader is not None
)
TWO_NORM_CYCLE_MODULE = importlib.util.module_from_spec(TWO_NORM_CYCLE_SPEC)
TWO_NORM_CYCLE_SPEC.loader.exec_module(TWO_NORM_CYCLE_MODULE)

BUFFERED_VISIT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "buffered_visit_feynman_kac_audit.py"
)
BUFFERED_VISIT_SPEC = importlib.util.spec_from_file_location(
    "buffered_visit_feynman_kac_audit", BUFFERED_VISIT_SCRIPT
)
assert (
    BUFFERED_VISIT_SPEC is not None
    and BUFFERED_VISIT_SPEC.loader is not None
)
BUFFERED_VISIT_MODULE = importlib.util.module_from_spec(BUFFERED_VISIT_SPEC)
BUFFERED_VISIT_SPEC.loader.exec_module(BUFFERED_VISIT_MODULE)

AXIAL_KILLING_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "axial_killing_buffered_visit_audit.py"
)
AXIAL_KILLING_SPEC = importlib.util.spec_from_file_location(
    "axial_killing_buffered_visit_audit", AXIAL_KILLING_SCRIPT
)
assert (
    AXIAL_KILLING_SPEC is not None
    and AXIAL_KILLING_SPEC.loader is not None
)
AXIAL_KILLING_MODULE = importlib.util.module_from_spec(AXIAL_KILLING_SPEC)
AXIAL_KILLING_SPEC.loader.exec_module(AXIAL_KILLING_MODULE)

FINITE_CYLINDER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "finite_cylinder_mode_audit.py"
)
FINITE_CYLINDER_SPEC = importlib.util.spec_from_file_location(
    "finite_cylinder_mode_audit", FINITE_CYLINDER_SCRIPT
)
assert (
    FINITE_CYLINDER_SPEC is not None
    and FINITE_CYLINDER_SPEC.loader is not None
)
FINITE_CYLINDER_MODULE = importlib.util.module_from_spec(
    FINITE_CYLINDER_SPEC
)
FINITE_CYLINDER_SPEC.loader.exec_module(FINITE_CYLINDER_MODULE)

FINITE_CYLINDER_PERTURBATION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "finite_cylinder_perturbation_margin_audit.py"
)
FINITE_CYLINDER_PERTURBATION_SPEC = importlib.util.spec_from_file_location(
    "finite_cylinder_perturbation_margin_audit",
    FINITE_CYLINDER_PERTURBATION_SCRIPT,
)
assert (
    FINITE_CYLINDER_PERTURBATION_SPEC is not None
    and FINITE_CYLINDER_PERTURBATION_SPEC.loader is not None
)
FINITE_CYLINDER_PERTURBATION_MODULE = importlib.util.module_from_spec(
    FINITE_CYLINDER_PERTURBATION_SPEC
)
FINITE_CYLINDER_PERTURBATION_SPEC.loader.exec_module(
    FINITE_CYLINDER_PERTURBATION_MODULE
)

FINITE_CYLINDER_KATO_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "finite_cylinder_kato_gate_audit.py"
)
FINITE_CYLINDER_KATO_SPEC = importlib.util.spec_from_file_location(
    "finite_cylinder_kato_gate_audit", FINITE_CYLINDER_KATO_SCRIPT
)
assert (
    FINITE_CYLINDER_KATO_SPEC is not None
    and FINITE_CYLINDER_KATO_SPEC.loader is not None
)
FINITE_CYLINDER_KATO_MODULE = importlib.util.module_from_spec(
    FINITE_CYLINDER_KATO_SPEC
)
FINITE_CYLINDER_KATO_SPEC.loader.exec_module(
    FINITE_CYLINDER_KATO_MODULE
)

GAUSSIAN_BOUNDARY_L2_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "gaussian_boundary_l2_transfer_audit.py"
)
GAUSSIAN_BOUNDARY_L2_SPEC = importlib.util.spec_from_file_location(
    "gaussian_boundary_l2_transfer_audit", GAUSSIAN_BOUNDARY_L2_SCRIPT
)
assert (
    GAUSSIAN_BOUNDARY_L2_SPEC is not None
    and GAUSSIAN_BOUNDARY_L2_SPEC.loader is not None
)
GAUSSIAN_BOUNDARY_L2_MODULE = importlib.util.module_from_spec(
    GAUSSIAN_BOUNDARY_L2_SPEC
)
GAUSSIAN_BOUNDARY_L2_SPEC.loader.exec_module(
    GAUSSIAN_BOUNDARY_L2_MODULE
)

GROUND_STATE_VISIT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ground_state_visit_transform_audit.py"
)
GROUND_STATE_VISIT_SPEC = importlib.util.spec_from_file_location(
    "ground_state_visit_transform_audit", GROUND_STATE_VISIT_SCRIPT
)
assert (
    GROUND_STATE_VISIT_SPEC is not None
    and GROUND_STATE_VISIT_SPEC.loader is not None
)
GROUND_STATE_VISIT_MODULE = importlib.util.module_from_spec(
    GROUND_STATE_VISIT_SPEC
)
GROUND_STATE_VISIT_SPEC.loader.exec_module(GROUND_STATE_VISIT_MODULE)

AXIAL_FORM_BOUNDARY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "axial_form_to_boundary_audit.py"
)
AXIAL_FORM_BOUNDARY_SPEC = importlib.util.spec_from_file_location(
    "axial_form_to_boundary_audit", AXIAL_FORM_BOUNDARY_SCRIPT
)
assert (
    AXIAL_FORM_BOUNDARY_SPEC is not None
    and AXIAL_FORM_BOUNDARY_SPEC.loader is not None
)
AXIAL_FORM_BOUNDARY_MODULE = importlib.util.module_from_spec(
    AXIAL_FORM_BOUNDARY_SPEC
)
AXIAL_FORM_BOUNDARY_SPEC.loader.exec_module(AXIAL_FORM_BOUNDARY_MODULE)

OFF_DIAGONAL_FORM_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "off_diagonal_form_transfer_audit.py"
)
OFF_DIAGONAL_FORM_SPEC = importlib.util.spec_from_file_location(
    "off_diagonal_form_transfer_audit", OFF_DIAGONAL_FORM_SCRIPT
)
assert (
    OFF_DIAGONAL_FORM_SPEC is not None
    and OFF_DIAGONAL_FORM_SPEC.loader is not None
)
OFF_DIAGONAL_FORM_MODULE = importlib.util.module_from_spec(
    OFF_DIAGONAL_FORM_SPEC
)
OFF_DIAGONAL_FORM_SPEC.loader.exec_module(OFF_DIAGONAL_FORM_MODULE)

WEIGHTED_CYLINDER_BUFFER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "weighted_cylinder_buffer_condition_audit.py"
)
WEIGHTED_CYLINDER_BUFFER_SPEC = importlib.util.spec_from_file_location(
    "weighted_cylinder_buffer_condition_audit",
    WEIGHTED_CYLINDER_BUFFER_SCRIPT,
)
assert (
    WEIGHTED_CYLINDER_BUFFER_SPEC is not None
    and WEIGHTED_CYLINDER_BUFFER_SPEC.loader is not None
)
WEIGHTED_CYLINDER_BUFFER_MODULE = importlib.util.module_from_spec(
    WEIGHTED_CYLINDER_BUFFER_SPEC
)
WEIGHTED_CYLINDER_BUFFER_SPEC.loader.exec_module(
    WEIGHTED_CYLINDER_BUFFER_MODULE
)

POISSON_CUTOFF_FORM_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "poisson_cutoff_form_transfer_audit.py"
)
POISSON_CUTOFF_FORM_SPEC = importlib.util.spec_from_file_location(
    "poisson_cutoff_form_transfer_audit", POISSON_CUTOFF_FORM_SCRIPT
)
assert (
    POISSON_CUTOFF_FORM_SPEC is not None
    and POISSON_CUTOFF_FORM_SPEC.loader is not None
)
POISSON_CUTOFF_FORM_MODULE = importlib.util.module_from_spec(
    POISSON_CUTOFF_FORM_SPEC
)
POISSON_CUTOFF_FORM_SPEC.loader.exec_module(
    POISSON_CUTOFF_FORM_MODULE
)

QUADRATIC_PARTITION_IMS_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "quadratic_partition_ims_budget_audit.py"
)
QUADRATIC_PARTITION_IMS_SPEC = importlib.util.spec_from_file_location(
    "quadratic_partition_ims_budget_audit",
    QUADRATIC_PARTITION_IMS_SCRIPT,
)
assert (
    QUADRATIC_PARTITION_IMS_SPEC is not None
    and QUADRATIC_PARTITION_IMS_SPEC.loader is not None
)
QUADRATIC_PARTITION_IMS_MODULE = importlib.util.module_from_spec(
    QUADRATIC_PARTITION_IMS_SPEC
)
QUADRATIC_PARTITION_IMS_SPEC.loader.exec_module(
    QUADRATIC_PARTITION_IMS_MODULE
)

RADIAL_CUBIC_PARTITION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "radial_cubic_partition_audit.py"
)
RADIAL_CUBIC_PARTITION_SPEC = importlib.util.spec_from_file_location(
    "radial_cubic_partition_audit",
    RADIAL_CUBIC_PARTITION_SCRIPT,
)
assert (
    RADIAL_CUBIC_PARTITION_SPEC is not None
    and RADIAL_CUBIC_PARTITION_SPEC.loader is not None
)
RADIAL_CUBIC_PARTITION_MODULE = importlib.util.module_from_spec(
    RADIAL_CUBIC_PARTITION_SPEC
)
RADIAL_CUBIC_PARTITION_SPEC.loader.exec_module(
    RADIAL_CUBIC_PARTITION_MODULE
)

CUBIC_LEVEL_TRANSFER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "cubic_level_transfer_audit.py"
)
CUBIC_LEVEL_TRANSFER_SPEC = importlib.util.spec_from_file_location(
    "cubic_level_transfer_audit",
    CUBIC_LEVEL_TRANSFER_SCRIPT,
)
assert (
    CUBIC_LEVEL_TRANSFER_SPEC is not None
    and CUBIC_LEVEL_TRANSFER_SPEC.loader is not None
)
CUBIC_LEVEL_TRANSFER_MODULE = importlib.util.module_from_spec(
    CUBIC_LEVEL_TRANSFER_SPEC
)
CUBIC_LEVEL_TRANSFER_SPEC.loader.exec_module(
    CUBIC_LEVEL_TRANSFER_MODULE
)

NAVIER_STOKES_COHERENCE_BUDGET_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "navier_stokes_coherence_budget_audit.py"
)
NAVIER_STOKES_COHERENCE_BUDGET_SPEC = (
    importlib.util.spec_from_file_location(
        "navier_stokes_coherence_budget_audit",
        NAVIER_STOKES_COHERENCE_BUDGET_SCRIPT,
    )
)
assert (
    NAVIER_STOKES_COHERENCE_BUDGET_SPEC is not None
    and NAVIER_STOKES_COHERENCE_BUDGET_SPEC.loader is not None
)
NAVIER_STOKES_COHERENCE_BUDGET_MODULE = importlib.util.module_from_spec(
    NAVIER_STOKES_COHERENCE_BUDGET_SPEC
)
NAVIER_STOKES_COHERENCE_BUDGET_SPEC.loader.exec_module(
    NAVIER_STOKES_COHERENCE_BUDGET_MODULE
)

GENERAL_AFFINE_SPECTRAL_FLOOR_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "general_affine_spectral_floor_audit.py"
)
GENERAL_AFFINE_SPECTRAL_FLOOR_SPEC = (
    importlib.util.spec_from_file_location(
        "general_affine_spectral_floor_audit",
        GENERAL_AFFINE_SPECTRAL_FLOOR_SCRIPT,
    )
)
assert (
    GENERAL_AFFINE_SPECTRAL_FLOOR_SPEC is not None
    and GENERAL_AFFINE_SPECTRAL_FLOOR_SPEC.loader is not None
)
GENERAL_AFFINE_SPECTRAL_FLOOR_MODULE = importlib.util.module_from_spec(
    GENERAL_AFFINE_SPECTRAL_FLOOR_SPEC
)
GENERAL_AFFINE_SPECTRAL_FLOOR_SPEC.loader.exec_module(
    GENERAL_AFFINE_SPECTRAL_FLOOR_MODULE
)

ANISOTROPIC_POISSON_TRANSFER_PILOT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "anisotropic_poisson_transfer_pilot.py"
)
ANISOTROPIC_POISSON_TRANSFER_PILOT_SPEC = (
    importlib.util.spec_from_file_location(
        "anisotropic_poisson_transfer_pilot",
        ANISOTROPIC_POISSON_TRANSFER_PILOT_SCRIPT,
    )
)
assert (
    ANISOTROPIC_POISSON_TRANSFER_PILOT_SPEC is not None
    and ANISOTROPIC_POISSON_TRANSFER_PILOT_SPEC.loader is not None
)
ANISOTROPIC_POISSON_TRANSFER_PILOT_MODULE = (
    importlib.util.module_from_spec(
        ANISOTROPIC_POISSON_TRANSFER_PILOT_SPEC
    )
)
ANISOTROPIC_POISSON_TRANSFER_PILOT_SPEC.loader.exec_module(
    ANISOTROPIC_POISSON_TRANSFER_PILOT_MODULE
)

REVERSIBLE_SHELL_RIGIDITY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reversible_shell_rigidity_audit.py"
)
REVERSIBLE_SHELL_RIGIDITY_SPEC = importlib.util.spec_from_file_location(
    "reversible_shell_rigidity_audit",
    REVERSIBLE_SHELL_RIGIDITY_SCRIPT,
)
assert (
    REVERSIBLE_SHELL_RIGIDITY_SPEC is not None
    and REVERSIBLE_SHELL_RIGIDITY_SPEC.loader is not None
)
REVERSIBLE_SHELL_RIGIDITY_MODULE = importlib.util.module_from_spec(
    REVERSIBLE_SHELL_RIGIDITY_SPEC
)
REVERSIBLE_SHELL_RIGIDITY_SPEC.loader.exec_module(
    REVERSIBLE_SHELL_RIGIDITY_MODULE
)

DIVERGENCE_FREE_SHELL_TAPER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "divergence_free_shell_taper_audit.py"
)
DIVERGENCE_FREE_SHELL_TAPER_SPEC = importlib.util.spec_from_file_location(
    "divergence_free_shell_taper_audit",
    DIVERGENCE_FREE_SHELL_TAPER_SCRIPT,
)
assert (
    DIVERGENCE_FREE_SHELL_TAPER_SPEC is not None
    and DIVERGENCE_FREE_SHELL_TAPER_SPEC.loader is not None
)
DIVERGENCE_FREE_SHELL_TAPER_MODULE = importlib.util.module_from_spec(
    DIVERGENCE_FREE_SHELL_TAPER_SPEC
)
DIVERGENCE_FREE_SHELL_TAPER_SPEC.loader.exec_module(
    DIVERGENCE_FREE_SHELL_TAPER_MODULE
)

DIVERGENCE_FREE_TAPER_TRANSFER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "divergence_free_taper_transfer_pilot.py"
)
DIVERGENCE_FREE_TAPER_TRANSFER_SPEC = importlib.util.spec_from_file_location(
    "divergence_free_taper_transfer_pilot",
    DIVERGENCE_FREE_TAPER_TRANSFER_SCRIPT,
)
assert (
    DIVERGENCE_FREE_TAPER_TRANSFER_SPEC is not None
    and DIVERGENCE_FREE_TAPER_TRANSFER_SPEC.loader is not None
)
DIVERGENCE_FREE_TAPER_TRANSFER_MODULE = importlib.util.module_from_spec(
    DIVERGENCE_FREE_TAPER_TRANSFER_SPEC
)
DIVERGENCE_FREE_TAPER_TRANSFER_SPEC.loader.exec_module(
    DIVERGENCE_FREE_TAPER_TRANSFER_MODULE
)

SECTORIAL_POISSON_TRANSFER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sectorial_poisson_transfer_audit.py"
)
SECTORIAL_POISSON_TRANSFER_SPEC = importlib.util.spec_from_file_location(
    "sectorial_poisson_transfer_audit",
    SECTORIAL_POISSON_TRANSFER_SCRIPT,
)
assert (
    SECTORIAL_POISSON_TRANSFER_SPEC is not None
    and SECTORIAL_POISSON_TRANSFER_SPEC.loader is not None
)
SECTORIAL_POISSON_TRANSFER_MODULE = importlib.util.module_from_spec(
    SECTORIAL_POISSON_TRANSFER_SPEC
)
SECTORIAL_POISSON_TRANSFER_SPEC.loader.exec_module(
    SECTORIAL_POISSON_TRANSFER_MODULE
)

MOVING_CUBIC_LABEL_TRANSPORT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "moving_cubic_label_transport_audit.py"
)
MOVING_CUBIC_LABEL_TRANSPORT_SPEC = importlib.util.spec_from_file_location(
    "moving_cubic_label_transport_audit",
    MOVING_CUBIC_LABEL_TRANSPORT_SCRIPT,
)
assert (
    MOVING_CUBIC_LABEL_TRANSPORT_SPEC is not None
    and MOVING_CUBIC_LABEL_TRANSPORT_SPEC.loader is not None
)
MOVING_CUBIC_LABEL_TRANSPORT_MODULE = importlib.util.module_from_spec(
    MOVING_CUBIC_LABEL_TRANSPORT_SPEC
)
MOVING_CUBIC_LABEL_TRANSPORT_SPEC.loader.exec_module(
    MOVING_CUBIC_LABEL_TRANSPORT_MODULE
)

CONTINUOUS_CUBIC_LOCALIZATION_NO_GO_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "continuous_cubic_localization_no_go_audit.py"
)
CONTINUOUS_CUBIC_LOCALIZATION_NO_GO_SPEC = (
    importlib.util.spec_from_file_location(
        "continuous_cubic_localization_no_go_audit",
        CONTINUOUS_CUBIC_LOCALIZATION_NO_GO_SCRIPT,
    )
)
assert (
    CONTINUOUS_CUBIC_LOCALIZATION_NO_GO_SPEC is not None
    and CONTINUOUS_CUBIC_LOCALIZATION_NO_GO_SPEC.loader is not None
)
CONTINUOUS_CUBIC_LOCALIZATION_NO_GO_MODULE = (
    importlib.util.module_from_spec(
        CONTINUOUS_CUBIC_LOCALIZATION_NO_GO_SPEC
    )
)
CONTINUOUS_CUBIC_LOCALIZATION_NO_GO_SPEC.loader.exec_module(
    CONTINUOUS_CUBIC_LOCALIZATION_NO_GO_MODULE
)

STOPPING_TIME_MOVING_VISIT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "stopping_time_moving_visit_audit.py"
)
STOPPING_TIME_MOVING_VISIT_SPEC = importlib.util.spec_from_file_location(
    "stopping_time_moving_visit_audit",
    STOPPING_TIME_MOVING_VISIT_SCRIPT,
)
assert (
    STOPPING_TIME_MOVING_VISIT_SPEC is not None
    and STOPPING_TIME_MOVING_VISIT_SPEC.loader is not None
)
STOPPING_TIME_MOVING_VISIT_MODULE = importlib.util.module_from_spec(
    STOPPING_TIME_MOVING_VISIT_SPEC
)
STOPPING_TIME_MOVING_VISIT_SPEC.loader.exec_module(
    STOPPING_TIME_MOVING_VISIT_MODULE
)

COHERENCE_ABORT_RENEWAL_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "coherence_abort_renewal_audit.py"
)
COHERENCE_ABORT_RENEWAL_SPEC = importlib.util.spec_from_file_location(
    "coherence_abort_renewal_audit",
    COHERENCE_ABORT_RENEWAL_SCRIPT,
)
assert (
    COHERENCE_ABORT_RENEWAL_SPEC is not None
    and COHERENCE_ABORT_RENEWAL_SPEC.loader is not None
)
COHERENCE_ABORT_RENEWAL_MODULE = importlib.util.module_from_spec(
    COHERENCE_ABORT_RENEWAL_SPEC
)
COHERENCE_ABORT_RENEWAL_SPEC.loader.exec_module(
    COHERENCE_ABORT_RENEWAL_MODULE
)

LERAY_MOLLIFIED_CELL_FRAME_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "leray_mollified_cell_frame_audit.py"
)
LERAY_MOLLIFIED_CELL_FRAME_SPEC = importlib.util.spec_from_file_location(
    "leray_mollified_cell_frame_audit",
    LERAY_MOLLIFIED_CELL_FRAME_SCRIPT,
)
assert (
    LERAY_MOLLIFIED_CELL_FRAME_SPEC is not None
    and LERAY_MOLLIFIED_CELL_FRAME_SPEC.loader is not None
)
LERAY_MOLLIFIED_CELL_FRAME_MODULE = importlib.util.module_from_spec(
    LERAY_MOLLIFIED_CELL_FRAME_SPEC
)
LERAY_MOLLIFIED_CELL_FRAME_SPEC.loader.exec_module(
    LERAY_MOLLIFIED_CELL_FRAME_MODULE
)

AFFINE_TAPER_RESIDUAL_NO_GO_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "affine_taper_residual_no_go_audit.py"
)
AFFINE_TAPER_RESIDUAL_NO_GO_SPEC = importlib.util.spec_from_file_location(
    "affine_taper_residual_no_go_audit",
    AFFINE_TAPER_RESIDUAL_NO_GO_SCRIPT,
)
assert (
    AFFINE_TAPER_RESIDUAL_NO_GO_SPEC is not None
    and AFFINE_TAPER_RESIDUAL_NO_GO_SPEC.loader is not None
)
AFFINE_TAPER_RESIDUAL_NO_GO_MODULE = importlib.util.module_from_spec(
    AFFINE_TAPER_RESIDUAL_NO_GO_SPEC
)
AFFINE_TAPER_RESIDUAL_NO_GO_SPEC.loader.exec_module(
    AFFINE_TAPER_RESIDUAL_NO_GO_MODULE
)

COMPACT_AFFINE_CAMPANATO_GATE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compact_affine_campanato_gate_audit.py"
)
COMPACT_AFFINE_CAMPANATO_GATE_SPEC = importlib.util.spec_from_file_location(
    "compact_affine_campanato_gate_audit",
    COMPACT_AFFINE_CAMPANATO_GATE_SCRIPT,
)
assert (
    COMPACT_AFFINE_CAMPANATO_GATE_SPEC is not None
    and COMPACT_AFFINE_CAMPANATO_GATE_SPEC.loader is not None
)
COMPACT_AFFINE_CAMPANATO_GATE_MODULE = importlib.util.module_from_spec(
    COMPACT_AFFINE_CAMPANATO_GATE_SPEC
)
COMPACT_AFFINE_CAMPANATO_GATE_SPEC.loader.exec_module(
    COMPACT_AFFINE_CAMPANATO_GATE_MODULE
)

LERAY_CONDITIONAL_OCCUPATION_NO_GO_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "leray_conditional_occupation_no_go_audit.py"
)
LERAY_CONDITIONAL_OCCUPATION_NO_GO_SPEC = importlib.util.spec_from_file_location(
    "leray_conditional_occupation_no_go_audit",
    LERAY_CONDITIONAL_OCCUPATION_NO_GO_SCRIPT,
)
assert (
    LERAY_CONDITIONAL_OCCUPATION_NO_GO_SPEC is not None
    and LERAY_CONDITIONAL_OCCUPATION_NO_GO_SPEC.loader is not None
)
LERAY_CONDITIONAL_OCCUPATION_NO_GO_MODULE = importlib.util.module_from_spec(
    LERAY_CONDITIONAL_OCCUPATION_NO_GO_SPEC
)
LERAY_CONDITIONAL_OCCUPATION_NO_GO_SPEC.loader.exec_module(
    LERAY_CONDITIONAL_OCCUPATION_NO_GO_MODULE
)

NONAUTONOMOUS_FULL_AFFINE_FORM_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "nonautonomous_full_affine_form_audit.py"
)
NONAUTONOMOUS_FULL_AFFINE_FORM_SPEC = importlib.util.spec_from_file_location(
    "nonautonomous_full_affine_form_audit",
    NONAUTONOMOUS_FULL_AFFINE_FORM_SCRIPT,
)
assert (
    NONAUTONOMOUS_FULL_AFFINE_FORM_SPEC is not None
    and NONAUTONOMOUS_FULL_AFFINE_FORM_SPEC.loader is not None
)
NONAUTONOMOUS_FULL_AFFINE_FORM_MODULE = importlib.util.module_from_spec(
    NONAUTONOMOUS_FULL_AFFINE_FORM_SPEC
)
NONAUTONOMOUS_FULL_AFFINE_FORM_SPEC.loader.exec_module(
    NONAUTONOMOUS_FULL_AFFINE_FORM_MODULE
)

ROTATING_AFFINE_VISIT_PILOT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rotating_affine_visit_pilot.py"
)
ROTATING_AFFINE_VISIT_PILOT_SPEC = importlib.util.spec_from_file_location(
    "rotating_affine_visit_pilot",
    ROTATING_AFFINE_VISIT_PILOT_SCRIPT,
)
assert (
    ROTATING_AFFINE_VISIT_PILOT_SPEC is not None
    and ROTATING_AFFINE_VISIT_PILOT_SPEC.loader is not None
)
ROTATING_AFFINE_VISIT_PILOT_MODULE = importlib.util.module_from_spec(
    ROTATING_AFFINE_VISIT_PILOT_SPEC
)
ROTATING_AFFINE_VISIT_PILOT_SPEC.loader.exec_module(
    ROTATING_AFFINE_VISIT_PILOT_MODULE
)

WEIGHTED_KERNEL_DYNAMIC_L2_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "weighted_kernel_dynamic_l2_audit.py"
)
WEIGHTED_KERNEL_DYNAMIC_L2_SPEC = importlib.util.spec_from_file_location(
    "weighted_kernel_dynamic_l2_audit",
    WEIGHTED_KERNEL_DYNAMIC_L2_SCRIPT,
)
assert (
    WEIGHTED_KERNEL_DYNAMIC_L2_SPEC is not None
    and WEIGHTED_KERNEL_DYNAMIC_L2_SPEC.loader is not None
)
WEIGHTED_KERNEL_DYNAMIC_L2_MODULE = importlib.util.module_from_spec(
    WEIGHTED_KERNEL_DYNAMIC_L2_SPEC
)
WEIGHTED_KERNEL_DYNAMIC_L2_SPEC.loader.exec_module(
    WEIGHTED_KERNEL_DYNAMIC_L2_MODULE
)

NONAUTONOMOUS_SCALAR_GAIN_GATE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "nonautonomous_scalar_gain_gate_audit.py"
)
NONAUTONOMOUS_SCALAR_GAIN_GATE_SPEC = importlib.util.spec_from_file_location(
    "nonautonomous_scalar_gain_gate_audit",
    NONAUTONOMOUS_SCALAR_GAIN_GATE_SCRIPT,
)
assert (
    NONAUTONOMOUS_SCALAR_GAIN_GATE_SPEC is not None
    and NONAUTONOMOUS_SCALAR_GAIN_GATE_SPEC.loader is not None
)
NONAUTONOMOUS_SCALAR_GAIN_GATE_MODULE = importlib.util.module_from_spec(
    NONAUTONOMOUS_SCALAR_GAIN_GATE_SPEC
)
NONAUTONOMOUS_SCALAR_GAIN_GATE_SPEC.loader.exec_module(
    NONAUTONOMOUS_SCALAR_GAIN_GATE_MODULE
)

RADIAL_PAYOFF_BELLMAN_PILOT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "radial_payoff_bellman_pilot.py"
)
RADIAL_PAYOFF_BELLMAN_PILOT_SPEC = importlib.util.spec_from_file_location(
    "radial_payoff_bellman_pilot",
    RADIAL_PAYOFF_BELLMAN_PILOT_SCRIPT,
)
assert (
    RADIAL_PAYOFF_BELLMAN_PILOT_SPEC is not None
    and RADIAL_PAYOFF_BELLMAN_PILOT_SPEC.loader is not None
)
RADIAL_PAYOFF_BELLMAN_PILOT_MODULE = importlib.util.module_from_spec(
    RADIAL_PAYOFF_BELLMAN_PILOT_SPEC
)
RADIAL_PAYOFF_BELLMAN_PILOT_SPEC.loader.exec_module(
    RADIAL_PAYOFF_BELLMAN_PILOT_MODULE
)

RADIAL_PAYOFF_SUPERSOLUTION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "radial_payoff_supersolution_audit.py"
)
RADIAL_PAYOFF_SUPERSOLUTION_SPEC = importlib.util.spec_from_file_location(
    "radial_payoff_supersolution_audit",
    RADIAL_PAYOFF_SUPERSOLUTION_SCRIPT,
)
assert (
    RADIAL_PAYOFF_SUPERSOLUTION_SPEC is not None
    and RADIAL_PAYOFF_SUPERSOLUTION_SPEC.loader is not None
)
RADIAL_PAYOFF_SUPERSOLUTION_MODULE = importlib.util.module_from_spec(
    RADIAL_PAYOFF_SUPERSOLUTION_SPEC
)
RADIAL_PAYOFF_SUPERSOLUTION_SPEC.loader.exec_module(
    RADIAL_PAYOFF_SUPERSOLUTION_MODULE
)

RADIAL_PAYOFF_INTERVAL_CERTIFICATE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "radial_payoff_interval_certificate.py"
)
RADIAL_PAYOFF_INTERVAL_CERTIFICATE_SPEC = (
    importlib.util.spec_from_file_location(
        "radial_payoff_interval_certificate",
        RADIAL_PAYOFF_INTERVAL_CERTIFICATE_SCRIPT,
    )
)
assert (
    RADIAL_PAYOFF_INTERVAL_CERTIFICATE_SPEC is not None
    and RADIAL_PAYOFF_INTERVAL_CERTIFICATE_SPEC.loader is not None
)
RADIAL_PAYOFF_INTERVAL_CERTIFICATE_MODULE = (
    importlib.util.module_from_spec(
        RADIAL_PAYOFF_INTERVAL_CERTIFICATE_SPEC
    )
)
RADIAL_PAYOFF_INTERVAL_CERTIFICATE_SPEC.loader.exec_module(
    RADIAL_PAYOFF_INTERVAL_CERTIFICATE_MODULE
)

RADIAL_BELLMAN_DOOB_PERTURBATION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "radial_bellman_doob_perturbation_audit.py"
)
RADIAL_BELLMAN_DOOB_PERTURBATION_SPEC = (
    importlib.util.spec_from_file_location(
        "radial_bellman_doob_perturbation_audit",
        RADIAL_BELLMAN_DOOB_PERTURBATION_SCRIPT,
    )
)
assert (
    RADIAL_BELLMAN_DOOB_PERTURBATION_SPEC is not None
    and RADIAL_BELLMAN_DOOB_PERTURBATION_SPEC.loader is not None
)
RADIAL_BELLMAN_DOOB_PERTURBATION_MODULE = (
    importlib.util.module_from_spec(
        RADIAL_BELLMAN_DOOB_PERTURBATION_SPEC
    )
)
RADIAL_BELLMAN_DOOB_PERTURBATION_SPEC.loader.exec_module(
    RADIAL_BELLMAN_DOOB_PERTURBATION_MODULE
)

CRITICAL_COLLAR_TRANSFER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "critical_collar_transfer_audit.py"
)
CRITICAL_COLLAR_TRANSFER_SPEC = importlib.util.spec_from_file_location(
    "critical_collar_transfer_audit",
    CRITICAL_COLLAR_TRANSFER_SCRIPT,
)
assert (
    CRITICAL_COLLAR_TRANSFER_SPEC is not None
    and CRITICAL_COLLAR_TRANSFER_SPEC.loader is not None
)
CRITICAL_COLLAR_TRANSFER_MODULE = importlib.util.module_from_spec(
    CRITICAL_COLLAR_TRANSFER_SPEC
)
CRITICAL_COLLAR_TRANSFER_SPEC.loader.exec_module(
    CRITICAL_COLLAR_TRANSFER_MODULE
)

RADIAL_BARRIER_CUTOFF_ENERGY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "radial_barrier_cutoff_energy_pilot.py"
)
RADIAL_BARRIER_CUTOFF_ENERGY_SPEC = importlib.util.spec_from_file_location(
    "radial_barrier_cutoff_energy_pilot",
    RADIAL_BARRIER_CUTOFF_ENERGY_SCRIPT,
)
assert (
    RADIAL_BARRIER_CUTOFF_ENERGY_SPEC is not None
    and RADIAL_BARRIER_CUTOFF_ENERGY_SPEC.loader is not None
)
RADIAL_BARRIER_CUTOFF_ENERGY_MODULE = importlib.util.module_from_spec(
    RADIAL_BARRIER_CUTOFF_ENERGY_SPEC
)
RADIAL_BARRIER_CUTOFF_ENERGY_SPEC.loader.exec_module(
    RADIAL_BARRIER_CUTOFF_ENERGY_MODULE
)

RADIAL_COLLAR_TRACE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "radial_collar_trace_pilot.py"
)
RADIAL_COLLAR_TRACE_SPEC = importlib.util.spec_from_file_location(
    "radial_collar_trace_pilot",
    RADIAL_COLLAR_TRACE_SCRIPT,
)
assert (
    RADIAL_COLLAR_TRACE_SPEC is not None
    and RADIAL_COLLAR_TRACE_SPEC.loader is not None
)
RADIAL_COLLAR_TRACE_MODULE = importlib.util.module_from_spec(
    RADIAL_COLLAR_TRACE_SPEC
)
RADIAL_COLLAR_TRACE_SPEC.loader.exec_module(
    RADIAL_COLLAR_TRACE_MODULE
)

RADIAL_COLLAR_FREQUENCY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "radial_collar_frequency_pilot.py"
)
RADIAL_COLLAR_FREQUENCY_SPEC = importlib.util.spec_from_file_location(
    "radial_collar_frequency_pilot",
    RADIAL_COLLAR_FREQUENCY_SCRIPT,
)
assert (
    RADIAL_COLLAR_FREQUENCY_SPEC is not None
    and RADIAL_COLLAR_FREQUENCY_SPEC.loader is not None
)
RADIAL_COLLAR_FREQUENCY_MODULE = importlib.util.module_from_spec(
    RADIAL_COLLAR_FREQUENCY_SPEC
)
RADIAL_COLLAR_FREQUENCY_SPEC.loader.exec_module(
    RADIAL_COLLAR_FREQUENCY_MODULE
)

PROTECTED_COLLAR_PARTITION_NO_GO_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "protected_collar_partition_no_go_audit.py"
)
PROTECTED_COLLAR_PARTITION_NO_GO_SPEC = (
    importlib.util.spec_from_file_location(
        "protected_collar_partition_no_go_audit",
        PROTECTED_COLLAR_PARTITION_NO_GO_SCRIPT,
    )
)
assert (
    PROTECTED_COLLAR_PARTITION_NO_GO_SPEC is not None
    and PROTECTED_COLLAR_PARTITION_NO_GO_SPEC.loader is not None
)
PROTECTED_COLLAR_PARTITION_NO_GO_MODULE = (
    importlib.util.module_from_spec(
        PROTECTED_COLLAR_PARTITION_NO_GO_SPEC
    )
)
PROTECTED_COLLAR_PARTITION_NO_GO_SPEC.loader.exec_module(
    PROTECTED_COLLAR_PARTITION_NO_GO_MODULE
)

INTERACTION_MARKED_LOCALIZATION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "interaction_marked_localization_audit.py"
)
INTERACTION_MARKED_LOCALIZATION_SPEC = importlib.util.spec_from_file_location(
    "interaction_marked_localization_audit",
    INTERACTION_MARKED_LOCALIZATION_SCRIPT,
)
assert (
    INTERACTION_MARKED_LOCALIZATION_SPEC is not None
    and INTERACTION_MARKED_LOCALIZATION_SPEC.loader is not None
)
INTERACTION_MARKED_LOCALIZATION_MODULE = importlib.util.module_from_spec(
    INTERACTION_MARKED_LOCALIZATION_SPEC
)
INTERACTION_MARKED_LOCALIZATION_SPEC.loader.exec_module(
    INTERACTION_MARKED_LOCALIZATION_MODULE
)

RADIAL_H1_PAYOFF_SUPERSOLUTION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "radial_h1_payoff_supersolution_pilot.py"
)
RADIAL_H1_PAYOFF_SUPERSOLUTION_SPEC = importlib.util.spec_from_file_location(
    "radial_h1_payoff_supersolution_pilot",
    RADIAL_H1_PAYOFF_SUPERSOLUTION_SCRIPT,
)
assert (
    RADIAL_H1_PAYOFF_SUPERSOLUTION_SPEC is not None
    and RADIAL_H1_PAYOFF_SUPERSOLUTION_SPEC.loader is not None
)
RADIAL_H1_PAYOFF_SUPERSOLUTION_MODULE = importlib.util.module_from_spec(
    RADIAL_H1_PAYOFF_SUPERSOLUTION_SPEC
)
RADIAL_H1_PAYOFF_SUPERSOLUTION_SPEC.loader.exec_module(
    RADIAL_H1_PAYOFF_SUPERSOLUTION_MODULE
)

RADIAL_H1_PAYOFF_INTERVAL_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "radial_h1_payoff_interval_certificate.py"
)
RADIAL_H1_PAYOFF_INTERVAL_SPEC = importlib.util.spec_from_file_location(
    "radial_h1_payoff_interval_certificate",
    RADIAL_H1_PAYOFF_INTERVAL_SCRIPT,
)
assert (
    RADIAL_H1_PAYOFF_INTERVAL_SPEC is not None
    and RADIAL_H1_PAYOFF_INTERVAL_SPEC.loader is not None
)
RADIAL_H1_PAYOFF_INTERVAL_MODULE = importlib.util.module_from_spec(
    RADIAL_H1_PAYOFF_INTERVAL_SPEC
)
RADIAL_H1_PAYOFF_INTERVAL_SPEC.loader.exec_module(
    RADIAL_H1_PAYOFF_INTERVAL_MODULE
)

AVERAGED_ENTRY_TRACE_GATE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "averaged_entry_trace_gate.py"
)
AVERAGED_ENTRY_TRACE_GATE_SPEC = importlib.util.spec_from_file_location(
    "averaged_entry_trace_gate", AVERAGED_ENTRY_TRACE_GATE_SCRIPT
)
assert (
    AVERAGED_ENTRY_TRACE_GATE_SPEC is not None
    and AVERAGED_ENTRY_TRACE_GATE_SPEC.loader is not None
)
AVERAGED_ENTRY_TRACE_GATE_MODULE = importlib.util.module_from_spec(
    AVERAGED_ENTRY_TRACE_GATE_SPEC
)
AVERAGED_ENTRY_TRACE_GATE_SPEC.loader.exec_module(
    AVERAGED_ENTRY_TRACE_GATE_MODULE
)

EXTERIOR_RETURN_TAIL_GATE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "exterior_return_tail_gate.py"
)
EXTERIOR_RETURN_TAIL_GATE_SPEC = importlib.util.spec_from_file_location(
    "exterior_return_tail_gate", EXTERIOR_RETURN_TAIL_GATE_SCRIPT
)
assert (
    EXTERIOR_RETURN_TAIL_GATE_SPEC is not None
    and EXTERIOR_RETURN_TAIL_GATE_SPEC.loader is not None
)
EXTERIOR_RETURN_TAIL_GATE_MODULE = importlib.util.module_from_spec(
    EXTERIOR_RETURN_TAIL_GATE_SPEC
)
EXTERIOR_RETURN_TAIL_GATE_SPEC.loader.exec_module(
    EXTERIOR_RETURN_TAIL_GATE_MODULE
)

CYLINDRICAL_BROWNIAN_RETURN_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "cylindrical_brownian_return_pilot.py"
)
CYLINDRICAL_BROWNIAN_RETURN_SPEC = importlib.util.spec_from_file_location(
    "cylindrical_brownian_return_pilot",
    CYLINDRICAL_BROWNIAN_RETURN_SCRIPT,
)
assert (
    CYLINDRICAL_BROWNIAN_RETURN_SPEC is not None
    and CYLINDRICAL_BROWNIAN_RETURN_SPEC.loader is not None
)
CYLINDRICAL_BROWNIAN_RETURN_MODULE = importlib.util.module_from_spec(
    CYLINDRICAL_BROWNIAN_RETURN_SPEC
)
CYLINDRICAL_BROWNIAN_RETURN_SPEC.loader.exec_module(
    CYLINDRICAL_BROWNIAN_RETURN_MODULE
)

BRANCH_RESOLVED_ENTRY_RENEWAL_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "branch_resolved_entry_renewal_audit.py"
)
BRANCH_RESOLVED_ENTRY_RENEWAL_SPEC = (
    importlib.util.spec_from_file_location(
        "branch_resolved_entry_renewal_audit",
        BRANCH_RESOLVED_ENTRY_RENEWAL_SCRIPT,
    )
)
assert (
    BRANCH_RESOLVED_ENTRY_RENEWAL_SPEC is not None
    and BRANCH_RESOLVED_ENTRY_RENEWAL_SPEC.loader is not None
)
BRANCH_RESOLVED_ENTRY_RENEWAL_MODULE = importlib.util.module_from_spec(
    BRANCH_RESOLVED_ENTRY_RENEWAL_SPEC
)
BRANCH_RESOLVED_ENTRY_RENEWAL_SPEC.loader.exec_module(
    BRANCH_RESOLVED_ENTRY_RENEWAL_MODULE
)

SPLIT_ENTRY_DENSITY_INHERITANCE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "split_entry_density_inheritance_audit.py"
)
SPLIT_ENTRY_DENSITY_INHERITANCE_SPEC = (
    importlib.util.spec_from_file_location(
        "split_entry_density_inheritance_audit",
        SPLIT_ENTRY_DENSITY_INHERITANCE_SCRIPT,
    )
)
assert (
    SPLIT_ENTRY_DENSITY_INHERITANCE_SPEC is not None
    and SPLIT_ENTRY_DENSITY_INHERITANCE_SPEC.loader is not None
)
SPLIT_ENTRY_DENSITY_INHERITANCE_MODULE = (
    importlib.util.module_from_spec(
        SPLIT_ENTRY_DENSITY_INHERITANCE_SPEC
    )
)
SPLIT_ENTRY_DENSITY_INHERITANCE_SPEC.loader.exec_module(
    SPLIT_ENTRY_DENSITY_INHERITANCE_MODULE
)

AFFINE_EXTERIOR_AXIAL_COMPENSATION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "affine_exterior_axial_compensation_audit.py"
)
AFFINE_EXTERIOR_AXIAL_COMPENSATION_SPEC = (
    importlib.util.spec_from_file_location(
        "affine_exterior_axial_compensation_audit",
        AFFINE_EXTERIOR_AXIAL_COMPENSATION_SCRIPT,
    )
)
assert (
    AFFINE_EXTERIOR_AXIAL_COMPENSATION_SPEC is not None
    and AFFINE_EXTERIOR_AXIAL_COMPENSATION_SPEC.loader is not None
)
AFFINE_EXTERIOR_AXIAL_COMPENSATION_MODULE = (
    importlib.util.module_from_spec(
        AFFINE_EXTERIOR_AXIAL_COMPENSATION_SPEC
    )
)
AFFINE_EXTERIOR_AXIAL_COMPENSATION_SPEC.loader.exec_module(
    AFFINE_EXTERIOR_AXIAL_COMPENSATION_MODULE
)

ANISOTROPIC_AFFINE_EXTERIOR_TAIL_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "anisotropic_affine_exterior_tail_gate.py"
)
ANISOTROPIC_AFFINE_EXTERIOR_TAIL_SPEC = (
    importlib.util.spec_from_file_location(
        "anisotropic_affine_exterior_tail_gate",
        ANISOTROPIC_AFFINE_EXTERIOR_TAIL_SCRIPT,
    )
)
assert (
    ANISOTROPIC_AFFINE_EXTERIOR_TAIL_SPEC is not None
    and ANISOTROPIC_AFFINE_EXTERIOR_TAIL_SPEC.loader is not None
)
ANISOTROPIC_AFFINE_EXTERIOR_TAIL_MODULE = (
    importlib.util.module_from_spec(
        ANISOTROPIC_AFFINE_EXTERIOR_TAIL_SPEC
    )
)
ANISOTROPIC_AFFINE_EXTERIOR_TAIL_SPEC.loader.exec_module(
    ANISOTROPIC_AFFINE_EXTERIOR_TAIL_MODULE
)

NEUTRAL_STRIP_STORAGE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_storage_gate.py"
)
NEUTRAL_STRIP_STORAGE_SPEC = importlib.util.spec_from_file_location(
    "neutral_strip_storage_gate", NEUTRAL_STRIP_STORAGE_SCRIPT
)
assert (
    NEUTRAL_STRIP_STORAGE_SPEC is not None
    and NEUTRAL_STRIP_STORAGE_SPEC.loader is not None
)
NEUTRAL_STRIP_STORAGE_MODULE = importlib.util.module_from_spec(
    NEUTRAL_STRIP_STORAGE_SPEC
)
NEUTRAL_STRIP_STORAGE_SPEC.loader.exec_module(
    NEUTRAL_STRIP_STORAGE_MODULE
)

NEUTRAL_STRIP_BRANCH_RESOLVENT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_branch_resolvent_pilot.py"
)
NEUTRAL_STRIP_BRANCH_RESOLVENT_SPEC = importlib.util.spec_from_file_location(
    "neutral_strip_branch_resolvent_pilot",
    NEUTRAL_STRIP_BRANCH_RESOLVENT_SCRIPT,
)
assert (
    NEUTRAL_STRIP_BRANCH_RESOLVENT_SPEC is not None
    and NEUTRAL_STRIP_BRANCH_RESOLVENT_SPEC.loader is not None
)
NEUTRAL_STRIP_BRANCH_RESOLVENT_MODULE = importlib.util.module_from_spec(
    NEUTRAL_STRIP_BRANCH_RESOLVENT_SPEC
)
NEUTRAL_STRIP_BRANCH_RESOLVENT_SPEC.loader.exec_module(
    NEUTRAL_STRIP_BRANCH_RESOLVENT_MODULE
)

NEUTRAL_STRIP_AXIAL_PATCH_BRANCH_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_axial_patch_branch_pilot.py"
)
NEUTRAL_STRIP_AXIAL_PATCH_BRANCH_SPEC = importlib.util.spec_from_file_location(
    "neutral_strip_axial_patch_branch_pilot",
    NEUTRAL_STRIP_AXIAL_PATCH_BRANCH_SCRIPT,
)
assert (
    NEUTRAL_STRIP_AXIAL_PATCH_BRANCH_SPEC is not None
    and NEUTRAL_STRIP_AXIAL_PATCH_BRANCH_SPEC.loader is not None
)
NEUTRAL_STRIP_AXIAL_PATCH_BRANCH_MODULE = importlib.util.module_from_spec(
    NEUTRAL_STRIP_AXIAL_PATCH_BRANCH_SPEC
)
NEUTRAL_STRIP_AXIAL_PATCH_BRANCH_SPEC.loader.exec_module(
    NEUTRAL_STRIP_AXIAL_PATCH_BRANCH_MODULE
)

GEOMETRIC_WALL_SPLIT_COMPATIBILITY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "geometric_wall_split_compatibility_audit.py"
)
GEOMETRIC_WALL_SPLIT_COMPATIBILITY_SPEC = importlib.util.spec_from_file_location(
    "geometric_wall_split_compatibility_audit",
    GEOMETRIC_WALL_SPLIT_COMPATIBILITY_SCRIPT,
)
assert (
    GEOMETRIC_WALL_SPLIT_COMPATIBILITY_SPEC is not None
    and GEOMETRIC_WALL_SPLIT_COMPATIBILITY_SPEC.loader is not None
)
GEOMETRIC_WALL_SPLIT_COMPATIBILITY_MODULE = importlib.util.module_from_spec(
    GEOMETRIC_WALL_SPLIT_COMPATIBILITY_SPEC
)
GEOMETRIC_WALL_SPLIT_COMPATIBILITY_SPEC.loader.exec_module(
    GEOMETRIC_WALL_SPLIT_COMPATIBILITY_MODULE
)

NEUTRAL_STRIP_SAME_SCALE_WIDTH_SWEEP_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_same_scale_width_sweep.py"
)
NEUTRAL_STRIP_SAME_SCALE_WIDTH_SWEEP_SPEC = importlib.util.spec_from_file_location(
    "neutral_strip_same_scale_width_sweep",
    NEUTRAL_STRIP_SAME_SCALE_WIDTH_SWEEP_SCRIPT,
)
assert (
    NEUTRAL_STRIP_SAME_SCALE_WIDTH_SWEEP_SPEC is not None
    and NEUTRAL_STRIP_SAME_SCALE_WIDTH_SWEEP_SPEC.loader is not None
)
NEUTRAL_STRIP_SAME_SCALE_WIDTH_SWEEP_MODULE = importlib.util.module_from_spec(
    NEUTRAL_STRIP_SAME_SCALE_WIDTH_SWEEP_SPEC
)
NEUTRAL_STRIP_SAME_SCALE_WIDTH_SWEEP_SPEC.loader.exec_module(
    NEUTRAL_STRIP_SAME_SCALE_WIDTH_SWEEP_MODULE
)

GEOMETRY_TRIGGERED_MIGRATING_CHILD_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "geometry_triggered_migrating_child_pilot.py"
)
GEOMETRY_TRIGGERED_MIGRATING_CHILD_SPEC = (
    importlib.util.spec_from_file_location(
        "geometry_triggered_migrating_child_pilot",
        GEOMETRY_TRIGGERED_MIGRATING_CHILD_SCRIPT,
    )
)
assert (
    GEOMETRY_TRIGGERED_MIGRATING_CHILD_SPEC is not None
    and GEOMETRY_TRIGGERED_MIGRATING_CHILD_SPEC.loader is not None
)
GEOMETRY_TRIGGERED_MIGRATING_CHILD_MODULE = importlib.util.module_from_spec(
    GEOMETRY_TRIGGERED_MIGRATING_CHILD_SPEC
)
GEOMETRY_TRIGGERED_MIGRATING_CHILD_SPEC.loader.exec_module(
    GEOMETRY_TRIGGERED_MIGRATING_CHILD_MODULE
)

MIGRATING_CORE_RESIDUAL_BUDGET_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "migrating_core_residual_budget_audit.py"
)
MIGRATING_CORE_RESIDUAL_BUDGET_SPEC = importlib.util.spec_from_file_location(
    "migrating_core_residual_budget_audit",
    MIGRATING_CORE_RESIDUAL_BUDGET_SCRIPT,
)
assert (
    MIGRATING_CORE_RESIDUAL_BUDGET_SPEC is not None
    and MIGRATING_CORE_RESIDUAL_BUDGET_SPEC.loader is not None
)
MIGRATING_CORE_RESIDUAL_BUDGET_MODULE = importlib.util.module_from_spec(
    MIGRATING_CORE_RESIDUAL_BUDGET_SPEC
)
MIGRATING_CORE_RESIDUAL_BUDGET_SPEC.loader.exec_module(
    MIGRATING_CORE_RESIDUAL_BUDGET_MODULE
)

WALL_STOPPING_TRACE_COMPOSITION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "wall_stopping_trace_composition_audit.py"
)
WALL_STOPPING_TRACE_COMPOSITION_SPEC = importlib.util.spec_from_file_location(
    "wall_stopping_trace_composition_audit",
    WALL_STOPPING_TRACE_COMPOSITION_SCRIPT,
)
assert (
    WALL_STOPPING_TRACE_COMPOSITION_SPEC is not None
    and WALL_STOPPING_TRACE_COMPOSITION_SPEC.loader is not None
)
WALL_STOPPING_TRACE_COMPOSITION_MODULE = importlib.util.module_from_spec(
    WALL_STOPPING_TRACE_COMPOSITION_SPEC
)
WALL_STOPPING_TRACE_COMPOSITION_SPEC.loader.exec_module(
    WALL_STOPPING_TRACE_COMPOSITION_MODULE
)

NEUTRAL_STRIP_RETURN_DENSITY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_return_density_pilot.py"
)
NEUTRAL_STRIP_RETURN_DENSITY_SPEC = importlib.util.spec_from_file_location(
    "neutral_strip_return_density_pilot",
    NEUTRAL_STRIP_RETURN_DENSITY_SCRIPT,
)
assert (
    NEUTRAL_STRIP_RETURN_DENSITY_SPEC is not None
    and NEUTRAL_STRIP_RETURN_DENSITY_SPEC.loader is not None
)
NEUTRAL_STRIP_RETURN_DENSITY_MODULE = importlib.util.module_from_spec(
    NEUTRAL_STRIP_RETURN_DENSITY_SPEC
)
NEUTRAL_STRIP_RETURN_DENSITY_SPEC.loader.exec_module(
    NEUTRAL_STRIP_RETURN_DENSITY_MODULE
)

NEUTRAL_STRIP_BOUNDARY_DISCRETIZATION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_boundary_density_discretization_audit.py"
)
NEUTRAL_STRIP_BOUNDARY_DISCRETIZATION_SPEC = (
    importlib.util.spec_from_file_location(
        "neutral_strip_boundary_density_discretization_audit",
        NEUTRAL_STRIP_BOUNDARY_DISCRETIZATION_SCRIPT,
    )
)
assert (
    NEUTRAL_STRIP_BOUNDARY_DISCRETIZATION_SPEC is not None
    and NEUTRAL_STRIP_BOUNDARY_DISCRETIZATION_SPEC.loader is not None
)
NEUTRAL_STRIP_BOUNDARY_DISCRETIZATION_MODULE = (
    importlib.util.module_from_spec(
        NEUTRAL_STRIP_BOUNDARY_DISCRETIZATION_SPEC
    )
)
NEUTRAL_STRIP_BOUNDARY_DISCRETIZATION_SPEC.loader.exec_module(
    NEUTRAL_STRIP_BOUNDARY_DISCRETIZATION_MODULE
)

NEUTRAL_STRIP_REVERSIBLE_BOUNDARY_FEM_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_reversible_boundary_fem_pilot.py"
)
NEUTRAL_STRIP_REVERSIBLE_BOUNDARY_FEM_SPEC = (
    importlib.util.spec_from_file_location(
        "neutral_strip_reversible_boundary_fem_pilot",
        NEUTRAL_STRIP_REVERSIBLE_BOUNDARY_FEM_SCRIPT,
    )
)
assert (
    NEUTRAL_STRIP_REVERSIBLE_BOUNDARY_FEM_SPEC is not None
    and NEUTRAL_STRIP_REVERSIBLE_BOUNDARY_FEM_SPEC.loader is not None
)
NEUTRAL_STRIP_REVERSIBLE_BOUNDARY_FEM_MODULE = (
    importlib.util.module_from_spec(
        NEUTRAL_STRIP_REVERSIBLE_BOUNDARY_FEM_SPEC
    )
)
NEUTRAL_STRIP_REVERSIBLE_BOUNDARY_FEM_SPEC.loader.exec_module(
    NEUTRAL_STRIP_REVERSIBLE_BOUNDARY_FEM_MODULE
)

NEUTRAL_STRIP_REVERSIBLE_SPECTRAL_TAIL_WIDTH_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_reversible_spectral_tail_width_audit.py"
)
NEUTRAL_STRIP_REVERSIBLE_SPECTRAL_TAIL_WIDTH_SPEC = (
    importlib.util.spec_from_file_location(
        "neutral_strip_reversible_spectral_tail_width_audit",
        NEUTRAL_STRIP_REVERSIBLE_SPECTRAL_TAIL_WIDTH_SCRIPT,
    )
)
assert (
    NEUTRAL_STRIP_REVERSIBLE_SPECTRAL_TAIL_WIDTH_SPEC is not None
    and NEUTRAL_STRIP_REVERSIBLE_SPECTRAL_TAIL_WIDTH_SPEC.loader is not None
)
NEUTRAL_STRIP_REVERSIBLE_SPECTRAL_TAIL_WIDTH_MODULE = (
    importlib.util.module_from_spec(
        NEUTRAL_STRIP_REVERSIBLE_SPECTRAL_TAIL_WIDTH_SPEC
    )
)
NEUTRAL_STRIP_REVERSIBLE_SPECTRAL_TAIL_WIDTH_SPEC.loader.exec_module(
    NEUTRAL_STRIP_REVERSIBLE_SPECTRAL_TAIL_WIDTH_MODULE
)

NEUTRAL_STRIP_REVERSIBLE_FINITE_TIME_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_reversible_finite_time_certificate.py"
)
NEUTRAL_STRIP_REVERSIBLE_FINITE_TIME_SPEC = (
    importlib.util.spec_from_file_location(
        "neutral_strip_reversible_finite_time_certificate",
        NEUTRAL_STRIP_REVERSIBLE_FINITE_TIME_SCRIPT,
    )
)
assert (
    NEUTRAL_STRIP_REVERSIBLE_FINITE_TIME_SPEC is not None
    and NEUTRAL_STRIP_REVERSIBLE_FINITE_TIME_SPEC.loader is not None
)
NEUTRAL_STRIP_REVERSIBLE_FINITE_TIME_MODULE = (
    importlib.util.module_from_spec(
        NEUTRAL_STRIP_REVERSIBLE_FINITE_TIME_SPEC
    )
)
NEUTRAL_STRIP_REVERSIBLE_FINITE_TIME_SPEC.loader.exec_module(
    NEUTRAL_STRIP_REVERSIBLE_FINITE_TIME_MODULE
)

NEUTRAL_STRIP_X_EXIT_CORRECTION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_x_exit_correction_audit.py"
)
NEUTRAL_STRIP_X_EXIT_CORRECTION_SPEC = importlib.util.spec_from_file_location(
    "neutral_strip_x_exit_correction_audit",
    NEUTRAL_STRIP_X_EXIT_CORRECTION_SCRIPT,
)
assert (
    NEUTRAL_STRIP_X_EXIT_CORRECTION_SPEC is not None
    and NEUTRAL_STRIP_X_EXIT_CORRECTION_SPEC.loader is not None
)
NEUTRAL_STRIP_X_EXIT_CORRECTION_MODULE = importlib.util.module_from_spec(
    NEUTRAL_STRIP_X_EXIT_CORRECTION_SPEC
)
NEUTRAL_STRIP_X_EXIT_CORRECTION_SPEC.loader.exec_module(
    NEUTRAL_STRIP_X_EXIT_CORRECTION_MODULE
)

NEUTRAL_STRIP_REVERSIBLE_FEM_CONSISTENCY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_reversible_fem_consistency_gate.py"
)
NEUTRAL_STRIP_REVERSIBLE_FEM_CONSISTENCY_SPEC = (
    importlib.util.spec_from_file_location(
        "neutral_strip_reversible_fem_consistency_gate",
        NEUTRAL_STRIP_REVERSIBLE_FEM_CONSISTENCY_SCRIPT,
    )
)
assert (
    NEUTRAL_STRIP_REVERSIBLE_FEM_CONSISTENCY_SPEC is not None
    and NEUTRAL_STRIP_REVERSIBLE_FEM_CONSISTENCY_SPEC.loader is not None
)
NEUTRAL_STRIP_REVERSIBLE_FEM_CONSISTENCY_MODULE = (
    importlib.util.module_from_spec(
        NEUTRAL_STRIP_REVERSIBLE_FEM_CONSISTENCY_SPEC
    )
)
NEUTRAL_STRIP_REVERSIBLE_FEM_CONSISTENCY_SPEC.loader.exec_module(
    NEUTRAL_STRIP_REVERSIBLE_FEM_CONSISTENCY_MODULE
)

NEUTRAL_STRIP_PARABOLIC_SPECTRAL_SPLIT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_parabolic_spectral_split_audit.py"
)
NEUTRAL_STRIP_PARABOLIC_SPECTRAL_SPLIT_SPEC = (
    importlib.util.spec_from_file_location(
        "neutral_strip_parabolic_spectral_split_audit",
        NEUTRAL_STRIP_PARABOLIC_SPECTRAL_SPLIT_SCRIPT,
    )
)
assert (
    NEUTRAL_STRIP_PARABOLIC_SPECTRAL_SPLIT_SPEC is not None
    and NEUTRAL_STRIP_PARABOLIC_SPECTRAL_SPLIT_SPEC.loader is not None
)
NEUTRAL_STRIP_PARABOLIC_SPECTRAL_SPLIT_MODULE = (
    importlib.util.module_from_spec(
        NEUTRAL_STRIP_PARABOLIC_SPECTRAL_SPLIT_SPEC
    )
)
NEUTRAL_STRIP_PARABOLIC_SPECTRAL_SPLIT_SPEC.loader.exec_module(
    NEUTRAL_STRIP_PARABOLIC_SPECTRAL_SPLIT_MODULE
)

NEUTRAL_STRIP_FIRST_WINDOW_BROWNIAN_MAJORANT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_first_window_brownian_majorant_audit.py"
)
NEUTRAL_STRIP_FIRST_WINDOW_BROWNIAN_MAJORANT_SPEC = (
    importlib.util.spec_from_file_location(
        "neutral_strip_first_window_brownian_majorant_audit",
        NEUTRAL_STRIP_FIRST_WINDOW_BROWNIAN_MAJORANT_SCRIPT,
    )
)
assert (
    NEUTRAL_STRIP_FIRST_WINDOW_BROWNIAN_MAJORANT_SPEC is not None
    and NEUTRAL_STRIP_FIRST_WINDOW_BROWNIAN_MAJORANT_SPEC.loader is not None
)
NEUTRAL_STRIP_FIRST_WINDOW_BROWNIAN_MAJORANT_MODULE = (
    importlib.util.module_from_spec(
        NEUTRAL_STRIP_FIRST_WINDOW_BROWNIAN_MAJORANT_SPEC
    )
)
NEUTRAL_STRIP_FIRST_WINDOW_BROWNIAN_MAJORANT_SPEC.loader.exec_module(
    NEUTRAL_STRIP_FIRST_WINDOW_BROWNIAN_MAJORANT_MODULE
)

NEUTRAL_STRIP_FIRST_WINDOW_MAXIMUM_BRIDGE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_first_window_maximum_bridge_certificate.py"
)
NEUTRAL_STRIP_FIRST_WINDOW_MAXIMUM_BRIDGE_SPEC = (
    importlib.util.spec_from_file_location(
        "neutral_strip_first_window_maximum_bridge_certificate",
        NEUTRAL_STRIP_FIRST_WINDOW_MAXIMUM_BRIDGE_SCRIPT,
    )
)
assert (
    NEUTRAL_STRIP_FIRST_WINDOW_MAXIMUM_BRIDGE_SPEC is not None
    and NEUTRAL_STRIP_FIRST_WINDOW_MAXIMUM_BRIDGE_SPEC.loader is not None
)
NEUTRAL_STRIP_FIRST_WINDOW_MAXIMUM_BRIDGE_MODULE = (
    importlib.util.module_from_spec(
        NEUTRAL_STRIP_FIRST_WINDOW_MAXIMUM_BRIDGE_SPEC
    )
)
NEUTRAL_STRIP_FIRST_WINDOW_MAXIMUM_BRIDGE_SPEC.loader.exec_module(
    NEUTRAL_STRIP_FIRST_WINDOW_MAXIMUM_BRIDGE_MODULE
)

NEUTRAL_STRIP_TRANSIENT_CONORMAL_LOW_BLOCK_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_transient_conormal_low_block_gate.py"
)
NEUTRAL_STRIP_TRANSIENT_CONORMAL_LOW_BLOCK_SPEC = (
    importlib.util.spec_from_file_location(
        "neutral_strip_transient_conormal_low_block_gate",
        NEUTRAL_STRIP_TRANSIENT_CONORMAL_LOW_BLOCK_SCRIPT,
    )
)
assert (
    NEUTRAL_STRIP_TRANSIENT_CONORMAL_LOW_BLOCK_SPEC is not None
    and NEUTRAL_STRIP_TRANSIENT_CONORMAL_LOW_BLOCK_SPEC.loader is not None
)
NEUTRAL_STRIP_TRANSIENT_CONORMAL_LOW_BLOCK_MODULE = (
    importlib.util.module_from_spec(
        NEUTRAL_STRIP_TRANSIENT_CONORMAL_LOW_BLOCK_SPEC
    )
)
NEUTRAL_STRIP_TRANSIENT_CONORMAL_LOW_BLOCK_SPEC.loader.exec_module(
    NEUTRAL_STRIP_TRANSIENT_CONORMAL_LOW_BLOCK_MODULE
)

NEUTRAL_STRIP_COMMON_CIRCLE_TIME_SLAB_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_common_circle_source_time_slab_certificate.py"
)
NEUTRAL_STRIP_COMMON_CIRCLE_TIME_SLAB_SPEC = (
    importlib.util.spec_from_file_location(
        "neutral_strip_common_circle_source_time_slab_certificate",
        NEUTRAL_STRIP_COMMON_CIRCLE_TIME_SLAB_SCRIPT,
    )
)
assert (
    NEUTRAL_STRIP_COMMON_CIRCLE_TIME_SLAB_SPEC is not None
    and NEUTRAL_STRIP_COMMON_CIRCLE_TIME_SLAB_SPEC.loader is not None
)
NEUTRAL_STRIP_COMMON_CIRCLE_TIME_SLAB_MODULE = (
    importlib.util.module_from_spec(
        NEUTRAL_STRIP_COMMON_CIRCLE_TIME_SLAB_SPEC
    )
)
NEUTRAL_STRIP_COMMON_CIRCLE_TIME_SLAB_SPEC.loader.exec_module(
    NEUTRAL_STRIP_COMMON_CIRCLE_TIME_SLAB_MODULE
)

WALL_MIGRATION_CHILD_RETURN_DENSITY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "wall_migration_child_return_density_pilot.py"
)
WALL_MIGRATION_CHILD_RETURN_DENSITY_SPEC = (
    importlib.util.spec_from_file_location(
        "wall_migration_child_return_density_pilot",
        WALL_MIGRATION_CHILD_RETURN_DENSITY_SCRIPT,
    )
)
assert (
    WALL_MIGRATION_CHILD_RETURN_DENSITY_SPEC is not None
    and WALL_MIGRATION_CHILD_RETURN_DENSITY_SPEC.loader is not None
)
WALL_MIGRATION_CHILD_RETURN_DENSITY_MODULE = (
    importlib.util.module_from_spec(
        WALL_MIGRATION_CHILD_RETURN_DENSITY_SPEC
    )
)
WALL_MIGRATION_CHILD_RETURN_DENSITY_SPEC.loader.exec_module(
    WALL_MIGRATION_CHILD_RETURN_DENSITY_MODULE
)


class CollisionIdentityTests(unittest.TestCase):
    def test_symbolic_audit(self) -> None:
        checks = MODULE.verify_identities()
        boolean_checks = {
            name: value for name, value in checks.items() if isinstance(value, bool)
        }
        self.assertTrue(boolean_checks)
        self.assertTrue(all(boolean_checks.values()), boolean_checks)

    def test_affine_pressure_hessian(self) -> None:
        checks = MODULE.verify_identities()
        self.assertEqual(
            checks["blowup_pressure_hessian"],
            "Matrix([[0, 0, 0], [0, 0, 0], [0, 0, -6/(T - t)**2]])",
        )

    def test_gram_boundary_dimensions(self) -> None:
        rows = GRAM_MODULE.boundary_table(3)
        self.assertEqual(
            [row["effective_bessel_dimension"] for row in rows], [3, 2, 1]
        )
        self.assertEqual(
            [row["driftless_boundary"] for row in rows],
            ["nonattainable", "nonattainable", "attainable"],
        )

    def test_determinant_has_no_ito_laplacian_drift(self) -> None:
        result = GRAM_MODULE.determinant_laplacian_audit(3)
        self.assertTrue(result["determinant_laplacian_zero"])

    def test_backward_affine_covariance(self) -> None:
        result = BACKWARD_MODULE.affine_covariance_audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["parallel_variance_limit"], "nu/a")

    def test_newtonian_boundary_defect(self) -> None:
        result = NEWTONIAN_MODULE.newtonian_audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(
            result["coincident_limit"], "sqrt(2)/(2*sqrt(pi)*sqrt(nu)*sqrt(t))"
        )

    def test_two_point_vorticity_audit(self) -> None:
        result = TWO_POINT_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["centre_laplacian_coefficient"], "1/2")
        self.assertEqual(result["relative_laplacian_coefficient"], "2")

    def test_strain_boundary_multiplier(self) -> None:
        result = MULTIPLIER_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["multiplier_limit_at_zero"], "0")
        self.assertEqual(result["multiplier_limit_at_infinity"], "1")
        self.assertEqual(
            result["small_scale_coefficient"], "8/(15*sqrt(pi))"
        )

    def test_heat_scale_cubic_cancellation(self) -> None:
        result = HEAT_CUBIC_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertLess(abs(result["cubic_orthogonality_residual"]), 1.0e-12)
        self.assertLess(result["second_order_relative_error"], 1.0e-3)
        self.assertGreater(result["explicit_triad_defect"], 0)

    def test_fourier_triad_collision_audit(self) -> None:
        result = TRIAD_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["allowed_amplitude_plane_rank"], 2)
        self.assertEqual(
            result["two_triad_pure_heat_derivative_at_nu_one"], "14"
        )

    def test_first_crossing_barrier_audit(self) -> None:
        result = BARRIER_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["base_palinstrophy"], "8")
        self.assertEqual(result["threshold_amplitude"], "16*nu/(x - 1)**2")

    def test_adaptive_scale_barrier_audit(self) -> None:
        result = ADAPTIVE_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["base_palinstrophy"], "139/2")
        self.assertAlmostEqual(float(result["stationary_heat_scale"]), 0.609377863436)

    def test_cumulative_defect_audit(self) -> None:
        result = CUMULATIVE_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["semigroup_primitive"], "1/decay_rate")
        self.assertEqual(result["lowpass_bound_scaling_exponent"], "1")

    def test_quartic_transfer_audit(self) -> None:
        result = QUARTIC_MODULE.audit(samples=2, heat_scale=0.5)
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["negative_sample_count"], 0)
        self.assertLess(result["maximum_heat_identity_residual"], 1.0e-9)

    def test_quartic_transfer_counterexample(self) -> None:
        result = QUARTIC_COUNTEREXAMPLE_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(
            result["pair_matrix_determinant"],
            "x**6 + 4*x**5 + 10*x**4 - 90*x**3 - 195*x**2 - 306*x - 224",
        )

    def test_quartic_transfer_helical_audit(self) -> None:
        result = QUARTIC_HELICAL_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertAlmostEqual(
            float(result["homochiral_threshold_scale"]),
            0.2730254800826943,
        )

    def test_quartic_transfer_helical_matrix_audit(self) -> None:
        result = QUARTIC_HELICAL_MATRIX_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(
            result["two_by_two_block_determinant"],
            "-(x**3 + 2*x**2 + 3*x + 3)**2/128",
        )

    def test_ns_trajectory_defect_audit(self) -> None:
        result = TRAJECTORY_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["linear_integrated_palinstrophy"], "8/viscosity")

    def test_galerkin_trajectory_audit(self) -> None:
        result = GALERKIN_MODULE.audit()
        self.assertTrue(result["solver_success"])
        self.assertTrue(result["constraint_checks_pass"])
        self.assertLess(result["maximum_identity_residual"], 1.0e-12)

    def test_galerkin_sweep_analysis(self) -> None:
        result_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "galerkin_trajectory_sweep.jsonl"
        )
        result = GALERKIN_ANALYSIS_MODULE.analyze(result_path)
        self.assertTrue(result["validation_checks_pass"])
        self.assertTrue(result["duplicate_rows_are_identical"])
        self.assertEqual(result["resolved_positive_reynolds"], [0.25, 0.5, 1.0, 2.0])
        self.assertEqual(
            result["negative_channel_transition_bracket"]["lower_reynolds"],
            0.922,
        )

    def test_helical_trajectory_channel_audit(self) -> None:
        result = HELICAL_TRAJECTORY_MODULE.audit()
        self.assertTrue(result["channel_decomposition_checks_pass"])
        self.assertLess(result["maximum_seed_matrix_residual"], 1.0e-10)
        self.assertLess(result["maximum_rank_one_residual"], 1.0e-10)

    def test_generated_mode_transfer_audit(self) -> None:
        result = GENERATED_TRANSFER_MODULE.audit()
        self.assertTrue(result["decomposition_checks_pass"])
        self.assertLess(result["maximum_decomposition_residual"], 1.0e-10)
        self.assertLess(abs(result["integral_sum_residual"]), 1.0e-10)

    def test_weak_generated_transfer_audit(self) -> None:
        result = WEAK_TRANSFER_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertAlmostEqual(
            result["scale_half_formal_sixth_order_crossing"],
            0.8616625288,
            places=9,
        )

    def test_second_normal_form_audit(self) -> None:
        result = SECOND_NORMAL_FORM_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["two_mode_resonant_quintet_count"], 0)
        self.assertEqual(result["two_mode_first_remainder_order"], 6)
        self.assertEqual(
            result["sparse_triad_exact_quintic"],
            "-(x - 1)**2*(9*x**3 + 18*x**2 + 27*x + 905)/600",
        )

    def test_third_normal_form_audit(self) -> None:
        result = THIRD_NORMAL_FORM_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["sextic_tree_count"], 60)
        self.assertEqual(
            result["sparse_triad_exact_quintic_primitive"],
            "(x - 1)**2*(31*x**3 + 62*x**2 + 93*x - 380)/1800",
        )
        self.assertEqual(
            result["negative_two_mode_exact_buckets"]["1"],
            "1772221/1404000",
        )

    def test_normal_form_resummation_audit(self) -> None:
        result = RESUMMATION_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["actual_tree_counts"], [1, 3, 12, 60, 360])
        self.assertEqual(result["two_mode_endpoints"]["0"], "0")
        self.assertEqual(result["two_mode_endpoints"]["2"], "0")
        self.assertEqual(
            result["galerkin_l1_smallness_parameter"],
            "q=K_max*||u||_1/nu",
        )

    def test_collision_coherence_weight_audit(self) -> None:
        result = COLLISION_COHERENCE_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(
            result["newtonian_q_one_boundary_damping"],
            "6*epsilon*nu/(epsilon + g**2)**2",
        )
        self.assertEqual(result["optimizer_switches_at_chi_three"], True)
        self.assertEqual(result["leray_strain_parabolic_scaling_index"], "5/2")

    def test_localized_strain_tube_audit(self) -> None:
        result = LOCALIZED_TUBE_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        rows = result["spectral_rows"]
        reynolds_two = next(row for row in rows if row["tube_reynolds"] == 2.0)
        self.assertAlmostEqual(reynolds_two["lambda_over_a"], 4.0, places=12)
        self.assertTrue(
            all(row["pair_decay_margin_over_a"] > 0.0 for row in rows)
        )
        self.assertEqual(
            result["effective_error_potential"],
            "-(a*error_x*x + a*error_y*y + divergence_error*nu "
            "- 2*nu*stretching_error)/(2*nu)",
        )
        self.assertLess(result["maximum_kummer_boundary_residual"], 1.0e-12)

    def test_moving_strain_tube_audit(self) -> None:
        result = MOVING_TUBE_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["pure_rotation_energy_error"], "0")
        reynolds_two = next(
            row
            for row in result["spectral_budget_rows"]
            if row["tube_reynolds"] == 2.0
        )
        self.assertAlmostEqual(
            reynolds_two["l2_budget_over_a_on_unit_disk"],
            1.0 / (3.0**0.5),
            places=12,
        )

    def test_strain_tube_reentry_audit(self) -> None:
        result = REENTRY_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["pure_brownian_3d_exponent"], 1)
        self.assertEqual(result["pure_brownian_2d_exponent"], 0)
        self.assertAlmostEqual(result["example_cycle_factor"], 8.0 / 15.0)
        self.assertAlmostEqual(result["example_renewal_bound"], 18.0 / 7.0)

    def test_three_dimensional_leray_gate_audit(self) -> None:
        result = THREE_DIMENSIONAL_GATE_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["full_plane_oscillator_ground_energy"], "2*a")
        self.assertEqual(
            result["sharp_sobolev_constant"],
            "2*2**(1/3)/(3*pi**(4/3))",
        )
        self.assertEqual(result["leray_potential_parabolic_index"], "2/2+3/2=5/2>2")
        self.assertEqual(result["single_return_deformation"], "eta")
        self.assertEqual(result["pair_return_deformation"], "eta**2")

    def test_strain_eigenframe_geometry_audit(self) -> None:
        result = EIGENFRAME_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(
            result["maximal_stretching_deficit"],
            "-lambda_1*xi_1**2 - lambda_2*xi_2**2 "
            "+ lambda_3*xi_1**2 + lambda_3*xi_2**2",
        )
        self.assertTrue(
            all(
                abs(row["full_space_excess"] - row["expected_excess"])
                < 1.0e-12
                for row in result["affine_spectral_rows"]
            )
        )
        self.assertEqual(
            result["tilt_harmonic_pressure_hessian"],
            "Matrix([[0, 0, gamma], [0, 0, 0], [gamma, 0, 0]])",
        )

    def test_pressure_collision_kernel_audit(self) -> None:
        result = PRESSURE_COLLISION_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(
            result["tracefree_riesz_frobenius_norm_squared"], "2/3"
        )
        self.assertEqual(result["pressure_mode_first_heat_derivative"], "-2/3")
        self.assertEqual(
            result["normalized_strain_local_reaction"],
            "(2*strain_ratio**2 + 2*strain_ratio - 1)/3",
        )
        self.assertAlmostEqual(
            result["low_pressure_projected_l1_to_linf_constant"],
            0.0032522411219488606,
            places=12,
        )

    def test_pressure_frame_pairing_audit(self) -> None:
        result = PRESSURE_PAIRING_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["random_seed"], 81)
        self.assertLess(result["maximum_eigenvalue_gradient_norm"], 1.0e-10)
        self.assertTrue(
            all(
                value < -1.0
                for value in result[
                    "maximum_eigenvalue_hessian_eigenvalues"
                ]
            )
        )
        self.assertGreater(result["reaction_without_scalar_diffusion"], 100.0)
        self.assertGreater(result["instantaneous_material_growth"], 10.0)
        self.assertLess(result["material_growth_identity_residual"], 1.0e-9)
        self.assertLess(result["pressure_split_residual"], 1.0e-10)

    def test_pressure_shell_commutator_audit(self) -> None:
        result = PRESSURE_SHELL_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertTrue(
            all(
                abs(row["identity_residual"]) < 1.0e-10
                for row in result["pressure_shell_split_rows"]
            )
        )
        self.assertEqual(
            result["shell_dimensionless_ratio"],
            "core_length**2*strain_scale/viscosity",
        )
        amplitude_rows = result["amplitude_scaling_rows"]
        self.assertLess(amplitude_rows[0]["pressure_to_dissipation_ratio"], 1.0)
        self.assertGreater(
            amplitude_rows[-1]["pressure_to_dissipation_ratio"], 1.0
        )
        spike_rows = result["leray_time_spike_rows"]
        self.assertTrue(
            all(row["time_l2_integral"] == 1.0 for row in spike_rows)
        )
        self.assertGreater(
            spike_rows[-1]["time_10_over_3_integral"],
            spike_rows[-2]["time_10_over_3_integral"],
        )

    def test_adaptive_reynolds_envelope_audit(self) -> None:
        result = REYNOLDS_ENVELOPE_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(
            result["envelope_effective_error"],
            result["envelope_effective_error_factorization"],
        )
        self.assertEqual(
            result["actual_local_reynolds"], "R_star*a/envelope"
        )
        self.assertAlmostEqual(
            result["R_two_single_principal_rate_over_A"], 4.0, places=12
        )
        self.assertAlmostEqual(
            result["R_two_single_decay_margin_over_A"], 2.0, places=12
        )
        self.assertAlmostEqual(
            result["R_two_pair_decay_margin_over_A"], 4.0, places=12
        )
        self.assertTrue(
            all(
                row["actual_local_reynolds"] <= 2.0
                for row in result["history_rows"]
            )
        )

    def test_shrinking_tube_renewal_audit(self) -> None:
        result = SHRINKING_RENEWAL_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(
            result["radial_laplacian_ratio"],
            "beta*(beta - dimension + 2)/radius**2",
        )
        self.assertEqual(
            result["brownian_beta_one_generator"],
            "(2*deformation*envelope - envelope_rate)/(2*envelope)",
        )
        self.assertEqual(result["capacity_budget_optimizer"], "1/2")
        self.assertEqual(result["maximum_static_capacity_budget"], "1/4")
        self.assertLess(result["example_cycle_factor"], 1.0)
        self.assertGreater(result["example_renewal_bound"], 0.0)

    def test_pressure_partition_flux_audit(self) -> None:
        result = PRESSURE_PARTITION_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertLess(result["maximum_partition_sum_error"], 1.0e-14)
        self.assertLess(result["maximum_partition_gradient_sum"], 1.0e-14)
        self.assertEqual(len(result["split_results"]), 3)
        self.assertTrue(
            all(
                len(row["cell_works"]) == 8
                and len(row["edge_rows"]) == 12
                and row["maximum_cell_identity_residual"] < 1.0e-10
                and abs(row["partition_total_work"]) < 1.0e-10
                and row["weighted_representation_residual"] < 1.0e-10
                and row["edge_representation_residual"] < 1.0e-10
                for row in result["split_results"]
            )
        )

    def test_intrinsic_radius_cover_audit(self) -> None:
        result = INTRINSIC_COVER_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertLessEqual(
            result["maximum_adjacent_lipschitz_slope"],
            result["kappa"] + 1.0e-12,
        )
        self.assertLessEqual(result["maximum_actual_local_reynolds"], 2.0)
        self.assertAlmostEqual(
            result["theoretical_neighbor_radius_ratio"], 5.0 / 3.0
        )
        self.assertAlmostEqual(
            result["theoretical_neighbor_reference_ratio"], 25.0 / 9.0
        )
        self.assertLessEqual(
            result["maximum_observed_overlapping_radius_ratio"],
            result["theoretical_neighbor_radius_ratio"] + 1.0e-12,
        )

    def test_monotone_dyadic_cover_audit(self) -> None:
        result = DYADIC_COVER_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["parent_to_child_side_ratio"], 2)
        self.assertEqual(result["parent_to_child_reference_envelope_ratio"], 4)
        self.assertTrue(
            all(
                abs(row["coverage_length"] - 4.0) < 1.0e-12
                and row["maximum_side_to_safe_radius"] <= 1.0 + 1.0e-12
                and row["maximum_cell_local_reynolds"] <= 2.0 + 1.0e-12
                and row["maximum_neighbor_side_ratio"] <= 2.0 + 1.0e-12
                for row in result["snapshot_rows"]
            )
        )

    def test_dyadic_gauge_transition_audit(self) -> None:
        result = DYADIC_TRANSITION_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(
            result["maximum_coordinate_exponent_difference"], "1/12"
        )
        self.assertEqual(
            result["maximum_parent_child_log_gauge_cost"],
            "R_star*dimension/48",
        )
        self.assertLess(result["audited_one_history_net_factor"], 0.57)
        self.assertLess(result["audited_pair_net_factor"], 0.33)
        self.assertGreater(
            result["generation_rows"][-1]["unpaid_transition_product"], 1.0
        )
        self.assertLess(
            result["generation_rows"][-1][
                "shrink_paid_one_history_product"
            ],
            1.0e-10,
        )

    def test_branching_transfer_operator_audit(self) -> None:
        result = BRANCHING_TRANSFER_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertAlmostEqual(result["single_branch_probability_sum"], 1.0)
        self.assertAlmostEqual(result["pair_branch_probability_sum"], 1.0)
        self.assertEqual(result["replica_pair_branch_count"], 64)
        self.assertAlmostEqual(
            result["single_interface_physical_l1_norm"], 1.0, places=12
        )
        self.assertAlmostEqual(
            result["pair_interface_physical_l1_norm"], 1.0, places=12
        )
        self.assertAlmostEqual(
            result["contracted_pair_branch_l1_norm"],
            result["pair_true_split_factor"],
            places=14,
        )
        self.assertGreater(result["weighted_pair_interface_l1_norm"], 1.0)
        self.assertGreater(result["pair_true_split_log_mismatch_budget"], 1.0)

    def test_interface_weight_no_go_audit(self) -> None:
        result = INTERFACE_WEIGHT_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(result["left_kernel_dimension"], 1)
        self.assertLess(
            result["normalized_left_kernel_maximum_constant_error"], 1.0e-12
        )
        self.assertGreater(
            result["weighted_single_logarithmic_l1_growth"], 0.0
        )
        self.assertAlmostEqual(
            result["weighted_pair_logarithmic_l1_growth"],
            2.0 * result["weighted_single_logarithmic_l1_growth"],
            places=12,
        )
        self.assertLess(result["R_two_physical_pair_sufficient_factor"], 1.0)
        self.assertGreater(result["R_two_remaining_logarithmic_budget"], 0.0)

    def test_two_norm_generation_cycle_audit(self) -> None:
        result = TWO_NORM_CYCLE_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        representative = result["R_one_beta_one_eta_two"]
        edge = result["R_two_beta_one_eta_two"]
        self.assertLess(representative["generation_factor"], 1.0)
        self.assertGreater(edge["generation_factor"], 1.0)
        self.assertGreater(
            edge["full_generation_required_visit_action"], 0.4
        )
        self.assertGreater(
            result["zero_action_R_threshold_for_beta_one_eta_two"], 1.0
        )
        self.assertLess(
            result["zero_action_R_threshold_for_beta_one_eta_two"], 1.5
        )

    def test_buffered_visit_feynman_kac_audit(self) -> None:
        result = BUFFERED_VISIT_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertEqual(
            result["one_history_inner_boundary_visit_gain"],
            "-1/(R_star*log(eta) - 1)",
        )
        self.assertTrue(
            result["R_quarter_beta_one_eta_two"][
                "complete_generation_closes"
            ]
        )
        self.assertFalse(
            result["R_half_beta_one_eta_two"][
                "complete_generation_closes"
            ]
        )
        threshold = result[
            "eta_two_beta_one_complete_generation_R_threshold"
        ]
        self.assertGreater(threshold, 0.35)
        self.assertLess(threshold, 0.45)

    def test_axial_killing_buffered_visit_audit(self) -> None:
        result = AXIAL_KILLING_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertTrue(
            all(
                abs(row["threshold_generation_criterion"] - 1.0) < 1.0e-9
                for row in result["threshold_rows"]
                if row["required_dimensionless_axial_killing"] > 0.0
            )
        )
        self.assertGreater(result["R_one_required_axial_killing"], 0.0)
        self.assertLess(result["R_one_maximum_half_height_over_L"], 2.0)
        self.assertTrue(
            all(
                row["axial_OU_killing"] < row["brownian_axial_killing"]
                for row in result["OU_vs_brownian_rows"]
            )
        )

    def test_finite_cylinder_mode_audit(self) -> None:
        result = FINITE_CYLINDER_MODULE.audit()
        boolean_checks = {
            name: value for name, value in result.items() if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertLess(
            abs(
                result["convergence_rows"][-1]["centre_visit_gain"]
                - result["convergence_rows"][-2]["centre_visit_gain"]
            ),
            1.0e-5,
        )
        finite_rows = [
            row
            for row in result["threshold_rows"]
            if row["R_star"] in (0.5, 1.0, 2.0)
        ]
        self.assertTrue(
            all(
                row["full_mode_half_height"]
                < row["principal_mode_half_height"]
                and row["maximum_occurs_at_centre"]
                and row["principal_eigenvalue_residual"] < 1.0e-5
                for row in finite_rows
            )
        )

    def test_finite_cylinder_perturbation_margin_audit(self) -> None:
        result = FINITE_CYLINDER_PERTURBATION_MODULE.audit()
        boolean_checks = {
            name: value
            for name, value in result.items()
            if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertLess(
            result["zero_potential_legacy_gain_residual"], 1.0e-11
        )
        self.assertLess(
            result["maximum_threshold_refinement_change"], 2.0e-4
        )
        rows = result["geometry_rows"]
        representative = next(
            row
            for row in rows
            if row["R_star"] == 0.5
            and row["half_height_over_L"] == 1.5
        )
        near_threshold = next(
            row
            for row in rows
            if row["R_star"] == 0.5
            and row["half_height_over_L"] == 2.0
        )
        self.assertLess(
            representative["critical_dimensionless_potentials"]["core"],
            representative["critical_dimensionless_potentials"]["shell"],
        )
        self.assertGreater(
            representative["critical_dimensionless_potentials"]["core"],
            0.8,
        )
        self.assertLess(
            near_threshold["critical_dimensionless_potentials"]["core"],
            0.2,
        )

    def test_finite_cylinder_kato_gate_audit(self) -> None:
        result = FINITE_CYLINDER_KATO_MODULE.audit()
        boolean_checks = {
            name: value
            for name, value in result.items()
            if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        working = result["geometry_rows"][0]
        near_threshold = result["geometry_rows"][2]
        self.assertGreater(
            working["critical_Kato_operator_norm"], 0.29
        )
        self.assertLess(
            working["critical_Kato_operator_norm"], 0.30
        )
        self.assertLess(
            near_threshold["critical_Kato_operator_norm"], 0.08
        )
        endpoint_rows = result[
            "endpoint_rows_normalized_to_unit_L3_over_2_mass"
        ]
        self.assertTrue(
            all(
                abs(row["L3_over_2_mass"] - 1.0) < 1.0e-12
                for row in endpoint_rows
            )
        )
        self.assertGreater(
            endpoint_rows[-1]["Newtonian_potential_at_centre"], 10.0
        )

    def test_gaussian_boundary_l2_transfer_audit(self) -> None:
        result = GAUSSIAN_BOUNDARY_L2_MODULE.audit()
        boolean_checks = {
            name: value
            for name, value in result.items()
            if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertLess(
            abs(result["dynamic_measure_single_Markov_L2_norm"] - 1.0),
            1.0e-12,
        )
        self.assertLess(
            abs(result["dynamic_measure_pair_Markov_L2_norm"] - 1.0),
            1.0e-12,
        )
        working = result["visit_rows"][0]
        self.assertLess(
            working["Gaussian_L2_complete_generation_criterion"], 0.4
        )
        self.assertGreater(
            working["maximum_one_history_entry_exit_measure_mismatch"],
            1.6,
        )
        self.assertLess(
            working["second_to_principal_multiplier_ratio"], 0.3
        )

    def test_ground_state_visit_transform_audit(self) -> None:
        result = GROUND_STATE_VISIT_MODULE.audit()
        boolean_checks = {
            name: value
            for name, value in result.items()
            if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        working = result["geometry_rows"][0]
        self.assertLess(
            working["maximum_row_sum_error"], 1.0e-12
        )
        self.assertLess(
            working["maximum_detailed_balance_error"], 1.0e-12
        )
        self.assertLess(
            working["mean_zero_L2_contraction_factor"], 0.3
        )
        refined = result["working_geometry_convergence_rows"][-1]
        self.assertGreater(
            refined[
                "minimum_kernel_density_relative_to_ground_state_measure"
            ],
            0.4136,
        )
        self.assertLess(
            refined[
                "maximum_kernel_density_relative_to_ground_state_measure"
            ],
            3.783,
        )

    def test_axial_form_to_boundary_audit(self) -> None:
        result = AXIAL_FORM_BOUNDARY_MODULE.audit()
        boolean_checks = {
            name: value
            for name, value in result.items()
            if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        working = result["geometry_rows"][0]
        self.assertGreater(
            working["critical_relative_form_bound"], 0.86
        )
        self.assertLess(
            working["critical_relative_form_bound"], 0.88
        )
        profile_rows = result["working_geometry_profile_rows"]
        constant = profile_rows[0]
        self.assertLess(
            abs(
                constant["critical_relative_form_bound"]
                - constant["universal_sufficient_relative_form_bound"]
            ),
            1.0e-7,
        )
        self.assertTrue(
            all(
                row["half_budget_actual_generation_criterion"]
                <= row["half_budget_form_generation_bound"] + 1.0e-9
                for row in profile_rows
            )
        )

    def test_off_diagonal_form_transfer_audit(self) -> None:
        result = OFF_DIAGONAL_FORM_MODULE.audit()
        boolean_checks = {
            name: value
            for name, value in result.items()
            if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        counterexample = result["counterexample"]
        self.assertGreater(counterexample["relative_amplification"], 100.0)
        self.assertGreater(
            counterexample["perturbed_cross_transfer"],
            counterexample["naive_relative_upper_bound"],
        )
        self.assertGreater(
            result["working_geometry_chi_two_alpha_budget"], 0.23
        )
        self.assertLess(
            result["working_geometry_chi_two_alpha_budget"], 0.25
        )

    def test_weighted_cylinder_buffer_condition_audit(self) -> None:
        result = WEIGHTED_CYLINDER_BUFFER_MODULE.audit()
        boolean_checks = {
            name: value
            for name, value in result.items()
            if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        half_radius = result["working_geometry_half_radius_surface"]
        quarter_collar = result[
            "working_geometry_quarter_collar_surface"
        ]
        self.assertGreater(
            half_radius["diagonal_to_cross_condition_number"], 1.62
        )
        self.assertLess(
            half_radius["diagonal_to_cross_condition_number"], 1.63
        )
        self.assertGreater(
            half_radius["allowable_relative_form_alpha"], 0.27
        )
        self.assertGreater(
            quarter_collar["diagonal_to_cross_condition_number"],
            half_radius["diagonal_to_cross_condition_number"],
        )
        angular_rows = result["working_geometry_angular_mode_rows"]
        self.assertLess(
            angular_rows[1]["cross_norm_at_this_angular_mode"],
            angular_rows[0]["cross_norm_at_this_angular_mode"],
        )

    def test_poisson_cutoff_form_transfer_audit(self) -> None:
        result = POISSON_CUTOFF_FORM_MODULE.audit()
        boolean_checks = {
            name: value
            for name, value in result.items()
            if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertLess(
            abs(
                result["finite_element_principal_visit_gain"]
                - result["exact_principal_visit_gain"]
            ),
            1.0e-5,
        )
        representative = next(
            row
            for row in result["profile_rows"]
            if row["perturbation_support_radius"] == 1.5
            and row["cutoff_taper_radius"] == 2.0
        )
        self.assertGreater(
            representative["allowable_relative_form_alpha"], 0.27
        )
        self.assertLess(
            representative["allowable_relative_form_alpha"], 0.28
        )
        self.assertGreater(
            representative[
                "conservative_L3_over_2_mass_budget_over_nu"
            ],
            1.26,
        )
        self.assertEqual(representative["maximizing_angular_mode"], 0)
        self.assertEqual(representative["maximizing_axial_mode"], 0)

    def test_quadratic_partition_ims_budget_audit(self) -> None:
        result = QUADRATIC_PARTITION_IMS_MODULE.audit()
        boolean_checks = {
            name: value
            for name, value in result.items()
            if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertLess(
            result["maximum_quadratic_partition_error"], 1.0e-14
        )
        self.assertFalse(
            result["unit_width_rows"][2]["IMS_cost_is_absorbable"]
        )
        self.assertTrue(
            result["sequential_unit_split_row"]["IMS_cost_is_absorbable"]
        )
        self.assertGreater(
            result["sequential_unit_split_row"]
            ["Poisson_compatible_L3_over_2_budget_over_nu"],
            1.1,
        )
        self.assertGreater(
            result["wide_tensor_octree_row"]
            ["Poisson_compatible_L3_over_2_budget_over_nu"],
            1.0,
        )

    def test_radial_cubic_partition_audit(self) -> None:
        result = RADIAL_CUBIC_PARTITION_MODULE.audit()
        boolean_checks = {
            name: value
            for name, value in result.items()
            if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertLess(
            result[
                "sampled_maximum_one_dimensional_IMS_at_unit_knot_spacing"
            ],
            0.785,
        )
        self.assertFalse(
            next(
                row
                for row in result["candidate_budget_rows"]
                if row["radial_support_radius_over_L"] == 1.5
            )["IMS_cost_is_absorbable"]
        )
        self.assertEqual(
            result["optimized_budget_row"]["radial_support_radius_over_L"],
            1.75,
        )
        self.assertGreater(
            result["optimized_budget_row"]
            ["Poisson_compatible_L3_over_2_budget_over_nu"],
            0.59,
        )
        self.assertEqual(
            result["optimized_full_tensor_budget_row"]
            ["radial_support_radius_over_L"],
            1.91,
        )
        self.assertGreater(
            result["optimized_full_tensor_budget_row"]
            ["full_Poisson_compatible_L3_over_2_budget_over_nu"],
            0.21,
        )

    def test_cubic_level_transfer_audit(self) -> None:
        result = CUBIC_LEVEL_TRANSFER_MODULE.audit()
        boolean_checks = {
            name: value
            for name, value in result.items()
            if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertLess(
            result["maximum_conditional_child_probability_sum_error"],
            2.0e-12,
        )
        self.assertAlmostEqual(
            result["replica_pair_probability_sum"], 1.0, places=14
        )
        self.assertLess(
            result["single_history_true_level_change_factor"], 0.65
        )
        self.assertLess(
            result["replica_pair_true_level_change_factor"], 0.42
        )

    def test_navier_stokes_coherence_budget_audit(self) -> None:
        result = NAVIER_STOKES_COHERENCE_BUDGET_MODULE.audit()
        boolean_checks = {
            name: value
            for name, value in result.items()
            if isinstance(value, bool)
        }
        self.assertTrue(all(boolean_checks.values()), boolean_checks)
        self.assertAlmostEqual(
            result["full_tensor_form_budget_over_nu"],
            0.21590032291866149,
            places=13,
        )
        self.assertLess(result["maximum_G_if_D_zero"], 0.05)
        self.assertGreater(result["maximum_H_if_F_zero"], 0.22)
        self.assertLess(
            result["fixed_z_translation_reaches_budget_at_UL_over_nu"],
            0.11,
        )

    def test_general_affine_spectral_floor_audit(self) -> None:
        result = GENERAL_AFFINE_SPECTRAL_FLOOR_MODULE.audit()
        self.assertTrue(result["all_positive_certificate_checks_pass"])
        self.assertFalse(result["general_affine_Poisson_transfer_certified"])
        self.assertAlmostEqual(
            result["uniform_general_affine_spectral_margin"],
            5.283185962946783,
            places=13,
        )
        self.assertGreater(
            result["dimensionless_uniform_margin_after_full_tensor_IMS"],
            0.44,
        )
        self.assertGreater(
            result["unit_relative_form_L3_over_2_budget_after_IMS"],
            1.68,
        )

    def test_anisotropic_poisson_transfer_pilot(self) -> None:
        result = ANISOTROPIC_POISSON_TRANSFER_PILOT_MODULE.audit()
        self.assertTrue(result["all_positive_pilot_checks_pass"])
        self.assertFalse(
            result["rigorous_general_affine_Poisson_transfer_certified"]
        )
        rows = result["spectrum_stress_rows"]
        self.assertEqual(rows[0]["t_parameter"], -0.5)
        self.assertEqual(rows[-1]["t_parameter"], 1.0)
        self.assertGreater(
            rows[-1]["allowable_relative_form_alpha"],
            rows[0]["allowable_relative_form_alpha"],
        )
        self.assertLess(
            rows[-1]["visit_operator_norm"],
            rows[0]["visit_operator_norm"],
        )
        self.assertTrue(
            result[
                "forced_full_affine_working_height_fails_for_some_spectra"
            ]
        )
        self.assertTrue(
            result[
                "optimized_compact_full_affine_all_sampled_spectra_close"
            ]
        )
        optimized = result["optimized_compact_full_affine_geometry_row"]
        self.assertAlmostEqual(
            optimized["half_height_over_L"], 0.85, places=12
        )
        self.assertGreater(
            optimized["diagnostic_final_mass_budget_over_nu"], 0.14
        )

    def test_reversible_shell_rigidity_audit(self) -> None:
        result = REVERSIBLE_SHELL_RIGIDITY_MODULE.audit()
        self.assertTrue(result["all_rigidity_checks_pass"])
        self.assertEqual(
            result["forced_anisotropic_shell_amplitude"], "kappa*r**2"
        )
        self.assertEqual(
            result["dirichlet_taper_normal_derivative_jump"],
            "-64*kappa/15",
        )

    def test_divergence_free_shell_taper_audit(self) -> None:
        result = DIVERGENCE_FREE_SHELL_TAPER_MODULE.audit()
        self.assertTrue(result["all_positive_taper_checks_pass"])
        self.assertFalse(
            result["rigorous_polynomial_taper_enclosure_certified"]
        )
        selected = result["selected_taper_row"]
        self.assertLess(
            selected["dense_validated_strain_amplification"], 1.00001
        )
        narrow = next(
            row
            for row in result["taper_radius_rows"]
            if row["taper_radius"] == 2.0
        )
        self.assertGreater(narrow["worst_t1_stretching_excess"], 1.3)

    def test_divergence_free_taper_transfer_pilot(self) -> None:
        result = DIVERGENCE_FREE_TAPER_TRANSFER_MODULE.audit()
        self.assertTrue(result["all_positive_transfer_checks_pass"])
        self.assertFalse(result["rigorous_nonsymmetric_transfer_certified"])
        rows = result["working_height_spectrum_rows"]
        self.assertLess(rows[-1]["complete_generation_criterion"], 0.26)
        self.assertGreater(
            rows[-1]["diagnostic_L3_over_2_mass_budget_over_nu"], 0.92
        )
        self.assertLess(
            rows[-1]["natural_uniform_round_trip_mismatch"],
            rows[-1][
                "maximum_one_history_measure_mismatch_for_natural_cycle"
            ],
        )
        self.assertAlmostEqual(
            rows[-1]["Doob_stationary_Markov_L2_norm"], 1.0, places=12
        )
        self.assertLess(
            result["axisymmetric_convergence_rows"][-1]
            ["axisymmetric_visit_gain_error"],
            3.0e-4,
        )

    def test_sectorial_poisson_transfer_audit(self) -> None:
        result = SECTORIAL_POISSON_TRANSFER_MODULE.audit()
        self.assertTrue(result["all_positive_sector_checks_pass"])
        self.assertLess(
            result["maximum_random_actual_to_bound_ratio"], 0.6
        )
        self.assertAlmostEqual(
            result["working_sharp_Sobolev_mass_budget_over_nu"],
            0.9292034203149158,
            places=12,
        )
        self.assertGreater(
            result["maximum_drift_L3_over_nu_if_alpha_zero"], 0.47
        )
        self.assertGreater(
            result["equal_share_potential_L3_over_2_mass_over_nu"],
            0.50,
        )

    def test_moving_cubic_label_transport_audit(self) -> None:
        result = MOVING_CUBIC_LABEL_TRANSPORT_MODULE.audit()
        self.assertTrue(result["all_positive_moving_label_checks_pass"])
        self.assertFalse(
            result["full_Navier_Stokes_localization_theorem_closed"]
        )
        partition = result["partition_stress_test"]
        self.assertLess(partition["maximum_partition_sum_error"], 1.0e-14)
        self.assertLess(
            partition["maximum_fisher_variance_identity_error"], 1.0e-12
        )
        self.assertAlmostEqual(
            partition["dynamic_measure_single_L2_norm"], 1.0, places=12
        )
        self.assertAlmostEqual(
            partition["dynamic_measure_pair_L2_norm"], 1.0, places=12
        )
        self.assertLess(
            result["parabolic_stress_test"]
            ["parabolic_intertwining_residual"],
            1.0e-12,
        )
        self.assertGreater(result["drift_only_E_budget"], 0.47)
        self.assertLess(
            result["maximum_unremoved_UL_over_nu_if_only_error"], 0.13
        )
        self.assertLess(
            result[
                "maximum_general_rotation_omegaL2_over_nu_from_bound"
            ],
            0.05,
        )

    def test_continuous_cubic_localization_no_go_audit(self) -> None:
        result = CONTINUOUS_CUBIC_LOCALIZATION_NO_GO_MODULE.audit()
        self.assertTrue(result["all_positive_cubic_no_go_checks_pass"])
        self.assertTrue(
            result["continuous_full_tensor_cubic_localization_ruled_out"]
        )
        self.assertGreater(
            result["full_tensor_Fisher_lower_bound_L2"], 3.67
        )
        self.assertLess(
            result["disk_ground_trial_form_upper_bound"], 1.53
        )
        self.assertLess(
            result["disk_ground_trial_after_full_Fisher_upper_bound"],
            -2.14,
        )
        self.assertLess(
            result[
                "maximum_Fisher_fraction_not_ruled_out_by_ground_trial"
            ],
            0.42,
        )

    def test_stopping_time_moving_visit_audit(self) -> None:
        result = STOPPING_TIME_MOVING_VISIT_MODULE.audit()
        self.assertTrue(result["all_positive_stopping_visit_checks_pass"])
        self.assertFalse(result["full_stopping_time_renewal_theorem_closed"])
        self.assertEqual(
            result["continuous_cubic_Fisher_cost_inside_visit"], 0.0
        )
        self.assertAlmostEqual(
            result["relabel_stress_test"]
            ["single_history_dynamic_L2_norm"],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            result["relabel_stress_test"]
            ["replica_pair_dynamic_L2_norm"],
            1.0,
            places=12,
        )
        self.assertLess(
            result["moving_coordinate_stress_test"]
            ["dimensionless_drift_identity_residual"],
            1.0e-13,
        )
        self.assertLess(
            result["critical_scaling_stress_test"]
            ["potential_scaling_residual"],
            1.0e-12,
        )
        self.assertAlmostEqual(
            result["equal_share_closure_value"], 1.0, places=12
        )

    def test_coherence_abort_renewal_audit(self) -> None:
        result = COHERENCE_ABORT_RENEWAL_MODULE.audit()
        self.assertTrue(result["all_positive_abort_renewal_checks_pass"])
        self.assertFalse(
            result["full_coherence_abort_bound_from_Navier_Stokes_closed"]
        )
        self.assertTrue(result["free_conservative_restarts_do_not_close"])
        self.assertTrue(
            result["split_paid_restart_closes_without_sector_error"]
        )
        self.assertGreater(
            result["maximum_unperturbed_restart_norm"], 0.49
        )
        self.assertGreater(
            result["maximum_extra_split_restart_mismatch"], 1.20
        )
        self.assertLess(
            result["split_paid_equal_alpha_beta"], 0.018
        )
        self.assertLess(
            result[
                "maximum_unit_norm_abort_probability_without_sector_error"
            ],
            0.25,
        )

    def test_leray_mollified_cell_frame_audit(self) -> None:
        result = LERAY_MOLLIFIED_CELL_FRAME_MODULE.audit()
        self.assertTrue(result["all_positive_mollified_frame_checks_pass"])
        self.assertFalse(result["full_Leray_stopping_visit_theorem_closed"])
        self.assertFalse(
            result["pointwise_velocity_or_eigenframe_derivative_required"]
        )
        self.assertLess(
            result["distributional_gradient_identity_residual"], 1.0e-12
        )
        self.assertLess(
            result["Galilean_remainder_invariance_residual"], 1.0e-12
        )
        self.assertLess(result["frame_orthogonality_residual"], 1.0e-13)
        self.assertLess(
            result["weighted_remainder_mean_skew_gradient_norm"],
            1.0e-13,
        )
        self.assertGreaterEqual(
            result["normalized_affine_spectrum_parameter_t"], -0.5
        )
        self.assertLessEqual(
            result["normalized_affine_spectrum_parameter_t"], 1.0
        )

    def test_affine_taper_residual_no_go_audit(self) -> None:
        result = AFFINE_TAPER_RESIDUAL_NO_GO_MODULE.audit()
        self.assertTrue(result["all_positive_affine_taper_no_go_checks_pass"])
        self.assertTrue(
            result["tapered_reference_fails_exact_affine_consistency"]
        )
        self.assertFalse(
            result["rigorous_compact_full_affine_transfer_certified"]
        )
        self.assertGreater(
            result["worst_t1_collar_L3_lower_bound_over_nu"], 6.4
        )
        self.assertGreater(
            result["worst_t1_dense_full_taper_mismatch_L3_over_nu"],
            result["worst_t1_collar_L3_lower_bound_over_nu"],
        )
        self.assertEqual(
            result["compact_full_affine_exact_affine_remainder"], 0.0
        )
        self.assertLess(
            result["maximum_t_parameter_allowed_by_collar_budget"], -0.38
        )

    def test_compact_affine_campanato_gate_audit(self) -> None:
        result = COMPACT_AFFINE_CAMPANATO_GATE_MODULE.audit()
        self.assertTrue(result["all_positive_compact_campanato_checks_pass"])
        self.assertFalse(result["full_bad_occupation_bound_closed"])
        self.assertLess(
            result["decomposition_stress_test"]
            ["maximum_remainder_decomposition_error"],
            1.0e-13,
        )
        self.assertTrue(
            result["mollifier_constants"]["normalization_verified"]
        )
        self.assertGreater(
            result["compact_cylinder_geometry"]
            ["linear_matrix_L3_coefficient"],
            4.0,
        )
        self.assertLess(
            result["no_restart_combined_T_equals_G_threshold"], 0.041
        )
        rotation = result["constant_spectrum_rotation_test"]
        self.assertTrue(
            rotation["constant_spectrum_can_fail_without_envelope_growth"]
        )
        self.assertLess(rotation["maximum_eigenvalue_change"], 1.0e-13)

    def test_leray_conditional_occupation_no_go_audit(self) -> None:
        result = LERAY_CONDITIONAL_OCCUPATION_NO_GO_MODULE.audit()
        self.assertTrue(
            result["all_positive_Leray_occupation_no_go_checks_pass"]
        )
        self.assertFalse(
            result["energy_only_conditional_occupation_bound_closed"]
        )
        self.assertGreater(
            result["critical_spatial_L3_over_2_norm"],
            result["compact_no_restart_potential_only_budget"],
        )
        self.assertAlmostEqual(
            result["Leray_dissipation_proxy_coefficient_times_epsilon"],
            0.0294,
            places=14,
        )
        self.assertGreater(
            result["conditional_cube_survival_probability"], 0.34
        )
        self.assertTrue(
            result["survival_exceeds_both_probability_allowances"]
        )

    def test_nonautonomous_full_affine_form_audit(self) -> None:
        result = NONAUTONOMOUS_FULL_AFFINE_FORM_MODULE.audit()
        self.assertTrue(
            result["all_positive_nonautonomous_form_checks_pass"]
        )
        self.assertFalse(
            result["uniform_nonautonomous_boundary_visit_certified"]
        )
        self.assertGreater(
            result["uniform_nonautonomous_coercive_floor"], 4.83
        )
        self.assertEqual(result["temporal_matrix_error_T"], 0.0)
        self.assertEqual(result["temporal_eigenvalue_error_G"], 0.0)
        self.assertGreater(result["Campanato_only_C3_threshold"], 0.30)
        self.assertGreater(
            result["spatial_strain_only_P_threshold"], 0.63
        )
        self.assertTrue(
            result["constant_spectrum_rotation_test"]
            ["instantaneous_reference_removes_rotation_error"]
        )

    def test_rotating_affine_visit_pilot(self) -> None:
        result = ROTATING_AFFINE_VISIT_PILOT_MODULE.audit()
        self.assertTrue(
            result["all_positive_rotating_visit_pilot_checks_pass"]
        )
        self.assertFalse(result["nonautonomous_boundary_visit_certified"])
        self.assertFalse(
            result["sampled_rotation_uniform_norm_increase_detected"]
        )
        self.assertLess(result["static_formulation_calibration_error"], 0.01)
        self.assertFalse(
            result["tilting_constant_payoff_monte_carlo"]
            ["sampled_rotation_payoff_increase_detected"]
        )
        tilt_rows = [
            row
            for row in result["tilting_constant_payoff_monte_carlo"]["rows"]
            if row["rotation_axis"] == "tilting"
        ]
        self.assertLess(
            tilt_rows[-1]["constant_outer_payoff_mean"], 0.05
        )

    def test_weighted_kernel_dynamic_l2_audit(self) -> None:
        result = WEIGHTED_KERNEL_DYNAMIC_L2_MODULE.audit()
        self.assertTrue(result["all_positive_weighted_kernel_checks_pass"])
        self.assertFalse(result["dynamic_weighted_boundary_theorem_closed"])
        self.assertFalse(
            result["fixed_Gaussian_boundary_conversion_required"]
        )
        self.assertLess(
            result["random_kernel_stress_test"]
            ["maximum_operator_norm_error_from_exact_gain"],
            1.0e-12,
        )
        self.assertTrue(
            result["Markov_and_pair_test"]
            ["pair_norm_equals_one_history_gain_squared"]
        )
        self.assertGreater(
            result["maximum_dynamic_one_history_gain_for_closure"], 1.23
        )
        self.assertLess(
            result["generation_criterion_if_gain_is_at_most_one"], 0.659
        )

    def test_nonautonomous_scalar_gain_gate_audit(self) -> None:
        result = NONAUTONOMOUS_SCALAR_GAIN_GATE_MODULE.audit()
        self.assertTrue(result["all_positive_scalar_gain_gate_checks_pass"])
        self.assertTrue(result["uniform_volume_gain_closes_ideal_cycle"])
        self.assertFalse(result["stationary_uniform_surface_gain_closes_ideal_cycle"])
        self.assertFalse(result["full_nonautonomous_boundary_gain_closed"])
        self.assertFalse(result["sampled_static_worst_comparison_supported"])
        self.assertFalse(result["optimistic_covariance_plus_Nash_closes"])
        self.assertLess(result["normalized_uniform_volume_gain_bound"], 1.207)
        self.assertLess(
            result["stationary_sector_constants"]
            ["normalized_uniform_surface_gain_bound"],
            1.27,
        )
        self.assertGreater(
            result["half_time_xy_switch_pilot"]
            ["increase_over_static_combined_standard_errors"],
            3.7,
        )

    def test_radial_payoff_bellman_pilot(self) -> None:
        result = RADIAL_PAYOFF_BELLMAN_PILOT_MODULE.audit()
        self.assertTrue(
            result["all_positive_radial_Bellman_pilot_checks_pass"]
        )
        self.assertFalse(
            result["ideal_nonautonomous_boundary_theorem_certified"]
        )
        self.assertLess(result["finest_inner_interface_maximum"], 0.688)
        self.assertLess(result["last_three_grid_spread"], 0.001)
        self.assertGreater(result["sampled_Bellman_margin_to_closure"], 0.54)

    def test_radial_payoff_supersolution_candidate(self) -> None:
        result = RADIAL_PAYOFF_SUPERSOLUTION_MODULE.audit()
        self.assertTrue(
            result["all_positive_supersolution_candidate_checks_pass"]
        )
        self.assertFalse(result["interior_residual_interval_certified"])
        self.assertFalse(
            result["ideal_nonautonomous_boundary_theorem_certified"]
        )
        self.assertFalse(
            result["candidate_would_close_if_residual_is_certified"]
        )
        self.assertGreater(
            result["candidate_complete_generation_criterion"], 1.18
        )
        self.assertLess(
            result["candidate"]["dense_grid"]["maximum_HJB_residual"],
            -0.009,
        )
        self.assertGreater(
            result["candidate"]["dense_grid"]["minimum_squared_margin"],
            0.02,
        )

    def test_radial_payoff_interval_certificate(self) -> None:
        result = RADIAL_PAYOFF_INTERVAL_CERTIFICATE_MODULE.audit()
        self.assertTrue(
            result["all_positive_compact_interval_checks_pass"]
        )
        self.assertTrue(result["whole_open_half_cylinder_certified"])
        self.assertTrue(
            result["ideal_nonautonomous_boundary_theorem_certified"]
        )
        self.assertFalse(result["certified_dynamic_cycle_closes"])
        self.assertTrue(
            result["symbolic_derivative_cross_check"]
            ["all_manual_derivatives_exact"]
        )
        self.assertFalse(
            result["compact_interior"]["box_budget_exhausted"]
        )
        self.assertEqual(
            result["compact_interior"]["unresolved_box_count"], 0
        )
        self.assertGreater(
            result["certified_complete_generation_criterion"], 1.18
        )
        self.assertEqual(result["certified_uniform_Doob_killing_rate"], 0.005)
        self.assertGreater(
            result["asymptotic_boundary_strips"]
            ["radial_case_margin_interval"][0],
            251.0,
        )

    def test_radial_Bellman_Doob_perturbation_audit(self) -> None:
        result = RADIAL_BELLMAN_DOOB_PERTURBATION_MODULE.audit()
        self.assertTrue(
            result["all_positive_Doob_perturbation_checks_pass"]
        )
        self.assertTrue(
            result["symbolic_Doob_audit"]
            ["weighted_error_cancellation_exact"]
        )
        self.assertFalse(
            result["pointwise_condition_follows_from_critical_Lp_norms"]
        )
        self.assertFalse(
            result["nonautonomous_critical_form_to_boundary_theorem_closed"]
        )
        self.assertLess(result["additive_gain_allowance"], -0.10)
        self.assertLess(result["remaining_generation_margin"], -0.18)

    def test_critical_collar_transfer_audit(self) -> None:
        result = CRITICAL_COLLAR_TRANSFER_MODULE.audit()
        self.assertTrue(
            result["all_positive_critical_collar_checks_pass"]
        )
        self.assertFalse(
            result["positive_entry_collar_implies_global_Kato_bound"]
        )
        self.assertFalse(
            result["dynamic_collar_condition_number_certified"]
        )
        self.assertTrue(
            result["first_insertion_constants_decrease_with_collar"]
        )
        self.assertLess(result["global_Kato_Neumann_budget"], 0.036)
        self.assertTrue(result["calibration_is_legacy_bare_halving"])
        self.assertFalse(result["current_cubic_split_baseline_closes"])
        self.assertGreater(
            result["endpoint_counterexample"]["rows"][-1]
            ["Newtonian_centre_potential"],
            19.0,
        )

    def test_radial_barrier_cutoff_energy_pilot(self) -> None:
        result = RADIAL_BARRIER_CUTOFF_ENERGY_MODULE.audit()
        self.assertTrue(
            result["all_positive_cutoff_energy_pilot_checks_pass"]
        )
        self.assertTrue(
            result["all_medium_fine_changes_below_one_percent"]
        )
        self.assertGreater(
            result["fine_rows"][1][
                "sqrt_energy_over_m0_over_barrier_gain"
            ],
            2.18,
        )

    def test_radial_collar_trace_pilot(self) -> None:
        result = RADIAL_COLLAR_TRACE_MODULE.audit()
        self.assertTrue(
            result["all_positive_stationary_collar_pilot_checks_pass"]
        )
        self.assertFalse(result["all_discrete_optimizers_nonnegative"])
        self.assertFalse(result["all_sampled_stationary_chi_below_two"])
        self.assertTrue(
            result[
                "all_sampled_d0p2_or_wider_stationary_chi_below_two"
            ]
        )
        self.assertLess(
            result["worst_fine_row_by_distance"][1]
            ["stationary_axisymmetric_chi_pilot"],
            1.89,
        )

    def test_radial_collar_frequency_pilot(self) -> None:
        result = RADIAL_COLLAR_FREQUENCY_MODULE.audit()
        self.assertTrue(result["all_positive_frequency_pilot_checks_pass"])
        self.assertFalse(result["static_worst_frequency_supported"])
        self.assertTrue(result["nonzero_frequency_resonance_detected"])
        self.assertTrue(
            result["all_sampled_d0p2_or_wider_chi_below_two"]
        )
        self.assertLess(
            max(
                row["time_harmonic_chi_pilot"]
                for row in result["rows"]
            ),
            1.90,
        )

    def test_protected_collar_partition_no_go(self) -> None:
        result = PROTECTED_COLLAR_PARTITION_NO_GO_MODULE.audit()
        self.assertTrue(
            result["all_positive_protected_collar_no_go_checks_pass"]
        )
        self.assertFalse(
            result["continuous_transverse_cubic_localization_viable"]
        )
        self.assertFalse(
            result["continuous_full_tensor_cubic_localization_viable"]
        )
        self.assertGreater(
            result["rows"][2]["full_cost_to_form_floor_ratio"],
            6.2,
        )

    def test_interaction_marked_localization_audit(self) -> None:
        result = INTERACTION_MARKED_LOCALIZATION_MODULE.audit()
        self.assertTrue(
            result["all_positive_interaction_marking_checks_pass"]
        )
        self.assertTrue(
            result[
                "interaction_marking_reconstructs_physical_resolvent"
            ]
        )
        self.assertFalse(
            result["interaction_marking_creates_a_collar_by_itself"]
        )
        self.assertFalse(
            result[
                "independent_labelwise_energy_estimates_preserve_global_skew"
            ]
        )
        self.assertFalse(
            result[
                "decoupled_quadratic_label_blocks_reconstruct_generator"
            ]
        )
        self.assertTrue(
            result["two_coordinate_counterexample"]
            ["each_exact_label_piece_has_adverse_real_part"]
        )

    def test_radial_H1_payoff_supersolution_pilot(self) -> None:
        result = RADIAL_H1_PAYOFF_SUPERSOLUTION_MODULE.audit()
        self.assertTrue(
            result["all_positive_H1_supersolution_pilot_checks_pass"]
        )
        self.assertTrue(result["candidate_is_in_H1"])
        self.assertFalse(result["HJB_residual_interval_certified"])
        self.assertFalse(result["finite_energy_supersolution_certified"])
        self.assertLess(result["entry_gain"], 1.15)
        self.assertLess(
            result["candidate_complete_generation_criterion"], 0.87
        )
        self.assertLess(
            result["dense_grid"]["maximum_HJB_residual"], -0.012
        )
        self.assertGreater(result["additive_gain_allowance"], 0.08)

    def test_radial_H1_payoff_interval_certificate(self) -> None:
        result = RADIAL_H1_PAYOFF_INTERVAL_MODULE.audit()
        self.assertTrue(result["all_positive_H1_interval_checks_pass"])
        self.assertTrue(result["whole_open_half_cylinder_certified"])
        self.assertTrue(result["finite_energy_supersolution_certified"])
        self.assertTrue(result["candidate_is_in_H1"])
        self.assertEqual(
            result["certified_uniform_Doob_killing_rate"], 0.005
        )
        self.assertLess(result["certified_entry_gain"], 1.15)
        self.assertLess(
            result["certified_complete_generation_criterion"], 0.87
        )
        self.assertTrue(
            all(
                row["unresolved_box_count"] == 0
                for row in result["finite_rectangles"]
            )
        )
        self.assertTrue(
            all(
                row["unresolved_box_count"] == 0
                for row in result["transformed_open_strips"]
            )
        )

    def test_averaged_entry_trace_gate(self) -> None:
        result = AVERAGED_ENTRY_TRACE_GATE_MODULE.audit()
        self.assertTrue(
            result["all_positive_averaged_entry_checks_pass"]
        )
        self.assertTrue(result["return_law_remains_unnormalized"])
        self.assertTrue(
            result["early_nonreturns_keep_sub_Markov_contraction"]
        )
        self.assertFalse(
            result["geometric_protected_support_required_by_this_route"]
        )
        self.assertFalse(
            result["actual_exterior_return_density_envelope_certified"]
        )
        self.assertTrue(result["H1_barrier_interval_certified"])
        self.assertFalse(result["full_Navier_Stokes_entry_theorem_closed"])
        self.assertGreater(
            result["one_error_threshold_rows"][-1]
            ["drift_L3_threshold"],
            0.013,
        )

    def test_exterior_return_tail_gate(self) -> None:
        result = EXTERIOR_RETURN_TAIL_GATE_MODULE.audit()
        self.assertTrue(result["all_positive_exterior_tail_checks_pass"])
        self.assertFalse(result["exponential_envelope_required"])
        self.assertTrue(
            result["sphere_tail_disproves_every_positive_exponential_rate"]
        )
        self.assertLess(result["surface_L4_trace_form_constant"], 0.68)
        self.assertFalse(result["actual_weighted_exterior_envelope_certified"])
        self.assertFalse(result["full_Navier_Stokes_return_gate_closed"])

    def test_cylindrical_brownian_return_pilot(self) -> None:
        result = CYLINDRICAL_BROWNIAN_RETURN_MODULE.audit()
        self.assertTrue(
            result["all_positive_cylindrical_return_pilot_checks_pass"]
        )
        patch = result["finite_axial_patch_return"]
        self.assertAlmostEqual(patch["probability"], 0.310135151371, places=10)
        self.assertLess(patch["cutoff_halving_change"], 1.0e-10)
        self.assertFalse(result["Brownian_cylinder_L2_envelope_certified"])
        self.assertFalse(result["weighted_Navier_Stokes_cylinder_envelope_certified"])

    def test_branch_resolved_entry_renewal(self) -> None:
        result = BRANCH_RESOLVED_ENTRY_RENEWAL_MODULE.audit()
        self.assertTrue(result["all_positive_branch_resolved_checks_pass"])
        self.assertTrue(result["branch_separated_renewal_identity_closed"])
        self.assertTrue(
            result["unnormalized_branch_mass_is_applied_exactly_once"]
        )
        self.assertLess(
            result["legacy_calibration"]["coefficient_difference"],
            1.0e-10,
        )
        brownian = result["Brownian_finite_patch_calibration"]
        self.assertLess(brownian["branch_sum_closure_criterion"], 0.67)
        self.assertFalse(
            result["true_split_entry_density_envelope_certified"]
        )
        self.assertFalse(
            result["weighted_return_entry_density_envelope_certified"]
        )
        self.assertFalse(result["full_Navier_Stokes_generation_gate_closed"])

    def test_split_entry_density_inheritance(self) -> None:
        result = SPLIT_ENTRY_DENSITY_INHERITANCE_MODULE.audit()
        self.assertTrue(result["all_positive_split_density_checks_pass"])
        self.assertTrue(
            result["pointwise_split_preserves_existing_physical_density"]
        )
        self.assertFalse(
            result["pointwise_split_creates_spatial_or_temporal_smoothing"]
        )
        self.assertFalse(
            result[
                "deterministic_split_time_covered_by_averaged_surface_trace"
            ]
        )
        self.assertGreater(
            result["conditional_volume_density_threshold_rows"][0]
            ["drift_L3_threshold"],
            0.27,
        )
        self.assertFalse(result["full_true_split_entry_gate_closed"])

    def test_affine_exterior_axial_compensation(self) -> None:
        result = AFFINE_EXTERIOR_AXIAL_COMPENSATION_MODULE.audit()
        self.assertTrue(
            result["all_positive_affine_axial_compensation_checks_pass"]
        )
        self.assertTrue(
            result[
                "axial_L2_exactly_cancels_affine_deformation_at_long_time"
            ]
        )
        self.assertTrue(
            result[
                "finite_axial_patch_removes_old_affine_L2_tail_obstruction"
            ]
        )
        self.assertGreater(
            result["spectral_tail_rows"]["1.0"]
            ["principal_radial_decay_rate"],
            1.44,
        )
        self.assertFalse(result["all_affine_spectra_and_orientations_covered"])
        self.assertFalse(result["full_weighted_exterior_return_gate_closed"])

    def test_anisotropic_affine_exterior_tail_gate(self) -> None:
        result = ANISOTROPIC_AFFINE_EXTERIOR_TAIL_MODULE.audit()
        self.assertTrue(
            result["all_positive_anisotropic_tail_checks_pass"]
        )
        self.assertEqual(result["trace_free_identity"], "0")
        self.assertEqual(
            result["residual_exponential_rate_after_axial_L2"],
            "1/2 - rho/2",
        )
        self.assertEqual(result["neutral_endpoint_spectral_bottom"], 0.0)
        self.assertFalse(
            result["uniform_positive_transverse_rate_over_full_affine_family"]
        )
        self.assertFalse(
            result["fixed_outer_start_kernel_nonsummability_fully_proved"]
        )
        self.assertFalse(result["full_Navier_Stokes_exterior_gate_closed"])

    def test_neutral_strip_storage_gate(self) -> None:
        result = NEUTRAL_STRIP_STORAGE_MODULE.audit()
        self.assertTrue(result["all_positive_neutral_strip_checks_pass"])
        self.assertGreater(
            result["maximum_admissible_half_width"], 2.22
        )
        self.assertGreater(result["working_net_margin"], 0.059)
        margins = {
            round(row["net_weighted_tail_margin_lower_bound"], 12)
            for row in result["parameter_rows"]
        }
        self.assertEqual(len(margins), 1)
        self.assertFalse(
            result["outer_wall_exit_identified_with_physical_true_split"]
        )
        self.assertFalse(
            result["boundary_flux_space_time_density_envelope_certified"]
        )
        self.assertFalse(result["full_Navier_Stokes_storage_gate_closed"])

    def test_neutral_strip_branch_resolvent_pilot(self) -> None:
        result = NEUTRAL_STRIP_BRANCH_RESOLVENT_MODULE.audit()
        self.assertTrue(result["all_positive_branch_resolvent_checks_pass"])
        self.assertTrue(result["probability_partition_verified_on_grids"])
        self.assertTrue(result["all_resolvents_nonnegative_on_grids"])
        rho_zero = next(
            row for row in result["finest_summary"] if row["rho"] == 0.0
        )
        self.assertGreater(
            rho_zero["maximum_residual_scalar_stress_criterion"], 1.2
        )
        self.assertFalse(result["raw_all_z_scalar_stress_closes_uniformly"])
        self.assertFalse(result["boundary_flux_space_time_L2_gains_computed"])
        self.assertFalse(result["full_Navier_Stokes_branch_gate_closed"])

    def test_neutral_strip_axial_patch_branch_pilot(self) -> None:
        result = NEUTRAL_STRIP_AXIAL_PATCH_BRANCH_MODULE.audit()
        self.assertTrue(
            result["all_positive_axial_patch_branch_checks_pass"]
        )
        self.assertTrue(
            result["all_sampled_complete_scalar_branch_stresses_close"]
        )
        self.assertFalse(
            result["all_sampled_scalar_stresses_close_without_split_payment"]
        )
        self.assertLess(
            result["maximum_sampled_complete_scalar_criterion"], 0.7
        )
        self.assertGreater(
            result["maximum_sampled_criterion_without_true_split_payment"],
            1.2,
        )
        self.assertGreater(
            result["minimum_sampled_principal_killed_rate"], 1.99
        )
        self.assertFalse(
            result["physical_wall_exit_to_true_split_identification_proved"]
        )
        self.assertFalse(
            result["boundary_flux_space_time_L2_error_gain_certified"]
        )
        self.assertFalse(result["full_Navier_Stokes_generation_gate_closed"])

    def test_geometric_wall_split_compatibility_audit(self) -> None:
        result = GEOMETRIC_WALL_SPLIT_COMPATIBILITY_MODULE.audit()
        self.assertTrue(
            result["all_positive_wall_split_compatibility_checks_pass"]
        )
        self.assertEqual(result["exact_probability_hit_upper_wall_before_lower_slab"], "5/6")
        self.assertFalse(result["wall_event_implies_true_level_change"])
        self.assertFalse(
            result["wall_point_lies_in_an_audited_direct_child_visit"]
        )
        self.assertGreater(
            result["working_wall_child_capture_gap_value"], 0.42
        )
        self.assertTrue(
            result["same_scale_wall_branch_must_be_used_in_current_architecture"]
        )
        self.assertFalse(result["full_Navier_Stokes_wall_split_gate_closed"])

    def test_neutral_strip_same_scale_width_sweep(self) -> None:
        result = NEUTRAL_STRIP_SAME_SCALE_WIDTH_SWEEP_MODULE.audit()
        self.assertTrue(result["all_positive_same_scale_width_checks_pass"])
        self.assertFalse(result["same_scale_width_sweep_finds_closure"])
        optimum = result["sampled_optimum"]
        self.assertGreaterEqual(optimum["strip_half_width"], 2.2)
        self.assertLessEqual(optimum["strip_half_width"], 2.4)
        self.assertGreater(optimum["maximum_same_scale_criterion"], 1.15)
        self.assertLess(result["maximum_mesh_refinement_spread"], 0.001)
        self.assertLess(result["time_refinement_change"], 0.001)
        self.assertLess(result["maximum_x_truncation_spread"], 0.001)
        self.assertFalse(result["width_tuning_repairs_current_architecture"])
        self.assertFalse(result["full_Navier_Stokes_wall_gate_closed"])

    def test_geometry_triggered_migrating_child_pilot(self) -> None:
        result = GEOMETRY_TRIGGERED_MIGRATING_CHILD_MODULE.audit()
        self.assertTrue(result["all_positive_migrating_child_checks_pass"])
        self.assertFalse(
            result["support_contained_child_reaches_entry_surface"]
        )
        self.assertFalse(result["minimum_capture_support_fits_current_buffer"])
        self.assertGreater(
            result["current_cubic_direct_capture_gap_value"], 0.42
        )
        self.assertTrue(result["translated_fine_partition_is_physical_Markov"])
        self.assertTrue(
            result["fixed_branch_full_partition_jump_is_conservative"]
        )
        self.assertTrue(
            result["fixed_branch_signed_pressure_commutator_cancels"]
        )
        self.assertTrue(
            result["abstract_extended_state_stopping_transfer_is_Markov"]
        )
        self.assertAlmostEqual(
            result["child_entry_to_next_wall_normalized_gap"], 0.1
        )
        self.assertTrue(result["bounded_terminal_Zeno_remainder_vanishes"])
        self.assertFalse(
            result["inward_concentric_split_closes_sampled_scalar_gate"]
        )
        self.assertTrue(
            result["migrating_wall_endpoint_closes_sampled_scalar_gate"]
        )
        self.assertTrue(
            result[
                "migrating_wall_with_one_conversion_closes_sampled_scalar_gate"
            ]
        )
        self.assertTrue(
            result["smooth_tracking_center_scale_identity_verified"]
        )
        self.assertTrue(result["smooth_tracking_closes_sampled_scalar_gate"])
        self.assertTrue(
            result[
                "smooth_tracking_with_one_conversion_closes_sampled_scalar_gate"
            ]
        )
        working = result["working_width_row"]
        self.assertLess(
            working["maximum_migrating_wall_endpoint_criterion"], 0.80
        )
        self.assertLess(
            working["maximum_migrating_wall_with_conversion_criterion"],
            0.90,
        )
        self.assertGreater(
            working["minimum_allowable_one_history_wall_transfer_mismatch"],
            1.30,
        )
        self.assertLess(
            working["maximum_smooth_tracking_wall_criterion"], 0.65
        )
        self.assertLess(
            working[
                "maximum_smooth_tracking_wall_with_conversion_criterion"
            ],
            0.70,
        )
        self.assertLess(result["maximum_mesh_refinement_spread"], 0.001)
        self.assertLess(result["time_refinement_change"], 0.001)
        self.assertLess(result["x_truncation_change"], 0.003)
        self.assertFalse(
            result["path_triggered_partition_common_PDE_localization_certified"]
        )
        self.assertFalse(
            result[
                "physical_shrink_payment_for_path_triggered_migration_certified"
            ]
        )
        self.assertFalse(
            result["full_Navier_Stokes_geometry_transition_gate_closed"]
        )

    def test_migrating_core_residual_budget_audit(self) -> None:
        result = MIGRATING_CORE_RESIDUAL_BUDGET_MODULE.audit()
        self.assertTrue(
            result["all_positive_migrating_residual_checks_pass"]
        )
        self.assertTrue(result["complete_residual_decomposition_verified"])
        self.assertTrue(result["pure_rotation_cancels_in_radial_gauge"])
        self.assertAlmostEqual(
            result["working_geometry_bracket_minimum"], 0.75
        )
        self.assertAlmostEqual(
            result["maximum_baseline_pair_criterion"],
            0.6721268902914064,
            places=8,
        )
        self.assertGreater(
            result["common_one_history_integrated_log_action_ceiling"],
            0.19,
        )
        self.assertLess(
            result["common_one_history_integrated_log_action_ceiling"],
            0.21,
        )
        self.assertGreater(
            result["common_one_history_multiplicative_ceiling"], 1.21
        )
        self.assertFalse(result["boundary_response_constants_certified"])
        self.assertFalse(
            result["actual_Navier_Stokes_residual_norms_certified"]
        )
        self.assertFalse(
            result["full_Navier_Stokes_migrating_residual_gate_closed"]
        )

    def test_wall_stopping_trace_composition_audit(self) -> None:
        result = WALL_STOPPING_TRACE_COMPOSITION_MODULE.audit()
        self.assertTrue(
            result["all_positive_wall_stopping_trace_checks_pass"]
        )
        self.assertTrue(
            result["square_tilted_positive_kernel_identity_verified"]
        )
        self.assertTrue(result["raw_and_square_tilted_forms_agree"])
        self.assertGreater(
            result["wall_only_integrated_log_action_ceiling"], 0.29
        )
        self.assertLess(
            result["wall_only_integrated_log_action_ceiling"], 0.30
        )
        self.assertGreater(
            result[
                "conditional_minimum_return_only_potential_L3_over_2_threshold"
            ],
            0.49,
        )
        self.assertGreater(
            result["conditional_minimum_return_only_drift_L3_threshold"],
            0.15,
        )
        self.assertFalse(result["raw_wall_flux_can_be_used_as_H1_trace_law"])
        self.assertFalse(
            result["actual_return_square_tilted_density_envelope_certified"]
        )
        self.assertFalse(
            result[
                "composite_wall_to_child_core_density_envelope_certified"
            ]
        )
        self.assertFalse(result["full_wall_stopping_trace_gate_closed"])

    def test_neutral_strip_return_density_pilot(self) -> None:
        result = NEUTRAL_STRIP_RETURN_DENSITY_MODULE.audit()
        self.assertTrue(
            result["all_positive_neutral_strip_return_density_checks_pass"]
        )
        self.assertTrue(result["boundary_edge_rate_reconstruction_exact"])
        self.assertTrue(
            result["finite_state_semigroup_exact_for_discrete_generator"]
        )
        self.assertLess(result["maximum_mesh_factor_change"], 0.003)
        self.assertLess(result["maximum_mesh_response_change"], 0.002)
        self.assertGreater(
            result[
                "conditional_minimum_return_only_potential_L3_over_2_threshold"
            ],
            0.36,
        )
        self.assertGreater(
            result["conditional_minimum_return_only_drift_L3_threshold"],
            0.10,
        )
        self.assertFalse(result["continuum_boundary_density_envelope_certified"])
        self.assertFalse(
            result["weighted_Navier_Stokes_return_density_certified"]
        )
        self.assertFalse(result["composite_wall_child_return_density_computed"])
        self.assertFalse(result["full_wall_stopping_trace_gate_closed"])

    def test_neutral_strip_boundary_density_discretization_no_go(self) -> None:
        result = NEUTRAL_STRIP_BOUNDARY_DISCRETIZATION_MODULE.audit(
            structural_meshes=(24, 30),
            density_mesh=24,
            bin_counts=(16, 32, 128, 512),
        )
        self.assertTrue(
            result["all_positive_boundary_discretization_checks_pass"]
        )
        self.assertFalse(result["fixed_mesh_histogram_L2_limit_finite"])
        self.assertFalse(
            result["shortley_weller_generator_reversible_on_all_meshes"]
        )
        self.assertGreater(
            result["density_row"][
                "response_ratio_finest_bins_to_32_bins"
            ],
            1.1,
        )
        self.assertFalse(
            result["previous_32_bin_response_is_a_continuum_upper_bound"]
        )
        self.assertFalse(result["continuum_boundary_density_certified"])

    def test_neutral_strip_reversible_boundary_fem_structure(self) -> None:
        result = NEUTRAL_STRIP_REVERSIBLE_BOUNDARY_FEM_MODULE.audit(
            spacings=(0.28, 0.22),
            run_density=False,
        )
        self.assertTrue(
            result["all_positive_reversible_boundary_fem_checks_pass"]
        )
        self.assertTrue(result["all_retained_off_diagonal_rates_positive"])
        self.assertTrue(result["exact_discrete_reversibility_verified"])
        self.assertTrue(result["probability_partition_verified"])
        self.assertTrue(
            result["inner_boundary_faces_partition_true_circle"]
        )
        self.assertTrue(result["exact_r2_entry_nodes_inserted"])
        self.assertFalse(result["physical_boundary_L2_flux_computed"])
        self.assertFalse(result["coupled_mesh_response_converged"])
        self.assertFalse(result["continuum_return_density_certified"])

    def test_neutral_strip_reversible_spectral_tail_width_structure(self) -> None:
        result = NEUTRAL_STRIP_REVERSIBLE_SPECTRAL_TAIL_WIDTH_MODULE.audit(
            spacing=0.28,
            x_half_widths=(4.2, 5.25),
            run_density=False,
        )
        self.assertTrue(
            result["all_positive_spectral_tail_width_checks_pass"]
        )
        self.assertTrue(
            result["finite_matrix_decay_enclosed_by_high_precision_barta"]
        )
        self.assertTrue(
            result["boundary_operator_norm_has_analytic_upper_bound"]
        )
        self.assertFalse(result["fitted_tail_eliminated"])
        self.assertFalse(result["finite_time_window_maxima_certified"])
        self.assertFalse(result["scalar_time_quadrature_certified"])
        self.assertFalse(result["x_truncation_analytically_removed"])
        self.assertFalse(result["continuum_return_response_certified"])
        json.dumps(result)

    def test_neutral_strip_reversible_finite_time_certificate(self) -> None:
        result = NEUTRAL_STRIP_REVERSIBLE_FINITE_TIME_MODULE.audit(
            spacing=0.28,
            x_half_widths=(4.2,),
            run_density=True,
        )
        self.assertTrue(result["all_finite_time_certificate_checks_pass"])
        self.assertTrue(result["positive_contractive_uniformization_verified"])
        self.assertTrue(result["finite_time_window_maxima_certified"])
        self.assertTrue(result["scalar_time_quadrature_certified"])
        self.assertFalse(result["x_truncation_analytically_removed"])
        self.assertFalse(result["continuum_return_response_certified"])
        row = result["width_rows"][0]["finite_time_certificate"]
        self.assertLess(row["maximum_state_l2_error_bound"], 1.0e-8)
        self.assertLess(row["maximum_finite_interval_enclosure_ratio"], 1.05)
        self.assertLess(row["maximum_certified_response"], 0.7)

        grid = NEUTRAL_STRIP_REVERSIBLE_BOUNDARY_FEM_MODULE._build_mesh(0.28)
        spectral = (
            NEUTRAL_STRIP_REVERSIBLE_SPECTRAL_TAIL_WIDTH_MODULE._spectral_row(
                grid
            )
        )
        operator = NEUTRAL_STRIP_REVERSIBLE_FINITE_TIME_MODULE._operator_data(
            grid, spectral
        )
        entries = np.asarray(grid["entry_states"])[[0, 6, 12, 18]]
        state = np.zeros((len(grid["state_mass"]), len(entries)))
        root_mass = operator["root_mass"]
        state[entries, np.arange(len(entries))] = 1.0 / root_mass[entries]
        duration = 0.02
        propagated, error, _ = (
            NEUTRAL_STRIP_REVERSIBLE_FINITE_TIME_MODULE._uniformization_step(
                operator["poisson_matrix"],
                operator["poisson_matrix_abs"],
                state,
                np.zeros(len(entries)),
                duration,
                operator["uniformization_rate"],
                spectral["principal_decay_barta_lower"],
            )
        )
        independent = expm_multiply(
            operator["symmetric_generator"] * duration, state
        )
        disagreement = np.linalg.norm(propagated - independent, axis=0)
        self.assertTrue(np.all(disagreement <= error + 2.0e-11))
        json.dumps(result)

    def test_neutral_strip_x_exit_correction_structure(self) -> None:
        result = NEUTRAL_STRIP_X_EXIT_CORRECTION_MODULE.audit(
            run_direct_certificate=False
        )
        self.assertTrue(result["all_x_exit_correction_checks_pass"])
        self.assertTrue(
            result["continuum_side_exit_probability_analytically_bounded"]
        )
        self.assertTrue(result["x_exit_interval_renewal_correction_proved"])
        self.assertTrue(result["x_exit_scalar_correction_proved"])
        self.assertFalse(
            result["x_truncation_removed_from_continuum_return_theorem"]
        )
        self.assertFalse(result["continuum_return_response_certified"])
        probabilities = [
            row["continuum_side_exit_probability_upper"]
            for row in result["side_exit_rows"]
        ]
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertGreater(probabilities[1], probabilities[2])
        self.assertLess(probabilities[2], 4.4e-8)

        for side_row in result["side_exit_rows"]:
            width = side_row["x_half_width"]
            reference_sum = 0.0
            for mode in range(side_row["retained_mode_count"]):
                wave_number = (
                    (2 * mode + 1)
                    * np.pi
                    / (2 * result["side_exit_rows"][0]["strip_half_width"])
                )
                parameter = wave_number**2 / 2.0
                transform = hyp1f1(
                    parameter, 0.5, 2.0
                ) / hyp1f1(parameter, 0.5, width**2 / 2.0)
                reference_sum += (
                    4.0 / (np.pi * (2 * mode + 1)) * transform
                )
            self.assertLessEqual(
                reference_sum,
                side_row["continuum_side_exit_probability_upper"],
            )
            self.assertLess(
                side_row["continuum_side_exit_probability_upper"]
                - reference_sum,
                max(1.0e-14, 1.0e-10 * reference_sum),
            )
        json.dumps(result)

    def test_neutral_strip_reversible_fem_consistency_gate(self) -> None:
        local_mass = np.asarray(
            [
                [2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0],
            ]
        )
        coercivity_minors = (
            NEUTRAL_STRIP_REVERSIBLE_FEM_CONSISTENCY_MODULE
            ._local_mass_coercivity_minors(local_mass, 0.15)
        )
        self.assertTrue(all(value > 0.0 for value in coercivity_minors))
        result = NEUTRAL_STRIP_REVERSIBLE_FEM_CONSISTENCY_MODULE.audit(
            spacings=(0.28,),
            low_mode_count=8,
            quadrature_order=8,
        )
        self.assertTrue(
            result["all_reversible_fem_consistency_gate_checks_pass"]
        )
        self.assertTrue(
            result["polygonal_circle_geometry_analytically_quantified"]
        )
        self.assertTrue(
            result["reference_quadrature_numerically_cross_checked"]
        )
        self.assertFalse(result["global_mass_form_near_identity"])
        self.assertFalse(
            result["whole_spectrum_multiplicative_perturbation_closes"]
        )
        self.assertTrue(result["low_mode_consistency_observed"])
        self.assertTrue(
            result["consistent_mass_transient_boundary_term_assembled"]
        )
        self.assertFalse(result["legacy_stiffness_only_boundary_map_complete"])
        self.assertTrue(result["parabolic_low_high_mode_split_required"])
        self.assertFalse(result["reference_quadrature_interval_certified"])
        self.assertFalse(result["continuum_boundary_flux_error_certified"])
        self.assertFalse(result["continuum_return_response_certified"])
        row = result["rows"][0]
        self.assertGreater(
            row["global_modified_to_reference_mass_ratio"][1], 2.0
        )
        self.assertLess(
            row["low_mode_modified_mass_ratio"][1], 1.15
        )
        self.assertLess(
            result["maximum_quadrature_order_cross_check"], 1.0e-10
        )
        json.dumps(result)

    def test_neutral_strip_parabolic_spectral_split(self) -> None:
        result = NEUTRAL_STRIP_PARABOLIC_SPECTRAL_SPLIT_MODULE.audit(
            run_modes=False,
            analytic_cutoff=60.0,
        )
        self.assertTrue(result["all_parabolic_spectral_split_checks_pass"])
        self.assertTrue(
            result["continuum_weighted_rellich_flux_bound_proved"]
        )
        self.assertTrue(
            result[
                "continuum_first_omitted_eigenvalue_lower_bounded_by_li_yau"
            ]
        )
        self.assertTrue(result["killed_kernel_bounded_by_full_ou_diagonal"])
        self.assertTrue(result["high_mode_half_time_factorization_proved"])
        self.assertTrue(
            result["all_later_window_high_mode_budget_bounded"]
        )
        self.assertFalse(result["production_low_mode_diagnostic_completed"])
        self.assertFalse(
            result["low_mode_variational_crimes_interval_certified"]
        )
        self.assertTrue(result["continuum_first_window_flux_certified"])
        self.assertTrue(result["first_window_response_budget_closed"])
        self.assertTrue(
            result["consistent_mass_transient_conormal_identity_proved"]
        )
        self.assertTrue(
            result["independent_low_block_algebra_regression_passes"]
        )
        regression = result["independent_low_block_algebra_regression"]
        self.assertLess(
            regression["off_block_residual_identity_error"], 2.0e-13
        )
        self.assertAlmostEqual(
            regression["one_dimensional_transient_conormal"], 3.0
        )
        boundary_regression = result["independent_boundary_L2_regression"]
        self.assertTrue(result["independent_boundary_L2_regression_passes"])
        self.assertLess(
            boundary_regression["pushed_gram_quadrature_error"], 2.0e-13
        )
        self.assertLess(
            boundary_regression["cross_gram_quadrature_error"], 2.0e-13
        )
        self.assertFalse(result["legacy_stiffness_only_boundary_map_complete"])
        self.assertTrue(result["polygon_flux_measure_pushforward_factor_proved"])
        self.assertTrue(
            result["polygon_flux_pushforward_conditional_on_L2_density"]
        )
        self.assertFalse(
            result["boundary_Riesz_reconstruction_interval_certified"]
        )
        self.assertTrue(result["boundary_Riesz_common_circle_geometry_assembled"])
        self.assertTrue(result["entry_source_projection_assembled"])
        self.assertTrue(result["legacy_raw_load_screen_below_one"])
        self.assertFalse(
            result["legacy_raw_load_screen_is_valid_boundary_L2_screen"]
        )
        self.assertFalse(
            result["time_zero_common_circle_one_for_one_screen_below_one"]
        )
        self.assertTrue(result["sampled_source_common_circle_screen_below_one"])
        self.assertGreater(
            result["sampled_source_common_circle_screen_headroom"], 0.03
        )
        self.assertFalse(
            result["later_window_source_time_suprema_interval_certified"]
        )
        self.assertFalse(
            result["post_terminal_source_discrepancy_tail_certified"]
        )
        self.assertTrue(result["time_slab_partition_nonoverlapping"])
        self.assertTrue(
            result["frozen_finite_block_time_slab_enclosure_proved"]
        )
        self.assertTrue(
            result[
                "frozen_finite_block_post_terminal_tail_enclosure_proved"
            ]
        )
        self.assertTrue(
            result["frozen_binary_endpoint_arithmetic_directed_enclosed"]
        )
        self.assertTrue(
            result["frozen_binary_endpoint_guard_dominates_derived_roundoff"]
        )
        self.assertTrue(
            result["frozen_binary_endpoint_inputs_treated_as_exact_binary64"]
        )
        self.assertTrue(result["stored_mass_row_lumped_coercivity_proved"])
        self.assertTrue(
            result["stored_matrix_eigenpair_residuals_directed_enclosed"]
        )
        self.assertTrue(
            result["reference_eigenvalue_proximity_intervals_proved"]
        )
        self.assertTrue(
            result["indexed_generalized_eigenvalue_inclusions_proved"]
        )
        self.assertTrue(result["stored_generalized_eigenvalues_indexed"])
        self.assertTrue(
            result["exact_polygon_generalized_eigenvalues_indexed"]
        )
        self.assertGreater(
            result[
                "exact_polygon_complement_generalized_eigenvalue_lower_bound"
            ],
            107.0,
        )
        self.assertFalse(
            result["endpoint_effect_of_eigenpair_residuals_certified"]
        )
        self.assertTrue(
            result["reference_finite_element_assembly_interval_enclosed"]
        )
        self.assertTrue(result["reference_quadrature_interval_certified"])
        self.assertLess(
            result["binary_frozen_reference_assembly_audit"][
                "absolute_mass_error_relative_to_stored_mass_form"
            ],
            6.0e-13,
        )
        self.assertLess(
            result["binary_frozen_reference_assembly_audit"][
                "absolute_stiffness_error_in_stored_mass_form_units"
            ],
            6.0e-9,
        )
        self.assertGreater(
            result["binary_frozen_eigensystem_residual_audit"][
                "retained_cutoff_proximity_interval_separation"
            ],
            0.6,
        )
        self.assertLess(
            result["binary_frozen_endpoint_roundoff_audit"][
                "maximum_directed_roundoff_norm_error_upper"
            ],
            5.0e-11,
        )
        self.assertFalse(
            result[
                "frozen_finite_block_coefficient_matrices_interval_enclosed"
            ]
        )
        self.assertLess(
            result["h006_refined_frozen_time_slab_combined_screen_total"],
            0.971,
        )
        self.assertFalse(
            result["static_boundary_screen_is_complete_low_block_comparison"]
        )
        self.assertFalse(result["retained_projected_dynamics_interval_certified"])
        self.assertFalse(result["modified_low_space_leakage_interval_bounded"])
        self.assertFalse(
            result["gap_free_contractive_Duhamel_leakage_screen_passes"]
        )
        self.assertFalse(result["low_block_source_trace_map_interval_certified"])
        self.assertFalse(result["polygon_to_circle_flux_map_certified"])
        self.assertFalse(result["continuum_return_response_certified"])
        budget = result["analytic_high_mode_budget"]
        self.assertGreater(
            result["continuum_li_yau_cutoff"][
                "li_yau_weighted_operator_lower"
            ],
            60.0,
        )
        self.assertGreater(
            budget["continuum_L2_source_to_inner_flux_constant"], 3.0
        )
        self.assertLess(
            budget["continuum_L2_source_to_inner_flux_constant"], 3.2
        )
        self.assertLess(
            budget["all_later_windows_high_interval_factor_upper"],
            0.01,
        )
        self.assertLess(
            budget["all_later_time_high_scalar_gain_upper"],
            0.01,
        )
        json.dumps(result)

    def test_neutral_strip_first_window_brownian_majorant(self) -> None:
        result = NEUTRAL_STRIP_FIRST_WINDOW_BROWNIAN_MAJORANT_MODULE.audit(
            run_pointwise_pilot=False,
            run_inversion=False,
        )
        self.assertTrue(result["all_first_window_majorant_checks_pass"])
        self.assertTrue(result["OU_stopped_hit_measure_domination_proved"])
        self.assertTrue(
            result["Bessel_absolute_continuity_mode_identity_proved"]
        )
        self.assertTrue(result["infinite_angular_mode_tail_summed"])
        self.assertTrue(result["continuum_first_window_flux_certified"])
        self.assertFalse(result["first_window_response_budget_closed"])
        budget = result["uniform_analytic_budget"]
        self.assertGreater(budget["uniform_raw_spatial_L2_upper"], 1.0)
        self.assertGreater(
            budget["first_window_interval_factor_upper"], 1.0
        )
        self.assertLess(budget["first_window_scalar_gain_upper"], 1.0)
        json.dumps(result)

    def test_neutral_strip_first_window_maximum_bridge(self) -> None:
        result = NEUTRAL_STRIP_FIRST_WINDOW_MAXIMUM_BRIDGE_MODULE.certificate()
        self.assertTrue(
            result["all_first_window_bridge_certificate_checks_pass"]
        )
        self.assertTrue(
            result["positive_bridge_maximum_quadrature_enclosed"]
        )
        self.assertTrue(result["finite_low_Brownian_modes_interval_enclosed"])
        self.assertTrue(result["complete_first_window_time_supremum_enclosed"])
        self.assertLess(
            result["complete_first_window_interval_factor_upper"], 0.96
        )
        self.assertGreater(
            result["complete_first_window_interval_factor_upper"], 0.94
        )
        self.assertLess(
            result["maximum_omitted_squared_mode_sum_upper"], 2.0e-9
        )
        self.assertTrue(result["first_window_response_budget_closed"])
        self.assertFalse(result["continuum_return_response_certified"])
        json.dumps(result)

    def test_neutral_strip_transient_conormal_low_block(self) -> None:
        result = NEUTRAL_STRIP_TRANSIENT_CONORMAL_LOW_BLOCK_MODULE.audit()
        self.assertTrue(result["all_transient_conormal_low_block_checks_pass"])
        self.assertTrue(
            result["consistent_mass_transient_conormal_identity_proved"]
        )
        self.assertFalse(result["legacy_stiffness_only_boundary_map_complete"])
        self.assertTrue(result["polygon_flux_measure_pushforward_factor_proved"])
        self.assertTrue(
            result["polygon_flux_pushforward_conditional_on_L2_density"]
        )
        self.assertFalse(
            result["finite_element_conormal_load_is_polygon_L2_density"]
        )
        self.assertFalse(
            result["boundary_Riesz_reconstruction_interval_certified"]
        )
        self.assertTrue(result["legacy_raw_load_screen_below_one"])
        self.assertFalse(
            result["legacy_raw_load_static_screen_is_valid_boundary_L2_screen"]
        )
        self.assertTrue(result["boundary_Riesz_common_circle_geometry_assembled"])
        self.assertTrue(result["entry_source_projection_assembled"])
        self.assertFalse(
            result["h006_time_zero_common_circle_one_for_one_screen_below_one"]
        )
        self.assertTrue(
            result["h006_sampled_source_common_circle_screen_below_one"]
        )
        self.assertFalse(
            result["later_window_source_time_suprema_interval_certified"]
        )
        self.assertFalse(
            result["post_terminal_source_discrepancy_tail_certified"]
        )
        self.assertTrue(result["time_slab_partition_nonoverlapping"])
        self.assertTrue(
            result["frozen_finite_block_time_slab_enclosure_proved"]
        )
        self.assertTrue(
            result[
                "frozen_finite_block_post_terminal_tail_enclosure_proved"
            ]
        )
        self.assertTrue(
            result["frozen_binary_endpoint_arithmetic_directed_enclosed"]
        )
        self.assertTrue(
            result["frozen_binary_endpoint_guard_dominates_derived_roundoff"]
        )
        self.assertTrue(
            result["frozen_binary_endpoint_high_precision_spot_check_covered"]
        )
        self.assertTrue(result["stored_mass_row_lumped_coercivity_proved"])
        self.assertTrue(
            result["stored_matrix_eigenpair_residuals_directed_enclosed"]
        )
        self.assertTrue(
            result["distinct_reference_eigenvalue_proximity_intervals_proved"]
        )
        self.assertTrue(
            result["indexed_generalized_eigenvalue_inclusions_proved"]
        )
        self.assertTrue(result["stored_generalized_eigenvalues_indexed"])
        self.assertTrue(
            result["exact_polygon_generalized_eigenvalues_indexed"]
        )
        self.assertTrue(
            result["reference_finite_element_mass_form_interval_enclosed"]
        )
        self.assertTrue(
            result[
                "reference_finite_element_stiffness_form_interval_enclosed"
            ]
        )
        self.assertTrue(
            result[
                "reference_finite_element_boundary_couplings_interval_enclosed"
            ]
        )
        self.assertTrue(
            result["reference_finite_element_assembly_interval_enclosed"]
        )
        self.assertTrue(result["reference_quadrature_interval_certified"])
        self.assertEqual(
            result["binary_frozen_endpoint_roundoff_audit"]["endpoint_count"],
            451,
        )
        self.assertFalse(
            result[
                "frozen_finite_block_coefficient_matrices_interval_enclosed"
            ]
        )
        self.assertGreater(
            result["h006_refined_frozen_time_slab_combined_screen_headroom"],
            0.029,
        )
        self.assertFalse(
            result[
                "naive_radial_map_one_for_one_additive_screen_below_one"
            ]
        )
        self.assertFalse(
            result["all_radial_map_perturbation_theorems_ruled_out"]
        )
        self.assertTrue(result["retained_projected_dynamics_diagnostic_completed"])
        self.assertFalse(
            result["static_boundary_screen_is_complete_low_block_comparison"]
        )
        self.assertFalse(result["retained_projected_dynamics_interval_certified"])
        self.assertFalse(result["modified_low_space_leakage_interval_bounded"])
        self.assertFalse(
            result["gap_free_contractive_Duhamel_leakage_screen_passes"]
        )
        self.assertFalse(result["low_block_source_trace_map_interval_certified"])
        self.assertGreater(
            result["h006_sampled_source_common_circle_screen_headroom"], 0.03
        )
        dynamics = result["h006_projected_dynamics_diagnostic"]
        self.assertLess(
            dynamics["retained_projected_semigroup_output_rows"][0][
                "full_output_discrepancy_over_same_time_reference"
            ],
            0.002,
        )
        self.assertGreater(
            dynamics[
                "retained_modified_invariance_residual_relative_Minv_spectral"
            ],
            0.06,
        )
        self.assertGreater(
            dynamics[
                "first_later_time_naive_contractive_Duhamel_leakage_upper"
            ],
            2.0,
        )
        self.assertTrue(result["reference_quadrature_interval_certified"])
        self.assertFalse(result["continuum_Ritz_projector_error_certified"])
        self.assertFalse(result["polygon_domain_perturbation_certified"])
        self.assertFalse(result["retained_continuum_low_block_certified"])
        self.assertFalse(result["continuum_return_response_certified"])
        summary_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "neutral_strip_common_circle_source_summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="ascii"))
        self.assertEqual(len(summary["common_circle_production_rows"]), 4)
        self.assertAlmostEqual(
            summary["h006_sampled_source_common_circle_screen"]["total"],
            result["h006_sampled_source_common_circle_screen_total"],
        )
        self.assertEqual(len(summary["frozen_time_slab_enclosures"]), 3)
        self.assertAlmostEqual(
            summary["frozen_time_slab_enclosures"][-1][
                "combined_screen_total"
            ],
            result["h006_refined_frozen_time_slab_combined_screen_total"],
        )
        self.assertTrue(
            summary["certification_flags"][
                "frozen_finite_block_time_slab_enclosure_proved"
            ]
        )
        self.assertTrue(
            summary["certification_flags"][
                "frozen_binary_endpoint_arithmetic_directed_enclosed"
            ]
        )
        self.assertTrue(
            summary["certification_flags"][
                "stored_matrix_eigenpair_residuals_directed_enclosed"
            ]
        )
        self.assertTrue(
            summary["certification_flags"][
                "reference_finite_element_assembly_interval_enclosed"
            ]
        )
        self.assertTrue(
            summary["certification_flags"][
                "indexed_generalized_eigenvalue_inclusions_proved"
            ]
        )
        self.assertTrue(
            summary["certification_flags"][
                "exact_polygon_reference_generalized_eigenvalues_indexed"
            ]
        )
        self.assertTrue(
            summary["certification_flags"][
                "reference_quadrature_interval_certified"
            ]
        )
        self.assertFalse(
            summary["certification_flags"][
                "frozen_finite_block_coefficient_matrices_interval_enclosed"
            ]
        )
        self.assertFalse(
            summary["certification_flags"][
                "continuum_return_response_certified"
            ]
        )
        json.dumps(result)

    def test_neutral_strip_common_circle_time_slab_interpolation(self) -> None:
        module = NEUTRAL_STRIP_COMMON_CIRCLE_TIME_SLAB_MODULE
        regression = module._independent_interpolation_regression()
        self.assertTrue(regression["passes"])
        self.assertGreater(regression["enclosure_margin"], 0.0)
        self.assertEqual(
            int(round(module.TERMINAL_TIME / module.WINDOW)) - 1,
            15,
        )
        self.assertEqual(int(round(module.TERMINAL_TIME / module.WINDOW)), 16)
        for start in (module.WINDOW, 3.0, module.TERMINAL_TIME):
            variance = math.expm1(2.0 * start)
            exact_axial = math.exp(start) * math.sqrt(
                math.erf(module.PATCH_HALF_HEIGHT / math.sqrt(variance))
                / (2.0 * math.sqrt(math.pi) * math.sqrt(variance))
            )
            self.assertGreaterEqual(
                module._axial_l2_global_upper(start),
                exact_axial,
            )
        results = Path(__file__).resolve().parents[1] / "results"
        coarse = json.loads(
            (
                results
                / "neutral_strip_h006_q12_k240_source_time_slab_certificate_v1.json"
            ).read_text(encoding="ascii")
        )
        refined = json.loads(
            (
                results
                / "neutral_strip_h006_q12_k240_source_time_slab_certificate_refined_v1.json"
            ).read_text(encoding="ascii")
        )
        self.assertTrue(coarse["all_frozen_time_slab_certificate_checks_pass"])
        self.assertTrue(refined["all_frozen_time_slab_certificate_checks_pass"])
        self.assertEqual(len(refined["window_rows"]), 15)
        self.assertEqual(refined["tail_first_window_index"], 16)
        self.assertLess(
            refined["later_low_block_interval_factor_upper"],
            coarse["later_low_block_interval_factor_upper"],
        )
        self.assertLess(refined["combined_screen_total"], 0.971)
        self.assertFalse(refined["coefficient_matrices_interval_enclosed"])
        roundoff = json.loads(
            (
                results
                / "neutral_strip_h006_q12_k240_endpoint_roundoff_audit_v1.json"
            ).read_text(encoding="ascii")
        )
        self.assertTrue(roundoff["all_endpoint_roundoff_checks_pass"])
        self.assertEqual(roundoff["endpoint_count"], 451)
        self.assertTrue(
            roundoff["existing_guard_dominates_all_derived_roundoff"]
        )
        self.assertTrue(
            roundoff["independent_high_precision_spot_check"][
                "covered_by_directed_roundoff_upper"
            ]
        )
        self.assertFalse(roundoff["assembled_coefficient_matrices_interval_enclosed"])
        self.assertFalse(
            roundoff["reference_generalized_eigenpairs_interval_enclosed"]
        )
        self.assertLess(
            roundoff["maximum_directed_roundoff_norm_error_upper"],
            5.0e-11,
        )
        eigensystem = json.loads(
            (
                results
                / "neutral_strip_h006_q12_k240_eigensystem_residual_audit_v1.json"
            ).read_text(encoding="ascii")
        )
        self.assertTrue(eigensystem["all_eigensystem_residual_checks_pass"])
        reference = eigensystem["reference_generalized_eigensystem"]
        self.assertEqual(len(reference["pair_rows"]), 241)
        self.assertTrue(
            reference["mass_coercivity"][
                "stored_mass_row_lumped_coercivity_proved"
            ]
        )
        self.assertTrue(reference["all_adjacent_proximity_intervals_disjoint"])
        self.assertGreater(
            reference["retained_cutoff_proximity_interval_separation"],
            0.6,
        )
        self.assertLess(
            reference["maximum_directed_inverse_mass_residual_upper"],
            1.0e-9,
        )
        self.assertFalse(eigensystem["generalized_eigenvalue_inclusions_proved"])
        self.assertFalse(
            eigensystem["endpoint_effect_of_eigenpair_residuals_certified"]
        )
        self.assertLess(
            max(
                row["absolute_difference"]
                for row in refined["cross_check_rows"]
            ),
            2.0e-14,
        )
        json.dumps(regression)

    def test_neutral_strip_gaussian_weighted_assembly_interval(self) -> None:
        result_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
        )
        result = json.loads(result_path.read_text(encoding="ascii"))
        self.assertTrue(result["complete_mesh_audit"])
        self.assertEqual(result["selected_triangle_count"], 30954)
        self.assertTrue(
            result["local_interval_checks"]["all_local_enclosures_valid"]
        )
        self.assertEqual(
            result["quadrature_diagnostics"][
                "q24_containment_failure_count"
            ],
            0,
        )
        self.assertTrue(
            result["stored_matrix_reconstruction"]["fingerprints_match"]
        )
        self.assertTrue(
            result["stored_matrix_reconstruction"][
                "all_four_matrix_fingerprints_match"
            ]
        )
        self.assertEqual(
            result["stored_matrix_reconstruction"][
                "reconstructed_matrix_fingerprint_sha256"
            ],
            "c80fdc5aa494fe1118aec4c38045df76151b2ccde3fded18f4af1161de7efd48",
        )
        self.assertLess(
            result["form_bounds"][
                "absolute_mass_error_relative_to_stored_mass_form"
            ],
            6.0e-13,
        )
        self.assertLess(
            result["form_bounds"][
                "absolute_stiffness_error_in_stored_mass_form_units"
            ],
            6.0e-9,
        )
        self.assertGreater(
            result["form_bounds"][
                "exact_mass_lower_relative_to_stored_mass"
            ],
            0.999999999999,
        )
        self.assertTrue(result["finite_element_assembly_interval_enclosed"])
        self.assertFalse(
            result["checkpoint"][
                "parked_for_repeated_cpu_threshold_breach"
            ]
        )
        crosscheck_path = (
            result_path.parent
            / "neutral_strip_h006_gaussian_assembly_reconstruction_crosscheck_v1.json"
        )
        crosscheck = json.loads(crosscheck_path.read_text(encoding="ascii"))
        self.assertTrue(crosscheck["all_reconstruction_cross_checks_pass"])
        self.assertTrue(crosscheck["all_four_matrices_bitwise_equal"])
        for row in crosscheck["matrix_rows"].values():
            self.assertTrue(row["bitwise_equal"])
            self.assertEqual(row["maximum_absolute_difference"], 0.0)

    def test_neutral_strip_sparse_inertia_indexing(self) -> None:
        results = Path(__file__).resolve().parents[1] / "results"
        primary = json.loads(
            (
                results
                / "neutral_strip_h006_q12_k240_sparse_inertia_audit_v1.json"
            ).read_text(encoding="ascii")
        )
        self.assertTrue(primary["complete_production_audit"])
        self.assertTrue(primary["all_sparse_inertia_audit_checks_pass"])
        self.assertTrue(
            primary["first_240_stored_generalized_eigenvalues_indexed"]
        )
        self.assertTrue(
            primary[
                "all_241_stored_generalized_eigenvalue_intervals_indexed"
            ]
        )
        self.assertEqual(
            primary["inertia_rows"]["retained_gap"][
                "negative_pivot_count"
            ],
            240,
        )
        self.assertEqual(
            primary["inertia_rows"]["post_241_interval"][
                "negative_pivot_count"
            ],
            241,
        )
        self.assertLess(
            primary["inertia_rows"]["post_241_interval"][
                "maximum_relative_pivot_interval_width"
            ],
            0.1,
        )

        crosscheck = json.loads(
            (
                results
                / "neutral_strip_h006_q12_k240_sparse_inertia_precision_crosscheck_v1.json"
            ).read_text(encoding="ascii")
        )
        self.assertTrue(
            crosscheck[
                "all_sparse_inertia_precision_crosscheck_checks_pass"
            ]
        )
        self.assertTrue(crosscheck["all_30422_pivot_signs_reproduced"])
        self.assertTrue(
            crosscheck["all_30422_crosscheck_intervals_nested"]
        )
        for row in crosscheck["rows"].values():
            self.assertTrue(row["all_row_crosscheck_checks_pass"])
            self.assertTrue(row["permutations_equal"])
            self.assertTrue(row["shifts_bitwise_equal"])

    def test_neutral_strip_exact_polygon_indexed_spectrum_transfer(
        self,
    ) -> None:
        result_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "neutral_strip_h006_exact_polygon_indexed_spectrum_transfer_v1.json"
        )
        result = json.loads(result_path.read_text(encoding="ascii"))
        self.assertTrue(
            result[
                "all_exact_polygon_indexed_spectrum_transfer_checks_pass"
            ]
        )
        self.assertEqual(result["indexed_interval_count"], 241)
        self.assertTrue(
            result[
                "first_240_exact_polygon_generalized_eigenvalues_indexed"
            ]
        )
        self.assertTrue(
            result[
                "all_241_exact_polygon_generalized_eigenvalues_indexed"
            ]
        )
        self.assertTrue(result["all_decimal_formula_values_contained"])
        self.assertGreater(
            result["minimum_adjacent_exact_interval_separation"],
            8.0e-5,
        )
        self.assertGreater(
            result["exact_retained_complement_separation"],
            0.6,
        )
        self.assertGreater(
            result[
                "exact_polygon_complement_generalized_eigenvalue_lower_bound"
            ],
            107.0,
        )
        self.assertFalse(result["continuum_Ritz_transfer_proved"])
        self.assertFalse(
            result["polygon_to_circle_domain_transfer_proved"]
        )

    def test_wall_migration_child_return_density_pilot(self) -> None:
        result = WALL_MIGRATION_CHILD_RETURN_DENSITY_MODULE.audit()
        self.assertTrue(
            result["all_positive_wall_child_composite_checks_pass"]
        )
        self.assertTrue(result["finite_state_composite_K_S_computed"])
        self.assertTrue(
            result["time_zero_wall_atom_propagated_through_child_return"]
        )
        self.assertLess(result["maximum_mesh_scalar_gain_change"], 0.02)
        self.assertLess(result["maximum_mesh_interval_factor_change"], 0.03)
        self.assertLess(result["maximum_mesh_response_change"], 0.005)
        working = result["mesh_rows"][-1]
        self.assertLess(
            working["maximum_composite_trace_response_at_alpha_zero"],
            0.08,
        )
        self.assertFalse(result["continuum_composite_K_S_certified"])
        self.assertFalse(
            result["nonaffine_Navier_Stokes_composite_K_S_certified"]
        )
        self.assertFalse(result["full_wall_stopping_trace_gate_closed"])


if __name__ == "__main__":
    unittest.main()
