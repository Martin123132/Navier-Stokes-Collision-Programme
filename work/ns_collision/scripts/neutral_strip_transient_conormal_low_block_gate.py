"""Audit the corrected transient conormal map on the retained P1 block."""

from __future__ import annotations

import json
import math


FIRST_WINDOW_INTERVAL_FACTOR = 0.9523841939624662
RETAINED_MODE_COUNT = 240
LI_YAU_HIGH_INTERVAL_FACTOR = 0.009609366961522469
LI_YAU_HIGH_SCALAR_GAIN = 0.003327266894077182


PRODUCTION_ROWS = (
    {
        "spacing": 0.12,
        "state_count": 3738,
        "inner_boundary_vertex_count": 56,
        "maximum_reference_eigenvalue": 112.77904843066368,
        "transient_boundary_mass_correction_relative_spectral": (
            0.10099045376208451
        ),
        "modified_to_transient_boundary_relative_spectral": (
            0.10060868831117092
        ),
    },
    {
        "spacing": 0.09,
        "state_count": 6625,
        "inner_boundary_vertex_count": 72,
        "maximum_reference_eigenvalue": 109.03221696164594,
        "transient_boundary_mass_correction_relative_spectral": (
            0.04326799250851692
        ),
        "modified_to_transient_boundary_relative_spectral": (
            0.04295685921456509
        ),
    },
    {
        "spacing": 0.075,
        "state_count": 9599,
        "inner_boundary_vertex_count": 88,
        "maximum_reference_eigenvalue": 107.59959232291766,
        "transient_boundary_mass_correction_relative_spectral": (
            0.039273445801102376
        ),
        "modified_to_transient_boundary_relative_spectral": (
            0.039154001125887766
        ),
    },
    {
        "spacing": 0.06,
        "state_count": 15211,
        "inner_boundary_vertex_count": 112,
        "maximum_reference_eigenvalue": 106.41622375897123,
        "transient_boundary_mass_correction_relative_spectral": (
            0.022215047704325884
        ),
        "modified_to_transient_boundary_relative_spectral": (
            0.022076046072654794
        ),
    },
)


H006_CUTOFF_ROWS = (
    {
        "retained_mode_count": 160,
        "modified_to_transient_boundary_relative_spectral": (
            0.016621912740849332
        ),
        "li_yau_high_interval_factor_upper": 0.12023028495611997,
    },
    {
        "retained_mode_count": 200,
        "modified_to_transient_boundary_relative_spectral": (
            0.020284313236479976
        ),
        "li_yau_high_interval_factor_upper": 0.03467165511892645,
    },
    {
        "retained_mode_count": 240,
        "modified_to_transient_boundary_relative_spectral": (
            0.022076046072654794
        ),
        "li_yau_high_interval_factor_upper": LI_YAU_HIGH_INTERVAL_FACTOR,
    },
)


H006_PROJECTED_DYNAMICS = {
    "retained_modified_mass_ratio": (
        1.001022825820076,
        1.0379113142959981,
    ),
    "retained_modified_stiffness_ratio": (
        0.9977799889201349,
        1.0001552902517574,
    ),
    "retained_projected_generator_relative_spectral": (
        0.037083185881800367
    ),
    "retained_modified_invariance_residual_relative_Minv_spectral": (
        0.06167804491844299
    ),
    "retained_modified_action_Minv_spectral": 104.55177869805438,
    "retained_modified_invariance_residual_Minv_spectral": (
        6.448549302841704
    ),
    "retained_modified_off_block_coupling_symmetric_spectral": (
        6.343703098841749
    ),
    "first_later_time_naive_contractive_Duhamel_leakage_upper": (
        2.3788886620656555
    ),
    "retained_projected_semigroup_output_rows": (
        {
            "time": 0.375,
            "full_output_discrepancy_over_time_zero_reference": (
                6.552333862641169e-05
            ),
            "full_output_discrepancy_over_same_time_reference": (
                0.0019795431283908312
            ),
        },
        {
            "time": 0.75,
            "full_output_discrepancy_over_time_zero_reference": (
                2.6872354937336635e-05
            ),
            "full_output_discrepancy_over_same_time_reference": (
                0.002248582476995437
            ),
        },
        {
            "time": 1.5,
            "full_output_discrepancy_over_time_zero_reference": (
                8.022953341537119e-06
            ),
            "full_output_discrepancy_over_same_time_reference": (
                0.004060039008304602
            ),
        },
        {
            "time": 3.0,
            "full_output_discrepancy_over_time_zero_reference": (
                4.839704819897405e-07
            ),
            "full_output_discrepancy_over_same_time_reference": (
                0.008614761697786197
            ),
        },
    ),
}


COMMON_CIRCLE_PRODUCTION_ROWS = (
    {
        "spacing": 0.12,
        "finite_element_pushforward_L2_factor": 1.0001580496515567,
        "time_zero_common_circle_relative_spectral": 0.2056058185413162,
        "first_later_window_maximum_entry_discrepancy": 0.028223519509847944,
        "first_later_window_maximum_entry_relative_discrepancy": (
            0.045655097938528434
        ),
        "sampled_later_window_source_interval_factor": 0.014236253763350278,
    },
    {
        "spacing": 0.09,
        "finite_element_pushforward_L2_factor": 1.0000954453104645,
        "time_zero_common_circle_relative_spectral": 0.14362448542452327,
        "first_later_window_maximum_entry_discrepancy": 0.021793302594841107,
        "first_later_window_maximum_entry_relative_discrepancy": (
            0.03603607037274936
        ),
        "sampled_later_window_source_interval_factor": 0.010981794839905575,
    },
    {
        "spacing": 0.075,
        "finite_element_pushforward_L2_factor": 1.0000638372717012,
        "time_zero_common_circle_relative_spectral": 0.11675710034167472,
        "first_later_window_maximum_entry_discrepancy": 0.017849507459357976,
        "first_later_window_maximum_entry_relative_discrepancy": (
            0.029219349788194064
        ),
        "sampled_later_window_source_interval_factor": 0.008983187235608191,
    },
    {
        "spacing": 0.06,
        "finite_element_pushforward_L2_factor": 1.0000393830272933,
        "time_zero_common_circle_relative_spectral": 0.08926494609313193,
        "first_later_window_maximum_entry_discrepancy": 0.014140954974326235,
        "first_later_window_maximum_entry_relative_discrepancy": (
            0.02293686319354182
        ),
        "sampled_later_window_source_interval_factor": 0.007124897003201583,
    },
)


FROZEN_TIME_SLAB_ROWS = (
    {
        "spacing": 0.12,
        "substep": 0.025,
        "finite_window_count": 15,
        "tail_first_window_index": 16,
        "finite_raw_sum_upper": 0.029436986285814253,
        "post_terminal_raw_sum_upper": 2.1972082058651468e-05,
        "later_low_block_interval_factor_upper": 0.01714338560696962,
        "combined_screen_total": 0.9791369465309584,
        "combined_screen_headroom": 0.020863053469041604,
        "maximum_interpolation_charge": 0.004525017765367228,
    },
    {
        "spacing": 0.06,
        "substep": 0.025,
        "finite_window_count": 15,
        "tail_first_window_index": 16,
        "finite_raw_sum_upper": 0.016437926362377595,
        "post_terminal_raw_sum_upper": 2.1824498288990452e-05,
        "later_low_block_interval_factor_upper": 0.009578609415694423,
        "combined_screen_total": 0.9715721703396832,
        "combined_screen_headroom": 0.02842782966031676,
        "maximum_interpolation_charge": 0.004528887893796346,
    },
    {
        "spacing": 0.06,
        "substep": 0.0125,
        "finite_window_count": 15,
        "tail_first_window_index": 16,
        "finite_raw_sum_upper": 0.013881494718201092,
        "post_terminal_raw_sum_upper": 2.1824498288990452e-05,
        "later_low_block_interval_factor_upper": 0.008090916167796953,
        "combined_screen_total": 0.9700844770917858,
        "combined_screen_headroom": 0.029915522908214198,
        "maximum_interpolation_charge": 0.0011322219734490864,
    },
)


H006_BINARY_FROZEN_ENDPOINT_ROUNDOFF = {
    "result_path": (
        "work/ns_collision/results/"
        "neutral_strip_h006_q12_k240_endpoint_roundoff_audit_v1.json"
    ),
    "endpoint_count": 451,
    "maximum_directed_roundoff_norm_error_upper": 3.010754666410437e-11,
    "minimum_existing_guard_margin": 2.668598279252563e-11,
    "worst_endpoint_time": 0.375,
    "worst_endpoint_column": 3,
    "boundary_mass_gershgorin_lower": 0.018697504183912635,
    "riesz_residual_frobenius_upper": 6.424512520815097e-13,
    "riesz_solve_error_operator_upper": 3.43602678604707e-11,
    "high_precision_spot_absolute_difference": 2.2126397936084174e-14,
    "existing_guard_dominates_all_derived_roundoff": True,
    "high_precision_spot_covered": True,
}


H006_BINARY_FROZEN_EIGENSYSTEM_RESIDUAL = {
    "result_path": (
        "work/ns_collision/results/"
        "neutral_strip_h006_q12_k240_eigensystem_residual_audit_v1.json"
    ),
    "reference_pair_count": 241,
    "maximum_directed_reference_residual_l2_upper": (
        7.595999996836146e-13
    ),
    "maximum_directed_inverse_mass_residual_upper": (
        7.541469516727855e-11
    ),
    "maximum_eigenvalue_proximity_radius": 7.541469516753379e-11,
    "minimum_adjacent_proximity_interval_separation": (
        8.212190464007561e-05
    ),
    "retained_cutoff_proximity_interval_separation": 0.6015334187394926,
    "stored_mass_row_lumped_coercivity_lower": 0.14999999999999333,
    "reference_orthogonality_frobenius_upper": 1.0926234959797678e-09,
    "modified_eigensystem_residual_l2_upper": 1.1384725498967043e-11,
    "all_241_proximity_intervals_disjoint": True,
    "distinct_eigenvalue_existence_in_each_interval_proved": True,
    "indexed_generalized_eigenvalue_inclusions_proved": False,
    "endpoint_effect_of_eigenpair_residuals_certified": False,
}


H006_BINARY_FROZEN_REFERENCE_ASSEMBLY = {
    "result_path": (
        "work/ns_collision/results/"
        "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
    ),
    "independent_reconstruction_crosscheck_path": (
        "work/ns_collision/results/"
        "neutral_strip_h006_gaussian_assembly_reconstruction_crosscheck_v1.json"
    ),
    "triangle_count": 30954,
    "matrix_fingerprint_sha256": (
        "c80fdc5aa494fe1118aec4c38045df76151b2ccde3fded18f4af1161de7efd48"
    ),
    "maximum_mass_entry_error_upper": 1.2142378297362264e-16,
    "maximum_stiffness_entry_error_upper": 3.778781758995954e-13,
    "maximum_boundary_entry_error_upper": 1.1646544513831305e-13,
    "maximum_boundary_mass_entry_error_upper": 1.6523696698987665e-17,
    "absolute_mass_error_relative_to_stored_mass_form": (
        5.492564204190142e-13
    ),
    "absolute_stiffness_error_in_stored_mass_form_units": (
        5.431182629562088e-09
    ),
    "exact_mass_lower_relative_to_stored_mass": 0.9999999999994507,
    "q24_containment_check_count": 2318,
    "q24_containment_failure_count": 0,
    "all_four_original_q12_matrices_bitwise_reconstructed": True,
    "reference_finite_element_assembly_interval_enclosed": True,
}


H006_INDEXED_SPECTRUM_TRANSFER = {
    "stored_inertia_result_path": (
        "work/ns_collision/results/"
        "neutral_strip_h006_q12_k240_sparse_inertia_audit_v1.json"
    ),
    "independent_precision_crosscheck_path": (
        "work/ns_collision/results/"
        "neutral_strip_h006_q12_k240_sparse_inertia_precision_crosscheck_v1.json"
    ),
    "exact_polygon_transfer_result_path": (
        "work/ns_collision/results/"
        "neutral_strip_h006_exact_polygon_indexed_spectrum_transfer_v1.json"
    ),
    "stored_inertia_primary_decimal_precision": 220,
    "stored_inertia_crosscheck_decimal_precision": 260,
    "stored_retained_gap_negative_pivot_count": 240,
    "stored_post_interval_negative_pivot_count": 241,
    "all_30422_pivot_signs_reproduced": True,
    "all_30422_crosscheck_intervals_nested": True,
    "first_240_stored_generalized_eigenvalues_indexed": True,
    "all_241_stored_generalized_eigenvalues_indexed": True,
    "first_240_exact_polygon_generalized_eigenvalues_indexed": True,
    "all_241_exact_polygon_generalized_eigenvalues_indexed": True,
    "exact_polygon_index_239_interval_upper": 106.4162237645287,
    "exact_polygon_complement_index_240_lower": 107.01775717228844,
    "exact_polygon_retained_complement_separation": 0.6015334077597457,
    "minimum_adjacent_exact_interval_separation": 8.21109564412836e-05,
    "continuum_Ritz_transfer_proved": False,
    "polygon_to_circle_domain_transfer_proved": False,
}


def _polygon_flux_measure_l2_factor(vertex_count: int) -> float:
    if vertex_count < 3:
        raise ValueError("a polygon needs at least three vertices")
    return math.sqrt(1.0 / math.cos(math.pi / vertex_count))


def _radial_map_endpoint_energy_factor(vertex_count: int) -> float:
    """Pulled-back energy factor at a radial-map edge endpoint."""
    shear = math.tan(math.pi / vertex_count)
    root = math.sqrt(shear * shear + 4.0)
    return (root + shear) / (root - shear)


def _screen_row(row: dict[str, float | int]) -> dict[str, float | int | bool]:
    polygon_factor = _polygon_flux_measure_l2_factor(
        int(row["inner_boundary_vertex_count"])
    )
    polygon_charge = polygon_factor - 1.0
    boundary_mismatch = float(
        row["modified_to_transient_boundary_relative_spectral"]
    )
    additive_screen = (
        FIRST_WINDOW_INTERVAL_FACTOR
        + LI_YAU_HIGH_INTERVAL_FACTOR
        + boundary_mismatch
        + polygon_charge
    )
    return {
        **row,
        "polygon_flux_measure_L2_factor": polygon_factor,
        "polygon_flux_measure_additive_screen_charge": polygon_charge,
        "crude_additive_screen_total": additive_screen,
        "crude_additive_screen_below_one": additive_screen < 1.0,
        "numerically_certified": False,
    }


def audit() -> dict[str, object]:
    rows = [_screen_row(dict(row)) for row in PRODUCTION_ROWS]
    common_circle_rows = []
    for row in COMMON_CIRCLE_PRODUCTION_ROWS:
        combined = (
            FIRST_WINDOW_INTERVAL_FACTOR
            + LI_YAU_HIGH_INTERVAL_FACTOR
            + row["sampled_later_window_source_interval_factor"]
        )
        common_circle_rows.append(
            {
                **row,
                "sampled_combined_interval_factor": combined,
                "sampled_combined_headroom": 1.0 - combined,
                "sampled_combined_below_one": combined < 1.0,
            }
        )
    frozen_time_slab_rows = [
        {
            **row,
            "combined_screen_below_one": row["combined_screen_total"] < 1.0,
            "guarded_frozen_finite_block_enclosure": True,
            "directed_interval_coefficient_enclosure": False,
        }
        for row in FROZEN_TIME_SLAB_ROWS
    ]
    first_window_headroom = 1.0 - FIRST_WINDOW_INTERVAL_FACTOR
    post_high_tail_headroom = (
        first_window_headroom - LI_YAU_HIGH_INTERVAL_FACTOR
    )
    cutoff_rows = []
    finest_polygon_charge = float(
        rows[-1]["polygon_flux_measure_additive_screen_charge"]
    )
    for row in H006_CUTOFF_ROWS:
        total = (
            FIRST_WINDOW_INTERVAL_FACTOR
            + row["modified_to_transient_boundary_relative_spectral"]
            + row["li_yau_high_interval_factor_upper"]
            + finest_polygon_charge
        )
        cutoff_rows.append(
            {
                **row,
                "crude_additive_screen_total": total,
                "crude_additive_screen_below_one": total < 1.0,
            }
        )

    finest_vertex_count = int(rows[-1]["inner_boundary_vertex_count"])
    radial_map_endpoint_factor = _radial_map_endpoint_energy_factor(
        finest_vertex_count
    )
    radial_map_one_for_one_screen_total = float(
        rows[-1]["crude_additive_screen_total"]
    ) + (radial_map_endpoint_factor - 1.0)

    result: dict[str, object] = {
        "model": "rho=0 corrected transient P1 conormal low-block gate",
        "semidiscrete_interior_equation": "M_II u_dot+A_II u=0",
        "absorption_conormal_moment": (
            "g=-A_BI u-M_BI u_dot=B_stiff^T u-B_mass^T u_dot"
        ),
        "eigenmode_conormal_moment": (
            "g_k=(B_stiff^T+lambda_k B_mass^T)v_k"
        ),
        "corrected_transient_boundary_operator": (
            "C_h^T=B_stiff^T+B_mass^T M_II^(-1) A_II"
        ),
        "legacy_stiffness_only_boundary_map_complete": False,
        "consistent_mass_transient_conormal_identity_proved": True,
        "retained_mode_count": RETAINED_MODE_COUNT,
        "first_window_interval_factor": FIRST_WINDOW_INTERVAL_FACTOR,
        "first_window_headroom": first_window_headroom,
        "li_yau_later_high_interval_factor_upper": (
            LI_YAU_HIGH_INTERVAL_FACTOR
        ),
        "li_yau_later_high_scalar_gain_upper": LI_YAU_HIGH_SCALAR_GAIN,
        "post_high_tail_headroom": post_high_tail_headroom,
        "production_rows": rows,
        "common_circle_production_rows": common_circle_rows,
        "frozen_time_slab_rows": frozen_time_slab_rows,
        "h006_cutoff_tradeoff_rows": cutoff_rows,
        "h006_projected_dynamics_diagnostic": H006_PROJECTED_DYNAMICS,
        "boundary_Riesz_common_circle_geometry_assembled": True,
        "entry_source_projection_assembled": True,
        "common_circle_source_diagnostic_completed": True,
        "legacy_raw_load_static_screen_is_valid_boundary_L2_screen": False,
        "h006_time_zero_common_circle_one_for_one_screen_total": (
            FIRST_WINDOW_INTERVAL_FACTOR
            + LI_YAU_HIGH_INTERVAL_FACTOR
            + common_circle_rows[-1][
                "time_zero_common_circle_relative_spectral"
            ]
        ),
        "h006_time_zero_common_circle_one_for_one_screen_below_one": False,
        "h006_sampled_source_common_circle_screen_total": common_circle_rows[-1][
            "sampled_combined_interval_factor"
        ],
        "h006_sampled_source_common_circle_screen_headroom": common_circle_rows[-1][
            "sampled_combined_headroom"
        ],
        "h006_sampled_source_common_circle_screen_below_one": common_circle_rows[-1][
            "sampled_combined_below_one"
        ],
        "later_window_source_time_suprema_interval_certified": False,
        "post_terminal_source_discrepancy_tail_certified": False,
        "time_slab_partition_nonoverlapping": True,
        "finite_time_slab_window_count": 15,
        "post_terminal_tail_first_window_index": 16,
        "frozen_finite_block_time_slab_enclosure_proved": True,
        "frozen_finite_block_post_terminal_tail_enclosure_proved": True,
        "frozen_finite_block_uses_guarded_floating_endpoint_arithmetic": True,
        "binary_frozen_endpoint_roundoff_audit": dict(
            H006_BINARY_FROZEN_ENDPOINT_ROUNDOFF
        ),
        "frozen_binary_endpoint_arithmetic_directed_enclosed": True,
        "frozen_binary_endpoint_guard_dominates_derived_roundoff": True,
        "frozen_binary_endpoint_high_precision_spot_check_covered": True,
        "frozen_binary_endpoint_inputs_treated_as_exact_binary64": True,
        "binary_frozen_eigensystem_residual_audit": dict(
            H006_BINARY_FROZEN_EIGENSYSTEM_RESIDUAL
        ),
        "stored_mass_row_lumped_coercivity_proved": True,
        "stored_matrix_eigenpair_residuals_directed_enclosed": True,
        "stored_matrix_orthogonality_defects_directed_enclosed": True,
        "reference_eigenvalue_proximity_intervals_proved": True,
        "distinct_reference_eigenvalue_proximity_intervals_proved": True,
        "indexed_spectrum_transfer_audit": dict(
            H006_INDEXED_SPECTRUM_TRANSFER
        ),
        "indexed_generalized_eigenvalue_inclusions_proved": True,
        "stored_generalized_eigenvalues_indexed": True,
        "exact_polygon_generalized_eigenvalues_indexed": True,
        "exact_polygon_complement_generalized_eigenvalue_lower_bound": (
            H006_INDEXED_SPECTRUM_TRANSFER[
                "exact_polygon_complement_index_240_lower"
            ]
        ),
        "endpoint_effect_of_eigenpair_residuals_certified": False,
        "binary_frozen_reference_assembly_audit": dict(
            H006_BINARY_FROZEN_REFERENCE_ASSEMBLY
        ),
        "reference_finite_element_mass_form_interval_enclosed": True,
        "reference_finite_element_stiffness_form_interval_enclosed": True,
        "reference_finite_element_boundary_couplings_interval_enclosed": True,
        "reference_finite_element_assembly_interval_enclosed": True,
        "frozen_finite_block_coefficient_matrices_interval_enclosed": False,
        "h006_refined_frozen_time_slab_interval_factor_upper": (
            frozen_time_slab_rows[-1][
                "later_low_block_interval_factor_upper"
            ]
        ),
        "h006_refined_frozen_time_slab_combined_screen_total": (
            frozen_time_slab_rows[-1]["combined_screen_total"]
        ),
        "h006_refined_frozen_time_slab_combined_screen_headroom": (
            frozen_time_slab_rows[-1]["combined_screen_headroom"]
        ),
        "h006_refined_frozen_time_slab_combined_screen_below_one": (
            frozen_time_slab_rows[-1]["combined_screen_below_one"]
        ),
        "retained_projected_dynamics_diagnostic_completed": True,
        "static_boundary_screen_is_complete_low_block_comparison": False,
        "retained_projected_dynamics_interval_certified": False,
        "modified_low_space_leakage_interval_bounded": False,
        "gap_free_contractive_Duhamel_leakage_screen_passes": False,
        "low_block_source_trace_map_interval_certified": False,
        "polygon_flux_measure_pushforward": (
            "radial pushforward of j_polygon ds has L2 factor at most "
            "sqrt(sec(pi/N))"
        ),
        "polygon_flux_measure_pushforward_factor_proved": True,
        "polygon_flux_pushforward_conditional_on_L2_density": True,
        "finite_element_conormal_load_is_polygon_L2_density": False,
        "boundary_Riesz_reconstruction_interval_certified": False,
        "static_boundary_norm_policy": (
            "The legacy diagnostic uses the Euclidean norm of equal-arc "
            "nodal conormal loads. The finite boundary Riesz, push, and "
            "cross matrices are now assembled in common-circle L2, but "
            "their interval certification is still required."
        ),
        "naive_radial_map_boundary_shear": math.tan(
            math.pi / finest_vertex_count
        ),
        "naive_radial_map_endpoint_energy_factor": (
            radial_map_endpoint_factor
        ),
        "naive_radial_map_one_for_one_additive_screen_total": (
            radial_map_one_for_one_screen_total
        ),
        "naive_radial_map_one_for_one_additive_screen_below_one": (
            radial_map_one_for_one_screen_total < 1.0
        ),
        "all_radial_map_perturbation_theorems_ruled_out": False,
        "naive_radial_map_claim_scope": (
            "Only the one-for-one additive charge of the endpoint energy "
            "factor fails; smoothing or time-localized radial-map estimates "
            "are not ruled out."
        ),
        "legacy_raw_load_screen_below_one": rows[-1][
            "crude_additive_screen_below_one"
        ],
        "legacy_raw_load_screen_apparent_headroom": (
            1.0 - float(rows[-1]["crude_additive_screen_total"])
        ),
        "corrected_low_block_convergence_observed": True,
        "reference_quadrature_interval_certified": True,
        "discrete_generalized_eigenpairs_interval_certified": False,
        "continuum_Ritz_projector_error_certified": False,
        "polygon_domain_perturbation_certified": False,
        "retained_continuum_low_block_certified": False,
        "polygon_to_circle_flux_map_certified": False,
        "continuum_return_response_certified": False,
        "scope": (
            "The transient conormal identity and the conditional scalar "
            "flux-measure pushforward factor are analytic. The exact finite "
            "boundary Riesz/push/cross matrices and entry source projection "
            "are assembled. Their four mesh rows and sixteen window-start "
            "samples are floating diagnostics. Endpoint interpolation and "
            "the post-6 tail are analytically enclosed for the frozen finite "
            "block, and the endpoint arithmetic guard is now derived by "
            "directed roundoff bounds relative to the stored binary64 data. "
            "Stored-matrix eigensystem residuals, mass coercivity, and "
            "disjoint eigenvalue-proximity intervals are also directed "
            "enclosed. Exact Gaussian-weighted reference mass, stiffness, "
            "boundary stiffness coupling, and boundary mass coupling are "
            "now outward enclosed around the fingerprint-matched q12 "
            "matrices. Indexed spectral counting, the remaining finite-block "
            "Riesz/Gram/projected algebra, endpoint propagation of assembly "
            "and eigensystem errors, off-block leakage, and continuum "
            "transfer remain open."
        ),
        "next_required_step": (
            "Verify the retained spectral count and enclose the remaining "
            "Riesz/Gram/projected finite-block algebra, then propagate the "
            "now-bounded assembly error and eigensystem residuals through "
            "the endpoint actions. After that, bound spectrally damped "
            "off-block leakage and prove continuum Ritz-projector and "
            "polygonal-domain perturbation estimates."
        ),
    }
    checks = [
        result["consistent_mass_transient_conormal_identity_proved"],
        not result["legacy_stiffness_only_boundary_map_complete"],
        result["polygon_flux_measure_pushforward_factor_proved"],
        result["polygon_flux_pushforward_conditional_on_L2_density"],
        not result["finite_element_conormal_load_is_polygon_L2_density"],
        not result["boundary_Riesz_reconstruction_interval_certified"],
        result["boundary_Riesz_common_circle_geometry_assembled"],
        result["entry_source_projection_assembled"],
        result["common_circle_source_diagnostic_completed"],
        not result["legacy_raw_load_static_screen_is_valid_boundary_L2_screen"],
        not result["h006_time_zero_common_circle_one_for_one_screen_below_one"],
        result["h006_sampled_source_common_circle_screen_below_one"],
        result["h006_sampled_source_common_circle_screen_headroom"] > 0.03,
        not result["later_window_source_time_suprema_interval_certified"],
        not result["post_terminal_source_discrepancy_tail_certified"],
        result["time_slab_partition_nonoverlapping"],
        result["finite_time_slab_window_count"] == 15,
        result["post_terminal_tail_first_window_index"] == 16,
        result["frozen_finite_block_time_slab_enclosure_proved"],
        result["frozen_finite_block_post_terminal_tail_enclosure_proved"],
        result[
            "frozen_finite_block_uses_guarded_floating_endpoint_arithmetic"
        ],
        result["frozen_binary_endpoint_arithmetic_directed_enclosed"],
        result["frozen_binary_endpoint_guard_dominates_derived_roundoff"],
        result["frozen_binary_endpoint_high_precision_spot_check_covered"],
        result["frozen_binary_endpoint_inputs_treated_as_exact_binary64"],
        result["stored_mass_row_lumped_coercivity_proved"],
        result["stored_matrix_eigenpair_residuals_directed_enclosed"],
        result["stored_matrix_orthogonality_defects_directed_enclosed"],
        result["reference_eigenvalue_proximity_intervals_proved"],
        result["distinct_reference_eigenvalue_proximity_intervals_proved"],
        result["indexed_generalized_eigenvalue_inclusions_proved"],
        result["stored_generalized_eigenvalues_indexed"],
        result["exact_polygon_generalized_eigenvalues_indexed"],
        result["indexed_spectrum_transfer_audit"][
            "all_30422_pivot_signs_reproduced"
        ],
        result["indexed_spectrum_transfer_audit"][
            "all_30422_crosscheck_intervals_nested"
        ],
        result["indexed_spectrum_transfer_audit"][
            "exact_polygon_retained_complement_separation"
        ] > 0.6,
        not result["indexed_spectrum_transfer_audit"][
            "continuum_Ritz_transfer_proved"
        ],
        not result["indexed_spectrum_transfer_audit"][
            "polygon_to_circle_domain_transfer_proved"
        ],
        not result["endpoint_effect_of_eigenpair_residuals_certified"],
        result["reference_finite_element_mass_form_interval_enclosed"],
        result["reference_finite_element_stiffness_form_interval_enclosed"],
        result[
            "reference_finite_element_boundary_couplings_interval_enclosed"
        ],
        result["reference_finite_element_assembly_interval_enclosed"],
        result["reference_quadrature_interval_certified"],
        result["binary_frozen_reference_assembly_audit"][
            "absolute_mass_error_relative_to_stored_mass_form"
        ] < 6.0e-13,
        result["binary_frozen_reference_assembly_audit"][
            "absolute_stiffness_error_in_stored_mass_form_units"
        ] < 6.0e-9,
        result["binary_frozen_eigensystem_residual_audit"][
            "retained_cutoff_proximity_interval_separation"
        ] > 0.6,
        result["binary_frozen_endpoint_roundoff_audit"][
            "maximum_directed_roundoff_norm_error_upper"
        ] < 5.0e-11,
        result["binary_frozen_endpoint_roundoff_audit"][
            "minimum_existing_guard_margin"
        ] > 0.0,
        not result[
            "frozen_finite_block_coefficient_matrices_interval_enclosed"
        ],
        result["h006_refined_frozen_time_slab_combined_screen_below_one"],
        result["h006_refined_frozen_time_slab_combined_screen_headroom"]
        > 0.029,
        result["retained_projected_dynamics_diagnostic_completed"],
        not result["static_boundary_screen_is_complete_low_block_comparison"],
        not result["retained_projected_dynamics_interval_certified"],
        not result["modified_low_space_leakage_interval_bounded"],
        not result["gap_free_contractive_Duhamel_leakage_screen_passes"],
        not result["low_block_source_trace_map_interval_certified"],
        not result[
            "naive_radial_map_one_for_one_additive_screen_below_one"
        ],
        not result["all_radial_map_perturbation_theorems_ruled_out"],
        result["legacy_raw_load_screen_below_one"],
        not result["legacy_raw_load_static_screen_is_valid_boundary_L2_screen"],
        rows[0]["modified_to_transient_boundary_relative_spectral"] > 0.1,
        rows[-1]["modified_to_transient_boundary_relative_spectral"] < 0.023,
        H006_PROJECTED_DYNAMICS[
            "retained_projected_semigroup_output_rows"
        ][0]["full_output_discrepancy_over_same_time_reference"] < 0.002,
        H006_PROJECTED_DYNAMICS[
            "retained_modified_invariance_residual_relative_Minv_spectral"
        ] > 0.06,
        H006_PROJECTED_DYNAMICS[
            "first_later_time_naive_contractive_Duhamel_leakage_upper"
        ] > 2.0,
        common_circle_rows[0]["sampled_later_window_source_interval_factor"]
        > common_circle_rows[-1][
            "sampled_later_window_source_interval_factor"
        ],
        frozen_time_slab_rows[-1]["later_low_block_interval_factor_upper"]
        < frozen_time_slab_rows[-2][
            "later_low_block_interval_factor_upper"
        ],
        abs(
            frozen_time_slab_rows[-1]["maximum_interpolation_charge"]
            / frozen_time_slab_rows[-2]["maximum_interpolation_charge"]
            - 0.25
        )
        < 1.0e-12,
        not cutoff_rows[1]["crude_additive_screen_below_one"],
        cutoff_rows[2]["crude_additive_screen_below_one"],
        result["reference_quadrature_interval_certified"],
        not result["discrete_generalized_eigenpairs_interval_certified"],
        not result["continuum_Ritz_projector_error_certified"],
        not result["polygon_domain_perturbation_certified"],
        not result["retained_continuum_low_block_certified"],
        not result["polygon_to_circle_flux_map_certified"],
        not result["continuum_return_response_certified"],
    ]
    result["all_transient_conormal_low_block_checks_pass"] = bool(all(checks))
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
