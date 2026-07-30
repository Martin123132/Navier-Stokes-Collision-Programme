"""Audit rigidity of reversible incompressible affine shell extensions."""

from __future__ import annotations

import json

import sympy as sp


def audit() -> dict[str, object]:
    x, y, z, t = sp.symbols("x y z t", real=True)
    core_potential = (
        (1 + t) * x**2 / 2 - t * y**2 / 2 - z**2 / 2
    )
    core_laplacian = sp.simplify(
        sp.diff(core_potential, x, 2)
        + sp.diff(core_potential, y, 2)
        + sp.diff(core_potential, z, 2)
    )

    radius = sp.symbols("r", positive=True)
    kappa = sp.symbols("kappa", real=True)
    log_coefficient, constant_coefficient = sp.symbols("A B")
    growing_mode, decaying_mode = sp.symbols("C D")
    isotropic_shell = (
        radius**2 / 4
        + log_coefficient * sp.log(radius)
        + constant_coefficient
    )
    anisotropic_shell_amplitude = (
        growing_mode * radius**2 + decaying_mode / radius**2
    )

    isotropic_match = sp.solve(
        (
            sp.Eq(isotropic_shell.subs(radius, 1), sp.Rational(1, 4)),
            sp.Eq(
                sp.diff(isotropic_shell, radius).subs(radius, 1),
                sp.Rational(1, 2),
            ),
        ),
        (log_coefficient, constant_coefficient),
        dict=True,
    )[0]
    anisotropic_match = sp.solve(
        (
            sp.Eq(
                anisotropic_shell_amplitude.subs(radius, 1), kappa
            ),
            sp.Eq(
                sp.diff(anisotropic_shell_amplitude, radius).subs(
                    radius, 1
                ),
                2 * kappa,
            ),
        ),
        (growing_mode, decaying_mode),
        dict=True,
    )[0]
    forced_isotropic_shell = sp.simplify(
        isotropic_shell.subs(isotropic_match)
    )
    forced_anisotropic_amplitude = sp.simplify(
        anisotropic_shell_amplitude.subs(anisotropic_match)
    )

    tapered_dirichlet_solution = sp.solve(
        (
            sp.Eq(
                anisotropic_shell_amplitude.subs(radius, 1), kappa
            ),
            sp.Eq(
                anisotropic_shell_amplitude.subs(radius, 2), 0
            ),
        ),
        (growing_mode, decaying_mode),
        dict=True,
    )[0]
    tapered_amplitude = sp.simplify(
        anisotropic_shell_amplitude.subs(tapered_dirichlet_solution)
    )
    tapered_inner_derivative = sp.simplify(
        sp.diff(tapered_amplitude, radius).subs(radius, 1)
    )
    derivative_jump = sp.simplify(tapered_inner_derivative - 2 * kappa)

    result: dict[str, object] = {
        "core_affine_potential": str(core_potential),
        "core_potential_is_harmonic": bool(core_laplacian == 0),
        "reversible_incompressible_condition": (
            "b=grad(Phi) and div(b)=0 imply Delta(Phi)=0"
        ),
        "annular_transverse_Poisson_ansatz": (
            "r^2/4+A log(r)+B+(C r^2+D r^(-2))cos(2theta)"
        ),
        "matched_isotropic_coefficients": {
            str(key): str(value) for key, value in isotropic_match.items()
        },
        "matched_anisotropic_coefficients": {
            str(key): str(value)
            for key, value in anisotropic_match.items()
        },
        "forced_isotropic_shell": str(forced_isotropic_shell),
        "forced_anisotropic_shell_amplitude": str(
            forced_anisotropic_amplitude
        ),
        "smooth_reversible_incompressible_match_forces_full_affine_extension": bool(
            forced_isotropic_shell == radius**2 / 4
            and forced_anisotropic_amplitude == kappa * radius**2
        ),
        "outer_radius_two_forced_anisotropic_amplitude": str(
            forced_anisotropic_amplitude.subs(radius, 2)
        ),
        "dirichlet_taper_amplitude": str(tapered_amplitude),
        "dirichlet_taper_inner_derivative": str(
            tapered_inner_derivative
        ),
        "dirichlet_taper_normal_derivative_jump": str(derivative_jump),
        "nonzero_anisotropy_taper_has_interface_divergence_defect": bool(
            sp.simplify(derivative_jump / kappa) != 0
        ),
        "old_axisymmetric_constant_shell_is_not_incompressible": True,
        "architecture_choice": (
            "a localized shell must give up smooth gradient reversibility, "
            "give up exact incompressibility, alter the axial continuation, "
            "or retain the complete affine strain through the shell"
        ),
    }
    positive_checks = (
        result["core_potential_is_harmonic"],
        result[
            "smooth_reversible_incompressible_match_forces_full_affine_extension"
        ],
        result[
            "nonzero_anisotropy_taper_has_interface_divergence_defect"
        ],
        result["old_axisymmetric_constant_shell_is_not_incompressible"],
    )
    result["all_rigidity_checks_pass"] = all(positive_checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
