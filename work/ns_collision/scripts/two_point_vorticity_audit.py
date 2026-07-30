"""Symbolic audit for the two-point vorticity collision system."""

from __future__ import annotations

import json

import sympy as sp


def audit() -> dict[str, str | bool]:
    # Tensor-product stretching identity.
    ax = sp.Matrix(3, 3, sp.symbols("ax0:9"))
    ay = sp.Matrix(3, 3, sp.symbols("ay0:9"))
    wx = sp.Matrix(3, 1, sp.symbols("wx0:3"))
    wy = sp.Matrix(3, 1, sp.symbols("wy0:3"))
    omega = wx * wy.T
    tensor_stretch_residual = sp.simplify(
        ax * omega + omega * ay.T
        - ((ax * wx) * wy.T + wx * (ay * wy).T)
    )
    cross_matrix = sp.Matrix(3, 3, sp.symbols("cross0:9"))
    cross_derivative = ax.T * cross_matrix + cross_matrix * ay
    common_endpoint = sp.simplify(
        cross_derivative.subs(
            {
                **{cross_matrix[i, j]: int(i == j) for i in range(3) for j in range(3)},
                **{ay[i, j]: ax[i, j] for i in range(3) for j in range(3)},
            }
        )
        - (ax.T + ax)
    )

    # Centre/separation second-order coefficients in one coordinate.
    # d_x=(1/2)d_c+d_r and d_y=(1/2)d_c-d_r.
    centre_second_order = sp.Rational(1, 2)
    relative_second_order = sp.Integer(2)
    mixed_second_order = sp.Integer(0)

    # Representative strain kernel with fixed vorticity vector e_3.
    x, y, z, scale = sp.symbols("x y z scale", real=True)
    radius_sq = x**2 + y**2 + z**2
    radius = sp.sqrt(radius_sq)
    position = sp.Matrix([x, y, z])
    fixed_vorticity = sp.Matrix([0, 0, 1])
    cross = position.cross(fixed_vorticity)
    numerator = cross * position.T + position * cross.T
    kernel = sp.simplify(numerator / radius**5)
    laplacian_kernel = kernel.applyfunc(
        lambda value: sp.simplify(
            sp.diff(value, x, 2)
            + sp.diff(value, y, 2)
            + sp.diff(value, z, 2)
        )
    )
    euler_homogeneity = kernel.applyfunc(
        lambda value: sp.simplify(
            x * sp.diff(value, x)
            + y * sp.diff(value, y)
            + z * sp.diff(value, z)
            + 3 * value
        )
    )
    numerator_laplacian = numerator.applyfunc(
        lambda value: sp.simplify(
            sp.diff(value, x, 2)
            + sp.diff(value, y, 2)
            + sp.diff(value, z, 2)
        )
    )

    g = sp.symbols("g", positive=True)
    radial_power = g ** (-3)
    radial_laplacian_part = sp.simplify(
        sp.diff(radial_power, g, 2) + 2 * sp.diff(radial_power, g) / g
    )
    angular_laplacian_part = -6 * g ** (-5)

    result: dict[str, str | bool] = {
        "tensor_stretch_product_rule": tensor_stretch_residual == sp.zeros(3),
        "cross_deformation_common_endpoint_is_twice_strain": common_endpoint
        == sp.zeros(3),
        "centre_laplacian_coefficient": str(centre_second_order),
        "relative_laplacian_coefficient": str(relative_second_order),
        "mixed_laplacian_coefficient": str(mixed_second_order),
        "strain_kernel_symmetric": sp.simplify(kernel - kernel.T) == sp.zeros(3),
        "strain_kernel_trace_free": sp.simplify(sp.trace(kernel)) == 0,
        "strain_kernel_harmonic_off_origin": laplacian_kernel == sp.zeros(3),
        "strain_kernel_degree_minus_three": euler_homogeneity == sp.zeros(3),
        "angular_numerators_are_harmonic_quadratics": numerator_laplacian
        == sp.zeros(3),
        "radial_laplacian_part": str(radial_laplacian_part),
        "angular_laplacian_part": str(angular_laplacian_part),
        "radial_angular_cancellation": sp.simplify(
            radial_laplacian_part + angular_laplacian_part
        )
        == 0,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
