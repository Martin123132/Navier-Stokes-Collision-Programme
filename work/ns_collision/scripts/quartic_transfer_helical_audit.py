"""Exact helical-channel audit for the two-mode quartic transfer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import sympy as sp


COUNTEREXAMPLE_SCRIPT = Path(__file__).with_name(
    "quartic_transfer_counterexample.py"
)
SPEC = importlib.util.spec_from_file_location(
    "quartic_transfer_counterexample_helpers", COUNTEREXAMPLE_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def _transfer(field: HELPERS.Field, x: sp.Symbol) -> sp.Expr:
    euler = {
        wave: -value for wave, value in HELPERS._nonlinearity(field).items()
    }
    return sp.factor(
        HELPERS._trilinear(euler, field, field, x, primitive=True)
        + HELPERS._trilinear(field, euler, field, x, primitive=True)
        + HELPERS._trilinear(field, field, euler, x, primitive=True),
        extension=sp.sqrt(2),
    )


def audit() -> dict[str, str | bool]:
    x = sp.symbols("x", positive=True)
    root_two = sp.sqrt(2)
    k = (1, 0, 0)
    m = (1, 1, 0)
    k_first = sp.Matrix([0, 1, 0])
    k_second = sp.Matrix([0, 0, 1])
    m_first = sp.Matrix([-1 / root_two, 1 / root_two, 0])
    m_second = sp.Matrix([0, 0, 1])

    transfers: dict[tuple[int, int], sp.Expr] = {}
    helicity_checks = []
    for sigma in (-1, 1):
        for tau in (-1, 1):
            k_value = (k_first + sp.I * sigma * k_second) / root_two
            m_value = (m_first + sp.I * tau * m_second) / root_two
            field = {
                k: k_value,
                tuple(-entry for entry in k): sp.conjugate(k_value),
                m: m_value,
                tuple(-entry for entry in m): sp.conjugate(m_value),
            }
            transfers[(sigma, tau)] = _transfer(field, x)
            helicity_checks.extend(
                [
                    sp.simplify(
                        sp.I * sp.Matrix(k).cross(k_value)
                        - sigma * k_value
                    )
                    == sp.zeros(3, 1),
                    sp.simplify(
                        sp.I * sp.Matrix(m).cross(m_value)
                        - tau * root_two * m_value
                    )
                    == sp.zeros(3, 1),
                ]
            )

    p = x**3 + 2 * x**2 + 3 * x
    homochiral_threshold = (25 * root_two + 19) / 14
    homochiral_expected = sp.factor(
        (sp.Rational(1, 8) - root_two / 10)
        * (1 - x) ** 2
        * (p - homochiral_threshold),
        extension=root_two,
    )
    heterochiral_expected = sp.factor(
        (sp.Rational(1, 8) + root_two / 10)
        * (1 - x) ** 2
        * (p + (25 * root_two - 19) / 14),
        extension=root_two,
    )
    threshold_x = sp.nsolve(
        p - homochiral_threshold, sp.Rational(3, 4), prec=50
    )
    threshold_scale = -sp.log(threshold_x)

    scale_half_x = sp.exp(-sp.Rational(1, 2))
    homochiral_at_half = sp.N(
        homochiral_expected.subs(x, scale_half_x), 30
    )
    heterochiral_at_half = sp.N(
        heterochiral_expected.subs(x, scale_half_x), 30
    )
    mixed_at_half = sp.N(
        ((1 - x) ** 2 * (p - 11) / 20).subs(x, scale_half_x), 30
    )

    result: dict[str, str | bool] = {
        "helical_eigenvector_equations_verified": all(helicity_checks),
        "homochiral_minus_transfer": str(transfers[(-1, -1)]),
        "homochiral_plus_transfer": str(transfers[(1, 1)]),
        "homochiral_channels_agree": bool(
            sp.simplify(transfers[(-1, -1)] - transfers[(1, 1)]) == 0
        ),
        "homochiral_closed_form_verified": bool(
            sp.simplify(transfers[(1, 1)] - homochiral_expected) == 0
        ),
        "heterochiral_minus_plus_transfer": str(transfers[(-1, 1)]),
        "heterochiral_plus_minus_transfer": str(transfers[(1, -1)]),
        "heterochiral_channels_agree": bool(
            sp.simplify(transfers[(-1, 1)] - transfers[(1, -1)]) == 0
        ),
        "heterochiral_closed_form_verified": bool(
            sp.simplify(transfers[(1, -1)] - heterochiral_expected) == 0
        ),
        "heterochiral_is_positive_at_every_positive_scale": bool(
            25 * root_two - 19 > 0
            and sp.Rational(1, 8) + root_two / 10 > 0
        ),
        "homochiral_threshold_p": str(homochiral_threshold),
        "homochiral_threshold_x": str(sp.N(threshold_x, 30)),
        "homochiral_threshold_scale": str(sp.N(threshold_scale, 30)),
        "homochiral_changes_sign_once": bool(
            0 < homochiral_threshold < 6
            and sp.discriminant(sp.diff(p, x), x) < 0
            and sp.Rational(1, 8) - root_two / 10 < 0
        ),
        "homochiral_transfer_at_scale_half": str(homochiral_at_half),
        "heterochiral_transfer_at_scale_half": str(heterochiral_at_half),
        "mixed_transfer_at_scale_half": str(mixed_at_half),
        "negative_mixed_field_requires_channel_interference": bool(
            homochiral_at_half > 0
            and heterochiral_at_half > 0
            and mixed_at_half < 0
        ),
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
