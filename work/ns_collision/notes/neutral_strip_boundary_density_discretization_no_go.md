# Neutral-strip boundary-density discretization no-go

## Purpose

The first return-density pilot represented the fitted-circle killing flux in
32 equal angular bins. Mesh refinement looked stable, but that comparison
held the number of boundary bins fixed. This note checks whether refining
the angular representation at fixed interior mesh can produce the required
boundary `L2` density.

It cannot. This is a no-go for the current discretization, not for the
continuum return-density strategy.

## The discrete boundary law is atomic

For a fixed Shortley-Weller grid, every killed coordinate edge has one hit
angle `theta_e`. At time `t` the forward chain therefore produces the
boundary measure

```text
mu_h(t)=sum_e m_e(t) delta_(theta_e),
m_e(t)=q_e p_(i(e))(t).                              (1)
```

Let `P_B mu_h` spread the mass in each of `B` equal bins uniformly over that
bin. Once the bins are narrow enough to separate all nonzero atoms,

```text
||P_B mu_h(t)||_2^2
  =B/(2 pi) sum_e m_e(t)^2.                          (2)
```

Consequently the histogram norm grows like `B^(1/2)`. The interval trace
factor is linear in that norm, and the response is its square root, so

```text
K_R,h,B proportional to B^(1/4).                    (3)
```

There is no finite fixed-mesh boundary-`L2` limit. Refining bins alone
resolves atoms rather than a continuum density.

## Numerical witness

The audit uses exactly one mesh-40 semigroup trajectory and changes only the
angular projection. It gives

```text
bins       interval factor       maximum response
  16        0.938012769061        0.615705137700
  32        0.967477581273        0.625300617306
  64        1.003633251954        0.636877525029
 128        1.419351756561        0.757379284149
 256        1.969134746552        0.892084760035
 512        2.702978731860        1.045176693009
1024        3.822589181403        1.242931559762.     (4)
```

The 1024-bin response is `1.987734419834` times the 32-bin value. The
scalar semigroup/resolvent recovery error is only `0.001057516652`, and the
terminal state is below `2.3e-15`; neither time integration nor the tail
causes the growth in (4).

## No canonical face length in the present chain

The fitted-edge generator is a monotone backward Shortley-Weller scheme. It
is not a conservative cut-cell generator with disjoint physical boundary
faces. Kolmogorov's detailed-balance cycle test makes this concrete. The
maximum absolute log cycle defects on meshes 30, 40, 50, and 60 are

```text
0.310868380940,
0.301937403536,
0.202471580772,
0.334378751459.                                     (5)
```

Thus there is no exact reversible node-volume measure from which physical
boundary-face weights can be recovered. The failure is localized to grid
cycles affected by the unequal fitted-circle stencil. It does not negate
the scheme's usefulness for scalar hitting probabilities.

## Consequences

The earlier `K_R` and `K_S` values remain finite-state diagnostics only.
They are not continuum upper bounds, so their conditional critical-norm
thresholds cannot be used as certified branch budgets. The same issue
propagates into `K_S` because its child-return factor uses the same
fitted-edge histogram.

The replacement must have all of the following:

1. a conservative boundary-fitted finite-volume or lumped-FEM generator;
2. exact discrete reversibility for the gradient affine drift;
3. disjoint inner-boundary faces with physical arclengths;
4. coupled interior and boundary refinement;
5. independent time-window and tail enclosures.

Only after that replacement stabilizes may the return and composite
responses be promoted beyond finite-state diagnostics. The calculation is
reproduced by
`scripts/neutral_strip_boundary_density_discretization_audit.py`.

**Later development.** The required positive reversible replacement is now
implemented and has a stable three-mesh physical-face pilot. It does not yet
provide a continuum upper bound. See
`neutral_strip_reversible_boundary_fem.md`.
