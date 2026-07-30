# Two-norm renewal within one dyadic generation

## Purpose

The true radius-halving contraction occurs once when the intrinsic scale moves
to the next dyadic generation. A history may make many buffered exits and
returns while the scale is unchanged. Assigning the true-split factor to every
ordinary return would be incorrect.

This note gives the correct two-level algebra: first sum repeated same-scale
visits, then apply one true generation transition.

## One same-scale visit

Let

```text
C_pair=exp(R_*/2)
```

be the conservative physical/gauge pair condition number. Write `D>=0` for
the net favorable action accumulated during one complete gauged buffered
visit: killed spectral margin plus favorable geometry, minus coherence and
shell errors. The physical visit norm is bounded by

```text
V=C_pair exp(-D).                                        (1)
```

Let the physical exterior pair-return norm satisfy

```text
r<=eta^(-2 beta).                                        (2)
```

This requires the moving return supersolution, including its deformation
term; (2) is not supplied by probability alone when exterior stretching is
present.

## Same-generation renewal

All repeated visits at this fixed scale sum to

```text
B=V[1+rV+(rV)^2+...]=V/(1-rV),                           (3)
```

provided `rV<1`. No true-split contraction appears inside (3).

After the same-scale block is complete, one genuine radius halving contributes

```text
gamma_pair=(1/4)exp(R_* d/24).                           (4)
```

The complete generation factor is

```text
G=gamma_pair V/(1-rV).                                   (5)
```

Elementary rearrangement shows that `G<1` is equivalent to

```text
D>log[C_pair(gamma_pair+r)].                             (6)
```

Condition (6) automatically implies the convergence condition for (3).
Additional visit, pressure, or return errors enter additively in this
logarithmic budget.

## Parameter audit

In three dimensions, take the strongest pure-Brownian return exponent
`beta=1` and buffer ratio `eta=2`, so `r=1/4`.

At `R_*=1`,

```text
C_pair approximately 1.64872,
gamma_pair approximately 0.283287,
r C_pair approximately 0.41218,
G(D=0) approximately 0.7946.
```

Thus the ideal algebra closes even before a positive visit action is used.

At `R_*=2`,

```text
C_pair approximately 2.71828,
gamma_pair approximately 0.321006,
r C_pair approximately 0.67957,
G(D=0) approximately 2.72.
```

It requires positive visit action

```text
D>log[C_pair(gamma_pair+1/4)] approximately 0.439.
```

Solving the zero-action threshold for `beta=1`, `eta=2` gives an `R_*`
strictly between one and `1.5`. Within this placeholder algebra, this makes
`R_*=1` look materially better than the spectral edge `R_*=2`.

## Capacity tradeoff

The choice `beta=1` maximizes geometric three-dimensional return contraction
but has

```text
beta(1-beta)=0.
```

It cannot pay any static positive exterior deformation through Brownian
capacity. The moving shrink term `A'/(2A)` or favorable radial drift must pay
that deformation. Choosing `beta=1/2` supplies the maximal static budget
`1/4`, but weakens pair return from `eta^(-2)` to `eta^(-1)` and increases the
visit action required by (6).

## Refined live gate

The combinatorial tree and closed interface transfer are now harmless in the
physical norm. The next analytic obligation is the complete gauged visit
operator in (1). A pointwise spectral rate is not by itself a uniform lower
bound on `D`, because a diffusion can make a very short visit. One needs
either:

1. a direct elliptic/Feynman-Kac bound for the inner-to-outer buffered visit;
2. a joint exit-time Laplace transform retaining the killed spectral margin;
3. a grouping rule that combines short visits before applying (6).

That calculation must include non-affine coherence errors. It is now the
nearest quantitative gate, while exterior Navier-Stokes geometry remains the
source of assumption (2).

The renewal algebra, parameter sweep, and zero-action Reynolds threshold are
reproduced by `scripts/two_norm_generation_cycle_audit.py`.

The complete visit is solved in `buffered_visit_feynman_kac.md`. Its exact
core-plus-shell gain is much larger than the placeholder condition number and
shows that `R_*=1`, `eta=2` does not close in the recurrent transverse model.
That later benchmark supersedes the provisional parameter recommendation
above without changing the two-level operator algebra.
