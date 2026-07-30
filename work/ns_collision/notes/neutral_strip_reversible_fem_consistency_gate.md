# Reversible FEM consistency gate

## Purpose

The finite-time certificate rigorously encloses the response of each stored
reversible matrix. The side-exit correction then pays paths omitted at the
artificial `x` boundary. Neither result proves that the matrix is an upper
approximation to the continuum killed diffusion.

This stage isolates the remaining discretization question. It independently
assembles the weighted conforming P1 forms on the same polygon, quantifies the
circle-polygon geometry, and tests whether one uniform form perturbation can
carry the finite-matrix certificate to the continuum.

## Forms being compared

On the polygonal domain, with `mu=exp(-x^2/2)`, the reference P1 forms are

```text
m_h(u,v)=int mu u v,
a_h(u,v)=int mu grad(u).grad(v).                    (1)
```

The positive reversible chain instead uses centroid-lumped mass and
midpoint-weighted Delaunay conductances:

```text
m_tilde(u,v)=sum_i m_i u_i v_i,
a_tilde(u,v)=sum_ij w_ij mu(mid_ij)(u_i-u_j)(v_i-v_j).  (2)
```

For a consistent-mass transient reference, the absorption conormal moment
also contains the boundary mass cross-block:

```text
g=-A_BI u-M_BI u_dot.                               (2a)
```

Thus an eigenmode uses `B_stiff+lambda B_mass`, not `B_stiff` alone. The
original stiffness-only boundary comparison is retained as a diagnostic but
is not the complete transient map.

The audit assembles (1) independently with tensor Gauss quadrature under a
Duffy map. Orders `q` and `q+4` are compared for every mass, stiffness, and
inner-boundary coupling matrix. This is a strong numerical cross-check, not
interval quadrature.

## Exact polygon geometry

For the regular inscribed `N`-gon, elementary geometry gives

```text
sagitta              =1-cos(pi/N),
chord / true arc     =sin(pi/N)/(pi/N),
normal-angle mismatch<=pi/N.                       (3)
```

At the production `h=0.12` mesh, `N=56`, so

```text
sagitta                         0.001573184982,
perimeter deficit fraction      0.000524450013,
maximum normal mismatch         0.056099868815 radians. (4)
```

The small sagitta controls position and area errors. The larger `O(h)`
normal mismatch matters directly to a flux theorem and cannot be replaced by
the `O(h^2)` sagitta.

## Whole-spectrum obstruction

The production `h=0.12` diagnostics give

```text
modified/reference stiffness form     0.932758 .. 1.050694,
modified/reference mass form          1.003586 .. 3.841141,
inner coupling Frobenius difference   0.004779.       (5)
```

The mass ratio near four is not a failed convergence test. For P1 elements,
consistent and vertex-lumped mass differ by an order-one factor on the
highest grid modes even as `h` tends to zero. Consequently a global estimate
of the form

```text
(1-epsilon)m_h <= m_tilde <= (1+epsilon)m_h          (6)
```

cannot have `epsilon` tending to zero. A whole-spectrum operator perturbation
would therefore spend several times the available response margin and is not
a valid route to a continuum certificate.

## Low modes

Parabolic smoothing changes the picture. On the first twenty generalized
eigenmodes of `(a_h,m_h)` at `h=0.12`, the same diagnostics give

```text
modified/reference mass               1.004179 .. 1.017130,
modified/reference stiffness          0.995857 .. 1.000369,
inner coupling spectral difference    0.002768.       (7)
```

Including (2a), the mass correction is `0.011948` and the modified-chain to
corrected-reference boundary discrepancy is `0.011473` on these 20 modes.
At cutoff 320 the corresponding `h=0.12` values are `0.138548` and
`0.138135`; the missing term is therefore essential in the production block.

Thus the modified chain is close precisely on the modes that survive after a
short positive time. This identifies the correct proof architecture:

1. enclose quadrature and geometry errors on a finite low spectral subspace;
2. use explicit parabolic decay to pay all higher discrete and continuum
   modes before each certified flux window;
3. separately bound the continuum flux before the split time, exploiting the
   unit distance between the entry circle and absorbing circle;
4. include the normal-angle and true-arc transformation in the boundary
   operator.

## Status

This stage proves the geometric formulas, identifies the complete transient
conormal map, and rules out the naive global-form argument. The matrix
comparisons in (5) and (7) remain high-order numerical diagnostics. They do
not yet enclose quadrature, continuum spectral projection, or domain
perturbation, so the continuum return response is still uncertified. The
common-circle boundary Riesz correction and source-aware refinement screen
are developed in
`neutral_strip_transient_conormal_low_block_gate.md`.

The audit is reproduced by
`scripts/neutral_strip_reversible_fem_consistency_gate.py`.
