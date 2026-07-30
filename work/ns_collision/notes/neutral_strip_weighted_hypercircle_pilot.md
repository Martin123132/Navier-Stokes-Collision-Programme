# Weighted hypercircle pilot for the continuum Ritz gate

## Scope

This stage asks whether the stored `h=0.06` polygon mesh has enough
quantitative room to certify a source-problem energy constant `C_h` with

`||T f-T_h f||_a <= C_h ||f||_m`

and, by weighted Aubin-Nitsche duality, the Ritz projection inequality

`||u-R_h u||_m <= C_h ||u-R_h u||_a`,

with `C_h < 0.08557115750643675`.

Here

`a(u,v)=integral mu grad(u).grad(v)`, `m(f,v)=integral mu f v`,
and `mu(x)=exp(-x^2/2)`.

The calculation is a floating feasibility pilot. It does not certify `C_h`.

## Correct source decomposition

Write the physical source as `g=mu f`, so
`||f||_m=||g||_(mu^-1)`. Let `g_0` be the ordinary, unweighted cell
average of `g` on every triangle.

The mean-zero residual obeys

`|integral (g-g_0)v| <= C_data ||g||_(mu^-1) ||v||_a`,

where the explicit mesh constant is

`C_data = max_T diameter(T)/pi * sqrt(mu_max,T/mu_min,T)`.

The projected physical source has the safe norm bound

`||g_0||_(mu^-1) <= alpha ||g||_(mu^-1)`,

with

`alpha = max_T sqrt(mu_max,T/mu_min,T)`.

For `g_0`, an RT0 flux can satisfy `div(p_h)+g_0=0` exactly. The weighted
Prager-Synge identity gives the computable finite-dimensional constant

`kappa_h = max_(g_0 != 0) min_(v_h,p_h) 
||p_h-mu grad(v_h)||_(mu^-1) / ||g_0||_(mu^-1)`.

Cea best approximation and the two auxiliary source problems therefore give
the safe additive bound

`C_h <= C_data + alpha kappa_h`.

For completeness, if `e=u-R_hu` and `z=T e`, Galerkin orthogonality gives

`||e||_m^2=a(z-R_hz,e)`.

Applying the source-problem energy bound to `z` and cancelling `||e||_m`
proves `||e||_m<=C_h||e||_a`. Thus the same computed constant is the one
required by the continuum cutoff theorem.

The tempting root-sum-square formula with the source space `f_h=c_T/mu` is
not valid here: its `L2(mu)` projection does not reproduce constants, so a
cell-local gradient-only Poincare estimate cannot be applied to it. No result
or flag in this stage uses that rejected shortcut.

The unweighted antecedent is Liu and Oishi, *SIAM Journal on Numerical
Analysis* 51 (2013), Theorems 3.2-3.3 and Section 3.3,
DOI `10.1137/120878446`. The weighted identities above are derived directly
for this operator.

## Discrete pilot

The implementation uses:

- the exact stored 30,954-triangle polygon mesh;
- continuous P1 potentials with zero boundary trace;
- globally oriented RT0 normal-flux degrees of freedom;
- `integral mu^-1 p_h.q_h` for the RT mass;
- exact P0 divergence constraints;
- `integral mu grad(v_h).grad(w_h)` for the P1 stiffness; and
- the diagonal P0 source mass `integral_T mu^-1`.

The matrix-free normalized hypercircle operator is applied with one sparse P1
solve and one sparse mixed RT0 solve. The largest floating eigenvalue is
`0.0005595307772313942`, hence

`kappa_h = 0.023654402914286256`.

The explicit geometry terms are

`C_data = 0.03035401892223142`,

`alpha = 1.1696923667355092`.

Thus the safe floating diagnostic is

`C_data + alpha kappa_h = 0.05802239345075824`,

which is below the strict target by `0.02754876405567851`. Equivalently, the
target permits `kappa_h < 0.04720654776803466`, nearly twice the observed
value.

All edge-incidence, orientation, symmetry, KKT residual, P1 residual,
operator-symmetry, eigen-residual, and independent objective-identity checks
pass. Quadrature orders 12 and 18 agree in `kappa_h` to
`1.100855517855103e-14`. The coarser `h=0.12` mesh fails the corrected
additive diagnostic (`0.13126653788302517`), which is a useful refinement
sanity check.

## What remains to certify

The continuum spectral flag remains false. A proof-grade upper bound needs:

1. directed enclosures for the `mu`, `mu^-1`, P1, RT0, and P0 matrix entries;
2. controlled sparse factorization/solve roundoff; and
3. a verified upper bound `kappa_h < 0.04720654776803466`.

The dense hypercircle matrix need not be formed. If `P` is the RT mass, `N`
the divergence matrix, `A` the P1 stiffness, `B` the P1-P0 load, `D` the
triangle-area diagonal, and `W` the `mu^-1` P0 mass, define

```text
K(beta) =
[ P   N^T   0    0        ]
[ N    0    0    D        ]
[ 0    0    A   -B        ]
[ 0    D   -B^T -beta^2 W ]
```

For `beta=0.04720654776803466`, Schur complements reduce the final block to
`Q-beta^2 W`, where `Q` is the hypercircle quadratic form. Therefore a
verified inertia

`positive = edge_count + state_count = 61908`,

`negative = 2*triangle_count = 61908`,

`zero = 0`

would prove `kappa_h < beta`, subject to the matrix-entry enclosures. This is
the next bounded certification target.

The positive-time point-source bridge, continuum conormal output, and
polygon-to-circle transfer remain separate later obligations. Nothing here is
a Navier-Stokes regularity or Clay-prize proof.
