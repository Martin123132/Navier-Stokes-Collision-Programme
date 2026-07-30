# Neutral-strip parabolic spectral split

## Purpose

The global lumped-mass ratio does not converge to one on the highest mesh
modes. A valid continuum comparison must retain a finite low spectral block
and pay the complementary modes through parabolic decay. This note proves the
continuum high-mode payment. The companion Brownian-majorant audit now gives
an analytic first-window bound; its initial constant is too coarse to close
the response budget. The finite low block remains a separate obligation.

## Weighted operator

On the true finite strip outside the unit circle, let

```text
P=-mu^(-1) div(mu grad),       mu=exp(-x^2/2),       (1)
```

with Dirichlet data on the circle, strip walls, and finite sides. The form is
self-adjoint in `L2(mu)`. Domain monotonicity against the infinite strip gives

```text
lambda_1(P) >= lambda_strip=pi^2/(4Y^2),  Y=2.1.    (2)
```

Conjugation by `sqrt(mu)` gives

```text
sqrt(mu) P sqrt(mu)^(-1)=-Delta+x^2/4-1/2.          (2a)
```

The two-dimensional Li-Yau inequality and `x^2/4-1/2>=-1/2` therefore give
an analytic cutoff after retaining `K` modes:

```text
lambda_(K+1)(P)
 >= 2 pi (K+1)/|D|-1/2,
|D|=4XY-pi.                                         (2b)
```

For `X=4.2`, `Y=2.1`, and `K=320`, (2b) is above `62`. Thus the high-mode
cutoff below does not rely on a floating FEM eigenvalue.

## Explicit inner-flux Rellich bound

Choose the radial Lipschitz multiplier

```text
q=-chi(r)e_r,
chi(r)=2-r for 1<=r<=2,
chi(r)=0 for r>=2.                                  (3)
```

It satisfies `q.n=1` on the inner circle and vanishes before the outer
boundary. The weighted Rellich identity for `Pu=f` is

```text
int_S1 mu (partial_n u)^2
 =int_D mu grad(u).[2 sym(grad q)-div_mu(q) I].grad(u)
  -2 int_D mu f q.grad(u).                          (4)
```

For (3), `||q||_infinity<=1` and the matrix in (4) has norm at most

```text
|chi'|+chi/r+chi x^2/r <= 4.                        (5)
```

Energy and (2) therefore give

```text
||mu partial_n u||_L2(S1)
 <= C_flux ||f||_L2(mu),
C_flux^2=2/sqrt(lambda_strip)+4/lambda_strip,
C_flux=3.134170665703... .                          (6)
```

The first norm in (6) is no larger than the weighted boundary term in (4)
because `0<mu<=1` on the circle. No FEM constant enters this estimate.

## Point source and high modes

Let `E_[Lambda,infinity)` be the spectral projector of `P`. Applying (6) to
the half-time factorization gives, for an entry point `z`,

```text
||Gamma exp(-tP) E_high delta_z||_2
 <= C_flux sup_(lambda>=Lambda)[lambda exp(-lambda t/2)]
    sqrt(k_D(t;z,z)).                                (7)
```

Positivity and domain monotonicity bound the killed diagonal by the full
transverse OU kernel with respect to `mu dx dy`. Uniformly over the entry
circle, `x_z^2<=4`,

```text
k_D(t;z,z) <= k_OU(t;z,z)
 <= exp(2[1-tanh(t/2)])
    /[2 sqrt(2) pi sqrt(t(1-exp(-2t)))].             (8)
```

When `Lambda>=2/t`, the spectral supremum in (7) is simply
`Lambda exp(-Lambda t/2)`.

## Later-window payment

The response uses windows of length `ell=3/8`. Reserve the complete first
window for an independent distance bound. For `Lambda=60`, (7)-(8), the
monotonic axial factor, and geometric decay between later windows give

```text
sum_(n>=1) sup_(t in [n ell,(n+1)ell]) rho_high(t)
 <= rho_high(ell)/(1-exp(-Lambda ell/2)).            (9)
```

The same bound controls scalar mass because boundary `L1` is at most
`sqrt(2pi)` times boundary `L2`; integrating the exponential in (7) cancels
the leading `Lambda`. The audit evaluates both payments. At cutoff `60` they
are each below `0.01`, well below the current response margin.

## Numerical low block

The optional production branch independently assembles the weighted P1
forms, computes the first `K+1` generalized modes, compares the numerical
first omitted eigenvalue with (2b), and measures modified/reference mass,
stiffness, and boundary coupling on the retained block. The high-mode payment
uses (2b), not the numerical eigenvalue. The comparison rows remain floating
diagnostics until eigenpairs and quadrature are interval enclosed.

For `K=320`, the production rows are

```text
spacing                  0.12                    0.09
P1 first omitted         152.6882937354          146.2442547422
modified/reference mass  1.00410..1.22019        1.00230..1.11827
modified/reference stiff 0.98842..1.00133        0.99366..1.00063
boundary spectral error  0.00334394              0.00207483.       (10)
```

Those last values compare only stiffness cross-blocks. The complete
consistent-mass transient conormal map is
`B_stiff+lambda B_mass` on an eigenmode. Including that term changes the
modified/reference boundary discrepancy at cutoff 320 to `0.138135` and
`0.0614455`, respectively. The companion transient-conormal audit therefore
uses cutoff 240 and extends the mesh sequence through `h=0.06`.

The numerical first omitted values comfortably exceed the analytic Li-Yau
lower bound `62.2567651958`, but only the latter is used in the theorem. It
gives the all-later-window payments

```text
interval factor high tail     0.000683411235,
scalar gain high tail         0.000177212109.       (11)
```

## First window and remaining gate

The companion calculation in
`neutral_strip_first_window_brownian_majorant.md` uses stopped Girsanov
domination, Bessel absolute continuity, and a summable radial-excursion split
to bound the complete `0<t<=3/8` continuum flux. The companion
maximum-bridge certificate integrates over the first-passage bridge maximum,
encloses the complete time supremum, and lowers the isolated first-window
interval factor to `0.952384193963` while retaining the analytic mode tail.

The regular-polygon boundary Riesz matrix, radial pushforward Gram, and
modified dual-cell cross Gram are now assembled analytically in a common
`L2(S1)` space. The source projections at the exact `r=2` entry vertices are
also included. This correction demotes the old raw-load screen `0.984266`:
the valid time-zero common-circle operator screen is above one. After actual
source projection and parabolic evolution, however, the later-window start
samples converge over four meshes. The companion time-slab gate now encloses
fifteen finite windows and the nonoverlapping post-`6` tail for the frozen
finite block. At `h=0.06`, `K=240`, and substep `0.0125`, its guarded factor is
`0.00809092`, giving combined screen `0.97008448` and headroom `0.02991552`.
The exact Gaussian-weighted reference finite-element forms are now enclosed
around the fingerprint-matched q12 matrices, with mass relative form error
`5.49257e-13` and stiffness error `5.43119e-9` in stored-mass units. The
indexed generalized eigenvalues are now directed enclosed for both the
stored pencil and exact forms on the stored polygon. The remaining
Riesz/Gram/projected finite-block algebra and generalized eigenvectors are
not yet fully directed-interval enclosed.

The symmetrized off-block coupling is `6.34370`, so discarding complement spectral
decay still gives the useless Duhamel bound `2.37889`. The preferred next
proof combines a spectrally damped leakage bound with a low-mode/shape polygon
perturbation. Endpoint propagation, continuum Ritz transfer, and interval
finite-block actions remain uncertified, so the complete return response is
still open.

The audit is reproduced by
`scripts/neutral_strip_parabolic_spectral_split_audit.py`.
