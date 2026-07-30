# General affine spectral floor

## Question

The axisymmetric reference strain is spectrally effective, but a general
Navier-Stokes strain need not have two equal eigenvalues. Requiring that
mismatch to be small spends almost the entire coherence budget. This note
asks a narrower question: does constant eigenvalue mismatch actually destroy
the localized spectral gap?

It does not. The spectral part of the construction extends uniformly to every
symmetric trace-free affine strain with a positive maximal eigenvalue. The
outer Poisson transfer and moving eigenframe do not yet extend automatically.

## One-parameter spectrum

Let the ordered eigenvalues of the forward strain be

```text
lambda_1<=lambda_2<=lambda_3,       sum lambda_i=0,
lambda_3 L^2/nu=1.
```

Every such spectrum has the form

```text
(lambda_1,lambda_2,lambda_3)=(-1-t,t,1),
-1/2<=t<=1.                                             (1)
```

The corresponding backward affine drift has diagonal coefficients
`(1+t,-t,-1)`. Conjugating its transverse drift on the unit disk produces

```text
-Delta_perp + [(1+t)^2 x^2+t^2 y^2]/4.                 (2)
```

The potential in (2) is nonnegative for the entire interval (1). On the full
plane its contribution after subtracting maximal stretching is

```text
(t+|t|)/2=max(t,0),                                    (3)
```

so even the unconfined affine model has no negative spectral excess.

## Uniform disk bound

For Dirichlet data on the unit transverse disk, discard the nonnegative
potential in (2). The disk Laplacian has ground value

```text
j_(0,1)^2=5.783185962946783.
```

The axial oscillator contributes `1/2` and maximal stretching subtracts
`1`. Hence every spectrum in (1) obeys

```text
m_aff L^2/nu >= j_(0,1)^2-1/2
              =5.283185962946783.                     (4)
```

This is a rigorous lower bound, not a finite-difference observation. It loses
only `0.013623662397` from the previously certified axisymmetric margin
`5.296809625343`.

## Cubic localization

On the existing optimized support `rho_s=1.91`, the certified full tensor
cubic IMS cost is

```text
I_IMS=4.838443634281468.
```

Combining this with (4) leaves

```text
m_aff-I_IMS=0.444742328665315.                         (5)
```

The sharp three-dimensional homogeneous Sobolev conversion therefore permits
the unit relative-form mass

```text
Q/nu < (m_aff-I_IMS)/[S_3(m_aff-I_IMS+1)]
     =1.686290885679306.                               (6)
```

Equation (6) is the internal localized form budget before paying any
finite-cylinder/Poisson boundary conversion.

## What this closes

Constant symmetric affine mismatch no longer needs to be charged to the tiny
`G` or `H` coherence budget. At the spectral stage, each dangerous cell can
use its complete locally fitted symmetric trace-free affine strain rather
than an axisymmetric approximation. The controlled remainder can therefore
begin with nonlinear spatial variation of the strain.

This removes a real obstruction from the earlier coherence formulation. It
does not yet produce a Navier-Stokes regularity theorem.

## What remains open

The old axisymmetric outer-Poisson relative-form factor is
`alpha=0.125398692617`. Multiplying (6) by it would give
`0.211458672437`, close to the old final budget. That multiplication is only a
diagnostic: the Poisson/cylinder transfer has not been proved uniformly for
the anisotropic drift (2).

Two gates remain:

1. extend the complete finite-cylinder and outer-Poisson boundary transfer to
   all `t` in (1), including orientation dependence;
2. transport locally translating and rotating eigenframes conservatively
   between overlapping cells and visits. A general skew component need not
   commute with an anisotropic affine core, although rotation about the
   symmetry axis was harmless in the old model.

The next decisive calculation is therefore an anisotropic Poisson transfer
audit over the compact one-parameter family (1), not another attempt to prove
that the eigenvalues are nearly axisymmetric.

All constants and algebraic checks are reproduced by
`scripts/general_affine_spectral_floor_audit.py`.
