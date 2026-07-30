# Localized strain tube: escape versus stretching

## Scope

This note isolates the missing effect in the uniform-affine stress test from
`collision_coherence_generator.md`. Relative separation by itself does not
control coherent stretching in an infinite strain field. If the stretching
region has finite transverse width, however, both backward histories must also
remain inside that region. Viscous escape then supplies a spectral loss.

For an ideal affine strain core with absorbing exit, that loss strictly beats
the largest coherent deformation growth for every finite tube Reynolds
number. This is an exact model theorem. It is not a Navier-Stokes regularity
theorem: a real strain region is nonuniform and time dependent, and histories
may leave it, stretch elsewhere, and re-enter.

## Ideal core

Take the incompressible affine strain

```text
S = diag(-a, -a, 2a),       a > 0,
```

inside a tube of transverse radius `L`. A backward stochastic history obeys

```text
dQ = -S Q dt + sqrt(2 nu) dW.
```

Its transverse coordinate `X=(Q_1,Q_2)` is therefore the outward
Ornstein-Uhlenbeck process

```text
dX = a X dt + sqrt(2 nu) dW,       |X| < L,
```

killed at the first exit from the disk. Its killed generator is

```text
L_perp = nu Delta + a x dot grad.
```

The stretching direction is the tube axis. A deformation vector aligned with
that direction grows at rate `2a`; two aligned factors can consequently grow
at rate `4a`. Thus the precise question is whether two-particle common
residence decays faster than `exp(-4at)`.

## Gauge transform

Write `U(x)=a|x|^2/2`, so that `L_perp=nu Delta+grad U dot grad`. Conjugating
the Dirichlet eigenvalue problem by `exp(-U/(2nu))` gives

```text
-L_perp  ~  H = -nu Delta + a^2 |x|^2/(4 nu) + a.
```

On the full plane, the oscillator part has ground energy `a`, with ground
state `exp(-a|x|^2/(4nu))`. Hence the bottom of `H` on the full plane is
exactly `2a`. Restriction to a finite disk with a Dirichlet boundary raises
the principal eigenvalue strictly:

```text
lambda_1(L) > 2a       for every finite L.
```

This can also be seen from the Rayleigh quotient. Equality would require the
strictly positive full-plane Gaussian ground state to satisfy a zero boundary
condition at `|x|=L`, which it does not.

For two independent histories, killed when either history exits, the product
semigroup has principal decay rate `2 lambda_1`. It follows that

```text
2 lambda_1 - 4a > 0.
```

Therefore common uninterrupted residence in this finite affine core loses
mass strictly faster than the maximum aligned two-factor deformation grows.
The collision-coherence damping derived in
`collision_coherence_generator.md` is an additional nonpositive contribution;
it is not needed for this strict ideal-core inequality.

This also explains why the earlier infinite affine calculation gave a
negative verdict. It retained only relative separation. In the finite tube,
each particle can escape transversely through the centre/common-coordinate
channel. In the infinite-width limit each particle contributes exactly `2a`
of spectral loss, so the pair loss `4a` only balances the deformation. The
finite boundary is what makes the inequality strict.

## Exact radial eigenvalue

Set

```text
R = a L^2/nu,       A = lambda_1/(2a).
```

The positive radial ground state reduces to Kummer's equation. Regularity at
the origin selects the confluent hypergeometric solution, and the Dirichlet
condition is

```text
M(A, 1, -R/2) = 0.
```

The smallest root with `A>1` gives the principal eigenvalue. The audited
values are:

| `R` | `lambda_1/a` | `(2 lambda_1-4a)/a` |
|---:|---:|---:|
| 0.1 | 58.8373109669 | 113.674621934 |
| 0.5 | 12.5936192507 | 21.1872385014 |
| 1 | 6.83762216791 | 9.67524433582 |
| 2 | 4 | 4 |
| 5 | 2.41975393542 | 0.839507870838 |
| 10 | 2.05494607999 | 0.109892159985 |
| 20 | 2.00080426703 | 0.00160853406775 |
| 50 | 2.000000000665 | 1.33069e-9 |

The positive margin becomes extremely small at large `R`. This is the main
robustness warning: a crude perturbation estimate will erase the effect in a
high-Reynolds tube even though the exact ideal-core inequality remains strict.

A finite axial boundary can only increase the first-exit loss in this killed
core model. The transverse disk is therefore the least favourable bounded
geometry among cylinders with the same transverse radius, as far as this
particular uninterrupted-residence estimate is concerned.

## A quantitative perturbation gate

There is a direct first robustness estimate on the same fixed disk. Let the
actual transverse backward drift be

```text
b(x,t) = a x + e(x,t)
```

and let the instantaneous deformation potential for one factor be
`c(x,t)=2a+delta_c(x,t)`. The ideal generator with deformation is
`L_perp+2a`. Conjugate by the same Gaussian as above and take the real
`L^2` energy. The first-order error satisfies

```text
Re integral psi (e dot grad psi)
    = -1/2 integral div(e) |psi|^2,
```

while conjugation contributes `-a x dot e/(2nu)`. All non-affine errors are
therefore combined in the effective potential

```text
q_eff = delta_c - 1/2 div(e) - a x dot e/(2 nu).
```

The ideal spectral bound then gives

```text
d/dt ||psi||_2^2
    <= -2 [lambda_1-2a-ess_sup(q_eff)] ||psi||_2^2.
```

Consequently, a sufficient one-history condition is

```text
ess_sup(q_eff) < lambda_1 - 2a.
```

For two histories with effective errors `q_eff,1` and `q_eff,2`, the summed
condition is

```text
ess_sup(q_eff,1 + q_eff,2) < 2(lambda_1 - 2a).
```

This is useful because it is an explicit, checkable non-affinity budget rather
than an appeal to qualitative stability. In units of `a`, the equal
single-history budget is half the final column in the table. It is about
`0.41975` at `R=5`, `0.05495` at `R=10`, `0.000804` at `R=20`, and
`6.65e-10` at `R=50`.

The bound is intentionally conservative. It uses a pointwise upper bound on
`q_eff`, a fixed circular cross-section, and a prescribed affine reference
rate. It does not yet cover a moving tube, rough Leray-level strain, boundary
flux caused by a moving localization, or re-entry. More refined form bounds
could permit errors that are large on small sets, but any such refinement must
still confront the collapsing high-`R` margin.

## What remains

The pointwise fixed-tube robustness estimate above is the first version of the
next proof gate. It now needs to be extended to a smooth, finite-energy,
moving strain region. That estimate must account for rotation of the
stretching direction, correlations between the two histories, and errors that
are controlled only in integral norms. After that, an exit-and-re-entry
renewal decomposition is needed: killing controls only one uninterrupted
visit, whereas an actual history can collect growth over many visits or
outside the selected core.

The model result is therefore positive but narrow. It identifies a real
viscous mechanism that the relative-gap observable missed, and it leaves two
concrete obligations rather than a claim of global closure:

1. stability of the spectral margin under integral-norm Navier-Stokes strain
   perturbations and moving localization;
2. summability of separated and re-entering history segments.

The symbolic gauge calculation and numerical Kummer roots are reproduced by
`scripts/localized_strain_tube_audit.py`.

The moving-coordinate and integral-norm extension is developed in
`moving_strain_tube_robustness.md`. Repeated visits are treated separately in
`strain_tube_reentry_renewal.md`.
