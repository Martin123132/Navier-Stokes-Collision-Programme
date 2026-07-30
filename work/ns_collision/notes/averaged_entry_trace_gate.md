# Averaged space-time entry trace gate

## Purpose

The finite-energy payoff candidate removes the need to localize the critical
forcing away from the absorbing boundary. Its positive overshoot is globally
controlled in the volume energy space, but an arbitrary point entry still
cannot be evaluated at the critical endpoint.

This note pairs that energy response with the actual **unnormalized**
outer-to-inner return law. The result is a conditional theorem with one
explicit exterior-kernel obligation. It preserves the probability loss from
early nonreturns and does not use a continuous partition or independent
interaction labels.

## Surface trace

Let

```text
D={r<2, |z|<0.75},             Sigma={r=1, |z|<0.75},
h[v]=||grad v||_2^2-||v||_2^2,
m_0=lambda_1(D)-1=4.832287335665.                    (1)
```

For `v` with zero trace at `r=2`, the radial fundamental theorem of calculus
gives

```text
int_Sigma |v|^2 d sigma
 <=int_{1<r<2}(|v|^2+|partial_r v|^2)dx.             (2)
```

Poincare then yields the explicit complete trace bound

```text
||T v||_2^2<=C_Sigma h[v],
C_Sigma=lambda_1/(lambda_1-1)+1/(lambda_1-1)
       =1.413882673168.                              (3)
```

No point evaluation occurs in (3).

## Time-averaged energy

Let `w=(u-U_H)_+`, let

```text
alpha<=0.220329037686 ||q||_(3/2),
a=1-alpha,
F=0.798968551320 ||q||_(3/2)
 +3.072840583265 ||e||_3.                            (4)
```

The nonautonomous positive-part estimate and its time-reversed version imply
on every interval `I` of length `ell`

```text
int_I h[w]
 <=F^2[ell/a^2+1/(a^3 m_0)].                         (5)
```

The endpoint term in the integrated energy inequality is paid by the
already-proved uniform bound

```text
sup_t ||w(t)||_2^2<=F^2/(a^2 m_0).                   (6)
```

If time is split into intervals of length `ell`, then for an exponential
envelope with `kappa>0`,

```text
int_0^infinity exp(-kappa s)h[w(t+s)]ds
 <=F^2 J(a,kappa),                                   (7)

J(a,kappa)=inf_(ell>0)
 [ell/a^2+1/(a^3m_0)]/[1-exp(-kappa ell)].           (8)
```

At `a=kappa=1`, the optimum is

```text
ell_*=0.581147187470,        J=1.788088519863.        (9)
```

The interval proof of (7) is insensitive to the orientation of the backward
payoff equation; it does not assume differentiability of the measurable
affine history.

## Unnormalized return law

Let `R` be a sub-Markov entry kernel from the preceding branch to the
space-time inner interface. For a same-scale excursion this is the exterior
return kernel. Assume its unnormalized law satisfies

```text
d nu/(ds d sigma)<=C_R exp(-kappa s).                (10)
```

Jensen, (3), and (7) give

```text
||R w||_(L2(mu))
 <=F sqrt[C_R C_Sigma J(a,kappa)].                   (11)
```

The law in (10) is not divided by its entry probability. Paths that do not
return simply contribute zero to the renewal branch. Thus small return mass
is retained as contraction rather than converted into a bad conditional
density ratio.

Since the candidate baseline obeys `U_H<=g_H` on the entry interface,

```text
||R u||_(L2(mu))
 <=g_H+F sqrt[C_R C_Sigma J(a,kappa)],               (12)

g_H=1.145614144998.                                  (13)
```

The dynamic pair cycle closes whenever the second term in (12) is below

```text
g_*-g_H=0.086519463497.                              (14)
```

## Exponential special-case budgets

Normalize the density envelope by

```text
M_R=C_R |Sigma|/kappa,       |Sigma|=3 pi.            (15)
```

`M_R=1` is the exponential-in-time, uniform-in-space reference density. For
`kappa=1`, the conditional one-error thresholds from (12)-(14) are:

| `M_R` | `||q||_(3/2)` | `||e||_3` |
|---:|---:|---:|
| 1 | 0.198745 | 0.0543636 |
| 2 | 0.142596 | 0.0384409 |
| 4 | 0.101889 | 0.0271818 |
| 8 | 0.0725854 | 0.0192204 |
| 16 | 0.0515988 | 0.0135909 |

These are scale-invariant normalized masses. They are implications of the
density hypothesis, not established Navier-Stokes tolerances. Moreover, an
unbounded Brownian exterior has a polynomial hitting-time tail, so this
exponential table applies only to a bounded storage region or an excursion
with additional killing. The raw exterior must use the summable-envelope
version proved in `exterior_return_tail_gate.md`.

## What changed

The route no longer requires

```text
critical perturbation support inside a protected collar,
continuous square-root localization and its IMS cost,
independent interaction labels or discarded cross-label blocks.          (16)
```

The first analytic certificate is now complete. The route requires:

1. the certified finite-energy HJB barrier above;
2. a summable interval-supremum envelope for the physical unnormalized
   exterior-return kernel, including exterior deformation weight;
3. an entry-law mechanism after every true dyadic split.

The third item cannot be omitted, but it need not be an independently
generated space-time envelope. Pointwise Markov relabeling preserves any
incoming physical density exactly. It creates no smoothing, however, and a
deterministic level-change time is a temporal atom outside this averaged
surface theorem. The split must either inherit a prior absolute space-time
density without collapsing it, or use the fixed-time bounded-volume route in
`split_entry_density_inheritance.md`. Split and return gains remain separate
in the renewal algebra.

The second item is a genuine Navier-Stokes exterior obligation. A Brownian
or affine model density is only a calibration until the residual physical
drift and stretching are included.

The trace constant, interval-energy factor, and density-budget table are
reproduced by `scripts/averaged_entry_trace_gate.py`. The barrier certificate
is reproduced independently by
`scripts/radial_h1_payoff_interval_certificate.py`.

The necessary polynomial-tail correction and exact half-space calibration
are in `exterior_return_tail_gate.md`.
