# Branch-resolved entry renewal

## Purpose

The finite-energy entry theorem must be applied to two physically different
transitions: a same-scale exterior return and the one genuine dyadic split
that starts the next generation. Combining them into one worst-case visit
gain hides branch mass and can count an unnormalized return probability
twice.

This note gives the exact branch-separated algebra and a Brownian cylinder
calibration. It does not prove the weighted Navier-Stokes entry envelopes.

## Exact renewal identity

Let `a_S` be the complete one-history gain for a true-split entry followed
by its visit, and let `a_R` be the complete one-history gain for a same-scale
return followed by its visit. Independent replicas square these gains.
Summing all returns before the next genuine split gives

```text
G=a_S^2[1+a_R^2+a_R^4+...]
 =a_S^2/(1-a_R^2).                                  (1)
```

Consequently,

```text
G<1  if and only if  a_S^2+a_R^2<1.                 (2)
```

Condition (2) also implies convergence of the same-scale Neumann series.
This is the branch-resolved form of the old criterion
`(gamma_pair+r_pair)g^2<1`.

## Unnormalized entry kernels

Let `T_j`, for `j=S,R`, be a positive unnormalized entry kernel and define

```text
p_j=||T_j 1||_(L2(mu)).                              (3)
```

For the certified baseline payoff `0<=U_H<=g_H` on the entry surface,
positivity gives

```text
||T_j U_H||_2<=p_j g_H.                              (4)
```

If the spatial `L2` density envelope of branch `j` gives the averaged trace
error

```text
delta_j<=F sqrt[C_4 J_(rho_j)],                      (5)
```

then a sufficient closure condition is

```text
(p_S g_H+delta_S)^2+(p_R g_H+delta_R)^2<1.           (6)
```

The mass in (3) is already inside the unnormalized kernel. It must not be
multiplied into (6) a second time. For a weighted Feynman-Kac kernel `p_j`
need not be a probability or be below one; it has to be estimated as part of
the physical branch theorem.

## Legacy calibration

At the compact calibration `R_*=1/2` in dimension three,

```text
gamma_pair=exp[(1/2)3/24]/4=0.266123614729,
p_S=sqrt(gamma_pair)=0.515871703750.
```

The legacy buffer estimate used `p_R=1/2`, so

```text
gamma_pair+p_R^2=0.516123614729,                     (7)
```

agreeing with the stored finite-element coefficient to `4.6e-12`. With the
certified finite-energy gain

```text
g_H=1.145614144998,
```

the branch baselines and criterion are

```text
a_S=0.590989920820,
a_R=0.572807072499,
a_S^2+a_R^2=0.677377028816.                          (8)
```

The exact remaining additive allowances are `0.12514984` on both branches,
`0.22870030` on the split branch alone, or `0.23387187` on the return branch
alone.

## Current cubic split calibration

The live parent-child transfer also pays the cubic recentering cost at
support radius `rho_s=1.91`. Its factors are

```text
p_S=0.639292608019,       p_S^2=0.408695038668.       (9)
```

With the legacy `p_R=1/2`, the corrected branch values are

```text
a_S=0.732382654539,
a_R=0.572807072499,
a_S^2+a_R^2=0.864492294975,
G=0.798319233762.                                   (10)
```

The equal-branch error allowance is `0.04999599`; the split-only and
return-only allowances are `0.08730757` and `0.10808620`. Equations (7)-(8)
are retained to identify the older bare-halving arithmetic, not as the live
cubic budget.

## Brownian finite-patch pilot

For Brownian motion with generator `Delta`, start at `(r,z)=(2,0)`, stop on
the infinite cylinder `r=1`, and accept the return only when `|z|<H`. Fourier
transform in `z` gives the exact representation

```text
p_H=(2/pi) int_0^infinity
    [sin(kH)/k] K_0(2k)/K_0(k) dk.                  (11)
```

The axial Poisson kernel is an even decreasing Gaussian mixture, so the
centered start maximizes acceptance among axial starting positions. At
`H=3/4`, numerical quadrature gives

```text
p_H=0.310135151371.                                  (12)
```

Replacing only the legacy return factor by this Brownian pilot gives

```text
a_S=0.732382654539,
a_R=0.355295216272,
a_S^2+a_R^2=0.662619043376,
G=0.613876915191.                                    (13)
```

The corresponding allowances become `0.13766768` on both branches,
`0.20237149` on the split branch alone, and `0.32559806` on the return branch
alone. The numerical budget therefore has room; the failure is not an
obvious Brownian return-probability obstruction.

Equation (12) is a convergence pilot, not an interval enclosure. More
importantly, it does not include the exterior deformation weight, physical
Navier-Stokes drift, moving frame, cap exits, or true-split entry law.

## Remaining theorem

The next proof obligation is not one common entry estimate. It has two
branch-specific parts:

1. a weighted same-scale return envelope with gain `p_R` and error
   `delta_R`;
2. a true-split law with gain `p_S` and error `delta_S`, obtained either by
   inheriting a prior absolute space-time density or by a bounded fixed-time
   child-volume density.

The return branch must have summable interval suprema in the spatial `L2`
density norm of `exterior_return_tail_gate.md`. A pointwise label split
preserves such an envelope but does not create one, and a deterministic
split-time atom needs the volume alternative. The exact dichotomy is proved
in `split_entry_density_inheritance.md`. Both gains enter (6) exactly once.

The identities, allowances, no-double-counting diagnostic, and Brownian
Fourier quadrature are reproduced by
`scripts/branch_resolved_entry_renewal_audit.py`.
