# Affine-to-taper residual no-go

## Question

The divergence-free streamfunction taper is a valid incompressible reference
field and its wide visit closes numerically. Can an actual locally affine
Navier-Stokes drift be treated as a small sector perturbation of that tapered
reference?

No, not uniformly over the affine spectrum. The outer collar alone produces
an `L^3` mismatch more than thirteen times the entire working drift budget at
the worst spectrum.

## Exact collar lower bound

For spectrum parameter `t`, put

```text
s=t+1/2,       0<=s<=3/2.                             (1)
```

The anisotropic part of the exact full-affine transverse drift is

```text
b_aff=s(x,-y).                                        (2)
```

The streamfunction taper is

```text
b_tap=s(x f+x y^2 f'/r,
        -y f-x^2 y f'/r).                             (3)
```

On the outer collar `2.65<=r<=2.75`, `f=f'=0`. Therefore

```text
|b_aff-b_tap|=s r.                                    (4)
```

For `|z|<1.2`, (4) gives the exact lower bound

```text
||b_aff-b_tap||_3/nu
 >=s[4 pi(1.2)(2.75^5-2.65^5)/5]^(1/3)
 =4.31235641799 s.                                    (5)
```

At `t=1`, `s=3/2`, so

```text
||b_aff-b_tap||_3/nu>=6.46853462698.                  (6)
```

The complete no-potential sector budget is only

```text
||e||_3/nu<0.478113110829.                            (7)
```

Thus the collar lower bound is `13.53` times the entire allowance before any
nonlinear velocity error, frame drift, or pressure-driven strain variation is
added. Equation (7) can tolerate (5) only for

```text
s<0.11087,       t<-0.38913.                          (8)
```

Most of the affine spectrum is excluded. Dense integration through the full
taper only increases the mismatch.

This is an affine-consistency test. A perturbative reference should give zero
error on the exact structure it is intended to model. The tapered shell does
not pass that test for an affine field that continues across the visit.

## Compact full-affine replacement

The fixed-label stopping-time architecture does not require a spatially
localized drift outside the stopped domain. It can therefore retain the
complete fitted affine field through a compact cylinder and stop at the
boundary. An exact affine field then has

```text
e=0.                                                   (9)
```

At the previously audited conservative compact height `H/L=0.75`, outer
radius `2L`, and worst spectrum `t=1`, the finite-element pilot gives

```text
visit norm                 0.55681307,
generation criterion       0.16001938,
sector condition number   11.49450692.                (10)
```

Without a restart, the combined sector intercept is

```text
d_aff=0.13048395.                                     (11)
```

Equal potential and drift shares permit approximately

```text
||q_+||_(3/2)/nu<0.33550,
||e||_3/nu<0.14335.                                   (12)
```

If every coherence abort is paid by the true pair split, the equal-share
budgets reduce to

```text
||q_+||_(3/2)/nu<0.11162,
||e||_3/nu<0.04769.                                   (13)
```

The values in (10)-(13) are finite-element calibrations, not enclosures.
They are structurally preferable because the reference is exact on a full
affine drift.

## Consequence

For the stopped moving-cylinder route:

1. use the compact full-affine field fitted at entry as the baseline;
2. put only nonlinear affine oscillation and temporal strain drift into the
   sector remainder;
3. retain the divergence-free taper as a separate model architecture, or use
   it only when taper coherence of the physical drift is independently
   established.

The next task is to derive the compact full-affine remainder from the
mollified Leray frame. The collar lower bound, dense mismatch check, and
compact transfer arithmetic are reproduced by
`scripts/affine_taper_residual_no_go_audit.py`.
