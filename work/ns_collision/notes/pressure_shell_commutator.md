# Pressure shell commutator and the local-Reynolds gate

## Purpose

Pointwise pressure-frame pairing fails, but pressure is globally orthogonal
to the strain. This note tests whether localizing that exact cancellation is
enough to absorb pressure in the transition shell of a strain tube.

The answer is deliberately split:

1. localization gives an exact pressure boundary commutator;
2. a universal fixed-scale absolute absorption estimate is impossible;
3. an adaptive moderate-local-Reynolds shell remains a legitimate, unproved
   route.

No regularity conclusion is claimed.

## Exact localized identity

Write

```text
-Delta p=f,
f=|S|^2-|omega|^2/2,
q_s=p-exp(s Delta)p.
```

The pressure collision defect is

```text
B_s^P=Hess q_s+(f-exp(s Delta)f)I/3.
```

Because `tr S=0`, the isotropic term vanishes. For every smooth cutoff
`phi`, symmetry of `Hess q_s` and incompressibility give

```text
integral phi B_s^P:S dx
 =integral phi partial_i partial_j q_s partial_j u_i dx
 =-integral partial_j q_s partial_j u_i partial_i phi dx
 =-integral grad q_s dot [(grad u)^T grad phi] dx.       (1)
```

The term containing `partial_i partial_j u_i` vanishes because it is a
derivative of `div u`. The same identity holds for the full pressure and the
heat-smoothed low-pressure piece separately. Thus pressure acts only where
`grad phi` is nonzero.

## Periodic identity audit

The deterministic trigonometric field from the pointwise pressure
counterexample is reused. On the `2pi`-periodic torus, set

```text
phi(x)=product_i (1+cos(x_i-x_i^*))/2,
```

where `x^*` is the refined strict maximum of `lambda_3`. This is a broad
periodic localization chosen to make all derivatives exact spectrally, not
an optimized tube cutoff. The normalized spatial integrals are

```text
piece                 integral phi P:S       boundary form       residual
full                  -1.778741171023481     -1.778741171023467  -1.38e-14
heat-smoothed low     -1.981008041028337     -1.981008041028333  -4.66e-15
collision defect       0.202266870004858      0.202266870004865  -7.27e-15
```

The low and defect terms recombine to the full term. The localized viscous
strain dissipation for viscosity one is

```text
nu integral phi |grad S|^2 dx =256.88787802985223.
```

Under `u -> -u`, pressure is unchanged while `S`, `grad u`, and both sides of
(1) reverse sign. Consequently the shell work has no universal favorable
sign, even on smooth divergence-free data.

## Absolute estimate and the time-integrability failure

On `R^3`, or on a fixed torus with the corresponding periodic constants,
the order-one pressure potential and Hardy-Littlewood-Sobolev estimate give

```text
|I_phi|
 <=C ||grad phi||_infinity ||S||_2 ||f-f_s||_(6/5)
 <=C ||grad phi||_infinity ||A||_2 ||f||_(6/5),          (2)
```

where `A=grad u`; heat contraction is absorbed into `C`. Since
`|f|<=C|A|^2`, interpolation between `L^2` and `L^6` gives

```text
||f||_(6/5)
 <=C ||A||_(12/5)^2
 <=C ||A||_2^(3/2) ||grad A||_2^(1/2).                  (3)
```

Combining (2) and (3),

```text
|I_phi|
 <=C ||grad phi||_infinity
      ||A||_2^(5/2) ||grad A||_2^(1/2).
```

Young's inequality can absorb the derivative factor only at the cost

```text
epsilon nu ||grad A||_2^2
 +C (epsilon nu)^(-1/3)
    ||grad phi||_infinity^(4/3) ||A||_2^(10/3).          (4)
```

Leray control supplies time integrability of `||A||_2^2`, not
`||A||_2^(10/3)`. The gap is real: a time spike of height `N^(1/2)` and
length `N^(-1)` has fixed `L_t^2` mass, while its `L_t^(10/3)` integral is
`N^(2/3)` and diverges.

This is the same critical gap previously found in the three-dimensional
moving-tube estimate, now reached directly through pressure localization.

## Dimensional decision gate

For a strain core with characteristic rate `a` and length `L`, a cutoff
changing across a comparable shell has

```text
pressure shell work       approximately a^3 L^3,
viscous shell dissipation approximately nu a^2 L.
```

Their ratio is

```text
R_local=a L^2/nu.                                         (5)
```

This is exactly the local strain-tube Reynolds number. Scaling the audited
periodic field at fixed spatial shape illustrates (5): pressure work scales
cubically and dissipation quadratically. Their ratio grows linearly with
velocity amplitude and crosses one at velocity RMS approximately `1444.21`.
It can exceed any prescribed universal absorption constant by further
amplitude scaling.

Therefore an estimate of the form

```text
|I_phi| <= c nu integral phi |grad S|^2
```

with fixed cutoff geometry and universal `c` cannot hold for all smooth
fields. This does not rule out estimates with additional lower-order terms,
signed cancellation over time, or cutoffs selected from the solution.

## Surviving route

Equation (5) identifies a useful stopping rule rather than merely a failed
inequality. A shell is perturbatively absorbable only while

```text
a(t)L(t)^2/nu <= R_*.
```

The next construction should therefore split visits into two classes:

1. moderate-`R_local` visits, where pressure-shell work is assigned to the
   viscous exit budget;
2. high-`R_local` visits, where absolute shell absorption is abandoned and
   one seeks signed cancellation from trajectory integration, heat-scale
   collision defects, or independent-replica residence estimates.

The unresolved obligation is to define this adaptive stopping-time
decomposition without introducing uncontrolled cutoff-transport terms, then
show that high-`R_local` visits have a summable occupation or renewal cost.

The first part of that obligation is addressed in
`adaptive_reynolds_envelope.md`. A tube scaled by a monotone running envelope
of the strain rate never expands, caps `R_local`, and has a nonpositive
amplitude/scale error in the ideal transverse gauge when `R_*<=2`. The live
obstruction is consequently shifted to geometric coherence, pressure tails,
centre motion, and shrinking-core renewal.

An alternative to bounding the pressure tails cell by cell is developed in
`pressure_partition_flux.md`. Across a complete partition of unity, all
pressure shell commutators cancel exactly; unequal neighboring localization
weights are the only surviving pressure defect.

All numerical identities and scaling checks are reproduced by
`scripts/pressure_shell_commutator_audit.py`.
