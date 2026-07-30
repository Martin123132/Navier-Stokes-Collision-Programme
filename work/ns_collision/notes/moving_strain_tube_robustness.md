# Moving strain tube and integral-norm robustness

## Purpose

The fixed affine-core estimate in `localized_strain_tube.md` has a strict
spectral margin, but a candidate Navier-Stokes concentration region will move,
rotate, change width, and depart from exact affine strain. This note computes
those errors in fixed coordinates and replaces the pointwise perturbation gate
by a first `L^2` form bound.

The result still assumes a smooth tube and a fixed circular cross-section
after mapping. It is a bridge toward finite-energy estimates, not a global
regularity theorem.

## Moving coordinates

Write the physical transverse coordinate as

```text
x = c(t) + L(t) O(t) y,       |y| < 1,
```

where `c` is the tube centre, `L` its transverse radius, and `O` a planar
rotation. If the physical backward drift is `b(x,t)`, the mapped diffusion and
drift are

```text
k(t) = nu/L(t)^2,

b_map = L^(-1) O^T (b-c') - (L'/L)y - Omega y,
Omega = O^T O'.
```

Choose an affine reference rate `a(t)>0` and set

```text
e(y,t) = b_map-a y,
R(t) = a(t)/k(t) = a(t)L(t)^2/nu.
```

For one deformation factor, write the stretching potential as
`2a+delta_s`. The mapped killed equation is

```text
partial_t f = [k Delta + (a y+e) dot grad + 2a+delta_s] f.
```

## Time-dependent gauge

Use

```text
f = exp(-R(t)|y|^2/4) psi.
```

The affine part becomes the harmonic-oscillator form from the fixed-tube
calculation. The derivative of the gauge adds `R'|y|^2/4`. After integrating
the first-order drift by parts, every moving and non-affine contribution is
contained in

```text
q_mov = delta_s
        - 1/2 div(e)
        - (R/2) y dot e
        + (R'/4)|y|^2.
```

This formula exposes several geometric effects.

1. Pure rotation has `e=-Omega y`, `div(e)=0`, and `y dot e=0`. It contributes
   exactly zero to the radial-gauge energy.
2. A centre-velocity mismatch enters through the linear term
   `-(R/2)y dot e`.
3. Tube expansion enters through both `e=-(L'/L)y` and `R'`. Expansion is a
   positive error because it delays boundary exit; contraction helps.
4. A changing reference strain also enters through `R'`.

Let `lambda_1(a,k)` be the principal killed eigenvalue on the unit disk and

```text
m = lambda_1-2a > 0.
```

The pointwise estimate is

```text
ess_sup q_mov < m.
```

## An L2 form gate

Pointwise control is too strong for the intended application. Let `K` be the
positive ideal shifted oscillator form, so `K>=m`, and put

```text
Q = ||(q_mov)_+||_L2(D).
```

Zero extension from the disk and the two-dimensional Ladyzhenskaya inequality
give

```text
integral q_mov |psi|^2
    <= Q ||psi||_4^2
    <= sqrt(2) Q ||psi||_2 ||grad psi||_2.
```

The oscillator form also gives

```text
k ||grad psi||_2^2 <= <psi,K psi> + a||psi||_2^2.
```

If `kappa=<psi,K psi>/||psi||_2^2`, then `kappa>=m`, and
`sqrt(kappa+a)/kappa` decreases with `kappa`. Consequently

```text
integral q_mov |psi|^2 <= alpha <psi,K psi>,

alpha = sqrt(2) Q sqrt(m+a)/(sqrt(k)m).
```

Thus `alpha<1`, equivalently

```text
Q < m sqrt(k)/(sqrt(2) sqrt(m+a)),
```

is a sufficient integral-norm robustness condition. The transformed norm then
decays at least at rate `(1-alpha)m` for one history. Two additive history
potentials obey the corresponding tensor-product estimate.

This is a form estimate, so errors may be large on a small part of the
cross-section. It is still stronger than Leray-level control: the formula
requires the positive effective error in transverse `L^2` at almost every
time, together with time integrability of the resulting decay rate.

## Numerical budgets

In units of `a`, on the unit disk, the sufficient `L^2` budget is

```text
Q/a < (m/a)/sqrt(2 R (m/a+1)).
```

| `R` | `m/a` | allowed `Q/a` |
|---:|---:|---:|
| 0.1 | 56.8373109669 | 16.7114632579 |
| 0.5 | 10.5936192507 | 3.11124952273 |
| 1 | 4.83762216791 | 1.41579037868 |
| 2 | 2 | 0.577350269190 |
| 5 | 0.419753935419 | 0.111400781107 |
| 10 | 0.0549460799925 | 0.011962076759 |
| 20 | 0.000804267033873 | 0.000127114676903 |
| 50 | 6.65346888695e-10 | 6.65346888474e-11 |

The form gate is useful at moderate `R` but becomes exceptionally fragile at
large `R`. Any successful high-Reynolds argument will need either sharper
local geometry, an averaged occupation mechanism, collision damping beyond
the escape margin, or a decomposition that prevents one tube from carrying a
large `R` for long.

## Remaining obligations

The next immediate issue is re-entry. A killed estimate controls one
uninterrupted visit. A complete history can exit, evolve elsewhere, and enter
again. The natural next object is a buffered renewal operator combining one
contractive tube visit with the weighted probability of returning through an
outer shell.

Beyond that, the `L^2` gate must be related to quantities available for an
actual Navier-Stokes solution. In particular, `q_mov` contains the divergence
and radial moment of the drift error, not only the strain magnitude. A moving
tube selected from a rough solution must therefore be constructed carefully;
these terms cannot be assumed small by definition.

The formulas and numerical budgets are reproduced by
`scripts/moving_strain_tube_audit.py`.
