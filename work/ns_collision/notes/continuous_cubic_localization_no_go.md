# Continuous cubic localization no-go

## Question

The moving-label construction preserves `sum(phi)=1`, but viscosity exposes
the Fisher cost

```text
I_phi=sum_j |grad sqrt(phi_j)|^2.                      (1)
```

Can the current divergence-free visit on

```text
rho<2.75L,       |z|<1.2L                             (2)
```

absorb a simultaneous full tensor cubic partition throughout every visit?

For the cardinal-cubic square-root localization already used in this corpus,
the answer is no. This is an analytic form obstruction, not just an adverse
finite-element run.

## Exact Fisher lower bound

On one unit knot interval the one-dimensional cubic Fisher density is

```text
I_3(x)=(1/4)sum_j N_3'(x-j)^2/N_3(x-j).
```

The exact rational expression from `radial_cubic_partition.md` satisfies

```text
I_3(x)-3/4
 =-3x(x-1)[9x^4-18x^3+9x^2-2]
   /{4(3x^3-6x^2+4)(3x^3-3x^2-3x-1)}.                (3)
```

For `0<=x<=1`,

```text
9x^2(1-x)^2-2<=9/16-2=-23/16.                        (4)
```

The signs of the remaining factors in (3) give the exact lower bound

```text
I_3(x)>=3/4.                                           (5)
```

This complements the earlier verified upper bound `I_3<157/200`: the cubic
Fisher density is nearly constant and cannot be made small by shifting the
lattice phase.

## Subordination to the visit cylinder

Let `h_perp` and `h_z` be the tensor knot spacings. A cubic support has
half-width two knot spacings. Requiring its transverse square and axial
interval to fit in (2) forces

```text
h_perp/L<=2.75/(2sqrt(2)),
h_z/L<=1.2/2.                                          (6)
```

Tensorizing (5), every point of the exact cardinal partition obeys

```text
L^2 I_phi
 >=2(3/4)(L/h_perp)^2+(3/4)(L/h_z)^2
 >=12/2.75^2+3/1.2^2
 =3.67011019284.                                       (7)
```

Using smaller knot spacings only increases this lower bound.

## A negative test direction

After axial OU separation, the real symmetric part of the worst-spectrum
visit form is

```text
h[v]=integral(|grad_perp v|^2
              +(zeta_0+1/2-c_t)|v|^2).                (8)
```

The optimized taper retains the axial stretching direction, so `c_t>=1`.
At `H/L=1.2`,

```text
zeta_0=1.26030185124.                                  (9)
```

Testing (8) on the ground state of the disk of radius `2.75L` gives

```text
h[v]/||v||^2
 <=j_(0,1)^2/2.75^2+zeta_0-1/2
 =1.52502065625.                                      (10)
```

Combining (7) and (10),

```text
(h-I_phi)[v]/||v||^2<=-2.14508953658.                 (11)
```

Hence the complete cubic Fisher term is not relatively form-small with
constant below one. It destroys coercivity in an explicit test direction.
At most `41.55%` of the nominal Fisher load escapes this particular necessary
test; the full load is excluded by a wide margin.

## Numerical cross-check

A nonsymmetric finite-element stress sweep independently reflects the same
crossing. Adding fractions of the nominal cubic Fisher potential to the
`t=1` visit gave stable mesh-converged behavior:

| Fisher fraction | visit norm | generation criterion |
|---:|---:|---:|
| 0 | 0.8040 | 0.2575 |
| 0.1 | 1.1440 | 0.5213 |
| 0.2 | 1.7792 | 1.2610 |
| 0.4 | 10.9219 | 47.5188 |
| 0.5 | sign-changing boundary map | not Markov-positive |

These are finite-element diagnostics, not part of the proof of (11).
They show that renewal contraction is lost much earlier than the analytic
coercivity ceiling.

## Consequence

The wide divergence-free taper solved the affine-shell problem, but it cannot
be combined with a continuously active full tensor cubic square-root
partition in the same visit form. Reintroducing that partition would undo the
gain.

This no-go is narrow. It does not exclude:

1. linear partitions used only to retain exact pressure-flux cancellation;
2. normalized cubic probabilities used at buffered entry and exit stopping
   times;
3. holding one selected cell label fixed through a complete visit;
4. a different partition or a substantially different visit geometry.

The revised architecture is therefore to separate roles. Use continuous
linear weights for pressure bookkeeping if needed, but use conservative
stopping-time labels for visit assignment and do not apply a simultaneous
square-root partition inside the visit. The selected moving cylinder then
sees only its fitted potential and divergence-free drift remainder, governed
by the sector budget in `moving_cubic_label_transport.md`.

The exact Fisher factorization, geometric lower bound, and disk-ground-state
certificate are reproduced by
`scripts/continuous_cubic_localization_no_go_audit.py`.
