# Quadratic partition IMS budget

## Purpose

The Poisson cutoff theorem requires the dangerous non-affine potential to be
inside a protected collar. An intrinsic cover can provide that support only
after localizing the energy. This note computes the exact IMS price of a
quadratic dyadic partition and checks whether it fits inside the available
spectral margin.

This calculation treats three simultaneously active tensor directions. It is
a conservative cube-localization benchmark, not the sharp geometry for the
later Poisson theorem: that theorem requires radial support but already
allows arbitrary axial dependence. `radial_cubic_partition.md` exploits this
distinction with a two-dimensional transverse partition whose square
supports are fitted inside the radial collar.

The standard simultaneous narrow octree partition fails this test. Wider
overlap survives. Sequential coordinate splits have a lower cost per stage,
but do not constitute a solution unless the stages are genuinely separated.

## Quadratic partition

Across a transition of width `omega L`, put

```text
chi_0=cos theta,
chi_1=sin theta,
theta'=pi/(2 omega L).                                 (1)
```

Then

```text
chi_0^2+chi_1^2=1,
|grad chi_0|^2+|grad chi_1|^2=pi^2/(4 omega^2 L^2).    (2)
```

For `d` simultaneously active tensor directions, the `2^d` products satisfy

```text
sum_bits chi_bits^2=1,
sum_bits |grad chi_bits|^2
 =d pi^2/(4 omega^2 L^2).                              (3)
```

The three-dimensional grid audit verifies both identities to approximately
`10^(-14)`.

For the reversible weighted cylinder form, the IMS identity is

```text
sum_j a_0[chi_j u,chi_j u]
 =a_0[u,u]
  +integral w sum_j|grad chi_j|^2 |u|^2.               (4)
```

The drift introduces no extra first-order term because it has already been
placed in divergence form.

## Available margin

At `R_*=0.5`, the conservative transverse one-history margin is

```text
m L^2/nu=5.29680963.                                   (5)
```

The strict minimum transition width that leaves positive margin is

```text
omega_min=(pi/2)sqrt(d/5.29680963).                    (6)
```

| active directions `d` | strict minimum `omega` |
|---:|---:|
| 1 | 0.682516 |
| 2 | 0.965223 |
| 3 | 1.182152 |

Thus a simultaneous three-direction transition of width `L` cannot be
absorbed.

## Unit-width comparison

At `omega=1`:

| active directions | IMS cost | remaining margin | Poisson-compatible `Q/nu` |
|---:|---:|---:|---:|
| 1 | 2.467401 | 2.829409 | 1.111897 |
| 2 | 4.934802 | 0.362007 | 0.399980 |
| 3 | 7.402203 | -2.105394 | 0 |

The final column first recomputes the sharp Sobolev relative-form budget from
the margin left after IMS, then applies the full Poisson factor
`alpha<0.274717`.

The two-direction unit transition is technically positive but leaves little
room. One active direction retains a serious scale-invariant mass budget.

## Wider simultaneous partition

A three-direction transition with width `1.5L` has

```text
IMS cost:             3.289868,
remaining margin:     2.006941,
admissible Q/nu:      1.004409.                        (7)
```

At width `1.25L`, it barely survives and permits only `Q/nu<0.53984`. At
width `2L`, the budget improves to `1.16642`.

The geometry therefore has one directly justified implementation and one
conditional alternative:

1. use a broad simultaneous octree partition, preferably around `1.5L` or
   wider;
2. investigate a hierarchical partition in which parent collar states remain
   active while children cover only the protected interior.

Merely writing the three tensor factors as sequential binary operations does
not save margin. If they are applied at the same time, the final cutoffs are
the same products and equation (3) adds all three costs back to `7.402203`.
The nominal one-direction budget `Q/nu<1.111897` becomes usable only if
parent-buffer states or complete intermediate visits genuinely prevent that
sum. The former needs a new non-leaf partition; the latter would incur extra
renewal and gauge bookkeeping. Neither has yet been proved.

## Pressure compatibility

Define the linear localization weights

```text
phi_j=chi_j^2.
```

Then `sum phi_j=1`, so all exact pressure partition identities remain valid.
Pressure continues to be an antisymmetric neighboring-cell flux; the IMS
calculation does not require taking independent absolute values of the cell
pressure terms.

## Remaining cover gate

The next construction must combine:

1. the monotone balanced dyadic cover;
2. quadratic overlaps satisfying the width constraints above;
3. broad simultaneous splits, or a separately justified hierarchy retaining
   parent collars;
4. one Gaussian/ground-state conversion per complete buffered visit;
5. pressure edge cancellation under the squared partition weights.

A time-dependent partition also creates transport terms. Those are not
included in the static IMS identity and must be bounded using the monotone
radius envelope or eliminated by fixed dyadic centers.

The tensor identities, minimum widths, residual margins, and adjusted
critical mass budgets are reproduced by
`scripts/quadratic_partition_ims_budget_audit.py`.
