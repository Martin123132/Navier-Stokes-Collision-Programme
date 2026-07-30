# Finite-cylinder Kato gate

## Purpose

Constant adverse potentials give a useful robustness calibration, but the
Navier-Stokes error is neither constant nor pointwise controlled by its
critical spatial norm. This note identifies the exact positive Green-operator
quantity that is sufficient for the finite-cylinder renewal and tests whether
bare `L^(3/2)` control can supply it.

It cannot. The failure is an endpoint phenomenon, and it sharply separates
the available critical form estimate from the pointwise visit estimate used
by the current renewal argument.

## Resolvent identity

Let `G_0` be the zero-boundary Green operator for the baseline piecewise
finite-cylinder generator, including the affine core stretching. Let `u_0`
be the baseline outer-radial boundary payoff and let `q_+` be an additional
nonnegative adverse potential. Then

```text
u_q=u_0+G_0(q_+ u_q).                                (1)
```

Define the positive Birman-Schwinger/Kato operator

```text
T_q f=G_0(q_+ f),
kappa(q)=||T_q||_(infinity to infinity)
        =sup_x G_0 q_+(x).                            (2)
```

If `kappa<1`, the Neumann series in (1) is positive and gives

```text
||u_q||_infinity <= ||u_0||_infinity/(1-kappa).       (3)
```

If `C_0` is the complete baseline generation criterion, the perturbed
criterion therefore obeys

```text
C_q <= C_0/(1-kappa)^2.                              (4)
```

Consequently the exact sufficient operator budget is

```text
kappa(q) < 1-sqrt(C_0).                              (5)
```

This controls all repeated same-scale visits before the genuine split; it is
not an instantaneous residence-time estimate.

## Budgets at the working geometries

| `R_*` | half-height `h` | `C_0` | critical `kappa` | allowed gain multiplier |
|---:|---:|---:|---:|---:|
| 0.5 | 1.5 | 0.493333 | 0.297623 | 1.42374 |
| 0.5 | 1.75 | 0.691792 | 0.168260 | 1.20230 |
| 0.5 | 2.0 | 0.864787 | 0.0700610 | 1.07534 |
| 1.0 | 1.0 | 0.288761 | 0.462635 | 1.86093 |
| 1.0 | 1.2 | 0.684048 | 0.172928 | 1.20908 |

The preferred `R_*=0.5`, `h=1.5` geometry can tolerate a Green-operator
perturbation of almost `0.3`. Again, choosing a cylinder close to the ideal
height threshold consumes most of the robustness before any PDE error is
introduced.

## Why critical L3/2 mass is insufficient

The local singularity of every smooth three-dimensional uniformly elliptic
Green kernel is Newtonian. It is therefore enough to test the free kernel
`(4 pi |x-z|)^(-1)` near an interior observation point.

For `2/3<alpha<=1`, put

```text
q_T(r)=c_T/[r^2 log(1/r)^alpha],
exp(-T)<r<exp(-1).                                  (6)
```

Its critical mass is determined by

```text
integral q_T^(3/2)
 =4 pi c_T^(3/2) integral_1^T t^(-3 alpha/2)dt.      (7)
```

Because `3 alpha/2>1`, (7) stays bounded as `T` tends to infinity. Choose
`c_T` so that `||q_T||_(3/2)=1` exactly. At the centre, however,

```text
integral q_T(z)/(4 pi |z|) dz
 =c_T integral_1^T t^(-alpha)dt,                    (8)
```

which diverges for `alpha<=1`.

For the audited choice `alpha=3/4`, the normalized centre potentials are:

| `T=log(1/epsilon)` | `||q_T||_(3/2)` | centre potential |
|---:|---:|---:|
| 4 | 1 | 0.260994 |
| 16 | 1 | 0.419489 |
| 256 | 1 | 0.881052 |
| 65,536 | 1 | 3.36184 |
| 100,000,000 | 1 | 19.6487 |

Thus there is no strong endpoint estimate

```text
sup_x G_0 q(x) <= C ||q||_(3/2)                     (9)
```

for all nonnegative `q`. The smooth drift and bounded baseline potential do
not remove this local singularity.

This does not contradict the earlier `L^(3/2)` form bound. That estimate
controls the quadratic form on `H_0^1`; point evaluation of the boundary
payoff is a stronger endpoint operation.

## Viable replacements

Three routes remain mathematically compatible with the result.

1. Prove the scale-invariant Kato bound (5) directly for the positive
   non-affine coherence error.
2. Prove a local Morrey or `L^p`, `p>3/2`, estimate strong enough to imply
   (5). For the free Newtonian kernel, Holder becomes finite exactly above
   this endpoint; a certified comparison constant is still needed for the
   drifted cylinder Green kernel.
3. Replace the pointwise interface maximum by an averaged physical/interface
   norm for which the existing critical form estimate propagates through the
   conservative branching operator.

The third route is especially relevant because the branching audit already
showed that physical mass is conserved and that nonconstant pointwise gauge
weights create the artificial interface amplification. An averaged closure
could align the PDE estimate and the branching norm instead of demanding
endpoint pointwise control.

## Remaining Navier-Stokes gate

The exact unresolved implication is now

```text
local strain/eigenframe coherence
    => kappa(q_+)<1-sqrt(C_0),                       (10)
```

or an averaged replacement for (10). Energy-class `L^(3/2)` form control
alone cannot establish it. The next productive calculation should therefore
construct the finite-cylinder Green operator in the natural Gaussian measure
and test weighted `L^2` or boundary-flux norms that survive the conservative
child/interface transfer.

The operator budgets, endpoint sequence, and supercritical free-kernel
Holder constants are reproduced by
`scripts/finite_cylinder_kato_gate_audit.py`.
