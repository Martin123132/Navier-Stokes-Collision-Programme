# Pressure as conservative inter-cell flux

## Purpose

The localized pressure commutator is not absolutely absorbable at a fixed
scale, and its collision defect retains nonlocal tails. Estimating every core
shell independently therefore discards the strongest exact fact: pressure is
globally orthogonal to strain.

A partition of unity retains that cancellation. It converts pressure from an
uncontrolled source in each cell into a conservative transfer between cells.
The remaining issue is not the absolute pressure tail but mismatch between
the weights assigned to neighboring cells.

## Exact partition identity

Let smooth functions `phi_alpha` satisfy

```text
sum_alpha phi_alpha=1,
sum_alpha grad(phi_alpha)=0.
```

For `P=Hess p` and incompressible `u`, each cell obeys

```text
I_alpha
 =integral phi_alpha P:S dx
 =-integral grad p dot [(grad u)^T grad(phi_alpha)] dx.
```

Summing gives

```text
sum_alpha I_alpha=integral P:S dx=0.                    (1)
```

The same statement holds separately for the heat-smoothed low pressure and
the collision defect. Thus the shell terms can be large and signed in an
individual cell, but their sum is exactly zero.

For cell coefficients `w_alpha`, define

```text
W=sum_alpha w_alpha phi_alpha.
```

Then

```text
sum_alpha w_alpha I_alpha
 =integral W P:S dx
 =-integral grad p dot [(grad u)^T grad W] dx.           (2)
```

Adding one constant to all `w_alpha` leaves (2) unchanged because of (1).
Only relative cell weights matter.

For the tensor-product partition this can be made genuinely local in the
cell graph. Pair the two cells that differ only in bit `j`. Since the
derivatives of `(1+cos)/2` and `(1-cos)/2` are opposite,

```text
sum_alpha w_alpha I_alpha
 =sum_(cell edges e) (w_e,- - w_e,+) J_e,                (3)
```

where `J_e` is the pressure flux formed from the common factors in the other
two coordinates. There are twelve edges in the eight-cell cube. Equation (3)
shows that only neighboring coefficient differences occur; no global
maximum-minus-minimum estimate is forced by the algebra.

## Periodic eight-cell audit

The smooth adversarial field from the pressure-frame counterexample is
partitioned into eight tensor-product cells. In each coordinate the two
factors are

```text
phi_+=(1+cos(x_i-x_i^*))/2,
phi_-=(1-cos(x_i-x_i^*))/2.
```

The eight products sum to one and their gradients sum to zero to machine
precision. The audit verifies, for full pressure, low pressure, and the
collision defect:

1. every cell contraction equals its boundary commutator;
2. individual cell pressure transfers are nonzero;
3. all eight transfers sum to zero;
4. the low and defect transfers recombine cell by cell;
5. arbitrary weighted sums agree with the single effective cutoff `W`;
6. a common shift of all cell weights is invisible.
7. the weighted defect equals the sum of twelve neighboring edge fluxes.

This stress test matters because it uses the same field that defeats the
pointwise maximum principle and fixed-shell sign. The cancellation is not a
feature of favorable data.

## Consequence for the adaptive programme

The adaptive envelope assigns different local scales and potentially
different gauge norms to different coherent cores. If each shell pressure
term is estimated in absolute value, the cubic local-Reynolds obstruction
returns. Equation (2) suggests a different bookkeeping rule:

```text
retain pressure transfers until neighboring cell contributions are paired.
```

With equal weights they cancel exactly. With unequal weights the surviving
term depends on the variation of `W`, hence on differences between neighboring
weights. A viable multiscale construction should therefore seek one of:

1. a common global deformation weight across all partition cells;
2. comparable neighboring envelope weights, with the mismatch paid by the
   strict `R_*<=2` spectral margin;
3. a flux-form discrete energy in which pressure transfers are antisymmetric
   across cell interfaces.

This does not close the problem. A time-dependent partition contributes
transport terms, overlapping cells must preserve affine coherence, and the
exterior cell still needs a weighted renewal estimate. It does remove one
unnecessary loss: nonlocal pressure need not be bounded independently in
every shell.

`intrinsic_radius_cover.md` constructs a Lipschitz minorant of the adaptive
diffusion radius. Overlapping enlarged cells then have quantitatively
comparable radii and reference envelopes, giving a concrete geometric input
for the neighboring edge-weight gate in (3).

The eight-cell identities and weighted mismatch checks are reproduced by
`scripts/pressure_partition_flux_audit.py`.
