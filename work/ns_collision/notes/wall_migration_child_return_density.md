# Wall-migration child-return density pilot

## Purpose

The raw strip-wall law lands at child radius `2`, while the H1 perturbation
trace lives at child radius `1`. This note computes the missing two-stage
model kernel

```text
B_S^core=B_wall M_migrate B_child-return.            (1)
```

It is the final bounded stage of this session. The result is a static
finite-state pilot, not a continuum or Navier-Stokes certificate.

**Later correction.** The child-return boundary law inherits the atomic
fitted-edge histogram obstruction proved in
`neutral_strip_boundary_density_discretization_no_go.md`. The `K_S` value
below is therefore a coarse finite-state diagnostic, not a continuum upper
bound, even before the stated time, tail, and mesh errors are addressed.

## Scale-normalized composition

Let `t_w` be parent-normalized time to the wall and `t_c` child-normalized
time from the migrated outer point to the child core. Since
`L_child=L_parent/2`, total time in child units is

```text
u=4t_w+t_c.                                          (2)
```

The transverse wall point `w=(x,+/-Y)` maps to child entry direction

```text
n=w/|w|,             y_child=2n.                     (3)
```

The axial coordinate doubles at the scale change. At `rho=0`, starting from
`z=0`, the variance at the child core is exactly

```text
V(t_w,t_c)
 =4 exp(2t_c)[exp(2t_w)-1]+exp(2t_c)-1.              (4)
```

The complete smooth-tracking and one-conversion factor is

```text
M_migrate=exp(1/8)2^(-3/4)=0.673774101371.           (5)
```

## Discrete kernel

The parent semigroup is propagated to boundary-resolved wall bins. Each bin
is mapped by (3) to one of 16 child-entry angles. A second semigroup propagates
those child starts to 16 fitted inner-circle bins. The parent and child time
steps are chosen so

```text
4 Delta t_w=Delta t_c=0.1,                           (6)
```

making (2) an exact discrete convolution index.

Near the top and bottom entry points, bilinear interpolation contains an
immediate wall atom. The audit carries that atom separately through the child
return semigroup. It is not discarded or treated as an absolutely continuous
parent wall time.

For fixed total time `u`, different wall times produce different axial
variances. Rather than combine those Gaussian mixtures optimistically, the
pilot sums their spatial `L2` norms by Minkowski. The resulting density is an
upper bound for the discrete time mixture.

## Three-mesh result

The `24`, `30`, and `36` interval rows were evaluated. On the two finest
meshes, the maxima change by

```text
composite scalar gain       0.00934425,
raw interval factor         0.01717318,
trace response K_S          0.00107099.               (7)
```

The working `36`-interval maxima are

```text
composite scalar gain       0.070758384246,
raw interval factor upper   0.120449905356,
K_S(alpha=0)                0.073608663429.            (8)
```

The small `K_S` is plausible: a wall path must survive migration and then
make a finite-patch child return before it can sample an H1 core error. It
should not be interpreted as a proved physical suppression.

The audit inserts the angle-resolved response into the existing pair
criterion while retaining the older, larger wall baseline. Potential rows
use the conservative factor `(1-alpha)^(-3)` for the interval response.
The resulting model-only wall-core thresholds are

```text
||q_res||_(3/2)<1.913793783533,
||e_res||_3      <1.131396524524.                    (9)
```

Their size records the small mass of the two-stage composite law. It is not
evidence that an actual migration residual is automatically small.

## Scope

The following remain numerical rather than certified:

1. boundary-angle and time-step limits;
2. the finite-time tail fit;
3. the continuum limit of both component semigroups and their convolution;
4. uniformity over `0<=rho<=1`, changing frames, and nonaffine drift;
5. the actual migration residual `q_res` during the interpolation.

Thus this stage supplies the previously missing finite-state `K_S` model,
but it does not close the wall-stopping trace theorem. The reproducible
calculation is in
`scripts/wall_migration_child_return_density_pilot.py`.
