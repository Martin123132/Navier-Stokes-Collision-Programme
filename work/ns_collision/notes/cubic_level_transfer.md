# Conservative cubic level transfer

## Purpose

The fixed-frame cubic tensor partition closes the static localization and IMS
problem.
The next question is whether a monotone change from one lattice level to its
dyadic child level creates a factor greater than one per generation.

For a globally uniform level, the cubic two-scale identity supplies an exact
pointwise Markov transfer. Its worst gauge recentering is paid by the genuine
radius-halving term. This removes balance-only interfaces by design; the
price is global rather than spatially local refinement.

## Positive refinement mask

The cardinal cubic identity is

```text
N_3(x-i)=sum_(r=0)^4 a_r N_3(2x-(2i+r)),
a=(1,4,6,4,1)/8.                                     (1)
```

The mask sums to two because a fine spline has half the integral of a coarse
spline. After integral normalization, the child probabilities are

```text
p=(1,4,6,4,1)/16,       sum_r p_r=1.                  (2)
```

The three-dimensional tensor product has 125 labels and total probability
one. Two independent replicas have 15625 label pairs and still total
probability one;
the branch count creates no mass factor.

More strongly, at a fixed point where the parent weight is positive,

```text
P(r|i,x)
 =a_r N_3(2x-(2i+r))/N_3(x-i),                        (3)
```

is nonnegative and sums to one by (1). Thus a history carrying its position
can change labels through a pointwise Markov kernel. No reconstruction of a
child observable from a parent cell average is being assumed.

## Gauge comparison

For the optimized full tensor support `rho_s=1.91`, the knot spacings are

```text
h_perp/L=rho_s/(2 sqrt(2)),       h_z/L=3/4.           (4)
```

A child center differs from its parent by

```text
delta_j=(r_j-2)h/2,
|delta_j|/L<=rho_s/(2 sqrt(2))                         (5)
```

in each transverse direction. Write a child point as
`x=c_child+(L/2)y`. In one coordinate the parent-minus-child gauge exponent,
before multiplication by `R_*/4`, is

```text
(d+y/2)^2-y^2=d^2+d y-3y^2/4.                         (6)
```

Its maximum is `4d^2/3`, attained at `y=2d/3`. The
critical point lies inside every relevant child support. The maximum axial
center offset is `3L/4`. Summing all three directions gives

```text
maximum exponent difference=rho_s^2/3+3/4,
maximum log gauge cost=(R_*/4)(rho_s^2/3+3/4).         (7)
```

At `R_*=0.5`, `rho_s=1.91`,

```text
gauge transition factor:        1.278585,
shrink-paid one-history factor: 0.639293,
shrink-paid replica-pair factor:0.408695.               (8)
```

The exact audit values should be used for calculations. Equation (8) is
rounded only for display. The pair product decays exponentially through 50
audited true level changes.

## Global monotone levels

Use one fixed cubic lattice level `L_n` and reference amplitude

```text
A_n=R_* nu/L_n^2.                                      (9)
```

Keep that level while the running global adverse envelope is below `A_n`.
When it reaches the threshold, change globally to

```text
L_(n+1)=L_n/2,       A_(n+1)=4A_n.                    (10)
```

Every change is then a genuine safety-triggered radius halving. The
contraction in (8) is not assigned to arbitrary extra refinement. Because a
single level covers the fixed-frame three-dimensional grid, there are no
balance-only coarse/fine interfaces and no moving centers between level
changes.

This global construction may over-refine regions far from the largest
strain. That is a complexity cost, not by itself an analytic loss: the
partition has bounded pointwise overlap and the physical label transfer is
Markov. The reference amplitude in (9) dominates the actual adverse
amplitude everywhere until the next threshold.

## Conversion discipline

The gauge comparison (7) acts directly between parent and child visit norms.
The physical-to-Poisson condition number must not be inserted at every level
change. It is paid once at complete buffered-visit entry or exit, while the
intermediate level changes remain in the gauged norm. Repeating a complete
conversion at every split would discard the contraction just proved.

Both lattice levels satisfy `sum phi=1`; equation (3) therefore preserves the
linear pressure partition at the change. Pressure edge flux is not replaced
by independent absolute cell bounds.

## Status and next gate

For envelope-triggered global levels, the parent/child branching and gauge
transfer are now contractive. This is a concrete time-coherent alternative
to the unresolved locally adaptive balanced octree. It does not yet prove
Navier-Stokes regularity.

The remaining decisive estimate is dynamical rather than combinatorial:

```text
||q_+||_(L^(3/2)(complete cubic support))/nu <0.2159   (11)
```

must be derived from actual strain, frame, pressure-Hessian, and transport
quantities on every complete visit. It cannot be inserted as a coherence
assumption. The next stage should decompose `q_+` into those exact
Navier-Stokes terms and test which pieces are controlled by the global
envelope, incompressibility, pressure flux, or accumulated spectral decay.

The mask conservation, pointwise Markov kernel, replica branching, gauge
maximum, and many-generation contraction are reproduced by
`scripts/cubic_level_transfer_audit.py`.
