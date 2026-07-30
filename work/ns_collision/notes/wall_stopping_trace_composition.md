# Wall-stopping trace composition

## Purpose

The migrating-core residual audit left two abstract response constants
`K_R,K_S`. This note identifies the exact measures behind those constants
and corrects an important surface mismatch: a same-scale return reaches the
H1 entry surface, while a migrated wall point reaches the child's outer
surface.

## Positive weighted branch kernel

Let `B_j` be a positive weighted branch kernel from an entry law `mu` to a
space-time exit surface, and put

```text
m_j(x)=B_j 1(x),
p_j^2=int m_j(x)^2 mu(dx).                            (1)
```

The square-tilted exit law is

```text
nu_j(dy)=p_j^(-2) int mu(dx)m_j(x)B_j(x,dy).         (2)
```

Positivity and Cauchy-Schwarz give the exact operator identity

```text
||B_j||_(L2(nu_j)->L2(mu))=p_j.                      (3)
```

If the H1 positive overshoot `w` satisfies the existing energy estimate and
the square-tilted law has spatial-`L2` interval factor `J_j(alpha)`, then

```text
||B_j w||_2
 <=p_j F sqrt[C_4 J_j(alpha)],                       (4)

C_4=0.674148137961.
```

Therefore the response constant is not free:

```text
K_j=p_j sqrt[C_4 J_j(alpha)].                        (5)
```

Equivalently, if `J_raw,j=p_j J_j` is computed from the raw unnormalized
density for a point entry,

```text
K_j=sqrt[p_j C_4 J_raw,j].                           (6)
```

Equations (5)-(6) show exactly where branch mass belongs. It is neither
discarded by conditioning nor multiplied into the response twice.

## Return calibration

The existing inflated Brownian exterior-cylinder pilot has

```text
J_B(0)=0.796475197211,
sqrt(C_4 J_B(0))=0.732763448277.                      (7)
```

Restricting its raw law to the finite axial patch of mass
`0.310135151371` gives the conservative raw-response calibration

```text
sqrt[p_H C_4 J_B(0)]=0.408074346823.                 (8)
```

This is a convergence pilot, not a certified affine-strip constant. For a
positive potential with relative-form parameter `alpha`, retaining the
`alpha=0` optimizing time window gives the safe algebraic inflation

```text
J_B(alpha)<=J_B(0)/(1-alpha)^3.                      (9)
```

Under the explicit hypothesis that the normalized migrating return laws
obey this stressed Brownian envelope, the audit computes angle-resolved
return-only critical-norm thresholds. They are model calibrations until the
actual weighted strip flux is bounded.

## The wall surface mismatch

A same-scale return lands on

```text
Sigma={r=1, |z|<3/4},                                (10)
```

which is exactly the H1 trace surface. A migrating wall point instead lands
at child radius

```text
r_child=2.                                           (11)
```

That is the child's outer storage surface, not `Sigma`. Consequently the raw
wall flux cannot be inserted into (4) as an H1 entry law. The relevant law
for a later child-core perturbation is the composite kernel

```text
B_S^core
 =B_wall M_migrate B_child-return.                   (12)
```

Only a space-time density estimate for (12) can define `K_S`. The migration
and relabeling maps preserve positive mass but do not create smoothing; the
child storage return is the stage that can create an H1 trace density.

## Two error channels

This separates errors that the previous unit-response calibration grouped
together:

1. nonaffine core errors are paid through (4) when a branch reaches `r=1`;
2. the residual generated during smooth centre/scale interpolation occurs
   only on the wall branch.

For the working scalar rows, a wall-only multiplicative residual has the
larger exact action allowance

```text
E_wall<0.296761039858,                               (13)
```

instead of the common-branch allowance `0.198654065723`. This does not bound
the physical migration residual; it puts it in the correct branch budget.

## Next theorem

The next calculation should augment the neutral-strip generator with
boundary location and time-resolved flux. It must produce:

```text
rho_R(t)=sup_entry ||k_R^tilt(t,.)||_L2(Sigma),
rho_S(t)=sup_entry ||k_(wall-migrate-child-return)^tilt(t,.)||_L2(Sigma),
```

with summable interval suprema. The return computation is one stopping
kernel. The wall computation is necessarily a two-stage composition. Once
those are available, (5) supplies actual `K_R,K_S` and the positive-root
budget can be evaluated without a unit-response placeholder.

The kernel identity, branch arithmetic, and stored Brownian calibration are
reproduced by `scripts/wall_stopping_trace_composition_audit.py`.
