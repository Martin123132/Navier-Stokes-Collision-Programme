# Conservative moving cubic labels

## Purpose

The divergence-free shell and sectorial Poisson theorem leave a concrete
geometric problem. Each coherent cell should translate with its local mean
velocity and rotate with its fitted affine frame, but independently moving
cubic templates no longer sum to one. Holding the cells fixed instead creates
a Galilean-dependent error.

This note gives an exact conservative construction and tracks its full
parabolic cost. The main conclusion is deliberately two-sided:

1. translation and rigid label motion can be removed without a cellwise
   potential charge;
2. viscosity produces a Fisher/IMS correction that cannot be erased by
   calling the labels Markovian.

The first point closes the algebraic Galilean gate. The second identifies the
next quantitative localization gate.

## Normalized rigid templates

At one fixed dyadic level let

```text
y_j=O_j(t)^T[x-c_j(t)]/L,
psi_j=Psi(y_j),
Z=sum_k psi_k,
phi_j=psi_j/Z.                                        (1)
```

Here `O_j` is orthogonal, `L` is fixed between true level changes, and `Psi`
is a nonnegative compact cubic tensor template. Whenever `Z>0`,

```text
sum_j phi_j=1,       sum_j grad(phi_j)=0.              (2)
```

Normalization does not enlarge a support: `supp(phi_j)` is contained in
`supp(psi_j)`. Consequently each active label still carries a definite
translated and rotated visit cylinder. A positive lower bound for `Z` is a
real coverage hypothesis; independent cell trajectories do not guarantee it
for arbitrarily long times.

The rigid template velocity is

```text
V_j=c_j'+O_j'O_j^T(x-c_j).                            (3)
```

Since `O_j'O_j^T` is skew, `V_j` is divergence-free. For the physical
backward drift `b`, direct differentiation gives

```text
D_b y_j=O_j^T(b-V_j)/L.                               (4)
```

Where `psi_j>0`, set `g_j=D_b log(psi_j)`. Then

```text
D_b phi_j=phi_j(g_j-g_bar),
g_bar=sum_k phi_k g_k.                                (5)
```

The right side is centered, so all moving-cutoff transport terms cancel when
the complete label family is retained. In particular, a common Galilean
velocity appears in both `b` and `c_j'` and disappears from (4).

## Conservative label flux

Equation (5) is a tangent vector `a_j=D_b phi_j` to the probability simplex.
Any such vector has an explicit positive flux representation. Put

```text
A=sum_k (a_k)_+=(1/2)sum_k |a_k|,
F_jk=(-a_j)_+(a_k)_+/A.                               (6)
```

Only losing labels send mass and only gaining labels receive it. For
`phi_j>0`, set `Q_jk=F_jk/phi_j` off the diagonal and choose the diagonal so
that each row sums to zero. Then

```text
phi Q=a.                                               (7)
```

At `phi_j=0`, nonnegativity implies that an admissible differentiable path
cannot have `a_j<0`; such a state only receives mass. Thus (6) extends by
continuity. The induced Markov map is contractive in the actual evolving
label laws by Jensen, for one history and for the independent replica pair.
No factor proportional to the number or rate of label switches appears.

This statement concerns conservative transport. It is not yet the complete
viscous localization identity.

## The viscous correction

For the parabolic generator

```text
L=partial_t+b dot grad+nu Delta,
```

choose `Q` so that `phi Q=L phi`. For label observables `F_j`, the exact
intertwining identity is

```text
L(sum_j phi_j F_j)
 =sum_j phi_j[L F_j
              +2nu grad(log phi_j) dot grad F_j
              +sum_k Q_jk F_k].                       (8)
```

The extra drift in (8) is the carré-du-champ of diffusion. It vanishes for
pure material transport but not for Navier-Stokes viscosity. In the quadratic
energy formulation the same term is the exact Fisher/IMS cost

```text
I_phi=sum_j |grad sqrt(phi_j)|^2
     =(1/4){sum_j phi_j|grad log psi_j|^2
             -|sum_j phi_j grad log psi_j|^2},         (9)

sum_j |grad(sqrt(phi_j)f)|^2
 =|grad f|^2+I_phi |f|^2.                              (10)
```

Thus normalized moving labels preserve mass and pressure cancellation, but
they do not make spatial localization free. The old fixed-lattice cubic IMS
constant is the special case `Z=1`. For independently moving frames, (9)
must be monitored or built into a recalibrated visit operator. Near the edge
of a cubic support `grad log(phi_j)` is singular, while its weighted Fisher
energy remains finite; treating it as an ordinary unweighted `L^3` drift
would be the wrong estimate.

## Fitted remainder and sector budget

Let `b_ref,j` be the divergence-free tapered affine shell in the moving frame
and define

```text
e_j=b-c_j'-O_j'O_j^T(x-c_j)-b_ref,j.                  (11)
```

With fixed `L`, every subtracted field is divergence-free. Therefore `e_j`
is a genuine first-order sector perturbation. If

```text
Q_j=||[c_actual-c_ref,j]_+||_(3/2)/nu,
E_j=||e_j||_3/nu,                                     (12)
```

the working sector theorem gives

```text
sqrt(S_3) E_j+(1+d)S_3 Q_j<d,
S_3=0.182551571487,
d=0.204278867283.                                     (13)
```

Equivalently,

```text
E_j+0.514540842593 Q_j<0.478113110829.                (14)
```

Translation is subtracted before (12), and frame rotation is charged once,
through `e_j`. Neither is also inserted into the zero-order potential.

On the current cylinder `rho<2.75L`, `|z|<1.2L`, the critical consequences
are sharp enough to matter:

```text
||U||_3/nu
 =3.84894909559 |U|L/nu,

||omega e_z cross (x-c)||_3/nu
 =7.79880733764 |omega|L^2/nu.                        (15)
```

If either were the only error, (14) would require respectively

```text
|U|L/nu<0.12421913,
|omega|L^2/nu<0.06130593.                             (16)
```

For an arbitrary rotation axis, the elementary cylinder bound gives the
stricter sufficient threshold `|omega|L^2/nu<0.04140063`. These numbers show
why local translation must be removed exactly and why uncontrolled
eigenframe rotation cannot be waved away as skew.

Continuous scale motion is also not free. If `L'/L=ell`, then
`V_scale=ell(x-c)` has divergence `3ell`; expansion creates the adverse
zero-order form term `3ell/2`. The present architecture therefore keeps `L`
fixed between envelope-triggered dyadic halvings and uses the already proved
positive cubic child kernel at those discrete events.

## What is now closed

The moving-cell bookkeeping has an exact answer:

1. normalize the independently translated and rotated templates;
2. retain their centered motion as conservative label flux;
3. subtract rigid cell velocity before defining the physical drift remainder;
4. charge only the stretching excess to `alpha` and only the divergence-free
   remainder to `beta`;
5. retain the Fisher/IMS term produced by viscosity.

The construction preserves the linear pressure partition at every time, so
the pressure terms remain antisymmetric inter-cell fluxes.

## Remaining theorem gates

This is not yet a Navier-Stokes regularity theorem. Three obligations remain
before the moving localization can be inserted into the visit renewal:

1. construct centers and frames at Leray regularity, preferably from smooth
   local averages, and handle repeated strain eigenvalues with projectors;
2. prove a positive lower bound for `Z` and an absorbable bound for the
   time-dependent Fisher cost (9), with conservative reseeding before either
   fails;
3. derive (14) from the actual velocity, strain, and pressure-driven frame
   evolution rather than assuming affine coherence.

The normalized partition, simplex flux, evolving-measure contraction,
parabolic intertwining, Fisher identity, and numerical sector constants are
reproduced by `scripts/moving_cubic_label_transport_audit.py`.
