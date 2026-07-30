# Annular two-shear static optimizer

## 1. Result

The modified two-shear witness does not inherit the old finite static
optimizer. The second shear creates an exact low-low interaction that is
visible to the `+++` partition weight.

For

```text
U_*=U_yz+U_xy
```

the exact weighted quantities are

```text
mean[Phi_+++ |grad U_*|^2]=17/16,

mean[F(U_*) dot grad Phi_+++]=-sqrt(2)/12,

mean[p[U_*,U_*] U_* dot grad Phi_+++]=-sqrt(2)/48.  (1.1)
```

The first flux is the complete local-energy flux. The second is its
pressure-only part after the kinetic/strain cancellation at `rho=0`.

With the favorable orientation

```text
u_N=h_N-a U_*,       a>=0,
lambda=t Phi_+++,    t>=0,
```

both normalized static objectives contain a positive cubic in `a`.
Consequently, for every fixed carrier `N>=3` and every fixed `t>0`, both
objectives tend to positive infinity as `a` tends to infinity.

This is a route guard. It is not a Navier-Stokes blow-up construction:
the static coefficient is not yet constrained by the backward adjoint, and
the low field's actual nonlinear evolution has not yet been imposed.

## 2. Exact low field

Use the two real plane waves

```text
ell_yz=(0,1,-1),      d_yz=(0,1,1)/sqrt(2),
ell_xy=(1,-1,0),      d_xy=(-1,-1,0)/sqrt(2),
```

with Fourier convention

```text
Uhat_j(+ell_j)=-i d_j,
Uhat_j(-ell_j)=+i d_j.
```

Each wave is divergence free and each shear separately has zero local
energy-flux load. Their Fourier supports are disjoint, so

```text
||U_*||_2^2=4.                                      (2.1)
```

The two individual weighted Fisher terms are each `1/2`. The cross pairs
whose difference is `+-(ell_yz+ell_xy)` lie in the partition stencil and
contribute `1/16`. Therefore

```text
M_*=1/2+1/2+1/16=17/16.                            (2.2)
```

The audit enumerates all ordered Fourier triples symbolically over
`Q(sqrt(2))`. At `+++`, it gives

```text
kinetic low-only load =-sqrt(2)/16,
pressure low-only load=-sqrt(2)/48,
complete low-only load=-sqrt(2)/12.                (2.3)
```

Thus the pressure contribution is one quarter of the complete load. The
remaining three quarters are exactly the kinetic contribution removed in
the `rho=0` generator.

## 3. Support decomposition

The positive high packet has

```text
k_x in [2N,3N-1].
```

The two low waves have first coordinates in `{-1,0,1}`. For `N>=3`:

```text
HHH: every mixed-sign output has |k_x|>=N+1>1;

HLL: every output has |k_x|>=2N-2>1;

high-low Fisher: every difference has |Delta k_x|>=2N-1>1.
```

Hence the `+++` stencil sees:

```text
the high Fisher term D_N,
the HHL term linear in U_*,
the low Fisher term a^2 M_*,
and the low-only cubic flux.
```

There is no HLL load and no high-low Fisher cross term. A complete
dictionary replay at `N=3` checks, for five signed low amplitudes `x`,

```text
L_complete(h_N+xU_*)=x B_comp,N+x^3 C_comp,

L_pressure(h_N+xU_*)=x B_p,N+x^3 C_p,

E_+++(h_N+xU_*)=D_N+x^2(17/16),                   (3.1)
```

to residuals below `9e-16`.

## 4. Exact objectives

Let

```text
B_comp,N=L_HHL,complete(h_N,U_*),
B_p,N=L_HHL,pressure(h_N,U_*),
D_N=mean[Phi_+++ |grad h_N|^2],
Q(delta_+++)=75/256.
```

The finite rows give `B_comp,N<0` and `B_p,N<0`. For `u_N=h_N-aU_*`,
the complete compatible bracket, after omitting the same harmless factor
three as in the preceding audit, is

```text
J_comp,N(a,t)
 =t[-a B_comp,N+(sqrt(2)/12)a^3
    -nu(D_N+(17/16)a^2)]
  -(nu/16)(75/256)t^3.                            (4.1)
```

After the kinetic/strain cancellation, the exact `rho=0` pressure
generator is

```text
J_p,N(a,t)
 =t[-a B_p,N+(sqrt(2)/48)a^3
    -nu(D_N+(17/16)a^2)]
  -(nu/16)(75/256)t^3.                            (4.2)
```

Every term in (4.1)-(4.2) is retained. In particular, the cubic coefficient
penalty has not been discarded.

For fixed `N` and fixed `t>0`,

```text
J_comp,N(a,t)~t*(sqrt(2)/12)a^3,
J_p,N(a,t)~t*(sqrt(2)/48)a^3.
```

Therefore

```text
sup_(a,t>=0) J_comp,N=+infinity,
sup_(a,t>=0) J_p,N=+infinity.                     (4.3)
```

The old stationarity equation

```text
a_N=|B_N|/(nu M)
```

is no longer an optimizer equation because it omitted the now nonzero
derivative of the low cubic.

## 5. Optimization at fixed amplitude

For either objective, let `A_N(a)` be the bracket multiplying `t`. When
`A_N(a)>0`, optimization in `t` alone is still exact:

```text
t_*(a)=sqrt[16 A_N(a)/(3nu Q)],

max_t J_N(a,t)=(2/3)A_N(a)t_*(a).                 (5.1)
```

As `a` tends to infinity,

```text
t_comp,*(a)
 ~sqrt[1024sqrt(2)/(675nu)] a^(3/2),

t_p,*(a)
 ~sqrt[256sqrt(2)/(675nu)] a^(3/2),               (5.2)
```

and both optimized objectives grow like `a^(9/2)`. The pressure scale
constant is one half of the complete constant; its objective constant is
one eighth.

## 6. Carrier scaling

Put

```text
a_N=kappa N^alpha,       kappa>0.
```

The powers in the coefficient-linear margin are

```text
HHL:             N^(alpha+1),
low Fisher:      N^(2alpha),
low self-flux:   N^(3alpha),
high Fisher:     N^(-3).
```

The HHL mechanism dominates the self-flux only when `alpha<1/2`. They have
the same power at `alpha=1/2`; the self-flux dominates for `alpha>1/2`.

The old optimizer used `alpha=1`. On that ray the new scales are

```text
A_N=Theta(N^3),       dominated by low self-flux,
t_*=Theta(N^(3/2)),
max_t J_N=Theta(N^(9/2)).                          (6.1)
```

With bounded `t`, the objective is `Theta(N^3)`. Keeping the old
`t=Theta(N)` instead gives `Theta(N^4)`, because its favorable `N^4` term
dominates the `N^3` coefficient penalty.

For `nu=1`, the finite replay with `a=N` first becomes positive among the
audited sizes at:

```text
complete objective: N=9,
pressure objective: N=49.
```

These finite thresholds are checks, not the source of the unboundedness
theorem.

## 7. HHL sign remains valid

The pressure-HH part still obeys

```text
B_p,N/N -> -(sqrt(2)/20)||b||_L2(D)^2
          =-0.001414088992406...<0.               (7.1)
```

The complete and pressure loads differ only by terms that vanish after
division by `N`. The finite row at `N=49` gives

```text
B_comp,49/49=-0.001514775291273...,
B_p,49/49   =-0.001514775180290....
```

Thus the two-shear replacement did not lose the favorable HHL sign. It
introduced a stronger low-frequency mechanism that changes which term
leads.

## 8. Restart consequence

The preceding one-shear restart gate used a finite static optimizer with

```text
a_N=Theta(N),       t_N=Theta(N),
```

to derive an `Omega(N^3)` reset tax and an `Omega(N^5)` required
time-average generator on a heat window. Those conclusions cannot simply
be copied.

For the two-shear field, Parseval changes the norm-only deficit bound to

```text
Delta_s
 >=(1/2)(
    sqrt(||h_N||_2^2+4a^2)-5t/16
   )_+^3.                                         (8.1)
```

More importantly, there is no canonical finite `a_N`, and coefficient
optimization at `a_N=Theta(N)` gives `t_N=Theta(N^(3/2))`. The old reverse
triangle lower bound can then become vacuous. This does not show that the
true deficit is small; it shows that the old proof of a reset tax no longer
decides it.

## 9. Route decision

Before porting the full finite jets, classify the phase and relative
polarization freedom of the two shears. The sharp next question is:

```text
Can the low-only pressure and complete self-flux be made exactly zero
while the static HHL load and the four-high square remain strictly negative?
```

If yes, the finite optimizer and restart analysis can be rebuilt on that
self-flux-free subfamily. If no, the backward-adjoint restart must retain
the low cubic and the actual nonlinear low-mode evolution from the start.

The production record is

```text
results/annular_two_shear_static_optimizer_audit_v1.json
```

and is reproduced by

```text
python work/ns_collision/scripts/annular_two_shear_static_optimizer_audit.py
```
