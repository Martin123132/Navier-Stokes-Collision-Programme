# Critical collar-to-form transfer gate

## Purpose

This note uses the legacy bare-halving closure value and is retained for its
off-diagonal and endpoint theorems. Under the current cubic split, the older
`1.342878685` barrier exceeds the closure gain `1.232133608`; its positive
sector-budget table is therefore not a live cycle allowance. A positive
collar does remove the critical endpoint singularity
from the first trip between the radial entry surface and the perturbation.
It does **not** turn bare `L^3/L^(3/2)` control into a global pointwise Kato
bound, because later perturbation iterates begin inside the rough support.

This note proves that distinction and identifies the mixed collar/energy
operator that can still close the cycle.

## Exact renewal budget

The certified numbers are

```text
g_0=1.342878684567,
g_*=1.391948395999,
g_*/g_0-1=0.036540688296.                             (1)
```

If a positive Kato Neumann estimate were available, its exact budget would
be

```text
kappa<1-g_0/g_*=0.035252536354.                       (2)
```

The difference between (1) and (2) is the denominator in the Neumann
series; both encode the same pair-cycle threshold.

## What the collar proves

Let

```text
D={r<2, |z|<0.75},
Sigma={r=1, |z|<0.75},
dist(Sigma,E)>=d,                                    (3)
```

where the adverse potential is supported in `E`. For every admissible
history

```text
B=B^T, tr B=0, B>=-I,
```

the top eigenvalue is at most two. If `Phi` is the affine fundamental
matrix and `Q` is the diffusion covariance, then

```text
||Phi(s)||<=exp(2s),
sigma_min(Phi(s))>=exp(-s),
(1-exp(-2s))I<=Q(s)<=[(exp(4s)-1)/2]I.               (4)
```

Every point of `Sigma` has norm at most `5/4`. Hence the affine mean remains
within `d/2` of its starting point up to

```text
s_d=(1/2)log[1+d/(2(5/4))].                           (5)
```

For `s<=s_d`, the full-space Gaussian and (4) give an explicit restricted
row bound

```text
||1_E k_B(s,x,.)||_3<=H_3(s,d),                       (6)
```

where

```text
H_3=e^s[2 pi q_-(s)]^(-3/2)
    {int_[|y|>=d/2] exp[-3|y|^2/(2q_+(s))]dy}^(1/3),
q_-=1-exp(-2s), q_+=(exp(4s)-1)/2.                   (7)
```

For the long tail, split the killed semigroup at `b=s_d/2`. The exact
Dirichlet floor is `lambda_1=5.832287335665`, so the generator including
the adverse baseline `+1` decays in volume `L^2` at

```text
m_0=lambda_1-1=4.832287335665.                        (8)
```

The free affine row and column both obey

```text
A_2(b)=e^b 2^(-3/2)pi^(-3/4)q_-(b)^(-3/4).           (9)
```

Two semigroup splits and interpolation between row `L^2` and `L^infinity`
therefore give

```text
C_entry(d)
 =int_0^(s_d) H_3(s,d)ds
  +A_2(b)^(4/3)exp[-2m_0b/3]/m_0,                    (10)

sup_(x in Sigma) int_0^infinity int_E
 k_B(s,x,y)q(s,y)dyds
 <=C_entry(d) sup_s||q(s)||_(3/2).                   (11)
```

The audited constants are:

| collar `d` | `C_entry(d)` | mass spending all of (2) |
|---:|---:|---:|
| 0.05 | 1.839900 | 0.0191600 |
| 0.10 | 0.985832 | 0.0357592 |
| 0.20 | 0.544615 | 0.0647293 |
| 0.40 | 0.311410 | 0.113203 |

This is a uniform nonautonomous estimate for the **first** potential
insertion. The last column is diagnostic, not a closure theorem.

## Why the pointwise Neumann route still fails

The global Kato norm required by a pointwise Neumann series is

```text
sup_(x in D) G_B q(x),                                (12)
```

not merely the supremum over `Sigma`. Put a small ball around an interior
point `x_0` of `E` and translate the normalized endpoint sequence

```text
q_T(r)=c_T/[r^2 log(1/r)^(3/4)],
exp(-T)<r<exp(-1).                                    (13)
```

Its `L^(3/2)` norm is one, while the Newtonian part of the Green potential
at `x_0` tends to infinity. Translation leaves the entry collar unchanged.
Thus the second perturbation iterate, whose source can lie in `E`, restores
the endpoint singularity. Equations (10)-(11) cannot be iterated in
`L^infinity` using only the critical norm.

This rules out the tempting implication

```text
positive entry collar + critical mass
    => global pointwise Kato smallness.               (14)
```

## Viable mixed norm theorem

The interior rough iterations must instead be resummed in the coercive
energy space. Put

```text
h[v]=||grad v||_2^2-||v||_2^2.
```

Poincare and sharp Sobolev give

```text
||grad v||_2^2<=(lambda_1/(lambda_1-1))h[v],
S_3=4^(2/3)/(3pi^(4/3)),                              (15)

alpha<=c_A S_3||q_+||_(3/2),
beta <=c_A sqrt(S_3)||e||_3,
c_A=lambda_1/(lambda_1-1)=1.206941336584.             (16)
```

Here `alpha` is the adverse relative form bound and `beta` is the
divergence-free drift sector bound.

There is a uniform nonautonomous interior estimate behind these parameters.
Let `zeta=1` on the perturbation support and let `w=(u-U)_+` be the positive
overshoot above the certified barrier. The baseline barrier residual is
nonpositive, the spatial boundary trace of `w` is zero, and the perturbation
forcing has energy-dual norm at most

```text
F=(alpha+beta)sqrt(h[zeta U]).                         (17)
```

The perturbed homogeneous form has real part at least `(1-alpha)h`; the
divergence-free drift error remains skew in this estimate. Standard
positive-part truncation on finite-horizon approximants gives

```text
(1/2)d||w_+||_2^2/ds+(1-alpha)h[w_+]
 <=F sqrt(h[w_+]),
h[w_+]>=m_0||w_+||_2^2.                              (18)
```

Young's inequality followed by the scalar comparison principle gives

```text
sup_t||w_+(t)||_2
 <=F/[(1-alpha)sqrt(m_0)].                            (19)
```

Passing through the bounded monotone horizon limit and using (17) therefore
proves the uniform causal response bound

```text
sup_t||w(t)||_2
 <=(alpha+beta)/(1-alpha)*sqrt(h[zeta U]/m_0).        (20)
```

This step is valid for arbitrary measurable affine histories. It uses only
the common form floor and the explicit time-independent barrier; it does not
freeze or average `B(t)`.

In the unperturbed collar, `w_+` is a subsolution of the homogeneous affine
equation. Define the remaining local trace constant by

```text
sup_(x in Sigma) w_+(t,x)
 <=C_col(d) sup_s||w_+(s)||_2.                        (21)
```

Uniform parabolic interior/boundary estimates make `C_col(d)` finite for a
positive collar, but its explicit calibrated value has not yet been proved.
Combining (20)-(21) gives the dynamic condition number

```text
chi_dyn=C_col(d)sqrt(h[zeta U]/m_0)/g_0               (22)
```

and the target bound

```text
g_(e,q)/g_0
 <=1+chi_dyn (alpha+beta)/(1-alpha).                  (23)
```

Consequently closure is equivalent to

```text
beta<d_sec-(1+d_sec)alpha,
d_sec=0.036540688296/chi_dyn.                         (24)
```

For illustration, the conditional one-error budgets are:

| `chi_dyn` | potential `||q_+||_(3/2)` | drift `||e||_3` |
|---:|---:|---:|
| 1 | 0.160000 | 0.0708594 |
| 1.5 | 0.107935 | 0.0472396 |
| 2 | 0.0814351 | 0.0354297 |
| 3 | 0.0546168 | 0.0236198 |
| 5 | 0.0329286 | 0.0141719 |
| 10 | 0.0165242 | 0.00708594 |

These are scale-invariant normalized masses. They are conditional targets,
not established Navier-Stokes tolerances.

## Revised gate

The correct architecture is now

```text
dynamic radial entry law
 -> positive unperturbed collar smoothing
 -> critical interior energy/sector resummation
 -> collar-to-dynamic-entry trace
 -> square-tilted pair renewal.                       (25)
```

The critical interior response in (20) is now closed. The next theorem must
certify the local homogeneous trace constant `C_col(d)` in (21), choose a
geometrically admissible cutoff, and evaluate `h[zeta U]`. The affine
covariance estimate (10) supplies a useful off-diagonal majorant for that
trace calculation, but it cannot substitute for the interior form
resummation.

The follow-up calibration finds that `d=0.2-0.3` is numerically plausible:
the minimum cutoff-energy and complete stationary axisymmetric trace pilots
give combined condition numbers below two, even after a small nonzero
temporal-frequency resonance. These are not enclosures and do not cover
switching full-affine histories. The same audit proves that the existing
continuous cardinal-cubic square-root partition cannot realize the protected
support: its IMS cost exceeds the complete form floor by more than a factor
six at `d=0.2`. The localization must therefore occur at stopping or marked
interaction times, with the global drift cancellation preserved.

The endpoint sequence, affine first-insertion constants, exact renewal
margin, and conditional sector table are reproduced by
`scripts/critical_collar_transfer_audit.py`.

## Superseded preferred route

The collar calculations remain valid conditional estimates, but they are no
longer the preferred transfer architecture. The later finite-energy payoff
barrier has both boundary exponents above `1/2`, is interval-certified at
entry gain `1.145614144998`, and permits the complete critical forcing to be
estimated globally without a support cutoff. Its positive overshoot is paired
with the actual unnormalized space-time return law in
`averaged_entry_trace_gate.md`. This bypasses both the protected-support IMS
obstruction and the labelwise skew-cancellation obstruction; it replaces them
with the explicit physical exterior return-density gate.
