# Finite-energy radial-payoff supersolution

## Purpose

The certified radial-payoff barrier closes the ideal affine visit, but its
axial exponent `7/20` puts its gradient outside global `L2`. That forces the
critical drift error to be separated from the absorbing boundary by a
protected collar. The continuous collar partition and naive interaction
marking audits show that such support cannot be obtained for free.

This note records a different radial-payoff supersolution with both corner
exponents above `1/2`. It has finite global Dirichlet energy and substantially
more renewal room. It was found by a dense-grid pilot and is now certified by
outward-rounded interval subdivision and transformed boundary-strip bounds.

## Candidate

Put

```text
s=r/2,                 c=cos(2 pi z/3),
A(s)=(197/200)s^2+(3/200)s^32,
Phi(c)=(13/10)c^(69/100)-(3/10)c^(169/100),

U_H(r,z)=A(s)+(11/10)(1-s^2)^(7/10) Phi(c).          (1)
```

The boundary inequalities are exact:

```text
U_H(2,z)=1,
U_H(r,+/-0.75)=A(r/2)>=0.                             (2)
```

Both singular exponents satisfy

```text
7/10>1/2,             69/100>1/2.                   (3)
```

Consequently the radial and axial boundary-gradient singularities are
square integrable. In particular `U_H` belongs to `H1(D)` on the complete
cylinder, unlike the first certified barrier.

## Dense HJB audit

For the strengthened Bellman operator

```text
F[U]=Delta U+(y.grad U)/2
      +(3/2)|y||grad U|+1.005 U,                     (4)
```

a grid containing 2,001 radial points and 1,551 points on the axial
half-domain, with geometric refinement to `10^-8` and `10^-9` at the two
boundary pieces, gives

```text
max F[U_H]                  = -0.01295988233,
max linear part             = -0.1987826054,
min(L^2-(9/4)|y|^2|grad U|^2)= 0.01070157267.        (5)
```

The worst residual occurs near `(r,z)=(1.36615,0.15300)`, well inside the
domain. These values first supplied falsification evidence.

The sign is now interval-certified on the complete open half-cylinder. Four
physical-coordinate regions cover `r<=1.999`, `z<=0.7495`. In the remaining
open strips, put

```text
x=1-r^2/4,              a=cos(2 pi z/3).              (6)
```

Multiplication by the positive weight

```text
x^(2-7/10) a^(2-69/100)                              (7)
```

removes every negative power. The two corner cases `a>=x` and `x>=a`, after
division by `a^2` or `x^2`, have continuous interval extensions to the
corner and strictly negative residual. Across all eight regions, 43,228
boxes were evaluated with no unresolved box and no exhausted budget. This
certifies the stronger `1.005 U_H` operator and a uniform Doob killing rate
`0.005`.

## Renewal gain

The maximum on the entry interface is explicit:

```text
g_H=U_H(1,0)=1.145614144998.                          (8)
```

After charging the current cubic parent-child recentering factor
`p_S=0.639292608019` and the legacy one-history return factor `p_R=1/2`,
the common-gain closure threshold is

```text
g_*=1.232133608495,
g_*-g_H=0.086519463497,
g_*/g_H-1=0.075522342208.                            (9)
```

The ideal complete-generation criterion becomes

```text
C_H=(0.408695038668+0.25)g_H^2
   =0.864492294975,                                  (10)
```

leaving `0.135507705025` at pair level. The former coefficient
`0.516123614725` omitted the cubic recentering cost and is only a legacy
bare-halving calibration. Under the corrected coefficient the first
certified candidate has criterion `1.187840019571` and no longer closes;
the finite-energy candidate is essential.

## Global critical forcing

Separated quadrature gives

```text
||U_H||_6       =1.702134272446,
||grad U_H||_2  =6.546424464264.                     (11)
```

Let

```text
h[v]=||grad v||_2^2-||v||_2^2,
c_A=lambda_1/(lambda_1-1)=1.206941336584,
S_3=4^(2/3)/(3 pi^(4/3)).                            (12)
```

For `v` with zero boundary trace, Holder, Sobolev, and Poincare give

```text
|<q U_H,v>|+|<e.grad U_H,v>|
 <=F_H sqrt(h[v]),                                   (13)

F_H=0.798968551320 ||q||_(3/2)
   +3.072840583265 ||e||_3.                          (14)
```

The adverse potential has relative form bound

```text
alpha<=0.220329037686 ||q||_(3/2).                   (15)
```

Thus the nonautonomous positive-part energy argument can be run globally;
no cutoff `zeta`, IMS cost, or protected perturbation support appears in
(11)-(13). The divergence-free drift error remains skew in the homogeneous
form.

## Remaining gate

Finite global energy does not give pointwise control on the entry surface.
The correct next norm is the actual unnormalized space-time return law. If
that law smooths an outer exit into a density on the inner interface, the
energy response can be paired with the ordinary `H1` surface trace and time
averaging. Early nonreturns retain their sub-Markov contraction because the
law is never conditioned on returning.

The resulting conditional theorem and density budgets are developed in
`averaged_entry_trace_gate.md`. The formula, dense HJB stress test, norm
quadrature, and renewal arithmetic are reproduced by
`scripts/radial_h1_payoff_supersolution_pilot.py`; the complete interval
certificate is reproduced by
`scripts/radial_h1_payoff_interval_certificate.py`.
