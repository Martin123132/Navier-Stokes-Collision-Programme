# Heat-Normal-Form Resummation

Date: 2026-07-17

Status: exact graded resolvent identity, rigorous finite-Galerkin smallness
majorant, and exact identification of the first two nonzero two-mode
coefficients. The resummation is perturbative under the available bounds and
does not provide a cutoff-uniform large-data construction.

## 1. Graded Functional Operators

Let the linear and Euler directional derivatives of a functional `F` be

```text
L_0 F(u)=DF(u)[Delta u],
L_1 F(u)=DF(u)[B(u,u)],
B(u,v)=-P[(u dot grad)v].                         (1.1)
```

For a mean-zero homogeneous Fourier polynomial, define the heat homotopy

```text
H F(u)=integral_0^infinity F(exp(r*Delta)u)dr.    (1.2)
```

Every monomial has positive total heat frequency, so

```text
L_0 H F=-F.                                       (1.3)
```

Start with

```text
F_0=D_s,
P_n=H F_n,
F_(n+1)=L_1 P_n.                                 (1.4)
```

The previous functionals are

```text
P_0=J_s,    F_1=X_s,
P_1=K_s,    F_2=Y_s,
P_2=M_s,    F_3=Z_s.                             (1.5)
```

Define the linear operator on the graded space of functionals

```text
A=H L_1.                                          (1.6)
```

Then

```text
P_(n+1)=A P_n,
P_n=A^n P_0.                                      (1.7)
```

Although `A` is linear as an operator on functionals, it raises homogeneous
degree by one.

## 2. Exact Telescoping and Formal Resolvent

Along Navier-Stokes,

```text
(nu*L_0+L_1)P_n=-nu*F_n+F_(n+1).                 (2.1)
```

For the partial correction

```text
S_N=sum_(n=0)^N nu^(-n)P_n,                      (2.2)
```

all intermediate transfers cancel exactly:

```text
(nu*L_0+L_1)S_N
 =-nu*D_s+nu^(-N)F_(N+1).                        (2.3)
```

Introduce the generating functional

```text
S(z)=sum_(n=0)^infinity z^n P_n.                 (2.4)
```

Formally,

```text
S(z)=P_0+z*A*S(z),
S(1/nu)=(I-A/nu)^(-1)P_0.                        (2.5)
```

If the series and the remainder in (2.3) converge appropriately, then

```text
(nu*L_0+L_1)S(1/nu)=-nu*D_s.                     (2.6)
```

The resummation is therefore exactly a Neumann resolvent for the nonlinear
Liouville generator, not a separate sign identity.

## 3. Tree Growth and Heat Denominators

At transfer order `n`, the homogeneous degree is `n+3`. Every derivative may
split any current velocity leaf, so the number of ordered trees is

```text
T_n=3*4*...*(n+2)=(n+2)!/2.                      (3.1)
```

The generated counts are

```text
n:       0    1    2    3    4
T_n:     1    3   12   60  360.                  (3.2)
```

This factorial count alone does not force factorial divergence. The newest
heat primitive divides a degree-`d` monomial by the sum of `d` nonzero
squared frequencies, which is at least `d`; the derivative that creates the
next transfer contributes exactly `d` leaf choices. These two factors cancel
in the elementary majorant.

## 4. Rigorous Finite-Galerkin Majorant

Restrict to mean-zero modes with Euclidean wave number at most `K_max`, and
use the Fourier `l^1` norm

```text
U=||u||_1=sum_k |u_k|.                            (4.1)
```

The projected Euler convolution obeys

```text
||B(u,v)||_1<=K_max*||u||_1*||v||_1.             (4.2)
```

Let `||F||_c` denote the coefficient norm of a homogeneous polynomial. For a
degree-`d` polynomial,

```text
||H F||_c<=||F||_c/d,                             (4.3)
```

while differentiation contributes at most `d`. Thus

```text
||F_(n+1)||_c<=K_max*||F_n||_c,
||F_n||_c<=||D_s||_c*K_max^n.                    (4.4)
```

Put

```text
q=K_max*U/nu.                                     (4.5)
```

Then

```text
nu^(-n)|P_n(u)|
 <=||D_s||_c*U^3*q^n/(n+3).                      (4.6)
```

The majorant series is

```text
sum_(n=0)^infinity q^n/(n+3)
 =[-log(1-q)-q-q^2/2]/q^3,       0<=q<1.         (4.7)
```

The residual in (2.3) satisfies

```text
nu^(-N)|F_(N+1)(u)|
 <=||D_s||_c*K_max*U^4*q^N.                      (4.8)
```

Therefore the hierarchy converges and telescopes in every fixed Galerkin
system under the explicit smallness condition

```text
K_max*||u||_1<nu.                                 (4.9)
```

This is a genuine theorem but not a Clay-scale advance. The condition becomes
worse as the cutoff grows and gives no uniform passage to arbitrary smooth
three-dimensional data.

## 5. Exact Two-Mode Coefficient Match

The denominator-preserving tree generator evaluates the primitive endpoints
on the negative two-mode seed exactly. Translation parity kills the odd
degrees:

```text
P_0=0,
P_2=0.                                            (5.1)
```

The first nonzero endpoint is

```text
P_1=(1-x)^2*(x^3+2*x^2+3*x-11)/120=c_4(x).       (5.2)
```

The next one is

```text
P_3=(1-x)^2/196560000
 *(6500*x^11+13000*x^10+19500*x^9+40625*x^8
   +61750*x^7+82875*x^6+104000*x^5+125125*x^4
   +689807*x^3+1254489*x^2+1819171*x+16386753)
 =c_6(x).                                         (5.3)
```

Equation (5.3) is reconstructed independently from the 60 primitive trees
using exact Gaussian-rational arithmetic. It agrees identically with the
order-six Duhamel coefficient.

Thus the previously observed expansion

```text
c_4 R^4+c_6 R^6+...                               (5.4)
```

is literally the beginning of the Neumann hierarchy (2.5). The positive
beat-mode return is the next resolvent coefficient, not evidence of a hidden
finite-stage positivity principle.

## 6. Nonperturbative Extension Is the Hard Problem

Suppose, only for this paragraph, that a global smooth decaying
Navier-Stokes flow `Phi_t(u)` is already available and that the following
integral converges. Then

```text
S_trajectory(u)=nu*integral_0^infinity
                 D_s(Phi_t(u))dt                 (6.1)
```

satisfies

```text
(nu*L_0+L_1)S_trajectory=-nu*D_s.                (6.2)
```

So a global nonperturbative solution of the homological equation is exactly
the future occupation functional we wanted to control in the first place.
Defining the resummation by (6.1) would assume global flow and convergence,
making the argument circular.

Analytic continuation of `(I-A/nu)^(-1)` beyond the Neumann disk is not
logically impossible. However, none of the computed signs supplies the
coercive estimate needed to construct such a continuation:

```text
X_s, Y_s, and Z_s are all sign-indefinite,
and every finite endpoint through P_2 is sign-indefinite.  (6.3)
```

## 7. Verdict and Pivot

The heat-normal-form hierarchy is now useful as:

```text
1. an exact perturbative expansion;
2. a diagnostic of which generated modes carry recurrence;
3. a finite-Galerkin small-data resolvent.
```

It is not, under the available estimates, a nonperturbative large-data
regularity mechanism. Blind computation of `P_4,P_5,...` would add
coefficients without changing this logical boundary.

The recommended pivot is back to collision geometry, but with a stricter
target informed by this failure:

```text
Construct a pair/separation observable whose full stochastic or two-point
generator contains viscosity and stretching together, rather than expanding
stretching as repeated Euler insertions around the heat flow.
```

The next concrete calculation should formulate the corresponding Poisson or
stopping problem on separation and orientation space and test whether the
Bessel noncollision drift controls its boundary flux before angular
absolute values are taken.
