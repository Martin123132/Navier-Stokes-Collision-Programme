# Second Heat Normal Form

Date: 2026-07-17

Status: exact second normal-form identity, exact sign-indefinite quintic
remainder, and exact explanation of the exceptional order-five cancellation
for the two-mode trajectory. This does not yet bound the remainder for
arbitrary solutions.

## 1. Polarized Definition

Let

```text
B(u,v)=-P[(u dot grad)v]
```

be the polarized Euler nonlinearity. Recall

```text
J_s(u)=integral_0^infinity D_s(exp(r*Delta)u)dr,
partial_t J_s=-nu*D_s+X_s,
X_s=DJ_s(u)[B(u,u)].                              (1.1)
```

Define the second heat primitive

```text
K_s(u)=integral_0^infinity X_s(exp(r*Delta)u)dr.  (1.2)
```

The Fourier denominator in (1.2) must be attached to the four original
inputs. For a term in which `B(u_p,u_q)` occupies a cubic slot and

```text
p+q+b+c=0,
```

the two denominators are

```text
Lambda_3=|p+q|^2+|b|^2+|c|^2,
Lambda_4=|p|^2+|q|^2+|b|^2+|c|^2.               (1.3)
```

Thus the kernel of `K_s` contains `1/(Lambda_3*Lambda_4)`. Replacing
`Lambda_4` by a receiving-mode frequency would be incorrect because the heat
flow acts before the bilinear convolution.

Polarizing the quartic form gives the quintic remainder

```text
Y_s(u)=DK_s(u)[B(u,u)].                           (1.4)
```

Every Fourier monomial in `K_s` has four inputs. Differentiation in the heat
direction multiplies it by `-Lambda_4`, so

```text
DK_s(u)[Delta u]=-X_s(u).                         (1.5)
```

Consequently every smooth Navier-Stokes solution satisfies

```text
partial_t K_s=-nu*X_s+Y_s.                       (1.6)
```

The finite-Fourier audit verifies (1.5) to `6.7e-16`, agrees with the
independent quartic evaluator to `2.3e-16`, and verifies (1.6) by a direct
Navier-Stokes directional difference to better than `1e-10`.

## 2. Combined Cumulative Identity

Set

```text
L_s=J_s+K_s/nu.                                  (2.1)
```

Equations (1.1) and (1.6) give

```text
partial_t L_s=-nu*D_s+Y_s/nu.                    (2.2)
```

Therefore

```text
integral_0^T D_s dt
 =[L_s(u_0)-L_s(u(T))]/nu
  +(1/nu^2)*integral_0^T Y_s(u(t))dt.            (2.3)
```

This supplies a second inverse heat frequency without taking an absolute
value. The unresolved transfer has degree five rather than four.

## 3. The Quintic Does Not Vanish Generally

Use the real divergence-free triad with independent positive coefficients

```text
u_(1,0,0)=(0,-1,-1),
u_(0,1,0)=(-1,0,-1),
u_(1,1,0)=-i*(1,-1,1),                           (3.1)
```

and conjugate coefficients at the negative waves. With

```text
x=exp(-s),       0<x<1,
```

exact symbolic convolution gives

```text
Y_s(u)=-(1-x)^2
       *(9*x^3+18*x^2+27*x+905)/600<0.           (3.2)
```

Because `Y_s` is homogeneous of degree five,

```text
Y_s(-u)=-Y_s(u)>0.                               (3.3)
```

Hence `Y_s` has both signs at every positive heat scale. The hoped-for
general quintic cancellation and every global sign-definite interpretation
are both false.

## 4. Why the Two-Mode Quintic Cancels

For the negative two-mode seed

```text
k=(1,0,0),       m=(1,1,0),
u_k=(0,-1,1),    u_m=(-1,1,-1),                  (4.1)
```

the Euler direction creates only

```text
+/-(m-k),       +/-(m+k).                        (4.2)
```

No wave in (4.2), together with any three waves from
`{+/-k,+/-m}`, sums to zero. Every polarized quintic term is therefore
forbidden by Fourier closure:

```text
Y_s(u_seed)=0.                                   (4.3)
```

The audit enumerates all possibilities and finds zero resonant quintets.
Equivalently, translation by `(pi,0,0)` changes the sign of both seed modes.
Translation invariance and odd parity of `Y_s` force (4.3).

The quartic endpoint is nonzero and exact:

```text
K_s(u_seed)
 =(1-x)^2*(x^3+2*x^2+3*x-11)/120<0.             (4.4)
```

This is the earlier weak coefficient `c_4`. The first nonlinear correction
creates the waves (4.2), so the first nonzero integrated remainder is order
six. The exact normal-form identity and the independently derived Duhamel
hierarchy give

```text
integral_0^infinity Y_s dt
 =nu^4*kappa^4*[c_6(x)*R^6+O(R^8)],             (4.5)
```

where `kappa` is the base wave-number scale, `R=A/(nu*kappa)`, and

```text
c_6(x)=(1-x)^2/196560000
 *(6500*x^11+13000*x^10+19500*x^9+40625*x^8
   +61750*x^7+82875*x^6+104000*x^5+125125*x^4
   +689807*x^3+1254489*x^2+1819171*x+16386753)
 >0.                                             (4.6)
```

At `s=0.5`, `c_6=0.0142862484162`. Thus the second normal form does not
remove the beat-mode return. It places the negative quartic contribution in
the endpoint `K_s(u_0)` and the positive sixth-order recurrence in
`integral Y_s`.

## 5. Closure Audit

Under Navier-Stokes scaling, with the heat scale transformed as
`s -> lambda^-2*s`, the functionals have exponents

```text
J_s, K_s, L_s, integral D_s dt, integral Y_s dt:  lambda^1,
D_s, X_s, Y_s:                                    lambda^3.   (5.1)
```

So (2.3) remains scale-critical. The difficulty is nonlinear energy growth,
not a scaling mismatch. Frequency counting makes the second primitive a
quartic order-zero multiplier. The natural candidate estimate is

```text
|K_s(u)| <= C*||u||_4^4
          <= C*||u||_2*||omega||_2^3.            (5.2)
```

Establishing the required multilinear multiplier bound uniformly is a
separate analytic task; it is not claimed here. Even if (5.2) holds, it is
critical under scaling but superlinear in enstrophy and cannot be inserted
into the continuation estimate as a harmless endpoint bound. Moreover,
(4.4) and its positive companion show exactly that no sign of `K_s` is
available even on the simplest two-mode cone.

The finite second normal form therefore does not close regularity by itself:

```text
1. K_s has an indefinite, enstrophy-superlinear endpoint;
2. Y_s is sign-indefinite in general;
3. on the exceptional two-mode cone, the dangerous positive recurrence
   reappears as the first surviving sixth-order integral.
```

## 6. Next Decision Gate

The normal-form recursion can be continued once more:

```text
M_s(u)=integral_0^infinity Y_s(exp(r*Delta)u)dr,
partial_t M_s=-nu*Y_s+Z_s,                       (6.1)
```

where `Z_s` is sextic and even under `u -> -u`. A blind iteration would only
raise the polynomial degree and move the obstruction. The useful next test
is narrower:

```text
1. construct the exact sextic Z_s kernel;
2. determine whether it has a sign or coercive pairing unavailable to the
   odd quintic Y_s;
3. test whether the modified endpoint J_s+K_s/nu+M_s/nu^2 has any lower
   bound compatible with the enstrophy estimate.
```

If the sextic is also indefinite and the endpoint remains superlinear, the
finite heat-normal-form route should be retired except as a convergent
small-Reynolds expansion. If a new even-degree coercivity appears, that is
the first plausible place for the collision-rigidity mechanism to survive.
