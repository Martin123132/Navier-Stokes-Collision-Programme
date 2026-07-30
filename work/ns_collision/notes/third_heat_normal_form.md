# Third Heat Normal Form

Date: 2026-07-17

Status: exact tree construction of the third primitive, exact evolution
identity, exact all-scale sign-indefiniteness of the even sextic remainder,
and exact sign-indefiniteness of the new endpoint. This retires blind finite
normal-form iteration as a coercivity mechanism, but not a resummed or
geometric use of the hierarchy.

## 1. Denominator-Preserving Tree Construction

Write the polarized Euler nonlinearity as

```text
B(u,v)=-P[(u dot grad)v].                         (1.1)
```

The cubic defect begins with three velocity leaves. Each Euler derivative
replaces one leaf by the ordered pair `B(u,u)`. The first three stages
therefore contain

```text
X_s:  3 trees,
Y_s:  3*4=12 trees,
Z_s:  3*4*5=60 trees.                            (1.2)
```

Every heat primitive records the frequency partition that existed when it
was introduced. If six final leaves have frequencies `k_1,...,k_6`, an
earlier partition may group two or three of them into one receiving wave.
The three denominators are consequently distinct:

```text
Lambda_3 = sum of squares of the three cubic-slot waves,
Lambda_4 = sum of squares of the four first-transfer inputs,
Lambda_5 = sum of squares of the five second-transfer inputs.  (1.3)
```

The sextic kernel contains

```text
1/(Lambda_3*Lambda_4*Lambda_5).                  (1.4)
```

The script stores each partition explicitly and updates its descendant sets
whenever a leaf splits. This avoids replacing an original-input heat
frequency by a receiving frequency after convolution.

## 2. Third Primitive and Cumulative Identity

Define

```text
M_s(u)=integral_0^infinity Y_s(exp(r*Delta)u)dr,
Z_s(u)=DM_s(u)[B(u,u)].                           (2.1)
```

The newest denominator `Lambda_5` cancels under differentiation in the heat
direction, so

```text
DM_s(u)[Delta u]=-Y_s(u),
partial_t M_s=-nu*Y_s+Z_s.                       (2.2)
```

Together with

```text
partial_t J_s=-nu*D_s+X_s,
partial_t K_s=-nu*X_s+Y_s,                       (2.3)
```

set

```text
P_s=J_s+K_s/nu+M_s/nu^2.                         (2.4)
```

Then

```text
partial_t P_s=-nu*D_s+Z_s/nu^2,

integral_0^T D_s dt
 =[P_s(u_0)-P_s(u(T))]/nu
  +(1/nu^3)*integral_0^T Z_s(u(t))dt.            (2.5)
```

The generated 12-tree quintic agrees with the independent second-normal-form
evaluator to `5e-16`. Direct directional checks give residuals below
`1.1e-11` for the heat identity and below `1.2e-12` for the full
Navier-Stokes identity.

## 3. The New Endpoint Is Sign-Indefinite

For the sparse triad used in the quintic counterexample, exact
Gaussian-rational convolution gives

```text
M_s(u)=(1-x)^2
       *(31*x^3+62*x^2+93*x-380)/1800,
x=exp(-s).                                        (3.1)
```

The cubic polynomial is increasing and its value at `x=1` is `-194`.
Therefore

```text
M_s(u)<0,       0<x<1.                            (3.2)
```

Since `M_s` is homogeneous of degree five,

```text
M_s(-u)=-M_s(u)>0.                                (3.3)
```

Thus the new endpoint in (2.5) has no universal sign at any positive heat
scale.

## 4. Exact Positive Sextic Channel

For the original negative two-mode field

```text
u_(1,0,0)=(0,-1,1),
u_(1,1,0)=(-1,1,-1),                             (4.1)
```

with equal coefficients at the negative waves, the exact sextic is

```text
Z_s^-(u)=(1-x)^2*P_-(x)/49140000,                (4.2)
```

where

```text
P_-(x)=16250*x^11+32500*x^10+48750*x^9
 +94250*x^8+139750*x^7+185250*x^6
 +230750*x^5+276250*x^4+1670249*x^3
 +3064248*x^2+4458247*x+33242991.                (4.3)
```

Every coefficient is positive. Hence

```text
Z_s^-(u)>0       for every s>0.                  (4.4)
```

At `s=0.5`, its value is `0.118206613517196...`. This is the instantaneous
even-degree source whose heat evolution produces the positive sixth-order
recurrence found in the weak trajectory expansion.

## 5. Exact Negative Sextic Channel

Change only the final polarization sign:

```text
u_(1,1,0)=(-1,1,1).                              (5.1)
```

This was the positive companion for the quartic transfer. Its sextic is

```text
Z_s^+(u)=(1-x)^2*P_+(x)/16380000,                (5.2)
```

where

```text
P_+(x)=153750*x^11+307500*x^10+461250*x^9
 +905550*x^8+1349850*x^7+1794150*x^6
 +2238450*x^5+2682750*x^4+4907*x^3
 -2672936*x^2-5350779*x-2361907.                 (5.3)
```

The derivative `P_+'` has exactly one sign variation, is negative at zero,
and has positive leading coefficient. Descartes' rule and continuity imply
that `P_+` has one positive critical point, which is a minimum. Its endpoint
values are

```text
P_+(0)=-2361907,
P_+(1)=-487465.                                  (5.4)
```

The maximum on `[0,1]` is therefore negative, and

```text
Z_s^+(u)<0       for every s>0.                  (5.5)
```

At `s=0.5`, its value is `-0.0556535279103182...`.

Equations (4.4) and (5.5) prove that the even sextic has both signs at every
positive heat scale on the same two-wave support. Even parity does not create
coercivity.

## 6. Closure Verdict

The third normal form remains scale-critical:

```text
P_s and integral D_s dt scale as lambda,
Z_s and D_s scale as lambda^3.                   (6.1)
```

It nevertheless fails the two hoped-for algebraic gates:

```text
1. the quintic endpoint M_s is sign-indefinite;
2. the even sextic remainder Z_s is sign-indefinite at every heat scale.
```

Continuing once more would create a degree-six endpoint and a degree-seven
odd remainder, but no evidence now supports a finite-stage sign transition.
The first three transfers show the alternating pattern

```text
quartic X_s: indefinite,
quintic Y_s:  indefinite,
sextic Z_s:   indefinite.                        (6.2)
```

This does not prove that every higher transfer is indefinite. It does prove
that stopping after the first, second, or third primitive cannot supply the
missing positivity.

## 7. Recommended Successor

Blind finite iteration should now be retired. Two materially different
routes remain:

```text
1. Resummed route: formulate the whole heat-normal-form hierarchy as a
   generating functional and determine its convergence parameter and
   large-Reynolds obstruction.

2. Geometric route: return to the pair determinant/helical structure and
   seek an estimate coupling all transfer orders before absolute values are
   taken.
```

The least speculative next calculation is the resummed route. It can tell us
whether the hierarchy contains information beyond an ordinary small-data
Neumann series. If it reduces exactly to a small-Reynolds expansion, the
finite normal-form branch should be archived and effort returned to the
nonperturbative collision geometry.
