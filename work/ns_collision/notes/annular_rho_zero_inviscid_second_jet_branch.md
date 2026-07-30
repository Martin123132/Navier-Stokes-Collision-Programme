# Annular rho-zero inviscid second-jet branch

## 1. Result

The preceding second-jet route guard certified the exact coupled
second-variation formula and a positive pure velocity-heat pressure
coefficient of order `N^7`. Its preliminary power ledger treated the
remaining nonlinear pressure sector as an `N^7` question.

That triage was too optimistic. The exact inviscid pressure sector reduces
to only two low-amplitude branches,

```text
J_inv,N(a,t)
 =t[c_1,N a+c_3,N a^3],                          (1.1)
```

but the four-high/one-low coefficient `c_1,N` is permitted the fixed-output
scale `N^7`. After the static optimizer contributes

```text
a_N t_N=O(N^2),
```

this is a candidate optimized `N^9` second-jet route. It is not yet a
certified `N^9` asymptotic law.

The crucial simplification is that the observed leading signal does not
come from a broad pressure shell. More than `99.9%` of the dominant
`N=29` contribution is carried by the finite set of pressure outputs

```text
|q|<4.                                             (1.2)
```

Thus the next obligation is a finite family of fixed-output Riemann-sum
limits rather than a full dealiased second-jet calculation.

## 2. Compact identity

Let

```text
T(x,y,z;phi)
 =integral p[x,y] z dot grad phi
```

and symmetrize the three velocity entries:

```text
S(x,y,z;phi)
 =(T(x,y,z;phi)+T(x,z,y;phi)+T(y,z,x;phi))/3.
```

Let

```text
B(x,y)
 =-P[((x dot grad)y+(y dot grad)x)/2],

C(x,phi)=-x dot grad phi.
```

Then

```text
E=B(u,u),
A=C(u,lambda),
lambda_2=C(E,lambda)+C(u,A).
```

The complete pressure-only second derivative along the coupled Euler and
transport directions is

```text
J_inv
 =6S(u,E,E;lambda)
  +6S(u,u,E;A)
  +6S(u,u,B(u,E);lambda)
  +S(u,u,u;lambda_2).                             (2.1)
```

The scalar chain-rule model verifies every multiplicity in (2.1)
symbolically.

## 3. Amplitude projection

Write

```text
u=H-aU,
lambda=t Phi.
```

The low field has wave and polarization

```text
ell=(0,1,-1),
d=(0,1,1)/sqrt(2),
ell dot d=0.
```

It is a stationary shear:

```text
(U dot grad)U=0,
p[U,U]=0,
B(U,U)=0.                                        (3.1)
```

Expression (2.1) is degree five in velocity and linear in the weight.
The annular support gap permits only zero, two, or four high velocity
legs. The five-low term vanishes by (3.1), leaving exactly (1.1).

The `a` coefficient has four high legs and requires only `8K` padding;
the `a^3` coefficient has two high legs and would require only `4K`.
The audit evaluates both together on `8K` grids. At `N=9`, a completely
independent `10K` full 20-channel second jet with `a=0.2,t=0.3` agrees
with (1.1) to `7.8e-16`.

The stored `N=3` and `N=5` 20-channel rows replay to better than
`6e-16`. Changing the projected grid from `8K` to `10K` changes either
coefficient by less than `2e-16`.

## 4. Finite carrier evidence

The projected rows are:

```text
N       c_1,N             c_1,N/N^7        c_3,N
5      -2.6297e-1         -3.3660e-6       5.2909e-2
9      -5.0157            -1.0485e-6       1.2842e-1
17     -2.3897e2          -5.8238e-7       2.3999e-1
25     -2.8833e3          -4.7241e-7       3.4296e-1
29     -7.6604e3          -4.4408e-7       3.9358e-1
```

The full log-log fit over `N=5,...,29` is pre-asymptotic and is not used
as a theorem. The trend says only that normalization by `N^5` is plainly
insufficient and that the natural fixed-output `N^7` route must be
resolved.

The two-high coefficient is much smaller:

```text
c_3,29/N^3=1.6138e-5,
```

and is declining under its danger normalization. It is postponed until
the four-high branch is decided.

## 5. Dominant forms

At `N=25`, nearly the entire four-high coefficient comes from

```text
-6S(B(H,H),B(H,H),U;Phi)

-12S(B(H,B(H,H)),H,U;Phi).                       (5.1)
```

Their values are

```text
+1631.2374132790,
-4513.5878176616.
```

All transported-weight terms together are below one in magnitude at this
carrier. This does not justify dropping them asymptotically, but it
identifies (5.1) as the leading route.

## 6. Pressure-output localization

Parseval decomposition of each form in (5.1) by the internal pressure
output gives, at `N=29`,

```text
combined bounded output |q|<4  = -7659.1534090488,
combined output outside |q|<4  =     0.0357952224,
full c_1,29                     = -7660.4031239650.
```

The shell calculation replays the direct forms to better than `2e-12`.
The bounded signal is a finite sum over explicit integer vectors such as

```text
(0,0,+/-1), (0,0,+/-2), (0,+/-1,-/+2),
(+/-1,0,+/-2), ...
```

Large individual mode contributions cancel substantially. Therefore a
sign claim requires the combined fixed-mode limit; checking the largest
mode separately would be invalid.

## 7. Correct scope

This stage proves:

```text
the compact coupled inviscid-pressure identity,
the exact t(c_1 a+c_3 a^3) amplitude reduction,
the eightfold branch-support ledger,
agreement with the full 20-channel formula,
and localization of the candidate leading signal to |q|<4.
```

It does not prove:

```text
that c_1,N/N^7 converges,
that its limit is nonzero or negative,
an optimized N^9 second-jet asymptotic,
a uniform Taylor remainder,
a positive or negative finite parabolic-window gain,
critical L^3 control, blowup, or global regularity.
```

The next theorem should write each bounded-output contribution in (5.1)
as a fixed-domain Riemann sum, prove convergence with a quantitative
remainder, and combine all modes before deciding the sign.
