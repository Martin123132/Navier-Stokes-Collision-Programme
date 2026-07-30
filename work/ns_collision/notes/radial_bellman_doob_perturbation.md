# Radial Bellman Doob perturbation gate

## Purpose

The first radial-payoff supersolution has pointwise gain `1.342878685` and
an exact Doob identity. Under the current cubic split it no longer closes.
This note records what its structural certificate still buys
for the spatial Campanato drift error `e` and adverse potential `q_+`.

The answer has two parts. It gives an exact smooth pointwise criterion and a
useful weighted cancellation, but it does not yet turn critical
`L^3/L^(3/2)` control into a boundary theorem.

## Exact transformed generator

Let

```text
L_B=Delta+(B(t)y).grad,
kappa_B=-[L_B U+U]/U.                                 (1)
```

The interval certificate proves

```text
kappa_B>=0.005                                        (2)
```

for every admissible symmetric trace-free affine history. For an actual
payoff equation with drift error `e`, potential `1+q_+`, and `u=Uv`, direct
expansion gives

```text
U^(-1)(L_B+e.grad+1+q_+)(Uv)
 =Delta v+[B y+e+2 grad log U].grad v
  +[-kappa_B+q_++e.grad log U]v.                     (3)
```

Thus the same supersolution remains valid under the signed pointwise gate

```text
q_++e.grad log U<=0.005.                              (4)
```

The absolute sufficient version replaces the drift term by
`|e||grad log U|`.

## Why the pointwise route stops

Put `x=1-r^2/4` and `a=cos(2 pi z/3)`. The certified barrier has

```text
|grad log U|~x^(-7/20)  near the radial wall,
|grad log U|~a^(-13/20) near an axial cap.            (5)
```

Consequently no nonzero uniform drift tolerance follows from (4), and
critical `L^3` control does not imply its pointwise version. Concentrating a
smooth error in a thin collar can keep its critical norm finite while making
the left side of (4) arbitrarily large. The new barrier must not be used to
smuggle a pointwise regularity assumption into the Leray argument.

## Weighted cancellation

The apparent `e.grad log U` potential in (3) is not an adverse real-form term.
In `L^2(U^2 dx)`, divergence-free `e` gives

```text
Re int U^2 v e.grad v
 =-int U^2 v^2 e.grad log U.                          (6)
```

This cancels the transformed zero-order contribution

```text
+int U^2 v^2 e.grad log U                             (7)
```

exactly. The cancellation is the weighted version of the original fact that
a divergence-free first-order error is skew on zero-boundary functions.
It removes `e` from coercivity, but not from the sector norm needed to compare
boundary operators.

## Remaining margin

The certified scalar numbers are

```text
g_barrier=1.342878684567,
g_closure=1.232133608495,
additive deficit=0.110745076072,
relative deficit=8.2468 percent,
C_ideal=1.187840019571.                               (8)
```

There is no perturbative margin for this older barrier. Its exact weighted
cancellation should instead be transferred to the lower-gain finite-energy
barrier:

```text
critical volume form control
 -> causal U-weighted interior response
 -> radial-boundary trace in the dynamic entry law.  (9)
```

The unresolved issue in (9) is again the boundary law. A uniform surface
trace estimate is insufficient for an arbitrary singular dynamic entry law;
either the perturbation must be separated from the interface by a collar, or
the exterior-return/entry kernel must supply the missing smoothing without
conditioning away its contraction.

The follow-up collar audit sharpens this statement. A positive collar gives
an explicit uniform bound for the first affine Green insertion, but not for
the global `L^infinity` Kato iteration: later iterates start in the rough
support and recover the critical Newtonian endpoint. The interior iterates
can instead be resummed by a nonautonomous positive-part energy estimate.
The remaining calibrated quantity is the homogeneous collar trace constant
defined in `critical_collar_transfer_gate.md`.

The product-rule identity, weighted cancellation, boundary asymptotics, and
margin arithmetic are reproduced by
`scripts/radial_bellman_doob_perturbation_audit.py`.
