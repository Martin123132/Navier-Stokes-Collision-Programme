# Collision-Defect Dynamics

Date: 2026-07-17

Status: exact smooth-solution and Fourier-triad identities, with explicit
sign obstructions. No unconditional collision-rigidity estimate is claimed.

## 1. Fixed-Scale Evolution

Let

```text
H_s=exp(s*Delta),
B_s=I-H_s,
D_s(u)=<B_s S(u),omega(u) tensor omega(u)>.
```

For divergence-free fields define the trilinear form

```text
V_s(a;b,c)
 =<B_s S(a),sym[omega(b) tensor omega(c)]>.
```

It is symmetric in its last two entries and `D_s(u)=V_s(u;u,u)`. Write
Navier-Stokes as

```text
partial_t u=nu*Delta u-N(u),
N(u)=P[(u dot grad)u],
```

where `P` is the Leray projection. Differentiating the cubic functional gives

```text
partial_t D_s
 =nu{V_s(Delta u;u,u)+2V_s(u;Delta u,u)}
  -{V_s(N(u);u,u)+2V_s(u;N(u),u)}.               (1.1)
```

The first brace is the exact viscous contribution. The second is a quartic
nonlinear transfer term. Formula (1.1) contains pressure only through the
bounded projection `P` and introduces no closure assumption.

For

```text
Q=||q||_2^2,
q=-Delta u=curl omega,
```

one also has

```text
(1/2)partial_t Q
 =-nu||grad q||_2^2+<q,Delta N(u)>.               (1.2)
```

Consequently, differentiating the collision-rigidity ratio `D_s/(nu*Q)`
immediately exposes a quartic transfer in both numerator and denominator.
There is no sign at this level from incompressibility alone.

## 2. Exact Triad Decomposition

On a periodic domain, fix a Fourier triad

```text
k+ell+m=0.
```

Let `A_k,A_ell,A_m` be the three cubic amplitudes obtained by placing the
strain factor on each of the three modes. The polarized Miller identity gives

```text
|k|^2 A_k+|ell|^2 A_ell+|m|^2 A_m=0.             (2.1)
```

The heat defect carried by the triad is

```text
d_s(k,ell,m)
 =sum_(p in {k,ell,m})[1-exp(-s|p|^2)]A_p.       (2.2)
```

Expanding (2.2) and using (2.1) removes the first heat moment:

```text
d_s
 =-(s^2/2)sum |p|^4 A_p+O(s^3).                 (2.3)
```

Under the linear heat equation, all three Fourier coefficients in the triad
acquire the same cubic decay factor. Hence

```text
partial_t d_s
 =-nu*Lambda*d_s,
Lambda=|k|^2+|ell|^2+|m|^2.                     (2.4)
```

Viscosity therefore damps every individual signed triad defect toward zero.

## 3. No Additional Triad Identity

For the triad

```text
k=(1,0,0), ell=(0,1,0), m=(-1,-1,0),
```

two explicit divergence-free polarization choices give, after removal of a
common Fourier phase,

```text
A^(1)=(-2,0,1),
A^(2)=(-1,1,0).                                  (3.1)
```

Both obey

```text
A_k+A_ell+2A_m=0,
```

and they are linearly independent. They span the complete plane permitted by
(2.1). Thus there is no second universal linear triad identity capable of
fixing the defect sign.

For the first polarization, (2.2) becomes exactly

```text
d_s=-[1-exp(-s)]^2.                              (3.2)
```

Changing the common phase or the sign of the physical field reverses the
cubic sign.

## 4. Why Viscosity Does Not Give a Global Barrier

Triadwise damping does not imply damping of the summed positive defect. Take
two Fourier triads with no mixed zero-sum interactions and amplitude-scale
them so that at the chosen collision scale

```text
d_low=1,       Lambda_low=4,
d_high=-1/2,  Lambda_high=36.
```

Then

```text
D_s=d_low+d_high=1/2>0,

(partial_t D_s)_heat
 =-nu[4*d_low+36*d_high]
 =14*nu>0.                                       (4.1)
```

Nothing in either triad grows. The negative high-frequency contribution
simply decays faster, exposing the positive low-frequency contribution. An
explicit noninteracting choice is the union of

```text
{+/-[(1,0,0),(0,1,0),(-1,-1,0)]}
```

and the scale-three rotated set

```text
{+/-[(0,3,0),(0,0,3),(0,-3,-3)]}.
```

Direct enumeration finds no mixed zero-sum triad between these supports.

This rules out the hoped-for instantaneous implication

```text
D_s>0 => (partial_t D_s)_heat<=0.                 (4.2)
```

It does not rule out a Navier-Stokes dynamical barrier, because the
palinstrophy denominator and nonlinear transfer have not yet been combined.

## 5. Exact Failure of the First-Crossing Barrier

The first-crossing test can be completed exactly for the trigonometric field

```text
u_0=(0,-1,-1)cos(x)
    +(-1,0,-1)cos(y)
    +(1,-1,1)sin(x+y).                            (5.1)
```

Write

```text
x_s=exp(-s),
h=1-x_s,
u=A*u_0.
```

Direct Fourier calculation gives

```text
Q=8*A^2,
D_s=(1/2)h^2*A^3.                                (5.2)
```

The threshold `F_s=nu*Q-D_s=0` is therefore reached at

```text
A_*(s)=16*nu/h^2.                                (5.3)
```

At time zero, the exact Navier-Stokes derivatives are

```text
partial_t Q=-28*nu*A^2+3*A^3,                    (5.4)

partial_t D_s
 =-4*nu*D_s+A^4*c_s,                             (5.5)

c_s=(7/4)(1-x_s)+(1-x_s^2)-(3/4)(1-x_s^5)
   =(h^2/4)(3*x_s^3+6*x_s^2+9*x_s+8).           (5.6)
```

Substituting (5.3) into `partial_t F_s` yields

```text
[h^4/(nu^2*A_*^2)] partial_t F_s
 =-4*h^2(
    48*x_s^3+95*x_s^2+146*x_s+115).              (5.7)
```

For every `s>0`, one has `0<x_s<1`, so the right side is strictly negative.
The vector field is a smooth finite Fourier polynomial, hence it generates a
local smooth Navier-Stokes solution. We have proved:

```text
F_s=0 does not imply partial_t F_s>=0.             (5.8)
```

By continuity, amplitudes just below `A_*(s)` start with `F_s>0` and retain
an outward-pointing derivative near the threshold. Therefore the condition
`D_s<=nu*Q` is not a forward-invariant region at any fixed collision scale.

This does not invalidate the continuation criterion: assuming the inequality
for the whole lifespan still proves regularity. It invalidates the proposed
route of deriving that assumption from an instantaneous first-crossing
barrier.

## 6. Stationary-Scale Obstruction to Adaptation

For a differentiable scale `s=s(t)`, the exact barrier derivative is

```text
d/dt[nu*Q-D_(s(t))]
 =[partial_t(nu*Q-D_s)]_(s fixed)
  -s'(t)*partial_s D_s.                           (6.1)
```

One might try to choose `s'(t)` so the second term prevents a crossing. A
second exact triad shows why this is not automatically possible. Take

```text
k=(1,0,0), ell=(1,1,0), m=(-2,-1,0),
```

with squared lengths `1,2,5` and the divergence-free polarizations recorded
in `scripts/adaptive_scale_barrier_audit.py`. After choosing the physical
phase and sign, its unit-amplitude defect is

```text
d_s=-(5/4)(1-x)+(5/4)(1-x^2)-(1/4)(1-x^5),
x=exp(-s).                                        (6.2)
```

The scale derivative factors as

```text
partial_s d_s
 =-(5/4)x(x-1)(x^3+x^2+x-1).                    (6.3)
```

The cubic in (6.3) is strictly increasing on `(0,1)` and has one root

```text
x_0=0.543689012692...,
s_0=-log(x_0)=0.609377863436....                  (6.4)
```

At this root, (6.2) is positive because its remaining factor reduces modulo
`x^3+x^2+x-1` to `x(x+2)`. Hence

```text
d_(s_0)>0,
partial_s d_(s_0)=0.                              (6.5)
```

The same exact Fourier calculation gives

```text
Q=(139/2)A^2,
partial_t Q=-651*nu*A^2-(15/2)A^3,               (6.6)

partial_t D_s=-8*nu*D_s+c_s*A^4,                 (6.7)
```

where

```text
c_s=(7/4)(1-x)+(15/8)(1-x^2)+(71/40)(1-x^5)
    -(5/8)(1-x^10)-(5/8)(1-x^13).                (6.8)
```

It factors as `(1-x)^2/40` times a degree-eleven polynomial whose coefficients
are all positive. At the threshold amplitude

```text
A_*=nu*(139/2)/d_(s_0),                           (6.9)
```

one obtains

```text
partial_t[nu*Q-D_(s_0)]
 =A_*^2[-95*nu^2-(15/2)nu*A_*-c_(s_0)A_*^2]
 <0.                                              (6.10)
```

But the adaptive correction in (6.1) is zero for every finite `s'(t)` by
(6.5). Therefore a differentiable adaptive scale cannot restore the local
first-crossing barrier in general.

This does not eliminate all solution-adapted scales. It eliminates the
specific strategy of enforcing `nu*Q-D_s>=0` by a local scale ODE. A
discontinuous scale selection, a multi-scale envelope, or a time-integrated
condition would be a genuinely different mechanism.

## 7. Revised Rigidity Target

The continuation condition from `heat_scale_cubic_cancellation.md` involves

```text
R_s(t)=[D_s(t)]_+/(nu*Q(t)).                      (6.1)
```

Any proof that `R_s` cannot cross one must use more than:

- the Miller first-moment identity;
- pair noncollision;
- or the sign of linear diffusion on the summed defect.

The exact remaining mechanisms are now:

```text
1. signed cancellation among triad scales;
2. quartic Navier-Stokes transfer between triads;
3. simultaneous evolution of palinstrophy Q;
4. the pair-determinant angular geometry.
```

The fixed-scale and differentiable adaptive first-crossing tests have now
failed. The viable descendants are narrower:

```text
1. a time-integrated bound on the positive pair determinant;
2. a multi-scale envelope that remains meaningful when partial_s D_s=0;
3. a signed triad measure whose nonlinear transfer has cross-scale rigidity;
4. a stochastic stopping scale using the full replica history, not a local
   ODE for one heat scale.
```

Historical supersession: the time-integrated signed triad measure proposed
here was subsequently constructed in `cumulative_collision_rigidity.md`.
The quartic transfer, two further heat normal forms, and the resummed
hierarchy were then audited. Their sign-indefiniteness and perturbative
large-Reynolds limitation are recorded in
`normal_form_resummation.md`. The current path-dependent frontier is the
projected-replica pressure edge, with the far-carrier contribution now
controlled in `floor_free_pressure_edge_tail_gate.md`.
