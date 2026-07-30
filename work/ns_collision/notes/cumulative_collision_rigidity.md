# Cumulative Collision Rigidity

Date: 2026-07-17

Status: exact smooth-solution criterion and exact triadwise primitive. The
required cumulative transfer estimate remains open.

## 1. Why Time Integration Changes the Gate

For fixed collision heat scale `s>0`, write

```text
S_s=exp(s*Delta)S,
D_s=<S-S_s,omega tensor omega>,
E=||omega||_2^2,
Q=||grad omega||_2^2.
```

The enstrophy identity is

```text
(1/2)E(T)+nu*integral_0^T Q dt
 =(1/2)E(0)
  +integral_0^T <S_s,omega tensor omega>dt
  +integral_0^T D_s dt.                          (1.1)
```

The heat-kernel estimate from `heat_scale_cubic_cancellation.md` gives

```text
||S_s(t)||_infinity
 <=C_0*s^(-5/4)||u_0||_2,
C_0=sqrt(3)/[2*(8*pi)^(3/4)].                    (1.2)
```

Unlike the pointwise argument, the time integral of enstrophy on the right is
already controlled by the kinetic-energy inequality:

```text
nu*integral_0^T E(t)dt<=(1/2)||u_0||_2^2.        (1.3)
```

Consequently,

```text
absolute_value(
 integral_0^T <S_s,omega tensor omega>dt)
 <=C_0*s^(-5/4)||u_0||_2^3/(2*nu).               (1.4)
```

This bound is uniform in `T`.

## 2. Cumulative Continuation Criterion

Suppose there exist a fixed `s>0`, a number `theta<1`, and a finite constant
`C_*` depending only on the initial data, viscosity, and `s`, such that every
smooth interval satisfies

```text
integral_0^T D_s(t)dt
 <=theta*nu*integral_0^T Q(t)dt+C_*.             (2.1)
```

Then (1.1)-(1.4) imply

```text
E(T)+(2-2*theta)nu*integral_0^T Qdt
 <=E(0)+C_0*s^(-5/4)||u_0||_2^3/nu+2*C_*.       (2.2)
```

Thus enstrophy stays uniformly bounded and the smooth solution continues.
This criterion survives both exact first-crossing counterexamples in
`collision_defect_dynamics.md`, because it does not assert that `D_s` is
pointwise small or monotone.

The signed integral in (2.1) is essential. Replacing it by
`integral |D_s|` discards the cross-scale cancellations that the collision
formulation was built to retain.

## 3. Triadwise Time Primitive

For a Fourier triad `alpha=(k,ell,m)`, let

```text
Lambda_alpha=|k|^2+|ell|^2+|m|^2,
d_alpha(t)=its signed contribution to D_s(t),
X_alpha(t)=its quartic Navier-Stokes transfer.
```

The exact evolution from `collision_defect_dynamics.md` is

```text
partial_t d_alpha
 =-nu*Lambda_alpha*d_alpha+X_alpha.               (3.1)
```

Integrating (3.1) gives

```text
integral_0^T d_alpha(t)dt
 =[d_alpha(0)-d_alpha(T)]/(nu*Lambda_alpha)
  +(1/(nu*Lambda_alpha))
    integral_0^T X_alpha(t)dt.                   (3.2)
```

Time integration therefore supplies one inverse triad frequency. This is the
first mechanism found here that weakens, rather than merely renames, the
quartic transfer obstruction.

## 4. Scale-Space Cubic Primitive

Define

```text
J_s(u)=sum_alpha d_alpha(u)/Lambda_alpha.         (4.1)
```

Since

```text
1/Lambda=integral_0^infinity exp(-r*Lambda)dr,
```

the same functional has the semigroup representation

```text
J_s(u)=integral_0^infinity
       D_s(exp(r*Delta)u)dr.                     (4.2)
```

Let

```text
X_s(u)=sum_alpha X_alpha(u)/Lambda_alpha.
```

Summing (3.2), or differentiating (4.1), yields

```text
partial_t J_s=-nu*D_s+X_s,

integral_0^T D_s dt
 =[J_s(u_0)-J_s(u(T))]/nu
  +(1/nu)integral_0^T X_s(u(t))dt.               (4.3)
```

Equation (4.3) converts the cumulative criterion into two concrete tasks:

```text
1. control the negative endpoint part of J_s;
2. control the positive time-integrated quartic primitive X_s.
```

## 5. Sharpness Warning

Heat smoothing and the Calderon-Zygmund estimate suggest the scale-correct
bound

```text
|J_s(u)|<=C*s^(1/4)||omega||_2^3.                (5.1)
```

The exponents are forced by Navier-Stokes scaling. Even if (5.1) is proved
with an optimal constant, it is superlinear in enstrophy and does not close
(2.2). Therefore the next advance must retain a sign, a transfer cancellation,
or a stronger relation between `J_s` and `X_s`. An absolute-value Sobolev
bound will return to the standard regularity barrier.

## 6. Quartic Transfer Is Sign-Indefinite

### 6.1 Exact Two-Mode Counterexample

The former conjecture `X_s(u)>=0` is false. Let the only independent nonzero
Fourier coefficients be

```text
k=(1,0,0),       u_k=(0,-1,1),
m=(1,1,0),       u_m=(-1,1,-1),                 (6.1)
```

with the equal real coefficients at `-k` and `-m`. This is a smooth real
divergence-free trigonometric polynomial. Direct symbolic convolution,
including the Leray projection, gives

```text
X_s=(1-x)^2/20*(x^3+2*x^2+3*x-11),
x=exp(-s).                                        (6.2)
```

For every `s>0`, `0<x<1` and
`0<x^3+2*x^2+3*x<6`, so (6.2) is strictly negative.

The full real two-polarization family

```text
u_k=(0,-a,b),       u_m=(-c,c,d)
```

depends only on `y=a*d` and `z=b*c`. Put

```text
p=x^3+2*x^2+3*x.
```

Then

```text
20*X_s/(1-x)^2
 =5*(p+1)*y^2+(14*p+36)*y*z+10*(p+2)*z^2.        (6.3)
```

The determinant of its symmetric `2 by 2` matrix is

```text
p^2-102*p-224<0,       0<p<6.                    (6.4)
```

Thus `X_s` has both signs at every positive heat scale. Choosing `d=1`
instead of `d=-1` in (6.1) gives the strictly positive companion

```text
X_s=(1-x)^2/20*(29*p+61).                         (6.5)
```

This is structural sign-indefiniteness, not numerical error. The symbolic
audit is `scripts/quartic_transfer_counterexample.py`, and the independent
finite-mode evaluator agrees with the adversarial optimizer to machine
precision.

### 6.2 Why the Random Sweep Missed It

For (6.1), the initial cubic defect and primitive both vanish:

```text
D_s(u)=J_s(u)=0.                                  (6.6)
```

The two occupied wave pairs do not themselves form a zero-sum cubic triad.
The Euler nonlinearity creates the missing third mode, and `X_s` measures the
initial signed rate at which that mode enters `J_s`. Dense random fields put
nonzero mass on the missing mode and almost never sample this support
boundary. This explains why 480 random tests and two selected exact triads
were positive while the analytic-gradient search found a negative direction
immediately.

### 6.3 Receiving-Mode Split

The Euler nonlinearity generated by the two occupied modes has two relevant
receiving pairs:

```text
q=m-k,       |q|^2=1,
r=m+k,       |r|^2=5.
```

Grouping every occurrence of the Euler direction by its receiving wave gives

```text
X_s^(q)=-(1-x)^2*y^2/2.                           (6.7)
```

Thus the difference-mode contribution is a negative semidefinite square.
This initially suggested a low-negative/high-positive frequency-flux split,
and dense random cube fields strongly displayed that pattern. The exact sum
mode rules it out. Its pair matrix is

```text
[[5*(p+3), 7*p+18],
 [7*p+18, 10*(p+2)]],
```

with determinant

```text
(p-6)*(p+4)<0.                                   (6.8)
```

Therefore even the higher receiving mode is sign-indefinite for every
`s>0`. A frequency-only classification cannot turn `X_s` into a positive
forward transfer. The polarization geometry must be retained.

### 6.4 Consequence for the Cumulative Route

The exact identity (4.3) remains valid, but no positive-measure or
sum-of-squares interpretation of `X_s` can hold in general. Negative transfer
helps the upper bound required in (2.1); the positive polarization channel in
(6.5) remains dangerous. Therefore replacing `X_s` by `|X_s|` or by a
positive norm would destroy precisely the cancellation now exposed.

The strongest surviving target is:

```text
Decompose X_s into geometrically meaningful signed interaction channels,
then prove that its positive cumulative part is absorbed by viscosity,
negative transfer, or an endpoint correction along Navier-Stokes flow;
or construct an exact trajectory-level obstruction to such an estimate.
```

The receiving-mode audit has now eliminated a frequency-only sign rule. A
helical audit sharpens the obstruction further. For unit helical coefficients
at `k` and `m`, the two same-helicity channels agree and the two
opposite-helicity channels agree. With `p=x^3+2*x^2+3*x`, their exact values
are

```text
X_s^same
 =(1/8-sqrt(2)/10)*(1-x)^2
   *(p-(25*sqrt(2)+19)/14),                       (6.9)

X_s^opposite
 =(1/8+sqrt(2)/10)*(1-x)^2
   *(p+(25*sqrt(2)-19)/14).                       (6.10)
```

The opposite-helicity channel is strictly positive for all `s>0`. The
same-helicity channel changes sign once at

```text
s=0.273025480082694... .                          (6.11)
```

For smaller `s` it is negative; for larger `s` it is positive. Crucially, at
`s=0.5` every pure helical channel is positive while the mixed real field
(6.1) is negative. Hence helicity labels alone do not diagonalize the quartic
form: coherent cross-channel interference creates additional negative
directions.

The full helical pair-interaction matrix can in fact be computed exactly. In
the pair order `(++,+-,-+,--)`, it is real symmetric. A parity basis splits it
into two scalar channels

```text
-sqrt(2)*(p-11)/80,       sqrt(2)*(p-11)/80,      (6.12)
```

and one `2 by 2` block whose determinant is

```text
-(p+3)^2/128.                                     (6.13)
```

Since `0<p<6`, the scalar channels have opposite signs and the block has one
eigenvalue of each sign. The pair matrix therefore has inertia `(2,2)` at
every positive scale. Exact complex-amplitude Fourier checks verify the phase
terms in `scripts/quartic_transfer_helical_matrix_audit.py`.

Physical pair vectors have the rank-one form

```text
z=(A_+,A_-) tensor (B_+,B_-),
z_++*z_--=z_+-*z_-+.                              (6.14)
```

Thus the ambient matrix eigenvectors cannot automatically be interpreted as
individual velocity fields. Nevertheless, (6.12)-(6.13), together with the
rank-one negative example (6.1), proves that coherent off-diagonal helicity
terms are essential. Any cumulative estimate must preserve those terms rather
than bounding the pure channels separately.

The next target is trajectory-level: determine whether viscosity controls
the positive rank-one cone after integrating in time, or whether another
finite-mode Navier-Stokes expansion disproves that possibility too.
