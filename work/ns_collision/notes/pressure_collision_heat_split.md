# Pressure Hessian collision heat split

## Objective

The maximal-strain equation isolated the nonlocal forcing

```text
-P_33+|omega_perp|^2/4.
```

This note puts the trace-free pressure Hessian into the same heat-scale and
collision-kernel framework used for strain. The pressure kernel has the same
degree-two attenuation law and collision increment cancellation. It does not
inherit the stronger cubic double zero of the global vortex-stretching
defect.

## Trace and trace-free pressure

Set

```text
f=|S|^2-|omega|^2/2=-Delta p.
```

Since `tr(Hess p)=-f`, write

```text
P=Hess p=P0-(f/3)I,
P0=P+(f/3)I.
```

The trace-free Hessian is the singular integral

```text
P0_ij(x)=PV integral T_ij(r)f(x-r)dr,

T_ij(r)
 =(3r_i r_j-|r|^2 delta_ij)/(4pi|r|^5).
```

Every component is a pure degree-two spherical harmonic with zero spherical
mean. In Fourier variables its multiplier is

```text
-k_i k_j/|k|^2+delta_ij/3.
```

The Frobenius norm squared of this matrix is `2/3`, independently of the
wave direction. Thus

```text
||P0||_2^2=(2/3)||f||_2^2
```

whenever `f` belongs to `L^2`.

## Exact heat attenuation

Let

```text
P0_s=exp(s Delta)P0,       L=2sqrt(s),       z=|r|/L.
```

The heat-regularized Newtonian potential is `erf(z)/|r|`. Its Hessian has an
anisotropic multiplier and a nonzero isotropic trace. The `f/3` trace
correction in `P0` cancels that heat trace exactly. The resulting trace-free
kernel is

```text
T_s(r)=M(z)T(r),

M(z)=erf(z)
     -(2z+(4/3)z^3)exp(-z^2)/sqrt(pi).
```

This is exactly the multiplier found for the strain kernel. Consequently,

```text
B_s^P=P0-P0_s
```

is a collision-boundary defect with kernel `(1-M)T`.

For a unit direction `e`,

```text
e dot T(r)e
 =[3(e dot r_hat)^2-1]/(4pi|r|^3).
```

The angular factor has zero spherical mean, so the defect has the exact
increment representation

```text
B_s^P,ee(x)
 =PV integral (1-M(|r|/L)) e dot T(r)e
    [f(x-r)-f(x)]dr.
```

This removes the constant collision jet. Bounding the remaining scalar
increment requires regularity of the quadratic source `f` that is not
provided by the Leray energy inequality.

## Maximal-strain reaction

Using `P=P0-(f/3)I`, the simple maximal-eigenvalue equation becomes

```text
(D_t-nu Delta)lambda_3
 =R_local-P0_33-F_frame,
```

where

```text
R_local
 =|S|^2/3-lambda_3^2
  +|omega_perp|^2/12-|omega_parallel|^2/6,

F_frame
 =2nu sum_(k,j<3)
    (lambda_3-lambda_j)|e_j dot partial_k e_3|^2.
```

Aligned vorticity contributes negatively, while transverse vorticity
contributes positively.

Put `r=lambda_2/lambda_3` and use
`lambda_1=-lambda_2-lambda_3`. The strain-only local reaction is

```text
[|S|^2/3-lambda_3^2]/lambda_3^2
 =(2r^2+2r-1)/3.
```

It is nonpositive when

```text
r<=(sqrt(3)-1)/2 approximately 0.366025.
```

It is strictly negative for the axisymmetric tube (`r=-1/2`) and can be
positive for strongly biaxial stretching.

## Low pressure and collision defect

Split

```text
P0=P0_s+B_s^P.
```

The projected regularized kernel is bounded. Direct optimization gives

```text
||P0_s,ee||_infinity
 <=C_P s^(-3/2)||f||_1,

C_P approximately 0.00325224112.
```

For divergence-free velocity on `R^3`,

```text
||S||_2^2=(1/2)||omega||_2^2,
```

and hence

```text
||f||_1<=||omega||_2^2.
```

This controls the regularized pressure by enstrophy, not kinetic energy. It
therefore does not independently prevent enstrophy growth.

The exact heat-scale maximum gate is

```text
R_local-P0_s,33-B_s^P,33 <=F_frame.
```

The live collision term is the signed high-frequency pressure defect
`-B_s^P,33`, expressed by the scalar source increment above.

## No pressure double zero

The strain-vorticity cubic defect from
`heat_scale_cubic_cancellation.md` has a double zero at `s=0`. The pressure
defect does not. For a scalar Fourier mode with wave vector parallel to
`e_3`, the trace-free multiplier has `33` component `-2/3`, so

```text
B_s^P,33=-(2/3)(1-exp(-s))f_mode
        =-(2/3)s f_mode+O(s^2).
```

The first heat derivative is nonzero. The shared degree-two collision kernel
therefore gives angular and increment cancellation, but not the Miller cubic
orthogonality.

More generally, commutation with the Laplacian gives the exact leading term

```text
B_s^P
 =s[Hess f-(Delta f/3)I]+O(s^2).
```

The missing estimate is therefore a signed trace-free source-Hessian estimate
at first heat order. The viscous frame penalty controls selected first
derivatives of the strain eigenframe, so it does not automatically dominate
this full second-derivative tensor.

## Verdict and next gate

Established:

1. the trace-free pressure Hessian has exactly the same heat attenuation
   multiplier as strain;
2. its collision defect removes constant source jets and depends on
   increments of `|S|^2-|omega|^2/2`;
3. aligned vorticity and sufficiently non-biaxial strain give favourable
   local reaction;
4. the pressure defect has only a simple heat-scale zero and is not controlled
   by energy-level norms.

The next useful question is whether the pressure source increment can be
paired with `F_frame` or with the two-replica collision damping without taking
an absolute value. A raw norm estimate will require unavailable derivatives
of `f`. The signed angular integral is essential.

The kernel identities, local reaction, low-frequency constant, and Fourier
stress test are reproduced by
`scripts/pressure_collision_kernel_audit.py`.

The proposed pointwise pairing with frame coherence is tested and disproved
in `pressure_frame_pairing_obstruction.md`, which also derives the surviving
localized pressure boundary identity.
