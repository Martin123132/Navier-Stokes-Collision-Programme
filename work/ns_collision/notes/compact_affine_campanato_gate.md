# Compact full-affine Campanato gate

## Purpose

The affine-to-taper no-go makes the compact full-affine stopped cylinder the
consistent baseline. This note fits that baseline from explicit mollified
Leray data and separates the physical error into four scale-critical pieces.

The decomposition is exact. It also proves that the monotone amplitude
envelope cannot classify every coherence failure: a symmetric strain can
rotate at fixed eigenvalues, exceed the drift budget, and never trigger a
level split.

## Explicit compact mollifier

Use the radial compact `C^3` kernel

```text
rho_0(y)=3465/(512 pi)(1-|y|^2)^4_+,

rho_(kL)(x)=(kL)^(-3)rho_0(x/(kL)),
k=3/4.                                                 (1)
```

Its support ball fits inside the compact cylinder

```text
r<2L,       |z|<3L/4.                                 (2)
```

Exact moments and derivative norms are

```text
int rho_0=1,
int y_i^2 rho_0=1/13,

||rho_0||_2^2=24255/(8398 pi),
||grad rho_0||_2^2=10395/(221 pi),
||D^2 rho_0||_2^2=31185/(26 pi).                      (3)
```

The scaled derivatives give explicit Caratheodory bounds for the centre and
spin-frame ODE. A compact `C^infinity` approximation can be used in a final
weak-limit construction; `C^3` is sufficient for the convolution identities
and spatial Lipschitz bounds used here.

## Exact full-affine decomposition

For the physical backward drift define

```text
U_L=int rho_(kL)b,
A_L=int rho_(kL)grad b=S_L+W_L.                       (4)
```

The frame follows `W_L`. At entry `tau`, diagonalize `S_L(tau)` and transport
that fixed body-frame matrix with the spin frame:

```text
S_ref(t)=O(t)O(tau)^T S_L(tau) O(tau)O(t)^T.          (5)
```

Define the instantaneous affine Campanato remainder

```text
R_L=b-U_L-A_L(x-c).                                   (6)
```

Subtracting the moving translation, spin, and full-affine reference gives
the exact identity

```text
e=R_L+[S_L-S_ref](x-c).                               (7)
```

Unlike the tapered reference, (7) vanishes for an exact affine drift whose
symmetric matrix is fixed in the spin-following frame.

For stretching, Weyl's inequality gives the sharper decomposition

```text
q_+<=[lambda_max(S(x))-lambda_max(S_L)]_+
     +[lambda_max(S_L)-lambda_ref]_+.                 (8)
```

Orientation drift appears in (7) but not automatically in (8); it should not
be double charged as eigenvalue growth.

## Critical functionals

Put

```text
C_3=||R_L||_3/nu,

P=||[lambda_max(S)-lambda_max(S_L)]_+||_(3/2)/nu,

T=L^2||S_L-S_ref||_op/nu,

G=L^2[lambda_max(S_L)-lambda_ref]_+/nu.               (9)
```

On (2), the exact geometry constants are

```text
V_hat=6 pi,
V_hat^(2/3)=7.08273101600,

[int_D |y|^3 dy]^(1/3)=4.07699407678.                 (10)
```

Consequently

```text
||e||_3/nu<=C_3+4.07699407678 T,

||q_+||_(3/2)/nu<=P+7.08273101600 G.                 (11)
```

Using the compact full-affine `H/L=0.75` transfer constants, the no-restart
sector condition becomes

```text
sqrt(S_3)C_3+(1+d)S_3 P
+sqrt(S_3)(4.07699408)T
+(1+d)S_3(7.08273102)G<d,

d=0.130483946925.                                    (12)
```

If temporal drift is the only error, (12) permits

```text
T<0.07491.                                             (13)
```

If the conservative bounds `T=G` are both charged, the threshold is
approximately `0.04073`. When every abort is paid by a true pair split, the
corresponding thresholds fall to approximately `0.0239` and `0.01347`.

At Leray regularity all four quantities in (9) are finite almost everywhere:
`u` is locally `L^6`, strain is `L^2`, and the cylinder has finite volume.
Nothing in the Leray inequality makes them uniformly small.

## Failure not caused by amplitude growth

Take

```text
D=diag(-1,0,1),
S(theta)=R(theta)D R(theta)^T,                         (14)
```

where `R` rotates the first and third axes. Every eigenvalue, including
`lambda_max=1`, is constant. Therefore the monotone amplitude envelope does
not move and no dyadic split occurs. Nevertheless

```text
||S(theta)-D||_op=2|sin theta|.                       (15)
```

The drift-only no-restart threshold is crossed after only about two degrees
of relative rotation; the split-paid threshold is tighter still. This is a
kinematic counterexample to the claim that every coherence failure forces an
amplitude split.

Spatial Campanato failure and spectral-shape change have the same issue.
They require either:

1. a dynamic `L^2` occupation estimate for the bad branch;
2. accumulated decay before same-level reference reset;
3. a nonautonomous visit theorem that lets the affine reference vary during
   the stopped visit.

The explicit mollifier constants, exact decomposition, cylinder integrals,
sector thresholds, and constant-spectrum rotation stress test are reproduced
by `scripts/compact_affine_campanato_gate_audit.py`.
