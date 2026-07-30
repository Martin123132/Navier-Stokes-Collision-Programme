# Two-Point Vorticity Collision System

Date: 2026-07-17

Status: exact smooth-solution identities and a precise kernel-level
obstruction. No unconditional regularity estimate is claimed.

## 1. Exact Tensor Equation

Write the three-dimensional vorticity equation as

```text
partial_t omega + u dot grad omega
  = A omega + nu Delta omega,
A=grad u.
```

For two independent spatial variables define

```text
Omega(x,y,t)=omega(x,t) tensor omega(y,t).
```

The product rule gives the exact six-dimensional equation

```text
partial_t Omega
 +u(x) dot grad_x Omega+u(y) dot grad_y Omega
 =nu(Delta_x+Delta_y)Omega
  +A(x)Omega+Omega A(y)^T.                         (1.1)
```

On the diagonal,

```text
tr Omega(x,x,t)=|omega(x,t)|^2.                   (1.2)
```

Thus the desired vorticity bound is a diagonal bound for a matrix-valued
parabolic system. The stretching potential is not scalar and has no fixed
sign.

## 2. Centre and Separation Coordinates

Set

```text
c=(x+y)/2,
r=x-y,
x=c+r/2,
y=c-r/2.
```

Then

```text
grad_x=(1/2)grad_c+grad_r,
grad_y=(1/2)grad_c-grad_r,
Delta_x+Delta_y=(1/2)Delta_c+2Delta_r.            (2.1)
```

With

```text
u_+=(u(x)+u(y))/2,
delta u=u(x)-u(y),
```

equation (1.1) becomes

```text
partial_t Omega
 +u_+ dot grad_c Omega+delta u dot grad_r Omega
 =nu[(1/2)Delta_c+2Delta_r]Omega
  +A(x)Omega+Omega A(y)^T.                        (2.2)
```

The relative diffusion generator is exactly `2*nu*Delta_r`. In radial and
angular variables `g=|r|`, `theta=r/g`,

```text
2*nu*Delta_r
 =2*nu[partial_gg+(2/g)partial_g+(1/g^2)Delta_S2]. (2.3)
```

The radial part of (2.3) is the squared-Bessel dimension-three generator and
produces the replica gap drift `4*nu/g`. The angular term is equally large at
small separation and cannot be dropped.

## 3. Backward Two-Copy Representation

For smooth coefficients, (2.2) has a matrix Feynman-Kac representation using
two independent backward diffusions. Schematically, with `Q_1(0)=x`,
`Q_2(0)=y`,

```text
dQ_i(s)=-u(Q_i(s),t-s)ds+sqrt(2*nu)dW_i(s),
J_i'(s)=J_i(s)A(Q_i(s),t-s),
J_i(0)=I,

Omega(x,y,t)
 =E[J_1(t) Omega_0(Q_1(t),Q_2(t)) J_2(t)^T].       (3.1)
```

Time ordering is implicit when the matrices do not commute. Formula (3.1)
agrees with the constant-affine check `omega(t)=exp(tA)omega_0`.

At `x=y`, the two histories begin in collision and separate under (2.3).
Their deformation cross-matrix satisfies

```text
C=J_1^T J_2,
C'=A(Q_1)^T C+C A(Q_2).                          (3.2)
```

At the common endpoint `Q_1=Q_2=x` and `C=I`, so initially

```text
C'=A(x)^T+A(x)=2S(x).                            (3.3)
```

More generally, if `C` is momentarily the identity but the paths have
separated, writing `A=S+W` gives

```text
C'=S(Q_1)+S(Q_2)+W(Q_2)-W(Q_1).                  (3.4)
```

At collision, differential rotation vanishes and strain doubles. Once the
replicas and their deformation matrices separate, equation (3.2) transports
the full cross-matrix and differential rotation can decorrelate the Cauchy
histories. This is an exact place where separation can affect the sign of the
cross-replica vorticity correlation.

## 4. Diagonal Curvature and Dissipation

For the scalar correlation

```text
K(c,r)=omega(c+r/2) dot omega(c-r/2),
```

one has at `r=0`

```text
grad_r K=0,
Delta_r K
  =(1/2)omega dot Delta omega-(1/2)|grad omega|^2,
Delta_c K
  =2 omega dot Delta omega+2|grad omega|^2.        (4.1)
```

Consequently,

```text
nu[(1/2)Delta_c+2Delta_r]K|_(r=0)
 =2*nu*omega dot Delta omega,                     (4.2)
```

as required by the usual enstrophy equation.

Equivalently, the vorticity structure function

```text
D(c,r)=|omega(c+r/2)-omega(c-r/2)|^2
```

satisfies

```text
Delta_r D(c,0)=2|grad omega(c)|^2.                (4.3)
```

Viscous enstrophy dissipation is therefore encoded in the curvature with
which two nearby vorticity values separate across the collision diagonal.

## 5. Exact Strain Kernel

The symmetric strain has the Biot-Savart representation

```text
S_ij(x)
 =3/(8*pi) PV integral
   [((r cross omega(y))_i r_j
     +(r cross omega(y))_j r_i)/|r|^5]dy,
r=x-y.                                             (5.1)
```

This formula is recorded, for example, in Appendix A.3 of the local reference
`references/constantin_vicol_wu_lagrangian_analyticity.pdf`.

For a fixed vector `w`, each component of the kernel

```text
K(r)w
 =3/(8*pi)
  [((r cross w) tensor r)+r tensor (r cross w)]/|r|^5
```

has the following exact properties away from `r=0`:

```text
K^T=K,
tr K=0,
K(lambda r)=lambda^(-3)K(r),
Delta_r K=0,
spherical mean of K=0.                            (5.2)
```

Its angular dependence is a pure degree-two spherical harmonic. In the form

```text
K(r)=g^(-3)Y_2(theta),
Delta_S2 Y_2=-6Y_2.                               (5.3)
```

## 6. Radial Repulsion Versus Angular Mixing

Apply the relative viscous generator to the strain kernel. The radial part is

```text
[partial_gg+(2/g)partial_g]g^(-3)=6g^(-5),
```

while the angular part is

```text
(1/g^2)Delta_S2[g^(-3)Y_2]=-6g^(-5)Y_2.          (6.1)
```

After multiplication by `2*nu`, the two contributions are

```text
+12*nu*g^(-5)Y_2,
-12*nu*g^(-5)Y_2.                                (6.2)
```

They cancel exactly. This is the tensor counterpart of the reciprocal-gap
criticality: the Newtonian strain kernel is harmonic off collision.

Therefore the `4*nu/g` radial repulsion cannot, by itself, produce a positive
coercive drift for vortex stretching. Any argument that keeps only the radial
gap and discards the angular generator gives a false sign advantage.

## 7. What Survives the Cancellation

The cancellation in (6.2) is local away from `r=0`. It does not remove the
distributional collision boundary. Heat evolution replaces the singular
Newtonian potential by

```text
1/g -> erf(g/sqrt(8*nu*tau))/g,
```

and the strain kernel by the corresponding derivatives of this regularized
potential. The difference is concentrated at `g` comparable to
`sqrt(nu*tau)` and has zero spherical mean but nontrivial tensor structure.

The live mechanism is now narrower and more precise:

```text
not a positive radial drift in the strain kernel,
but a strict near-collision boundary defect combined with
angular cancellation and cross-replica decorrelation.                (7.1)
```

### 7.1 Exact Heat-Attenuation Multiplier

The regularization can be computed without losing the angular tensor. Put

```text
L=sqrt(8*nu*tau),
z=g/L,
f_tau(g)=erf(g/L)/g.
```

For a radial function, the anisotropic coefficient of its Hessian is

```text
b_tau(g)=f_tau''(g)-f_tau'(g)/g.
```

For the unsmoothed Newtonian potential `f(g)=1/g`, this coefficient is
`3/g^3`. Direct differentiation gives

```text
b_tau(g)=[3/g^3]M(z),

M(z)=erf(z)
     -(2*z+(4/3)*z^3)exp(-z^2)/sqrt(pi).          (7.2)
```

The isotropic part of the radial Hessian cancels against the skew
cross-product matrix in the strain formula. Consequently the entire
regularized strain kernel is

```text
K_tau(r)=M(|r|/L)K(r),
B_tau(r)=[1-M(|r|/L)]K(r).                        (7.3)
```

This is stronger than qualitative localization: heat flow preserves the
exact degree-two angular tensor and changes only its radial amplitude. The
multiplier obeys

```text
M'(z)=8*z^4*exp(-z^2)/(3*sqrt(pi)) > 0,
M(0)=0,
M(infinity)=1,
M(z)=8*z^5/(15*sqrt(pi))+O(z^7).                 (7.4)
```

Thus `0<M(z)<1` for `z>0`, and the regularized `g^(-3)` kernel vanishes like

```text
K_tau(r)=O(g^2/L^5)
```

at collision. The defect retains the original sign-indefinite angular
tensor, however, so positive radial attenuation is not by itself a coercive
vortex-stretching estimate.

## 8. Strongest Current Target

Let `K_tau` be the heat-regularized strain kernel and

```text
B_tau=K-K_tau
```

the collision-boundary defect. A useful non-circular theorem would estimate
the contraction of `B_tau` with the two-copy Cauchy covariance using Leray
energy-class data, while treating the heat-regularized far part by standard
Calderon-Zygmund bounds.

Schematically, the missing estimate is

```text
integral_0^t E[
  deformation_1 * B_tau(Q_1-Q_2)
  * deformation_2 * vorticity_data
]d tau
 <= controlled critical quantity.                 (8.1)
```

The right side must not contain `integral ||grad u||_infinity`, an assumed
vorticity supremum, or any equivalent continuation criterion.

## 9. Outcome of This Stage

Established:

- exact two-point tensor PDE;
- exact centre/separation diffusion coefficients;
- exact appearance of the Bessel radial generator;
- exact degree-two harmonic structure of the strain kernel;
- exact cancellation of radial repulsion by angular diffusion off collision;
- exact survival of a distributional near-collision boundary defect.
- exact positive scalar attenuation of the strain kernel under relative heat
  flow, including quadratic removal of its collision singularity.

Open:

- a signed or norm-coercive estimate for the tensor boundary defect;
- control of deformation weights at Leray regularity;
- conversion of replica decorrelation into a critical continuation bound.

The next-stage file `heat_scale_cubic_cancellation.md` sharpens this last
point. It proves that the global high-pass stretching defect has a double
zero at heat scale zero, rewrites its derivative as an exact Gaussian
two-copy determinant, and isolates a collision-rigidity continuation
criterion with all low-frequency strain controlled by kinetic energy.
