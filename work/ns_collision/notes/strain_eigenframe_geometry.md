# Strain eigenframe geometry and the pressure gate

## Purpose

The three-dimensional tube audit showed that energy estimates alone do not
provide the affine coherence needed by the localized collision mechanism.
This note asks whether the Navier-Stokes evolution of the strain eigenframe
supplies that missing geometry.

The answer is mixed. Viscosity produces an exact nonpositive penalty for
spatial eigenframe variation, and vorticity misalignment can only reduce
stretching below the maximal strain eigenvalue. The trace-free pressure
Hessian remains an uncontrolled nonlocal forcing of both eigenvalue growth
and frame rotation.

All pointwise eigenvector formulas below are stated where the maximal strain
eigenvalue is simple. Eigenvalue collisions require projector or viscosity-
solution formulations and are a separate technical obligation.

## Strain evolution

Write

```text
A=grad u=S+Omega,       P=Hess p.
```

The velocity-gradient equation and its symmetric part are

```text
D_t A=-A^2-P+nu Delta A,

D_t S=-S^2-Omega^2-P+nu Delta S
     =-S^2+(|omega|^2 I-omega tensor omega)/4
       -P+nu Delta S.
```

Let `lambda_1<=lambda_2<lambda_3` and let `e_i` be an orthonormal strain
eigenframe. Direct projection onto `e_3` gives

```text
D_t lambda_3
  =-lambda_3^2
   +|omega_perp|^2/4
   -P_33
   +nu e_3 dot Delta S e_3,
```

where

```text
|omega_perp|^2=|omega|^2-(omega dot e_3)^2.
```

## Viscous eigenframe penalty

Second-order eigenvalue perturbation gives

```text
e_3 dot Delta S e_3
 =Delta lambda_3
  -2 sum_(k,j<3)
      |e_j dot (partial_k S)e_3|^2/(lambda_3-lambda_j).
```

Since

```text
e_j dot partial_k e_3
 =e_j dot (partial_k S)e_3/(lambda_3-lambda_j),
```

the maximal-eigenvalue equation is

```text
(D_t-nu Delta)lambda_3
 =-lambda_3^2-P_33+|omega_perp|^2/4
  -2nu sum_(k,j<3)
     (lambda_3-lambda_j)|e_j dot partial_k e_3|^2.
```

The final term is nonpositive. Viscosity therefore penalizes spatial
rotation of the stretching eigenvector, weighted by the spectral gap. This
is an equation-derived coherence mechanism, not an assumed tube property.

It is strongest where the eigenvalue gaps are large. Near a repeated maximal
eigenvalue, the vector `e_3` is not well defined and this representation must
be replaced by the maximal eigenspace projector.

## Vorticity alignment

For `omega=|omega| xi`, define the actual stretching rate

```text
alpha=xi dot S xi.
```

In the strain eigenframe,

```text
lambda_3-alpha
 =(lambda_3-lambda_1)xi_1^2
 +(lambda_3-lambda_2)xi_2^2
 >=0.
```

Thus vorticity misalignment never increases stretching above `lambda_3`.
If the affine reference rate is chosen as the maximal eigenvalue at the core
centre, then for any deformation direction `n`,

```text
[n dot S(x)n-lambda_3(center)]_+
 <=[lambda_3(x)-lambda_3(center)]_+.
```

The positive stretching error in the form estimate is therefore controlled
by spatial excess of `lambda_3`, not by orientation misalignment.

The vorticity direction equation is

```text
D_t xi
 =(I-xi tensor xi)Sxi
  +nu[Delta xi+|grad xi|^2 xi
      +2 grad(log|omega|) dot grad xi].
```

If `mu=xi dot e_3`, its strain-only contribution is

```text
(D_t mu)_strain=mu(lambda_3-alpha).
```

Material stretching therefore tends to align `xi` with `e_3` in magnitude.
Viscous direction diffusion and eigenframe rotation supply the remaining
terms.

## Eigenframe rotation

For `j<3`,

```text
e_j dot D_t e_3
 =[-omega_j omega_3/4-P_j3
   +nu e_j dot Delta S e_3]
  /(lambda_3-lambda_j).
```

The pressure Hessian and viscous off-diagonal strain Laplacian can rotate the
frame. Small eigenvalue gaps amplify that rotation. The vorticity term
vanishes when vorticity is exactly parallel to `e_3`.

## General affine spectrum

The localized spectral mechanism is not restricted to
`(-a,-a,2a)`. For any constant symmetric trace-free strain with ordered
eigenvalues, the backward gauge produces an anisotropic oscillator with
full-space ground energy

```text
E_0=(|lambda_1|+|lambda_2|+lambda_3)/2.
```

Relative to maximal deformation growth `lambda_3`,

```text
E_0-lambda_3=max(lambda_2,0).
```

If `lambda_2<=0`, full space gives the same exact balance seen in the
axisymmetric model. If `lambda_2>0`, there is already a positive gauged
`L^2` excess. Any bounded Dirichlet localization adds a strict margin in
either case.

This does not remove the need to compare physical and gauged norms or to
control re-entry. It shows that transverse axisymmetry is not essential to
the local spectral calculation.

## The pressure obstruction

The pressure trace is fixed locally:

```text
Delta p=-|S|^2+|omega|^2/2.
```

Its trace-free Hessian is not. Two local harmonic pressure jets demonstrate
the missing information:

```text
h_tilt=gamma*x*z,
Hess h_tilt has P_13=P_31=gamma;

h_axial=gamma*(z^2-x^2)/2,
Hess h_axial=diag(-gamma,0,gamma).
```

Both have zero Laplacian. The first changes eigenframe rotation by an
arbitrarily signed amount; the second changes maximal-eigenvalue forcing.
Thus the pressure Poisson trace cannot close either equation pointwise.

Globally,

```text
P_ij=R_i R_j (|S|^2-|omega|^2/2),
```

so this is a nonlocal far-field problem rather than a free local pressure.
Any closure must exploit cancellation, coherence, or a critical-space bound
for this Riesz-transform contribution.

## Conditional gate

At a spatial maximum of a simple `lambda_3`, the diffusion term is
nonpositive. A sufficient pointwise condition preventing positive material
growth there is

```text
-P_33+|omega_perp|^2/4
 <=lambda_3^2
   +2nu sum_(k,j<3)
      (lambda_3-lambda_j)|e_j dot partial_k e_3|^2.
```

This condition is not established by Leray energy. It identifies the next
object to test: the pressure-vorticity forcing excess relative to self-strain
and viscous frame coherence. A useful next calculation is to express its
pressure part through the heat-regularized two-point strain kernel already
developed in this corpus and test whether collision cancellation improves its
critical norm.

The identities and pressure stress tests are reproduced by
`scripts/strain_eigenframe_geometry_audit.py`.

The trace-free pressure Hessian is put into the collision heat split in
`pressure_collision_heat_split.md`.
