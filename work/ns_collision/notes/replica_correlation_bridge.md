# Backward-Replica Correlation Bridge

Date: 2026-07-17

Status: exact representation identities and a proposed regularity bridge.
The decisive decorrelation estimate is open.

## 1. Why Replicas Enter the Actual Equation

For a smooth solution, the Constantin-Iyer stochastic flow is

```text
dX_t(a)=u(X_t(a),t)dt+sqrt(2*nu)dW_t,
A_t=X_t^(-1).
```

The velocity and vorticity representations are

```text
u_t=E P[(grad A_t)^T (u_0 o A_t)],
omega_t=E [((grad X_t) omega_0) o A_t].            (1.1)
```

The primary source explicitly identifies `grad X` as the source of possible
three-dimensional velocity and vorticity growth. See
`references/constantin_iyer/detsns.tex`, equations `e:vort-trans` and the
discussion immediately following it.

Define the random Cauchy history at a fixed Eulerian endpoint by

```text
Z_W(x,t)=((grad X_t) omega_0) o A_t(x).
```

Then, for two independent Wiener replicas,

```text
omega(x,t)=E Z_W(x,t),
|omega(x,t)|^2=E[Z_1(x,t) dot Z_2(x,t)].            (1.2)
```

Equation (1.2) is an exact algebraic reason to study two independent flows:
regularity is tied to a cross-replica correlation, not merely to the motion of
an arbitrary auxiliary pair.

The finite-replica formulation of Iyer and Mattingly uses precisely
independent Wiener copies, but also warns that one-point motions are
insufficient: the inverse and the whole flow of diffeomorphisms are required.
See `references/iyer_mattingly/particle_method.tex`, equations `eFlowN`,
`euN`, and `eOmegaN`.

## 2. Backward Histories Start in Collision

The one-point law of an inverse history ending at `(x,t)` can be viewed,
formally and for smooth periodic incompressible drift, through the backward
diffusion

```text
dQ_tau=-u(Q_tau,t-tau)d tau+sqrt(2*nu)dW_tau,
Q_0=x.
```

Two independent histories start at the same endpoint. With

```text
R_tau=Q_tau^(1)-Q_tau^(2),
G_tau=|R_tau|,
alpha_tau=R_tau dot delta u/(G_tau^2),
```

their radial equation is

```text
dG_tau=[4*nu/G_tau-alpha_tau*G_tau]d tau
       +2*sqrt(nu)d beta_tau.                      (2.1)
```

Thus the same `4*nu/G` term now has a direct interpretation:

- viscosity separates two histories that meet at the observation point;
- positive forward longitudinal strain acts backward as attraction;
- unbounded coherent stretching requires the attraction to defeat viscous
  replica separation often enough to keep the Cauchy histories correlated.

This is closer to the zeta mechanism than deterministic parcel injectivity.
The event is a square-root birth of separated histories from a common
endpoint, with a singular outward drift.

## 3. Affine Anisotropy Audit

Take a constant symmetric strain matrix `S` and the backward relative SDE

```text
dR=-S R d tau+2*sqrt(nu)dB,
R_0=0.
```

Its covariance `C=E[R R^T]` obeys

```text
C'=-S C-C S+4*nu I.                               (3.1)
```

Along an eigenvector with forward strain eigenvalue `lambda`,

```text
C_lambda(tau)=
  (2*nu/lambda)(1-exp(-2*lambda*tau)), lambda!=0,
  4*nu*tau,                              lambda=0. (3.2)
```

For the incompressible strain

```text
S=diag(-a,-a,2a), a>0,
```

the two transverse variances and stretching-direction variance are

```text
C_perp=(2*nu/a)(exp(2*a*tau)-1),
C_parallel=(nu/a)(1-exp(-4*a*tau)).                (3.3)
```

Consequences:

1. Radial distance grows because backward transverse separation is strong.
2. Along the forward stretching direction, separation saturates at viscous
   width `sqrt(nu/a)`.
3. Radial non-collision alone does not control vortex stretching.
4. If the high-strain region is spatially localized, transverse escape can
   end the common stretching history.
5. Uniform affine strain evades this escape mechanism, but it has infinite
   energy on `R^3` and is not periodic.

The first potentially Clay-relevant quantity is therefore not just `G`; it is
the joint residence time of two independent histories inside a localized,
directionally coherent stretching region.

## 4. Candidate Residence-Time Functional

Let `E_kappa(s)` be a superlevel region where the positive strain eigenvalue
is at least `kappa`, with a direction field `e(x,s)`. For histories ending at
`(x,t)`, define schematically

```text
R_kappa(x,t)
 = E integral_0^t
     1_{Q_1,Q_2 in E_kappa(t-tau)}
     |e(Q_1) dot e(Q_2)|
     kappa(t-tau) d tau.                           (4.1)
```

A useful theorem would bound the exponentially weighted version of (4.1)
uniformly in `x,t` using only quantities available from the Leray energy
inequality plus incompressibility and the pressure relation.

This would express a physical statement:

```text
viscosity separates backward histories faster than a finite-energy,
spatially localized strain field can keep both histories coherently trapped.
```

No such unconditional bound is claimed here.

## 5. Newtonian Boundary Defect

The three-dimensional reciprocal gap is critical. Applying Ito's formula to
`F=1/G` in (2.1) gives, away from `G=0`,

```text
d(1/G)=alpha_tau*(1/G)d tau
       -2*sqrt(nu)/G^2 d beta.                    (5.1)
```

The viscous drift cancels exactly because `1/|x|` is harmonic away from the
origin. More generally,

```text
d(G^(-q)) has viscous drift
  2*nu*q*(q-1)*G^(-q-2)d tau.                     (5.2)
```

Thus `q=1` is the exact radial critical exponent. Fractional inverse gaps with
`0<q<1` receive damping, while more singular positive powers receive the
opposite Ito drift before angular cancellation is considered.

For pure relative diffusion started at separation `r>0`, `1/G` is a positive
strict local martingale, not a true martingale. Its expectation is

```text
E_r[1/G_tau]
  = erf(r/sqrt(8*nu*tau))/r,

1/r-E_r[1/G_tau]
  = erfc(r/sqrt(8*nu*tau))/r > 0.                 (5.3)
```

The process almost surely never reaches `G=0`, yet arbitrarily close
approaches create a nonzero boundary defect. At an initially coincident
endpoint,

```text
E_0[1/G_tau]=1/sqrt(2*pi*nu*tau).                 (5.4)
```

This is exactly the heat regularization of the Newtonian kernel. The
connection matters because the pressure, Biot-Savart law, and strain are
built from the Newtonian potential and its derivatives. Those derivatives
are harmonic off the diagonal as well, although their stronger singularity
requires principal-value and angular cancellation rather than absolute-value
estimates.

This sharpens the proposed mechanism:

```text
viscosity does not need particles to hit the collision set;
near-collision boundary flux already regularizes the Newtonian interaction.
```

The open question is whether the tensorial cancellation in the actual
Navier-Stokes kernels turns this boundary defect into a coercive estimate for
the cross-replica deformation correlation.

## 6. Exact Target and Non-Promotion Gates

The strongest clean target is a two-copy deformation-correlation estimate

```text
sup_(x,t<T) E[Z_1(x,t) dot Z_2(x,t)] <= C(u_0,nu,T), (5.1)
```

or a scale-critical integral consequence strong enough to invoke a standard
Navier-Stokes continuation criterion.

The following statements are insufficient by themselves:

- `Q_1(tau)!=Q_2(tau)` for every `tau>0`;
- positive expected squared separation;
- almost-everywhere deterministic parcel injectivity;
- preservation of `det grad X=1` while the flow is already smooth;
- a bound that assumes `integral ||grad u||_infinity dt<infinity`.

The last condition would be circular. The target must be derived from
energy-class information and Xi-like structural cancellation, here meaning
incompressibility, Biot-Savart/pressure nonlocality, and replica geometry.

## 7. Immediate Research Gates

1. Derive (1.2) and (2.1) in a common backward-flow convention with all
   transposes and time orientations checked against the primary formula.
2. Express the cross-replica deformation product as a two-point parabolic
   system on `(x,y)` and identify its boundary data on `x=y`.
3. Split the stretching contribution into common-residence and separated
   histories.
4. Test whether the separated term has cancellation from zero-mean vorticity,
   vorticity-direction decorrelation, or the Leray projection.
5. Rewrite the Newtonian and strain kernels along the relative Bessel process
   and isolate the strict-local-martingale boundary defect before taking
   principal values.
6. Stress-test any proposed estimate on affine strain, Burgers vortices,
   shear flows, and concentrated self-similar profiles.
7. Determine the exact scale at which the energy inequality fails to control
   common residence. That failure term, rather than a generic norm, becomes
   the next theorem target.

