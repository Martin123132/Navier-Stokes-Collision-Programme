# Three-dimensional form and Leray gate

## Objective

The moving transverse-tube estimate gives a workable model criterion, but it
does not yet use the natural three-dimensional quantities of a Leray-Hopf
solution. This note performs the full three-dimensional gauge transform,
derives the critical form bound for one localized visit, and tests whether
finite energy and dissipation automatically close the visit and return gates.

They do not. The calculation nevertheless identifies the missing hypotheses
precisely: local affine coherence inside the core and a weighted
return-versus-stretching condition outside it.

## Full affine core

For the forward strain

```text
S = diag(-a,-a,2a),       a>0,
```

the backward drift is

```text
b_0(x) = -Sx = (a x, a y, -2a z).
```

It is divergence-free and is the gradient of

```text
U(x) = a(x^2+y^2)/2-a z^2.
```

For one deformation factor aligned with the stretching axis, the killed
generator is

```text
A_0 = nu Delta+b_0 dot grad+2a.
```

Conjugation by `exp(-U/(2nu))` gives

```text
A_0  ~  -K,

K = -nu Delta
    + a^2(x^2+y^2+4z^2)/(4nu)
    - 2a.
```

The unshifted anisotropic oscillator has full-space ground energy exactly
`2a`. Hence `K>=0` on the full space and, by strict Dirichlet domain
monotonicity,

```text
K >= m > 0
```

on every bounded core. The transverse Kummer margin from
`localized_strain_tube.md` is already a lower bound for a finite cylinder;
finite axial killing can only increase it.

## Three-dimensional perturbation

Write the actual backward drift and stretching potential as

```text
b = b_0+e,       c=2a+delta_s.
```

After the same gauge and an energy integration by parts, the effective error
is

```text
q = delta_s - 1/2 div(e) - b_0 dot e/(2nu).
```

Because both the Navier-Stokes drift and `b_0` are divergence-free,
`div(e)=0`. Thus

```text
q = delta_s - b_0 dot e/(2nu).
```

This is a genuine improvement over the transverse projection. It also shows
why affine coherence matters: the gauge couples the large reference drift to
the non-affine velocity remainder. Finiteness of strain alone does not make
that product small.

## Critical L3/2 form bound

For zero-boundary functions on a bounded three-dimensional core, zero
extension and the sharp Sobolev inequality give

```text
||psi||_6^2 <= S3 ||grad psi||_2^2,

S3 = 4^(2/3)/(3 pi^(4/3)).
```

Put `Q=||q_+||_(3/2)`. Holder and Sobolev imply

```text
integral q |psi|^2 <= S3 Q ||grad psi||_2^2.
```

If `kappa=<psi,Kpsi>/||psi||_2^2`, then `kappa>=m` and

```text
nu||grad psi||_2^2
    <= <psi,Kpsi>+2a||psi||_2^2.
```

It follows that the perturbation is relatively form-bounded by

```text
alpha = (S3 Q/nu)(m+2a)/m.
```

The explicit sufficient visit condition is therefore

```text
Q < nu m/[S3(m+2a)].
```

This is the scale-critical spatial norm. Using the conservative transverse
margin, the permitted dimensionless budgets are:

| tube `R` | `m/a` | allowed `Q/nu` |
|---:|---:|---:|
| 0.1 | 56.8373109669 | 5.29169897582 |
| 0.5 | 10.5936192507 | 4.60795495410 |
| 1 | 4.83762216791 | 3.87562073575 |
| 2 | 2 | 2.73895204477 |
| 5 | 0.419753935419 | 0.950250257174 |
| 10 | 0.0549460799925 | 0.146470683209 |
| 20 | 0.000804267034 | 0.002201963354 |
| 50 | 6.653468887e-10 | 1.822353221e-9 |

## Translation to local Leray quantities

Let the core have volume `|Omega|` and diameter `ell`. Choose the centre
velocity so that the remainder `e` has zero mean, and let `C_P` be its
Poincare constant after scaling by `ell`. Then

```text
||delta_s||_(3/2)
    <= |Omega|^(1/6)||delta_s||_2,

||b_0 dot e/(2nu)||_(3/2)
    <= |Omega|^(1/6)
       C_P ||B|| ell^2 ||grad e||_2/(2nu).
```

Both terms are finite at almost every time for a Leray-Hopf solution. The
required inequality is smallness, however, and the energy inequality supplies
no such pointwise-in-time local affine-coherence bound.

Trying to use the stronger spatial `L^2` norm without pointwise smallness also
exposes the known parabolic gap. Three-dimensional interpolation and Young's
inequality produce a scalar growth term proportional to

```text
||q(t)||_2^4/nu^3.
```

Leray control provides only `||grad u(t)||_2` in time `L^2`. A spike of height
`sqrt(N)` and duration `1/N` has fixed time `L^2` norm but time `L^4` fourth
power equal to `N`. Equivalently, the potential class `L^2_t L^2_x` has
parabolic index

```text
2/2+3/2 = 5/2 > 2.
```

Thus ordinary energy control does not close the visit estimate.

## Weighted return obstruction

The exterior return estimate fails for a related structural reason. Consider
the local incompressible backward affine drift

```text
b_ret = diag(-kappa,-kappa,2kappa)x.
```

On the equatorial plane it transports a history inward from radius
`L_out=eta L` to `L` in deterministic time

```text
tau = log(eta)/kappa.
```

The corresponding forward strain is
`diag(kappa,kappa,-2kappa)`. A deformation vector in the returning radial
direction gains

```text
exp(kappa tau)=eta
```

per history, or `eta^2` for two histories. At the deterministic return time,
the transverse and axial noise variances are

```text
nu(1-eta^(-2))/kappa,
nu(eta^4-1)/(2kappa),
```

so they vanish as `kappa` grows. Weighted return therefore approaches the
deterministic gain rather than a contraction.

This drift is trace-free and has finite gradient energy density `6kappa^2` on
every bounded shell. It also has a global smooth finite-energy solenoidal
extension. Indeed,

```text
A=(-kappa*y*z,kappa*x*z,0)
```

satisfies `curl A=b_ret`; if a smooth compactly supported cutoff `chi` equals
one on the return shell, then `curl(chi A)` is compactly supported,
divergence-free, and agrees with `b_ret` there. The local and global norms
appearing in the Leray energy inequality therefore do not exclude this drift
configuration.

The example remains a drift-level stress test, not a stationary
Navier-Stokes solution: it does not prove that an actual Navier-Stokes
exterior maintains this behavior for the complete excursion. It proves only
that a bare finite-energy estimate, without using further PDE evolution or
geometry, cannot imply the required weighted-capacity contraction.

There is an important later refinement. The argument above concerns total
weighted return mass and omits the finite axial entry patch. In the spatial
`L2` density norm used by the finite-energy trace theorem, the outward axial
OU spreading for this same affine drift exactly cancels `exp(kappa t)`. The
remaining transverse killed-OU tail is summable. Thus this counterexample
still blocks a contraction proof from energy alone, but it does not block the
new finite-patch density route; see
`affine_exterior_axial_compensation.md`.

## Closure verdict

The localized collision mechanism survives the full three-dimensional audit,
but it is not closed by standard Leray bounds. The exact sufficient package is
now:

1. a finite three-dimensional core with spectral margin `m>0`;
2. local affine coherence strong enough to satisfy the critical `L^(3/2)`
   error budget;
3. a buffered exterior estimate that controls return probability and
   deformation together, not return probability alone;
4. summability over all candidate concentration cores.

The next viable research direction is to test whether Navier-Stokes geometry
itself supplies items 2 and 3 near a hypothetical first singularity. Vorticity
direction coherence, strain-eigenframe persistence, or local pressure/Hessian
constraints are possible sources. Reapplying the energy inequality without
new geometry will reproduce the supercritical gap above.

The symbolic gauge, sharp Sobolev constant, critical budgets, time spikes, and
return-strain obstruction are reproduced by
`scripts/three_dimensional_leray_gate_audit.py`.

The first attempt to derive the missing coherence directly from the strain
and vorticity equations is developed in `strain_eigenframe_geometry.md`.
