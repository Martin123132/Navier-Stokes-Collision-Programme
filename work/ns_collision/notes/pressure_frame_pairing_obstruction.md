# Pressure-frame pairing obstruction and boundary replacement

## Question

The pressure collision split isolated the signed high-frequency term
`-B_s^P,33`. A natural hope was that this adverse term might be paid
pointwise by the viscous eigenframe penalty, perhaps together with the local
self-strain reaction, at a spatial maximum of `lambda_3`.

That pointwise route is false. A smooth divergence-free trigonometric
polynomial gives a strict continuous maximum at which pressure defeats all
those terms and the maximal strain eigenvalue grows materially. A different
exact cancellation survives after contraction with the full strain tensor and
spatial integration. Localizing that cancellation moves pressure to a
boundary commutator.

## Reproducible periodic stress test

The audit constructs a deterministic field on the `2pi`-periodic
three-torus as follows:

```text
grid used for coefficients: 20^3
random seed: 81
retained wave numbers: 0<|k|<=3
velocity RMS: 10
viscosity: 1
pressure heat scale: 0.08
```

The random coefficients are projected exactly onto `k dot u_hat=0`; the
physical field is a finite trigonometric polynomial. Products have wave
number at most six, below the grid Nyquist frequency, so the pressure source
and direct Navier-Stokes time derivative are de-aliased.

Starting from a grid maximum, continuous BFGS refinement locates

```text
x approximately (3.5797413640, 2.7714330414, 5.1374362804).
```

At this point,

```text
strain eigenvalues
 approximately (-15.28169044, 0.97136564, 14.31032480),

|grad lambda_3| approximately 8.1e-14,

eigenvalues of Hess(lambda_3)
 approximately (-48.75992710, -33.09972142, -24.71248705).
```

It is therefore a strict continuous local maximum with a simple maximal
eigenvalue.

## Failure of the pointwise gate

At that maximum, the terms in

```text
(D_t-nu Delta)lambda_3
 =R_local-P0_33-F_frame
```

are

```text
R_local                         approximately  27.43377673,
-P0_s,33                        approximately  58.84643009,
-B_s^P,33                       approximately  54.18482031,
F_frame                         approximately  15.38870214.
```

Thus

```text
R_local-P0_33-F_frame
 approximately 125.07632500 > 0.
```

The remaining scalar diffusion at the maximum is

```text
nu Delta lambda_3 approximately -106.57213557,
```

so the complete material growth is still positive:

```text
D_t lambda_3 approximately 18.50418943.
```

An independent Fourier calculation of
`partial_t S+u dot grad S`, using the projected Navier-Stokes time derivative,
agrees to about `1.4e-12`.

This is smooth initial data, so it generates a local smooth Navier-Stokes
solution. It does not demonstrate blow-up. It disproves a universal
pointwise maximum principle and the proposed pointwise pairing of the
pressure collision defect with frame coherence.

Amplitude scaling explains the failure. Pressure and local quadratic terms
scale like `A^2`, whereas scalar diffusion and the viscous frame penalty scale
like `nu A`. For this field the material-growth threshold is velocity RMS
approximately `8.68`; the audited RMS is ten.

## Global pressure orthogonality

The failed pointwise statement has an exact integrated replacement. Since
`P=Hess p` is symmetric and `div u=0`,

```text
integral P:S dx
 =integral partial_i partial_j p partial_j u_i dx
 =0.
```

The cancellation survives the heat split separately:

```text
integral P0_s:S dx=0,
integral B_s^P:S dx=0.
```

The trace corrections in `P0` do not contribute because `tr S=0`. The audit
verifies all three contractions on the counterexample field.

This cancellation is different from the cubic Miller identity. It is a
pressure-gradient integration-by-parts identity and holds at every heat
scale.

## Localization moves pressure to the boundary

Let `phi` be a smooth scalar localization and write

```text
B_s^P=Hess(p-p_s)+(f-f_s)I/3.
```

The isotropic term again vanishes against `S`. Integration by parts gives

```text
integral phi B_s^P:S dx
 =-integral grad(p-p_s) dot [(grad u)^T grad phi] dx.
```

Therefore pressure disappears in any region where `phi` is constant and is
paid only where the localization changes. For a buffered strain core, that
is precisely the transition shell used in the exit/re-entry decomposition.

This suggests a replacement for the failed maximal-eigenvalue route:

1. use a localized matrix or strain-energy observable rather than the
   pointwise eigenvalue;
2. choose `phi` constant on the coherent core and changing only across the
   exit buffer;
3. combine viscous core dissipation with the exact pressure boundary
   commutator;
4. pay the shell term using the same capacity/exit estimate that controls
   repeated visits.

## New gate

The next estimate is no longer a pointwise pressure sign. It is the shell
commutator bound

```text
|integral grad(p-p_s) dot [(grad u)^T grad phi] dx|
```

relative to localized viscous strain dissipation and buffered escape. A raw
Holder estimate is likely supercritical. The pressure heat defect and the
geometry of `grad phi` must be retained, and the estimate should be tested on
the periodic counterexample before being promoted.

The field construction, continuous maximum refinement, pressure split,
direct Navier-Stokes derivative, and global orthogonality checks are
reproduced by `scripts/pressure_frame_pairing_audit.py`.

The proposed shell estimate has now been audited in
`pressure_shell_commutator.md`. Its exact boundary identity survives, but
fixed-scale absolute absorption fails at large local Reynolds number. The
remaining route is an adaptive moderate-Reynolds shell decomposition with a
separate mechanism for high-Reynolds visits.
