# Affine exterior axial compensation

## Purpose

The earlier Leray audit used the incompressible affine return drift

```text
b=(-kappa x,-kappa y,2 kappa z)                     (1)
```

to show that finite energy alone does not force weighted return contraction.
That stress test remains valid for total return mass, but it omitted the
finite axial entry patch from the newer spatial `L2` density route. This note
restores the axial geometry exactly.

## Exact axial cancellation

Starting from `z=0`, the axial coordinate in (1) is an outward OU process
with variance

```text
sigma_z^2(t)=[exp(4 kappa t)-1]/(2 kappa).           (2)
```

The returning radial deformation direction gains `exp(kappa t)`. The
full-line Gaussian density has

```text
||Gaussian_(sigma_z)||_2
 =1/[sqrt(2)pi^(1/4)sigma_z^(1/2)].                  (3)
```

Combining (2)-(3) gives the exact identity

```text
exp(kappa t)||Gaussian_(sigma_z)||_2
 =(2kappa)^(1/4)/[sqrt(2)pi^(1/4)]
  [1-exp(-4kappa t)]^(-1/4).                        (4)
```

Thus axial `L2` dilution cancels the complete affine deformation
exponentially. The remaining long-time behavior is the transverse first-hit
density. Restriction to `|z|<3/4` can only lower this full-line norm.

This does not contradict the old deterministic stress test. At its return
time the axial variance vanishes as `kappa` grows, so total weighted return
mass need not contract. Spatial `L2` density sees the axial spreading over
the whole excursion and asks a different, stronger-resolved question.

## Exact transverse modes

For the transverse inward OU generator

```text
L_perp=Delta-kappa r partial_r,       r>1,            (5)
```

put `y=kappa r^2/2`. The Laplace transform of angular hitting mode `n` is

```text
F_n(lambda;r)
 =(y/y_a)^(n/2)
  U[n/2+lambda/(2kappa),n+1,y]
  /U[n/2+lambda/(2kappa),n+1,y_a],                  (6)

y_a=kappa/2.
```

The principal radial decay rate is the first positive solution of

```text
U[-lambda_0/(2kappa),1,kappa/2]=0.                  (7)
```

The killed inward OU exterior is confining, so `lambda_0>0`; angular modes
add a nonnegative centrifugal form. Equations (4)-(7) imply an exponentially
summable weighted spatial `L2` tail for every `kappa>0`. At `kappa=0`, the
Brownian cylinder calculation supplies the summable
`t^(-5/4)/log(t)^2` endpoint.

High-precision Tricomi root pilots give:

| `kappa` | `lambda_0` |
|---:|---:|
| 0.125 | 0.0870177 |
| 0.25 | 0.214770 |
| 0.5 | 0.547233 |
| 1 | 1.44777 |

## Working-scale inversion pilot

At the normalized axisymmetric endpoint `kappa=1`, Gaver-Stehfest orders 12
and 14, angular modes `0..12`, and times `0.05..3` differ by at most about
one percent. The sampled weighted
spatial `L2` envelope peaks near

```text
t=0.15,       ||k_weighted(t)||_2 approximately 1.004. (8)
```

This is larger than the deliberately inflated Brownian peak `0.731`, but it
is finite and followed by the faster tail `exp(-1.44777t)`. The inverse
transform and spectral constants are pilots, not interval enclosures.

## Consequence and remaining gate

The old inward-affine example no longer rules out the finite-patch `L2`
density strategy. In the exact axisymmetric model, incompressibility couples
radial attraction to precisely the axial expansion needed to cancel the
deformation weight in `L2`.

This is not yet the exterior theorem. The next obligations are:

1. certify a global-in-time envelope and angular tail at `kappa=1`;
2. cover every admissible affine spectrum, orientation, and measurable time
   history, including noncommuting frames;
3. perturb the affine kernel by the actual critical Navier-Stokes exterior
   drift and deformation errors;
4. integrate the resulting branch gain into the corrected cubic renewal.

Summability alone does not prove return contraction or any Navier-Stokes
regularity statement.

The exact transforms, axial identity, spectral roots, and inversion pilot
are reproduced by
`scripts/affine_exterior_axial_compensation_audit.py`.
