# Rotating full-affine visit pilot

## Question

The instantaneous full-affine reference removes orientation drift from the
physical error, but replaces the static elliptic visit by a nonautonomous
one. Does a rotating constant-spectrum strain immediately produce a worse
compact-cylinder transfer?

The sampled answer is no. Constant-rate rotations about the cylinder axis
decrease the uniform angular `L2` visit norm, and constant-rate tilts decrease
a pointwise Feynman-Kac payoff. This is numerical evidence for a static-worst
comparison principle, not a proof of one.

## Axis-preserving rotation

For normalized forward spectrum `(-1-t,t,1)`, the backward transverse affine
drift is

```text
B_0=diag(1+t,-t).                                     (1)
```

If the strain rotates about the cylinder axis at constant rate `omega`, the
rotating coordinate frame leaves the cylinder fixed and converts the drift
to

```text
B_omega=B_0-omega J.                                 (2)
```

Equation (2) is time independent but nonreversible. A direct nonsymmetric
finite-element solve on the compact `H/L=0.75`, `R/L=2` cylinder gives:

| `t` | static uniform norm | largest sampled norm | maximizing `omega` |
|---:|---:|---:|---:|
| -0.5 | 0.335274 | 0.335274 | invariant |
| 0.0 | 0.360489 | 0.360489 | 0 |
| 0.5 | 0.419637 | 0.419637 | 0 |
| 1.0 | 0.493982 | 0.493982 | 0 |

Rates `0, 0.25, 0.5, 1, 2, 4, 8, 16` were sampled. At every anisotropic
spectrum the norm decreases with rotation. At `t=-1/2` the transverse drift
is isotropic, so exact invariance is the required sanity check.

The new nonsymmetric discretization and the old reversible discretization
give static `t=1` weighted norms `0.54870` and `0.55676` on the working mesh.
Refinement reduces that gap toward a common value near `0.549`; the old
`0.5568` renewal calibration is conservative at its mesh.

## Why the old Gaussian weight cannot rotate

If the same solutions are measured in the static anisotropic Gaussian
boundary weights, the reported diagnostic norm grows with `omega` and
eventually exceeds one. That growth is a measure-conversion artifact: the
static reversible weight is not invariant for (2). It confirms that a
nonautonomous proof must carry the actual evolving hitting law or establish
a common admissible boundary norm.

## Tilting rotation

A rotation that mixes a transverse direction with the cylinder axis loses
axial separation. For the worst static spectrum `t=1`, use

```text
B_omega=diag(2,-1,-1)-omega J_tilt.                  (3)
```

The audit advances the exact linear SDE over each timestep, starts at
`(1,0,0)`, stops at `r=2` or `|z|=0.75`, and measures

```text
E[exp(tau); radial exit occurs first].                (4)
```

With `60,000` paths per row and `dt=0.00125`:

| rotation | `omega` | payoff | standard error |
|---|---:|---:|---:|
| static | 0 | 0.59091 | 0.00260 |
| cylinder axis | 1 | 0.58445 | 0.00261 |
| cylinder axis | 2 | 0.56938 | 0.00260 |
| cylinder axis | 4 | 0.52916 | 0.00262 |
| cylinder axis | 8 | 0.46526 | 0.00262 |
| tilt | 1 | 0.55623 | 0.00257 |
| tilt | 2 | 0.46356 | 0.00247 |
| tilt | 4 | 0.25261 | 0.00201 |
| tilt | 8 | 0.04284 | 0.00091 |

No sampled rotation increases (4). Timestep refinement moves all values
downward because discrete boundary detection misses some between-step cap
exits, but preserves the ordering with a wide margin. The rows are
statistical diagnostics, not enclosures.

## Piecewise-switching search

An exploratory search then sampled 200 deterministic eight-block histories:
half rotated only in the transverse plane and half used full sphere
orientations, with block durations `0.025, 0.05, 0.1, 0.2, 0.5`. Curated
`x/y`, `x/z`, and three-axis switches were included separately. Rapid
switching reduced the payoff strongly. Long blocks approached the static
row because many paths exited before the first switch.

The largest coarse random row was selected after looking at all 200 rows, so
its apparent excess was tested on an independent 500,000-path sample. The
holdout result was

```text
static       0.5903404 +/- 0.0008998,
candidate    0.5858682 +/- 0.0008942.                 (5)
```

Thus the only apparent switching excess was selection noise. This search is
still finite and does not establish that static strain is universally worst.

## Consequence

Constant-spectrum orientation motion is no longer an evident obstruction to
the compact visit once it is included in the baseline. The next theorem
should target a common dynamic boundary norm or comparison principle under
arbitrary measurable coefficient histories. The analytic task is now to
prove a uniform constant-payoff estimate for those histories.

The nonsymmetric finite-element sweep, static calibration, exact linear-SDE
steps, and tilt statistics are reproduced by
`scripts/rotating_affine_visit_pilot.py`.
