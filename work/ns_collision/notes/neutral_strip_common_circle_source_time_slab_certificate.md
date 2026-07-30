# Common-circle source time-slab certificate

## Target

For the retained source-resolved discrepancy, write

```text
d_z(t)=D_K exp(-L_K t) S_mod,z
       -C_K exp(-Lambda_K t) S_ref,z
```

as an element of the common boundary Hilbert space constructed by the exact
polygon trace-mass, radial-push, and dual-cell cross Grams. The later-window
quantity to be bounded is

```text
sum_(n>=1) sup_(t in [n ell,(n+1)ell]) a_x(t)||d_z(t)||,
ell=3/8.                                                   (1)
```

There are fifteen finite windows with indices `1,...,15`, covering
`[3/8,6]`. The geometric tail begins with index `16` at `t=6`. The earlier
sixteen-point diagnostic sampled indices `1,...,16`; its last value is the
first tail sample, not an additional finite window before a post-`6` tail.

## Slab interpolation

For a Hilbert-space-valued twice differentiable function `d` on `[a,b]`,
linear interpolation and the Peano remainder give

```text
sup_[a,b] ||d(t)||
 <=max(||d(a)||,||d(b)||)+(b-a)^2 sup_[a,b]||d''(t)||/8.   (2)
```

This avoids assuming that the discrepancy norm is monotone. For a diagonal
spectral chain `B exp(-Lambda t)S`, split the second derivative at half time:

```text
||B Lambda^2 exp(-Lambda t)S_z||
 <=||B Lambda exp(-Lambda a/2)||
   ||Lambda exp(-Lambda a/2)S_z||,     t>=a.               (3)
```

The reference and modified bounds from (3) are added. Each `3/8` window is
split into `0.025` slabs, so (2) needs only endpoint actions and one analytic
second-derivative bound per full window.

The axial factor is enclosed without a monotonicity assumption on its exact
formula. The elementary inequality `erf(x)<=2x/sqrt(pi)` gives, for every
`t>=a>0`,

```text
a_x(t)<=sqrt(H/(pi(1-exp(-2a)))).                         (4)
```

## Projected action and tail

Let `M_K=LL^T` and

```text
H_K=L^(-1) A_K L^(-T).
```

The implementation diagonalizes the symmetric `H_K`. If `H_0` is the
reconstructed diagonalization and `delta=||H_K-H_0||`, contractive Duhamel
gives

```text
||exp(-H_K t)-exp(-H_0 t)||<=t delta.                     (5)
```

This gap-free defect is added to every finite slab. It is a numerical-linear-
algebra guard for the projected action and is unrelated to the much larger
off-block leakage from the retained space into its complement.

At `T=6`, the two chains are bounded separately. If `lambda_1` and
`mu_1-delta` are their decay floors and `A_ref,A_mod` are the terminal
boundary amplitudes, then (4) and contraction give

```text
tail <=a_x,upper(T) [
 A_ref/(1-exp(-lambda_1 ell))
 +A_mod/(1-exp(-(mu_1-delta)ell)) ].                      (6)
```

Equation (6) starts at window `16`, so it composes with exactly the fifteen
finite rows without overlap.

## Implementation status

The executable is
`scripts/neutral_strip_common_circle_source_time_slab_certificate.py`.
It uses fingerprinted eigenbasis caches, runs below normal priority, writes
results atomically, independently cross-checks the common-circle endpoint
norm by the original Gram route, and includes an unrelated vector-valued
interpolation regression.

The cache-backed production rows now pass every internal check:

```text
h      substep   finite raw sum   tail raw sum    low factor    combined     headroom
0.12   0.025     0.0294369863     2.19721e-5     0.0171433856  0.9791369465 0.0208630535
0.06   0.025     0.0164379264     2.18245e-5     0.0095786094  0.9715721703 0.0284278297
0.06   0.0125    0.0138814947     2.18245e-5     0.0080909162  0.9700844771 0.0299155229. (7)
```

Halving the substep reduced the maximum interpolation charge by exactly a
factor of four, from `0.0045288879` to `0.0011322220`, as (2) predicts. The
three independent endpoint Gram comparisons at `h=0.06` agree to at worst
`1.97e-15`. The post-`6` contribution is already negligible relative to the
finite-window sum.

Equation (7) proves the time-interpolation and terminal-tail inequalities for
the frozen finite block. The companion endpoint-roundoff audit now derives the
floating guard operation by operation for all 451 `h=0.06` endpoints. Its
maximum directed roundoff upper is `3.0107546665e-11`, leaving at least
`2.6685982793e-11` under the existing guard; an independent 80-digit worst-case
reconstruction is also covered. This statement is relative to the stored
binary64 inputs. A companion Gaussian-weighted assembly audit now encloses
the exact reference mass, stiffness, boundary stiffness coupling, and
boundary mass coupling around the fingerprint-matched q12 matrices. The
remaining Riesz/Gram/projected algebra and generalized eigenpairs are still
input-level floating data, so the result is not yet a full discrete interval
certificate. A companion residual audit proves
stored-mass coercivity and encloses all cached generalized-eigenpair
residuals. A directed sparse-inertia companion now indexes all 241 intervals,
and min-max transfers them to the exact forms on the stored polygon.
Eigenvector/projector enclosure and propagation of those residuals to
endpoint output remain open. Off-block leakage, continuum Ritz transfer, and polygon-domain
perturbation are also still open. See
`neutral_strip_common_circle_endpoint_roundoff_audit.md`,
`neutral_strip_common_circle_eigensystem_residual_audit.md`, and
`neutral_strip_gaussian_weighted_assembly_interval_audit.md`, and
`neutral_strip_sparse_inertia_indexed_spectrum.md`.
