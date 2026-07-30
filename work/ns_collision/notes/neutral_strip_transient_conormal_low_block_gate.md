# Transient conormal low-block gate

## Purpose

The first-window bridge certificate and the continuum high-mode estimate
leave a finite low spectral block. Earlier diagnostics compared the positive
Markov-chain boundary coupling with the stiffness cross-block of a conforming
P1 discretization. That is not the complete boundary flux for a
consistent-mass semidiscrete heat equation.

This note derives the missing transient mass term, recomputes the retained
boundary comparison, proves the scalar flux-measure conversion from the
inscribed polygon to the true circle, and tests whether a finer mesh leaves
enough numerical margin for an eventual interval proof.

## Correct transient conormal moment

Partition the consistent P1 mass and stiffness matrices into interior and
Dirichlet boundary vertices. The interior coefficients satisfy

```text
M_II u_dot+A_II u=0.                                (1)
```

Testing the weak equation with a boundary lift gives the positive absorption
conormal moment

```text
g=-A_BI u-M_BI u_dot
  =B_stiff^T u-B_mass^T u_dot,                      (2)
B_stiff=-A_IB,  B_mass=M_IB.                        (3)
```

For a generalized eigenmode

```text
A_II v_k=lambda_k M_II v_k,
u_k(t)=exp(-lambda_k t)v_k,                          (4)
```

equation (2) becomes

```text
g_k=(B_stiff^T+lambda_k B_mass^T)v_k.               (5)
```

Without choosing eigenvectors, the corrected finite-dimensional output is

```text
C_h^T=B_stiff^T+B_mass^T M_II^(-1)A_II.            (5a)
```

This is the operator that an interval calculation must enclose. Working only
mode by mode is convenient numerically but is not part of the definition.

The previous comparison used only `B_stiff`. That is exact for the
centroid-lumped Markov chain because its boundary mass cross-block vanishes,
but not for the consistent reference discretization.

On the first 20 modes at `h=0.12`, the correction is already `1.19%`; the
modified-to-corrected discrepancy is `1.15%`, compared with `0.28%` in the
stiffness-only comparison. At mode cutoff 320 the correction grows to
`13.85%`.

## Cutoff tradeoff

Reducing the retained mode count reduces (5), but raises the analytic Li-Yau
high-mode payment. At `h=0.09` the principal rows are

```text
K     corrected boundary mismatch    later high interval factor
160          0.0325639                       0.1202303
200          0.0396636                       0.0346717
240          0.0429569                       0.0096094
280          0.0506965                       0.0025894
320          0.0614455                       0.0006834.              (6)
```

The useful balance lies near `K=240`. Production refinement at that cutoff
gives

```text
h        states    corrected mismatch    mass correction
0.12      3738        0.1006086883         0.1009904538
0.09      6625        0.0429568592         0.0432679925
0.075     9599        0.0391540011         0.0392734458
0.06     15211        0.0220760461         0.0222150477.             (7)
```

These are high-order floating diagnostics, not interval enclosures.

## Projected dynamics and off-block leakage

The static boundary ratio in (7) is not a complete low-block comparison. Let
`V` contain the first `K` reference eigenvectors, normalized in the reference
mass, and set

```text
M_K=V^T M_tilde V,       A_K=V^T A_tilde V,
L_K=M_K^(-1)A_K,         D_K=B_tilde^T V.            (7a)
```

For the same initial coefficient vector, the two projected output actions
are

```text
T_ref(t)=C_K exp(-Lambda t),
T_proj(t)=D_K exp(-L_K t).                           (7b)
```

At `h=0.06`, `K=240`, the floating diagnostics are

```text
modified/reference mass on V             1.001023 .. 1.037911
modified/reference stiffness on V        0.997780 .. 1.000155
||L_K-Lambda||/||Lambda||                 0.0370832

t        ||T_proj-T_ref||/||C_K||    relative to ||T_ref(t)||
0.375          0.0000655233                 0.00197954
0.750          0.0000268724                 0.00224858
1.500          0.0000080230                 0.00406004
3.000          0.0000004840                 0.00861476.             (7c)
```

Thus the actual projected discrepancy after the first window is much smaller
than the time-zero `0.022076` screen charge. This is the expected parabolic
smoothing, but the calculation still omits escape from the retained space.

There is an exact way to isolate that obligation. Symmetrize the full
modified operator as

```text
H=M_tilde^(-1/2) A_tilde M_tilde^(-1/2),
Q=M_tilde^(1/2)V M_K^(-1/2),
E=(I-QQ^T)HQ.                                       (7d)
```

Here `Q^TQ=I`, and `||E||` is the coupling from the retained space to its
orthogonal complement. Duhamel and contractivity prove

```text
||exp(-tH)Q-Q exp(-t Q^T H Q)|| <= t||E||.          (7e)
```

The normalized residual in (7d) is computed without forming `H`:

```text
E=M_tilde^(-1/2)
  (A_tilde V-M_tilde V L_K) M_K^(-1/2).             (7f)
```

At `h=0.06`, `||E||=6.3437031`. The gap-free bound (7e) is therefore
`2.3788887` at `t=3/8`, even though the direct projected discrepancy in (7c)
is tiny. This rules out plain contractive Duhamel as the leakage estimate.
A successful enclosure must retain complement spectral decay, ideally in a
two-block comparison, and apply the existing half-time flux smoothing after
the leakage has entered the high block. The source trace into `V` must also
be enclosed; neither (7b) nor (7e) does that.

As an independent orientation check, the executable parent audit constructs
an unrelated deterministic `11`-state SPD system with a four-dimensional
trial space. It forms `E` both from `(I-QQ^T)HQ` and from the residual in
(7f), obtaining a spectral-norm difference below `9e-15`. It also verifies
(7e) at four times and checks the transient conormal sign on a one-dimensional
P1 element, where both formulas give the positive load `3`. This regression
checks the algebra, not the production floating constants or continuum
certification.

## Polygon flux measure

Consider one edge of a regular inscribed `N`-gon. In local polar angle
`|phi|<=alpha=pi/N`, its radius and arclength Jacobian are

```text
r_p(phi)=cos(alpha)/cos(phi),
ds/dphi=cos(alpha)sec(phi)^2.                        (8)
```

Push a scalar conormal measure `j_p ds` radially to the unit circle and call
its density `j_c`. Conservation of boundary mass gives

```text
j_c(phi)=j_p(phi) ds/dphi.                          (9)
```

Therefore

```text
||j_c||_L2(circle)^2
 <=max(ds/dphi)||j_p||_L2(polygon)^2
 <=sec(alpha)||j_p||_2^2,                           (10)
```

so the exact output-measure factor is at most

```text
C_polygon=sqrt(sec(pi/N)).                           (11)
```

At `h=0.06`, `N=112` and `C_polygon=1.000196744856`. The apparently larger
normal-angle mismatch is not an additional factor for a scalar conormal
*measure*: it is absorbed by the pushforward Jacobian. This does not compare
the polygonal and circular domains or their semigroups.

## Common-circle finite boundary map

The P1 calculation returns a nodal conormal load vector `g`, meaning a linear
functional on the boundary trace space, rather than an `L2` density. On each
regular polygon edge of length `ell=2sin(alpha)`, the exact trace mass is

```text
M_edge=ell/6 [[2,1],[1,2]],
M_Gamma j_h=g.                                       (11a)
```

The pushed density is `j_c=j_h ds/dphi`. Put `q=tan(alpha)` and
`c=cos(alpha)`. Direct polynomial integration in `q` gives the local Gram
matrix of its true-circle `L2` norm:

```text
P_edge=[[p_d,p_o],[p_o,p_d]],
p_d=c^2(2q/3+4q^3/15),
p_o=c^2(q/3+q^3/15).                                (11b)
```

The modified chain assigns each killed load to the true-circle dual arc of
length `a=2alpha`. Each dual cell contains half of each adjacent polygon
edge. Consequently its cross inner product with the pushed P1 density has
the exact row stencil

```text
H_i=(ell/a)[(1/8)e_(i-1)+(3/4)e_i+(1/8)e_(i+1)].    (11c)
```

For reference and modified load maps `G` and `D`, let `X=M_Gamma^(-1)G`.
Both outputs now live in the same `L2(S1)` space, and the input-space Gram
matrix of their difference is exactly

```text
E=D^T(aI)^(-1)D-D^T H X-X^T H^T D+X^T P_Gamma X.   (11d)
```

The implementation forms (11d) without sampling the boundary. An independent
32-point split-edge Gauss check gives pushed-Gram error `6.2e-15` and zero
cross-Gram error. At `N=112`, the P1-subspace push factor is
`1.000039383027`, smaller than the unrestricted factor `1.000196744856` in
(11).

For an entry vertex `z`, the reference point-source coefficients are
`V^T e_z`. The modified Galerkin projection uses the different mass metric:

```text
S_ref=V^T e_z,              S_mod=M_K^(-1)V^T e_z.  (11e)
```

Thus the two source-resolved load maps at time `t` are

```text
G_z(t)=C_K exp(-Lambda t)S_ref,
D_z(t)=D_K exp(-L_K t)S_mod.                        (11f)
```

Equations (11a)-(11f) close the finite geometric reconstruction algebra.
Their production evaluation is still floating, and neither equation compares
the polygonal-domain solution with the true-circle-domain solution.

## Naive radial additive-screen obstruction

For a concrete polygon-to-circle comparison, map each polygonal radial
segment by

```text
R=1+(r-r_p(phi))/(2-r_p(phi))                       (11g)
```

and use the identity outside `r=2`. At an edge endpoint, in
radial-tangential orthonormal coordinates, its derivative is the shear

```text
J=[[1,-q],[0,1]],        q=tan(pi/N).                (11h)
```

The pulled-back Dirichlet-energy factor at that endpoint is

```text
kappa_N=(sqrt(q^2+4)+q)/(sqrt(q^2+4)-q).            (11i)
```

For `N=112`, `q=0.0280572933` and `kappa_N=1.0284536599`. Charging
the full endpoint coefficient distortion one-for-one to the already crude
screen raises (13) to `1.0127200117`. Thus that naive additive accounting
fails. This does not prove that every perturbation theorem using the radial
map must fail: time localization or smoothing could convert coefficient
distortion into a smaller response cost. It does show why the cleaner next
route is a low-mode or shape perturbation estimate that exploits the
`O(N^-2)` sagitta rather than paying the endpoint shear directly.

## Feasibility screen

The first-window certificate leaves headroom

```text
1-0.952384193962=0.047615806038.                    (12)
```

For `K=240`, subtracting the analytic later high-mode payment leaves
`0.038006439076`. The previous raw-load calculation gave

```text
0.952384193962   first window
+0.009609366962  later high modes
+0.022076046073  raw conormal-load mismatch
+0.000196744856  separate unrestricted push factor
=0.984266351853.                                    (13)
```

Equation (13) is now demoted. A nodal-load norm plus a separate general push
factor is not the physical common-circle `L2` discrepancy. At `h=0.06`, the
correct time-zero operator discrepancy from (11d) is `0.0892649461`; charging
that one-for-one would give `1.0512585070`, above one.

The response does not use an arbitrary time-zero low-block vector. Applying
the actual source maps (11e), evolving through (11f), and summing the maximum
entry-column errors at the sixteen later-window starts gives

```text
h       time-zero common-circle   t=3/8 max entry   sampled interval factor
0.12          0.20560582             0.02822352          0.01423625
0.09          0.14362449             0.02179330          0.01098179
0.075         0.11675710             0.01784951          0.00898319
0.06          0.08926495             0.01414095          0.00712490. (14)
```

The source-aware quantity converges cleanly. At `h=0.06` its floating screen
is

```text
0.952384193962  certified first window
+0.009609366962 certified later high modes
+0.007124897003 sampled later low-block transfer error
=0.969118457927.                                    (15)
```

This leaves `0.030881542073` sampled headroom. It is materially better than
the valid time-zero operator screen because it includes parabolic damping and
the actual entry map. It is not yet a certificate: values in (14) are window
starts rather than interval suprema, omit the post-`6` discrepancy tail, and
compare only the projected modified dynamics.

The companion time-slab gate now replaces those point samples, for the frozen
finite block, by the non-monotonicity-safe endpoint interpolation bound and a
nonoverlapping geometric tail. The correct partition has finite window indices
`1,...,15` on `[3/8,6]` and tail index `16` starting at `6`. At `h=0.06` and
substep `0.0125`, the guarded low-block factor is `0.008090916168`, giving

```text
0.952384193962  certified first window
+0.009609366962 certified later high modes
+0.008090916168 guarded frozen low block including tail
=0.970084477092.                                    (16)
```

The interpolation remainder falls by exactly four when the substep is halved,
and independent endpoint Gram evaluations agree within `1.97e-15`. Equation
(16) is stronger than the sampled time treatment, but is still conditional on
the floating frozen coefficient data.

The endpoint arithmetic inside that condition is now closed. A directed
binary64 error audit covers all 451 substep endpoints and bounds the worst norm
error by `3.0107546665e-11`, leaving at least `2.6685982793e-11` under the
existing guard. An independent 80-digit reconstruction of the worst endpoint
and source column differs by only `2.21264e-14`. The audit treats the stored
matrices, modes, eigenvalues, and sources as exact binary inputs; it does not
enclose their assembly or generalized-eigenpair errors.

The next residual audit closes part of the latter statement. Directed local
Sylvester tests and global duplicate-summation bounds prove the stored mass
coercivity `M>=0.14999999999999333 D`. This converts the worst cached
generalized-eigenpair residual to an inverse-mass proximity radius
`7.54147e-11`. All 241 resulting intervals are disjoint, with
retained/omitted separation `0.601533418739`. A subsequent sparse-inertia
audit now certifies counts 240 and 241 at two separating shifts, reproduces
all 30,422 pivot signs at precision 260, and indexes all 241 stored-pencil
intervals.

The Gaussian-weighted assembly audit now closes the reference-form part of
the input condition. Directed geometry, analytic barycentric moments, and
Hermite remainder bounds enclose the exact mass, stiffness, boundary
stiffness coupling, and boundary mass coupling around matrices with the same
eigensystem fingerprint. The mass-form relative error is
`5.49257e-13`; the stiffness error is `5.43119e-9` in stored-mass units.
Min-max transfer then indexes all 241 exact-polygon finite-element
eigenvalues and gives complement lower bound `107.01775717228844`.

## Remaining certification obligations

Four gaps prevent promotion of (16):

1. Endpoint arithmetic, stored-matrix eigensystem residuals, indexed stored
   eigenvalues, weighted reference finite-element forms, and indexed
   exact-polygon eigenvalues are directed enclosed. Finite common-circle
   Riesz/Gram and projected algebra, eigenvector/projector enclosure, and
   endpoint propagation of assembly and eigensystem errors remain open. The
   time-slab and tail inequalities are analytic for the frozen binary data.
2. The projected action (7b) does not bound the off-block coupling (7d). The
   gap-free estimate (7e) fails numerically, so a spectrally damped leakage
   enclosure is required.
3. A conforming P1 eigenspace is not the continuum spectral projector. An
   a posteriori Ritz-projector error theorem is required; conforming
   Rayleigh upper bounds alone are insufficient.
4. Equations (11a)-(11f) transfer the reconstructed polygon boundary output
   to the circle, but do not compare diffusion on the two domains. A
   domain-perturbation theorem remains necessary.

Consequently the retained continuum low block, full polygon-to-circle flux
map, later low-mode composition, and continuum return response remain
uncertified. The next finite calculation should directed-interval enclose the
remaining Riesz/Gram/projected operations and propagate the now-bounded
assembly error plus the already-enclosed indexed spectral residuals through
(11f). The production eigenbases,
assembly, endpoint, residual, and guarded time-slab results are stored in
atomic fingerprinted caches and JSON outputs. In parallel, the analytic step
is a two-block spectral leakage theorem replacing (7e), followed by a
low-mode shape derivative estimate for the polygon-to-circle domain change.

The audit is reproduced by
`scripts/neutral_strip_transient_conormal_low_block_gate.py`.
