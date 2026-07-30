# Navier-Stokes Collision Programme

This directory tests whether the non-collision mechanism encountered in the
de Bruijn-Newman zero flow has a useful analogue for three-dimensional
incompressible Navier-Stokes.

Status: exploratory research. Nothing here proves global regularity or finite
time blow-up.

Research direction update (2026-07-30, restart-time third-jet closure):
the exact depth-three tree expansion has total absolute coefficient mass
`1412`. Its dangerous assignment ledger contains 579 protected pressure
rows, 30 expanded bounded-output pressure exceptions, and 81 protected
velocity-Fisher rows. After resonance elimination, exactly four protected
four-high assignments have one bounded internal Euler output, whereas all
20 six-high pressure assignments have only nonempty, strictly nested free
complement shells. Boundary-safe tensor packet differences and a
finite/dyadic Euler-output estimate now give the needed one carrier gain
for every protected four-high row and four raw gains for the all-high row.
This proves the constructed restart family satisfies the explicit bound
`|g_N'''(0)|<=C0 max(nu,nu^-1)^13 N^11` for odd `N>=3`; the deliberately
coarse integer `C0` has 422 digits. The proof does not add gains across
possibly correlated four-high shells. The live gate is still the uniform
version on `0<=s<=T/N^2`; the restart-time theorem alone gives no
turnaround or Clay-prize conclusion.

Research direction update (2026-07-30, complete second-jet asymptotic):
the exact 20-channel second variation contains 69 atomic subterms, now
partitioned without overlap into 21 pure Euler/transport, 31 one-heat, and
17 two-heat rows. The one-heat block is the exact cross derivative
`X(Yg)+Y(Xg)`, while the two-heat block is `D^2g[Y,Y]+Dg[DY Y]`. Incidence
and raw carrier counts put every heat row below `N^9` except the one-heat
pressure HHHH branch. Integrating its weight derivatives back to the base
`Phi` vertex leaves two compatible differences against an order-four
kernel; fixed-output pressure and an internal finite/dyadic split give
`O(N^8)` after optimization. All one- and two-heat rows are therefore
`O(N^8)=o(N^9)`. Combined with the preceding pure `E,A` Fisher exclusions,
the complete restart-time second jet now has the certified strict negative
`N^9` inviscid-pressure limit. The live gate has moved to a uniform Taylor
remainder on `0<=s<=T/N^2`; no parabolic-window turnaround or Clay-prize
claim follows yet.

Research direction update (2026-07-30, Euler-transport Fisher exclusion):
five apparently separate viscosity-bearing second-jet rows combine exactly
as `-nu d^2/ds^2 mean(lambda |grad u|^2)` along Euler velocity and
transported weight. The material identity removes all weight derivatives,
so the parity-gauged `Phi` vertex retains at least two compatible
differences against every degree-four Fourier monomial. Boundary-safe
`l1` estimates use
`||Delta h_N||_1=O(N)` and `||Delta^2 h_N||_1=O(1)`; no pointwise `C^2`
zero-extension claim is made. Holding the outer Riesz output fixed and
splitting the sole internal degree-one Euler singularity into finite and
dyadic shells gives an `O(N^7)` fixed-weight HHHH bound. Both HHHH and HHLL
are therefore `O(N^8)=o(N^9)` after the static optimizer. The companion
transported weight-self rows combine as
`-nu d^2/ds^2 mean(lambda |grad lambda|^2)` and are only
`O(N^6)=o(N^9)`. Every pure `E,A` viscosity-bearing Fisher row is now
excluded at leading order. The complete second-jet limit remains open for
the mixed `V,D` heat blocks.

Research direction update (2026-07-29, two-shear full `c1` port): the
complete fourteen-profile amplitude-one tail theorem now ports to the
modified witness. Its multiplier retains the sharp bounds
`|hhat_N|<=1/(2N)` and
`|Delta_j[parity*hhat_N]|<4/N^2`. Every stored tail profile is linear in
the low field, so doubling the low Fourier `l1` mass doubles the old tail
constant and gives
`|c1_*,N-D_*,N|<=70,657,920 N^6` for odd `N>=5`. Compact
`L1 cap L2` convergence of the modified packet and the exact 58-output
matrix give `D_*,N/N^7->L_*`; hence
`c1_*,N/N^7->L_*<0`. The amplitude-one continuum and tail obstruction is
closed on this branch. The static optimizer, complete finite jets, uniform
Taylor remainder, and parabolic window remain open. All `474` standalone
tests pass.

Research direction update (2026-07-29, two-shear square gate): an additive
modified annular witness now bypasses the unresolved numerical sign of the
original one-shear continuum constant. The new high profile
`b=S*(x^2+y^2)/(x*r^3)*(-z,0,x)` is divergence free and has `b_y=0`.
Combining the original `yz` low shear with a sign-flipped `xy` shear gives
the exact fixed-output matrix
`Q_*=(sqrt(2)/40)diag(1,-2,1)`. The Euler component trace then reduces the
leading four-high continuum coefficient to the strict negative square
`L_*=-(3sqrt(2)/20)||v_y||_2^2<0`; strictness follows analytically from the
small-output covariance expansion. The same two shears reduce the static
HHL load to `-(sqrt(2)/20)||b||_L2(D)^2<0`. Exact rational stencil
enumeration and `N=8,16,32` FFT replays pass. The exact odd finite packet is
also divergence free and has negative static HHL rows through `N=25`.
This certifies the two continuum signs for the modified witness, not the old
`L_EE`; at that checkpoint the quantitative finite remainder, optimizer,
tail ledger, and parabolic-window estimates still needed to be ported. All
`470` standalone tests passed.

Research direction update (2026-07-29, continuum sign gate): the remaining
`L_EE` sign problem now has a direct exact-box tensor-trapezoid rule, rather
than only a shifted-packet inverse-`N` fit. Its `N=8,16,32,64` rows converge
at second order to the stable candidate near `-2.99386e-7`, with the exact
`h^11` normalization and corrected `G_code=-g` sign independently checked
by the Euler energy trace. The two continuum pieces occupy disjoint output
bands. An explicit Euler-Maclaurin face measure
`mu_2=(1/12) sum orientation*partial_normal(a)` reproduces the complete
leading `h^2` coefficient; subtracting it gives fourth-order convergence.
The residual loss of smoothness is now localized to the internal expansion
`v(rho)=P_rho M rho+O(|rho|^3)`, where
`M=integral a tensor a` is diagonal by parity. The elementary bound
`||a||_2^2<=13/288` gives an explicit `7.800e-8` budget for the cube
`|rho_j|<=1/20`. The open task is a directed enclosure of the regular
complement, including FFT roundoff; the sign is still not certified.
At that checkpoint, all `464` standalone tests passed.

Research direction update (2026-07-29): the complete four-high
amplitude-one tail has now been closed analytically. Expanding the seven
nonleading forms and the three omitted dominant permutations gives fourteen
atomic rows with total absolute coefficient mass `94`. Every tail
permutation has a high leaf on the test side, so one exact parity-gauged
vertex difference can be absorbed there while the degree-zero outer pressure
projector is held fixed. The zero-extended sine packet is used only through
the boundary-safe estimate `|Delta h_N|<=4/N^2`; no `C^2` or higher
zero-extension claim is made. Treating each internal projected Euler
operator as its full globally Lipschitz degree-one symbol gives the explicit
bound
`|c_1,N-D_N|<=35,328,960 N^6` for odd `N>=5`. Combined with the prior
fixed-output remainder, this proves
`c_1,N/N^7 -> L_EE` and the quantitative bound
`|c_1,N/N^7-L_EE|<=35,578,960/N` for odd `N>=128`. All `457` corpus tests
pass. The numerical candidate `L_EE about -2.99386e-7` is still not a sign
theorem; the live gate is a deterministic joint interval enclosure for its
two cancelling continuum integrals.

Research direction update (2026-07-28): the exact compatible twelve-edge
penalty does not repair the annular HHL obstruction once every term in the
instantaneous objective is retained. For the explicit field
`u_N=h_N-a_N U` and weight `lambda_N=t_N Phi_+++`, the low-wave Fisher cost
is exactly `a_N^2/2`, the coefficient penalty is exactly
`(nu/16)(75/256)t_N^3`, and all non-HHL flux terms are stencil-invisible or
have zero load. Joint optimization gives `a_N,t_N=Theta(N)` and a positive
objective of order `N^3`; the finite replay is positive at `N=25`. Even the
bounded choice `t=1` becomes positive, directly at audited size `N=137`.
Fixed rays carrying positive `---` mass are suppressed by its `N^3` Fisher
cost, but rays on the face `z_---=0` with nonzero limiting load escape.
Thus the static arbitrary-coefficient coercivity route is closed. The live
question is whether actual backward-adjoint dynamics or a state-coupled
coefficient law forces a critical endpoint tax that this static cone omits.

The first dynamic check answers that endpoint question for the
static-optimal witness. At `rho=0`, the exact restart formula is stronger
than the stored deficit-dropped inequality:
`||u(T)||_3^3=||u(s)||_3^3+sup_lambda[J_0-Delta_s]`, where
`Delta_s=integral(|u| - lambda_s)^2(|u|+lambda_s/2)`. The true generator
also uses pressure only because strain cancels the kinetic HHL term.
Pressure-only replay leaves the static escape intact, but backward `L^3`
contraction and exact partition moments give `Delta_s=Omega(N^3)` for its
optimal `t_N Phi_+++` weight. On a `T/N^2` restart window, survival
therefore requires an order-`N^5` average generator, an `N^2` amplification
over its initial value. The static escape is not by itself a dynamic
counterexample.

The exact first time jet has now been separated into Euler, velocity
viscosity, weight advection, and weight antidiffusion. Dealiased Fourier
replay gives a negative complete derivative for
`N=25,29,33,37,41`. More decisively, an independent heat-weighted HHL
identity proves that the viscous-pressure component has the strictly
negative limit
`D_u g_pressure[nu Delta u_N]/N^5 -> -1.0442344590e-7/nu`.
At `N=41` the sum of all other components is only `1.80%` of that pressure
term. The compatible tensor stencils, odd-high support gap, exact
mixed-difference Fisher identity, and low-pressure-output shell split now
prove that the full remainder is `O(N^4)=o(N^5)`. Hence the complete first
jet has the same strictly negative `N^5` limit and is eventually negative.
This closes initial `N^5` amplification for the static-optimal witness, but
does not control a later turnaround. The live gate is now a second-time-jet
or Taylor-remainder bound on the full `T/N^2` window.

Research direction update (2026-07-26): the signed projected route now has
an exact backward restart dual, a scale law for its pressure edges, and a
finite-window correlation no-go theorem. For every deterministic
nonnegative terminal weight and canonical `0<=rho<=1` replica pair, the
complete generator difference is exactly
`(3/2) integral lambda_T(C_rho-C_0)`, the nonnegative terminal Wiener-chaos
variance tax. Thus `rho=0` is globally optimal in this backward-dual class;
the formal short-time crossover near `0.0756123` cannot become a net
positive-correlation advantage. A stochastic finite-window solver is no
longer the next sign test, although correlated replicas may still expose
useful structure outside this scoped no-go.

The remaining signed projected route also has an exact high-pressure-tail
decomposition. It proves
`||Q_m p||_2 <= 2||u||_infinity||grad u||_2/m`, so the tail is absorbed at
`m` comparable to amplitude over viscosity when the terminal weight has a
comparable positive floor. An exact co-scaled Taylor-Green family confirms
that this intrinsic frequency is necessary. The floor cannot be assumed:
the explicit zero-face family
`epsilon+sin(x/2)^2` has weighted singular-integral constants diverging like
`epsilon^(-1/4)`. Thus uniform weighted Calderon-Zygmund localization cannot
preserve the full terminal supremum. The live route is now a signed dyadic
pressure-flux Carleson estimate that sums antisymmetric neighboring
transfers before charging coefficient mismatch.

The balanced single-band part of that edge is now closed without a weight
floor. Combining the exact eight-shift pressure identity, the annular
residue-chain toggle theorem, and high-pass Poincare gives
`|P_v|<=2sqrt(2)(1+C^2)^3 m||u||_infinity K^(-2)E_v` for the complete
pressure generated and transported by one band
`K<=|k|<=Lambda K`. Nonnegative compatible vertex coefficients sum this
bound directly, while the cubic terminal-weight Fisher term remains
unspent. Taylor-Green and seed-81 sparse replays pass, including exact
co-scaling. The constant is conservative and the theorem does not compare
component Fisher costs with the physical multiband weighted Fisher form.
That finite-overlap multiband graph, together with cross-shell HHL pressure,
is the next obstruction.

The naive multiband Fisher recombination step is now ruled out exactly.
For the compatible zero-face weight `sin(x_1/2)^2`, a smooth divergence-free
pressure-free shear split into `J` dyadic annuli has physical weighted
Fisher energy `1/4` but component sum `J/4`. Every omitted neighboring-band
interface contributes exactly `-1/4`. The ratio remains `J` under partition
and amplitude co-scaling, and strictly positive floors approaching zero do
not provide a uniform constant. Thus independent annular estimates cannot
be charged and summed after absolute values. The next viable gate must test
whether the actual HHL pressure vanishes on, or is controlled transverse to,
the near-null residue-chain directions of the complete signed Fisher graph.

That compatibility gate is now positive for one canonical affine chain.
For frequencies `k_n=(n,1,0)`, both divergence-free polarizations, the low
wave `(0,-1,0)`, and the compatible weight
`phi_-(x_1)phi_+(x_2)`, the complete kinetic-plus-pressure HHL load is an
exact skew nearest-neighbor form. The high-high pressure coefficient
cancels its anisotropic kinetic partner, and a positive-polynomial
certificate bounds each remaining skew edge by one eighth of its signed
Fisher edge. Consequently `|B_HHL|<=E_lambda/2` uniformly in chain length.
Real constant and first-Dirichlet modes have zero load; phase tilts activate
pressure and remain controlled. This does not yet cover arbitrary residues,
low waves, partition phases, or cross-residue couplings. The next gate is
the corresponding finite low-wave/vertex block Schur estimate.

The scalar primitive-chain generalization is now closed, albeit with a
conservative constant. For every primitive partition step
`q in {-1,0,1}^3\{0}`, one isolated one-sided residue chain, arbitrary
transverse residue, arbitrary complex low polarization and phase, and any
translated compatible tensor vertex, the complete HHL load obeys
`|B_chain|<=108(|Uhat|/m)E_chain`. Two-coordinate steps improve to `27/2`
and three-coordinate steps to `6`. The proof combines the exact `9/2`
ordered complete-HHL symbol budget, an exhaustive six-partner resonance
count, and matrix-valued discrete Hardy. The omitted orthogonal sine-phase
block is nonzero but has generalized norm tending to `2/3`; it is not a
no-go. This result still cannot be summed over chains or primitive steps:
the next gate must assemble their shared Fisher matrix once in a joint
low-wave/vertex Schur block.

At the conditional-edge level, the apparent reciprocal singularity at a zero
partition face has now been removed exactly: optimizing the nonnegative face
coefficients against the cubic Fisher penalty gives the sharp envelope
`8|e|^(3/2)/(3sqrt(3)m sqrt(nu))`. The worst case is itself a zero face,
so no positive floor is needed at this stage. Conditional Holder reduces the
remaining term to the natural pressure `L^(3/2)` scale, and its ratio to
Fisher is `Re_cell^(3/2)`. This is a genuine improvement over the loose
reciprocal Young ratio `Re_cell^2`, but the resulting
`(U^2/nu)||u||_3^3` budget is not controlled by Leray energy. The globally
compatible eight-cell problem now has an exact seven-load projective
reduction, and its cubic energy is nonconvex. Compatibility annihilates the
Taylor-Green pressure load, but a smooth abstract vertex flux saturates all
three directionwise envelopes simultaneously, so compatibility alone gives
no uniform strict gain. The Fourier-triad range question is now settled:
seven spectrally isolated divergence-free blocks make the instantaneous
velocity-to-pressure map onto the complete zero-sum seven-load space. A
`42`-mode field realizes the exact saturating Hamming ray, while an
independent sparse seed-81 calculation reproduces the stored pressure work
to `4.9e-15`. This rules out an algebraic missing-ray argument, but not a
quantitative estimate. The first realization-cost gate is now also settled
for the explicit lacunary architecture. Under the exact co-scaling
`b_m[u_(a,m)]=a^3m b_1[u]`, the single-block `L2` and velocity-Fisher minima
are explicit. Unit-polarization couplings have nonzero high-carrier limits,
so one fixed bad load has a `42`-mode realization sequence with bounded `L2`
and bounded critical-`L3` upper cost as carrier rises. Thus critical `L3`
size alone is not carrier-coercive. In the same architecture the
block-optimal velocity-Fisher cost grows quadratically; an exact
correction-aware support certificate shows all `21` zero-leading quadratic
pairs are conjugate opposites, so every vertex weight sees exactly one eighth
of that gradient cost. A subsequent pure-high-pass theorem now gives the
global unweighted replacement: if `u_hat(k)=0` for `|k|<K`, then Holder,
the pressure Riesz bound, high-pass Poincare, and Sobolev interpolation imply
`|b_Phi|<=C||grad Phi||_infinity K^(-3/2)||grad u||_2^3`. Fixed nonzero load
therefore costs at least linearly in `K` in unweighted enstrophy. The
quadratic block power is not promoted globally.

That theorem cannot simply be weighted by a vertex basis. For
`phi_-=sin(x/2)^2`, an exact sine-window packet supported on
`{N+1,...,2N}` has unit `L2`, unweighted Dirichlet growth at least `(N+1)^2`,
but weighted Dirichlet below `38.86` uniformly in `N`. Its real shear
embedding is divergence free, although pressure-free. The correctly weighted
high-pass quantity nevertheless remains coercive. Factoring
`Phi_v=psi_v^2` gives the exact identity
`||grad(psi_v u)||_2^2=mean[Phi_v|grad u|^2]+(3m^2/4)||psi_v u||_2^2`.
Because multiplication by `psi_v` shifts carrier by at most
`sqrt(3)m/2`, for `K>sqrt(3)m` this yields
`||psi_v u||_2^2<=mean[Phi_v|grad u|^2]/[K(K-sqrt(3)m)]`, together with
an explicit weighted-Fisher bound for `u grad psi_v`. A separate
alias-free finite-Fourier pilot uses a three-dimensional Fejer-windowed
carrier triad, exact Leray projection, and its induced Poisson pressure.
After normalizing a vertex pressure load to one over `N=2,3,4,5`, weighted
Fisher falls from `125.37` to `96.80`, unweighted Fisher rises from `864.75`
to `2612.64`, and `U/K` rises from `3.71` to `5.10`. This is a numerical
mechanism test, not an asymptotic counterexample. Both the exact concentration
scaling and the PDE pilot preserve the adverse ratio `U/(nu K)`.

The proposed lower-carrier-only pressure commutator estimate is now settled
negatively. Multiplication by `psi_v` has an exact eight-shift Walsh
expansion: derivative order is Hamming distance on the vertex cube, and the
double-Riesz symbol has genuine distance-two and distance-three terms. A
two-scale counterexample makes those terms unavoidable. For the peak-one
Fejer packet shifted into Fourier coordinates `{3,...,2N+1}`, the exact curl
field `u_N=N^(-1)(partial_2 a_N,-partial_1 a_N,0)` is divergence free, has
fixed lowest mode `3sqrt(3)`, and has uniformly bounded amplitude while
concentrating at a triple partition zero. For a fixed smooth high-output
pressure tail,
`||psi p_N^H||_2>=cN^(-3)`, whereas
`||psi u_N||_2<=CN^(-9/2)` and
`||u_N grad psi||_2<=CN^(-7/2)`. The discarded diagonal ratio therefore
grows at least as `N^(1/2)`, and amplitude rescaling shows that the intrinsic
condition `K>=C||u||_infinity/nu` does not repair it. The exact surviving
bound is coupled across all eight Hamming vertices. The next target is an
annular or dyadic-shell estimate, where upper and lower carrier are
comparable, followed by cross-shell paraproduct summation.

That annular single-shell target is now proved for a smooth high-output
pressure multiplier. Fourier coefficients split into residue chains modulo
the partition frequency `m`; on a chain of length `N`, multiplication by
`sin(mx/2)` and `cos(mx/2)` is the zero-boundary difference/sum pair. Both
toggle inequalities have the sharp constant
`cot(pi/[2(N+1)])`. For coordinate degree `L`, the longest chain has length
`ceil((2L+1)/m)`, and tensorization bounds a Hamming-distance `d` vertex mass
by the original mass times the `d`th power of that constant. On
`K<=|k|<=Lambda K`, the multiplier difference factor `m/K` cancels the
chain length `O(K/m)`; the dimensionless toggle is at most
`2(Lambda+1)/pi`. Inserting this into the exact eight-shift identity gives
`||psi_v p_H||_2<=C_Lambda||u||_infinity||psi_v u||_2` and an explicit
single-vertex absorption condition for the smooth high-output self-shell
pressure. This does not cover a sharp cutoff, the low-output high-high beat,
mixed or separated shells, or the full pressure. The next live gate is the
signed low-output self-shell interaction before absolute values.

The complete self-shell pressure is now closed as well. The apparent
low-output obstruction disappears at the load level: every Fourier summand
requires `q+k+r=0`, while the partition gradient has
`|r|<=sqrt(3)m`. Hence pressure output below `Q` is exactly invisible to a
velocity shell above `K` whenever `Q+sqrt(3)m<=K`. For every
`K>sqrt(3)m`, an adaptive smooth split at half the spectral gap makes the
low load vanish and places the complement under the annular commutator
theorem. This gives a gap-dependent bound for the actual uncut self-shell
pressure. When `K>=2sqrt(3)m`, a fixed `K/2` split gives uniform constants.
Sparse divergence-free adversaries have substantial low pressure and
nonzero total work but exactly zero low-load resonances, while a separate
field below the fixed-split threshold has a nonzero low channel. The first
genuine nonlocal obstruction is therefore cross-shell
`high+high -> low` pressure tested against low velocity. Its
slowly-modulated Reynolds-stress limit is the next gate.

That cross-shell carrier-gain gate is now settled negatively. An exact
two-sideband divergence-free field uses high waves `(1,1,H)` and `(0,0,H)`,
whose difference is the fixed low wave `(1,1,0)`. Their low Reynolds stress
converges to `2e_1 tensor e_1`, and the induced pressure load against a
fixed low velocity and the all-cosine partition vertex converges to
`1/144`. The complete cubic high-high-low local-energy flux has the same
nonzero limit after kinetic transport and both cross-pressure terms are
included: isotropic kinetic energy cancels the pressure scalar, but the
anisotropic Reynolds stress survives. Exact sparse calculations through
`H=1024` reconstruct the cubic coefficient independently to `2.3e-16`.
Thus frequency separation alone supplies no positive `L/H` power. The live
route is an exact dyadic interaction atlas retaining shell amplitudes,
conservative transfer telescoping, eight-cell cancellation, and
time-integrated viscous payment.

The dyadic interaction atlas is now exact. Every occupied cubic load obeys
`k_1+k_2+k_3+r=0`, `|r|<=sqrt(3)m`; consequently the largest two velocity
frequencies are within a factor four above the partition scale. Only `HHH`
and `HHL` patterns survive. Global shell transfer is antisymmetric, but its
localized defect is precisely the physical boundary flux, so pure
fixed-vertex shell telescoping is false. The modulated `HHL` family is a pure
top Walsh character across the eight cells: equal weights cancel, while a
nonnegative selector retains half its `L1` load. Five separated high shells
accumulate coherently with load exactly linear in their Fourier `L2` energy.
This nevertheless leaves a positive amplitude theorem:
`|B_(v,L;HHL)|<=C m L^(3/2)a_L sum_(H>=4L)a_H^2`, and dyadic summation gives
`C sqrt(2m)||u||_2||grad u||_2^2`. That is absorbed only at small global
Reynolds size. The next gate is a joint scale-cell Carleson or
time-integrated improvement capable of beating this large-data coefficient.

Propagating a nonnegative terminal Legendre weight retains replica
dissipation and the critical Fisher term `lambda|grad lambda|^2`. At
partition frequency `m`, exact pressure/Fisher balance scales with the local
Reynolds number `a/(nu m)`, while the loose edge Young remainder scales with
its square. Fixed-scale universal absorption is therefore impossible, and
`m` must track local amplitude. On the seed-81 finite-mode stress,
frequencies `1` through `6` couple to pressure and `7` through `12` are
spectrally silent; this supports adaptation but is not by itself a
pressure-tail theorem. The first complete short-time correlation expansion
is retained as a consistency check: its reset loss is `2361.3578258`, its
first time coefficient is `-62459.6458230`, and its uncontrolled quadratic
truncation
crosses near `0.0756123`. The exact terminal-tax theorem now explains why
that truncation cannot certify a sign change. The intrinsic pressure-tail
bound at `rho=0`, the sharp cubic zero-face envelope, and the new
triple-zero curl-Fejer no-go now shift the live target to a bounded-annulus
commutator theorem. It must control the exact distance-two and
distance-three Hamming leakage when `K<=|k|<=Lambda K`; only after that
single-shell gate survives should comparable and separated shell
paraproducts be summed. Earlier universal-sign no-goes,
`G_rho>=|grad u|^2`, factor-`422` projection loss, and harmonic variance
remain route-selection foundations. No general absorption theorem, signed
critical estimate, or regularity conclusion is claimed.

Current checkpoint: a second computer-assisted supersolution now certifies
the ideal nonautonomous full-affine radial-payoff gain at `1.145614144998`,
so the square-tilted ideal pair cycle closes at `0.864492294975` after the
current cubic parent-child recentering cost and legacy return bound are both
charged. The older `0.677377028810` figure used only the bare halving factor
and is retained as a legacy calibration, not the live budget. Its radial
and axial corner exponents are both above `1/2`, giving finite global
Dirichlet energy and explicit critical forcing coefficients. This replaces
the preferred protected-collar route: exact IMS arithmetic rules out the
continuous cubic realization, while a separate audit shows that naive
interaction marking loses the global skew cancellation. A rigorous
conditional averaged-entry theorem now retains early nonreturns as
sub-Markov contraction and gives quantitative `L^3/L^(3/2)` budgets from a
space-time return-density envelope. Finite axial extent exactly repairs the
old axisymmetric affine long-tail objection, while a neutral-direction Weyl
sequence proves that this repair is not a uniform spectral theorem over the
full trace-free affine family. A neutral-strip stopping geometry restores a
positive model survival-tail margin for the obstructed half-family. Its
branch-resolved axial-patch pilot reaches `0.653278728734` when every wall exit
is paid the existing cubic true-split factor, but an exact zero-solution
counterexample shows that a geometric wall hit does not imply that physical
split. Without the payment, strip-width optimization bottoms out at
`1.159483564573`; tuning the present strip therefore does not close the scalar
model. A first geometry-triggered audit proves that no support-contained
half-scale child can reach an admissible outer wall, and paying a concentric
inward split still gives `1.210225321505` at `Y=2.1`. A wall-following
migrating child is different: its exact endpoint gauge factor is
`0.611720063380` at the axial wall point, while the correct unbounded
`x`-dependent gauge payoff remains integrable in the pilot. Its full-wall
criterion is `0.762477784893`, and one physical-to-gauge conversion gives
`0.845167851953`. A smoother boundary-tracking interpolation has the exact
geometric factor `2^(-3/4)=0.594603557501`; its corresponding criteria are
`0.627713348448` and `0.672126890291` after one conversion. Positive
translated partition transfer and fixed-branch pressure cancellation are
exact. The moving-core PDE now has an exact residual decomposition: amplitude
deficit is favorable, radial-frame rotation cancels, and the adverse part is
isolated as `q_res`. The scalar criterion permits a common integrated residual
action below `0.1986540657`, but the branch boundary-response constants and
the actual swept-core critical norms are not yet proved. Those are now the
live geometry gates, together with migration-centre moments and adapted PDE
localization. A branch-resolved trace audit further shows that the weighted
response constant is fixed by the square-tilted stopping law,
`K_j=p_j sqrt(C_4 J_j)`, and corrects a surface mismatch: raw wall flux lands
at child radius `2`, so its H1 response requires composition with a child
return to radius `1`. Migration residual is wall-only and has the sharper
integrated-action ceiling `0.296761039858`. A boundary-resolved semigroup
pilot now computes the actual static `rho=0` strip-return response on two
meshes; the conditional return-only thresholds are `0.366023205860` in
`L^(3/2)` potential mass and `0.107962714456` in `L^3` drift mass. The
continuum flux remains open. A subsequent discretization audit shows why
the fixed-bin mesh comparison cannot certify it: at fixed mesh the fitted
edge law is atomic, so after its atoms separate the histogram response grows
as `B^(1/4)`. On mesh 40 it rises from `0.625300617306` at 32 bins to
`1.242931559762` at 1024 bins, while the fitted Shortley-Weller generator
also fails exact detailed balance near the curved boundary. Thus the quoted
return thresholds are coarse finite-state diagnostics, not upper bounds.
A constrained-Delaunay, lumped weighted-FEM replacement now fixes those two
defects at `rho=0`: every retained rate is positive, detailed balance holds
to `3.5e-16`, the three hitting branches partition probability to `5e-15`,
and disjoint inner dual arcs sum exactly to `2 pi`. Coupled spacings
`0.16, 0.12, 0.09` give physical-face responses `0.618929914899`,
`0.620117729331`, and `0.621054704444`, with finest change
`0.000936975112`. This is meaningful convergence evidence, not yet a
continuum upper bound: polygonal, time-window, tail, and `x`-truncation
errors remain unenclosed.
A spectral-tail and axial-width audit now removes the fitted terminal slope
from each stored FEM model. High-precision Barta bounds give principal decay
at least `2.324559286697`, and the largest rigorous finite-matrix tail
payments at `T=6` are `3.758716458935e-5` in the interval factor and
`4.929230751102e-5` in scalar mass. At `X=4.2, 5.25, 6.3`, the explicitly
stressed responses are `0.623248235706`, `0.623596956316`, and
`0.623957665810`, a spread of `0.000709430104`, while truncation probability
falls from `1.664244080768e-3` to `3.374752264798e-8`. This closes the fitted
tail only for the stored finite matrices. A subsequent contractive
uniformization certificate replaces both empirical finite-time stresses.
Poisson remainders, sparse-arithmetic roundoff, maxima between time nodes,
an upper scalar Darboux sum, and the post-`T` spectral tail give responses
`0.619681476353`, `0.620016242618`, and `0.620371178275` at the same three
widths. The maximum propagated weighted-state error is `5.01e-12`, and an
independent Krylov checkpoint agrees within its error allowance. This closes
finite-time maxima and scalar quadrature for the symmetrized stored matrices;
the continuum FEM error remains open.
The artificial-side branch is now controlled analytically as well. Removing
the inner disk and separating the rectangle gives continuum side-exit upper
bounds `2.250904005638e-3`, `1.731311139515e-5`, and
`4.360755393701e-8`. A positive renewal inequality pays all later side
excursions by `1/(1-2p_X)`, while a global axial scalar bound pays missing
mass. The resulting stored-matrix responses are `0.622430322078`,
`0.620037342720`, and `0.620371231439`. This resolves the analytic `x`-tail
mechanism but still is not a continuum bound: polygonal-circle and weighted
FEM consistency errors remain open.
The first consistency gate now quantifies the regular inscribed-polygon
geometry exactly and independently assembles high-order weighted P1 mass,
stiffness, and inner-flux forms on the same meshes. At `h=0.12`, the first
twenty modes differ by at most `1.713%` in mass, `0.414%` in stiffness, and
`0.277%` in the boundary coupling; at `h=0.09` these fall to `0.950%`,
`0.221%`, and `0.133%`. The whole-space lumped/consistent mass ratio remains
near `3.82`, however, so a uniform operator perturbation is invalid. The
continuum route must split low modes from a damped high-mode tail and bound
the early-time continuum flux separately.
The high-mode half of that split is now analytic. A weighted Rellich identity
gives continuum source-to-inner-flux constant `3.134170665703`; conjugation
and Li-Yau put mode `321` above `62.256765195769` without using FEM
eigenvalues. The full-OU diagonal then bounds all high modes after `t=3/8`
by `0.000683411235` in interval factor and `0.000177212109` in scalar gain.
The retained 320-mode diagnostics improve under `h=0.12` to `0.09`, but the
first window, interval low block, and polygon-to-circle flux map remain open.
A final bounded pilot now composes wall flux,
the exact half-scale time/axial map, and one child return. Its three-mesh
working composite response is `K_S=0.073608663429`, with finest-mesh change
`0.00107099`; this supplies the missing finite-state model but not a
continuum or Navier-Stokes certificate, and it inherits the same child-return
 boundary histogram obstruction. Its conditional wall-core-only
 calibrations are `1.913793783533` in potential mass and `1.131396524524` in
 drift mass. All 113 regression tests pass.

A later continuum-certification branch has now completed the missing
positive-exponential hypercircle assembly on all 30,954 triangles of the
stored `h=0.06` mesh. All 216,678 q12 entry checks and 854 independent q18
checks are contained. The hash-bound archive stores an exactly symmetric
232,421-nonzero RT0 center/radius pair, 30,954-entry W and D diagonals, 91,124
P1-P0 loads, and 92,862 exact signed divergence incidences. This closes the
matrix-entry stage only. Full threshold-pencil inertia, `kappa_h<0.045`, the
global Ritz constant, and continuum spectral capture remain false.

The ensuing full central-factorization audit shows that the matrix size is not
the immediate obstacle. `MMD_AT_PLUS_A` with ten-step symmetric Ruiz scaling
factors the 123,816-dimensional pencil with 5,630,594 L+U entries and about
69.5 MB of factor storage, while consistently observing the target central
pivot count `61908/61908/0`. MMD-ATA and COLAMD are numerically rejected. The
global norm roundoff proxy still exceeds the smallest selected pivot by a
factor of about 2,270, so inertia remains uncertified and the next route must
be componentwise directed LDL or a verified residual bound.

Both replacement routes now certify the bounded leading problem through pivot
33,280. Directed interval LDL at precisions 50 and 80 gives 31,971 negative
and 1,309 positive pivots, minimum margin
`0.003304358585502960725532827806`, and complete cross-precision nesting. An
independent directed congruence-residual proof at precisions 60 and 100
obtains the same signs with transformed-residual/minimum-diagonal ratio below
`2.940e-8`. Pivot 32,849 is the first strongly cancellation-sensitive point;
the actual six-descendant transition at 33,224 remains well separated from
zero. A complete symbolic scan shows that the arithmetic is strongly
backloaded: the final 9,128 pivots carry about `95.39%` of all off-diagonal
recurrence terms. The next bounded target is a standalone residual pilot
through first state entry at 63,680. Full inertia and every continuum claim
remain false.

A genuinely standalone residual runner now removes the remaining logical
dependency on a matching directed-LDL audit. Hash-bound replays at 2,304,
32,064, and 33,280 reproduce the historical certificate values exactly. At
63,680 pivots, precision-60 and precision-100 runs certify 32,392 negative and
31,288 positive reference diagonals with transformed-residual/minimum-
diagonal ratio below `2.462e-4`. The first 11 state pivots, beginning at
63,644, are all positive with minimum absolute diagonal `0.636695710069665`.
The global minimum `0.000815079692889409` instead occurs at edge-metric pivot
63,629. The original separated norm product fails first when edge pivot
64,039 is appended and reaches ratio `95.5495` by 64,064 pivots. A sharper
directed componentwise theorem evaluates `max(Q R Q^T 1)` rather than
`||Q||_inf ||R||_inf ||Q||_1`, preserving the residual/inverse path geometry.
It certifies the 64,064-pivot interval family at precisions 60 and 100 as
32,500 negative and 31,564 positive, with ratio `0.270366` and safety factor
`3.69869`. A separately reconstructed precision-60/100 extension now certifies
64,128 pivots as 32,564 negative and 31,564 positive. All precision-100
control metrics are exactly flat from 64,064 to 64,128: the 64 added
edge-metric pivots are negative and comfortably separated from zero, the
leading factor is bitwise unchanged, and the controlling row remains pivot
64,040. The next bounded target is only 64,256; full inertia and every
continuum claim remain false.

The initial audit keeps three notions separate:

1. deterministic labelled fluid parcels;
2. two labels in one stochastic Lagrangian flow, driven by common noise;
3. independent stochastic Lagrangian replicas.

The first exact structural match occurs in case 3. If `g` is the distance
between two independent viscous replicas in three dimensions, Ito's formula
produces the outward drift

```text
4*nu/g.
```

This has exactly the singular order and sign of the neighbouring-zero barrier.
It is not yet a Navier-Stokes regularity theorem: independent-replica
separation does not automatically control deformation inside one stochastic
flow, and almost-everywhere non-collision does not exclude an exceptional
singularity.

## Files

- `notes/first_principles_collision_audit.md`: derivations, stress tests, and
  proof gates.
- `notes/replica_correlation_bridge.md`: the proposed bridge from backward
  replica separation to control of coherent vortex stretching.
- `notes/two_point_vorticity_collision_system.md`: exact two-point tensor PDE,
  centre/separation generator, and angular-kernel obstruction.
- `notes/heat_scale_cubic_cancellation.md`: exact heat multiplier split,
  double-zero cubic cancellation, pair-determinant formula, and the current
  collision-rigidity continuation criterion.
- `notes/collision_defect_dynamics.md`: fixed-scale Navier-Stokes evolution,
  exact Fourier-triad constraints, and the cross-scale sign obstruction.
- `notes/cumulative_collision_rigidity.md`: time-integrated continuation
  criterion and inverse-frequency triad primitive.
- `notes/trajectory_collision_defect.md`: exact Navier-Stokes Taylor jet,
  resummed weak response, and the high-Reynolds decision gate.
- `notes/galerkin_collision_trajectory.md`: validated Reynolds sweep,
  truncation gates, and loss of the negative cumulative channel.
- `notes/helical_transfer_recurrence.md`: exact helical/generation split and
  the positive sixth-order difference-mode return.
- `notes/second_heat_normal_form.md`: exact second heat primitive, general
  quintic obstruction, and exceptional two-mode order-six recurrence.
- `notes/third_heat_normal_form.md`: denominator-preserving normal-form trees,
  exact sextic sign obstruction, and finite-iteration closure verdict.
- `notes/normal_form_resummation.md`: graded Neumann resolvent, finite-Galerkin
  convergence majorant, and nonperturbative circularity gate.
- `notes/collision_coherence_generator.md`: joint replica-separation and
  deformation generator, strict collision damping, and occupation criterion.
- `notes/localized_strain_tube.md`: exact spectral escape theorem for an ideal
  finite strain core, its positive margin, and the nonuniform/re-entry gates.
- `notes/moving_strain_tube_robustness.md`: moving-coordinate gauge,
  rotation cancellation, and an integral-norm perturbation budget.
- `notes/migrating_core_residual_budget.md`: exact tracked-core residual
  decomposition and angle-resolved scalar error allowances.
- `notes/wall_stopping_trace_composition.md`: square-tilted branch response,
  return calibration, and the wall-to-child-core surface correction.
- `notes/neutral_strip_return_density.md`: boundary-resolved strip semigroup,
  spatial-L2 return envelope, and two-mesh response calibration.
- `notes/neutral_strip_boundary_density_discretization_no_go.md`: proof that
  fixed-mesh fitted-edge histograms have no finite boundary-`L2` limit and
  audit of the curved-stencil detailed-balance defect.
- `notes/neutral_strip_reversible_boundary_fem.md`: positive reversible
  body-fitted replacement, physical circle-face norm, and coupled mesh
  response pilot.
- `notes/neutral_strip_reversible_spectral_tail_width.md`: high-precision
  Barta decay enclosure, analytic finite-matrix tail, and axial-width stress.
- `notes/neutral_strip_reversible_finite_time_certificate.md`: contractive
  uniformization, finite-window maxima, and scalar Darboux certificate for
  the stored reversible FEM matrices.
- `notes/neutral_strip_x_exit_correction.md`: interval-enclosed Kummer side
  probability and positive renewal correction for all later side excursions.
- `notes/neutral_strip_reversible_fem_consistency_gate.md`: exact polygon
  geometry, independently assembled weighted P1 forms, and the low/high-mode
  route forced by the global lumped-mass obstruction.
- `notes/neutral_strip_parabolic_spectral_split.md`: explicit weighted
  Rellich flux constant, full-OU diagonal majorant, and analytic continuum
  high-mode payment after the first time window.
- `notes/neutral_strip_first_window_brownian_majorant.md`: stopped
  OU-to-Brownian domination, Bessel absolute continuity, and a summable
  radial-excursion bound for the complete first return window.
- `notes/neutral_strip_first_window_maximum_bridge_certificate.md`: positive
  bridge-maximum integration, outward-rounded radius/time slabs, and a
  certified first-window interval factor below one.
- `notes/neutral_strip_transient_conormal_low_block_gate.md`: corrected
  consistent-mass transient conormal identity, projected-dynamics and
  off-block leakage diagnostics, exact common-circle boundary reconstruction,
  source-aware convergence screen, and radial additive-screen warning.
- `results/neutral_strip_common_circle_source_summary.json`: machine-readable
  four-mesh common-circle/source diagnostic, cache fingerprints, and explicit
  non-certification flags.
- `notes/neutral_strip_common_circle_source_time_slab_certificate.md`:
  endpoint-interpolation and post-`6` geometric-tail enclosure for the frozen
  source-resolved finite block, with the `15 + tail` indexing correction.
- `notes/neutral_strip_common_circle_endpoint_roundoff_audit.md`: directed
  binary64 arithmetic enclosure for all 451 frozen `h=0.06` endpoints, plus an
  independent 80-digit reconstruction and explicit input-level caveats.
- `notes/neutral_strip_common_circle_eigensystem_residual_audit.md`: directed
  stored-mass coercivity, generalized-eigenpair residuals, and 241 disjoint
  spectral proximity intervals, with indexed counting left explicit.
- `notes/neutral_strip_gaussian_weighted_assembly_interval_audit.md`:
  directed element geometry, analytic Gaussian moment/remainder enclosures,
  exact q12 fingerprint reconstruction, and global reference-form errors.
- `notes/neutral_strip_continuum_ritz_dependency_audit.md`: exact stored-mesh
  P1 conformity, corrected conforming eigenvalue direction, and a fail-closed
  cutoff-resolvent route with explicit solution-operator and projection-
  constant thresholds.
- `notes/neutral_strip_weighted_hypercircle_pilot.md`: corrected weighted
  source decomposition, exact-mesh P1/RT0 pilot, and the sparse threshold-
  pencil route for a global Ritz projection bound.
- `notes/neutral_strip_weighted_hypercircle_sparse_inertia_pilot.md`:
  four-block inertia identity, coarse Schur-complement validation, and
  full-mesh sparse fill diagnostics.
- `notes/neutral_strip_positive_exponential_rt_interval_pilot.md`: degree-22
  directed moments, distributed RT0/P0 checks, and complete directed geometry
  budget at exact decimal `beta=0.045`.
- `notes/neutral_strip_positive_exponential_complete_assembly.md`: resumable
  all-triangle P/W/D/B/N assembly, sparse radius aggregation, and independent
  hash replay.
- `notes/neutral_strip_weighted_hypercircle_central_factorization_audit.md`:
  full central ordering/scaling sweep, fixed elimination fingerprints, and
  the fail-closed interval-roundoff decision.
- `notes/neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.md`:
  hash-bound adaptive directed-Decimal LDL on the first 1,024 pivots,
  independent 80-digit nesting replay, and the exact first recurrence
  interaction at pivot 1,738.
- `notes/neutral_strip_weighted_hypercircle_directed_ldl_interaction2048_audit.md`:
  verified interaction-bearing extension through pivot 2,048, CPU-safe
  precision-80 resume/nesting replay, and the next fill transition at
  pivots 2,270/2,274.
- `notes/neutral_strip_weighted_hypercircle_transition32064_feasibility.md`:
  two-route certification through pivot 32,064, complete symbolic workload
  map, full-run checkpoint feasibility, and the remaining continuum gates.
- `notes/neutral_strip_weighted_hypercircle_transition33280_audit.md`:
  two-route certification through pivot 33,280, the cancellation-sensitive
  pivot at 32,849, the six-descendant transition, and the standalone residual
  state-entry gate.
- `notes/neutral_strip_weighted_hypercircle_standalone_state_entry63680.md`:
  directed-independent residual provenance, exact historical regression,
  first-state-entry certification, and inverse-majorant risk growth.
- `notes/neutral_strip_weighted_hypercircle_state_region64064_obstruction.md`:
  precision-stable fail-closed residual result, transition-cluster audit, and
  localization of the inverse-majorant obstruction to the 64,039--64,040
  edge chain.
- `notes/neutral_strip_weighted_hypercircle_componentwise_recovery64064.md`:
  exact old-bound closure-loss boundary, directed componentwise residual
  theorem, and standalone recovery of the full 64,064-pivot state region.
- `notes/neutral_strip_weighted_hypercircle_componentwise_growth64128.md`:
  precision-nested 64,128-pivot certificate, exact leading-factor and source
  nesting, flat-control explanation, and conservative 64,256 next gate.
- `notes/wall_migration_child_return_density.md`: scale-correct wall-to-child
  convolution, wall-atom handling, and three-mesh composite response.
- `notes/strain_tube_reentry_renewal.md`: buffered visit decomposition,
  three-dimensional return barrier, and renewal convergence gate.
- `notes/three_dimensional_leray_gate.md`: full 3D gauge, critical
  `L^(3/2)` visit bound, Leray scaling gap, and weighted-return obstruction.
- `notes/strain_eigenframe_geometry.md`: maximal-strain evolution, viscous
  frame coherence, alignment deficit, and nonlocal pressure-Hessian gate.
- `notes/pressure_collision_heat_split.md`: trace-free pressure collision
  kernel, local reaction split, simple-zero obstruction, and maximum gate.
- `notes/pressure_frame_pairing_obstruction.md`: smooth periodic counterexample
  to pointwise pairing and the surviving localized boundary identity.
- `notes/pressure_shell_commutator.md`: exact localized pressure identity,
  fixed-scale scaling obstruction, and adaptive local-Reynolds gate.
- `notes/adaptive_reynolds_envelope.md`: monotone shrinking tube whose
  amplitude and scale-motion errors are nonpositive for `R_*<=2`.
- `notes/shrinking_tube_renewal.md`: moving Newtonian return barrier,
  shrinkage budget, and conditional two-history renewal lemma.
- `notes/pressure_partition_flux.md`: partition-of-unity pressure
  conservation and the neighboring-weight mismatch gate.
- `notes/intrinsic_radius_cover.md`: Lipschitz minorant of the diffusion
  radius, preserved Reynolds cap, and comparable overlapping cells.
- `notes/monotone_dyadic_cover.md`: fixed-centre, split-only balanced cover
  and the parent-child gauge transition gate.
- `notes/dyadic_gauge_transition.md`: exact recentering cost, radius-halving
  shrink payment, and the balance-refinement warning.
- `notes/branching_transfer_operator.md`: conservative octree and replica-pair
  transfer, harmonic weight inheritance, and the weighted closure criterion.
- `notes/interface_weight_no_go.md`: proof that closed interface transfer
  forces constant weights and the resulting physical/gauged two-norm route.
- `notes/two_norm_generation_cycle.md`: correct same-scale renewal followed
  by one true dyadic split and the resulting visit-action threshold.
- `notes/buffered_visit_feynman_kac.md`: exact transverse core-shell visit
  gain, logarithmic moment threshold, and corrected Reynolds regime.
- `notes/axial_killing_buffered_visit.md`: inward-OU axial eigenvalue,
  separated-mode visit improvement, and finite-core aspect-ratio gate.
- `notes/finite_cylinder_mode_expansion.md`: complete axial OU boundary
  expansion and corrected compact-core aspect-ratio thresholds.
- `notes/finite_cylinder_perturbation_margin.md`: exact core/shell adverse-
  potential thresholds and scale-invariant mass calibration.
- `notes/finite_cylinder_kato_gate.md`: exact Green-operator robustness
  budget and endpoint no-go for pointwise `L^(3/2)` control.
- `notes/gaussian_boundary_l2_transfer.md`: exact Gaussian boundary visit
  norm, dynamic-measure Markov contraction, and entry/exit density budget.
- `notes/ground_state_visit_transform.md`: exact visit Doob transform,
  principal scalar growth, reversible mixing spectrum, and kernel
  minorization audit.
- `notes/axial_form_to_boundary.md`: rigorous relative-form propagation to
  the Gaussian visit norm for nonconstant separable axial perturbations.
- `notes/off_diagonal_form_transfer.md`: correct resolvent cross-block form
  bound, counterexample to the naive relative estimate, and renewal budget.
- `notes/weighted_cylinder_buffer_condition.md`: converged internal Green
  condition numbers, collar dependence, and full angular-mode audit.
- `notes/poisson_cutoff_form_transfer.md`: complete outer Poisson form bound,
  cutoff-energy constants, full mode search, and critical mass budgets.
- `notes/quadratic_partition_ims_budget.md`: exact weighted IMS identity,
  overlap-width obstruction, and sequential versus octree localization.
- `notes/radial_cubic_partition.md`: compact cubic B-spline partition fitted
  to the radial Poisson collar, exact Fisher/IMS bound, optimized collar
  tradeoff, and the remaining level-transition gate.
- `notes/cubic_level_transfer.md`: positive cubic refinement mask,
  pointwise Markov child labels, gauged radius-halving contraction, and a
  global monotone-level alternative to balance-only refinement.
- `notes/navier_stokes_coherence_budget.md`: explicit critical and
  Leray-level affine-coherence constants on the optimized support, plus the
  Galilean no-go for charging cell translation as a potential.
- `notes/general_affine_spectral_floor.md`: spectrum-uniform Dirichlet floor
  for every symmetric trace-free affine core, post-IMS margin, and the
  remaining anisotropic Poisson and moving-frame gates.
- `notes/anisotropic_poisson_transfer_pilot.md`: coupled-angular finite-element
  stress test over the full affine spectrum, including boundary, cutoff, and
  axial-mode diagnostics.
- `notes/reversible_shell_rigidity.md`: exact no-taper theorem for smooth
  reversible incompressible shells and the compact full-affine fallback.
- `notes/divergence_free_shell_taper.md`: streamfunction-localized affine
  shell, exact strain reduction, minimax taper, and nonsymmetric visit pilot.
- `notes/sectorial_poisson_transfer.md`: conditional nonsymmetric
  form-to-boundary theorem and calibrated critical-mass budget.
- `notes/moving_cubic_label_transport.md`: normalized translating and rotating
  cubic labels, conservative simplex flux, the exact parabolic Fisher/IMS
  correction, and moving-frame sector budgets.
- `notes/continuous_cubic_localization_no_go.md`: exact lower bound for the
  cardinal-cubic Fisher cost and a disk-ground-state certificate ruling out
  continuous full tensor localization in the wide tapered visit.
- `notes/stopping_time_moving_visit.md`: fixed-label moving-cylinder SDE,
  scale-critical remainder identities, and conservative relabeling only at
  buffered stopping times.
- `notes/coherence_abort_renewal.md`: Neumann restart theorem for visits that
  lose affine coherence early, including true-split and bad-occupation
  thresholds.
- `notes/leray_mollified_cell_frame.md`: Galilean-covariant mollified centre,
  spin-following `SO(3)` frame, and entry affine tensor defined at Leray
  regularity without eigenframe derivatives.
- `notes/nonautonomous_scalar_gain_gate.md`: arbitrary-history volume bound,
  stationary surface estimate, and the failure of static-worst comparison.
- `notes/weighted_kernel_dynamic_l2.md`: exact square-tilted law and dynamic
  positive-kernel norm identity.
- `notes/radial_payoff_bellman_gate.md`: exact affine-control Hamiltonian,
  radial-payoff HJB pilot, and certified explicit supersolution.
- `notes/radial_bellman_doob_perturbation.md`: certified common killing,
  exact weighted drift cancellation, and the critical boundary-form gate.
- `notes/critical_collar_transfer_gate.md`: affine off-diagonal collar
  estimate, translated endpoint no-go for global Kato iteration, and the
  conditional dynamic sector budget.
- `notes/radial_collar_trace_calibration.md`: optimal barrier-cutoff energy,
  complete stationary collar trace norm, and temporal-frequency stress test.
- `notes/protected_collar_partition_no_go.md`: exact IMS obstruction to
  realizing the protected core with the continuous cardinal-cubic weights.
- `notes/interaction_marked_localization_gate.md`: exact Dyson marking,
  labelwise skew-commutator obstruction, and the coupled-label dichotomy.
- `notes/radial_h1_payoff_supersolution.md`: certified finite-energy HJB
  barrier, improved renewal gain, and global critical forcing constants.
- `notes/averaged_entry_trace_gate.md`: unnormalized space-time return law,
  interval-energy trace theorem, and conditional density budgets.
- `notes/exterior_return_tail_gate.md`: polynomial-tail correction,
  summable-envelope theorem, and spatial-L2 Poisson-kernel calibration.
- `notes/cylindrical_brownian_return_pilot.md`: exact cylinder mode formula,
  stress-envelope pilot, and recurrent-tail budget.
- `notes/branch_resolved_entry_renewal.md`: exact split/return renewal
  algebra, unnormalized mass bookkeeping, and finite-patch Brownian pilot.
- `notes/split_entry_density_inheritance.md`: Markov density inheritance,
  deterministic-time atom obstruction, and fixed-time volume alternative.
- `notes/affine_exterior_axial_compensation.md`: exact outward-OU axial
  cancellation, Tricomi return modes, and weighted affine tail pilot.
- `notes/anisotropic_affine_exterior_tail_gate.md`: exact affine exponent
  balance and neutral-direction Weyl obstruction to a uniform tail rate.
- `notes/neutral_strip_storage_gate.md`: exact stopped-strip Poincare repair
  and the remaining wall-exit-to-true-split identification.
- `notes/neutral_strip_branch_resolvent_pilot.md`: branch-resolved finite-state
  return/wall resolvents and the raw scalar-residual failure.
- `notes/neutral_strip_axial_patch_branch_pilot.md`: exact axial-patch return
  transform, deformation-weighted wall moment, and conditional scalar margin.
- `notes/geometric_wall_split_compatibility.md`: exact event-separation
  counterexample and direct-child geometry obstruction.
- `notes/neutral_strip_same_scale_width_sweep.md`: unpaid wall-branch width
  optimization and the remaining above-one minimum.
- `notes/geometry_triggered_migrating_child.md`: exact nested-child
  obstruction, wall-following endpoint gauge, translated Markov transfer,
  and the remaining adapted-localization gate.
- `scripts/verify_collision_identities.py`: symbolic checks of the main exact
  identities.
- `scripts/gram_boundary_audit.py`: dimension/codimension audit for pair,
  triangle, and tetrahedron collapse under independent viscosity.
- `scripts/backward_replica_audit.py`: exact anisotropic separation in an
  affine strain field.
- `notes/correlated_replica_tangent_gramian_bridge.md`: exact Brownian
  correlation homotopy, common-noise tangent limit, forward/inverse Gramian
  congruence, determinant floor, deformation bounds, and non-promotion
  gates.
- `scripts/correlated_replica_tangent_gramian_audit.py`: symbolic
  correlation identities and independent affine, shear, rotation,
  noncommuting, and scaling stress tests.
- `notes/parabolic_gramian_continuation_gate.md`: exact local restart laws,
  critical `L^3` continuation hierarchy, scalar-trace falsifier, and the
  surviving tensorial theorem target.
- `scripts/parabolic_gramian_continuation_audit.py`: Burgers-axis, exact
  periodic shear, exact periodic ABC, cocycle, determinant, and small-window
  strain checks for the continuation hierarchy.
- `notes/projected_weber_replica_gate.md`: joint tangent/covector generator,
  smooth tensor proxy, affine single-weight obstruction, exact shear
  projection losses, and signed projected replica target.
- `scripts/projected_weber_replica_gate_audit.py`: symbolic generator and
  shear identities, tensor-power checks, deterministic magnetization sweep,
  and exact common-path harmonic-variance certificate.
- `notes/signed_projected_replica_generator.md`: correlated projected
  two-replica balance, Gaussian correlation homotopy, weighted critical
  dual, three-replica tensor generator, and pressure-flux gate.
- `scripts/signed_projected_replica_generator_audit.py`: exact generator
  algebra, shear/ABC/Burgers stresses, Hermite quadrature, and four-grid
  smooth critical-pressure falsifier.
- `notes/adjoint_replica_pressure_edge_gate.md`: backward terminal dual,
  reset correlation ordering, exact cubic edge penalty, smooth sign no-go,
  and reciprocal-weight pressure gate.
- `scripts/adjoint_replica_pressure_edge_gate_audit.py`: symbolic adjoint
  algebra, exact shear and ABC checks, four-grid amplitude stress, and
  scalar partition-edge/Fisher identities.
- `notes/scale_adapted_edge_rho_expansion.md`: local-Reynolds edge scaling,
  partition-frequency sweep, first-chaos pressure linearization, and the
  complete first short-time rho coefficient.
- `scripts/scale_adapted_edge_rho_expansion_audit.py`: exact scale algebra,
  twelve-frequency edge audit, replica pressure operator, and three-grid
  short-time correlation expansion.
- `notes/finite_window_rho_terminal_tax_no_go.md`: exact endpoint-tax
  identity, Wiener-chaos generator ordering, formal-crossover
  reinterpretation, and the boundary of the finite-window no-go.
- `scripts/finite_window_rho_terminal_tax_audit.py`: symbolic endpoint
  algebra, weighted spatial chaos stress, independent Hermite quadrature,
  and route-decision gates.
- `notes/intrinsic_pressure_tail_gate.md`: low-low-free pressure-tail
  decomposition, conditional intrinsic absorption, exact Taylor-Green
  scaling family, and zero-face weighted-localization obstruction.
- `scripts/intrinsic_pressure_tail_gate_audit.py`: symbolic tail estimate,
  exact amplitude-frequency stress, and explicit weighted
  singular-integral norm lower bound.
- `notes/pressure_hamming_commutator_gate.md`: exact eight-shift Walsh
  calculus, Hamming-cube pressure leakage, and the divergence-free
  lower-carrier diagonal counterexample.
- `scripts/pressure_hamming_commutator_gate_audit.py`: rational multiplier
  reconstruction and an alias-free curl-Fejer pressure replay at a triple
  partition zero.
- `notes/annular_vertex_commutator_gate.md`: sharp residue-chain toggle
  theorem, tensor Hamming collapse, and smooth self-shell pressure
  absorption with exact scope limits.
- `scripts/annular_vertex_commutator_gate_audit.py`: generalized-eigenvalue
  replay and an alias-free shellized curl-Fejer stress.
- `notes/self_shell_pressure_closure.md`: exact far-low orthogonality,
  adaptive spectral-gap decomposition, and the complete self-shell pressure
  theorem.
- `scripts/self_shell_pressure_closure_audit.py`: lattice support replay,
  sparse divergence-free adversaries, and fixed-split threshold stress.
- `notes/cross_shell_modulated_wave_gate.md`: exact two-sideband
  Reynolds-stress limit and pressure-plus-transport carrier-decay no-go.
- `scripts/cross_shell_modulated_wave_gate_audit.py`: sparse Fourier
  reconstruction of every cubic HHL component through carrier `1024`.
- `notes/dyadic_three_shell_atlas.md`: exact `HHH/HHL` support atlas,
  localized shell-skew defect, top-Walsh cell flux, and amplitude summation.
- `scripts/dyadic_three_shell_atlas_audit.py`: transfer-identity replay,
  coherent five-shell stress, and dyadic sequence envelope.
- `notes/joint_scale_cell_viscous_occupation.md`: cumulative low-output
  stress, common Fourier-Walsh coherence no-go, and the surviving
  time-integrated viscous occupation theorem.
- `scripts/joint_scale_cell_viscous_occupation_audit.py`: exact eight-shell
  channel replay, Stokes heat law, Gram-Schur estimate, and conditional
  Duhamel forcing gate.
- `notes/nonlinear_stress_regeneration_gate.md`: exact projected stress
  evolution, the HHL sweeping commutator theorem, the HHH anisotropic
  pressure-strain obstruction, and sparse parabolic pulse scaling.
- `scripts/nonlinear_stress_regeneration_gate_audit.py`: paired-triad
  symbol reconstruction, random polarization stress, HHH limiting witness,
  and the time-summability replay.
- `notes/dense_annular_hhh_packet_gate.md`: exact rational HHH channel,
  dense coherent lattice count, sharp `H^(5/2)` Bernstein loss, top-Walsh
  coupling, and the Leray-input-only parabolic no-go.
- `scripts/dense_annular_hhh_packet_gate_audit.py`: exhaustive annular
  lattice replay through `226981` coherent triples and exact scaling
  certificate.
- `notes/scalar_local_energy_regeneration_gate.md`: exact scalar trace
  cancellation, complete differentiated HHL transfer, frequency-isolated
  dense top-Walsh witness with corrected continuous-domain positivity,
  and the sharp Leray-controlled `H^(-3/2)` forcing norm.
- `scripts/scalar_local_energy_regeneration_gate_audit.py`: independent
  quartic polarization check, all-term low-mode evolution, dense spaced
  packet replay, and weighted viscous response certificate.
- `scripts/dense_spaced_continuum_positivity_audit.py`: outward-rounded
  interval proof of uniform complete-quartic positivity over a relaxed
  six-dimensional offset domain.
- `notes/smooth_galerkin_shell_response_gate.md`: exact pair-rate
  high-shell stress evolution, heat-weighted HHL/filter commutator, direct
  sharp-Galerkin-boundary payment, complete `H^(-3)` forcing square, and
  forced plus initial tail theorem for fixed finite low channels.
- `scripts/smooth_galerkin_shell_response_gate_audit.py`: pairwise
  evolution reconstruction, exact heat-rate replay, dyadic forcing-square
  checks, and response/initial-stress summation audit.
- `notes/scale_uniform_low_output_tail_gate.md`: scale-uniform summation of
  every low-output Fourier channel, `H^(-1-epsilon)` Galerkin stress-tail
  compactness, and the exact channel-saturated `H^(-1)` endpoint pulse.
- `scripts/scale_uniform_low_output_tail_gate_audit.py`: exact lattice
  multiplicity, dyadic Sobolev threshold, Galerkin-tail rates, and endpoint
  relaxation audit.
- `notes/direct_h_minus_one_stress_tail_gate.md`: physical-space correction
  of the channelwise endpoint inference and a Galerkin-uniform
  `L2_t H_x^(-1)` high-high stress-tail theorem.
- `scripts/direct_h_minus_one_stress_tail_gate_audit.py`: exact five-shell
  overlap arithmetic, deterministic sequence replay, and admissibility
  classification of the old saturated pulse.
- `notes/dense_low_output_block_gate.md`: positive-volume low-output theorem
  for the dense HHH packet, retained as an instantaneous spatial diagnostic
  after the direct endpoint correction.
- `scripts/dense_low_output_block_gate_audit.py`: outward-rounded fixed-channel
  interval certificate, exact output multiplicity, and finite projected-symbol
  replay.
- `notes/floor_free_pressure_edge_tail_gate.md`: terminal-time-uniform,
  floor-free removal of far-carrier low-output high-high pressure beats from
  each smooth projected-replica partition edge.
- `scripts/floor_free_pressure_edge_tail_gate_audit.py`: pinned endpoint
  prerequisite, Fourier Sobolev-duality replay, and scale-adapted diagonal
  tail certificate.
- `notes/balanced_annular_pressure_edge_gate.md`: complete-output,
  floor-free intrinsic absorption of one bounded annular self-pressure edge,
  with exact multiband scope limits.
- `scripts/balanced_annular_pressure_edge_gate_audit.py`: prerequisite hash
  gate, sparse Taylor-Green and seed-81 replay, compatible graph-Fisher
  check, and co-scaling certificate.
- `notes/multiband_weighted_fisher_recombination_no_go.md`: exact
  divergence-free dyadic-chain counterexample to floor-free component
  Fisher recombination and the signed-interface route decision.
- `scripts/multiband_weighted_fisher_recombination_no_go_audit.py`: exact
  rational chain identities, positive-floor and co-scaling stress, plus
  Taylor-Green and seed-81 finite-field replays.
- `notes/pressure_active_fisher_null_compatibility_gate.md`: exact
  two-polarization residue-chain HHL symbol, pressure/kinetic cancellation,
  positive-polynomial edge bound, and strict generalization limits.
- `scripts/pressure_active_fisher_null_compatibility_gate_audit.py`: symbolic
  coefficient certificate, sparse constant/Dirichlet and pressure-active
  phase-tilt replays, generalized chain spectra, and mandatory adversaries.
- `notes/primitive_hhl_chain_hardy_envelope.md`: uniform isolated-chain
  extension to every primitive cube step, transverse residue, low
  polarization and phase, with exact Hardy and Fisher-floor constants.
- `scripts/primitive_hhl_chain_hardy_envelope_audit.py`: exhaustive resonance
  count, exact complete-symbol budget, long-chain Hardy spectra, translated
  sparse primitive atlas, co-scaling stress, and mandatory adversaries.
- `notes/joint_primitive_hhl_incidence_schur_gate.md`: shared-Fisher finite
  block assembly, direct pressure-dominated growth witnesses, and the
  fail-closed route from numerical slabs to an explicit asymptotic family.
- `scripts/joint_primitive_hhl_incidence_schur_gate_audit.py`: all 52 real
  cube-low coordinates, sparse complete-HHL incidence blocks, exact-null
  Fisher quotient, Schur spectra, resumable windows, and direct witnesses.
- `notes/separable_annular_pressure_schur_no_go.md`: explicit bounded-annulus
  divergence-free family, exact mixed-difference Fisher identity, strict
  pressure limit, complete-HHL asymptotics, and fail-closed route decision.
- `scripts/separable_annular_pressure_schur_no_go_audit.py`: exact
  `Q(sqrt(2))` pressure/kinetic certificates, vectorized annular replays,
  continuum quadrature, fixed-transverse control, and dictionary crosscheck.
- `notes/annular_eight_vertex_heat_window_gate.md`: exact six-channel
  Walsh response, equal-weight cancellation, vertex Fisher scaling, heat
  persistence, and perturbative Navier-Stokes shadowing consequence.
- `scripts/annular_eight_vertex_heat_window_gate_audit.py`: all-vertex
  incidence matrices, stable difference/sum Fisher transforms, continuum
  sign certificate, heat-window replay, and eight-vertex dictionary check.
- `notes/compatible_edge_annular_escape.md`: exact joint low-amplitude and
  coefficient-ray optimization, full-field support proof, fixed-ray
  classification, and bounded-weight escape theorem.
- `scripts/compatible_edge_annular_escape_audit.py`: rational edge-penalty
  replay, complete flux/Fisher dictionary check, asymptotic constants, and
  finite optimized and bounded-coefficient crossings.
- `notes/deficit_retaining_annular_restart_gate.md`: exact reset-time
  Legendre deficit, pressure-only correction, annular tax theorem, and
  parabolic `N^2` amplification gate.
- `scripts/deficit_retaining_annular_restart_gate_audit.py`: symbolic
  endpoint identity, exact partition norms, pressure-only finite replay,
  and fail-closed reset-tax scaling audit.
- `notes/annular_rho_zero_first_jet.md`: exact generator variations,
  dealiased finite replay, heat-weighted HHL identity, and the strictly
  negative viscous-pressure `N^5` limit.
- `scripts/annular_rho_zero_first_jet_audit.py`: symbolic directional
  derivatives, rectangular sixfold de-aliasing, independent weighted-load
  replay, continuum sign certificate, and five-carrier decomposition.
- `notes/annular_rho_zero_first_jet_remainder_gate.md`: compatible-stencil
  factorization, support incidence, pressure-shell lemma, complete
  `O(N^4)` remainder ledger, and negative total first-jet limit.
- `scripts/annular_rho_zero_first_jet_remainder_gate_audit.py`: exact stencil
  orders, amplitude-parity replay, mixed-difference Fisher identity,
  dangerous-branch extraction, and fail-closed total-limit certificate.
- `notes/annular_rho_zero_second_jet_route_guard.md`: exact coupled
  second-variation formula, support ledger, positive double-heat pressure
  `N^7` limit, and the two nonlinear channel groups still left open.
- `scripts/annular_rho_zero_second_jet_route_guard_audit.py`: 20-channel
  Hessian/acceleration decomposition, tenfold de-aliasing replay,
  independent second difference, and sparse second-heat asymptotics.
- `notes/annular_rho_zero_inviscid_second_jet_branch.md`: compact coupled
  Euler/transport pressure identity, exact `a+a^3` branch projection,
  corrected candidate `N^9` route, and bounded-output localization.
- `scripts/annular_rho_zero_inviscid_second_jet_branch_audit.py`: eightfold
  branch-only de-aliasing, full-form replay, amplitude coefficient
  extraction, and mode-resolved pressure-output diagnostics.
- `notes/annular_rho_zero_euler_maclaurin_cusp_gate.md`: direct exact-box
  continuum rule, explicit packet-face corrections, internal Leray-cusp
  expansion, and the closed small-cube error budget.
- `scripts/annular_rho_zero_direct_continuum_quadrature.py`: resumable
  `h^11` tensor-trapezoid rows, output-sector diagnostics, energy trace,
  Richardson replay, and small-mode cusp checks.
- `scripts/annular_rho_zero_euler_maclaurin_boundary_pilot.py`: explicit
  face directional derivative, corrected packet measures, `B_4` face and
  `B_2 B_2` edge pilots, and fail-closed certification state.
- `notes/annular_two_shear_square_gate.md`: modified divergence-free high
  profile, exact two-shear matrix, static negative norm, and four-high
  negative-square reduction.
- `scripts/annular_two_shear_square_gate_audit.py`: rational low-stencil and
  strain enumeration plus exact-box component-energy replays for the
  modified witness.
- `notes/annular_two_shear_full_c1_port.md`: modified multiplier and profile
  bounds, fixed-output continuity, doubled fourteen-profile tail ledger,
  and the strict complete `c1` limit.
- `scripts/annular_two_shear_full_c1_port_audit.py`: dependency traversal,
  explicit one-difference constants, low-field linearity checks, and
  fail-closed full-limit certificate.
- `notes/annular_two_shear_static_optimizer.md`: exact two-shear low
  self-flux, complete and pressure-only static objectives, loss of the old
  finite optimizer, and corrected carrier/restart scaling.
- `scripts/annular_two_shear_static_optimizer_audit.py`: symbolic
  four-mode enumeration, complete finite-field support replay, HHL/Fisher
  rows, coefficient penalty, and fail-closed unboundedness certificate.
- `notes/annular_parallel_shear_phase_repair.md`: exact scalar-phase no-go,
  full-polarization factorization, stationary parallel-shear repair,
  reflection-protected strict signs, and restored optimizer/reset scaling.
- `scripts/annular_parallel_shear_phase_repair_audit.py`: symbolic
  phase/polarization classification, exact stencil and curvature symmetry,
  complete finite replay, and fail-closed repair certificate.
- `scripts/run_full_regression_checkpoint.py`: one-worker below-normal pytest
  runner with exact collection counting and an atomic structured closeout.
- `scripts/newtonian_boundary_audit.py`: reciprocal-gap strict-local-
  martingale and heat-kernel boundary defect.
- `scripts/two_point_vorticity_audit.py`: symbolic centre/separation and
  strain-kernel harmonic checks.
- `scripts/strain_boundary_multiplier_audit.py`: exact heat attenuation of
  the degree-two Newtonian strain kernel at the collision boundary.
- `scripts/heat_scale_cubic_cancellation_audit.py`: de-aliased spectral
  check of the first heat-scale moment cancellation in vortex stretching.
- `scripts/fourier_triad_collision_audit.py`: exact triad amplitude plane,
  heat-defect formula, and two-scale viscous sign stress test.
- `scripts/first_crossing_barrier_audit.py`: exact smooth Fourier
  counterexample to forward invariance of the fixed-scale rigidity region.
- `scripts/adaptive_scale_barrier_audit.py`: unequal-triad obstruction where
  the scale derivative vanishes at an outward-pointing threshold.
- `scripts/cumulative_defect_audit.py`: energy-level cumulative criterion,
  scaling, and semigroup primitive checks.
- `scripts/quartic_transfer_audit.py`: exact finite-mode primitive evaluator
  and resumable random search for the quartic transfer sign.
- `scripts/quartic_transfer_optimizer.py`: divergence-free finite-Galerkin
  evaluator and analytic-gradient adversarial sign search.
- `scripts/quartic_transfer_counterexample.py`: symbolic Fourier derivation
  of the exact two-mode counterexample to quartic-transfer positivity.
- `scripts/quartic_transfer_helical_audit.py`: exact same-helicity,
  opposite-helicity, and coherent-channel sign audit.
- `scripts/quartic_transfer_helical_matrix_audit.py`: full Hermitian pair
  matrix, parity block decomposition, and phase-sensitive checks.
- `scripts/ns_trajectory_defect_audit.py`: exact trajectory coefficients,
  weakly nonlinear sign reversal, and cumulative dissipation ratio.
- `scripts/galerkin_trajectory_audit.py`: locked, resumable projected
  Fourier-Galerkin trajectory sweep with exact structural diagnostics.
- `scripts/galerkin_sweep_analysis.py`: duplicate audit, convergence
  classification, and persisted sweep summary.
- `scripts/helical_trajectory_channel_audit.py`: physical rank-one
  parity/helicity decomposition along Galerkin trajectories.
- `scripts/generated_mode_transfer_audit.py`: difference mode, sum mode,
  first-shell interaction, and higher-generation transfer split.
- `scripts/weak_generated_transfer_audit.py`: exact order-four/order-six
  Duhamel law, with an optional full symbolic derivation.
- `scripts/second_normal_form_audit.py`: polarized four-frequency primitive,
  exact quintic counterexample, and two-mode support-selection audit.
- `scripts/third_normal_form_audit.py`: exact 60-tree sextic kernel,
  Gaussian-rational sign certificates, and third evolution identities.
- `scripts/normal_form_resummation_audit.py`: hierarchy telescoping, tree-count
  growth, small-data majorant, and exact `c4`/`c6` endpoint recovery.
- `scripts/collision_coherence_weight_audit.py`: regularized inverse-gap
  generator, optimized damping, affine stress test, and energy scaling gate.
- `scripts/localized_strain_tube_audit.py`: gauge transform, Kummer principal
  eigenvalue, and two-history escape-versus-stretching audit.
- `scripts/moving_strain_tube_audit.py`: moving gauge terms and the transverse
  `L^2` form-bound budget.
- `scripts/strain_tube_reentry_audit.py`: weighted exterior barrier,
  finite-three-dimensional return factors, and renewal arithmetic.
- `scripts/three_dimensional_leray_gate_audit.py`: anisotropic oscillator,
  sharp Sobolev budget, time-concentration test, and return-strain audit.
- `scripts/strain_eigenframe_geometry_audit.py`: strain eigenvalue/frame
  identities, general affine spectrum, and harmonic pressure stress tests.
- `scripts/pressure_collision_kernel_audit.py`: trace-free pressure heat
  multiplier, projected kernel bound, and first-order Fourier stress test.
- `scripts/pressure_frame_pairing_audit.py`: continuously refined strain
  maximum, direct Navier-Stokes growth, and pressure orthogonality audit.
- `scripts/pressure_shell_commutator_audit.py`: periodic boundary-identity
  check, amplitude stress test, and critical time-integrability obstruction.
- `scripts/adaptive_reynolds_envelope_audit.py`: envelope-gauge
  factorization, shrinking-cutoff sign, and `R_*=2` spectral cross-check.
- `scripts/shrinking_tube_renewal_audit.py`: moving return barrier,
  Brownian capacity optimization, and scale-uniform renewal arithmetic.
- `scripts/pressure_partition_flux_audit.py`: eight-cell full/low/defect
  pressure-flux cancellation and weighted partition checks.
- `scripts/intrinsic_radius_cover_audit.py`: nonuniform-envelope test of the
  Lipschitz radius, monotonicity, and neighboring-gauge bounds.
- `scripts/monotone_dyadic_cover_audit.py`: dynamic split-only tree test of
  safe radii, Reynolds caps, and 2:1 neighboring balance.
- `scripts/dyadic_gauge_transition_audit.py`: parent-child gauge maximum and
  contraction after a genuine envelope-driven radius halving.
- `scripts/branching_transfer_operator_audit.py`: stochastic child/interface
  operators, weighted mismatch, and multi-generation branching stress test.
- `scripts/interface_weight_no_go_audit.py`: left-kernel no-go, logarithmic
  weighted growth, and conservative physical-to-gauge cycle budget.
- `scripts/two_norm_generation_cycle_audit.py`: two-level renewal algebra,
  Reynolds/buffer sweep, and the zero-action closure threshold.
- `scripts/buffered_visit_feynman_kac_audit.py`: symbolic piecewise visit PDE,
  exact replica gain, and optimized complete-generation stress test.
- `scripts/axial_killing_buffered_visit_audit.py`: Kummer/Bessel radial
  transfer, axial OU eigenvalues, and required finite-height thresholds.
- `scripts/finite_cylinder_mode_audit.py`: full axial oscillator expansion,
  convergence checks, and centreline visit maxima.
- `scripts/finite_cylinder_perturbation_margin_audit.py`: generalized
  Kummer/Bessel transfer, exact robustness roots, and critical
  `L^(3/2)` constant-potential masses.
- `scripts/finite_cylinder_kato_gate_audit.py`: finite-cylinder Kato budgets,
  normalized endpoint concentration sequence, and supercritical Holder
  comparison.
- `scripts/gaussian_boundary_l2_transfer_audit.py`: reversible cylinder
  measure, exact axial visit spectrum, dynamic Markov `L^2` contractions,
  and density-mismatch thresholds.
- `scripts/ground_state_visit_transform_audit.py`: positive ground-state
  factorization, reversible visit kernel, spectral mixing, and refined
  density bounds.
- `scripts/axial_form_to_boundary_audit.py`: generalized axial form
  eigenvalues, concentrated-profile thresholds, and nonlinear renewal
  bounds.
- `scripts/off_diagonal_form_transfer_audit.py`: Loewner resolvent theorem,
  cross-coupling counterexample, random stress tests, and `chi` budgets.
- `scripts/weighted_cylinder_buffer_condition_audit.py`: reversible radial
  finite elements, internal Green blocks, angular modes, and collar margins.
- `scripts/poisson_cutoff_form_transfer_audit.py`: same-boundary resolvent
  identity, cutoff Poisson energies, non-axisymmetric stress test, and
  `L^(3/2)` budget conversion.
- `scripts/quadratic_partition_ims_budget_audit.py`: tensor quadratic
  partition identities, IMS spectral costs, minimum overlap widths, and
  post-localization mass budgets.
- `scripts/radial_cubic_partition_audit.py`: exact cardinal-cubic partition
  identities, radial support geometry, collar optimization, and residual
  critical-mass budget.
- `scripts/cubic_level_transfer_audit.py`: conservative cubic child kernel,
  replica branching, recentering maximum, and true-level split contraction.
- `scripts/navier_stokes_coherence_budget_audit.py`: effective-error
  decomposition, exact support integrals, Campanato thresholds, and
  translation-invariance stress test.
- `scripts/general_affine_spectral_floor_audit.py`: one-parameter affine
  spectrum reduction, disk Bessel floor, and uniform post-IMS form budget.
- `scripts/anisotropic_poisson_transfer_pilot.py`: weighted annular
  finite-element Poisson map for the anisotropic reversible shell model.
- `scripts/reversible_shell_rigidity_audit.py`: symbolic harmonic-continuation
  rigidity and interface-divergence checks.
- `scripts/divergence_free_shell_taper_audit.py`: exact divergence/strain
  identities and polynomial minimax taper sweep.
- `scripts/divergence_free_taper_transfer_pilot.py`: nonsymmetric finite-element
  boundary visit for the localized incompressible shell.
- `scripts/sectorial_poisson_transfer_audit.py`: coercive-sector perturbation
  theorem, closure algebra, and large-skew matrix stress tests.
- `scripts/moving_cubic_label_transport_audit.py`: normalized moving-label
  identities, Markov contraction, parabolic intertwining, and explicit
  translation/rotation/scale thresholds.
- `scripts/continuous_cubic_localization_no_go_audit.py`: symbolic cubic
  Fisher lower bound and the current visit-form coercivity obstruction.
- `scripts/stopping_time_moving_visit_audit.py`: moving-cylinder coordinate
  transform, critical scaling, stopping-time Markov transfer, and closure
  arithmetic.
- `scripts/coherence_abort_renewal_audit.py`: restart-resolvent bounds and
  calibrated split-paid and probability-paid abort budgets.
- `scripts/leray_mollified_cell_frame_audit.py`: convolution identities,
  distributional affine fit, Galilean test, and spin-frame preservation.
- `scripts/affine_taper_residual_no_go_audit.py`: exact affine collar
  mismatch, compact full-affine replacement, and calibrated visit budgets.
- `scripts/compact_affine_campanato_gate_audit.py`: explicit compact
  mollifier, exact affine remainder, critical thresholds, and the
  constant-spectrum rotation obstruction.
- `scripts/leray_conditional_occupation_no_go_audit.py`: parabolic
  concentration scaling and exact conditional cube-survival obstruction.
- `scripts/nonautonomous_full_affine_form_audit.py`: instantaneous affine
  reference, reduced physical error, and uniform nonautonomous form floor.
- `scripts/rotating_affine_visit_pilot.py`: nonsymmetric finite-element
  rotation sweep and exact-step tilting-strain Monte Carlo stress test.
- `scripts/weighted_kernel_dynamic_l2_audit.py`: exact square-tilted dynamic
  boundary law, positive-kernel norm identity, and replica-pair lift.
- `scripts/nonautonomous_scalar_gain_gate_audit.py`: arbitrary-history volume
  gain theorem, stationary surface-sector estimate, switched-strain
  counterexample pilot, and covariance/Nash no-go.
- `scripts/radial_payoff_bellman_pilot.py`: exact affine-control Hamiltonian
  and axisymmetric policy-iteration stress test for the true radial payoff.
- `scripts/radial_payoff_supersolution_audit.py`: explicit rational
  boundary-layer candidate and dense residual/squared-margin audit.
- `scripts/radial_payoff_interval_certificate.py`: symbolic derivative
  cross-check, outward-rounded interior/collar subdivision, asymptotic strip
  proof, and certified ideal dynamic-cycle closure.
- `scripts/radial_bellman_doob_perturbation_audit.py`: transformed-generator
  identity, weighted divergence-free cancellation, and remaining margins.
- `scripts/critical_collar_transfer_audit.py`: first-insertion affine
  Gaussian constants, critical endpoint sequence, and conditional
  collar-to-form sector budgets.
- `scripts/radial_barrier_cutoff_energy_pilot.py`: variational minimum-energy
  cutoff calibration for protected radial/axial cores.
- `scripts/radial_collar_trace_pilot.py`: complete discrete stationary
  interface-to-entry evaluation norm over the axisymmetric affine family.
- `scripts/radial_collar_frequency_pilot.py`: time-harmonic collar stress
  test and nonzero-frequency resonance diagnostic.
- `scripts/protected_collar_partition_no_go_audit.py`: protected-support
  cubic spacing and IMS no-go arithmetic.
- `scripts/interaction_marked_localization_audit.py`: finite-dimensional
  label split, direct-sum compression, and Dyson cancellation stress tests.
- `scripts/radial_h1_payoff_supersolution_pilot.py`: rational finite-energy
  candidate, dense HJB stress grid, norm quadrature, and renewal margins.
- `scripts/radial_h1_payoff_interval_certificate.py`: outward-rounded
  finite-region subdivision and weighted two-case open-corner certificate.
- `scripts/averaged_entry_trace_gate.py`: surface trace, time-averaged energy,
  unnormalized return-law composition, and density-budget audit.
- `scripts/exterior_return_tail_gate.py`: polynomial-tail no-go, general
  interval-envelope theorem, and exact half-space calibrations.
- `scripts/cylindrical_brownian_return_pilot.py`: Bessel-mode inversion,
  spatial-L2 stress envelope, and finite axial-patch return quadrature.
- `scripts/branch_resolved_entry_renewal_audit.py`: branch-separated Neumann
  renewal identity, error allowances, and no-double-counting checks.
- `scripts/split_entry_density_inheritance_audit.py`: density-preserving
  label split, temporal-atom counterexample, and volume-density budgets.
- `scripts/affine_exterior_axial_compensation_audit.py`: exact affine
  separation, radial spectral roots, and weighted-L2 inversion pilot.
- `scripts/anisotropic_affine_exterior_tail_gate.py`: trace-free affine
  endpoint balance and explicit zero-gap Weyl sequence.
- `scripts/neutral_strip_storage_gate.py`: uniform neutral-strip spectral
  margin and physical branch scope audit.
- `scripts/neutral_strip_branch_resolvent_pilot.py`: boundary-fitted return and
  wall resolvent pilot for the static neutral affine family.
- `scripts/neutral_strip_axial_patch_branch_pilot.py`: implicit semigroup
  integration of the exact finite axial-patch branch payoff.
- `scripts/geometric_wall_split_compatibility_audit.py`: exact wall-event,
  true-split, and child-capture compatibility checks.
- `scripts/neutral_strip_same_scale_width_sweep.py`: same-scale wall criterion
  sweep with mesh, timestep, and truncation stresses.
- `scripts/geometry_triggered_migrating_child_pilot.py`: direct-capture
  geometry, inward/migrating branch comparison, and transfer-mismatch audit.
- `scripts/migrating_core_residual_budget_audit.py`: symbolic moving-core
  split and sharp multiplicative/additive residual budget audit.
- `scripts/wall_stopping_trace_composition_audit.py`: exact weighted-kernel
  response identity and branch-resolved trace-law audit.
- `scripts/neutral_strip_return_density_pilot.py`: time-resolved inner flux,
  axial-OU density composition, and return-response pilot.
- `scripts/neutral_strip_boundary_density_discretization_audit.py`: angular
  bin-refinement no-go and detailed-balance cycle audit for the fitted-edge
  law.
- `scripts/neutral_strip_reversible_boundary_fem_pilot.py`: constrained-
  Delaunay lumped-FEM generator and physical boundary-flux refinement pilot.
- `scripts/neutral_strip_reversible_spectral_tail_width_audit.py`: symmetric
  generator decay enclosure, boundary-operator bound, and axial-width sweep.
- `scripts/neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py`:
  resumable componentwise directed-Decimal LDL prefix certification.
- `scripts/neutral_strip_weighted_hypercircle_directed_ldl_precision_crosscheck.py`:
  hash-bound pivot and lower-interval nesting replay.
- `scripts/neutral_strip_weighted_hypercircle_congruence_residual_pilot.py`:
  independent congruence-residual inertia certificate without interval pivot
  divisions.
- `scripts/neutral_strip_weighted_hypercircle_congruence_residual_precision_crosscheck.py`:
  higher-precision nesting replay of every residual upper bound.
- `scripts/neutral_strip_weighted_hypercircle_symbolic_transition_map.py`:
  exact full-order potential-fill and recurrence-work map.
- `scripts/neutral_strip_weighted_hypercircle_full_feasibility_audit.py`:
  storage, kernel-work, launch-readiness, and continuum-dependency audit.
- `scripts/neutral_strip_weighted_hypercircle_transition33280_audit.py`:
  hash-bound incremental risk and transition-class audit.
- `scripts/neutral_strip_weighted_hypercircle_standalone_residual.py`:
  directed-independent, hash-bound congruence-residual certificate.
- `scripts/neutral_strip_weighted_hypercircle_standalone_residual_regression.py`:
  exact replay against the historical 2,304, 32,064, and 33,280 results.
- `scripts/neutral_strip_weighted_hypercircle_state_entry63680_audit.py`:
  block-resolved first-state-entry and inverse-growth audit.
- `scripts/neutral_strip_weighted_hypercircle_state_region64064_audit.py`:
  independent factor-compatibility, transition, and inverse-majorant
  obstruction audit.
- `scripts/neutral_strip_weighted_hypercircle_closure_boundary64039_audit.py`:
  adjacent pass/fail reconstruction and one-pivot recurrence localization.
- `scripts/neutral_strip_weighted_hypercircle_componentwise_residual.py`:
  direct `Q R Q^T 1` transformed-residual spectral certificate.
- `scripts/neutral_strip_weighted_hypercircle_componentwise_state_region64064_audit.py`:
  precision/provenance audit of the componentwise state-region recovery.
- `scripts/neutral_strip_weighted_hypercircle_componentwise_growth64128_audit.py`:
  exact source/factor nesting and componentwise-growth audit through 64,128.
- `scripts/wall_migration_child_return_density_pilot.py`: resolved wall
  migration followed by one child-return density and composite K_S pilot.
- `results/quartic_transfer_sweep_summary.json`: persisted 480-field sign
  sweep summary; historical empirical evidence that missed the sparse
  counterexample.
- `tests/test_collision_identities.py`: regression tests for the symbolic,
  trajectory, helical, and generated-mode audits.

## Commands

```text
python work/ns_collision/scripts/verify_collision_identities.py
python work/ns_collision/scripts/correlated_replica_tangent_gramian_audit.py --output work/ns_collision/results/correlated_replica_tangent_gramian_audit_v1.json
python -m unittest work/ns_collision/tests/test_correlated_replica_tangent_gramian.py
python work/ns_collision/scripts/signed_projected_replica_generator_audit.py --output work/ns_collision/results/signed_projected_replica_generator_audit_v1.json
python -m unittest work/ns_collision/tests/test_signed_projected_replica_generator.py
python work/ns_collision/scripts/adjoint_replica_pressure_edge_gate_audit.py --output work/ns_collision/results/adjoint_replica_pressure_edge_gate_audit_v1.json
python -m unittest work/ns_collision/tests/test_adjoint_replica_pressure_edge_gate.py
python work/ns_collision/scripts/scale_adapted_edge_rho_expansion_audit.py --output work/ns_collision/results/scale_adapted_edge_rho_expansion_audit_v1.json
python -m unittest work/ns_collision/tests/test_scale_adapted_edge_rho_expansion.py
python work/ns_collision/scripts/gram_boundary_audit.py
python work/ns_collision/scripts/backward_replica_audit.py
python work/ns_collision/scripts/newtonian_boundary_audit.py
python work/ns_collision/scripts/two_point_vorticity_audit.py
python work/ns_collision/scripts/strain_boundary_multiplier_audit.py
python work/ns_collision/scripts/heat_scale_cubic_cancellation_audit.py
python work/ns_collision/scripts/fourier_triad_collision_audit.py
python work/ns_collision/scripts/first_crossing_barrier_audit.py
python work/ns_collision/scripts/adaptive_scale_barrier_audit.py
python work/ns_collision/scripts/cumulative_defect_audit.py
python work/ns_collision/scripts/quartic_transfer_audit.py --samples 8
python work/ns_collision/scripts/quartic_transfer_optimizer.py --family triad
python work/ns_collision/scripts/quartic_transfer_counterexample.py
python work/ns_collision/scripts/quartic_transfer_helical_audit.py
python work/ns_collision/scripts/quartic_transfer_helical_matrix_audit.py
python work/ns_collision/scripts/ns_trajectory_defect_audit.py
python work/ns_collision/scripts/galerkin_trajectory_audit.py --modes 2,3 \
  --reynolds 0.25,0.5,1,2,4 \
  --output work/ns_collision/results/galerkin_trajectory_sweep.jsonl
python work/ns_collision/scripts/galerkin_sweep_analysis.py \
  --input work/ns_collision/results/galerkin_trajectory_sweep.jsonl \
  --output work/ns_collision/results/galerkin_trajectory_summary.json
python work/ns_collision/scripts/helical_trajectory_channel_audit.py \
  --output work/ns_collision/results/helical_trajectory_channels.jsonl
python work/ns_collision/scripts/generated_mode_transfer_audit.py \
  --reynolds 0.922 \
  --output work/ns_collision/results/generated_mode_transfer_R0922.json
python work/ns_collision/scripts/weak_generated_transfer_audit.py
python work/ns_collision/scripts/second_normal_form_audit.py
python work/ns_collision/scripts/third_normal_form_audit.py
python work/ns_collision/scripts/normal_form_resummation_audit.py
python work/ns_collision/scripts/collision_coherence_weight_audit.py
python work/ns_collision/scripts/localized_strain_tube_audit.py
python work/ns_collision/scripts/moving_strain_tube_audit.py
python work/ns_collision/scripts/strain_tube_reentry_audit.py
python work/ns_collision/scripts/three_dimensional_leray_gate_audit.py
python work/ns_collision/scripts/strain_eigenframe_geometry_audit.py
python work/ns_collision/scripts/pressure_collision_kernel_audit.py
python work/ns_collision/scripts/pressure_frame_pairing_audit.py
python work/ns_collision/scripts/pressure_shell_commutator_audit.py
python work/ns_collision/scripts/pressure_hamming_commutator_gate_audit.py
python work/ns_collision/scripts/annular_vertex_commutator_gate_audit.py
python work/ns_collision/scripts/self_shell_pressure_closure_audit.py
python work/ns_collision/scripts/cross_shell_modulated_wave_gate_audit.py
python work/ns_collision/scripts/dyadic_three_shell_atlas_audit.py
python work/ns_collision/scripts/joint_scale_cell_viscous_occupation_audit.py
python work/ns_collision/scripts/nonlinear_stress_regeneration_gate_audit.py
python work/ns_collision/scripts/dense_annular_hhh_packet_gate_audit.py
python work/ns_collision/scripts/scalar_local_energy_regeneration_gate_audit.py
python work/ns_collision/scripts/dense_spaced_continuum_positivity_audit.py
python work/ns_collision/scripts/smooth_galerkin_shell_response_gate_audit.py
python work/ns_collision/scripts/scale_uniform_low_output_tail_gate_audit.py
python work/ns_collision/scripts/direct_h_minus_one_stress_tail_gate_audit.py
python work/ns_collision/scripts/dense_low_output_block_gate_audit.py
python work/ns_collision/scripts/floor_free_pressure_edge_tail_gate_audit.py
python work/ns_collision/scripts/balanced_annular_pressure_edge_gate_audit.py
python work/ns_collision/scripts/multiband_weighted_fisher_recombination_no_go_audit.py
python work/ns_collision/scripts/pressure_active_fisher_null_compatibility_gate_audit.py
python work/ns_collision/scripts/primitive_hhl_chain_hardy_envelope_audit.py
python work/ns_collision/scripts/joint_primitive_hhl_incidence_schur_gate_audit.py
python work/ns_collision/scripts/separable_annular_pressure_schur_no_go_audit.py
python work/ns_collision/scripts/annular_eight_vertex_heat_window_gate_audit.py
python work/ns_collision/scripts/compatible_edge_annular_escape_audit.py
python work/ns_collision/scripts/deficit_retaining_annular_restart_gate_audit.py
python work/ns_collision/scripts/annular_rho_zero_first_jet_audit.py
python work/ns_collision/scripts/annular_rho_zero_first_jet_remainder_gate_audit.py
python work/ns_collision/scripts/annular_rho_zero_second_jet_route_guard_audit.py
python work/ns_collision/scripts/annular_rho_zero_inviscid_second_jet_branch_audit.py
python work/ns_collision/scripts/annular_rho_zero_fixed_output_continuum_gate_audit.py
python work/ns_collision/scripts/annular_rho_zero_continuum_convolution_quadrature.py --sizes 9,13,17,21,25,29,33,37,41,45,49,53,57,61,65
python work/ns_collision/scripts/annular_rho_zero_direct_continuum_quadrature.py --sizes 8,16,32,64
python work/ns_collision/scripts/annular_rho_zero_euler_maclaurin_boundary_pilot.py --sizes 8,16,32,64
python work/ns_collision/scripts/annular_two_shear_square_gate_audit.py --sizes 8,16,32
python work/ns_collision/scripts/annular_two_shear_full_c1_port_audit.py
python work/ns_collision/scripts/annular_two_shear_static_optimizer_audit.py
python work/ns_collision/scripts/annular_parallel_shear_phase_repair_audit.py
python work/ns_collision/scripts/annular_parallel_shear_finite_jet_port_audit.py
python work/ns_collision/scripts/annular_parallel_shear_euler_transport_fisher_exclusion_audit.py
python work/ns_collision/scripts/annular_parallel_shear_heat_block_exclusion_audit.py
python work/ns_collision/scripts/annular_parallel_shear_third_jet_route_guard_audit.py
python work/ns_collision/scripts/annular_parallel_shear_third_internal_shell_lemma_audit.py
python work/ns_collision/scripts/run_full_regression_checkpoint.py --expected-count 519
python work/ns_collision/scripts/adaptive_reynolds_envelope_audit.py
python work/ns_collision/scripts/shrinking_tube_renewal_audit.py
python work/ns_collision/scripts/pressure_partition_flux_audit.py
python work/ns_collision/scripts/intrinsic_radius_cover_audit.py
python work/ns_collision/scripts/monotone_dyadic_cover_audit.py
python work/ns_collision/scripts/dyadic_gauge_transition_audit.py
python work/ns_collision/scripts/branching_transfer_operator_audit.py
python work/ns_collision/scripts/interface_weight_no_go_audit.py
python work/ns_collision/scripts/two_norm_generation_cycle_audit.py
python work/ns_collision/scripts/buffered_visit_feynman_kac_audit.py
python work/ns_collision/scripts/axial_killing_buffered_visit_audit.py
python work/ns_collision/scripts/finite_cylinder_mode_audit.py
python work/ns_collision/scripts/finite_cylinder_perturbation_margin_audit.py
python work/ns_collision/scripts/finite_cylinder_kato_gate_audit.py
python work/ns_collision/scripts/gaussian_boundary_l2_transfer_audit.py
python work/ns_collision/scripts/ground_state_visit_transform_audit.py
python work/ns_collision/scripts/axial_form_to_boundary_audit.py
python work/ns_collision/scripts/off_diagonal_form_transfer_audit.py
python work/ns_collision/scripts/weighted_cylinder_buffer_condition_audit.py
python work/ns_collision/scripts/poisson_cutoff_form_transfer_audit.py
python work/ns_collision/scripts/quadratic_partition_ims_budget_audit.py
python work/ns_collision/scripts/radial_cubic_partition_audit.py
python work/ns_collision/scripts/cubic_level_transfer_audit.py
python work/ns_collision/scripts/navier_stokes_coherence_budget_audit.py
python work/ns_collision/scripts/general_affine_spectral_floor_audit.py
python work/ns_collision/scripts/anisotropic_poisson_transfer_pilot.py
python work/ns_collision/scripts/reversible_shell_rigidity_audit.py
python work/ns_collision/scripts/divergence_free_shell_taper_audit.py
python work/ns_collision/scripts/divergence_free_taper_transfer_pilot.py
python work/ns_collision/scripts/sectorial_poisson_transfer_audit.py
python work/ns_collision/scripts/moving_cubic_label_transport_audit.py
python work/ns_collision/scripts/continuous_cubic_localization_no_go_audit.py
python work/ns_collision/scripts/stopping_time_moving_visit_audit.py
python work/ns_collision/scripts/coherence_abort_renewal_audit.py
python work/ns_collision/scripts/leray_mollified_cell_frame_audit.py
python work/ns_collision/scripts/affine_taper_residual_no_go_audit.py
python work/ns_collision/scripts/compact_affine_campanato_gate_audit.py
python work/ns_collision/scripts/leray_conditional_occupation_no_go_audit.py
python work/ns_collision/scripts/nonautonomous_full_affine_form_audit.py
python work/ns_collision/scripts/rotating_affine_visit_pilot.py
python work/ns_collision/scripts/weighted_kernel_dynamic_l2_audit.py
python work/ns_collision/scripts/nonautonomous_scalar_gain_gate_audit.py
python work/ns_collision/scripts/radial_payoff_bellman_pilot.py
python work/ns_collision/scripts/radial_payoff_supersolution_audit.py
python work/ns_collision/scripts/radial_payoff_interval_certificate.py
python work/ns_collision/scripts/radial_bellman_doob_perturbation_audit.py
python work/ns_collision/scripts/critical_collar_transfer_audit.py
python work/ns_collision/scripts/radial_barrier_cutoff_energy_pilot.py
python work/ns_collision/scripts/radial_collar_trace_pilot.py
python work/ns_collision/scripts/radial_collar_frequency_pilot.py
python work/ns_collision/scripts/protected_collar_partition_no_go_audit.py
python work/ns_collision/scripts/interaction_marked_localization_audit.py
python work/ns_collision/scripts/radial_h1_payoff_supersolution_pilot.py
python work/ns_collision/scripts/radial_h1_payoff_interval_certificate.py
python work/ns_collision/scripts/averaged_entry_trace_gate.py
python work/ns_collision/scripts/exterior_return_tail_gate.py
python work/ns_collision/scripts/cylindrical_brownian_return_pilot.py
python work/ns_collision/scripts/branch_resolved_entry_renewal_audit.py
python work/ns_collision/scripts/split_entry_density_inheritance_audit.py
python work/ns_collision/scripts/affine_exterior_axial_compensation_audit.py
python work/ns_collision/scripts/anisotropic_affine_exterior_tail_gate.py
python work/ns_collision/scripts/neutral_strip_storage_gate.py
python work/ns_collision/scripts/neutral_strip_branch_resolvent_pilot.py
python work/ns_collision/scripts/neutral_strip_axial_patch_branch_pilot.py
python work/ns_collision/scripts/geometric_wall_split_compatibility_audit.py
python work/ns_collision/scripts/neutral_strip_same_scale_width_sweep.py
python work/ns_collision/scripts/geometry_triggered_migrating_child_pilot.py
python work/ns_collision/scripts/migrating_core_residual_budget_audit.py
python work/ns_collision/scripts/wall_stopping_trace_composition_audit.py
python work/ns_collision/scripts/neutral_strip_return_density_pilot.py
python work/ns_collision/scripts/neutral_strip_boundary_density_discretization_audit.py
python -m pip install -r work/ns_collision/requirements-reversible-fem.txt
python work/ns_collision/scripts/neutral_strip_reversible_boundary_fem_pilot.py
python work/ns_collision/scripts/neutral_strip_reversible_spectral_tail_width_audit.py
python work/ns_collision/scripts/neutral_strip_reversible_finite_time_certificate.py
python work/ns_collision/scripts/neutral_strip_x_exit_correction_audit.py
python work/ns_collision/scripts/neutral_strip_reversible_fem_consistency_gate.py
python work/ns_collision/scripts/neutral_strip_parabolic_spectral_split_audit.py
python work/ns_collision/scripts/neutral_strip_first_window_brownian_majorant_audit.py
python work/ns_collision/scripts/neutral_strip_first_window_maximum_bridge_certificate.py
python work/ns_collision/scripts/neutral_strip_transient_conormal_low_block_gate.py
python work/ns_collision/scripts/neutral_strip_common_circle_sparse_inertia_audit.py --decimal-precision 220 --output work/ns_collision/results/neutral_strip_h006_q12_k240_sparse_inertia_audit_v1.json
python work/ns_collision/scripts/neutral_strip_sparse_inertia_precision_crosscheck.py --output work/ns_collision/results/neutral_strip_h006_q12_k240_sparse_inertia_precision_crosscheck_v1.json
python work/ns_collision/scripts/neutral_strip_exact_polygon_indexed_spectrum_transfer.py --output work/ns_collision/results/neutral_strip_h006_exact_polygon_indexed_spectrum_transfer_v1.json
python work/ns_collision/scripts/neutral_strip_continuum_ritz_dependency_audit.py --output work/ns_collision/results/neutral_strip_h006_continuum_ritz_dependency_audit_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_pilot.py --spacing 0.06 --output work/ns_collision/results/neutral_strip_h006_weighted_hypercircle_pilot_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_sparse_inertia_pilot.py
python work/ns_collision/scripts/neutral_strip_positive_exponential_rt_interval_pilot.py
python work/ns_collision/scripts/neutral_strip_positive_exponential_complete_assembly.py
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_central_factorization_audit.py
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py --maximum-pivots 32064 --precisions 50,80,120 --checkpoint-batch 512 --checkpoint work/ns_collision/results/neutral_strip_h006_hypercircle_directed_ldl_transition32064_checkpoint_v1.json --output work/ns_collision/results/neutral_strip_h006_hypercircle_directed_ldl_transition32064_audit_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py --maximum-pivots 32064 --precisions 80 --checkpoint-batch 512 --checkpoint work/ns_collision/results/neutral_strip_h006_hypercircle_directed_ldl_transition32064_p80_checkpoint_v1.json --output work/ns_collision/results/neutral_strip_h006_hypercircle_directed_ldl_transition32064_p80_audit_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_directed_ldl_precision_crosscheck.py --lower-audit work/ns_collision/results/neutral_strip_h006_hypercircle_directed_ldl_transition32064_audit_v1.json --lower-checkpoint work/ns_collision/results/neutral_strip_h006_hypercircle_directed_ldl_transition32064_checkpoint_v1.json --higher-audit work/ns_collision/results/neutral_strip_h006_hypercircle_directed_ldl_transition32064_p80_audit_v1.json --higher-checkpoint work/ns_collision/results/neutral_strip_h006_hypercircle_directed_ldl_transition32064_p80_checkpoint_v1.json --lower-precision 50 --higher-precision 80 --label transition32064 --output work/ns_collision/results/neutral_strip_h006_hypercircle_directed_ldl_transition32064_precision_crosscheck_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_congruence_residual_pilot.py --directed-audit work/ns_collision/results/neutral_strip_h006_hypercircle_directed_ldl_transition32064_audit_v1.json --maximum-pivots 32064 --decimal-precision 60 --output work/ns_collision/results/neutral_strip_h006_hypercircle_congruence_residual_pilot32064_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_congruence_residual_pilot.py --directed-audit work/ns_collision/results/neutral_strip_h006_hypercircle_directed_ldl_transition32064_audit_v1.json --maximum-pivots 32064 --decimal-precision 100 --output work/ns_collision/results/neutral_strip_h006_hypercircle_congruence_residual_pilot32064_p100_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_congruence_residual_precision_crosscheck.py --lower work/ns_collision/results/neutral_strip_h006_hypercircle_congruence_residual_pilot32064_v1.json --higher work/ns_collision/results/neutral_strip_h006_hypercircle_congruence_residual_pilot32064_p100_v1.json --lower-precision 60 --higher-precision 100 --output work/ns_collision/results/neutral_strip_h006_hypercircle_congruence_residual_precision_crosscheck32064_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_symbolic_transition_map.py --maximum-pivots 123816 --checkpoints 32768,49152,65536,81920,98304,114688,123816 --prior-scan-pivots 32768 --output work/ns_collision/results/neutral_strip_h006_hypercircle_symbolic_transition_map123816_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_full_feasibility_audit.py
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_transition33280_audit.py
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_standalone_residual_regression.py
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_standalone_residual.py --maximum-pivots 63680 --decimal-precision 60 --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual63680_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_standalone_residual.py --maximum-pivots 63680 --decimal-precision 100 --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual63680_p100_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_congruence_residual_precision_crosscheck.py --lower work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual63680_v1.json --higher work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual63680_p100_v1.json --lower-precision 60 --higher-precision 100 --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual_precision_crosscheck63680_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_state_entry63680_audit.py
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_standalone_residual.py --maximum-pivots 64064 --decimal-precision 60 --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual64064_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_standalone_residual.py --maximum-pivots 64064 --decimal-precision 100 --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual64064_p100_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_congruence_residual_precision_crosscheck.py --lower work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual64064_v1.json --higher work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual64064_p100_v1.json --lower-precision 60 --higher-precision 100 --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual_precision_crosscheck64064_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_state_region64064_audit.py
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_closure_boundary64039_audit.py
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_componentwise_residual.py --maximum-pivots 64040 --decimal-precision 60 --separated-result work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual64040_v1.json --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_componentwise_residual64040_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_componentwise_residual.py --maximum-pivots 64040 --decimal-precision 100 --separated-result work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual64040_p100_v1.json --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_componentwise_residual64040_p100_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_componentwise_residual.py --maximum-pivots 64064 --decimal-precision 60 --separated-result work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual64064_v1.json --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_componentwise_residual64064_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_componentwise_residual.py --maximum-pivots 64064 --decimal-precision 100 --separated-result work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual64064_p100_v1.json --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_componentwise_residual64064_p100_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_componentwise_state_region64064_audit.py
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_standalone_residual.py --maximum-pivots 64128 --decimal-precision 60 --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual64128_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_standalone_residual.py --maximum-pivots 64128 --decimal-precision 100 --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual64128_p100_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_componentwise_residual.py --maximum-pivots 64128 --decimal-precision 60 --separated-result work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual64128_v1.json --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_componentwise_residual64128_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_componentwise_residual.py --maximum-pivots 64128 --decimal-precision 100 --separated-result work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_residual64128_p100_v1.json --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_componentwise_residual64128_p100_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_congruence_residual_precision_crosscheck.py --lower work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_componentwise_residual64128_v1.json --higher work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_componentwise_residual64128_p100_v1.json --output work/ns_collision/results/neutral_strip_h006_hypercircle_standalone_componentwise_residual_precision_crosscheck64128_v1.json
python work/ns_collision/scripts/neutral_strip_weighted_hypercircle_componentwise_growth64128_audit.py
python work/ns_collision/scripts/wall_migration_child_return_density_pilot.py
python -m unittest work/ns_collision/tests/test_collision_identities.py
```
