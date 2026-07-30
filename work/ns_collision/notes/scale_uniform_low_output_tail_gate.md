# Scale-uniform low-output stress-tail gate

Supersession notice: the scalar-envelope calculation and saturated pulse in
this note remain valid, but the inference that the actual Reynolds-stress
endpoint is open is superseded by
`direct_h_minus_one_stress_tail_gate.md`. Restoring the physical-space product
structure proves the actual comparable high-high stress tail vanishes in
`L2_t H_x^(-1)`.

Status: the fixed-channel smooth Galerkin response theorem extends to the
complete low-output Fourier family. After paying the three-dimensional
output multiplicity, the high-high Reynolds-stress tail tends to zero in
`L2_t H_x^(-s)` for every `s>1`, uniformly over Galerkin truncations with
common Leray bounds. Standard fixed-shell compactness therefore passes this
stress series through a Galerkin limit in `H^(-1-epsilon)`.

The endpoint `H^(-1)` is not closed. An exact channel-saturated parabolic
pulse keeps a nonzero `H^(-1)` response while satisfying the sharp scalar
forcing envelope. This proves that the endpoint cannot follow from that
envelope alone. It is not a Navier-Stokes solution or counterexample.

## 1. Low-output stress series

Let `C_H` be the smooth pair-shell Reynolds stress from the exact response
gate. With a smooth output cutoff, define

```text
R_K^lo
 =sum_(H>=K) P_(|q|_infinity<=H/8) C_H.             (1.1)
```

The constants below absorb the finite overlap of both multiplier families.
This max-norm output cutoff is contained in the Euclidean `H/4` region
required by the preceding HHL theorem.
Write

```text
A=E(0)/sqrt(nu),

B=E_*sqrt(D)/nu,                                   (1.2)

E_*=sup_t ||u(t)||_2^2,

D=integral ||grad u(t)||_2^2 dt.
```

For a Fourier output `q`, set

```text
J_q=max(K,8<q>_infinity),

<q>_infinity=max(1,|q|_infinity).                  (1.3)
```

The preceding exact pair-rate proof is uniform in `q`. Leray projectors and
the low-output double Riesz multiplier have operator norm at most one, the
pair selectors have uniform rescaled derivative bounds, and the HHL
commutator depends only on the inequalities `|q|,|c|<=L`. No count of
low-output channels entered that proof.

Consequently, applying the fixed-channel theorem beginning at `J_q` gives

```text
||Rhat_K^lo(q)||_(L2_t)

 <=C[A/J_q+B/sqrt(J_q)].                            (1.4)
```

The first term is the initial high stress. The second is the complete HHH,
paired HHL, filter-commutator, and sharp Galerkin-boundary response. The
constant is independent of `q`, `K`, and the Galerkin cutoff.

## 2. Littlewood-Paley channel count

Use max-norm dyadic output blocks

```text
Omega_Q={q in Z^3: Q<=|q|_infinity<2Q}.             (2.1)
```

Their exact cardinality is

```text
N_Q
 =(4Q-1)^3-(2Q-1)^3

 =56Q^3-36Q^2+6Q

 <=56Q^3.                                          (2.2)
```

Euclidean Littlewood-Paley blocks have the same `Q^3` bound. Parseval and
(1.4) therefore give

```text
||Delta_Q R_K^lo||_(L2_t L2_x)^2

 <=C Q^3[
          A^2/max(K,Q)^2
          +B^2/max(K,Q)
        ].                                         (2.3)
```

The mixed term is absorbed by `(x+y)^2<=2x^2+2y^2`.
The zero Fourier mode is one additional channel and is bounded directly by
(1.4); it does not alter any dyadic exponent.

The factor `Q^(3/2)` in the unsquared norm is not an invented loss. It is the
square root of the number of low Fourier outputs. A dense high-frequency
autocorrelation can fill a positive fraction of those outputs.

## 3. Scale-uniform Sobolev theorem

For the inhomogeneous `H^(-s)` norm, (2.3) reduces to two dyadic series:

```text
I_s(K)
 =sum_Q Q^(3-2s)/max(K,Q)^2,                        (3.1)

F_s(K)
 =sum_Q Q^(3-2s)/max(K,Q).                          (3.2)
```

Splitting at `Q=K` and summing the two geometric tails gives, for `s>1`,

```text
I_s(K)<=C_s *
  {
    K^(1-2s),          1<s<3/2,
    log(2K)/K^2,       s=3/2,
    K^(-2),            s>3/2,
  }                                                     (3.3)

F_s(K)<=C_s *
  {
    K^(2-2s),          1<s<3/2,
    log(2K)/K,         s=3/2,
    K^(-1),            s>3/2.
  }                                                     (3.4)
```

Thus

```text
||R_K^lo||_(L2_t H_x^(-s))

 <=C_s[A sqrt(I_s(K))+B sqrt(F_s(K))] ->0           (3.5)
```

for every `s>1`.

Equivalently, with `s=1+epsilon`:

```text
0<epsilon<1/2:

 ||R_K^lo||_(L2_t H^(-1-epsilon))

 <=C_epsilon[
      A K^(-1/2-epsilon)
      +B K^(-epsilon)
    ],                                             (3.6)

epsilon=1/2:

 <=C sqrt(log(2K))[
      A/K+B/sqrt(K)
    ],                                             (3.7)

epsilon>1/2:

 <=C_epsilon[A/K+B/sqrt(K)].                        (3.8)
```

This is the first estimate in the project that sums every low-output Fourier
channel with constants independent of the output scale.

## 4. Galerkin stability

Consider smooth Fourier Galerkin solutions with common Leray bounds. For a
fixed wave `k`, incompressibility gives

```text
uhat(a) dot (k-a)=uhat(a) dot k.
```

Therefore Fourier Cauchy-Schwarz and the contraction property of the Leray
projector give

```text
|Nhat(k)|
 <=|k| sum_a |uhat(a)||uhat(k-a)|
 <=|k| E_*.                                        (4.1)
```

The mode equation then bounds `|partial_t uhat(k)|` uniformly in the
Galerkin cutoff. Every fixed finite Fourier family is uniformly bounded and
equi-Lipschitz, so Arzela-Ascoli gives strong convergence after a diagonal
subsequence. The uniform `L_infinity_t` energy bound consequently gives
strong `L2_t` convergence of every fixed-shell quadratic stress
coefficient.

Given `delta>0`, choose `K` so that the right side of (3.5) is below
`delta`, uniformly in the Galerkin cutoff. The stress below `K` is
finite-dimensional and converges. The two high tails cost at most
`2delta`. This diagonal argument proves:

> For every `s>1`, the complete low-output high-high stress series is
> Galerkin-stable in `L2_t H_x^(-s)`.

Sharp Galerkin top-boundary leakage is already included in `B`; no smooth
approximation of the Galerkin projector is being assumed here.

This passes one quadratic stress component. It does not yet pass every
cubic term in the local-energy inequality.

## 5. Exact endpoint thought experiment

At `s=1`, the high-output part of (3.2) becomes

```text
sum_(Q>K) Q^(3-2)/Q
 =sum_(Q>K) 1,                                    (5.1)
```

over dyadic `Q`. The estimate is not uniform in the Galerkin cutoff. To
check whether this is only poor bookkeeping, use the exact forced
relaxation model below.

Fix one carrier `H`, put `Q=H/16`, and assign every Fourier channel in
`Omega_Q` the same scalar equation

```text
dot c_q+H^2 c_q
 =H^(5/2) 1_[0,H^(-2)](t),

c_q(0)=0.                                          (5.2)
```

The sharp forcing envelope is exactly one:

```text
H^(-3)
 ||H^(5/2)1_[0,H^(-2)]||_(L2_t)^2
 =1.                                               (5.3)
```

Solving (5.2) on the pulse and its heat tail gives the exact identity

```text
||c_q||_(L2_t)^2=1/(eH).                           (5.4)
```

Since `N_Q=56Q^3-36Q^2+6Q`,

```text
Q^(-2) sum_(q in Omega_Q)||c_q||_(L2_t)^2

 =N_Q/(eH Q^2)

 ->7/(2e).                                         (5.5)
```

The left side of (5.5) is the standard Littlewood-Paley equivalent squared
`H^(-1)` block norm. It therefore remains nonzero as `H` tends to infinity.
In `H^(-1-epsilon)`, the same expression gains `Q^(-2epsilon)` and
vanishes.

This establishes a precise envelope-level endpoint obstruction:

- viscosity and the `H^(-3/2)` shell forcing weight are exactly critical;
- three-dimensional low-output multiplicity consumes the remaining gain;
- no argument using only the scalar forcing envelope can prove `H^(-1)`
  compactness.

The model deliberately explores the mathematically possible saturated
channel array. It does not prove that an unforced Navier-Stokes trajectory
can populate every channel in (5.2) with the same sign for a full parabolic
time.

## 6. Scope and next gate

Established:

- a Fourier-channel constant uniform in the low output;
- exact three-dimensional Littlewood-Paley multiplicity;
- vanishing of the full low-output stress tail in
  `L2_t H_x^(-1-epsilon)` for every `epsilon>0`;
- stability of this stress series under smooth Galerkin limits;
- an exact endpoint pulse showing that the certified scalar envelope alone
  cannot yield `H^(-1)` compactness.

Not established:

- an actual Navier-Stokes realization of the channel-saturated pulse;
- an `H^(-1)` endpoint theorem or counterexample;
- passage of every cubic local-energy defect;
- exceptional-set removal or global regularity.

The next decisive test is the dense-output realization gate. The certified
dense HHH packet already reaches `H^(5/2)` in one low tensor channel. We
must determine whether its directed positivity persists over a
positive-volume block of low outputs. If it does, the endpoint obstruction
is structurally close to Navier-Stokes forcing. If it does not, the failure
must expose an output-space cancellation strong enough to revisit
`H^(-1)`.

The deterministic arithmetic and pulse identities are generated by
`scripts/scale_uniform_low_output_tail_gate_audit.py`.
