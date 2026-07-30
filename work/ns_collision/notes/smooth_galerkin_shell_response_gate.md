# Smooth Galerkin shell-response gate

Status: for every smooth finite Fourier Galerkin solution, the exact
low-output high-shell stress evolution can be summed without replacing its
pairwise viscous rates by an artificial shell rate. Smooth shell-selector
leakage and the heat-weighted HHL sweeping commutator retain a low
derivative. A sharp Galerkin top cutoff can break that pairing, but its
worst unpaired boundary term is paid directly by the same `H^(-3)` norm.
The complete comparable-shell HHH plus separated HHL forcing obeys the
spacetime square estimate, and both the forced Duhamel tail and
initial-stress heat tail vanish at every fixed finite family of low
Fourier/tensor channels. This is not yet a scale-uniform localization
theorem or a suitable-weak-solution result.

## 1. Exact pairwise evolution

On the three-torus, use

```text
dot uhat(k)+nu|k|^2 uhat(k)=Nhat(k).               (1.1)
```

For a fixed low output `q`, one ordered stress pair is

```text
C_(k,q)=uhat(k) tensor uhat(q-k).                  (1.2)
```

Direct differentiation gives the exact identity

```text
(d/dt+nu lambda_(k,q))C_(k,q)

 =Nhat(k) tensor uhat(q-k)
  +uhat(k) tensor Nhat(q-k),                      (1.3)

lambda_(k,q)=|k|^2+|q-k|^2.                       (1.4)
```

The rate in (1.4) depends on the pair. No single shell rate is inserted.
For a static smooth symmetric pair-shell selector `sigma_H`, Duhamel gives

```text
C_(H,q)(t)
 =sum_k sigma_H(k,q-k)
        exp[-nu lambda_(k,q)t] C_(k,q)(0)

  +integral_0^t sum_k sigma_H(k,q-k)
        exp[-nu lambda_(k,q)(t-s)]G_(k,q)(s) ds.  (1.5)
```

The Galerkin truncation makes every sum finite. The estimates below are
uniform in that truncation.

Assume the selectors have finite dyadic overlap, are supported where both
pair frequencies are comparable to `H`, and satisfy

```text
|grad sigma_H|<=C_chi/H.                          (1.6)
```

Because they are static Fourier multipliers, they commute with `d/dt` and
the Laplacian. Their only leakage is the finite neighboring-shell
assignment inside the nonlinear term.

The orthogonal Galerkin projector itself may have a sharp top-frequency
boundary. A low shift can then retain one regenerated high mode and discard
its paired partner. Section 4 includes this worst-case unpaired contribution
instead of assuming that boundary is smooth.

## 2. Exact nonlinear atlas

Expanding `G_(k,q)` in (1.3) produces three velocity frequencies whose sum
is `q`. Above the fixed low-output scale, the triangle inequality leaves
only:

```text
HHH: all three frequencies are comparable;

HHL: two comparable high frequencies and one low frequency. (2.1)
```

The comparable family includes a fixed number of neighboring dyadic
shells. Write

```text
a_J(t)=||u_J(t)||_2,

A_H(t)^2=sum_(|log_2(J/H)|<=2)a_J(t)^2.            (2.2)
```

There is no separated `HLL` block.

## 3. Heat-weighted HHL commutator

Let high waves `a,b`, low wave `c`, and low output `q` satisfy

```text
a+b+c=q,

H<=|a|,|b|<=2H,

|c|,|q|<=L,       H>=4L.                          (3.1)
```

For divergence-free polarizations, set

```text
G_1=B(a,c;U_a,U_c) odot U_b,

G_2=B(b,c;U_b,U_c) odot U_a.                      (3.2)
```

The unweighted sweeping theorem already gives

```text
||G_1+G_2||_F
 <=18L||U_a||||U_b||||U_c||.                      (3.3)
```

Exact Duhamel evolution gives the two terms different rates:

```text
lambda_1=|a+c|^2+|b|^2,

lambda_2=|b+c|^2+|a|^2.                           (3.4)
```

Introduce the common reference

```text
lambda_0=|a|^2+|b|^2.                             (3.5)
```

The rate differences are exact:

```text
lambda_1-lambda_0=2a dot c+|c|^2,

lambda_2-lambda_0=2b dot c+|c|^2,

lambda_1-lambda_2=2c dot (a-b).                   (3.6)
```

Thus they are `O(HL)`, while all three rates are bounded below by a
constant times `H^2`.

At delay `tau`, define

```text
m_1=sigma_H(a+c,b)exp(-nu lambda_1 tau),

m_2=sigma_H(b+c,a)exp(-nu lambda_2 tau),

m_0=sigma_H(a,b)exp(-nu lambda_0 tau).             (3.7)
```

The exact decomposition

```text
m_1G_1+m_2G_2

 =m_0(G_1+G_2)
  +(m_1-m_0)G_1+(m_2-m_0)G_2                     (3.8)
```

separates the old cancellation from its two commutators.

The first term costs `18L`. By (1.6), each selector difference costs
`C_chi L/H`; each unpaired `G_j` is at most `5H` times the product of the
three amplitudes. The mean-value theorem and (3.6) give

```text
|exp(-nu lambda_j tau)-exp(-nu lambda_0 tau)|

 <=C nu tau HL exp(-c nu H^2 tau).                (3.9)
```

After multiplication by the unpaired `O(H)` term, (3.9) is

```text
CL(nu H^2 tau)exp(-c nu H^2 tau).                 (3.10)
```

The polynomial is absorbed into a slightly weaker exponential. Therefore

```text
||m_1G_1+m_2G_2||_F

 <=C(1+C_chi)L exp(-c'nu H^2 tau)
      ||U_a||||U_b||||U_c||.                      (3.11)
```

This proves that exact pairwise heat rates and smooth shell boundaries do
not restore the apparent high derivative.

At a sharp Galerkin top boundary, one term in (3.8) can be absent and this
paired estimate is unavailable. The individual term has its original
`O(H)` symbol. It is retained as a separate leakage envelope below.

## 4. Complete forcing square

For the low shells in the HHL block, define

```text
S_H(t)=sum_(L<=H/4)L^(5/2)a_L(t),

T_H(t)=sum_(L<=H/4)L^(3/2)a_L(t).                 (4.1)
```

The factor consists of the low derivative in (3.11) and the
three-dimensional Bernstein bound
`||u_L||_infinity<=CL^(3/2)a_L`. The second envelope omits the
commutator derivative and is reserved for a worst-case unpaired sharp
Galerkin boundary term.

The complete heat-weighted shell forcing is bounded by an integrable
kernel times

```text
g_H(t)
 =C[H^(5/2)A_H(t)^3
    +A_H(t)^2S_H(t)
    +H A_H(t)^2T_H(t)].                           (4.2)
```

The first term is the comparable-shell HHH Bernstein estimate. The second
is the complete paired HHL estimate, including smooth selector leakage.
The third pays the sharp Galerkin boundary even if every affected HHL term
is treated separately.

Set

```text
E_*=sup_t sum_J a_J(t)^2,

D=integral sum_J J^2a_J(t)^2 dt.                 (4.3)
```

For dyadic shells,

```text
S_H(t)^2
 <=[sum_(L<=H/4)L^3]
    [sum_(L<=H/4)L^2a_L(t)^2]

 <=H^3 D_(<H)(t)/56.                              (4.4)
```

Similarly,

```text
T_H(t)^2
 <=[sum_(L<=H/4)L]
    [sum_(L<=H/4)L^2a_L(t)^2]

 <=H D_(<H)(t)/2.                                 (4.5)
```

Finite comparable-shell overlap gives

```text
sum_H H^2A_H^2<=21.3125 sum_J J^2a_J^2,

sum_H A_H^2<=5 sum_J a_J^2.                       (4.6)
```

Consequently,

```text
sum_H H^(-3)
       ||H^(5/2)A_H^3||_(L2_t)^2
 <=21.3125 E_*^2D,                               (4.7)

sum_H H^(-3)||A_H^2S_H||_(L2_t)^2
 <=25E_*^2D/56,                                  (4.8)

sum_H H^(-3)||H A_H^2T_H||_(L2_t)^2
 <=25E_*^2D/2.                                   (4.9)
```

Using `(x+y+z)^2<=3(x^2+y^2+z^2)`,

```text
sum_H H^(-3)||g_H||_(L2_t)^2
 <=104 C^2 E_*^2D.                               (4.10)
```

The multiplier and finite low-channel constants are contained in `C`.
Equations (4.7)-(4.9) cover all comparable HHH and separated HHL
interactions; the additional (4.9) makes the result uniform across sharp
Galerkin top cutoffs. It does not rely on shell-label orthogonality or
finite-packet extrapolation.

## 5. Forced response and initial stress

The kernel in (3.11) has time `L1` norm at most

```text
C/(nu H^2).                                       (5.1)
```

Young's inequality therefore gives

```text
||C_H^F||_(L2_t)
 <=C/(nu H^2)||g_H||_(L2_t).                      (5.2)
```

For dyadic `H>=H_0`, with `H_0` at least four times the largest retained
low-output/low-input scale, Cauchy-Schwarz and (4.10) yield

```text
||sum_(H>=H_0)C_H^F||_(L2_t)

 <=C E_*sqrt(D)/(nu sqrt(H_0)).                   (5.3)
```

The initial term in (1.5) obeys, by Fourier Cauchy-Schwarz,

```text
||C_H^0(t)||_F
 <=C exp(-c nu H^2t)A_H(0)^2.                    (5.4)
```

Its time `L2` norm and the dyadic sum satisfy

```text
||sum_(H>=H_0)C_H^0||_(L2_t)

 <=C E(0)/(sqrt(nu)H_0).                          (5.5)
```

Combining (5.3) and (5.5),

```text
||sum_(H>=H_0)C_H||_(L2_t)

 <=C[
      E(0)/(sqrt(nu)H_0)
      +E_*sqrt(D)/(nu sqrt(H_0))
     ].                                           (5.6)
```

For every fixed finite set of low Fourier/tensor channels, the complete
high-shell stress tail therefore tends to zero as `H_0` tends to infinity,
uniformly over smooth Galerkin truncations with the same Leray bounds.

## 6. Scope and next gate

Established here:

- the exact pairwise stress Duhamel formula;
- retention of every pairwise viscous rate;
- the heat-weighted HHL sweeping commutator;
- payment of smooth shell-selector leakage;
- direct payment of sharp Galerkin top-cutoff leakage;
- the complete HHH plus HHL `H^(-3)` forcing-square estimate;
- the forced response and initial-stress tail bounds;
- vanishing high-shell stress tails at fixed finite low channels for
  smooth Galerkin solutions.

Not established:

- constants uniform over an increasing low-output channel family;
- a scale-uniform physical-space partition or Carleson theorem;
- compactness of every nonlinear term needed for suitable weak passage;
- exceptional-set removal or global regularity.

The next gate is to replace the fixed finite low-channel family by a
scale-uniform low-output Littlewood-Paley or partition-space norm, then
prove that estimate is stable under Galerkin limits. Weak-solution claims
must wait for that step.

The identities and finite-sequence replays are generated by
`scripts/smooth_galerkin_shell_response_gate_audit.py`.
