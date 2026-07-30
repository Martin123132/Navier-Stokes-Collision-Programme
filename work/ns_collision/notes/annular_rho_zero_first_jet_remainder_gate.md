# Annular rho-zero first-jet remainder gate

## 1. Result

The preceding first-jet theorem proved that the viscous-pressure component of
the static-optimal annular restart family has a strictly negative `N^5`
limit. It left open whether the Euler, viscous weighted-Fisher, and two
backward-weight directions could contain another `N^5` term.

They do not. If

```text
u_N=h_N-a_N U,
lambda_N=t_N Phi_+++,
```

and `R_N` denotes the complete first derivative with the viscous-pressure
component removed, then

```text
R_N=O(N^4)=o(N^5).                                (1.1)
```

Consequently the complete normalized generator derivative satisfies

```text
g'_N/N^5
 -> -1.0442344590350905e-7/nu<0.                  (1.2)
```

Thus the static-optimal annular witness initially moves in the wrong
direction at exactly the scale required to overcome its reset-time Legendre
deficit.

Equation (1.2) is a restart-time asymptotic theorem. It is not a
finite-window theorem: a negative first derivative does not exclude a later
turnaround on a window of length `T/N^2`.

## 2. Starting point

Remove the inessential outer factor three from the replica identity and set

```text
g_0(u,lambda)
 =integral[
    p(u)u dot grad lambda
    -nu lambda|grad u|^2
    -nu lambda|grad lambda|^2].
```

The exact first derivative is

```text
g'_0
 =D_u g_0[v_E]
  +D_u g_0[v_nu]
  +D_lambda g_0[mu_A]
  +D_lambda g_0[mu_nu],                           (2.1)
```

where

```text
v_E  =-(u dot grad)u-grad p,
v_nu = nu Delta u,
mu_A  =-u dot grad lambda,
mu_nu =-nu Delta lambda.
```

The predecessor audit established the exact directional formulas and proved

```text
D_u g_pressure[v_nu]/N^5 -> c_*,

c_*=-1.0442344590350905e-7/nu<0.                  (2.2)
```

The task here is to bound every other term in (2.1).

The static optimizer gives

```text
a_N=O(N),
t_N=O(N),                                        (2.3)
```

with explicit limits

```text
a_N/N -> |b_0|/nu,
t_N/N -> 64|b_0|/(15sqrt(2)nu).
```

## 3. Compatible tensor stencils

In one coordinate,

```text
phi(x)=(1+cos x)/2
```

has Fourier coefficients `1/4,1/2,1/4` at `-1,0,1`. The annular coefficient
contains the alternating factor `(-1)^n`. After this parity is gauged away,
the value, first-derivative, and second-derivative stencils have symbols

```text
S_0(z)=1/2-(z+z^-1)/4
       =-(z-1)^2/(4z),

S_1(z)=(z^-1-z)/4
       =-(z-1)(z+1)/(4z),

S_2(z)=-(z+z^-1)/4
       =-(z^2+1)/(4z).                            (3.1)
```

Their vanishing orders at `z=1` are respectively two, one, and zero.
Tensorization gives:

```text
Phi                  six differences,
grad Phi             five differences,
D^2 Phi              four differences,
Delta Phi            four differences,
grad Delta Phi       at least three differences. (3.2)
```

These are exact algebraic factorizations. They are the Fourier form of the
sixth-order zero of `Phi_+++` at the packet centre `(pi,pi,pi)`.

## 4. Frequency incidence

The positive high packet has first coordinate

```text
2N<=k_1<=3N-1,
```

and the negative packet has the reflected interval. Every low field and
weight product used in the first jet has bounded first-coordinate support.

One high leg cannot return to bounded frequency. For three high legs, the
closest two-positive/one-negative sum has first-coordinate magnitude at
least

```text
4N-(3N-1)=N+1.
```

Therefore, for `N>=5`, every integrated monomial containing an odd number of
high legs vanishes exactly. Only zero, two, or four high legs survive.

This also fixes the low-amplitude parity:

```text
quartic pressure channels        even in a_N,
cubic Fisher channels            odd in a_N,
antidiffusive cubic pressure     odd in a_N,
antidiffusive quadratic Fisher   even in a_N.      (4.1)
```

The dealiased `a` versus `-a` replay verifies (4.1) with maximum residual
`5.41e-16`.

## 5. Viscous weighted-Fisher term

For the positive high packet define the parity-gauged matrix

```text
F_N(a,b,c)
 =(-1)^(a+b+c) k_abc tensor hhat_N(k_abc).
```

The exact compatible Fisher identity is

```text
E_Phi(h_N)
 =(1/32) sum ||Delta_1 Delta_2 Delta_3 F_N||_F^2.
```

Applying `nu Delta` to the high velocity inserts `-|k|^2`. Hence the high
part of the viscous weighted-Fisher remainder is

```text
2nu^2 t_N P_N,

P_N
 =(1/32) sum
   Delta_123 F_N : Delta_123(|k|^2 F_N).          (5.1)
```

The gauged tensor is the restriction of a uniformly smooth function on a
fixed compact annulus separated from zero. Cellwise finite-difference
estimates give

```text
sum ||Delta_123 F_N||^2          =O(N^-3),

sum ||Delta_123(|k|^2 F_N)||^2  =O(N).
```

Cauchy-Schwarz in (5.1) therefore yields

```text
P_N=O(N^-1).                                      (5.2)
```

The low plane wave has squared frequency two and exact weighted Fisher cost
`a_N^2/2`. Its heat derivative contributes exactly

```text
2nu^2 t_N a_N^2.
```

Combining this with (2.3) and (5.2),

```text
R_F,nu(N)
 =2nu^2t_N(P_N+a_N^2)
 =O(N^3)=o(N^5).                                  (5.3)
```

The direct mixed-difference replay through `N=65` has

```text
N P_N=21.7732,22.1775,22.4880,22.7340,
      22.9337,23.2381,23.6269.
```

At the five FFT carriers, (5.1) plus the exact low term reproduces the
dealiased derivative with maximum residual `8.83e-15`.

## 6. The pressure-shell lemma

The pure four-high pressure terms require care because pressure and the
Leray projection are not smooth at zero output.

Write

```text
P_N(r)=Fourier coefficient of p[h_N,h_N] at r,
V_N(r)=Fourier coefficient of v_E[h_N,h_N] at r.
```

Since there are at most `C N^3` contributing pairs and every high
coefficient is `O(N^-1)`,

```text
|P_N(r)|<=C N,
|V_N(r)|<=C N |r|.                                (6.1)
```

The second estimate uses the divergence form of the projected Euler
nonlinearity.

Split the internal output into the finite shell and dyadic shells

```text
K<=|r|<2K,  1<=K<=CN.
```

The finite shell is handled directly. Divergence freedom gives
`r dot hhat=O(N^-1)` there, so the zero-output pressure convention creates
no singular term.

On a positive shell, eliminate one resonant index. The coefficient profile
is a zero-extended uniformly `C^6` lattice function: the sine factors vanish
at every boundary. Apply the five differences from `grad Phi` by discrete
summation by parts. A difference landing on the profile costs `C/N`; a
difference landing on a pressure or Leray projector costs `C/K`. Thus

```text
|Delta^(1,2,2) V_N(r)|
 <=C N K (N^-1+K^-1)^5.                           (6.2)
```

There are at most `C K^3` outputs in the shell. Equations (6.1)-(6.2) give

```text
sum_(|r|~K)
 |P_N(r) Delta^(1,2,2)V_N(-r)|
 <=C N^2 K^4(N^-1+K^-1)^5
 <=C N^2/K.                                       (6.3)
```

Summing the finite and dyadic shells gives the deliberately conservative
bound

```text
Q_HHHH<=C t_N N^2 log(2+N)=O(N^3 log N).          (6.4)
```

The pressure-variation term is put in the same form by self-adjointness of
`R_iR_j`: move the pressure multiplier onto the external
`h dot grad Phi`, then apply the same five differences. The
weight-advection pressure term is identical after writing

```text
mu_A=-u dot grad lambda
```

in Fourier space and applying the gradient stencil to the shifted high
coefficient.

The logarithm in (6.4) is harmless and can be absorbed into one additional
power of `N`.

## 7. Two-high/two-low pressure branch

At high output, one mixed pressure coefficient obeys

```text
|p[U,h_N](k)|<=C N^-2.                            (7.1)
```

Indeed, one factor `(k+ell) dot hhat(k)` reduces to
`ell dot hhat(k)=O(N^-1)`, while the pressure denominator is order `N^2`.

The mixed Euler coefficient obeys

```text
|v_E[U,h_N](k)|<=C.                               (7.2)
```

Only `O(N^3)` high outputs occur. Restoring the two low amplitudes and the
outer weight scale gives

```text
Q_HHLL
 <=C t_N a_N^2 N
 =O(N^4).                                         (7.3)
```

For the nested pressure variation, move `R_iR_j` by self-adjointness and use
the same discrete summation by parts. No additional carrier power appears.

This is the worst branch in the complete remainder ledger. A direct
amplitude extraction corroborates (7.3): between `N=9` and `N=25`,

```text
Euler pressure a^2 coefficient / N:
-0.005405,-0.004959,-0.004723,-0.004577,-0.004477,

weight-advection pressure a^2 coefficient / N:
 0.000979, 0.000878, 0.000828, 0.000798, 0.000779.
```

These finite values are diagnostics; the bound follows from (7.1)-(7.3).

## 8. Remaining weighted channels

The exact stencil orders and the support parity now give the following
ledger.

```text
channel                                      optimized bound
velocity-viscous weighted Fisher             O(N^3)
Euler pressure                               O(N^4)
Euler weighted Fisher                        O(N^4)
weight-advection pressure                    O(N^4)
weight-advection velocity Fisher             O(N^4)
weight-advection weight self                 O(N^4)
weight-antidiffusion pressure                O(N^3)
weight-antidiffusion velocity Fisher         O(N^3)
weight-antidiffusion weight self             O(N^3)
```

For the quadratic high correlations, `Phi`, `grad Phi`, and `Delta Phi`
apply six, five, and four differences. The low-low terms contain only fixed
Fourier supports and are bounded directly using (2.3). The only possible
low-only quartic size is `a_N t_N^3=O(N^4)`.

Every branch is therefore `o(N^5)`, proving (1.1).

## 9. Total limit and interpretation

Combining (1.1) with the predecessor pressure theorem,

```text
g'_N
 =D_u g_pressure[nu Delta u_N]+R_N,

g'_N/N^5 -> c_*<0.
```

The five complete finite rows are already negative:

```text
N=25,29,33,37,41.
```

The theorem now promotes this from a finite observation to eventual
large-carrier negativity.

What it rules out:

- the static-optimal annular `+++` witness cannot obtain its required
  `N^5` amplification from a positive initial first jet;
- none of the previously unbounded remainder channels changes the leading
  sign.

What it does not rule out:

- a rapid second-order turnaround during `T/N^2`;
- a different terminal weight close to the Legendre optimizer;
- nonlinear amplification later in the restart window;
- singularity formation or global regularity.

The next gate is a second-time-jet or Taylor-remainder estimate strong enough
to control

```text
g_N(t)-g_N(0)-t g'_N(0)
```

uniformly for `0<=t<=T/N^2`.

## 10. Reproducibility

Run:

```text
python work/ns_collision/scripts/annular_rho_zero_first_jet_remainder_gate_audit.py
```

The production record is
`results/annular_rho_zero_first_jet_remainder_gate_audit_v1.json`.
