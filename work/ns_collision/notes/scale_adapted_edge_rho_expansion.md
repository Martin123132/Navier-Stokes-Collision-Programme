# Scale-adapted pressure edges and the short-time rho expansion

## Scope and outcome

The preceding adjoint stage left two possible routes:

1. adapt the pressure partition scale so its cubic edge penalty can compete
   with the pressure transfer;
2. find a genuinely finite-time benefit from positively correlated
   projected replicas.

This note tests the first necessary conditions for both routes.

The conclusions are:

- A fixed partition scale cannot absorb pressure uniformly under amplitude
  scaling. The loose edge Young ratio grows like the square of the local
  Reynolds number.
- Co-scaling the partition frequency with local amplitude is dimensionally
  viable: the exact pressure/Fisher ratio is proportional to
  `Re_cell=a/(nu m)`.
- On the stored finite-Fourier adversary, increasing partition frequency
  strongly improves the resolved edge budget, and frequencies `7` through
  `12` are silent. This is a band-limited diagnostic, not a pressure-tail
  theorem.
- Positive replica correlation is strictly worse through the leading
  short-time order after every nontrivial reset.
- For the stored field, the first full time coefficient bends strongly in
  the favorable direction. Its quadratic Taylor truncation crosses near
  `0.07561`, which identifies a finite-window test scale but does not certify
  a sign change.

Everything is derived for smooth periodic data. No critical bound,
low-regularity partition, or global regularity result is claimed.

## 1. Frequency-dependent partition identity

Replace each one-dimensional partition factor by

```text
phi_+=(1+cos(m(x-x_*)))/2,
phi_-=(1-cos(m(x-x_*)))/2,
```

where integer `m>=1` is the partition frequency. Condition on the other two
coordinates and write

```text
lambda=A phi_+ +B phi_-,
D=A-B,
H=A+B.
```

Define

```text
e_m=mean_normal[p u_j partial_j phi_+].
```

The exact directional pressure and adjoint-Fisher terms are

```text
P_j=mean_other[D e_m],                              (1.1)

D_lambda,j=(m^2/16)mean_other[H D^2].              (1.2)
```

The factor `m^2` in (1.2) is essential. Pointwise Young optimization gives

```text
P_j-nu D_lambda,j
 <=4/(nu m^2) mean_other[e_m^2/H].                 (1.3)
```

Formula (1.3) is valid when `H>0`, with the usual extended-value convention
at a zero face. It remains too lossy unless the scale is chosen together
with the local amplitude.

## 2. Exact scaling law

Co-scale a base field, pressure, and weight by

```text
u_(a,m)(x)=a u(mx),
p_(a,m)(x)=a^2 p(mx),
lambda_(a,m)(x)=a lambda(mx).
```

On the fixed periodic domain, which contains repeated scaled cells,

```text
pressure flux              scales as a^4 m,
velocity and weight Fisher scale as nu a^3 m^2.
```

Therefore

```text
pressure/Fisher = a/(nu m)=Re_cell.                (2.1)
```

The Young remainder in (1.3) scales as `a^5/nu`, so relative to velocity
Fisher it scales as

```text
(a/(nu m))^2=Re_cell^2.                            (2.2)
```

Consequences:

- No estimate based on (1.3) can be uniform at fixed `m`.
- A necessary scale choice is `m` proportional to at least `a/nu`.
- This is exactly the bounded local-Reynolds condition already encountered
  in the collision/localization programme, now derived from the signed
  adjoint pressure edge.

Equations (2.1)-(2.2) are dimensional necessities, not sufficient
pressure-tail estimates.

## 3. Resolved frequency sweep

The seed-81 field is resampled on a `96^3` grid. The smooth positive
eight-cell coefficient vector from the adjoint stage is used at frequencies
`m=1,...,12`.

For each frequency, direct pressure, conditional pressure, direct Fisher,
and conditional cubic-edge Fisher agree within `2e-10`.

Selected values:

```text
m   pressure      Young remainder   exact sign alpha   edge alpha
1   1.28045350    3483.13607        614.8179           0.4754
2   7.58659156    2848.81034        102.8929           0.5233
3   3.00498342    1276.24927        262.6968           0.7859
4   7.58404776     603.42891        104.0992           1.1424
5   2.82429911     180.07515        279.6891           2.0902
6   0.93454801      26.75064        846.7377           5.4232
```

Here:

- `exact sign alpha` is the fixed-frequency amplitude at which the exact
  pressure-Fisher flux changes sign;
- `edge alpha` is the amplitude up to which the separated Young remainder
  can be paid by the weighted velocity Fisher term.

The Young route fails even at base amplitude for `m=1,2,3`, becomes
nontrivially payable at `m=4`, and improves further at `m=5,6`.

For this particular finite-mode field, direct pressure and the edge
remainder vanish to numerical precision for `m=7,...,12`. This spectral
silence confirms the implementation and the value of scale separation, but
general Navier-Stokes pressure has no finite Fourier cutoff. A rigorous route
must pay the high-frequency tail rather than assume it vanishes.

## 4. First-chaos replica expansion

Reset the projected replicas at time `s`. For elapsed time `tau`,

```text
V_r
 =u-sqrt(2nu) partial_k u Delta W_r^k+O(tau).
```

For Brownian correlation `rho`,

```text
E[Delta W_1^k Delta W_2^l]
 =rho tau delta_kl.
```

The first correlation corrections are therefore

```text
R_rho
 =u^T S u
  +2nu rho tau sum_k (partial_k u)^T S(partial_k u)
  +O(tau^2),

G_rho
 =|grad u|^2
  +2nu rho tau |grad^2 u|^2
  +O(tau^2).                                      (4.1)
```

For fixed `u`, the replica pressure is the linear operator

```text
Delta Pi_u[V]
 =-div((u dot grad)V+(grad u)^T V).                (4.2)
```

Thus

```text
F_rho
 =2(p-|u|^2/2)u
  +4nu rho tau sum_k Pi_u[partial_k u] partial_k u
  +O(tau^2).                                      (4.3)
```

The audit implements (4.2) spectrally. Substituting `V=u` recovers
`p-|u|^2/2` up to a constant with maximum residual below `5.7e-13`.

## 5. Short-time correlation ordering

Let `Q_rho` be the adjoint dual generator. At the reset,

```text
partial_rho Q_rho|_(rho=0,t=s)
 =3nu integral lambda|grad u|^2
 =K_0>0.                                           (5.1)
```

Hence, for every nontrivial positive weight,

```text
integral_s^(s+h)(Q_rho-Q_0)dt
 =rho K_0 h+O(rho h^2)>0                           (5.2)
```

for all sufficiently small `h`. Positive correlation is not merely worse
at the reset instant; it is worse on an entire short restart interval.

The next coefficient includes deterministic evolution of
`lambda|grad u|^2` and the first-chaos strain, pressure, and Hessian terms:

```text
K_1
 =3nu d_t integral lambda|grad u|^2
  -3 integral lambda R_(rho,1)
  +(3/2) integral grad lambda dot F_(rho,1)
  -3nu integral lambda G_(rho,1).                  (5.3)
```

For the seed-81 field and smooth positive partition weight, grids
`48,64,80` agree to displayed precision:

```text
K_0 = 2361.35782576588,
K_1 = -62459.6458230194.
```

The components at grid `64` are

```text
3nu d_t D                  = -31207.6179311623,
first-chaos strain term    =    615.5025409729,
first-chaos pressure term  =    -15.0445977978,
first-chaos Hessian term   = -31852.4858350322.
```

The negative curvature is dominated by deterministic gradient decay and the
replica Hessian correction, not pressure.

The quadratic truncation

```text
rho[K_0 h+(1/2)K_1 h^2]
```

has formal integrated zero at

```text
h=0.0756122707598066.
```

This is not a sign certificate. Terms of order `h^3` are uncontrolled at
that duration. It is a principled location for a finite-window
nonperturbative experiment, replacing an arbitrary time sweep.

## 6. What this stage establishes

Established under smooth assumptions:

- the exact frequency-dependent edge and cubic Fisher identities;
- the local-Reynolds scaling laws (2.1)-(2.2);
- impossibility of fixed-scale universal Young absorption;
- resolved improvement of the edge budget with partition frequency;
- the first-chaos formulas for `R_rho`, `F_rho`, and `G_rho`;
- the complete first time coefficient in the rho derivative of the dual
  generator;
- strict inferiority of `rho>0` on sufficiently short restart windows.

Still open:

- a pressure-tail estimate uniform at the intrinsic local-Reynolds scale;
- construction of an adapted multilevel partition at that scale;
- a nonperturbative finite-window test near `h=0.0756`;
- proof that any finite-time rho advantage exists;
- preservation of the terminal dual supremum through scale changes;
- low-regularity and exceptional-set arguments;
- global regularity.

## 7. Next theorem target

The next bounded stage should not extend the Taylor series blindly. It should:

1. integrate the deterministic two-replica correlation PDE, or an equivalent
   projected stochastic ensemble, over a controlled window around
   `h=0.0756`;
2. compare the full `rho`-dependent signed dual flux with `rho=0`, including
   the variance majorization paid at the terminal endpoint;
3. simultaneously formulate the intrinsic scale
   `m approximately a/nu` and bound the pressure frequencies above that
   scale.

A useful rho route must overcome the exact short-time loss. A useful edge
route must control genuine pressure tails, not rely on the spectral silence
of the finite-mode stress field.
