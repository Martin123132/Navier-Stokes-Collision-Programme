# Nonautonomous scalar-gain gate

## Purpose

The square-tilted kernel theorem reduces the compact visit to the scalar
constant-payoff gain `g=||K1||_(L2(entry law))`. This note tests three ways
to control it and separates what closes from what does not.

The main positive result is a uniform arbitrary-history estimate in volume
`L2`. The remaining obstruction is transporting a boundary entry law into
that volume norm without conditioning away the contraction.

## All-exit majorant

Let `tau` be the first exit from

```text
D={r<2, |z|<0.75}.                                    (1)
```

The radial-exit constant payoff is bounded by the stronger all-exit moment

```text
K1(y)<=m(t,y)
      =E_(t,y) exp[int_t^tau lambda(s)ds].             (2)
```

Here `lambda(s)<=1` is the normalized maximal stretching and the affine
drift is trace-free.

## Uniform volume theorem

Write `m=1+w`. With zero boundary trace for `w`, Duhamel gives

```text
w(t)=int_t^infinity U(t,s)lambda(s) ds.                (3)
```

Every affine first-order term is skew in Lebesgue `L2`. The cylinder
Dirichlet floor is

```text
lambda_1(D)=j_(0,1)^2/4+pi^2/[4(0.75)^2]
           =5.832287335665.                            (4)
```

Therefore, for every measurable admissible strain history,

```text
||U(t,s)||_(2->2)
 <=exp[-(lambda_1-1)(s-t)],

lambda_1-1=4.832287335665.                             (5)
```

Since `||lambda||_2<=sqrt(|D|)`, (3) gives the pointwise-in-time bound

```text
||m(t)||_(L2(D))/sqrt(|D|)
 <=1+1/(lambda_1-1)
 =1.206941336584.                                      (6)
```

The current cubic renewal permits `g<1.232133608`. Thus (6) closes narrowly.
It tolerates a volume density bounded by `1.04218` times normalized
Lebesgue measure.

This is a genuine nonautonomous theorem. It uses the common trace-free
history before taking the norm and therefore cannot imitate inward drift in
all three directions.

## Stationary surface theorem

For a time-independent divergence-free drift, put

```text
H=-Delta-1,       (H+K)w=1,                           (7)
```

where `K` is skew. Testing against `w` gives

```text
sqrt(h[w])<=||1||_(H^-1).                             (8)
```

Separation on (1) yields

```text
||1||_(H^-1)^2=2.472575248261.                        (9)
```

The exact inner-cylinder trace constant is the principal radial Green
diagonal

```text
C_T^2=I_0(k) [K_0(k)-K_0(2k)I_0(k)/I_0(2k)]
     =0.275842983250,

k=sqrt(pi^2/[4(0.75)^2]-1).                          (10)
```

After normalizing the interface area, (8)-(10) give

```text
||m||_(L2(uniform inner surface))<=1.269011078449.     (11)
```

This no longer closes under the current cubic split: `1.26901>1.23213`.
It remains a useful stationary comparison but cannot solve the entry gate.

## Static-worst comparison is false numerically

The stronger all-exit payoff was simulated from the contracting transverse
direction of the worst spectrum. With exact linear-SDE steps, boundary
interpolation, `dt=0.00125`, and 500,000 paths:

```text
static t=1                 1.32527670 +/- 0.00042034,
x/y switch every 0.5       1.32753720 +/- 0.00043479. (12)
```

The increase is `3.74` combined standard errors. This is numerical evidence,
not an enclosure, but it is enough to reject using an unproved claim that a
static affine endpoint must be worst. Both sampled values exceed the current
common-gain closure threshold.

An inadmissible position-dependent controller can imitate isotropic inward
drift and gives approximately

```text
1.39123942 +/- 0.00071980,                             (13)
```

also exceeds the current allowance. A Bellman supersolution that lets
the orientation react separately to each particle is therefore structurally
too pessimistic.

## Covariance/Nash route

For a common trace-free affine history, the transition covariance obeys

```text
Q(t)>=(1-exp(-2t))I,
det Q(t)>=(2t)^3.                                      (14)
```

The determinant follows from `det F(t,s)=1` and Minkowski's determinant
inequality. Combining optimistic Gaussian cylinder occupancy with the sharp
free-space Nash smoothing constant and the Dirichlet tail still gives only

```text
m<=1.48294666.                                         (15)
```

Since (15) exceeds `1.23213361`, this coarse route should not be promoted as
a proof strategy without another geometric input.

## Revised gate

The arbitrary-history estimate is already good enough in volume. The next
construction should insert a buffered sub-Markov smoothing phase:

```text
boundary law
 -> unnormalized killed smoothing into a volume law
 -> nonautonomous affine visit with gain 1.20694134
 -> square-tilted exit law.                            (16)
```

The smoothing law must remain unnormalized so that trajectories exiting
before the volume phase contribute contraction rather than a bad conditional
density factor. A storage-augmented boundary renewal is the alternative.

The volume theorem, separated stationary constants, switching diagnostics,
covariance identities, and failed Nash budget are reproduced by
`scripts/nonautonomous_scalar_gain_gate_audit.py`.
