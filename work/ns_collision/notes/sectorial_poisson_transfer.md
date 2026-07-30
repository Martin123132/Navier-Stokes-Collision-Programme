# Sectorial Poisson form transfer

## Abstract theorem

Let the zero-boundary baseline form split as

```text
a_0=h+k,
```

where `h` is symmetric and coercive and `Re k(v,v)=0`. Let a nonnegative
potential obey

```text
q[v]<=alpha h[v],                 0<=alpha<1.          (1)
```

For equal outer boundary data, write `w=u_q-u_0`. Then

```text
(A_0-Q)w=Q u_0.                                        (2)
```

Testing (2) against `w` and using Cauchy-Schwarz in the `q` form gives

```text
(1-alpha)h[w]
 <=sqrt(q[u_0]q[w])
 <=sqrt(q[u_0] alpha h[w]).                            (3)
```

If `q` is supported where a cutoff `zeta` equals one, then

```text
q[u_0]=q[zeta u_0]<=alpha h[zeta u_0].                 (4)
```

Combining (3)-(4),

```text
sqrt(h[w])
 <=alpha/(1-alpha) sqrt(h[zeta u_0]).                  (5)
```

For an inner trace satisfying `||Tv||<=C_T sqrt(h[v])`, define

```text
E_zeta=sup h[zeta P_0 f],
chi_sec=C_T sqrt(E_zeta)/||B_0||.                      (6)
```

Then

```text
||B_q||/||B_0||
 <=1+chi_sec alpha/(1-alpha).                          (7)
```

This is the nonsymmetric counterpart of the positive-resolvent Poisson
bound. The size of the skew part does not occur in (5). Six random matrix
trials with skew strengths through `1000` verify the finite-dimensional
operator inequality.

## Application to the streamfunction shell

For the divergence-free three-dimensional drift, axial conjugation leaves
the transverse real energy

```text
h[v]=integral(|grad v|^2+(zeta+1/2-c_t)|v|^2).         (8)
```

At the selected taper, `c_t<=1.00000165`. For `H/L=1.2`, the worst spectrum
therefore has positive constant lower mass

```text
zeta+1/2-c_t=0.76030020.                               (9)
```

The trace norm of this constant lower energy is the radial modified-Bessel
Green diagonal at `r=1`, giving `C_T^2=0.57519847`.

The cutoff identity does not require symmetry. Since `A_0u_0=0`, testing with
`zeta^2u_0` yields

```text
h[zeta u_0]
 =integral |grad zeta|^2 u_0^2                        (10)
```

for the real part of the form. The finite-element pilot gives, at `t=1`,

```text
||B_0||=0.80264472,
E_zeta=25.46110,
chi_sec=4.76787,
C_0=0.25663566.                                        (11)
```

Equation (7) retains renewal contraction for

```text
alpha<0.16962754.                                      (12)
```

Because (9) makes `h[v]>=||grad v||_2^2`, the sharp homogeneous Sobolev
inequality converts (12) to the diagnostic critical mass

```text
||q_+||_(3/2)/nu < alpha/S_3 =0.92920342.              (13)
```

The abstract implication (1)-(7) is rigorous. The numbers in (9)-(13) are
finite-element and dense-polynomial calibrations, not certified enclosures.

## Remaining first-order gate

Let a divergence-free first-order mismatch have sector bound

```text
|k_e(u,v)|<=beta sqrt(h[u]h[v]).                       (14)
```

Repeating (3) with the forcing `(Q-K_e)u_0` gives

```text
sqrt(h[w])
 <=(alpha+beta)/(1-alpha) sqrt(h[zeta u_0]).           (15)
```

Thus the two errors obey the explicit working budget

```text
beta < d-(1+d)alpha,
d=[C_0^(-1/2)-1]/chi_sec=0.20427887.                  (16)
```

For a divergence-free drift error `e`, Holder and Sobolev give

```text
beta<=sqrt(S_3)||e||_3/nu.                             (17)
```

With no zero-order error, (16) permits
`||e||_3/nu<0.47811311`. Giving equal sector shares to `alpha` and `beta`
permits

```text
alpha=beta<0.09267379,
||q_+||_(3/2)/nu<0.50765811,
||e||_3/nu<0.21690228.                                 (18)
```

This controls a first-order mismatch through an explicit critical norm; it is
not harmless merely because it vanishes from `Re a(v,v)`. The remaining PDE
task is to derive (17), with the required support/cutoff and moving-frame
properties, from the actual velocity relative to the fitted shell.

The theorem algebra, closure equality, Sobolev conversion, and random skew
stress tests are reproduced by
`scripts/sectorial_poisson_transfer_audit.py`.
