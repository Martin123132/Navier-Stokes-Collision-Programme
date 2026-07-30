# Multiband weighted-Fisher recombination no-go

Status: certified. The production audit and all five focused tests pass.
This result falsifies a specific auxiliary summation step, not the preceding
single-band pressure theorem.

## 1. The proposed recombination

The balanced-annular pressure theorem controls one Fourier band by that
band's vertex-weighted velocity Fisher energy. To sum bandwise estimates,
one might seek

```text
sum_j integral lambda|grad u_j|^2
 <=C integral lambda|grad(sum_j u_j)|^2,            (1.1)
```

with `C` independent of the number of bands and valid for every compatible
nonnegative partition weight, including weights with zero faces.

Equation (1.1) is false even when:

- every `u_j` is a smooth finite Fourier divergence-free field;
- every component lies in an annulus of radial width less than two;
- the weighted Fourier graph has nearest-neighbor degree two;
- the full field is a pressure-free Navier-Stokes shear.

## 2. Exact Fourier graph

For a periodic weight,

```text
E_lambda(u)
 =sum_(k,l)(k dot l) lambdahat(l-k)
   uhat(k) dot conjugate(uhat(l)).                  (2.1)
```

Choose the compatible zero-face weight

```text
lambda(x)=sin(x_1/2)^2=(1-cos x_1)/2.              (2.2)
```

It is obtained from the eight-cell partition by taking coefficient one at
the four vertices with first sign minus and zero at the other four.
Its Fourier support is `{0,+e_1,-e_1}`.

For a scalar one-dimensional Fourier series with

```text
d_n=n a_n,
```

equation (2.1) becomes exactly

```text
E_lambda(f)
 =(1/4)sum_n |d_(n+1)-d_n|^2.                      (2.3)
```

Thus weighted Fisher is the discrete Dirichlet form on each frequency
residue chain. Constant chain data are an interior null direction; only
chain endpoints are charged.

## 3. Divergence-free dyadic counterexample

For an integer `J>=1`, put

```text
N=2^J-1,

u_J(x)
 =(0,sum_(n=1)^N sin(nx_1)/n,0).                   (3.1)
```

This is smooth, finite Fourier, divergence free, and pressure free:

```text
(u_J dot grad)u_J=0.
```

Split it into dyadic annuli

```text
u_j
 =(0,sum_(2^j<=n<2^(j+1)) sin(nx_1)/n,0),

j=0,...,J-1.                                       (3.2)
```

Every block satisfies

```text
2^j<=|k|<=2^(j+1)-1<2*2^j.                         (3.3)
```

For the sine coefficients in (3.1), `d_n` is constant on every occupied
positive and negative chain. Formula (2.3) gives

```text
E_lambda(u_J)=1/4,

E_lambda(u_j)=1/4 for every j.                     (3.4)
```

Therefore

```text
sum_j E_lambda(u_j)=J/4,

[sum_j E_lambda(u_j)]/E_lambda(u_J)=J.             (3.5)
```

No constant in (1.1) can be uniform.

## 4. The missing terms are exactly the interfaces

Expanding the physical quadratic form gives

```text
E_lambda(sum_j u_j)
 =sum_j E_lambda(u_j)
  +2sum_(j<k) Re integral lambda grad u_j.grad u_k. (4.1)
```

Only neighboring dyadic blocks interact for this example. The complete
signed correction is

```text
-(J-1)/4,                                          (4.2)
```

and every one of the `J-1` adjacent interfaces contributes exactly

```text
-1/4.                                              (4.3)
```

Finite graph degree therefore does not imply coercive recombination. The
neighboring edges carry essentially all of the long-chain cancellation.
Taking bandwise absolute values deletes the mechanism.

## 5. Positive floors and weight Fisher

For

```text
lambda_epsilon
 =epsilon+sin(x_1/2)^2,
```

the exact ratio becomes

```text
[J+2epsilon(2^J-1)]/
[1+2epsilon(2^J-1)].                               (5.1)
```

Every finite example can have a strictly positive floor. Choosing

```text
epsilon=(2^J-1)^(-2)
```

still makes (5.1) asymptotic to `J`. This is a floor-uniform no-go, not a
contradiction to a fixed-positive-floor weighted estimate.

The terminal-weight Fisher term does not repair (1.1) as an additive
weight-only allowance:

```text
integral lambda|grad lambda|^2=1/16.               (5.2)
```

Velocity scaling `u ->alpha u` multiplies both sides of (1.1) by
`alpha^2` while (5.2) is unchanged. Any fixed additive multiple of (5.2)
is defeated by taking `alpha` large.

## 6. Co-scaling

For integer `m` and amplitude `a`, set

```text
lambda_m=sin(mx_1/2)^2,

u_(J,a,m)(x)=a u_J(mx).                            (6.1)
```

Every physical and component velocity Fisher energy scales by `a^2m^2`,
and their ratio remains exactly `J`. The field remains pressure free. Thus
partition-frequency adaptation and amplitude co-scaling do not rescue the
discarded recombination step.

## 7. Consequence for the proof route

This no-go does not invalidate:

- the complete pressure estimate for one aggregated bounded annular field;
- the time-integrated floor-free far-carrier `H^(-1)` pressure-tail theorem;
- a future joint signed pressure-Fisher block estimate.

It does rule out summing independent single-band pressure estimates after
charging each one to its isolated weighted Fisher energy.

The next live target must retain the off-diagonal Fisher interfaces in the
same block form as the cross-shell `HHL` pressure. A useful estimate must
control the joint signed pressure-minus-Fisher Schur complement, or remove
far carriers by the `H^(-1)` tail theorem before any bandwise Fisher split.
Taylor-Green, seed-81, and amplitude co-scaling remain mandatory checks.

The deterministic certificate is generated by
`scripts/multiband_weighted_fisher_recombination_no_go_audit.py`; its
production result is
`results/multiband_weighted_fisher_recombination_no_go_audit_v1.json`, with
SHA-256
`47b8704985671f0dac66ae38ff87a186acd6b938928828d3299a571337a7f087`.
