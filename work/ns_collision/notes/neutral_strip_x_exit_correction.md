# Analytic neutral-strip x-exit correction

## Purpose

The reversible FEM certificate stops trajectories at artificial sides
`|x|=X`. Width stability and a small discrete side probability do not remove
that boundary: inward OU paths can hit a side and return later. This note
bounds the continuum side event and composes all later returns through a
positive renewal inequality.

The correction is analytic. The direct return term is still supplied by the
stored finite matrix, so this stage does not certify the continuum FEM limit.

## Rectangle side probability

At `rho=0`, the transverse process is

```text
dX_t=-X_t dt+sqrt(2)dB_t^x,
dY_t=       sqrt(2)dB_t^y.                           (1)
```

Remove the absorbing unit disk. This can only increase the probability of
hitting `|x|=X` before `|y|=Y`, where `Y=2.1`. In the resulting rectangle,
expand the constant side value in Dirichlet `y` modes

```text
k_n=(2n+1)pi/(2Y),
1=sum_(n>=0) 4(-1)^n/[(2n+1)pi] cos(k_n y).          (2)
```

The even `x` coefficient solves

```text
v_n''-x v_n'-k_n^2 v_n=0,       v_n(+-X)=1.          (3)
```

Putting `z=x^2/2` gives Kummer's equation and therefore

```text
v_n(x)=M(k_n^2/2,1/2,x^2/2)
       /M(k_n^2/2,1/2,X^2/2).                       (4)
```

All terms in the defining series for `M(a,1/2,z)` are positive. The audit
uses interval arithmetic for every recurrence operation, sums until the
term ratios are proved decreasing, and encloses the remaining geometric
tail. Since `M` increases with `x^2`, every point on the `r=2` entry circle
obeys

```text
p_X <=sum_(n>=0) 4/[(2n+1)pi] v_n(2).               (5)
```

For the mode tail, comparison with

```text
cosh(k_n x)/cosh(k_n X)                              (6)
```
is valid because applying `L_x-k_n^2` to (6) gives a nonpositive result on
`0<=x<=X`. Hence

```text
v_n(2)<=2 exp[-k_n(X-2)],                            (7)
```

which supplies a closed geometric tail after twelve modes.

The resulting continuum upper bounds are

```text
X       p_X upper             stored FEM pilot
4.20    2.250904005638e-3     1.664244080768e-3
5.25    1.731311139515e-5     1.320720817810e-5
6.30    4.360755393701e-8     3.374752264798e-8.     (8)
```

The analytic bounds exceed the corresponding discrete losses, providing a
useful independent direction check. At `X=6.3`, the omitted twelve-mode tail
is below `1.23e-36`.

## Return renewal

Let `R_X` be the inner-return kernel before either a physical wall or an
artificial side, and let `R_infinity` be the kernel without the `x` sides.
Every omitted return must, after its first side hit, cross `r=2` before it can
reach `r=1`. Let `B_X` be that side-to-`r=2` continuation. It is sub-Markov
and

```text
||B_X 1||_infinity<=p_X.                             (9)
```

The positive-kernel decomposition is

```text
R_infinity=R_X+B_X R_infinity.                      (10)
```

The axial spatial-`L2` factor is

```text
A(t)=exp(t)||g_t 1_|z|<H||_2.                       (11)
```

For `H=3/4`, `A` is nonincreasing. In the variable
`q=H/sqrt(exp(2t)-1)`, this reduces to monotonicity of

```text
(q/H+H/q) erf(q).                                   (12)
```

For `q>=H` both derivative terms are nonnegative. For `q<H`, use
`erf(q)<=2q/sqrt(pi)` and

```text
exp(-q^2)(H^2+q^2)>=H^2-q^2,                        (13)
```

which follows from `H^2<2` and
`(1-r)/(1+r)<=exp(-2r)`. Thus a delayed restart cannot increase the axial
factor.

For a nonnegative time envelope `rho`, define

```text
N_ell(rho)=sum_j sup_(j ell<=t<(j+1)ell) rho(t).     (14)
```

Translating the interval grid by an arbitrary delay makes each interval
meet at most two old intervals. Minkowski and (9) therefore give

```text
N_ell(B_X*rho)<=2p_X N_ell(rho).                    (15)
```

Equations (10) and (15) imply

```text
N_ell(R_infinity)<=N_ell(R_X)/(1-2p_X).             (16)
```

All three denominators are positive; at `X=6.3` the multiplier is only
`1.000000087216`.

## Scalar correction

The scalar axial weight satisfies

```text
W(t)=exp(t) erf[H/sqrt(2(exp(2t)-1))]
 <=sqrt(1+2H^2/pi)=1.165374884729.                  (17)
```

This follows by taking the minimum of `erf<=1` and
`erf(x)<=2x/sqrt(pi)` and optimizing their intersection. Every missing
return belongs to the first-side event, so no renewal multiplier is needed:

```text
S_infinity<=S_X+1.165374884729 p_X.                 (18)
```

Combining (16), (18), and the existing trace constant gives the corrected
stored-matrix rows

```text
X       direct response      x-corrected response
4.20    0.619681476353        0.622430322078
5.25    0.620016242618        0.620037342720
6.30    0.620371178274        0.620371231439.       (19)
```

## Scope

This closes the analytic side-excursion mechanism: the correction includes
arbitrarily many side excursions and later returns. It does not turn (19)
into a continuum return bound because `R_X` is still the symmetrized stored
FEM kernel. The next gates are:

1. polygonal-circle geometry error;
2. invariant-weighted lumped mass and stiffness consistency;
3. boundary-flux trace consistency under mesh refinement;
4. only then, uniform extension over `0<=rho<=1`.

The interval calculation and corrected rows are reproduced by
`scripts/neutral_strip_x_exit_correction_audit.py`.
