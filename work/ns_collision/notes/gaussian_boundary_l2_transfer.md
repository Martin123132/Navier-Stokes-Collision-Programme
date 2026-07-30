# Gaussian boundary L2 transfer

## Purpose

The pointwise finite-cylinder route demands a Kato norm that is not controlled
by critical `L^(3/2)` mass. This note changes the interface norm rather than
strengthening the PDE hypothesis prematurely.

The piecewise ideal generator has a natural Gaussian `L^2` boundary space.
In that space the full axial visit operator is exactly diagonal, and Markov
branching/interface transfer is contractive when measured against the actual
evolving probability laws. This produces a plausible operator architecture
compatible with the critical interior form estimate.

It does not yet prove closure for Navier-Stokes. The two remaining constants
are an entry/exit density comparison and an interior-form-to-boundary-operator
estimate.

## Global reversible measure

In cylindrical variables, let

```text
w(rho,y)=exp[-R_* y^2+(R_*/2)min(rho^2,1)].            (1)
```

This density is continuous across `rho=1`. Away from that interface,

```text
w^(-1) div(w grad)
 =Delta+R_*(x_1 partial_1+x_2 partial_2)-2R_*y partial_y
                                                        in the core,
 =Delta-2R_*y partial_y                                 in the shell. (2)
```

Thus the drift part of the piecewise model is symmetric in `L^2(w dx)`.
The core stretching `2R_* 1_(rho<1)` is a real multiplication potential and
preserves symmetry. On each radial interface, the induced normalized axial
measure is

```text
dmu_h(y)=Z_h^(-1) exp(-R_*y^2)dy,   -h<y<h.            (3)
```

## Exact boundary operator norm

Let `phi_n` be the Dirichlet axial OU modes, orthonormal in (3), with
eigenvalues `zeta_n`. Separation of variables gives the outer-to-inner visit
operator `B` as

```text
B phi_n=U(zeta_n) phi_n.                               (4)
```

The radial multiplier `U(zeta)` is positive and strictly decreasing in
`zeta` by the Feynman-Kac comparison principle. Therefore

```text
||B||_(L2(mu_h) to L2(mu_h))=U(zeta_0).                (5)
```

This explains the earlier discrepancy. The principal-mode calculation was a
surrogate for the pointwise constant-boundary payoff, which sums all modes at
one point. It is the exact norm of the complete boundary operator in the
Gaussian `L^2` space.

For two independent histories,

```text
||B tensor B||_2=U(zeta_0)^2.                          (6)
```

The first 81 audited multipliers are positive and strictly decreasing at all
working geometries. The second/principal ratios range from approximately
`0.126` to `0.326`, leaving a substantial axial spectral gap.

## L2 renewal budget

Using the same true-split and exterior-return factors as before, define

```text
C_2=U(zeta_0)^2[gamma+eta^(-2)].                       (7)
```

The exact ideal `L^2` benchmarks are:

| `R_*` | half-height `h` | `U(zeta_0)` | `C_2` | pointwise `C_infinity` |
|---:|---:|---:|---:|---:|
| 0.5 | 1.5 | 0.856817 | 0.378904 | 0.493333 |
| 0.5 | 1.75 | 1.04334 | 0.561826 | 0.691792 |
| 0.5 | 2.0 | 1.19705 | 0.739570 | 0.864787 |
| 1.0 | 1.0 | 0.631959 | 0.212980 | 0.288761 |
| 1.0 | 1.2 | 1.00311 | 0.536604 | 0.684048 |

Every `L^2` criterion is strictly smaller than its pointwise counterpart.
No modes have been discarded in deriving (5); the finite mode family only
audits the monotonic multiplier law and the numerical values.

## Markov transfer in evolving measures

Let `P` be a forward column-stochastic kernel, let `mu` be the source law,
and set `nu=P mu`. The backward observable map is `P^T`. Jensen gives, for
every `p>=1`,

```text
||P^T f||_(Lp(mu)) <= ||f||_(Lp(nu)).                  (8)
```

Stationarity and reversibility are unnecessary. The measure simply evolves
with the process. The independent pair kernel satisfies the same statement
in `mu tensor mu` and `nu tensor nu`.

The audit verifies (8) for the asymmetric cube interface semigroup. Both the
one-history and 64-state pair `L^2` norms are one to roundoff. Child
expectation with probabilities `p_i`, and its pair lift `p_i p_j`, also have
exact norm one.

This is the correct way to retain conservative branching in `L^2`: carry the
actual probability law through every Markov phase rather than repeatedly
projecting onto one fixed reference weight.

## Entry and exit mismatch

The spectral visit uses the Gaussian measure `m`, while the actual hitting
laws entering and leaving a buffered core are generally `nu_out` and
`nu_in`. If the relevant Radon-Nikodym derivatives are bounded, one complete
one-history visit costs

```text
M = [ess sup(dm_out/dnu_out)
     ess sup(dnu_in/dm_in)]^(1/2).                     (9)
```

The pair visit costs `M^2`, so (7) closes provided

```text
M^2 C_2<1,
M<1/sqrt(C_2).                                        (10)
```

The permitted one-history mismatches are:

| `R_*` | `h` | maximum `M` |
|---:|---:|---:|
| 0.5 | 1.5 | 1.62456 |
| 0.5 | 1.75 | 1.33413 |
| 0.5 | 2.0 | 1.16281 |
| 1.0 | 1.0 | 2.16686 |
| 1.0 | 1.2 | 1.36513 |

This is again a reason to keep geometric slack. The preferred compact
geometry allows about a `62%` one-history norm mismatch; the near-threshold
geometry allows only about `16%`.

As a stress test, converting the deliberately nonuniform child law from the
branching audit to the unrelated stationary cube-graph law and back costs
`3.40338` for one history and `11.5830` for the pair. That failure is not a
property of branching. It is the cost of imposing incompatible fixed
measures. Dynamic Markov measures avoid it exactly.

If an actual hitting law is singular relative to (3), (9) is infinite. The
buffer must therefore provide genuine smoothing before the Gaussian
conversion; this cannot be assumed at an instantaneous interface crossing.

## Relation to the critical form estimate

The earlier `L^(3/2)` inequality controls the symmetric interior quadratic
form in `L^2(w dx)`. Unlike point evaluation, the trace from `H^1` in the
three-dimensional cylinder to `L^2` on its two-dimensional interfaces is
continuous. It is therefore reasonable to seek an estimate of the form

```text
||B_q||_(L2(mu_h) to L2(mu_h))
 <= M_form(alpha) ||B_0||_2,                           (11)
```

where `alpha<1` is the relative critical form bound for the positive
non-affine potential. Establishing (11) with an explicit constant is the next
analytic task. It requires a boundary Poisson/trace estimate for the
piecewise weighted operator; it does not follow merely by naming the form
bound.

## Revised proof gate

The candidate generation cycle is now:

```text
dynamic physical probability L2
 -> one entry conversion to Gaussian boundary L2
 -> complete perturbed buffered visit
 -> one exit conversion to dynamic physical probability L2
 -> conservative Markov transfer and true split.       (12)
```

Two obligations remain:

1. bound (9) for the actual finite-cylinder hitting laws within the values in
   the table;
2. derive (11) from the local Navier-Stokes coherence error and the existing
   critical form inequality.

The natural measure, complete visit spectrum, Markov contractions, and
measure-mismatch budgets are reproduced by
`scripts/gaussian_boundary_l2_transfer_audit.py`.
