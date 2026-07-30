# Exterior return tail correction

## Purpose

The first averaged-entry theorem used an exponential space-time density
envelope as a convenient sufficient hypothesis. That envelope is impossible
in an unbounded exterior even for pure Brownian motion. This note corrects
the theorem to the natural summable polynomial class and calibrates the exact
three-dimensional half-space kernel.

The correction does not damage the energy-to-entry mechanism. It sharpens
the unresolved issue: exterior deformation must preserve a summable weighted
Poisson-kernel envelope, not manufacture exponential killing where none
exists.

## Exponential-tail no-go

For three-dimensional Brownian motion with generator `Delta`, started at
radius `R>a`, the first hitting time of the sphere of radius `a` has defective
density

```text
f(t)=(a/R)(R-a)/(2 sqrt(pi))
     t^(-3/2) exp[-(R-a)^2/(4t)].                     (1)
```

Its total mass is `a/R`, and

```text
f(t)~(a/R)(R-a)/(2sqrt(pi)) t^(-3/2).                (2)
```

Therefore no finite `C` and positive `kappa` can satisfy

```text
f(t)<=C exp(-kappa t)                                (3)
```

on an unbounded exterior. In particular, the exponential rows in the first
entry audit can describe a bounded storage domain or an additionally killed
excursion, but not the raw exterior return.

## General summable-envelope theorem

Let the unnormalized weighted return law obey

```text
dnu/(ds d sigma)<=rho(s),                            (4)
```

where `rho` need not be exponential. Split time into

```text
I_n=[n ell,(n+1)ell].                                (5)
```

The interval energy estimate for the positive overshoot `w` gives

```text
int_(I_n) h[w]
 <=F^2[ell/a_0^2+1/(a_0^3 m_0)],
a_0=1-alpha.                                         (6)
```

Combining the surface trace with (4)-(6),

```text
int |Tw|^2 dnu<=C_Sigma F^2 J_rho(alpha),            (7)

J_rho(alpha)=inf_(ell>0)
 [ell/a_0^2+1/(a_0^3m_0)]
 sum_(n>=0) sup_(s in I_n)rho(s).                    (8)
```

Thus the required hypothesis is the summability of the interval suprema.
Every polynomial envelope `rho(s)=O(s^(-p))` with `p>1` works. The proof
still uses the unnormalized law, so nonreturns remain contraction.

## Exact half-space benchmark

At boundary distance `d`, the heat Poisson kernel of the three-dimensional
half-space is

```text
K(t,y)=d(4pi)^(-3/2)t^(-5/2)
       exp[-(d^2+|y|^2)/(4t)].                       (9)
```

Its spatial supremum is attained at `y=0`; its peak time is `d^2/10`. For
`d=1`, direct integration recovers total hitting mass one. Rigorous interval
suprema plus an integral-test tail give, at `alpha=0`,

```text
ell_*=0.32200834,
J_half=0.45308836.                                  (10)
```

Using the certified finite-energy barrier and `C_Sigma=1.413882673168`, the
conditional one-error thresholds are

```text
||q||_(3/2)<0.13064831,
||e||_3      <0.03517836.                            (11)
```

These are Brownian flat-interface calibrations, not cylindrical or
Navier-Stokes bounds.

## Lowering the spatial norm

The pointwise density envelope is stronger than necessary. For every
zero-boundary `v`, radial integration gives

```text
||T v||_4^4
 <=4 int_(1<r<2)|v|^3|grad v| dx
 <=4 ||v||_6^3||grad v||_2.                          (12)
```

The sharp three-dimensional Sobolev inequality and Poincare therefore imply

```text
||T v||_4^2<=C_4 h[v],
C_4=2 S_3^(3/4)c_A=0.674148137961.                   (13)
```

If the return density has only a spatial `L2(Sigma)` envelope

```text
||k(s,.)||_2<=rho_2(s),                              (14)
```

Holder pairs `k` with `(Tw)^2` and gives the same theorem with `C_4` and
`rho_2`. For the exact half-space kernel,

```text
||K(t,.)||_2
 =d/(4sqrt(2)pi)t^(-2)exp[-d^2/(4t)].                (15)
```

At `d=1`, its interval factor and conditional budgets are

```text
ell_*=0.42059264,        J_2=0.52343692,
||q||_(3/2)<0.17416208,
||e||_3      <0.04739843.                            (16)
```

This `L2` kernel target is preferable for the exterior cylinder: it demands
less angular pointwise control and gives a smaller trace constant.

## Deformation obstruction

Multiplying the Brownian return kernel by a constant positive deformation
weight `exp(ct)` destroys every polynomial tail. Because (1) has no positive
exponential moment,

```text
int_0^infinity exp(ct)f(t)dt=infinity,       c>0.     (17)
```

This is the time-resolved form of the earlier weighted-return obstruction.
A bound on total unweighted return probability is insufficient.

The axisymmetric affine stress test has a more favorable answer in the
spatial `L2` density norm once the finite axial patch is retained. Its
outward axial OU spreading exactly cancels the radial deformation exponent,
leaving a summable killed transverse-OU tail. This exact model calculation is
in `affine_exterior_axial_compensation.md`; it does not yet cover general
time-dependent or nonaffine exterior drift.

A useful weighted barrier can target polynomial moments. For

```text
h(s,r)=(1+s)^gamma(L/r)^beta,                        (18)
```

the exact identity in three dimensions is

```text
(partial_s+nu Delta+b.grad+c)h/h
 =gamma/(1+s)
  +beta[nu(beta-1)-b.x]/r^2+c.                       (19)
```

The extra positive term records the price of a `gamma`-moment. It cannot be
paid at spatial infinity by static Brownian capacity alone; signed radial
drift, deformation cancellation, scale motion, or a bounded storage/split
mechanism must supply it.

## Revised gate

The next object is the actual weighted exterior-cylinder Poisson kernel. The
axisymmetric affine model is now positive, but a successful estimate must
provide a summable pointwise envelope, or directly
bound the interval-supremum series in (8), while retaining:

1. the Navier-Stokes drift relative to the moving mollified frame;
2. the exterior deformation weight;
3. the finite axial entry patch;
4. the unnormalized nonreturn mass.

The Brownian formulas, exponential no-go, `L2` trace improvement,
interval-supremum factors, and conditional budgets are reproduced by
`scripts/exterior_return_tail_gate.py`.
