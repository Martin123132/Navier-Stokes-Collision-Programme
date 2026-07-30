# Annular parallel-shear heat-block exclusion

## 1. Result

Write the coupled state as `z=(u,lambda)` and split its vector field into

```text
X(z)=(E,A)=(-P[(u dot grad)u],-u dot grad lambda),
Y(z)=(V,D)=(nu Delta u,-nu Delta lambda).
```

For the complete generator `g`, the second derivative separates exactly:

```text
pure E/A:
  D^2g[X,X]+Dg[DX X],

one heat:
  2D^2g[X,Y]+Dg[DX Y+DY X],

two heat:
  D^2g[Y,Y]+Dg[DY Y].                            (1.1)
```

The preceding stage proves that every viscosity-bearing pure `E/A` Fisher
row is `o(N^9)`. This stage proves

```text
one-heat block=O(N^8)=o(N^9),
two-heat block=O(N^8)=o(N^9).                    (1.2)
```

The complete second jet therefore has the already certified strict negative
inviscid-pressure limit

```text
J''_N/N^9
 ->-[1024 beta_0^2/(405 nu^2)]
    [sqrt(3)/10] ||v_y||_2^2
 <0.                                             (1.3)
```

This is a restart-time second-derivative theorem. It is not yet a uniform
Taylor estimate on a parabolic time window.

## 2. Exhaustive channel partition

The finite chain-rule engine has 20 second-variation channels and 69 atomic
subterms. Two additional `second::aggregate::*` polynomial rows are derived
diagnostics and are not counted as atoms.

The exact partition is

```text
block                   pressure  velocity Fisher  weight self  total
pure E/A                       9                5            7     21
one heat                     14                8            9     31
two heat                      8                4            5     17
                                                                  --
                                                                  69
```

The channel sets are:

```text
pure E/A:
  H_uu[E,E]
  2H_u_lambda[E,A]
  H_lambda_lambda[A,A]
  D_u[u2_EE]
  D_lambda[lambda2_E0]
  D_lambda[lambda2_0A]

one heat:
  2H_uu[E,V]
  2H_u_lambda[E,D]
  2H_u_lambda[V,A]
  2H_lambda_lambda[A,D]
  D_u[u2_EV]
  D_u[u2_VE]
  D_lambda[lambda2_V0]
  D_lambda[lambda2_0D]
  D_lambda[lambda2_DA]

two heat:
  H_uu[V,V]
  2H_u_lambda[V,D]
  H_lambda_lambda[D,D]
  D_u[u2_VV]
  D_lambda[lambda2_DD].
```

These sets are pairwise disjoint and exhaust all 20 channels.

The one-heat accelerations are exactly

```text
DX Y+DY X:
  velocity =u2_EV+u2_VE,
  weight   =lambda2_V0+lambda2_0D+lambda2_DA.
```

The two-heat acceleration is

```text
DY Y=(u2_VV,lambda2_DD).
```

Thus the partition is the chain rule for `X+Y`, not a post hoc grouping.

## 3. Incidence and degrees

The nine block/category groups have fixed velocity and weight degrees:

```text
block       category          velocity degree  weight degree
pure E/A    pressure                        5              1
pure E/A    velocity Fisher                 4              1
pure E/A    weight self                     2              3
one heat    pressure                        4              1
one heat    velocity Fisher                 3              1
one heat    weight self                     1              3
two heat    pressure                        3              1
two heat    velocity Fisher                 2              1
two heat    weight self                     0              3.
```

The packet and all low/weight modes have the same first-coordinate
separation used by the predecessor incidence theorem. Hence:

```text
degree four: only 0, 2, or 4 high leaves;
degree three: only 0 or 2 high leaves;
degree two:   only 0 or 2 high leaves;
degree one:   no high leaf.
```

This immediately makes every heat block except the one-heat HHHH pressure
branch subcritical by a raw carrier count.

## 4. One-heat pressure HHHH branch

The pressure generator is linear in `lambda`. In the complete one-heat
combination, integrate every derivative in

```text
A, D, V0, 0D, DA
```

off the weight until the base vertex is undifferentiated `Phi`. The
resulting quartic velocity kernel has total differential order at most four:
one Euler derivative and one heat Laplacian, including the original
derivative on the weight.

After parity gauging, `Phi` supplies six tensor-product differences. The
degree-four vertex lemma from the preceding stage proves that every
multiindex of total degree at most four leaves at least two exact compatible
differences.

At fixed vertex output the raw count is

```text
free high tuples                    O(N^9),
four high coefficients              O(N^-4),
order-four kernel                   O(N^4).
```

The two compatible differences reduce this to `O(N^7)` at fixed weight
scale.

The outer degree-zero pressure projector is not differentiated. Every
one-heat HHHH atom has a high leaf on its test side:

```text
pressure second variation:  p[direction,direction] * test,
pressure first variation:   p[u,direction] * test,
pressure direction:         p[u,u] * direction.
```

Choose a high test leaf as the dependent resonance variable. A vertex shift
then changes that test side while the outer pressure output and projector
remain fixed.

If both compatible differences hit an internal projected Euler symbol, use
the full degree-one symbol rather than its pressure and advection pieces.
Its second difference is split by internal output:

```text
K<=|r|<2K.
```

As in the preceding Fisher exclusion, the shell contribution is

```text
O(N^5 K^2),
```

and its dyadic sum is `O(N^7)`. No `C^2` extension through output zero is
assumed.

Restoring `t_N=O(N)` gives

```text
one-heat pressure HHHH=O(N^8).                   (4.1)
```

For HHLL there are `O(N^3)` high pairs, coefficient product `O(N^-2)`,
and kernel at most `O(N^4)`. Its fixed-amplitude size is `O(N^5)`;
`a_N^2t_N=O(N^3)` again gives `O(N^8)`. The low-only term is at most
`a_N^4t_N=O(N^5)`.

## 5. Remaining one-heat rows

The one-heat velocity-Fisher group has velocity degree three and at most
five carrier derivatives. Its only high branch is HHL:

```text
high pair sum after coefficients     O(N),
kernel                               O(N^5),
a_N t_N                              O(N^2).
```

Therefore

```text
one-heat velocity Fisher=O(N^8).                  (5.1)
```

The one-heat weight-self group has velocity degree one. A one-high term
cannot return to bounded output, so only the fixed low field survives:

```text
one-heat weight self=O(a_N t_N^3)=O(N^4).         (5.2)
```

Equations (4.1)-(5.2) prove the first line of (1.2).

## 6. Two-heat rows

The two-heat pressure group has velocity degree three. Its HHL branch has
high-pair sum `O(N)` and at most four high-carrier derivatives from the two
velocity Laplacians. Thus its fixed-amplitude coefficient is `O(N^5)`, and

```text
two-heat pressure=O(N^5 a_N t_N)=O(N^7).         (6.1)
```

The two-heat velocity-Fisher group has velocity degree two. The worst atom
is

```text
H_uu[V,V] weighted Fisher,
```

with two `grad V` factors and therefore six high-carrier derivatives. The
HH high-pair sum after coefficients is `O(N)`, so

```text
two-heat velocity Fisher
 =O(N^7 t_N)
 =O(N^8).                                        (6.2)
```

The two-heat weight-self group has velocity degree zero and only fixed
weight modes:

```text
two-heat weight self=O(t_N^3)=O(N^3).             (6.3)
```

Equations (6.1)-(6.3) prove the second line of (1.2).

## 7. Complete second-jet limit

The 69 atomic subterms are now covered as follows:

```text
9 pure E/A pressure rows:
  strict negative N^9 limit already certified;

12 pure E/A Fisher rows:
  O(N^8) or lower;

31 one-heat rows:
  O(N^8) or lower;

17 two-heat rows:
  O(N^8) or lower.
```

Therefore every term outside the inviscid-pressure block is `o(N^9)`, and
(1.3) follows.

The positive double-heat pressure curvature remains present at order
`N^7`; it cannot change the strict negative `N^9` limit.

## 8. Scope

This stage proves:

```text
the exhaustive 69-subterm partition,
all viscosity-bearing second rows are o(N^9),
and the complete restart-time second jet has a negative N^9 limit.
```

It does not prove:

```text
a uniform Taylor remainder on 0<=s<=T/N^2,
that the adjoint/static optimizer remains frozen over that window,
critical L3 control,
finite-time blowup,
or Navier-Stokes global regularity.
```

The next gate is an exact third-jet degree/channel ledger followed by a
uniform integral-remainder estimate along the evolving
Navier-Stokes/adjoint pair.

## 9. Reproducibility

Run:

```text
python work/ns_collision/scripts/annular_parallel_shear_heat_block_exclusion_audit.py
python -m pytest -q work/ns_collision/tests/test_annular_parallel_shear_heat_block_exclusion.py
```

The production record is
`results/annular_parallel_shear_heat_block_exclusion_audit_v1.json`.
