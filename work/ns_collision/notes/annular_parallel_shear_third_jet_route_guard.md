# Annular parallel-shear third-jet route guard

## 1. Result

Write the coupled state and vector field as

```text
z=(u,lambda),
F=X+Y,
X=(E,A)=(-P[(u dot grad)u],-u dot grad lambda),
Y=(V,D)=(nu Delta u,-nu Delta lambda).
```

This stage gives an exact third-flow heat split, an exhaustive carrier
ledger, and an independent finite spectral replay. It proves:

```text
28 sector/heat/incidence rows are exhaustive;
22 rows are O(N^10) or lower by direct counting;
6 high-incidence rows require compatible differences;
13 bounded-output pressure families are possible;
only 5 of those families can reach O(N^11).
```

It does not yet prove a complete `O(N^11)` third-derivative bound. The
remaining restart-time step is a depth-three discrete
Leibniz/dyadic-shell lemma for nested Euler and Leray outputs. A second,
logically separate step must propagate the resulting estimate along the
evolving trajectory on `0<=s<=T/N^2`.

## 2. Exact third-flow split

The second state derivative separates by heat count:

```text
z20=DX X,
z21=DX Y+DY X,
z22=DY Y.
```

Because `Y` is linear and `X` is quadratic/bilinear, the third state
derivative is

```text
z30=D^2X[X,X]+DX z20,

z31=2D^2X[X,Y]+DX z21+DY z20,

z32=D^2X[Y,Y]+DX z22+DY z21,

z33=DY z22.                                      (2.1)
```

Substitution into

```text
g'''=D^3g[F,F,F]+3D^2g[F,z2]+Dg[z3]
```

gives the four exact scalar blocks

```text
r=0:
  D^3g[X,X,X]+3D^2g[X,z20]+Dg[z30],

r=1:
  3D^3g[X,X,Y]
 +3D^2g[Y,z20]+3D^2g[X,z21]+Dg[z31],

r=2:
  3D^3g[X,Y,Y]
 +3D^2g[Y,z21]+3D^2g[X,z22]+Dg[z32],

r=3:
  D^3g[Y,Y,Y]+3D^2g[Y,z22]+Dg[z33].             (2.2)
```

The finite engine implements (2.1) and also computes the unsplit formula

```text
z3=D^2X[F,F]+DX z2+DY z2.
```

At `N=5` the two coefficient arrays agree to

```text
velocity: 5.83e-11,
weight:   2.04e-14.
```

## 3. Degrees and carrier orders

After three time derivatives, a block with `r` heat directions has:

```text
sector             velocity degree  weight degree  differential order
pressure                     6-r              1                 <=4+r
velocity Fisher              5-r              1                 <=5+r
weight self                  3-r              3                 <=5+r.
```

The annular incidence rule permits only an even number `h` of high
velocity leaves.

For `h>=2`, the high tuple sum after inserting the packet coefficients has
power

```text
2h-3.
```

Adding the low optimizer amplitudes, weight scale, and differential order
gives the same naive exponent in all three sectors:

```text
naive optimized power=h+8.                       (3.1)
```

Consequently:

```text
h=2: O(N^10), no extra gain required;
h=4: naive O(N^12), two gains required;
h=6: naive O(N^14), four gains required.
```

The six dangerous group rows are exactly:

```text
sector             heat count  high leaves  gains required
pressure                    0            4               2
pressure                    0            6               4
pressure                    1            4               2
pressure                    2            4               2
velocity Fisher             0            4               2
velocity Fisher             1            4               2.
```

Every other one of the 28 rows is already `O(N^10)` or lower.

## 4. Projector-safe rows

After parity gauging, the base vertex `Phi` supplies two differences in
each coordinate, hence six differences in total.

If a high leaf occurs outside the two outer pressure inputs, choose an
external high leaf as the dependent resonance variable. The outer
pressure output then either:

1. remains fixed; or
2. remains at carrier scale, where the degree-zero projector has regular
   finite differences.

An explicit power of the bounded vertex variable `q` is not a lost
carrier estimate. A term containing `q^alpha` has already lost
`|alpha|` carrier powers compared with the leading homogeneous term.
Thus the discrete product rule trades explicit vertex degree against
carrier degree.

This leaves the expected two gains on protected four-high rows and four
gains on the all-high six-leaf row. What is not yet written is the
complete depth-three shell proof when several differences cross nested
Euler/Leray outputs close to zero. The existing second-jet
two-difference lemma cannot simply be cited for that stronger statement.

## 5. Bounded-output pressure exceptions

Use `U_j^r` and `Lambda_j^r` for the heat-count-`r` component of the
`j`th state derivative. A bounded-output exception is possible when all
four high leaves lie inside the two pressure inputs and every velocity
leaf in the test and transported-weight slots is low.

Up to symmetry of the two pressure slots, the complete inventory is:

```text
r  pressure pair       test       weight       power bound
0  U0^0,U2^0           U0^0       Lambda1^0           10
0  U0^0,U2^0           U1^0       Lambda0^0           10
0  U0^0,U3^0           U0^0       Lambda0^0           11
0  U1^0,U1^0           U0^0       Lambda1^0           10
0  U1^0,U1^0           U1^0       Lambda0^0           10
0  U1^0,U2^0           U0^0       Lambda0^0           11

1  U0^0,U2^0           U0^0       Lambda1^1            9
1  U0^0,U2^0           U1^1       Lambda0^0            9
1  U0^0,U3^1           U0^0       Lambda0^0           11
1  U1^0,U1^0           U0^0       Lambda1^1            9
1  U1^0,U1^0           U1^1       Lambda0^0            9
1  U1^0,U2^1           U0^0       Lambda0^0           11
1  U1^1,U2^0           U0^0       Lambda0^0           11.
```

There is no heat-count-two exception. At `r=2` the pressure sector has
velocity degree four. The three total jet orders cannot place both heat
operations outside the pressure pair, so that pair has capacity at most
three and an external high leaf is forced.

For an exceptional family, let `m_p` be the total high differential order
inside the pressure pair. Direct counting gives

```text
four-high tuple and coefficient power       5,
low velocity amplitude power              2-r,
weight scale power                          1,
pressure-side differential power          m_p.
```

Hence

```text
optimized power <=8-r+m_p<=11.              (5.1)
```

This estimate never differentiates the bounded outer projector. It is the
reason an `O(N^11)` target is natural even if no additional cancellation
occurs in the five saturating families.

## 6. The correct Taylor target

The preceding stage proves, for a constant `c2>0`,

```text
g_N''(0)<=-c2 N^9
```

for all sufficiently large `N`. Suppose one proves the uniform estimate

```text
sup_(0<=s<=T/N^2) |g_N'''(s)| <= C3 N^11.        (6.1)
```

Then

```text
g_N''(s)
 <=-(c2-C3 T)N^9.
```

Choosing

```text
0<T<=c2/(2C3)
```

gives

```text
g_N''(s)<=-(c2/2)N^9.                            (6.2)
```

Thus `O(N^11)` with an explicit uniform constant is sufficient.
The previously stated target `o(N^11)` is stronger than necessary.

With `delta=T/N^2`, integrating (6.2) yields

```text
integral_0^delta g_N(s) ds
 <=g_N(0)delta+g_N'(0)delta^2/2
   -c2 N^9 delta^3/12.                           (6.3)
```

The last term is

```text
-(c2 T^3/12)N^3,
```

whereas the stored `g_N(0)=O(N^3)` and `g_N'(0)=O(N^5)` contribute only
`O(N)` after the corresponding integrations. A uniform version of
(6.1) would therefore make the early parabolic-window behavior decisively
negative for this restart family.

## 7. Finite spectral replay

The production replay uses:

```text
N=5,
(a_yz,a_xy,t)=(0.7,0.7,0.9),
nu=1,
dealias factors 14 and 16.
```

The exact multilinear heat-block totals on the `14K` grid are:

```text
r=0      -0.09128004971843723
r=1     371.6106292397538
r=2    7134.080941621970
r=3  389894.2588980738
```

Their sum is

```text
g'''=397399.8591888858.
```

A centered third difference with Richardson extrapolation gives

```text
397399.8413929942,
```

with relative residual

```text
4.48e-8.
```

Changing the padding from `14K` to `16K` changes the total by

```text
4.39e-16
```

relatively. The maximum relative divergence residual among the four
third-velocity heat blocks is `5.06e-11`.

These checks validate the chain-rule and heat-block assembly. Their
finite signs and magnitudes are not used as asymptotic evidence.

## 8. Scope and next gate

Established:

```text
the exact third-flow heat split;
the 28-row degree/incidence ledger;
the 22 automatically subcritical rows;
the complete 13-family bounded-output inventory;
the five possible O(N^11) saturating families;
and an independent finite spectral replay.
```

Still open:

```text
the depth-three compatible-difference shell lemma;
a complete restart-time O(N^11) bound with a constant C3;
uniform propagation of that bound on 0<=s<=T/N^2;
critical L3 control;
finite-time blowup;
and Navier-Stokes global regularity.
```

The next proof stage should close the protected four- and six-high rows
one tree topology at a time, retaining an explicit constant. Only then
should the work move to the dynamic parabolic-window bootstrap.

## 9. Reproducibility

Run:

```text
python work/ns_collision/scripts/annular_parallel_shear_third_jet_route_guard_audit.py
python -m pytest -q work/ns_collision/tests/test_annular_parallel_shear_third_jet_route_guard.py
```

The production record is
`results/annular_parallel_shear_third_jet_route_guard_audit_v1.json`.
