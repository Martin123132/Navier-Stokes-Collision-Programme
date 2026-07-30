# Annular parallel-shear third internal-shell lemma

## 1. Result and scope

Let

```text
z=(u,lambda),
F=X+Y,
X=(E,A)=(-P[(u dot grad)u],-u dot grad lambda),
Y=(V,D)=(nu Delta u,-nu Delta lambda),
```

and let `g_N(s)` be the scalar generator along the repaired
parallel-shear restart family constructed in the preceding stages. Put

```text
mu=max(nu,nu^-1).
```

Within the stored Fourier normalization, this stage proves the
restart-time estimate

```text
|g_N'''(0)| <= C0 mu^13 N^11                    (1.1)
```

for every fixed `nu>0` and every odd `N>=3`. The integer `C0` is explicit
and is recorded in the audit result. It is deliberately enormous; its
purpose is finiteness and independence of `N`, not numerical sharpness.

This closes the depth-three internal-output obligation left by the
third-jet route guard. It does not prove

```text
sup_(0<=s<=T/N^2) |g_N'''(s)| <= C3 N^11.        (1.2)
```

The coefficients and packet geometry evolve when `s>0`, so (1.2) is a
separate dynamic bootstrap problem. In particular, this note proves no
parabolic-window turnaround, critical `L3` estimate, finite-time blowup,
or Navier-Stokes regularity theorem.

## 2. Exact third-tree ledger

Use the symmetric projected Euler bilinear operator

```text
B(x,y)=-1/2 P[(x dot grad)y+(y dot grad)x],
```

the velocity heat operator `H=nu Delta`, the scalar transport bilinear
operator `C`, and the scalar heat operator `D=-nu Delta`. Expanding the
first three state derivatives as rooted trees gives the following exact
tree counts and absolute coefficient masses:

```text
velocity block  tree count  coefficient mass
U00                       1                 1
U10,U11                   1                 1
U20                       1                 2
U21                       2                 3
U22                       1                 1
U30                       2                 6
U31                       4                12
U32                       4                 7
U33                       1                 1

weight block    tree count  coefficient mass
L00                       1                 1
L10,L11                   1                 1
L20                       2                 2
L21                       3                 3
L22                       1                 1
L30                       4                 6
L31                       9                12
L32                       6                 7
L33                       1                 1.
```

Inserting these trees into the pressure, velocity-Fisher, and
weight-self sectors produces:

```text
sector             heat r   atom counts by r   coefficient masses by r
pressure             0..3       20,49,44,13          120,300,244,64
velocity Fisher      0..3       11,26,22,6            60,144,111,27
weight self          0..3       recorded trees         60,144,111,27.
```

The total absolute functional coefficient mass is exactly `1412`.
These are symbolic tree identities. No finite spectral replay or
floating-point threshold enters the count.

The predecessor carrier ledger shows that 22 of its 28
sector/heat/incidence rows are already `O(N^10)` or lower. The only rows
that require attention here have four or six high velocity leaves. The
expanded assignment ledger is:

```text
sector/heat/high       route       assignments   coefficient mass
pressure/0/4           exception            19                156
pressure/0/4           protected           281               1644
pressure/0/6           protected            20                120
pressure/1/4           exception            11                 90
pressure/1/4           protected           234               1410
pressure/2/4           protected            44                244
Fisher/0/4             protected            55                300
Fisher/1/4             protected            26                144.
```

The 30 expanded pressure exceptions collapse to the 13
bounded-output slot families already listed in the route guard. Direct
counting bounds every such family by `O_nu(N^11)`.

## 3. Resonance geometry along a dependent path

In a protected row, choose a high leaf outside the two outer pressure
inputs. This keeps the degree-zero outer pressure projector fixed while
the vertex stencil is transferred to that dependent leaf. If `q` is a
vertex mode, resonance writes the dependent high frequency as

```text
k_d=q-fixed low modes-sum_(j != d) k_j.          (3.1)
```

Consider an Euler `B` node on the ancestry path from `k_d` to its tree
root. Let `S` be the set of high leaves in that subtree. After (3.1), its
output has the form

```text
r_S=q+fixed low modes-sum_(j notin S) k_j.       (3.2)
```

There are two cases.

1. If the high complement is empty, `r_S=O(1)` throughout the finite
   vertex stencil and is independent of the free high carriers. The
   associated Euler symbol is `O(1)`, one carrier power smaller than its
   generic `O(N)` bound. There is no output-shell sum.

2. If the high complement is nonempty, `r_S` is a free three-dimensional
   output. One can replace one free high frequency by `r_S` with unit
   lattice Jacobian and split `r_S` into a finite shell and dyadic shells.

The exhaustive dependent-path depths are:

```text
row                    depth 0  depth 1  depth 2  depth 3
pressure r=0,h=4           170       84       26        1
pressure r=0,h=6            14        5        1        0
pressure r=1,h=4           162       66        6        0
pressure r=2,h=4            32       12        0        0
Fisher r=0,h=4              46        8        1        0
Fisher r=1,h=4              23        3        0        0.
```

Exactly four protected four-high assignments have one bounded path
node: two in pressure `r=0` and two in Fisher `r=0`. No protected
assignment has more than one.

For every one of the 20 all-high pressure assignments:

```text
every path-node high complement is nonempty;
the complements are strictly nested along the path;
the maximum path depth is two.
```

Indeed, before resonance the path-node leaf sets are strictly increasing.
Taking their complements in the full six-leaf set makes the free shell
sets strictly decreasing. Choosing one leaf from each successive layer
turns the map from free high frequencies to the path outputs into a
block-triangular integer map with identity `3x3` diagonal blocks. Thus
shell multiplicity gains may be added in the six-high row.

No corresponding independence assertion is made for four-high rows. A
path layer can add only low leaves, so two four-high complements can
coincide. Section 6 deliberately uses only one factor there.

## 4. Boundary-safe packet differences

For the zero-extended one-dimensional sine sequence

```text
s_n=sin(pi n/(N+1)),  1<=n<=N,
```

direct summation gives the sufficient bounds

```text
||s||_1 <= 2N,
||Delta s||_1 <= 2,
||Delta^2 s||_1 <= 4pi/N.                       (4.1)
```

The gauged high packet is the tensor product of three such sequences
times a polarization multiplier `m(k)` homogeneous of degree `-1`.
Its support obeys `|k|>=2N`, so `m` is smooth on a fixed annulus and

```text
|partial^beta m(k)| <= M_|beta| N^(-1-|beta|).  (4.2)
```

Apply the discrete product rule to (4.1)-(4.2). For every
`alpha in {0,1,2}^3`,

```text
||Delta^alpha h_N||_1
 <= C_H N^(2-|alpha|).                           (4.3)
```

Thus each of the six possible tensor differences supplies one carrier
power in `l1`. Equation (4.3) is a zero-extension estimate. It does not
claim that the packet has six pointwise derivatives across its boundary.

After parity gauging, the three-dimensional `Phi` vertex is the tensor
product of

```text
(-1/4,1/2,-1/4).
```

It therefore transfers exactly two unit differences in each coordinate
to the dependent packet/kernel product. Discrete Leibniz expansion
distributes a total budget of six differences. If a difference lands on
an explicit polynomial frequency factor, that factor loses one carrier
power; this is the same gain as (4.3).

## 5. Internal Euler shell estimate

For output frequency `r`, write the projected Euler symbol as

```text
b_r(x,y)
 =-(i/2) P_r[(r dot x)y+(r dot y)x],   b_0=0.    (5.1)
```

It obeys the global bounds

```text
|b_r(x,y)| <= |r||x||y|,
|b_r(x,y)-b_s(x,y)| <= 7|r-s||x||y|. (5.2)
```

For a dyadic shell `K<=|r|<2K` with `K>=4p`, differentiating the rational
symbol away from zero and using the finite-difference integral formula
gives

```text
|Delta^p b_r(x,y)|
 <= E_p K^(1-p)|x||y|,   0<=p<=6.                (5.3)
```

The finitely many outputs with `K<4p` are bounded directly, including
the convention `b_0=0`.

An unrestricted internal output has `O(N^3)` choices. Restricting it to
`|r|~N^kappa` has `O(N^(3kappa))` choices and therefore gains
`3(1-kappa)` carrier powers. Relative to the generic degree-one
symbol, (5.3) gains `1+(p-1)kappa`. The combined gain is

```text
4+(p-4)kappa.
```

Minimizing over `0<=kappa<=1` gives

```text
p differences on one free Euler output gain at least min(p,4). (5.4)
```

At `p=4` all dyadic shells have the same power, so a factor
`log(32N)` can occur. For odd `N>=3`, the coarse inequality

```text
log(32N) <= 32N
```

absorbs it with one spare carrier power.

## 6. Protected four-high rows

A protected four-high row has naive optimized size `O_nu(N^12)`. Only one
carrier gain is needed for (1.1).

Expand the six vertex differences by the discrete Leibniz rule.

* If a difference reaches the packet profile, (4.3) gives one power.
* If a difference reaches a regular polynomial factor, its degree drops
  by one and gives one power.
* Otherwise at least one path `B` factor receives a difference. If its
  output is bounded as in case 1 of Section 3, the generic `O(N)` symbol
  is replaced by `O(1)`. If its output is free, apply (5.4) to that one
  output and bound all other path factors crudely.

This argument uses no product of four-high shell gains. At the one-power
endpoint there is no logarithmic loss:

```text
bounded output: no shell sum;
packet/polynomial: no shell sum;
free output with p=1: the largest dyadic shell dominates.
```

If `p=4` occurs, there are four raw gains, so one may absorb its logarithm
and retain three. Consequently every protected four-high pressure or
Fisher row satisfies

```text
O_nu(N^12) * N^-1 = O_nu(N^11).                 (6.1)
```

## 7. All-high pressure row

The all-high pressure row has naive optimized size `O_nu(N^14)`.
Section 3 proves that every realized path output is free and that the
path shells are jointly block-triangular. Hence gains can be added across
all path factors.

For path depth `d`, distribute the six differences among

```text
[packet profile, regular polynomial pool, d Euler outputs].
```

The gain of an allocation `(p_0,...,p_(d+1))` is

```text
p_0+p_1+sum_(j=2)^(d+1) min(p_j,4).             (7.1)
```

Exhausting the weak compositions of six gives:

```text
path depth  allocation count  minimum raw gain
0                         7                 6
1                        28                 4
2                        84                 4
3                       210                 4.
```

Only depths zero, one, and two occur in the all-high row. Thus the raw
gain is at least four. Spending one power on the possible `p=4` shell
logarithm gives

```text
O_nu(N^14) * N^-4 * N = O_nu(N^11).             (7.2)
```

Combining (6.1), (7.2), the 13 direct pressure exceptions, and the 22
automatic rows proves the exponent in (1.1).

## 8. Explicit constant

The static load and exact optimizer masses give

```text
|B_N| <= 64N,
a_N <= 57N/nu <= 64mu N,
t_N <= 183N/nu <= 256mu N,
||low velocity||_Fourier,l1 <= 256mu N,
||weight||_Fourier,l1 <= 256mu N.                (8.1)
```

All packet frequencies are below `4N`, every third-tree intermediate
frequency is below `32N`, and the maximum scalar differential order is
eight. The audit uses the majorants

```text
C_profile = 2^120 9!,
C_Euler   = 2^60  9!,
C_regular = 2^64  9!,
L=max(C_profile,C_Euler,C_regular).
```

It then defines

```text
C_diff = 32 * 9^6 * (49^3 L)^8,

C0 = 1412 * 64 * 32 * 32^8 * 256^9 * C_diff.    (8.2)
```

Here the factors respectively dominate the exact tree coefficient mass,
high/low assignments, resonant tuple constants, frequency powers,
low/weight Fourier masses, finite output shells, Leibniz allocations,
path factors, and the possible dyadic logarithm. Charging `49^3` inside
each of the eight possible factor slots deliberately covers simultaneous
finite internal outputs. The resulting `C0` has 422 decimal digits and
begins

```text
2746219328370245...
```

The powers of `nu`, `nu^-1`, low amplitude, and weight scale are all
dominated by `mu^13`. Equations (8.1)-(8.2) therefore make (1.1)
quantitative.

## 9. Reproducibility and next gate

Run:

```text
python work/ns_collision/scripts/annular_parallel_shear_third_internal_shell_lemma_audit.py
python -m unittest work/ns_collision/tests/test_annular_parallel_shear_third_internal_shell_lemma.py
```

The production record is:

```text
work/ns_collision/results/annular_parallel_shear_third_internal_shell_lemma_audit_v1.json
```

The next mathematical gate is dynamic, not another restart-time carrier
count. One must propagate packet difference and shell bounds along the
actual coupled Navier-Stokes/adjoint trajectory for
`0<=s<=T/N^2`, with a constant independent of `N`, and then verify that
the resulting `C3` permits a nonzero choice `T<=c2/(2C3)`.
