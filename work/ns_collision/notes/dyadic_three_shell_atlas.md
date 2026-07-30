# Dyadic three-shell atlas and the localized transfer gate

Status: the cubic shell geometry, localized transfer identities, and a
summable high-high-low amplitude envelope are now explicit. Global shell
transfer is conservative, but fixed-vertex shell telescoping fails because
localization converts its defect into physical boundary flux. Equal
eight-cell weights cancel, while nonconstant nonnegative weights can retain
the bad channel. The amplitude bound is perturbative only at small global
Reynolds size. No regularity conclusion is claimed.

## 1. Cubic frequency constraint

Every kinetic contribution to a vertex load has three velocity frequencies
and one partition frequency:

```text
k_1+k_2+k_3+r=0.                                  (1.1)
```

The same is true for pressure. If `q=k_1+k_2` is the pressure output, then
the load constraint `q+k_3+r=0` is again (1.1).

For the tensor partition at frequency `m`,

```text
|r|<=R=sqrt(3)m.                                   (1.2)
```

Order the three velocity magnitudes as

```text
M_1>=M_2>=M_3.                                     (1.3)
```

The triangle inequality gives the exact support rule

```text
M_1
 <=M_2+M_3+R
 <=2M_2+R,

M_2>=(M_1-R)/2.                                   (1.4)
```

If `M_1>=2R`, then

```text
M_2>=M_1/4.                                        (1.5)
```

For disjoint dyadic annuli `[2^j,2^(j+1))`, (1.5) means the largest two
shell indices differ by at most two.

## 2. Exact interaction atlas

Above the partition scale, all occupied cubic interactions reduce to:

```text
HHH: all three velocity scales are comparable;

HHL: two comparable high scales and one possibly much lower scale. (2.1)
```

A genuinely separated `HLL` interaction is impossible: its single high
frequency could not be balanced by two low frequencies and the fixed
partition stencil.

The `HHH` pressure block can be grouped into a bounded annulus and placed
under the complete self-shell pressure theorem, with finite overlap between
neighboring annuli. The genuinely nonlocal block is `HHL`.

An exhaustive finite support stress using carriers `16,32,64,128` finds
`120` ordered occupied triples. Its largest ratio `M_1/M_2` is
`2.001952172256`, below the theorem bound `4`.

## 3. Localized shell transfer identity

For divergence-free `a`, define

```text
T_Phi(a;b,c)
 =mean[Phi c dot (a dot grad)b].                   (3.1)
```

Integration by parts gives

```text
T_Phi(a;b,c)+T_Phi(a;c,b)
 =-mean[(a dot grad Phi)(b dot c)].                (3.2)
```

For `Phi=1`, the right side vanishes:

```text
T_1(a;b,c)=-T_1(a;c,b).                           (3.3)
```

Thus global shell transfer is exactly antisymmetric. Localization does not
destroy conservation; it moves the uncancelled amount to a spatial
partition boundary.

For one high field `H` and one low field `L`, the three kinetic `HHL`
transfers satisfy

```text
T_Phi(L;H,H)
+T_Phi(H;L,H)
+T_Phi(H;H,L)

=-mean[
   ((|H|^2/2)L+(L dot H)H) dot grad Phi].          (3.4)
```

The sparse audit verifies (3.2) over all eight vertices to
`1.57e-17`, reconstructs (3.4) to `7.81e-18`, and verifies global
antisymmetry exactly.

## 4. Why pure shell telescoping fails locally

Equation (3.3) can telescope transfers in global shell energies. Equation
(3.2) shows why the same argument cannot make a fixed-cell local flux
vanish: the remainder is the physical kinetic boundary flux.

Pressure has the same spatial structure:

```text
mean[Phi U dot grad p]
 =-mean[p U dot grad Phi].                         (4.1)
```

The preceding modulated-wave family proves that the complete signed `HHL`
boundary flux, after kinetic and all pressure terms are combined, tends to
`1/144`. Therefore the fixed-vertex sum cannot be a pure internal
shell-index difference whose compactly supported total vanishes.

This does not contradict global conservation. It identifies where the
conserved transfer exits: through the spatial cell boundary.

## 5. Exact eight-cell structure

Let `b_v` be the complete `HHL` vertex load of the two-sideband family. At
carrier `64`, the only occupied partition frequencies are

```text
r=(1,1,1) and -r.
```

Consequently the eight loads form the pure top Walsh character

```text
b_v=chi_{123}(v)b_{+++},

chi_{123}(v)=v_1v_2v_3.                            (5.1)
```

The exact audit gives

```text
b_{+++}=0.007048320351647,

sum_v b_v=0,                                       (5.2)
```

and every Walsh coefficient except order three is zero.

Equal cell weights therefore cancel. But take the nonnegative selector

```text
w_v=[1+chi_{123}(v)]/2.                            (5.3)
```

Then

```text
sum_v w_v b_v
 =4b_{+++}
 =0.028193281406587,

sum_v |b_v|
 =8b_{+++}.                                        (5.4)
```

The selector retains exactly one half of the `L1` load. Complete-cell
cancellation alone cannot control arbitrary nonconstant nonnegative
coefficients.

## 6. Coherent accumulation across high shells

Place the same fixed low modulation under carriers

```text
H=16,32,64,128,256.                                (6.1)
```

Cross terms between distinct high carriers cannot meet the fixed
low/stencil resonance. The complete vertex load is therefore the exact sum
of the five individual loads:

```text
number of shells    combined load

1                   0.006660287038012
2                   0.013694262551373
3                   0.020742582903020
4                   0.027753726547925
5                   0.034735197530534.             (6.2)
```

The largest sum-versus-individual residual is below `7e-18`. There is no
automatic sign alternation or high-shell telescoping bonus.

Each high field has Fourier `L2` energy proxy `4`, so the accumulation in
(6.2) is linear in the high-shell energy. This points to the correct
surviving estimate.

## 7. Single-low-shell amplitude estimate

Write

```text
a_J=||u_J||_2.                                     (7.1)
```

For `L>=m` and `H>=4L`, the complete `HHL` load obeys

```text
|B_(v,L;HHL)|
 <=C m L^(3/2)a_L
      sum_(H>=4L)a_H^2.                            (7.2)
```

The four terms are controlled as follows.

For kinetic transport, Bernstein gives

```text
||u_L||_infinity<=CL^(3/2)a_L,                    (7.3)
```

and the two high factors contribute `a_H^2`.

For `p[H,H]L`, Fourier support of `L grad Phi_v` restricts the pressure
output to size `O(L)`. The smooth low double-Riesz kernel has
`L1-to-L2` norm `CL^(3/2)`, while

```text
||u_H tensor u_H||_1<=a_H^2.                      (7.4)
```

For cross pressure `p[L,H]H`, the pressure output remains at scale `H`.
`L2` Riesz boundedness and (7.3) give

```text
||p[L,H]||_2
 <=C||u_L||_infinity a_H,                         (7.5)
```

and the final high factor contributes another `a_H`.

Comparable high neighbors have finite multiplicity and are absorbed using
`a_Ha_(H')<=(a_H^2+a_(H')^2)/2`. This proves (7.2).

## 8. Dyadic summation

Define

```text
E=sum_J a_J^2=||u||_2^2,

D=sum_J J^2a_J^2 approximately ||grad u||_2^2.    (8.1)
```

For every low shell,

```text
sum_(H>=4L)a_H^2
 <=L^(-2)D.                                       (8.2)
```

Therefore

```text
sum_L |B_(v,L;HHL)|

 <=C mD sum_(L>=m)L^(-1/2)a_L.                    (8.3)
```

Cauchy-Schwarz and the dyadic geometric series give

```text
sum_(L>=m)L^(-1/2)a_L
 <=E^(1/2)[sum_(L>=m)L^(-1)]^(1/2)
 <=sqrt(2/m)E^(1/2).                              (8.4)
```

Combining (8.3)-(8.4),

```text
sum_L |B_(v,L;HHL)|
 <=C sqrt(2m)||u||_2||grad u||_2^2.               (8.5)
```

Four deterministic finite-sequence replays satisfy every intermediate tail
and Cauchy bound and preserve the cubic scaling exactly.

## 9. Cell-weighted extension

For nonnegative cell coefficients `w_v`, set

```text
W=sum_v w_v Phi_v.                                 (9.1)
```

The constant component of `w` is invisible because

```text
sum_v grad Phi_v=0.                                (9.2)
```

Repeating (7.2) replaces `m` by

```text
||grad W||_infinity,                              (9.3)
```

which depends only on nonconstant Walsh coefficients, equivalently edge
variation on the cell cube. This is the natural joint scale-cell norm.

## 10. Large-data gate

Viscosity pays `nu D`. Equation (8.5) is directly absorbed only under

```text
C sqrt(m)||u||_2<nu.                               (10.1)
```

This is a small global Reynolds condition. Under amplitude scaling
`u->Au`, the `HHL` load scales as `A^3`, while `nu D` scales as `A^2`.
Leray energy control therefore does not turn (8.5) into a universal
large-data estimate.

Established:

- the exact `HHH/HHL` atlas;
- global shell-transfer antisymmetry and its localized boundary defect;
- failure of pure fixed-vertex shell telescoping;
- exact equal-weight eight-cell cancellation;
- a pure top-Walsh `HHL` channel retained by nonnegative weights;
- coherent accumulation across separated high shells;
- the shell-amplitude bound (7.2) and dyadic sum (8.5).

Still open:

- a joint scale-cell Carleson improvement over (8.5);
- time-integrated compensation using viscous occupation;
- a critical signed large-data bound;
- low-regularity passage, exceptional-set removal, and regularity.

## 11. Next theorem target

The next candidate must beat the coefficient in (10.1). Define the
cumulative high-shell Reynolds stress above each low scale and test whether
its joint scale-cell flux is a Carleson measure controlled by dissipation,
with cell dependence measured by `||grad W||` or Walsh edge variation.

If a pointwise Carleson estimate fails, retain time and ask whether the
large values of that joint flux have summable occupation under the viscous
energy budget. Another shell-only cancellation search would simply repeat
the no-go proved here.

The identities and finite-mode replays are generated by
`scripts/dyadic_three_shell_atlas_audit.py`.
