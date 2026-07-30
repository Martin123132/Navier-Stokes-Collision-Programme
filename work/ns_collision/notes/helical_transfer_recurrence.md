# Helical Transfer Recurrence

Date: 2026-07-17

Status: exact trajectory decomposition and exact weak-amplitude Duhamel
expansion for the loss of the negative cumulative channel. This identifies a
specific recurrence mechanism but does not bound it for arbitrary solutions.

## 1. Seed Pair Versus Generated Modes

For the negative two-mode initial field, write the full Galerkin velocity as

```text
u(t)=u_seed(t)+u_generated(t),
```

where `u_seed` contains only the original wave pairs

```text
k=(1,0,0),       m=(1,1,0).
```

The quartic polynomial has the exact grouping

```text
X_s(u)=X_s(u_seed)+[X_s(u)-X_s(u_seed)].          (1.1)
```

The first term is evaluated using the full `4 by 4` helical pair matrix from
`quartic_transfer_helical_matrix_audit.py`. The second term contains every
quartic monomial with at least one generated-mode factor.

At the cumulative sign transition:

```text
 R       integral X_seed    integral X_generated    integral X_total
0.900    -0.00677810          0.00651076             -0.000267336
0.922    -0.00745664          0.00748132              0.0000246774
0.940    -0.00804822          0.00835999              0.000311768
1.000    -0.0102732           0.0119125               0.00163933
```                                                        

Thus the original pair remains cumulatively negative. The sign loss is caused
by generated-mode transfer overtaking it. Near the crossing, the seed term
has local power `R^3.95`, while the generated term has local power `R^5.74`.

The decomposition retains the physical rank-one constraint exactly. Across
all snapshots, the maximum pair-matrix residual was below `1e-14` and the
maximum Fourier rank-one residual was below `1e-14`.

## 2. Parity/Helicity Content of the Seed

At `R=0.922`, the integrated seed channels are

```text
symmetric homochiral                    0.000273573,
symmetric heterochiral                 -0.00750609,
antisymmetric homochiral diagonal      -0.00558433,
antisymmetric heterochiral diagonal     0.00322034,
antisymmetric interference              0.00213986.
```                                                     

Their sum is `-0.00745664`. The largest negative term is the coherent
symmetric heterochiral channel. This does not contradict the positivity of a
single pure heterochiral pair: the parity channel contains coherent
off-diagonal coupling between the `+-` and `-+` pair amplitudes.

## 3. Generated-Mode Hierarchy

Let

```text
q=m-k=(0,1,0),       r=m+k=(2,1,0).
```

For a full trajectory, define the exact inclusion-exclusion split

```text
X_s(u)=X_seed
      +Delta_q X+Delta_r X+Delta_(q,r) X
      +Delta_higher X.                              (3.1)
```

Here `Delta_q` is the increment from adding the difference-mode pair to the
seed, `Delta_r` is the corresponding sum-mode increment,
`Delta_(q,r)` is their non-additive interaction, and `Delta_higher` contains
all modes beyond that first generated shell.

The integrated values are

```text
component                         R=0.922        R=0.940
seed                             -0.00745664    -0.00804822
difference-mode increment         0.00503153     0.00560871
sum-mode increment                0.0000470244   0.0000524141
difference/sum interaction       -0.000105559   -0.000116882
higher-generation remainder       0.00250832     0.00281575
total                              0.0000246774   0.000311768
```                                                     

At `R=0.922`, the difference mode supplies `67.25 percent` of the positive
generated remainder. Higher generations supply `33.53 percent`; the sum mode
supplies only `0.63 percent`; and the first-shell interaction removes
`1.41 percent`.

This is the sharpest connection found so far to the original collision
intuition: the dominant feedback is carried by the Fourier difference or
beat mode `m-k`. It is not a literal fluid-particle collision statement.

## 4. Exact Weak-Amplitude Law

Set

```text
x=exp(-s*K^2),       R=A/(nu*K).
```

The exact heat-Duhamel hierarchy gives

```text
integral_0^infinity X_s(t)dt
 =nu^3*K^4*[c_4(x)*R^4+c_6(x)*R^6+O(R^8)].        (4.1)
```

The quartic seed coefficient is

```text
c_4=(1-x)^2*(x^3+2*x^2+3*x-11)/120<0.            (4.2)
```

Every order-five term cancels exactly. The difference-mode part of the
order-six feedback is

```text
c_6^q=(1-x)^2/4800
      *(17*x^3+34*x^2+51*x+263)>0.                (4.3)
```

The sum-mode order-six term is also positive but much smaller. Their first
interaction is negative at order six and positive at order eight. After
combining the complete first generated shell,

```text
c_6^first>0                                        (4.4)
```

for every `0<x<1`.

The next Duhamel field `u^(3)` contributes at the same order. Its numerator is

```text
3640*x^11+7280*x^10+10920*x^9+23335*x^8
+35750*x^7+48165*x^6+60580*x^5+72995*x^4
+45931*x^3+18867*x^2-8197*x+11213799.             (4.5)
```

It is strictly positive on `[0,1]`: the constant alone exceeds the magnitude
of its only negative monomial. Therefore

```text
c_6=c_6^first+c_6^second>0                         (4.6)
```

at every positive heat scale. Its fully positive combined numerator is
recorded in `weak_generated_transfer_audit.py`.

At `s=0.5`,

```text
c_4=-0.01060700105,
c_6^first=0.00985777027,
c_6^second=0.00442847814,
c_6=0.01428624842.                                (4.7)
```

The first shell supplies `69.00 percent` of `c_6`; the second supplies
`31.00 percent`, closely matching the trajectory decomposition. The formal
sixth-order crossing is

```text
R=sqrt(-c_4/c_6)=0.8616625288.                    (4.8)
```

The full trajectory crosses near `R=0.92`, showing that order-eight and higher
terms delay, but do not create, the sign conversion.

## 5. Consequence and Next Gate

The negative quartic channel cannot be used as a persistent cancellation
reservoir for this family. Navier-Stokes mode creation produces a sixth-order
return that is positive at every heat scale and eventually overtakes it. This
is an exact obstruction for the two-mode family, not an empirical conjecture
about arbitrary flows.

The natural next normal-form step is to define a heat primitive of `X_s`,

```text
K_s(u)=integral_0^infinity X_s(exp(r*Delta)u)dr,   (5.1)
```

so that

```text
partial_t K_s=-nu*X_s+Y_s.                       (5.2)
```

Combining (5.2) with `partial_t J_s=-nu*D_s+X_s` supplies a second inverse
frequency and moves the unresolved recurrence into the higher transfer
`Y_s`. The immediate questions are:

```text
1. Does the quintic part of Y_s vanish generally or only on this family?
2. Is the first surviving sextic part sign-definite or viscosity-controlled?
3. Can J_s+K_s/nu absorb the positive difference-mode return without
   introducing a supercritical endpoint term?
```

This second normal-form identity is the next analytic decision gate.
