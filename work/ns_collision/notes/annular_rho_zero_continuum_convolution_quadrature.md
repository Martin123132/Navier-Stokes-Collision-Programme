# Annular rho-zero continuum convolution quadrature

## 1. Purpose

The fixed-output gate reduced the two leading `N^7` contractions to

```text
L_EE=L_VV+L_GH,

L_VV=(sqrt(2)/20) integral (v_z^2-v_y^2),

L_GH=(sqrt(2)/10) integral (g_y a_y-g_z a_z).
```

This audit evaluates those two continuum pieces without rerunning the
eightfold five-field second-jet grid. A separate termwise tail ledger is
still required to identify this leading limit with the complete
`c_1,N/N^7`.

The result is numerically very stable:

```text
L_VV approximately  1.722407e-7,
L_GH approximately -4.716268e-7,
L_EE approximately -2.993861e-7.                 (1.1)
```

These are quadrature candidates. They are not interval enclosures.

## 2. Fourfold de-aliasing

Let

```text
V=B(H,H),
G=B(H,V).
```

The two discrete continuum diagnostics are quartic means:

```text
sum_r |V_j(r)|^2,
sum_r G_j(r) conjugate(H_j(r)).
```

No monomial contains more than four copies of the one-field support.
Therefore each rectangular grid length only needs to be strictly larger
than four times the corresponding one-field maximum. This is half the
linear padding of the full branch audit and one eighth of its grid-point
count at the same carrier.

For carrier `N`, the audit uses

```text
next_fast_len(4 K_j+1)
```

in each coordinate.

## 3. Discrete functional

The parity-gauged scaling is

```text
Vhat_N(r)=-i sigma_r N^2 v_N(r/N),

Ghat_N(r)=   sigma_r N^5 g_N(r/N),

Hhat_N(r)=-  sigma_r N^-1 a_N(r/N).
```

Consequently

```text
L_VV,N
 =(sqrt(2)/20) N^-7
   [sum |V_z|^2-sum |V_y|^2],                    (3.1)

L_GH,N
 =(sqrt(2)/10) N^-7
   [sum G_z conjugate(H_z)
    -sum G_y conjugate(H_y)].                    (3.2)
```

Equations (3.1)-(3.2) converge to the two fixed-domain integrals.

## 4. Independent trace check

The Euler energy identity gives

```text
sum |V|^2+2 sum G dot conjugate(H)=0.             (4.1)
```

This checks:

- the factor `1/2` in `G=B(H,V)`;
- the pressure projection in both nonlinear fields;
- the sign in the `G-H` pairing;
- and the absence of quartic wraparound.

Across all 15 production rows, the relative residual in (4.1) is at most

```text
2.90e-16.
```

The largest divergence residual is also at roundoff scale.

## 5. Production rows

The combined quadrature is:

```text
N      L_EE,N
9     -9.5160955031e-7
13    -6.8463757083e-7
17    -5.7007846809e-7
21    -5.0729914312e-7
25    -4.6789330997e-7
29    -4.4093177801e-7
33    -4.2135339050e-7
37    -4.0650403753e-7
41    -3.9486153740e-7
45    -3.8549180886e-7
49    -3.7779049913e-7
53    -3.7134958473e-7
57    -3.6588378289e-7
61    -3.6118770424e-7
65    -3.5710975830e-7
```

The last nine rows fitted as polynomials in `1/N` give:

```text
degree 2: -2.9984920528e-7,
degree 3: -2.9937363100e-7,
degree 4: -2.9938615166e-7.                       (5.1)
```

At degree four, the two pieces are

```text
L_VV:  1.7224070914e-7,
L_GH: -4.7162686079e-7.
```

The maximum replay residual of the degree-four combined fit on those nine
rows is `4.62e-17`. This smooth agreement is strong numerical evidence,
not a rigorous error bar.

## 6. Cross-replay

The predecessor obtained the candidate coefficient by projecting the full
eightfold branch calculation onto the 36 active pressure outputs. This
audit instead replaces each fixed `q/N` shift by its zero-shift continuum
counterpart and computes the resulting quartic means.

At the largest common carrier,

```text
N=29,

36-mode active coefficient/N^7 =-4.440149827e-7,
continuum lattice quadrature   =-4.409317780e-7,
absolute difference            = 3.083204662e-9.
```

The agreement supplies an independent replay of the parity, pressure, and
symmetrization factors.

## 7. Correct scope

This stage establishes:

```text
a much cheaper exact quartic quadrature,
machine-precision Euler trace closure,
agreement with the original fixed-output route,
and a stable negative numerical candidate for L_EE.
```

It does not establish:

```text
the full nonleading c_1,N tail bound,
an interval enclosure for L_EE,
L_EE<0 as a theorem,
a certified nonzero N^9 second-jet coefficient,
a parabolic-window Taylor bound,
critical L^3 control, blowup, or global regularity.
```

The next stage must bound the continuum discretization error without
deducing it from fit quality. The positive and negative pieces must be
enclosed jointly tightly enough to keep their combined upper endpoint
below zero.

## 8. Reproducibility

Run:

```text
python work/ns_collision/scripts/annular_rho_zero_continuum_convolution_quadrature.py --sizes 9,13,17,21,25,29,33,37,41,45,49,53,57,61,65
```

Rows are written atomically after each carrier. Re-running the command
reuses matching completed rows.
