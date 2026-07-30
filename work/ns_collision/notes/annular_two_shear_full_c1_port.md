# Annular two-shear full c1 port

## 1. Result

The two-shear square gate proved that the modified annular witness has the
strict continuum identity

```text
L_*=-(3*sqrt(2)/20)||v_y||_2^2<0.
```

This stage ports the complete amplitude-one four-high tail theorem to that
witness. If `D_*,N` denotes the two retained fixed-output contractions and
`c1_*,N` denotes the complete coefficient, then for odd `N>=5`,

```text
|c1_*,N-D_*,N|<=70,657,920 N^6.                 (1.1)
```

The modified packet converges in `L1 cap L2`, the active output set remains
finite, and the continuum bilinear map is continuous. Hence

```text
D_*,N/N^7 -> L_*,
c1_*,N/N^7 -> L_*<0.                             (1.2)
```

This closes the old amplitude-one continuum-sign obstruction on the
modified branch. It does not close the optimizer, uniform Taylor remainder,
parabolic window, or critical `L3` estimates.

## 2. Modified finite multiplier

On the positive packet, write

```text
m_*(k)
 =(k_x^2+k_y^2)/(k_x*|k|^3)*(-k_z,0,k_x).
```

The exact finite coefficient is the old parity-gauged sine product times
`m_*`. Since

```text
k_x>=2N,
|k_y|,|k_z|<=(N-1)/2,
```

one has

```text
|m_*(k)|
 <=(k_x^2+k_y^2)|k|/(k_x|k|^3)
 <=1/k_x
 <=1/(2N).                                       (2.1)
```

Thus the coefficient-size hypothesis in the old tail theorem is unchanged.

## 3. One first difference

Set

```text
r^2=x^2+y^2+z^2,
m_z=(x^2+y^2)/r^3,
m_x=-(z/x)m_z.
```

The coordinate derivatives of `m_z` obey

```text
|partial_x m_z|<=5/r^2,
|partial_y m_z|<=5/r^2,
|partial_z m_z|<=3/r^2.                          (3.1)
```

On the packet,

```text
|z|/x<=1/4,
r/x<=sqrt(9/8).
```

The product rule therefore gives the following constants for the three
derivatives of `m_x`:

```text
partial_x: 5/4+3/(4sqrt(8)) <1.516,
partial_y: 5/4              =1.250,
partial_z: sqrt(9/8)+3/4    <1.811.
```

Combining `m_x` and `m_z` componentwise yields

```text
|partial_j m_*|<6/r^2<=3/(2N^2).                 (3.2)
```

The sine-product first difference is at most `pi/N`. Equations (2.1) and
(3.2) give

```text
|Delta_j[parity(k) hhat_*,N(k)]|
 <=[pi/2+3/2]/N^2
 <4/N^2.                                         (3.3)
```

At a packet face, the boundary sine itself is at most `pi/N`, so (3.3)
also holds for the zero extension. No second or higher boundary derivative
is used.

## 4. Profile convergence

The continuum multiplier satisfies the same size and derivative bounds on
`D`. Multiplication by the sine cutoff gives coordinate Lipschitz constant

```text
pi/2+3/2.
```

The effective sine sample and a point in its centred lattice cell are
separated by less than `2.1/N`. A rounded pointwise bound `25/N` is
therefore valid. The union of the sampled and continuum supports has volume
at most four, so

```text
||b_N-b||_1 <=100/N,
||b_N-b||_2 <= 50/N,

epsilon_N:=||b_N-b||_1+||b_N-b||_2<=256/N.       (4.1)
```

The deliberately rounded constant is used only to certify convergence.

## 5. Fixed-output port

The two low shears have a combined exact stencil with

```text
58 active outputs,
max |q|^2=6,
Q_*=(sqrt(2)/40)diag(1,-2,1).
```

Every finite shift is `q/N` and therefore tends to zero. For compactly
supported profiles, the existing continuum Euler bilinear bound is

```text
||B_c(f,g)||_p
 <=(R_f+R_g)/2[
    ||f||_1||g||_p+||g||_1||f||_p],
p in {1,2}.                                      (5.1)
```

Apply (5.1) to the first Euler field and then to the second Taylor field.
Equation (4.1), finite translation continuity, and the exact matrix sum
give

```text
D_*,N/N^7 -> L_*.                                (5.2)
```

No inverse-`N` fit is used in (5.2).

## 6. Tail-ledger port

The stored full tail theorem has fourteen structural profiles and total
absolute atomic coefficient mass `94`. A machine traversal confirms that
every one of those profiles contains exactly one low leaf. Consequently,

```text
c1[U_yz+U_xy]=c1[U_yz]+c1[U_xy]                 (6.1)
```

at amplitude one.

The old tail proof uses only:

```text
even coordinate sum of the low wave,
|ell|^2=2,
divergence-free unit low polarization,
finite low Fourier l1 mass,
the annular support radius and mode count,
|hhat_N|<=1/(2N),
one packet difference <=4/N^2,
and a fixed outer degree-zero pressure projector.
```

Both new low waves satisfy those hypotheses. The new low Fourier `l1` mass
is four instead of two, while all other constants are unchanged. The old
per-atomic bound therefore doubles:

```text
375,840 N^6 -> 751,680 N^6.
```

Multiplying by the unchanged mass `94` gives (1.1):

```text
751,680*94=70,657,920.
```

In normalized form,

```text
|c1_*,N-D_*,N|/N^7<=70,657,920/N ->0.            (6.2)
```

Combining (5.2), (6.2), and the strict square identity proves (1.2).

## 7. Scope

This stage proves:

```text
the modified packet coefficient and one-difference bounds;
fixed-output convergence D_*,N/N^7 -> L_*;
the complete fourteen-profile tail port;
and c1_*,N/N^7 -> L_*<0.
```

It does not prove:

```text
that the old one-shear L_EE is negative;
the modified static joint optimizer;
the modified complete first or second jet at finite N;
a uniform second-jet Taylor remainder;
the required parabolic-window amplification;
critical L3 control;
finite-time blowup;
or global regularity.
```

The next gate is to port the static optimizer and the complete finite jet
formulas to the two-mode low field. Those stages must retain all low-low and
mixed terms rather than importing only the favorable amplitude-one
coefficient.
