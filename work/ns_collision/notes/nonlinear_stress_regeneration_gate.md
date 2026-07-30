# Nonlinear stress regeneration gate

Status: the exact Fourier evolution of the low-output high-shell Reynolds
stress is now explicit. For every `HHL` regeneration triad, the two
apparently carrier-sized terms cancel after both high legs are retained;
the complete forcing is bounded by the low scale. This gain does not extend
to all `HHH` tensor channels: an explicit family leaves a traceless
pressure-strain term of order `H`. A sparse parabolic pulse family shows
that viscosity can make this surviving term square-summable in time, but a
dense annular mode-multiplicity bound remains open. No regularity conclusion
is claimed.

## 1. Exact projected evolution

Use the Fourier Navier-Stokes convention

```text
dot uhat(k)+nu|k|^2 uhat(k)=Nhat(k),               (1.1)

Nhat(k)
 =-i P_k sum_(a+b=k)
   [(uhat(a) dot b)uhat(b)].                       (1.2)
```

For a fixed low output `q`, define one ordered stress pair

```text
C_(k,q)=uhat(k) tensor uhat(q-k).                  (1.3)
```

Direct differentiation gives

```text
(d/dt+nu[|k|^2+|q-k|^2])C_(k,q)

 =Nhat(k) tensor uhat(q-k)
  +uhat(k) tensor Nhat(q-k).                       (1.4)
```

Summing (1.4) over pairs in a high annulus gives the exact low-output
Reynolds-stress evolution. There is no single scalar viscous rate before
the individual pair rates are bounded by a shell heat envelope.

The nonlinear term in (1.4) contains three velocity frequencies. Their sum
is `q`, so the previous atlas applies:

```text
HHH: all three frequencies are comparable;

HHL: two frequencies are high and one is low.      (1.5)
```

## 2. Bilinear symbol

For divergence-free polarizations `U` at wave `a` and `V` at wave `b`,
write the symmetrized Navier-Stokes interaction as

```text
B(a,b;U,V)
 =-i P_(a+b)[(U dot b)V+(V dot a)U].               (2.1)
```

For vectors `X,Y`, set

```text
X odot Y=X tensor Y+Y tensor X.                    (2.2)
```

These definitions include both ordered convective interactions and both
ordered stress pairs.

## 3. HHL sweeping commutator

Let high waves `a,b` and a low wave `c` satisfy

```text
a+b+c=q,                                          (3.1)

H<=|a|,|b|<=2H,

|c|,|q|<=L,        H>=4L.                          (3.2)
```

The complete contribution that regenerates the high-high stress at output
`q` is

```text
G_HHL
 =B(a,c;U_a,U_c) odot U_b
  +B(b,c;U_b,U_c) odot U_a.                       (3.3)
```

The separate terms in (3.3) can be `O(H)`. Taken together, they obey

```text
||G_HHL||_F
 <=18 L ||U_a||||U_b||||U_c||.                    (3.4)
```

The constant is independent of `H`.

### Proof

First omit the Leray projectors. The two sweeping terms combine exactly:

```text
(U_c dot a)(U_a odot U_b)
+(U_c dot b)(U_b odot U_a)

 =(U_c dot (a+b))(U_a odot U_b)
 =(U_c dot q)(U_a odot U_b),                      (3.5)
```

because `a+b=q-c` and `U_c dot c=0`. Thus their derivative is low.
The two strain terms already contain `U_a dot c` or `U_b dot c`.

It remains to estimate the pressure projections. For

```text
k=a+c=q-b,
```

transversality to `a` gives

```text
||(P_k-I)U_a||
 =|c dot U_a|/|a+c|
 <=L||U_a||/(H-L).                                (3.6)
```

Multiplying (3.6) by the apparent sweeping factor at most `2H` still costs
only `O(L)`. The same calculation holds for `b+c`. Using

```text
||X odot Y||_F<=2||X||||Y||                       (3.7)
```

gives a constant below `17`; (3.4) retains `18`.

This is the Fourier form of the sweeping/Galilean commutator: a low
velocity can transport a high stress rapidly, but it cannot deform the
paired stress at the carrier rate.

## 4. Exact coherent pump replay

Take

```text
q=(1,1,0),          c=(1,0,0),

a=(0,1,H),          b=(0,0,-H),                   (4.1)
```

with polarizations

```text
U_c=i e_3,          U_a=U_b=e_1.                  (4.2)
```

Each generated contribution has an `e11` coefficient of carrier size and
opposite sign. Their paired value is exactly

```text
(G_HHL)_11
 =4H/[(H^2+1)(H^2+2)].                            (4.3)
```

At `H=64`, the sum of the absolute unpaired `e11` terms is more than
`1.67e7` times the paired remainder. The audit also tests `48` random
polarization triples at each of eight carriers. The largest observed value
of

```text
||G_HHL||_F/L
```

is `2.99650`, well below the analytic constant `18`.

## 5. HHH pressure-strain obstruction

The HHL theorem cannot be promoted to every nonlinear triad. Set

```text
a_H=(H,0,0),

b_H=(-H,H,0),

c_H=(1,1-H,0),                                    (5.1)

a_H+b_H+c_H=q=(1,1,0).
```

All three waves have size comparable to `H`. For a fixed divergence-free
polarization family, form the complete three-leg stress forcing

```text
G_HHH
 =B(a_H,b_H) odot U_c
  +B(a_H,c_H) odot U_b
  +B(b_H,c_H) odot U_a.                           (5.2)
```

The unprojected transport part is a low-output divergence: divided by `H`,
it tends to zero. Its trace also tends to zero after Leray projection, as
required by local kinetic-energy conservation.

The traceless tensor does not vanish:

```text
||G_HHH/H||_F -> 2.27653869169.                    (5.3)
```

At carrier `2048`, the residual from the limiting matrix is
`0.00139346`. The surviving term is anisotropic pressure-strain. Therefore

```text
||G_HHH||<=C L [amplitudes]                        (5.4)
```

is false for the full tensor stress, even though its trace has the
low-output cancellation.

This does not conflict with the self-shell pressure closure. That theorem
controls a complete localized pressure load; (5.3) concerns the time
regeneration of one anisotropic stress channel.

## 6. Parabolic pulse experiment

The carrier factor in (5.3) is instantaneous. To test whether viscosity can
pay for its duration, use

```text
H_j=16*4^j,

A_H=H^(-1/3),

tau_H=H^(-2).                                      (6.1)
```

The scaled forcing has size

```text
||G_HHH||~H A_H^3~1.                              (6.2)
```

Thus every shell feeds the same nonzero tensor channel. The coherent sum
over its shell square function grows like `sqrt(N)`, while

```text
sum_H A_H^2<infinity.                              (6.3)
```

So energy alone cannot bound the instantaneous forcing square function.

Retaining the parabolic duration changes the scaling:

```text
shell enstrophy-time cost
 ~H^2 A_H^2 tau_H
 ~A_H^2,                                          (6.4)

||f_H||_(L2_t)^2
 ~H^2 A_H^6 tau_H
 ~A_H^6.                                          (6.5)
```

Both series converge. Across seven audited shells:

```text
cumulative energy proxy                 1.56424829

cumulative enstrophy-time cost          2.04447874

cumulative forcing L2-time norm^2       0.02076436

coherent pointwise ratio                2.64489314. (6.6)
```

For one normalized triad per shell, (6.4)-(6.5) imply the conditional
sequence bound

```text
sum_H ||f_H||_(L2_t)^2
 <=C E_*^2 sum_H(enstrophy-time cost).             (6.7)
```

This confirms the mechanism behind the preceding viscous occupation
theorem for sparse triads.

## 7. Remaining multiplicity gate

Equation (6.7) is not yet a Navier-Stokes theorem. A full annulus contains
many modes and many triples feeding the same low output. Coherent
convolution could introduce the Bernstein factor

```text
H^(3/2)                                            (7.1)
```

that a one-triad model cannot detect.

Established:

- the exact pairwise stress evolution (1.4);
- the `HHH/HHL` regeneration atlas;
- the carrier-independent HHL theorem (3.4);
- exact cancellation of the coherent HHL pump;
- the `O(H)` traceless HHH pressure-strain witness;
- sparse parabolic summability of the surviving HHH forcing.

Still open:

- dense same-shell mode multiplicity;
- a full `ell2_H L2_t` regeneration bound from Leray data;
- critical signed large-data closure;
- low-regularity passage and global regularity.

The next decisive test is a divergence-free dense annular packet normalized
to fixed shell energy and feeding one low Fourier/tensor/Walsh channel.

The identities and finite-mode replays are generated by
`scripts/nonlinear_stress_regeneration_gate_audit.py`.
