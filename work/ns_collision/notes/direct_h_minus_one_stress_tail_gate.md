# Direct H-minus-one stress-tail gate

Status: certified. The exact arithmetic audit and focused dependency replay
pass. This note corrects the route decision in
`scale_uniform_low_output_tail_gate.md`; it does not alter that note's valid
scalar-envelope calculation.

## 1. Correction

The channelwise response theorem proved uniform decay in
`L2_t H_x^(-s)` for every `s>1` and then exhibited a channel-saturated pulse
showing that its scalar forcing envelope alone cannot close `s=1`.

That conclusion about the envelope is correct. The inference that the actual
high-high Reynolds stress still has an open `H^(-1)` endpoint is not.
Channelwise absolute values discarded a physical-space product estimate that
couples all outputs before counting them.

For the actual comparable-shell stress, this missing estimate closes the
endpoint directly from the standard Leray quantities.

## 2. Comparable-shell stress

Let

```text
a_J(t)=||u_J(t)||_2,

A_H(t)^2=sum_(|log_2(J/H)|<=2) a_J(t)^2.          (2.1)
```

Let `C_H` denote the canonical factorized smooth comparable-pair shell piece
of the Reynolds stress, with its bounded low-output Fourier cutoff applied
afterward. Such a piece is a finite sum of uniformly bounded products

```text
P_lo(u_J tensor u_J'),       J,J' comparable to H. (2.2)
```

The estimate below is pointwise in time and does not use the Navier-Stokes
equation yet.

## 3. Endpoint product estimate

On the three-torus,

```text
H^1 embeds in L^6,

L^(6/5) embeds in H^(-1).                          (3.1)
```

For comparable shell factors, Holder and Bernstein give

```text
||u_J tensor u_J'||_(H^-1)

 <=C||u_J tensor u_J'||_(6/5)

 <=C||u_J||_2||u_J'||_3

 <=C H^(1/2)||u_J||_2||u_J'||_2.                 (3.2)
```

Summing the fixed number of comparable pairs and using Cauchy-Schwarz over
that finite family yields

```text
||C_H(t)||_(H^-1)<=C H^(1/2)A_H(t)^2.             (3.3)
```

An orthogonal low-output Fourier projection is contractive in `H^(-1)`, and
the canonical smooth cutoff has the same uniform multiplier bound, so (3.3)
applies to the exact low-output stress used in the earlier response gate.
The same conclusion extends to nonfactorized pair selectors only when their
rescaled bilinear multipliers have a uniform
`L2 times L3 -> L^(6/5)` bound; no broader selector class is assumed here.

## 4. Dyadic tail

For dyadic shells and the five-neighbor definition (2.1), exact overlap
arithmetic gives

```text
sum_H A_H^2<=5 sum_J a_J^2,                       (4.1)

sum_H H A_H^2
 <=(1/4+1/2+1+2+4)sum_J J a_J^2
 =(31/4)sum_J J a_J^2.                           (4.2)
```

At a fixed time, (3.3) and dyadic Cauchy-Schwarz imply

```text
||sum_(H>=K) C_H||_(H^-1)^2

 <=C^2[sum_(H>=K)A_H^2]
       [sum_(H>=K)H A_H^2].                      (4.3)
```

If `H>=K` and `J` occurs in `A_H`, then `J>=K/4`, so

```text
J<=4J^2/K.                                       (4.4)
```

Writing

```text
E_*=sup_t sum_J a_J(t)^2,

D=integral sum_J J^2a_J(t)^2 dt,                 (4.5)
```

and integrating (4.3) gives the explicit estimate

```text
||sum_(H>=K) C_H||_(L2_t H_x^-1)^2

 <=155 C^2 E_*D/K.                               (4.6)
```

Therefore

```text
sum_(H>=K) C_H ->0 in L2_t H_x^(-1)              (4.7)
```

uniformly over smooth Galerkin truncations sharing the same Leray bounds.
The estimate is trajectory-level: it applies to the actual stress at every
time, not to an independently prescribed forcing array.

## 5. Why the pulse does not contradict this

The previous thought experiment assigned every channel

```text
dot c_q+H^2c_q=H^(5/2)1_[0,H^(-2)],     c_q(0)=0. (5.1)
```

At the end of the pulse,

```text
c_q(H^-2)=H^(1/2)(1-e^(-1)).                     (5.2)
```

An actual Fourier coefficient of a unit-energy Reynolds stress is bounded
independently of `H` by Fourier Cauchy-Schwarz. Thus (5.2) eventually
exceeds the admissible stress amplitude.

The pulse still proves the narrow statement for which it was designed:
the scalar `H^(-3/2)` forcing envelope, by itself, does not imply endpoint
decay. It is not an admissible evolution once the algebraic relation
`C_H=P_Hu tensor P_Hu` is restored.

## 6. Dense forcing remains compatible

The dense HHH packet may still have instantaneous derivative

```text
<T,G_HHH(q)> approximately H^(5/2)               (6.1)
```

on many outputs. Equation (4.6) does not disprove that derivative. It says
that the complete stress cannot follow the channel-saturated parabolic pulse
independently in every output. Sign reversal, nonlinear deformation,
viscous response, or another exact correlation must intervene in the
trajectory-level stress.

The staged dense-output certificate therefore remains useful as a spatial
diagnostic, but it no longer decides the `H^(-1)` stress endpoint.

## 7. Scope and next gate

This correction establishes:

- endpoint `L2_t H_x^(-1)` vanishing of the complete comparable high-high
  stress tail;
- uniformity over smooth Galerkin cutoffs;
- inadmissibility of the old saturated pulse as an actual Reynolds-stress
  trajectory;
- endpoint passage for this quadratic stress component.

It would not establish:

- compactness of every cubic term in the local-energy defect;
- equation-level control of all HHL and partition commutators after
  localization;
- suitable-weak exceptional-set removal;
- finite-time blow-up or global regularity.

The standard suitable-weak compactness passage is not itself a regularity
mechanism, so the project should not spend the next stage presenting that
known closure as progress toward the prize. After (4.6) is validated, this
quadratic stress detour is closed.

The time-integrated signed triad measure named in the earlier roadmap was
already constructed in `cumulative_collision_rigidity.md`. Its quartic,
quintic, and sextic transfers were subsequently proved sign-indefinite, and
`normal_form_resummation.md` identified the available hierarchy as
perturbative. It must not be restarted as though it were unfinished.

The live nonstandard obligation is the path-dependent projected-replica
pressure edge, or a genuinely multiscale defect envelope. The first new
consequence of (4.6) is installed in
`floor_free_pressure_edge_tail_gate.md`: arbitrarily far low-output
high-high pressure beats vanish in the time-integrated edge without requiring
a positive partition-weight floor. The near-carrier signed edge remains open.
Any successful route must produce new rigidity capable of shrinking or
eliminating the suitable-weak singular set; ordinary quadratic stress
compactness cannot do that.

The exact overlap arithmetic and endpoint-pulse classification are generated
by `scripts/direct_h_minus_one_stress_tail_gate_audit.py`. The production
certificate is
`results/direct_h_minus_one_stress_tail_gate_audit_v1.json`; the focused
dependency replay contains 15 passing tests.
