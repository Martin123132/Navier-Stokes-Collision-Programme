# Leray conditional-occupation scaling no-go

## Question

Can the standard Leray energy inequality force the conditional probability
of a same-level bad visit below the restart allowance?

Not by itself. A critical strain burst can have vanishing total Leray cost
while retaining a fixed, excessive occupation probability for a diffusion
conditioned to enter its parabolic cell. This is a scaling obstruction to an
energy-only implication, not a Navier-Stokes counterexample.

## Critical burst

Normalize the viscosity so that the diffusion generator is `Delta`. For
`epsilon>0`, set

```text
Q_epsilon(t,x)=a epsilon^(-2)

on 0<t<tau epsilon^2 and |x_i|<h epsilon.             (1)
```

The critical spatial potential norm is independent of `epsilon`:

```text
||Q_epsilon(t)||_(3/2)=4 a h^2.                       (2)
```

In contrast, the spacetime quantity seen by Leray enstrophy is

```text
||Q_epsilon||_(L2_t,x)^2
 =8 a^2 tau h^3 epsilon.                              (3)
```

Thus the standard energy cost tends to zero at small scales while the
critical visit error remains fixed. This is exactly the supercritical
scaling `2/2+3/2=5/2>2` in concrete form.

## Exact conditional survival

Start the diffusion at the centre of the cube. For generator `Delta`, the
one-dimensional probability of remaining in `(-h epsilon,h epsilon)` for
time `tau epsilon^2` is

```text
p_1=(4/pi) sum_(n>=0) (-1)^n/(2n+1)
    exp[-(2n+1)^2 pi^2 tau/(4h^2)].                   (4)
```

The coordinates are independent, so the cube survival probability is
`p_1^3`; it is also independent of `epsilon`.

Choose

```text
a=0.7,       h=0.5,       tau=0.06.                  (5)
```

Then

```text
||Q_epsilon(t)||_(3/2)=0.7,
||Q_epsilon||_(L2_t,x)^2=0.0294 epsilon,
p_cube=0.34624485261.                                 (6)
```

The spatial mass exceeds the compact full-affine no-restart potential-only
budget `0.63227660`. The survival probability exceeds both the earlier
probability-paid allowance `0.24345125` and the candidate compact-affine
allowance

```text
(1-0.55681307)^2=0.19641465.                          (7)
```

Therefore no bound depending only on the global `L2_t L2_x` strain cost and
tending to zero with that cost can supply the required conditional restart
probability uniformly over cells and entry states.

## Navier-Stokes scaling

The same obstruction is built into the exact Navier-Stokes rescaling

```text
u_epsilon(x,t)
 =epsilon^(-1)u(x/epsilon,t/epsilon^2).               (8)
```

On a correspondingly scaled time interval, kinetic energy and integrated
enstrophy both acquire a factor `epsilon`. The critical Campanato/strain
functionals and the conditional diffusion law acquire factor one. Hence a
successful theorem cannot obtain a scale-independent conditional branch
bound from global energy magnitude alone.

This does **not** construct a singular solution, and it does not prove that
an arbitrary scalar profile (1) is realized as the strain error of a
Navier-Stokes solution. It proves only that the Leray inequality, considered
as the proposed input estimate, lacks the scaling strength needed for the
conditional conclusion.

## Consequence

The bad-occupation branch must use at least one ingredient not present in
the bare energy inequality:

1. equation-specific strain geometry or pressure cancellation;
2. smoothing and density control of the actual entry law;
3. collision killing or another pathwise damping term;
4. an additional scale-critical Morrey/Kato hypothesis;
5. a nonautonomous full-affine reference that avoids treating coherent
   orientation motion as an abort.

The fifth route attacks the constant-spectrum counterexample directly and
is the next calculation. The scaling identities, exact cube survival
series, and numerical thresholds are reproduced by
`scripts/leray_conditional_occupation_no_go_audit.py`.
