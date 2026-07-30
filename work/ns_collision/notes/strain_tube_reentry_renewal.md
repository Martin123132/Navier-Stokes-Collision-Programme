# Buffered exit and re-entry renewal

## Why a buffer is necessary

The localized strain estimate kills a history at tube exit. A real backward
history is not killed: it can leave the concentration region and return.
Simply counting every crossing of one boundary as a new visit is unusable,
because a continuous diffusion can chatter across that boundary immediately.

Use two localization surfaces instead. The inner region contains the strain
core. A visit continues through the surrounding shell and ends only when the
history reaches the outer surface. A new visit begins only after a history
that reached the outer surface returns to the inner region. This hysteresis
makes the excursion decomposition well defined.

## Three-dimensional return barrier

First use a spherical benchmark with inner radius `L` and outer radius
`L_out>L`. Let the backward generator outside the core be

```text
A = nu Delta + b dot grad + c,
```

where `b` is measured relative to the moving centre and `c` is the positive
deformation potential accumulated during the excursion. For

```text
h(r) = (L/r)^beta,
```

the three-dimensional weighted generator satisfies

```text
(A h)/h
  = beta[nu(beta-1)-b(x) dot x]/r^2 + c(x).
```

Therefore `h` is a Feynman-Kac supersolution whenever

```text
c_+(x) r^2
    <= beta [b(x) dot x + nu(1-beta)].
```

Optional stopping then bounds the weighted return operator for one history by

```text
r_one <= (L/L_out)^beta.
```

For two histories driven by independent Brownian noises, with additive
excursion deformation weights, conditional independence gives

```text
r_pair <= (L/L_out)^(2 beta).
```

For pure three-dimensional Brownian motion, `beta=1` is exact. If the radial
backward drift obeys `b dot x >= -kappa nu` with `kappa<1` and `c=0`, any
`0<beta<=1-kappa` remains valid. Strong inward drift or positive deformation
outside the core can destroy this elementary return contraction; the displayed
supersolution condition records the precise obstruction.

## Renewal algebra

Let `V` be the weighted propagator for one buffered common visit, including
the core, shell, and final outer exit. Let `R` propagate an outer exit to the
next common inner return. Multiple visits produce the operator series

```text
V [I + RV + (RV)^2 + ...].
```

It converges if

```text
||R|| ||V|| < 1,
```

and then

```text
total norm <= ||V||/[1-||R||||V||].
```

In the pure Brownian spherical benchmark, writing
`eta=L_out/L`, the two-history return norm is at most `eta^(-2)`. The permitted
visit gain is therefore `||V||<eta^2`.

| `eta` | one-history return | pair return | maximum visit gain |
|---:|---:|---:|---:|
| 1.1 | 0.9090909091 | 0.8264462810 | 1.21 |
| 1.25 | 0.8 | 0.64 | 1.5625 |
| 1.5 | 0.6666666667 | 0.4444444444 | 2.25 |
| 2 | 0.5 | 0.25 | 4 |
| 3 | 0.3333333333 | 0.1111111111 | 9 |

This is compatible with the moving-tube result: if each buffered visit is
contractive in the gauged norm, any strict three-dimensional return
contraction makes the renewal series summable. Gauge changes and shell
stretching may instead give `||V||>1`; the table quantifies the room available.

## A decisive geometry distinction

For Brownian motion in dimension `d`, the decaying radial harmonic exponent is
`beta=d-2`. In three dimensions it is `1`. In two dimensions it is `0`, and
the return probability to a disk is one.

An ideal infinite vortex tube has a two-dimensional transverse return problem.
It therefore has no geometric return contraction, even though a finite-width
tube has a strict killed spectral margin during one visit. A finite
three-dimensional localization, finite axial extent, or another mechanism
that suppresses returns is essential for this elementary renewal route.

For a general bounded region, Newtonian capacity replaces the spherical
formula. The relevant question is not whether the core looks exactly like a
ball, but whether its buffered three-dimensional return operator has norm
strictly below one after deformation weights are included.

## What is and is not closed

This decomposition closes the algebra of repeated visits once estimates for
`V` and `R` are available. It does not yet prove those estimates uniformly for
Navier-Stokes. Three obligations remain:

1. control stretching in the transition shell so the complete buffered visit
   operator satisfies the required bound;
2. construct a weighted exterior supersolution, or a capacity estimate, for
   the actual moving drift and deformation;
3. relate every potentially singular stretching region to a family of finite
   three-dimensional cores without losing summability over the family.

The main gain is diagnostic. Re-entry is not an unspecified loophole anymore:
it is represented by a return operator with a precise contraction condition,
and the infinite-cylinder geometry is identified as a genuine failure mode.

The symbolic barrier and renewal arithmetic are reproduced by
`scripts/strain_tube_reentry_audit.py`.

The attempt to control both visit and return operators using only
three-dimensional Leray-Hopf quantities is carried out in
`three_dimensional_leray_gate.md`.

The moving-radius version is derived in `shrinking_tube_renewal.md`.
Monotone envelope shrinkage adds a favorable term to the return barrier and
keeps the geometric return factor uniform in the absolute core radius.
