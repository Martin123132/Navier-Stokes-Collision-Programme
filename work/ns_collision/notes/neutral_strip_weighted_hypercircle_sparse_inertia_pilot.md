# Sparse inertia route for the weighted hypercircle constant

## Purpose

The weighted hypercircle pilot found

`kappa_h = 0.023654402914286256`

in floating arithmetic. The continuum gate does not need a sharp enclosure.
This stage fixes the simpler production candidate

`beta = 0.045`.

Using the current floating geometry terms, `kappa_h<beta` would give

`C_h <= C_data + alpha beta = 0.08299017542532933`,

below the strict target `0.08557115750643675` by
`0.002580982081107422`. The geometry terms still need directed enclosure, so
this headroom is not yet certified.

## Threshold pencil

Let:

- `P` be the RT0 mass for `integral mu^-1 p.q`;
- `N` be the signed P0 divergence matrix;
- `A` be the weighted P1 stiffness;
- `B` be the P1-P0 physical-load matrix;
- `D` be the triangle-area diagonal; and
- `W` be the P0 mass for `integral_T mu^-1`.

For a threshold `beta`, define

```text
K(beta) =
[ P   N^T   0    0        ]
[ N    0    0    D        ]
[ 0    0    A   -B        ]
[ 0    D   -B^T -beta^2 W ]
```

Eliminating the positive blocks `P` and `A`, then the negative flux Schur
block `-N P^-1 N^T`, leaves

`Q-beta^2 W`,

where `Q` is the dense hypercircle quadratic form. Sylvester inertia
additivity therefore gives

```text
inertia(K(beta)) =
inertia(P) + inertia(A) + inertia(-N P^-1 N^T)
                 + inertia(Q-beta^2 W).
```

Consequently, `kappa_h<beta` is equivalent to the full threshold-pencil
inertia

`positive=edge_count+state_count`,

`negative=2*triangle_count`,

`zero=0`.

## Coarse identity validation

On the `h=0.6` mesh:

- edges: `505`;
- states: `127`;
- triangles: `316`;
- pencil dimension: `1264`; and
- floating `kappa_h`: `0.2316506434954726`.

The direct dense inertia of `K(beta)` agrees exactly with the inertia predicted
from the independently formed dense Schur complement in three regimes:

1. `beta=0.045`, where all 316 shifted source directions remain positive;
2. `beta=0.8*kappa_h`, where 12 shifted directions remain positive; and
3. `beta=1.2*kappa_h`, where the shifted source block is strictly negative.

Every row has zero unresolved eigenvalues at a sign tolerance at least 33
times smaller than its nearest direct pencil eigenvalue. This validates the
block signs and inertia accounting, not the full-mesh inertia.

## Sparse resource probe

The `h=0.12` central binary pencil has:

- dimension `30,968`;
- `198,098` nonzeros;
- `1,038,256` nonzeros in SuperLU `L+U`;
- fill ratio `5.2411230804955125`;
- factor time `0.1379728999454528` seconds; and
- equal row and column permutations.

SuperLU is used only as a central fill probe. It is not a verified inertia
calculation.

The exact `h=0.06` topology reconstructs:

- edges: `46,697`;
- states: `15,211`;
- triangles: `30,954`;
- pencil dimension: `123,816`;
- pencil nonzeros: `798,384`; and
- target inertia: `(61,908 positive, 61,908 negative, 0 zero)`.

No full-mesh factorization was launched in this stage.

## Interval inventory

The full certificate separates into the following obligations:

1. Reuse the completed directed intervals for the Gaussian-weighted P1
   stiffness `A`.
2. Treat `N` as exact integer topology with entries `-1` and `+1`.
3. Directed-enclose the binary-vertex determinants in `D`; obtain `B` from
   `D/3`.
4. Build new analytic local enclosures for the RT0 matrix `P` and source mass
   `W`, whose moments use `exp(+x^2/2)`.
5. Treat `beta=0.045` as an exact decimal and direct all products in
   `beta^2 W`.
6. Outward-enclose `C_data` and `alpha` and retain positive continuum
   headroom.
7. Obtain a verified symmetric-indefinite sparse LDL inertia with every pivot
   interval excluding zero.

The next bounded step should implement and test the local `exp(+x^2/2)`
moment enclosures on a distributed triangle sample, reusing the established
Gaussian assembly audit architecture. It should not yet launch the full
123,816-pivot directed LDL.

The global Ritz constant, continuum spectral capture, point-source bridge,
conormal output, polygon-to-circle transfer, and Navier-Stokes regularity
claim all remain false.
