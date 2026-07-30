# Certified modified-chain two-block state leakage

## Purpose

The inertia-Schur certificate proves that the modified complementary block is
bounded below by `102.7`. To use the damped two-block theorem, the low-block
floor and the off-block coupling must belong to the same stored binary
modified operator and the same frozen trial split.

## Directed finite-block parameters

For the frozen reference columns `V`, define

```text
M_K=V^T M_tilde V,
A_K=V^T A_tilde V,
L_K=M_K^(-1)A_K.
```

Sparse-dense products are bounded entrywise with binary64 `gamma_n`
estimates. Dense eigensystem reconstruction and orthogonality defects then
certify

```text
A_K-2.36 M_K > 0.
```

Thus the modified low block has lower bound `alpha=2.36`.

The unnormalized invariance residual is

```text
R=A_tilde V-M_tilde V L_K.
```

The checker bounds the solve error in `L_K`, certifies the spectral norm of
`M_tilde^(-1/2)R` by an SVD reconstruction, and divides by the certified
square root of the minimum eigenvalue of `M_K`. This gives a directed upper
bound for

```text
epsilon=||(I-QQ^T)HQ||,
Q=M_tilde^(1/2)V M_K^(-1/2).
```

The bound is intentionally a little larger than the earlier floating value
`6.343703098841749`; no floating diagnostic is silently promoted.

## Two-block theorem

Together with the independently certified high floor `beta=102.7`, the
parameters are inserted into

```text
d/dt [x;y] <= [[-alpha,epsilon],[epsilon,-beta]][x;y].
```

The result certifies state-space high leakage and low feedback at the first
half-window and at all later production times. It is not yet charged directly
to the boundary response. That requires the next half-time flux-smoothing
operator estimate and the low-projector source/trace mismatch.

The executable is
`scripts/neutral_strip_modified_two_block_leakage_certificate.py`.

## Certified production values

The directed parameter bounds are

```text
alpha                                      2.36
beta                                     102.7
epsilon upper                              6.44525405835444
minimum eigenvalue of A_K-alpha M_K        0.010163242064110413
```

At `t=3/8`, the high component is at most `0.030690921574191874`, the
low-block feedback is at most `0.06704678749964414`, and their orthogonal
combination is at most `0.07373740150761779`. The corresponding gap-free
high-component estimate would be `2.4169702718829154`.
