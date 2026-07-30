# Protected-collar cubic localization no-go

## Purpose

The radial collar calibration is numerically viable for protected supports

```text
E_d={r<=1-d, |z|<=0.75-d},       d>=0.2.              (1)
```

This note checks whether the existing cardinal-cubic square-root partition
can keep every physical perturbation inside such a support during a visit.
It cannot: the exact IMS cost is already larger than the complete compact
form floor before any Navier-Stokes error is charged.

## Exact support spacings

The cardinal cubic has support half-width `2h` in one coordinate. To fit a
transverse tensor square inside `r<=1-d`, its corner must obey

```text
2 sqrt(2) h_perp=1-d.                                (2)
```

To fit the axial support in `|z|<=0.75-d`, one needs

```text
2h_z=0.75-d.                                         (3)
```

The proved Fisher bound `I_3<157/200` then gives

```text
IMS_perp=314/[25(1-d)^2],
IMS_z=157/[50(0.75-d)^2].                            (4)
```

The arbitrary-history compact affine form floor is only

```text
m_0=4.832287335665.                                   (5)
```

## Numerical consequences

| `d` | transverse IMS | axial IMS | full IMS | full/floor |
|---:|---:|---:|---:|---:|
| 0 | 12.5600 | 5.58222 | 18.1422 | 3.75438 |
| 0.10 | 15.5062 | 7.43195 | 22.9381 | 4.74685 |
| 0.20 | 19.6250 | 10.3802 | 30.0052 | 6.20931 |
| 0.30 | 25.6327 | 15.5062 | 41.1388 | 8.51332 |
| 0.40 | 34.8889 | 25.6327 | 60.5215 | 12.5244 |

Even with no positive collar, the transverse partition alone costs more
than twice (5). Enlarging the collar shrinks the protected core and makes
the Fisher cost worse. Omitting the axial partition does not repair the
transverse failure.

## Consequence

The new collar cannot be realized by continuously applying the current
square-root cubic weights to the evolving energy. That route would spend the
spectral floor several times over before reaching the perturbation estimate.

The surviving possibilities are structurally different:

1. choose and change labels only at stopping times, so no partition gradient
   enters the interior generator;
2. mark individual Duhamel/form interactions by a conservative linear
   partition while retaining the global divergence-free cancellation;
3. find another localization mechanism with a proved cost below (5).

The second option is not yet a theorem. In particular, localizing the drift
pieces separately destroys their individual skewness unless the label sum
is kept intact through the energy estimate.

This is a no-go for the audited cardinal-cubic continuous partition, not for
all possible partitions or stopping constructions. The exact arithmetic is
reproduced by `scripts/protected_collar_partition_no_go_audit.py`.
