# Reversible incompressible shell rigidity

## Structural question

The anisotropic Poisson pilot continues the affine potential constantly along
radial shell rays. That preserves a reversible weighted form, but the shell
drift is not divergence-free. Can one taper the affine reference smoothly
while retaining both reversibility and incompressibility?

With the axial drift fixed, the answer is no. The obstruction is an elementary
harmonic continuation rigidity, not a numerical defect.

## Harmonic affine core

For the normalized trace-free spectrum `(-1-t,t,1)`, the backward affine
potential is

```text
Phi_t=(1+t)x^2/2-t y^2/2-z^2/2.                       (1)
```

It is harmonic because its Hessian has trace zero. More generally,

```text
b=grad Phi,       div b=0
```

implies `Delta Phi=0`. Reversibility and incompressibility therefore turn the
shell extension into a harmonic Cauchy problem.

Write `kappa=(1+2t)/4`. The transverse core potential is

```text
phi_core=r^2/4+kappa r^2 cos(2 theta).                 (2)
```

With axial potential `-z^2/2`, a reversible incompressible annular extension
must satisfy `Delta_perp phi=1`. Its general axisymmetric plus mode-two form
is

```text
phi_shell=r^2/4+A log r+B
          +(C r^2+D r^(-2))cos(2 theta).               (3)
```

Matching both value and radial derivative at `r=1` gives

```text
A=B=D=0,       C=kappa.                                (4)
```

Thus (3) is exactly (2) throughout the shell. The full affine field is the
unique smooth reversible incompressible continuation in this class.

If one instead matches the anisotropic value at `r=1` and forces it to zero
at `r=2`, the harmonic amplitude is

```text
kappa(16-r^4)/(15r^2).                                 (5)
```

Its inner derivative is `-34 kappa/15`, whereas the core derivative is
`2 kappa`. The jump is `-64 kappa/15`, producing an interface divergence
defect. The old axisymmetric constant-radial shell likewise retains axial
divergence without the compensating transverse divergence.

## Forced full-affine stress test

The anisotropic finite-element pilot also evaluates the forced continuation.
Stretching is retained throughout `r<2`, and the inner and outer weighted
angular measures use anisotropy amplitudes `kappa` and `4 kappa`.

At the old half-height `H/L=1.5`, the axisymmetric finite-element visit gain
`1.22122` agrees with the exact full-affine value `1.22215`. Across the
spectrum:

| `t` | visit norm | generation criterion | `chi_P` |
|---:|---:|---:|---:|
| -0.50 | 1.2212 | 0.7697 | 4.2163 |
| 0.00 | 1.2988 | 0.8707 | 4.3500 |
| 0.25 | 1.3832 | 0.9874 | 4.5195 |
| 0.50 | 1.4833 | 1.1355 | 4.7613 |
| 1.00 | 1.6952 | 1.4832 | 5.4825 |

The unperturbed cycle therefore fails before the most anisotropic endpoint.
Continuing the full affine field through the old shell is not a uniform
solution at the old geometry.

## Axial compaction

Shortening the cylinder raises the principal axial Dirichlet eigenvalue and
reduces the visit norm. It also raises the axial cubic IMS cost. These effects
must be charged together. If `h=H/L` and the axial cubic knot spacing is
`h/2`, the conservative spectral reserve is

```text
m_res=(j_(0,1)^2-1/2)+zeta_0(h)
      -I_IMS,perp-0.785/(h/2)^2.                       (6)
```

The previous localization paid the axial IMS term but did not credit
`zeta_0(h)`, so (6) is a consistent strengthening rather than free margin.

For the worst sampled spectrum `t=1`, representative pilot rows are:

| `h` | `zeta_0` | generation | spectral reserve | final diagnostic `Q/nu` |
|---:|---:|---:|---:|---:|
| 0.75 | 3.9049 | 0.1600 | 0.1629 | 0.0886 |
| 0.80 | 3.3762 | 0.2108 | 0.3103 | 0.1318 |
| 0.85 | 2.9387 | 0.2693 | 0.4329 | 0.1463 |
| 0.90 | 2.5726 | 0.3352 | 0.5364 | 0.1450 |
| 1.00 | 2.0000 | 0.4874 | 0.7003 | 0.1183 |
| 1.20 | 1.2603 | 0.8561 | 0.9200 | 0.0322 |
| 1.25 | 1.1299 | 0.9575 | 0.9606 | 0.00935 |
| 1.30 | 1.0149 | 1.0611 | 0.9972 | 0 |

On the sampled grid, `h=0.85` maximizes the combined provisional budget.
The audit separately checks all seven spectrum samples at that height; the
worst remains `t=1` and all unperturbed cycles close.

## Meaning

There are now two honest candidate routes:

1. retain reversible incompressibility by using a compact full-affine buffer;
   the numerical optimum has less perturbation room but does not fail;
2. construct a divergence-free, non-gradient localized shell and replace the
   self-adjoint Poisson argument by a sectorial or probabilistic transfer.

Neither is yet a proof. The full-affine calculation assumes the actual drift
is coherent with one affine matrix across the entire buffer, and all
anisotropic finite-element values remain non-rigorous convergence evidence.
The shell rigidity itself is exact and reproduced by
`scripts/reversible_shell_rigidity_audit.py`; the compact numerical scan is
in `scripts/anisotropic_poisson_transfer_pilot.py`.
