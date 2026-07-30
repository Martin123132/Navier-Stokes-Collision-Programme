# Finite-cylinder perturbation margins

## Purpose

The complete axial mode calculation showed that compact ideal strain cores
close the dyadic renewal inequality. This note asks how much adverse
zero-order error that conclusion can tolerate before the complete generation
factor reaches one.

The calculation is exact within the piecewise finite-cylinder model. It does
not replace a variable Navier-Stokes error by its average: constant potentials
are calibration probes, not extremizers for arbitrary `L^(3/2)` data.

## Perturbed separated equations

Scale radius by `L` and time by `L^2/nu`. Write

```text
delta_core = q_core L^2/nu,
delta_shell = q_shell L^2/nu.
```

For axial OU eigenvalue `zeta_n`, the radial core equation is

```text
u_n''+u_n'/rho+R_* rho u_n'
 +(2R_*+delta_core-zeta_n)u_n=0.                 (1)
```

Its regular solution is a Kummer function with parameter

```text
A_n=1+(delta_core-zeta_n)/(2R_*),
u_n(rho)=M(A_n,1,-R_*rho^2/2).
```

After normalizing at `rho=1`, its interface logarithmic derivative is

```text
d_n=-R_* A_n M(A_n+1,2,-R_*/2)/M(A_n,1,-R_*/2). (2)
```

The shell equation is

```text
u_n''+u_n'/rho+(delta_shell-zeta_n)u_n=0.        (3)
```

The audit uses `I_0,K_0` when `zeta_n>delta_shell`, the harmonic basis at
equality, and `J_0,Y_0` when `zeta_n<delta_shell`. Thus it remains valid when
an adverse shell potential crosses an axial mode instead of extrapolating the
modified-Bessel formula beyond its domain.

## Full boundary operator and renewal

For every axial mode, equations (1)-(3) give the outer-to-inner gain `U_n`.
The complete inner-boundary profile is reconstructed as in the preceding
finite-cylinder audit:

```text
u(1,y)=sum_n c_n U_n exp(R_*y^2/2) psi_n(y).
```

Let `G=max_y u(1,y)` and `V=G^2` for the two histories. At `eta=2`,
`beta=1`,

```text
r=eta^(-2)=1/4,
gamma=exp(3R_*/24)/4,
C=V(gamma+r).                                      (4)
```

The same-generation returns renew to `gamma V/(1-rV)`. Hence complete
generation closure is equivalent to `C<1`; each reported critical potential
is the root `C=1` using the full axial mode family.

## Exact constant-potential thresholds

Here `h=H/L` is the half-height. `delta_c`, `delta_s`, and `delta_u` mean a
constant adverse potential in the core only, shell only, or both regions.

| `R_*` | `h` | base `C` | `delta_c` | `delta_s` | `delta_u` |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.5 | 0.493333 | 0.834367 | 0.859356 | 0.442662 |
| 0.5 | 1.75 | 0.691792 | 0.399520 | 0.417702 | 0.208722 |
| 0.5 | 2.0 | 0.864787 | 0.147525 | 0.156385 | 0.0765393 |
| 1.0 | 1.0 | 0.288761 | 1.56800 | 1.68494 | 0.880524 |
| 1.0 | 1.2 | 0.684048 | 0.403549 | 0.443422 | 0.215951 |

Every threshold maximum occurs at the axial centreline. Repeating the roots
with 401/61 and 801/81 grid/mode discretizations changes the worst threshold
by approximately `1.21e-5`. At zero potential, the generalized transfer
recovers the preceding finite-cylinder gain to machine precision.

The main geometric lesson is quantitative. The `R_*=0.5`, `h=1.5` core has
meaningful room for error. At `h=2`, much closer to the ideal threshold
`h=2.2558`, only a small adverse potential is enough to lose closure. A proof
cannot choose an almost-critical aspect ratio and then treat perturbations as
lower-order bookkeeping.

For equal constant amplitude, core error is slightly more damaging than shell
error in every tested geometry. A potential present in both regions is much
more damaging than either separate perturbation, as residence in the two
regions accumulates in the same visit operator.

## Scale-invariant mass calibration

For a constant physical potential `q=delta nu/L^2` on a region of volume
`V_0 L^3`,

```text
||q||_(3/2)/nu=delta V_0^(2/3).                    (5)
```

The dimensionless volumes are

```text
V_core  =2 pi h,
V_shell =2 pi h(eta^2-1),
V_full  =2 pi h eta^2.
```

The corresponding critical constant-potential masses are:

| `R_*` | `h` | core `Q/nu` | shell `Q/nu` | uniform `Q/nu` |
|---:|---:|---:|---:|---:|
| 0.5 | 1.5 | 3.72281 | 7.97569 | 4.97691 |
| 0.5 | 1.75 | 1.97553 | 4.29628 | 2.60068 |
| 0.5 | 2.0 | 0.797392 | 1.75826 | 1.04247 |
| 1.0 | 1.0 | 5.33908 | 11.9340 | 7.55500 |
| 1.0 | 1.2 | 1.55169 | 3.54655 | 2.09235 |

These numbers are scale invariant and can be compared with the earlier
critical-form budget. They are not a sufficient condition for an arbitrary
potential with the same `L^(3/2)` norm. Concentration changes the Green
operator seen by a path, and the full generation condition is stricter than
positivity of one killed-core form.

## Next PDE gate

The useful next object is the positive finite-cylinder Green operator at a
working geometry such as `R_*=0.5`, `h=1.5`, not another constant-potential
sweep. For variable `q_+`, a rigorous route must control an operator such as

```text
K_q f(x)=integral G_0(x,z) q_+(z) f(z) dz
```

through a Birman-Schwinger, Kato, or sharp form bound that also propagates to
the outer-boundary visit operator. This would turn the calibrated margin into
a legitimate condition on the non-affine strain/frame error. The outstanding
difficulty is then to derive that condition from local Navier-Stokes
coherence without assuming the desired regularity.

The transfer formulas, mode reconstruction, thresholds, refinement checks,
and mass conversions are reproduced by
`scripts/finite_cylinder_perturbation_margin_audit.py`.
