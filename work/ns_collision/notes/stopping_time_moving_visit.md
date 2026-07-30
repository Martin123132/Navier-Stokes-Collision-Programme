# Stopping-time moving-cylinder visit

## Purpose

The continuous cubic Fisher no-go rules out keeping a full tensor
square-root partition active throughout the wide divergence-free visit. It
does not rule out using the cubic partition as a conservative probability law
at stopping times.

This note separates those two roles. A label is chosen at buffered entry and
held fixed through one moving-cylinder visit. Relabeling occurs only at exit
or at a separately declared coherence-failure stopping time. Because no
spatial cutoff is differentiated inside the visit, the continuous cubic
Fisher term is absent.

## Moving stopped domain

Let the physical diffusion be

```text
dX_t=b(X_t,t)dt+sqrt(2nu)dW_t.                         (1)
```

For label `j`, prescribe an absolutely continuous centre and orthogonal frame
and define

```text
C_j(t)=c_j(t)+L O_j(t)D,                              (2)
```

where `D` is the fixed dimensionless cylinder and `L` is constant during one
visit. At the first buffered entry time `tau`, choose one admissible label
from a normalized cubic law `pi_j(X_tau,tau)`. Hold that label until

```text
sigma_j=inf{t>=tau:
            O_j(t)^T[X_t-c_j(t)]/L not in D}.          (3)
```

The label is a state of the stopped process, not a continuously applied
factor `sqrt(pi_j(x,t))`.

## Exact moving-coordinate SDE

Put

```text
Y=O_j^T(X-c_j)/L,
Omega_j=O_j^T O_j',
s=nu(t-tau)/L^2.                                      (4)
```

Since `Omega_j` is skew, Ito's formula gives

```text
dY_s=[(L/nu)O_j^T(b-c_j')
      -(L^2/nu)Omega_j Y_s]ds+sqrt(2)dB_s.             (5)
```

The rotated stochastic integral is Brownian: a predictable orthogonal
finite-variation rotation preserves quadratic covariance. The domain in (5)
is the fixed cylinder `D`.

Write the physical fitted shell as

```text
b_ref,j=(nu/L)O_j b_hat_ref(Y)                         (6)
```

and define

```text
e_j=b-c_j'-O_j'O_j^T(x-c_j)-b_ref,j.                  (7)
```

Then the dimensionless drift mismatch in (5) is exactly

```text
e_hat_j=(L/nu)O_j^T e_j.                              (8)
```

Translation and frame motion have therefore been removed before any form
estimate. Rigid frame velocity is divergence-free, so `e_j` remains
divergence-free when `b` and `b_ref,j` are.

## Critical scaling

The sector norms are unchanged by (4):

```text
||e_hat_j||_(L3(D))
 =||e_j||_(L3(C_j))/nu,

||q_hat_j||_(L3/2(D))
 =||q_j||_(L3/2(C_j))/nu.                             (9)
```

Thus no scale factor has been hidden in the moving-domain conversion. With

```text
alpha=S_3 ||q_+||_(3/2)/nu,
beta=sqrt(S_3)||e||_3/nu,                             (10)
```

the existing sector theorem applies directly:

```text
||B_(q,e)||/||B_0||
 <=1+chi_sec (alpha+beta)/(1-alpha).                  (11)
```

For the current pilot constants, renewal closes when

```text
beta<0.2042788673-1.2042788673 alpha.                 (12)
```

## Relabeling at exit

At `sigma_j`, choose the next label from the new admissible law
`pi_k(X_sigma,sigma)`. The row-stochastic kernel

```text
K_jk=pi_k                                               (13)
```

pushes every old label law to `pi`. Jensen makes its backward observable map
contractive from the new dynamic `L^p` law to the old one. The independent
replica-pair lift is contractive as well. The number of old and new labels may
differ.

This transition has no Fisher cost because it is made at a stopping time; no
spatial derivative of `pi` enters the interior generator. A continuous
linear partition may still be retained separately in the global pressure
identity. Since it is not used as a square-root energy localization, its role
is only to keep pressure as conservative inter-cell flux.

## What this repairs

The revised cycle is

```text
dynamic physical boundary law
 -> conservative label choice at buffered entry
 -> one fixed-label moving-cylinder visit
 -> sector-controlled boundary operator
 -> conservative relabeling at exit
 -> exterior return or true dyadic split.              (14)
```

Inside the visit there is no continuous cubic partition and hence no
`3.6701` Fisher load. The translation, rotation, potential, and drift terms
are each charged once in (7)-(12).

## Remaining gates

This exact stopped-process identity is still conditional. A proof must now
resolve three dynamical issues.

1. **Early coherence failure.** If the fitted affine budget fails before the
   geometric exit (3), an immediate relabel may carry no visit contraction.
   Such failures must be paid by a true scale split, accumulated decay, or a
   bounded bad-occupation estimate.
2. **Hitting-law conversion.** The actual moving-cylinder entry and exit laws
   must be compared with the Perron/Doob boundary measure within the existing
   mismatch allowance.
3. **Leray-level geometry.** Centres and frames must be constructed from
   deterministic local averages of the solution, with absolutely continuous
   motion and a projector treatment of repeated strain eigenvalues.

The moving-coordinate identity, critical scaling, stopping-time Markov
contraction, and sector closure arithmetic are reproduced by
`scripts/stopping_time_moving_visit_audit.py`.
