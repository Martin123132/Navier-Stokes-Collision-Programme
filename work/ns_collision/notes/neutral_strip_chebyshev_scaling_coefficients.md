# Certified Chebyshev scaling and coefficients

## Purpose

The source-oriented boundary pilot uses a degree-320 Chebyshev expansion for
one time step of length `3/8`. Before the matrix recurrence can be certified,
the exact stored normalized operator must lie inside the scaling interval and
the scalar coefficients, including their infinite tail, must be enclosed.

This stage closes those two scalar/operator-input gates only.

## Spectral scaling

For the exact stored binary pencil,

```text
H=M^(-1/2) A M^(-1/2).
```

The previously certified two-block constants imply

```text
lambda_min(H)
 >= (alpha+beta)/2
    - sqrt(((beta-alpha)/2)^2+epsilon^2)
 = 1.9476888638257819.
```

The normalized binary64 matrix is reconstructed from the hash-checked stored
mass and stiffness. IEEE `gamma_n` bounds cover inverse square roots,
diagonal scaling, symmetrization, and row summation. The resulting directed
Gershgorin upper bound is checked to be below `7988`. Therefore the exact
stored spectrum lies inside `[1.9,8000]`.

## Coefficient enclosure

For `h=3/8`, `c=(8000+1.9)/2`, and
`a=h(8000-1.9)/2`, the expansion is

```text
exp(-hH)
 = exp(-h*1.9)
   [q_0 T_0(X)+2 sum_(k>=1) (-1)^k q_k T_k(X)],

q_k=exp(-a) I_k(a).
```

The positive series

```text
q_k = exp(-a) sum_(j>=0)
      (a/2)^(2j+k)/(j!(j+k)!)
```

is evaluated with 80-decimal interval arithmetic. Orders 0 and 1 seed the
three-term recurrence through order 320; every resulting interval remains
strictly positive. The stored SciPy pilot coefficients are not assumed to be
correctly rounded. Their distance from the exact coefficient intervals is
enclosed order by order, and the resulting coefficient-error bounds are
summed in `l1`.

For the infinite order tail, orders 321 through 752 are evaluated directly by
the positive series. Its term ratio decreases in `j`, so the omitted series
tail is geometric. For the remaining Bessel orders, the same positive series
shows

```text
I_(k+1)(a)/I_k(a) <= a/(2(k+1)).
```

At order 752 this is below one and decreases thereafter, yielding the final
geometric order-tail enclosure.

## Scope

This certificate proves the scaling interval and scalar degree-320
coefficients. It does not yet enclose sparse recurrence roundoff, reduced
semigroup evaluation, boundary output multiplication, within-window suprema,
or the post-time-6 contribution. The production screen is unchanged.

The executable is
`scripts/neutral_strip_chebyshev_scaling_coefficients_certificate.py`.
