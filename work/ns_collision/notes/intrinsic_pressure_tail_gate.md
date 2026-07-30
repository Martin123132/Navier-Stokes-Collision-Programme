# Intrinsic pressure tail and the zero-face gate

Status: exact smooth periodic tail decomposition, conditional intrinsic-scale
absorption theorem, and a no-go for obtaining the required zero-face result
by a uniform weighted singular-integral estimate. The floor-free signed edge
bound, critical estimate, and Navier-Stokes regularity remain open.

## 1. The high pressure tail

On the periodic domain, with zero mean pressure,

```text
p=-R_i R_j(u_i u_j).
```

Let `Q_m` retain Fourier modes with `|k|>m`. Split

```text
u=u_<+u_>,

supp(u_<) contained in {|k|<m/2}.
```

The low-low product has support below `m`, so

```text
Q_m p
 =-R_iR_j Q_m[
    u_<i u_>j+u_>i u_<j+u_>i u_>j].               (1.1)
```

This is the dyadic/paraproduct reason that a genuine pressure tail must
contain at least one high velocity input.

There is also a direct derivative estimate. The pressure multiplier has
matrix norm one, and

```text
|grad(u tensor u)|<=2|u||grad u|.
```

Consequently

```text
||Q_m p||_2
 <=m^(-1)||grad p||_2
 <=m^(-1)||grad(u tensor u)||_2
 <=2m^(-1)||u||_infinity||grad u||_2.              (1.2)
```

For any smooth terminal weight,

```text
|integral Q_m p u dot grad lambda|
 <=2||u||_infinity^2 m^(-1)
   ||grad u||_2||grad lambda||_2

 <=||u||_infinity^2 m^(-1)
   [||grad u||_2^2+||grad lambda||_2^2].            (1.3)
```

Equation (1.3) is scale-correct and completely rigorous. Its localization
is the delicate part.

## 2. Conditional intrinsic absorption

If

```text
lambda(x)>=lambda_*>0,
```

then

```text
nu integral lambda[
  |grad u|^2+|grad lambda|^2]

>=nu lambda_*[
  ||grad u||_2^2+||grad lambda||_2^2].
```

Thus the high pressure tail is absorbed whenever

```text
m>=||u||_infinity^2/(nu lambda_*).                 (2.1)
```

If the weight floor is comparable to amplitude,

```text
lambda_*>=theta||u||_infinity,
```

then (2.1) is exactly the intrinsic local-Reynolds condition

```text
||u||_infinity/(nu m)<=theta.                      (2.2)
```

This proves that the proposed intrinsic frequency is sufficient for the
unweighted tail once a comparable positive weight floor is available.
It does not supply that floor.

## 3. Exact amplitude-frequency gate

Consider the two-dimensional Taylor-Green field embedded in three
dimensions:

```text
u=(sin(x)cos(y),-cos(x)sin(y),0),
p=(cos(2x)+cos(2y))/4.
```

It is divergence-free and

```text
(u dot grad)u+grad p=0.
```

Set

```text
g=u dot grad p
 =(cos(3x)cos(y)-cos(x)cos(3y))/4,

lambda=L-beta g.
```

Since `|g|<=1/2`, the weight is positive when `L>beta/2`. Direct periodic
integration gives

```text
mean[p u dot grad lambda]       =beta/32,
mean[lambda|grad u|^2]          =L,
mean[lambda|grad lambda|^2]     =5L beta^2/16.
```

All nonzero pressure modes have `|k|=2`, so this is entirely a pressure tail
above cutoff one.

Co-scale by

```text
u_(a,n)=a u(nx),
p_(a,n)=a^2 p(nx),
lambda_(a,n)=a lambda(nx).
```

Then

```text
tail pressure flux =a^4 n beta/32,

total Fisher
 =nu a^3 n^2 L(1+5beta^2/16).
```

Their ratio is

```text
[a/(nu n)] beta/[32L(1+5beta^2/16)].               (3.1)
```

Thus fixed-frequency universal absorption is impossible even when the
entire pressure is in the designated tail. Choosing `n` proportional to
`a/nu` makes the ratio scale invariant. Intrinsic adaptation is necessary;
this example does not disprove it.

## 4. Why the positive floor cannot be inserted

The terminal dual supremum contains nonnegative weights approaching zero.
A standard attempt to localize (1.2) would use weighted
Calderon-Zygmund bounds. Those constants cannot be uniform in the zero-face
limit.

On the circle take

```text
w_epsilon(x)=epsilon+sin(x/2)^2.
```

Its whole-circle averages are

```text
mean(w_epsilon)=epsilon+1/2,

mean(1/w_epsilon)=1/sqrt[epsilon(epsilon+1)].
```

Therefore its `A_2` characteristic is at least

```text
(epsilon+1/2)/sqrt[epsilon(epsilon+1)]
 approximately 1/(2sqrt(epsilon)).                 (4.1)
```

The blow-up can be seen without invoking an abstract theorem. For

```text
f_epsilon=1/w_epsilon,
```

the periodic Hilbert transform is explicitly

```text
Hf_epsilon
 =sin(x)/[
   2sqrt(epsilon(epsilon+1))w_epsilon].
```

The ratio of weighted output norm squared to weighted input norm squared is

```text
(epsilon+1/2)/sqrt[epsilon(epsilon+1)]-1.           (4.2)
```

Hence the explicit operator-norm lower bound grows like

```text
epsilon^(-1/4)/sqrt(2).
```

This proves that a uniform arbitrary-weight singular-integral localization
cannot bridge (1.2) to the zero-face Fisher terms. It is a no-go for that
proof route, not a pressure-specific counterexample to every possible signed
edge estimate.

## 5. What is established

Established:

- the exact low-low-free dyadic pressure-tail identity;
- the scale-correct unweighted `L^2` pressure-tail estimate;
- intrinsic absorption under a comparable positive weight floor;
- an exact co-scaled tail family reconfirming local-Reynolds necessity;
- explicit divergence of the weighted singular-integral constant as the
  terminal weight approaches a zero face.

Not established:

- a uniform floor-free pressure-tail estimate;
- preservation of the full terminal dual supremum;
- absorption of the exact signed partition edge;
- low-regularity passage or global regularity.

## 6. Next theorem target

The conservative partition identity must now be used before absolute values.
Across a neighboring edge, pressure transfer is antisymmetric; only the
coefficient difference survives. The next target is:

> A signed dyadic pressure-flux Carleson estimate on the balanced intrinsic
> cover, where neighboring transfers are summed first and only coefficient
> mismatch is charged.

The `2:1` balanced cover and Lipschitz intrinsic radius should be tested for
square summability of these mismatches without an `A_2` floor. Any candidate
must survive both the Taylor-Green co-scaling family and the zero-face weight
limit above.
