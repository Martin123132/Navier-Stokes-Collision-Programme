# Sharp cubic envelope at a zero pressure face

Status: exact smooth-class optimization of the conditional pressure edge
against the cubic terminal-weight Fisher penalty. The reciprocal zero-face
singularity is removed. The resulting critical `L^(3/2)` pressure remainder
is not yet absorbed, and no Navier-Stokes regularity conclusion is claimed.

## 1. Why the reciprocal remainder was avoidable

For one partition direction write

```text
H=A+B,
D=A-B,
```

where the two conditional face values satisfy `A,B>=0`. Consequently

```text
H>=|D|,

H D^2>=|D|^3.                                      (1.1)
```

At partition frequency `m`, the exact conditional pressure/Fisher pair is

```text
P_j=mean_other[D e_j],

D_(lambda,j)
 =(m^2/16)mean_other[H D^2].                       (1.2)
```

The previous quadratic Young split gave

```text
D e-(nu m^2/16)H D^2
 <=4e^2/(nu m^2 H),
```

which diverges when `H` approaches zero. That estimate treats `H` and `D`
as independent, discarding (1.1).

## 2. Exact coefficient optimization

For fixed edge flux `e`, put

```text
c=nu m^2/16.
```

Using (1.1),

```text
D e-cH D^2
 <=|D||e|-c|D|^3.                                  (2.1)
```

The right side is maximized at

```text
|D|=sqrt[|e|/(3c)],

sign(D)=sign(e).
```

The exact value is

```text
sup_(A,B>=0)[
  (A-B)e-c(A+B)(A-B)^2]

=2|e|^(3/2)/(3sqrt(3c))

=8|e|^(3/2)/(3sqrt(3)m sqrt(nu)).                   (2.2)
```

This is sharp. Equality in `H>=|D|` requires `H=|D|`, so the optimizer has
one of `A,B` exactly zero. The zero face is therefore the extremizer, but
its optimized cost is finite.

Pointwise application of (2.2) gives the floor-free bound

```text
P_j-nu D_(lambda,j)

<=8/[3sqrt(3)m sqrt(nu)]
  mean_other |e_j|^(3/2).                           (2.3)
```

Equation (2.3) preserves the full nonnegative directionwise face supremum.
It does not require an `A_2` weight or a positive lower bound.

## 3. Natural pressure exponent

For

```text
e_j(x_hat_j)
 =mean_(x_j)[p u_j partial_j phi_+(m x_j)]
```

and

```text
partial_j phi_+=-(m/2)sin(m x_j),
```

normalized one-dimensional integration gives

```text
mean |partial_j phi_+|^3=m^3/(6pi).
```

Conditional Holder therefore yields, when `|u|<=U`,

```text
|e_j|^(3/2)
 <=U^(3/2)m^(3/2)/sqrt(6pi)
   mean_(x_j)|p|^(3/2).                             (3.1)
```

Combining (2.3)-(3.1), one direction costs at most

```text
8/[9sqrt(2pi)]
 U^(3/2)sqrt(m/nu)
 integral |p|^(3/2).                               (3.2)
```

Summing three directions gives

```text
8/[3sqrt(2pi)]
 U^(3/2)sqrt(m/nu)
 integral |p|^(3/2).                               (3.3)
```

At the intrinsic frequency `m=U/nu`, this becomes

```text
8/[3sqrt(2pi)]
 (U^2/nu) integral |p|^(3/2).                      (3.4)
```

The pressure estimate

```text
||p||_(3/2)<=C_R||u||_3^2
```

then converts (3.4) to

```text
8 C_R^(3/2)/[3sqrt(2pi)]
 (U^2/nu)||u||_3^3.                                (3.5)
```

This is critical and floor-free. It is not closed: the Leray energy
inequality does not control `U^2` in time.

## 4. Scaling check

Under

```text
u_(a,m)=a u(mx),
p_(a,m)=a^2p(mx),
lambda_(a,m)=a lambda(mx),
```

the conditional edge density scales as

```text
e ->a^3m e.
```

The cubic envelope in (2.3) scales as

```text
a^(9/2)m^(1/2)/sqrt(nu),
```

while the velocity and weight Fisher terms scale as

```text
nu a^3m^2.
```

Their ratio is exactly

```text
[a/(nu m)]^(3/2)=Re_cell^(3/2).                    (4.1)
```

Thus the sharp zero-face route retains the same intrinsic frequency, with a
`3/2` Reynolds power instead of the reciprocal Young route's square.

## 5. Taylor-Green edge stress

For

```text
u=(sin x cos y,-cos x sin y,0),
p=(cos 2x+cos 2y)/4,
```

and partition frequency one, direct conditional integration gives

```text
e_x(y)=cos(y)[1/32-cos(2y)/16],

e_y(x)=-cos(x)[1/32-cos(2x)/16],

e_z=0.
```

At each conditional point, choosing the zero-face optimizer from Section 2
reproduces (2.2) to floating-point roundoff. This confirms the sharpness on
an actual incompressible pressure field, rather than only scalar algebra.

The directionwise optimizer permits independent face coefficients for every
conditional point. It therefore upper-bounds the globally compatible
eight-cell coefficient supremum but need not equal it.

## 6. What changed

Established:

- the reciprocal zero-face blow-up is an artifact of quadratic Young;
- the full nonnegative directionwise edge supremum is finite and exact;
- its natural pressure exponent is `3/2`;
- its pressure/Fisher scaling is `Re_cell^(3/2)`;
- the Taylor-Green pressure edge attains the scalar zero-face envelope.

Still open:

- compatibility of all conditional optimizers with one global partition
  coefficient vector;
- quantitative absorption of the pressure `L^(3/2)` remainder;
- signed cancellation among neighboring edges and levels;
- low-regularity passage and global regularity.

## 7. Next theorem target

The next bounded stage should evaluate the globally compatible nonnegative
coefficient supremum on the complete eight-cell graph. Its cubic objective
is coupled because one cell coefficient participates in three neighboring
edges.

That graph optimization must:

1. retain antisymmetric edge cancellation;
2. compare its value with the directionwise envelope (2.3);
3. impose the `2:1` intrinsic-scale adjacency rule;
4. survive Taylor-Green co-scaling and the seed-81 pressure adversary;
5. determine whether compatibility produces a strict constant gain or
   merely reproduces the `L^(3/2)` obstruction.
