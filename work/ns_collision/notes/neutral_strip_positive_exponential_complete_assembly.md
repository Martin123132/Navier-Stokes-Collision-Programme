# Complete positive-exponential hypercircle assembly

## Scope

This stage extends the passing 512-triangle analytic moment pilot to every
triangle of the stored `h=0.06` mesh. It assembles the positive-exponential
blocks needed by the weighted hypercircle threshold pencil:

- the globally oriented RT0 mass `P`;
- the P0 source mass diagonal `W`;
- the triangle-area diagonal `D`;
- the P1-P0 load `B`; and
- the exact signed divergence incidence `N`.

The existing directed Gaussian P1 stiffness block `A` is a separate artifact.
No threshold-pencil factorization or inertia claim is made in this stage.

## Resumable contract

The mesh has 30,954 triangles, 46,697 global edges, and 15,211 interior P1
states. Its fingerprint is

`174d325adf2b1a7f6c70a023982060bc492dbb279d267e4cdc2a2a85e9270835`.

Assembly uses degree-22 directed simplex moments for `exp(+x^2/2)`, q12
central values on every triangle, and independent q18 cross-checks at stride
257 and at the final triangle. Each 512-triangle chunk is written to an atomic
NPZ checkpoint. The companion JSON binds the checkpoint hash, mesh
fingerprint, dimensions, quadrature contract, and next complete triangle.
Resume rejects changed contracts, hashes, counts, indices, non-finite values,
negative radii, and non-unit divergence signs.

For each globally signed RT0 contribution, the local analytic interval is
stored as a q12 center and an outward radius. Duplicate global entries are
summed in CSR form and their radii are inflated for binary summation using the
maximum two-triangle incidence count. The central and error CSR structures
must then agree exactly.

## Complete result

The run completed all 30,954 triangles. It performed:

- 216,678 q12 containment checks with zero failures; and
- 854 independent q18 containment checks with zero failures.

The complete-mesh maxima are:

- area interval width: `4.7704895589362195e-18`;
- load interval width: `2.0599841277224584e-18`;
- P0 source-mass width: `1.5489831639570184e-12`;
- RT0 mass-entry width: `4.8066794988699257e-10`;
- q12/q18 source difference: `7.993605777301127e-15`; and
- q12/q18 RT0 difference: `1.6370904631912708e-11`.

The smallest local certified lower bounds are
`0.0005695216635132152` for P0 source mass and
`0.14539799271158366` for an RT0 diagonal entry.

Global sparse assembly gives:

- `P`: shape `46697 x 46697`, 232,421 nonzeros;
- `B`: shape `15211 x 30954`, 91,124 nonzeros;
- `N`: shape `30954 x 46697`, 92,862 exact `+1/-1` entries;
- `W`: 30,954 diagonal values; and
- `D`: 30,954 diagonal values.

Both the central `P` matrix and its radius matrix are exactly symmetric. The
largest aggregated `P` entry radius is `5.02617718315517e-10`. Stored lower
endpoints for every W, D, and B entry are strictly positive.

The deterministic matrix-array hash is

`6deb9b9c41f842320cfeee2abf64a81047a45e234fe98b9cd66bbb55127b0274`.

The final matrix archive SHA-256 is

`c2ab8e9f0ffaa217abdc8b6c48dccc54e57e98daeb353f21df8be60906bd376b`.

The completed contribution checkpoint SHA-256 is

`5190a21d4b2b746fbf0867842b05d7a487e1a1ba146cf592ae07619a79f9fa43`.

An independent replay loaded the completed checkpoint, rebuilt every sparse
array, and matched the stored archive entry-for-entry and hash-for-hash.

## What this closes

The complete-mesh positive-exponential `P/W/D/B` entry-enclosure flag and its
hash-bound checkpoint flag are now true. Together with exact `N`, this removes
the first missing matrix-assembly obligation identified by the hypercircle
pilot.

It does not prove the full pencil inertia, `kappa_h < 0.045`, the global
weighted Ritz projection bound, or continuum spectral capture. The next stage
is a bounded central factorization and symbolic-fill audit for the
123,816-dimensional pencil, followed only if viable by a directed verified
inertia computation. Point-source, conormal, polygon-to-circle, and the larger
Navier-Stokes regularity obligations remain separate. Nothing here is a
regularity or Clay-prize proof.
