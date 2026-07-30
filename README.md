# Navier-Stokes Collision Programme

[![Focused checks](https://github.com/Martin123132/Navier-Stokes-Collision-Programme/actions/workflows/focused-checks.yml/badge.svg)](https://github.com/Martin123132/Navier-Stokes-Collision-Programme/actions/workflows/focused-checks.yml)

This is the standalone workspace for the Navier-Stokes research fork that grew
out of the non-collision mechanism studied in the Riemann-Hypothesis project.
The conceptual origin is preserved as provenance; neither workspace is a
runtime or file dependency of the other.

Status: exploratory research. Nothing in this workspace currently proves
global regularity or finite-time blow-up for three-dimensional incompressible
Navier-Stokes, and nothing here constitutes a Clay-prize solution.

## Current checkpoint

For the explicit repaired parallel-shear restart family, the current corpus
proves the restart-time estimate

```text
|g_N'''(0)| <= C0 max(nu,nu^-1)^13 N^11
```

for odd `N>=3`, with an explicit deliberately coarse constant. The proof uses
an exhaustive third-tree topology ledger, boundary-safe packet differences,
and separate four-high and six-high internal-output estimates.

This is not yet the uniform estimate on `0<=s<=T/N^2`. The live problem is to
propagate the restart-time bounds along the evolving Navier-Stokes/adjoint
trajectory without exceeding the `N^11` budget.

## Canonical paths

- Research corpus: `work/ns_collision`
- Research overview: `work/ns_collision/README.md`
- Session bookmark: `work/ns_collision/results/session_bookmark.json`
- Runtime policy: `work/ns_collision/session_runtime_policy.json`
- Ownership and path map: `PROJECT_PATHS.json`
- Migration record: `MIGRATION_MANIFEST.json`
- Source snapshot hashes: `migration/SOURCE_TREE_SHA256.json`
- Split validator: `tools/validate_workspace_split.py`
- Large-data manifest: `release-manifests/research-data-v1.json`
- Data restoration tool: `tools/restore_research_archives.py`
- GitHub publication audit: `tools/audit_github_publication.py`

All historical `work/ns_collision/...` paths were retained deliberately, so
existing scripts and resume commands run from this workspace root without
rewriting the research corpus.

## Clone and restore

The Git tree contains all source, notes, tests, and small audit records. The 29
largest numerical checkpoints are divided among five hashed release assets so
GitHub does not truncate a directory or accumulate a 1.2 GB initial history.
Restore their exact original paths with:

```text
python tools/restore_research_archives.py
```

The full data layout and integrity model are described in `RESEARCH_DATA.md`.

The latest focused theorem check does not require the large archives:

```text
python -m unittest work/ns_collision/tests/test_annular_parallel_shear_third_internal_shell_lemma.py
```

After restoring the release data, the complete recorded regression is:

```text
python work/ns_collision/scripts/run_full_regression_checkpoint.py --expected-count 519
```

The active obligation, validated checkpoint, resume command, and next action
live in `work/ns_collision/results/session_bookmark.json`.

## Migration provenance

The NS corpus contains one historical comparison to the zeta non-collision
mechanism. That comparison is mathematical provenance, not an import, path
dependency, shared checkpoint, or claim that the two problems are equivalent.
Any future theorem or computation used by both projects must be copied into
each workspace with its source and hash recorded.

`MIGRATION_MANIFEST.json` and `migration/` preserve the verified 2026-07-23
split from the RH workspace. `tools/validate_workspace_split.py` additionally
checks the original preservation copy when run on the source workstation; a
normal GitHub clone is not expected to contain that external directory.
