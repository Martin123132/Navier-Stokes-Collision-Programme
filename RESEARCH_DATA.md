# Research data layout

This project has a reviewable Git layer and a lossless research-data layer.

## Git layer

The repository tracks all source code, mathematical notes, tests, migration
records, session state, and small JSON audit results. Python caches, temporary
console captures, and locally built release packages are excluded.

No tracked directory is allowed to contain more than 500 direct files, and no
tracked file is allowed to exceed 10 MiB. The publication audit enforces both
limits before a push.

## Release layer

Twenty-nine large numerical files, chiefly compressed NumPy checkpoints and
directed-LDL JSON records, retain their original paths through five GitHub
release assets. The assets are split by calculation family:

```text
hypercircle transition 32064
hypercircle transition 33280
sparse-inertia checkpoints
reference eigensystems
other neutral-strip support checkpoints
```

The authoritative file and asset hashes are in
`release-manifests/research-data-v1.json`.

Restore the complete workspace:

```text
python tools/restore_research_archives.py
```

For an offline restore from pre-downloaded assets:

```text
python tools/restore_research_archives.py --archive-dir D:\path\to\assets --no-download
```

After restoration, the historical workspace paths are identical to those used
when the 519-test checkpoint was produced.
