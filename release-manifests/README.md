# Research data releases

The Git history contains source code, notes, tests, control manifests, and
small audit results. Large numerical checkpoints are stored as GitHub release
assets so the repository remains reviewable and does not approach GitHub's
per-file or repository-size limits.

`research-data-v1.json` is the authoritative map. It records every archived
file's original workspace-relative path, byte size, SHA-256 digest, containing
asset, and the SHA-256 digest of each asset.

Build the deterministic assets from a complete local workspace:

```text
python tools/build_research_release_archives.py
```

Restore a clone after the release has been published:

```text
python tools/restore_research_archives.py
```

The restore tool refuses path traversal, mismatched assets, unexpected archive
members, and conflicting local files.
