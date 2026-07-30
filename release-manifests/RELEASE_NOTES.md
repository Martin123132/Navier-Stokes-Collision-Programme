# Research data checkpoint 2026-07-30

This release preserves the large numerical checkpoints omitted from ordinary
Git history. It is a storage companion to the initial public research
checkpoint, not a theorem announcement or Clay-prize claim.

The assets are divided by calculation family so no single browser directory or
release package becomes unwieldy. Each ZIP stores files at their canonical
workspace-relative paths.

Integrity and restoration:

1. Read `release-manifests/research-data-v1.json`.
2. Verify each asset SHA-256.
3. Run `python tools/restore_research_archives.py`.
4. Run the 519-test regression command recorded in the root README.

The restore tool verifies both archive and extracted-file hashes and refuses
to replace a conflicting local file.
