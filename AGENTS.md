# Navier-Stokes Workspace Session Contract

This contract applies to every manual or goal-triggered Codex turn in this
standalone workspace.

- Treat `work/ns_collision` as the canonical research tree for this project.
- Treat `work/ns_collision/results/session_bookmark.json` as the only active
  Navier-Stokes session bookmark.
- Do not use, update, or depend on the Riemann-Hypothesis bookmark at
  `../l/work/rh_compute/results/session_bookmark.json`. It is migration
  provenance only.
- Do not move, delete, or rewrite the preserved source tree at
  `../l/work/ns_collision` unless the user explicitly requests a later cleanup.
- Preserve genuinely shared mathematical foundations by copying them with
  provenance. Never make either project depend on mutable files owned by the
  other project.
- Keep commands, artifact references, and checkpoint paths workspace-relative.
  Existing `work/ns_collision/...` paths are intentionally preserved.
- A single reply has a hard wall-clock ceiling of 4 hours from turn start.
- At 3 hours 15 minutes, do not begin another long calculation or proof stage.
- At 3 hours 30 minutes, enter checkpoint mode: finish only the current atomic
  row, batch, checker, or file edit; flush caches; cancel queued work; and
  prepare the report.
- Reserve the final 30 minutes for integrity checks, bookmark updates, process
  cleanup, and the user-facing report. Send the final reply by 4 hours.
- Long computations must be resumable. Prefer append-only JSONL caches,
  deterministic task order, explicit runtime limits, and atomic checkpoints.
- The default daytime resource mode uses one active Navier-Stokes compute
  worker and has a hard maximum of two. Sample total CPU for at least five
  seconds before launching compute. Use two workers only when the sampled
  baseline is at most 40%; use one worker when it is at most 60%; otherwise
  defer compute and report that the machine is already busy.
- When the user says they are going to bed, switch to unattended night mode for
  subsequent compute: use up to four active Navier-Stokes workers, adjusted
  downward when the sampled baseline would put expected total CPU above 75%.
  Keep all workers below normal priority and never exceed four.
- When the user says they are back for the day, switch immediately to daytime
  mode. If a night run has more than two active workers, stop only that
  Navier-Stokes run, retain its last fsynced atomic checkpoint, validate it,
  and resume with at most two workers.
- Run long Navier-Stokes compute below normal process priority when the
  platform supports it. Controller processes must remain effectively idle and
  must not be used to hide extra active compute workers.
- While computing, sample total CPU periodically. In daytime mode, if it
  remains above 75% for two consecutive samples, park after the current atomic
  unit, shut down only this turn's workers, validate the flushed checkpoint,
  and report back. The corresponding unattended-night threshold is 85%.
- Never leave Codex-owned workers running after the reply unless the user has
  explicitly requested that exact background process.
- Record every parked calculation in
  `work/ns_collision/results/session_bookmark.json`, including the validated
  checkpoint, resume command, unfinished obligation, and next action.
- A resumed goal starts a fresh reply budget. An active goal may trigger the
  next bounded cycle; it does not extend the current cycle past 4 hours.
- A paused goal means no autonomous background research or computation.
- Do not stop, reprioritize, or otherwise interfere with RH, field-theory,
  stock-algorithm, or other user processes. Track and clean up only processes
  launched by this workspace turn.

Machine-readable values are in
`work/ns_collision/session_runtime_policy.json`. Project paths and ownership
boundaries are in `PROJECT_PATHS.json`.
