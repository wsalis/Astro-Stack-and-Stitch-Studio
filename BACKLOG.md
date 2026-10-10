# Backlog

## Handoff Notes

- Keep the current big-rig stacking and finished-tile mosaic workflows stable; do not introduce a separate Seestar mode.
- Preserve the stacking-centric main GUI. Advanced distributed controls, if implemented, belong in a separate window opened by a button beside Mosaic.
- Reuse the existing GPUStacker pipeline, data formats, settings, and test conventions. Avoid unrelated refactors and avoid changing scientific behavior without benchmark evidence.
- Before implementation, inspect current code/tests because these notes describe desired behavior, not guarantees about the current implementation. Each item should be implemented and validated independently where practical.

## Items

- [ ] **Resume interrupted runs (priority: large, high-frame-count sessions).** Avoid repeating completed stages after an 8–16-hour run is interrupted.
	- Checkpoint candidate boundaries: completed analysis; registered frames plus normalization/weight metadata; first-pass global stack before local-normalization fitting; fitted local-normalization maps before the final stack; completed master before optional products.
	- Persist a versioned manifest containing stage, input identities, reference, relevant settings, software/schema version, expected artifact paths/shapes, and completion/integrity markers. Validate it before reuse; explain incompatibilities and offer a clean restart rather than silently mixing stale artifacts.
	- Write checkpoints atomically (temporary artifact then commit marker/rename); treat partial writes as incomplete. Never infer completion only from file existence.
	- Make checkpoint retention/cleanup explicit. A successful run may clean temporary checkpoints according to the chosen setting; failed/interrupted runs must preserve valid checkpoints for recovery.
	- Acceptance: simulate interruption after each supported boundary and verify resume skips completed stages, produces output equivalent to a clean run within defined tolerances, rejects changed inputs/settings, and safely handles truncated/corrupt checkpoint files. Mid-band resume is explicitly deferred unless separately designed.

- [ ] **Stack pre-registered frames.** Provide a path for feeding compatible registered frames directly into stacking without calibration, analysis, or registration.
	- Define supported inputs first: preferably GPUStacker's retained registered-frame store and its manifest/index, rather than guessing normalization/registration metadata from arbitrary images. A folder picker may select that store.
	- Validate common channel count, dimensions, reference grid, normalization metadata, and required files; fail before stacking with a filename-specific explanation if incompatible.
	- Preserve frame identity and available metadata in reports. Clarify which weighting/filter controls can be recomputed from stored data and which require original analysis results.
	- Acceptance: an existing retained store can produce a stack matching the ordinary pipeline's stack stage; malformed, incomplete, or incompatible stores fail preflight without modifying source files.

- [ ] **Visual reference-frame review.** Let the user inspect the automatic best-reference choice and candidate frames before registration, so they can confirm the registration anchor.
	- After Analyze & rank, expose the ranked candidates with enough identity/quality information to distinguish them (filename, rank, stars, FWHM, and existing ranking metrics). Selecting a candidate for preview must not silently change the chosen reference.
	- Show a full-frame preview with a non-destructive display stretch and relevant metadata. Load/decode only the selected candidate on demand; do not retain full-resolution previews for the entire input set. Preview the image using the same calibration/debayer interpretation as the pending run where applicable, without modifying source files.
	- Preview the automatic top-ranked reference by default. Allow previewing any candidate, including the manually selected reference, and provide an explicit action to use a candidate as the reference. Clearly distinguish automatic/manual reference selection from the currently previewed frame.
	- Acceptance: tests verify ranking and preview selection do not mutate reference settings; explicit selection does; preview handles supported FITS/XISF inputs, CFA/debayer settings, missing/corrupt files, and large images without loading all candidates into memory. GUI test confirms the run uses exactly the reference shown as selected.

- [ ] **Large-run resource preflight.** Estimate temporary and output storage from actual input groups and their dimensions/channel layouts, then compare estimates with free space on each relevant volume.
	- Do not assume all frames match the first file. Include registered arrays, optional rejection masks/maps, drizzle/deconvolution intermediates, checkpoint retention, and planned outputs; document assumptions and uncertainty.
	- For batch runs, report per-group and total estimates. Distinguish a hard insufficiency from a configurable low-space warning.
	- Acceptance: tests cover mixed dimensions/channels, optional products, batch groups, and paths on separate volumes; estimates are conservative against measured peak usage on representative runs.

- [ ] **Maybe: distributed stacking for very large datasets.** Optional multi-Windows worker execution for exceptionally large stacks (for example, thousands of Seestar subs), only if shared-storage benchmarks show a worthwhile speedup.
	- Coordinator (main PC) performs analysis and registration and stores every registered frame plus required metadata on a shared network SSD. Workers read the same store via a shared UNC path; do not depend on machine-specific drive letters.
	- Keep this as stacking-only distribution: workers execute globally normalized first-pass band tasks; coordinator assembles the first-pass master and fits local-normalization maps; workers execute final-stack band tasks. All frames must contribute to each assigned spatial band so rejection semantics match the single-machine path.
	- Dynamically schedule work from measured worker throughput rather than asking the user for a fixed percentage split. A disconnected worker's uncommitted task is requeued; completed task outputs are validated and not overwritten while in use.
	- Verify worker GPU/backend availability, software version, settings compatibility, shared paths, and output write access before dispatch. Initially support Windows workers using the same GPUStacker checkout; keep the Mac/MPS path out of scope.
	- UI: advanced button beside Mosaic opens a dedicated worker/setup window; do not add distributed controls to the main stacking form. Include a single-machine fallback and clear worker/run status.
	- Gate implementation on a representative shared-SSD benchmark (read throughput, GPU utilization, total wall time) and define acceptable single-vs-multi-worker output tolerances. Test worker loss, duplicate/stale task results, coordinator restart, and storage disconnect before calling it reliable.

- [ ] **Interruption-safe run and outputs.** Make cancellation and failures leave a clear, recoverable state and protect completed deliverables.
	- Define safe cancellation boundaries for long stages and distinguish cancel from abrupt process/machine loss. Integrate with checkpoint/resume rather than building a competing recovery mechanism.
	- Save each final product atomically so a crash cannot leave a partial FITS/CSV/JSON that looks complete. Record per-product completion and failure in the run report.
	- If an optional step (for example, drizzle or deconvolution) fails, retain the successful master and other completed products; report the failed step and a way to retry it where supported.
	- Acceptance: inject cancellation/failure during representative stages; verify valid completed files remain readable, incomplete files are not presented as complete, and rerun/recovery behavior is documented.

- [ ] **Post-stack CSV/JSON filtering and rejection diagnostics.** Keep diagnostics in generated files; do not add a metrics browser or filtering interface to the stacking-centric GUI.
	- Reuse existing per-frame CSV/JSON fields where possible. Summarize input/eligible/stacked/excluded counts and exclusion reasons, with the effective filter thresholds and relevant session baselines.
	- Ensure per-frame records include star count, FWHM, noise, background, transparency, PSFSW weight, photometry-star count, and registration confidence/residuals when available. Represent unavailable values explicitly, not as misleading zeroes.
	- Clearly distinguish whole-frame exclusion from per-pixel low/high sample rejection. Add per-frame rejection attribution only if it can be computed without ambiguous or excessive-cost inference; otherwise report aggregate/maps and state that per-frame attribution is unavailable.
	- Version the report schema and document metric definitions so users can compare filter-setting experiments. Acceptance: fixtures verify counts reconcile, reasons are stable, and CSV/JSON values agree.

- [ ] **Regression coverage for large short-exposure and big-rig data.** Protect both high-frame-count throughput and established image quality.
	- Add a reproducible many-file short-exposure stress/performance workload; avoid requiring the full multi-hour live dataset for routine CI. Record hardware, storage, settings, and wall-time/throughput context for performance claims.
	- Retain representative big-rig quality benchmarks and compare outputs using established metrics/tolerances, not visual judgment alone. Keep reference inputs immutable and document provenance/access requirements.
	- Acceptance: functional tests run in normal CI; expensive stress/real-data benchmarks are opt-in or scheduled. A performance improvement is not accepted if quality metrics regress beyond agreed tolerances.
