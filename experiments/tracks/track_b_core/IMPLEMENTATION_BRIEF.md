# Track B unified r1 implementation brief

## Authority

Implement the reviewed local files in the parent workspace:

- `TRACK_B_UNIFIED_R1_EXPLORATORY_DESIGN_V3_20260824.json`
- `TRACK_B_PROTOCOL_DECISIONS_20260824.json` through `TRACK_B_PROTOCOL_DECISIONS_V8_20260824.json`
- `TRACK_B_UNIFIED_R1_MATRIX_CATALOG_DRAFT_20260824.json`
- `EXPERIMENT_PREP_FREEZE_20260824.md`
- `ORIGINAL_PLAN_UNCHANGED_20260824.txt`

The latest decision wins where older documents require internal SHA, publication-grade control machinery, or the old B7 loss/exposure rule.

## Implementation rules

- Build one coherent new implementation in this root. Do not upgrade candidate1--candidate3 in place and do not copy their control-plane, READY, receipt, manifest, run, or checkpoint identity.
- Pure scientific helpers from candidate3 may be adapted only after checking them against V3; known old defects include variable batch below64, incomplete B4 snapshots, incomplete B6 transition state, old B7 sampling/exposure, and incomplete B8 comparator evidence.
- No formal training, remote access, dependency installation, archive creation, or SHA generation/comparison during implementation.
- Use stable paths `runs/<family>/<condition>/seed_<seed>/attempt_<NNN>` and matching `reviews/...`; never overwrite attempts.
- Runtime outputs must retain resolved config, environment, status, key checkpoints, raw prediction/truth, producer metrics and reviewer metrics.
- Keep code direct and testable. Do not add signing, hash chains, distributed ledgers, permission systems, or publication-oriented abstractions.

## Required scientific paths

1. Common B1/B2/B3/B5 and B4 reference CNN training.
2. B4 paired AdamW/SGD 20k path with shared initialization/stream and first-gate/3k/20k evidence.
3. B6 new G64 upstream, exact ridge, head-only, LP→FT 500/501, layer4 and full.
4. B7 matched 128-endpoint dense absolute and relative displacement+anchor paths.
5. B8 degree2, fixed-lr/no-scheduler explicit-XY MLP, and CNN with heldout appearance fields.
6. Independent evaluator and family aggregate using raw fields and preregistered contrasts only.

## Required local verification

- Semantic validation of matrix, indices, queries and streams; no internal hashes.
- Unit tests for renderers, targets, fixed batch64, schedules, FrozenBN, B4, B6, B7, B8, metrics and support gate.
- Tiny end-to-end smoke for every distinct execution path; no formal 3000/20000-step run.
- Representative resume/state-continuity tests.
