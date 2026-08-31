# Research timeline

This file gives a navigation-level view of the project. Detailed results and evidence limits are recorded in `EXPERIMENT_SUMMARY_CN.md` and `S1_SUMMARY_CN.md`.

## Early phase-sensitivity experiments

The initial experiments used synthetic coordinate regression to examine whether ResNet18 stride/downsampling introduced translation-phase sensitivity, and compared baseline, anti-aliased, and no-GAP variants. A longer baseline run showed that early periodic error should not be treated as a fixed localization ceiling.

## Bézier inverse problems

The next experiments changed the target from a single position to Bézier control points. Cubic, quadratic, in-frame, and off-canvas settings were used to separate training-budget limitations from renderer identifiability and out-of-support behavior.

## Phase 1–1.8

This stage studied how frozen pooled representations carry translation and relative geometry, including linear probes, variance decomposition, subspace removal, robustness checks, and unseen translation/shape combinations.

## Phase 2

Phase 2 expanded the controlled factors: support coverage, backbone and head choices, padding, content, symmetry, dense rendering, and recovery behavior. The aim was to compare explanations under matched synthetic protocols rather than rely on one architecture.

## Phase 3 and mechanism probes

Phase 3 introduced operator and group-law tests, head-semantics checks, equivariance diagnostics, controlled CNNs, and more explicit representation analyses. Separate mechanism probes examined spectral structure, optimizer behavior, tangent-space effects, and small MLP controls.

## S0 / S1 Neural Affine work

S0 and S1 focused on clean-room materialization, fixed protocols, asset validation, Neural Affine controls, crossover analyses, and independent checks. Execution closure, support-gate eligibility, and scientific promotion were recorded separately.

## Track A

Track A tested a narrowly specified finite-boundary mechanism with small models and controlled visible/safe regions. Its conclusions are limited to that mechanism and protocol.

## Track B

Track B organized a broader function-selection matrix, formal evaluation code, scientific oracles, and post-hoc diagnostics. The public snapshot keeps the core implementation and tests while omitting formal run archives and machine-specific orchestration.

## Project closeout

The experimental phase has ended. The local workspace remains the complete evidence archive; this repository is the readable, public-facing subset.
