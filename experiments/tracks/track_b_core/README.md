# Track B core archive

This directory is a public, archival subset of the Track B function-selection work.

Included here are the core `fsx` implementation, protocol definitions, scientific checks, and unit tests needed to understand the experiment design. Machine-specific orchestration, formal-run archives, recovery handoffs, upload/download instructions, and live execution reports are intentionally omitted.

The original code was developed for a bounded formal execution environment. Some entry points can therefore refer to orchestration components that are not part of this public subset. The archive should be read as historical implementation evidence rather than an immediately runnable package.

## Layout

- `fsx/`: data, models, evaluation, aggregation, schedules, and scientific diagnostics.
- `protocols/`: frozen experiment definitions retained for interpretation.
- `tests/`: selected configuration, data-semantics, evaluation, model, and science tests.
- `IMPLEMENTATION_BRIEF.md`: compact description of the implementation contract.

For project-level context and evidence limits, see `../../../docs/EXPERIMENT_SUMMARY_CN.md` and `../../../docs/ARCHIVE_AND_PATH_NOTE.md`.
