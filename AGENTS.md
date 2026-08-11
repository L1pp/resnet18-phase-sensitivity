# AGENTS.md

## Cursor Cloud specific instructions

This repo is a self-contained set of offline PyTorch research experiments (CLI
scripts only). There are no servers, databases, ports, or long-running services
to start; "running the app" means invoking the experiment scripts.

### Environment
- Python dependencies are installed into a virtualenv at `.venv/` (git-ignored).
  It is NOT auto-activated. Run scripts with `.venv/bin/python ...` (or
  `source .venv/bin/activate` first).
- PyTorch is the CPU-only build (`+cpu`). `torch.cuda.is_available()` is `False`;
  all scripts auto-fall back to CPU. Full-size training profiles are slow on CPU,
  so prefer the fast validation paths below unless you specifically need full runs.

### Running / validating (see README.md for the full command list)
- Fast, no-training sanity checks: the `check` stages, e.g.
  `.venv/bin/python resnet_phase_experiment.py check` and
  `.venv/bin/python supplemental_baseline_long.py --check-only`.
- Fast end-to-end smoke pipelines (render → train → analyze → identifiability):
  `.venv/bin/python quadratic_bezier_minimal_experiment.py all --profile smoke`
  (~20s) and `.venv/bin/python bezier_inverse_experiment.py all --profile smoke`
  (~1min).

### Outputs / gotchas
- Generated artifacts land in `outputs/` (resnet phase experiment) and in
  `results/<experiment>/smoke/` for smoke runs; both are git-ignored, as are
  `*.pt`/`*.npz` data and checkpoints. The curated files already committed under
  `results/` are lightweight and intentionally tracked — do not overwrite them.
- To regenerate curated (non-smoke) results you must pass `--force`; otherwise
  stages refuse to overwrite existing completed results (e.g.
  `quadratic_bezier_minimal_experiment.py all --profile minimal --force`).
- There is no lint config and no automated test suite in this repo; validation is
  done by running the `check`/smoke pipelines above.
