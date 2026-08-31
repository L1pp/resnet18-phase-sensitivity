# S1 Neural Affine clean minimal reproduction

This directory is a standalone clean-room implementation. It intentionally does not
import any historical `today_shortcycle`, `phase*`, or closeout training code.

The CPU and GPU entry points are in `scripts/`. Formal configuration and decision
thresholds live in `protocol.json`; generated state is written below a separate run
root and never into historical result directories.

