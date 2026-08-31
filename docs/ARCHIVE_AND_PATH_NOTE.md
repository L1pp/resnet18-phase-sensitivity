# Archive and path note

This repository is a selected public layer over a much larger local research archive.

## Public snapshot versus local archive

The public snapshot keeps source code, protocols, factual summaries, selected figures, and compact tables. It intentionally excludes weights, generated datasets, full formal outputs, machine inventories, network records, deployment scripts, transfer packages, logs, and temporary data.

Absence from this repository does not mean that an experiment never produced those materials. It means that the material was not necessary or appropriate for the public research record.

## Historical paths

The experiments were executed across several local and cloud environments. Historical code may therefore contain absolute paths or environment-specific defaults, including paths associated with different GPU machines.

Those paths were left unchanged because global rewriting would alter historical code without proving that the replacement paths were correct. A future user or AI should inspect each entry point and adapt it in a separate working copy.

Historical paths are not a declaration of the current repository layout.

## Interpretation order

When documents disagree, prefer evidence in this order:

1. final runtime metrics, summaries, histories, and manifests;
2. run metadata and protocol locks;
3. factual closeout or audit reports;
4. README files and planned configuration fields.

Planned fields can remain stale after an experiment changes or stops.

## Scientific status

An execution marked complete is not automatically a proof of every scientific hypothesis. Likewise, review of an artifact is not the same as independently rerunning the experiment.

The public summaries preserve controls, anomalies, missing items, and claim limits where these affect interpretation.
