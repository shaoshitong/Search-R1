# Through13: actual A/B observations

Both same h100-11 native runs completed13/200. The runtime remains
`2e3976230ac9425d66d8553afa9962821e964932`; this publication changes only
read-only reporting code and small results. It does not alter the experiment.

| Outer | A selected mixed | B selected mixed | A whole wall (s) | B whole wall (s) |
|---|---:|---:|---:|---:|
|1|396/512 = 77.3438%|402/512 = 78.5156%|589.547|612.038|
|2|392/512 = 76.5625%|392/512 = 76.5625%|974.194|976.123|
|3|383/512 = 74.8047%|392/512 = 76.5625%|784.153|814.824|
|4|394/512 = 76.9531%|409/512 = 79.8828%|748.519|738.386|
|5|408/512 = 79.6875%|380/512 = 74.2188%|899.092|898.213|
|6|404/512 = 78.9062%|361/512 = 70.5078%|720.921|718.562|
|7|405/512 = 79.1016%|332/512 = 64.8438%|787.663|785.689|
|8|368/512 = 71.8750%|336/512 = 65.6250%|806.226|806.789|
|9|366/512 = 71.4844%|291/512 = 56.8359%|779.165|785.590|
|10|512/512 = 100.0000%|512/512 = 100.0000%|1085.732|1074.770|
|11|347/512 = 67.7734%|307/512 = 59.9609%|776.805|795.214|
|12|353/512 = 68.9453%|296/512 = 57.8125%|766.868|802.281|
|13|330/512 = 64.4531%|299/512 = 58.3984%|787.143|774.692|

Outer1 reused authentic sampling after the authorized migration and is excluded
from fresh speed comparisons. Outer5 includes scheduled QA. Outer10 is a full
2560-question periodic: actual-EM mixed-first selection gave100% selected
mixed in both arms. That point is not evidence that either probe predicted
every group correctly, and is excluded from ordinary selection averages.

For ordinary outers6–13 excluding10 (seven observations), selected mixed
averages are A71.7913%, B61.9978%; B is currently lower. Whole-wall means are
A774.970s/B781.259s, including persistence and shared-node variation. These
are descriptive early observations, not final QA or a controlled speed claim.

The unchanged QA5 results remain A141/512 EM27.5391%, F1 36.1807%;
B146/512 EM28.5156%, F1 37.3150%. There is no QA10 or QA13. Scheduled
QA50/100/150/200 and final51713 are still pending. The first fresh periodic10
probe diagnostic is reported separately in `period10-paired-results.md`;
its classification metrics are not QA EM/F1.

## Probe records and interpretation

A has zero online updates. B has170 actual updates per rank and170 EMA
updates through13. All170 globally question-weighted loss rows were read as
small files: actual Adam and EMA counters are sequential1..170, LR5e-6,
and row counts match committed per-outer update budgets. These counts are
per rank, not a sum of the four ranks. Original rank-journal verification
remains the separate actual-progress evidence.

B ordinary13 balanced its own current512 groups by replacement. Its metrics
on the original unresampled selected groups differ from metrics on the balanced
training draws. On outer12, raw selected-train balanced accuracy51.9144%→56.5878%,
trajectory BCE .743853→.620066 and mixed BCE .761085→.626553. EMA selected-train
balanced accuracy remained50%; trajectory BCE .719792→.675167 improved while
mixed BCE1.150888→1.153391 worsened. Thus neither all losses nor all accuracies
improved. High ordinary accuracy in a selected population cannot replace
balanced accuracy, fresh diagnostic results or final QA.

Each ordinary round uses a changing actor, changing selected questions and
new trajectory labels. Its per-step loss sequence is not repeated evaluation
of one fixed dataset. Uncontinued candidate groups have no true labels, so
these records do not assert full5120-candidate classification accuracy.

The one-shot collector reads committed metadata, global probe loss rows,
whole-wall records and existing QA summaries. It creates no watcher and never
reads weights/hidden/full outputs or changes a trainer. Normal50 acceptance
and grounding50 actual adoption remain future requirements.

Source small audit: `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix5-ab-small-report-through013-20261017T233707927027Z.json`, SHA256 `2f4e6e17a7a7d0f28aff39276f19aefe57333ad447ed1ed54abc39f5c8129923`.
The full small record is `through13-small-metrics-and-loss-curves.json`.
