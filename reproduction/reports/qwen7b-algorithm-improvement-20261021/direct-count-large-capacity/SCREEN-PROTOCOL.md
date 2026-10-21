# Frozen large-head screen, registered before GPU execution

The staged training implementation/config0be60cc8 is unchanged. This separate
CPU screen adds class-specific reporting and a single predeclared internal
selection rule. It never fits a threshold, calibrator, new LR or loss weight.

The12 candidates are raw and EMA at each epoch0/1/2/4/8/20. Every node reports
Mixed BA/TPR/TNR/AUC, AllWrong and AllCorrect false-positive rates separately,
K0..5, and exact25% ranking coverage. Zero class denominators are NA. NQ2560,
HotpotQA2560 and pooled5120 are reported separately; pooled/Hotpot scores do not
enter the NQ gate. Fit-primary q/K statistics are supplementary and never gate.

Baseline is the same frozen raw E1 legacy trajectory head, using its genuine
cached logits Z on exactly the holdout's committed new-seed own5 indices5..9.
The score is `1-product(sigmoid(Z))-product(1-sigmoid(Z))`, evaluated in float64
from the saved float32 Z with stable sigmoid. Threshold is fixed0.5; ranking is
`(-score,qid)`. No extra model forward, new rollout or external data is needed.

Before GPU execution freeze the2560 NQ IDs by SHA256 of canonical
`["large-count-NQ-halves",9137,qid]`, ties by qid. The first/last1280 are the two
fixed halves. Only identity/domain metadata freezes them; neither labels nor
scores participates. Each candidate must satisfy all of:

1. NQ BA strictly exceeds the same-cache E1 baseline BA.
2. NQ Top640 contains strictly more true Mixed questions than E1 Top640.
3. Each fixed1280 half is noninferior to its own E1 baseline on both BA and
   Top320 true Mixed count.

Wait for all12 candidates to complete. Among eligible candidates choose exactly
one by NQ Top640 count descending, BA descending, earlier epoch, then EMA before
raw. Ties do not create extra candidates. If none qualifies, the result remains
a negative capacity diagnostic and no external candidate is packaged/scored.
These internal questions were seen by original warmup; this is internal recipe
selection, not final evidence of unseen-question generalization.

Even a passing internal screen only permits preparing one possible external
candidate. It does not authorize external scoring by itself: actually reload
that checkpoint and qualify same-ID/five-input numerical compatibility through
the online micro8 API first. Native cache micro8 replay is exact at the same
shape. Subsequent cached head encode uses320 trajectories per32-question batch;
it is a common diagnostic backend and is not claimed bitexact to online micro8.
The existing external/final pools stay unopened by this screen.

`direct_count_large_screen.py` is independent of the staged training runner.
It checks own new5 labels against frozen source metadata and computes metrics
from accepted q/K files. Those files do not repeat all five IDs; forward-input
binding is inherited explicitly from the qualified scoring source/config and
cache provenance, rather than misrepresented as a new per-row ID execution
receipt. No model weights are loaded by the screen. Full head checkpoint
reloads remain a separate actual allocation qualification.
