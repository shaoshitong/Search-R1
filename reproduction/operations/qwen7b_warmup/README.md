# Independent official Qwen2.5-7B probe warmup

## Current accepted scope, 2026-10-20

The newer direct human request superseded the original20-epoch budget: complete
**four epochs**, then compare EMA checkpoints1/2/3/4 on the same2048 previously
unseen questions x5 authentic trajectories. All four epochs are accepted.
Epoch2 wins this independent NQ-dev selection validation:181/512 true Mixed
(35.3516%), versus641/2048 (31.2988%) in the full candidate pool.

The immutable training config retains `fit_epochs=20` for checkpoint SHA
compatibility. **Do not resume epochs5..20 from that stale field or an old goal
replay.** The boundary controller stopped the original trainer after four full
epochs and preserved a separate four-update tail with its full recovery state.
Selection uses only the sealed checkpoints1..4. This stage never starts student RL.

Original data scope retained: full canonical169028 training questions x5,
independent512 diagnostic x5; no student RL. Official actor revision
`a09a35458c702b33eeacc393d103063234e8bc28` stays frozen throughout.

Layer numbers are **zero-based source indices**: capture native post-block
`model.layers[26]` (vLLM hidden+residual), initialize/train `model.layers[27]`
and a flattened question+first5 Linear. All first20 policy tokens/native hidden,
original RoPE positions and masks are retained remotely, but positions6..20 do
not enter the Prefix5 forward. Short answers are zero padded with false masks.

Eight independent two-GPU array tasks use one TP1 generation engine plus one
E5/FP16 Flat retriever each. They split canonical identities modulo8 and seal
bounded128-question G5 batches, preserving accepted batches on interruption.
No ten-to-five prefix selection and no actual-EM data selection occur here.
Continuation resumes those exact five original trajectories. The reference
first-complete-action/search interaction and original strict EM are retained.

Feature storage is lossless BF16-bit packing: one question base per question,
compressed per-trajectory XOR differences, and all actual policy states. Even
small native batching differences are preserved and each reconstructed row is
checked against its producer SHA. No assumption of bit-identical siblings.

After all847700 trajectories (including diagnostic) pass identity/file SHA
acceptance, a four-GPU fit completed four full balanced epochs. Each includes all
original training questions, oversampling minority whole-G5 classes. Adam5e-6,
weight decay0, EMA0.99, trajectory BCE + Mixed BCE1:1, no actor gradient/update.
Global51 groups and local micro2 groups; unequal rank counts are gradient
weighted to reproduce the global mean. Diagnostic never enters gradients.

Every epoch retains raw/EMA/Adam/rank RNG and an immutable checkpoint; every100
updates seals a replaceable inflight checkpoint with an exact plan cursor.
Each write checks all-rank model/EMA/Adam hashes and separately reloads on CPU.
Each epoch evaluates the natural full train and diagnostic banks and retains
per-question p_j/Mixed probability/own EM on remote storage. The original512
diagnostic bank measures learning curves; it no longer chooses the checkpoint.
The human-requested selection ranks predicted q on the same independently
excluded2048-question bank, selects512 questions, and only then reads their real
EM labels. Highest true Mixed count wins; exact ties prefer the earlier epoch.
No final QA data is used. The NQ-only validation is not a final seven-set QA score.

Remote base:
`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019`.
Own repo, cache, tmp, outputs and locks reside there; existing qualified Python,
libraries and canonical retrieval assets are reused read-only.

`runtime.sh -m reproduction.operations.qwen7b_warmup.qualify` tests real CPU
packing, nonidentical sibling features, Prefix5 future-token invariance,
whole-G5 balance, global/DDP gradient equivalence, and Adam/RNG next-step
recovery. Native GPU capture and four-rank checkpoint recovery remain runtime
acceptance requirements; the CPU command alone does not prove them. The completed
run also passed real native capture, four-rank GPU next-update recovery and
independent CPU/SHA audits of every complete epoch checkpoint.

Final delivery is `BASE/delivery-four-epoch-selection`: `complete.json`,
`REPORT.md`, `epoch-curves.png`, `epoch-curves.csv`,
`metrics-and-acceptance.json`, and `artifact-paths-and-publication.json`.
All p_j, q and same-trajectory EM remain under
`BASE/selection-2048-20261020/results`; the recommended full checkpoint is
`BASE/probe/epoch-0002.pt`, weight key `ema`. `BASE` is the remote base below.

`coordinator` reconciles its own submission intent and receipt, waits for
actual model fresh-SHA complete, prepares canonical partitions, and submits
ordinary acd_u/normal sampling. It submits fit only after all accepted bank
receipts, with no dependency on another experiment. Unknown submission
outcomes require squeue/sacct reconciliation; never blindly resubmit.

Data route: HKUST2 remote download -> HKUST3 independent SHA -> HPC3 fresh SHA.
No model/trajectory/features pass through Mac. Public small source metadata
and code are the only control-plane payloads. No Mac HF proxy is used.

Reference provenance: PORT.json and the supplied best-probe handoff. This is an
independent port, with explicit7B layer26/27 and Prefix5/capture20 differences;
3B trained weights and labels are not reused.
