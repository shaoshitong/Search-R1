# Layer, pooling, and G10 experiment: static source discovery

Scope: local source and existing small configuration snapshots only, 2026-10-21.
No remote file existence/hash check, GPU job, actor update, submission, or active
training modification was performed for this report. A path below identifies a
source contract, not a fresh claim that every remote payload is present.

## Exact layer semantics

`operations/qwen7b_warmup/native_capture.py` registers a hook on
`runner.model.model.layers[26]` (zero-based). vLLM returns `(hidden,residual)`;
the code saves `hidden+residual`, the post-block26 residual stream, which is the
input to the next block27. `model.py` constructs
`Qwen2DecoderLayer(config,layer_idx=27)` and loads the official
`model.layers.27.*` weights. This is not HF `hidden_states[26]`: under standard
HF indexing the matching post-block26 output is `hidden_states[27]`.

For candidate trainable blocks with zero-based indices21/24/27, matching frozen
capture boundaries are post-block20/23/26. Record both the explicit zero-based
index and semantic boundary in every config/feature/checkpoint manifest; do not
silently switch to human one-based numbering.

The existing packed feature bank only saves layer26, hidden size3584,
question width273, policy prefix20. It cannot reconstruct layer20/23 from
layer26. `model.py` hard-codes block27 and a flattened linear readout of
`(QMAX+prefix)*3584`; the requested layer/pooling comparison needs an isolated
new model implementation and new checkpoints.

## Existing trajectory/token and identity contracts

Canonical base B:
`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019`.

Official actor in source/config snapshots:
`B/assets/qwen25-7b-instruct-a09a35458c70`, model
`Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`.
`shard.py` requires model `complete.json.status ==
MIGRATION_ALL_SELECTED_ASSETS_VERIFIED` and exact revision. A new recapture or
extra-rollout run must additionally revalidate the verified file inventory and
hashes; an `actor_version` string alone is insufficient.

Relevant immutable source chain:

1. `B/inputs/manifest.json`: `batches[]` entries include path/SHA/questions/split/
   shard; `ordered_train_qid_sha256`, source SHA, model revision.
2. `B/sampling/shard-XX/complete.json`: `run_id`, `model_revision`, `batches[]`.
3. Each batch `complete.json`: `input_sha256`, `prefix`, `full`,
   `actor_optimizer_updates=0`. Derive payload paths from these manifests.
4. `prefix` receipt entries `states`, `prefix`, `actor_hidden`, each path/bytes/
   SHA. Their filenames are `rank-0-states.jsonl.gz`,
   `rank-0-prefix.jsonl.gz`, `rank-0-packed/complete.json` under the batch
   `outers/outer-000001/prefix/` directory.
5. `full` receipt entries `full`, `own_em`, each path/bytes/SHA. Filenames are
   `rank-0-full.jsonl.gz` and `rank-0-own-em.jsonl` in the matching `full/`.
6. `B/bank/groups.json`: `train`169028 and `diagnostic`512, every group has
   `qid`, `question_hash`, `trajectory_ids`, own `labels`, source ordinal.
   `B/bank/features.json` lists packed feature complete manifests;
   `B/bank/complete.json` pins groups/features fingerprints and actor metadata.

Prefix rows already include enough causal inputs to re-encode their first20:
`trajectory_id`, `qid`, `actor_version`, `initial_prompt_token_ids`,
`prefix_segments[]` with `context_tokens`, `request_length`, `policy_start`,
`policy_length`, `position_start`, plus `policy_token_ids`,
`policy_position_ids`, `retained_policy_length`, `prefix_sha256`.
Each segment contains that turn's actual request plus only the retained policy
tokens. Prompt-only or concatenating policy tokens without observation contexts
would be a different representation and must not be substituted.

Full rows additionally retain `response_token_ids`, `response_info_mask`,
`behavior_turns`, `sampling_journal`, `response_log_probs`, `strict_em`,
`policy_snapshot_id`, prefix/scan/selection bindings, retrieval calls and terminal
reason. `behavior_turns` preserves actual request IDs/tokens, generated IDs,
kept-token counts, cuts and native chunks. Thus longer-prefix reconstruction is
possible in principle from existing complete rows, without fresh answers, after
verifying actual records and SHA. This is a source capability, not a new remote
availability claim.

`prefix20_online/features.py` demonstrates frozen causal context re-encoding;
`prefix_only_correctness_ab/features.py:retained_segments` reconstructs policy
tokens across actual tool turns. **Neither can run unchanged for7B**: hard-coded
3B revision/36layers/2048hidden/layer27/eightGPU and strict row-count contracts
must not be bypassed. A new7B recapture entrypoint should load the pinned actor,
`eval()+requires_grad_(False)`, use exact saved context/positions and capture
post-block20/23/26 in one pass. No token generation/retrieval is needed for this
recapture. HF/vLLM numerical paths can differ in BF16; label it frozen replay,
and qualify layer26 replay against the existing native bank with quantified
error rather than falsely require/promise bit identity across runtimes.

## Pooling and context are separate axes

Use a new head whose trainable block index and pooling are explicit. Three
readouts should operate on valid policy tokens only: last valid policy hidden,
masked policy mean, learned attention over valid policy hiddens. Short-EOS
padding contributes neither attention weight nor pooling mass.

For question+prefix versus prefix-only, define whether the question is visible
to the **trainable block**. Existing frozen prefix features already contain
question/past-observation context in both variants. It is impossible to remove
that historical context simply by dropping question slots from a probe. The
prefix-only probe can omit question keys/values in its trainable block; the
question+prefix probe retains original question tokens and masks QMAX padding.
Use original RoPE positions. Current `Probe.forward` asserts first slot valid,
so a compact prefix-only sequence needs a proper new implementation rather
than leaving question slots false under that assertion. A separate question-
inclusive readout would add another axis and should not be conflated with
question visibility in the trainable block.

## Five additional real trajectories per question

Existing sampling hard-codes G5 at `_states(range(5))`, shard receipts, merge,
selection manifests and packed writer lifecycle (question base released after
five). Merely changing a G constant does not produce an accepted G10 bank.

Safe addition plan:

- Freeze question identities and original G5 manifest/hashes first. Preserve the
  existing final G5 and its own labels. Extra sampling only covers declared
  fitting/development questions, never the frozen final question pool.
- Create an independent run ID and `sampling_attempt_id`, explicit response
  indices5..9 and immutable seed derivation. `initial_state`/`seed_for` already
  record run/trajectory/turn/chunk/seed and accept attempt identity, but a new
  G5-extra adapter is needed because the existing `_states` generates0..4.
- Use exactly the same official initial actor file identity, tokenizer,
  interaction protocol/retriever/budgets/strict-label function as old G5. No
  actor optimizer or policy update; labels are recomputed from each new complete
  answer, never copied from its siblings, predicted or synthetically assigned.
- New trajectory IDs and native sampling seeds must be disjoint; identical
  sampled text is allowed and must not be secretly resampled away.
- Store extra G5 independently. Join old5+new5 only after ten unique trajectory
  IDs and own-label bindings have passed source SHA/identity checks. Each G10
  bank records ordered IDs/labels0..9 and two generation event manifests.
- To reuse the existing packed writer without changing its five-completion
  lifecycle, keep original and extra G5 packed as separate feature sources,
  then build a G10 grouping manifest; never pretend it already supports10.

Canonical upstream split source in existing snapshots is
`/data/user/jxu957/experiments/topd-20261008/runtime/assets/common-instruct-0b002b/`
with `train/validation/stage/final` Parquet+JSONL and manifest SHA
`0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`.
`prepare_heldout2048.py` excludes all four by source ID, qid and NFKC/casefold/
whitespace-normalized question before sampling an external NQ-dev pool. The
current `selection-2048-20261020` is explicitly checkpoint-selection validation,
has been evaluated repeatedly, and is **not an untouched final test**. For a
new untouched final pool, freeze its ordered qids/normalized hashes before any
scores/EM are inspected; exclude canonical splits and all already-used
selection/development questions. Static code does not establish that enough
unused questions exist; count from actual source before promising2048.

## G10 -> G5 samplers with equal per-question total weight

For a G10 question with K correct answers, there are252 distinct five-subsets.
Uniform-random G5 has exact Mixed probability
`1-[C(K,5)+C(10-K,5)]/252`, with C(n,5)=0 for n<5.
Compute the label from each chosen subset's five own EMs, never from whether
the entire G10 has both classes.

One explicit **Mixed/nonMixed-balanced G5** rule is: partition all252 subsets
by their own Mixed label; if both partitions exist, choose positive/negative
with probability1/2, then uniformly within that partition. For K=0 or10 there
is no positive subset; keep the question and sample its valid negative subsets
instead of fabricating labels. This rule balances available subgroup labels
within a question, not global questions; report the resulting empirical prior.
It differs from choosing2positive+3negative or3positive+2negative trajectories,
which produces only Mixed subgroups and repeats the earlier class-bias problem.

If evaluating m random or balanced views of the same question, their weights
must sum to1: question loss is `sum_v w_v loss(G5_v)`, `sum_v w_v=1`, then mean
over unique questions. Each G5 trajectory BCE remains mean over five, plus one
group term; do not count all252 combinations as252 new questions or let
questions with more valid Mixed combinations dominate. A fixed deterministic
view RNG/manifest allows exact arm comparison and restore.

Training-only label-aware subgroup sampling is a supervised augmentation.
Final scoring must use its original fixed five **without label-aware subset
selection**, because deployment receives five ordinary trajectories and cannot
know EM before choosing questions. Evaluate all arms on identical frozen G5:
Mixed balanced accuracy/TPR/TNR/AUC, and real Mixed@512 from score-only2048
selection. A gain on balanced G10 views alone can reflect a changed class prior
or easier artificial subset distribution rather than better final selection.
