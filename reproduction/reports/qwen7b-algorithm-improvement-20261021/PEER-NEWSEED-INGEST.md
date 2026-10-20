# Received newseed raw G10 acceptance

`peer_newseed_ingest.py` is CPU-only. It never loads actor weights or hidden
features, submits Slurm, generates a rollout, changes the original B bank, or
updates a probe. Missing receiver `complete.json` returns NOT_READY before an
output directory is created. HF upload completion alone is insufficient.

The accepted source is the fixed 244-sealed-batch NAS10 snapshot
`newseed-20261021-185747-445fc691`. Future source batches require a separate
snapshot; they are not appended to this cohort.

## Real remote calls after source publication

Use the experiment's existing remote runtime (Python 3.10 with tokenizer and
pyarrow); the login system Python is not sufficient for the final tokenizer
strict-EM pass. Set `QWEN7B_CODE_ROOT` to the new immutable published code root.
All paths below are remote shared storage.

```sh
B=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019
S=newseed-20261021-185747-445fc691
R="$B/peer-newseed-h100-20261021/$S"
O="$B/algorithm-improvement-20261021/peer-newseed-ingest-$S"
bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -m reproduction.operations.qwen7b_warmup.peer_newseed_ingest --action prepare --receiver "$R" --output "$O"
bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -m reproduction.operations.qwen7b_warmup.peer_newseed_ingest --action ingest --config "$O/config.json"
```

`prepare` binds fresh source metadata SHA, the actual receiver catalog and
complete, original B config/bank/input manifest, canonical train parquet, and
official tokenizer acceptance. It projects three separate **identity-only**
exclusion manifests: diagnostic512, recipe-selection2048 and frozen final4096.
It reads only the existing identity-only final plan, never its source_refs,
labels or rollouts. The currently fitted6528 cohort is **not** an exclusion.

`ingest` inherits the receiver's completed fresh archive/member SHA acceptance;
it does not perform another all-archive or states-payload hash pass. It fresh
hashes only the actual input/seal/prefix/full/own-EM files it consumes, checks
pre/post file identity, rederives canonical train qid/hash/source ordinal from
the SHA-bound original parquet, and then rechecks all accepted old5/new5 rows:

- ten genuine globally unique own IDs and each own integer strict EM;
- original/new identical frozen actor, exact question token prompt/gold/source,
  sampling contract, action boundaries, interaction protocol and dtype;
- every native chunk's actual request, raw output, logprobs and derived seed;
- full token/mask/context reconstruction and own Prefix20 causal positions;
- tokenizer-based strict EM using the unchanged production `strict_label`;
- no same-question seed collision within/between actual original/new events.

Native capture appends `feature_*` locators *after* sealing the causal prefix;
the prefix SHA therefore uses the same body as the already qualified replay
normalizer, excluding those locators. It never opens native hidden payloads.

The final `complete.json` points to raw groups, source events and overlap/class
statistics. Old5/new5 remain separate genuine events; no virtual G5 is chosen
here. The output is **not** a trained model or feature bank, and these question
identities are not a warmup holdout: initial warmup saw all169028 train items.

## Subsequent HF replay interface

`raw-g10/groups.json` has `train` rows with qid/hash, ten own IDs and labels,
and two `real_G5_events` preserving original then newseed order.
`raw-g10/source-events.json` records each event's input/seal/prefix/full/own-EM
SHA refs. A later label-blind capture preparation can project only qid, own IDs
and Prefix20 refs from these files, normalize via `replay_probe_layers`, and
use `ReplayFeatureReader` for **both** old and new training plus evaluations.
The existing matched9088 `replay_bank_capture` has an exact fixed9088 gate;
this raw receipt deliberately does not weaken or monkeypatch that gate.
No native/HF bit-exact equivalence or GPU replay completion is claimed.
Actual frozen final identity rows also contain canonical_question_sha256 and
pool; these metadata fields are whitelist-validated then projected to qid/hash.
Labels and source_proofs are never followed.

Seven narrow CPU tests cover target complete/SHA gates, identity-only exclusions,
own G10 IDs/EM, genuine tokenizer strict-EM with tool observations, Prefix20
multisegment positions, actual seed overlap, and sampling protocol mismatch.

## Authorized8192-question feature-only follow-up

`peer_newseed_replay.py` now implements the separate HF replay adapter. It
selects exactly8192 accepted raw training questions using identity hash seed1729,
excluding controlled6528 and all three evaluation scopes by qid **and** hash.
Insufficient eligible questions fail closed with the real available count.
Labels/probe scores do not enter selection or the projected prefix inputs.
Each selected question keeps all10 genuine old5/new5 trajectories.

`deploy_peer_newseed_replay.py` provides explicit `stage`, `ingest-prepare`,
`ingest-start`, `features-start`, `finalize`, per-shard `submit` and `status`.
It stages28 exact published files in4KiB SHA chunks; every rendered control
command is below10KB. Unknown launch/submission receipts refuse duplication.
Finalization creates four independent normal/acd_u scripts, each1GPU/8CPU/64G,
2h and **no dependency**. No retriever/second GPU is requested.

Feature preparation checks actual shared free space at least600GiB. The
uncompressed dense retained-feature upper bound for8192×10×three layers is
516GB; lossless per-question/XOR packing usually reduces it. All features stay
on shared storage. Workers use the exact Q1-qualified R2 replay primitive and
official model view;256-trajectory chunks are immutable and complete chunks
can be verified/reused. They preserve per-trajectory causal tool contexts and
positions, import the real ReplayFeatureReader, and check actual row coverage.

The reused R2 evaluation features cover only R2's **original G5** diagnostic512
and already-viewed selection2048. They do not cover the new validation G10
random fixed5; those distinct tokens require a separate same-backend replay.
Untouched final pools are neither generated nor opened by this feature adapter.
