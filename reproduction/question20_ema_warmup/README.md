# Question + policy20 joint probe warmup — 2026-10-16

This isolated core implements the user's current **200 complete merged-data
epochs** probe experiment. It starts no job and does not train an actor. Source
identity verification, merged dataset construction, remote resource admission,
launch, progress receipts and publication belong to the runner.

The architecture and original-position layout follow
`prefix20_question_replay/model.py`; the complete-G5 joint objective follows
`prefix20_group_joint100/loss.py` in the reference checkout. This implementation
is self-contained and imports no reference experiment package.

## Scientific contract

- A frozen official Qwen2.5-3B-Instruct encoder (revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`) supplies **layer27 post-block**
  representations. Encode each authentic full causal context; retain all
  initial-question tokens and this same trajectory's first20 generated policy
  tokens, including their original positions/turn segments. No decode-retokenize,
  future tokens, invented continuation, or old EM for a new continuation.
- The trainable probe is the **native official full Qwen2DecoderLayer35** and
  native Qwen2RotaryEmbedding, followed by masked flatten and a fresh Linear to
  one correctness logit. Block weights are copied exactly from official source;
  all block parameters and the Linear are trainable FP32 masters. No old learned
  probe/Adam/RNG is imported at fresh initialization.
- Storage is `[Qmax question slots, 20 policy slots]`, hidden2048. `Qmax` comes
  from the complete actual dataset scan. Short questions are right padded;
  policy tokens stay at the tail. Real RoPE positions remain independent of
  storage slots. No question truncation is allowed.
- For same-event G5 logits `z[i,j]`, original strict EM `c[i,j]`, `p=sigmoid(z)`:
  `m=1[0<sum(c)<5]`, `h=1-prod(p)-prod(1-p)`. The loss is
  `mean_trajectory_BCEWithLogits + mean_group_BCE(h,m)`, with **both weights1**.
  K=1 is mixed. All32 Bernoulli outcome patterns are summed in log space, giving
  a mixed-event logit without clamping, probability floors, or an unstable
  `1 - near1` subtraction. Production loss arithmetic is FP32.
- Each complete event stays on one rank and within one head microbatch. The
  default global pattern is `51,51,51,51,52` G5 events, averaging256 trajectories
  per Adam. The final remainder is processed once. No replacement, class
  reweighting, drop-last, or duplicate padding samples are introduced.
- EMA starts as an exact copy of the fresh raw model. After **each returned,
  counter-verified successful Adam**, once:
  `EMA = 0.99 * EMA + 0.01 * raw`. It includes floating parameters/buffers;
  integer buffers are copied exactly. Native nonpersistent buffers are saved
  explicitly. A failed/nonfinite pre-step does not advance Adam or EMA; a
  post-Adam failure retains the real raw update and is not called a rollback.
- Train and fixed-test full diagnostics use **the same EMA Head version** in
  `eval`/`no_grad`. The raw model is never swapped. Raw online pre-own-update
  losses are reported separately and are not called fixed-head train metrics.

EMA does not establish absence of overfitting. The requested train/test mixed
balanced-accuracy curves provide the evidence.

## Reader/model API

Each `groups` entry is one immutable sampling event:

```python
{"event_group_id": "unique-source-event-and-question", "qid": "...",
 "trajectory_ids": ["t0", "t1", "t2", "t3", "t4"],
 "labels": [0, 1, 0, 0, 0]}
```

`event_id` is accepted as an alias. Grouping by `qid` alone is forbidden because
the same question sampled in another event has a different real G5 label.
Call `validate_groups(groups)` once for each full split before planning.

`reader.get_batch(flat_trajectory_ids, device=...)` preserves exact input order
and returns:

| Key | Shape/type |
| --- | --- |
| `features` | `[N,Qmax+20,2048]`, detached ordinary BF16 tensor |
| `attention_mask`, `question_mask`, `policy_mask` | `[N,Qmax+20]`, bool |
| `position_ids` | `[N,Qmax+20]`, int64, original valid-token positions |
| `question_lengths` | `[N]`, int64, actual initial question lengths |
| `policy_lengths` | optional `[N]`, int64, actual retained lengths1..20 |
| `policy_segments` | list per trajectory; each segment has integer `policy_start`, `policy_length`, `position_start`, `turn_index` |
| `trajectory_ids` | exact supplied list in exact order |
| `labels` | optional `[N]`, original strict EM; checked against group labels |

The frozen encoder must create ordinary detached output tensors outside
`inference_mode`; the trainable head may save these tensors for backward.

## Training API

```python
from reproduction.question20_ema_warmup.model import Question20CorrectnessProbe
from reproduction.question20_ema_warmup import ParameterEMA, TrainCounters
from reproduction.question20_ema_warmup.training import (
    build_epoch_plan, train_update, evaluate_groups, validate_groups)
from reproduction.question20_ema_warmup.schedule import cosine_learning_rate

# Seed fresh Python/NumPy/Torch/private sampler RNG in the bound runner first.
raw = Question20CorrectnessProbe(official_model_dir, scanned_Qmax,
                               attention_implementation="sdpa").to(device)
if world > 1:
    raw = torch.nn.parallel.DistributedDataParallel(
        raw, device_ids=[local_rank], broadcast_buffers=False)
# DDP's initial broadcast finishes before making the exact initial EMA copy.
optimizer = torch.optim.Adam(raw.parameters(), lr=3e-5, weight_decay=0)
ema, counters = ParameterEMA(raw), TrainCounters()
# Every rank must use the same official/fresh initialization and epoch plan.
plan = build_epoch_plan(len(train_groups), epoch=epoch, seed=seed, world=world)
total_updates = len(plan) * 200
for step in plan:
    rate = cosine_learning_rate(counters.optimizer_steps, total_updates)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = rate
    receipt = train_update(raw, optimizer, ema, counters, train_groups, step,
                           rank=rank, reader=train_reader,
                           micro_groups=micro_groups, device=device)
    # Persist small actual receipt; no new evaluation inside train_update.
# Only after exact-once epoch coverage and all ranks' real boundaries pass:
counters.complete_epochs += 1
train_metrics = evaluate_groups(ema.model, train_groups, rank=rank, world=world,
                               reader=train_reader, micro_groups=micro_groups,
                               device=device, weight_kind="ema0.99")
test_metrics = evaluate_groups(ema.model, test_groups, rank=rank, world=world,
                              reader=test_reader, micro_groups=micro_groups,
                              device=device, weight_kind="ema0.99")
```

EMA construction follows DDP's initial parameter broadcast, as shown above.
`train_update` scales every micro mean by
`world * actual_micro_groups / actual_global_groups`. Unequal rank counts,
unequal micro counts and the last partial batch therefore equal the direct
global objective. A rank with zero final groups calls the model's zero graph
to participate in the same DDP reduction and consumes **zero** data.

Each real merged-data epoch covers the entire event set, not one historical
source shard. A provisional445440-G5 union would produce8700 Adam per epoch
and1740000 over200 epochs; actual validated merged count determines both clocks.
The cosine schedule uses real successful Adam updates, from3e-5 to3e-6.

`GroupMetrics.result()` distinguishes `mixed_group.balanced_accuracy` (primary)
from `trajectory_correctness.balanced_accuracy`. Group prediction is `h>=0.5`;
the positive label is `0<K<5`. BA is `(TPR + TNR)/2`; if a class is absent,
BA is explicitly `None`, not fabricated as zero or one.

## Persistence and continuation

`checkpoint.py` captures raw model, EMA including nonpersistent native buffers,
complete Adam, rank Python/NumPy/Torch/CUDA/private-sampler RNG, actual counters,
epoch plan SHA and cursor. `save_checkpoint` writes an atomic fsync'ed file and
returns actual bytes/SHA. Checkpoints and tensors must stay on real remote
artifact paths. The caller records artifact/source/run/execution bindings and
never continues a prior experiment's trained head.

Restore model/Adam/EMA, construct DDP, then restore each rank's saved RNG last.
Checkpoint recovery is the same new experiment; it does not add another200
epochs. A partial source state is not a complete-state resume.

## Necessary CPU verification

`python -m reproduction.question20_ema_warmup.selfcheck` passed on2026-10-16
18:01:13Asia/Shanghai (Torch2.8CPU): joint loss/gradient equivalence; extreme
±1000 logits finite with no clamp; causal padding; exact-once complete-G5 plan;
exact EMA initialization/update; distinct group/trajectory diagnostics;
raw/EMA/Adam/RNG roundtrip; two-rank Gloo unequal3/2 micro reductions and final
1/0 group update equal direct global-mean Adam/EMA. No dataset reads, LLM loads,
GPU acceptance, or extra evaluation pipeline was performed.
