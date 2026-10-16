# Prefix20 joint4 independent CPU consumer — integration draft

This consumer is an independent operator. Its local CPU fixture tests do not accept current training, real checkpoints, transport or CUDA restoration. It never changes the healthy training runtime.

## Entry point

Use explicit values from the actual new run. There is no old-base, old-job,
old-source or old-run default:

```text
python /absolute/deployed/cpu_audit/audit_prefix20.py \
  --base /data/user/jxu957/experiments/ACTUAL-PREFIX20-BASE \
  --commit ACTUAL-40-HEX-PRODUCER-COMMIT \
  --run-id ACTUAL-32-HEX-RUN-ID \
  --output /absolute/owned/audit-prefix20-UNIQUE.json
```

The output is exclusive-create. A failed audit writes `REJECTED`, returns 1,
and never replaces an earlier receipt. The exact fresh 1..200 update ledger is
checked before Arrow/tokenizer imports, any Torch ZIP reads or large source and
checkpoint traversal. Missing 200 and any 201 both fail immediately.

The reader imports no Torch, Transformers, training modules, GPU modules or
remote clients. It has no subprocess, network, actor reload, generation,
training, scheduler, allocation or Slurm operation. Arrow, sandboxed Jinja2 and `tokenizers`
are only imported after the 200 gate. Torch ZIP files are read using a restricted
unpickler whose known globals resolve to local inert constructors. It cannot
execute experiment `__setstate__` methods. Numeric tensor allocation is avoided;
Adam step scalars and storage-byte hashes are read directly from ZIP members.
Unknown GLOBALs, unsupported archive layouts and unrecognized data fail closed.

## Existing producer evidence consumed

| Requirement | Existing evidence and consumer check |
|---|---|
| Actual 200, native4, no201 | exact updates1..200; every returned attempt; four scheduler/worker ledgers, counts, consumed LR and no suffix; bound config total_steps201 means stop after outer200 |
| Same scientific settings | explicit source commit/run ID; original immutable and runtime source SHA; resolved config wrapper/digest; epoch68 head SHA; fresh Adam/no teacher preparation |
| Top512 all | all5120 eligible/canonical question offsets and hashes; exact independent `1-prod(p)-prod(1-p)`; stable ordinal ties; exploration0 |
| Authentic prefix20 | all25600 saved prefix rows reconstructed from original per-state native chunk/token/position metadata; original sampler and effective stop/request/source provenance |
| Same2560 continuation | selected original ordinals/IDs; already completed prefix turns identical; paused native chunks identical; resumed raw native requests/tokens/LP lengths; actor prompt/model/environment masks |
| Own strict EM | CPU decode final authored model span using pinned tokenizer bytes; independent terminal-action parser and alias-normalized EM against pinned canonical training gold; compare each of2560 labels |
| Frozen actor during acquisition | before-prefix/after-prefix/after-prediction/after-continuation worker snapshots; actual Adam/scheduler/outer counters; actual actor RNG equality around probe prediction |
| Actor then probe | actual completed actor attempt, its snapshot, probe event own transaction and version; immutable actor-before-probe execution branch |
| Probe BCE/onepass/global256/local64/LR1e-6 | actual unique2560 own-trajectory transaction, four returned rank640/10-step/BCE receipts; final serialized actual Adam2000; immutable executed single-pass/BCE-mean partition code and resolved config |
| Saved states | five boundaries; complete-file full SHA; safe serialized actor Adam/scheduler counters and probe rank actual counters; four probe model/Adam storage-byte equality; actor/probe candidate state equality |
| Fixed5/50/100/150/200 | actual ordered dev512 identity ledger. Step5 also has an independent stage64 evaluation; it is not renamed as512 |
| Native51713 / seven sources | exact ordered canonical occurrence IDs and sources, retaining duplicate normalized questions; reject padding; finite row-aligned EM/F1; independently recompute seven source EM/F1, macro and weighted metrics |

The candidate stream is additionally checked for no repeated eligible index
within a real permutation cycle and full membership of completed cycles. Its
actual consumed source count must equal5120 per returned attempt, including
fully skipped actor attempts. Fully skipped attempts must leave a probe abort
receipt, not a completed probe update.

No new per-step trajectory-ID logging requirement is imposed on the submitted
healthy training job. One-pass and BCE execution acceptance combines retained
raw own2560 data, actual four-rank returned counts, serialized Adam steps and
immutable executed source/config bindings. It is not mislabeled as independent
gradient recomputation.

## Scope and remaining gates

`accepted=true` / `RECORDED_EVIDENCE_PASS` means this CPU conjunction passed.
It deliberately emits `goal_complete=false`:

- The existing producer does not retain the actor advantage tensors. Strict EM
  all-correct gating is recomputed, and actual enabled code/config with sample
  std correction1 and epsilon1e-6 is bound. The consumer cannot independently
  compare every applied normalized advantage element and does not claim that.
- Model/Adam/RNG ZIP metadata and storage bytes are parsed. The producer's saved
  finite-all-tensors receipt is consumed; this reader does not reload a model
  or allocate/recompute full model tensors.
- Fresh cross-host archive SHA acceptance is a separate controller gate. The
  CPU consumer does not initiate or implicitly certify transport.
- An actual final200 native4 restore-only producer, with two independent actual
  readbacks and zero updates/generation, is still required. No live job or JSON
  Boolean alone is a fresh physical GPU authentication.

The optional `--recovery-receipt` plus `--recovery-sha256` consumes a strictly
bound separate recovery result. Its currently proposed adapter schema is:

```json
{
  "kind": "prefix20_final200_native4_actual_restore",
  "source_base": "/explicit/actual/source/base",
  "source_run_id": "32 hex",
  "source_commit": "40 hex",
  "completed_outer": 200,
  "world_size": 4,
  "accepted": true,
  "restore_only": true,
  "optimizer_step_increments": 0,
  "scheduler_step_increments": 0,
  "probe_adam_step_increments": 0,
  "generation_calls": 0,
  "training_updates": 0,
  "dataequal": true,
  "selector_rng_equal": true,
  "driver_rng_equal": true,
  "eligible_mapping_equal": true,
  "candidate_pending_cursor_equal": true,
  "actual_rank_readbacks": [
    {
      "rank": 0,
      "actor_model_optimizer_scheduler_rng_equal": true,
      "probe_model_optimizer_private_rng_equal": true,
      "probe_actual_adam_steps": 2000,
      "readback_count": 2
    }
  ]
}
```

The rank array must actually contain0,1,2,3. This is an integration adapter,
not a fabricated existing production receipt. No supplied receipt yields
`PENDING_ACTUAL_NATIVE4_FINAL200_RECOVERY_RECEIPT`; no GPU recovery claim is made.
The final restore producer should additionally bind actual resource/process
birth, target file SHA and readback artifacts; those stay its responsibility.

## Validation performed locally

```text
python3 -m unittest discover -s /absolute/path/cpu_audit -v
```

16 synthetic adversarial tests pass. Tests cover missing200 early refusal and
immutable failure receipts, update201 refusal, duplicate keys/truncated JSONL,
Top512 ordinal ties, legacy448+64 rejection, changed eligible mapping,
canonical duplicate/order preservation, extra padding rejection, nested or
nonterminal answer rejection, malicious pickle GLOBAL rejection without
execution, and actual scalar Adam2000 extraction from a Torch-style ZIP.
A meta-path guard rejects any attempted Torch import during these tests.

No real current-run200 artifact exists in this subtask's accessible local
workspace. Consequently the full reader, remote ZIP variant compatibility,
large actual raw trajectories, actual final metrics and real recovery have
**not** been accepted or represented as tested. First retained raw evidence
should be used for a read-only schema smoke check before final200 integration;
unknown archive globals must be investigated explicitly, never broadly allowed.
