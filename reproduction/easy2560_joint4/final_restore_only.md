# Easy2560 final200 actual recovery

This is a separate post-training tool. It does not change the pinned production
allocation, continue training, generate answers, or execute update201. The source
is the actual outer200 checkpoint from the authenticated Easy2560 continuation.
Schema2 receipts recursively retain every full-state restore edge and its real
execution segment; the implementation is not limited to a single CP5 restart.

The existing CPU final consumer must first be able to authenticate all five
fixed512 boundaries, the ordered native51713 result, the complete authenticated
outers1–200 across all source segments, and the five complete remote checkpoint
archives at5/50/100/150/200. Historical archives and evaluation rows are resolved
to their actual ancestor bases rather than required in the newest base. For
current job750491, the inherited prefix is1–70 and its new suffix is71–200;
its new archive consumers cover100/150/200 while5/50 remain ancestor evidence. The publisher calls that consumer directly; a second preliminary invocation is not
needed. CPU publication alone is not a CUDA recovery pass.

Current job750491 must finish with4000 actual Adam writes on every one of its
four ranks, scheduler last_epoch200 and no write4001/outer201. Scheduler
_step_count201 includes its initialization call and is not an update201.

## Produce the source receipt

Use an explicitly pushed operator source, the existing remote Python runtime,
and a new small output path. Keep weights and optimizer states remote. Substitute
the actual final paths; these commands are not a claim that outer200 exists yet.

```bash
python -B -m search_g1.easy2560_final_checkpoint create \
  --base "$EASY_SOURCE_BASE" \
  --checkpoint "$EASY_SOURCE_BASE/checkpoints/search-g1-common8/actor/global_step_200" \
  --output "$EASY_SOURCE_RECEIPT"
```

The receipt records every actual checkpoint member, all four CPU Adam/scheduler
states, saved RNG/driver/eligible-pool identities, actual write/call/rollback
counts, the authenticated execution history, and the saved round150 grounding
dependencies. Its RNG boundary is the saved checkpoint before final evaluation;
it does not invent a post-evaluation RNG snapshot.

## Prepare and execute an independent target

Deploy the exact pushed operator Git archive into the target's `source/COMMIT`
directory with the existing source deployer. The target must be a new dedicated
`/data/user/jxu957/experiments/easy2560-corollout-joint4-*` directory. Preparation
reuses the source environment and data, binds the published target source, and
copies only the small saved grounding heads. It does not copy checkpoint tensors.

```bash
python -B -m reproduction.easy2560_joint4.final_restore_only prepare \
  --base "$EASY_RESTORE_TARGET" \
  --source-receipt "$EASY_SOURCE_RECEIPT" \
  --commit "$EASY_OPERATOR_COMMIT"

sbatch --partition=emergency_acd --qos=formal-user --account=jxu957 \
  --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=48 --mem=512G \
  --time=01:00:00 --export=NONE \
  --output="$EASY_RESTORE_TARGET/logs/restore-%j.log" \
  "$EASY_RESTORE_TARGET/repo/reproduction/easy2560_joint4/final_restore_only.sh" \
  "$EASY_RESTORE_TARGET"
```

The entry may instead run in the same authorized four-GPU allocation after its
production workers have released the devices. It still requires a separate
target and private Ray execution. Do not submit another recovery while the
first is live or its outcome is uncertain.

The entry actually loads the four saved states, compares the full FSDP model,
Adam, scheduler, worker RNG, driver RNG, candidate order/pending data and saved
counters, then reads the same live state twice without another reload. It
records process births, assigned CUDA UUIDs, source-ledger invariance, successful
task completion and owned resource release. Training, optimizer/scheduler steps,
generation and checkpoint saves are disabled in this role.

## Consume the completed GPU evidence

After successful actual execution, the small proof is
`$EASY_RESTORE_TARGET/logs/final-restore-only-complete.json`.

```bash
python -B -m reproduction.easy2560_joint4.final_cpu_acceptance \
  --base "$EASY_SOURCE_BASE" --training-commit "$EASY_TRAINING_COMMIT" \
  --final-restore-proof "$EASY_RESTORE_TARGET/logs/final-restore-only-complete.json" \
  --output "$EASY_FINAL_CPU_REPORT"
```

This final consumer only authenticates existing evidence. It does not allocate
GPUs, import Torch, reload a model or regenerate evaluation results. Without
`--final-restore-proof`, its previous CPU-only acceptance semantics are unchanged.
Even a combined accepted report does not mark the overall user goal complete;
the final requirement-by-requirement audit remains necessary.
