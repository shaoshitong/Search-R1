# Recovery after the actual physical16 OOM in job731063

Job731063 failed at outer41 on 2026-10-18 19:11:44 CST while allocating
response logits in the actor update. Completed6..40 remain historical evidence;
there is no saved checkpoint40. The recovery boundary is the original job730226
checkpoint5, including its four native model/Adam/scheduler/RNG states and driver
candidate stream. Regenerate and recompute6 onward. Do not stitch job731063's
completed suffix or partial41 into the resumed update ledger.

The only execution change is physical actor micro16 -> 8 per rank, four GPUs,
global physical micro32, accumulation4, global mini128. Each physical8 forward
contains two logical4 masked means; their equal combination remains unchanged.
Virtual0 remains off. The optimizer and scheduler restore at100 writes and
outer5, then continue at6; the warmup remains57 outers. This does not claim
bitwise identical computation after changing physical grouping.

The recovery reader retains the original source's physical16 requirement. The
target may use16,8,4; upward changes are rejected. Among budget fields, only
`ppo_micro_batch_size` and `per_rank_actor_micro_batch` may differ, and their
geometry must match the target. Other scientific/data/model/retrieval/rollout
fields retain the existing exact checks. The original CP5-specific saved probe
version exception remains in the pipeline.

After committing and pushing the recovery fix, publish the exact new commit
using the existing deploy_source entry with an explicit `--commit` so unrelated
in-progress audit files are excluded. On the pinned HPC3 login, the commands are:

```bash
BASE=/data/user/jxu957/experiments/easy2560-corollout-joint4-micro8-after731063-20261018
ORIGINAL=/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018
COMMIT=FULL_PUSHED_RECOVERY_COMMIT
cd "$BASE/source/$COMMIT"
"$ORIGINAL/env/bin/python" -B -m reproduction.easy2560_joint4.control bootstrap \
  --base "$BASE" --commit "$COMMIT" --predecessor "$ORIGINAL"
"$ORIGINAL/env/bin/python" -B -m reproduction.easy2560_joint4.control submit \
  --base "$BASE" --commit "$COMMIT" --predecessor "$ORIGINAL" \
  --actor-micro 8 \
  --restore-receipt "$ORIGINAL/preparation/native4-recovery5/source-receipt.json"
```

Submission is once-only. Inspect the existing new-base claim/job on an unknown
outcome. The existing launcher requests jxu957/emergency_acd/formal-user,
4 GPUs,48 CPUs,512GiB,7 days. Do not change the source of another live job.

The first real acceptance must include native4 saved model/Adam/scheduler/RNG
and driver/data readback plus actual complete outer6 and two measured progress
points. Slurm RUNNING alone is insufficient. At milestones, observe existing
fixed512 QA, full native4 checkpoints and archives, and grounding refreshes;
the final requirements remain native51713, all five milestone results and
archives, actual final state recovery, and no optimizer update201.

CPU regression command (no GPU allocation):

```bash
python3 -B -m unittest -v \
  reproduction.easy2560_joint4.test_micro8_restore \
  reproduction.easy2560_joint4.test_restore_entry \
  search_g1.test_easy2560_restore \
  reproduction.common8.test_pipeline_easy_restore \
  search_g1.test_logical_micro_loss
```
