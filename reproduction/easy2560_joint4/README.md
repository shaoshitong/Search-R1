# Easy2560 selection with Co rollout, HPC3 native4

This entry runs a fresh official Qwen2.5-3B-Instruct actor for 200 outer updates.
It performs no Prefix20 feature warmup, never loads the epoch68 selection head,
and disables virtual0. Each candidate pool contains 2560 eligible questions and
12800 full G5 trajectories. The selector chooses 512 groups, mixed first and
all-correct second in original candidate order, and reuses their real trajectories.

The rollout/context/action/retrieval protocol and grounding recipe are inherited
from the current Co implementation. Grounding round0 reuses its original fitted
official-model artifacts with explicit origin and a new run/version binding;
round50/100/150 are genuinely refitted during this run. No unused terminal200 refit.

## Entry

After committing, pushing and verifying the remote branch:

```bash
python -m reproduction.easy2560_joint4.deploy_source \
  --base /data/user/jxu957/experiments/easy2560-corollout-joint4-20261017 \
  --bridge-source /absolute/path/to/local_account_bridge.py \
  --control-work /absolute/path/to/control-work \
  --output /absolute/path/to/small-deployment-receipt.json
```

On the existing pinned HPC3 login, from the deployed source:

```bash
python -m reproduction.easy2560_joint4.control submit \
  --base /data/user/jxu957/experiments/easy2560-corollout-joint4-20261017 \
  --commit FULL_PUSHED_SHA --actor-micro 32
```

The once-only submission claim is durable before sbatch. An unknown observation
or timeout does not authorize resubmission; inspect the saved claim and actual
Slurm job. The allocation is jxu957, emergency_acd, formal-user, four GPUs,
48 CPUs and 512 GiB. No additional GPU smoke jobs are submitted.

Physical batch32/rank gives global micro128 and accumulation1. Logical loss
normalization stays micro4/rank. Only an actually failed owned OOM attempt may
be relaunched in a new independent run directory using physical16,8,4 with
accumulation2,4,8; never hot-change or restart a live process.

All outputs/cache/checkpoints resolve to HPC3. Existing Python, sklearn, FAISS,
data, official model and retrieval assets are reused read-only, with no download.
Personal storage remains capped at decimal2TB; report measured usage while the
job runs. Do not raise the cap or silently omit checkpoint milestones.

## Full-state archival

New Easy launches enable `easy2560_rolling_checkpoint=true` and save a full
native-four checkpoint after **every completed outer**, with model, rank-local
Adam/scheduler/RNG, driver RNG and exact candidate stream. After scheduled QA
and grounding adoption, small worker/driver boundary snapshots pin the actual
next-outer state. `preparation/rolling-checkpoints/latest.json` advances only
after the complete new files and SHA manifest are flushed. Only then may the
previous managed non-milestone checkpoint be retired. A partial/new failed save
leaves the preceding recovery point; checkpoints5/50/100/150/200 are protected.
New and previous saves temporarily coexist on HPC3; include this peak in the
2TB budget. No tensor payload is downloaded to the Mac.

Recovery accepts any completed outer before200, not only5, including a checkpoint
from an already resumed run. `python -m search_g1.easy2560_restore --base BASE
--checkpoint latest --output NEW_RECEIPT.json` selects the most recent complete
rolling checkpoint; pass its immutable receipt to the existing `control submit
--restore-receipt` entry in a new dedicated base. Restore all state and continue
at the saved outer+1; retain each real execution segment in the historical ledger.
An obsolete rolling checkpoint's tensors are not a permanent ancestor dependency.

This is a prospective recovery change. Job734689/source4cc has no dynamic
checkpoint hook and remains unchanged. Per the user's2026-10-19 instruction,
enable the repaired implementation if this job genuinely fails; do not restart
a healthy job merely to deploy it, and do not pretend unsaved outers have a
recoverable checkpoint. Once enabled, verify consecutive actual saves and a real
restore before claiming the recovery mechanism has passed runtime acceptance.

`checkpoint_archive snapshot` runs CPU-only on HPC3 after a milestone's complete
four-rank state and actual outer ledger row exist. It dynamically inventories all
saved HF shards, four rank-local Adam/scheduler/RNG files, driver candidate stream,
data, source and grounding artifacts/encoder dependencies. There is no Prefix20
paired-head, 33-file or eight-rank assumption.

`archive_controller` runs persistently on HKUST3, takes the HPC3 password only via
stdin JSON, and streams data directly HPC3 to HKUST3. Model payloads never pass
through the Mac. It observes a specific source job and keeps one controller claim.
Its per-file progress and actual partial file sizes permit measured transfer checks.
It saves a failure receipt on error; retry requires inspecting actual process birth
and partial/receipt state, not relaunching on an observation timeout.

After receiving every file, HKUST3 freshly rereads every SHA and fsyncs files and
directories. HPC3 then freshly rereads the original files before final acceptance.
The receipt explicitly does not claim an actual CUDA restore. No checkpoint is
automatically deleted by these tools. Any later source retirement must preserve
the accepted HKUST3 copy, active grounding encoder and required recovery sources.

```bash
# On HKUST3, with this source already published/deployed there:
python -B -m reproduction.easy2560_joint4.archive_controller \
  --base /data/user/jxu957/experiments/easy2560-corollout-joint4-20261017 \
  --commit FULL_TRAINING_SHA --job ACTUAL_JOB_ID
```

Provide the secret over an existing RAM-only stdin transport; never place it in
argv, source, configuration, logs or credential files.

The common lifecycle retains fixed512 checks at5/50/100/150/200 and final exact
native51713 evaluation with seven-source and macro/weighted EM/F1; no update201.
Submission/RUNNING/first log is not acceptance. Verify at least two actual measured
progress points, one completed outer, error/exit status and four-rank update counts.

## Independent CPU maintenance and final reports

`candidate_raw_maintenance.py` binds one explicit `--base`, `--run-id`,
`--source-job` and full40 `--training-commit`; its legacy defaults identify
job730080/source93eda66. A different published source requires matching source
publication, runtime-copy and run-manifest SHA maps. It never infers HEAD or
combines a failed job's history with a fresh retry. Its helper retains the latest two completed
outers, every failed/partial pool, selected token/probability traces, selection
JSON, RNG and checkpoints. Only the five raw tensors of older full-success
pools may be retired. Retired unselected candidates cannot be replayed in full.
The process has one durable launch claim; a timeout never authorizes a duplicate
or a training restart. Block-release receipts do not claim a fresh quota reading.

After a milestone's existing full archive has passed its actual source and
destination SHA checks, `checkpoint_evidence seal` creates a separate immutable
bundle of exact ledger prefixes, eligible membership and saved callback phase.
It leaves producer checkpoints and accepted receipts unchanged. CP5 was saved
before the extra fixed512 callback; CP50/100/150 were saved after fixed512 and
before grounding refresh; CP200 precedes final51713 and has no further refit.
No sidecar claims post-callback RNG capture or an actual CUDA restore.

`final_cpu_acceptance --base BASE --training-commit FULL40 --output NEW_REPORT.json` consumes the five
fixed512 points and exact ordered native51713 scores, preserving repeated
canonical questions, and reports per-source, macro and weighted EM/F1. It rejects
missing200, incomplete final ledgers and any update201. It also authenticates
existing complete checkpoint archive receipts, including through sealed small
metadata after an explicitly authorized source retirement. It performs no
training, answer generation, model reload or GPU allocation. Non93 sources require
the same explicit published-source binding; sealed evidence retains the small
publication records when source retirement was separately authorized. The original93
pipeline still uses its original verifier; this independent report does not
rewrite that pipeline's marker or claim whole-goal/recovery acceptance.

`retire_old_cosearch200.py` applies the prior authorization for failed experiment
checkpoints to exactly25 source tensors from failed job713379/CP200. Its pinned
plan preserves the full HK33 archive and all small recovery/receipt files. It
requires fresh full33 hashes on both hosts, exact source identities, both UID
reader observations and durable per-file unlink records; independent source and
archive checks close the operation. It leaves both old watchers running and
does not create the old archive worker's recursive-delete marker.
Non-target tokenizer assets larger than2MiB remain in place with strict file
identity comparison; their contents are not read for hashes. Only the exact
published d45 HK worker SHA is accepted as compatible archive evidence; source
evidence must match the current operator, with the HK hashing path checked by
AST against d45. `recover_cosearch200_source_audit.py` defaults to read-only
inspection of the one failed d45 source attempt. Its explicit `--apply` moves
only that attempt's six CPU audit files into a fixed failure directory after
confirming its original process birth exited and no deletion began. It preserves
the plan and all checkpoint files, never starts another worker, and refuses to
retry a partial audit move.
