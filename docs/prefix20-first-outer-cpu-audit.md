# First completed outer — independent read-only CPU audit

This is a separate interim operator. It never calls or relaxes the final consumer's hard `COMPLETE200` gate, and always emits `final_acceptance=false` and `goal_complete=false`. It allocates no GPU, imports no Torch, reloads no model, performs no training, sends no SSH/Slurm request and writes no checkpoint or tensor. Only one small JSON result is written, exclusively, under the actual run's existing `logs` or `preparation` directory.

The main operator must publish this runner and verify its source SHA first. Invoke it on HPC using the actual source base, full producer commit and run ID:

```sh
/ACTUAL/BASE/env/bin/python -B /PUBLISHED/first_outer_audit.py \
  --base /ACTUAL/BASE \
  --commit ACTUAL_FULL_40_HEX_PRODUCER_COMMIT \
  --run-id ACTUAL_32_HEX_RUN_ID \
  --output /ACTUAL/BASE/logs/first-outer-cpu-audit-UNIQUE.json
```

There is no old-run, job, base or commit default. The runner dynamically loads only `audit_prefix20.py` and `safe_pt.py` from `base/source/<commit>/reproduction/prefix20_joint4/cpu_audit`. Their exact file SHA must match the current run's source manifest before import. Existing final audit files are not changed.

## Readiness and concurrent appends

Exit codes and result status:

- **0 / `FIRST_OUTER_RECORDED_EVIDENCE_PASS`**: actual first completed outer and any preceding zero-write aborted attempts passed this interim evidence conjunction.
- **2 / `WAITING_FIRST_ACTUAL_COMPLETE`**: the complete update1 row, causal rank rows or atomic raw artifacts are not yet visible. This is readiness, not a training failure.
- **1 / `REJECTED`**: visible evidence violates its expected identity, schema, counts or raw-data checks. This does not cancel or change training; the main operator inspects the reason.

Each live ledger is read once up to its initial opened size, with inode equality and no truncation required. Later appends are allowed. Only complete newline-terminated rows are decoded; an incomplete trailing row is explicitly counted and excluded. The receipt records the exact complete prefix byte count and SHA, rather than pretending to hash the eventual whole file. A causally incomplete, otherwise correct scheduler/Adam prefix yields WAITING. Existing output is never replaced; choose a new result filename for every observation.

The fully audited window ends at update1's actual completed attempt ID. It includes any earlier zero-write aborts at logical outer1. Later in-flight attempts are excluded from the first-window conjunction. Their observed complete rows are checked for run/rank/job identity and update201, but are not represented as fully audited. Any observed update201 is rejected.

## Evidence consumed

- Real four-rank actor attempt results and cumulative retained Adam write counts; at least one actual actor write completes the outer. A partial safe-progress outer is reported explicitly, without assuming 20 writes.
- Four scheduler sequences: BEGIN/ABORT for preceding zero-write attempts, then BEGIN/COMMIT_PREPARED/COMPLETE for the completed attempt; exact outer1, physical write counts, scheduler epoch0→1 / count1→2 and actual consumed LR0 from the original warmup57 schedule.
- Four physical optimizer/opportunity journals: actual sequential retained writes, minibatch indices, learning rates, opportunity status and before/after counters. Rollback and skipped opportunities do not become retained writes.
- Existing `source_and_config`, `canonical_inputs` and `audit_outer` helpers verify unchanged native4 configuration, official fresh actor/epoch68 head binding, Top512 from5120 predictions, all25600 authentic prefix20 trajectories, the same2560 continuations, canonical strict EM, four frozen actor snapshots, prequential metrics and actual four-rank probe10/global256/local64/LR1e-6/no-replay receipts.
- Candidate source consumption includes every aborted attempt; `.pt` files remain remote and are only read by the restricted CPU ZIP parser.

This is CPU recorded-evidence acceptance. It does not independently authenticate GPU probabilities or recompute gradients. Full200, saved probe Adam2000, five checkpoints, fixed5/50/100/150/200, final native51713, archive SHA and actual native4 recovery remain separate final acceptance requirements.

## Local validation

```sh
python3 -B -m unittest reproduction.prefix20_joint4.cpu_audit.test_first_outer_audit -v
```

17 small adversarial fixtures pass, including abort→partial3-write success, later in-flight rows, partial JSONL tails, append during read, causal WAITING, wrong rank counters/LR/outer labels, update201, stale job/helper identity, immutable small receipts, and reuse of the actual bound helpers while their final missing200 gate still rejects outer1. A meta-path guard forbids Torch imports. An independent read-only reviewer also reran the preceding 16-test set and found no remaining deterministic fatal issue.

The complete raw artifact path has not yet been exercised against actual outer1 in this subtask. The main operator performs that check after the actual completed update appears; no remote operation was performed while preparing this operator.
