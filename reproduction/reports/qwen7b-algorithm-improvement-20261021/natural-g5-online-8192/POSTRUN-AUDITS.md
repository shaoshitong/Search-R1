# Natural-G5 post-run acceptance and matched comparison

Root submitted the two original jobs once:749459=L21/P20 and749460=L27/P20, normal/acd_u, two GPUs,30min. At the last permitted fresh read both were PENDING; current state has not been queried after network access was suspended. No SSH, push or Slurm operation is part of this local deliverable.

## Actual checkpoint acceptance

`audit_natural_g5_online_checkpoint.py` is standalone CPU code intended to run on HPC3 after the exact original allocation finishes `COMPLETED 0:0`. It hashes and deserializes each requested checkpoint on remote storage, verifies epoch/cursor/own-plan binding, raw and EMA keys/shapes/finite values/separate contents, all native+head Adam moments and actual clocks, two rank RNG states, epoch question coverage and fixed training/validation evaluation presence. CPU Python/NumPy/Torch RNG states are restored to isolated generators. CUDA RNG byte states are inspected, but GPU restore or next-update equality is explicitly not claimed. EMA recursion is bound to the published unchanged runtime source, not reconstructed from a single final checkpoint.

The final expected cursor and all Adam clocks are192. Per-epoch cursors are64/128/192. Checkpoint payloads must remain remote. Full deserialization can use several GiB CPU RAM; the script does not request resources, GPUs or another Slurm job.

## Fair paired effect comparison

`compare_natural_g5_online_matched.py` uses only already viewed independent validation2048 and fixed training2048 outputs. It matches qid, own five trajectory IDs and true EM exactly across raw/EMA, epochs and both sampling policies before comparison. Stable BCE is recomputed from logits, along with Mixed balanced accuracy/recall, nonMixed false-positive rate, all-correct/all-wrong/Mixed errors and exact Top512 Mixed. The recomputed values must match the producer. It reports paired right/wrong changes and newly selected/dropped Mixed counts; baseline logit differences are shown separately. No new model inference, threshold fitting or final4096 access occurs. Missing requested files produce a partial/wait result; `--require-complete` requires all requested matched outputs. Stable generalization gain is not inferred merely from completion or lower training loss.

## Deploy separately after access returns

The local `stage_natural_g5_postrun_audits.py` generator reads scripts from a local persisted Git commit and emits a small remote staging script. After the commit is pushed and its remote SHA verified, root can send that script through the existing private control. It writes only a new `postrun-audit-code-<commit>` directory under the new natural-G5 suite, leaving active training source/entries/locks untouched. It does not execute audits or submit jobs.

Example after publication:

```sh
python3 -m reproduction.operations.qwen7b_warmup.stage_natural_g5_postrun_audits --commit <published-commit> --output <small-local-stage.py>
```

On HPC3, use the existing qualified runtime Python to run each staged file directly:

```sh
python <postrun-audit-code>/audit_natural_g5_online_checkpoint.py --layer 21 --epoch 3
python <postrun-audit-code>/audit_natural_g5_online_checkpoint.py --layer 27 --epoch 3
python <postrun-audit-code>/compare_natural_g5_online_matched.py --layers 21 27 --epochs 0 1 2 3 --require-complete
```

Local checks completed:all three scripts compile; in-memory2048-row paired metric cases match exact counts; BCE remains finite at logits±2000. No production checkpoint or prediction was loaded locally. Actual remote acceptance and result comparison remain pending.
