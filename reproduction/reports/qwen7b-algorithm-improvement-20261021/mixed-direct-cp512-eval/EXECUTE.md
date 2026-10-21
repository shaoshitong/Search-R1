CP512 independent evaluation is actually READY; this subagent has not submitted GPU work. Root pin-complete5173B/SHA375be8ff was accepted without another pin operation. The pinned inode/UID/size/mtime and all receipt fields were checked; checkpoint payload was never read, rehashed or loaded on CPU. The former rolling source path may already be collected and is not required.

Original fast source-config14625B/SHA22098d5a and base are unchanged. Original25 runtime files match exact accepted origins/archive91d005991a95; the only additional runtime module is qualified evaluate_mixed_direct_checkpoint512.py from3573e88b. All26 remote SHA checks and Python3.10 compile passed. Event two-fixture CPU qualification passed before actual preparation.

Root single-submit command:

```sh
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_cp512_eval.py --action submit --submit-authorized
```

Normal/acd_u oneH100,8CPU,128GiB,onehour,no dependency. Immutable intent and same-name checks prevent duplicate submission; unknown submit results require actual scheduler reconciliation. GPU worker fresh-checks official files and the whole pinned CP512 before complete nine-state restoration, then freezes raw/EMA/twoAdam/RNG/buffers/sampler while scoring fixed512 and2048 only. No training-feature reader, actor/probe update, threshold tuning, or final4096 scoring. No evaluation result or performance gain is claimed by preparation.
