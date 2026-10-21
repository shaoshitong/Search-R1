# Effective-B128 official Mixed continuation

Controller actions are separate. Stage/prepare never signal or submit; Root alone uses the explicit retirement/submission flags. The active B32 job746220 and its immutable source/config are unchanged until Root requests the graceful handoff.

```sh
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_b128_continuation.py --action stage
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_b128_continuation.py --action prepare
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_b128_continuation.py --action retire --handoff-authorized
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_b128_continuation.py --action status
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_b128_continuation.py --action submit --submit-authorized
```

The new namespace is `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1`. Normal/acd_u, one H100, 8 CPU, 128GiB, 48h, no scheduler dependency. Existing submission intent or job identity prevents another submit; reconcile an unknown result against the same job name before any retry.

The controller verifies the current source746220 identity/module/base and training lock, then sends only SIGUSR1 to that verified own Python process inside its allocation. It waits for the exact main-job `COMPLETED 0:0`, lock release, and equality of latest/progress/update-intent/interrupted cursors with a fully published CPU-reloaded checkpoint. It never substitutes an older snapshot. Pin is an immutable hard link with same-inode/size/UID metadata binding; there is no CPU tensor load or full checkpoint rehash. The new GPU worker must fresh-verify and restore the original complete source state before its explicit B128 sampler/binding migration.

The original source config is retained byte-for-byte: 14625B, SHA256 `22098d5a4678db2f9c56d56cb5709829c3a4619c626e0b52d16ce726844402ac`. Runtime is the exact accepted25-file fast bundle plus the two published B128 modules. The small control-only handoff helper is carried in the published controller command and does not enter the27 runtime files.

The sole intended training change is effective batch128. Source question visits are `s*32` at the actual retired cursor `s`; each subsequent accepted update uses128 question visits. The original four-epoch visit budget is676224, with epoch visits169056 and explicit remaining epoch-end cursors computed from that source. LR5e-6, loss, EMA.99, ownG5/labels, exclusions, features, historical continuation provenance, and saved model/twoAdam/RNG remain bound to the source. No fixture, profile, new evaluation, or GPU task was added by this preparation.
