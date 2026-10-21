# Same-job runtime limit and exact continuation preparation

The single permitted attempt to extend job745501 from24h to48h was **denied** by Slurm (`Access/permission denied for job745501`, returncode1). The partition acd_u permits7days and normal QoS reports no MaxWall, but this account cannot extend its running job. No retry was made after the explicit denial. The current job, GPU/CPU/memory, QoS/priority, immutable source, bound config and original sbatch remain unchanged.

Two later real observations show continued updates143→166→180, no failure, under the same PID300719/birth116543024 on ACD1-28. At epoch1792588360.047, the job was RUNNING44:53 with24h limit (about23h15min remaining); Adam was14598. No new evaluation was performed.

The actual periodic full checkpoint **CP128** is3750958234B, SHA `66b17e0458f7d68d025f5895c8544aebca5030476565970f636903c9a88556ce`, at `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1/checkpoints/.rolling-accepted-cursor-000128-b62802132f4346bdb42f23621f2c3a81.pt`. Its producer receipt confirms all six CPU state hashes, sampler/cursor and historical four-rank RNG reload. This is the already-accepted producer proof, not an additional login tensor load. Accepted progress beyond128 is currently an unsaved tail; the prepared recovery refuses to discard it.

The published controller `reproduction.operations.qwen7b_warmup.resume_joint_balanced_native_main` is terminal-only and uses the same namespace, immutable runtime3b56e583, bound config5aaa63c9 and E4-derived current optimizer. Actual invocation of its `prepare` action while745501 was RUNNING returned `WAITING_FOR_ORIGINAL_WRITER_TERMINAL_AND_EXACT_HIGHEST_CHECKPOINT`, cursor180/latest complete128, `GPU_submitted=false`. It did not create a future fake checkpoint/ready receipt or load checkpoint tensors.

Before any future submit the controller requires:

- No original or recovery same-name active/pending job; exact original main terminal; free training lock.
- A graceful interrupted receipt matching the highest complete checkpoint, progress and accepted update intent exactly, with no later unresolved/unsaved transaction and no failure receipt.
- Published rolling transaction, fresh full-file checkpoint SHA, source/config hashes, sampler state/cursor and all six component receipt bindings. Real model/Adam/RNG/sampler restoration and next-update double replay still run inside the future owned allocation.
- Every prior epoch evaluation retained; a current pending epoch is bound to the exact cursor/checkpoint and is evaluated before the next optimizer update. No previous epoch may be filled using later weights.
- An immutable separate `recovery-attempt-001` submission identity/intent chain. The original submission files are retained. A submitted or uncertain attempt cannot be repeated blindly.

Future commands, **not executed now**:

```sh
python3 -m reproduction.operations.qwen7b_warmup.resume_joint_balanced_native_main --action prepare
python3 -m reproduction.operations.qwen7b_warmup.resume_joint_balanced_native_main --action submit
```

The future continuation entry requests normal/acd_u1H1008CPU128GiB48h with no dependency and the exact existing runtime/config. It remains blocked until the original writer safely exits. No current signal, cancellation, resubmission, new scoring or training change was performed. Actual future checkpoint restoration is not claimed to have happened yet.
