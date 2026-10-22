# Network permission interruption handoff

Root reported the application network permission was revoked and network grant returned null. From that notice onward this agent stopped all SSH, remote reads and pushes, made no alternative-route attempt, and did not mutate Slurm. Existing jobs are not classified as exited from this local access failure.

Last successful actual remote observation: **2026-10-22T23:19:51.112381+08:00**, epoch1792682391.1123812. These are now historical values, not current remote progress.

| Variant | Job | Last actual Slurm state | Cursor | complete | CPU accepted |
|---|---:|---|---:|---|---|
| layer27-prefix20 | 749077 | COMPLETED | 192 | True | True |
| layer21-prefix20 | 749078 | COMPLETED | 192 | True | True |
| layer27-prefix80 | 749079 | RUNNING | 192 | False | False |
| layer21-prefix80 | 749369 | RUNNING | 64 | False | False |
| layer15-prefix20 | 749398 | COMPLETED | 192 | True | True |
| layer18-prefix20 | 749399 | COMPLETED | 192 | True | True |
| layer24-prefix20 | 749400 | RUNNING | 128 | False | False |

The four accepted checkpoints have complete.json plus terminal COMPLETED/0:0, fresh entire-file SHA, finite distinct raw/EMA, strict native-block/head load, all46 Adam states at192 and both rank RNG byte states. They were not reloaded again after unchanged acceptance. CUDA runtime restoration/next-update equivalence remains untested.

| Accepted variant | Checkpoint SHA256 |
|---|---|
| layer15-prefix20 | `e7ad37b5e4058c97b0d8350289863de6adcaf49e4a16ebca173fb9d14a2f7da6` |
| layer18-prefix20 | `b997d0e7e7a978ad473e090fe7bbca44ea2045ced55822aeb2a2b33d0dc74a95` |
| layer21-prefix20 | `053442f84894b13af3377932b4c954401da5505f42f954dda650368796d066cc` |
| layer27-prefix20 | `f28778eff8d5ad0bab4f0d82db7506b779f21439ee2b1606408708e6e32e236c` |

Last backup pushed and remote SHA verified before the permission notice: **718127ece46e3393bd9097b49594c0e1dd415275** on codex/q7b-completed-pilot-cpu-acceptance-20261022. It includes L15/L18 actual acceptance and metadata observers. This interruption note is local-only until network authorization returns.

When authorization returns, first fresh-read current seven-cell complete/Slurm status, then use the published validator9c5ac710f49a966bd8f7719b974647b26f85f9c5 on newly terminal completed variants only. It reuses old accepted receipts; mount device st_dev is login-node-local while inode/size/times remain checked. Do not run a duplicate validator, infer task exit, reset or resubmit a training job, open running checkpoint files, or use final4096 scores. Final scorer v3 and seven-cell CPU entry are already staged; no final candidate was frozen.
