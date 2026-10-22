# All seven pilot checkpoint acceptances

Fresh acceptance ledger: 2026-10-22T23:57:34.558280+08:00, epoch 1792684654.5582802. All seven Slurm jobs were terminal COMPLETED/0:0 with actual epoch3/cursor192 complete receipts before their immutable checkpoints were opened. The final L21/P80 acceptance finished at epoch1792684650.323415; the CPU validator exited normally with no failure.

| Variant | Cursor | Checkpoint bytes | SHA256 |
|---|---:|---:|---|
| layer15-prefix20 | 192 | 3770208838 | `e7ad37b5e4058c97b0d8350289863de6adcaf49e4a16ebca173fb9d14a2f7da6` |
| layer18-prefix20 | 192 | 3770208838 | `b997d0e7e7a978ad473e090fe7bbca44ea2045ced55822aeb2a2b33d0dc74a95` |
| layer21-prefix20 | 192 | 3770208838 | `053442f84894b13af3377932b4c954401da5505f42f954dda650368796d066cc` |
| layer21-prefix80 | 192 | 3770454598 | `31b9527b571ca5b90ab6b6f83f2f25c541a314c4ab23a09e7300c0e79811305c` |
| layer24-prefix20 | 192 | 3770208838 | `e5d9a524da55160ed4b555d2adb7a0fdee26795b6e8f5b2a91db4699ab80c84c` |
| layer27-prefix20 | 192 | 3770208838 | `f28778eff8d5ad0bab4f0d82db7506b779f21439ee2b1606408708e6e32e236c` |
| layer27-prefix80 | 192 | 3770454598 | `07f884372aaff321a3b72478f45d12b852d4090db23d37ce5e6f1ea15f57911c` |

Every receipt validates finite separate raw/EMA tensors, strict native Qwen block plus head model loading, all46 Adam states with step192 and compatible finite moments, actual optimizer-state loading, and both ranks' Python/Numpy/CPU torch RNG restoration. CUDA byte states are present; GPU next-update restoration was not tested and is not claimed.

Existing accepted files were reused without repeating large checkpoint reads when inode/bytes/mtime/ctime and the latest SHA receipt were unchanged. Shared mount device numbers differ after login reboot or across hosts; st_dev alone is not a content change. New files always received a fresh complete SHA.

L21/P80 required10.73s actual checkpoint work and peaked at4.420GB RSS. The earlier network-permission interruption and login reboot/shared-mount absence are resolved historical observations; no missing mount was recreated. No running job/source was altered. No final4096 data or model scores were read.

Validator source9c5ac710f49a966bd8f7719b974647b26f85f9c5 was pushed and remotely verified before execution. Source and receipt history are retained on codex/q7b-completed-pilot-cpu-acceptance-20261022.
