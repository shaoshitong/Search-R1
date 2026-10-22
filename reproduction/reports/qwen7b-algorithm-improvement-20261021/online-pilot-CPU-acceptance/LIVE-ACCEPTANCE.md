# Completed pilot checkpoint acceptance — current ledger

Network was explicitly reauthorized. The original private SSH route first succeeded at epoch1792683088.5436; local-only interruption commit1393fa8597df771c991154f0932a3b93b33dc931 was then pushed and remote SHA verified. No alternate connection route, GPU allocation or Slurm mutation was used.

Latest fresh observation: 2026-10-22T23:33:06.891401+08:00, epoch1792683186.8914008. Six of seven cells are terminal-complete and independently CPU accepted. L21/P80 job749369 remains RUNNING at121 with no complete.json; its running checkpoint was not opened.

| Accepted variant | Job | Cursor | Checkpoint bytes | Fresh SHA256 |
|---|---:|---:|---:|---|
| layer15-prefix20 | 749398 | 192 | 3770208838 | `e7ad37b5e4058c97b0d8350289863de6adcaf49e4a16ebca173fb9d14a2f7da6` |
| layer18-prefix20 | 749399 | 192 | 3770208838 | `b997d0e7e7a978ad473e090fe7bbca44ea2045ced55822aeb2a2b33d0dc74a95` |
| layer21-prefix20 | 749078 | 192 | 3770208838 | `053442f84894b13af3377932b4c954401da5505f42f954dda650368796d066cc` |
| layer24-prefix20 | 749400 | 192 | 3770208838 | `e5d9a524da55160ed4b555d2adb7a0fdee26795b6e8f5b2a91db4699ab80c84c` |
| layer27-prefix20 | 749077 | 192 | 3770208838 | `f28778eff8d5ad0bab4f0d82db7506b779f21439ee2b1606408708e6e32e236c` |
| layer27-prefix80 | 749079 | 192 | 3770454598 | `07f884372aaff321a3b72478f45d12b852d4090db23d37ce5e6f1ea15f57911c` |

Acceptance requires actual complete.json plus Slurm COMPLETED/0:0 before opening the immutable epoch3 file. Every accepted checkpoint has finite separate raw/EMA, strict loading into the real native block plus head, all46 Adam parameter states at192 with finite compatible moments and actual optimizer-state load, and both source-rank Python/Numpy/CPU torch RNG states loadable with CUDA byte-states present. CUDA runtime restore or next-update equivalence remains untested; full trajectory-sampler resume is not claimed.

Existing accepted files were reused without repeating model reads when inode, bytes, mtime, ctime and the latest checkpoint SHA receipt remained unchanged. Shared-mount st_dev differs across login nodes (48/50), so it is ignored only for reuse; fresh new-file acceptance still reads the complete3.77GB and checks SHA.

The post-restoration validator PID1650713 completed normally with no failure. It accepted L27/P80 in10.76s and L24/P20 in21.91s actual checkpoint work, peakRSS4.419GB. Six accepted records are now remote and mirrored as small local JSONs. Source9c5ac710f49a966bd8f7719b974647b26f85f9c5 was already pushed/verified before use. No final4096 input, EM aggregate, probe score, actual candidate freeze or final inference occurred.
