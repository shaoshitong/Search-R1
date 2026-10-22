# Completed pilot checkpoint acceptance — live ledger

CPU-only acceptance proceeds only after actual complete.json plus Slurm COMPLETED/0:0. No final4096 data, inference or Slurm mutation. Old accepted files are reused without repeating large reads when inode, bytes, mtime, ctime and saved SHA receipt are unchanged; node-local mount device numbers are not file-content identity. Actual ACD-Login-3 st_dev50 versus prior48 differed while every stable file field and latest SHA remained identical. Fix source9c5ac710f49a966bd8f7719b974647b26f85f9c5 was pushed/verified before new CPU passes.

| Variant | Job | Cursor | Checkpoint bytes | Fresh SHA256 |
|---|---:|---:|---:|---|
| layer15-prefix20 | 749398 | 192 | 3770208838 | `e7ad37b5e4058c97b0d8350289863de6adcaf49e4a16ebca173fb9d14a2f7da6` |
| layer21-prefix20 | 749078 | 192 | 3770208838 | `053442f84894b13af3377932b4c954401da5505f42f954dda650368796d066cc` |
| layer27-prefix20 | 749077 | 192 | 3770208838 | `f28778eff8d5ad0bab4f0d82db7506b779f21439ee2b1606408708e6e32e236c` |

Each accepted model has finite separate raw/EMA, strict load into the actual native block plus head, all46 Adam states at192, matching first/second moment shapes and exact actual optimizer state load, both source rank Python/Numpy/CPU torch RNG states loadable and serialized CUDA RNG present. CUDA runtime restore and next-update equivalence remain untested; full trajectory sampler resume is not claimed. The original two acceptance records remain unchanged.

L15/P20 was accepted at epoch1792682082.9425 in11.51s actual checkpoint work, peakRSS4.419GB. Its published depth-expansion source metadata SHA was independently checked. L18/P20 became complete/0:0 afterward and the next locked pass has been launched. L27/P80 is at192 but still completing evaluation; it has no complete.json and has not been opened. L21/P80 andL24/P20 are also still running.

L18/P20 job749399 subsequently passed independently at epoch1792682174.4431: fresh CP SHA b997d0e7e7a978ad473e090fe7bbca44ea2045ced55822aeb2a2b33d0dc74a95, cursor192. Four accepted variants are now L27/P20, L21/P20, L15/P20 andL18/P20. At epoch1792682229.2684, P80 remained RUNNING after192 without complete, L24/P20 was73 andL21/P80 was62. No unfinished CP was opened.
