# Full MSE family selection and paired development analysis

Prepared only; no actual checkpoint is selected yet.

After both actual750189/750190 finish all10 epochs with successful accounting, this CPU entry applies the already-published seed1-only rule (maxTop512 count, then earliest epoch, then EMA), fixes exactly that epoch/state on seed2, checks identical qid/orderedtrajectory/strictEM, and computes matched development/TrainB diagnostics and2000 paired-question bootstrap intervals. Re-ranking preserves the original qid tie-break. No seed2 retuning, no model inference, no CP read, no final4096 access.

The selected epoch may differ from E10; its actual CPU checkpoint acceptance is still required before cross-family candidate freeze. This output is not that final freeze. Original historical E1 trainA remains separate.
