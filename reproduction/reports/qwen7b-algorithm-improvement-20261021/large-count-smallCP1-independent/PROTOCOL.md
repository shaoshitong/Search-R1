# Independent small new-head CP1 acceptance

The E1 block and legacy trajectory head are frozen in this experiment. CP1
therefore contains only new-head raw/EMA/Adam plus RNG, metadata, and the original
four rank digest/RNG archive. It is not a joint nine-component checkpoint.

This new auditor never opens the old 3.75GB E1 model/optimizer checkpoint, or
cache hidden/mask tensors. Before CPU materialization, require actual file size
at most64MiB, bounded ZIP storage ranges, and a meta-device load showing exactly
the seven tensors of the512518-parameter head and its matching Adam moments.
Then independently hash/load/restore raw, EMA, Adam, Python/NumPy/torch CPU RNG;
hash stored CUDA RNG bytes without initializing CUDA or claiming kernel restore.

Compare actual CP1 four hashes with the producer first-update exact replay;
verify Adam1, config/cache/data cursor, and original four-rank E1 digest/RNG
archive inherited unchanged from the allocation's independently reloaded small
CP0. The original E1 source descriptor is bound as metadata, not reloaded.
The CPU auditor performs zero optimizer updates and does not replay feature/model
forwards; the producer's separate actual first-update replay remains required.

All real loading occurs only in owned job743978 via an overlapping CPU2/gpu0
Slurm step, with CUDA_VISIBLE_DEVICES empty. Check job owner/name/command before
creating the step. First run one synthetic small-state fixture that cross-checks
producer hash conventions, CPU/meta reload and rejection of old-weight scope.
Actual CP1 absence must return waiting, never acceptance. No row metric auditor
or active training code is edited.
