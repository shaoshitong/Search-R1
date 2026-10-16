# Recovery after the original reward-probe CPU dependency failure

Actual job727148 ran four ranks on ACD1-25, passed CUDA/JIT and the initial
model smoke, copied and fully verified78,952,654,893 retrieval bytes, and served
the real retrieval HTTP check. It failed before actor/question-probe updates:
the original G1 reward calibration could not import joblib. Its cleanup then
failed because the reused Python/kernel lacks the pidfd interface. Preserve the
original failure logs, source, claims and process ledger.

Restore the successful sigma3 private CPU runtime binding: joblib1.4.2,
scikit-learn1.5.2, scipy1.13.1 and threadpoolctl3.5.0 with base NumPy1.26.4.
The existing private vendor is reused read-only after full file SHA and its
actual CPU fit/serialization proof; it precedes the shared model runtime on
PYTHONPATH. No shared Torch environment is changed. Real target imports and
CPU fit/roundtrip must run before the next GPU submission.

Keep original G1 reward calibration300/G5/layers21,24,27/Xa/flip/CV2 and later
50/100/150 refits. This is separate from the new question probe, which still
loads epoch68 weights with fresh Adam and no teacher warmup/replay. No reward,
actor loss, precision, denominator, batch, LR or selection change is made.

Restore the old safe missing-pidfd compatibility route. Only a retained,
unreaped direct-child Popen handle may pin its verified session via wait lock
and waitid(WNOWAIT) before signaling. A recovered ledger without that handle
must refuse this route. The four actual Linux tests cover foreign-session
preservation, reaped/changed births, and recovered-ledger rejection. Mac skips
Linux process tests; Mac fixtures are not target cleanup acceptance.

The cluster rejected a zero-GPU CPU-only cleanup submission without allocating
a job. To keep the training allocation at four GPUs, the fresh recovery is
explicitly placed on ACD1-25. Its pure-CPU prepare step binds the actual new
four-card Slurm job, then an exact failed727148 cleanup worker runs before
run_pipeline or any CUDA/model work. It never signals processes. It verifies
recorded births and all old UID/session members are absent, copies original
run/ledger/cache/staging claims by SHA into a new cleanup shadow, cleans corpus
cache then staging with the original exact-inode cleanup functions, checks the
old tmpfs root/session absence, and verifies the original evidence stayed
byte-identical. Only then can the new pipeline proceed. Unknown or live readers
reject startup; it does not silently change nodes or repeat submission.

Personal budget stays2TB. An explicitly measured cumulative4GiB allowance
replaces the previous cumulative2GiB for all failed/fresh source/input staging
copies together. The original complete global observation keeps its original
timestamp; no fresh whole-root measurement or hard quota verification is claimed.
The once-only recovery records the four known roots' actual allocated du plus
the next copied inputs/source and metadata margin before creating the target.
