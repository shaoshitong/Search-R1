# Actual startup metadata repair, before any model/cache/update

Job743933 failed after12 seconds at startup with `KeyError: bytes` because the
original model_config descriptor intentionally contains only path and SHA256.
The generic full-file descriptor checker required a byte count. The error
occurred before CUDA/model/head/cache/optimizer initialization: cache0, updates0.
The full v1 source/config/failure/log/job receipts remain unchanged.

A separate v2 target accepts the exact SHA-only schema, verifies actual file SHA
and optional declared byte count, and normalizes the small model-config proof.
CPU prepare now executes all small startup descriptor guards using actual refs;
only the3.75GB model checkpoint full-SHA check stays inside the GPU allocation.
Four narrow metadata tests include correct SHA-only acceptance, wrong SHA/size
rejection, real geometry/cohort own-G5 binding and unchanged20-epoch batch order.

The scientific source/data/E1/head initialization/938 order/20epochs/holdout IDs/
NQ-half rule/screen/FP32-BF16 split remain identical. This is a fresh diagnostic
attempt after0 actual updates; it is not a checkpoint resume or lost-update
recovery. New target: direct-count-frozenE1-fit30016-hold5120-v2.
