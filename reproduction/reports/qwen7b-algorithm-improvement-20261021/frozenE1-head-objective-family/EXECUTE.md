# Reuse the finite metadata resolver

Run from this repository after actual cache/CP0/epoch0 assets appear:

```sh
python3 reproduction/operations/qwen7b_warmup/deploy_head_family_metadata.py \
  --control-root /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5 \
  --commit 08b028bf41c45658f1f62f09c7545b9537dc911a --action resolve
```

The source commit is already published and verified; later report commits on
the same branch do not invalidate this pinned worker. The helper requires it
to be an ancestor of the published branch and checks exact local/Git worker
bytes. No new GPU, model forward, optimizer, threshold or score is created.

The actual family protocol is A/frozenE1-head-objective-family-v1/family-protocol.json,
7114B/SHA19f5e62fa39f4bde9790cadb29137a7e09cc91bd3fc1698a19f3a444326eed60.
The future packet, ONLY after all real checks pass, is in the same namespace:
coarse-fork-inputs.json. Its role/keys match group coarse_event_cached_train.
The current actual result is WAIT, with no packet written; never treat it as
accepted cache or CP0. `--action status` checks only protocol/packet existence.

Actual small CP0 raw/EMA/Adam/RNG restore must still run in each future fork
allocation before an update. The published verifier exports pure small-state
functions for that stage; never invoke its Count743978-specific CP1 wrapper
from another job. The raw E1 model/optimizer payload stays unopened.
