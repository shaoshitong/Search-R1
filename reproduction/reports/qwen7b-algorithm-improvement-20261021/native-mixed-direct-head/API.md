# Official block27 + direct Mixed classifier (current source contract)

This supersedes the earlier E4-inherited design. Current production reads no trained probe checkpoint or old optimizer. Source is official Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`, native block27; both optimizers start at0, LR5e-6. EMA initially copies the complete raw block+head, then uses decay.99. Prior E4-only fixture metadata remains historical in Git and is not a production qualification.

Use the existing `model.Probe(model_dir,273,20)` official block27 loader (not the whole7B model). It checks the official28/3584/28/4/18944 configuration and reads only the mapped block27 tensors. Do not call a checkpoint restoration path. Then call:

```python
state = fresh_official_training_state(
    official_probe,
    official_inventory={"path": absolute_small_json, "sha256": expected_sha, "bytes": expected_bytes},
    expected_block_hash=block_state_hash(official_probe.block),
    official_revision=OFFICIAL_REVISION,
    head_seed=frozen_seed, learning_rate=5e-6,
)
model, ema = state["model"], state["ema"]
block_adam, head_adam = state["block_adam"], state["head_adam"]
```

`official_inventory` may be the already qualified official file inventory or official safetensors index metadata. It must be an absolute JSONdict<=16MiB; the factory freshly checks its SHA/optional bytes. The production deployment must bind its actual official inventory/model-file provenance, including the actual block tensors loaded by Probe. The factory checks in-memory block hash and real7B native geometry, and neither opens nor inherits a trained probe checkpoint. New-head construction seeds only the CPU default generator within a restored RNG scope; it does not touch CUDA RNG. The adapter registers `block.*`, `rotary.*`, `group_head.*`, never the loader's unused scalar `output.*`.

The returned dict contains model, EMA, native-block Adam0, new-head Adam0 and provenance. Both optimizers are fresh `torch.optim.Adam` with LR5e-6. Move the entire model/EMA and real batches to the allocated device; no source four-rank optimizer/RNG archive is inherited. Full current checkpoint/RNG/cursor management remains the production worker's job. The generic old mapping utilities and `from_restored_probe` remain historical helpers only, with no E4/14418 guard; current production must use the official factory.

`encode_trajectories(batch)` follows the original causal native forward and retains question273+Prefix20 geometry, mask and native positions. Its returned flat `hidden`, `attention_mask`, `position_ids` keep native-block gradients, while input features are detached. Encode the10 own trajectories once; gather the chosen5 for each of four real G5 views as `[views,5,293,3584]`, `[views,5,293]`, `[views,5,293]`; `group_logits(encoded)` returns one direct Mixed logit per view. No EM enters the model.

Question slots average only the chosen5 trajectories, with identical question masks/positions but numerical hidden differences permitted. All100 policy outputs remain separate. Type and within-question/within-policy slot embeddings are allowed; no trajectory-number embedding exists. Two bidirectional attention layers (dim256/8heads/dropout0) plus CLS produce one logit. G5 permutation invariance is mathematical; finite floating results are checked within tolerance. Padding is erased before projection and excluded as attention keys, including zero-prefix trajectories.

`mixed_only_bce(logits, own_em5, question_weights=None)` is the sole objective, target `0<sum(ownEM)<5`. Four view losses average per question before scaling by32-question batch. Sampler/worker must preserve authentic own IDs and 50/50 Mixed/nonMixed class balance. There is no trajectory BCE, scalar trajectory prediction or product formula.

Six tiny synthetic CPU tests passed in0.699s: two-layer256/8-head geometry; all100 prefix gradient paths; G5 permutation with different question hidden values; padding/zero-prefix/NaN erasure; one Mixed BCE and joint native gradients with detached input features; official-source block identity, fresh Adam0→1 for both modules, initial raw=EMA; incorrect revision/inventory/block hash rejected. No real7B tensor/checkpoint or GPU qualification was performed.
