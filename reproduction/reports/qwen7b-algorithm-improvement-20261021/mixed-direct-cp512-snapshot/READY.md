# CP512 direct-Mixed evaluation ready

Worker origin `3573e88b8d320baa3f5fb1f442787777f42591a5`; two CPU fixtures passed (19.650s total, 6.774s tests), no real model/checkpoint/tensor reads. Keep the original fast25-file runtime unchanged and append only `evaluate_mixed_direct_checkpoint512.py` (10198B, SHA095a762dbea04b1e5507aa36b0567a304903c4d53135e10ffd71076f96b30e86). The prior CP128 snapshot module is not an import dependency.

Entry: `python -m reproduction.operations.qwen7b_warmup.evaluate_mixed_direct_checkpoint512 --base /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-official-cp512-eval-v1`.

`evaluation-config.json` fields: mode=read_only_checkpoint_snapshot, source_config (snapshot/source-config.json, exact original14625bytes), source_main_config (fast namespace/config.json), checkpoint (pin), source_checkpoint_original (only pathname differs), and code_hashes. Source config SHA22098d5a4678db2f9c56d56cb5709829c3a4619c626e0b52d16ce726844402ac. It retains its original fast base and continuation provenance.

Production requires fresh full pinned checkpoint SHA, all official file SHA, source `fast.typed_config` and `fast.continuation_binding`, actual nine-component restore, both Adam clocks512 and sampler512. It constructs only fixed512/2048 readers, never the276GB training reader. Raw and EMA predictions preserve ownG5 IDs/EM and use only direct Mixed logits. All nine saved components must remain unchanged afterward. No final4096, threshold tuning or trajectory accuracy.

This implementation task did not submit a job or mutate the source training. Root controls the single evaluation submission; actual GPU completion and independent ownEM/metrics audit remain required.
