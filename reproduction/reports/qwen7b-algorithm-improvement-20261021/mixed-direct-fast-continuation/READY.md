# Asynchronous Mixed-only continuation ready

Published worker/updater source: `24c689334b40da9e40c9052f3a40348ba572da0b`. Five small CPU fixtures passed in 18.716 seconds (6.854 seconds in tests), with no real model/checkpoint/tensor reads.

Entry: `python -m reproduction.operations.qwen7b_warmup.joint_mixed_direct_fast_continuation --base /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-fast-v1 --action train`. `--action audit-first` is the allocation-only CPU full-state audit child.

New config uses `mode=fast_continuation`, `architecture.micro_questions=32`, `prefetch_batches=8`, `parallel_workers=8`, and `reader_integrity_mode=inherited_full_seal_with_fresh_metadata_and_each_row_SHA`. `continuation` contains `source_config`, `source_main_config`, `source_checkpoint`, `source_checkpoint_original`, and `source_handoff_receipt`. Source config remains the exact original 9615 bytes/SHA/base. Pinned source checkpoint changes only its pathname. The handoff must prove original job746083 terminal0:0, lock-free, exact highest complete/accepted cursor and no unresolved tail.

All original scientific leaves remain equal: B32, real four G5 views per question at weight1/4, 50/50 actual Mixed selection, two Adam, EMA.99, LR5e-6, native/head architecture, fixed evaluation and global21132 budget. The original nine saved components are restored under the old binding, compared exactly, then saved under the explicit new runtime binding.

Reader origin `c2fdc9598ad55fea11bc5a5d2298ba4560807c24`. Every update calls `prepare(sampler)`; updater checks `accept_batch(batch)` before feature/model work. Restore/replay/evaluation disposes prefetch. Producer full payload seals are inherited explicitly, manifests/index SHA/size are fresh, every used retained row SHA is checked. This does not claim another fresh276GB scan or a background whole-bank scan.

Production qualification remains mandatory: first complete fast-update replay, independent allocation CPU full-state checkpoint reload, durable next-update replay, six actual complete updates. No job was submitted or signaled by this implementation task. Root/group own the single graceful handoff and submission.
