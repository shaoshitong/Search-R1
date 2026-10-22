Future-only full169028 final scoring stage

Admission wrapper is exactly82c352abfbea8e6411b2a52727ef1d13555a264f. The mature architecture scorer closure is reused fromb680a97bc5965910402092f7957ccb4155f934cf, with independent fullMSE BASE. No training/scoring code is changed.

The template remains FUTURE_ONLY_NOT_FROZEN: epoch, primary state, seeds and actual checkpoint/proofs are unset. Root must later freeze the same selected epoch/state for both complete, CPU-accepted training seeds. The CPU acceptance must recursively bind actual accepted_input.fixture, accepted_input.physical_source_contract and accepted_input.P20_alignment. Both seeds must share the same physically accepted fixture and full-P20 alignment chain.

This preparation does not load CPs, read final journals/labels, start GPUs, submit Slurm or create a real candidate freeze.

Actual source-stage code: a4d4ec0e0e65d641522e6dec4d04f4d6035ffffe. Full remote base: /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-full169028-L15P80-MSE-20261023-v2.

Only after Root makes a real frozen descriptor and verifies both completed/accepted seeds, the existing entry takes explicit REMOTE_FREEZE_JSON and SHA256; this preparation did not call sbatch.

Publication: /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-full169028-L15P80-MSE-20261023-v2/source-publication.json; SHA256 3651b977567457f5f1d60f3bb126f5d01b701b3900836771a8506dfb3561900e

Entry: /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-full169028-L15P80-MSE-20261023-v2/scoring.sbatch; SHA256 49b62c2ed86adc7eb5fb591c1f97902c614c342dc562664023dd267875dbeeeb

Template: /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-full169028-L15P80-MSE-20261023-v2/candidate-freeze-TEMPLATE.json; SHA256 6d6bf1050464d8833dacc9b1f84b7181fc79d5a8d7a746319374ad74a42baa79

All13 source files were independently rehashed; wrapper SHA65d853fefd922b85ec723c9194b8e1d823703652925059a8d74e898e2349d791 remains exact82c.
