Future-only full169028 final scoring stage

Admission wrapper is exactly82c352abfbea8e6411b2a52727ef1d13555a264f. The mature architecture scorer closure is reused fromb680a97bc5965910402092f7957ccb4155f934cf, with independent fullMSE BASE. No training/scoring code is changed.

The template remains FUTURE_ONLY_NOT_FROZEN: epoch, primary state, seeds and actual checkpoint/proofs are unset. Root must later freeze the same selected epoch/state for both complete, CPU-accepted training seeds. The CPU acceptance must recursively bind actual accepted_input.fixture, accepted_input.physical_source_contract and accepted_input.P20_alignment. Both seeds must share the same physically accepted fixture and full-P20 alignment chain.

This preparation does not load CPs, read final journals/labels, start GPUs, submit Slurm or create a real candidate freeze.
