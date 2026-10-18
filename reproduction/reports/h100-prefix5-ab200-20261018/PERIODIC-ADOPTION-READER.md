# Progressive B periodic-result observation

The standalone small-metadata reader in `once_records_20261018/read_progressive_period_pair.py.txt` previously required prefix20 and an identical periodic-checkpoint/following-score SHA. That works within51–100, but would falsely reject the authorized100→101 and150→151 readout expansions. This was an observation limitation, not evidence of a training failure.

The reader now uses the actual completed stage20/40/80. At an expansion boundary it reads the existing immutable `prefix-migration-OLD-to-NEW-at-N.json`, binds its before receipt exactly to the completed periodic checkpoint, checks the preserved run/outer/Adam/EMA/LR clocks and recorded no-reset flags, and matches the actual following unlabelled EMA selection to the migration's after payload and the completed actor version. Within a stage it still requires the exact completed checkpoint SHA.

These are small producer receipts and actual score metadata. They do not independently establish tensor equality, moment preservation or numerical forward equivalence. Existing source-state CPU proofs and the implementation's expansion checks are separate evidence. Missing or contradictory evidence is reported by this reader only; it never stops or modifies training.

Four focused CPU fixture tests cover unchanged-stage adoption, both authorized expansion boundaries, unrelated/reset receipts, and incorrect following-score identities. No model, optimizer, hidden state or full rollout is loaded. The actual100→101 and150→151 observations remain pending until those records exist. Frozen A/B training sources and the live progress observer are unchanged.
