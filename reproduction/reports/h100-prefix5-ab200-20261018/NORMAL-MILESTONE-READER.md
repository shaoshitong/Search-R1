# Read-only normal milestone result reader

`once_records_20261018/read_normal_milestone_qa_grounding_actual.py.txt` generalizes the already-used100 QA/grounding metadata reader for actual150 and200 results. Supply `PARAMS={base,arm,outer}` inside an empty-CUDA private retained namespace. It only reads committed original output; missing files return pending, with no new GPU inference, calibration, trainer modification or intent.

At100/150 it binds the newly fitted grounding head to the corresponding complete actor and original300 calibration identities, and reports the next complete actor's actual four-rank reward-version use when available. At200 it verifies actual actor200 reward use against the retained150 fit, not an invented200 refit or outer201. Both arms retain the shared new BF16 fixed512 QA contract, canonical qids/ordinals/source identities, original four-worker exit0 and score-aggregate checks.

The actual100 source evidence remains the previously published immutable records. This new reader has been compiled locally; it has not executed against unproduced150/200 results, and does not make those stages complete. Large source checkpoint proofs are separate once-only CPU controls and are not repeated by this reader. Live A/B training runtimes are unchanged.
