# Final51713: original-output CPU acceptance

This separate end-of-run reader is for the current four-rank A/B actor experiments. The older probe100-epoch/Slurm-specific final-audit scripts do not qualify this run. It changes no generation, scoring rule, prompt, training logic or frozen runtime source.

Only after native outer200 and final QA have produced their complete markers, use:

```sh
python3 -m reproduction.operations.h100_reference_replica.final_qa_acceptance_control status --arm both
python3 -m reproduction.operations.h100_reference_replica.final_qa_acceptance_control start --arm A
python3 -m reproduction.operations.h100_reference_replica.final_qa_acceptance_control start --arm B
```

The once controller stages only a SHA-pinned reader into each arm's `preparation/final-qa-tools/`, with empty CUDA visibility and the arm's frozen repo first on PYTHONPATH. It detaches a CPU process and retains its exact PID/birth, intent, receipt and bounded log. A start does not retry a transport; inspect existing state after uncertainty. Status never starts a worker. Existing proof/receipt/intent prevents blind replay. A missing final marker reports pending without loading a tokenizer or staging a worker.

The reader verifies the actual final actor/run, frozen scoring/prompt source publication, shared new BF16 QA contract, and all four actual QA worker exit0 records. It binds canonical input SHA `55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1` and exact source counts:

| Source | Questions |
|---|---:|
| 2wikimultihopqa | 12576 |
| bamboogle | 125 |
| hotpotqa | 7405 |
| musique | 2417 |
| nq | 3610 |
| popqa | 14267 |
| triviaqa | 11313 |
| Total | 51713 |

Every ordinal/qid/source must match the canonical row and its original rank partition. Actual initial prompt token IDs are checked against the frozen prompt assembly using the final actor's tokenizer. Each original stored response/mask is rescored against its canonical gold aliases using the frozen strict EM/F1 scorer. Worker metric rows, aggregate EM/F1 and every source aggregate must agree. Each compressed original shard is independently hashed in the same single pass used to decode it, with file identity held stable. Full responses, questions, model state and prediction files stay remote; the proof is small metadata. No neural weights, retrieval or model forward is loaded/run by this reader.

Successful output is `preparation/final51713-original-output-cpu-once.json` with status `FINAL51713_ORIGINAL_OUTPUT_SCORE_READBACK_PASS`. An existing proof is reused with small-binding/file-stat checks rather than repeating completed large-file reads. It explicitly does **not** certify native leader exit0, all200 checkpoint states, retention, owned GPU/process cleanup, or the entire goal. Those independent final obligations remain mandatory. A Python pipeline-complete marker is not substituted for actual native exit/cleanup evidence.

Nine focused local CPU tests pass using tiny synthetic gzip/metric fixtures: out-of-order arrival, canonical identity/prompt errors, wrong EM versus gold, missing coverage, compressed SHA mismatch, per-source aggregate mismatch, actual frozen-code binding, no-op pending state, and preparation-only CPU launch. They validate the tool, not real final QA or model quality. The actual final51713 output audit is pending and must execute once on each completed arm.
