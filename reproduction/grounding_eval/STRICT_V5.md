# Conservative typed strict-v5

This independent executable protocol corrects known v4 intervention/classification defects. It does not overwrite v4, and it does not claim exact alignment with an author's evaluator. A changed answer alone is not grounding success. New interventions require **new generation**; old v4 answers cannot be relabeled into a v5 result.

## Frozen behavior

- Keep the original fixed NQ 3,610 occurrences, original normal trajectories and strict original EM. No normal inference, teacher, training or retrieval changes.
- Parse raw gold aliases before any QA EM normalization. Preserve decimal points, currency, signs, units, dates and qualifiers. Unsupported/conflicting alias types, distinct entities, dates/events or precision are `INVALID_INTERVENTION` with a reason; no generic institution fallback.
- A finite explicit type inventory supports numbers/units, unambiguous years/dates/day intervals, countries/continents, limited US regions, colors, named anatomical subtypes and conservatively identified people. A question containing “Organ Mountains” cannot turn its location into a body organ. `who` alone never forces a person type. This inventory intentionally sacrifices coverage where identity is uncertain.
- Match longest original spans once, with lexical and numeric-slot boundaries. Naked 7 cannot alter 3.7, 7.5, −7, $7, 7,000 or twenty-seven. Actual mappings and modified-document hashes are stored. Residual declared original aliases or unresolved person surnames invalidate the intervention.
- Only actual injected values, explicit finite equivalents and complete dates grounded around actually replaced year spans become CF aliases. An unseen full-date candidate is not a success target. Bare age is licensed only if the original raw gold annotation includes a bare age.
- Finite whole-answer equality covers digits/number words, valid calendar representations, exact months/years conversion, percent units, and under/beneath. Articles are ignored only after parsing for comparison. No substring matching, LLM judge, free synonym guessing or unsupported numerical rounding.
- `Hall`: CF final answer matches any explicitly equivalent original gold. `True`: it matches an actually injected alias and not an original equivalent. `Amb`: any other successfully generated answer, including no legal terminal answer. Errors/retry exhaustion never become Amb.
- Greedy BF16, TP1, same 8,192 context / 512 new-token budget, exact seen documents, no retrieval, stop `</answer>` included for the frozen terminal parser. Context overflow is an explicit error/exclusion, not silent truncation.

The source hash, checkpoint full inventory, protocol/config, code-file hashes and occurrence identity bind request IDs. `strict_v5` rejects v4 manifests/cache. It preserves atomic results, task/GPU leases, bounded batching, attempt logs, token cost, final full-file SHA verification and independent classification recomputation. `identity.py` imports only hash/read helpers from the unchanged v4 core; all preparation, intervention, classification and aggregation use `core_v5`.

## Coverage and interpretation

Report all original correct/direct/searched counts, READY count, excluded reasons, completed/pending/errors, TC/DInv denominators and token costs. Unsupported answers are not model failures. Different coverage across methods must be displayed; do not compare conditional TC alone as an unconditional quality ranking.

The fixed 40-case diagnostic set is stratified from **v4 Amb**, not randomly sampled from NQ. Its exclusion ratio is a debugging result, not estimated population prevalence. Its old CF generations are never reused as v5 generations.

## Deployment files

New scientific implementation: `strict_v5.py`, `core_v5.py`, `replacements_v5.py` in `reproduction/grounding_eval/`. Also deploy unchanged `identity.py`, `core.py` (identity hash helpers only), `normal_a800.py` (UUID lease), and frozen `search_r1/qa_protocol.py`, with empty package `__init__.py` files. The manifest hashes all these executable dependencies. `operations/strict_v5_audit40.py` is a CPU-only optional diagnostic; it is not a GPU runner.

Use a **new immutable runtime and output directory**, not a running v4 directory. Run prepare/full SHA detached with logs; poll through short connections. The root operator reviews deployment and the real pilot before six-method rollout.

```bash
# From the new runtime, using the established model environment.
python -m reproduction.grounding_eval.strict_v5 prepare \
  --source /REMOTE/METHOD-normal/normal.jsonl \
  --normal-config /REMOTE/METHOD-normal/normal-config.json \
  --checkpoint /REMOTE/HF/global_step_200 \
  --output /REMOTE/METHOD-cf-strict-v5 \
  --method METHOD --run-id VERIFIED_ORIGINAL_RUN_ID --expected-count 3610
# Omit --seed when the historical method seed is unknown; intervention seed=1729.
python -m reproduction.grounding_eval.strict_v5 pilot \
  --manifest /REMOTE/METHOD-cf-strict-v5/manifest.json --gpu-uuid GPU-... --max-batch 256
python -m reproduction.grounding_eval.strict_v5 run \
  --manifest /REMOTE/METHOD-cf-strict-v5/manifest.json --gpu-uuid GPU-... --max-batch 256
python -m reproduction.grounding_eval.strict_v5 verify \
  --manifest /REMOTE/METHOD-cf-strict-v5/manifest.json
```

On shared H100/HPC3 allocation, use its existing scheduler allocation/device binding and independently verified conservative memory budget; do not provide an A800-only physical lease path on another cluster. Normal source identity and all scientific budgets remain unchanged. Pilot resumes in the same new output, preserving its records. `--expected-count 128` is explicitly pilot-only and cannot count as the final 3,610 result. Full runs can optionally reuse same-protocol pilot results with exact method/checkpoint/config/source-record identities via `--reuse-results`; v4 reuse is refused.

CPU checks: `python -m pytest -q tests/reproduction/test_grounding_strict_v5.py`. No GPU was used during implementation validation.
