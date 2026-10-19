# Strict NQ grounding (2026-10-19)

## Protocol audit correction — 2026-10-19

Frozen v4 outputs are retained for reproducibility, but their TC/D-Inv are
**diagnostic, not accepted paper-strict measurements**. Stratified response audits
found invalid type substitutions/date ranges, EM-normalized aliases corrupting
numeric type parsing, and missed original/counterfactual semantic aliases. An
execution `complete` or matching SHA proves successful computation and data
integrity, not semantic validity of the intervention. Do not put frozen v4 TC in a
paper main table as an accepted strict result. Normal QA EM and actual token
accounting are independent of these counterfactual intervention defects.

The separately published official changed-answer replay keeps the existing
responses/interventions and changes only the classification rule (optionally its
answer parser). It does not reproduce the official fake pool, prompt, judge model,
missing-intervention fallback or independent seeds. Label it
`official-changed-rule-replay-on-v4-responses`, never `official-code-legacy`
full reproduction. Its classes and outputs remain separate from strict caches.
A corrected strict implementation must have a new protocol/config/source hash,
new request IDs and a new output root; v4 runtime and accepted normal traces are
immutable. The implementation description below documents frozen v4 intent and
operation, not a new scientific acceptance.

This offline evaluator never imports or changes a trainer. Its normal-input adapter
must preserve the selected historical checkpoint's normal QA interaction protocol.
Only the fixed 3,610 NQ occurrences are supplemented when historical full traces
are unavailable. A question ID is not a unique occurrence: retain original ordinal.

`paper-strict-v1` implements the paper definition, rather than legacy “any answer
change = True” or its service-error fallback. Operational choices not specified by
that definition are explicit in `CF_CONFIG`: deterministic boundary-aware alias
replacement (longest first, no cascading), a single user message containing the
question and modified actual seen docs, greedy BF16,512 generated tokens,8,192
context, no new retrieval, strict terminal nonnested answer parsing. Search tokens
are not suppressed. These choices are not presented as exact author settings.

The frozen v4 intervention stops at `</answer>` and retains that closing tag.
Replacement aliases include terminal-punctuation variants. Question-guided types
handle quantities/dates before person cues; unknown-type fallback is marked in the
record. Exact normalized injected values and explicit aliases determine True;
unlisted semantic paraphrases remain Amb. Earlier diagnostic pilots are separate
and must not be reused by the formal run.

Commands (run from the repository, all paths on the remote persistent filesystem):

```
python -m reproduction.grounding_eval prepare --source normal.jsonl --checkpoint HF_DIR --output NEW_DIR --method NAME --run-id ID --normal-config normal-config.json
python -m reproduction.grounding_eval pilot --manifest NEW_DIR/manifest.json
python -m reproduction.grounding_eval run --manifest NEW_DIR/manifest.json --max-batch 256
python -m reproduction.grounding_eval verify --manifest NEW_DIR/manifest.json
```

Use `--max-batch 256` (space between flag and value). Pilot uses16 valid
interventions. Full run resumes those atomic result files, starts with16 requests,
and raises batch size to32/64/128/256. vLLM bounds KV cache and chunked prefill;
actual OOM reduces the request batch, never converts an error to Amb. Locks prevent
duplicate workers. Contexts exceeding the frozen budget remain explicitly invalid,
never silently truncate the original observations. Full model SHA is checked at
prepare and independent final verification; source/config/code hashes bind caches.

Each normal input JSONL row contains `data_source="nq"`, original
`evaluation_ordinal`, `qid`, `question`, `gold` (list of aliases), `model_spans`
(policy actions, excluding observations), `seen_docs` (exact post-truncation tool
observations), `actual_retrieval_calls`, and `source_shard_sha256`. Adapters retain
raw token provenance on the remote side. Model spans cannot be reconstructed from
untrusted concatenated model+tool text by searching for answer tags.

Token accounting records each model call and the whole retained trajectory:
initial prompt, retained model generation, observations, and environment corrections.
Raw generated tokens additionally include tails discarded by the action parser.
Cumulative model input counts repeated history on every call; it is distinct from
the single retained trajectory length. Normal and counterfactual usage are reported
separately. Unavailable per-request KV-cache hits are null, never guessed. Local
audits contain small aggregates only; raw token IDs and trajectories stay remote.

Independent aggregation recounts Direct/True/Amb/Hall globally and recomputes
classification from actual injected values. TC includes correct direct answers;
D-Inv does not. Empty denominators are null. Unperturbable/invalid/source/error
entries are excluded with coverage counts. Recoverable missing/error entries imply
provisional. One seed has no sample SD. No model/raw-output/credential goes in Git
or onto the Mac; only code/config and compact audit summaries may be local.

Do not launch all methods before the first checkpoint's pilot and full acceptance.
Record Slurm allocation or authorized shared-GPU reservation, engine metadata,
two actual completed-entry/byte measurements, exit status, then final full-SHA and
classification acceptance. Always keep remote persistent copies of artifacts.
