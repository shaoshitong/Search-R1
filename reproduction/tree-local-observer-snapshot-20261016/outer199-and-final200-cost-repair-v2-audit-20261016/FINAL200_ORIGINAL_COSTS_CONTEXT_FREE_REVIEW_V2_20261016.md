# V2 — targeted static repair verification

Final targeted result for the exact source revision below: **PASS — all four V1 findings resolved by static inspection**. The missing executable input-preparation dependency now has a concrete `prepare-pins` entry. PASS applies to the targeted source review, not runtime or actual200 acceptance.

Source root: `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-speed-repair-20261008/`.

| File | Bytes | SHA256 |
|---|---:|---|
| final_original_costs200.py | 51307 | 2dc7f95b417204862b1207e1e1bca0b4db7f9f08533d5b688dad763ed95d875a |
| final_original_costs200_remote.py | 31870 | 16433eb3a9ab97eb3ea82e695a43964205ecb3a206f538a6f1ed6e28a1493466 |
| variance200_original_record_coverage.py | 33932 | 7648b529a61dae49c2846b2d59976d4ec3bf2e2708ddfd78bf3b1f30651d5706 |

Only local text and referenced small metadata were read. No entry, import of the entry, compilation, test, collector, SSH, or training operation was run. Writer sources were not modified. The contract/publication update was still being finalized when this revision was supplied; this report does not attest future contract or commit bytes.

Final publication binding supplied by root and confirmed in the local receipt: commit `c969768583a78ca74c6962eb675f414a97202da6`; `outputs/tree-nested-branch-handoff-20261013/LOCAL_FINAL200_ORIGINAL_COST_REPAIR_V2_PUBLICATION_20261016.json`, 21742B, SHA256 `4cd6f397dccf33f5cd2244c097e838a341567cd6b17f56b441f365a05ee7a9ab`. The receipt was locally read and its bytes/SHA matched. No new remote verification was performed by this reviewer.

## Last P2 repair, verified against final bytes

**Resolved.** The first V2 pass reviewed master51229B/SHAa1fd0a3d05f00336c08e7d0875cd943d5968d2869a3c6713d245aa5ae7ae371e and found that a BASE CAPTURE_READBACK left by a failed mirror write was only inspected during recovery; its absent mirrors were skipped. The final master51307B/SHA2dc7f95b417204862b1207e1e1bca0b4db7f9f08533d5b688dad763ed95d875a now reads `previous_raw`, validates its product, and calls `publish(readback.name, previous_raw)` (`final_original_costs200.py:508–510`). The publisher preserves those exact bytes, fills missing mirrors, and registers the receipt in GENERATED; `checks()` immediately afterward (`:511`) therefore verifies all three copies. It does not regenerate timestamp/readback observations or call SSH. Only this precise final receipt-republish change was rechecked; the full review was not restarted. No remaining actionable defect was identified in the requested targeted scope.

## V1 repair verification

- **Atomic once-only ownership: resolved.** `claim_attempt():199–211` uses `os.open(..., O_CREAT|O_EXCL)` on the authoritative BASE attempt before submission. `collect():527–545` does not invoke the helper when ownership is denied. `prepare_pins():573–586` acquires its separately named attempt before its one helper call. A mirror failure leaves the primary attempt in place. The general publisher uses a unique temporary file, fsync, and `os.link` rather than overwriting existing immutable files (`:177–196`).
- **Original dev phase: resolved.** `development_record():99–104` takes phase from the original control value. Aggregate uses that function (`:664–667`) and describes coverage as 40 dev records while retaining acceptance/periodic phases. Remote checks still require periodic only for the two new controls.
- **Partial known sums: resolved.** `aggregate_values():77–96` separately sums known values, retains a null complete total when any required value is absent, and exposes partial-known outers plus record coverage. Aggregate branch fields pass `candidate_round_ledger_known_sums`, observed field counts, and unique-round required counts (`:642–647`). Candidate outcomes pass their own `field_known_sums` and per-outcome record counts (`:652–658`). Remote candidate output now includes `unique_round_count` (`final_original_costs200_remote.py:208`). Missing values have not been replaced with zero.
- **Response preservation and publishing boundary: resolved in the requested scope, including the final P2 above.** `persist_response():214–222` frames complete stdout/stderr bytes with SHA, lengths, and header; the header includes actual exit code, input/source bindings, timing, transport, and one actual call. Persistence precedes semantic parsing and dependent rendering. The collect try block includes the complete `finish_collect` publication stage (`:537–547`); prepare likewise includes `finish_prepare` (`:578–588`). Recovery requires actual response bytes/SHA, verifies framing and source/input bindings, and uses only saved response and local publication (`:591–603`), with zero new helper calls. Unknown transport failures without a returned response remain fail-closed errors and do not establish an exit0 result or permission to replay.

## New executable input preparation

`final_original_costs200_remote.py:230–341` is a concrete prepare implementation, not just contract pseudocode. The local entry (`final_original_costs200.py:568–588`) assembles and submits that mode once after validating the real actual200 boundary (`:333–346`). Native rolling200 without the complete original metric200 is explicitly rejected before preparation. It does not manufacture actual200 evidence.

The implementation statically covers the requested preparation scope:

- It fixes the three own run identities and effective segments to outer1, 2..5, 6..200, validates resolved config SHA, scans accepted original metric lines, rejects conflicting/incomplete selected lines, and binds the real last accepted-line byte end and prefix SHA after a stable prefix reread (`remote:235–252`).
- It maps child credited attempt-results by actual accepted outer, joins Controller identity/attempt, enumerates rank filenames into eight selected ranks per outer, and rejects ambiguous accepted selections (`:253–289`). The rows contain 200 effective outers and 1600 rank refs by construction; the later collector validates actual per-file identities and physical opportunity records.
- It reuses prior audit3..190 byte/SHA/name bindings; other outer audits use exact own filenames and reject ambiguity rather than selecting newest. Ambiguous audit selection reads only a bounded header for actual global_step/attempt_id, without returning audit payload (`:290–316`).
- It inventories admitted own native JSON patterns and binds currently retained metadata refs, with unavailable historical JSON handled by the collector. It collects relevant own noncredited metadata refs and does not traverse PT/tensors (`:317–327`, `native_inventory():215–227`).
- It skips the old38 controls by their bound original filenames and locates exactly COMPLETE periodic195+200, returning their actual path/bytes/SHA (`:328–337`). The collector later checks split coverage, source/config/model/data, metric equality, and source200 poll ledger binding. Old38 originals remain local reuse.

`finish_prepare():550–565` binds the actual saved preparation response into the frozen manifest and records preparation1 separately from collector0. The eventual cost summary records prepare calls separately from collector1/local aggregation1 (`:709–712`); old source1ed is retained7/7 with zero new calls. Actual extra input downloads/reads outside these entries still require their own truthful accounting; prepare1+collect1 is not a claim of one SSH for the full workflow.

Both modes pass through actual compressed argv gates before their attempt/helper submission (`transport_program():418–426`): target command at most32768 bytes, management command at most65536, complete local argv at most80000. Exact future manifest/program bytes still determine whether those gates pass; no guessed fit is attested by this static review. Failure does not trigger another SSH invocation. Preparation and collection output limits are explicit and do not truncate original records or path lists.

The returned material remains bounded audit metadata and selected scalar lines, not raw audit leaves/question groups, token/action/logprob/text payloads, weights/checkpoints, tensors, environment, or FAISS. The previously withdrawn Mac-mirror-only P1 remains withdrawn.

## Scope and limits

This is a targeted re-verification of V1 plus the requested preparation/recovery additions. No training algorithm review, new tests, governance chain, or remote operation was introduced. Source variance remains byte-identical to V1; the new capture/scalar schema and complete-field protocol remain compatible with its existing bindings by inspection.

The supplied training state was actual199 complete with200 sampling underway. This review does not independently measure that state, claim actual200/dev200, execute prepare/collect/recovery/variance, or attest final HF/tokenizer/51713/cleanup acceptance. All new runtime counts remain zero as supplied. Final master, remote, and variance bytes were locally read back and bound to the table above and the final publication receipt. This review did not independently run remote Git verification; root's published commit/remote observations are retained as supplied evidence. Any later source changes need their own precise diff/SHA binding and must not inherit this PASS silently.
