# Peer sealed-data delta: read-only two-observation plan

Fresh NAS10 metadata observations at epochs **1792538188.614480** and **1792538287.046558** (98.43 seconds apart) agreed on **388 sealed batches / 49,664 questions / 248,320 new-seed trajectories**. The accepted HPC3 snapshot remains **244 batches / 31,232 questions / 156,160 new-seed trajectories**, across16 previously verified archives. Its old5+new5 raw-G10 ingest is still31232questions/312320trajectories.

| Scope | Batches | Questions | New-seed trajectories | NQ | HotpotQA |
| :-- | --: | --: | --: | --: | --: |
| Total delta, absent from imported snapshot | 144 | 18,432 | 92,160 | 12,428 | 6,004 |
| Existing independent proof, all current payload fstats still match | 122 | 15,616 | 78,080 | 12,300 | 3,316 |
| Sealed, pending a matching existing independent audit | 22 | 2,816 | 14,080 | 128 | 2,688 |

The two observations have exactly identical144delta `(input path,input SHA,seal SHA)` bindings. Previously imported244seals/input bindings remain unchanged. Delta versus imported membership has **zero overlapping batch paths, qids, question hashes or trajectory IDs**. All delta input JSON received fresh SHA checks; per-batch domain counts, question-identity digests, expected own-G5 ID digests, seal/config refs and proof status are recorded. No labels or probe scores selected these batches.

## Exact scope and preserved contracts

- Canonical source: `/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020`, actual hostnameh100-10. The old physical alias is recorded only for mapping old receipts.
- Same run`70188b8e50fd4b45bc420ce58014f46e`, modelrevision`a09a35458c702b33eeacc393d103063234e8bc28`, seed20261021/source seed1729, attempt`independent-seed20261021-g5-20261020`, Prefix20, L26, question cap273, warmup-only, hidden disabled and actorupdates0.
- Source config SHA`856a5580c9faf6968aee9535bbc600d9c07f0b018dc13e85611fa76105bba789` and canonical1322-batch/169028-question input-manifest SHA match the imported snapshot. Every delta batch config also matches the root contract. The entire169028 generation is **incomplete**; remaining934batches were unsealed in these finite observations.
- Current append training uses the previously imported **NQ-only** enrichment. This future delta contains both NQ and HotpotQA; it has **not** been exported, ingested, featurized or trained. Original warmup already saw the canonical training questions, so these are additional seeds/trajectories and newly imported scope, not unseen-to-warmup questions.
- Fresh HPC3 small receipts bind original complete SHA`8a0f5128f7f613d4f6ee9cc5d6135f461f66feb66e73b65bbfa5e78ada875d5d` and raw-ingest complete SHA`1fd5c29862267bc202419973ba6238d9c7b66de7ef2199dba725dda7fecf0804`. Source16archive hashes/cohort identities match the actually imported catalog. Previous1.37GB archive/member SHA acceptance is inherited here; archives were not rehashed again.

## What “existing audited” means

The source auditor's last aggregate is INCOMPLETE,366verifiedbatches/46848questions, zero failures, epoch1792537096.437206. Its historical pending count2 predates the current388seals. This read recomputed current proof-to-file **fstat** bindings and found366matching proofs in total; the122/22 delta split is based on current metadata, not the stale aggregate pending count.

No prefix/full/own-EM payload, archive, hidden tensor or model was opened. Existing audit proofs check own-EM/full correspondence and native seed/protocol provenance, but do **not** claim tokenizer strict-EM rerun or deep states-shard audit. Matching cached fstats is **not a fresh payload SHA** and **not a new tokenizer strict-EM pass**. Future real export/ingest must freshly verify archive/member SHA and own strict EM/causal-seed correspondence, plus the existing training/evaluation identity exclusions; this metadata plan alone does not authorize acceptance or training.

## Frozen artifacts

- Complete current delta plan SHA: `104d1a7b246814419a03959ebc43c99e2097a44d214971dc8eba317b56ec263b`.
- [Existing-audited122-batch plan](existing-audited-delta-plan.json), batch-record SHA`673b8cb6abe7108e1d03a98b518121dafcd1ec10b156ec78f37c5faacfb7cc21`.
- [Pending22-batch plan](sealed-pending-audit-delta-plan.json), batch-record SHA`b98b44e8cb7b2456540d8c9e97c3af0cf6b34ad1efb931885a8f020fa360c50e`.
- [First observation](source-observation-1.json), [second observation](source-observation-2.json), [fresh imported receipt binding](imported-snapshot-fresh-binding.json).
- Worker source commit`cb42e5af954427e6993c1ef3bbf468fa3a2bc536`, worker source SHA`154e22b150562baf3945308c300fd6292b1724988d093c8d9d92c9a5cad798e5`. Three CPU fixtures passed; both real NAS10 metadata workers exited0 (0.937s and0.950s).
- Operator: `observe-source-delta.py`; strict existing H100 control route, source/binding via stdinRAM, CPU≤2/AS3GiB/CPU120s. No remote source/claim/lease/service/process was changed. Only code and small identity/config/audit metadata are saved on Mac; no raw rows are returned.

Current job743517 was untouched. No payload export/download, GPU request or additional training was started by this task.
