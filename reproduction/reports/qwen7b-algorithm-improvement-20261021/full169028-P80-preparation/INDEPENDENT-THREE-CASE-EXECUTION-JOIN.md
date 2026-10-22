# Independent three-case physical-execution / strictEM audit

The mismatch is an incorrect physical-execution join, not evidence of a corrected EM postprocess version. Canonical labels1/1/0 are independently reproduced from the execution actually selected by the canonical bank; donor RAW labels0/0/1 are also independently correct for that other execution.

| question | donor answer / strictEM | chosenHPC3 answer / strictEM | first20 equal | first80 equal | full response equal |
|---|---|---|---|---|---|
| hotpotqa:train:70604 | United Kingdom / 0 | England / 1 | True | False | False |
| hotpotqa:train:2094 | None / 0 | No / 1 | True | False | False |
| hotpotqa:train:81793 | Lea & Perrins / 1 | Mr & Mrs T / 0 | True | False | False |

For Hotpot2094 the donor final authored segment contains an orphan `</think>` before `<answer>No</answer>`; the current strict protocol parser rejects that segment, hence0. The chosenHPC3 final segment has a valid terminal answer action and normalizes to groundtruth`no`, hence1. This is a concrete strict-format difference, not an alternate gold answer.

All three canonical groups originate in `HPC3_first9344`, whose source bank groups retain `chosen_new5_execution_run_id`, `chosen_new5_seal`, `old5_seal`, and `source_ordinal`. The final joint169028 groups project away those extra fields. The selected source resolution contract explicitly states `same_protocol_tid_not_payload_identity=true`, wholeG5 perqid selection, and binding to the chosen physical source/seal/retained-row SHA. The metadata execution run is36b784a3bf964e69a6130bcc41d68809; payload protocol trajectory IDs/run_id retain70188… in both executions. Thus payload IDs alone cannot resolve the physical source.

The official tokenizer and current exact `search_g1.signals`, `search_r1.qa_protocol`, `qa_em` file hashes were checked; the local source hashes exactly match the remote code used for both recomputations. No model weights, checkpoint, final4096, label updates or training mutations were used. Only three small example summaries remain local. Full-cohort mismatch prevalence is not inferred from these cases.

Evidence SHA256s:

- `independent-three-case-lineage.json`: `3c35b36856b3c270d614dc59a26d222dabb491c013fd40261ed5ef8fa30a1805`.
- `independent-three-RAW-strictEM-recomputed.json`: `7f97c151ceb5ef931a09d2827b46db251ac1c8f770510c3640363ace7a552472`.
- `independent-three-chosenHPC3-strictEM-recomputed.json`: `047f32330b8e8cdd7e427000bea96d57a85b17968f378d07de24a444bba41445`.
