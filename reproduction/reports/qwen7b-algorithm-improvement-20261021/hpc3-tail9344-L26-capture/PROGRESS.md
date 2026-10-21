# Fixed9344 source-resolved L26 capture: actual progress

Job **745731**, normal/acd_u **1H1008CPU128GiB2h**, no dependency, is RUNNING on ACD1-36. It captures only the fixed9344 own-HPC3 G10 input, four2336-question shards sequentially on one GPU. It performs no training, new response generation, evaluation or change to main745501.

Actual observations (same PID3742092 / birth132330345):

| Epoch | Completed trajectory rows | Complete chunks | Actual packed bytes |
|---|---:|---:|---:|
|1792591756.656|2855|8|477846136|
|1792591875.144|9020|28|1506787164|

Both measurements have confirmed a visible single H100, no failure receipt and exact main RUNNING. The dataset goal is93440 rows and292 chunks; final feature-bank acceptance remains outstanding. No improved probe score is claimed.

The immutable initial CPU preparation0d872886/v1 rejected an extra authenticated `count` descriptor key before any row/GPU work. Its logs/partials remain preserved. Fix772381da/v2 verifies path/bytes/SHA strictly and retains count metadata; four focused source-resolution tests passed. Current producer code remains immutable772381da; core replay uses the unchanged qualified243 runtime db5b5ac04173, core50091134 and golden3eeeb79b. The official actor revision is a09a35458c702b33eeacc393d103063234e8bc28, question+Prefix20, captureL26, BF16, batch32/padded-token budget16384. HF/native bitexact equivalence is not claimed. No postblock cache is used.

Input qualification is actual9344 questions /93440 authentic old5+new5, all strict EM recomputed with0 mismatches and no same-question old/new seed collisions. The source is fixed HPC3 execution, not a donor union. Whole G5 physical execution is resolved before capture. CPU projection completed in27.36s; final streamed physical-source readback took14.68s/RSS54MiB on CPU[0,1]. It freshly checked149 original prefix files and149 seals after projection, bound all93440 canonical prefix rows to physical file/execution, and sent one actual old5 plus one actual HPC3-new5 prefix through the qualified causal normalizer. No tensor/model was loaded for that check.

The final owned-allocation join is configured to fresh-hash every feature payload, reconstruct every retained row and verify its rowSHA/finite values, then bind source-prefix bodySHA, canonical physical prefix rowSHA and retained-feature rowSHA in separate source-binding sidecars before writing bank complete. A successful submit or RUNNING does not satisfy that final gate.

Small actual control artifacts (all code/config/metadata only):

| File | Bytes | SHA256 |
|---|---:|---|
|source-publication.json|1092|`421f41227bbc387ea3f114059e8172d6a9ef92af2495cae063458d0865a74d82`|
|preparation.json|3805|`c63010438f4571efd247441168a43c4966e8c2017d099fd0f452f42947b6e77c`|
|capture.sbatch|2047|`1534ac7743e17f8a3fb958aec252fb9cd7d043cfd5c3293fbf2fbc93664ebfad`|
|submission-plan.json|1113|`727636126d24ac926ef535ac8b924683f2c4df1d126b431e68a09b986118e12b`|
|submission-intent.json|1206|`e96848fafb04ef685b14c51391ffa58c9275b362eef1605ca20955964399792c`|
|submission-identity.json|2891|`fe4a277b5f563cf59055ea735abdabaf4af296ac7fcc55df6bbf80595db76d33`|
|submission-result.json|401|`f79b8d987f7de92e9a52f4c6b23e6a9ab927ae67f501576c8557293c1b4a3e34`|
|pre-submit-physical-readback.json|1579|`62ed01ed8547cf4cd7cceecf1bba9bdd8f48c78a141c76a9bbb3ab7287ecff28`|

Source raw complete:24395B SHA `0898b54516068d5b2c0fd1e2d95befb47d58ed2e96ca1065fc9b556387abbbfd`; source-resolution contract:485B SHA `229710cbfccc53c92be96081535eeeee308cc52b969b07713258212c2b0d86c2`.

Outputs remain entirely on HPC3 under `algorithm-improvement-20261021/hpc3-tail9344-independent-L26-prepared-v2`. Former warmup text exposure is not presented as unseen-question evaluation. Current124194 training bank/config remain unchanged. This is progress, not a final feature-ready receipt.
