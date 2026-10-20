# DynRPO iter200 NQ grounding, 2026-10-20

Historical source session: `01a0fcbe-15f8-7600-ae84-9487d924560a`.
Protocol record recovered from pinned **审查七种算法实现 (2)**, session
`01a14ab9-289c-7e30-ad26-542dea4f30ed`, local `xia/outputs/grounding-eval-20261019`.

Original training733898 is complete; this is a separate single-H100 evaluation742547
on HPC3 ACD1-67 / emergency_acd. No actor training, new retrieval or normal QA regeneration.
Live evaluation source commit: `7c881cc413b916aedfe288443cd9fca838fb1c8b`.
Remote task root `/data/user/jxu957/experiments/dynrpo-nq-grounding-20261020`.
Exact HF checkpoint is the retained R4 outer0200 attempt d3eadfec8ccd440f82136d69c35cf237.

The existing native final51713 outputs provide exact policy masks, generated token IDs,
post-truncation observation IDs and BF16 actor-version evidence. Export only canonical
NQ3610; require same question/gold/order SHA `23c0a8307f38a7a8d1cbd9e71fb115b7bd091421a3d3b10952fb71c8210b551c`
and original1706 correct. Tool markup cannot be parsed as a model answer.

Frozen original modules provide two independent counterfactual runs:
* v4: broad typed replacement; changed-answer-rule replay is a diagnostic only.
* v5: conservative finite typed replacement and whole-answer equivalence; report
  exclusions and eligible subset, not an unconditional grounding ranking.

Both decode same iter200 checkpoint greedily BF16/TP1, seed1729, context8192,
new-token512, included stop `</answer>`, actual modified seen documents, no new retrieval.
Classify True only for explicitly injected correct alternatives; Hall for invariant
original-correct answers; Amb for other successful responses. Inference errors are errors.
TC = (Direct + True + 0.6*Amb + 0.3*Hall)/(Direct + True + Amb + Hall).
D-Inv = Hall/(True + Amb + Hall). Excluded/unperturbable examples are outside these
conditional denominators and must be reported. v4's separate changed-rule replay
follows the historical six-method reporting without pretending scientific equivalence
with corrected v5. Do not use old user-edited table cells as measured results.

Full checkpoint SHA at preparation/final verification; separate frozen manifests,
independent prepared-field/prompt/classification recounts, source and results hashes.
Raw trajectories/checkpoints remain remote. No checkpoint copying or model download.

`submit_once.py` HAS RUN. Existing submit claim/job.id are authoritative; never replay
submission. `deploy_exact_source.py` HAS RUN with the Git archive on stdin.
`observe_readonly.py` reads the same job and task only. No credentials are in these files.
