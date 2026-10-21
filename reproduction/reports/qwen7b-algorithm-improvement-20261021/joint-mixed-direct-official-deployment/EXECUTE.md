# Official Mixed-only production entry

Controller `4f4cf5e5b64735eb51f70e7c65708b6f4a249fcc` is published and verified. The actual immutable21-file runtime was staged and compiled with qualified Python3.10 at `A/joint-mixed-direct-official-169028-v1/code-dd7ecdf7060f`. Its source-publication is6458B/SHA`cdae3137aec68c9521afe2bb11e60fb67fc5922b7db00bb076b0e2f787bfce80`; the backed-up file has identical bytes/SHA.

Per-file origins are worker/checkpoint`e2fd6e7fbe4fbd6d62fc86923565c01765d02794` (4 actual tiny CPU fixtures passed), model`8c0ccaf4eda6d97d46c9f0ea82620aa1a6d9a973`, evaluation`7380f3bfa82e4d699a617590d25d56bf4ef8b5ad`, sampler`c98ddcd7861b7023f3efa0d82d3c1fbea0664a13`, and other qualified helpers`3b56e583efe2469ac9108873c31de7144a5201e6`. The current controller is separately bound; it does not rename or modify that runtime/publication.

Actual source stage, idempotent re-verification and prepare have run. Preparation currently returns`WAITING_FOR_ACTUAL_FULL169028_SOURCE_RESOLVED_BANK`, `config_written=false`, `GPU_submitted=false`. The root owner is completing the independent full-bank join and owns the final prepare/submit. This deployment task has not submitted a GPU.

From the repository root, after the actual bank complete appears:

```sh
python3 -m reproduction.operations.qwen7b_warmup.deploy_joint_mixed_direct_official --action prepare
```

Only after`OFFICIAL_MIXED_ONLY_FULL169028_METADATA_READY_NOT_SUBMITTED` and the root's actual-submit authorization:

```sh
python3 -m reproduction.operations.qwen7b_warmup.deploy_joint_mixed_direct_official --action submit --submit-authorized
python3 -m reproduction.operations.qwen7b_warmup.deploy_joint_mixed_direct_official --action status
```

The entry requests normal/acd_u1H100/8CPU/128G/48h, no dependency, withUSR1@900. Unknown submission retains its intent and requires same-name/job inspection before retry. It trains only the direct real-G5 Mixed logit: official native block27 and fresh group head/Adam0, LR5e-6/EMA.99,169028 realG10 questions,5283 accepted updates per epoch×4, four quarter-weight G5 views/Q32, seed1729/head9137. It keeps CP0/1/2, epoch pins and atomic rolling128; fixed512/2048 raw+EMA are checkpoint-first, and all6656 true evaluation identities remain excluded. Official inventory is the explicit external contract ref. Full official weight and feature payload SHAs are verified inside the GPU allocation before the first forward. No E4 probe/Adam/RNG, trajectory loss, old dual-BCE entry or final4096 scoring is enabled.
