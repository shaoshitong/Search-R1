# OPSD3304 A800 standby: prepared, not started

The preserved historical Python/config bundle is deployed at
`/data/shaoshitong/grounding-eval-20261019/runtime-opsd-legacy-2ac4463afa45`.
Its 208 files contain 1,742,022 bytes of Python/config only; every destination
file was reread and matched its source SHA. No weights, parquet, environment,
wheel, credential, or raw trajectory passed through Mac.

CPU imports passed with CUDA hidden, using `/data/search-g1/env/bin/python`:
torch 2.4.0, vLLM 0.6.3, transformers 4.46.3, pandas 2.3.3, pyarrow 25.0.1,
OmegaConf 2.3.1, tensordict 0.5.0. The original generation module SHA is
`91e281dc853c077dfcda3f05b64a714c009c8993fe05b2069fcc2def8f114470`.
Existing A800 final.parquet matches
`55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1`,
with 51,713 occurrences and 3,610 NQ occurrences at ordinals 0–3609.

Normal protocol remains the original `_validate` LLMGenerationManager:
4 search turns, start 2048/context 4096/action 500/observation 500 tokens,
top 3, greedy, BF16 TP1, **force_final_answer=False**. This is distinct from
the authored-prefix Easy adapter and from the OPSD training rollout controller.
Model-generated policy tokens, raw sampled tokens, observation/prompt tokens,
trajectory totals and cumulative repeated-context input tokens are recorded.

Before any inference, the separate asset worker must finish fresh full-file
checkpoint validation, and the root must bind one currently idle GPU UUID and
its physical numeric index. The adapter takes the same OFD UUID lease namespace
as A800 training/CF, rejects existing compute or bootstrap processes, and uses a
verified **numeric** CUDA_VISIBLE_DEVICES mapping (not a UUID string).

Prepared command shape only; replace the explicit placeholders from fresh
resource admission and an immutable deployed evaluation runtime:

```sh
/data/search-g1/env/bin/python -B /ABS/REVIEWED_RUNTIME/reproduction/grounding_eval/normal_h100.py \
  --method opsd \
  --runtime-repo /data/shaoshitong/grounding-eval-20261019/runtime-opsd-legacy-2ac4463afa45/repo \
  --source-config /data/shaoshitong/grounding-eval-20261019/runtime-opsd-legacy-2ac4463afa45/source-config.json \
  --data /data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512/data/final.parquet \
  --checkpoint /data/shaoshitong/grounding-eval-20261019/assets/opsd3304-iter200 \
  --output /data/shaoshitong/grounding-eval-20261019/opsd3304-normal-v4 \
  --run-id 'H100 native training job3304; formal-3304; actor/global_step_200' \
  --retrieval-url http://127.0.0.1:19000/retrieve \
  --gpu-index PHYSICAL_INDEX --gpu-uuid FROZEN_UUID \
  --gpu-lock-directory /data/shaoshitong/.search-g1-gpu-uuid-locks \
  --batch-size 128 --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.60 --min-start-free-gib 48 --min-running-free-gib 12
```

Start with a bounded pilot, then select a larger safe batch under the
same scientific protocol after actual memory/throughput evidence.

Retrieval caveat: original GPU Flat/E5 were FP16 with 8 index shards; the owned
A800 service uses the same source/spec and 2 shards. Its 16-query exact cropped
observation parity was established for G1 only. This is not evidence of OPSD
historical output parity. Keep the topology difference in provenance and do not
label the rerun an exact reproduction of the historical service output.
