# Scalar versus batched HF causal replay: actual bounded qualification

Job **743322** completed with Slurm exit **0:0** on one normal/acd_u H100 in 81 seconds. The fixture used eight label-blind selected R2 training questions and their unchanged five actual Prefix20 records each. It read no correctness labels and changed no model parameters.

| Frozen HF prefill | 40-trajectory replay | CUDA peak |
|---|---:|---:|
| Scalar | 4.049 s | 14.288 GB |
| Batch 16, padded-token budget 16,384 | 1.642 s | 15.145 GB |
| Batch 32, same token budget | 0.798 s | 16.063 GB |

Retained masks, positions and lengths were identical. L26 retained-feature relative L2 difference was 0.01307 and global cosine 0.9999146. The same frozen E4 EMA then produced maximum absolute trajectory-probability difference 0.04071 and Mixed-score difference 0.01696. Neither fixed 0.5 classification changed. All eight question ranks and the top four stayed the same. Batch 16 and 32 produced identical numeric results on this fixture.

This proves bounded functional and numeric qualification, not bit equality or unchanged ordering on 2,048 questions. Actual fixture contexts were 153–184 tokens and contained no short or multisegment Prefix20 record. CPU tests separately cover those alignment paths; broader actual records remain subject to feature-reader and causal-position checks.

The parent explicitly accepted batch **32 / 16,384** as a new non-bitexact HF causal feature branch. New training and new frozen-validation capture use that same branch. E4 baseline and candidate must be evaluated on the same new validation features. Prior scalar R2 diagnostics remain diagnostic and cannot supply an unqualified matched gate against native-feature E4 scores.

The completion reference is `algorithm-improvement-20261021/replay-batched-golden-v3/output/complete.json`, SHA256 `3eeeb79b750d2e0bfee81c9a283c7428d0629b0d56815f91909c93e2bd9f5715`. E4 checkpoint SHA256 is `e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8`; batched core SHA256 is `500911347444a07693fb4405a6f6fbbcad7b70375522a52833953b426970f506`. Exact source/config/fixture descriptors and the explicit acceptance receipt are retained remotely; only small aggregates are mirrored locally.
