# Accepted122 real G10 Prefix20 layer26 capture

Source runtime: `57bd9e337d6d2a75d2603b684d28752bfe3bf520` on `codex/peer-audited122-L26-replay-20261021`; 37 code files independently SHA-bound, five remote CPU tests passed.

The accepted15616 questions (NQ12300/HotpotQA3316;156160 authentic old5+new5 trajectories) are projected without label/score selection. Every sealed prefix was checked by the original causal normalizer. Hotpot initial prompt126–248 and maximum full causal context717; NQ prompt127–151 and maximum context656. Zero overlength;19 real short-terminal prefixes retained. This is additional to current30016 data, but584 questions overlap older controlled6528 training, so there is no claim of independence from all historical fitting.

Two normal/acd_u jobs743669/743670 each allocate2H100,16CPU,128G,2h, no dependency. Four disjoint workers each3904 questions/39040 trajectories. Each srun task must actually see one H100; GPU visibility is not inferred from a literal CUDA_VISIBLE_DEVICES number. The unchanged qualified batch32/16384 core computes postblocks20/23/26; only26 is persisted using `lossless_question_base_xor_policy_v1`. Actor/probe/optimizer updates0. Existing743619/frozen30016 state unchanged.

Remote root: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-audited122-L26-replay-v1`.

CPU preparation is complete, GPU completion remains unaccepted until actual packed files, source IDs, reader and final Slurm exit pass. No model/rollout/tensor artifacts are local.
