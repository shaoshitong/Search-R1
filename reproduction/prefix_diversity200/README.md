# Search-G1 Top1024 → actual EM diversity Top512 (200 outer)

This independent experiment extends source932cb6a16c3ff1e811a235b4a4322a397ee99393. The prior agent owns its Top512 baseline; do not alter or cancel it. No new warmup. Probe initialization uses actual retained CP100 (fixed512 balanced accuracy59.87592805%, independently reloadable); actor starts at official Qwen2.5-3B-Instruct revisionaa8e72537993ba99e69dfaafa59ed015b17504d1. Warmup Adam/RNG are not inherited.

Per outer:5120 distinct diagnostic-excluded questions ×5 independently sampled20-policy-token prefixes. Freeze actor/probe versions for selection. Score h=1-prod(p)-prod(1-p), stably select1024, and continue exactly those original5120 trajectories with native tokens, tool context, log-probabilities and budgets. Rank complete G5s by binary population strict-EM varianceK(5-K)/25, then frozen prefix h descending, then original candidate ordinal. Select exactly512 questions/2560 trajectories for the unchanged Search-G1 grounding soft-reward actor objective. EM is only used for this selection and probe supervision, never substituted for soft reward. No extra outcome-seeking rollout or guarantee of mixed labels; if fewer than512 mixed groups exist, the rest are honest same-label groups.

Human clarification2026-10-16: the probe uses ALL1024 actually completed questions (5120 trajectories), not just actor512. One natural pass, LR1e-6, trajectory-mean BCE + group-mean mixed BCE (weights1/1, denominators5120 and1024). Eight ranks,20 synchronous Adam writes each,16 global batches255 +4 batches260, average256. Each rank naturally visits128 completeG5/640 trajectories. Actor retains20 writes/rank/globalmini128/epochs1/initialmicro128 (local16, noaccumulation), LR1e-6/outerwarmup57. Reduce actor micro only after an actual typed OOM; preserve prior committed state and failed partial evidence.

Report actual Top1024 and actor512 mixed fractions, predicted h, K histograms and EM variance; post-update probe full1024 train and actor-selected512 diagnostics plus fixed512 test ordinary/balanced mixed accuracy; all20 actual per-step global/rank losses and stage wall times. Different batches mean loss need not decrease monotonically. Uncontinued4096 questions have no outcomes: never claim all5120 candidate accuracy. Selection uses labels and changes actor distribution; no unproven unbiasedness or guaranteed QA gain claim.

Execution is on retained18/h100-10/10.33.0.18, independent run namespace with actual8GPU Slurm allocation/UUID leases and owned process cleanup. Use retained18 existing official model/FAISS/runtime, same-cluster shared warmup/diagnostic/data/heads, and isolated missing CPU reward libraries. Loopback HTTP only; no Mac model/tensor/rollout transfer; no cross-cluster asset access. Source and task config must be committed/pushed before dispatch. Outputs are visible from h100-8 via its existing NFS mount of retained18; no global mount or credentials changes.

Save rolling joint actor/probe/optimizer/RNG/pool identity and5/50/100/150/200 milestones, retain raw trajectories and scalar costs remotely, fixed512 QA at prior milestones and final51713. Goal completion requires actual200 joint outer, complete final QA, checkpoint/HF/tokenizer validation, source/config/data identity and owned cleanup. Deployment, a RUNNING job or tests alone are not completion.

Entry: `python -m reproduction.prefix_diversity200.control deploy|submit|observe --target h100-10 ...`. Preflight helper only stages own CPU dependencies and read-only input verification; tests are minimal CPU checks and never rerun warmup/baselines. New code inherits synchronous vLLM token/logprob snapshot fix95f27fd7.

Runtime acceptance2026-10-16: actual joint1/2/3 saved;8rank actor/probe60 each after true Adam40/scheduler2 restoration and20 additional writes. Probe has20 additional failed unsaved physical writes/rank, separately preserved. Normal outer3 phase sum419.492678102s. Current3421 uses frozen9ce4b14a; later observer/report commits are not hot deployed. Updated scalar/control evidence lives in control_evidence, complete assets remain remote.


### QA initialization repair (2026-10-17)

QA uses a CUDA-free parent and eight fresh TP1 worker processes. A constructor
CUDA OOM or the exact observed invalid-memory-profile assertion can be retried
at most four times per rank. Every previous child is waited/reaped first; OOM
backoff reduces KV/profile concurrency only. Model, question split, seeds, BF16,
greedy decoding, interaction budgets and strict scorer remain fixed.

A locked, bound journal preserves the total attempt budget across clean partial
resumes. Completed rank receipts are checked against canonical questions and raw
shard hashes. Generation failures and partial answer shards are preserved and
never automatically resampled. An ambiguous old owner requires reconciliation.
Existing aggregate results are also revalidated against all eight rank receipts.

Independent local review and CPU lifecycle/canonical-negative fixtures are under
`i1i6_evidence/QA_FRESH_PROCESS_REVIEW.json`; they are not GPU acceptance. This
repair is deployed separately from frozen training and does not change a live
trainer's source.

Standalone milestone QA uses `qa_standalone` and `qa_standalone_control` on a
separately authorized H100 node. Its systemd cgroup has32 CPU cores/256GiB,
physical8GPU access and explicitly zero Slurm GPU reservation. It does not change
the trainer or other jobs. The memory/UUID guard stops only its own unit. The
original committed28 Arrow shards are read-only memory-mapped in original order;
full8GPU FP16 FAISS/E5 and BF16 TP1 inference run on the same node over loopback.
The extracted corpus is a same-NFS-inode hardlink; models/data/index never cross
Mac. Source config, immutable actor identity, fixed512 source and original corpus
ready receipt are bound separately. Constructor/partial-output rules above apply.
Native terminal cgroup emptiness and actual QA output are deployment acceptance,
not implied by the independent local CPU review.

The first standalone GPU run hit the own-memory guard during model loading:
all eight model workers had loaded on physicalGPU0 after CUDA-capable imports
ran before per-rank binding. No QA answers were generated. Its terminal/cgroup
empty/GPU-empty and cost evidence are retained. The repair binds the physical
rank before importing torch/vLLM/verl, then requires one visible CUDA device.
A new standalone namespace/unit performs actual QA; frozen training stays intact.

Isolated QA continuation after the I1–I6 repair:
- `qa_standalone_identity.py` binds fixed512 to saved5/50/100/150/200, or full51713 to actor200 in a distinct `-final-` namespace. Later checkpoints require the immutable outer `complete.json` SHA and its actual actor/probe20×outer per-rank boundary; a version label alone is insufficient.
- Build a new config from the already accepted QA5 template, retain canonical data hashes and the shared interaction contract, set the actual saved model/joint/source-evaluation receipt, and obtain fresh UUID/headroom evidence on the chosen authorized H100 node. All output stays on retained18 NFS; retrieval stays on the inference node's loopback. Do not use this as permission to stop or hot-edit native training3433.
- `qa_standalone_control.deployment_script(config, published_commit)` deploys a new code/config namespace; `submit_script(base)` records one submission intent and refuses replay. Resolve any unknown intent against its exact unit before further action. The repaired overlay includes the CPU raw-output verifier.
- `qa_standalone_observe --root ... --deployment <specific deployment receipt>` observes that exact unit/node. `qa_acceptance_cpu --base ... --expected-count 512|51713` requires the already complete8 rank receipts and aggregate; it recomputes raw strict scores without generation.
- Register accepted saved QA only after raw-output CPU reproduction and native unit/cgroup cleanup. Registry final entries additionally carry `expected_count=51713` and `independent_saved_raw_cpu_recompute=true`; final results do not resolve the distinct dev512 milestone200. Prepared support is not actual milestone/final acceptance.
