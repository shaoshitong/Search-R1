# Search-G1 prefix selection, formal actor200

Authorized 2026-10-16. Run `5cba22ca409a4571a1e6b87d89c86b09` on h100-8.
Remote root: `/mnt/data/shaoshitong/search-g1-prefix-selection200-20261016-5cba22ca409a`.

Initialize the correctness probe **model only** from retained warmup CP100:
`8df6bd445945ffbbf6a94171f24934495f0ce3ff17a3d2577bbd53d4b8fefd56`.
This was the best actually retained checkpoint on fixed512 mixed balanced
accuracy (0.598759280482336); epoch45 had a higher metric but no retained weight.
Online probe Adam starts at zero. Actor starts from official
Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.

Each outer reserves5120 distinct questions, generates5 actual policy20 prefixes,
scores them with the fixed official layer27 encoder plus question20 probe,
selects stable Top512 (no exploration), and continues those exact2560 prefixes.
Tools retain their actual context and positions. Natural early EOS stays short;
a zero-retained response remains in its G5 and contributes no fabricated PPO token.
Unselected candidates have no outcome labels and no claimed classification accuracy.

The training sampler pauses at the twentieth retained policy token and stops each
logical action at the first complete action actually seen in its native token
stream. It resumes the same token IDs and records selected native log probabilities.
This new training stream is explicitly not claimed bitwise equivalent to generating
all500 tokens before cutting an action. QA evaluation separately retains the
confirmed pre-d00 raw observation500/no answer reserve/no early-final protocol,
FP32 greedy inference and asynchronous eight-TP1 execution.

Actor updates use the original grounding flip/boundary soft reward, separate from
the new correctness head, GRPO own-G5 advantages, mini128, one epoch,20 actual
Adam writes/rank. Preserve the earlier outer scheduler clock: completed0 uses
LR0, completed57 uses1e-6; outer58 onwards is constant. No reference forward is
required when reference KL coefficient is zero. FP32 master/Adam and BF16 FA2
forward are retained, including explicit packed sequence boundaries.

All eight cards participate in actor and probe phases. Initial actor micro128
(local16) has no accumulation. A **real typed CUDA OOM** ends that attempt;
the controller records abandoned unsaved writes, restores this run's preceding
complete actor state, and retries the same cached rollout with micro halved.
No partial writes are stitched into the continuous clock. Probe has ten whole-G5
steps (eight255-trajectory batches and two260), local30/35 with exact DDP weights,
one natural pass, constant LR1e-6. Its loss is trajectory BCE plus own-G5 mixed
BCE, weight1, mixed event0<K<5. Global weighted losses are recorded at each of
the ten updates; different batches need not show monotone loss.

Inference starts at1024 concurrent sequences/card and32768 batched tokens,
using actual free GPU memory minus2GiB, capped at0.95 total-card utilization.
True initialization OOM reduces cache/concurrency. Frozen encoder scoring starts
at micro256/card. The actual batch limits, memory and utilization are evidence,
not a promise that all phases or the entire pipeline remain at100% utilization.

The original grounding reward is served by eight rank-local replicas with separate
GPU locks. Full E5/FAISS retrieval stays local across eight GPU shards. No foreign
task is stopped. Native Slurm uses eight H100s,96 CPU and512GiB RAM.

Each joint outer commits actor HF/master Adam/scheduler/RNG, probe Adam/eight RNG,
candidate permutation/cursor and token/label records. Milestones5/50/100/150/200
are retained. Only consumed derived features and unreferenced nonmilestone rotating
states from **this run** are pruned, with producer hash deletion receipts. Original
source caches, old experiments, trajectories, questions and metrics are retained.
All large assets remain remote; Mac contains code/config/docs/small summaries only.

QA validation is fixed512 (256NQ+256Hotpot), at5/50/100/150/200; final QA is all
51713 across seven datasets. Completion requires actual200, QA counts, milestone
state validation and retention, native exit0 and owned-resource cleanup. Submission
or finite updates alone do not establish model quality or completion.

Implementation tests:41 focused CPU tests covering pool reservations, same-prefix
continuity across tool turns, native LP identity, exact group math/DDP weighting,
zero-retained responses, OOM recovery, legacy evaluation and owned storage pruning.
These do not claim actual GPU restore/forward success; production progress is
verified separately after launch. No extra GPU test or pretraining baseline gate.
