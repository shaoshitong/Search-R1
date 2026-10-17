# H100 Prefix5 A/B actor200 — approved execution protocol

This run implements the activated thread goal, after the preparation-only reference audit. The common baseline is staged `7679bb8ced2c13922fa9f5c9050ed0ce7fa54dde` and its published preparation/audit `8bd6a7661cc904b8f4e938df80807fed2c3f9420`. No old actor or online optimizer is imported. Both actors start from the same official Qwen2.5-3B-Instruct revision; selection heads start from exactly the same retained warmup epoch5 EMA model-only export (SHA `06ffd8c21b6208ff5a0beb11cded62da7b9551c3e1357a1d7321e69eb06d994e`). No new warmup.

## Preserved common science

Each arm has four physical ranks, colocated exact-index FAISS/E5 and grounding services, independent UUID leases, run directories, ports, pool/optimizer/RNG states. A uses0–3; B uses4–7. Foreign tasks are preserved. No new Slurm/GRES or global policy changes.

Current sampling actor question+5 policy token index27 postblock hidden is used, preserving actual IDs/masks/positions. The sixth sampled token is discarded after processing token5. The copied native block35 and flattened Linear remain unchanged. Rank score is h=1-product(p)-product(1-p); deterministic ties and original prefix continuation remain unchanged. No independent encoder forward.

Candidates: outer1–5=2560; ordinary6+=5120; periodic10/20/...190=2560 complete G5 with original actual-mixed-first/probe-score-fill actor selection. Outer200 is ordinary. Actor always512G5 with original Search-G1 reward/PPO, 20 update opportunities/rank, original virtual8/micro2 equal-mean objective on four physical ranks, global mini128, LR1e-6/57-complete-outer warmup, original finite-state safeguards and actual-write accounting. No data augmentation/new reward/loss changes to actor.

Boundary/commit grounding probes retain qualified round0 weights and actual50/100/150 refits/adoption. They are independent from the selection heads. Unified new BF16 interaction/QA protocol and fixed512 QA at5/50/100/150/200 plus final51713 remain unchanged. No diagnostic question enters candidates.

## A — frozen warmup

No selection-probe optimizer, parameter update or online EMA. Original periodic candidate completion/selection and diagnosis still run. Probe actual Adam stays0. Actor and grounding updates remain enabled.

## B — current-outer balanced raw training, EMA selection

Raw and EMA start equal. Only raw receives Adam updates at constant5e-6. After each actual Adam write, all floating EMA state follows0.99*EMA+0.01*raw; integer state is copied. Nonpersistent native buffers are retained. EMA selects and its pre/post evaluation is primary; raw paired metrics are also recorded.

Ordinary data are this outer's selected512G5; periodic data are this outer's entire2560G5. No historical data or extra rollout. Group labels are mixed iff0<K<5, includingK1. Resample whole G5 within current classes with replacement to256+256 ordinary groups or1280+1280 periodic groups. One pass over this sequence gives10 or50 actual synchronized writes; mean global batch256 (255/260), original true-global DDP weighting. If a class is absent, record a skip, preserve head/EMA and keep actor progressing. Final200 gets ordinary probe training. Maximum probe/EMA writes per rank=2760, reduced only by recorded missing-class skips.

Loss = mean own-trajectory BCEWithLogits +5*mean tied mixed-event BCE. Same32-pattern log-domain computation; no clamp/detach, no independent head, DIST, mixup, auxiliary KL or new soft labels. Threshold remains0.5.

Full unresampled current data are evaluated with raw/EMA before and after fitting. Periodic fixed512 questions receive fresh G5/hidden/labels from the same actor; paired metrics share those trajectories and never train. Record ordinary/balanced accuracy, both recalls and loss terms. Balanced-sample metrics do not substitute for original-distribution metrics. Uncontinued ordinary candidates have no claimed accuracy.

## State and verification

Every joint commits actor/masterAdam/scheduler/RNG/pool and selection head state. B also saves EMA/Adam/actual EMA clock and complete sampling indices, multiplicities and RNG. Sampling metadata have separate hashes; large state stays remote. Normal5/50/100/150/200 state readback uses CPU-only `ab_checkpoint_once`; no GPU gate or repeated completed large reads. Same-run recovery preserves latest genuine complete joint and actual clocks, never rolls back for an old hardcoded helper. Original code remains immutable once started.

Implementation: `ab_policy.py`, `ab_probe.py`, `ab_lifecycle.py`, `ab_checkpoint_once.py`, `ab_control.py`; explicit opt-in integration in probe/launch/storage and same-run recovery readers. Original reference mode remains available unchanged unless AB config is explicitly present.

Focused CPU validation: 37 tests passed for resampling/wholeG5/DDP equivalence, EMA state and integer/nonpersistent buffers, weighted mixed loss/K1 target, actual skip clocks, A/B joint commits including outer200, original actor recipe, candidate selection, grounding lifecycle, interaction protocol and recovery. Native GPU progression remains an execution acceptance item, not a CPU-test claim.
