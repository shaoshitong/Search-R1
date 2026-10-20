# Peer data import and current evidence — 2026-10-21 07:03 CST

The requested peer snapshot was copied through the authorized remote route and verified: 31,232 NQ questions, 156,160 genuine additional trajectories; old5 + new5 form 312,320 real trajectories. Source files and peer jobs were preserved. No model, raw rollout, feature tensor or optimizer payload passed through the Mac.

After question identity/text-hash exclusion and full B32 batches, the frozen fit cohort has 30,016 questions and 300,160 real trajectories. The existing 8,192 questions were preserved exactly; 21,824 additional experiment questions were captured. These are not unseen questions from the original native169,028-question warmup. All four capture jobs completed normally and the unified bank has independent input/ID/EM acceptance. Tensor payloads have producer SHA receipts; the actual trainer additionally rehashed the full feature manifests before its first update.

The original no-ranking CP256 is the predeclared continuation source. New job743517 uses the same trajectory1 + three-event1 loss, Prefix20, block27, and inherited raw model/EMA/Adam/RNG. It appends only682 B32 updates; the old256 updates are not repeated. Actual CP257 passed double replay and a separate CPU full-SHA/six-component reload. Six complete new updates and a later progressing update were independently accepted. Root measured286 then310, Adam10750 then10774; no failure. Required saved-checkpoint-first fixed-HF evaluations are cumulative512/768/938. Current startup success is not evidence of improved selection.

## Fair fixed-HF2048 recipe-validation results

This fixed G5 pool contains663 real Mixed questions (32.37%). Same backend and exact same five real trajectory identities/EM are required for comparison. The reference frontier is E2 BA56.2505% and E4 Mixed196/512=38.28125%; neither validation nor frontier is untouched final confirmation.

| Recipe | Mixed BA | Mixed selected /512 |
|---|---:|---:|
| E2 baseline |56.2505%|181|
| E4 baseline |55.7714%|196|
| Old8192, trajectory + three-event, CP256 |54.6555%|182|
| Matched within-question rank weight1 |54.7012%|184|
| Matched inter-question rank weight1 |55.4754%|189|
| Matched both ranking weight1 |55.0855%|186|

All three matched ranking variants completed256 updates with full-state checkpoint acceptance. None jointly exceeds the frontier. They were not added to the new-data continuation. Additional data and additional optimization budget change together in the682-step continuation; its result alone will not isolate their causal contributions.

## Frozen readout diagnosis

Actual production p/q for both E4 and peerCP256 exactly reproduce all sealed prediction rows. The fixed linear head was decomposed after block27 into question terms (including original bias) and Prefix20 terms (no bias). Question terms have exactly zero variance within a question. Prefix-only within-question correct/wrong AUC is only0.53325 (E4) and0.53576 (peerCP256). E4 Mixed@512 is196 with the full head,177 with the question term,143 with the prefix term; peer values are182/176/149. Simply deleting the question term worsens selection. Prefix hidden states already contain question context, so this does not remove question conditioning or prove an information-theoretic limit. FP32 additive reconstruction and BF16 production rounding were audited separately.

An exact G20-all15504-subsets branch has completed all136 updates and saved CP136; final native-backend evaluation and independent acceptance are pending. Its native results must not be compared as if they were the HF numbers above.

A CPU-only causal Prefix5/20/40/80 collision study is being prepared on the same real8192 G10 cohort. P20 must reconstruct the accepted tokens, positions and causal contexts exactly. Longer prefixes may use only actually generated kept policy tokens and observations already available at that cutoff. No active training setting is changed. Input uniqueness by itself will not be described as learnability.

No candidate has yet passed the joint goal. Untouched final4096 remains unread. No actor training was started by these probe-only experiments.
