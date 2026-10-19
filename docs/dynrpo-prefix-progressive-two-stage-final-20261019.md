# DynRPO progressive Prefix / two-stage selection: final report, 2026-10-19

Training completed **200 full outer iterations**. Final seven-set macro strict EM is **43.3040%**, macro F1 **51.7888%**. Question-weighted overall EM is **47.8738%**, F1 **55.0934%**, with **24,757 / 51,713** strictly correct. Fixed512 at200 is **47.2656% EM /55.3801% F1**.

Job **733898** finished **COMPLETED,0:0**, including batch/extern and CPU audit steps0..7. All task-owned process births are gone, remaining-owned list is empty, entry lock is released. No foreign process was signalled. This report and the companion requirement index are the final publication deliverables; the publication receipt records their pushed commit and exact remote mirror SHA before the goal is closed.

## Experimental identity and authorized changes

- Run `cc0de51f2d68495c9664aa888074839c`; R4 execution `8185f59b57894bf9a4b8812e9f998699`.
- Final runtime `2000d46a530c64d78be7802bf11175a68b65effe`; immutable science source `c33aadcb42e6bd152ef284abc426ff4d56bfae0a`.
- Four physical H100 on ACD1-6, shared by actor/retrieval/grounding; partition `emergency_acd`, QoS normal,48CPU/512G. The later four-card instruction supersedes the old eight-card goal text. R4 elapsed1-02:57:07 includes evaluation; it is not the wall time of the entire multi-stage historical experiment.
- Student started from official Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Warmup reused epoch5 EMA model/buffers, SHA `0bf01eddd3cdaec02f91ce8a09d8c5d49d795428e634972791c99179c862d9aa`; online Adam started at0. No warmup was rerun and no historical online Adam was imported.
- Initial actual job729343 passed five complete outers plus positive6. Authorized CP50 continuation introduced longer Prefix/three epochs. R3 job733342 was explicitly stopped after complete61; qualified full CP61 restored into R4 job733898, which completed62..200 without replaying the student prefix.

This is an **evolving, user-authorized protocol**, not a fixed5120->1024->512 experiment from iteration0. Earlier dynamic budgets, later caps, and the two-stage switch must remain explicit in comparisons.

## Final selection and Prefix protocol

For ordinary outer>=62: **5120 distinct questions -> original five Prefix trajectories/question -> existing probe q Top1024 -> finish those same original G5 trajectories -> compute original strict EM -> prioritize512 true Mixed questions**. Mixed is exactly `0 < sum(EM_j) < 5`. Within class, q then original ordinal resolves ordering. If fewer than512 Mixed exist, keep every Mixed question and fill with highest-q nonMixed. No second sampling or relaxed scorer is used. At200 the selected batch contained137Mixed and375fills,26.7578125%Mixed.

The original feedback formula was actually exercised before the authorized replacement. The actual first-seven reservation audit verifies selected512 Mixed feedback, source joint SHA, distinct IDs, ceil-to-four and cumulative pool cursor. After50 the candidate upper bound was5120; from62 ordinary candidates are fixed5120. These historical fields may remain in backed-up launchconfig; `two_stage_selection` and actual selection receipts establish the effective later behavior.

| Student outer | Prefix | Probe schedule |
|---|---:|---|
|1..50|5|Original P10 one epoch; old completed50 probe reached250 writes/rank|
|51..100|20|Additional CP50 three-epoch fit250->400; then P10 three epochs|
|101..150|40|Post100 capacity expansion and three-epoch fit|
|151..200|80|Post150 capacity expansion; last fit190,200 saves/evaluates only|

Features are current actor layer27 post-block hidden from the actual original generation stream. Capture lookahead is discarded. Short responses are padded/masked only in probe features; no synthetic PPO action tokens are appended. Capacity changes preserve the transformer block, old readout slots, buffers, Adam moments/clocks and RNG through the approved zero expansion. Runtime migration checks, qualified post-fit payloads and actual next-outer use passed.

P10 uses2560 distinct questions xG5 and, from newCP50 onward,3epochs/150actual Adam writes per rank at LR5e-6. Student state is saved before fitting; probe state is inherited and frozen between P10 updates. Ordinary P10 data is collected from frozen actor(n-1) before actor n updates; boundary extra refresh50/100/150 uses saved actor n. Thus each feature/label pool belongs to one frozen actor snapshot, but not all pools are generated after PPO n. Final probe count is2500/rank and remains unchanged190..200.

Probe loss remains mean trajectory BCE plus equal-weight G5 Mixed BCE, with `q=1-prod(p)-prod(1-p)` evaluated stably through log-domain Bernoulli event masses. Good training fit does not establish good out-of-sample Mixed prediction: at190, train loss1.4066->0.7618 while diagnostic1.9565->1.9591. Diagnostic accuracy48.05%->50.78%; no generalization claim is made.

## Student objective and data

Actor uses512questions x5responses=2560trajectories, minibatch128, PPOepoch1 and20update opportunities/outer. Final actual Adam count is4000 per rank, scheduler200. Late audited outers have20real writes and0skips. Mathematical reduction preserves reference virtual8/micro2 equal masked-token means despite physical4/micro4 execution; it is not one aggregate token mean over the enlarged physical batch.

LR1e-6,57outer warmup then constant, clip0.2,DualClip3,gradclip1, entropy0 and referenceKL0 remain. Only all-five-correct groups receive a virtual zero in advantage statistics; no fake actor trajectory is added. BF16 compute is shared by train and QA, FP32 actor master/Adam retained. Strict parsing/scoring was not relaxed.

Common manifest `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`; train169028, fixeddev512, final51713 unchanged. Train parquet SHA `5d40180ba54d35513bb780408caa068d7d60123214a49b0739f835a2ae47c8a1`; final parquet SHA `55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1`. Existing heldout/dedup eligibility168510 is not a newly repartitioned dataset.

## Fixed512 evaluation

Canonical NQ256+Hotpot256 source SHA `3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb`. Every milestone has512 unique identities, original strict scores, BF16/common protocol and all four worker exits0. Rank inference clocks exclude initialization; full phase clocks include it.

| Outer | Strict correct | EM | F1 | Rank0/1/2/3 inference seconds | Full phase seconds |
|---|---:|---:|---:|---|---:|
| 5 | 141/512 | 27.5391% | 36.7957% | 11.124/10.463/10.471/11.496 | 90.940 |
| 50 | 223/512 | 43.5547% | 52.2269% | 8.834/8.827/7.970/9.569 | 80.481 |
| 100 | 237/512 | 46.2891% | 54.4009% | 10.293/10.343/10.143/10.664 | 86.878 |
| 150 | 249/512 | 48.6328% | 56.8096% | 9.532/9.857/9.512/9.453 | 80.240 |
| 200 | 242/512 | 47.2656% | 55.3801% | 10.619/10.899/10.922/10.109 | 103.097 |

The lower score at200 than150 is retained as measured; no checkpoint or scorer was selected retrospectively to hide it.

## Full final51713 evaluation at200

| Dataset | Questions | Strict correct | EM | F1 |
|---|---:|---:|---:|---:|
| NQ | 3610 | 1706 | 47.2576% | 55.5172% |
| TriviaQA | 11313 | 7069 | 62.4856% | 70.1421% |
| PopQA | 14267 | 6476 | 45.3915% | 50.0737% |
| HotpotQA | 7405 | 3173 | 42.8494% | 54.5833% |
| 2Wiki | 12576 | 5806 | 46.1673% | 52.5544% |
| MuSiQue | 2417 | 478 | 19.7766% | 28.6379% |
| Bamboogle | 125 | 49 | 39.2000% | 51.0133% |
|Seven-set macro|—|—|43.3040%|51.7888%|
|Question-weighted overall|51713|24757|47.8738%|55.0934%|

Actual rank inference seconds:1089.951/1087.576/1119.331/1118.445. Full phase1221.781s (20.36min), exit0. All four raw compressed output files were streamed and freshly SHA-verified against canonical51713 distinct identities/seven-set counts, reaggregating stored original strict EM/F1. This audit did not rerun generation or independently rescore every answer token. Stale every16-step progress counters are superseded by complete receipts and the verified full raw stream.

Macro and weighted metrics are different and must not be mixed in the user's seven-benchmark table. No all-baseline/SOTA claim or single-variable causal gain is supported by this evolving experiment.

## Checkpoints, restoration and retained locations

Final root:
`/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4`

Final HF model:
`/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4/actor/outers/outer-0200-attempt-d3eadfec8ccd440f82136d69c35cf237/huggingface`

Coupled final checkpoint entry:
`/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4/outers/outer-000200/complete.json`

Final evaluation entry:
`/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4/final-evaluation/complete.json`

Joint200 SHA `4ff93382532c98c8b05a5e2dc351795ca4d1c71cffb97ca51b60632df4de9fde`. Final probe SHA `cab9401fabf7bebd40a653917b7e45c0b0a514ec889c3c1bafee261077da9e97`,933674012B,14Adam states,4rank RNG,2500writes/rank.

Final200 independent CPU acceptance freshly hashed33files and loaded HF tensors, four-rank model/Adam/scheduler/RNG, driver/pool and probe/grounding. It took78.973s, peakRSS6834036736B, using the owned allocation's CPU-only step,2CPU/12G/no extraGPU. It did not perform a GPU forward or stopped-writer final200 recovery rehearsal. Actual HF200 GPU inference subsequently passed both fixed512 and full51713.

5/50/100/150/200 milestones retain33qualified files each. Their earlier independent full-SHA/CPU seals were revalidated by fresh inode/size/mtime/shared-fsid checks (165files), with fresh proof SHA. All40fixedQA output files were freshly SHA checked. CP50 additionally retains the inherited P20/400-step probe, full SHA `b3e74c1e5b17147ee3bb6080203940770218388783c499240f4da6b43adaa748`, paired with the unchanged original50 actor/model/driver entries. Original50joint SHA005be1... and post-P20joint SHA`cc6a8d6ea7bf979190781532701f3e9dc8a7b7041ea5dbfc44b7eb0cdae89ddc` are distinct legitimate stages.

Genuine stopped CP61 passed whole-joint SHA/CPU qualification before R4. All4ranks restored Adam1220/scheduler61/probe550; complete62..66 plus positive67 passed original-trajectory and real-update checks. This proves actual resume advancement without pretending that final200 was separately stopped and resumed again.

## I1–I6 and remaining requirement evidence

The companion JSON index records exact SHA/bytes for every local small audit artifact and the scope of each requirement. Historical `PREFINAL` or `full_goal_complete=false` fields remain immutable snapshots; later final acceptance closes their enumerated work without rewriting history.

- **I1:** shared wrapper/context4096/action500/observation500/reserve500, same train/QA functions; exact source equivalence and recorded stream tests plus actual QA contracts.
- **I2:** actual executed CPU loss/gradient-equivalence tests include unequal token counts and empty virtual ranks; exact tested source matches deployment. This proves the mathematical reduction, not native-GPU bitwise equality.
- **I3:** imported real round0 current manifest92f938... and actual heads, distinct from selection probe; real50/100/150 refits and all4replica adoption checked at51/101/151. Original round0 CV/features were not regenerated.
- **I4:** real before/after monitor invocation and healthy dispositions, thresholds targetKL0.2/negative-tail0.01/logbound30/DualClip3 unchanged.194 had40calls/rank (21fresh+19same-state reuse),42actual forwards/rank, KL0.000969..0.041403,0guardskips/rollbacks. Frozen32length-stratified samples are not a full2560 KL scan.200 again40calls/rank.
- **I5:** actual BF16 train/QA bindings, FP32master/Adam CPU states and successfulnativeHF200 inference.
- **I6:** first complete action in native chunk stream, original boundary token preservation; real101/151 prefix IDs/LP/masks/positions match their original G5 continuation and final projected actor input. Hidden padding tensors were not independently reread.

| ID | Requirement | Authoritative small evidence artifacts |
|---|---|---|
| identity | Latest authorized four shared H100/emergency_acd, isolated R4, actual job733898 lineage | `native-initial-five-acceptance.json`, `two-stage-five-actual-next-use.json`, `two-stage-final-owned-cleanup-and-slurm-acceptance.json` |
| initialization | Official Qwen revision, qualified EMA5 model/buffers; onlineAdam0 initially | `native-initial-five-acceptance.json`, `two-stage-external-entry-config-backup-publication.json` |
| budget-history | Historical feedback formula/distinct candidate cursor; authorized later caps and fixedtwo-stage replacement | `native-budget-reservation-chain-audit.json`, `progressive-cp50-first51-feature-budget-binding.json`, `two-stage-five-actual-next-use.json`, `two-stage-actual200-complete-ledgers.json` |
| selection | Currentactor onlineprefix, sameoriginalG5 continuation, trueMixed-priority512 | `two-stage-five-actual-next-use.json`, `two-stage-prefix-continuity-000101.json`, `two-stage-prefix-continuity-000151.json`, `two-stage-actual200-complete-ledgers.json` |
| prefix | P5->P20->P40->P80, maskedshortpadding, discardlookahead, retainblock/optimizer | `two-stage-final-cp50-progressive-probe-retention.json`, `two-stage-transition-000100-actual-next-complete.json`, `two-stage-transition-000150-actual-next-complete.json`, `two-stage-prefix-continuity-000151.json` |
| P10 | Every10through190; fromnewCP50 threeepochs/150writes/rank/LR5e-6;200nofit | `two-stage-prefinal-p10-history-consistency.json`, `two-stage-prefinal-p10-history-joints.json`, `two-stage-live-milestone-000200-v1-full-sha-cpu.json` |
| probe-state | Inheritblock/readout/Adam/RNG, savebeforefit, remoteactorfeatures/no actorgradient | `two-stage-final-cp50-progressive-probe-retention.json`, `two-stage-transition-000100-actual-next-complete.json`, `two-stage-transition-000150-actual-next-complete.json`, `two-stage-periodic-000190-actual-next-use.json` |
| loss | TrajectoryBCE+equalG5MixedBCE; nativeactor512G5/min128/ep1/LR/clip/zeroanchor | `two-stage-external-entry-config-backup-publication.json`, `two-stage-prefinal-i1-i6-evidence-index-v3.json`, `two-stage-actual200-complete-ledgers.json` |
| I1 | Sharedinformationwrapper/context4096/action500/obs500/reserve500 | `two-stage-prefinal-i1-i6-evidence-index-v3.json`, `two-stage-prefinal-existing-protocol-test-log-evidence.json`, `two-stage-final-milestone-retention-qa-lineage.json` |
| I2 | Physical4 loss+gradient equalreferencevirtual8/micro2 | `two-stage-prefinal-existing-gradient-test-log-evidence.json`, `two-stage-prefinal-i1-i6-evidence-index-v3.json` |
| I3 | Realround0; groundingrefits50/100/150adopt51/101/151 | `two-stage-final-initial-grounding-provenance.json`, `grounding50-independent-fit-and-adoption-acceptance.json`, `two-stage-transition-000100-actual-next-complete.json`, `two-stage-transition-000150-actual-next-complete.json` |
| I4 | Realbefore/afteroldpolicyKL/negative-tailmonitorandunchangedthresholds | `two-stage-prefinal-i4-actual194-dispositions.json`, `two-stage-actual200-complete-ledgers.json`, `logic-audit-tests-final.json` |
| I5 | Train/QABF16,FP32masterandAdam | `two-stage-live-milestone-000200-v1-full-sha-cpu.json`, `two-stage-final-milestone-retention-qa-lineage.json`, `two-stage-final51713-integrity-and-stored-strict-metrics.json` |
| I6 | Train/QAfirstcompleteactionnativeboundaryandoriginaltokens | `two-stage-prefinal-existing-protocol-test-log-evidence.json`, `two-stage-prefix-continuity-000101.json`, `two-stage-prefix-continuity-000151.json` |
| checkpoint | Completejointperouter; latest+5/50/100/150/200retention; fullmodel/Adam/scheduler/RNG/driver/pool/probe/grounding | `two-stage-final-milestone-retention-qa-lineage.json`, `two-stage-final-current-terminal-readback.json`, `two-stage-final-cp50-progressive-probe-retention.json`, `two-stage-live-milestone-000200-v1-full-sha-cpu.json` |
| recovery | ActualSHA/CPUqualifiedjointrestoreandnextupdates, no resetor prefixreplay | `two-stage-five-actual-next-use.json`, `two-stage-actual200-complete-ledgers.json`, `pending-recovery-cpu-tests-final.json` |
| data | Unchangedmanifest/train169028/dev512/final51713 | `two-stage-external-entry-config-backup-publication.json`, `two-stage-final-milestone-retention-qa-lineage.json`, `two-stage-final51713-integrity-and-stored-strict-metrics.json` |
| fixedQA | Actual5/50/100/150/200fixed512strictEM/F1/inferencetime | `two-stage-final-milestone-retention-qa-lineage.json` |
| finalQA | After200actualfull51713strictEM/F1/sevencoverage | `two-stage-final51713-integrity-and-stored-strict-metrics.json` |
| tests | Necessaryformula/gradient/period/budget/recoverytestsandactualnativeverification | `logic-audit-tests-final.json`, `adaptive-candidates-tests.json`, `pending-recovery-cpu-tests-final.json`, `two-stage-five-actual-next-use.json`, `two-stage-prefinal-existing-gradient-test-log-evidence.json` |
| source | Latestimplementation/config/entrycommittedpushed/hashverified | `two-stage-prefinal-runtime-all-source-fresh-sha.json`, `two-stage-external-entry-config-backup-publication.json` |
| advancement | Atleasttwoobservationsandfivefullouters+positivecontinuation | `native-initial-five-acceptance.json`, `two-stage-five-actual-next-use.json`, `two-stage-actual200-complete-ledgers.json` |
| storage-isolation | Remoteonlymodel/tensor/rollout/env; noMacproxy/modelrelay/foreignprocesschange | `two-stage-external-entry-config-backup-publication.json`, `two-stage-final-owned-cleanup-and-slurm-acceptance.json` |
| closure | 200complete,allscheduledfits/evals/retentionandownedresourcesreleased | `two-stage-final-current-terminal-readback.json`, `two-stage-final-owned-cleanup-and-slurm-acceptance.json` |

## Validation limits and publication

- Intermediate P140 canonical probe was legitimately pruned under latest+milestones retention before independent CPU reread; an alternate epoch serialization was not substituted. P140 fitting metadata/updates are accepted, canonical P140 CPU-readback is not claimed. Final190/200probe and required milestones are qualified.
- Prefix100/150 pre-fit old/new tensors were not independently compared again; approved runtime equality checks, post-fit CPU and actual next-use are the evidence.
- P10 history audit is fresh actual joint metadata, not a fresh tensor/epoch-ledger reread for all19periods. Required full milestone payloads were qualified separately.
- Previously executed78logic tests,85adaptive-budget tests and80pending-recovery tests are reused with explicit scope. Real firstfive/next, resumedfive/next, boundary transitions, P10fits and finalstate are additional runtime evidence; no claim that every fault branch was forced on GPUs.
- All model/checkpoint/rawoutput/env bytes remain remote. Local artifacts are code/config/docs/small scalar audits. No Mac proxy/model-data relay or foreign process mutation was introduced during final acceptance.

All1419 deployed runtime sourcefiles (19821686B) passed fresh SHA against original publication. Runtime2000d46 and sciencec33aad remain immutable. Before resource release, exact external sbatch/Python wrapper/launchconfig/resumeconfig/compatibility/authorization originals were backed up and pushed at `202b210a7e9b2acbbe9816f0495bdf1cffbb85c2`, branch `codex/prefix20-40-80-cp50-20261018`, with Gitblob/original SHA checks. Final report/index commit and shared-storage mirror SHA are recorded in the final publication receipt. No canceled automation is restored.
