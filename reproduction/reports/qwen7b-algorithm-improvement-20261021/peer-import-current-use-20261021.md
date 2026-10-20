# Peer data is imported and used — 2026-10-21 07:44 CST

The human-authorized source is thread `01a157ec-c811-7c51-b270-db2042410464`, official Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`, independent source seed20261021. The first frozen snapshot is **31,232 NQ questions /156,160 new genuine trajectories**. All16 archives and3466 members were accepted by source/HF/remote receiver and a separate fresh HPC3 SHA readback. Raw destination is `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/peer-newseed-h100-20261021/newseed-20261021-185747-445fc691/raw`. Target complete SHA: `8a0f5128f7f613d4f6ee9cc5d6135f461f66feb66e73b65bbfa5e78ada875d5d`. No payload was stored on the Mac or management-host disk.

CPU ingest paired each question with its original five responses, giving312,320 genuine old5+new5 trajectories, and recomputed strict EM with zero mismatches. This source subset is NQ-only; the original warmup already used all169,028 canonical training questions, so it is not an unseen-question corpus. Four completed normal-priority feature jobs captured only the21824 newly selected questions; old8192 features were reused. Frozen identity-only training membership contains30016 questions /300160 G10 trajectories. Data and all model/optimizer/feature payloads remain remote.

## Actual training and qualified partial result

Independent append job743517 inherits the complete prior peer CP256 and Adam10720, preserving the first256 actual question batches. It adds682 new question batches. Two fresh root observations at epochs1792539812.837430 and1792539896.447957 showed **683→696 complete updates /938**, **Adam11147→11160**, latest complete checkpoint **672→688**, and no failure record. Slurm RUNNING is supporting evidence only; the actual update and checkpoint increments establish progress. FullCP688 SHA: `382387fc74327c2372f906beb13c6a35ae9372765e97745d7cc59b785aadad76`. The first independent full-state reload and continuous five-plus-next transactions were already accepted. No actor/student update or active-source hotpatch occurred.

The independently accepted checkpoint512 validation uses the identical fixed-HF2048 G5 identities and backend as its E2/E4 comparisons. It is a partial result, not the final938 result:

| Recipe | Mixed BA | Mixed recall | nonMixed recall | allwrong false-Mixed | allcorrect false-Mixed | AUC | Mixed@512 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Append checkpoint512 | 55.6880% | 70.4374% | 40.9386% | 50.9005% | 85.1515% | 0.56660 | 185/512 (36.1328%) |
| Same-HF E4 | 55.7714% | 67.5716% | 43.9711% | 51.1848% | 71.5152% | 0.57648 | 196/512 (38.2813%) |

The stronger fair BA frontier is E2 at56.2505%. The candidate has not jointly improved BA and selection. Its selected K0..5 composition is `[197,44,30,48,63,130]`. Independent paired2000-bootstrap/four-ID-strata analysis also found no consistent advantage; pointwise BA/selection CIs overlap zero. These CIs are descriptive, not a multiplicity-corrected generalization claim. Training continues to768/938 without altering this frozen dataset.

## Completed prefix-confidence check is negative

The official actor's actual chosen-token logprobs were aligned to81920 real Prefix20 trajectories, using only already generated retained policy tokens. A fixed ID-only6144/2048 internal split fitted one eight-scalar confidence extension and its two-scalar calibration control. This uses **new-seed five-response internal holdout**, distinct from the external fixed-HF2048 recipe pool above. Both fits completed18.59seconds/exit0; independent small auxiliary checkpoint, optimizer, RNG, split, fit-only normalizer and own-EM probability/metric checks passed.

| Internal newseed5 holdout | Mixed BA | Mixed@512 |
| :-- | --: | --: |
| Frozen raw E4 | 64.0519% | 238/512 |
| Matched two-scalar control | 63.4370% | 229/512 |
| Prefix confidence full8 | 63.3607% | 232/512 |

The fit loss fell2.06695→1.93435, but full8 increased allcorrect false-Mixed errors68.69%→91.69%. It did not pass the preregistered internal gate, so no external candidate or untouched-final score was run. Independent terminal audit SHA: `f77c10b3868ad895c799574b06cd47191e4968fad8cf90f0d141860249b6cdd6`; auditor source `39d5af483d5d25871f500fd09987d2a9a5f69dbb`. Real learned scalars, normalizers, optimizer state, probabilities and rows remain remote.

## Further source data

Two read-only metadata passes found18432 further sealed questions, including an existing-audited subset of15616 (12300 NQ +3316 HotpotQA). This delta is **preparation only** at this report time: no archive/member SHA acceptance, ingest, feature capture or training is claimed. Its staging will be separate from the running append dataset. Source production and foreign jobs remain untouched.

A direct six-count head from the actual five trajectory hidden embeddings is being prepared, with separate explicit new Adam and inherited legacy block/head/Adam. It has not been submitted to GPU. Whole scientific goal remains active; there is no jointly improved recipe and the untouched4096 final pool remains unread.
