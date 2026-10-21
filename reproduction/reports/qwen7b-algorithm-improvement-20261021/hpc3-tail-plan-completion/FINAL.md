The two original tail jobs 745518/745519 completed with exit 0:0 and released their worker locks: 241 batches, 30,754 questions, 153,770 authentic new trajectories. No GPU was kept alive or restarted. All raw files and receipts remain remote.

The three scopes (9,344 + 14,592 + 6,818 questions) passed full old5/new5 payload SHA, own-ID, Prefix20-to-full causal, native-seed and tokenizer strict-EM checks. The final scopes recomputed 145,920 and 68,180 trajectories with zero mismatch or same-question old/new seed collision. The last partial batch of 34 questions is retained. The final CPU workers used one CPU2/GPU0 step at a time inside the existing owned 745501 allocation, adding no GPU.

Qualified whole-G5 source selection gives 124,194 existing questions + 14,080 donor-only + 30,754 HPC3 tail = exactly canonical 169,028 questions / 1,690,280 trajectories. Missing and extraneous qids: 0. Own trajectory IDs are unique; all old5 IDs and EM match canonical. The 3,584 donor/HPC3 overlap questions use one qualified complete HPC3 execution; no trajectory mixing or premature packed-feature union.

| Metadata scope | NQ | HotpotQA | Questions |
|---|---:|---:|---:|
| Current main | 78,871 | 45,323 | 124,194 |
| All donor | 0 | 17,664 | 17,664 |
| Selected donor-only | 0 | 14,080 | 14,080 |
| Complete HPC3 tail | 0 | 30,754 | 30,754 |
| Canonical / final chosen union | 78,871 | 90,157 | 169,028 |

Canonical and chosen union each contain 169,022 distinct question hashes: six repeated hashes, each representing two authentic qids (12 questions / six extra rows). They are retained. Only identity metadata from the true 512 + 2048 + 4096 evaluation exclusions was used; qid and hash overlaps are zero. No evaluation labels/outcomes were read. Texts were already seen during original warmup, so these are not unseen-text questions.

The final metadata proof took 5.51 seconds / 806 MiB, checked exact qid-set SHA plus 166 final-scope audit receipts and released locks. It inherited completed raw payload acceptance, without repeating bulk response hashes or loading a model. Full remote descriptors are in FINAL.json and actual-full169028-qualified-raw-union.json.

This establishes complete raw G10, not complete L26 features or training. The current 124,194-question training source, configuration, data and evaluation plan are unchanged.
