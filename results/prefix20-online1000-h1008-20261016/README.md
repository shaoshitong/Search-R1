# Prefix20 online1000: five-fresh-epoch initial acceptance

Job3408 on h100-8, run8a6b24dc19664cb3a3fb97101f21c54e, actually completed fresh epochs2–6 and started real generation on all eight ranks for epoch7. The executed runtime remains frozen at ca1c60f49a29c30ca299c1d6073ee9a4bb86655b. The same job continues toward1000 epochs; this initial acceptance does not finish the goal or request a stop.

Each fresh epoch completed5120 distinct questions × G5 =25600 complete trajectories. Head training uses global batch256, eight ranks × local32, no accumulation, one natural without-replacement pass:100 synchronized actual Adam updates per rank per epoch. Original ledgers prove101–200,201–300,301–400,401–500 and501–600. Across these five fresh epochs this is500 additional writes per rank, cumulative600 per rank; rank counters are not summed.

The five fresh selections contain25600 distinct question IDs and normalized question identities, without overlap across these five epochs or with cached epoch1. Direct intersection with the fixed512 diagnostic question IDs is zero. The normalized diagnostic-exclusion binding is consistent and enforced by the frozen pool initializer; this audit does not independently reread every canonical question text. All fixed diagnostics retain the same manifest and2560 trajectories (559 correct,2001 incorrect), without regenerating diagnostic trajectories.

## Epoch1–6 metrics

Epoch1 is retained historical cached data, not one of the five fresh acceptance epochs. Training metrics are online predictions before each own minibatch update, not a separate evaluation of the final epoch model over all training data. Diagnostic metrics below are actual per-epoch measurements, not best-epoch selection or proof of probability-quality success.

| Epoch | Source | Adam/rank cumulative | Online train BCE | Fixed diagnostic BCE | Accuracy | Balanced accuracy | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | cached original5120 | 100 | 1.787934 | 0.586365 | 73.6719% | 55.6349% | 0.628386 |
| 2 | fresh5120 | 200 | 0.593701 | 0.578590 | 74.5703% | 55.3072% | 0.622379 |
| 3 | fresh5120 | 300 | 0.559183 | 0.686397 | 63.5938% | 57.3098% | 0.623538 |
| 4 | fresh5120 | 400 | 0.548148 | 0.590366 | 72.9297% | 53.9999% | 0.611562 |
| 5 | fresh5120 | 500 | 0.535492 | 0.544847 | 76.6016% | 53.8993% | 0.644028 |
| 6 | fresh5120 | 600 | 0.527516 | 0.558564 | 77.3047% | 53.1888% | 0.655170 |

## Actual phase costs, fresh epochs2–6

The following are console stage wall intervals, including startup, polling, shutdown and publication. They must not be confused with the internal update-loop or diagnostic-loop times.

- Fresh generation: 391.242–401.680 seconds.
- Prefix packing: 6.615–6.820 seconds.
- Frozen feature encoding: 28.426–28.739 seconds.
- Complete training phase: 28.585–28.801 seconds.

The update loop alone took4.811–4.954s for100 actual Adam writes/rank; the fixed diagnostic loop took0.134–0.152s.

## Evidence and retention boundaries

`five-fresh-actual-facts.json` contains the per-epoch eight-rank ledgers, small complete/metric/question/selection/pool-boundary SHA readbacks, phase times and observed epoch7 progression. `epoch2-actual-facts.json` retains the earlier single-fresh-epoch observation. Questions, gold answers and compactprefix20 payloads remain at their recorded remote paths for reuse.

Large compact/full rollout payloads were only checked for existence and byte count against producer receipts. Their hashes are producer-reported, not independent payload readbacks. Model checkpoint hashes are likewise historical producer receipt values; mutable latest.pt may now represent a later epoch. No model was independently hashed, loaded on CPU, or semantically reloaded by this audit. No question text, tokens, model/tensor or raw rollout payload was downloaded to Mac.

The archived collector was consumed once and is code backup, not an instruction to replay blindly. This publication changes only small facts, documentation and archival audit code; runtime, job, state and automation remain unchanged.
