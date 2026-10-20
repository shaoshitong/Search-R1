# Exact all252 G10 completed: joint improvement gate failed

Job **743417** completed **136/136 real probe updates**, Adam10464->10600. Slurm **COMPLETED 0:0**, elapsed00:29:02. No actor updates, optimizer reset, additional job, or production promotion. The full checkpoint was CPU-reloaded and component-validated before evaluation; an independent remote CPU audit reread its complete SHA.

## Same native validation protocol

All rows use the same2048 recipe-selection validation questions, precommitted label-blind realG5 IDs, own strictEM, and fixed threshold0.5. No threshold was fit to this validation. Its Mixed baseline is663/2048=32.373%. E4 is the current-probe reference; random4 is the matched E1/fullstate,6528-question,B48x136/order/update-budget control.

| Probe | Mixed BA | AUC | Top512 real Mixed | Mixed recall | NonMixed recall | Allwrong FPR | Allcorrect FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native E4 reference | 55.5291% | 57.6101% | 200/512 (39.0625%) | 66.3650% | 44.6931% | 49.8578% | 72.7273% |
| Matched G10 random4 | 55.1478% | 57.3218% | 194/512 (37.8906%) | 89.1403% | 21.1552% | 73.4597% | 96.0606% |
| Matched G10 exact all252 | 55.3908% | 57.4850% | 197/512 (38.4766%) | 88.6878% | 22.0939% | 72.5118% | 95.1515% |

All252 gains only **0.2431pp BA**, **0.1631pp AUC**, and **3 Top512 Mixed** over matched random4. All three remain below nativeE4. **The joint improvement gate failed; this is not a production replacement.** In this bounded matched experiment, removing Monte Carlo grouping noise did not solve the observed issue. This does not establish that every grouping scheme/data budget is ineffective.

All252 confusion: TP588/FN75/FP1079/TN306. NonMixed FPR77.9061%; **314/330 allcorrect** and **765/1055 allwrong** questions are called Mixed. Its Top512 contains **183 allwrong + 197 Mixed + 132 allcorrect**. The false-positive bias remains; lower training loss or trajectory accuracy is not treated as success.

The original512 diagnostic gives BA52.7822%/AUC52.9451%. No frozen confirmation/stability final pool was read. The2048 pool is recipe-selection validation already used for method development, not an untouched final test.

## Only grouping changed; actual qualification

- Four randomG5 views/question became allC(10,5)=252 uniformly averaged realG5 views, original-question totalweight1. No fabricated trajectories/labels.
- Retained E1 raw/EMA/Adam10464 and all4source-rank RNG archives; current1GPU uses canonical rank0. No4rank bitexact-resume claim.
- Same nativeL26/Prefix20,6528questions,originalB48x136 order,LR5e-6,trajectoryBCE1+three-eventNLL1,EMA and other settings. No within/inter/stat auxiliary term.
- The first real GPU Adam update was executed twice from the same fullstate; raw/EMA/Adam/two native buffers/RNG hashes matched exactly. FullCP1 freshSHA and further positive updates were accepted, then all136 were followed.
- Independent audit checked all136 consecutive cursors,Adam10464+cursor,all252/totalweight1,two-pass identical logits,and zero inter-view matrices.
- Actual update mean9.1647s,range3.4414-13.6740s. Real H100 peak allocated6655523328B,reserved8885633024B; these are actual process measurements, not synthetic estimates.
- Runtime48eb944281d2e3d6fa5ad66e4d6f0f9eaf18aeec; independent final-audit source693543a8d042e7296df8bdfcea9bbc61d56672b9. Finite observer exited; allocation released naturally. Source assets and checkpoints retained.

## Remote artifacts

Root: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g10-exact-uniform252-v1`. Model/Adam/features/raw predictions remain remote; local files contain only code and small audits.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `complete.json` | 19592 | `dda15062827abf68c69b5ca8d28a6d82b4aeea15d1f9b30e18b25e588e7aa76f` |
| `checkpoints/cursor-000136.pt` | 3745829722 | `e2a60561bdb1e39c6f5352cf6e93ecb59182537eb26c40bc5f1bbe7b49dda6bb` |
| `independent-final-audit.json` | 14059 | `e7c8d8f40013fb55a31e97e1866cc5fba3e3cdccb38043711c39060db761b418` |
| `predictions/selection2048FreshG5.jsonl.gz` | 149334 | `7362b873f0171345395b7df18b102ef61de89b0731ecff6791224906ccd42a1f` |
| `predictions/diagnostic512.jsonl.gz` | 38648 | `a27cc1ec7018785a5c72456a9ff4de258588175e088aa0cf00890f8930bb2a3b` |

Small metrics: `actual-final-independent-audit.json`; terminal state: `actual-observer/latest-good.json`. Independent stdlib metrics verified gzip readability,ownEM/fixed5 IDs,plan binding,save-before-evaluation,native backend,and predictionSHA. No G5 reselection,external threshold fitting,or retraining occurred.
