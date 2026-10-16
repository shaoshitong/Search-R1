# G5 mixed decision: balanced accuracy for all70 replay iterations

Balanced accuracy is `0.5*(TP/(TP+FN)+TN/(TN+FP))`: the mixed and nonmixed class recalls receive equal weight. This is a posthoc evaluation metric, not balanced sampling or a training change. The decision threshold remains `q>=0.5`. No model inference or training was rerun.

Every source metric SHA matches the previously published report provenance. The fixed test contains183 mixed and329 nonmixed G5 groups. Always predicting nonmixed has balanced accuracy50%.

| Head | Own-batch train balanced accuracy | Fixed-test balanced accuracy |
|---|---:|---:|
| Final replay70 | 73.9199% | 60.9107% |
| Highest fixed-test, replay30 | 66.9282% | 64.9410% |

Final fixed-test mixed recall is95/183=51.91256831%; nonmixed recall is230/329=69.90881459%. Their equal-weight mean is60.91069145%. Final train mixed recall is1253/1846=67.87648971%; nonmixed recall is2618/3274=79.96334759%, giving73.91991865%.

Train uses each iteration own5120G5 batch evaluated after its updates; these batches differ. It is not a fixed-training-set curve. Train evaluation was not recorded for iterations1..6 and remains missing. Test is the unchanged fixed512G5 cache.

Ordinary accuracy below the majority-class baseline does not imply absence of discrimination. Here the final balanced test accuracy exceeds the constant-class50% baseline; this is a statement about the fixed diagnostic, not a demonstrated policy-selection or QA improvement.

[Exact small balanced summary](BALANCED-ACCURACY-SUMMARY.json)

| Replay iteration | Own-batch train balanced accuracy | Fixed-test balanced accuracy |
|---:|---:|---:|
| 1 | not recorded | 62.8357% |
| 2 | not recorded | 59.8452% |
| 3 | not recorded | 62.4412% |
| 4 | not recorded | 63.7849% |
| 5 | not recorded | 61.8948% |
| 6 | not recorded | 62.5982% |
| 7 | 69.1119% | 62.3300% |
| 8 | 64.1984% | 62.4720% |
| 9 | 66.0343% | 63.7816% |
| 10 | 65.7350% | 60.3285% |
| 11 | 64.6674% | 62.9312% |
| 12 | 62.1979% | 59.9482% |
| 13 | 64.3248% | 64.2957% |
| 14 | 66.3945% | 62.9022% |
| 15 | 65.9493% | 60.9074% |
| 16 | 64.4470% | 62.8374% |
| 17 | 64.4257% | 64.7225% |
| 18 | 65.5539% | 64.3605% |
| 19 | 65.6677% | 64.8131% |
| 20 | 62.7714% | 61.4355% |
| 21 | 66.3467% | 64.0324% |
| 22 | 66.0905% | 63.4162% |
| 23 | 63.7369% | 62.5301% |
| 24 | 67.2918% | 63.1496% |
| 25 | 66.2729% | 62.3623% |
| 26 | 66.4867% | 62.6323% |
| 27 | 66.2388% | 60.4224% |
| 28 | 66.1660% | 62.3233% |
| 29 | 67.5603% | 62.9345% |
| 30 | 66.9282% | 64.9410% |
| 31 | 64.1142% | 63.2285% |
| 32 | 67.0505% | 64.0308% |
| 33 | 67.1388% | 64.4834% |
| 34 | 65.1951% | 63.4436% |
| 35 | 62.9826% | 59.6409% |
| 36 | 65.1507% | 62.3474% |
| 37 | 69.4376% | 63.7575% |
| 38 | 66.7007% | 62.5044% |
| 39 | 65.4046% | 63.9884% |
| 40 | 68.5696% | 63.2044% |
| 41 | 67.7688% | 63.2966% |
| 42 | 69.2886% | 62.7518% |
| 43 | 69.5495% | 63.7575% |
| 44 | 69.3892% | 61.5120% |
| 45 | 69.4704% | 64.2392% |
| 46 | 66.0100% | 64.0482% |
| 47 | 69.5002% | 60.9672% |
| 48 | 70.3276% | 64.6013% |
| 49 | 67.4832% | 62.4703% |
| 50 | 69.0549% | 62.1473% |
| 51 | 67.8378% | 63.7202% |
| 52 | 70.4574% | 60.7280% |
| 53 | 70.8852% | 60.3900% |
| 54 | 70.2359% | 62.1182% |
| 55 | 71.6130% | 61.1482% |
| 56 | 70.9110% | 63.1820% |
| 57 | 68.5738% | 63.5067% |
| 58 | 69.5278% | 62.6048% |
| 59 | 67.0849% | 63.2626% |
| 60 | 70.6250% | 63.3016% |
| 61 | 71.0773% | 63.0865% |
| 62 | 70.2898% | 63.3274% |
| 63 | 68.1100% | 59.8809% |
| 64 | 71.8423% | 63.8190% |
| 65 | 71.5670% | 61.0286% |
| 66 | 70.9893% | 63.0541% |
| 67 | 73.5196% | 61.9953% |
| 68 | 75.2046% | 59.3635% |
| 69 | 74.8313% | 61.9646% |
| 70 | 73.9199% | 60.9107% |
