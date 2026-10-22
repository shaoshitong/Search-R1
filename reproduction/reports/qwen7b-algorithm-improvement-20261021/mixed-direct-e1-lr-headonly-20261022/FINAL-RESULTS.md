# Three-epoch phased continuation results

Job747723 completed0:0 in3:43:10. E1 full state was retained; new E2 native+head LR5e-7; new E3 raw native frozen/headLR5e-5. Final actual clocks native3898/head5219 and accepted raw-native hash unchanged during head-only phase. Full checkpoint5219 is retained and CPU reload accepted. All169028 source questions cumulatively seen, not a guarantee of unique full pass every epoch.

| Snapshot | Fixed2048 EMA MixedBCE | MixedBA | MixedTPR | nonMixedTNR | Top512 Mixed |
|---|---:|---:|---:|---:|---:|
| E1 retained | .653016 |52.8846%|21.8703%|83.8989%|205/512=40.0391%|
| E2 LR5e-7 | .946060 |48.7836%|90.3469%|7.2202%|176/512=34.3750%|
| E3 nativefrozen/head5e-5 | .626374 |50.8674%|5.1282%|96.6065%|194/512=37.8906%|

The proposed schedule did not exceed retainedE1 selection. E2 still predicts Mixed for92.78% of nonMixed; E3 misses94.87% of trueMixed. The lower E3 totalBCE does not establish Mixed discrimination improvement. Final4096 remains untouched. OwnG5 prediction CPU recomputation is separately in progress, with no new GPU scoring or new training submitted.
