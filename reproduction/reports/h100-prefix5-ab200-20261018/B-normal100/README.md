# B normal100: actual QA, joint state and native40-token transition

Both current four-GPU arms run on h100-11. Frozen training sources remain A2e397623 and Bc5f1a07e. These are actual intermediate results, not completion of the two200-outer/final51713 objectives.

## Same fixed512 QA / same new BF16 contract

| Actual actor100 | Correct /512 | QA EM | QA F1 |
|---|---:|---:|---:|
| A | 215 | 41.9922% | 50.2885% |
| B | 232 | 45.3125% | 54.0760% |

B answers17 more questions correctly than A at100: +3.3203 percentage points EM and +3.7875 points F1. Canonical question bytes, every qid/ordinal/source, original worker receipt SHA and score aggregation were checked; all four actual B workers exited0 with128 rows each. Source question SHA is `3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb`.

| Source | A EM | B EM | A F1 | B F1 |
|---|---:|---:|---:|---:|
| nq (256 each) | 41.7969% | 40.6250% | 53.4212% | 51.8006% |
| hotpotqa (256 each) | 42.1875% | 50.0000% | 47.1559% | 56.3514% |

The original B50 QA was221/512 EM43.1641%, F151.3998%; an already-existing recovered B50 execution was220/512 EM42.9688%, F151.6048%. Their previously recorded execution reproducibility difference remains unresolved despite identical identities/protocol/source. Both values remain visible; no new QA50 was generated. B100 improves by11 or12 correct questions relative to those existing executions.

Post50 B also changes prefix length and expanded-admission/outcome-assisted reranking, in addition to raw/EMA fitting. The observed A/B difference cannot isolate an EMA effect or establish statistical significance. QA100 uses actor100; the new probe100 is for subsequent selection and cannot itself explain actor100 QA.

## Actual100 reward refit

The original ordered300 calibration questions were used with actor100. The completed immutable manifest and both small grounding heads were actually read and SHA checked; all four real HTTP replicas adopted the newly fitted round100 version. This is separate from the selection probe. Subsequent completed actor101 receipts also confirm all four actual reward-RPC records used this round100 version.

Grounding version: `d9e5385ebec24ea8ab6cb8db8d2f825a:100:a5fafdc6dfef4262960164b80c990a09`.

## Source joint checkpoint100

The once-only CPU reader independently hashed all25 source files totaling39539134667 bytes and actually deserialized the saved states: actor2000/rank, scheduler100, probe raw/EMA/Adam1400/rank, EMA1400, and saved four-rank RNG. Source probe is the correct20-token/293-position checkpoint. The proof is `/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/search-g1-prefix-selection200-reference-replica-b-20261018-d9e5385ebec2-progressive-cp50-r2/preparation/ab-joint-0100-independent-full-sha-cpu.json`, SHA `5c42967db2df36577b2484f51c2549d979a8c252d1449f73dd31ba40b8ddd508`. No GPU forward/restore or independent backup was performed; completed hashes/loads must not be repeated.

## Original-distribution probe fit at100

Sampling actor0099 supplied the original2560 trainingG5 and fresh fixed512 heldoutG5. No qid or normalized-content overlap. Before/after uses identical trajectories and labels, not teacher trajectories or QA100 greedy responses.

| Model / original distribution | Mixed BA before → after | Trajectory BCE before → after | Mixed BCE before → after |
|---|---:|---:|---:|
| raw / train | 51.1043% → 70.9124% | 1.387744 → 0.505107 | 0.689433 → 0.835606 |
| raw / test | 50.1265% → 55.2573% | 1.690824 → 0.797697 | 0.606807 → 1.282754 |
| ema / train | 48.5875% → 54.9454% | 0.784499 → 0.635804 | 0.997547 → 1.012398 |
| ema / test | 48.5530% → 56.6920% | 1.073064 → 0.906007 | 0.889256 → 1.091456 |

EMA heldout BA improves48.55%→56.69%, but its heldout mixedBCE worsens0.889256→1.091456. Raw heldout mixedBCE also worsens despite higher BA. This is not uniform improvement. Heldout has74mixed/438nonmixed (majority85.5469%); post EMA ordinary accuracy45.1172%, mixed recall72.9730%, nonmixed recall40.4110%, with261 false positives. Mean predicted mixture0.58734 remains above true prevalence0.14453.

Original train has301mixed/2259nonmixed. Its one balanced pass is1280+1280 groups,1278 unique,50.0781% repetitions. All four actual probe journals show50 Adam/EMA writes1350→1400 atLR5e-6/EMA.99, trajectoryBCE+5×mixedBCE; actor journals show20 writes1980→2000.

| Actual changing-batch window | Trajectory BCE | Mixed BCE | Weighted total |
|---|---:|---:|---:|
| first10 | 1.053664 | 0.986779 | 5.987558 |
| last10 | 0.670266 | 0.652587 | 3.933202 |

Windows use different balanced mini-batches and actual global-question weights, not repeated fixed-data loss. The probe100 loop takes7.008537s; whole training-wall phase takes1162.969228s (19.38min).

Periodic100 selects301/512=58.7891% mixed using the existing actual-label-assisted rule. This is not probe classification accuracy.

## Actual101 native40-token / EMA adoption

Original next-prefix scan actually contains25600 trajectories from actor0100, max retained policy prefix40 and head313 positions, actor-hidden-native=true, independent-encoder-forward=false, labels_present=false. The legacy string identifiers containing “policy5/prefix5” remain unchanged; the actual numeric dimension/source fields above record this40-token phase.

The immutable20→40 migration binds the completed100 checkpoint to a new40-token payload, preserving the recorded raw/EMA/Adam1400 clocks, LR, run and no-reset flags. Actual101 unlabelled EMA scoring uses the migrated payload `5994fc1aaf5ebe5ee14f6fba9dd85721c4b57e05ea2c94bbadbfbb8cc5a25654` and actor0100. Migration receipts and actual score metadata are not an independent tensor-preservation/forward audit.

Actual101 is now complete: all four actor journals record20 writes to2020 and all four probe journals10 writes to1410. Its actual four-rank reward records use the fitted100 version. Later150/200 QA, final51713 and native exit0/resource closure are not yet achieved. Models, optimizer, hidden and complete trajectories remain remote.
