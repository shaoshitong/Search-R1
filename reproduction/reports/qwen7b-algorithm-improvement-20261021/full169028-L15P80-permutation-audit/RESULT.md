# Actual full169028 L15/P80 MSE trajectory-order audit

Job **750476 completed successfully (COMPLETED0:0), elapsed2m28s**. This was the existing two-GPU viewed-NQ audit of the already selected **E9/raw** checkpoints from both training seeds. No actor/probe update or optimizer was created. The original/reverse/cyclic1 orders were fixed before execution; all three are reported and no favorable order is selected.

| Training seed | Order | True Mixed in Top512 | Mixed fraction | BA at v≥0.08 | ΔMixed vs original |
|---|---|---:|---:|---:|---:|
| seed1 | original | 329/512 | 64.257812% | 70.113748% | 0 |
| seed1 | reverse | 325/512 | 63.476562% | 69.660497% | -4 |
| seed1 | cyclic1 | 312/512 | 60.937500% | 69.605121% | -17 |
| seed2 | original | 321/512 | 62.695312% | 71.331711% | 0 |
| seed2 | reverse | 318/512 | 62.109375% | 71.046877% | -3 |
| seed2 | cyclic1 | 314/512 | 61.328125% | 70.547887% | -7 |

The packed probe is sensitive to trajectory order: reversing the five trajectories lowers Mixed@512 by4 and3 questions; cyclic1 lowers it by17 and7 questions for seed1/seed2. The original fixed order remains the inference protocol. This observation does not justify changing the order after viewing these results.

| Seed | Permutation | Score Spearman vs original | Top512 overlap | Mean absolute variance drift | Maximum absolute variance drift | P95 absolute variance drift | Changed threshold labels |
|---|---|---:|---:|---:|---:|---:|---:|
| seed1 | reverse | 0.963141919 | 442/512 | 0.0106826529 | 0.1609560829 | 0.0345296748 | 158 |
| seed1 | cyclic1 | 0.972830079 | 450/512 | 0.0091138212 | 0.1626574416 | 0.0294146143 | 123 |
| seed2 | reverse | 0.962937022 | 439/512 | 0.0102536200 | 0.1392908227 | 0.0335894341 | 151 |
| seed2 | cyclic1 | 0.971777428 | 451/512 | 0.0087114807 | 0.1464541834 | 0.0272392567 | 147 |

Each original-order replay covers the same2048 questions, ordered own five trajectory IDs and strict EM labels as the checkpoint’s saved development evaluation. **Both seeds have maximum and mean absolute4v-score delta0**, passing the fixed0.001 tolerance exactly. The replay is an actual numeric adapter check, not just matching Top512 counts.

All orders preserve the same question, the same five own trajectory/strict-EM/real Prefix80-token associations and the same target variance; only their order changes. `same_own_ID_EM_P80_tokens_and_targets_all_orders=true` for both seeds. The per-order identity SHA is identical across the two seeds:

- original: `267190b86736f48716a7bb812bab7e6f7dad21a3fdad7aeadea5b7bd76723c25`.
- reverse: `2c96ecdedc454a6287e944610f504a4a1282639b373320310efb8afb3f0e80ba`.
- cyclic1: `28de554416598859076d4b629da7f827331f08f6ba9fbf53beb158a081342ea0`.

The original question/prefix inputs bind the accepted physical-v2 fixture SHA `745bdef27a1098603142a3e9f5506b4bd2777f8b699933923dbc1a208f44f1f8`, physical execution/source contract and complete P20-alignment proofs. The two selected checkpoint SHA values are:

- seed1: `5f90fb17bee993353f47a7d687bdf2ea235dcd3d2eba287465567f4c286b13d7`; CPU acceptance receipt SHA `2b2be91c7df977762d9889cb7c969e1f66a13811ad5c5ec55e1c79ad66942d96`.
- seed2: `4203a480330b5f64def9dcf641ad6e66067c1810a46c113b778b6c79cdc90145`; CPU acceptance receipt SHA `b00c8cad06b1eee0e135ca890aaebb00beea913866ba3bce62367ac6e1024d40`.

Scores are population variance v; the saved evaluation uses4v, and replay comparisons explicitly account for that scale. Thresholdv0.08 is fixed, not fitted. Labels and targets are used only to evaluate these unchanged groups, never to reorder or choose trajectories.

This is **repeatedly viewed NQ development order sensitivity**, not unseen-question generalization or final4096 confirmation. `order_selected_by_audit=false`, `final4096_read=false`, actor/probe updates0. Root has separately reported an actual candidate freeze with config SHA `2606a04636e282df2e011c962fe47c29c00be4460f3581f3fc03b58fd5d428ca`; final submission was invoked but its job handle/outcome was not established for this report. **No final result or completion is claimed here.**

Source: local [actual-audit-live.json](../fullP80-selected-root-control/actual-audit-live.json), snapshot `1792715958.004006`, file SHA `aca10629480f8db6ebf77d5fc36c6e8d090fcd89b44d34fc34c32946eba490c4`; bound audit descriptor SHA `1bca7df0a5ba108758f7082469b2352e260fbc711047382a99ea1f8925a1acdc`. This report was assembled locally from the existing small receipt; no remote process, checkpoint load, final-data access, new inference, Slurm action or push was performed.
