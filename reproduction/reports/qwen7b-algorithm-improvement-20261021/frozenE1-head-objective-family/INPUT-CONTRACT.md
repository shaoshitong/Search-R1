# Actual shared input resolver and coarse row adapter

`deploy_head_family_metadata.py --action resolve` resolves the already frozen
family. Missing cache/CP0/epoch0 completion returns the literal status
WAITING_FOR_VERIFIED_BASELINE_CACHE_CP0 and writes no consumer packet.
It creates no GPU job and does not wait for assets or fabricate future hashes.

Once real files exist, the consumer packet role is
FROZEN_E1_SHARED_CACHE_CP0_COARSE_FORK_INPUTS. Required refs are baseline_config,
baseline_publication, cache_complete, baseline_CP0, baseline_epoch0_evaluation,
family_protocol and frozen_train_prior. Extra refs bind cohort/orders, fixed
view epochs/grouping, held own-five, exclusions and the independent auditors.

Controller acceptance rehashes metadata, all1373 native chunk receipts, the
small raw frozen-logit file and small CP0; checks351360 exact ordered own IDs,
real tensor file sizes, finite float32 logits, BF16 geometry, original938
B32 batches repeated20times, own-newseed labels and cache→CP0→evaluation order.
The50GB hidden tensor is only statted. Its full payload SHA remains the
producer-complete claim; this controller never calls torch.load or loads that
large payload. CP0 only receives bounded ZIP/storage inspection here.

Initial CP0 raw/EMA/Adam/RNG hashes come from the actual epoch0 accepted-save
receipt, with Adam0 and initial raw=EMA. They are metadata until the future
fork allocation independently CPU-loads and restores all four components
before ANY update. The packet explicitly marks this mandatory unperformed
gate. The staged independent small checkpoint helper exports geometry,
component_hashes, archive_signature and restore_small for this purpose; its
Count743978-specific audit wrapper must not be called from a fork.

The coarse row schema keeps qid,q,K,domain,trajectory_ids,strict_em,
unadjusted_six_count_probabilities,unadjusted_event_probabilities,p_j and
coarse_event_NLL. Common score_semantics comes from the evaluation receipt.
The new direct_head_family_row_audit.py adapter independently checks own IDs,
labels, probability aggregation and semantics then reuses the original
independent BA/recall/AW-AC/AUC/Top25 implementation. It never requires countNLL
or predicted-K accuracy and does not claim coarse objectives identify the six
count categories. The original Count row auditor remains immutable.

Four synthetic CPU fixtures passed: full36-node gate/tie selection, missing
actual assets, bounded ZIP-only inspection, and coarse own-label/probability/
semantic mismatches. No real scoring, parameter fitting or GPU launch occurred.
