# Detached unique-question score FIFO

`reproduction.operations.qwen7b_warmup.detached_score_window.DetachedScoreWindow(capacity=2018,target_k=500,reference_population=2018)` owns deterministic JSON metadata only. No model/feature tensors, sampling randomness, distributed collective or Slurm operation occurs inside the helper.

`prepare(current_entries, model_version=actual_Adam_updates)` takes global current rows in the exact order of the runtime's autograd logit vector. Entry fields: qid, actual trajectory_ids, actual selected-view strict-EM labels, true_mixed, finite detached cached_score, PRE-update scored_step, scored_batch_attempt, score_model='raw'; optional full view metadata may be retained but features/activations/Tensors are rejected. TrueMixed is verified from precisely these labels, never the full G10 when a subset was selected.

Before assembling, prior entries for every current qid are removed. The most recent up-to2018−B past rows are followed by all B current rows. Return fields are cached_rows/current_rows/rows, cached_scores, cached_labels(binaryMixed), current_labels(binaryMixed), N_eff, K, current_offset and diagnostics. Runtime concatenates its no-grad cached scalar tensor with the current global autograd logits. K=ceil(N_eff×500/2018), including warmfill. capacity0 keeps no cache but uses the same reference denominator, so a current64 batch getsK16.

After the runtime either performs one Adam update or skips it, call `commit(post_update_step=actual_new_count)` on every data attempt. Enqueued values retain the PRE-update version; a skipped update keeps the same model version. FIFO eviction does not inspect EM or scores. The ring itself retains up-to2018 records so its checkpoint covers the complete FIFO, not only the1954 chosen for this particular window.

`state_dict()` and `DetachedScoreWindow.from_state_dict(state)` preserve exact ordered JSON metadata, committed version/count and optional pending phase. Normal runtime checkpointing occurs after commit with pending=None; optimizer and sampler/RNG remain the common runtime's responsibility. This helper also roundtrips a prepared pending phase, without claiming an independently implemented GPU optimizer recovery for that phase.

Diagnostics include N/K, duplicate/removal counts, age min/mean/max in actual Adam updates, scored-version range, ring/window SHA256. They are deterministic across ranks when the runtime supplies the same gathered rows. Cache age is not a measured numerical score-drift bound.

For a fixed window with unchanged views and binary labels, if fresh scores s and the stale-mixed scores t differ by at-mostdelta uniformly, both K-score sums in the structured margin objective change by at-mostK×delta. Thus L_K(s)≤L_K(t)+2delta/gamma and fresh empirical regret/K≤cached L_K+2delta/gamma. This is a loss-value bound under an actual uniform-drift assumption; it does not make cache age a delta bound or prove that current-only gradients scaledN/B are unbiased full-window gradients.

Five pure CPU metadata tests passed in0.680seconds: full warmfill1954+64/K500; qid replacement and label-blind FIFO; cache0/K16 and skipped update; exact committed/pending checkpoint roundtrip; rejection of duplicate questions, wrong view labels and features/nonfinite scores. No GPU, model, remote IO or final4096 data was used.
