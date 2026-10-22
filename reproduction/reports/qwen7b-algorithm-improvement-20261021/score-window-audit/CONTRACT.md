# Detached score-window audit

Entry: `reproduction.operations.qwen7b_warmup.score_window_audit.audit_score_window(entries, *, current_step, score_fn, k, gamma=1.0, loss_fn=precision_at_k_ramp)`. It imports the exact existing `group_topk_losses.precision_at_k_ramp`; no alternative same-name objective is defined.

Each entry has qid, own trajectory_ids, selected-subset binary strict-EM labels, cached_score (unbounded raw logit), scored_step (actual Adam update counter), optional scored_batch_attempt, and score_model=raw. Every rank calls the audit if the scorer uses collectives. The callback receives new identity-only dicts containing qid, trajectory_ids and M, runs under `torch.no_grad()`, and returns the complete global window in its original order. The runtime owns capture, gather and eval/train mode restoration.

The audit never refreshes the queue or updates parameters, and checks that critical caller-owned cache metadata remains unchanged. Only aggregate Python values and SHA digests are returned. TopK ties use descending logit then supplied window index, matching the loss module's reference regret ordering.

Let b=min(K,totalMixed). The exact training loss is `relu(gamma*b + sumTopK(s-gamma*y) - C_star)/(gamma*K)`, with C_star the maximal score sum among selections containing the attainable b positives. Both the loss-augmented TopK sum and C_star change by at most K*delta under supnorm score drift delta. ReLU is 1-Lipschitz, giving `abs(Lfresh-Lcached)<=2*delta/gamma`. Thus `fresh_attainable_regret=b/K-Pfresh<=Lcached+2*delta/gamma`, and `Pfresh>=max(0,b/K-bound)`. Loss need not be at most1. This is not `Pfresh>=1-bound` when totalMixed<K, and it certifies only the exact frozen finite window.

Main aggregate fields: cached_L/cached_bound, fresh_L, fresh_bound, delta_max/delta_mean, oracle_precision, cached/fresh_attainable_regret, cached/fresh_P_at_K, old_fresh_TopK_overlap_count/fraction, bound-holds flags, age_actual_updates_histogram, view/qid coverage SHA digests and duplicate counts. No per-question logits, labels, raw rollouts, answer text or feature tensors are returned.

Actual CPU validation: six tests passed, including100 randomized finite-window drift checks, fewer-thanK positives, unbounded ramp loss, all-negative window, no-grad/parameter/cache preservation, label stripping, empty-window no-forward, and detection of a corrupting callback. No GPU, Slurm or final4096 access was involved.
