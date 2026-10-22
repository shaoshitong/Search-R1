# Input and detached-window semantics

This note records existing source behavior; it does not change or restart any experiment.

The full L15/P80 MSE input contains one question/template prompt and five selected own-trajectory prefixes, each capped at 80 genuine retained policy tokens, with fixed separators and masked padding for shorter completions. No extra future response suffix, gold answer, EM or Mixed target is appended. A generated completion shorter than 80 tokens can naturally fit completely inside that fixed prefix. This follows the user's existing short-response padding instruction; it is not correct to claim that a prefix can never contain a complete generated answer.

Authoritative source: `online_packed_full169028_MSE_P80.py` `pack_group` and `capture`; physical producer `prepare_full169028_policy80_physical_v2.py` `policy80` and `compact_raw`. The pilot acceptance recorded 43,567 of 92,160 own rows as shorter than 80 tokens; this is a pilot statistic, not an estimate of the full169028 population. No raw source rescan or additional model inference was performed for this clarification.

The separate WINDOW2018 mechanism has a **total** score window of 2,018 questions. Once filled, it contains 1,954 historical detached scores plus 64 current autograd scores. The history holds scalar scores, actual selected-view EM labels, identities and score versions; current question IDs replace old cached views. Historical rows participate in the TopK-ramp comparison without a backward graph. Current64 alone contributes parameter gradients. After the step, FIFO commits the detached pre-update current scores. Before the FIFO is full, the effective window grows with available history.

Authoritative source: `detached_score_window.py` `prepare`/`commit`; `online_packed_window_topk_seeded_v2.py` current all-gather, history tensor construction, concatenation and backward. Separate epoch-end `no_grad` recomputation diagnoses score staleness and does not backpropagate through the history. This is not the exact gradient of recomputing all2,018 samples under the current model, and does not guarantee unseen-question improvement. The latest user clarification required no new job, source hot edit or duplicate experiment.

Observed gains from the full L15/P80 MSE runs belong to that complete configuration. They cannot be attributed to the loss alone, or to WINDOW2018, which is a separate matched pilot comparison.
