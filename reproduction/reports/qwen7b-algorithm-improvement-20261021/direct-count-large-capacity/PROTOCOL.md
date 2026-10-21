# Frozen-E1 large-cohort count-head capacity protocol

Prepared implementation only. No GPU has been submitted, no model tensors have
been loaded by this preparation. The5120 source metadata is now actually
complete (source receipt SHA8a4ed0b2…); exact five descriptors and original30016/938
proofs are frozen in INPUT-PACKET.json. CPU-only prepare is separately recorded. This diagnostic does not change743809 or743619.

The question is whether a small count head can generalize after fitting a much
larger training cohort when the original E1 native block27 and trajectory head
are entirely frozen. The earlier512-question diagnostic established capacity to
fit a tiny set but did not establish useful heldout performance.

## Fixed data and objective

- Fit: exactly the existing30016 G10 questions, original938 batches of32, all20
  epochs in the same original question order. The four original real G5 views
  per question retain their source grouping epoch and each weight1/4.
- Internal holdout: a separately frozen5120 training-pool question set,2560 NQ
  and2560 HotpotQA. Its sole primary G5 is the precommitted new-seed trajectory
  indices5..9 with those exact five own strict-EM labels. No score/label chooses
  its G5. Question and question-hash identities must be disjoint from fit,
  controlled6528, existing diagnostic/selection and final pools before use.
- Both fit and internal holdout texts were seen by original warmup. The holdout
  is independent of this head fit, not an unseen-original-warmup question pool.
- One fresh head from seed1729, identical structure/initialization to production;
  Adam0, LR1e-4, weight decay0, EMA0.99. Do not reuse the tinyfit-trained head.
- Only count NLL is differentiated. Frozen original trajectory BCE is a constant
  recorded for context. Original legacy raw parameters/buffers are hashed before
  and after native caching; no legacy optimizer is created and actor updates=0.
-20 epochs =18760 true head Adam updates. Full head raw/EMA/Adam/RNG/source
  archive/cache/data cursor checkpoints precede scoring. First complete update
  is replayed with exact full state, and initial1..6 progress is required.

## Arithmetic, geometry and memory

Use the qualified native API, micro8, question273 + actual masked prefix20.
Cache its genuine postblock hidden as BF16, without FP16 conversion or added
quantization. Head encode remains CUDA BF16 autocast and returns float32
embeddings; count classifier/NLL stay FP32 outside autocast. Raw/EMA are scored
separately. No public reader or active runner is edited.

35136 questions ×10 trajectories ×20 tokens ×3584 hidden ×2 bytes =
50,370,969,600 bytes (46.912GiB); masks add7,027,200 bytes and frozen logits
1,405,440 bytes. The isolated runner uses remote raw mmap cache files,256-row
accepted chunks and per-span SHA. It resumes only verified accepted spans and
never calls a partially written cache complete. Packed source features are
verified by the original strict reader, and own retained rows remain verified.

After cache completion, release the frozen legacy module; only then preload
cache to H100. Measured GPU free memory must exceed the cache bytes plus16GiB;
otherwise fail without truncating contexts, reducing precision or changing the
experiment. Planned allocation if separately authorized:1H10080GB,8CPU,128GiB
host memory. No allocation is requested here.

The prior1536 diagnostic spent173.73 seconds actually caching15360 trajectories.
A linear native-forward-only extrapolation is3974 seconds (66.2 minutes), not a
runtime promise: strict packed-source SHA/read initialization, metadata and
large-cache I/O add costs. The prior whole three-LR job took6m32s, which cannot
isolate a transferable per-update timing for this18760-step run. Resource and
walltime choices therefore remain subject to root's separate execution decision.
No physical-order optimization or native batch-size change is introduced now.

## Fixed measurements and interpretation

At epochs0/1/2/4/8/20, save full state first, then separately report raw and EMA:
fit-all4 NLL/count accuracy (related views, not independent questions), fit
primary-G5 NLL/count accuracy/Mixed BA/TPR/TNR/AUC, and internal new5 equivalents.
The fit primary G5 is the label-blind lexicographically smallest source-index
view. Holdout remains its exact new5. Report original global top25% composition
and each domain's separately declared25% coverage, with N and K0..5 histograms.
No external or final predictions are opened and this diagnostic does not select
an external candidate, threshold, LR grid or production winner.

Input packet keys are SHA descriptors: merged_groups(train G10),
merged_features_layer26(manifest descriptors), internal_holdout_selection,
internal_holdout_groups(internal_holdout exact new5), and
identity_exclusion_acceptance. No placeholders pass prepare. All raw cache,
models, optimizer states and probability rows remain remote. The current local
metadata tests cover exact memory/update arithmetic, own-new5 identity/EM
binding, disjoint questions and unchanged20-epoch original batch order.

Actual CPU preparation completed successfully with three metadata tests (0.736s),
followed by a separate status readback. Config0be60cc8… binds actual30016 fit and
5120 holdout, with own new5 labels and original938 byte/record controls verified.
See PREPARED.json and actual-config.json. No GPU job or sbatch was created.
