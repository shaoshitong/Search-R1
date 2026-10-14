# Reward selection comparison figures, 2026-10-14

Native editable draw.io diagrams, no bitmap-wrapped diagrams. Labels: Soft Reward; Soft Reward -> Hard Reward; Easy Selection+Soft Reward. Fixed512 EM is from immutable native evaluation identities at outer50/100/150/200. Native training reward means are from actual update ledgers. The two reward definitions have different scales.

Diversity is the sample standard deviation (correction1) within each original question's five responses. A=(r-group_mean)/(group_std+1e-6), not token weighted. Strict EM component distributions are reconstructed from source-bound token/scorer or immutable selection receipts, and are not the actual soft-policy advantages. Historical Soft Reward reconstruction is available at1/20/40/60/80/100/120/140. Its per-row soft rewards use frozen historical encoders and readout heads, not guaranteed bitwise online recovery.

Independent Easy Selection reward replay was blocked before GPU model loading when the original experiment's final evaluation3395 acquired all8 GPUs. No trainer, source manifest or original probe service was changed. The isolated replay script is a candidate, not validated GPU inference; aggregate values from failed attempts must not be published as data. Raw tokens, model weights and per-row replay outputs stay remote. Local artifacts only contain code, diagrams and compact aggregates.

The source root/work paths in audit tools locate existing deployments. These scripts do not launch or restart any original training/evaluation job. Do not interpret code backup as execution authorization.
