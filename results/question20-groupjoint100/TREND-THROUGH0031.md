# Fresh joint100 trend through complete epoch31

Same native3412 h100-8 RUNNING. Complete31/100 at3100 Adam/rank; epoch32 observed59 partial writes,3159 actualAdam/rank. LR at the last complete boundary2.4090546027993275e-5. Do not count32 as complete.

Window values are arithmetic means of complete post-update evaluations. Train uses the current epoch own5120G5 batch, which differs each epoch. Test is the unchanged512G5 cache at threshold0.5.

| Epochs | Train balanced accuracy | Test balanced accuracy | Test trajectory BCE | Test G5 BCE |
|---|---:|---:|---:|---:|
|1-10|57.1033%|52.3510%|0.876505|0.961823|
|11-20|60.4680%|54.1203%|0.709876|0.890624|
|21-30|66.1689%|57.1552%|0.548208|0.787419|

The latest epoch31 train balanced accuracy is67.40455292%; fixed-test balanced accuracy60.65656817%, the highest observed so far. Latest ordinary mixed accuracy is65.0390625% train and58.0078125% test. Both test loss components and window mean balanced accuracy improve across the first three ten-epoch windows. Individual epochs fluctuate; this is not a monotone trend or an established final gain.

Current loss is mean own-trajectory BCE plus weight1 mean ownG5 mixed BCE. The observed mixed label is0<sum(ownstrictEM)<5, includingK1. The event score uses the same five probabilities:1-prod(p)-prod(1-p). Encoder stays frozen and group loss is differentiable through the five logits.

Only small producer metric aggregates and their full-file SHA values were read, with completion coordinates checked. No new model forward, rollout, trajectory/prediction transfer or training change. All100 and final acceptance remain incomplete.
