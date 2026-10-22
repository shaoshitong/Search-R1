# Native Prefix80, existing matched layer runs

All values are already published fixed-pool evaluations; no new model execution or final4096 read. Training plan each epoch contains3489Mixed and4703nonMixed questions, with inverse-frequency BCE weights. The fixed training probe has888Mixed/210all-correct/950all-wrong; fixed viewed validation has663Mixed/330all-correct/1055all-wrong.

| Layer | State | Epoch | Train BA | Train BCE | Train Top512 Mixed | Viewed dev BA | Dev BCE | Dev Top512 Mixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
|15|raw|0|50.00%|0.746826|225|50.00%|0.795846|171|
|15|raw|1|52.52%|0.668043|288|51.58%|0.645266|207|
|15|raw|2|61.35%|0.673686|307|57.32%|0.729698|217|
|15|raw|3|65.46%|0.631542|327|59.80%|0.678623|219|
|15|EMA|0|50.00%|0.746826|225|50.00%|0.795846|171|
|15|EMA|1|50.00%|0.713097|247|50.00%|0.740837|190|
|15|EMA|2|58.53%|0.684924|292|55.79%|0.702893|212|
|15|EMA|3|63.02%|0.661615|307|59.16%|0.680578|213|
|18|raw|0|50.00%|0.750246|218|50.00%|0.800762|179|
|18|raw|1|54.34%|0.662729|293|52.64%|0.640853|209|
|18|raw|2|61.18%|0.680793|313|57.34%|0.750368|224|
|18|raw|3|63.13%|0.670230|323|59.12%|0.759905|223|
|18|EMA|0|50.00%|0.750246|218|50.00%|0.800762|179|
|18|EMA|1|50.03%|0.713969|254|50.04%|0.743681|191|
|18|EMA|2|58.92%|0.681336|297|56.29%|0.702029|212|
|18|EMA|3|62.26%|0.658012|311|59.14%|0.682262|222|
|21|raw|0|50.00%|0.778502|209|50.00%|0.844597|168|
|21|raw|1|54.46%|0.659688|296|53.66%|0.640664|202|
|21|raw|2|62.58%|0.667388|314|58.10%|0.736454|219|
|21|raw|3|66.19%|0.623105|337|59.94%|0.686762|215|
|21|EMA|0|50.00%|0.778502|209|50.00%|0.844597|168|
|21|EMA|1|50.13%|0.724932|247|50.18%|0.764415|183|
|21|EMA|2|59.21%|0.683680|297|56.02%|0.712993|203|
|21|EMA|3|64.03%|0.656175|313|58.75%|0.690154|214|

At E3, raw dev BA is59.80/59.12/59.94% for L15/L18/L21, with Top512 realMixed219/223/215. EMA dev BA is59.16/59.14/58.75%, with Top213/222/214. The highest BA and highest Top are different metrics; neither establishes generalization from this viewed pool. Complete E0…3 raw/EMA class confusion counts and candidate/prediction paths are in the JSON.

All three use a frozen official actor and one trainable copied native block plus a dim256 two-layer eight-head noncausal group head, direct Mixed BCE, LR5e-6, EMA.99, 8192questions×3epochs,192updates,global128/two GPUs. Runtime layer numbers are exact zero-based actor.layers indices; each capture uses actor.layers[:layer]. These are five authentic causal trajectory prefills, not the newer repacked variable-M model.

For a new independent seed, the current hardcoded model/head seed9137 and own-view-plan seed20261022 need an explicit immutable override and independent roots. Pair the same new model/plan seed across depth conditions; do not replace fixed dev G5. No second-seed job has been created by this report.

An existing small CPU hidden-fixture test checks native G5 permutation invariance. No actual candidate checkpoint permutation audit was found. The full packed sampler audit checks target invariance only, explicitly not prediction invariance. The packed causal architecture itself is order-sensitive.

Existing final4096 v3 loads generic native block/prefix geometry, but its predeclared allowed tuple gate currently lacks(15,80) and(18,80). If either is later frozen as a final candidate, a new immutable scorer publication must explicitly add that geometry before final access. This report does not alter that gate or unblind final data.
