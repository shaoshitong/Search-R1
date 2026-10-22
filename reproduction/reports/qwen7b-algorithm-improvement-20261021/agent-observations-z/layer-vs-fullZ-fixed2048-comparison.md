# Existing fixed2048: layer pilots versus full Z

No new inference, training or final4096 read. All rows come from already published fixed2048 evaluations. This viewed validation pool has2048questions and663trueMixed under the same fixed ownG5. These are single-seed within-run trends, not repeated-run stability evidence.

| Experiment | State | Independent BA by epoch | Top512 real Mixed by epoch |
|---|---|---|---|
|L15P20|raw|50.23% → 54.75% → 58.58%|204 → 204 → 200|
|L15P20|EMA|49.85% → 54.56% → 57.10%|176 → 194 → 205|
|L18P20|raw|50.45% → 55.74% → 58.65%|203 → 205 → 206|
|L18P20|EMA|50.35% → 56.59% → 57.22%|183 → 198 → 205|
|L21P80|raw|53.66% → 58.10% → 59.94%|202 → 219 → 215|
|L21P80|EMA|50.18% → 56.02% → 58.75%|183 → 203 → 214|
|full_Z_L21P20_packed_M3to10|raw|53.78% → 55.58% → 54.91% → 59.67% → 54.44%|218 → 224 → 217 → 223 → 224|
|full_Z_L21P20_packed_M3to10|EMA|57.27% → 54.94% → 55.18% → 55.01% → 55.56%|215 → 223 → 225 → 220 → 221|

The layer15/18 Prefix20 pilots show increasingly useful classification BA, but their final Top512 remains200–206(raw) or205(EMA). Layer21 Prefix80 has the strongest BA trend among these8192 pilots: raw53.66→58.10→59.94%, EMA50.18→56.02→58.75%. Its ranking improves from the initial epoch but is not monotonic: raw202→219→215; EMA183→203→214.

Full Z shows higher Top512 counts than the final layer15/18 Prefix20 pilots, but its last three completed raw/EMA counts remain around220–225 rather than a continuing rise. Its raw best BA59.67% at E4 coexists with Top223; the EMA best Top225 at E3 coexists with BA55.18%. Those maxima belong to different checkpoints and should not be combined as one result. E5 trainTop399raw/365EMA contrasts with independent224/221, demonstrating a substantial training-versus-independent ranking gap. The observed data therefore support learning, but not a stable independent ranking breakthrough or final convergence.

Pilot fits use8192questions×3epochs, directMixedBCE with the original plan counts and inverse-frequency weighting, EMA.99, and authentic per-trajectory causal contexts. FullZ uses169028questions per epoch, randomM3…10, one repacked question+Mprefix input, normalized4vMSE+absolute4vMSE, and EMA.9995. The diagnostic thresholds are respectively Mixedprobability.5 and predicted variance.08. The same validation question/trajectory labels permit descriptive selection comparison, but this is not a single-factor causal ablation or equal-budget comparison. No final candidate is frozen by this report.

Exact row-level source hashes, complete train/independent BA and Top counts, separate metric-specific maxima and last-three-epoch ranges are in layer-vs-fullZ-fixed2048-comparison.json.
