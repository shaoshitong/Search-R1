# Through35: ordinary selection and current probe fitting

Both runs have35 complete outer records in this report. Each physical actor rank has700 actual writes; A selection head has0 online writes and B raw/EMA has470. Actual36 used frozen A0 and B EMA470, matching each prior35 head hash with outcome_labels_used=false. Training runtime2e397623 is unchanged.

## Ordinary selection, excluding periodic oracle rounds

| Arm | Ordinary6–35 count | Mean mixed6–35 | Mean mixed31–35 | Mean predicted mixing31–35 | Mean whole wall31–35 |
|---|---:|---:|---:|---:|---:|
|A|27|53.8701%|36.6016%|93.1998%|780.043s|
|B|27|49.3996%|38.7109%|92.8683%|793.929s|

These are original selected512 groups per round. The other4608 ordinary candidates have no final labels; there is no full-pool accuracy or enrichment claim. Candidate schedule and periodic actual-EM-priority selection remain unchanged. Recent B mixing exceeds A, while its full ordinary6–35 mean remains lower. This neither proves final QA benefit nor isolates EMA from the rest of the B treatment.

| Outer35 selected | A frozen | B EMA |
|---|---:|---:|
|Actual mixed|32.6172%|41.0156%|
|All wrong|8.0078%|31.2500%|
|All correct|59.3750%|27.7344%|
|Mean predicted mixing|93.2003%|92.7940%|

## B35 paired fitting on original unresampled512 G5

Probe LR5e-6;10 actual updates once after balanced256 mixed+256 nonmixed draws, EMA.99 after every write. Metrics below use all original selected512 groups, not the repeated balanced draws.

| Head | BA before→after | Ordinary accuracy before→after | Traj BCE before→after | Mixed BCE before→after | Weighted total before→after |
|---|---:|---:|---:|---:|---:|
|raw|59.5743%→65.6701%|58.9844%→65.2344%|1.015669→1.014183|0.689185→0.630194|4.461595→4.165155|
|ema|50.0000%→50.0000%|41.0156%→41.0156%|0.678474→0.677330|1.582931→1.542594|8.593130→8.390299|

EMA classifies all of its Top512 as mixed at the fixed0.5 threshold before and after fitting, giving selected-set BA50% and ordinary accuracy equal to the actual selected mixed fraction. This selected high-score subset does not establish that the head collapsed over the entire candidate or diagnostic distribution. Its mean prediction nevertheless greatly exceeds the observed selected mixed fraction; report the calibration discrepancy rather than declaring success. Ranking effectiveness and full-pool/fresh-diagnostic discrimination require their own evidence. There is no fresh diagnostic35 and no QA35; the next periodic diagnostic is40.

The diagnostic question identities are fixed but trajectories/labels are fresh at each periodic actor snapshot. The most recent complete paired diagnostic remains30; see `period30-paired-results.md`. QA5 remains A EM27.5391%/F1 36.1807%, B EM28.5156%/F1 37.3150%. QA50/100/150/200 and final51713 are pending. Probe BA and trajectory classification are not QA EM/F1.

All470 stored global B loss rows have sequential actual Adam/EMA clocks and total=trajectory BCE+5×mixed BCE within FP32 error. There were no added rollouts, encoder passes, GPU tests or changes to training to produce this report. Each arm continues200/no201; final QA, normal saved-state acceptance, retention and owned-resource closure are still required.

Full per-outer selected outcomes, original-distribution before/after metrics, balancing counts,470 loss rows and QA source receipts are in `through35-comparison/comparison-progress.json`; its Markdown companion provides a scan-friendly view. Source:

- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix5-ab-small-report-through035-20261018T043926380501Z.json`, SHA256 `c1287df95e47f1982bb17bed7df7b3d56a7c521aa3085d13499008ce9fe1f95c`.
