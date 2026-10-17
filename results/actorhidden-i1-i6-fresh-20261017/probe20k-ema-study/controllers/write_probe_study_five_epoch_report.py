from pathlib import Path
import json
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
repo=Path('/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016');out=repo/'results/actorhidden-i1-i6-fresh-20261017/probe20k-ema-study'
prior=json.loads((root/'work/probe20k-bc5-matched-final-aggregate.json').read_text());new=json.loads((root/'work/probe5120-g5mixup-final-aggregate.json').read_text());conflict=json.loads((root/'work/probe5120-mixup-target-conflict-summary.json').read_text())
text=['# Frozen actor0050 probe: five-epoch results','Date:2026-10-18 Asia/Shanghai. All figures are probe classification, not actor QA EM/F1.','',
'Latest human setting:5120 distinct train questionsG5, fixed512 heldout questionsG5, full question+20 readout; copied block35+Linear; source head40/Adam200; LR5e-5; EMA.99; loss trajectory BCE +5*mixed BCE; compatible hidden/true soft-label mixup Beta(.2,.2), training only. All evaluations use original unaugmented features/labels. No new rollout or actor update. Five epochs100 actual Adam/rank each; final700/rank and EMA500. Native3473 COMPLETED0:0, final fullSHA/CPUstate/eightRNG/ownedrelease verified.','',
'## Latest5120 group5+mixup, LR5e-5','', '|Epoch|EMA train mixed BA|EMA test mixed BA|Raw train mixed BA|Raw test mixed BA|EMA train trajectory acc|EMA test trajectory acc|','|---|---:|---:|---:|---:|---:|---:|']
a=new['arms'][0]
for e in a['epochs']:
 vals=[e[mode][split]['balanced_accuracy']for mode,split in [('ema','train'),('ema','test'),('raw','train'),('raw','test')]]+[e['ema'][sp]['trajectory_accuracy']for sp in ('train','test')]
 text.append('|'+str(e['epoch'])+'|'+'|'.join(f'{v*100:.2f}%'for v in vals)+'|')
text+=['','|Epoch|EMA train traj BCE|EMA test traj BCE|EMA train mixed BCE|EMA test mixed BCE|EMA train mixed ordinary acc|EMA test mixed ordinary acc|','|---|---:|---:|---:|---:|---:|---:|']
for e in a['epochs']:
 vals=[e['ema'][sp][key]for key in ('trajectory_bce','group_bce')for sp in ('train','test')]
 text.append('|'+str(e['epoch'])+'|'+'|'.join(f'{v:.4f}'for v in vals)+'|'+'|'.join(f"{e['ema'][sp]['accuracy']*100:.2f}%"for sp in ('train','test'))+'|')
text+=['','## Earlier matched20480-question comparison','',
'Identical frozen actor50, source head40/Adam200/RNG, dataset/order/test; LR1e-5, EMA.99,400 updates per epoch. A baseline is intentionally stopped; use E0..5 only. B and C each completed5 epochs,2000 new/final2200 actual Adam/rank, native3472 COMPLETED0:0; final states/fullSHA/CPU/eightRNG and ownrelease verified.','',
'|Epoch|A EMA test mixed BA|B group2 EMA test mixed BA|C mixup EMA test mixed BA|','|---|---:|---:|---:|']
for i in range(6):text.append('|'+str(i)+'|'+'|'.join(f"{arm['epochs'][i]['ema']['test']['balanced_accuracy']*100:.2f}%"for arm in prior['arms'])+'|')
for arm in prior['arms']:
 text+=['','### '+arm['arm']['name'],'','|Epoch|EMA train BA|EMA test BA|Raw test BA|EMA train traj BCE|EMA test traj BCE|EMA train mixed BCE|EMA test mixed BCE|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for e in arm['epochs']:
  vals=[e['ema'][sp]['balanced_accuracy']for sp in ('train','test')]+[e['raw']['test']['balanced_accuracy']]
  losses=[e['ema'][sp][k]for k in ('trajectory_bce','group_bce')for sp in ('train','test')]
  text.append('|'+str(e['epoch'])+'|'+'|'.join(f'{v*100:.2f}%'for v in vals)+'|'+'|'.join(f'{v:.4f}'for v in losses)+'|')
text+=['','## Mixup target incompatibility found in actual plans','',
'For an all-correct G5 and an all-wrong G5, convex trajectory targets are all lambda, while both real mixed labels are0 so their convex mixed target is0. Fitting trajectory soft targets gives q=1-lambda^5-(1-lambda)^5>0. At lambda=.5 this is .9375 despite mixed target0. With q tied to the five probabilities, both targets cannot simultaneously attain their individual BCE minima. The group5 weight amplifies the tradeoff. This is a proved target inconsistency for these synthetic samples; it does not establish its sole causal role in all observed quality.','',
'Analysis reads actual saved epoch plans, actual strict EM, and reconstructs the exact seeded lambdas; no model forward or new training.','',
'|Epoch|Pairs|Allcorrect/allwrong pairs|Fraction|Those with lambda .2.. .8|','|---|---:|---:|---:|---:|']
for r in conflict['records']:text.append(f"|{r['epoch']}|{r['actual_planned_pairs']}|{r['allcorrect_allwrong_pairs']}|{r['allcorrect_allwrong_fraction']*100:.2f}%|{r['lambda_between_02_08']}|")
text+=['','Do not silently change targets in these completed runs. A future consistent variant could retain real-group BCE on unaugmented G5 and apply mixup to trajectory supervision separately; that requires a separately authorized comparison.','',
'## Interpretation and retention','',
'The5120 run changes data amount, group weight, learning rate and number of updates per epoch together; do not attribute effects to a single factor. EMA train BA54.49→60.99%, heldout51.65→52.37%; raw final train50.01/test48.45%. This setting has optimization/target-fit problems as well as weak generalization, not merely the earlier high-train/low-test pattern. Full raw/EMA losses, both accuracy definitions, AUC/Brier and probability-by-K summaries are in the small JSON aggregates. No claim of quality success. Formal actor training remains human-stopped.','',
'Weights/Adam/native hidden/raw trajectories/full predictions remain remote. Mac retains only code/config/docs and small aggregate/proof JSON. Final normal checkpoint CPU load and fullSHA are not an independent persistent checkpoint backup or a post-load GPU forward.','',
'Frozen runtime for3472:dbf426f739227c7650848b69c9ad0b50778770cc. Frozen runtime for3473:c93f55a0382ff3191b91755fa8309b1b0af4bb70. Later report/controller commits do not hotedit either runtime.']
(out/'FIVE-EPOCH-RESULTS.md').write_text('\n'.join(text)+'\n')
print(out/'FIVE-EPOCH-RESULTS.md')
