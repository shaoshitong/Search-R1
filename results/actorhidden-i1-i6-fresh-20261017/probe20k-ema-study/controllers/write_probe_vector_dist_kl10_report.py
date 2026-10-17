from pathlib import Path
import json,sys
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control');repo=Path('/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
s=json.loads((root/'work/probe5120-vector-dist-kl10-final-aggregate.json').read_text());a=s['arms'][0];e=a['epochs'];out=repo/'results/actorhidden-i1-i6-fresh-20261017/probe20k-ema-study';out.mkdir(exist_ok=True)
lines=['# G5 DIST + group mean KL + diversity KL: all 10 epochs','',
'Frozen actor0050; same source head40/Adam200; 5120G5 train, fixed512G5 heldout. Question+20 readout. LR5e-5; EMA.99; train-only continuous dominant hidden mixup; no new rollout. Raw/EMA full unaugmented train/test evaluation after each epoch. Training loss = within-G5 Pearson DIST + mean-correctness Bernoulli KL + 5×finite-G diversity Bernoulli KL. No individual-trajectory or binary-mixed BCE in training.','',
'`q=1-prod(p)-prod(1-p)` is the historical binary mixed-event score. `D=4/25*sum_{j!=k}p_j(1-p_k)` is expected finite-G diversity intensity; real K=0/5→0, K=1/4→.64, K=2/3→.96. Both threshold.5 balanced accuracies are reported separately. The synthetic target is one five-vector r; both scalar targets come from r. Constant-target groups omit Pearson only; the two KL terms remain.','']
for mode in ('ema','raw'):
 lines+=['## '+mode.upper()+' post-update mixed classification','',
 '| Epoch | Train q BA % | Test q BA % | Train D BA % | Test D BA % | Train total | Test total |','|---:|---:|---:|---:|---:|---:|---:|']
 for row in e:
  t,v=row[mode]['train'],row[mode]['test'];lines.append(f"| {row['epoch']} | {t['balanced_accuracy']*100:.2f} | {v['balanced_accuracy']*100:.2f} | {t['diversity_mixed_balanced_accuracy']*100:.2f} | {v['diversity_mixed_balanced_accuracy']*100:.2f} | {t['total_loss']:.4f} | {v['total_loss']:.4f} |")
 lines+=['','## '+mode.upper()+' component losses on real unaugmented data','',
 '| Epoch | Train DIST | Test DIST | Train mean KL | Test mean KL | Train diversity KL | Test diversity KL | Train vector MAE | Test vector MAE |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for row in e:
  t,v=row[mode]['train'],row[mode]['test'];vals=[t['dist_loss'],v['dist_loss'],t['mean_correctness_kl'],v['mean_correctness_kl'],t['diversity_kl'],v['diversity_kl'],t['vector_mae'],v['vector_mae']];lines.append('| '+str(row['epoch'])+' | '+' | '.join(f'{x:.4f}'for x in vals)+' |')
 lines+=['']
lines+=['## Online loss on augmented training batches','',
'These batch losses are recorded before each batch update and use synthetic inputs; they are not the post-update fulltrain/heldout losses above.','',
'| Epoch | Updates/rank | Mean DIST | Mean mean-KL | Mean diversity-KL | Mean total | First total | Last total |','|---:|---:|---:|---:|---:|---:|---:|---:|']
for row in a['update_curve_summary']:
 m=row['mean'];lines.append(f"| {row['epoch']} | {row['actual_updates']} | {m['dist_loss']:.4f} | {m['mean_correctness_kl']:.4f} | {m['diversity_kl']:.4f} | {m['total_loss']:.4f} | {row['first']['total_loss']:.4f} | {row['last']['total_loss']:.4f} |")
best=max(e,key=lambda r:r['ema']['test']['balanced_accuracy']);bestd=max(e,key=lambda r:r['ema']['test']['diversity_mixed_balanced_accuracy'])
lines+=['',f"Best EMA heldout q BA: {best['ema']['test']['balanced_accuracy']*100:.2f}% at epoch{best['epoch']}; best EMA heldout D BA: {bestd['ema']['test']['diversity_mixed_balanced_accuracy']*100:.2f}% at epoch{bestd['epoch']}. These are observed maxima on the same repeatedly monitored diagnostic set, not an independent final quality estimate.",'',
'All model/optimizer/native-hidden/trajectories/fullpredictions remain remote. Native terminal, final checkpoint fullSHA/CPU state and owned-resource-release evidence are separate small summaries; they do not establish model quality or an independent persistent backup. Formal actor training remains stopped.','']
(out/'TEN-EPOCH-VECTOR-DIST-KL-RESULTS.md').write_text('\n'.join(lines))
(out/'vector-dist-kl10-small-aggregate.json').write_text(json.dumps(s,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
fields=[('balanced_accuracy','Mixed-event balanced accuracy (q)',100),('diversity_mixed_balanced_accuracy','Diversity-threshold balanced accuracy (D)',100),('total_loss','Total objective on real inputs',1),('dist_loss','Within-G5 DIST',1),('mean_correctness_kl','Group mean correctness KL',1),('diversity_kl','Finite-G diversity KL',1)]
for ax,(key,title,scale)in zip(axs.flat,fields):
 for split,col in [('train','#1667aa'),('test','#dc6b25')]:
  for mode,style,alpha in [('raw','--',.45),('ema','-',1.)]:
   ax.plot([r['epoch']for r in e],[r[mode][split][key]*scale for r in e],style,color=col,alpha=alpha,marker='o'if mode=='ema'else None,markersize=3,label=f'{split} {mode.upper()}')
 if scale==100:ax.axhline(50,color='grey',linestyle=':',linewidth=.7)
 ax.set(title=title,xlabel='Epoch (0 = common source)',ylabel='Percent'if scale==100 else'Loss');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Frozen actor50 | 5120 G5 train / fixed512 G5 test | question+20\nDIST + mean KL + 5 diversity KL; dominant hidden mixup; LR5e-5; EMA0.99')
fig.savefig(out/'vector-dist-kl10-curves.png',dpi=150);fig.savefig(out/'vector-dist-kl10-curves.pdf');plt.close(fig)
print(json.dumps(dict(report=str(out/'TEN-EPOCH-VECTOR-DIST-KL-RESULTS.md'),curve=str(out/'vector-dist-kl10-curves.png'),best_ema_q=dict(epoch=best['epoch'],BA=best['ema']['test']['balanced_accuracy']),best_ema_D=dict(epoch=bestd['epoch'],BA=bestd['ema']['test']['diversity_mixed_balanced_accuracy']))))
