from pathlib import Path
import json
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control');repo=Path('/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016');out=repo/'results/actorhidden-i1-i6-fresh-20261017/probe20k-ema-study'
s=json.loads((root/'work/probe5120-test15360-dist-divkl10-final-aggregate.json').read_text());a=s['arms'][0];epochs=a['epochs']
assert s['questions']==5120 and s['test_questions']==15360 and a['arm']['mean_weight']==0
text=['# DIST + diversity KL: 5120 train / 15360 test, all 10 epochs','',
'Original20480 actor0050 cached questions are partitioned by question with fixed seed20261018, label-blind:5120G5 training,15360G5 test. All five sibling trajectories stay together. The original512 diagnostic is not used. Same matched initial head40/Adam200/eightRNG; prior fitted weights are not imported. These are internal heldout data from the earlier20480 corpus, not a new external final test.','',
'LR5e-5,EMA.99,question+20,8GPU,100newAdam/rank/epoch. Training-only dominant continuous hidden mixup. No rollout or encoder forward. Loss = within-G5 Pearson DIST + 5×Bernoulli diversity KL. Groupmean KL and individual BCE are absent from optimization; evaluator retains them only as inactive diagnostics. D(v)=4/25 sum_{j!=k}v_j(1-v_k); the mixed soft5vector supplies the entire target. Constant-label groups have no Pearson term and D=0 for both all-correct/all-wrong, so these terms do not identify their absolute correctness scale.','',
'q=1-prod(p)-prod(1-p) is the historical mixed-event probability; D is diversity intensity. Threshold.5 ordinary/balanced accuracy is computed separately for both. All tables use real, unaugmented data, fulltrain/test post-update evaluation. Loss and data partition both changed compared with the previous experiment; differences cannot be attributed solely to removing mean KL.','']
for mode in ('ema','raw'):
 text += ['## '+mode.upper()+' classification','', '| Epoch | Train q BA % | Test q BA % | Train D BA % | Test D BA % | Train q accuracy % | Test q accuracy % |','|---:|---:|---:|---:|---:|---:|---:|']
 for e in epochs:
  t,v=e[mode]['train'],e[mode]['test'];vals=[t['balanced_accuracy'],v['balanced_accuracy'],t['diversity_mixed_balanced_accuracy'],v['diversity_mixed_balanced_accuracy'],t['accuracy'],v['accuracy']];text.append('| '+str(e['epoch'])+' | '+' | '.join(f'{x*100:.2f}'for x in vals)+' |')
 text += ['','## '+mode.upper()+' loss and vector relation','', '| Epoch | Train DIST | Test DIST | Train diversity KL | Test diversity KL | Train total | Test total | Train mixed-group Pearson | Test mixed-group Pearson |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for e in epochs:
  t,v=e[mode]['train'],e[mode]['test'];vals=[t['dist_loss'],v['dist_loss'],t['diversity_kl'],v['diversity_kl'],t['total_loss'],v['total_loss'],t['vector_pearson_mixed_groups'],v['vector_pearson_mixed_groups']];text.append('| '+str(e['epoch'])+' | '+' | '.join(f'{x:.5f}'for x in vals)+' |')
 text+=['']
text += ['## Online synthetic-batch objective','', 'Before-update batch losses on mixed hidden inputs; do not compare directly to full real-data evaluation above.','', '| Epoch | Actual new updates/rank | Mean DIST | Mean diversity KL | Mean total | First total | Last total |','|---:|---:|---:|---:|---:|---:|---:|']
for e in a['update_curve_summary']:
 m=e['mean'];text.append(f"| {e['epoch']} | {e['actual_updates']} | {m['dist_loss']:.5f} | {m['diversity_kl']:.5f} | {m['total_loss']:.5f} | {e['first']['total_loss']:.5f} | {e['last']['total_loss']:.5f} |")
best=max(epochs,key=lambda e:e['ema']['test']['balanced_accuracy']);bestd=max(epochs,key=lambda e:e['ema']['test']['diversity_mixed_balanced_accuracy']);text+=['',f"Best observed EMA test q BA: {best['ema']['test']['balanced_accuracy']*100:.2f}% at epoch{best['epoch']}; D BA: {bestd['ema']['test']['diversity_mixed_balanced_accuracy']*100:.2f}% at epoch{bestd['epoch']}. These maxima use the repeatedly monitored internal test set.",'', 'All tensors/weights/optimizer/nativehidden/fullpredictions/split identities remain remote. Mac retains only code/config/docs/small aggregates and figures. Formalactor remains stopped; this is the authorized finite probe ablation.','']
(out/'RESPLIT-DIST-DIVERSITY-KL-10E-RESULTS.md').write_text('\n'.join(text));(out/'resplit-dist-divkl10-small-aggregate.json').write_text(json.dumps(s,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
fields=[('balanced_accuracy','Mixed event balanced accuracy (q)',100),('diversity_mixed_balanced_accuracy','Diversity threshold balanced accuracy (D)',100),('total_loss','Total: DIST + 5 diversity KL',1),('dist_loss','Within-G5 DIST',1),('diversity_kl','Finite-G diversity KL',1),('group_mean_mae','Mean correctness MAE (diagnostic only)',1)]
for ax,(key,title,scale)in zip(axs.flat,fields):
 for split,color in [('train','#1667aa'),('test','#dc6b25')]:
  for mode,style,alpha in [('raw','--',.45),('ema','-',1.)]:
   ax.plot([e['epoch']for e in epochs],[e[mode][split][key]*scale for e in epochs],style,color=color,alpha=alpha,marker='o'if mode=='ema'else None,markersize=3,label=f'{split} {mode.upper()}')
 if scale==100:ax.axhline(50,color='grey',linestyle=':',linewidth=.7)
 ax.set(title=title,xlabel='Epoch (0 = matched source)',ylabel='Percent'if scale==100 else'Loss / MAE');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Frozen actor50 | original20480 repartitioned:5120 G5 train /15360 G5 test\nNo mean KL; no individual BCE; question+20; LR5e-5; EMA.99; dominant hidden mixup train-only')
fig.savefig(out/'resplit-dist-divkl10-curves.png',dpi=150);fig.savefig(out/'resplit-dist-divkl10-curves.pdf');plt.close(fig)
print(json.dumps(dict(report=str(out/'RESPLIT-DIST-DIVERSITY-KL-10E-RESULTS.md'),curve=str(out/'resplit-dist-divkl10-curves.png'),best_epoch=best['epoch'],best_test_q_BA=best['ema']['test']['balanced_accuracy'])))
