from pathlib import Path
import json
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control');out=Path('/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/results/actorhidden-i1-i6-fresh-20261017/probe20k-ema-study')
s=json.loads((root/'work/probe5120-test15360-softmix50-lr1e5-final-aggregate.json').read_text());a=s['arms'][0];epochs=a['epochs'];assert a['arm']['objective']=='group_vector_dist_softmix_kl'and a['arm']['soft_mixed_weight']==50 and s['learning_rate']==1e-5
text=['# Same h in training and selection: softmixedKL50, LR1e-5, all10epochs','',
'Same original20480 cached actor0050 G5 questions, partition seed20261018:5120train/15360test, fullG5 question-disjoint. Old512 diagnostic not used. Same matched sourcehead40/Adam200/eightRNG, not earlier fitted weights. No new rollout/encoder forward. Question+20,8GPU,globalmeanBS256/fullG5/noaccum,EMA.99,train-only dominant continuous hidden mixup.','',
'PRIMARY prediction in BOTH training and evaluation: h=1-prod(p)-prod(1-p). Soft training targetbyK=[0,.9,1,1,.9,0]. Loss=DIST(ordered5probabilities,orderedownEMvector)+50*KL(Bern(t(K))||Bern(h)). IndividualBCE,groupmeanKL,D-intensityKL are excluded. Training mixup uses the same lambda>.5 for hidden,orderedownEMvector,and convex combination of the original group targets t(K_A),t(K_B). No hardK inferred from fractional vectors. Test/realtrain evaluation uses unaugmented inputs and the original soft target, while binary mixed accuracy uses0<K<5 and h>=.5. Legacy D metrics are diagnostics only and are not primary results.','',
'All raw/EMA epoch0..10 results below are full real post-update evaluations. Online synthetic-batch losses are separate. LR,groupobjective/softtargets,and groupweight differ from the earlier run; this is not a single-factor LR ablation. The test set is an internal heldout partition of the previously used20480 corpus, not a newly collected external final test.','']
for mode in ('ema','raw'):
 text+=['## '+mode.upper()+' mixed decision using h','', '| Epoch | Train h balanced % | Test h balanced % | Train h accuracy % | Test h accuracy % | Train mean h | Test mean h |','|---:|---:|---:|---:|---:|---:|---:|']
 for e in epochs:
  t,v=e[mode]['train'],e[mode]['test'];text.append(f"| {e['epoch']} | {100*t['balanced_accuracy']:.2f} | {100*v['balanced_accuracy']:.2f} | {100*t['accuracy']:.2f} | {100*v['accuracy']:.2f} | {t['mean_q']:.5f} | {v['mean_q']:.5f} |")
 text+=['','## '+mode.upper()+' current-objective losses','', '| Epoch | Train DIST | Test DIST | Train soft h KL | Test soft h KL | Train total | Test total | Train mixed-group Pearson | Test mixed-group Pearson |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for e in epochs:
  t,v=e[mode]['train'],e[mode]['test'];values=[t['dist_loss'],v['dist_loss'],t['soft_mixed_kl'],v['soft_mixed_kl'],t['total_loss'],v['total_loss'],t['vector_pearson_mixed_groups'],v['vector_pearson_mixed_groups']];text.append('| '+str(e['epoch'])+' | '+' | '.join(f'{x:.5f}'for x in values)+' |')
 text+=['']
for mode in ('ema','raw'):
 text+=['## '+mode.upper()+' test recalls and calibration','', '| Epoch | Mixed recall % | Nonmixed recall % | Mean h | Mean soft target |','|---:|---:|---:|---:|---:|']
 for e in epochs:
  v=e[mode]['test'];text.append(f"| {e['epoch']} | {100*v['mixed_recall']:.2f} | {100*v['nonmixed_recall']:.2f} | {v['mean_q']:.5f} | {v['soft_mixed_target_mean']:.5f} |")
 text+=['']
text+=['## Online loss on augmented batches','', 'This is before-update loss on mixed hidden tensors; not fulltrain/test evaluation.','', '| Epoch | New updates/rank | Mean DIST | Mean soft h KL | Mean total | First total | Last total |','|---:|---:|---:|---:|---:|---:|---:|']
for e in a['update_curve_summary']:
 m=e['mean'];text.append(f"| {e['epoch']} | {e['actual_updates']} | {m['dist_loss']:.5f} | {m['soft_mixed_kl']:.5f} | {m['total_loss']:.5f} | {e['first']['total_loss']:.5f} | {e['last']['total_loss']:.5f} |")
best=max(epochs[1:],key=lambda e:e['ema']['test']['balanced_accuracy']);text+=['',f"Best trained EMA test h BA: {100*best['ema']['test']['balanced_accuracy']:.2f}% at epoch{best['epoch']}. This is a maximum on a repeatedly monitored internal diagnostic set, not independent final QA quality.",'', 'All models/Adam/nativehidden/trajectory/fullpredictions/split IDs remain remote. Local files contain only code/config/docs/small aggregate and figures. Formal actor training remains stopped.','']
(out/'SOFTMIX50-LR1E5-10E-RESULTS.md').write_text('\n'.join(text));(out/'softmix50-lr1e5-small-aggregate.json').write_text(json.dumps(s,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
fields=[('balanced_accuracy','h mixed balanced accuracy',100),('accuracy','h mixed ordinary accuracy',100),('total_loss','Total: DIST +50 soft h KL',1),('dist_loss','Within-G5 DIST',1),('soft_mixed_kl','Soft mixed-event KL (training target)',1),('mean_q','Mean predicted h',1)]
for ax,(key,title,scale)in zip(axs.flat,fields):
 for split,color in [('train','#1667aa'),('test','#dc6b25')]:
  for mode,style,alpha in [('raw','--',.45),('ema','-',1.)]:
   ax.plot([e['epoch']for e in epochs],[e[mode][split][key]*scale for e in epochs],style,color=color,alpha=alpha,marker='o'if mode=='ema'else None,markersize=3,label=f'{split} {mode.upper()}')
 if key=='balanced_accuracy':ax.axhline(50,color='grey',linestyle=':',linewidth=.7)
 ax.set(title=title,xlabel='Epoch (0 = matched source)',ylabel='Percent'if scale==100 else'Loss / Probability');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Same h in training and testing |5120 G5 train /15360 G5 test |question+20\nDIST +50 soft mixed KL; target byK=[0,.9,1,1,.9,0]; LR1e-5;EMA.99;train-only dominant hidden mixup')
fig.savefig(out/'softmix50-lr1e5-curves.png',dpi=150);fig.savefig(out/'softmix50-lr1e5-curves.pdf');plt.close(fig)
print(json.dumps(dict(report=str(out/'SOFTMIX50-LR1E5-10E-RESULTS.md'),curve=str(out/'softmix50-lr1e5-curves.png'),best_trained_epoch=best['epoch'],best_test_BA=best['ema']['test']['balanced_accuracy'])))
