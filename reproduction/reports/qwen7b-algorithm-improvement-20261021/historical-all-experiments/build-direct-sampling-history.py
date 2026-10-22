"""Local existing-summary extraction only; no remote access or fresh scoring."""
import hashlib,json,pathlib,re
R=pathlib.Path(__file__).resolve().parents[1];OUT=pathlib.Path(__file__).resolve().parent
rows=[];windows=[];inventory=[]
def read(name):
 p=R/name;raw=p.read_bytes();d=json.loads(raw);inventory.append(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));return d,str(p)
def first(d,*keys):
 for k in keys:
  if k in d:return d[k]
 return None
def metric(d):
 v=d.get('pool',d);sel=first(d,'selected512','selected512_composition','selected512composition','Top512','top512')or{}
 n=first(sel,'mixed','Mixed','mixed_questions','actual_Mixed');frac=first(sel,'fraction','Mixed_fraction','mixed_fraction');k=first(sel,'questions','selected_questions','selection_budget')
 if n is None:n=first(d,'selected512_Mixed','Top512_Mixed','Top512Mixed')
 if frac is None:frac=first(d,'selected512_Mixed_fraction','Top512_Mixed_fraction')
 if n is not None and k is None:k=512
 return dict(BA=first(v,'mixed_balanced_accuracy','Mixed_balanced_accuracy','balanced_accuracy','BA','Mixed_BA','MixedBA'),BCE=first(d,'binaryMixedBCE','binaryMixedBCE_from_probability','producer_stable_binaryMixedBCE','BCE','Mixed_BCE'),AUC=first(v,'mixed_roc_auc','Mixed_ROC_AUC','AUC'),TPR=first(v,'mixed_recall','Mixed_recall','TPR'),FPR=first(v,'nonmixed_false_positive_rate','nonMixed_FPR','nonMixed_false_positive_rate'),questions=first(v,'questions','n'),Mixed=first(v,'mixed_questions','true_Mixed','Mixed'),top=n,top_k=k,top_fraction=frac,pool_identity=first(d,'own_G5_and_EM_identity_sha256','ordered_qids_sha256','pool_identity_SHA256'))
def add(exp,cp,state,d,pool,source,pointer,note,train=None,train_source=None,budget=None,status='measured_existing_checkpoint',head=False):
 m=metric(d);t=metric(train)if train else{};is_train=pool.startswith('fixed_train');ind='2048'in pool and not is_train
 rows.append(dict(experiment=exp,checkpoint=str(cp),state=state,train_BA=m['BA']if is_train else t.get('BA'),test_BA=None if is_train else m['BA'],top_mixed=m['top'],top_k=m['top_k'],top_fraction=m['top_fraction'],pool_questions=m['questions']or(2048 if'2048'in pool else 512 if'512'in pool else None),pool_identity=dict(name=pool,recorded_identity_sha256=m['pool_identity'],proof_source=source+'#'+pointer,identity_not_invented=True),evaluation_pool=pool,train_pool=pool if is_train else('fixed_train2048_E1_seen_labelblind_G5'if train else None),train_top_mixed=m['top']if is_train else t.get('top'),train_top_k=m['top_k']if is_train else t.get('top_k'),train_BCE=m['BCE']if is_train else t.get('BCE'),test_BCE=None if is_train else m['BCE'],AUC=m['AUC'],Mixed_recall=m['TPR'],nonMixed_FPR=m['FPR'],only_head=head,budget=budget,source=source+'#'+pointer,training_source=train_source,comparability_note=note,status=status))
# The latest complete packet retains E0, CP128 and CP512, both states and pools.
d,src=read('mixed-direct-cp512-eval/TERMINAL-PACKET.json')
for cp,states in d['metrics'].items():
 for state,pools in states.items():
  for pool,m in pools.items():add('official-direct-Mixed-fresh-block27',cp,state,m,pool,src,'/metrics/'+cp+'/'+state+'/'+pool,'Direct Mixed BCE only; native block+group head; official initial block; 50:50 real G5 role sampler with repeated questions. diagnostic512 selecting512 is the entire pool, not a2048→512 ranking.',budget=dict(cursor=int(cp[2:])if cp.startswith('CP')else 0,question_batch=32,LR=5e-6,EMA=.99))
# Original E1–E4 full budget, all states, independently recomputed ownG5.
d,src=read('mixed-direct-b128-final-metrics/actual-independent-metrics.json')
full,fullsrc=read('mixed-direct-b128-continuation/actual-evaluations-readonly-2.json')
base,basrc=read('mixed-direct-e1-unique-g5/E1-matched-training-independent-baseline.json')
for e,states in d['metrics'].items():
 for state,m in states.items():
  train=base['fixed_training']['metrics'][state]['fixed_train2048']if e=='1'else None
  note='Original question-visit epochs; repeated role sampling, four G5 views/question occurrence, previous-hard-quartile replay after boundaries. Not fair evidence for LR causality; no fixed train eval for E2–4. E1 fixed-train result was subsequently obtained on the exact matched E1 checkpoint.'
  add('official-direct-Mixed-original-hard-replay','E'+e,state,m,'fixed_selection2048',src,'/metrics/'+e+'/'+state,note,train,basrc+'#/fixed_training/metrics/'+state+'/fixed_train2048'if train else None,dict(cursor=m.get('cursor'),LR=5e-6,EMA=.99,global_question_batch=128,earlier_B32_updates=1675))
  ep=full.get('epochs',{}).get('epoch-'+e.zfill(2),{}).get('models',{}).get(state,{})
  if 'diagnostic512'in ep:add('official-direct-Mixed-original-hard-replay','E'+e,state,ep['diagnostic512'],'diagnostic512',fullsrc,'/epochs/epoch-'+e.zfill(2)+'/models/'+state+'/diagnostic512',note+' Whole512 pool selection is not ranking.',train,basrc if train else None,dict(cursor=m.get('cursor'),LR=5e-6,EMA=.99))
# Low LR/head-high-LR repeated-hard branch, distinct from corrected branch.
d,src=read('mixed-direct-e1-lr-headonly-results/actual-independent-metrics.json')
for state,m in d['reference_E1_metrics'].items():
 add('E1-fork747723-hard-replay-lowLR-then-head10x','E1',state,m,'fixed_selection2048',src,'/reference_E1_metrics/'+state,'Exact shared official E1 baseline, not a new independently trained model.',base['fixed_training']['metrics'][state]['fixed_train2048'],basrc,dict(cursor=2577,LR=5e-6,EMA=.99))
for e,states in d['metrics'].items():
 for state,m in states.items():add('E1-fork747723-hard-replay-lowLR-then-head10x','E'+e,state,m,'fixed_selection2048',src,'/metrics/'+e+'/'+state,'Continues exact E1 raw+EMA+Adam; old hard-replay/repeated-question sampler remains. E2 joint5e-7; E3 native frozen/head5e-5. Not fair isolated LR comparison. Fixed-train checkpoint BA absent; online train windows are separate.',budget=dict(cursor=m.get('cursor'),LR=5e-7 if e=='2'else 5e-5,EMA=.99,global_question_batch=128,updates_per_phase=1321),head=e=='3')
for phase,v in d.get('training_curve',{}).items():
 if isinstance(v,dict):windows.append(dict(experiment='747723-hard-replay-lowLR-head10x',phase=phase,metric_scope='changing_online_raw_not_fixed_checkpoint',summary=v,source=src+'#/training_curve/'+phase))
# Additional old lowLR/head-only diagnostic512 receipts, kept separate.
old,oldsrc=read('mixed-direct-e1-lr-headonly-20261022/actual-progress-7.json')
for e in(2,3):
 for state,pools in old['epoch_%d_evaluation'%e]['metrics'].items():
  add('E1-fork747723-hard-replay-lowLR-then-head10x','E'+str(e),state,pools['diagnostic512'],'diagnostic512',oldsrc,'/epoch_%d_evaluation/metrics/%s/diagnostic512'%(e,state),'Same repeated-hard branch; entire diagnostic512 selected, not2048→512 ranking.',budget=dict(LR=5e-7 if e==2 else 5e-5,EMA=.99),head=e==3)
# Corrected unique-question and natural label-blind controls: all saved pools/states.
for label,filename in [('corrected-unique-class-balanced749015','live-observations/main-latest.json'),('corrected-labelblind-uniformG5-classweighted749120','live-observations/uniform-latest.json')]:
 d,src=read(filename)
 for name,v in d.items():
  if not name.startswith('independent')or not isinstance(v,dict):continue
  e=int(name[len('independent'):]);train=d.get('training'+str(e),{}).get('metrics',{})
  for state,pools in v['metrics'].items():
   for pool,m in pools.items():
    tr=train.get(state,{}).get('fixed_train2048');note='E1 raw/EMA separately restored; raw updates EMA(.99). E2 joint and E3 head-only both5e-7. Each new epoch has169028 unique questions, randomized order and authentic own-G10→G5 draws, tail68. '
    note+=('Class-balanced84514/84514 true G5 with unique questions; no previous-hard replay.'if label.startswith('corrected-unique')else'Label-blind uniform real G5, class weights; no forced50:50 subset selection and no hard replay. Snapshot may predate E3 evaluation.')
    if pool=='diagnostic512':note+=' Whole512 selection is not2048→512 ranking.'
    add(label,'E'+str(e),state,m,pool,src,'/'+name+'/metrics/'+state+'/'+pool,note,tr,src+'#/training'+str(e)+'/metrics/'+state+'/fixed_train2048'if tr else None,dict(cursor=v['cursor'],new_phase_updates=1321,LR=5e-7,EMA=.99,global_batch=128),head=e==3)
# Historical block+trajectory BCE model: E0 is historical E4, not untrained.
d,src=read('joint124194-root-latest/actual-epoch-quality-audit.json')
for e in d['epoch_results']:
 for state,pools in e['states'].items():
  for pool,m in pools.items():
   if not isinstance(m,dict)or'questions'not in m:continue
   add('joint124194-native-block27-plus-readout','newE'+str(e['epoch'])+'=historical-balanced-P20-E4',state,m,pool,src,'/epoch_results/'+str(e['epoch'])+'/states/'+state+'/'+pool,'Trajectory BCE+product-formula Mixed BCE1:1; current start already historicalE4 Adam14418. Mixed/nonMixed50:50 role sampler with hard mining. No posttraining epoch1–4 evaluation in this completed-audit snapshot.',budget=dict(source_Adam=14418,new_cursor=e['checkpoint']['cursor'],questions=124194,updates_per_new_epoch=3882,LR=5e-6,EMA=.99))
for cp in d['pending_epochs']:
 for state in('raw','EMA'):add('joint124194-native-block27-plus-readout','newE'+str(cp),state,{},'fixed_selection2048',src,'/pending_epochs','No completed evaluation in saved local audit; not invented.',status='missing_posttraining_evaluation')
# Actual seven-update native qualification and prepare-only full169028 branch.
d,src=read('joint-balanced-native/actual-terminal-qualification.json')
for state in('raw','EMA'):add('joint-native-real-seven-update-qualification','CP7',state,{},'not_evaluated',src,'/external_evaluation_performed','Real7-update restore qualification; explicitly no external evaluation.',budget=dict(updates=7,source_Adam=14418),status='real_training_qualification_no_external_metrics')
p=R/'joint-balanced-native-full169028/PREPARATION.md';inventory.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
for state in('raw','EMA'):add('joint-native-full169028-prepare-only','not_started',state,{},'not_evaluated',str(p),'','Prepared continuation from E4, no completed metric receipt in local namespace. Not counted as a scored run.',budget=dict(questions=169028,planned_updates=21132,planned_epochs=4),status='prepared_only_no_result')
# Online curves preserved separately; never substituted for fixed train BA.
for filename,exp in [('mixed-direct-b128-continuation/actual-final-training-diagnostic.json','official-direct-Mixed-original-hard-replay'),('joint124194-root-latest/actual-train-test-current-metrics.json','joint124194-native-block27-plus-readout')]:
 d,src=read(filename);windows.append(dict(experiment=exp,metric_scope='changing_online_raw_not_fixed_checkpoint',source=src,summary=d.get('epochs',d.get('current_train_windows'))))
status=dict(status='LOCAL_EXISTING_DIRECT_MIXED_AND_SAMPLING_HISTORY_EXTRACTED',rows=rows,online_training_windows_separate=windows,sources=inventory,all_BA_units='fraction_0_to_1',missing_is_null=True,no_new_scoring=True,no_remote_access=True,no_model_tensor_read=True,no_final4096_read=True,scope_notes=['API/synthetic native-mixed-direct-head qualification is not empirical model evaluation; no fabricated metric rows.','Mixed-only-direct-group and official-direct-fast/B128 are one continuous source chain, not separate independently initialized experiments.','Other historical fixed-feature/classifier candidates assigned to separate report.'])
(OUT/'direct-mixed-and-sampling.json').write_text(json.dumps(status,indent=2)+'\n')
lines=['# 7B direct Mixed / sampling / joint-block history','', 'All numbers are extracted from existing local receipts, no new scoring. BA means binary Mixed balanced accuracy. Training BA is a fixed checkpoint on fixed train2048 only when available; changing online training windows are separately retained in JSON. Missing is `—`, never zero. Pool size512 rows rank the full512 and must not be read as2048→512 enrichment.','', '| Experiment | Checkpoint | State | Fixed train BA % | Evaluation pool | Eval BA % | AUC % | BCE | Selected true Mixed | Only head | Status |','|---|---|---|---:|---|---:|---:|---:|---:|---|---|']
for i,v in enumerate(rows):
 fmt=lambda k:'—'if v[k]is None else'%.2f'%(100*v[k]);top='—'if v['top_mixed']is None else'%s/%s (%.2f%%)'%(v['top_mixed'],v['top_k'],100*v['top_mixed']/v['top_k']);bce='—'if v['test_BCE']is None else'%.6f'%v['test_BCE'];lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |'%(v['experiment'],v['checkpoint'],v['state'],fmt('train_BA'),v['evaluation_pool'],fmt('test_BA'),fmt('AUC'),bce,top,'yes'if v['only_head']else'no',v['status']))
lines+=['', '## Per-row source and comparability','']
for i,v in enumerate(rows):lines+=['%d. **%s / %s / %s / %s** — %s Source: `%s`.%s'%(i+1,v['experiment'],v['checkpoint'],v['state'],v['evaluation_pool'],v['comparability_note'],v['source'],' Fixed train source: `'+v['training_source']+'`.'if v['training_source']else'')]
lines+=['','## Online training windows (not fixed checkpoint train BA)','','These changing raw-model/sample windows are not substituted into fixed-train BA above. Repeated questions and four correlated G5 views per occurrence make them incomparable with independent2048 checkpoint metrics.','','| Experiment | Epoch/window | Raw online BA % | BCE | Updates | Source |','|---|---|---:|---:|---:|---|']
original=json.loads((R/'mixed-direct-b128-continuation/actual-final-training-diagnostic.json').read_text())
for ep,v in original['epochs'].items():
 for tag in('online_raw','online_raw_last100'):
  w=v[tag];lines.append('| original hard-replay | %s/%s | %.2f | %.6f | %d | `%s#/epochs/%s/%s` |'%(ep,tag,100*w['MixedBA'],w['binary_Mixed_BCE'],w['updates'],R/'mixed-direct-b128-continuation/actual-final-training-diagnostic.json',ep,tag))
low=json.loads((R/'mixed-direct-e1-lr-headonly-results/actual-independent-metrics.json').read_text())
for ep,values in low['training_curve']['phases'].items():
 for tag,w in values.items():lines.append('| 747723 old hard-replay LR/head-only | E%s/%s | %.2f | %.6f | %d | `%s#/training_curve/phases/%s/%s` |'%(ep,tag,100*w['Mixed_BA'],w['binaryMixedBCE'],w['updates'],R/'mixed-direct-e1-lr-headonly-results/actual-independent-metrics.json',ep,tag))
(OUT/'direct-mixed-and-sampling.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(rows=len(rows),experiments=sorted({v['experiment']for v in rows}))))
