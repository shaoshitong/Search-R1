from pathlib import Path
import json,gzip,hashlib,math,importlib.util,subprocess,time,collections
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-frozen-calibration-scores-v2');pub=json.loads((b/'source-publication.json').read_text());spec=importlib.util.spec_from_file_location('frozen_score',str(Path(pub['code_root'])/'score_peer_g10_calibration.py'));s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);c=s.read(b/'config.json')
if not (b/'complete.json').exists():print(json.dumps(dict(status='WAITING_BOTH_ACTUAL_FROZEN_SCORING_COMPLETES')));raise SystemExit(0)
done=s.read(b/'complete.json');assert done['status']==s.STATUS and done['normal_exit'] and done['role']==s.ROLE and done['groups']==c['groups'] and done['actor_updates']==done['probe_updates']==0
groups=s.read(s.checked(c['groups']));identity=s.identities(groups);assert identity==done['identities']==c['identities'];assert [r['name'] for r in c['checkpoints']]==['E4','peer256']
train=s.read(s.TRAIN/'complete.json');latest=s.read(s.TRAIN/'latest.json');assert train['cursor']==256 and train['steps']==10720 and train['checkpoint']==latest and latest['sha256']==s.CP_SHA
baseline=s.read(s.TRAIN/'E4-same-HF-validation/complete.json');expected={'E4':baseline['checkpoints']['E4']['ema_sha256'],'peer256':latest['accepted_hashes']['ema']};models={}
for name in ('E4','peer256'):
 r=done['checkpoints'][name];assert r==s.read(b/name/'complete.json') and r['ema_sha256']==expected[name] and r['parameters_unchanged'] and r['optimizer_constructed'] is False and r['actor_updates']==r['probe_updates']==0 and r['config']==s.proof(b/'config.json')
 pred=s.checked(r['predictions']);verified=s.verify_predictions(pred,groups);stats={k:dict(questions=0,p_sum=0.,q10_sum=0.,correct_p_sum=0.,correct_count=0,wrong_p_sum=0.,wrong_count=0) for k in ('allwrong','mixed','allcorrect')};K=collections.Counter();bce=0.;rows=0
 with gzip.open(pred,'rt') as stream:
  for group,line in zip(groups,stream):
   row=json.loads(line);k=sum(row['strict_em']);K[k]+=1;tag='allwrong' if k==0 else 'allcorrect' if k==10 else 'mixed';st=stats[tag];st['questions']+=1;st['p_sum']+=sum(row['p_j'])/10;st['q10_sum']+=row['predicted_mixed_G10'];rows+=1
   for z,p,y in zip(row['logits'],row['p_j'],row['strict_em']):
    bce+=max(z,0)-z*y+math.log1p(math.exp(-abs(z)));st['correct_p_sum' if y else 'wrong_p_sum']+=p;st['correct_count' if y else 'wrong_count']+=1
 assert rows==8192
 for st in stats.values():
  st['mean_p_j']=st.pop('p_sum')/st['questions'] if st['questions'] else None;st['mean_q_G10']=st.pop('q10_sum')/st['questions'] if st['questions'] else None
  for tag in ('correct','wrong'):st['mean_p_'+tag]=st.pop(tag+'_p_sum')/st[tag+'_count'] if st[tag+'_count'] else None
 models[name]=dict(predictions=r['predictions'],ema_sha256=r['ema_sha256'],verification=verified,training_calibration_only=True,trajectory_bce=bce/81920,G10_K=dict(K),classwise=stats,timing=r['timing'])
jid=(b/'job.id').read_text().strip();p=subprocess.run(['sacct','-j',jid,'-n','-P','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25);assert p.returncode==0;actual=p.stdout.strip();job=next(x.split('|') for x in actual.splitlines() if x.split('|')[0]==jid)
result=dict(status='BOTH_FROZEN_G10_MODELS_OWN_EM_LOGITS_SIGMOID_8192_IDS_AND_FILES_INDEPENDENTLY_VERIFIED',job_id=jid,sacct=actual,terminal_normal_exit=job[1:3]==['COMPLETED','0:0'],config=s.proof(b/'config.json'),complete=s.proof(b/'complete.json'),groups=c['groups'],identities=identity,models=models,peer_training_unchanged_cursor=latest['cursor'],peer_training_unchanged_Adam_steps=latest['steps'],no_optimizer_created=True,actor_updates=0,probe_updates=0,evaluation_or_final_scored=False,unseen_performance_claim=False,time=time.time());s.atomic(b/'independent-score-audit.json',result)
if result['terminal_normal_exit']:
 sources=[]
 for name in ('E4','peer256'):
  external=baseline['checkpoints']['E4']['predictions'] if name=='E4' else train['metrics']['selection2048FreshG5HF']['predictions']
  fresh=s.proof(external['path']);assert fresh['sha256']==external['sha256']
  sources.append(dict(name=name,predictions=done['checkpoints'][name]['predictions'],external_predictions=fresh,checkpoint=done['checkpoints'][name]['checkpoint'],ema_sha256=done['checkpoints'][name]['ema_sha256']))
 inputs=dict(status='TWO_FROZEN_MODELS_TRAINING_CALIBRATION_INPUTS_ACCEPTED',role=s.ROLE,groups=c['groups'],sources=sources,scoring_complete=s.proof(b/'complete.json'),independent_audit=s.proof(b/'independent-score-audit.json'),config=s.proof(b/'config.json'),seen_training_data=True,external_predictions_newly_generated=False,final_pool_read=False,actor_updates=0,probe_updates=0)
 s.immutable(b/'CPU-compatible-inputs.json',inputs);result['CPU_compatible_inputs']=s.proof(b/'CPU-compatible-inputs.json')
print(json.dumps(result))
