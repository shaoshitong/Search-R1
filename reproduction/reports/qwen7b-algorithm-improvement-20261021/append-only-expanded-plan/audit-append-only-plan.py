import hashlib,json,pathlib,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-expanded-append-only-plan-v1')
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(r):
 if fp(r['path'])!=r:raise ValueError('Source receipt bytes/SHA changed')
 return read(r['path'])
complete=fp(B/'complete.json');assert complete['sha256']=='0239d04138aafe4be2dceab0cfe5f0a7c4b91aa3627868adc860d1d3772cac7e'
v=read(B/'complete.json');plan=checked(v['plan']);old=checked(v['old_plan']);identity=checked(v['identity_plan'])
assert plan['batches'][:256]==old['batches'] and plan['batch_epochs'][:256]==old['batch_epochs']
assert plan['cohort_qids']==[g['qid'] for g in identity['groups']]
before=[q for b in plan['batches'][:256] for q in b];after=[q for b in plan['batches'][256:] for q in b]
assert len(before)==len(set(before))==8192 and len(after)==len(set(after))==21824
assert set(before).isdisjoint(after) and set(before+after)==set(plan['cohort_qids'])
assert len(plan['batches'])==938 and all(len(b)==32 for b in plan['batches']) and plan['batch_epochs']==[0]*938
assert v['source_Adam']==10720 and v['target_Adam']==11402 and v['new_updates']==682
assert not v['labels_or_scores_read'] and not v['model_tensors_read'] and not v['source_checkpoint_and_loss_recipe_selected'] and not v['probe_training_submitted']
result=dict(status='APPEND_ONLY_PLAN_INDEPENDENT_OLD256_PRESERVATION_NEW682_DISJOINT_FULL_COVERAGE_SHA_PASS',complete=complete,plan=v['plan'],source_commit=v['source_commit'],historical_questions=8192,new_questions=21824,cumulative_questions=30016,new_updates=682,cumulative_updates=938,source_Adam=10720,target_Adam=11402,model_or_optimizer_tensors_verified=False,training_submitted=False,final_read=False,time=time.time())
p=B/'independent-plan-audit.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(result,sort_keys=True)+'\n');temp.replace(p)
print(json.dumps(dict(audit=fp(p),**result),sort_keys=True))
