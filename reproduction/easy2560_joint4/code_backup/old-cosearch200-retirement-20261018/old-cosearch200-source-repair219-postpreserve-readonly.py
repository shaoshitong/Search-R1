import hashlib,json,os,stat,time
from pathlib import Path
code_root=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018/preparation/old-cosearch200-retirement-code/219686d225feb9ba3c94e776c1294384a14d809a');audit=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011/preparation/cp200-tensor-retirement/19ec0e1bc6224d8e85bbcba0b8ba9eb8');dest=audit/'failed-source-attempt-d45-pid110273-start96708929'
raw=(code_root/'retire_old_cosearch200.py').read_bytes();assert hashlib.sha256(raw).hexdigest()=='fbeec1fba246222622b07c681ede4dec3d0c04992581ec2f35e5c269ffb15b42'
scope={'__name__':'read_only_operator','__file__':str(code_root/'retire_old_cosearch200.py')};exec(compile(raw,scope['__file__'],'exec'),scope)
result=json.loads((dest/'preserved.json').read_text());moved=[]
for name in result['confirmed_moved_names']:
 p=dest/name;record=scope['identity'](p);expected=result['files'][name];assert record=={k:v for k,v in expected.items() if k!='sha256'} and hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256'];assert not os.path.lexists(audit/name);moved.append(name)
assert scope['identity'](audit/'plan.json')=={k:v for k,v in result['files']['plan.json'].items() if k!='sha256'};assert hashlib.sha256((audit/'plan.json').read_bytes()).hexdigest()==result['retained_plan_sha256']
for name in ('deletion-claim.json','actions.jsonl','deletion-result.json','deletion-failure.json','complete.json','source-validation.json','progress.json'):assert not os.path.lexists(audit/name)
plan=scope['load_plan'](code_root/'candidate.json');core={}
for name,row in plan['core'].items():
 p=Path(plan['base'])/'checkpoints/global_step_200'/name;r=scope['identity'](p);assert r['bytes']==row['bytes'] and r['mtime_ns']==row['source_mtime_ns'] and r['uid']==21930 and r['links']==1;core[name]=r
print(json.dumps({'checked_unix':time.time(),'uid':os.getuid(),'six_original_audit_files_preserved_byte_and_identity_equal':moved,'retained_plan_sha256':result['retained_plan_sha256'],'checkpoint33_stat_matches_pinned_candidate':len(core),'checkpoint33_identity':core,'checkpoint_tensor_contents_read':False,'deletion_not_started':True,'new_source_claim_absent':True,'operator_sha256':hashlib.sha256(raw).hexdigest()},sort_keys=True))
