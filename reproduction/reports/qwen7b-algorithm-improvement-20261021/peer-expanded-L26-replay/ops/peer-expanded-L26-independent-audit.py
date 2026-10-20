"""Independent bank-ownID/EM and source-manifest verification; no tensor IO."""
from pathlib import Path
import json,hashlib,subprocess,time
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');b=A/'peer-expanded-L26-replay-v1';out=b/'bank'
def read(p):return json.loads(Path(p).read_text())
def fp(p):
 p=Path(p);data=p.read_bytes();return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def checked(v):
 a=fp(v['path']);assert a['sha256']==v['sha256'] and ('bytes' not in v or a['bytes']==v['bytes']);return Path(v['path'])
done=read(out/'complete.json');assert done['status']=='EXPANDED_REAL_G10_LAYER26_REUSED8192_PLUS_ADDITIONAL21824_ACCEPTED' and done['questions']==30016 and done['trajectories']==300160
plan=read(checked(done['identity_plan']));assert done['identity_plan']['sha256']=='b2bcc13b542c3efbd90077a94138c02917940f286fde8f6ffeab2abe7542a047'
prior=read(checked(plan['prior_identity']))['groups'];assert plan['groups'][:8192]==prior and plan['groups'][8192:]==plan['remaining_groups'] and len(prior)==8192 and len(plan['remaining_groups'])==21824
raw=read(checked(done['raw_complete']));truth={g['qid']:g for g in read(checked(raw['outputs']['groups']))['train']};groups=read(checked(done['outputs']['groups.json']))['train'];ids=set();hashes=set();k=[0]*11
for g,wanted in zip(groups,plan['groups']):
 assert {key:g[key] for key in ('qid','question_hash','trajectory_ids')}==wanted and json.loads(g['qid'])[0]=='nq' and g['labels']==truth[g['qid']]['labels'] and g['trajectory_ids']==truth[g['qid']]['trajectory_ids']
 assert len(g['labels'])==len(g['trajectory_ids'])==10 and len(set(g['trajectory_ids']))==10 and all(type(y) is int and y in (0,1) for y in g['labels']) and not ids.intersection(g['trajectory_ids']) and g['question_hash'] not in hashes
 ids.update(g['trajectory_ids']);hashes.add(g['question_hash']);k[sum(g['labels'])]+=1
assert len(groups)==len(plan['groups'])==30016 and len(ids)==300160
features=read(checked(done['outputs']['features-layer26.json']));sources=read(checked(done['outputs']['sources.json']));assert len(sources)==8
expected=[];newjobs=[];prior_refs=[]
for source in sources:
 c=read(checked(source['complete']));selection=read(checked(source['selection']));assert c['source_selection']==source['selection'] and c['actor_updates']==c['probe_updates']==0 and c['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'
 assert c['batching']['core_sha256']=='500911347444a07693fb4405a6f6fbbcad7b70375522a52833953b426970f506' and c['batching']['batch_size']==32 and c['batching']['padded_token_budget']==16384
 refs=c['feature_manifests_by_capture_layer']['26'];expected+=refs
 if source['kind']=='prior':prior_refs+=refs;assert selection['question_plan']==plan['prior_identity'] and c['questions']==2048
 else:
  assert selection['question_plan']==done['identity_plan'] and c['questions']==5456 and c['compute_layers']==[20,23,26] and c['persist_layers']==[26] and c['existing8192_recaptured'] is False
  newjobs.append(c['job_id'])
assert expected==features and len({v['path'] for v in features})==len(features)
for item in features:
 m=read(checked(item));assert m['capture_layer']==m['feature_layer']==26 and m['dimension']==3584 and m['prefix_tokens']==20 and m['question_max_tokens']==273
jobs=subprocess.run(['sacct','-X','-n','-P','-j',','.join(newjobs),'--format=JobIDRaw,State,ExitCode,Elapsed'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25);assert jobs.returncode==0
states={row.split('|')[0]:row.split('|')[1:3] for row in jobs.stdout.splitlines() if row};assert all(states[j]==['COMPLETED','0:0'] for j in newjobs)
result=dict(status='INDEPENDENT_30016_REAL_G10_OWN_ID_EM_AND_LAYER26_SOURCE_SHA_AUDIT_PASSED',bank_complete=fp(out/'complete.json'),outputs=done['outputs'],identity_plan=done['identity_plan'],questions=30016,trajectories=300160,prior8192_order_and_features_preserved=True,new_questions=21824,new_trajectories=218240,all_own_EM_equal_canonical_raw=True,all_question_hashes_unique=True,G10_K_histogram=k,manifest_count=len(features),prior_layer26_manifests_reused=len(prior_refs),same_batch32_core_and_golden=True,new_capture_normal_exits=jobs.stdout.strip(),producer_payload_SHA_and_reader_checks_retained=True,independent_manifest_SHA_freshly_verified=True,no_payload_tensor_bulk_rehash_in_this_audit=True,actor_updates=0,probe_updates=0,new_rollouts_generated=0,training_submitted=False,untouched_final_read=False,time=time.time());(out/'independent-input-audit.json').write_text(json.dumps(result,sort_keys=True));print(json.dumps(result))
