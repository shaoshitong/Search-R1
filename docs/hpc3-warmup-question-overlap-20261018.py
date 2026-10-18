"""Exact question-identity overlap with EMA5's real warmup groups, metadata only.

Only canonical row indices and hashes travel between controls. No hidden states,
weights, raw trajectories or environment packages leave either remote machine.
"""
import json,shlex,subprocess,sys,time
from pathlib import Path
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import PYTHON,OUTPUT
TARGET=r'''
import json
from pathlib import Path
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r3');c=json.loads((b/'preparation/launch-config.json').read_text())
roots={'50':b/'probe-refreshes/round-000050/outers/outer-000050','60':b/'outers/outer-000060'}
print(json.dumps(dict(canonical_train_sha256=c['canonical_train_sha256'],source_question_pool_sha256=c['source_question_pool_sha256'],fixed_test_identity_sha256=c['fixed_test_identity_sha256'],feature_manifest_sha256=c['fixed_test_manifest_sha256'],round_indices={n:[q['source_index'] for q in json.loads((r/'questions.json').read_text())['rows']] for n,r in roots.items()})))
'''
SOURCE=r'''
import json,hashlib,time,socket,os
from pathlib import Path
from collections import Counter
started=time.time();target=TARGET_METADATA
base=Path('/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-54ac33dcebe64e62a11fe4177f41f116')
def rd(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=rd(base/'preparation/launch-config.json')
assert c['canonical_train_sha256']==target['canonical_train_sha256']
assert c['source_question_pool_sha256']==target['source_question_pool_sha256'] and sha(c['source_question_pool'])==target['source_question_pool_sha256']
assert c['fixed_test_identity_sha256']==target['fixed_test_identity_sha256'] and sha(c['fixed_test_identity_path'])==target['fixed_test_identity_sha256']
feature=Path(c['fixed_test_features'])/'manifest.json';assert sha(feature)==target['feature_manifest_sha256'];manifest=rd(feature)
train=Counter();test=Counter();events=Counter();shard_proofs=[]
for item in manifest['shards']:
 p=Path(item['path']);raw=(p/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==item['manifest_sha256'];m=json.loads(raw)
 meta=m['files']['groups.json'];raw=(p/'groups.json').read_bytes();assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256'];groups=json.loads(raw);assert len(groups)==item['groups']
 for g in groups:
  assert g['split'] in ('train','test') and len(g['normalized_question_hash'])==64
  (train if g['split']=='train' else test)[g['normalized_question_hash']]+=1;events[g['split']]+=1
 shard_proofs.append(dict(path=str(p/'groups.json'),sha256=meta['sha256'],bytes=meta['bytes']))
assert events=={'train':445440,'test':512}
pool=rd(c['source_question_pool']);byindex={r['source_index']:r['normalized_question_hash'] for r in pool['records']}
fixed={r['normalized_question_hash'] for r in rd(c['fixed_test_identity_path'])['records']};assert set(test)==fixed
rounds={}
for n,indices in target['round_indices'].items():
 assert len(indices)==len(set(indices))==2560 and all(i in byindex for i in indices)
 matched=[train[byindex[i]] for i in indices]
 rounds[n]=dict(current_fit_questions=2560,seen_in_EMA5_warmup=sum(x>0 for x in matched),unseen_in_EMA5_warmup=sum(x==0 for x in matched),seen_fraction=sum(x>0 for x in matched)/2560,mean_warmup_G5_events_per_current_question=sum(matched)/2560)
result=dict(status='EXACT_WARMUP_QUESTION_MEMBERSHIP_AUDIT_PASS',checked_unix=time.time(),host=socket.gethostname(),source_feature_manifest_sha256=sha(feature),source_groups_files=len(shard_proofs),source_group_metadata_total_bytes=sum(r['bytes'] for r in shard_proofs),source_groups_metadata_manifest_sha256=hashlib.sha256(json.dumps(shard_proofs,sort_keys=True,separators=(',',':')).encode()).hexdigest(),warmup_train_G5_events=events['train'],warmup_train_unique_normalized_questions=len(train),warmup_test_G5_events=events['test'],warmup_test_unique_normalized_questions=len(test),train_test_question_overlap=len(set(train)&set(test)),eligible_source_pool_questions=len(byindex),eligible_source_pool_seen_in_warmup=sum(train[r['normalized_question_hash']]>0 for r in pool['records']),rounds=rounds,fixed_diagnostic_questions_seen_in_warmup_train=len(fixed&set(train)),same_canonical_train_sha256=target['canonical_train_sha256'],source_question_pool_sha256=target['source_question_pool_sha256'],fixed_identity_sha256=target['fixed_test_identity_sha256'],new_rollouts=0,tensor_payloads_read=0,elapsed_seconds=time.time()-started)
print(json.dumps(result))
'''

def main():
    addr=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip()
    r=run(PYTHON+' -c '+shlex.quote(TARGET),account='mou176',source_address=addr,timeout=25)
    if r.returncode:raise RuntimeError('HPC3 metadata read failed: '+r.stderr.decode()[-500:])
    target=json.loads(r.stdout)
    code=SOURCE.replace('TARGET_METADATA',repr(target))
    root=Path('/Users/shaoshitong/Documents/remote_codex_control')
    r=subprocess.run([sys.executable,str(root/'control.py'),'run','baai-a800','--','python3','-c',code],cwd=root,text=True,capture_output=True,timeout=150)
    if r.returncode:raise RuntimeError('Source metadata read failed: '+r.stderr[-700:]+r.stdout[-700:])
    result=next(json.loads(line) for line in reversed(r.stdout.splitlines()) if line.startswith('{'))
    (Path(OUTPUT)/'probe-warmup-current-question-overlap-exact.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
