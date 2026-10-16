"""Published code-only deployment; every mutation has a distinct intent/receipt."""
import argparse,base64,hashlib,io,json,subprocess,tarfile,time
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('deploy','submit','query','observe'))
    p.add_argument('--base',required=True);p.add_argument('--commit');p.add_argument('--config');p.add_argument('--receipts',required=True)
    a=p.parse_args()
    if a.phase=='deploy':
        archive=subprocess.check_output(['git','-C',str(REPO),'archive','--format=tar',a.commit])
        with tarfile.open(fileobj=io.BytesIO(archive))as t:
            files={x.name:hashlib.sha256(t.extractfile(x).read()).hexdigest()for x in t if x.isfile()}
        c=json.loads(Path(a.config).read_text());c.update(base=a.base,source_commit=a.commit)
        payload=dict(base=a.base,config=c,commit=a.commit,files=files,archive=base64.b64encode(archive).decode())
        code=r'''
import base64,hashlib,io,json,os,pathlib,sys,tarfile,time
p=PAYLOAD;b=pathlib.Path(p['base']);c=p['config']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(p,x):
 with p.open('x')as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
assert str(b).startswith('/mnt/data/shaoshitong/search-g1-question20-replay70-')
if b.exists():
 print(json.dumps(dict(status='EXISTING_DEPLOY_NAMESPACE_NO_REPLAY',intent=(b/'preparation/deploy-intent.json').exists(),receipt=read(b/'preparation/deploy-receipt.json')if(b/'preparation/deploy-receipt.json').exists()else None)));raise SystemExit
evidence=pathlib.Path(c['source_freeze_proof_path']);assert sha(evidence)==c['source_freeze_proof_sha256']
source=read(evidence)
assert source['source_epoch']==c['source_epoch']and source['source_checkpoint_sha256']==c['source_checkpoint_sha256']
assert source['owned_births_present']==0 and source['native_cgroup_exists']is False
assert source['actual_adam_steps_per_rank']==c['source_adam_steps_per_rank']
assert c['iterations']==70 and c['global_batch']==256 and c['learning_rate']==3e-6
b.mkdir(mode=0o700);prep=b/'preparation';prep.mkdir();(b/'repo').mkdir();(b/'logs').mkdir()
once(prep/'deploy-intent.json',dict(commit=p['commit'],run_id=c['run_id'],file_count=len(p['files']),created_unix=time.time()))
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(p['archive'])))as t:
 for x in t:
  path=pathlib.PurePosixPath(x.name);assert not path.is_absolute()and'..'not in path.parts
  if x.isdir():(b/'repo'/x.name).mkdir(parents=True,exist_ok=True)
  elif x.isfile():
   q=b/'repo'/x.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(t.extractfile(x).read());q.chmod(x.mode)
for name,digest in p['files'].items():assert sha(b/'repo'/name)==digest,name
sources=[]
for item in source['replay_sources']:
 d=dict(split='train',epoch=item['source_data_epoch'])
 if item['kind']=='cached_inputs':d['inputs']=item['inputs']
 else:d.update(rollout_complete=item['rollout_complete'],sha256=item['sha256'])
 sources.append(d)
sources.append(dict(inputs=c['cached_inputs'],split='test'))
once(prep/'replay-sources.json',dict(sources=sources))
once(prep/'launch-config.json',c);once(prep/'source-files.json',p['files'])
result=dict(status='QUESTION20_REPLAY70_CODE_DEPLOYED_NOT_SUBMITTED',base=str(b),run_id=c['run_id'],source_commit=p['commit'],target_code_files=len(p['files']),all_target_code_full_sha_match=True,config_sha256=sha(prep/'launch-config.json'),source_epoch=c['source_epoch'],source_checkpoint_sha256=c['source_checkpoint_sha256'],source_adam_steps_per_rank=c['source_adam_steps_per_rank'],learning_rate=c['learning_rate'],new_rollouts=0,large_payload_transferred_from_Mac=False)
once(prep/'deploy-receipt.json',result);print(json.dumps(result))
'''.replace('PAYLOAD',repr(payload))
    else:
        code='import sys,json;sys.path.insert(0,'+repr(a.base+'/repo')+');from reproduction.prefix20_question_replay.launch import '+a.phase+';print(json.dumps('+a.phase+'('+repr(a.base)+'),allow_nan=False))'
    from reproduction.h1008_qprobe.management_transport import remote
    path=Path(a.receipts)/('question20-replay70-'+a.phase+'-'+str(time.time_ns())+'.json')
    out=json.loads(remote(code,path,control_path=None,target='h100-8'))
    result=path.with_name(path.stem+'-result.json');result.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(path=str(result),sha256=hashlib.sha256(result.read_bytes()).hexdigest(),result=out),allow_nan=False))

if __name__=='__main__':main()
