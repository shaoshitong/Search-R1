"""Code-only exact deployment and bounded metadata observation."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

from reproduction.actor50_probe100.control import archive_source,route,ROOT


REMOTE=r'''
import base64,hashlib,io,json,os,pathlib,re,subprocess,sys,tarfile,time,shutil
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=b/'repo'
assert re.fullmatch('[0-9a-f]{32}',h['run_id']) and re.fullmatch('[0-9a-f]{32}',h['execution_id'])
assert str(b)=='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+h['run_id']+'-actor50-prefix20-reuse20'
def read(path):
 path=pathlib.Path(path);assert path.resolve()==path and not path.is_symlink() and path.stat().st_size<64*1024**2
 return json.loads(path.read_text())
def sha(path):
 out=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):out.update(x)
 return out.hexdigest()
def once(path,value):
 with pathlib.Path(path).open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False)
def who(pid):
 try:
  path=pathlib.Path('/proc')/str(pid);raw=(path/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
  return dict(pid=pid,sid=int(v[3]),uid=path.stat().st_uid,start_ticks=int(v[19]),state=v[0])
 except (FileNotFoundError,ProcessLookupError):return None
def same(identity):
 actual=who(identity['pid'])
 return actual if actual and all(actual[k]==identity[k] for k in ('pid','sid','uid','start_ticks')) else None
if p['action']=='stage':
 assert not b.exists()
 parent=pathlib.Path(p['source_handle']['base']);clean=read(parent/'logs/cleanup-status.json')
 assert clean['cleaned'] and clean['owned_births_all_gone'] and same(read(parent/'preparation/submit-receipt.json')['leader']) is None
 cp20=read(parent/'training/cycle-0010/complete.json');assert cp20['completed_epochs']==20 and cp20['ema_updates']==276
 assert read(parent/'probe/latest.json')['completed_epochs']==20
 assert p['human_intent']['scope']['new_prefix_policy_tokens']==20 and p['human_intent']['new_prefix20_epoch_count']==20
 b.mkdir();(b/'preparation').mkdir();(b/'logs').mkdir();repo.mkdir()
 raw=base64.b64decode(p['archive_b64']);assert hashlib.sha256(raw).hexdigest()==h['archive_sha256']
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
  members=archive.getmembers();assert all(not pathlib.PurePosixPath(x.name).is_absolute() and '..' not in pathlib.PurePosixPath(x.name).parts and (x.isfile() or x.isdir()) for x in members)
  archive.extractall(repo,members=members)
 for name,digest in h['source_files'].items():assert sha(repo/name)==digest
 assert shutil.disk_usage(b).free>160*1024**3
 c=read(parent/'preparation/launch-config.json')
 for key in list(c):
  if key.startswith('resume_') or key in ('warmup_optimizer_restored','candidate_schedule'):c.pop(key)
 c.update(base=str(b),run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
  role='frozen_actor50_prefix20_reuse_matched20',prefix_policy_tokens=20,target_probe_epochs=20,cycles=10,
  source_prefix5_base=str(parent),source_prefix5_complete20_sha256=sha(parent/'training/cycle-0010/complete.json'),
  source_prefix5_checkpoint20_sha256=cp20['sha256'],feature_micro_batch=8,
  actor_training_enabled=False,actor_updates=0,new_rollouts=0,probe_learning_rate=5e-6,ema_decay=.99,
  source_model_kind='same_probe50_raw_plus_zero15_head_positions',source_optimizer_imported=False,
  new_optimizer=True,auxiliary_gpu_indices=[],auxiliary_gpu_uuids=[])
 assert c['actor_version']=='54ac33dcebe64e62a11fe4177f41f116-actor0050'
 once(b/'preparation/deployment.json',h);once(b/'preparation/config.json',c);once(b/'preparation/human-intent.json',p['human_intent'])
 value=dict(status='PREFIX20_EXACT_SOURCE_STAGED_NOT_LAUNCHED',source_commit=h['source_commit'],source_files_verified=len(h['source_files']),
  config_sha256=sha(b/'preparation/config.json'),source_prefix5_epochs=20,source_prefix5_Adam_EMA=276,
  source_prefix5_checkpoint20_sha256=cp20['sha256'],output_remote=str(b),free_remote_bytes=shutil.disk_usage(b).free,checked_unix=time.time())
 once(b/'preparation/source-receipt.json',value)
elif p['action']=='submit':
 assert read(b/'preparation/deployment.json')==h
 for name,digest in h['source_files'].items():assert sha(repo/name)==digest
 assert not (b/'preparation/submit-intent.json').exists(),'Onceclaim exists: inspect, do not replay'
 once(b/'preparation/submit-intent.json',dict(run_id=h['run_id'],checked_unix=time.time()))
 c=read(b/'preparation/config.json');env=dict(os.environ,PYTHONPATH=str(repo),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
 command=[str(pathlib.Path(c['environment_root'])/'bin/python'),'-m','reproduction.actor50_prefix20_reuse20.pipeline','--base',str(b)]
 with (b/'logs/pipeline.log').open('xb') as f:child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 identity=who(child.pid);assert identity and identity['sid']==child.pid
 value=dict(status='PREFIX20_ONCE_SUBMITTED_NOT_ACCEPTED',leader=identity,checked_unix=time.time(),command=command)
 once(b/'preparation/submit-receipt.json',value)
else:
 assert read(b/'preparation/deployment.json')==h
 small=lambda path:read(path) if pathlib.Path(path).exists() else None
 value=dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],checked_unix=time.time(),
  leader=same(read(b/'preparation/submit-receipt.json')['leader']),status=small(b/'logs/status.json'),
  latest=small(b/'probe/latest.json'),complete=small(b/'complete.json'),failure=small(b/'logs/failure.json'),
  exit=small(b/'logs/exit.json'),cleanup=small(b/'logs/cleanup-status.json'),allocation=small(b/'logs/native-allocation.json'),
  initial_model=small(b/'preparation/initial-model-qualified.json'))
 datasets=[]
 for n in range(11):
  name='test' if n==0 else 'cycle-%04d'%n;d=b/'data'/name;meta=small(d/'prepared.json')
  if meta:datasets.append(dict(dataset=name,repetition=meta['repetition'],feature_progress=[small(d/('rank-%d-progress.json'%rank)) for rank in range(4)],feature_completed_ranks=sum((d/('rank-%d-complete.json'%rank)).exists() for rank in range(4))))
 value['datasets']=datasets
 if (b/'logs/pipeline.log').exists():
  path=b/'logs/pipeline.log'
  with path.open('rb') as f:f.seek(max(0,path.stat().st_size-4096));value['log_tail']=f.read().decode(errors='replace')
raw=json.dumps(value,allow_nan=False);assert len(raw.encode())<128*1024;print(raw)
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','submit','observe'),required=True)
    p.add_argument('--handle',type=Path,required=True);a=p.parse_args();workdir=Path(__file__).resolve().parents[2]
    payload=dict(action=a.action)
    if a.action=='stage':
        assert not a.handle.exists() and not subprocess.check_output(['git','status','--porcelain'],cwd=workdir).strip()
        archive,files=archive_source(workdir);run=uuid.uuid4().hex;execution=uuid.uuid4().hex
        h=dict(run_id=run,execution_id=execution,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workdir,text=True).strip(),
            base='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+run+'-actor50-prefix20-reuse20',
            source_files=files,archive_sha256=hashlib.sha256(archive).hexdigest(),created_unix=time.time())
        a.handle.write_text(json.dumps(h,indent=2)+'\n')
        payload.update(archive_b64=base64.b64encode(archive).decode(),
            source_handle=json.loads((ROOT/'preparation/actor50-probe100-resume-deployment-20261018.json').read_text()),
            human_intent=json.loads((ROOT/'preparation/actor50-prefix20-human-intent-20261018.json').read_text()))
    else:h=json.loads(a.handle.read_text())
    payload['handle']=h
    r=subprocess.run(route(REMOTE),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    out=ROOT/'preparation'/('actor50-prefix20-'+a.action+'-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode(),checked_unix=time.time()),indent=2)+'\n')
    if r.returncode:raise RuntimeError('Inspect once-operation receipt: '+str(out))
    print(json.dumps(dict(receipt=str(out),result=json.loads(r.stdout))))


if __name__=='__main__':main()
