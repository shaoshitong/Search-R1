"""Prepare or inspect eight-card code/environment only; no training action."""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

from .control import REPO, OUT, PHYSICAL as STOPPED, remote

PHYSICAL='/mnt/data/shaoshitong/search-g1-prefix-selection200-reference-replica-eight-prepared-20261018'
CANONICAL='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'+PHYSICAL.rsplit('/',1)[1]
REFERENCE='c5f1a07e6a9f4ccee35f45ddf69ab4aa3d28e910'
BASELINE='b52590d0a068b7d56d92908adc138aba0180b9cd'

REMOTE=r'''
import base64,gzip,hashlib,io,json,os,pathlib,subprocess,tarfile,time,uuid
p=pathlib.Path(PHYSICAL);old=pathlib.Path(STOPPED);canonical=CANONICAL
def read(f):return json.loads(f.read_text()) if f.exists() else None
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def same(who):
 q=pathlib.Path('/proc')/str(who['pid'])
 if not q.exists():return False
 fields=(q/'stat').read_text().rsplit(')',1)[1].split()
 return fields[0]!='Z' and int(fields[19])==who['start_ticks'] and int(fields[3])==who['sid'] and q.stat().st_uid==who['uid']
old_receipt=read(old/'preparation/submit-receipt.json');ledger=read(old/'logs/process-ledger.json')
live=[who for job in ledger['jobs'].values() for who in job.get('ledger',{}).values() if same(who)]
assert not same(old_receipt['leader']) and not live,'Own four-card predecessor still executing'
cleanup=read(old/'logs/cleanup-status.json');assert cleanup and cleanup['cleaned']
if ACTION=='stage' and not p.exists():
 p.mkdir();(p/'repo').mkdir();(p/'preparation').mkdir();(p/'logs').mkdir()
 with (p/'preparation/stage-intent.json').open('x')as f:json.dump(dict(source_commit=COMMIT,created_unix=time.time(),preparation_only=True),f)
 with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(ARCHIVE))))as tf:
  for member in tf.getmembers():
   assert not member.name.startswith('/') and '..'not in pathlib.Path(member.name).parts
   assert member.isfile()or member.isdir()
  tf.extractall(p/'repo')
 files=json.loads(gzip.decompress(base64.b64decode(MANIFEST)))
 for row in files:
  relative=pathlib.Path(row['path']);assert not relative.is_absolute()and'..'not in relative.parts
  target=p/'repo'/relative
  if not target.exists():
   source=old/'repo'/relative;assert source.is_file()and not source.is_symlink()
   data=source.read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256'],str(relative)
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  assert target.is_file()and not target.is_symlink()and sha(target)==row['sha256'],str(relative)
 config=read(old/'preparation/launch-config.json');oldbase=config['base']
 def rebase(x):
  if isinstance(x,dict):return{k:rebase(v)for k,v in x.items()}
  if isinstance(x,list):return[rebase(v)for v in x]
  if isinstance(x,str)and(x==oldbase or x.startswith(oldbase+'/')):return canonical+x[len(oldbase):]
  return x
 config=rebase(config)
 for key in ('startup_sampling_resume','node_migration','resume_import','resume_joint','resume_pending_probe','source_compatibility','source_compatibility_sha256','b_progressive'):
  config.pop(key,None)
 inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
 mapping={int(row.split(',')[0]):row.split(',')[1].strip()for row in inventory.splitlines()}
 cards=list(range(8));uuids=[mapping[i]for i in cards];run=uuid.uuid4().hex
 config.update(base=canonical,run_id=run,execution_id=uuid.uuid4().hex,source_commit=COMMIT,
  actor_version=run+'-unassigned',world_size=8,physical_gpu_indices=cards,auxiliary_gpu_indices=cards,
  physical_gpu_uuids=uuids,auxiliary_gpu_uuids=uuids,execution_hardware_layout='h100_eight_rank_colocated_prepared_v1',
  retrieval_gpu_shards=8,grounding_physical_gpu=7,grounding_physical_gpu_by_rank=cards,grounding_refit_visible_device=7,
  actor_global_micro_batch_size=16,expected_execution_node='h100-10',retrieval_port=19610,
  search_url='http://127.0.0.1:19610/retrieve',grounding_probe_base_port=19620,
  preparation_only=True,experiment_assignment_pending=True,training_authorized=False,
  environment_only_8gpu=True,reference_B_snapshot_commit=REFERENCE,
  initialization_assignment_pending=True,checkpoint_imported=False,
  progressive_B_available=dict(prefix_tokens=[20,40,80],ordinary_continuation_questions=[768,1024,1536],
   candidate_questions=5120,selected_questions=512,reference_resume_outer=50))
 (p/'preparation/launch-config.json').write_text(json.dumps(config,sort_keys=True))
 (p/'preparation/source-full-sha.json').write_text(json.dumps(dict(commit=COMMIT,reference_B_commit=REFERENCE,files=files)))
 (p/'preparation/human-preparation-only.json').write_text(json.dumps(AUTH,ensure_ascii=False,indent=2))
 (p/'preparation/stage-complete.json').write_text(json.dumps(dict(source_commit=COMMIT,files=len(files),prepared_unix=time.time(),training_started=False)))
assert (p/'preparation/stage-complete.json').exists(),'Partial preparation must be reconciled, never retried blindly'
for name in ('submit-intent.json','submit-receipt.json'):
 assert not(p/'preparation'/name).exists(),'Preparation unexpectedly submitted'
assert not(p/'actor').exists()and not(p/'probe').exists()and not(p/'question-pool').exists()
manifest=read(p/'preparation/source-full-sha.json')
for row in manifest['files']:assert sha(p/'repo'/row['path'])==row['sha256'],row['path']
wrapper=p/'repo/reproduction/operations/b_replica_retained18/private_paths.sh'
command=['unshare','--mount','--propagation','private','bash',str(wrapper),'/usr/bin/env',
 'SEARCH_G1_PHYSICAL_WORLD_SIZE=8','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',
 'PYTHONPATH='+canonical+'/repo','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1',
 'HF_HOME='+canonical+'/cache/hf','XDG_CACHE_HOME='+canonical+'/cache',
 '/dataset/shaoshitong/Search-G1-common8/env/bin/python','-m',
 'reproduction.operations.b_replica_retained18.environment','--config',canonical+'/preparation/launch-config.json']
result=subprocess.run(command,text=True,capture_output=True,timeout=120)
assert result.returncode==0,dict(stdout=result.stdout,stderr=result.stderr)
validation=json.loads(result.stdout)
receipt=dict(status='EIGHT_GPU_ENVIRONMENT_PREPARED_NOT_STARTED',checked_unix=time.time(),physical=str(p),canonical=canonical,
 source_commit=manifest['commit'],reference_B_commit=manifest['reference_B_commit'],source_files=len(manifest['files']),
 config_sha256=sha(p/'preparation/launch-config.json'),validation=validation,
 predecessor_stopped=True,predecessor_joint=(read(old/'latest-joint.json')or{}).get('outer'),predecessor_cleanup=cleanup,
 training_started=False,submit_intent_exists=False,checkpoint_imported=False,
 model_or_environment_installed=False,large_assets_transferred=False,new_gpu_allocation=False,
 source_code_mutable_hardlinks=False)
(p/'preparation/environment-inspection-'+str(time.time_ns())+'.json').write_text(json.dumps(receipt))
print(json.dumps(dict(receipt=receipt,config=read(p/'preparation/launch-config.json'))))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('stage','inspect'))
    args=parser.parse_args()
    auth=json.loads((OUT/'HUMAN_PREPARE_ONLY_8GPU.json').read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    archive=b'';files=[]
    if args.action=='stage':
        if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
            raise ValueError('Publish clean source before remote preparation')
        published=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/search-g1-b-replica-20261018'],cwd=REPO,text=True).split()[0]
        if published!=commit:raise ValueError('Latest source not pushed')
        tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().rstrip('\0').split('\0')
        changed=set(subprocess.check_output(['git','diff','--name-only',BASELINE,commit],cwd=REPO,text=True).splitlines())
        files=[dict(path=name,sha256=hashlib.sha256((REPO/name).read_bytes()).hexdigest())for name in tracked]
        buf=io.BytesIO()
        with tarfile.open(fileobj=buf,mode='w')as tf:
            for name in tracked:
                if name in changed:tf.add(REPO/name,arcname=name,recursive=False)
        archive=gzip.compress(buf.getvalue(),compresslevel=6)
    script=REMOTE
    values=dict(PHYSICAL=PHYSICAL,CANONICAL=CANONICAL,STOPPED=STOPPED,ACTION=args.action,COMMIT=commit,
        REFERENCE=REFERENCE,AUTH=auth,ARCHIVE=base64.b64encode(archive).decode(),
        MANIFEST=base64.b64encode(gzip.compress(json.dumps(files).encode())).decode())
    script='\n'.join(name+'='+repr(value)for name,value in values.items())+'\n'+script
    compile(script,'eight-card-prepare-'+args.action,'exec')
    evidence=OUT/('eight-'+args.action+'-'+str(time.time_ns())+'-transport.json')
    result=json.loads(remote(script,evidence,target='h100-10',established_controller=True))
    output=OUT/('EIGHT_ENVIRONMENT_'+args.action.upper()+'.json');output.write_text(json.dumps(result['receipt'],indent=2))
    (OUT/'EIGHT_PREPARED_CONFIG.json').write_text(json.dumps(result['config'],indent=2))
    print(json.dumps(dict(evidence=str(output),receipt=result['receipt'])))


if __name__=='__main__':main()
