"""Publish three exact Git CPU-reader files, then read existing scalar logs once.

No training source, tensor, model, GPU, allocation or external task is changed.
An uncertain deployment is inspected rather than automatically replayed.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

W = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/easy2560-co-rollout-hpc3-4gpu-20261017')
OUT = W.parents[1]/'outputs/easy2560-co-rollout-hpc3-4gpu-20261017'
SIGMA = W.parent/'qprobe-sigma3-loss-20261015'
BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730961-20261018'
ORIGINAL = '/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018'
COMMIT = '509c6689fbedf84e833d5afb24f901e89fda9e7e'
NAMES = ('reproduction/easy2560_joint4/__init__.py',
         'reproduction/easy2560_joint4/summarize_saved_telemetry.py',
         'reproduction/easy2560_joint4/test_summarize_saved_telemetry.py')
REMOTE = r'''import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=PAYLOAD
b=Path(p['base']);root=b/'preparation/cpu-saved-telemetry-code'/p['commit']
assert os.getuid()==14090 and b.resolve()==b and not root.exists()
assert (b/'job.id').read_text().strip()=='731063'
run=json.loads((b/'logs/run-manifest.json').read_text())
assert run['source_commit']=='a61b48139386b346e4fef12511ece1ac0d09a8e3'
root.mkdir(parents=True)
manifest={}
for name,record in p['files'].items():
 path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
 raw=base64.b64decode(record['base64']);assert hashlib.sha256(raw).hexdigest()==record['sha256'] and len(raw)==record['bytes']
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert path.read_bytes()==raw
 manifest[name]={k:record[k] for k in ('bytes','sha256')}
publication=dict(source_commit=p['commit'],files=manifest,all_deployed_files_fresh_SHA_verified=True,training_source_modified=False,checked_unix=time.time())
with (root/'source-publication.json').open('x') as f:json.dump(publication,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
cmd=[str(b/'env/bin/python'),'-B',str(root/'reproduction/easy2560_joint4/summarize_saved_telemetry.py'),'--base',p['original'],'--base',str(b),'--through-outer',str(p['through'])]
started=time.time();child=subprocess.run(cmd,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root)),capture_output=True,text=True,timeout=180)
evidence=b/'preparation/derived-telemetry';evidence.mkdir(exist_ok=True)
receipt=dict(source_commit=p['commit'],source_root=str(root),through_outer=p['through'],started_unix=started,checked_unix=time.time(),returncode=child.returncode,stderr=child.stderr,stdout=child.stdout,training_source_modified=False,new_GPU_allocations=0,model_or_tensor_loading_performed=False)
receipt_path=evidence/('through-%d-%s-execution.json'%(p['through'],p['commit'][:7]))
with receipt_path.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
assert child.returncode==0,child.stderr[-2000:]
report=json.loads(child.stdout);assert report['missing_outers']==[] and report['through_outer']==p['through']
target=evidence/('through-%d-%s.json'%(p['through'],p['commit'][:7]))
with target.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='ACTUAL_SAVED_TELEMETRY_CPU_REPORT_COMPLETE',source_commit=p['commit'],source_root=str(root),publication=publication,report_path=str(target),report_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),execution_receipt=str(receipt_path),returncode=child.returncode,checked_unix=time.time(),report=report,training_modified=False,new_GPU_allocations=0)))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through-outer',type=int,required=True)
    a=parser.parse_args();assert 1<=a.through_outer<=200
    files={}
    for name in NAMES:
        raw=subprocess.check_output(['git','show',COMMIT+':'+name],cwd=W)
        files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
    payload=dict(base=BASE,original=ORIGINAL,commit=COMMIT,through=a.through_outer,files=files)
    code=REMOTE.replace('p=PAYLOAD','p='+repr(payload))
    command=shlex.join([BASE+'/env/bin/python','-B','-c',code])
    source=SIGMA/'reproduction/hpc3_qprobe/local_account_bridge.py'
    spec=importlib.util.spec_from_file_location('existing_account_bridge',source)
    bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    result=bridge.run('jxu957',command,'',str(W.parent),login_host='10.120.48.31',observation_timeout=55,
        expected_host_identity=dict(hostname='ACD-Login-1',boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',pid_namespace='pid:[4026531836]'))
    record=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,operator_source_commit=COMMIT,through_outer=a.through_outer,checked_unix=time.time())
    path=OUT/('actual-telemetry-through-%d-%s-%d.json'%(a.through_outer,COMMIT[:7],int(time.time())))
    with path.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    if result.returncode:print(json.dumps(dict(audit=str(path),**record)));return
    d=json.loads(result.stdout)
    report=d['report']
    print(json.dumps(dict(audit=str(path),status=d['status'],remote_report=d['report_path'],source_commit=COMMIT,
        through_outer=report['through_outer'],missing=report['missing_outers'],rows=[dict(step=x['step'],
            training_EM=x['selected_training_em']['fraction'],mixed=x['selected_mixed_fraction'],
            reward=x['selected_sequence_reward_mean'],Adam=x['retained_adam_writes_per_rank'],
            actor_peak_mib=[g['sampled_peak_used_mib'] for g in x['gpu_memory']['per_gpu']]) for x in report['outers']],training_modified=False)))


if __name__=='__main__':main()
