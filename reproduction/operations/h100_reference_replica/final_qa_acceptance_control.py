"""Once launch/status of final-output CPU acceptance; never launch QA/training."""
import argparse
import base64
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from .follow_h10011_ab_normal_boundary import ROOT,PATH_LAYOUTS
from .readonly_management_route import MODES,run_readonly,lan5_argv

INNER=r'''
import base64,hashlib,json,os,subprocess,time
from pathlib import Path
ARMS=__ARMS__
ACTION=__ACTION__
WORKER=__WORKER__
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def birth(receipt):
 try:
  fields=Path('/proc/'+str(receipt['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
  return int(fields[19])==receipt['start_ticks'] and fields[0]!='Z'
 except FileNotFoundError:return False
result={}
for arm,entry in ARMS.items():
 b=Path(entry['base']);prep=b/'preparation';proof=prep/'final51713-original-output-cpu-once.json'
 intent=prep/'final-qa-cpu-control-intent.json';receipt=prep/'final-qa-cpu-control-receipt.json';log=prep/'final-qa-cpu-control.log'
 value=dict(base=str(b),run_id=entry['run_id'])
 if proof.exists():
  saved=json.loads(proof.read_text())
  if saved['run_id']!=entry['run_id'] or saved['evaluation_config_sha256']!=sha(prep/'final-evaluation-config.json') or saved['final_receipt_sha256']!=sha(b/'final-evaluation/complete.json'):
   raise ValueError('Existing final QA proof changed binding')
  value.update(status='EXISTING_FINAL_QA_CPU_PROOF_NO_REPLAY',proof=str(proof),proof_sha256=sha(proof),result=saved,
   cpu_worker_same_birth=birth(json.loads(receipt.read_text()))if receipt.exists()else None)
 elif receipt.exists():
  saved=json.loads(receipt.read_text());value.update(status='EXISTING_CPU_RECEIPT_NO_REPLAY',receipt=saved,same_birth=birth(saved))
 elif intent.exists() or proof.with_suffix('.intent.json').exists():value['status']='EXISTING_INTENT_RECONCILE_NO_REPLAY'
 elif any(not (b/name).is_file()for name in ('logs/pipeline-complete.json','latest-joint.json','final-evaluation/complete.json','preparation/final-evaluation-config.json')):
  value['status']='PENDING_FINAL_OUTPUT_NOT_COMMITTED_NO_AUDIT'
 elif ACTION=='status':value['status']='FINAL_OUTPUT_COMMITTED_CPU_ACCEPTANCE_PENDING'
 else:
  raw=(prep/'launch-config.json').read_bytes();config=json.loads(raw)
  if hashlib.sha256(raw).hexdigest()!=entry['config_sha256'] or config['run_id']!=entry['run_id']:raise ValueError('Actual frozen launch binding differs')
  payload=base64.b64decode(WORKER['base64']);checksum=hashlib.sha256(payload).hexdigest()
  if checksum!=WORKER['sha256']:raise ValueError('CPU helper source SHA differs')
  worker=prep/'final-qa-tools'/checksum/'final_qa_acceptance.py';worker.parent.mkdir(parents=True,exist_ok=True)
  if not worker.exists():
   with worker.open('xb')as f:f.write(payload)
  if sha(worker)!=checksum:raise ValueError('Staged CPU helper SHA differs')
  claim=dict(run_id=entry['run_id'],created_unix=time.time(),cpu_only=True,source=str(worker),source_sha256=checksum,
   source_runtime_commit=entry['commit'],frozen_repo_changed=False,transport_auto_retry=False,whole_goal_complete=False)
  with intent.open('x')as f:json.dump(claim,f)
  env=dict(os.environ,PYTHONPATH=str(b/'repo'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
   OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
  command=['/dataset/shaoshitong/Search-G1-common8/env/bin/python',str(worker),'--base',str(b)]
  with log.open('ab')as f:process=subprocess.Popen(command,cwd=b/'repo',env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  fields=Path('/proc/'+str(process.pid)+'/stat').read_text().rsplit(')',1)[1].split()
  saved=dict(claim,status='FINAL_CPU_ACCEPTANCE_STARTED_NOT_PASS',pid=process.pid,start_ticks=int(fields[19]),command=command,proof=str(proof),log=str(log))
  with receipt.open('x')as f:json.dump(saved,f)
  value.update(status=saved['status'],receipt=saved)
 if log.exists():
  with log.open('rb')as f:f.seek(max(0,log.stat().st_size-2500));value['log_tail']=f.read().decode(errors='replace')
 result[arm]=value
print(json.dumps(result))
'''


def remote_inner(arms,action):
    if action not in ('start','status'):raise ValueError('Explicit start/status required')
    payload=Path(__file__).with_name('final_qa_acceptance.py').read_bytes()
    values={'__ARMS__':repr(arms),'__ACTION__':repr(action),'__WORKER__':repr(dict(base64=base64.b64encode(payload).decode(),sha256=hashlib.sha256(payload).hexdigest()))}
    return re.sub('|'.join(map(re.escape,values)),lambda match:values[match.group()],INNER)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('start','status'));p.add_argument('--arm',choices=('A','B','both'),default='both');p.add_argument('--transport',choices=MODES,default='retained18');p.add_argument('--path-layout',choices=tuple(PATH_LAYOUTS),default='retained-cache-v2');a=p.parse_args()
    if a.action=='start'and a.transport=='retained18_fallback_lan5':p.error('Once start cannot auto-retry; query status after uncertain outcomes')
    arms=json.loads((ROOT/'work/prefix5-ab-assigned-arms.json').read_text());arms={k:v for k,v in arms.items()if a.arm=='both'or k==a.arm}
    if not all(v['node']=='h100-11'for v in arms.values()):raise ValueError('Current qualified h100-11 placement only')
    inner=remote_inner(arms,a.action)
    script='import subprocess,os\np=subprocess.run('+repr(['unshare','--mount','--propagation','private','bash',PATH_LAYOUTS[a.path_layout],'/usr/bin/python3','-c',inner])+',env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),text=True,capture_output=True,timeout=40)\nassert p.returncode==0,p.stderr\nprint(p.stdout)'
    spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
    tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');evidence=ROOT/('work/prefix5-ab-final-qa-cpu-'+a.action+'-'+tag+'.json')
    if a.action=='status':result,route=run_readonly(ctl,'h100-11','/usr/bin/python3 -',input_text=script,timeout=60,mode=a.transport)
    else:
        argv=lan5_argv(ctl,'h100-11','/usr/bin/python3 -')if a.transport=='lan5'else ctl.route('h100-11','/usr/bin/python3 -')
        try:result=subprocess.run(argv,input=script,text=True,capture_output=True,timeout=60)
        except subprocess.TimeoutExpired:
            evidence.write_text(json.dumps(dict(status='UNCERTAIN_START_QUERY_EXISTING_ONCE_STATE_DO_NOT_REPLAY',training_modified=False))+'\n');raise
        route=dict(route=a.transport,auto_retry=False)
    value=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,transport=route,training_modified=False)
    evidence.write_text(json.dumps(value,indent=2)+'\n')
    if result.returncode:raise RuntimeError(str(evidence)+' '+result.stderr[-2500:])
    print(json.dumps(dict(evidence=str(evidence),result=json.loads(result.stdout))))


if __name__=='__main__':main()
