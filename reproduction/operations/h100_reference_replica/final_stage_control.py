"""Observe actual final QA or once-start CPU-only completed-run closure."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from .assigned_cpu_control import execute
from .follow_h10011_ab_normal_boundary import ROOT


REMOTE = r'''
import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
e=ENTRY;b=Path(e['base']);prep=b/'preparation';sys.path.insert(0,str(b/'repo'))
from reproduction.a800.owned_processes import same,identity
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(prep/'launch-config.json')==e['config_sha256']
submitted=read(prep/'submit-receipt.json');assert submitted['run_id']==e['run_id']
v=dict(base=str(b),run_id=e['run_id'],observed_unix=time.time(),
       native_leader=submitted['leader'],native_same_birth=same(submitted['leader']))
for name in ('logs/stage.json','logs/pipeline-status.json','logs/cleanup-status.json','preparation/native-actual-waitpid-exit.json'):
 p=b/name
 if p.exists():v[name]=read(p)
v['pipeline_complete_exists']=(b/'logs/pipeline-complete.json').is_file()
for role,directory in [('qa200',b/'outers/outer-000200/validation'),('final51713',b/'final-evaluation')]:
 value=dict(directory=str(directory),progress=[])
 for rank in range(4):
  p=directory/('rank-%d-progress.json'%rank)
  if p.is_file():value['progress'].append(read(p))
  else:value['progress'].append(None)
 p=directory/'complete.json'
 if p.is_file():
  q=read(p);value['complete']={k:q.get(k)for k in ('run_id','actor_version','qa_count','em','f1','per_source','actual_worker_exit_codes','actual_compute_dtype','question_source_sha256','completed_utc')};value['complete_sha256']=sha(p)
 value['original_output_bytes']={str(rank):(directory/('rank-%d-full.jsonl.gz'%rank)).stat().st_size for rank in range(4)if(directory/('rank-%d-full.jsonl.gz'%rank)).is_file()}
 v[role]=value
proof=prep/'final-completion-retention-owned-release-once.json'
intent=prep/'final-closure-control-intent.json';receipt=prep/'final-closure-control-receipt.json';log=prep/'final-closure-control.log'
if proof.exists():
 r=read(proof);v['closure']=dict(status='EXISTING_FINAL_CLOSURE_NO_REPLAY',proof=str(proof),sha256=sha(proof),summary={k:r[k]for k in ('status','complete_outers','actual_native_returncode','no_outer201','retained_selected_question_groups','retained_selected_trajectories','historical_other_execution_residue_is_separate')})
elif receipt.exists():
 r=read(receipt);v['closure']=dict(status='EXISTING_CLOSURE_RECEIPT_NO_REPLAY',receipt=r,same_birth=same(r['worker']),proof_exists=False)
elif intent.exists()or proof.with_suffix('.intent.json').exists():v['closure']=dict(status='EXISTING_CLOSURE_INTENT_RECONCILE_NO_REPLAY')
elif ACTION=='close':
 required=[prep/'ab-joint-0200-independent-full-sha-cpu.json',prep/'final51713-original-output-cpu-once.json',prep/'native-actual-waitpid-exit.json',b/'logs/pipeline-complete.json']
 if not all(p.is_file()for p in required):v['closure']=dict(status='PENDING_FINAL_PROOFS_OR_NATIVE_TERMINAL_NO_AUDIT')
 else:
  assert not v['native_same_birth'] and read(required[2])['actual_returncode']==0
  payload=base64.b64decode(WORKER['base64']);assert hashlib.sha256(payload).hexdigest()==WORKER['sha256']
  path=prep/'final-closure-tools'/WORKER['sha256']/'final_completion_worker.py';path.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists():
   with path.open('xb')as f:f.write(payload)
  assert sha(path)==WORKER['sha256']
  claim=dict(run_id=e['run_id'],created_unix=time.time(),worker_sha256=WORKER['sha256'],cpu_only=True,frozen_repo_changed=False,transport_auto_retry=False)
  with intent.open('x')as f:json.dump(claim,f)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  command=['/dataset/shaoshitong/Search-G1-common8/env/bin/python',str(path),'--base',str(b),'--run-id',e['run_id'],'--config-sha256',e['config_sha256']]
  with log.open('xb')as f:p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  r=dict(claim,worker=identity(p.pid),command=command,log=str(log),proof=str(proof))
  with receipt.open('x')as f:json.dump(r,f)
  v['closure']=dict(status='CPU_FINAL_CLOSURE_STARTED_NOT_PASS',receipt=r)
if log.exists():
 with log.open('rb')as f:f.seek(max(0,log.stat().st_size-2200));v['closure_log_tail']=f.read().decode(errors='replace')
print(json.dumps({ARM:v}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('observe', 'close'))
    parser.add_argument('--arm', choices=('A', 'B'), default='B')
    parser.add_argument('--transport', choices=('retained18', 'lan5', 'retained18_fallback_lan5'), default='retained18')
    args = parser.parse_args()
    if args.action == 'close' and args.transport == 'retained18_fallback_lan5':
        parser.error('A once CPU start cannot retry transport; query existing receipts')
    entry = json.loads((ROOT / 'work/prefix5-ab-assigned-arms.json').read_text())[args.arm]
    payload = Path(__file__).with_name('final_completion_worker.py').read_bytes()
    prefix = ('ENTRY=' + repr(entry) + '\nARM=' + repr(args.arm) + '\nACTION=' + repr(args.action)
              + '\nWORKER=' + repr(dict(base64=base64.b64encode(payload).decode(), sha256=hashlib.sha256(payload).hexdigest())) + '\n')
    result = execute({args.arm: entry}, lambda _: prefix + REMOTE,
                     'status' if args.action == 'observe' else 'start', args.transport, 'final-stage-' + args.action)
    value = result['arms'][args.arm]
    output = ROOT / ('work/prefix5-ab-final-stage-' + args.arm + '-latest-small.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    # Full original outputs and the large final inventory always remain remote.
    print(json.dumps(dict(output=str(output), evidence=result['evidence'], result=value)))


if __name__ == '__main__':
    main()
