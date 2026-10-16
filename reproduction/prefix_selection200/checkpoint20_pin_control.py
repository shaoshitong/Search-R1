"""Arm/query one CPU-only complete20 retention worker, outside frozen runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from reproduction.h1008_qprobe.management_transport import remote


def start_script(base, job, code, commit):
    spec = dict(base=base, job=str(job), code=code, code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                control_commit=commit)
    return r'''
import pathlib,json,hashlib,os,subprocess,fcntl,time
p=SPEC;b=pathlib.Path(p['base']);root=b/'preparation/retained-joint20-once-v1'
assert b.parent==pathlib.Path('/mnt/data/shaoshitong') and b.resolve()==b and b.name.startswith('search-g1-prefix-selection200-')
c=json.loads((b/'preparation/launch-config.json').read_text());sub=json.loads((b/'preparation/submit-receipt.json').read_text())
assert c['run_id']=='a008d7ce983347bbb6e81051b25a7c4e' and c['probe_learning_rate']==1e-5 and str(sub['job_id'])==p['job']
with(b/'preparation/checkpoint20-pin-control.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if root.exists():
  print(json.dumps(dict(status='EXISTING_PIN_NAMESPACE_NO_REPLAY',root=str(root),receipt=json.loads((root/'RECEIPT.json').read_text())if(root/'RECEIPT.json').exists()else None,result=json.loads((root/'RESULT.json').read_text())if(root/'RESULT.json').exists()else None)));raise SystemExit
 if (b/'logs/cleanup-status.json').exists() and not (b/'outers/outer-000020/complete.json').exists():
  print(json.dumps(dict(status='SOURCE_EXECUTION_ENDED_BEFORE20_NO_WORKER_STARTED',training_modified=False)));raise SystemExit
 root.mkdir(mode=0o700);script=root/'checkpoint20_pin.py';script.write_text(p['code']);script.chmod(0o400)
 assert hashlib.sha256(script.read_bytes()).hexdigest()==p['code_sha256']
 intent=dict(run_id=c['run_id'],job=p['job'],target_outer=20,config_sha256=hashlib.sha256((b/'preparation/launch-config.json').read_bytes()).hexdigest(),code_sha256=p['code_sha256'],control_commit=p['control_commit'],created_unix=time.time(),operation='WAIT_REAL20_THEN_SAME_FILESYSTEM_PIN_ONLY')
 def once(path,value):
  with path.open('x')as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 once(root/'INTENT.json',intent)
 with(root/'worker.log').open('xb')as log:
  worker=subprocess.Popen(['/usr/bin/python3','-E','-B','-S',str(script),'--base',str(b),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','CUDA_VISIBLE_DEVICES':''})
 tail=(pathlib.Path('/proc')/str(worker.pid)/'stat').read_text().rsplit(')',1)[1].split()
 receipt=dict(status='COMPLETE20_RETENTION_ARMED_NOT_COMPLETE',root=str(root),pid=worker.pid,start_ticks=int(tail[19]),sid=os.getsid(worker.pid),uid=os.getuid(),code_sha256=p['code_sha256'],control_commit=p['control_commit'],training_modified=False,gpu_used=False,independent_backup=False)
 once(root/'RECEIPT.json',receipt);print(json.dumps(receipt))
'''.replace('SPEC', repr(spec))


def status_script(base):
    return r'''
import pathlib,json,os
b=pathlib.Path(BASE);r=b/'preparation/retained-joint20-once-v1';out=dict(root=str(r),exists=r.exists(),read_only=True)
if r.exists():
 for name in ('RECEIPT.json','PROGRESS.json','RESULT.json'):
  if(r/name).exists():out[name]=json.loads((r/name).read_text())
 receipt=out.get('RECEIPT.json')
 if receipt:
  p=pathlib.Path('/proc')/str(receipt['pid'])/'stat'
  if p.exists():
   tail=p.read_text().rsplit(')',1)[1].split();out['worker']=dict(pid=receipt['pid'],same_birth=int(tail[19])==receipt['start_ticks'],state=tail[0])
  else:out['worker']=dict(pid=receipt['pid'],gone=True)
print(json.dumps(out))
'''.replace('BASE', repr(base))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('start', 'status'))
    parser.add_argument('--base', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--receipts', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.phase == 'start':
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        relative = 'reproduction/prefix_selection200/checkpoint20_pin.py'
        code = subprocess.check_output(['git', 'show', commit + ':' + relative], cwd=repo).decode()
        if code != (repo / relative).read_text():
            raise ValueError('Publish the exact retention code first')
        script = start_script(args.base, args.job, code, commit)
    else:
        script = status_script(args.base)
    path = Path(args.receipts) / ('actorhidden-pin20-' + args.phase + '-' + str(time.time_ns()) + '-transport.json')
    result = json.loads(remote(script, path, control_path=None, target='h100-8'))
    actual = path.with_name(path.name.replace('-transport.json', '-actual.json'))
    actual.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(original=str(actual), sha256=hashlib.sha256(actual.read_bytes()).hexdigest(), result=result)))
