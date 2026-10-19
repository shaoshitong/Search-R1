"""Prepare/explicitly launch Easy's independently frozen diagnostic and strict CF."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

BASE = '/data/shaoshitong/grounding-eval-20261019'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--worktree', required=True)
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    repo, audit = Path(args.worktree), Path(args.audit_dir)
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    origin = subprocess.check_output(['git','-C',str(repo),'ls-remote','origin',
        'refs/heads/codex/grounding-eval-strict-20261019'], text=True).split()[0]
    subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',commit,origin], check=True)
    sources = {}
    for name in ('easy_dual_cf_controller.py', 'verify_completed_cf.py',
                 'verify_completed_strict_v5.py', 'official_changed_rule_replay.py'):
        sources[name] = subprocess.check_output(['git','-C',str(repo),'show',
            commit+':reproduction/grounding_eval/operations/'+name]).decode()
    gate_raw = (audit/'g1-strict-v5-root-acceptance-gate.json').read_bytes()
    gate = json.loads(gate_raw)
    assert gate['accepted'] and 'Easy Selection strict-v5 after normal complete' in gate['authorized_next']
    runtime = BASE+'/easy-cf-control-runtime-'+commit[:8]
    control = BASE+'/easy-selection-dual-cf-control'
    normal = BASE+'/easy-selection-normal3610'
    job = dict(method='Easy Selection', run_id='15bb15365346401da1d919e9cde8f813',
        python='/data/search-g1/env/bin/python', checkpoint=BASE+'/assets/easy-selection-iter200',
        normal_root=normal, control_output=control, normal_recompute_authorized=False,
        gpu_uuid='GPU-21508d9d-bd43-ff94-7bee-1227ec7b007b',
        g1_acceptance_gate=control+'/g1-root-gate.json',
        g1_acceptance_gate_sha256=hashlib.sha256(gate_raw).hexdigest(),
        rule_replay=runtime+'/official_changed_rule_replay.py', protocols={
            'v4':dict(source_commit='21b9ccf4939803c41ed3351415cd90616709735c',
                runtime=BASE+'/runtime-21b9ccf4',module='reproduction.grounding_eval',
                output=BASE+'/easy-selection-cf-full-v4',verifier=runtime+'/verify_completed_cf.py'),
            'v5':dict(source_commit='16c36aa777f684f245b1d38cb2939ef175131831',
                runtime=BASE+'/runtime-16c36aa7',module='reproduction.grounding_eval.strict_v5',
                output=BASE+'/easy-selection-cf-strict-v5',verifier=runtime+'/verify_completed_strict_v5.py')})
    payload = dict(job=job, controller_runtime=runtime, sources=sources,
        source_commit=commit, gate=base64.b64encode(gate_raw).decode())
    audit.mkdir(exist_ok=True)
    (audit/'easy-dual-cf-prepared-job.json').write_text(json.dumps(job,indent=2)+'\n')
    if not args.execute:
        print(json.dumps(dict(state='PREPARED_NOT_EXECUTED', job=job, controller_source_commit=commit)))
        return
    remote = r'''
import json,sys,pathlib,hashlib,os,fcntl,subprocess,time
p=json.load(sys.stdin);job=p['job'];root=pathlib.Path(job['control_output']);root.mkdir(exist_ok=True)
lock=(root/'launch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not(root/'launch.json').exists(),'Existing launch must be inspected; never duplicate'
normal=pathlib.Path(job['normal_root']);assert json.loads((normal/'normal-independent-verification.json').read_text())['state']=='VERIFIED'
assert json.loads((normal/'complete.json').read_text())['completed']==3610
assert json.loads((normal/'control/exit.json').read_text())['exit_code']==0
rt=pathlib.Path(p['controller_runtime']);rt.mkdir(exist_ok=True)
def put(path,data):
 if path.exists():assert path.read_bytes()==data
 else:
  with path.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 assert path.read_bytes()==data
for name,source in p['sources'].items():put(rt/name,source.encode())
import base64
gate=base64.b64decode(p['gate']);assert hashlib.sha256(gate).hexdigest()==job['g1_acceptance_gate_sha256'];put(pathlib.Path(job['g1_acceptance_gate']),gate)
for version,config in job['protocols'].items():
 runtime=pathlib.Path(config['runtime']);receipt=json.loads((runtime/'deployment-receipt.json').read_text());assert receipt['source_commit']==config['source_commit']
 for name,value in receipt['files'].items():assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==value['sha256']
sys.path.insert(0,job['protocols']['v4']['runtime']);from reproduction.grounding_eval.normal_a800 import gpu_lease
fd,index=gpu_lease(job['gpu_uuid']);assert index==5
jobpath=root/'job.json';raw=json.dumps(job,sort_keys=True,indent=2).encode()+b'\n';put(jobpath,raw)
command=[job['python'],'-B',str(rt/'easy_dual_cf_controller.py'),'--job',str(jobpath)];os.close(fd)
with(root/'controller.log').open('xb')as log:child=subprocess.Popen(command,cwd=root,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=(pathlib.Path('/proc')/str(child.pid)/'stat').read_text();birth=int(stat[stat.rfind(')')+2:].split()[19])
receipt=dict(pid=child.pid,birth_ticks=birth,command=command,controller_source_commit=p['source_commit'],source_sha256={k:hashlib.sha256(v.encode()).hexdigest()for k,v in p['sources'].items()},job_sha256=hashlib.sha256(raw).hexdigest(),normal_recomputed=False,gpu_uuid=job['gpu_uuid'],at=time.time())
put(root/'launch.json',json.dumps(receipt,indent=2).encode()+b'\n');print(json.dumps(receipt))
'''
    compile(remote,'launch-easy-dual-cf','exec')
    spec = importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    command = ctl.route('baai-a800','/usr/bin/python3 -B -c '+shlex.quote(remote))
    command[1:1] = ['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6']
    result = subprocess.run(command,input=json.dumps(payload),capture_output=True,text=True,timeout=45)
    report = dict(transport_exit=result.returncode, stderr=result.stderr[-1500:],
        receipt=json.loads(result.stdout.strip().splitlines()[-1]) if not result.returncode else result.stdout[-2000:])
    (audit/'easy-dual-cf-launch.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if result.returncode:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
