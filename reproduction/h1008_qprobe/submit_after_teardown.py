"""Once-only control after completed predecessor evaluation and verified cleanup."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


REMOTE = '''import pathlib,json,hashlib,os,time,subprocess,sys
P=pathlib.Path;meta=json.loads(METADATA)
b=P('/mnt/data/shaoshitong/search-g1-qprobe3-h1008-77579a4-20261015-v3');prep=b/'preparation';config=json.loads((prep/'h1008-launch-config.json').read_text());old=P(config['predecessor_evaluation_base'])
def read(p):return json.loads(p.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 return h.hexdigest()
def once(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
if (prep/'submit-intent.json').exists() or (prep/'submit-receipt.json').exists():
 print(json.dumps({'action':'inspect existing outcome; no replay','receipt':read(prep/'submit-receipt.json') if (prep/'submit-receipt.json').exists() else None}));raise SystemExit
assert not (prep/'teardown-closure-submit-intent.json').exists(),'Prior closure control claim; inspect before retry'
dispatcher=read(prep/'start-after-predecessor-receipt.json');stat=P('/proc')/str(dispatcher['pid'])/'stat'
assert not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[19]!=dispatcher['start_ticks'],'Original dispatcher must have exited'
terminal_path=P(config['predecessor_terminal_evidence_base'])/'preparation/predecessor-exit0.json';terminal=read(terminal_path)
assert terminal['job_id']==3395 and terminal['state']=='FAILED' and terminal['exit_code']=='1:0'
result=read(old/'journal/final-result.json');cleanup=read(old/'logs/cleanup-status.json');pipeline=read(old/'logs/final-pipeline-status.json');ledger=read(old/'logs/process-ledger.json')
assert result['qa_count']==result['question_count']==sum(result['source_counts'].values())==51713
assert result['run_id']==cleanup['run_id']==ledger['run_id']==config['predecessor_run_id']
assert pipeline['error']=="AssertionError('final-evaluator left owned subprocesses')" and cleanup['cleaned'] is True
assert ledger['jobs']['final-evaluator']['exit_code']==0
live=[];count=0
for job in ledger['jobs'].values():
 for row in job['ledger'].values():
  count+=1;p=P('/proc')/str(row['pid'])/'stat'
  if p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==str(row['start_ticks']):live.append(row['pid'])
assert count>0 and not live
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);assert not gpu.strip()
cgroup=P('/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3395');assert not cgroup.exists()
closure=dict(job_id=3395,error=pipeline['error'],evaluator_exit_code=0,owned_birth_count=count,original_births_still_live=live,gpu_processes=gpu,cgroup_exists=False,final_result_sha256=digest(old/'journal/final-result.json'),cleanup_sha256=digest(old/'logs/cleanup-status.json'),process_ledger_sha256=digest(old/'logs/process-ledger.json'),native_log_sha256=digest(old/'logs/native-3395.log'),actual_native_terminal_sha256=digest(terminal_path),checked_unix=time.time(),published_control_commit=meta['commit'],native_exit0_claimed=False)
assert hashlib.sha256(meta['submit_source'].encode()).hexdigest()==meta['submit_sha256']
once(prep/'teardown-closure-submit-intent.json',dict(control_commit=meta['commit'],submit_source_sha256=meta['submit_sha256'],closure=closure,created_unix=time.time()))
if not (prep/'predecessor-exit0.json').exists():
 with (prep/'predecessor-exit0.json').open('xb') as f:f.write(terminal_path.read_bytes());f.flush();os.fsync(f.fileno())
once(prep/'predecessor-teardown-closure.json',closure)
p=prep/'submit-after-teardown-published.py'
with p.open('x') as f:f.write(meta['submit_source']);f.flush();os.fsync(f.fileno())
assert digest(p)==meta['submit_sha256']
sys.path.insert(0,str(b/'repo'));namespace={'__name__':'published_submission_control','__file__':str(p)};exec(compile(meta['submit_source'],str(p),'exec'),namespace)
receipt=namespace['submit'](b,read(prep/'source-files.json'))
once(prep/'teardown-closure-submit-receipt.json',dict(submission=receipt,published_control_commit=meta['commit'],native_predecessor_exit0_claimed=False,checked_unix=time.time()));print(json.dumps({'submission':receipt,'closure':closure,'training_frozen_source':config['source_commit']}))
'''


def main():
    repo = Path(__file__).resolve().parents[2]
    proof = json.loads(Path(sys.argv[1]).read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode().strip()
    assert proof['published_control_commit'] == commit and proof['all_blobs_full_sha_match']
    source = (repo / 'reproduction/h1008_qprobe/submit.py').read_text()
    sha = hashlib.sha256(source.encode()).hexdigest()
    assert sha == proof['files']['reproduction/h1008_qprobe/submit.py']
    meta = json.dumps(dict(commit=commit, submit_source=source, submit_sha256=sha))
    sys.path.insert(0, 'work')
    import search_g1_restore5_transport as transport
    original = subprocess.run
    def direct(args, *a, **kw):
        if isinstance(args, list) and 'mgmt-host' in args:
            args = list(args)
            option = 'ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p'
            if option in args:
                index = args.index(option)
                del args[index-1:index+1]
        return original(args, *a, **kw)
    transport.subprocess.run = direct
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    print(transport.remote('/usr/bin/python3 -c ' + shlex.quote(REMOTE.replace('METADATA', repr(meta))),
          'outputs/gigpo-e1-instruct200/QPROBE3-CLOSURE-SUBMIT-' + stamp + '.json', emit=False))


if __name__ == '__main__':
    main()
