"""Stage/review/submit a bounded or persistent Slurm continuation.

`plan` is local only. `stage` copies only these small sources to physical NAS10.
`submit` requires exact Git publication verification, a successful stage receipt,
and a fresh intent; every later invocation reconciles that intent, never repeats
sbatch. Worker admission additionally requires the confirmed r6 stop handoff.
The persistent mode requests its own eight-hour allocation; it does not extend
existing backfill allocations or change their queue entries. Both modes use
the same one-GPU worker, shared UUID/batch locks, and eligible two eight-GPU
nodes, bounding their combined physical GPU concurrency at sixteen.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


ROOT = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
REPO = Path(__file__).resolve().parents[3]
PREFIX = 'reproduction/operations/qwen7b_resample/'
BASE = '/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'
PHYSICAL = '/mnt/data/shaoshitong/' + BASE.rsplit('/', 1)[1]
RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
SEED = 20261021
FROZEN_COMMIT = '82062fc12c6c0bdb50b9e8681e1af9e6e763af10'
FROZEN_REPO = BASE + '/code/repo-' + FROZEN_COMMIT
JOB_NAME = 'sst-q7b-g5-pool-' + RUN_ID[:12]
BEGIN = 'now'  # Actual sealed r6 terminal handoff is mandatory before GPU sampling.
QUEUE_MODES = {
    'backfill': dict(job_name=JOB_NAME, control_directory='pool-control',
                     walltime='30', run_seconds=1500, batch_reserve_seconds=300),
    'persistent': dict(job_name='sst-q7b-g5-persistent-' + RUN_ID[:12],
                       control_directory='pool-control-persistent',
                       walltime='08:00:00', run_seconds=28500, batch_reserve_seconds=300,
                       array='0-15%16'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def bootstrap_source(queue_mode='backfill'):
    profile = QUEUE_MODES[queue_mode]
    return ("BASE = " + repr(BASE) + "\nFROZEN_REPO = " + repr(FROZEN_REPO)
            + "\nQUEUE_MODE = " + repr(queue_mode) + "\nRUN_SECONDS = " + repr(profile['run_seconds'])
            + "\nBATCH_RESERVE_SECONDS = " + repr(profile['batch_reserve_seconds']) + "\n" + r'''
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
stage=Path(__file__).resolve().parent
manifest=json.loads((stage/'manifest.json').read_text())
assert os.geteuid()==0 and socket.gethostname().split('.')[0] in ('h100-5','h100-10')
assert os.stat('/proc/self/ns/mnt').st_ino != os.stat('/proc/1/ns/mnt').st_ino
for name,row in manifest['files'].items():
    path=stage/name;data=path.read_bytes()
    assert path.is_file() and not path.is_symlink()
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],name
base=Path(BASE);assert (base/'config.json').is_file() and Path(FROZEN_REPO).is_dir()
job=os.environ['SLURM_JOB_ID'];assert job.isdigit()
task=os.environ['SLURM_ARRAY_TASK_ID'];assert task.isdigit() and 0<=int(task)<128
launch=base/'pool/launch'/('job-'+job+'-task-'+task)
launch.mkdir(parents=True,exist_ok=False)
log=(launch/'bootstrap.log').open('ab',buffering=0)
os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
detail=subprocess.check_output(['scontrol','show','job','-dd',job],text=True,timeout=30)
evidence=launch/'allocation-evidence.json'
with evidence.open('x') as stream:
    json.dump(dict(job_id=job,node=socket.gethostname().split('.')[0],scontrol_job_detail=detail,
                   captured_epoch=time.time(),euid=os.geteuid(),array_task_id=task),stream)
os.chmod(evidence,0o400)
env=dict(os.environ,PYTHONPATH=FROZEN_REPO+':/dataset/shaoshitong/Search-G1-common8/retriever-gpu-lib',
    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
    SEARCH_G1_B_PREFIX_TOKENS='20',SEARCH_G1_PHYSICAL_WORLD_SIZE='4',SEARCH_G1_BASE=BASE,
    HF_HOME=BASE+'/cache/huggingface',TORCH_EXTENSIONS_DIR=BASE+'/cache/torch_extensions',
    TMPDIR=str(launch/'tmp'),VLLM_ATTENTION_BACKEND='XFORMERS',VLLM_HOST_IP='127.0.0.1',
    CPATH='/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:'
          '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:'
          '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10')
for name in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
    env.pop(name,None)
for name in ('HF_HOME','TORCH_EXTENSIONS_DIR','TMPDIR'):
    Path(env[name]).mkdir(parents=True,exist_ok=True)
os.chdir(FROZEN_REPO)
command=['unshare','--net','/bin/bash','-c',
    'set -euo pipefail; ip link set dev lo up; exec "$@"','pool-private-net',
    '/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u',str(stage/'pool_worker.py'),
    '--base',BASE,'--handoff-marker',BASE+'/preparation/pool-handoff.json',
    '--allocation-evidence',str(evidence),
    '--encoder',BASE.rsplit('/',1)[0]+'/search-g1-prefix5-h10011-runtime-assets-20261018/common8/assets/e5-base-v2',
    '--run-seconds',str(RUN_SECONDS),'--batch-reserve-seconds',str(BATCH_RESERVE_SECONDS)]
print(json.dumps(dict(status='EXACT_STAGE_VERIFIED_ALLOCATION_CAPTURED_ENTERING_PRIVATE_NET',
    source_bundle_sha256=manifest['source_bundle_sha256'],queue_mode=QUEUE_MODE,job_id=job,array_task_id=task,
    worker=str(stage/'pool_worker.py'),allocation_evidence=str(evidence),epoch=time.time())),flush=True)
os.execvpe(command[0],command,env)
''').encode()


def bundle(queue_mode='backfill'):
    profile = QUEUE_MODES[queue_mode]
    sources = {name: Path(__file__).with_name(name).read_bytes()
               for name in ('pool_control.py', 'pool_worker.py', 'pool_paths.sh')}
    source_rows = {name: dict(bytes=len(data), sha256=digest(data)) for name, data in sources.items()}
    identity = digest(encoded(source_rows if queue_mode == 'backfill' else
                             dict(queue_mode=queue_mode, source_files=source_rows)))
    stage = BASE + '/preparation/' + profile['control_directory'] + '/' + identity
    files = dict(sources, **{'pool_bootstrap.py': bootstrap_source(queue_mode)})
    # Slurm delivers the embedded wrapper itself, so a compute node need not
    # expose NAS10 until the private wrapper has mounted it.
    command = ['unshare', '--mount', '--propagation', 'private', '/bin/bash', '-c',
               sources['pool_paths.sh'].decode(), 'pool-private-mount', '${NODE}',
               '/usr/bin/python3', stage + '/pool_bootstrap.py']
    script = '#!/bin/bash\nset -euo pipefail\nNODE=$(hostname -s)\ncase "$NODE" in h100-5|h100-10) ;; *) exit 64 ;; esac\n'
    script += '[[ $(id -u) == 0 ]]\nexec ' + ' '.join('"$NODE"' if x == '${NODE}' else shlex.quote(x) for x in command) + '\n'
    files['pool.sbatch'] = script.encode()
    manifest = dict(version=1, source_bundle_sha256=identity, queue_mode=queue_mode,
                    queue_profile=profile, run_id=RUN_ID, seed=SEED,
                    frozen_runtime_commit=FROZEN_COMMIT, canonical_base=BASE,
                    source_files=source_rows, files={name: dict(bytes=len(data), sha256=digest(data))
                                                    for name, data in files.items()})
    assert sum(len(x) for x in files.values()) < 256 * 1024, 'Small code only'
    return identity, files, manifest


def sbatch_arguments(stage, queue_mode='backfill'):
    profile = QUEUE_MODES[queue_mode]
    return ['sbatch', '--parsable', '--job-name=' + profile['job_name'], '--partition=gpu',
            '--array=' + profile.get('array', '0-127%16'), '--begin=' + BEGIN, '--exclude=h100-8,h100-11',
            '--nodes=1', '--ntasks=1', '--gres=gpu:h100:1', '--cpus-per-task=8',
            '--mem=96G', '--time=' + profile['walltime'], '--no-requeue', '--signal=B:TERM@60',
            '--chdir=/tmp', '--output=/tmp/' + profile['job_name'] + '-%A_%a.log', str(stage / 'pool.sbatch')]


def verify_publication(commit, reference, source_rows):
    if not re.fullmatch(r'[0-9a-f]{40}', commit or '') or not re.fullmatch(r'refs/heads/[A-Za-z0-9._/-]+', reference or ''):
        raise ValueError('submit requires full --published-commit SHA and --published-ref refs/heads/BRANCH')
    for name, row in source_rows.items():
        data = subprocess.check_output(['git', 'show', commit + ':' + PREFIX + name], cwd=REPO)
        if digest(data) != row['sha256'] or len(data) != row['bytes']:
            raise ValueError('Published commit lacks exact local source: ' + name)
    remote = subprocess.check_output(['git', 'ls-remote', '--exit-code', 'origin', reference], cwd=REPO, text=True).strip()
    if remote != commit + '\t' + reference:
        raise ValueError('Origin branch does not resolve to the explicitly published commit')
    return dict(status='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED', commit=commit, remote='origin',
                remote_ref=reference, verified_remote_sha=commit, source_files=source_rows,
                verified_epoch=time.time())


REMOTE = r'''
import base64,fcntl,hashlib,json,os,re,socket,subprocess,time
from pathlib import Path
assert os.geteuid()==0 and socket.gethostname().split('.')[0]=='h100-10'
base=Path(PHYSICAL);root=base/'preparation'/CONTROL_DIRECTORY;stage=root/IDENTITY
config=json.loads((base/'config.json').read_text())
assert config['run_id']==RUN_ID and config['seed']==SEED
def read(p):return json.loads(p.read_text()) if p.exists() else None
def publish(p,value):
    temp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
    with temp.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)
def query():
    commands=[['squeue','--noheader','--user=root','--name='+JOB_NAME,'--format=%A|%i|%T|%N'],
              ['scontrol','show','jobs','--oneliner']]
    reports=[];ids=set()
    for command in commands:
        try:r=subprocess.run(command,text=True,capture_output=True,timeout=30)
        except subprocess.TimeoutExpired:reports.append(dict(command=command,status='TIMEOUT'));continue
        text=r.stdout
        if command[0]=='scontrol':
            lines=[x for x in text.splitlines() if re.search(r'\bJobName='+re.escape(JOB_NAME)+r'(?:\s|$)',x)]
            text='\n'.join(lines)
            for line in lines:
                match=re.search(r'\b(?:ArrayJobId|JobId)=(\d+)',line)
                if match:ids.add(match.group(1))
        else:
            for line in text.splitlines():
                if line.split('|')[0].isdigit():ids.add(line.split('|')[0])
        reports.append(dict(command=command,returncode=r.returncode,stdout=text[:24000],stderr=r.stderr[-3000:]))
    return dict(job_ids=sorted(ids),reports=reports)
def verify_stage():
    saved=read(stage/'manifest.json');assert saved==MANIFEST,'Different stage manifest'
    for name,row in saved['files'].items():
        p=stage/name;data=p.read_bytes()
        assert p.is_file() and not p.is_symlink() and len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],name
    assert (read(stage/'stage-receipt.json') or {}).get('status')=='EXACT_SMALL_CODE_STAGE_SHA_VERIFIED'
value=dict(action=ACTION,queue_mode=QUEUE_MODE,source_bundle_sha256=IDENTITY,physical_stage=str(stage),epoch=time.time())
if ACTION=='stage':
    root.mkdir(parents=True,exist_ok=True);stage.mkdir(exist_ok=True)
    intent=stage/'stage-intent.json'
    if not intent.exists():
        with intent.open('x') as f:json.dump(dict(intent_id=INTENT,action=ACTION,queue_mode=QUEUE_MODE,manifest=MANIFEST,epoch=time.time()),f)
    assert read(intent)['manifest']==MANIFEST
    for name,text in FILES.items():
        assert '/' not in name and name not in ('.','..')
        p=stage/name;data=base64.b64decode(text)
        if not p.exists():
            with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==MANIFEST['files'][name]['sha256'],name
        os.chmod(p,0o555)
    manifest=stage/'manifest.json'
    if not manifest.exists():publish(manifest,MANIFEST)
    assert read(manifest)==MANIFEST
    receipt=dict(status='EXACT_SMALL_CODE_STAGE_SHA_VERIFIED',intent_id=INTENT,queue_mode=QUEUE_MODE,source_bundle_sha256=IDENTITY,
                 manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),files=MANIFEST['files'],epoch=time.time())
    publish(stage/'stage-receipt.json',receipt);verify_stage();value['receipt']=receipt
elif ACTION=='submit':
    verify_stage();assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
    assert PUBLICATION['source_files']==MANIFEST['source_files']
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        intent=root/'pool-submit-intent.json';receipt=root/'pool-submit-receipt.json'
        old=read(intent);observed=query();value['observed']=observed
        if old:
            value.update(status='EXISTING_INTENT_NEVER_RESUBMITTED',intent=old,receipt=read(receipt))
            if not value['receipt'] and observed['job_ids']:
                recovered=dict(status='RECOVERED_EXISTING_SUBMISSION_BY_JOB_NAME',queue_mode=QUEUE_MODE,intent=old,observed=observed,epoch=time.time())
                publish(receipt,recovered);value['receipt']=recovered
            elif not observed['job_ids'] and not value['receipt']:
                value['status']='SUBMIT_OUTCOME_UNKNOWN_NO_RETRY'
        elif observed['job_ids']:
            value.update(status='EXISTING_JOB_NAME_FOUND_NO_SUBMISSION',observed=observed)
        else:
            assert all(x.get('returncode')==0 for x in observed['reports']),'Cannot prove no prior job name'
            inventory=subprocess.check_output(['sinfo','--noheader','--partition=gpu','--Node','--format=%N'],text=True,timeout=30)
            nodes=set(inventory.split());assert nodes-{'h100-8','h100-11'} <= {'h100-5','h100-10'} and nodes & {'h100-5','h100-10'},inventory
            args=SBATCH
            assert not any(x.startswith('--account') for x in args)
            proposed=dict(status='SBATCH_INTENT_RECORDED',intent_id=INTENT,queue_mode=QUEUE_MODE,run_id=RUN_ID,seed=SEED,
                source_bundle_sha256=IDENTITY,job_name=JOB_NAME,argv=args,publication=PUBLICATION,epoch=time.time())
            with intent.open('x') as f:json.dump(proposed,f,indent=2);f.flush();os.fsync(f.fileno())
            try:
                submitted=subprocess.run(args,text=True,capture_output=True,timeout=60)
                result=dict(status='SBATCH_RETURNED' if submitted.returncode==0 else 'SBATCH_FAILED_NO_AUTOMATIC_RETRY',queue_mode=QUEUE_MODE,
                    returncode=submitted.returncode,stdout=submitted.stdout,stderr=submitted.stderr,intent=proposed,epoch=time.time())
                if submitted.returncode==0:
                    job=submitted.stdout.strip().split(';')[0];assert job.isdigit(),submitted.stdout
                    result.update(status='SLURM_POOL_SUBMITTED_ONCE',job_id=job)
            except subprocess.TimeoutExpired as error:
                result=dict(status='SUBMIT_OUTCOME_UNKNOWN_NO_RETRY',queue_mode=QUEUE_MODE,intent=proposed,error=str(error),epoch=time.time())
            result['observed']=query();publish(receipt,result);value.update(status=result['status'],receipt=result)
else:
    value.update(status='READ_ONLY_POOL_STATUS',intent=read(root/'pool-submit-intent.json'),
                 receipt=read(root/'pool-submit-receipt.json'),stage_receipt=read(stage/'stage-receipt.json'),observed=query())
    counts={};latest=[]
    for p in sorted((base/'pool/jobs').glob('job-*/attempt-*')):
        progress=read(p/'progress.json');complete=read(p/'complete.json');failure=read(p/'failure.json')
        status='FAILED' if failure else 'COMPLETE' if complete else 'PROGRESS' if progress else 'STARTING'
        counts[status]=counts.get(status,0)+1
        latest.append(dict(directory=str(p),status=status,progress=progress,complete=complete,failure=failure))
    value.update(job_attempt_counts=counts,latest_attempts=latest[-16:],global_goal_complete=False)
print(json.dumps(value,sort_keys=True))
'''


def transport(script):
    modules = {}
    for name, path in [('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')]:
        spec = importlib.util.spec_from_file_location('pool_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    # Match the configured direct mgmt-host route; no task-forced Mac SOCKS.
    argv = modules['ctl'].route('h100-10', '/usr/bin/python3 -')
    return subprocess.run(argv, input=script, text=True, capture_output=True, timeout=110)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'stage', 'submit', 'status'))
    parser.add_argument('--queue-mode', choices=tuple(QUEUE_MODES), default='backfill',
                        help='Independent queue/intent namespace; default retains the 30-minute backfill mode')
    parser.add_argument('--intent-id', help='Explicit operator intent label; mandatory for stage/submit')
    parser.add_argument('--published-commit', help='Full SHA containing the exact three source files')
    parser.add_argument('--published-ref', help='Origin refs/heads/BRANCH verified immediately before submit')
    args = parser.parse_args()
    profile = QUEUE_MODES[args.queue_mode]
    identity, files, manifest = bundle(args.queue_mode)
    stage = Path(PHYSICAL) / 'preparation' / profile['control_directory'] / identity
    if args.action == 'plan':
        print(json.dumps(dict(action='LOCAL_ONLY_PLAN',queue_mode=args.queue_mode,manifest=manifest,physical_stage=str(stage),
            sbatch=sbatch_arguments(stage,args.queue_mode),handoff_marker=BASE+'/preparation/pool-handoff.json'),indent=2))
        return
    if args.action in ('stage', 'submit') and not re.fullmatch(r'[A-Za-z0-9._-]{1,120}', args.intent_id or ''):
        parser.error('stage/submit require --intent-id (letters, digits, dots, hyphens, underscores)')
    publication = verify_publication(args.published_commit, args.published_ref, manifest['source_files']) if args.action == 'submit' else None
    values = dict(PHYSICAL=PHYSICAL,IDENTITY=identity,BASE=BASE,RUN_ID=RUN_ID,SEED=SEED,
                  JOB_NAME=profile['job_name'],QUEUE_MODE=args.queue_mode,CONTROL_DIRECTORY=profile['control_directory'],
                  ACTION=args.action,INTENT=args.intent_id,PUBLICATION=publication,
                  MANIFEST=manifest,FILES={name:base64.b64encode(data).decode() for name,data in files.items()},
                  SBATCH=sbatch_arguments(stage,args.queue_mode))
    script = '\n'.join(name+'='+repr(value) for name,value in values.items()) + '\n' + REMOTE
    record_prefix = 'qwen7b-pool-control' if args.queue_mode == 'backfill' else 'qwen7b-pool-persistent-control'
    record = ROOT / 'work' / ('%s-%s-%d.json' % (record_prefix,args.action,time.time_ns()))
    try:
        result = transport(script)
    except subprocess.TimeoutExpired:
        record.write_text(json.dumps(dict(action=args.action,queue_mode=args.queue_mode,status='TRANSPORT_TIMEOUT_QUERY_STATUS_DO_NOT_REPEAT_SUBMIT',
                                         source_bundle_sha256=identity,epoch=time.time()),indent=2)+'\n')
        raise
    envelope = dict(action=args.action,queue_mode=args.queue_mode,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                    source_bundle_sha256=identity,local_receipt=str(record),epoch=time.time())
    record.write_text(json.dumps(envelope,indent=2)+'\n')
    print(json.dumps(envelope,indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
