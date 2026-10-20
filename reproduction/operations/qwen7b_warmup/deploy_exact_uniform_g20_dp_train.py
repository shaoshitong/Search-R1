"""One explicit matched G20 all15504 native training job; no grid or dependencies.

Immutable published source, bounded4KiB transport, CPU prepare, held submission
ID capture then release. Unknown prior outcomes and active locks fail closed.
Existing CPU qualification and previous random/balanced native jobs stay intact.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
from .deploy_confirmation_materialization import make_bundle
from . import deploy_peer_newseed_replay as transport
from .probe_exact_uniform_g20_dp_train import TARGET, ENTRY

JOB_NAME = 'q7b-G20-all15504-DP'
PACKAGE = 'reproduction/operations/qwen7b_warmup/'
SEEDS = tuple(PACKAGE + name for name in ('probe_exact_uniform_g20_dp_train.py',
    'test_probe_exact_uniform_g20_dp_train.py', 'test_exact_uniform_g5_dp_loss.py', 'runtime.sh'))
MANIFEST = '_G20_all15504_DP_training_source_manifest.json'
COMMON = transport.COMMON
INIT = r'''
guard=lock();uploads.mkdir(exist_ok=True);(b/'logs').mkdir(exist_ok=True)
value={key:inputs[key] for key in ('commit','bundle_sha256','bundle_bytes','chunks','manifest_sha256')};p=b/'upload-controller.json'
if p.exists():assert read(p)==value,'Independent immutable source differs; no active source replacement'
else:atomic(p,value)
print(json.dumps(dict(status='G20_ALL15504_DP_SOURCE_UPLOAD_BOUND',chunks=inputs['chunks'])))
'''
CHUNK = transport.CHUNK
EXTRACT = r'''
guard=lock();data=b''.join(p.read_bytes() for p in sorted(uploads.glob('part-*')))
assert len(data)==inputs['bundle_bytes'] and hashlib.sha256(data).hexdigest()==inputs['bundle_sha256']
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
 raw=tar.extractfile(inputs['manifest']).read();assert hashlib.sha256(raw).hexdigest()==inputs['manifest_sha256']
 manifest=json.loads(raw);files=manifest['files'];assert {m.name for m in tar.getmembers()}==set(files)|{inputs['manifest']}
 for name in set(files)|{inputs['manifest']}:
  relative=pathlib.Path(name);assert not relative.is_absolute() and '..' not in relative.parts;member=tar.getmember(name);assert member.isfile()
  raw=tar.extractfile(member).read()
  if name in files:assert len(raw)==files[name]['bytes'] and hashlib.sha256(raw).hexdigest()==files[name]['sha256']
  p=code/relative;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==raw
  else:p.write_bytes(raw);p.chmod(0o444)
value=dict(commit=inputs['commit'],remote_sha=inputs['remote_sha'],code_root=str(code),files=files,bundle_sha256=inputs['bundle_sha256']);p=b/'source-publication.json'
if p.exists():assert read(p)==value
else:atomic(p,value)
publication();print(json.dumps(dict(status='ACTUAL_G20_DP_GPU_RUNTIME_SOURCE_SHA_VERIFIED',code_root=str(code),files=len(files))))
'''
CPU = r'''
guard=lock();pub,root=publication();env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
args=runtime(root,['unittest','-v','reproduction.operations.qwen7b_warmup.test_exact_uniform_g5_dp_loss','reproduction.operations.qwen7b_warmup.test_probe_exact_uniform_g20_dp_train'])
r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=50)
(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-3500:]
args=runtime(root,[inputs['entry'],'--base',str(b),'--action','prepare'])
r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=50)
(b/'actual-cpu-prepare.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-4000:]
done=read(b/'preparation.json');assert done['GPU_training_authorized'] and not done['actual_GPU_qualification']
script=b/'train.sbatch'
if script.exists():assert script.read_text()==inputs['sbatch']
else:script.write_text(inputs['sbatch'])
value=dict(status='ACTUAL_G20_ALL15504_DP_SINGLE_NORMAL_JOB_READY_NOT_SUBMITTED',preparation=fp(b/'preparation.json'),config=done['config'],sbatch=fp(script),job_name=inputs['job_name'],source_commit=pub['commit'],source_publication=fp(b/'source-publication.json'),GPU_requested=0,epoch=time.time())
atomic(b/'submission-plan.json',value);print(json.dumps(value))
'''
SUBMIT = r'''
guard=lock();pub,root=publication();item=read(b/'submission-plan.json')
assert fp(item['config']['path'])==item['config'] and fp(item['sbatch']['path'])==item['sbatch'] and item['source_commit']==pub['commit']
if (b/'job.id').exists():print(json.dumps(dict(status='EXISTING_EXACT_G20_DP_JOB_PRESERVED',job_id=(b/'job.id').read_text().strip())))
else:
 assert not (b/'submission-intent.json').exists(),'Unknown previous submit outcome; inspect same-name job, never blind repeat'
 assert not any((b/name).exists() for name in ('controller.json','progress.json','latest.json','update-intent.json','failure.json')),'Existing training state requires explicit exact restore'
 assert not command(['squeue','-h','-u','mou176','-n',inputs['job_name'],'-o','%i|%j|%T|%R']),'Unknown same-name job exists'
 with (b/'g20-train.lock').open('a') as train_lock:fcntl.flock(train_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 intent=dict(status='G20_ALL15504_DP_SINGLE_SUBMIT_INTENT',job_name=inputs['job_name'],config=item['config'],script=item['sbatch'],source_publication=item['source_publication'],partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time_limit='02:00:00',dependency=None,epoch=time.time());atomic(b/'submission-intent.json',intent)
 try:r=subprocess.run(['sbatch','--hold','--parsable',item['sbatch']['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 except subprocess.TimeoutExpired:atomic(b/'submission-unknown.json',dict(status='UNKNOWN_OUTCOME_DO_NOT_REPEAT',epoch=time.time()));raise
 if r.returncode:atomic(b/'submission-rejected.json',dict(stderr=r.stderr[-2000:],epoch=time.time()));raise RuntimeError(r.stderr[-2000:])
 job=r.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown submission reply, preserve intent'
 with (b/'job.id').open('x') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
 state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
 command(['scontrol','release',job]);result=dict(intent,status='ACTUAL_G20_ALL15504_DP_NORMAL_JOB_RELEASED_NOT_PROGRESS_VERIFIED',job_id=job);atomic(b/'submission.json',result);print(json.dumps(result))
'''
STATUS = r'''
def optional(p):
 try:return read(p)
 except (FileNotFoundError,json.JSONDecodeError):return None
result=dict(epoch=time.time(),base=str(b))
try:job=(b/'job.id').read_text().strip();result['job_id']=job
except FileNotFoundError:job=None
for filename in ('preparation.json','progress.json','next-update-replay.json','latest.json','failure.json','interrupted.json','gpu-memory.json','complete.json'):
 value=optional(b/filename)
 if value is not None:result[filename]=value
controller=optional(b/'controller.json')
if controller is not None:result['controller']={key:controller.get(key) for key in ('pid','job_id','status','source_steps','global_original_questions','recipe','plan_sha256')}
if job:
 result['squeue']=command(['squeue','-h','-j',job,'-o','%i|%j|%u|%T|%R|%N'])
 result['sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode,Elapsed,NodeList'])
 path=b/'logs'/('train-'+job+'.log')
 if path.exists():result['log_tail']=path.read_text(errors='replace')[-2500:]
print(json.dumps(result))
'''


def render_sbatch(code_root):
    return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-G20-all15504-DP
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=@@BASE@@/logs/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
python3 - @@CODE@@ <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);pub=json.loads((root.parent/'source-publication.json').read_text());assert str(root)==pub['code_root']
for name,value in pub['files'].items():
 path=root/name;assert path.stat().st_size==value['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==value['sha256'],name
VERIFY
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m @@ENTRY@@ --base @@BASEQ@@ --action train
'''.replace('@@BASEQ@@', shlex.quote(str(TARGET))).replace('@@BASE@@', str(TARGET)).replace('@@CODE@@', shlex.quote(str(code_root))).replace('@@ENTRY@@', ENTRY)


def bundle(repo):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/codex/qwen7b-probe-warmup-20261019'], cwd=repo, text=True).split()[0]
    if commit != remote:
        raise ValueError('Publish exact latest source before GPU preparation')
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=repo, text=True).splitlines()
    cache = {}
    def load(name):
        if name not in cache:
            cache[name] = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=repo)
        return cache[name]
    raw, report = make_bundle(paths, load, commit, seeds=SEEDS,
        resource_files=('verl/version/version',), manifest_name=MANIFEST)
    for name in SEEDS + (PACKAGE + 'deploy_exact_uniform_g20_dp_train.py',):
        if (repo / name).read_bytes() != load(name):
            raise ValueError('Unpublished source: ' + name)
    return commit, remote, raw, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-root', type=Path)
    parser.add_argument('--action', choices=('prepare', 'submit', 'status'), default='status')
    parser.add_argument('--submit-authorized', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    inputs = dict(base=str(TARGET), target=str(TARGET), commit='a' * 40, remote_sha='a' * 40,
        bundle_sha256='b' * 64, bundle_bytes=10, chunks=1,
        manifest=MANIFEST, manifest_sha256='c' * 64, entry=ENTRY, job_name=JOB_NAME)
    blocks = dict(init=INIT, chunk=CHUNK, extract=EXTRACT, cpu=CPU, submit=SUBMIT, status=STATUS)
    def source(action, extra=None):
        values = dict(inputs); values.update(extra or {})
        return 'inputs=' + repr(values) + '\n' + COMMON + '\n' + blocks[action]
    if args.self_test:
        sizes = {}
        script = render_sbatch(TARGET / 'code-a')
        for action in blocks:
            code = source(action, dict(chunk_index=0, chunk_data=base64.b64encode(b'x' * 4096).decode(), chunk_sha256='d' * 64, sbatch=script))
            compile(code, action, 'exec');sizes[action] = len(('python3 -c ' + shlex.quote(code)).encode())
            if sizes[action] >= 16000:
                raise ValueError('Command size bound exceeded')
        subprocess.run(['bash', '-n'], input=script, text=True, check=True)
        exports = '\n'.join(row for row in script.splitlines() if row.startswith('export '))
        subprocess.run(['bash', '-euc', exports + '\ntest "$QWEN7B_CODE_ROOT" = ' + shlex.quote(str(TARGET / 'code-a'))], check=True)
        assert '--dependency' not in script and '--qos=normal' in script and '--time=02:00:00' in script
        print(json.dumps(dict(status='EXACT_G20_DP_COMMAND_SHELL_AND_DEFAULT_NORMAL_CHECKS_PASS', command_bytes=sizes, remote_actions_executed=False)))
        return
    if args.control_root is None:
        parser.error('--control-root required')
    if args.action == 'submit' and not args.submit_authorized:
        parser.error('Explicit --submit-authorized required for the single new job')
    raw = None
    if args.action == 'prepare':
        commit, remote, raw, report = bundle(repo)
        inputs.update(commit=commit, remote_sha=remote, bundle_sha256=hashlib.sha256(raw).hexdigest(),
            bundle_bytes=len(raw), chunks=(len(raw) + 4095) // 4096, manifest_sha256=report['source_manifest_sha256'])
    spec = importlib.util.spec_from_file_location('exact_g20_dp_control', args.control_root / 'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    address = subprocess.check_output(['/usr/sbin/ipconfig', 'getifaddr', 'en0'], text=True).strip()
    def call(action, extra=None):
        command = 'python3 -c ' + shlex.quote(source(action, extra))
        assert len(command.encode()) < 16000
        result = ctl.run(command, account='mou176', source_address=address, timeout=110 if action == 'cpu' else 55)
        rows = [json.loads(row) for row in result.stdout.decode(errors='replace').splitlines() if row.startswith('{')]
        if result.returncode or not rows:
            raise RuntimeError('Unconfirmed read/prepare/submit; inspect before repeating: ' + result.stdout.decode(errors='replace')[-3000:] + result.stderr.decode(errors='replace')[-1000:])
        return rows[-1]
    if args.action == 'prepare':
        call('init')
        for i, start in enumerate(range(0, len(raw), 4096)):
            chunk = raw[start:start + 4096]
            call('chunk', dict(chunk_index=i, chunk_data=base64.b64encode(chunk).decode(), chunk_sha256=hashlib.sha256(chunk).hexdigest()))
        call('extract')
        result = call('cpu', dict(sbatch=render_sbatch(TARGET / ('code-' + inputs['commit'][:12]))))
    else:
        result = call(args.action)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
