"""Published-only CPU qualification/preparation; intentionally no GPU actions.

Immutable import closure, 4KiB payload chunks, every command <16KiB. Owned
background CPU benchmark has intent/PID birth/exit receipts and cannot be
blindly restarted after an unknown result. This module never calls Slurm.
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
from .prepare_exact_uniform_g20_dp import TARGET

PACKAGE = 'reproduction/operations/qwen7b_warmup/'
SEEDS = tuple(PACKAGE + name for name in ('exact_uniform_g5_dp_loss.py',
    'probe_exact_uniform_g5_dp.py', 'test_exact_uniform_g5_dp_loss.py',
    'benchmark_exact_uniform_g5_dp.py', 'prepare_exact_uniform_g20_dp.py', 'runtime.sh'))
MANIFEST = '_exact_g5_dp_source_manifest.json'
COMMON = transport.COMMON
INIT = r'''
guard=lock();uploads.mkdir(exist_ok=True)
value={key:inputs[key] for key in ('commit','bundle_sha256','bundle_bytes','chunks','manifest_sha256')}
p=b/'upload-controller.json'
if p.exists():assert read(p)==value,'Different source or unknown prior CPU activity; preserve original directory'
else:atomic(p,value)
print(json.dumps(dict(status='DP_CPU_IMMUTABLE_STAGE_BOUND',chunks=inputs['chunks'])))
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
value=dict(commit=inputs['commit'],remote_sha=inputs['remote_sha'],code_root=str(code),files=files,bundle_sha256=inputs['bundle_sha256'])
p=b/'source-publication.json'
if p.exists():assert read(p)==value
else:atomic(p,value)
publication();print(json.dumps(dict(status='ACTUAL_DP_CPU_CLOSURE_ALL_SOURCE_SHA_VERIFIED',code_root=str(code),files=len(files))))
'''
TEST = r'''
guard=lock();pub,root=publication();env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
args=runtime(root,['unittest','-v','reproduction.operations.qwen7b_warmup.test_exact_uniform_g5_dp_loss'])
r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=50)
(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-3000:]
value=dict(status='ACTUAL_REMOTE_DP_SIX_CPU_TESTS_PASS',log=fp(b/'actual-cpu-tests.log'),source_publication=fp(b/'source-publication.json'),model_tensors_loaded=False,GPU_requested=0,epoch=time.time())
atomic(b/'cpu-tests.json',value);print(json.dumps(value))
'''
WRAPPER = r'''
import json,os,pathlib,subprocess,sys,time
b=pathlib.Path(sys.argv[1]);root=pathlib.Path(sys.argv[2]);out=b/'cpu-benchmark'
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
try:
 with (b/'cpu-benchmark.log').open('ab') as log:
  r=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.benchmark_exact_uniform_g5_dp','--base',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1560)
 value=dict(returncode=r.returncode,epoch=time.time(),GPU_requested=0)
except BaseException as e:value=dict(returncode=None,error_type=type(e).__name__,error=str(e),epoch=time.time(),GPU_requested=0)
p=b/'cpu-exit.json';q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(value)+'\n');q.replace(p)
'''
START = r'''
guard=lock();pub,root=publication();assert (b/'cpu-tests.json').exists()
if (b/'cpu-launch.json').exists():
 v=read(b/'cpu-launch.json');print(json.dumps(dict(status='EXISTING_OWNED_CPU_LAUNCH_PRESERVED',launch=v,same_process_birth=ticks(v['pid'])==v['start_ticks'])))
else:
 assert not (b/'cpu-intent.json').exists(),'Unknown prior CPU result; do not repeat'
 value=dict(status='DP_CPU_BENCHMARK_INTENT',epoch=time.time(),source_publication=fp(b/'source-publication.json'),CPU_limit=2,memory_cap_bytes=3*1024**3,GPU_requested=0)
 atomic(b/'cpu-intent.json',value);wrapper=b/'cpu-wrapper.py'
 if wrapper.exists():assert wrapper.read_text()==inputs['wrapper']
 else:wrapper.write_text(inputs['wrapper'])
 with (b/'cpu-wrapper.log').open('ab') as log:p=subprocess.Popen(['python3','-u',str(wrapper),str(b),str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 value.update(status='CPU_STARTED_NOT_COMPLETION',pid=p.pid,start_ticks=ticks(p.pid),wrapper=fp(wrapper));atomic(b/'cpu-launch.json',value);print(json.dumps(value))
'''
PREPARE = r'''
guard=lock();pub,root=publication();bench=b/'cpu-benchmark/complete.json'
assert bench.exists() and read(b/'cpu-exit.json')['returncode']==0,'Real benchmark not complete; no launch or automatic retry'
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
args=runtime(root,['reproduction.operations.qwen7b_warmup.prepare_exact_uniform_g20_dp','--base',str(b),'--benchmark',str(bench)])
r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=50)
(b/'actual-cpu-preparation.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-4000:]
done=read(b/'preparation.json');assert done['GPU_requested']==0 and not done['actual_GPU_qualification'];print(json.dumps(done))
'''
STATUS = r'''
def optional(p):
 try:return read(p)
 except (FileNotFoundError,json.JSONDecodeError):return None
result=dict(epoch=time.time(),base=str(b),GPU_requested=0)
for filename in ('cpu-launch.json','cpu-exit.json','preparation.json'):
 value=optional(b/filename)
 if value is not None:result[filename]=value
launch=result.get('cpu-launch.json')
if launch:result['wrapper_same_process_birth']=ticks(launch['pid'])==launch['start_ticks']
bench=b/'cpu-benchmark'
for filename in ('controller.json','progress.json','complete.json','failure-controller.json','failure-old32.json','failure-dp.json'):
 value=optional(bench/filename)
 if value is not None:result[filename]=value
result['worker_progress']={p.name:read(p) for p in bench.glob('*-progress.json')}
print(json.dumps(result))
'''


def bundle(repo):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/codex/qwen7b-probe-warmup-20261019'], cwd=repo, text=True).split()[0]
    if commit != remote:
        raise ValueError('Exact current source must be pushed before CPU execution')
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=repo, text=True).splitlines()
    cache = {}
    def load(name):
        if name not in cache:
            cache[name] = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=repo)
        return cache[name]
    raw, report = make_bundle(paths, load, commit, seeds=SEEDS,
        resource_files=('verl/version/version',), manifest_name=MANIFEST)
    for name in SEEDS + (PACKAGE + 'deploy_exact_g5_dp_cpu.py',):
        if (repo / name).read_bytes() != load(name):
            raise ValueError('Unpublished runtime/operator change: ' + name)
    return commit, remote, raw, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-root', type=Path)
    parser.add_argument('--action', choices=('stage', 'test', 'start', 'prepare', 'status'), default='status')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    inputs = dict(base=str(TARGET), target=str(TARGET), commit='a' * 40,
        remote_sha='a' * 40, bundle_sha256='b' * 64, bundle_bytes=10,
        chunks=1, manifest=MANIFEST, manifest_sha256='c' * 64)
    actions = dict(init=INIT, chunk=CHUNK, extract=EXTRACT, test=TEST,
        start=START, prepare=PREPARE, status=STATUS)
    def source(action, extra=None):
        values = dict(inputs); values.update(extra or {})
        return 'inputs=' + repr(values) + '\n' + COMMON + '\n' + actions[action]
    if args.self_test:
        sizes = {}
        for action in actions:
            code = source(action, dict(chunk_index=0, chunk_data=base64.b64encode(b'x' * 4096).decode(),
                chunk_sha256='d' * 64, wrapper=WRAPPER))
            compile(code, action, 'exec')
            sizes[action] = len(('python3 -c ' + shlex.quote(code)).encode())
            if sizes[action] >= 16000:
                raise ValueError('Remote command too large: ' + action)
        print(json.dumps(dict(status='DP_CPU_COMMAND_RENDER_CHECKS_PASS', command_bytes=sizes,
            GPU_requested=0, remote_actions_executed=False)))
        return
    if args.control_root is None:
        parser.error('--control-root required')
    data = None
    if args.action == 'stage':
        commit, remote, data, report = bundle(repo)
        inputs.update(commit=commit, remote_sha=remote, bundle_sha256=hashlib.sha256(data).hexdigest(),
            bundle_bytes=len(data), chunks=(len(data) + 4095) // 4096,
            manifest_sha256=report['source_manifest_sha256'])
    spec = importlib.util.spec_from_file_location('dp_cpu_control', args.control_root / 'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    address = subprocess.check_output(['/usr/sbin/ipconfig', 'getifaddr', 'en0'], text=True).strip()
    def call(action, extra=None):
        command = 'python3 -c ' + shlex.quote(source(action, extra))
        if len(command.encode()) >= 16000:
            raise ValueError('Remote command exceeds16KiB')
        result = ctl.run(command, account='mou176', source_address=address, timeout=55)
        rows = [json.loads(line) for line in result.stdout.decode(errors='replace').splitlines() if line.startswith('{')]
        if result.returncode or not rows:
            raise RuntimeError('Unconfirmed CPU outcome; inspect without blind repeat: ' + result.stdout.decode(errors='replace')[-3000:] + result.stderr.decode(errors='replace')[-1000:])
        return rows[-1]
    if args.action == 'stage':
        call('init')
        for i, start in enumerate(range(0, len(data), 4096)):
            chunk = data[start:start + 4096]
            call('chunk', dict(chunk_index=i, chunk_data=base64.b64encode(chunk).decode(),
                chunk_sha256=hashlib.sha256(chunk).hexdigest()))
        result = call('extract')
    else:
        result = call(args.action, dict(wrapper=WRAPPER) if args.action == 'start' else None)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
