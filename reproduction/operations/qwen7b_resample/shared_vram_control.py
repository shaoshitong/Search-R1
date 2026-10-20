"""Once native systemd control for authorized spare-VRAM sharing on h100-5.

Stage only small exact sources on NAS10. The independent bounded unit uses
CPUQuota=1600%, MemoryMax=160G and RuntimeMaxSec=28800; it changes no Slurm job.
Private mounts and networking preserve the original frozen asset/config paths.
Only this unit's processes belong to its cgroup. Review/publication is required
before stage or submit, and unknown submission outcomes are never replayed.
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
JOB_NAME = 'sst-q7b-g5-sharedvram-v6-' + RUN_ID[:12]
CONTROL_DIRECTORY = 'shared-vram-native-control-v6'
QUEUE_MODE = 'native-spare-vram-bounded-systemd-v6'
SOURCES = ('shared_vram_control.py', 'shared_vram_worker.py', 'shared_vram_retriever.py',
           'shared_vram_vllm_adapter.py', 'pool_worker.py', 'pool_paths.sh')
UNIT = JOB_NAME + '.service'
PREDECESSOR_UNIT = 'sst-q7b-g5-sharedvram-v3-' + RUN_ID[:12] + '.service'
PREDECESSOR_UNITS = tuple('sst-q7b-g5-sharedvram-v%d-' % version + RUN_ID[:12] + '.service' for version in (3, 4, 5))
RUN_SECONDS = 28500
BATCH_RESERVE_SECONDS = 300


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def bootstrap_source():
    return ("BASE = " + repr(BASE) + "\nFROZEN_REPO = " + repr(FROZEN_REPO)
            + "\nQUEUE_MODE = " + repr(QUEUE_MODE) + "\nRUN_SECONDS = " + repr(RUN_SECONDS)
            + "\nBATCH_RESERVE_SECONDS = " + repr(BATCH_RESERVE_SECONDS) + "\n" + r'''
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
stage=Path(__file__).resolve().parent
manifest=json.loads((stage/'manifest.json').read_text())
assert os.geteuid()==0 and socket.gethostname().split('.')[0]=='h100-5'
assert os.stat('/proc/self/ns/mnt').st_ino != os.stat('/proc/1/ns/mnt').st_ino
for name,row in manifest['files'].items():
    path=stage/name;data=path.read_bytes()
    assert path.is_file() and not path.is_symlink()
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],name
base=Path(BASE);assert (base/'config.json').is_file() and Path(FROZEN_REPO).is_dir()
unit=os.environ['Q7B_NATIVE_UNIT_INSTANCE'];assert unit==manifest['native_unit']
launch=base/'shared-vram/launch'/('native-'+str(os.getpid()))
launch.mkdir(parents=True,exist_ok=False)
log=(launch/'bootstrap.log').open('ab',buffering=0)
os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
properties=['Id','MainPID','ControlGroup','ActiveState','CPUQuotaPerSecUSec','MemoryMax','RuntimeMaxUSec']
detail=subprocess.check_output(['systemctl','show',unit,*['--property='+x for x in properties]],text=True,timeout=30)
values=dict(line.split('=',1)for line in detail.splitlines()if'='in line)
assert values['Id']==unit and int(values['MainPID'])==os.getpid()
assert values['ControlGroup'] and values['ControlGroup']in Path('/proc/self/cgroup').read_text()
assert values['MemoryMax']=='171798691840'
evidence=launch/'allocation-evidence.json'
with evidence.open('x') as stream:
    json.dump(dict(mode='NATIVE_SHARED_VRAM',node=socket.gethostname().split('.')[0],unit=unit,
        main_pid=os.getpid(),cgroup=values['ControlGroup'],captured_epoch=time.time(),unit_show=detail,
        resource_limits=dict(cpu_quota_percent=1600,memory_max_bytes=171798691840,runtime_max_seconds=28800),
        original_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),stream)
os.chmod(evidence,0o400)
env=dict(os.environ,PYTHONPATH=FROZEN_REPO+':/dataset/shaoshitong/Search-G1-common8/retriever-gpu-lib',
    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
    SEARCH_G1_B_PREFIX_TOKENS='20',SEARCH_G1_PHYSICAL_WORLD_SIZE='4',SEARCH_G1_BASE=BASE,
    HF_HOME=BASE+'/cache/huggingface',TORCH_EXTENSIONS_DIR=BASE+'/cache/torch_extensions',
    TMPDIR=str(launch/'tmp'),VLLM_ATTENTION_BACKEND='XFORMERS',VLLM_HOST_IP='127.0.0.1',
    CUDA_VISIBLE_DEVICES='3,4,6,7',
    CPATH='/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:'
          '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:'
          '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10')
for name in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
    env.pop(name,None)
for name in ('HF_HOME','TORCH_EXTENSIONS_DIR','TMPDIR'):
    Path(env[name]).mkdir(parents=True,exist_ok=True)
os.chdir(FROZEN_REPO)
command=['unshare','--net','/bin/bash','-c',
    'set -euo pipefail; ip link set dev lo up; exec "$@"','shared-vram-private-net',
    '/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u',str(stage/'shared_vram_worker.py'),
    '--base',BASE,'--handoff-marker',BASE+'/preparation/pool-handoff.json',
    '--allocation-evidence',str(evidence),'--pool-worker',str(stage/'pool_worker.py'),
    '--retriever-script',str(stage/'shared_vram_retriever.py'),
    '--model-gpus','3,4,6,7','--retrieval-gpus','0,1,2,5','--retrieval-weights','15,9,20,20',
    '--tensor-parallel-size','4',
    '--retrieval-shards','64','--model-max-seqs','128','--model-batch-tokens','16384','--model-kv-block-cap','16384',
    '--encoder',BASE.rsplit('/',1)[0]+'/search-g1-prefix5-h10011-runtime-assets-20261018/common8/assets/e5-base-v2',
    '--run-seconds',str(RUN_SECONDS),'--batch-reserve-seconds',str(BATCH_RESERVE_SECONDS)]
print(json.dumps(dict(status='EXACT_SHARED_VRAM_STAGE_VERIFIED_ENTERING_PRIVATE_NET',
    source_bundle_sha256=manifest['source_bundle_sha256'],queue_mode=QUEUE_MODE,unit=unit,main_pid=os.getpid(),
    model_devices='3,4,6,7',model_groups=[[3,4,6,7]],tensor_parallel_size=4,retrieval_devices='0,1,2,5',retrieval_weights='15,9,20,20',
    retrieval_logical_shards=64,model_max_num_seqs=128,model_max_batched_tokens=16384,model_kv_block_cap=16384,
    worker=str(stage/'shared_vram_worker.py'),allocation_evidence=str(evidence),epoch=time.time())),flush=True)
os.execvpe(command[0],command,env)
''').encode()


def native_paths_source(original):
    text = original.decode()
    expected = '[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]'
    old_stash = '/run/q7b-pool-nas10-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}'
    assert text.count(expected) == 1 and text.count(old_stash) == 1
    return text.replace(expected, '[[ ${Q7B_NATIVE_UNIT_INSTANCE:-} =~ ^[a-zA-Z0-9_.-]+$ ]]').replace(
        old_stash, '/run/q7b-shared-vram-nas10-${Q7B_NATIVE_UNIT_INSTANCE}').encode()


def bundle():
    sources = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    source_rows = {name: dict(bytes=len(data), sha256=digest(data)) for name, data in sources.items()}
    identity = digest(encoded(dict(queue_mode=QUEUE_MODE, source_files=source_rows)))
    files = dict(sources, **{'shared_vram_bootstrap.py': bootstrap_source(),
                            'shared_vram_native_paths.sh': native_paths_source(sources['pool_paths.sh'])})
    manifest = dict(version=1, source_bundle_sha256=identity, queue_mode=QUEUE_MODE,
                    native_unit=UNIT, explicit_native_spare_vram_sharing=True,
                    resource_limits=dict(cpu_quota_percent=1600,memory_max_bytes=171798691840,runtime_max_seconds=28800),
                    model_gpus=[3,4,6,7], model_groups=[[3,4,6,7]], tensor_parallel_size=4, model_engine_count=1,
                    retrieval_gpus=[0,1,2,5], retrieval_weights=[15,9,20,20],
                    retrieval_logical_shards=64, model_max_num_seqs=128, model_max_batched_tokens=16384,
                    model_kv_block_cap=16384,
                    run_id=RUN_ID, seed=SEED, frozen_runtime_commit=FROZEN_COMMIT, canonical_base=BASE,
                    source_files=source_rows, files={name: dict(bytes=len(data), sha256=digest(data))
                                                    for name, data in files.items()})
    assert sum(len(x) for x in files.values()) < 256 * 1024, 'Small code only'
    return identity, files, manifest


def launch_arguments(stage, native_paths):
    return ['systemd-run','--unit='+UNIT,'--property=KillMode=control-group',
            '--property=TimeoutStopSec=60','--property=Restart=no','--property=CPUQuota=1600%',
            '--property=MemoryMax=160G','--property=RuntimeMaxSec=28800','--property=TasksMax=2048',
            '--setenv=Q7B_NATIVE_UNIT_INSTANCE='+UNIT,
            'unshare','--mount','--propagation','private','/bin/bash','-c',native_paths.decode(),
            'shared-vram-private-mount','h100-5','/usr/bin/python3',str(stage/'shared_vram_bootstrap.py')]


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
import base64,fcntl,hashlib,json,os,socket,subprocess,time
from pathlib import Path
assert os.geteuid()==0 and socket.gethostname().split('.')[0]==('h100-10'if ACTION=='stage'else'h100-5')
base=Path(PHYSICAL if ACTION=='stage'else BASE);root=base/'preparation'/CONTROL_DIRECTORY;stage=root/IDENTITY
config=json.loads((base/'config.json').read_text())
assert config['run_id']==RUN_ID and config['seed']==SEED
def read(p):return json.loads(p.read_text()) if p.exists() else None
def publish(p,value):
    temp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
    with temp.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)
def query(unit=UNIT):
    props=['Id','LoadState','MainPID','ActiveState','SubState','Result','ExecMainStatus','ControlGroup',
           'RuntimeMaxUSec','CPUQuotaPerSecUSec','MemoryMax']
    command=['systemctl','show',unit,*['--property='+x for x in props]]
    r=subprocess.run(command,text=True,capture_output=True,timeout=30)
    values=dict(x.split('=',1)for x in r.stdout.splitlines()if'='in x)
    known=values.get('LoadState')in('loaded','not-found')
    return dict(unit=unit,present=values.get('LoadState')=='loaded',known=known,
        properties=values,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,epoch=time.time())
def verify_stage():
    saved=read(stage/'manifest.json');assert saved==MANIFEST,'Different stage manifest'
    for name,row in saved['files'].items():
        p=stage/name;data=p.read_bytes()
        assert p.is_file()and not p.is_symlink()and len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
    assert(read(stage/'stage-receipt.json')or{}).get('status')=='EXACT_SMALL_CODE_STAGE_SHA_VERIFIED'
value=dict(action=ACTION,queue_mode=QUEUE_MODE,source_bundle_sha256=IDENTITY,stage=str(stage),epoch=time.time())
if ACTION=='stage':
    assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
    assert PUBLICATION['source_files']==MANIFEST['source_files']
    root.mkdir(parents=True,exist_ok=True);stage.mkdir(exist_ok=True)
    intent=stage/'stage-intent.json'
    if not intent.exists():
        with intent.open('x')as f:json.dump(dict(intent_id=INTENT,action=ACTION,manifest=MANIFEST,publication=PUBLICATION,epoch=time.time()),f)
    assert read(intent)['manifest']==MANIFEST
    for name,text in FILES.items():
        assert'/'not in name and name not in('.','..')
        p=stage/name;data=base64.b64decode(text)
        if not p.exists():
            with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==MANIFEST['files'][name]['sha256'],name
        os.chmod(p,0o555)
    if not(stage/'manifest.json').exists():publish(stage/'manifest.json',MANIFEST)
    assert read(stage/'manifest.json')==MANIFEST
    receipt=dict(status='EXACT_SMALL_CODE_STAGE_SHA_VERIFIED',intent_id=INTENT,queue_mode=QUEUE_MODE,
        source_bundle_sha256=IDENTITY,files=MANIFEST['files'],epoch=time.time())
    publish(stage/'stage-receipt.json',receipt);verify_stage();value['receipt']=receipt
elif ACTION=='submit':
    verify_stage();assert PUBLICATION['status']=='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'
    assert PUBLICATION['source_files']==MANIFEST['source_files']
    with(root/'submit.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        intent=root/'native-submit-intent.json';receipt=root/'native-submit-receipt.json'
        old=read(intent);observed=query();value['observed']=observed
        if old:
            value.update(status='EXISTING_INTENT_NEVER_RESTARTED',intent=old,receipt=read(receipt))
            if not value['receipt']and observed['present']:
                recovered=dict(status='RECOVERED_EXISTING_NATIVE_UNIT_BY_NAME',intent=old,observed=observed,epoch=time.time())
                publish(receipt,recovered);value['receipt']=recovered
            elif not observed['present']and not value['receipt']:value['status']='NATIVE_START_OUTCOME_UNKNOWN_NO_RETRY'
        elif observed['present']:value.update(status='EXISTING_NATIVE_UNIT_FOUND_NO_START',observed=observed)
        else:
            assert observed['known'],'Cannot prove unique unit name is absent'
            value['predecessors']=[]
            for previous_unit in PREDECESSOR_UNITS:
                previous=query(previous_unit);value['predecessors'].append(previous)
                assert previous['known'] and int(previous['properties'].get('MainPID','0'))==0 \
                    and previous['properties'].get('ActiveState')not in('active','activating','deactivating'), \
                    'Previous own native sampler is still active; keep it running and do not create a second intent'
            proposed=dict(status='NATIVE_UNIT_INTENT_RECORDED',intent_id=INTENT,queue_mode=QUEUE_MODE,
                run_id=RUN_ID,seed=SEED,source_bundle_sha256=IDENTITY,unit=UNIT,
                argv=LAUNCH,publication=PUBLICATION,epoch=time.time())
            with intent.open('x')as f:json.dump(proposed,f,indent=2);f.flush();os.fsync(f.fileno())
            try:
                r=subprocess.run(LAUNCH,text=True,capture_output=True,timeout=60)
                result=dict(status='BOUNDED_NATIVE_SHARED_VRAM_UNIT_STARTED_ONCE'if r.returncode==0 else'NATIVE_START_FAILED_NO_RETRY',
                    returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,intent=proposed,epoch=time.time())
            except subprocess.TimeoutExpired as error:
                result=dict(status='NATIVE_START_OUTCOME_UNKNOWN_NO_RETRY',intent=proposed,error=str(error),epoch=time.time())
            result['observed']=query();publish(receipt,result);value.update(status=result['status'],receipt=result)
else:
    value.update(status='READ_ONLY_NATIVE_SHARED_VRAM_STATUS',intent=read(root/'native-submit-intent.json'),
        receipt=read(root/'native-submit-receipt.json'),stage_receipt=read(stage/'stage-receipt.json'),observed=query())
    latest=[]
    for p in sorted((base/'pool/jobs').glob('job-shared5-*/attempt-*')):
        latest.append(dict(directory=str(p),progress=read(p/'progress.json'),complete=read(p/'complete.json'),failure=read(p/'failure.json')))
    value.update(latest_attempts=latest[-8:],global_goal_complete=False)
print(json.dumps(value,sort_keys=True))
'''


def transport(script, node):
    modules = {}
    for name, path in [('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')]:
        spec = importlib.util.spec_from_file_location('pool_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    # Honor the user's configured direct mgmt-host alias; never force Mac7892.
    argv = modules['ctl'].route(node, '/usr/bin/python3 -')
    from .management_session import apply
    argv = apply(argv)
    return subprocess.run(argv, input=script, text=True, capture_output=True, timeout=110)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'stage', 'submit', 'status'))
    parser.add_argument('--intent-id', help='Explicit operator intent label; mandatory for stage/submit')
    parser.add_argument('--published-commit', help='Full SHA containing the exact six small source files')
    parser.add_argument('--published-ref', help='Origin refs/heads/BRANCH verified before stage/submit')
    args = parser.parse_args()
    identity, files, manifest = bundle()
    stage = Path(PHYSICAL) / 'preparation' / CONTROL_DIRECTORY / identity
    if args.action == 'plan':
        print(json.dumps(dict(action='LOCAL_ONLY_PLAN',queue_mode=QUEUE_MODE,manifest=manifest,
            physical_stage=str(stage),native_unit=UNIT,resource_limits=manifest['resource_limits'],
            handoff_marker=BASE+'/preparation/pool-handoff.json'),indent=2))
        return
    if args.action in ('stage', 'submit') and not re.fullmatch(r'[A-Za-z0-9._-]{1,120}', args.intent_id or ''):
        parser.error('stage/submit require --intent-id (letters, digits, dots, hyphens, underscores)')
    publication = verify_publication(args.published_commit,args.published_ref,manifest['source_files']) if args.action in ('stage','submit') else None
    values = dict(PHYSICAL=PHYSICAL,IDENTITY=identity,BASE=BASE,RUN_ID=RUN_ID,SEED=SEED,
                  JOB_NAME=JOB_NAME,QUEUE_MODE=QUEUE_MODE,CONTROL_DIRECTORY=CONTROL_DIRECTORY,
                  PREDECESSOR_UNITS=PREDECESSOR_UNITS,
                  ACTION=args.action,INTENT=args.intent_id,PUBLICATION=publication,
                  MANIFEST=manifest,FILES={name:base64.b64encode(data).decode() for name,data in files.items()},
                  UNIT=UNIT,LAUNCH=launch_arguments(Path(BASE)/'preparation'/CONTROL_DIRECTORY/identity,files['shared_vram_native_paths.sh']))
    script = '\n'.join(name+'='+repr(value) for name,value in values.items()) + '\n' + REMOTE
    record = ROOT / 'work' / ('qwen7b-shared-vram-control-%s-%d.json' % (args.action,time.time_ns()))
    try:
        node = 'h100-10' if args.action == 'stage' else 'h100-5'
        if node == 'h100-5':
            mount_command=['unshare','--mount','--propagation','private','/bin/bash','-c',
                files['shared_vram_native_paths.sh'].decode(),'shared-vram-control-mount','h100-5','/usr/bin/python3','-']
            script='import os,subprocess\nr=subprocess.run('+repr(mount_command)+',input='+repr(script)+',env=dict(os.environ,Q7B_NATIVE_UNIT_INSTANCE='+repr(UNIT)+'),text=True,capture_output=True,timeout=95)\nprint(r.stdout,end="");print(r.stderr,end="",file=__import__("sys").stderr);raise SystemExit(r.returncode)\n'
        result = transport(script,node)
    except subprocess.TimeoutExpired:
        record.write_text(json.dumps(dict(action=args.action,queue_mode=QUEUE_MODE,
            status='TRANSPORT_TIMEOUT_QUERY_STATUS_DO_NOT_REPEAT_SUBMIT',
            source_bundle_sha256=identity,epoch=time.time()),indent=2)+'\n')
        raise
    envelope = dict(action=args.action,queue_mode=QUEUE_MODE,returncode=result.returncode,
                    stdout=result.stdout,stderr=result.stderr,source_bundle_sha256=identity,
                    local_receipt=str(record),epoch=time.time())
    record.write_text(json.dumps(envelope,indent=2)+'\n')
    print(json.dumps(envelope,indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
