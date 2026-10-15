"""One CPU-only continuation of the diagnosed 927 prelaunch claim collision.

Preserve runtime/source/assets and every old claim. Call the published helper's
prepare_launch directly under a new exclusive claim; never call its CLI main.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
OPS = Path(__file__).resolve().parents[2]
FORMAL_PATH = 'reproduction/operations/a800_qprobe_qwen_native_formal.py'
FORMAL_SHA = '89aa7841a8290e3b0e3bcf64f6cc96b54a06037ec484c6a2caa492247ae371f8'
SOURCE = '927163041cf0ddf2c2c517efc466177d76dc2a18'
FAILED_BASE = '/data/shaoshitong/Search-G1-a800-fresh200-92bb58c18ae74ca9bc59fe1ba5dc02b6'
HANDLE = ROOT/'preparation/qprobe-qwen-native-formal-cpu-repair-v2-deployment.json'
HANDLE_SHA = '425da14c8d8aa5afc37e9ec0c71853fbf5c23e801b6e05713195352c25542e25'
DIAGNOSIS = ROOT/'preparation/qprobe-qwen-native-formal-92bb-prelaunch-entry-terminal-diagnosis.json'
DIAGNOSIS_SHA = 'ceb78ca00c91e9814bbf1f7def42ede2763f6d89efe066f16c2c54598385a440'
CLAIM = 'native-prelaunch-qualification-repair-v1'

REMOTE = r'''
bind(json.load(sys.stdin));sys.path.insert(0,str(repo))
assert intent['source_commit']=='927163041cf0ddf2c2c517efc466177d76dc2a18'
assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-92bb58c18ae74ca9bc59fe1ba5dc02b6'
from reproduction.a800_common import qprobe_qwen_formal_ops as helper
loaded_base,loaded_repo,loaded_intent=helper.load(base)
assert (loaded_base,loaded_repo,loaded_intent)==(base,repo,intent)
helper.unlaunched(base)
conflict_path=base/'logs/prepare-launch-once.json';conflict_raw=sealed(conflict_path)
conflict=json.loads(conflict_raw)
assert conflict['action']=='prepare' and conflict['source_commit']==intent['source_commit'] and conflict['no_automatic_retry'] is True
failed=document(base/'logs/prepare-launch-exit.json');assert failed['action']=='prepare-launch' and failed['exit_code']==1
log=sealed(base/'logs/prepare-launch-worker.log',1024*1024)
assert len(log)==1221 and hashlib.sha256(log).hexdigest()=='1794c33fa68d8bf05cdc2b7062f8fd1e6c59df6116b9422f87b2d6194a1828bf'
launch=document(base/'logs/prepare-launch-launch.json')
assert launch['action']=='prepare-launch' and launch['source_commit']==intent['source_commit'] and not process(launch['process'])['alive']
assert not (base/'logs/native-prelaunch.json').exists()
assert helper.required_cpu_checks(base,repo,intent)['tests_run']==23
os.environ.update(helper.environment(base,repo));assert os.environ['CUDA_VISIBLE_DEVICES']==''
claim=base/'logs/native-prelaunch-qualification-repair-v1-once.json'
once(claim,dict(source_commit=intent['source_commit'],operator_commit=p['operator_commit'],producer_identity=birth(os.getpid()),failed_worker_sha256=hashlib.sha256(log).hexdigest(),no_automatic_retry=True,claimed_unix=time.time()))
started=time.time()
try:
 value=helper.prepare_launch(base,repo,intent)
 helper.unlaunched(base);assert sealed(conflict_path)==conflict_raw
 assert value['stage']=='NATIVE_FRESH_FORMAL200_PRELAUNCH_QUALIFIED' and value['source_commit']==intent['source_commit']
 assert value['actor_started'] is False and value['actual_cuda_restore_accepted'] is False
 once(base/'logs/native-prelaunch-qualification-repair-v1-exit.json',dict(exit_code=0,source_commit=intent['source_commit'],started_unix=started,finished_unix=time.time(),prelaunch_sha256=sha(base/'logs/native-prelaunch.json'),no_automatic_retry=True))
 emit(dict(stage='NATIVE_PRELAUNCH_CLAIM_COLLISION_REPAIR_QUALIFIED',source_commit=intent['source_commit'],base=str(base),qualification=value,prelaunch_sha256=sha(base/'logs/native-prelaunch.json'),old_claim_preserved=True,actor_started=False))
except BaseException as failure:
 import traceback
 frames=[dict(file=Path(item.filename).name,line=item.lineno,function=item.name) for item in traceback.extract_tb(failure.__traceback__)[-8:]]
 once(base/'logs/native-prelaunch-qualification-repair-v1-failure.json',dict(failure_type=type(failure).__name__,frames=frames,source_commit=intent['source_commit'],started_unix=started,finished_unix=time.time(),no_automatic_retry=True))
 once(base/'logs/native-prelaunch-qualification-repair-v1-exit.json',dict(exit_code=1,source_commit=intent['source_commit'],started_unix=started,finished_unix=time.time(),no_automatic_retry=True))
 raise
'''


def support():
    commit = subprocess.check_output(['git','-C',str(OPS),'rev-parse','HEAD'], text=True).strip()
    own = Path(__file__).resolve(); relative = str(own.relative_to(OPS))
    assert subprocess.check_output(['git','-C',str(OPS),'show',commit+':'+relative]) == own.read_bytes(), 'Commit this exact continuation before execution'
    path = OPS/FORMAL_PATH; raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FORMAL_SHA
    assert subprocess.check_output(['git','-C',str(OPS),'show',commit+':'+FORMAL_PATH]) == raw
    spec = importlib.util.spec_from_file_location('published_native_formal_continuation_support', path)
    formal = importlib.util.module_from_spec(spec); spec.loader.exec_module(formal)
    warmup, _, base = formal.support()
    common = formal.remote_code(base)
    assert common.endswith('\n'+formal.REMOTE)
    code = common[:-len('\n'+formal.REMOTE)] + '\n' + REMOTE
    ast.parse(code)
    return formal, warmup, base, commit, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handle', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    formal, warmup, base, commit, code = support()
    handle = args.handle.resolve(); receipt = args.receipt.resolve()
    assert handle == HANDLE and base.digest(handle) == HANDLE_SHA
    assert base.digest(DIAGNOSIS) == DIAGNOSIS_SHA
    diagnosis = base.document(DIAGNOSIS)
    assert diagnosis['expected_source_commit'] == diagnosis['git_head'] == SOURCE and diagnosis['git_status'] == '' and diagnosis['actor_claim_exists'] is False
    intent = base.document(handle); formal.authority(warmup,base,intent)
    assert intent['source_commit'] == SOURCE and intent['base'] == FAILED_BASE
    assert receipt.parent == ROOT/'preparation' and receipt.suffix == '.json' and not receipt.exists()
    base.once(handle.with_name(handle.stem+'.'+CLAIM+'.claim.json'), dict(source_commit=SOURCE,operator_commit=commit,no_automatic_retry=True,claimed_unix=time.time()))
    route = base.runtime_module().ssh_command(code)
    assert route[-1].startswith('python3 -c ')
    route[-1] = base.PYTHON+' -c '+route[-1][len('python3 -c '):]
    result = dict(role=CLAIM,operator_commit=commit,source_commit=SOURCE,base=FAILED_BASE,handle_sha256=HANDLE_SHA,diagnosis_sha256=DIAGNOSIS_SHA,no_automatic_retry=True,observation=None)
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            done = subprocess.run(route,input=base.canonical(dict(intent=intent,operator_commit=commit)),stdout=out,stderr=err,timeout=300)
            result['exit_code'] = done.returncode
        except (OSError,subprocess.TimeoutExpired) as failure:
            result.update(exit_code=None,failure_type=type(failure).__name__,status='TRANSPORT_UNKNOWN_OBSERVE_SAME_BASE_NO_RETRY')
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(1024*1024+1);err.seek(0)
        result['bounded_stderr_sha256']=hashlib.sha256(err.read(256*1024)).hexdigest()
    if len(raw)<=1024*1024:
        rows=[json.loads(line.split(':',1)[1]) for line in raw.decode(errors='strict').splitlines() if line.startswith('TOKEN_WARMUP_JSON:')]
        assert len(rows)<=1
        if rows:result['observation']=rows[0]
    result['checked_unix']=time.time();base.once(receipt,result)
    print(json.dumps(dict(receipt=str(receipt),**result),allow_nan=False))
    return 0 if result['exit_code']==0 and result['observation'] is not None else 1


if __name__=='__main__':raise SystemExit(main())
