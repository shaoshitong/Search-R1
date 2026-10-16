"""One CPU storage refresh while the exact training job waits in the queue.

Use the original observer/completer in an isolated stage. Only a complete
admitted receipt replaces the live pointer; no Slurm or training mutation.
Each invocation needs a new output. An uncertain result requires inspection,
never replay. A long queue needs another real refresh before six-hour expiry.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

IDENTITY = dict(hostname='ACD-Login-1', boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a', pid_namespace='pid:[4026531836]')

STATE = '''
def state():
 r=subprocess.run(['scontrol','show','job','-d',job],capture_output=True,text=True,timeout=25)
 assert r.returncode==0 and not r.stderr
 f=dict(re.findall(r'(\\w+)=(\\S+)',r.stdout))
 assert f['JobId']==job and f['UserId']=='jxu957(14090)' and f['Account']=='jxu957'
 assert f['Command']==str(base/'source'/commit/'reproduction/prefix20_joint4/train.sh') and f['WorkDir']==str(base/'repo')
 return f['JobState']
assert state()=='PENDING' and not (base/'logs/run-manifest.json').exists(), 'Training has left the queue; do not publish'
'''


def main():
    p=argparse.ArgumentParser()
    for name in ('base','job','training-commit','bridge-source','control-work','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();base=Path(a.base)
    assert re.fullmatch('[0-9a-f]{40}',a.training_commit) and a.job.isdecimal()
    assert re.fullmatch('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-[A-Za-z0-9._-]+',str(base))
    output=Path(a.output);claim=output.with_name(output.name+'.claim.json')
    assert not output.exists() and not claim.exists(), 'Inspect the existing once-only operation'
    operator=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True)
    remote=subprocess.check_output(['git','ls-remote','git@github.com:shaoshitong/Search-R1.git','refs/heads/codex/qprobe-prefix20-hpc3-4gpu-20261016'],text=True).split()[0]
    assert remote==operator, 'Publish operator source first'
    tag=uuid.uuid4().hex
    stage=base.parent/('qprobe-teacher-warmup200-prefix20-storage-refresh'+a.job+'-'+tag)
    target=base/'preparation'/('queued-storage-refresh-'+tag)
    public=dict(base=str(base),job=a.job,training_commit=a.training_commit,operator_commit=operator,
                stage=str(stage),target=str(target),started_unix=time.time(),training_job_changed=False)
    with claim.open('x') as f:json.dump(public,f,indent=2)
    spec=importlib.util.spec_from_file_location('existing_account_bridge',a.bridge_source)
    bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    source=base/'source'/a.training_commit
    setup='import json,os,re,shutil,subprocess,time\nfrom pathlib import Path\n'
    setup+='base=Path('+repr(str(base))+');stage=Path('+repr(str(stage))+');target=Path('+repr(str(target))+');job='+repr(a.job)+';commit='+repr(a.training_commit)+'\n'
    observations=[]

    def call(user,code,label,timeout=55):
        command='CUDA_VISIBLE_DEVICES= QPROBE_SOURCE='+shlex.quote(str(source))+' PYTHONPATH='+shlex.quote(str(source))+' '+shlex.quote(str(base/'env/bin/python'))+' -B -c '+shlex.quote(setup+code)
        r=bridge.run(user,command,'',a.control_work,login_host='10.120.48.31',expected_host_identity=IDENTITY,observation_timeout=timeout)
        record=dict(phase=label,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,checked_unix=time.time())
        observations.append(record)
        output.with_name(output.name+'.'+label+'.json').write_text(json.dumps(dict(public,observation=record),indent=2)+'\n')
        if r.returncode:raise RuntimeError('Inspect saved '+label+' outcome; do not replay')
        return json.loads(r.stdout)

    try:
        call('jxu957',STATE+"os.umask(0o007)\nstage.mkdir(mode=0o770)\n(stage/'preparation').mkdir(mode=0o770)\nassert stage.stat().st_gid==21919\nprint(json.dumps(dict(stage=str(stage),state=state())))",'stage')
        # Slurm hides jxu957's jobs from mou176. Only the owner checks queue
        # state; this separate account only measures the isolated CPU stage.
        root=call('mou176',"from reproduction.prefix20_joint4.storage_observe import observe\nobserve(stage,commit)",'root',1790)
        if not root['measurement_complete']:
            assert root['returncode']==1 and not root['timed_out']
            result=Path(root['stdout_file']).parent/'result.json'
            call('jxu957',STATE+"from reproduction.prefix20_joint4.storage_complete import complete\ncomplete(stage,Path("+repr(str(result))+"),commit)",'complements',600)
        value=call('jxu957',STATE+'''
from reproduction.prefix20_joint4.control import publish,storage_admission
from reproduction.prefix20_joint4.assets import atomic,sha
from reproduction.prefix20_joint4.recover726927 import relocate_global_storage
from reproduction.prefix20_joint4.storage_complete import validate_upper_bound
target.mkdir();(target/'preparation').mkdir()
original=json.loads((stage/'preparation/personal-storage-observation.json').read_text())
if original.get('measurement_kind'):
 relocate_global_storage(stage,target,additional_prelaunch_reserve_bytes=4*2**30)
else:
 assert original['measurement_complete'] and original['returncode']==0 and not original['stderr']
 for key in ('stdout','stderr'):
  destination=target/'preparation'/('root.'+key)
  shutil.copyfile(original[key+'_file'],destination)
  assert sha(destination)==original[key+'_sha256']
  original[key+'_file']=str(destination)
 original['additional_prelaunch_reserve_bytes']=4*2**30
 publish(target/'preparation/personal-storage-observation.json',original)
proof=json.loads((target/'preparation/personal-storage-observation.json').read_text())
if proof.get('measurement_kind'):validate_upper_bound(proof,base)
admission=storage_admission(target)
assert admission['personal_budget_bytes']==2*10**12 and admission['accepted']
assert state()=='PENDING' and not (base/'logs/run-manifest.json').exists()
live=base/'preparation/personal-storage-observation.json'
shutil.copyfile(live,target/'preparation/live-before.json')
atomic(live,proof)
descriptor=os.open(live.parent,os.O_RDONLY)
try:os.fsync(descriptor)
finally:os.close(descriptor)
assert json.loads(live.read_text())==proof
assert storage_admission(base)['accepted']
result=dict(status='COMPLETE_STORAGE_RECEIPT_REFRESHED_WHILE_PENDING',job=job,admission=admission,
 root_measured_unix=proof.get('original_root_checked_unix',proof['checked_unix']),
 pointer=str(live),state_after=state(),training_job_changed=False,actual_gpu_allocation=False,checked_unix=time.time())
publish(target/'preparation/refresh-result.json',result)
print(json.dumps(result))
''','publish',90)
        public.update(result=value,observations=observations,finished_unix=time.time())
    except BaseException as error:
        public.update(error_type=type(error).__name__,error=str(error),observations=observations,finished_unix=time.time())
        output.write_text(json.dumps(public,indent=2)+'\n')
        raise
    output.write_text(json.dumps(public,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),**value)))


if __name__=='__main__':main()
