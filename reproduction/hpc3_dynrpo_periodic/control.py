"""Publish/fetch-verify code, deploy it, and reconcile one Slurm submission.

This local program transports source code and small receipts only. Secrets
remain inside the existing private-control stdin/RAM bridge.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from .common import BASE, PIN
from .local_control import run

REPO = Path(__file__).resolve().parents[2]
BRANCH = 'codex/dynrpo-periodic25-hpc3-20261016'
CONTROL_PYTHON = '/data/user/jxu957/experiments/topd-20261008/environment/topd-py310/bin/python'


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])


def publish(output):
    if git('status', '--porcelain').strip():
        raise ValueError('Commit the actual implementation before publication')
    commit = git('rev-parse', 'HEAD').decode().strip()
    subprocess.run(['git','-C',str(REPO),'push','origin','HEAD:refs/heads/'+BRANCH],check=True)
    subprocess.run(['git','-C',str(REPO),'fetch','origin',BRANCH],check=True)
    remote = git('rev-parse','FETCH_HEAD').decode().strip()
    if remote != commit:
        raise ValueError('Remote branch does not contain the current implementation')
    files = {}
    archive = io.BytesIO()
    names = git('ls-tree','-r','--name-only',commit).decode().splitlines()
    with tarfile.open(fileobj=archive,mode='w:gz') as tar:
        for name in names:
            if Path(name).suffix not in ('.py','.sh','.yaml','.yml','.json','.md') and name not in (
                    'verl/version/version','verl/single_controller/version/version'):
                continue
            content = git('show', commit+':'+name)
            fetched = git('show', remote+':'+name)
            if content != fetched:
                raise ValueError('Fetched source blob differs: '+name)
            files[name] = hashlib.sha256(content).hexdigest()
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(content), 0o755 if name.endswith('.sh') else 0o644
            tar.addfile(item,io.BytesIO(content))
    record = dict(commit=commit,remote_commit=remote,branch=BRANCH,pinned_base=PIN,
        pushed_and_fetched=True,files=files,checked_unix=time.time())
    output.mkdir(parents=True,exist_ok=True)
    (output/'source-publication.json').write_text(json.dumps(record,indent=2)+'\n')
    (output/'code-only.tar.gz').write_bytes(archive.getvalue())
    return record,archive.getvalue()


def deploy(publication, archive, output):
    plan = dict(base=str(BASE),source_commit=publication['commit'],
                target_outer=200,probe_boundaries=list(range(0,200,25)),
                save_boundaries=list(range(25,201,25)),fit_epochs=20,
                inherited_warmup_epochs=10,student_parameters_copied=False,
                preparation_partition='acd_u',formal_gpus=4)
    header = dict(publication=publication,plan=plan,
                  archive_sha256=hashlib.sha256(archive).hexdigest())
    script = (REPO/'reproduction/hpc3_dynrpo_periodic/bootstrap.py').read_text()
    result = run(CONTROL_PYTHON+' -c '+shlex.quote(script),
                 json.dumps(header).encode()+b'\n'+archive)
    receipt = dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode())
    (output/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode:
        raise RuntimeError('Deployment failed: '+receipt['stderr'])
    return receipt


def submit(phase, commit, output, partition='acd_u'):
    if phase not in ('preparation','formal') or partition not in ('acd_u','acd_ue','emergency_acd'):
        raise ValueError('Unknown phase or priority tier')
    cpus, memory, gpus = (12,'128G',1) if phase=='preparation' else (48,'512G',4)
    # This lock is independent of both the other chat and the runtime lock.
    script = '''import fcntl,json,os,subprocess,time
from pathlib import Path
b=Path(BASE);p=b/'preparation'/(PHASE+'-submission.json')
with (b/'preparation'/(PHASE+'-submit.lock')).open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 name='dynrpo-periodic5-'+PHASE
 active=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'--name='+name,'-o','%i|%T|%P|%R'],text=True).strip()
 if active:
  print(json.dumps(dict(status='EXISTING_OWN_JOB_NO_DUPLICATE',queue=active)));raise SystemExit(0)
 if p.exists():
  print(json.dumps(dict(status='EXISTING_SUBMISSION_REQUIRES_RECONCILIATION',receipt=json.loads(p.read_text()))));raise SystemExit(0)
 claim=b/'preparation'/(PHASE+'-submit-claim.json')
 with claim.open('x') as f:json.dump(dict(phase=PHASE,commit=COMMIT,partition=PARTITION,claimed_unix=time.time()),f);f.flush();os.fsync(f.fileno())
 entry=str(b/'source'/COMMIT/'reproduction/hpc3_dynrpo_periodic/run.sh')
 args=['sbatch','--parsable','--job-name='+name,'--partition='+PARTITION,'--time=7-00:00:00','--nodes=1','--ntasks=1','--cpus-per-task='+str(CPUS),'--mem='+MEMORY,'--gres=gpu:'+str(GPUS),'--chdir='+str(b),'--output='+str(b/'logs'/(PHASE+'-%j.log')),'--export=ALL','--wrap=bash '+entry+' '+PHASE+' '+str(b)+' '+COMMIT]
 if PHASE=='formal':args.insert(2,'--exclude=ACD1-21')
 r=subprocess.run(args,text=True,capture_output=True)
 value=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,phase=PHASE,partition=PARTITION,commit=COMMIT,submitted_unix=time.time())
 if r.returncode==0 and r.stdout.strip().split(';')[0].isdigit():value['job']=r.stdout.strip().split(';')[0]
 with p.open('x') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
 print(json.dumps(value))
'''
    prefix = '\n'.join(k+'='+repr(v) for k,v in dict(BASE=str(BASE),PHASE=phase,COMMIT=commit,
        PARTITION=partition,CPUS=cpus,MEMORY=memory,GPUS=gpus).items())+'\n'
    result = run(CONTROL_PYTHON+' -c '+shlex.quote(prefix+script))
    receipt = dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode())
    (output/(phase+'-submission.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode:
        raise RuntimeError('Submission control failed: '+receipt['stderr'])
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('publish-deploy','preparation','formal'))
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.action=='publish-deploy':
        record,archive=publish(args.output)
        result=deploy(record,archive,args.output)
    else:
        record=json.loads((args.output/'source-publication.json').read_text())
        result=submit(args.action,record['commit'],args.output)
    print(json.dumps(result))


if __name__=='__main__':main()
