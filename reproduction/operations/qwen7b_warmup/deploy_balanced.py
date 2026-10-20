"""Publish isolated balanced/rank1 phase and submit only after CPU preparation."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip(),'Commit and push latest code first'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]==commit
    spec=importlib.util.spec_from_file_location('control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();out=a.control_root/'outputs/qwen7b-warmup-20261019/balanced-deployment.txt'
    def call(code,timeout=45):
        r=control.run('python3 -c '+shlex.quote(code),account='mou176',source_address=address,timeout=timeout)
        with out.open('ab') as h:h.write(r.stdout);h.write(r.stderr)
        print(r.stdout.decode(),flush=True)
        if r.returncode:print(r.stderr.decode(),flush=True);raise SystemExit(r.returncode)
        return r.stdout.decode()
    prefix='commit='+repr(commit)+'\n'
    call(prefix+INIT)
    names=['balanced_sampling.py','balanced_loss.py','balanced_finetune.py','balanced_prepare.py','balanced_fit.sbatch','test_balanced_policy.py','deploy_balanced.py']
    for name in names:
        rel='reproduction/operations/qwen7b_warmup/'+name;data=(repo/rel).read_bytes();digest=hashlib.sha256(data).hexdigest()
        code=prefix+'rel='+repr(rel)+'\ndata='+repr(base64.b64encode(data).decode())+'\ndigest='+repr(digest)+'\n'+UPLOAD
        assert len(code.encode())<64000
        call(code)
    call(prefix+PREPARE,240)
    call(prefix+SUBMIT,50)


COMMON=r'''
import pathlib,json,subprocess,hashlib,base64,time,shutil,fcntl,os
parent=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');old=parent/'mixed-only-ft-cp2-prefix20-20261020';b=parent/'balanced-rank1-from-epoch1-prefix20-20261020';root=b/('repo-'+commit[:12])
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);assert r.returncode==0,(args,r.stderr);return r.stdout.strip()
'''
INIT=COMMON+r'''
assert not (b/'job.id').exists(),'Inspect existing submission; do not duplicate'
assert '742046' not in command(['squeue','-h','-u','mou176','-o','%i']).splitlines(),'Old writer still live'
with (old/'probe/fit.lock').open('a') as h:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not root.exists(),'Inspect partial deployment before retry'
intent=json.loads((b/'preparation/handoff.json').read_text());assert intent['source_checkpoint']['steps']==10464
pub=json.loads((old/'preparation/source-publication.json').read_text())
for rel,digest in pub['files'].items():assert hashlib.sha256((pathlib.Path(pub['code_root'])/rel).read_bytes()).hexdigest()==digest
shutil.copytree(pub['code_root'],str(root));pub.update(commit=commit,remote_sha=commit,code_root=str(root),base=str(b),time=time.time())
atomic(b/'preparation/source-publication.json',pub);print(json.dumps(dict(status='ISOLATED_NEW_SOURCE_ROOT_CREATED',commit=commit,code_root=str(root))))
'''
UPLOAD=COMMON+r'''
assert not (b/'job.id').exists();raw=base64.b64decode(data);assert hashlib.sha256(raw).hexdigest()==digest
(root/rel).write_bytes(raw);assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest
pub=json.loads((b/'preparation/source-publication.json').read_text());pub['files'][rel]=digest;atomic(b/'preparation/source-publication.json',pub)
print(json.dumps(dict(uploaded=rel,sha256=digest)))
'''
PREPARE=COMMON+r'''
pub=json.loads((b/'preparation/source-publication.json').read_text())
for rel,digest in pub['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
runtime=str(root/'reproduction/operations/qwen7b_warmup/runtime.sh');env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
for args in [['-m','unittest','reproduction.operations.qwen7b_warmup.test_balanced_policy','reproduction.operations.qwen7b_warmup.test_restore_isolation'],['-u','-m','reproduction.operations.qwen7b_warmup.balanced_prepare','--base',str(b),'--source-commit',commit]]:
 r=subprocess.run(['bash',runtime]+args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=180);assert r.returncode==0,r.stdout+r.stderr;print(r.stdout+r.stderr,flush=True)
print(json.dumps(dict(status='ACTUAL_CPU_RUNTIME_TESTS_AND_REAL_BANK_PREPARATION_PASSED',commit=commit,time=time.time())))
'''
SUBMIT=COMMON+r'''
assert not (b/'job.id').exists(),'Unknown prior submit must be reconciled rather than repeated'
assert not command(['squeue','-h','-u','mou176','-n','q7b-bal-rank1','-o','%i']),'Same-name job already present'
assert '742046' not in command(['squeue','-h','-u','mou176','-o','%i']).splitlines()
with (b/'probe/fit.lock').open('a') as h:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
pub=json.loads((b/'preparation/source-publication.json').read_text());assert pub['commit']==commit
for rel,digest in pub['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
acceptance=json.loads((b/'preparation/acceptance.json').read_text());assert acceptance['final_target_steps']==15736
record=json.loads((b/'preparation/handoff.json').read_text());record.update(status='BALANCED_RANK1_SUBMIT_INTENT',time=time.time(),source_commit=commit);atomic(b/'preparation/handoff.json',record)
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root))
r=subprocess.run(['sbatch','--hold','--parsable',str(root/'reproduction/operations/qwen7b_warmup/balanced_fit.sbatch')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);assert r.returncode==0,r.stderr
new=r.stdout.strip().split(';')[0];assert new.isdigit()
(b/'job.id').write_text(new+'\n');submission=dict(job_id=new,code_root=str(root),source_commit=commit,gpus=4,partition='emergency_acd',source_epoch=1,start_epoch=2,end_epoch=5,pair_weight=1.,time=time.time())
atomic(b/'submission.json',submission);record.update(status='BALANCED_RANK1_JOB_HELD',new_job=new,time=time.time());atomic(b/'preparation/handoff.json',record)
state=command(['squeue','-h','-j',new,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
command(['scontrol','release',new]);record.update(status='BALANCED_RANK1_JOB_RELEASED',time=time.time());atomic(b/'preparation/handoff.json',record)
print(json.dumps(submission));print(command(['squeue','-h','-j',new,'-o','%i|%u|%j|%T|%P|%R']))
'''


if __name__=='__main__':main()
