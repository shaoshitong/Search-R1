"""Repair only failed, uncommitted CP2 trial; preserve all immutable artifacts."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]==commit
    files={}
    for name in ['mixed_finetune.py','mixed_prepare.py','test_restore_isolation.py']:
        rel='reproduction/operations/qwen7b_warmup/'+name;data=(repo/rel).read_bytes();files[rel]=dict(data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest())
    sp=importlib.util.spec_from_file_location('control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    script='new_commit='+repr(commit)+'\nnew_files='+repr(files)+'\n'+REMOTE
    result=m.run('python3 -c '+shlex.quote(script),account='mou176',source_address=address,timeout=170)
    out=a.control_root/'outputs/qwen7b-warmup-20261019/mixed-alias-repair-handoff.txt';out.write_bytes(result.stdout)
    print('returncode',result.returncode);print(result.stdout.decode());print(result.stderr.decode())
    if result.returncode:raise SystemExit(result.returncode)

REMOTE=r'''
import pathlib,json,subprocess,hashlib,shutil,base64,fcntl,time,os
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/mixed-only-ft-cp2-20261020')
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=45);assert r.returncode==0,(args,r.stderr);return r.stdout.strip()
old=(b/'job.id').read_text().strip();assert old=='741944'
assert not command(['squeue','-h','-j',old,'-o','%T'])
accounting=command(['sacct','-X','-n','-P','-j',old,'--format=JobID,State,ExitCode']);assert accounting.startswith(old+'|FAILED|1:0'),accounting
assert not any((b/'probe'/n).exists() for n in ['progress.json','latest.json','next-update-proof.json']),'Preserve real accepted updates, do not reinitialize'
with (b/'probe/fit.lock').open('a') as h:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
record_path=b/'alias-repair-handoff.json';assert not record_path.exists(),'Resolve previous repair outcome before submitting again'
c=json.loads((b/'config.json').read_text());pub=json.loads((b/'preparation/source-publication.json').read_text());old_pub=json.loads(json.dumps(pub))
record=dict(status='FAILED_TRIAL_REPAIR_INTENT',old_job=old,old_job_state=accounting,time=time.time(),root_cause='CPU Adam state tensors shared with reusable checkpoint object in next-update replay; source file immutable; no accepted continuation updates',source_steps=9146,new_epochs=5,new_source_commit=new_commit,goal_created_at=1792496306)
atomic(record_path,record)
h=hashlib.sha256()
with open(c['source_checkpoint']['path'],'rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==c['source_checkpoint']['sha256'];record['source_cp2_fresh_sha256']=h.hexdigest();atomic(record_path,record)
root=b/('repo-'+new_commit[:12]);assert not root.exists();shutil.copytree(pub['code_root'],str(root))
for rel,proof in new_files.items():
 data=base64.b64decode(proof['data']);assert hashlib.sha256(data).hexdigest()==proof['sha256'];(root/rel).write_bytes(data);pub['files'][rel]=proof['sha256']
for rel,digest in pub['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
cpu=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','reproduction.operations.qwen7b_warmup.test_restore_isolation'],env=dict(os.environ,QWEN7B_CODE_ROOT=str(root)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=45)
assert cpu.returncode==0,cpu.stdout+cpu.stderr
record['hpc3_cpu_regression']=cpu.stdout.strip();print(cpu.stdout,flush=True)
history=b/'preparation/failure-741944';history.mkdir(exist_ok=True)
for name in ['restore-acceptance.json']:
 p=b/'probe'/name
 if p.exists():p.rename(history/name)
atomic(history/'source-publication.json',old_pub);shutil.copyfile(str(b/'config.json'),str(history/'config.json'))
c.update(fit_epochs=5,source_commit=new_commit,fit_file_sha256=new_files['reproduction/operations/qwen7b_warmup/mixed_finetune.py']['sha256'])
pub.update(commit=new_commit,remote_sha=new_commit,code_root=str(root),time=time.time())
atomic(b/'config.json',c);atomic(b/'preparation/source-publication.json',pub)
r=subprocess.run(['sbatch','--hold','--parsable','--partition=emergency_acd','--job-name=q7b-mixed-cp2-high',str(root/'reproduction/operations/qwen7b_warmup/mixed_fit.sbatch')],env=dict(os.environ,QWEN7B_CODE_ROOT=str(root)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40);assert r.returncode==0,r.stderr
new=r.stdout.strip().split(';')[0];assert new.isdigit();record.update(new_job=new,new_code_root=str(root),status='REPAIR_FIVE_EPOCH_JOB_HELD');atomic(record_path,record)
shutil.copyfile(str(b/'submission.json'),str(b/('submission-'+old+'.json')))
atomic(b/'submission.json',dict(job_id=new,partition='emergency_acd',gpus=4,code_root=str(root),source_commit=new_commit,replaces_failed_trial=old,additional_epochs=5,time=time.time()))
t=b/'job.id.tmp';t.write_text(new+chr(10));t.replace(b/'job.id');command(['scontrol','release',new])
record.update(status='REPAIRED_FIVE_EPOCH_JOB_RELEASED',time=time.time());atomic(record_path,record)
print(json.dumps(record));print(command(['squeue','-h','-u','mou176','-o','%i|%j|%T|%P|%R']))
'''
if __name__=='__main__':main()
