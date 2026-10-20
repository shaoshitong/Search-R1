"""Amend a never-started pending continuation to five epochs, preserving source."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[3];assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]==commit
    files={}
    for name in ['mixed_finetune.py','mixed_prepare.py']:
        rel='reproduction/operations/qwen7b_warmup/'+name;data=(repo/rel).read_bytes();files[rel]=dict(data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest())
    spec=importlib.util.spec_from_file_location('control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    script='new_commit='+repr(commit)+'\nnew_files='+repr(files)+'\n'+REMOTE
    result=m.run('python3 -c '+shlex.quote(script),account='mou176',source_address=address,timeout=150)
    out=a.control_root/'outputs/qwen7b-warmup-20261019/mixed-five-epoch-budget-handoff.txt';out.write_bytes(result.stdout)
    print('returncode',result.returncode);print(result.stdout.decode());print(result.stderr.decode())
    if result.returncode:raise SystemExit(result.returncode)

REMOTE=r'''
import json,pathlib,shutil,hashlib,base64,subprocess,fcntl,os,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/mixed-only-ft-cp2-20261020')
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40);assert r.returncode==0,(args,r.stderr);return r.stdout.strip()
def row(job):
 s=command(['squeue','-h','-j',job,'-o','%i|%u|%j|%T|%P|%R']);assert s and len(s.splitlines())==1,s;return s.split('|')
old=(b/'job.id').read_text().strip();assert old=='741944'
r=row(old);assert r[1:5]==['mou176','q7b-mixed-cp2-high','PENDING','emergency_acd'],r
assert not any((b/'probe'/name).exists() for name in ['progress.json','restore-acceptance.json','latest.json'])
with (b/'probe/fit.lock').open('a') as h:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
record_path=b/'five-epoch-amendment.json';assert not record_path.exists(),'Inspect previous outcome first'
publication=json.loads((b/'preparation/source-publication.json').read_text());config=json.loads((b/'config.json').read_text());assert config['fit_epochs']==4
record=dict(status='GOAL_FIVE_EPOCH_AMENDMENT_INTENT',goal_created_at=1792496306,goal='监督直到完成探针训练5epoch，如果探针训练5epoch还没过拟合，那么继续训练，然后检查2048选择512后的混合比例，如果还是有问题，那么继续思考，解决过拟合问题，甚至增加数据集',recorded_time=time.time(),scope='Only replace never-started owned pending741944, no running process cancellation',old_job=old,old_source_commit=publication['commit'],new_source_commit=new_commit,previous_epochs=4,new_epochs=5)
atomic(record_path,record);command(['scontrol','hold',old]);assert 'JobHeldUser' in row(old)[-1]
new_root=b/('repo-'+new_commit[:12]);assert not new_root.exists();shutil.copytree(publication['code_root'],str(new_root))
for rel,proof in new_files.items():
 data=base64.b64decode(proof['data']);assert hashlib.sha256(data).hexdigest()==proof['sha256'];(new_root/rel).write_bytes(data);publication['files'][rel]=proof['sha256']
for rel,digest in publication['files'].items():assert hashlib.sha256((new_root/rel).read_bytes()).hexdigest()==digest,rel
atomic(b/'preparation/source-before-five-epoch.json',publication)
publication.update(commit=new_commit,remote_sha=new_commit,code_root=str(new_root),time=time.time())
config.update(fit_epochs=5,source_commit=new_commit,fit_file_sha256=new_files['reproduction/operations/qwen7b_warmup/mixed_finetune.py']['sha256'])
old_config_path=b/'preparation/config-before-five-epoch.json';shutil.copyfile(str(b/'config.json'),str(old_config_path))
atomic(b/'config.json',config);atomic(b/'preparation/source-publication.json',publication)
env=dict(os.environ,QWEN7B_CODE_ROOT=str(new_root))
r=subprocess.run(['sbatch','--hold','--parsable','--partition=emergency_acd','--job-name=q7b-mixed-cp2-high',str(new_root/'reproduction/operations/qwen7b_warmup/mixed_fit.sbatch')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40);assert r.returncode==0,r.stderr
new=r.stdout.strip().split(';')[0];assert new.isdigit();record.update(new_job=new,new_code_root=str(new_root),status='NEW_FIVE_EPOCH_PENDING_HELD');atomic(record_path,record)
assert row(new)[3]=='PENDING' and 'JobHeldUser' in row(new)[-1]
assert row(old)[3]=='PENDING' and 'JobHeldUser' in row(old)[-1]
command(['scancel',old]);shutil.copyfile(str(b/'submission.json'),str(b/('submission-'+old+'.json')))
atomic(b/'submission.json',dict(job_id=new,partition='emergency_acd',gpus=4,code_root=str(new_root),source_commit=new_commit,replaces_pending_job=old,additional_epochs=5,time=time.time()))
t=b/'job.id.tmp';t.write_text(new+'\n');t.replace(b/'job.id')
command(['scontrol','release',new]);record.update(status='FIVE_EPOCH_GOAL_ALIGNED_JOB_RELEASED',time=time.time(),config_sha256=hashlib.sha256((b/'config.json').read_bytes()).hexdigest());atomic(record_path,record)
print(json.dumps(record));print(command(['squeue','-h','-u','mou176','-o','%i|%j|%T|%P|%R']))
'''

if __name__=='__main__':main()
