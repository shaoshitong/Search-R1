"""Recover failed, unaccepted replay with deterministic kernels and diagnostics."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]==commit
 files={}
 for name in ['mixed_finetune.py','mixed_prefix20_fit.sbatch','replay_diagnostics.py','test_reducer_warmup.py']:
  rel='reproduction/operations/qwen7b_warmup/'+name;data=(repo/rel).read_bytes();files[rel]=dict(data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest())
 sp=importlib.util.spec_from_file_location('c',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 r=m.run('python3 -c '+shlex.quote('commit='+repr(commit)+'\nfiles='+repr(files)+'\n'+REMOTE),account='mou176',source_address=address,timeout=280)
 (a.control_root/'outputs/qwen7b-warmup-20261019/prefix20-deterministic-recovery.txt').write_bytes(r.stdout);print('returncode',r.returncode);print(r.stdout.decode());print(r.stderr.decode())
 if r.returncode:raise SystemExit(r.returncode)

REMOTE=r'''
import pathlib,json,subprocess,time,shutil,base64,hashlib,fcntl,os
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/mixed-only-ft-cp2-prefix20-20261020')
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=45);assert r.returncode==0,(args,r.stderr);return r.stdout.strip()
old=(b/'job.id').read_text().strip();assert old=='742009'
assert not command(['squeue','-h','-j',old,'-o','%T']);accounting=command(['sacct','-X','-n','-P','-j',old,'--format=JobID,State,ExitCode']);assert accounting.startswith(old+'|FAILED|1:0'),accounting
assert not any((b/'probe'/n).exists() for n in ['progress.json','latest.json','next-update-proof.json']),'Do not discard accepted progress'
with (b/'probe/fit.lock').open('a') as h:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
record_path=b/'preparation/deterministic-recovery.json';assert not record_path.exists(),'Resolve existing attempt first'
record=dict(status='DETERMINISTIC_REPLAY_RECOVERY_INTENT',old_job=old,time=time.time(),source_commit=commit,source_steps=9146,accepted_continuation_updates=0,exact_hash_gate_retained=True,fixes=['FlashAttention deterministic=True explicitly supported by actual installed Transformers4.46.3/FlashAttention2.7.4','CUBLAS4096:8 and torch deterministic algorithms','Gradient-only DDP bucket warmup without Adam or EMA update','All-rank exact tensor diagnostics retained on mismatch'])
atomic(record_path,record);c=json.loads((b/'config.json').read_text());pub=json.loads((b/'preparation/source-publication.json').read_text());old_pub=json.loads(json.dumps(pub));assert c['fit_epochs']==5 and c['prefix_tokens']==20
h=hashlib.sha256()
with open(c['source_checkpoint']['path'],'rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==c['source_checkpoint']['sha256'];record['fresh_source_cp2_sha256']=h.hexdigest();atomic(record_path,record)
root=b/('repo-'+commit[:12]);assert not root.exists();shutil.copytree(pub['code_root'],str(root))
for rel,item in files.items():
 data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256'];(root/rel).write_bytes(data);pub['files'][rel]=item['sha256']
for rel,digest in pub['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
runtime=str(root/'reproduction/operations/qwen7b_warmup/runtime.sh');env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),FLASH_ATTENTION_DETERMINISTIC='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
r=subprocess.run(['bash',runtime,'-m','reproduction.operations.qwen7b_warmup.test_reducer_warmup'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40);assert r.returncode==0,r.stdout+r.stderr;record['actual_runtime_cpu_warmup_test']=r.stdout.strip();print(r.stdout,flush=True)
history=b/'preparation/failure-742009';history.mkdir(exist_ok=True)
for name in ['restore-acceptance.json','prefix20-initialization-proof.json','deterministic-warmup.json']:
 p=b/'probe'/name
 if p.exists():p.rename(history/name)
atomic(history/'source-publication.json',old_pub);shutil.copyfile(str(b/'config.json'),str(history/'config.json'))
c.update(source_commit=commit,fit_file_sha256=files['reproduction/operations/qwen7b_warmup/mixed_finetune.py']['sha256']);pub.update(commit=commit,remote_sha=commit,code_root=str(root),time=time.time())
atomic(b/'config.json',c);atomic(b/'preparation/source-publication.json',pub)
r=subprocess.run(['sbatch','--hold','--parsable',str(root/'reproduction/operations/qwen7b_warmup/mixed_prefix20_fit.sbatch')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40);assert r.returncode==0,r.stderr
new=r.stdout.strip().split(';')[0];assert new.isdigit();record.update(new_job=new,new_code_root=str(root),status='DETERMINISTIC_REPLAY_JOB_HELD');atomic(record_path,record)
shutil.copyfile(str(b/'submission.json'),str(b/('submission-'+old+'.json')));atomic(b/'submission.json',dict(job_id=new,code_root=str(root),source_commit=commit,gpus=4,partition='emergency_acd',prefix_tokens=20,epochs=5,time=time.time(),replaces_failed_trial=old))
(b/'job.id').write_text(new+chr(10));command(['scontrol','release',new]);record.update(status='DETERMINISTIC_REPLAY_JOB_RELEASED',time=time.time());atomic(record_path,record)
print(json.dumps(record));print(command(['squeue','-h','-u','mou176','-o','%i|%j|%T|%P|%R']))
'''
if __name__=='__main__':main()
