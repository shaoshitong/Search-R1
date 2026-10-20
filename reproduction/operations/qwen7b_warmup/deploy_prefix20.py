"""Prepare isolated Prefix20 continuation; replace only the held Prefix5 queue."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]==commit
 files={}
 for name in ['model.py','mixed_finetune.py','mixed_prepare.py','prefix_migration.py','test_prefix_migration.py','mixed_prefix20_fit.sbatch']:
  rel='reproduction/operations/qwen7b_warmup/'+name;data=(repo/rel).read_bytes();files[rel]=dict(data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest())
 sp=importlib.util.spec_from_file_location('control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 code='new_commit='+repr(commit)+'\nnew_files='+repr(files)+'\n'+REMOTE
 assert len(code.encode())<65000,'Use separate bounded code uploads if needed'
 result=m.run('python3 -c '+shlex.quote(code),account='mou176',source_address=address,timeout=380)
 (a.control_root/'outputs/qwen7b-warmup-20261019/prefix20-deployment.txt').write_bytes(result.stdout)
 print('returncode',result.returncode);print(result.stdout.decode());print(result.stderr.decode())
 if result.returncode:raise SystemExit(result.returncode)

REMOTE=r'''
import pathlib,json,subprocess,hashlib,shutil,base64,fcntl,time,os
parent=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');oldbase=parent/'mixed-only-ft-cp2-20261020';b=parent/'mixed-only-ft-cp2-prefix20-20261020'
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=45);assert r.returncode==0,(args,r.stderr);return r.stdout.strip()
old=(oldbase/'job.id').read_text().strip();assert old=='741973'
s=command(['squeue','-h','-j',old,'-o','%i|%u|%j|%T|%P|%R']);assert s.split('|')[1:5]==['mou176','q7b-mixed-cp2-high','PENDING','emergency_acd'] and 'JobHeldUser' in s,s
assert not any((oldbase/'probe'/n).exists() for n in ['progress.json','latest.json','next-update-proof.json'])
with (oldbase/'probe/fit.lock').open('a') as h:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not b.exists(),'Inspect partial prior Prefix20 outcome before reuse';b.mkdir();(b/'preparation').mkdir()
record=dict(status='PREFIX20_MIGRATION_INTENT',time=time.time(),previous_held_job=old,source_checkpoint_epoch=2,source_steps=9146,prefix_tokens=20,source_prefix_tokens=5,epochs=5,source_commit=new_commit,scope='Latest human says cannot remainPrefix5; only replace the held not-started job, preserve CP2 block/Adam and expand old readout with zero columns')
atomic(b/'preparation/handoff.json',record);pub=json.loads((oldbase/'preparation/source-publication.json').read_text());root=b/('repo-'+new_commit[:12]);shutil.copytree(pub['code_root'],str(root))
for rel,proof in new_files.items():
 data=base64.b64decode(proof['data']);assert hashlib.sha256(data).hexdigest()==proof['sha256'];(root/rel).write_bytes(data);pub['files'][rel]=proof['sha256']
for rel,digest in pub['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
pub.update(commit=new_commit,remote_sha=new_commit,code_root=str(root),base=str(b),time=time.time());atomic(b/'preparation/source-publication.json',pub)
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root));runtime=str(root/'reproduction/operations/qwen7b_warmup/runtime.sh')
prep=subprocess.run(['bash',runtime,'-u','-m','reproduction.operations.qwen7b_warmup.mixed_prepare','--base',str(b),'--source-commit',new_commit,'--prefix-tokens','20'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=150);assert prep.returncode==0,prep.stdout+prep.stderr
print(prep.stdout,flush=True)
for module in ['test_prefix_migration','test_restore_isolation']:
 r=subprocess.run(['bash',runtime,'-m','reproduction.operations.qwen7b_warmup.'+module],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35);assert r.returncode==0,r.stdout+r.stderr;print(r.stdout,flush=True)
# Real production checkpoint geometry is tested in the existing CPU runtime.
# No actor forward, GPU request, or source optimizer mutation is performed.
proof_code="""import json,torch,time
from pathlib import Path
from reproduction.operations.qwen7b_warmup.model import Probe,restore_buffers
from reproduction.operations.qwen7b_warmup.prefix_migration import extend,state_hash
b=Path(PHASE);c=json.loads((b/'config.json').read_text());saved=torch.load(c['source_checkpoint']['path'],map_location='cpu',mmap=True,weights_only=False)
model=Probe(c['model_dir'],273,20);names=[n for n,_ in model.named_parameters()];derived,report=extend(saved,names)
params=dict(model.named_parameters());ids=[p for g in derived['optimizer']['param_groups'] for p in g['params']]
for name,i in zip(names,ids):
 for key in ['exp_avg','exp_avg_sq']:assert derived['optimizer']['state'][i][key].shape==params[name].shape
model.load_state_dict(derived['raw'],strict=True);restore_buffers(model,derived['raw_buffers']);assert state_hash(model.state_dict())==report['raw_sha256']
model.load_state_dict(derived['ema'],strict=True);restore_buffers(model,derived['ema_buffers']);assert state_hash(model.state_dict())==report['ema_sha256']
assert all(int(s['step'])==9146 for s in saved['optimizer']['state'].values())
report.update(status='REAL_CP2_PREFIX20_CPU_GEOMETRY_HASH_AND_ADAM_MIGRATION_PASSED',CPU_only=True,GPU_next_update_not_yet_tested=True,source_cp2_sha256=c['source_checkpoint']['sha256'],time=time.time())
(b/'preparation/real-cp2-prefix20-cpu-proof.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
""".replace('PHASE',repr(str(b)))
r=subprocess.run(['bash',runtime,'-u','-c',proof_code],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=150);assert r.returncode==0,r.stdout+r.stderr;print(r.stdout,flush=True)
record.update(status='PREFIX20_PREPARED_REAL_CP2_CPU_CHECKED',time=time.time());atomic(b/'preparation/handoff.json',record)
r=subprocess.run(['sbatch','--hold','--parsable',str(root/'reproduction/operations/qwen7b_warmup/mixed_prefix20_fit.sbatch')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40);assert r.returncode==0,r.stderr
new=r.stdout.strip().split(';')[0];assert new.isdigit();record.update(status='NEW_PREFIX20_JOB_HELD',new_job=new);atomic(b/'preparation/handoff.json',record)
state=command(['squeue','-h','-j',new,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
state=command(['squeue','-h','-j',old,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
command(['scancel',old]);atomic(b/'submission.json',dict(job_id=new,code_root=str(root),source_commit=new_commit,gpus=4,partition='emergency_acd',prefix_tokens=20,epochs=5,time=time.time()))
(b/'job.id').write_text(new+chr(10));command(['scontrol','release',new]);record.update(status='PREFIX20_FIVE_EPOCH_JOB_RELEASED',time=time.time());atomic(b/'preparation/handoff.json',record)
print(json.dumps(record));print(command(['squeue','-h','-u','mou176','-o','%i|%j|%T|%P|%R']))
'''
if __name__=='__main__':main()
