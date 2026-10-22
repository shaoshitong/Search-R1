"""Stage fourth matched L21/P80 cell only; no Slurm submission or job mutation."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1]
BRANCH='codex/q7b-online-L21P80-factorial-20261022'
WORKER_COMMIT='5a3cfa1588c883d1c816d77ff95191acaa71eb35'
SUBMIT='reproduction/reports/qwen7b-algorithm-improvement-20261021/online-layer-prefix/submit-l21p80-factorial-once.py'
REMOTE='''import pathlib,json,hashlib,fcntl,os,re,subprocess,time
a=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');suite=a/'online-mixed-layer-prefix-20261022-2gpu-v1';b=suite/'layer21-prefix80';b.mkdir(exist_ok=True);guard=(b/'stage.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(p.read_bytes())
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):assert proof(ref['path'])=={k:ref[k]for k in('path','bytes','sha256')};return read(pathlib.Path(ref['path']))
def once(p,x):
 raw=(json.dumps(x,sort_keys=True,separators=(',',':'))+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Different prepared factorial binding must be preserved'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
pubref=dict(path=str(suite/'source-publication.json'),bytes=1133,sha256='6b285b3aedc993d18caf26af240864381ba84d6f7502595df6a1192a193c6ff3');pub=checked(pubref);assert pub['commit']==v['worker_commit'];code=pathlib.Path(pub['code_root'])
for name,ref in pub['files'].items():assert proof(code/name)==dict(path=str(code/name),**ref)
fixture_root=a/'online-mixed-layer-prefix-20261022-v1';fixture=read(fixture_root/'fixture-complete.json');assert fixture['status']=='ONLINE_MIXED_MATCHED8192_G10_FIXED2048_TOKENS_READY'and fixture['training_questions']==8192 and fixture['validation_questions']==2048 and fixture['training_trajectories']==81920 and fixture['fixed_G5_sha256']=='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'
source=suite/'layer27-prefix80/training.sbatch';text=source.read_text();assert'--layer 27 --prefix 80'in text and'--nproc-per-node=2'in text and'--gres=gpu:2'in text
text=text.replace('L27P80','L21P80').replace('/layer27-prefix80/','/layer21-prefix80/').replace('--layer 27 --prefix 80','--layer 21 --prefix 80')
text=re.sub(r'(?m)^#SBATCH --partition=\\S+$','#SBATCH --partition=acd_u',text);text=re.sub(r'(?m)^#SBATCH --time=\\S+$','#SBATCH --time=02:00:00',text)
assert'L27P80'not in text and'/layer27-prefix80/'not in text and'--layer 27'not in text
for required in('--job-name=q7b-online2-L21P80','--partition=acd_u','--time=02:00:00','--cpus-per-task=12','--mem=128G','--gres=gpu:2','--nproc-per-node=2','--layer 21 --prefix 80 --epochs 3 --lr 0.000005',str(code)):assert required in text,required
entry=b/'training.sbatch'
if entry.exists():assert entry.read_text()==text,'Preserve already staged L21P80 entry'
else:entry.write_text(text)
subprocess.check_call(['bash','-n',str(entry)])
inputs=fixture_root/'inputs';metadata=dict(fixture_complete=proof(fixture_root/'fixture-complete.json'),train_groups=proof(inputs/'train-groups.json'),validation_groups=proof(inputs/'validation-groups.json'),fixed_train2048=proof(inputs/'fixed-train2048.json'),epoch_plans=[proof(inputs/('epoch-%02d-plan.json'%e))for e in(1,2,3)],token_shards_original_frozen_proofs=fixture['token_shards'])
binding=dict(status='L21P80_MATCHED_FACTORIAL_FOURTH_CELL_READY_NOT_SUBMITTED',layer=21,capture_layer=20,prefix=80,
 job_name='q7b-online2-L21P80',entry=proof(entry),source_entry_copied_without_mutation=proof(source),source_publication=pubref,
 runtime_commit=v['worker_commit'],code_root=str(code),metadata=metadata,
 training=dict(questions=8192,real_trajectories_G10=81920,epochs=3,actual_updates_budget=192,global_batch=128,world_size=2,local_batch=64,micro_questions=8,gradient_accumulation_microbatches=8,lr=5e-6,EMA_decay=.99,model_and_head_seed=9137,official_block_fresh=True,official_actor_frozen=True,own_G5_epoch_plans_identical=True,hidden_features_persisted=False),
 resources=dict(partition='acd_u',qos='normal',GPUs=2,CPUs=12,memory_GiB=128,time='02:00:00'),
 isolation=dict(only_new_variant_directory_written=True,existing_entries_or_code_changed=False,existing_jobs_stopped_or_modified=False,final4096_read=False),
 stage_controller=v['controller'],root_submit_controller=v['submit_controller'],root_must_single_submit_after_same_name_and_intent_check=True,Slurm_submitted=False)
once(b/'factorial-ready.json',binding)
print(json.dumps(dict(status=binding['status'],ready=proof(b/'factorial-ready.json'),entry=proof(entry),runtime_commit=v['worker_commit'],Slurm_submitted=False)))
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
 actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,actual],cwd=R)
 own=str(Path(__file__).resolve().relative_to(R));source=subprocess.check_output(['git','show',a.commit+':'+own],cwd=R);assert source==Path(__file__).read_bytes()
 submit=subprocess.check_output(['git','show',a.commit+':'+SUBMIT],cwd=R)
 v=dict(worker_commit=WORKER_COMMIT,controller=dict(commit=a.commit,path=own,sha256=hashlib.sha256(source).hexdigest()),submit_controller=dict(commit=a.commit,path=SUBMIT,sha256=hashlib.sha256(submit).hexdigest()))
 spec=importlib.util.spec_from_file_location('L21P80_factorial_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 script='v='+repr(v)+'\n'+REMOTE;compile(script,'stage_factorial_remote','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=45)
 if r.returncode:raise RuntimeError(r.stdout[-1600:]+r.stderr[-3000:])
 result=json.loads(r.stdout.splitlines()[-1]);dest=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/online-layer-prefix';(dest/'L21P80-actual-stage.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


compile(REMOTE,'stage_factorial_static','exec')
if __name__=='__main__':main()
