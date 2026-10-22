"""Stage L15/L18/L24 P20 matched pilots, without Slurm mutation."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1]
P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-online-matched-depth-expansion-20261022'
ORIGINAL='5a3cfa1588c883d1c816d77ff95191acaa71eb35'
NAMES=['online_mixed_depth_expansion.py','deploy_online_matched_depth_expansion.py']
REPORT='reproduction/reports/qwen7b-algorithm-improvement-20261021/online-matched-depth-expansion/'
REMOTE='''import pathlib,json,hashlib,fcntl,os,re,subprocess,time,base64,zlib
suite=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');guard=(suite/'depth-expansion-stage.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def put(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw,'Immutable staged artifact differs: '+str(p)
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def once(p,x):put(p,(json.dumps(x,sort_keys=True,separators=(',',':'))+'\\n').encode())
pubpath=suite/'source-publication.json';assert proof(pubpath)==dict(path=str(pubpath),bytes=1133,sha256='6b285b3aedc993d18caf26af240864381ba84d6f7502595df6a1192a193c6ff3');pub=read(pubpath);assert pub['commit']==v['original'];old=pathlib.Path(pub['code_root']);code=suite/('depth-expansion-code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
files={}
for name in('peer_long_prefix_diagnostic.py','native_mixed_group_classifier.py','runtime.sh'):
 rel=v['prefix']+name;raw=(old/rel).read_bytes();ref=pub['files'][rel];assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'];put(code/rel,raw);files[rel]=ref
for rel,packed in v['payload'].items():
 raw=zlib.decompress(base64.b64decode(packed));assert hashlib.sha256(raw).hexdigest()==v['files'][rel]['sha256'];put(code/rel,raw);compile(raw,rel,'exec');files[rel]=v['files'][rel]
original=(old/(v['prefix']+'online_mixed_prefix_explore.py')).read_bytes();expanded=(code/(v['prefix']+'online_mixed_depth_expansion.py')).read_bytes();assert original.count(b'choices=[21,27]')==1 and expanded==original.replace(b'choices=[21,27]',b'choices=[15,18,21,24,27]')
publication=dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=files,original_runtime_commit=v['original'],only_runtime_change='CLI accepts block indices 15/18/24 as well as original21/27; computation unchanged',controller=v['controller']);pubnew=suite/'depth-expansion-source-publication.json';once(pubnew,publication)
f=suite.parent/'online-mixed-layer-prefix-20261022-v1';fixture=read(f/'fixture-complete.json');assert fixture['status']=='ONLINE_MIXED_MATCHED8192_G10_FIXED2048_TOKENS_READY'and fixture['training_questions']==8192 and fixture['validation_questions']==2048 and fixture['fixed_G5_sha256']=='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'
inputs=f/'inputs';metadata=dict(fixture_complete=proof(f/'fixture-complete.json'),train_groups=proof(inputs/'train-groups.json'),validation_groups=proof(inputs/'validation-groups.json'),fixed_train2048=proof(inputs/'fixed-train2048.json'),epoch_plans=[proof(inputs/('epoch-%02d-plan.json'%e))for e in(1,2,3)],token_shards_original_frozen_proofs=fixture['token_shards']);source=suite/'layer27-prefix20/training.sbatch';template=source.read_text();assert '--layer 27 --prefix 20'in template
results=[]
for layer in(15,18,24):
 b=suite/('layer%d-prefix20'%layer);b.mkdir(exist_ok=True);entry=b/'training.sbatch';text=template.replace('L27P20','L%dP20'%layer).replace('/layer27-prefix20/','/layer%d-prefix20/'%layer).replace('--layer 27 --prefix 20','--layer %d --prefix 20'%layer).replace(str(old),str(code)).replace('qwen7b_warmup.online_mixed_prefix_explore','qwen7b_warmup.online_mixed_depth_expansion');text=re.sub(r'(?m)^#SBATCH --partition=\\S+$','#SBATCH --partition=acd_u',text);text=re.sub(r'(?m)^#SBATCH --time=\\S+$','#SBATCH --time=00:30:00',text)
 for item in('--job-name=q7b-online2-L%dP20'%layer,'--partition=acd_u','--time=00:30:00','--gres=gpu:2','--cpus-per-task=12','--mem=128G','--nproc-per-node=2','--layer %d --prefix 20 --epochs 3 --lr 0.000005'%layer,str(code)):assert item in text,item
 put(entry,text.encode());subprocess.check_call(['bash','-n',str(entry)])
 binding=dict(status='MATCHED_DEPTH_EXPANSION_READY_NOT_SUBMITTED',layer=layer,capture_layer=layer-1,prefix=20,job_name='q7b-online2-L%dP20'%layer,entry=proof(entry),source_entry_copied_without_mutation=proof(source),source_publication=proof(pubnew),runtime_commit=v['commit'],original_runtime_commit=v['original'],code_root=str(code),metadata=metadata,training=dict(questions=8192,real_trajectories_G10=81920,epochs=3,actual_updates_budget=192,global_batch=128,world_size=2,local_batch=64,micro_questions=8,gradient_accumulation_microbatches=8,lr=5e-6,EMA_decay=.99,model_and_head_seed=9137,official_block_fresh=True,official_actor_frozen=True,own_G5_epoch_plans_identical=True,hidden_features_persisted=False),resources=dict(partition='acd_u',qos='normal',GPUs=2,CPUs=12,memory_GiB=128,time='00:30:00'),isolation=dict(existing_entries_or_code_changed=False,existing_jobs_stopped_or_modified=False,final4096_read=False),root_submit_controller=v['submit_controller'],Slurm_submitted=False)
 once(b/'depth-ready.json',binding);results.append(dict(layer=layer,prefix=20,entry=proof(entry),ready=proof(b/'depth-ready.json')))
print(json.dumps(dict(status='THREE_MATCHED_DEPTH_CELLS_STAGED_NOT_SUBMITTED',runtime_commit=v['commit'],publication=proof(pubnew),entries=results,Slurm_submitted=False)))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args();actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];assert actual==a.commit
 payload={};files={}
 for name in NAMES:
  rel=P+name;raw=subprocess.check_output(['git','show',a.commit+':'+rel],cwd=R);assert raw==(R/rel).read_bytes();payload[rel]=base64.b64encode(zlib.compress(raw)).decode();files[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 own=P+Path(__file__).name;submit=REPORT+'submit-depth-cells-once.py';raw=subprocess.check_output(['git','show',a.commit+':'+submit],cwd=R);assert raw==(R/submit).read_bytes()
 v=dict(prefix=P,original=ORIGINAL,commit=a.commit,branch=BRANCH,payload=payload,files=files,controller=dict(commit=a.commit,path=own,sha256=files[own]['sha256']),submit_controller=dict(commit=a.commit,path=submit,sha256=hashlib.sha256(raw).hexdigest()))
 spec=importlib.util.spec_from_file_location('depth_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();script='v='+repr(v)+'\n'+REMOTE;compile(script,'depth_stage_remote','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000,len(command.encode());r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
 if r.returncode:raise RuntimeError(r.stdout[-2000:]+r.stderr[-3000:])
 result=json.loads(r.stdout.splitlines()[-1]);dest=R/REPORT;dest.mkdir(parents=True,exist_ok=True);(dest/'actual-stage.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
compile(REMOTE,'depth_stage_static','exec')
if __name__=='__main__':main()
