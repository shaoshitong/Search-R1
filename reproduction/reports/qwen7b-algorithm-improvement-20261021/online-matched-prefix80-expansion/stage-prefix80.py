"""Stage two new geometries using the unchanged, published depth runtime."""
import pathlib,json,hashlib,subprocess,re,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
S=A/'online-mixed-layer-prefix-20261022-2gpu-v1'
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
pubpath=S/'depth-expansion-source-publication.json';pub=read(pubpath)
assert proof(pubpath)['sha256']=='cc76d46721f9416b7c32a5c38b2e88a68b7e4eb2acbc608ec3bd71bb7e39a6cf'
assert pub['commit']=='c366ad4bf11c32f78b2622775c2a7bf83e1a5299'
sourcepub=read(S/'source-publication.json');old=sourcepub['code_root'];new=pub['code_root'];P='reproduction/operations/qwen7b_warmup/'
for rel,ref in pub['files'].items():
 raw=(pathlib.Path(new)/rel).read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
template=S/'layer21-prefix80/training.sbatch';text0=template.read_text();assert '--layer 21 --prefix 80'in text0
out=[]
for layer in (15,18):
 b=S/('layer%d-prefix80'%layer);b.mkdir(exist_ok=True);entry=b/'training.sbatch'
 text=text0.replace('L21P80','L%dP80'%layer).replace('/layer21-prefix80/','/layer%d-prefix80/'%layer).replace('--layer 21 --prefix 80','--layer %d --prefix 80'%layer).replace(old,new).replace('qwen7b_warmup.online_mixed_prefix_explore','qwen7b_warmup.online_mixed_depth_expansion')
 text=re.sub(r'(?m)^#SBATCH --partition=\S+$','#SBATCH --partition=acd_u',text)
 text=re.sub(r'(?m)^#SBATCH --time=\S+$','#SBATCH --time=00:30:00',text)
 for needed in ['--job-name=q7b-online2-L%dP80'%layer,'--gres=gpu:2','--cpus-per-task=12','--mem=128G','--layer %d --prefix 80 --epochs 3 --lr 0.000005'%layer,new]:assert needed in text,needed
 if entry.exists():assert entry.read_text()==text
 else:entry.write_text(text)
 subprocess.check_call(['bash','-n',str(entry)])
 ready=dict(status='PREFIX80_MATCHED_GEOMETRY_READY_NOT_SUBMITTED',entry=proof(entry),job_name='q7b-online2-L%dP80'%layer,layer=layer,prefix=80,runtime_commit=pub['commit'],source_publication=proof(pubpath),source_entry=proof(template),training=dict(questions=8192,epochs=3,updates=192,global_batch=128,world_size=2,micro=8,accumulation=8,lr=5e-6,EMA_decay=.99,official_fresh_block_head_Adam=True,same_frozen_own_G5_plans=True,hidden_features_persisted=False),resources=dict(partition='acd_u',GPUs=2,CPUs=12,memGiB=128,time='00:30:00'),rationale='L15/L18 P20 improved BA; matched P80 isolates longer prefix against own P20 and L21P80.',final4096_read=False,Slurm_submitted=False)
 p=b/'prefix80-ready.json'
 if p.exists():assert read(p)==ready
 else:p.write_text(json.dumps(ready,sort_keys=True)+'\n')
 out.append(ready)
print(json.dumps(dict(time=time.time(),entries=out,training_runtime_changed=False,Slurm_submitted=False)))
