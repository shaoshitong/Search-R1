import pathlib,json,subprocess,time,hashlib,fcntl,re,os
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-v1')
def read(p):return json.loads(p.read_text())
def atomic(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,sort_keys=True)+'\n');q.replace(p)
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if r.returncode:raise RuntimeError(r.stderr[-1800:])
 return r.stdout.strip()
lock=(b/'suite-submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
fixture=read(b/'fixture-complete.json');assert fixture['training_questions']==8192 and fixture['validation_questions']==2048 and not fixture['question_hash_train_validation_overlap']
pub=read(b/'source-publication.json');code=pathlib.Path(pub['code_root'])
for n,x in pub['files'].items():
 raw=(code/n).read_bytes();assert len(raw)==x['bytes']and hashlib.sha256(raw).hexdigest()==x['sha256']
worker=(code/'reproduction/operations/qwen7b_warmup/online_mixed_prefix_explore.py').read_text();assert 'requires_grad_(True)'in worker and 'native_gradient_L2'in worker
results=[]
for layer,prefix in ((27,20),(21,20),(27,80)):
 d=b/('layer%d-prefix%d'%(layer,prefix));name='q7b-online-L%dP%d'%(layer,prefix);entry=d/'training.sbatch';receipt=d/'root-submission.json'
 if receipt.exists():results.append(read(receipt));continue
 active=command(['squeue','-h','-u','mou176','-n',name,'-o','%i|%T|%R']);assert not active,'Owned existing job: reconcile '+active
 assert not(d/'root-submission-intent.json').exists(),'Unknown prior submit: inspect same job name before retry'
 assert not list(d.glob('progress-rank-*.json')),'Existing experiment must not duplicate'
 atomic(d/'root-submission-intent.json',dict(time=time.time(),entry=str(entry),publication_commit=pub['commit'],name=name))
 result=command(['sbatch','--parsable',str(entry)]);assert re.fullmatch(r'[0-9]+(?:;[^ ]+)?',result),result;job=result.split(';')[0]
 x=dict(job_id=job,layer=layer,prefix=prefix,name=name,publication_commit=pub['commit'],state='SUBMITTED_NOT_PROGRESS_VERIFIED',time=time.time());atomic(receipt,x)
 x['slurm_identity']=command(['scontrol','show','job','-o',job]);assert all(v in x['slurm_identity']for v in ('JobName='+name,'QOS=normal','Partition=acd_u','NumCPUs=24','MinMemoryNode=256G','Dependency=(null)'));assert 'gres/gpu=4'in x['slurm_identity']or 'TresPerNode=gres/gpu:4'in x['slurm_identity'];atomic(receipt,x);results.append(x)
print(json.dumps(dict(status='THREE_MATCHED_FOUR_GPU_JOBS_SUBMITTED',results=results)))
