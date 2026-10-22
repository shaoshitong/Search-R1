import pathlib,json,subprocess,time,fcntl,hashlib,re
a=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');old=a/'online-mixed-layer-prefix-20261022-v1';new=a/'online-mixed-layer-prefix-20261022-2gpu-v1'
def read(p):return json.loads(p.read_text())
def atomic(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,sort_keys=True)+'\n');q.replace(p)
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if r.returncode:raise RuntimeError(r.stderr[-1200:])
 return r.stdout.strip()
lock=(old/'two-gpu-handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pub=read(new/'source-publication.json');code=pathlib.Path(pub['code_root'])
for name,r in pub['files'].items():
 raw=(code/name).read_bytes();assert len(raw)==r['bytes']and hashlib.sha256(raw).hexdigest()==r['sha256']
worker=(code/'reproduction/operations/qwen7b_warmup/online_mixed_prefix_explore.py').read_text();assert 'requires_grad_(True)'in worker and 'assert world==2'in worker
fixture=read(old/'fixture-complete.json');assert fixture['training_questions']==8192 and fixture['validation_questions']==2048
handoff=old/'two-gpu-human-authorized-handoff.json';out=read(handoff)if handoff.exists()else dict(observed_time=time.time(),human_instruction='哦，对了，这个如果现在排不到的话，就是时间这么久的话，可以两卡，两张卡，然后把梯度累积增多一倍嘛，然后去跑这个实验',instruction_source='latest direct human turn; original message timestamp not exposed',scope='only pending three online 4GPU pilots; preserve RUNNING full169028 job749015',new_global_batch=128,new_local_batch=64,micro_questions=8,new_accumulation=8,previous_accumulation=4,publication=pub['commit'],jobs=[])
atomic(handoff,out)
for layer,prefix in ((27,20),(21,20),(27,80)):
 previous=old/('layer%d-prefix%d'%(layer,prefix));target=new/('layer%d-prefix%d'%(layer,prefix));prior=read(previous/'root-submission.json');jid=prior['job_id'];expected='q7b-online2-L%dP%d'%(layer,prefix);record=target/'root-submission.json'
 if record.exists():continue
 identity=command(['scontrol','show','job','-o',jid]);assert 'UserId=mou176'in identity and 'JobName='+prior['name']in identity
 entry=target/'training.sbatch';text=entry.read_text();assert '--gres=gpu:2'in text and '--nproc-per-node=2'in text and '--cpus-per-task=12'in text and '--mem=128G'in text
 item=dict(old_job=jid,layer=layer,prefix=prefix,before=identity,human_instruction_recorded_before_cancel=True);out['jobs'].append(item);atomic(handoff,out)
 if 'JobState=PENDING'in identity:command(['scancel','--state=PENDING',jid])
 after=command(['scontrol','show','job','-o',jid]);item['old_after']=after;atomic(handoff,out)
 if 'JobState=RUNNING'in after:item['action']='KEPT_ALREADY_RUNNING_FOUR_GPU_NO_DUPLICATE';atomic(handoff,out);continue
 assert 'JobState=CANCELLED'in after,'Old terminal must be reconciled before new submit: '+after
 same=command(['squeue','-h','-u','mou176','-n',expected,'-o','%i|%T|%R']);assert not same,'Do not duplicate '+same
 intent=target/'root-submission-intent.json';assert not intent.exists(),'Uncertain prior submit; inspect same-name jobs before retry'
 atomic(intent,dict(time=time.time(),entry=str(entry),name=expected,publication_commit=pub['commit'],replaces_cancelled_job=jid))
 answer=command(['sbatch','--parsable',str(entry)]);assert re.fullmatch(r'[0-9]+(?:;[^ ]+)?',answer);job=answer.split(';')[0];result=dict(job_id=job,layer=layer,prefix=prefix,name=expected,publication_commit=pub['commit'],replaces_cancelled_pending_job=jid,time=time.time(),state='SUBMITTED_NOT_PROGRESS_VERIFIED');atomic(record,result)
 result['slurm_identity']=command(['scontrol','show','job','-o',job]);assert all(s in result['slurm_identity']for s in ('JobName='+expected,'Dependency=(null)','Partition=acd_ue','NumCPUs=12','MinMemoryNode=128G'));assert 'gres/gpu=2'in result['slurm_identity']or 'TresPerNode=gres/gpu:2'in result['slurm_identity'];atomic(record,result);item['new_job']=result;item['action']='PENDING_FOUR_GPU_CANCELLED_TWO_GPU_SUBMITTED';atomic(handoff,out)
print(json.dumps(dict(time=time.time(),jobs=[dict(old_job=x['old_job'],action=x.get('action'),new_job=x.get('new_job',{}).get('job_id'))for x in out['jobs']])))
