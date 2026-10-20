import pathlib,json,hashlib,subprocess,fcntl,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');b=B/'algorithm-improvement-20261021/training-extraG5-128-20261021/materialized';p=b/'prepared.json'
read=lambda p:json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25);assert r.returncode==0,r.stdout+r.stderr;return r.stdout.strip()
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
with (b/'sampling-submit.lock').open('a') as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);proof=read(p);assert proof['status']=='EXTRA_REAL_G5_PILOT_PREPARED_NOT_SUBMITTED' and proof['questions']==128 and proof['trajectories']==640
 for key in ['config','input_manifest','sampling_script']:assert sha(proof[key]['path'])==proof[key]['sha256']
 config=read(proof['config']['path']);assert config['warmup_only'] and not config['actor_updates_authorized'] and config['extra_response_indices']==list(range(5,10))
 script=pathlib.Path(proof['sampling_script']['path']);command(['bash','-n',str(script)])
 exports='\n'.join(x for x in script.read_text().splitlines() if x.startswith('export '));assert command(['bash','-euc',exports+'\nprintf %s "$QWEN7B_CODE_ROOT"'])==config['code_root']
 name='q7b-G10pilot-128-20261021'
 if (b/'job.id').exists():
  j=(b/'job.id').read_text().strip();print(json.dumps(dict(status='EXISTING_PILOT_NOT_RESUBMITTED',job_id=j,accounting=command(['sacct','-X','-n','-P','-j',j,'-o','JobID,State,ExitCode']))))
 else:
  existing=command(['squeue','-h','-u','mou176','-n',name,'-o','%i|%T|%R']);assert not existing,'Inspect same-name array before submission';assert not (b/'submission-intent.json').exists(),'Unknown previous submission: inspect rather than duplicate'
  atomic(b/'submission-intent.json',dict(job_name=name,proof_sha256=sha(p),config=proof['config'],array='0-3%1',time=time.time()))
  j=command(['sbatch','--hold','--parsable','--job-name='+name,'--array=0-3%1','--time=01:00:00',str(script)]).split(';')[0];assert j.isdigit()
  with (b/'job.id').open('x') as f:f.write(j+'\n')
  rec=dict(status='EXTRA_G5_PILOT_ARRAY_HELD',job_id=j,base=str(b),questions=128,new_real_trajectories=640,gpus_per_task=2,max_concurrent_tasks=1,requested_partition='acd_u',source_commit=config['source_commit'],config=proof['config'],sampling_script=proof['sampling_script'],no_dependency=True,time=time.time());atomic(b/'submission.json',rec)
  q=command(['squeue','-h','-j',j,'-o','%T|%R']);assert 'PENDING|' in q and 'JobHeldUser' in q;command(['scontrol','release',j]);rec.update(status='EXTRA_G5_PILOT_ARRAY_RELEASED_VERIFY_REAL_BATCH',queue=command(['squeue','-h','-j',j,'-o','%i|%j|%T|%P|%R']));atomic(b/'submission.json',rec);print(json.dumps(rec))
